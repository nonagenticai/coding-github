"""
GitHub webhook handler endpoints.
Handles GitHub webhook events for automated code analysis and PR automation.
"""

# Removed to fix TypeAdapter error with Request parameter
# from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from coding_client.auth.tenant import DEFAULT_TENANT_ID, TENANT_ID_HEADER
from coding_client.models.responses import (  # TODO: verify coding-client import
    internal_error,
)
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from config.settings import get_settings, resolve_webhook_mode
from interfaces.webhook_interface import WebhookEventHandlerRegistry  # TODO: verify
from schemas.github_api_models import GitHubWebhookResponse, ResponseStatus
from services.andon import require_andon_released
from services.ci_waits.manager import CI_WAIT_EVENTS, get_ci_wait_manager
from services.github.github_client import PRAnalysisDict
from services.github.github_service import (  # TODO: verify - was src.app.core.github
    GitHubConfig,
    GitHubService,
)
from services.github.secure_webhook_handler import SecureWebhookHandler
from services.github_integration_core import GitHubIntegrationHub
from services.outbound import optional_sibling_headers

router = APIRouter()
logger = logging.getLogger(__name__)

# Per-push analysis caps. A push can carry far more commits/files than is worth
# analysing inline on a webhook; these bound the work. They are intentional, but
# must NOT be silent — when a push exceeds a cap we only analyse the first N and
# log a warning so partial coverage is observable rather than looking complete.
_MAX_COMMITS_ANALYZED = 10
_MAX_FILES_SCANNED = 20


def _webhook_github_token(repo_full_name: str | None, feature: str) -> str | None:
    """The credential for a webhook-driven GitHub call — there is none.

    ⛔ There is no platform personal access token (owner ruling 2026-09-30).
    GitHub access is the internal App (platform repos, platform TENANTS only) or
    the public App through the fr tenant binding, and a GitHub webhook carries
    no fr tenant — so neither App can be resolved here, and a tenant must not be
    invented. Every webhook feature that needs GitHub API access is therefore
    DISABLED: this logs and returns ``None``, and each caller skips. That is the
    behaviour production already had while the old PAT setting was empty.
    """
    logger.info(
        "skipped: webhook %s for %s needs a GitHub credential; webhooks carry no tenant "
        "for the GitHub App and there is no PAT fallback",
        feature,
        repo_full_name,
    )
    return None


# ⛔ GATED BY THE ANDON BRAKE, and this is the least obvious of the gated routes,
# so read why before removing it. This receiver's LIVE path posts to GitHub:
# SecureWebhookHandler.handle_pull_request_event (secure_webhook_handler.py:327,
# the registered one) falls through to ``routers.github_webhooks
# .analyze_pull_request_changes`` (:365-367 — its own comment says "this is the
# live path"), which calls ``_post_pr_comment`` → ``create_issue_comment``. So an
# inbound webhook writes to a repository, and a pulled cord must stop it.
# ⚠️ THE TRADE, stated rather than hidden: while the cord is pulled GitHub sees a
# 503 on the delivery and the code-graph update for that event does not happen.
# GitHub does not auto-retry, so recovery is the Redeliver button on the webhook's
# delivery log after the cord is released. That is deliberate: a lost analysis
# comment is recoverable, a GitHub write during a brake is not.
@router.post(
    "/webhook",
    operation_id="handle_github_webhook",
    response_model=GitHubWebhookResponse,
    dependencies=[Depends(require_andon_released)],
)
async def github_webhook_handler(
    request: Request,
    x_github_event: str = Header("ping", alias="X-GitHub-Event"),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
    x_github_delivery: str | None = Header(None, alias="X-GitHub-Delivery"),
) -> GitHubWebhookResponse | JSONResponse:
    """
    Secure GitHub webhook handler with signature verification.

    Handles all GitHub webhook events with proper security validation
    and routes them to appropriate processors.

    ⛔ NO ``require_tenant`` dependency, deliberately. This endpoint is called
    by GitHub itself, which cannot send an ``X-Tenant-Id`` header or a licence
    token -- so a ``Depends(require_tenant)`` here 401s every real delivery
    (measured: with no ``TRUSTED_CIDRS``/``DEFAULT_TENANT_ID`` configured, which
    is this service's live default, ``require_tenant`` has no fallback and
    always raises 401 for an unauthenticated caller). The correct guard for an
    inbound GitHub delivery is the HMAC signature -- GitHub's actual credential
    -- checked below by ``SecureWebhookHandler``, not a tenant header GitHub was
    never going to send. This mirrors ``routers.github_internal``, which guards
    its machine-to-machine route with a shared secret instead of
    ``require_tenant`` for the identical reason.
    """
    try:
        payload = await request.body()
        # Use registry pattern to avoid circular dependency
        registry_handler = WebhookEventHandlerRegistry.get_handler("secure_github")
        webhook_handler: SecureWebhookHandler
        if registry_handler is None:
            # Fallback to direct import if handler not registered
            from services.github.secure_webhook_handler import (
                get_secure_webhook_handler,
            )

            webhook_handler = get_secure_webhook_handler()
        else:
            # Cast since SecureWebhookHandler should be registered
            webhook_handler = cast(SecureWebhookHandler, registry_handler)

        outcome_only = resolve_webhook_mode(get_settings().webhook_mode) == "outcome_only"
        ci_wait = get_ci_wait_manager()
        verified = False
        if outcome_only or ci_wait is not None:
            # HMAC FIRST (unchanged contract: bad/missing secret or signature
            # => 401). Only an authenticated delivery is then filtered, so an
            # unsigned caller cannot probe which events are ignored -- and only an
            # authenticated one may trigger a CI-wait evaluation.
            await webhook_handler.verify_security(request, payload, x_hub_signature_256)
            verified = True
        if ci_wait is not None:
            # CI-checks waits: every verified delivery is an evaluation trigger
            # (its repo's open waits for a check/status/workflow/PR event, stale
            # waits for any). Background task, reads GitHub only -- it wakes NO
            # other handler, so outcome_only keeps every dormant one dormant.
            ci_wait.schedule_delivery(x_github_event, payload)
        if outcome_only:
            if x_github_event == "pull_request":
                return await _handle_pull_request_outcome_only(
                    payload=payload, delivery_id=x_github_delivery
                )
            if x_github_event in _CI_WAIT_ONLY_EVENTS and ci_wait is not None:
                return _ci_wait_trigger_only(x_github_event, x_github_delivery)
            if x_github_event != "workflow_run":
                return _ignored_outcome_only(x_github_event, x_github_delivery)

        if x_github_event == "workflow_run":
            # `workflow_run` is not a member of coding_client's
            # GitHubEventTypeEnum (an external package this repo does not own),
            # so it cannot flow through SecureWebhookHandler.handle_webhook()'s
            # enum-gated dispatch -- that call 400s "Unsupported event type" for
            # it, AFTER verifying the signature. Verification is reused
            # directly here via the public `verify_security` wrapper (same
            # method, same configured secret, same constant-time compare, same
            # fail-closed-on-empty-secret behaviour) so a workflow_run delivery
            # gets IDENTICAL authentication to every other event; only the
            # enum-based routing is bypassed.
            return await _handle_workflow_run_webhook(
                request=request,
                webhook_handler=webhook_handler,
                payload=payload,
                signature=x_hub_signature_256,
                delivery_id=x_github_delivery,
                already_verified=verified,
            )

        result = await webhook_handler.handle_webhook(
            request=request,
            payload=payload,
            event_type=x_github_event,
            signature=x_hub_signature_256,
            delivery_id=x_github_delivery,
        )
        return GitHubWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="Webhook processed successfully",
            event_type=result.event_type,
            action=result.action,
            processed=result.processed,
            processing_time_ms=result.processing_time_ms,
            delivery_id=x_github_delivery,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise internal_error(f"Webhook processing failed: {e}")


#: Events outcome_only otherwise ignores that now trigger a CI-wait evaluation.
_CI_WAIT_ONLY_EVENTS = CI_WAIT_EVENTS - {"pull_request", "workflow_run"}


def _ci_wait_trigger_only(event: str, delivery_id: str | None) -> JSONResponse:
    """202 for a verified check/status delivery used ONLY as a CI-wait trigger."""
    logger.info("CI-wait trigger %s webhook (delivery: %s)", event, delivery_id)
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "reason": "ci_wait_trigger"},
    )


def _ignored_outcome_only(event: str, delivery_id: str | None) -> JSONResponse:
    """202 for an authenticated delivery filtered out by webhook_mode=outcome_only."""
    logger.info("Ignored %s webhook (delivery: %s): webhook_mode=outcome_only", event, delivery_id)
    return JSONResponse(
        status_code=202,
        content={"status": "ignored", "reason": "webhook_mode=outcome_only"},
    )


async def _handle_pull_request_outcome_only(
    *, payload: bytes, delivery_id: str | None
) -> GitHubWebhookResponse | JSONResponse:
    """outcome_only: forward a verified ``pull_request`` ``closed``; ignore the rest."""
    from services.outcome_forward import build_pull_request_outcome, forward_outcome_event

    start_time = time.time()
    try:
        event_data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}") from e
    action = event_data.get("action")
    if action != "closed":
        return _ignored_outcome_only("pull_request", delivery_id)
    outcome = build_pull_request_outcome(event_data, delivery_id)
    processed = False
    if outcome is not None:
        asyncio.create_task(forward_outcome_event(outcome))
        processed = True
    return GitHubWebhookResponse(
        status=ResponseStatus.SUCCESS,
        message="Webhook processed successfully",
        event_type="pull_request",
        action=action,
        processed=processed,
        processing_time_ms=(time.time() - start_time) * 1000,
        delivery_id=delivery_id,
    )


async def _handle_workflow_run_webhook(
    *,
    request: Request,
    webhook_handler: SecureWebhookHandler,
    payload: bytes,
    signature: str | None,
    delivery_id: str | None,
    already_verified: bool = False,
) -> GitHubWebhookResponse:
    """Verify + handle a ``workflow_run`` delivery outside the enum-gated dispatch.

    ``action == "completed"`` on the repository's DEFAULT branch forwards a
    ``ci_completed`` outcome event to n8n (see ``services.outcome_forward``),
    replacing the "AA Observability - GitHub Outcome Poll" 30-minute timer for
    post-merge CI results. Every other action is verified and acknowledged
    (``processed=False``) and dropped, matching how the enum-gated path treats
    an event type with no registered interest.

    The forward runs as a background task: GitHub's response does not wait on
    it, so n8n being down (or slow) never turns a verified delivery into
    anything but a 2xx.
    """
    from services.outcome_forward import build_workflow_run_outcome, forward_outcome_event

    start_time = time.time()
    if not already_verified:
        await webhook_handler.verify_security(request, payload, signature)
    try:
        event_data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}") from e

    action = event_data.get("action")
    processed = False
    if action == "completed":
        outcome = build_workflow_run_outcome(event_data, delivery_id)
        if outcome is not None:
            asyncio.create_task(forward_outcome_event(outcome))
            processed = True

    processing_time_ms = (time.time() - start_time) * 1000
    logger.info(
        "Processed workflow_run webhook (action: %s, delivery: %s) in %.2fms",
        action,
        delivery_id,
        processing_time_ms,
    )
    return GitHubWebhookResponse(
        status=ResponseStatus.SUCCESS,
        message="Webhook processed successfully",
        event_type="workflow_run",
        action=action,
        processed=processed,
        processing_time_ms=processing_time_ms,
        delivery_id=delivery_id,
    )


async def handle_pull_request_event(payload: dict[str, Any]) -> None:
    """Handle pull request events with hybrid validation."""
    action = payload.get("action")
    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {})
    repo_name = repo.get("full_name")
    pr_number = pr.get("number")
    logger.info("PR event: %s on PR #%s in %s", action, pr_number, repo_name)

    if action in ["opened", "synchronize"]:
        try:
            # Use hybrid validation: fast check + GitHub Actions
            from github import Github

            from config.settings import get_settings
            from services.pr_validation.pr_validation_orchestrator import (
                get_pr_validation_orchestrator,
            )

            settings = get_settings()
            token = _webhook_github_token(repo_name, "pr_validation")
            if not token:
                return

            # Get PR details
            github_client = Github(token)
            full_repo = github_client.get_repo(repo_name)
            github_pr = full_repo.get_pull(pr_number)
            pr_files = list(github_pr.get_files())

            # Extract repo owner and name
            repo_parts = repo_name.split("/")
            if len(repo_parts) != 2:
                logger.error(f"Invalid repo name format: {repo_name}")
                return

            repo_owner, repo_name_only = repo_parts

            # Prepare PR metadata
            pr_metadata = {
                "number": pr_number,
                "title": github_pr.title,
                "body": github_pr.body,
                "user": {"login": github_pr.user.login if github_pr.user else "unknown"},
                "base": {"sha": github_pr.base.sha, "ref": github_pr.base.ref},
                "head": {"sha": github_pr.head.sha, "ref": github_pr.head.ref},
                "created_at": (github_pr.created_at.isoformat() if github_pr.created_at else None),
                "updated_at": (github_pr.updated_at.isoformat() if github_pr.updated_at else None),
            }

            # Trigger hybrid validation
            orchestrator = get_pr_validation_orchestrator()
            result = await orchestrator.validate_pr_with_actions(
                repo_owner=repo_owner,
                repo_name=repo_name_only,
                pr_number=pr_number,
                pr_files=pr_files,
                pr_metadata=pr_metadata,
                trigger_comprehensive=True,  # Enable GitHub Actions validation
            )

            # Post fast validation results to PR
            validation_id = result.get("validation_id")
            fast_val = result.get("fast_validation", {})
            comp_val = result.get("comprehensive_validation", {})

            comment = "## PR Validation Results\n\n"
            comment += f"**Validation ID:** `{validation_id}`\n\n"

            # Fast validation results
            comment += "### Fast Validation\n"
            comment += f"- **Status:** {'Passed' if fast_val.get('passed') else 'Failed'}\n"
            comment += f"- **Score:** {fast_val.get('score', 0):.1f}/100\n\n"

            if fast_val.get("results"):
                for val in fast_val["results"]:
                    status = "PASS" if val.passed else "FAIL"
                    comment += f"- {status} **{val.validation_type}**: {val.score:.1f}/100\n"

            # Comprehensive validation status
            comment += "\n### Comprehensive Validation\n"
            comp_status = comp_val.get("status", "not_triggered")

            if comp_status == "triggered":
                comment += "- **Status:** In Progress\n"
                comment += f"- **Message:** {comp_val.get('message', 'Running comprehensive analysis...')}\n"
                comment += "\nComprehensive validation is running via GitHub Actions. "
                comment += "Results will be posted when complete.\n"
            elif comp_status == "not_triggered":
                reason = comp_val.get("reason", "unknown")
                comment += "- **Status:** Skipped\n"
                comment += f"- **Reason:** {reason}\n"
            else:
                comment += f"- **Status:** {comp_status}\n"

            gateway_base_url = settings.coding_gateway_url.rstrip("/")
            comment += (
                "\n---\n_Powered by Code Graph Helper_ "
                f"[View Status]({gateway_base_url}/api/validation/{validation_id})"
            )

            # Post comment
            github_pr.create_issue_comment(comment)

            logger.info(f"[{validation_id}] Posted validation results to PR #{pr_number}")

        except (ValueError, RuntimeError, ConnectionError) as e:
            logger.error("Failed to validate PR #%s: %s", pr_number, e)
        except Exception as e:
            logger.error("Unexpected error validating PR #%s: %s", pr_number, e, exc_info=True)

    elif action == "closed" and pr.get("merged"):
        logger.info("PR #%s was merged in %s", pr_number, repo_name)


async def handle_review_comment_event(payload: dict[str, Any]) -> None:
    """Handle PR review comments."""
    comment = payload.get("comment", {})
    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {})
    repo_name = repo.get("full_name")
    pr_number = pr.get("number")
    comment_body = comment.get("body", "")
    logger.info(
        "Review comment on PR #%s in %s: %s...",
        pr_number,
        repo_name,
        comment_body[:100],
    )
    if any(trigger in comment_body.lower() for trigger in ["@codehelper", "/analyze", "/review"]):
        try:
            await analyze_pull_request_changes(repo_name, pr_number, "comment_trigger")
        except (ValueError, RuntimeError, ConnectionError) as e:
            logger.error("Failed to analyze PR #%s from comment: %s", pr_number, e)


async def handle_push_event(payload: dict[str, Any]) -> None:
    """Handle push events."""
    ref = payload.get("ref", "")
    commits = payload.get("commits", [])
    repo = payload.get("repository", {})
    repo_name = repo.get("full_name")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    logger.info("Push event to %s in %s with %s commits", branch, repo_name, len(commits))
    if branch in ["main", "master", "develop"]:
        try:
            await analyze_repository_changes(repo_name, branch, commits)
        except (ValueError, RuntimeError, ConnectionError) as e:
            logger.error("Failed to analyze push to %s/%s: %s", repo_name, branch, e)


async def handle_issue_comment_event(payload: dict[str, Any]) -> None:
    """Handle issue/PR comments."""
    issue = payload.get("issue", {})
    comment = payload.get("comment", {})
    repo = payload.get("repository", {})
    repo_name = repo.get("full_name")
    comment_body = comment.get("body", "")
    if "pull_request" in issue:
        pr_number = issue.get("number")
        logger.info("PR comment on #%s in %s: %s...", pr_number, repo_name, comment_body[:100])
        if any(trigger in comment_body.lower() for trigger in ["@codehelper", "/analyze", "/help"]):
            try:
                await analyze_pull_request_changes(repo_name, pr_number, "comment_trigger")
            except (ValueError, RuntimeError, ConnectionError) as e:
                logger.error("Failed to analyze PR #%s from comment: %s", pr_number, e)


def _generate_pr_analysis_summary(analysis_result: PRAnalysisDict, trigger: str) -> str:
    """Generate markdown summary for PR analysis results."""
    summary_parts = [
        "## Code Analysis Results",
        "",
        f"**Trigger:** {trigger}",
        f"**Analyzed at:** {datetime.now(UTC).isoformat()}",
        "",
        "### Statistics",
        f"- **Total files:** {analysis_result['statistics']['total']}",
        f"- **Added:** {analysis_result['statistics']['added']}",
        f"- **Modified:** {analysis_result['statistics']['modified']}",
        f"- **Deleted:** {analysis_result['statistics']['removed']}",
        f"- **Total additions:** +{analysis_result['total_additions']}",
        f"- **Total deletions:** -{analysis_result['total_deletions']}",
        "",
    ]
    if analysis_result["languages"]:
        summary_parts.extend(["### Languages", ""])
        for language, stats in analysis_result["languages"].items():
            summary_parts.append(
                f"- **{language}:** {stats['files']} files (+{stats['additions']}/-{stats['deletions']})"
            )
        summary_parts.append("")
    if analysis_result["statistics"]["total"] <= 10:
        summary_parts.extend(["### Changed Files", ""])
        for file in analysis_result["files"]:
            emoji = {"added": "+", "modified": "~", "removed": "-"}.get(file.change_type, "?")
            summary_parts.append(f"- {emoji} `{file.file_path}`")
        summary_parts.append("")
    summary_parts.extend(["---", "*Analysis provided by CodeGraph Helper*"])
    return "\n".join(summary_parts)


async def _post_pr_comment(repo_name: str, pr_number: int, comment: str) -> None:
    """Post a comment to a pull request."""
    token = _webhook_github_token(repo_name, "pr_comment")
    if token:
        config = GitHubConfig(token_id="webhook", token=token)
        async with GitHubService(config) as github_service:
            await github_service.add_pull_request_comment(
                repo_name=repo_name, pr_number=pr_number, comment=comment
            )
        logger.info("Posted analysis comment to PR #%s in %s", pr_number, repo_name)
    else:
        logger.warning("No GitHub credential for webhook comments, cannot post comment")


async def analyze_pull_request_changes(repo_name: str, pr_number: int, trigger: str) -> None:
    """Analyze changes in a pull request and provide feedback.

    ⛔ Needs a GitHub credential to read the PR and post, and a webhook has none
    (no PAT exists; no tenant for the App), so it is skipped before GitHub is
    touched (reachable from both the PR webhook and the review-comment trigger;
    neither carries an fr tenant).
    """
    if not _webhook_github_token(repo_name, "pr_analysis"):
        return
    try:
        github_hub = GitHubIntegrationHub()
        repo_url = f"https://github.com/{repo_name}"
        analysis_result = await github_hub.analyze_pull_request(repo_url, pr_number)
        comment_body = _generate_pr_analysis_summary(analysis_result, trigger)
        await _post_pr_comment(repo_name, pr_number, comment_body)
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Error analyzing PR #%s in %s: %s", pr_number, repo_name, e)
        error_comment = f"**Analysis Error**\n\nFailed to analyze changes: {str(e)}"
        try:
            await _post_pr_comment(repo_name, pr_number, error_comment)
        except (ValueError, RuntimeError, ConnectionError) as post_error:
            logger.error("Failed to post error comment: %s", post_error)


async def _analyze_single_commit(
    github_service: "GitHubService", repo_name: str, commit: dict[str, Any]
) -> dict[str, Any] | None:
    """Analyze a single commit and return its analysis data."""
    try:
        commit_sha = commit.get("id", "")
        author = commit.get("author", {}).get("name", "Unknown")
        message = commit.get("message", "")
        timestamp = commit.get("timestamp", "")
        changed_files = await github_service.get_commit_files(repo_name, commit_sha)
        file_changes = []
        for file_info in changed_files:
            filename = file_info.get("filename", "")
            status = file_info.get("status", "")
            additions = file_info.get("additions", 0)
            deletions = file_info.get("deletions", 0)
            file_changes.append(
                {
                    "filename": filename,
                    "status": status,
                    "additions": additions,
                    "deletions": deletions,
                }
            )
        return {
            "sha": commit_sha,
            "author": author,
            "message": message[:100],
            "timestamp": timestamp,
            "files_changed": len(file_changes),
            "total_additions": sum(f["additions"] for f in file_changes),
            "total_deletions": sum(f["deletions"] for f in file_changes),
            "file_changes": file_changes,
        }
    except (ValueError, RuntimeError, ConnectionError) as commit_error:
        logger.warning("Could not analyze commit %s: %s", commit_sha, commit_error)
        return None


async def _update_code_graph(
    github_hub: "GitHubIntegrationHub",
    repo_name: str,
    branch: str,
    changed_files: list[Any],
    commit_shas: list[Any],
    tenant_id: str | None = None,
) -> None:
    """Ask coding-analysis to refresh the repo's code graph.

    This service does not update a graph itself — see
    ``services.github.webhook_handler``. The statuses below are the ones that
    handler now distinguishes; the old ``"success"`` was returned by a sqlite
    path that could not run at all.
    """
    try:
        update_result = await github_hub.handle_repository_update(
            repo_name=repo_name,
            branch=branch,
            changed_files=changed_files,
            commit_shas=commit_shas,
            tenant_id=tenant_id,
        )
        status = update_result["status"]
        if status == "scan_requested":
            logger.info(
                "Code-graph re-scan requested for %s (%d changed files); "
                "coding-analysis runs it in the background",
                repo_name,
                len(changed_files),
            )
        elif status == "no_container":
            logger.warning(
                "No container for %s — no code graph to refresh (nothing was updated)",
                repo_name,
            )
        else:
            logger.error(
                "Code-graph refresh for %s failed: %s",
                repo_name,
                update_result.get("update_stats", {}).get("error"),
            )
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Failed to request a code-graph refresh for %s: %s", repo_name, e)


def _get_repository_summary(commits_data: list[Any], changed_files: set[Any]) -> dict[str, Any]:
    """Calculate repository analysis summary statistics."""
    total_files = len(changed_files)
    total_commits = len(commits_data)
    total_additions = sum(c["total_additions"] for c in commits_data)
    total_deletions = sum(c["total_deletions"] for c in commits_data)
    return {
        "total_files": total_files,
        "total_commits": total_commits,
        "total_additions": total_additions,
        "total_deletions": total_deletions,
    }


async def _process_commits_analysis(
    github_service: GitHubService, repo_name: str, commits: list[Any]
) -> tuple[Any, ...]:
    """Process commits and return analysis data and changed files.

    Only the first ``_MAX_COMMITS_ANALYZED`` commits of the push are analysed; a
    larger push is truncated (and logged) rather than fully analysed inline.
    """
    all_changed_files = set[Any]()
    commit_analysis = []
    if len(commits) > _MAX_COMMITS_ANALYZED:
        logger.warning(
            "Push to %s has %d commits; analysing only the first %d (truncated)",
            repo_name,
            len(commits),
            _MAX_COMMITS_ANALYZED,
        )
    for commit in commits[:_MAX_COMMITS_ANALYZED]:
        analysis = await _analyze_single_commit(github_service, repo_name, commit)
        if analysis:
            commit_analysis.append(analysis)
            for file_change in analysis["file_changes"]:
                all_changed_files.add(file_change["filename"])
    return (commit_analysis, all_changed_files)


async def analyze_repository_changes(
    repo_name: str, branch: str, commits: list[Any]
) -> dict[str, Any]:
    """Analyze changes from push events."""
    try:
        logger.info("Analyzing %s commits pushed to %s/%s", len(commits), repo_name, branch)
        token = _webhook_github_token(repo_name, "push_analysis")
        if not token:
            return {"error": "GitHub token not configured"}
        github_config = GitHubConfig(token_id="webhook", token=token)
        github_service = GitHubService(github_config)
        github_hub = GitHubIntegrationHub()
        commit_analysis, all_changed_files = await _process_commits_analysis(
            github_service, repo_name, commits
        )
        await _update_code_graph(
            github_hub,
            repo_name,
            branch,
            list[Any](all_changed_files),
            [c["sha"] for c in commit_analysis],
        )
        try:
            issues_found = await _analyze_files_for_issues(list[Any](all_changed_files))
            if issues_found:
                logger.warning("Found %s potential issues in changed files", len(issues_found))
                critical_issues = [
                    issue for issue in issues_found if issue.get("severity") == "critical"
                ]
                if critical_issues:
                    await _send_issue_notifications(repo_name, branch, critical_issues)
        except (ValueError, RuntimeError, ConnectionError) as issue_error:
            logger.error("Failed to analyze files for issues: %s", issue_error)
        summary = _get_repository_summary(commit_analysis, all_changed_files)
        logger.info(
            "Repository analysis complete for %s/%s: %s commits, %s files changed, +%s/-%s lines",
            repo_name,
            branch,
            summary["total_commits"],
            summary["total_files"],
            summary["total_additions"],
            summary["total_deletions"],
        )
        return {
            "repository": repo_name,
            "branch": branch,
            "commits_analyzed": summary["total_commits"],
            "files_changed": summary["total_files"],
            "total_additions": summary["total_additions"],
            "total_deletions": summary["total_deletions"],
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Error analyzing repository changes: %s", e)
        return {"error": str(e), "repository": repo_name, "branch": branch}


def _get_code_files(changed_files: list[Any]) -> list[Any]:
    """Filter files to only include code files for analysis."""
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".java",
        ".cpp",
        ".c",
        ".h",
    }
    return [f for f in changed_files if any(f.endswith(ext) for ext in code_extensions)]


async def _analyze_security_vulnerabilities(
    code_files: list[Any],
) -> list[dict[str, Any]]:
    """Analyze code files for security vulnerabilities via coding-security service.

    Only the first ``_MAX_FILES_SCANNED`` changed code files are scanned; a larger
    changeset is truncated (and logged) rather than scanned in full inline.

    Sends ``X-Tenant-Id`` when one is configured. Without it this call was
    reaching coding-security only via that service's trusted-network fallback
    (any RFC1918 source IP, which covers the whole pod CIDR) — a network
    position, not a credential, and one that grants exactly the
    ``DEFAULT_TENANT_ID`` context named explicitly here. The push-event chain
    (``analyze_repository_changes`` -> ``_analyze_files_for_issues`` -> here)
    carries no per-request tenant, so this is the service's own identity; the
    scan is over file paths, not tenant-scoped records. Mirrors
    ``services/github_provisioning._scan_headers``.

    ⚠️ ``DEFAULT_TENANT_ID`` is ``str | None`` as of coding-client 1.9.0 (#36),
    which removed the built-in ``"n8n-dev"`` fallback. This service does not set
    the variable on its Deployment, so unset is the live case and the header
    value would otherwise be a literal ``None``. Unlike ``_scan_headers`` this
    site is not type-checked into failing — the dict is handed to a client that
    accepts a loose mapping — so it has to be guarded explicitly.

    The call also carries this service's own licence in ``Authorization`` so it
    survives coding-security running with ``TRUST_TENANT_HEADER=false``. That
    matters more here than anywhere else in this module: the whole body is
    wrapped in ``except Exception -> return []``, so a refused scan is reported
    to the push-event pipeline as **"no vulnerabilities found"**. Dropping the
    credential would not break a push event; it would make every push event lie.
    """
    try:
        from main import get_security_client

        if len(code_files) > _MAX_FILES_SCANNED:
            logger.warning(
                "Changeset has %d code files; scanning only the first %d (truncated)",
                len(code_files),
                _MAX_FILES_SCANNED,
            )

        if DEFAULT_TENANT_ID is None:
            logger.warning(
                "DEFAULT_TENANT_ID is unset; sending no %s on the push-event "
                "vulnerability scan. coding-security will reject it. Set "
                "DEFAULT_TENANT_ID on the Deployment if this scan should run.",
                TENANT_ID_HEADER,
            )
        scan_headers = optional_sibling_headers(DEFAULT_TENANT_ID)

        client = get_security_client()
        result = await client.analyze_vulnerabilities(
            file_paths=[str(f) for f in code_files[:_MAX_FILES_SCANNED]],
            max_files=_MAX_FILES_SCANNED,
            headers=scan_headers,
        )
        issues: list[dict[str, Any]] = []
        if result and isinstance(result, dict) and "vulnerabilities_by_type" in result:
            for vuln_type, vulns in result["vulnerabilities_by_type"].items():
                for vuln in vulns:
                    issues.append(
                        {
                            "type": "security",
                            "severity": vuln.get("severity", "medium"),
                            "file": vuln.get("file_path", "unknown"),
                            "message": vuln.get("description", "Security vulnerability detected"),
                            "line": vuln.get("line_start", 0),
                            "tool": "security_detector",
                        }
                    )
        return issues
    except Exception as e:
        logger.debug("Security vulnerability analysis not available: %s", e)
        return []


def _analyze_file_patterns(changed_files: list[Any]) -> list[dict[str, Any]]:
    """Analyze file patterns for potential issues."""
    issues = []
    sensitive_patterns = [
        ".env",
        "config.json",
        "secrets.json",
        "credentials",
        ".pem",
        ".key",
        ".p12",
        "password",
        "token",
    ]
    binary_extensions = (".zip", ".tar.gz", ".exe", ".dll", ".so", ".dylib")
    for file_path in changed_files:
        if any(pattern in file_path.lower() for pattern in sensitive_patterns):
            issues.append(
                {
                    "type": "sensitive_file",
                    "severity": "high",
                    "file": file_path,
                    "message": f"Potentially sensitive file modified: {file_path}",
                    "tool": "file_pattern_analysis",
                }
            )
        if file_path.endswith(binary_extensions):
            issues.append(
                {
                    "type": "binary_file",
                    "severity": "medium",
                    "file": file_path,
                    "message": f"Binary file committed: {file_path}",
                    "tool": "file_pattern_analysis",
                }
            )
    return issues


async def _analyze_files_for_issues(changed_files: list[Any]) -> list[Any]:
    """Analyze changed files for potential issues and security vulnerabilities."""
    try:
        code_files = _get_code_files(changed_files)
        issues = []
        if code_files:
            issues.extend(await _analyze_security_vulnerabilities(code_files))
        issues.extend(_analyze_file_patterns(changed_files))
        logger.info(
            "Found %s potential issues in %s changed files",
            len(issues),
            len(changed_files),
        )
        return issues
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Error analyzing files for issues: %s", e)
        return []


def _format_notification_message(critical_issues: list[Any], branch: str) -> str:
    """Format notification message for critical issues."""
    issue_summary = []
    for issue in critical_issues[:5]:
        file_name = issue.get("file", "unknown")
        message = issue.get("message", "Critical issue detected")
        issue_summary.append(f"- **{file_name}**: {message}")
    return f"\n**Critical Issues Detected**\n\nThe following critical issues were found in recent commits to `{branch}`:\n\n{chr(10).join(issue_summary)}\n\nPlease review and address these issues promptly.\n\n_This is an automated notification from the code analysis system._\n"


async def _try_create_github_issue(
    github_service: "GitHubService",
    repo_name: str,
    branch: str,
    notification_body: str,
    critical_issues: list[Any],
) -> bool:
    """Try to create a GitHub issue for critical problems."""
    try:
        issue_title = f"Critical Issues Detected in {branch} Branch"
        await github_service.create_issue(
            repo_name=repo_name,
            title=issue_title,
            body=notification_body,
            labels=["critical", "automated-analysis", "security"],
        )
        logger.info("Created issue notification for %s critical issues", len(critical_issues))
        return True
    except (ValueError, RuntimeError, ConnectionError) as issue_error:
        logger.warning("Could not create GitHub issue: %s", issue_error)
        return False


async def _try_fallback_notifications(
    github_service: "GitHubService",
    repo_name: str,
    branch: str,
    notification_body: str,
    critical_issues: list[Any],
) -> None:
    """Try fallback notification methods when GitHub issue creation fails."""
    try:
        recent_prs = await github_service.get_pull_requests(repo_name, state="open")
        if recent_prs:
            latest_pr = recent_prs[0]
            comment_body = f"**Critical Issues Detected**\n\n{notification_body}\n\n*Posted via automated analysis*"
            await github_service.create_pr_comment(
                repo_name=repo_name, pr_number=latest_pr["number"], comment=comment_body
            )
            logger.info("Posted critical issues comment on PR #%s", latest_pr["number"])
        else:
            logger.warning(
                "Critical issues notification for %s/%s",
                repo_name,
                branch,
                extra={
                    "repo_name": repo_name,
                    "branch": branch,
                    "critical_issues_count": len(critical_issues),
                    "issues": critical_issues[:3],
                    "notification_body": notification_body[:500],
                },
            )
            await _save_notification_to_file(repo_name, branch, critical_issues, notification_body)
    except (ValueError, RuntimeError, ConnectionError) as fallback_error:
        logger.error("All notification methods failed: %s", fallback_error)


async def _save_notification_to_file(
    repo_name: str, branch: str, critical_issues: list[Any], notification_body: str
) -> None:
    """Save notification to file as last resort."""
    try:
        notifications_dir = Path("/tmp/notifications")
        notifications_dir.mkdir(exist_ok=True)
        notification_file = (
            notifications_dir / f"{repo_name.replace('/', '_')}_{branch}_issues.json"
        )
        notification_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "repo_name": repo_name,
            "branch": branch,
            "critical_issues": critical_issues,
            "notification_body": notification_body,
        }
        notification_file.write_text(json.dumps(notification_data, indent=2))
        logger.info("Saved notification to %s", notification_file)
    except (OSError, ValueError, RuntimeError) as file_error:
        logger.debug("Could not save notification file: %s", file_error)


async def _send_issue_notifications(repo_name: str, branch: str, critical_issues: list[Any]):
    """Send notifications for critical issues found in commits."""
    try:
        token = _webhook_github_token(repo_name, "issue_notifications")
        if not token:
            return
        github_config = GitHubConfig(token_id="webhook", token=token)
        github_service = GitHubService(github_config)
        notification_body = _format_notification_message(critical_issues, branch)
        success = await _try_create_github_issue(
            github_service, repo_name, branch, notification_body, critical_issues
        )
        if not success:
            await _try_fallback_notifications(
                github_service, repo_name, branch, notification_body, critical_issues
            )
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Error sending issue notifications: %s", e)
