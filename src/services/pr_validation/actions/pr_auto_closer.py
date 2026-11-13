"""
PR Auto-Closer for validation failures.

This module provides functionality to automatically close or comment on
pull requests that fail validation based on configurable rules.
"""

import logging
from collections.abc import Coroutine
from typing import Any

from github import Github, GithubException
from github.PullRequest import PullRequest
from pydantic import BaseModel, Field

from schemas.github_api_models import ValidatePRResponse

logger = logging.getLogger(__name__)

# GitHub's 422 text when an identity reviews a PR it authored itself.
_OWN_PR_REVIEW_MARKER = "own pull request"

# ``PRActionOutcome.changes_request_degraded`` values: ``<why>:<what we did>``.
# ``self_authored_pr`` — the PR author is exactly the login that posted our
# validation comment; ``own_pr_422`` — GitHub refused REQUEST_CHANGES with 422
# "own pull request". Nothing else is ever presumed to be our own PR.
REASON_SELF_AUTHORED = "self_authored_pr"
REASON_OWN_PR_422 = "own_pr_422"
_SKIPPED_COMMENT_POSTED = "review_skipped_comment_already_posted"
_COMMENT_REVIEW_POSTED = "comment_review_posted"


class PRActionOutcome(BaseModel):
    """What the auto-closer ACTUALLY did to a pull request.

    Every flag is True only when the action was both *attempted* and *confirmed
    successful* by the GitHub API. A flag is False when the action was not
    requested by config, when the code path was not taken (e.g. a PR that got
    closed never has changes requested), or when the GitHub call raised.

    ``errors`` collects one human-readable entry per failure, including a fatal
    failure to reach the repository or pull request at all — in which case no
    action flag can possibly be True. Callers MUST treat a non-empty ``errors``
    as "this did not fully happen"; reporting success on it is a lie.
    """

    labels_added: bool = False
    comment_posted: bool = False
    changes_requested: bool = False
    pr_closed: bool = False
    # Set when a REQUEST_CHANGES review was due but the acting identity AUTHORED
    # the PR (an App reviewing its own conveyor PR): GitHub refuses that with a
    # 422, so the verdict went out as a comment instead. ``changes_requested``
    # stays False — no changes were requested — and this is not an error.
    changes_request_degraded: str | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """True only when nothing failed."""
        return not self.errors

    def action_flags(self) -> dict[str, Any]:
        """The action flags, for use as an ``actions_taken`` payload."""
        return {
            "labels_added": self.labels_added,
            "comment_posted": self.comment_posted,
            "changes_requested": self.changes_requested,
            "changes_request_degraded": self.changes_request_degraded,
            "pr_closed": self.pr_closed,
        }


class PRAutoCloser:
    """
    Automatically manages pull requests based on validation results.

    This class can:
    - Close PRs that fail critical validation
    - Add warning labels to PRs
    - Post detailed validation reports as comments
    - Request changes through PR reviews
    """

    def __init__(self, github_token: str, config: dict[str, Any] | None = None):
        """
        Initialize PR auto-closer with GitHub token and configuration.

        Args:
            github_token: GitHub personal access token with repo permissions
            config: Configuration for auto-close behavior. Merged OVER the
                defaults, so a partial config (the common case — callers of
                /process-validation send arbitrary JSON) no longer raises
                KeyError on the first missing key and silently no-ops.
        """
        self.github = Github(github_token)
        self.config = {**self._get_default_config(), **(config or {})}
        # Login of the identity this token acts as, learned from the author of
        # the comment we post (an installation token cannot call GET /user).
        self._acting_login: str | None = None

    def _get_default_config(self) -> dict[str, Any]:
        """Get default auto-close configuration."""
        return {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": True,
            "request_changes": True,
            "labels": {
                "critical": "validation-failed-critical",
                "high": "validation-failed-high",
                "medium": "validation-warning",
                "low": "validation-info",
            },
            "close_message_template": "\n## 🚫 Pull Request Auto-Closed Due to Critical Validation Failures\n\nThis pull request has been automatically closed because it failed critical validation checks.\n\n**Validation Score:** {score}/100\n**Risk Level:** {risk_level}\n**Recommendation:** {recommendation}\n\n### Critical Issues Found:\n{critical_issues}\n\n### Required Actions:\n1. Review and resolve the critical issues listed above\n2. Ensure all validation checks pass\n3. Create a new pull request with the resolves\n\n### Validation Details:\n{validation_summary}\n\n---\n*This action was performed automatically by the PR validation system.*\n",
            "warning_message_template": "\n## ⚠️ Pull Request Validation Warning\n\nThis pull request has validation issues that should be addressed.\n\n**Validation Score:** {score}/100\n**Risk Level:** {risk_level}\n**Recommendation:** {recommendation}\n\n### Issues Found:\n{issues_summary}\n\n### Suggestions:\n{suggestions}\n\n---\n*Please address these issues before merging.*\n",
        }

    async def _attempt(
        self,
        outcome: PRActionOutcome,
        error_prefix: str,
        coro: Coroutine[Any, Any, None],
    ) -> bool:
        """Await one PR action, returning whether it actually succeeded.

        On failure the error is logged AND recorded on ``outcome`` so it can
        reach the caller. Swallowing it here without recording it is what let a
        total no-op be reported as a success.
        """
        try:
            await coro
            return True
        except Exception as e:
            message = f"{error_prefix}: {e}"
            logger.error(message)
            outcome.errors.append(message)
            return False

    async def process_validation_result(
        self,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        validation_response: ValidatePRResponse,
    ) -> PRActionOutcome:
        """
        Process validation results and take appropriate actions.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            pr_number: Pull request number
            validation_response: Validation response from the API

        Returns:
            PRActionOutcome: exactly which actions actually succeeded, plus any
            errors. This method never raises; the caller is responsible for
            turning a non-empty ``errors`` into a visible failure rather than
            reporting success.
        """
        outcome = PRActionOutcome()
        try:
            repo = self.github.get_repo(f"{repo_owner}/{repo_name}")
            pr = repo.get_pull(pr_number)
            should_close = self._should_close_pr(validation_response)
        except Exception as e:
            # Fatal: we never reached the PR, so no action can have happened.
            message = f"Error processing validation result for PR #{pr_number}: {e}"
            logger.error(message)
            outcome.errors.append(message)
            return outcome

        if self.config["add_labels"]:
            outcome.labels_added = await self._attempt(
                outcome,
                "Error adding labels to PR",
                self._add_validation_labels(pr, validation_response),
            )
        if self.config["post_comment"]:
            outcome.comment_posted = await self._attempt(
                outcome,
                "Error posting comment on PR",
                self._post_validation_comment(pr, validation_response, should_close),
            )
        if should_close and self.config["auto_close_enabled"]:
            outcome.pr_closed = await self._attempt(
                outcome,
                "Error closing PR",
                self._close_pr(pr, validation_response),
            )
        elif self.config["request_changes"] and validation_response.recommendation == "reject":
            await self._request_changes_or_degrade(outcome, pr, validation_response)
        return outcome

    def _is_self_authored(self, pr: PullRequest) -> bool:
        """True when the acting identity authored ``pr`` (GitHub would 422 a
        REQUEST_CHANGES review).

        Only an exact match against the login that authored our own validation
        comment counts. With no comment posted the acting login is unknown and
        nothing is presumed (a ``[bot]`` author may be dependabot, not us); the
        REQUEST_CHANGES attempt's 422 "own pull request" handles that case.
        """
        author_login = getattr(getattr(pr, "user", None), "login", None)
        if not isinstance(author_login, str) or not isinstance(self._acting_login, str):
            return False
        return author_login.lower() == self._acting_login.lower()

    async def _request_changes_or_degrade(
        self,
        outcome: PRActionOutcome,
        pr: PullRequest,
        validation_response: ValidatePRResponse,
    ) -> None:
        """Request changes — or, on the acting identity's OWN PR, degrade.

        GitHub refuses REQUEST_CHANGES on your own PR (422 "Can not request
        changes on your own pull request"). That is a property of who authored
        the PR, not a failure: the verdict still reaches the PR as a comment
        (the validation comment already posted, else a COMMENT review with the
        same body), and ``changes_request_degraded`` says so.
        """
        if self._is_self_authored(pr):
            await self._degrade_self_review(outcome, pr, validation_response, REASON_SELF_AUTHORED)
            return
        try:
            await self._request_changes(pr, validation_response)
            outcome.changes_requested = True
        except GithubException as e:
            if e.status == 422 and _OWN_PR_REVIEW_MARKER in str(e).lower():
                logger.warning(
                    "REQUEST_CHANGES refused on own PR #%s (%s); degrading", pr.number, e
                )
                await self._degrade_self_review(outcome, pr, validation_response, REASON_OWN_PR_422)
                return
            self._record_error(outcome, "Error requesting changes on PR", e)
        except Exception as e:
            self._record_error(outcome, "Error requesting changes on PR", e)

    async def _degrade_self_review(
        self,
        outcome: PRActionOutcome,
        pr: PullRequest,
        validation_response: ValidatePRResponse,
        reason: str,
    ) -> None:
        if outcome.comment_posted:
            outcome.changes_request_degraded = f"{reason}:{_SKIPPED_COMMENT_POSTED}"
            logger.info(
                "PR #%s is own PR (%s); review skipped (validation comment already posted)",
                pr.number,
                reason,
            )
            return
        if await self._attempt(
            outcome,
            "Error posting comment review on self-authored PR",
            self._submit_review(pr, validation_response, event="COMMENT"),
        ):
            outcome.changes_request_degraded = f"{reason}:{_COMMENT_REVIEW_POSTED}"

    @staticmethod
    def _record_error(outcome: PRActionOutcome, error_prefix: str, e: Exception) -> None:
        message = f"{error_prefix}: {e}"
        logger.error(message)
        outcome.errors.append(message)

    def _should_close_pr(self, validation_response: ValidatePRResponse) -> bool:
        """Determine if PR should be closed based on validation results."""
        if validation_response.recommendation == "reject":
            critical_issues = 0
            for validation in validation_response.validations:
                critical_issues += len([i for i in validation.issues if i.severity == "critical"])
            if critical_issues > 0:
                return True
        if validation_response.overall_score < self.config["close_on_score_below"]:
            return True
        return False

    async def _add_validation_labels(
        self, pr: PullRequest, validation_response: ValidatePRResponse
    ) -> None:
        """Add appropriate labels to the PR based on validation results.

        Raises on GitHub failure so the caller can report the truth; see
        ``_attempt``.
        """
        existing_labels = [label.name for label in pr.labels]
        validation_labels = list(self.config["labels"].values())
        labels_to_remove = [lbl for lbl in existing_labels if lbl in validation_labels]
        if labels_to_remove:
            for label in labels_to_remove:
                pr.remove_from_labels(label)
        if validation_response.recommendation == "reject":
            label = self.config["labels"]["critical"]
        elif validation_response.recommendation == "needs_work":
            label = self.config["labels"]["high"]
        else:
            label = self.config["labels"]["low"]
        pr.add_to_labels(label)
        logger.info(f"Added label '{label}' to PR #{pr.number}")

    async def _post_validation_comment(
        self,
        pr: PullRequest,
        validation_response: ValidatePRResponse,
        should_close: bool,
    ) -> None:
        """Post a detailed validation comment on the PR.

        Raises on GitHub failure so the caller can report the truth; see
        ``_attempt``.
        """
        issues_by_severity: dict[str, list[str]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }
        all_suggestions = []
        for validation in validation_response.validations:
            for issue in validation.issues:
                issues_by_severity[issue.severity].append(
                    f"- **{issue.file}:{issue.line}** - {issue.message}"
                )
            all_suggestions.extend(validation.suggestions)
        critical_issues = "\n".join(issues_by_severity["critical"][:5])
        issues_summary = []
        for severity, issues in issues_by_severity.items():
            if issues:
                issues_summary.append(f"**{severity.title()}:** {len(issues)} issues")
        template = (
            self.config["close_message_template"]
            if should_close
            else self.config["warning_message_template"]
        )
        comment = template.format(
            score=validation_response.overall_score,
            risk_level=validation_response.recommendation.upper(),
            recommendation=validation_response.recommendation,
            critical_issues=critical_issues or "No critical issues",
            issues_summary="\n".join(issues_summary),
            suggestions="\n".join(f"- {s}" for s in all_suggestions[:5]),
            validation_summary=self._format_validation_summary(validation_response),
        )
        posted = pr.create_issue_comment(comment)
        acting_login = getattr(getattr(posted, "user", None), "login", None)
        if isinstance(acting_login, str):
            self._acting_login = acting_login
        logger.info(f"Posted validation comment on PR #{pr.number}")

    async def _close_pr(self, pr: PullRequest, validation_response: ValidatePRResponse) -> None:
        """Close the pull request.

        Raises on GitHub failure so the caller can report the truth; see
        ``_attempt``.
        """
        pr.edit(state="closed")
        logger.info(f"Closed PR #{pr.number} due to validation failures")

    async def _request_changes(
        self, pr: PullRequest, validation_response: ValidatePRResponse
    ) -> None:
        """Request changes on the PR through GitHub's review system.

        Raises on GitHub failure so the caller can report the truth; see
        ``_attempt``.
        """
        await self._submit_review(pr, validation_response, event="REQUEST_CHANGES")
        logger.info(f"Requested changes on PR #{pr.number}")

    async def _submit_review(
        self, pr: PullRequest, validation_response: ValidatePRResponse, *, event: str
    ) -> None:
        """Submit the validation review with ``event``; raises on GitHub failure."""
        commits = pr.get_commits()
        latest_commit = list(commits)[-1]
        # NB: this was a plain string with an un-substituted `{...}` placeholder,
        # so every review body literally read "{validation_response.overall_score}".
        review_body = (
            "\n## Changes Requested\n\n"
            "The automated validation has found issues that must be addressed:\n\n"
            f"**Validation Score:** {validation_response.overall_score}/100\n\n"
            "Please resolve the identified issues and update the pull request.\n"
        )
        pr.create_review(commit=latest_commit, body=review_body, event=event)

    def _format_validation_summary(self, validation_response: ValidatePRResponse) -> str:
        """Format a summary of validation results."""
        summary_lines = []
        for validation in validation_response.validations:
            status = "✅" if validation.passed else "❌"
            summary_lines.append(
                f"{status} **{validation.validation_type}**: Score {validation.score}/100 - {len(validation.issues)} issues"
            )
        return "\n".join(summary_lines)


async def auto_process_pr_validation(
    repo_owner: str,
    repo_name: str,
    pr_number: int,
    github_token: str,
    validation_response: ValidatePRResponse,
    config: dict[str, Any] | None = None,
) -> PRActionOutcome:
    """
    Convenience function to automatically process PR validation results.

    This can be called from:
    - GitHub Actions after validation
    - Webhook handlers
    - CI/CD pipelines

    Returns:
        PRActionOutcome: which actions actually succeeded, plus any errors.
    """
    auto_closer = PRAutoCloser(github_token, config)
    return await auto_closer.process_validation_result(
        repo_owner, repo_name, pr_number, validation_response
    )
