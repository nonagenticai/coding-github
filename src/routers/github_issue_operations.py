"""
GitHub Issue Operations Endpoints

Create, list, read, comment on, label and close GitHub issues.

WHY THIS MODULE EXISTS
----------------------
Measured 2026-08-26: the coding API exposed **20 routes** — repositories, pull requests,
forks, file operations, webhooks — and **not one issue route**, while
`GitHubService.create_issue` had existed all along, reachable only from the webhook
handler's critical-analysis notification path. Issues are how work is described before it
becomes a branch, so the gap sat directly under the delivery pipeline.

Route naming follows the verb-kebab idiom already used by the PR module
(`/create-pull-request`, `/mark-pull-request-ready`), and every route declares an explicit
`operation_id` because the generated `@nonagentic/n8n-nodes-coding` node derives its
operation names from exactly that field.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import bad_request, internal_error, not_found
from fastapi import APIRouter, Depends, HTTPException

from schemas.github_api_models import (
    AddIssueLabelsRequest,
    CloseIssueRequest,
    CommentIssueRequest,
    CreateIssueRequest,
    GetIssueRequest,
    IssueCommentResponse,
    IssueResponse,
    ListIssuesRequest,
    ListIssuesResponse,
)
from services.andon import require_andon_released
from services.github.github_service import GitHubConfig, GitHubService
from services.github_app_auth import GitHubAppMintError
from services.github_integration import GitHubAuthError, resolve_github_token
from services.tenant_binding import (
    PERMS_ISSUES,
    PERMS_PR_ACTIONS,
    TenantRepoTokenError,
    github_auth_http_exception,
    tenant_error_http_exception,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@asynccontextmanager
async def _github(
    owner: str | None = None,
    repo: str | None = None,
    tenant_id: str | None = None,
    permissions: dict[str, str] | None = None,
) -> AsyncIterator[GitHubService]:
    """Open a GitHubService on the App-first credential ladder.

    This used to read the platform PAT setting directly (since removed, 2026-09-30).
    That env var was EMPTY in production, so every route
    here -- merge, PR checks and the five issue operations -- answered 400 "No GitHub
    access token configured" without ever reaching GitHub. Meanwhile the same service
    already authenticates the PR-CREATE flow through
    :func:`services.github_integration.resolve_github_token`, which mints a GitHub App
    installation token. The credential was mounted and correct the whole time; only this
    helper was looking in the wrong place.

    ``owner``/``repo`` matter: the ladder mints from the App's installation on THAT owner,
    and for a platform-owned repo the App is MANDATORY -- it raises ``GitHubAuthError``
    naming the failed rung rather than silently downgrading to a PAT that cannot write.
    Callers surface that as a 412.

    ⛔ For a CUSTOMER owner (not ``system_github_owner``) the credential is the
    tenant-bound public-App token scoped to ``repo`` with ``permissions``; every
    refusal (no binding, installation mismatch, fr unreachable) is raised HERE as
    an HTTPException carrying ``error_code`` (e.g. 409 ``github_app_not_installed``)
    so no route's generic ``except Exception`` can turn it into a 500.
    """
    try:
        resolved = resolve_github_token(
            owner=owner, repo=repo, tenant_id=tenant_id, permissions=permissions
        )
    except TenantRepoTokenError as e:
        raise tenant_error_http_exception(e) from e
    if not resolved.token:
        raise bad_request(
            f"No GitHub credential resolved for {owner}/{repo} (rung '{resolved.rung}'): "
            "the GitHub App is not configured and no token is set."
        )
    # The rung is logged, never the token: a silent downgrade to a weaker credential
    # surfaces later as an opaque GitHub 403 that names the credential TYPE and not the
    # reason a weaker one was chosen.
    logger.info(
        "github op auth rung=%s is_app=%s for %s/%s", resolved.rung, resolved.is_app, owner, repo
    )
    async with GitHubService(
        GitHubConfig(token_id=f"rung:{resolved.rung}", token=resolved.token)
    ) as service:
        yield service


def _is_ungranted_permission_refusal(e: BaseException) -> bool:
    """True iff GitHub refused a scoped-token mint with 422 (permission not granted).

    The mint failure reaches the route as an ``HTTPException`` raised from a
    ``TenantRepoTokenError`` raised from a ``GitHubAppMintError``; walk the cause
    chain for that last one carrying ``status_code == 422``. Anything else — a 5xx,
    a network failure, not-installed, a tenant refusal — is NOT this case.
    """
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GitHubAppMintError) and cur.status_code == 422:
            return True
        cur = cur.__cause__
    return False


def _github_refusal(e: Exception, full_name: str) -> HTTPException | None:
    """GitHub's own 401/403/404 as OUR typed answer, never a 500.

    A GitHub 403 from the App ("Resource not accessible by integration", e.g. a PR
    number read through the issues API with an issues-only token) used to fall
    into the generic ``except Exception`` below and surface as
    ``500 Failed to get issue``. Lazy import: that module imports ``_github`` here.
    """
    from routers.github_pr_status_operations import _access_denied, _is_github_404

    denied = _access_denied(e, full_name)
    if denied is not None:
        return denied
    if _is_github_404(e):
        return not_found("Repository or issue", full_name)
    return None


@router.post(
    "/create-issue",
    operation_id="create_issue",
    dependencies=[Depends(require_andon_released)],
    response_model=IssueResponse,
    summary="Create Issue",
    description="Open a GitHub issue on the given repository, optionally labelled.",
)
async def create_issue(
    request: CreateIssueRequest, tenant_id: str = Depends(require_tenant)
) -> IssueResponse:
    """Open a new issue."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ISSUES) as service:
            created = await service.create_issue(
                repo_name=request.full_name(),
                title=request.title,
                body=request.body,
                labels=request.labels,
            )
        return IssueResponse(**created)
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        raise not_found("Repository", request.full_name()) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to create issue: {e}") from e


@router.post(
    "/list-issues",
    operation_id="list_issues",
    response_model=ListIssuesResponse,
    summary="List Issues",
    description=(
        "List issues on a repository. **Pull requests are excluded.** GitHub models a PR "
        "as an issue, so an unfiltered listing answers with the open PRs — usually the "
        "majority on an active repo — which reads as a working endpoint returning "
        "plausible data. `count` is the number returned, not the repository total."
    ),
)
async def list_issues(
    request: ListIssuesRequest, tenant_id: str = Depends(require_tenant)
) -> ListIssuesResponse:
    """List issues, excluding pull requests."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ISSUES) as service:
            issues = await service.list_issues(
                repo_name=request.full_name(),
                state=request.state,
                labels=request.labels,
                limit=request.limit,
            )
        return ListIssuesResponse(issues=[IssueResponse(**i) for i in issues], count=len(issues))
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        raise not_found("Repository", request.full_name()) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to list issues: {e}") from e


@router.post(
    "/get-issue",
    operation_id="get_issue",
    response_model=IssueResponse,
    summary="Get Issue",
    description=(
        "Fetch a single issue by number. Returns 400 if the number refers to a pull "
        "request, rather than silently returning PR data shaped like an issue."
    ),
)
async def get_issue(
    request: GetIssueRequest, tenant_id: str = Depends(require_tenant)
) -> IssueResponse:
    """Fetch one issue."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ISSUES) as service:
            issue = await service.get_issue(
                repo_name=request.full_name(), issue_number=request.issue_number
            )
        return IssueResponse(**issue)
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        # list_issues/get_issue raise ValueError for "this is a PR"; distinguish it from
        # a genuinely absent repo so the caller can tell a wrong number from a wrong repo.
        if "pull request" in str(e):
            raise bad_request(str(e)) from e
        raise not_found("Issue", str(request.issue_number)) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to get issue: {e}") from e


@router.post(
    "/comment-on-issue",
    operation_id="comment_on_issue",
    dependencies=[Depends(require_andon_released)],
    response_model=IssueCommentResponse,
    summary="Comment On Issue",
    description="Post a markdown comment on an existing issue.",
)
async def comment_on_issue(
    request: CommentIssueRequest, tenant_id: str = Depends(require_tenant)
) -> IssueCommentResponse:
    """Comment on an issue."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ISSUES) as service:
            result = await service.comment_on_issue(
                repo_name=request.full_name(),
                issue_number=request.issue_number,
                body=request.body,
            )
        return IssueCommentResponse(**result)
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        raise not_found("Issue", str(request.issue_number)) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to comment on issue: {e}") from e


@router.post(
    "/add-issue-labels",
    operation_id="add_issue_labels",
    dependencies=[Depends(require_andon_released)],
    response_model=IssueResponse,
    summary="Add Issue Labels",
    description=(
        "Add labels to an EXISTING issue and return the issue with its resulting label "
        "set. Measured 2026-09-17 against the live API: re-adding a label the issue "
        "already carries is a **200 no-op** (no duplicate, no error), so the call is safe "
        "to replay; and a label the REPOSITORY does not define is **created** by GitHub "
        "on add, grey `ededed` with no description, then attached — a typo'd name does "
        "not fail, it adds a new repo label."
    ),
)
async def add_issue_labels(
    request: AddIssueLabelsRequest, tenant_id: str = Depends(require_tenant)
) -> IssueResponse:
    """Add labels to an existing issue.

    `create_issue` accepts labels only at creation time, which left the conveyor's
    GitHub-issue intake stamping `conveyor:queued` / `conveyor:failed` through a raw
    `POST /issues/{n}/labels` with a PAT that has NO Issues permission (measured: a
    plain `GET /issues/162` on that credential is 403 `need=issues=read`). This route
    goes through `_github()`, i.e. the App-first credential ladder, and the App holds
    `issues: write`.
    """
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ISSUES) as service:
            labelled = await service.add_issue_labels(
                repo_name=request.full_name(),
                issue_number=request.issue_number,
                labels=request.labels,
            )
        return IssueResponse(**labelled)
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        raise not_found("Issue", str(request.issue_number)) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to add labels to issue: {e}") from e


@router.post(
    "/close-issue",
    operation_id="close_issue",
    dependencies=[Depends(require_andon_released)],
    response_model=IssueResponse,
    summary="Close Issue",
    description=(
        "Close an issue, optionally leaving a comment first. The comment is posted "
        "BEFORE the state change, so a failure to comment leaves the issue open and the "
        "call retryable rather than closing it with the explanation lost."
    ),
)
async def close_issue(
    request: CloseIssueRequest, tenant_id: str = Depends(require_tenant)
) -> IssueResponse:
    """Close an issue (or a PR: GitHub models a PR as an issue)."""
    try:
        async with AsyncExitStack() as stack:
            # PERMS_PR_ACTIONS first: the conveyor closes stale PRs through this route,
            # and an issues-only token can 403 when the number is a pull request. An
            # installation never granted pull_requests answers the mint with 422; then,
            # and only then, retry once with PERMS_ISSUES so plain issues still close.
            scope = "pr_actions"
            try:
                service = await stack.enter_async_context(
                    _github(request.owner, request.repo, tenant_id, PERMS_PR_ACTIONS)
                )
            except HTTPException as e:
                if not _is_ungranted_permission_refusal(e):
                    raise
                logger.warning(
                    "close_issue: pull_requests:write not granted on %s; retrying with "
                    "issues:write only",
                    request.full_name(),
                )
                scope = "issues"
                service = await stack.enter_async_context(
                    _github(request.owner, request.repo, tenant_id, PERMS_ISSUES)
                )
            logger.info(
                "close_issue token scope=%s for %s#%s",
                scope,
                request.full_name(),
                request.issue_number,
            )
            closed = await service.close_issue(
                repo_name=request.full_name(),
                issue_number=request.issue_number,
                comment=request.comment,
            )
        return IssueResponse(**closed)
    except HTTPException:
        # A 400 from _github() (no token configured) must reach the caller AS a 400.
        # Without this, the generic handler below reclassifies it as a 500 and the
        # actual cause — an unconfigured service — is lost behind "Failed to ...".
        raise
    except ValueError as e:
        raise not_found("Issue", str(request.issue_number)) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        refusal = _github_refusal(e, request.full_name())
        if refusal is not None:
            raise refusal from e
        raise internal_error(f"Failed to close issue: {e}") from e
