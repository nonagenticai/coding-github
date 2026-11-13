"""
GitHub PR Status + Merge Endpoints

The last two steps of the delivery loop — read CI state, then merge — had no
implementation anywhere in the coding stack. Measured 2026-08-26 with an unshimmed
grep and a positive control: `check_runs`, `combined_status`, `merge_method` and
`merge_pull_request` were 0 hits across `coding-*/src`, while `check_workflow_status()`
sat at `github_actions_trigger.py:190` with no route and no caller.

⛔ THE PROPERTY THIS MODULE EXISTS TO PROTECT: "no failures yet" is not "passed".
The verdict is three-valued, and `pending` is refused for merge exactly as firmly as
`failed`. See GitHubService.get_pull_request_checks for the measurement that motivates it.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import error_response, internal_error, not_found
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from routers.github_issue_operations import _github
from schemas.github_api_models import (
    GetPullRequestRequest,
    ListPullRequestsRequest,
    ListPullRequestsResponse,
    MergePRRequest,
    MergePRResponse,
    PRChecksRequest,
    PRChecksResponse,
    PullRequestState,
)
from services.andon import require_andon_released
from services.github.github_service import GithubException
from services.github_integration import GitHubAuthError
from services.repo_visibility_guard import RepoVisibilityRefusedError, visibility_refused_http
from services.tenant_binding import (
    PERMS_MERGE,
    PERMS_PR_CHECKS,
    PERMS_PR_STATE,
    github_auth_http_exception,
)
from utils.github_utils import parse_rate_limit_reset

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/pr-checks",
    operation_id="get_pull_request_checks",
    response_model=PRChecksResponse,
    summary="Get Pull Request Checks",
    description=(
        "Read CI state for a PR's head commit. **`verdict` is three-valued** — "
        "`passed` | `failed` | `pending`. Never infer `passed` from `failed == 0`: an "
        "unrun pipeline has no failures. `passed` requires every check COMPLETED, no "
        "failing conclusion, and at least `expected_checks` (default 1) runs with "
        "conclusion `success`; skipped/neutral runs never count toward it. Every name in "
        "`required_checks` must have a latest run concluding `success`."
    ),
)
async def get_pull_request_checks(
    request: PRChecksRequest, tenant_id: str = Depends(require_tenant)
) -> PRChecksResponse:
    """Read a PR's CI state."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_PR_CHECKS) as service:
            state = await service.get_pull_request_checks(
                repo_name=request.full_name(),
                pr_number=request.pr_number,
                expected_checks=request.expected_checks,
            )
        return PRChecksResponse(**state)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        # NOT a 404: the PR may exist perfectly well: we could not authenticate to look.
        raise github_auth_http_exception(e) from e
    except ValueError as e:
        raise not_found("Pull request", str(request.pr_number)) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        raise internal_error(f"Failed to read PR checks: {e}") from e


@router.post(
    "/merge-pull-request",
    operation_id="merge_pull_request",
    dependencies=[Depends(require_andon_released)],
    response_model=MergePRResponse,
    summary="Merge Pull Request",
    description=(
        "Merge a pull request. By default refuses unless the check verdict is `passed` "
        "— a `pending` verdict is refused as firmly as a failure, because on a "
        "single-node CI pool an unrun pipeline is the common case, not the rare one. "
        "A refusal returns `merged: false` with a machine-readable `reason` "
        "(`checks_pending`, `checks_failed`, `already_merged`, `conflicts`, `head_moved`, "
        "`base_moved`, `merge_blocked`) rather than an error, so a caller can branch on it. "
        "Pass `sha` to merge only that exact head commit, and `require_up_to_date` to "
        "refuse when that head is behind the base branch. `already_merged` returns the "
        "existing merge commit as `sha` and the merged head as `head_sha`. On a "
        "successful merge an `aa/*` head branch in the same repository is deleted; no "
        "other branch ever is."
    ),
)
async def merge_pull_request(
    request: MergePRRequest, tenant_id: str = Depends(require_tenant)
) -> MergePRResponse:
    """Merge a PR, gated on CI."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_MERGE) as service:
            result = await service.merge_pull_request(
                repo_name=request.full_name(),
                pr_number=request.pr_number,
                merge_method=request.merge_method,
                commit_title=request.commit_title,
                commit_message=request.commit_message,
                require_checks=request.require_checks,
                expected_checks=request.expected_checks,
                sha=request.sha,
                require_up_to_date=request.require_up_to_date,
                required_checks=request.required_checks,
            )
        return MergePRResponse(**result)
    except HTTPException:
        raise
    except RepoVisibilityRefusedError as e:
        raise visibility_refused_http(e) from e
    except GitHubAuthError as e:
        # NOT a 404: the PR may exist perfectly well: we could not authenticate to merge it.
        raise github_auth_http_exception(e) from e
    except ValueError as e:
        raise not_found("Pull request", str(request.pr_number)) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        raise internal_error(f"Failed to merge PR: {e}") from e


def _is_github_404(e: Exception) -> bool:
    """PyGithub raises ``UnknownObjectException`` (a ``GithubException``, status 404)
    for a missing repo or PR — NOT ``ValueError`` — so it must be mapped explicitly or
    it falls through to the generic 500."""
    return isinstance(e, GithubException) and getattr(e, "status", None) == 404


def _github_status(e: Exception) -> int | None:
    """The HTTP status PyGithub carries, or ``None`` if this is not a GitHub error."""
    if not isinstance(e, GithubException):
        return None
    status = getattr(e, "status", None)
    return status if isinstance(status, int) else None


def _rate_limit_from_headers(e: Exception) -> dict[str, Any] | None:
    """GitHub's ``X-RateLimit-*`` off an ERROR response, or ``None`` when absent.

    ⛔ ``None`` means THE HEADERS WERE NOT THERE, which is not the same fact as
    ``remaining == 0``. A caller that backs off on an exhausted quota must never be
    handed a missing header as an exhausted one — the same rule
    :class:`GitHubRateLimit` states for the success path.

    Read case-insensitively on purpose: the casing PyGithub carries is whatever the
    transport handed it, and a caller that matched ``"X-RateLimit-Remaining"``
    exactly would silently see every lowercased header as absent — a blindness that
    fails in the *permissive* direction (rate limit read as a permission denial).

    ⛔ ``reset`` goes through the SHARED total conversion
    :func:`utils.github_utils.parse_rate_limit_reset`, never ``datetime.fromtimestamp``
    inline. An inline call sat OUTSIDE ``_int``'s try/except and raised ValueError on a
    millisecond reset, escaping the route's own ``except Exception`` clause entirely —
    so the 403 this whole function exists to make legible arrived with NO
    ``error_code``. ⚠️ ``remaining`` and ``limit`` are returned whatever the reset does:
    losing ``remaining: 0`` because a date would not parse would throw away the
    "do not rotate the token" signal, which is the point.
    """
    headers = getattr(e, "headers", None)
    if not isinstance(headers, dict):
        return None
    lower = {str(k).lower(): v for k, v in headers.items()}

    def _int(name: str) -> int | None:
        try:
            return int(lower[name])
        except (KeyError, TypeError, ValueError):
            return None

    remaining = _int("x-ratelimit-remaining")
    if remaining is None:
        return None
    reset_epoch, reset_at = parse_rate_limit_reset(lower.get("x-ratelimit-reset"))
    return {
        "remaining": remaining,
        "limit": _int("x-ratelimit-limit"),
        "reset_epoch": reset_epoch,
        "reset_at": reset_at,
    }


#: GitHub's exact message when an App installation token lacks the permission a
#: resource needs. Distinct from org policy / SSO / a repo the token cannot see.
APP_PERMISSION_DENIED_MESSAGE = "Resource not accessible by integration"


def _is_app_permission_denial(e: Exception) -> bool:
    """True when GitHub's 403 body names a missing App permission."""
    data = getattr(e, "data", None)
    message = data.get("message") if isinstance(data, dict) else None
    return isinstance(message, str) and APP_PERMISSION_DENIED_MESSAGE in message


def _access_denied(e: Exception, full_name: str) -> HTTPException | None:
    """Map GitHub's 401/403 to OUR 401/403, keeping it distinct from 404 — and
    keeping an EXHAUSTED QUOTA distinct from a permission denial.

    Four outcomes, because the callers take different actions:

    * ``401 github_unauthorized`` — the credential was rejected. ROTATE IT.
    * ``403 github_rate_limited`` — GitHub answered 403 **and said
      ``X-RateLimit-Remaining: 0``**. The credential is fine and the quota is spent:
      ⛔ WAIT until ``reset_at``, DO NOT ROTATE. A rotation here burns a good token
      and does not help, because the quota is per-account, not per-token.
    * ``403 github_app_permission_denied`` — GitHub said "Resource not accessible
      by integration": the App token lacks the permission for this resource. Do
      not rotate; the permission set (or the request) must change.
    * ``403 github_forbidden`` — the repo is visible and access is refused anyway
      (org policy, SAML). Rotating MAY help.

    The status stays GitHub's own (a rate limit really is a 403 on this API), so the
    discriminator is ``error_code`` plus ``details.rate_limit`` — a caller reading
    only the status still behaves exactly as it does today.

    ⛔ MUTANTS this must refuse: fold 401/403 into ``not_found`` (-> 404) or into the
    generic 500; or collapse the rate-limit branch into the generic auth branch, so
    the conveyor rotates a healthy credential every time the hour's quota runs out.
    """
    status = _github_status(e)
    if status not in (401, 403):
        return None
    if status == 401:
        return error_response(
            status_code=401,
            message=(
                f"GitHub rejected the credential for '{full_name}' (HTTP 401). The "
                "repository may exist; we could not authenticate to look."
            ),
            error_code="github_unauthorized",
            details={"repository": full_name, "error_code": "github_unauthorized"},
        )

    rate_limit = _rate_limit_from_headers(e)
    if rate_limit is not None and rate_limit["remaining"] == 0:
        return error_response(
            status_code=403,
            message=(
                f"GitHub's API rate limit is exhausted (HTTP 403 with "
                f"X-RateLimit-Remaining: 0) while reading '{full_name}'. The "
                "credential is NOT the problem — do not rotate it; retry after "
                f"{rate_limit['reset_at'] or 'the reset window'}."
            ),
            error_code="github_rate_limited",
            details={
                "repository": full_name,
                "error_code": "github_rate_limited",
                "rate_limit": rate_limit,
                "credential_rotation_would_not_help": True,
            },
        )
    if _is_app_permission_denial(e):
        return error_response(
            status_code=403,
            message=(
                f"GitHub refused the App credential for '{full_name}' (HTTP 403 "
                f"'{APP_PERMISSION_DENIED_MESSAGE}'): the installation token does not "
                "hold the permission this resource needs (e.g. a pull request read "
                "through the issues API needs pull_requests access). Rotating will not "
                "help; the permission set or the request must change."
            ),
            error_code="github_app_permission_denied",
            details={
                "repository": full_name,
                "error_code": "github_app_permission_denied",
                "rate_limit": rate_limit,
            },
        )
    return error_response(
        status_code=403,
        message=f"'{full_name}' exists but this credential may not read it (HTTP 403).",
        error_code="github_forbidden",
        details={
            "repository": full_name,
            "error_code": "github_forbidden",
            "rate_limit": rate_limit,
        },
    )


def _shape_drift(route: str, e: ValidationError) -> HTTPException:
    """Response-shape drift is OUR bug: a 500 naming the fields, never a 404.

    Only field locations and error types are reported — ``include_input=False`` keeps
    the offending values (which could carry upstream data) out of logs and response.
    """
    fields = sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()})
    logger.error(
        "%s response shape drift: %s",
        route,
        e.errors(include_url=False, include_input=False, include_context=False),
    )
    return internal_error(f"{route}: upstream PR data did not match the response model {fields}")


@router.post(
    "/get-pull-request",
    operation_id="get_pull_request",
    response_model=PullRequestState,
    summary="Get Pull Request",
    description=(
        "Read one pull request: state, merged/merged_at, live `mergeable` (`null` = "
        "GitHub has not computed it yet, NOT 'unmergeable'), head SHA, base/head refs, "
        "title, url, draft, and `base_default_branch` (the BASE repo's default branch, "
        "so `base != base_default_branch` is answerable without a second "
        "`get-repository` call; `null` = UNKNOWN, never 'not the default branch'). "
        "**The four failure modes stay distinct**: `404` the repo "
        "or PR does not exist for this credential; `401 github_unauthorized` the "
        "credential was rejected — rotate it; `403 github_rate_limited` the quota is "
        "spent (GitHub sent `X-RateLimit-Remaining: 0`) — wait for "
        "`details.rate_limit.reset_at`, ⛔ do NOT rotate; `403 github_forbidden` the "
        "repo is visible and access is refused anyway. `rate_limit` carries GitHub's "
        "`X-RateLimit-*` from this call."
    ),
)
async def get_pull_request_state(
    request: GetPullRequestRequest, tenant_id: str = Depends(require_tenant)
) -> PullRequestState:
    """Read one PR."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_PR_STATE) as service:
            state = await service.get_pull_request_state(
                repo_name=request.full_name(), pr_number=request.pr_number
            )
        return PullRequestState(**state)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        # NOT a 404: the PR may exist perfectly well: we could not authenticate to look.
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        # BEFORE ValueError: pydantic's ValidationError IS a ValueError, so shape drift
        # between the service dict and PullRequestState would otherwise read as a 404.
        raise _shape_drift("get-pull-request", e) from e
    except ValueError as e:
        raise not_found("Pull request", str(request.pr_number)) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Pull request", f"{request.full_name()}#{request.pr_number}") from e
        raise internal_error(f"Failed to read pull request: {e}") from e


@router.post(
    "/list-pull-requests",
    operation_id="list_pull_requests",
    response_model=ListPullRequestsResponse,
    summary="List Pull Requests",
    description=(
        "List pull requests, most recently updated first. `state` is "
        "`open` | `closed` | `all` (default `open`); `limit` 1..100 (default 30); "
        "optional `head` branch filter. Items share the get-pull-request shape, except "
        "`mergeable` is always `null` (GitHub's list endpoint does not compute it) and "
        "`rate_limit` is all-null. `base_default_branch` IS populated on list items — "
        "GitHub's list payload carries `base.repo.default_branch` — so the trunk "
        "comparison works here too, at no extra API call."
    ),
)
async def list_pull_request_states(
    request: ListPullRequestsRequest, tenant_id: str = Depends(require_tenant)
) -> ListPullRequestsResponse:
    """List PRs."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_PR_STATE) as service:
            items = await service.list_pull_request_states(
                repo_name=request.full_name(),
                state=request.state,
                limit=request.limit,
                head=request.head,
            )
        return ListPullRequestsResponse(items=[PullRequestState(**i) for i in items])
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("list-pull-requests", e) from e
    except ValueError as e:
        raise not_found("Repository", request.full_name()) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Repository", request.full_name()) from e
        raise internal_error(f"Failed to list pull requests: {e}") from e
