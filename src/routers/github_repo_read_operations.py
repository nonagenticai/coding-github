"""
GitHub Phase B Read Endpoints — repository metadata, PR changed files, Actions runs.

WHY THIS MODULE EXISTS
----------------------
Six n8n workflows (delivery-sweep, landing-reconciler, github-outcome-poll and
friends) still call ``api.github.com`` directly with a raw org PAT
(``AA_GITHUB_TOKEN``), because these three reads had no tenant-scoped route. Every
such call carries the platform's org token through n8n's execution store. The three
routes here close that gap; nothing else about them is new — they follow the
credential seam, the error mapping and the route idiom already used by
``github_pr_status_operations``.

⛔ THE PROPERTY THIS MODULE EXISTS TO PROTECT: **a refusal must not read as a
404, and a 404 must not read as a refusal.**

``get-repository`` is the one the callers branch on. GitHub answers three ways and
they mean three different things:

* **404** — the repository does not exist for this credential. ⚠️ Read that
  precisely: GitHub returns 404 (not 403) for a PRIVATE repo the credential cannot
  see, deliberately, so that existence is not leaked. "Absent" and "invisible" are
  therefore NOT distinguishable from the outside, and this route does not pretend
  otherwise — it reports 404 and says so in the message.
* **401** — the credential itself was rejected. The repo may exist perfectly well.
* **403** — the repo exists and is visible, and access is forbidden anyway (org
  policy, SAML, a secondary rate limit).

Collapsing any of those into another is a silent regression for the callers, which
is why each has its own branch and its own test. The generic ``except Exception ->
500`` at the bottom of each handler must never be the thing that answers an auth
question.

TOKEN RESOLUTION — ⛔ no new seam, and no PAT fallback. Every handler opens
``routers.github_issue_operations._github``, exactly as the PR-status routes do.
That helper calls ``services.github_integration.resolve_github_token``, which for a
platform-owned owner runs ``services.tenant_binding.check_platform_repo_tenant``
(the platform-repo tenant gate) and for a customer owner runs
``resolve_customer_repo_token`` (tenant-bound public App). A refusal raises a
``TenantRepoTokenError``, which ``_github`` maps through
``tenant_error_http_exception`` into the established 403
``platform_repo_forbidden`` envelope — so it must reach the client AS that, never
as a 500. Each handler therefore re-raises ``HTTPException`` before its generic
clause, and that ordering is pinned by a test.

Tokens are never logged here; the helper logs the RUNG, never the credential.
"""

from __future__ import annotations

import logging

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import error_response, internal_error, not_found
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from routers.github_issue_operations import _github
from routers.github_pr_status_operations import (
    _access_denied,
    _is_github_404,
    _shape_drift,
)
from schemas.github_api_models import (
    BranchHeadResponse,
    CompareCommitsRequest,
    CompareCommitsResponse,
    CompareFilesRequest,
    CompareFilesResponse,
    GetBranchRequest,
    GetRepositoryRequest,
    ListPullRequestFilesRequest,
    ListPullRequestFilesResponse,
    ListWorkflowRunsRequest,
    ListWorkflowRunsResponse,
    RepositoryStateResponse,
)
from services.github.github_service import CompareFilesTruncatedError
from services.github_integration import GitHubAuthError
from services.tenant_binding import (
    PERMS_ACTIONS_READ,
    PERMS_PR_STATE,
    PERMS_REPO_READ,
    github_auth_http_exception,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/get-repository",
    operation_id="get_repository",
    response_model=RepositoryStateResponse,
    summary="Get Repository",
    description=(
        "Read one repository: `default_branch`, `archived`, `disabled`, `private`, "
        "`visibility`, `fork` and the acting credential's `permissions`. "
        "**The three failure modes stay distinct**: `404` the repo does not exist "
        "*for this credential* (GitHub also answers 404 for a private repo it may not "
        "see, so absent and invisible are not separable); `401` the credential was "
        "rejected — the repo may exist; `403` the repo exists and access is forbidden. "
        "`permissions` flags are tri-state — `null` means GitHub did not say, which is "
        "not `false`. `rate_limit` carries GitHub's `X-RateLimit-*` from this call."
    ),
)
async def get_repository(
    request: GetRepositoryRequest, tenant_id: str = Depends(require_tenant)
) -> RepositoryStateResponse:
    """Read one repository's metadata."""
    full_name = request.full_name()
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_REPO_READ) as service:
            state = await service.get_repository_state(repo_name=full_name)
        return RepositoryStateResponse(**state)
    except HTTPException:
        # Includes the platform-repo tenant refusal (403 platform_repo_forbidden)
        # raised inside _github. It must reach the caller in ITS shape, not as a 500.
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        # BEFORE ValueError: pydantic's ValidationError IS a ValueError, so shape
        # drift would otherwise be answered as a 404 "repository not found".
        raise _shape_drift("get-repository", e) from e
    except ValueError as e:
        raise not_found("Repository", full_name) from e
    except Exception as e:
        denied = _access_denied(e, full_name)
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Repository", full_name) from e
        raise internal_error(f"Failed to read repository: {e}") from e


@router.post(
    "/list-pull-request-files",
    operation_id="list_pull_request_files",
    response_model=ListPullRequestFilesResponse,
    summary="List Pull Request Files",
    description=(
        "One PAGE of a pull request's changed files: `filename`, `status`, "
        "`additions`, `deletions`, `changes`, and `patch` when GitHub supplies one "
        '(`null` for binary files and over-size diffs — never `""`). `page` is '
        "1-based; `per_page` is 1..100 (GitHub's cap), default 100. `has_more` is "
        "`count == per_page`, i.e. another page MAY exist — GitHub sends no total, so "
        "there is no honest way to promise one. Page until a short or empty page."
    ),
)
async def list_pull_request_files(
    request: ListPullRequestFilesRequest, tenant_id: str = Depends(require_tenant)
) -> ListPullRequestFilesResponse:
    """Read one page of a PR's changed files."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_PR_STATE) as service:
            page = await service.list_pull_request_file_page(
                repo_name=request.full_name(),
                pr_number=request.pr_number,
                page=request.page,
                per_page=request.per_page,
            )
        return ListPullRequestFilesResponse(**page)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("list-pull-request-files", e) from e
    except ValueError as e:
        raise not_found("Pull request", str(request.pr_number)) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Pull request", f"{request.full_name()}#{request.pr_number}") from e
        raise internal_error(f"Failed to list pull request files: {e}") from e


@router.post(
    "/list-workflow-runs",
    operation_id="list_workflow_runs",
    response_model=ListWorkflowRunsResponse,
    summary="List Workflow Runs",
    description=(
        "GitHub Actions runs whose head commit is `head_sha`, newest first. "
        "**This is NOT `/pr-checks`** — that one is keyed by PR number and reads check "
        "runs; this is keyed by a commit, which may have no PR at all. Each run "
        "carries `id`, `name`, `status`, `conclusion`, `html_url`, `head_sha`, "
        "`created_at` (plus `event`, `path`, `workflow_id`, `run_number`, "
        "`run_attempt`, `head_branch`, `updated_at`). `conclusion` is `null` until "
        '`status == "completed"`. ⛔ `count == 0` means no run has been CREATED for '
        "this SHA yet (or they aged out of retention) — it is never 'passed'."
    ),
)
async def list_workflow_runs(
    request: ListWorkflowRunsRequest, tenant_id: str = Depends(require_tenant)
) -> ListWorkflowRunsResponse:
    """Read Actions runs for one head SHA."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_ACTIONS_READ) as service:
            runs = await service.list_workflow_runs_for_sha(
                repo_name=request.full_name(),
                head_sha=request.head_sha,
                per_page=request.per_page,
            )
        return ListWorkflowRunsResponse(**runs)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("list-workflow-runs", e) from e
    except ValueError as e:
        raise not_found("Repository", request.full_name()) from e
    except Exception as e:
        denied = _access_denied(e, request.full_name())
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Repository", request.full_name()) from e
        raise internal_error(f"Failed to list workflow runs: {e}") from e


@router.post(
    "/get-branch",
    operation_id="get_branch",
    response_model=BranchHeadResponse,
    summary="Get Branch",
    description=(
        "Read one branch: its HEAD commit `sha` (full 40 hex) and whether it is "
        "`protected`. `branch` may contain `/` (e.g. `feature/x`); whitespace and `..` "
        "are refused with 422. **Failure modes stay distinct**, as on `/get-repository`: "
        "`404` the repository OR the branch does not exist *for this credential* "
        "(GitHub also answers 404 for a private repo it may not see); `401` the "
        "credential was rejected; `403` access is forbidden. `rate_limit` carries "
        "GitHub's `X-RateLimit-*` from this call."
    ),
)
async def get_branch(
    request: GetBranchRequest, tenant_id: str = Depends(require_tenant)
) -> BranchHeadResponse:
    """Read one branch's HEAD commit SHA."""
    full_name = request.full_name()
    ref = f"{full_name}@{request.branch}"
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_REPO_READ) as service:
            head = await service.get_branch_head(repo_name=full_name, branch=request.branch)
        return BranchHeadResponse(owner=request.owner, repo=request.repo, **head)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("get-branch", e) from e
    except ValueError as e:
        raise not_found("Branch", ref) from e
    except Exception as e:
        denied = _access_denied(e, full_name)
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Branch", ref) from e
        raise internal_error(f"Failed to read branch: {e}") from e


@router.post(
    "/compare",
    operation_id="compare_commits",
    response_model=CompareCommitsResponse,
    summary="Compare Commits",
    description=(
        "Compare `base` against `head` (each a commit SHA or a ref name): GitHub's "
        "`status` (`ahead` | `behind` | `identical` | `diverged`), `ahead_by`, "
        "`behind_by`, the `merge_base_sha` (40 hex), and `base_is_ancestor_of_head` — "
        "`true` iff `status` is `ahead` or `identical`, i.e. every commit of `base` is "
        "already in `head`. ⛔ A `diverged` pair is `false` even with `ahead_by > 0`. "
        "Whitespace and `..` are refused with 422. **Failure modes stay distinct**, as "
        "on `/get-branch`: `404` the repository, sha or ref does not exist *for this "
        "credential* — never a `200` with `false`; `401` the credential was rejected; "
        "`403` access is forbidden. `rate_limit` carries GitHub's `X-RateLimit-*`."
    ),
)
async def compare_commits(
    request: CompareCommitsRequest, tenant_id: str = Depends(require_tenant)
) -> CompareCommitsResponse:
    """Compare two commits and report whether ``base`` is contained in ``head``."""
    full_name = request.full_name()
    ref = f"{full_name}@{request.base}...{request.head}"
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_REPO_READ) as service:
            result = await service.compare_commits(
                repo_name=full_name, base=request.base, head=request.head
            )
        return CompareCommitsResponse(
            owner=request.owner,
            repo=request.repo,
            base=request.base,
            head=request.head,
            **result,
        )
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("compare", e) from e
    except ValueError as e:
        raise not_found("Commit", ref) from e
    except Exception as e:
        denied = _access_denied(e, full_name)
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Commit", ref) from e
        raise internal_error(f"Failed to compare commits: {e}") from e


@router.post(
    "/compare/files",
    operation_id="compare_files",
    response_model=CompareFilesResponse,
    summary="Compare Files",
    description=(
        "List the EXACT files changed between `base` and `head` (each a commit SHA or a "
        "ref name), sorted by `filename`, with `files_count`, plus the same comparison "
        "fields as `/compare`: `status` (`ahead` | `behind` | `identical` | "
        "`diverged`), `ahead_by`, `behind_by`, `merge_base_sha` (40 hex) and "
        "`base_is_ancestor_of_head` (`true` iff `status` is `ahead` or `identical`). "
        "Each file carries `filename`, `status` and `previous_filename` (set only when "
        "`renamed`). ⛔ **Never a partial list**: GitHub's compare returns at most 300 "
        "files with no truncation flag, so 300 or more answers `422` with "
        "`error_code` `compare_files_truncated` and the observed `count`, and no "
        "`files`. Whitespace and `..` are refused with 422. **Failure modes stay "
        "distinct**, as on `/compare`: `404` the repository, sha or ref does not exist "
        "*for this credential*; `401` the credential was rejected; `403` access is "
        "forbidden. `rate_limit` carries GitHub's `X-RateLimit-*`."
    ),
)
async def compare_files(
    request: CompareFilesRequest, tenant_id: str = Depends(require_tenant)
) -> CompareFilesResponse:
    """List the files changed between two commits — exactly, or not at all."""
    full_name = request.full_name()
    ref = f"{full_name}@{request.base}...{request.head}"
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_REPO_READ) as service:
            result = await service.compare_files(
                repo_name=full_name, base=request.base, head=request.head
            )
        return CompareFilesResponse(
            owner=request.owner,
            repo=request.repo,
            base=request.base,
            head=request.head,
            **result,
        )
    except HTTPException:
        raise
    except CompareFilesTruncatedError as e:
        # A possibly-truncated list must never be returned: refuse, carrying the count.
        raise error_response(
            status_code=422,
            message=(
                f"GitHub returned {e.count} changed files for '{ref}', at or above its "
                f"{e.cap}-file compare cap; the list may be truncated and is not returned."
            ),
            error_code="compare_files_truncated",
            details={
                "repository": full_name,
                "error_code": "compare_files_truncated",
                "count": e.count,
                "cap": e.cap,
            },
        ) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except ValidationError as e:
        raise _shape_drift("compare-files", e) from e
    except ValueError as e:
        raise not_found("Commit", ref) from e
    except Exception as e:
        denied = _access_denied(e, full_name)
        if denied is not None:
            raise denied from e
        if _is_github_404(e):
            raise not_found("Commit", ref) from e
        raise internal_error(f"Failed to compare files: {e}") from e
