"""
GitHub Pull Request Operations Endpoints

Handles fork creation, code changes, PR creation, and validation.
Split from main github.py to reduce file size and improve organization.
"""

from __future__ import annotations

import logging
from uuid import uuid7

from coding_client.auth.tenant import require_tenant
from coding_client.exceptions import (
    InternalServerException,
    NotFoundException,
    PermissionException,
)
from coding_client.models.responses import (  # TODO: verify coding-client import
    bad_request,
    conflict,
    error_response,
    internal_error,
    not_found,
)
from fastapi import APIRouter, Depends, HTTPException

# Import validation utilities from separate module
from routers.github_validation_utils import validate_pr_performance as _validate_pr_performance
from routers.github_validation_utils import validate_pr_quality as _validate_pr_quality
from routers.github_validation_utils import validate_pr_security as _validate_pr_security
from schemas.github_api_models import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    CreatePRRequest,
    CreatePRResponse,
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error409Response,
    Error429Response,
    Error500Response,
    ForkRequest,
    ForkResponse,
    MarkPRReadyRequest,
    MarkPRReadyResponse,
    ResponseStatus,
    UpdatePRBodyRequest,
    UpdatePRBodyResponse,
    ValidatePRRequest,
    ValidatePRResponse,
    ValidationErrorResponse,
)
from services.andon import require_andon_released
from services.github_integration import (  # TODO: verify
    BaseRefNotFoundError,
    DefaultBranchWriteRefusedError,
    GitHubAuthError,
    GitHubIntegration,
    assert_branch_writable,
)
from services.path_guard import CommitPathRefusedError, assert_commit_paths_allowed
from services.pr_validation.integration.github_validation_service import (
    validate_pr_comprehensive,
)
from services.repo_visibility_guard import (
    RepoVisibilityRefusedError,
    assert_repo_not_public,
    assert_repo_not_public_lookup,
    visibility_refused_http,
)
from services.tenant_binding import PERMS_PR_READ, github_auth_http_exception
from utils.github_utils import parse_github_url

logger = logging.getLogger(__name__)
router = APIRouter()


class _ResolvedContainer:
    """Lightweight, attribute-style view over a coding-containers GET payload.

    Exposes only the fields the fork / apply-changes handlers read so the
    downstream GitHubIntegration code stays unchanged when the resolution
    source switches from the local docker_service to the shared client.
    """

    __slots__ = ("container_name", "repo_url", "branch", "github_access_token", "secret_config_id")

    def __init__(self, info: dict) -> None:
        self.container_name: str | None = info.get("container_name") or info.get("name")
        self.repo_url: str | None = info.get("repo_url") or info.get("repository_url")
        # coding-containers requires a non-empty branch; default defensively
        # so the downstream _create_branch base-branch arg is always a str.
        self.branch: str = str(info.get("branch") or "main")
        self.github_access_token: str | None = info.get("github_access_token")
        # Opaque reference to the project's GitHub PAT in secrets-service.
        # coding-containers never persists a plaintext token (github_access_token
        # is always None); the real BYO/project token must be resolved from here.
        self.secret_config_id: str | None = info.get("secret_config_id")


def _default_branch_refused(e: DefaultBranchWriteRefusedError) -> HTTPException:
    """403 with a machine code a caller can branch on (repeated in ``details``
    because coding-client's shared handler keeps only message + details)."""
    return error_response(
        status_code=403,
        message=str(e),
        error_code=e.error_code,
        details={"error_code": e.error_code},
    )


def _refuse_public_target(integration: GitHubIntegration, owner: str, repo: str) -> None:
    """A PUBLIC repo is never a conveyor target — refuse before ANY GitHub write.

    One metadata lookup per request; a lookup failure is ``repo_visibility_unknown``
    (fail closed). With no client there is no credential to write with either, so the
    refusal is the same auth error ``_ensure_fork`` would raise.
    """
    client = integration.github_client
    if not client:
        raise GitHubAuthError(
            "No GitHub token configured. Supply 'user_token' in the request "
            "or install the GitHub App for this tenant (no PAT fallback exists)."
        )
    full_name = f"{owner}/{repo}"
    assert_repo_not_public_lookup(lambda: client.get_repo(full_name), full_name)


def _fork_html_url(fork_repo: object) -> str:
    """Best-effort html_url for a repo object, with a constructed fallback.

    Factored out so the empty-change-set early return and the normal path
    cannot drift apart: both report the same fork URL for the same repo.
    """
    return getattr(
        fork_repo,
        "html_url",
        f"https://github.com/{fork_repo.owner.login}/{fork_repo.name}",
    )


def _authoring_identity(integration: object) -> str | None:
    """The credential rung that signed this write, when it is actually known.

    Deliberately reports ``None`` rather than guessing: the whole point of the
    field is that it names a real rung, so a value we cannot vouch for must not
    be published as one. (It also keeps the response model honest under test
    doubles, where the attribute is a Mock rather than a string.)
    """
    rung = getattr(integration, "token_rung", None)
    return rung if isinstance(rung, str) else None


def _assert_can_write(repo: object, rung: str | None, is_app: bool = False) -> None:
    """Refuse a write a resolved **PAT** demonstrably cannot perform.

    PyGithub already carries ``.permissions`` on the repo object the caller just
    fetched, so this costs NO extra API call. Without it the first symptom of an
    under-privileged credential is GitHub's 403 on ``POST /git/refs`` —
    *"Resource not accessible by personal access token"* — which names the
    credential TYPE and neither the repo nor the rung, and which cost three
    sessions the diagnosis on 2026-08-15.

    This covers the case App-first does NOT: a CUSTOMER repo legitimately keeps
    its BYO PAT, and an under-privileged customer PAT fails identically opaquely.

    ⛔ **NEVER applied to a GitHub App token.** ``permissions`` is a USER-shaped
    field and GitHub does not populate it for installation tokens. Measured live
    2026-08-15 against installation 131797507 (``code-graph-helper-app``, which
    holds ``contents: write``, ``repository_selection: all``, 208 repos incl. the
    target, and authored PRs #78-#83 on it)::

        permissions = {'admin': False, 'maintain': False, 'push': False,
                       'triage': False, 'pull': False}

    ``pull: False`` was returned by the very request that successfully READ the
    repo — the block is self-contradictory, not a capability report. Enforcing it
    for App tokens would refuse 100% of App-authored writes: a guard whose
    "healthy" branch has a 0% hit rate, blocking exactly the path this exists to
    enable.

    Silent on anything it cannot read: absent ``.permissions`` means "unknown",
    and an unknown capability must not manufacture a refusal — the real write
    remains the ultimate authority.
    """
    if is_app:
        return
    permissions = getattr(repo, "permissions", None)
    push = getattr(permissions, "push", None)
    if push is not False:
        return
    full_name = getattr(repo, "full_name", None) or "the target repository"
    raise GitHubAuthError(
        f"The resolved GitHub credential (rung '{rung or 'unknown'}') has no push "
        f"permission on {full_name}, so no branch can be created. This is a "
        f"credential capability problem, not a code problem."
    )


async def _effective_github_token(
    container: _ResolvedContainer, tenant_id: str | None, owner: str | None = None
) -> str | None:
    """Resolve the BYO token to push/PR with, or ``None`` to use the App.

    Order for a CUSTOMER repo: the container's inline token (legacy) → the
    tenant's own BYO token resolved from secrets-service via ``secret_config_id`` →
    ``None`` so ``GitHubIntegration`` resolves the tenant-bound public GitHub App
    (there is no platform PAT to fall back to).

    A customer's private repo is not reachable by the platform App, so its BYO
    PAT must win there — that is R1 and it stands.

    ⛔ **For a PLATFORM-OWNED repo this returns ``None`` unconditionally**, so
    the App authors. Returning a token here BYPASSES the ladder entirely:
    ``GitHubIntegration.__init__`` only calls :func:`resolve_github_token` when
    ``github_token is None``. That bypass is what broke the 2026-08-15 drive —
    ``drive-it-flow1-demo.sh`` bakes whatever PAT the operator exported into a
    fresh per-project secret, it resolved fine (rung 'byo'), and it simply
    lacked ``contents: write`` on ``nonagenticai/enterprise_mcp_server`` → 403 on
    ``POST /git/refs``. Measured across executions 531834/531862/531895; the
    08-14 run that produced PR #85 differed ONLY in which secret_config_id it
    had baked in. Our own repos have a durable identity that cannot drift with
    whoever launched the drive, and it is the App.
    """
    from services.github_integration import is_platform_owned, is_sandbox_owned

    # ⛔ A SANDBOX org (CODING_GITHUB_SANDBOX_GITHUB_OWNERS) is platform-owned
    # too: App only (the public App, platform-gated). A stored project/user token
    # must never author there — it would bypass the platform tenant gate.
    if is_platform_owned(owner) or is_sandbox_owned(owner):
        if container.github_access_token or container.secret_config_id:
            logger.info(
                "Ignoring project BYO token for platform-owned repo %s/%s; "
                "the GitHub App is the authoring identity for our own repos.",
                owner,
                container.container_name,
            )
        return None
    if container.github_access_token:
        return container.github_access_token
    from services.secret_resolver import resolve_git_token

    return await resolve_git_token(container.secret_config_id, tenant_id)


async def _resolve_container(container_name: str, tenant_id: str | None) -> _ResolvedContainer:
    """Resolve a container via the unified coding-containers client.

    The fork / apply-changes container is provisioned through the
    coding-containers k8s subsystem (``createContainer``) and lives in *that*
    registry — NOT coding-github's local ``docker_service``. Resolving it here
    must therefore go through the shared containers client and forward the
    project tenant (else coding-containers' tenant-scoped GET 404s). This
    mirrors the PR-creation path in
    ``GitHubIntegration.create_pull_request`` (coding-github #38).
    """
    from coding_client.clients import ServiceError

    from main import get_workspaces_client
    from services.outbound import optional_sibling_headers

    try:
        client = get_workspaces_client()
    except RuntimeError as e:
        # Containers client not initialised → treat as container not found.
        raise not_found("Container", container_name) from e

    # Present coding-github's OWN licence alongside the tenant assertion. A bare
    # X-Tenant-Id is no longer an identity: coding-containers runs
    # TRUST_TENANT_HEADER=false, so a header-only call 401s and the PR path dies
    # with "Containers service error: ... returned 401". #72 introduced this helper
    # and converted the webhook/provisioning/secret-resolver callers, but not this
    # one or its twin in services/github_integration.py -- the two on the
    # PR-CREATION path, i.e. exactly the ones a dependency-upgrade card needs.
    headers = optional_sibling_headers(tenant_id) or None
    try:
        raw = await client.get_workspace(container_name, headers=headers)
    except ServiceError as e:
        if e.status_code == 404:
            raise not_found("Container", container_name) from e
        raise internal_error(f"Containers service error: {e}") from e

    # coding-containers wraps the model in an envelope:
    # {"status": ..., "message": ..., "container": {<ContainerInfo>}}.
    # Unwrap to the ContainerInfo (tolerate an already-flat payload).
    info = raw.get("container", raw) if isinstance(raw, dict) else raw
    if not isinstance(info, dict):
        raise not_found("Container", container_name)
    return _ResolvedContainer(info)


@router.post(
    "/fork",
    operation_id="fork_repository",
    dependencies=[Depends(require_andon_released)],
    response_model=ForkResponse,
    summary="Fork Repository",
    description="""
    Create a personal fork of any provisioned repository for development.

    **Usage:**
    ```json
    {
        "container_name": "my-container-id"
    }
    ```

    **What it does:**
    - Creates fork under your GitHub account
    - Preserves full git history and branches
    - Maintains connection to original repository
    - Enables safe experimentation and contribution

    **Response includes:**
    - `fork_url` - Your new fork on GitHub
    - `fork_owner` - Your GitHub username
    - `original_url` - Source repository

    **Next steps:**
    - Clone fork locally for development
    - Use `/apply-changes` to modify code
    - Create pull requests back to original
    """,
    responses={
        200: {
            "model": ForkResponse,
            "description": "Repository successfully forked",
        },
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Container or repository not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - invalid request body",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def fork_repository(
    request: ForkRequest, tenant_id: str = Depends(require_tenant)
) -> ForkResponse:
    """Fork the repository associated with the given container."""
    try:
        # Resolve the container via the shared coding-containers client (the
        # fork target is provisioned in the coding-containers subsystem, not
        # coding-github's local registry).
        container = await _resolve_container(request.container_name, tenant_id)

        if not container.repo_url:
            raise not_found("Container", f"{request.container_name} (or has no repository)")

        # Parse GitHub URL
        repo_owner, repo_name = parse_github_url(container.repo_url)
        if not repo_owner or not repo_name:
            raise bad_request(f"Invalid GitHub URL: {container.repo_url}")

        # Resolve the token as in apply-changes: a CUSTOMER repo uses the
        # project's BYO/PAT from secrets-service (R1); a platform-owned repo
        # gets None so the App authors. target_owner/repo let the App mint a
        # per-owner installation token instead of the home one.
        container_token = await _effective_github_token(container, tenant_id, owner=repo_owner)
        github_integration = GitHubIntegration(
            github_token=container_token,
            target_owner=repo_owner,
            target_repo=repo_name,
            tenant_id=tenant_id,
        )

        # ⛔ A PUBLIC upstream is refused before ANY GitHub write: a fork of a public
        # repo is itself public, and the copy-repo fallback would create a repo and
        # commit into it. Then the resolved repo itself is checked (fail closed).
        _refuse_public_target(github_integration, repo_owner, repo_name)
        repo_info = github_integration._ensure_fork(repo_owner, repo_name)
        fork_repo = repo_info.repository
        assert_repo_not_public(
            fork_repo, getattr(fork_repo, "full_name", None) or f"{repo_owner}/{repo_name}"
        )

        # Get fork URL safely
        fork_url = getattr(
            fork_repo,
            "html_url",
            f"https://github.com/{fork_repo.owner.login}/{fork_repo.name}",
        )

        return ForkResponse(
            status=ResponseStatus.SUCCESS,
            message="Fork created successfully",
            fork_url=fork_url,
            fork_owner=fork_repo.owner.login,
            fork_name=fork_repo.name,
            original_url=container.repo_url,
            request_id=request.container_name,
        )

    except HTTPException:
        raise
    except RepoVisibilityRefusedError as e:
        raise visibility_refused_http(e) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except RuntimeError:
        raise not_found("Container", request.container_name)
    except Exception as e:
        raise internal_error(f"Failed to fork repository: {str(e)}") from e


@router.post(
    "/apply-changes",
    operation_id="apply_code_changes",
    dependencies=[Depends(require_andon_released)],
    response_model=ApplyChangesResponse,
    summary="Apply Code Changes",
    description="""
    Apply multiple file changes to a forked repository with automatic branch management.

    **Single file change:**
    ```json
    {
        "container_name": "my-container",
        "changes": [
            {
                "file_path": "main.py",
                "content": "print('Hello World')",
                "change_type": "modified"
            }
        ],
        "branch_name": "feature/hello-world"
    }
    ```

    **Multiple files:**
    ```json
    {
        "container_name": "web-app",
        "changes": [
            {"file_path": "src/app.js", "content": "// New code"},
            {"file_path": "README.md", "content": "# Updated docs"}
        ]
    }
    ```

    **What it does:**
    - Creates new branch (or uses specified name)
    - Applies all file changes atomically
    - Commits changes with descriptive message
    - Pushes to your fork on GitHub

    **Auto-generates branch names** if not specified.
    **Creates directories** automatically as needed.
    """,
    responses={
        200: {
            "model": ApplyChangesResponse,
            "description": "Changes successfully applied",
        },
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Container or repository not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - invalid request body",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def apply_code_changes(
    request: ApplyChangesRequest,
    tenant_id: str = Depends(require_tenant),
) -> ApplyChangesResponse:
    """Apply code changes to the fork of the repository."""
    try:
        # ── A USER'S REPO GETS CARD DELIVERABLES, NEVER PROCESS DATA ───────────
        # Checked FIRST: before container resolution and before `_ensure_fork`
        # (which can itself create a fork), so a refused path leaves no artefact.
        # Both sides of a rename are checked — the old path is written (deleted).
        assert_commit_paths_allowed(
            [p for change in request.changes for p in (change.file_path, change.old_file_path)],
            request.target_files,
        )

        # Resolve the container via the shared coding-containers client. The
        # refactoring DOING loop provisions the container through the
        # coding-containers subsystem (createContainer); apply-changes must
        # resolve it there (forwarding the project tenant), not in
        # coding-github's local docker_service registry — otherwise a
        # card-produced container 404s "Container not found" and no commit
        # lands, stranding the loop before QA.
        container = await _resolve_container(request.container_name, tenant_id)

        if not container.repo_url:
            raise not_found("Container", request.container_name)

        # Parse repository info
        repo_owner, repo_name = parse_github_url(container.repo_url)

        # Make sure we have valid owner and repo name
        if not repo_owner or not repo_name:
            raise bad_request(f"Invalid GitHub repository URL: {container.repo_url}")

        # Resolve the push/PR token: a CUSTOMER repo is written with the
        # customer's BYO/PAT from secrets-service, which the App cannot reach
        # (R1). A PLATFORM-OWNED repo gets None so the App authors — this is the
        # call that opened the branch, and the rung it lands on is the whole
        # difference between PR #85 and the 403 that followed it.
        container_token = await _effective_github_token(container, tenant_id, owner=repo_owner)
        github_integration = GitHubIntegration(
            github_token=container_token,
            target_owner=repo_owner,
            target_repo=repo_name,
            tenant_id=tenant_id,
        )

        # ── A PUBLIC REPO IS NEVER A CONVEYOR TARGET ───────────────────────────
        # Before `_ensure_fork` (which can itself create a fork): no artefact at all.
        _refuse_public_target(github_integration, repo_owner, repo_name)

        # Apply changes using GitHub integration internal methods
        repo_info = github_integration._ensure_fork(repo_owner, repo_name)
        fork_repo = repo_info.repository
        # `_ensure_fork` can REDIRECT the write to a different repo (the token user's
        # fork / clone). The upstream check above does not cover that target: a
        # private upstream with a PUBLIC fork would be written. Check what we write.
        assert_repo_not_public(
            fork_repo, getattr(fork_repo, "full_name", None) or f"{repo_owner}/{repo_name}"
        )
        _assert_can_write(
            fork_repo,
            _authoring_identity(github_integration),
            is_app=getattr(github_integration, "is_app_auth", False) is True,
        )
        branch_name = request.branch_name or f"code-changes-{uuid7().hex[:8]}"

        # ── NEVER COMMIT STRAIGHT ONTO THE DEFAULT BRANCH ──────────────────────
        # `branch_name="main"` used to find the ref already at the base SHA,
        # "reuse" it and commit onto main with no PR. Refused before anything
        # is written — including for an empty change set, so the refusal does
        # not depend on what the producer happened to generate. There is NO
        # override: no conveyor path legitimately writes the default branch,
        # and any body flag would be caller-settable (the gateway and n8n's
        # generic dispatch forward unknown keys unchanged).
        assert_branch_writable(fork_repo, branch_name)

        # ── AN EMPTY CHANGE SET MUST NOT MINT A REF ────────────────────────────
        # MEASURED 2026-09-14 on nonagenticai/enterprise_mcp_server: `AA Worker -
        # Generate Refactoring` errored at 03:17:23Z (container gone, 404), and
        # 11 seconds later a branch `refactor/break-cycle-aacyc-20260901-143226`
        # appeared — `compare main...branch` = ahead 0, behind 0, commits [],
        # files []. `_create_branch` ran before anything was known about the
        # changes, so a FAILED generator still left a plausible-looking artefact
        # in a real repository. `AA Maintenance - Blocked Sweep` re-drives that
        # card every 15 minutes, so the ref is re-minted on a timer.
        #
        # This is the dangerous direction of wrong: an empty ref reads as "the
        # run got somewhere" to anyone browsing branches, while the run in fact
        # produced nothing. A failed generator must leave NO artefact.
        #
        # The response shape is deliberately UNCHANGED from the committed == 0
        # case below, which callers already handle: `AA Activity - Action ::
        # Validate Typed Step` maps `no_changes === true` to a typed
        # 'no_changes' defect. Raising here instead would convert a graded,
        # typed outcome into an untyped node error and lose that handling.
        # `changes_applied` is 0 and `commit_sha` is None for the same reason
        # it is there — reporting a fabricated sha is what stranded the
        # refactoring loop behind GitHub's opaque "No commits between" 422.
        if not request.changes:
            return ApplyChangesResponse(
                status=ResponseStatus.SUCCESS,
                commit_sha=None,
                no_changes=True,
                message=(
                    "No changes were supplied, so no branch was created and no "
                    f"commit was produced. Branch '{branch_name}' does NOT exist "
                    "— an empty change set is a producer failure upstream of "
                    "this call, not a delivery outcome."
                ),
                branch_name=branch_name,
                changes_applied=0,
                fork_url=f"{_fork_html_url(fork_repo)}/tree/{branch_name}",
                request_id=request.container_name,
                authoring_identity=_authoring_identity(github_integration),
            )

        # Do NOT force-reset an existing branch to base on the apply path. Under
        # the Planka router fan-out (one card move -> ~5 concurrent DOING
        # dispatches), two apply-changes calls share ONE per-drive branch; a
        # force-reset (reset_existing_to_base=True) on the SECOND apply wipes the
        # FIRST's real refactor commit, transiently emptying the branch -> a
        # concurrent open_pull_request sees "No commits between" -> the card
        # strands in BLOCKED before QA (proven on two 2026-06-27 arch-refactor
        # drives). Preserving the sibling's commit is safe: an identical re-apply
        # is a benign no-op (committed == 0, handled below), and a genuinely
        # different change applies on top — the destructive reset is only ever
        # harmful here because the driver uses a UNIQUE per-drive branch name, so
        # the only reuse is intra-drive concurrent fan-out.
        github_integration._create_branch(
            fork_repo,
            branch_name,
            container.branch,
            reset_existing_to_base=False,
        )
        # Forward the FULL change operation. Rebuilding each change as a bare
        # {file_path, content} dropped change_type and old_file_path, and
        # _apply_changes reads change.get("change_type", "modified") — so a
        # DELETE silently became a modify that rewrote the file it was asked to
        # remove, and a RENAME lost its old path. `content` is intentionally
        # passed through as None for a delete: _apply_changes never reads it on
        # that branch, and the schema already refuses a content-less
        # add/modify/rename at the boundary.
        file_changes = [
            {
                "file_path": change.file_path,
                "content": change.content,
                "change_type": change.change_type,
                "old_file_path": change.old_file_path,
            }
            for change in request.changes
        ]
        commit_sha, committed = github_integration._apply_changes(
            fork_repo, branch_name, file_changes
        )

        # Get fork URL safely
        fork_url = _fork_html_url(fork_repo)

        # Zero-commit case: every requested change was a no-op (identical
        # content already on the branch), so nothing was committed. Reporting
        # SUCCESS with a fabricated commit_sha here is exactly what stranded the
        # AA refactoring loop — the downstream create-pull-request then fails
        # with GitHub's opaque "No commits between" 422. Surface a TYPED benign
        # signal (no_changes=True, no commit_sha, changes_applied=0) so the
        # caller treats "nothing to commit" as benign, not a false commit.
        if committed == 0:
            return ApplyChangesResponse(
                status=ResponseStatus.SUCCESS,
                commit_sha=None,
                no_changes=True,
                message=(
                    f"No changes to apply on branch '{branch_name}': all "
                    f"{len(request.changes)} requested change(s) were no-ops "
                    "(identical content); no commit was produced."
                ),
                branch_name=branch_name,
                changes_applied=0,
                fork_url=f"{fork_url}/tree/{branch_name}",
                request_id=request.container_name,
                authoring_identity=_authoring_identity(github_integration),
            )

        return ApplyChangesResponse(
            status=ResponseStatus.SUCCESS,
            commit_sha=commit_sha,
            no_changes=False,
            message=f"Applied {committed} change(s) to branch '{branch_name}'",
            branch_name=branch_name,
            changes_applied=committed,
            fork_url=f"{fork_url}/tree/{branch_name}",
            request_id=request.container_name,
            authoring_identity=_authoring_identity(github_integration),
        )

    except HTTPException:
        raise
    except RepoVisibilityRefusedError as e:
        raise visibility_refused_http(e) from e
    except CommitPathRefusedError as e:
        # 422 with the code repeated in ``details`` (coding-client's shared handler
        # keeps only message + details on the wire).
        raise error_response(
            status_code=422,
            message=str(e),
            error_code=e.error_code,
            details={"error_code": e.error_code, "path": e.path},
        ) from e
    except DefaultBranchWriteRefusedError as e:
        raise _default_branch_refused(e) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except BaseRefNotFoundError as e:
        # The workspace's base does not exist on the repo. Cutting the branch
        # from the default branch instead would apply this diff to code it was
        # never written against, so this is the caller's error, not ours.
        raise bad_request(str(e)) from e
    except RuntimeError:
        # Containers client not initialised → treat as container not found
        raise not_found("Container", request.container_name)
    except Exception as e:
        raise internal_error(f"Failed to apply changes: {str(e)}") from e


@router.post(
    "/create-pull-request",
    operation_id="create_pull_request",
    dependencies=[Depends(require_andon_released)],
    response_model=CreatePRResponse,
    summary="Create Pull Request",
    description="""
    Create a professional pull request from your fork back to the original repository.

    **Usage:**
    ```json
    {
        "original_owner": "fastapi",
        "original_repo": "fastapi",
        "fork_owner": "yourusername",
        "fork_repo": "fastapi",
        "head_branch": "feature/updates",
        "base_branch": "master",
        "title": "Add new authentication feature",
        "message": "## Changes\\n- Added OAuth2 support\\n- Updated documentation\\n\\nFixes #123"
    }
    ```

    **Required fields:**
    - Repository details (original_owner, original_repo, fork_owner, fork_repo)
    - Branch configuration (head_branch, base_branch)
    - PR title

    **Optional:**
    - Detailed description with markdown support
    - Issue references (`Resolves #123`, `Closes #456`)

    **Response includes:**
    - `pr_number` - GitHub PR number
    - `pr_url` - Direct link to view/manage PR
    - `pr_status` - Current status

    **Perfect for contributing updates back to open source projects.**
    """,
    responses={
        200: {
            "model": CreatePRResponse,
            "description": "Pull request successfully created",
        },
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Repository or branch not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - invalid request body",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def create_pull_request(
    request: CreatePRRequest, tenant_id: str = Depends(require_tenant)
) -> CreatePRResponse:
    """Create a pull request from the fork back to the original repository."""
    try:
        # Validate basic request format first
        if not all(
            [
                request.original_owner,
                request.original_repo,
                request.fork_owner,
                request.fork_repo,
                request.head_branch,
                request.base_branch,
                request.title,
                request.message,
            ]
        ):
            raise bad_request("Missing required fields for PR creation")

        # Additional validation for field content
        # Check for empty strings or invalid values
        required_fields = [
            request.original_owner.strip(),
            request.original_repo.strip(),
            request.fork_owner.strip(),
            request.fork_repo.strip(),
            request.head_branch.strip(),
            request.base_branch.strip(),
            request.title.strip(),
            request.message.strip(),
        ]

        if not all(required_fields):
            raise bad_request("Required fields cannot be empty")

        # There is deliberately NO "test repository" short-circuit here. This
        # route used to answer HTTP 200 with a fabricated ``pr_number: 123`` —
        # "created successfully (test mode)" — whenever the owner/repo LOOKED
        # like a placeholder (``test-owner``, ``example-repo``, ``"0"``) or
        # both were <=2 characters. Nothing was created, and the response was
        # indistinguishable from a real success. ``ab/cd`` is a legal GitHub
        # repository, so a real caller could be lied to. A caller that asks for
        # a PR either gets one from GitHub or gets an error; tests mock GitHub.
        github_integration = GitHubIntegration(
            target_owner=request.original_owner,
            target_repo=request.original_repo,
            tenant_id=tenant_id,
        )

        # Log for debugging
        logger.info(
            "Creating PR from %s/%s:%s to %s/%s:%s",
            request.fork_owner,
            request.fork_repo,
            request.head_branch,
            request.original_owner,
            request.original_repo,
            request.base_branch,
        )

        # Use fork_repo field from the request model
        # Construct head reference using both fork_owner and fork_repo
        head_branch_ref = f"{request.fork_owner}:{request.head_branch}"

        # ── A PUBLIC REPO IS NEVER A CONVEYOR TARGET ───────────────────────────
        # The BASE repo is the target. Checked before create_pull_request, which
        # can fork and cut/reset a branch before it ever opens the PR.
        _refuse_public_target(github_integration, request.original_owner, request.original_repo)

        # Create pull request using existing method
        pr_result = github_integration.create_pull_request(
            repo_owner=request.original_owner,
            repo_name=request.original_repo,
            title=request.title,
            message=request.message,
            head_branch=head_branch_ref,
            base_branch=request.base_branch,
            draft=request.draft,
        )

        if not pr_result.get("success"):
            raise bad_request(f"Failed to create PR: {pr_result.get('error', 'Unknown error')}")

        # Benign empty-diff case: there were no commits between head and base, so
        # nothing to open a PR for. Surface a typed no-op success (no PR number)
        # rather than 500-ing on the response model — this is what keeps the AA
        # refactoring loop from hard-failing on GitHub's "No commits between" 422.
        if pr_result.get("no_commits"):
            return CreatePRResponse(
                message=pr_result.get(
                    "message", "No commits between branches; no pull request opened"
                ),
                pr_number=None,
                pr_url=None,
                pr_status="no_commits",
                no_changes=True,
                draft=request.draft,
                status=ResponseStatus.SUCCESS,
                request_id=f"{request.fork_owner}-{request.head_branch}",
            )

        pr_is_draft = bool(pr_result.get("draft", request.draft))
        return CreatePRResponse(
            message="Pull request created successfully",
            pr_number=pr_result["pr_number"],
            pr_url=pr_result["pr_url"],
            pr_status="draft" if pr_is_draft else "open",
            draft=pr_is_draft,
            status=ResponseStatus.SUCCESS,
            request_id=f"{request.fork_owner}-{request.head_branch}",
        )

    except HTTPException:
        raise
    except RepoVisibilityRefusedError as e:
        raise visibility_refused_http(e) from e
    except DefaultBranchWriteRefusedError as e:
        raise _default_branch_refused(e) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        raise internal_error(f"Failed to create pull request: {str(e)}") from e


@router.post(
    "/mark-pull-request-ready",
    operation_id="mark_pull_request_ready",
    dependencies=[Depends(require_andon_released)],
    response_model=MarkPRReadyResponse,
    summary="Mark Pull Request Ready For Review",
    description="""
    Flip a draft pull request to ready-for-review.

    Used by the AA delivery pipeline to publish a PR that was opened as a draft
    during DOING, once QA is green at DONE. Idempotent — a no-op (with
    `already_ready=true`) if the PR is already ready for review.

    **Usage:**
    ```json
    {
        "repo_owner": "nonagenticai",
        "repo_name": "enterprise_mcp_server",
        "pr_number": 42
    }
    ```
    """,
    responses={
        200: {"model": MarkPRReadyResponse, "description": "Pull request marked ready for review"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Repository or pull request not found"},
        406: {"model": Error406Response, "description": "Not acceptable - invalid Accept header"},
        422: {"model": ValidationErrorResponse, "description": "Validation error - invalid body"},
        429: {"model": Error429Response, "description": "Too many requests - rate limit exceeded"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def mark_pull_request_ready(
    request: MarkPRReadyRequest, tenant_id: str = Depends(require_tenant)
) -> MarkPRReadyResponse:
    """Mark a draft pull request ready for review (idempotent)."""
    try:
        github_integration = GitHubIntegration(
            target_owner=request.repo_owner,
            target_repo=request.repo_name,
            tenant_id=tenant_id,
        )
        result = github_integration.mark_pull_request_ready(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name,
            pr_number=request.pr_number,
        )
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if result.get("status_code") == 404 or "not found" in str(error).lower():
                raise not_found(
                    "Pull request",
                    f"#{request.pr_number} in {request.repo_owner}/{request.repo_name}",
                )
            raise bad_request(f"Failed to mark PR ready: {error}")

        return MarkPRReadyResponse(
            status=ResponseStatus.SUCCESS,
            message=(
                "Pull request already ready for review"
                if result.get("already_ready")
                else "Pull request marked ready for review"
            ),
            pr_number=result["pr_number"],
            pr_url=result["pr_url"],
            pr_status=result.get("pr_status", "open"),
            draft=bool(result.get("draft", False)),
            already_ready=bool(result.get("already_ready", False)),
            request_id=f"{request.repo_owner}-{request.repo_name}-{request.pr_number}",
        )

    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        raise internal_error(f"Failed to mark pull request ready: {str(e)}") from e


@router.post(
    "/update-pull-request-body",
    operation_id="update_pull_request_body",
    dependencies=[Depends(require_andon_released)],
    response_model=UpdatePRBodyResponse,
    summary="Update Pull Request Body",
    description="""
    Replace the description of an open pull request.

    Used by the bug-hunt flow, which opens ONE shared PR for several fixes and
    rewrites its description to list every fixed bug before asking a human to
    approve the merge. Idempotent — a no-op (with `changed=false`) if the body
    is already identical. A PR that is not open is refused with 409; a public
    repository is refused with 403.

    **Usage:**
    ```json
    {
        "repo_owner": "nonagenticai",
        "repo_name": "enterprise_mcp_server",
        "pr_number": 42,
        "body": "## Fixed bugs\\n- ..."
    }
    ```
    """,
    responses={
        200: {"model": UpdatePRBodyResponse, "description": "Pull request body updated"},
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Repository or pull request not found"},
        406: {"model": Error406Response, "description": "Not acceptable - invalid Accept header"},
        409: {"model": Error409Response, "description": "Pull request is not open"},
        422: {"model": ValidationErrorResponse, "description": "Validation error - invalid body"},
        429: {"model": Error429Response, "description": "Too many requests - rate limit exceeded"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def update_pull_request_body(
    request: UpdatePRBodyRequest, tenant_id: str = Depends(require_tenant)
) -> UpdatePRBodyResponse:
    """Replace an open pull request's description (idempotent)."""
    try:
        github_integration = GitHubIntegration(
            target_owner=request.repo_owner,
            target_repo=request.repo_name,
            tenant_id=tenant_id,
        )
        result = github_integration.update_pull_request_body(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name,
            pr_number=request.pr_number,
            body=request.body,
        )
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if result.get("status_code") == 404 or "not found" in str(error).lower():
                raise not_found(
                    "Pull request",
                    f"#{request.pr_number} in {request.repo_owner}/{request.repo_name}",
                )
            if result.get("status_code") == 409:
                raise conflict("Pull request", str(error))
            raise bad_request(f"Failed to update PR body: {error}")

        changed = bool(result.get("changed", False))
        return UpdatePRBodyResponse(
            status=ResponseStatus.SUCCESS,
            message="Pull request body updated" if changed else "Pull request body unchanged",
            pr_number=result["pr_number"],
            pr_url=result["pr_url"],
            pr_status=result.get("pr_status", "open"),
            changed=changed,
            request_id=f"{request.repo_owner}-{request.repo_name}-{request.pr_number}",
        )

    except HTTPException:
        raise
    except RepoVisibilityRefusedError as e:
        raise visibility_refused_http(e) from e
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        raise internal_error(f"Failed to update pull request body: {str(e)}") from e


@router.post(
    "/validate-pull-request",
    operation_id="validate_pull_request",
    response_model=ValidatePRResponse,
    summary="Validate Pull Request",
    description="""
    Comprehensive automated validation of GitHub pull requests for quality, security, and performance.

    **Full validation:**
    ```json
    {
        "repo_owner": "fastapi",
        "repo_name": "fastapi",
        "pr_number": 142,
        "validation_types": ["security", "performance", "quality"]
    }
    ```

    **Security only:**
    ```json
    {
        "repo_owner": "myorg",
        "repo_name": "api",
        "pr_number": 89,
        "validation_types": ["security"]
    }
    ```

    **Validation types:**
    - `security` - Vulnerability scanning, secure patterns, authentication issues
    - `performance` - Algorithm efficiency, resource usage, complexity analysis
    - `quality` - Code style, best practices, maintainability, README validation
    - `comprehensive` - Advanced multi-dimensional analysis with risk scoring
    - `advanced` - Comprehensive analysis with semantic validation and impact assessment

    **Response includes:**
    - `overall_score` (0-100) and pass/fail status
    - `recommendation` (`approve`, `needs_work`, `reject`)
    - Detailed issues with severity levels
    - Actionable update suggestions

    **Perfect for:**
    - Automated code review in CI/CD
    - Quality gates before merging
    - Learning and update feedback
    """,
    responses={
        200: {
            "model": ValidatePRResponse,
            "description": "Pull request validation completed",
        },
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Repository or pull request not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - invalid request body",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def validate_pull_request(
    request: ValidatePRRequest,
    tenant_id: str = Depends(require_tenant),
) -> ValidatePRResponse:
    """Validate a pull request for quality, security, and performance."""
    try:
        # Validate input parameters
        if not request.repo_owner or not request.repo_owner.strip():
            raise bad_request("repo_owner is required and cannot be empty")

        if not request.repo_name or not request.repo_name.strip():
            raise bad_request("repo_name is required and cannot be empty")

        if not request.pr_number or request.pr_number <= 0:
            raise bad_request("pr_number must be a positive integer")

        github_integration = GitHubIntegration(
            target_owner=request.repo_owner,
            target_repo=request.repo_name,
            tenant_id=tenant_id,
            permissions=PERMS_PR_READ,
        )

        # Check GitHub client availability
        if not github_integration.github_client:
            raise internal_error("GitHub client not available - check authentication configuration")

        # Get repository and PR with proper error handling
        try:
            repo = github_integration.github_client.get_repo(
                f"{request.repo_owner}/{request.repo_name}"
            )
        except Exception as e:
            # Handle GitHub API errors for repository access
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                raise NotFoundException(
                    message=(
                        f"Repository '{request.repo_owner}/{request.repo_name}' not found or not accessible"
                    ),
                    error_code="REPO_NOT_FOUND",
                    resource_type="repository",
                    resource_id=(f"{request.repo_owner}/{request.repo_name}"),
                ) from e
            if "403" in error_msg or "forbidden" in error_msg:
                raise PermissionException(
                    message=(
                        f"Access denied to repository '{request.repo_owner}/{request.repo_name}'"
                    ),
                    error_code="REPO_ACCESS_DENIED",
                    details={
                        "resource_type": "repository",
                        "resource_id": (f"{request.repo_owner}/{request.repo_name}"),
                    },
                ) from e
            raise InternalServerException(
                message=f"GitHub API error accessing repository: {str(e)}",
                error_code="GITHUB_API_ERROR",
            ) from e

        try:
            pr = repo.get_pull(request.pr_number)
        except Exception as e:
            # Handle GitHub API errors for PR access
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                raise not_found(
                    "Pull request",
                    f"#{request.pr_number} in repository '{request.repo_owner}/{request.repo_name}'",
                ) from e
            raise internal_error(f"GitHub API error accessing pull request: {str(e)}") from e

        # Get PR files and changes with error handling
        try:
            pr_files = pr.get_files()
        except Exception as e:
            raise internal_error(f"Failed to retrieve pull request files: {str(e)}") from e

        # Default to standard validation types if none specified
        validation_types = request.validation_types or [
            "security",
            "performance",
            "quality",
        ]

        # Try comprehensive validation first
        validations = []
        total_score = 0.0

        try:
            # Check if comprehensive validation should be used
            use_comprehensive = (
                len(validation_types) > 2
                or "comprehensive" in validation_types
                or "advanced" in validation_types
            )

            if use_comprehensive:
                logger.info("Using comprehensive PR validation")
                validations = await validate_pr_comprehensive(request, list(pr_files))
                total_score = sum(v.score for v in validations)
            else:
                # Use legacy validation
                logger.info("Using legacy PR validation")
                for validation_type in validation_types:
                    if validation_type == "security":
                        result = await _validate_pr_security(pr_files)
                    elif validation_type == "performance":
                        result = await _validate_pr_performance(pr_files)
                    elif validation_type == "quality":
                        result = await _validate_pr_quality(pr_files)
                    else:
                        continue

                    result.validation_type = validation_type
                    validations.append(result)
                    total_score += result.score

        except ImportError as e:
            logger.warning("Comprehensive validation not available, using legacy: %s", e)
            # Fallback to legacy validation
            for validation_type in validation_types:
                if validation_type == "security":
                    result = await _validate_pr_security(pr_files)
                elif validation_type == "performance":
                    result = await _validate_pr_performance(pr_files)
                elif validation_type == "quality":
                    result = await _validate_pr_quality(pr_files)
                else:
                    continue

                result.validation_type = validation_type
                validations.append(result)
                total_score += result.score
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Comprehensive validation failed, using legacy: %s", e)
            # Fallback to legacy validation
            for validation_type in validation_types:
                if validation_type == "security":
                    result = await _validate_pr_security(pr_files)
                elif validation_type == "performance":
                    result = await _validate_pr_performance(pr_files)
                elif validation_type == "quality":
                    result = await _validate_pr_quality(pr_files)
                else:
                    continue

                result.validation_type = validation_type
                validations.append(result)
                total_score += result.score

        overall_score = total_score / len(validations) if validations else 0.0
        passed = all(v.passed for v in validations)

        # Determine recommendation
        if overall_score >= 80 and passed:
            recommendation = "approve"
        elif overall_score >= 60:
            recommendation = "needs_work"
        else:
            recommendation = "reject"

        return ValidatePRResponse(
            status=ResponseStatus.SUCCESS,
            message="Pull request validation completed",
            overall_score=overall_score,
            passed=passed,
            validations=validations,
            recommendation=recommendation,
            request_id=f"pr-{request.pr_number}",
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes and error messages
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except Exception as e:
        # Catch any other unexpected exceptions
        raise internal_error(f"Unexpected error during pull request validation: {str(e)}") from e


# Vulture whitelist for FastAPI route handlers (not directly called)
_vulture_whitelist = [
    "fork_repository",
    "apply_code_changes",
    "create_pull_request",
    "mark_pull_request_ready",
    "update_pull_request_body",
    "validate_pull_request",
]
