"""Tests for PR operations endpoints via real HTTP requests.

Covers fork, apply-changes, create-pull-request, validate-pull-request.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BASE = "/api/repositories"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container_payload(**overrides):
    """A coding-containers GET envelope — what the shared client returns.

    fork / apply-changes now resolve the container through the unified
    coding-containers client (not coding-github's local docker_service), so
    the test double is the wrapped GET payload.
    """
    container = {
        "container_name": "test-container",
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "status": "running",
        "github_access_token": None,
    }
    container.update(overrides)
    return {"status": "success", "message": "ok", "container": container}


def _mock_workspaces_client(get_result=None, error=None):
    """Mock the shared coding-containers client used for unified resolution."""
    client = MagicMock()
    if error is not None:
        client.get_workspace = AsyncMock(side_effect=error)
    else:
        client.get_workspace = AsyncMock(
            return_value=get_result if get_result is not None else _make_container_payload()
        )
    return client


def _mock_fork_repo():
    fork = MagicMock()
    fork.html_url = "https://github.com/fork-owner/repo"
    fork.owner = SimpleNamespace(login="fork-owner")
    fork.name = "repo"
    fork.private = True  # the resolved WRITE target is private too (#118)
    return fork


# ===========================================================================
# Fork tests
# ===========================================================================


class TestForkSuccess:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_success(self, mock_gh_cls, mock_get_client, async_client):
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "test-container"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["fork_owner"] == "fork-owner"
        assert body["status"] == "success"


class TestForkNotFound:
    @patch("main.get_workspaces_client")
    async def test_fork_not_found(self, mock_get_client, async_client):
        from coding_client.clients import ServiceError

        mock_get_client.return_value = _mock_workspaces_client(
            error=ServiceError(404, "container not found")
        )

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "ghost-container"})
        assert resp.status_code == 404


class TestForkPermissionError:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_permission_error(self, mock_gh_cls, mock_get_client, async_client):
        mock_get_client.return_value = _mock_workspaces_client()

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.side_effect = Exception("403 Forbidden")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "test-container"})
        assert resp.status_code == 500


class TestForkRuntimeError:
    @patch("main.get_workspaces_client")
    async def test_fork_runtime_error_returns_404(self, mock_get_client, async_client):
        """RuntimeError from an uninitialised containers client returns 404."""
        mock_get_client.side_effect = RuntimeError("client not init")

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "test-container"})
        assert resp.status_code == 404


class TestForkMissingToken:
    """Regression: fork without a GitHub token returns 412 (was 500 pre-fix).

    The container exists, but carries no `container.github_access_token` (and
    no PAT fallback exists), so GitHubIntegration._ensure_fork raises
    GitHubAuthError. The handler must surface that as HTTP 412 with the
    actionable hint, not a generic 500.
    """

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_no_token_returns_412(self, mock_gh_cls, mock_get_client, async_client):
        from services.github_integration import GitHubAuthError

        mock_get_client.return_value = _mock_workspaces_client()

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.side_effect = GitHubAuthError(
            "No GitHub token configured. Supply 'user_token' in the request "
            "or install the GitHub App for this tenant (no PAT fallback exists)."
        )
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "test-container"})
        assert resp.status_code == 412
        # coding-client's exception handler wraps HTTPException.detail as
        # {"error": {"code": "HTTP_412", "message": ...}}.
        body = resp.json()
        assert body["error"]["code"] == "HTTP_412"
        assert "GitHub token" in body["error"]["message"]


# ===========================================================================
# Apply changes tests
# ===========================================================================


class TestApplyChangesSuccess:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_changes_success(self, mock_gh_cls, mock_get_client, async_client):
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha123", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/main.py", "content": "print('hi')"}],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["changes_applied"] == 1
        assert body["branch_name"] == "feature-branch"
        assert body["commit_sha"] == "sha123"
        assert body["no_changes"] is False


class TestApplyChangesPreservesConcurrentBranch:
    """Under the Planka router fan-out, two concurrent apply-changes calls share
    ONE per-drive refactor branch. The apply path MUST NOT force-reset an existing
    branch to base (reset_existing_to_base=False) — a force-reset on the second
    apply wipes the first's real refactor commit, transiently emptying the branch
    so a concurrent open_pull_request sees 'No commits between' and the card
    strands in BLOCKED before QA. Offline twin of the 2026-06-27 concurrent-apply
    clobber (§4.4)."""

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_does_not_force_reset_existing_branch(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha123", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/main.py", "content": "print('hi')"}],
                "branch_name": "refactor/break-cycle-arch-refactor-drive-1",
            },
        )
        assert resp.status_code == 200
        # The apply path preserves a sibling apply's commit instead of resetting.
        assert mock_instance._create_branch.call_args.kwargs["reset_existing_to_base"] is False


class TestApplyChangesNoOp:
    """An identical-content apply commits nothing. The endpoint must surface a
    TYPED no_changes signal (not a fabricated SUCCESS with a real commit_sha) so
    the caller treats 'nothing to commit' as benign — the empty-diff path that
    otherwise strands the AA refactoring loop on create-PR's 'No commits between'."""

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_changes_noop_returns_no_changes(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        # Zero commits landed: every change was an identical-content no-op.
        mock_instance._apply_changes.return_value = ("", 0)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/main.py", "content": "unchanged"}],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Still SUCCESS (benign), but honestly typed as "no changes".
        assert body["status"] == "success"
        assert body["no_changes"] is True
        # No fabricated commit sha, and the count is honest (0, not len(changes)).
        assert body["commit_sha"] is None
        assert body["changes_applied"] == 0
        assert body["branch_name"] == "feature-branch"


class TestApplyChangesPreservesChangeType:
    """The router must forward the FULL change operation to ``_apply_changes``.

    Regression guard for the conveyor's "cannot delete or rename a file" defect:
    the route used to rebuild every inbound change as a bare
    ``{"file_path", "content"}``, dropping ``change_type`` and
    ``old_file_path``. ``GitHubIntegration._apply_changes`` then read
    ``change.get("change_type", "modified")`` — so a delete arrived as a MODIFY
    and silently rewrote the file it was asked to remove, and a rename lost the
    old path and became an add.

    ⚠️ These assertions must go through the ROUTE. Exercising ``_apply_changes``
    directly cannot see this bug: the service layer has supported all four
    change types the whole time; the router was the only thing discarding them.
    """

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_delete_reaches_apply_changes_as_deleted(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """A delete has NO content — n8n's generate-code-fix omits the field
        entirely — and must still arrive as change_type='deleted'."""
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("shaDEL", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/dead_code.py", "change_type": "deleted"}],
                "branch_name": "feature-branch",
            },
        )
        # A content-less delete is a VALID request: content is required for
        # add/modify/rename, never for a delete.
        assert resp.status_code == 200, resp.text

        forwarded = mock_instance._apply_changes.call_args.args[2]
        assert forwarded == [
            {
                "file_path": "src/dead_code.py",
                "content": None,
                "change_type": "deleted",
                "old_file_path": None,
            }
        ]
        # _apply_changes reads change.get("change_type", "modified"): anything
        # other than the literal "deleted" is silently a modify.
        assert forwarded[0]["change_type"] == "deleted"

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_rename_reaches_apply_changes_with_old_file_path(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """A rename carries old_file_path + the post-rename content; dropping
        old_file_path makes _apply_changes raise, or degrades it to an add."""
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("shaMV", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [
                    {
                        "file_path": "src/new_name.py",
                        "old_file_path": "src/old_name.py",
                        "content": "print('moved')",
                        "change_type": "renamed",
                    }
                ],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 200, resp.text

        forwarded = mock_instance._apply_changes.call_args.args[2]
        assert forwarded == [
            {
                "file_path": "src/new_name.py",
                "content": "print('moved')",
                "change_type": "renamed",
                "old_file_path": "src/old_name.py",
            }
        ]

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_default_change_type_is_still_modified(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """A change that names no change_type keeps the historical 'modified'
        default, so existing callers are unaffected by the fix."""
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha123", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/main.py", "content": "print('hi')"}],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 200, resp.text

        forwarded = mock_instance._apply_changes.call_args.args[2]
        assert forwarded[0]["change_type"] == "modified"
        assert forwarded[0]["old_file_path"] is None

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_missing_content_on_modify_is_still_rejected(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """Relaxing `content` for deletes must NOT make it optional for the
        types that genuinely need it — _apply_changes raises ValueError on a
        content-less add/modify, and that belongs at the boundary as a 422, not
        as a 500 from deep inside the service."""
        mock_get_client.return_value = _mock_workspaces_client()
        mock_gh_cls.return_value = MagicMock()

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/main.py", "change_type": "modified"}],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 422, resp.text
        mock_gh_cls.return_value._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_rename_without_old_file_path_is_rejected(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """A rename with no old_file_path cannot be executed: _apply_changes
        raises 'old_file_path required for rename operation'. Refuse it at the
        boundary instead of surfacing a 500."""
        mock_get_client.return_value = _mock_workspaces_client()
        mock_gh_cls.return_value = MagicMock()

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [
                    {
                        "file_path": "src/new_name.py",
                        "content": "print('moved')",
                        "change_type": "renamed",
                    }
                ],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 422, resp.text
        mock_gh_cls.return_value._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_unrecognised_change_type_is_refused_not_written(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """An unrecognised ``change_type`` must be refused at the boundary.

        ``_apply_changes`` branches on the four exact strings and ends in an
        ``else`` that treats ANYTHING else as a content write
        (``_create_or_update_file``). ``change_type`` carried only
        ``pattern=SAFE_STRING_PATTERN`` — ``^[\\s\\S]*$``, which every string
        satisfies — so the singular ``"delete"`` validated, fell to that
        ``else``, and OVERWROTE the file it was asked to remove. Silently: the
        else branch only logs a warning.

        This is reachable, not theoretical. The change spec is LLM-generated and
        ``AA_IMPLEMENTATION/doing_action_procedures.json`` forwards
        ``generate_code_change_spec.output.changes`` verbatim; n8n's own worker
        prompt has to warn the model that ``"delete"``, ``"remove"`` and
        ``"removed"`` are NOT recognised, which is precisely the spelling a
        model reaches for.

        Destroying a file is not a defensible response to a typo — refuse it.
        """
        mock_get_client.return_value = _mock_workspaces_client()
        mock_gh_cls.return_value = MagicMock()

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [
                    {
                        "file_path": "src/dead_code.py",
                        "change_type": "delete",
                        "content": "# this content must never be written",
                    }
                ],
                "branch_name": "feature-branch",
            },
        )
        assert resp.status_code == 422, resp.text
        # The refusal must name the offending field, not be an opaque 422.
        assert "change_type" in resp.text
        # The load-bearing half: the write never happened.
        mock_gh_cls.return_value._apply_changes.assert_not_called()


class TestApplyChangesContainerNotFound:
    @patch("main.get_workspaces_client")
    async def test_container_not_found_returns_404(self, mock_get_client, async_client):
        from coding_client.clients import ServiceError

        mock_get_client.return_value = _mock_workspaces_client(
            error=ServiceError(404, "container not found")
        )

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "ghost",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 404


class TestUnifiedContainerResolution:
    """The fork / apply container is resolved through the shared
    coding-containers client (k8s subsystem), forwarding the project tenant —
    NOT coding-github's local docker_service registry. Regression guard for
    the architecture_refactoring "produce a PR" blocker (#557)."""

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_resolves_via_unified_client_and_forwards_tenant(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        client = _mock_workspaces_client()
        mock_get_client.return_value = client

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("shaABC", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "card-produced-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 200
        # Resolved via the shared client, not the local docker_service.
        client.get_workspace.assert_awaited_once()
        call = client.get_workspace.await_args
        assert call.args[0] == "card-produced-container"
        # The require_tenant-resolved tenant is forwarded so coding-containers'
        # tenant-scoped GET resolves the container (the test harness resolves
        # it to "test-tenant").
        # Case-insensitive, content-based: these headers are now built by
        # services.outbound.optional_sibling_headers, which emits coding-client's
        # canonical `X-Tenant-Id` spelling and also attaches this service's own
        # licence (coding-containers stopped trusting a bare tenant header, which
        # was 401ing the PR path). Pinning the exact dict froze the old hand-rolled
        # spelling and said nothing about the credential.
        sent = call.kwargs["headers"] or {}
        assert {k.lower(): v for k, v in sent.items()}["x-tenant-id"] == "test-tenant"

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_unresolvable_base_returns_400_and_commits_nothing(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        from services.github_integration import BaseRefNotFoundError

        mock_get_client.return_value = _mock_workspaces_client()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=_mock_fork_repo())
        mock_instance._create_branch.side_effect = BaseRefNotFoundError(
            "Base branch 'release-9' not found on o/r"
        )
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "card-produced-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 400
        assert "release-9" in resp.text
        mock_instance._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    async def test_apply_service_error_non_404_returns_500(self, mock_get_client, async_client):
        from coding_client.clients import ServiceError

        mock_get_client.return_value = _mock_workspaces_client(
            error=ServiceError(503, "containers down")
        )

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "x",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 500


class TestByoTokenResolution:
    """R1 regression: the commit/PR path must use the project's BYO PAT.

    coding-containers never persists a plaintext token (``github_access_token``
    is always None); the real token lives in secrets-service behind
    ``secret_config_id``. If apply-changes/fork don't resolve it they fall back
    to the GitHub App / system token, which has no write access to a customer's
    private repo → "Repository not found" 500 and no PR. These guard that the
    resolved BYO token is passed to GitHubIntegration, and that absence of a
    secret cleanly falls back to the platform credential (own-repo path)."""

    @patch("services.secret_resolver.resolve_git_token", new_callable=AsyncMock)
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_uses_resolved_byo_token(
        self, mock_gh_cls, mock_get_client, mock_resolve, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client(
            get_result=_make_container_payload(github_access_token=None, secret_config_id="sc-123")
        )
        mock_resolve.return_value = "byo-pat-xyz"

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=_mock_fork_repo())
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha777", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "byo-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
                "branch_name": "aa/diag",
            },
        )
        assert resp.status_code == 200
        # Resolved from secrets-service with (secret_config_id, project tenant)...
        mock_resolve.assert_awaited_once_with("sc-123", "test-tenant")
        # ...and handed to GitHubIntegration instead of the App/system fallback.
        assert mock_gh_cls.call_args.kwargs.get("github_token") == "byo-pat-xyz"

    @patch("services.secret_resolver.resolve_git_token", new_callable=AsyncMock)
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_no_secret_falls_back_to_platform_token(
        self, mock_gh_cls, mock_get_client, mock_resolve, async_client
    ):
        # No inline token and no secret → resolver returns None → GitHubIntegration
        # gets github_token=None and uses the platform credential (own/EXEC path).
        mock_get_client.return_value = _mock_workspaces_client()
        mock_resolve.return_value = None

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=_mock_fork_repo())
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha1", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "own-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 200
        assert mock_gh_cls.call_args.kwargs.get("github_token") is None


# ===========================================================================
# Create PR tests
# ===========================================================================


_PLACEHOLDER_NAMES = [
    ("test-owner", "test-repo"),
    ("original-owner", "awesome-project"),
    ("example-owner", "example-repo"),
    ("0", "0"),
    # both <=2 chars: a perfectly legal GitHub repository, which the removed
    # short-circuit also answered with a fabricated PR.
    ("ab", "cd"),
]


def _pr_body(owner: str, repo: str, **extra):
    return {
        "original_owner": owner,
        "original_repo": repo,
        "fork_owner": "fork-user",
        "fork_repo": repo,
        "head_branch": "feature/x",
        "base_branch": "main",
        "title": "My PR",
        "message": "Some changes",
        **extra,
    }


class TestCreatePRNeverFabricates:
    """There is no "test mode": placeholder-looking names go to GitHub like any
    other, and a failure is an error — never a 200 with an invented PR."""

    @pytest.mark.parametrize(("owner", "repo"), _PLACEHOLDER_NAMES)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_placeholder_names_reach_github(self, mock_gh_cls, owner, repo, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": True,
            "pr_number": 7,
            "pr_url": f"https://github.com/{owner}/{repo}/pull/7",
            "draft": False,
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/create-pull-request", json=_pr_body(owner, repo))

        assert resp.status_code == 200
        body = resp.json()
        assert body["pr_number"] == 7
        assert "test mode" not in body["message"]
        mock_instance.create_pull_request.assert_called_once()

    @pytest.mark.parametrize(("owner", "repo"), _PLACEHOLDER_NAMES)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_placeholder_names_failure_is_an_error(
        self, mock_gh_cls, owner, repo, async_client
    ):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": False,
            "error": "Not Found",
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/create-pull-request", json=_pr_body(owner, repo))

        assert resp.status_code >= 400
        assert "123" not in resp.text

    @pytest.mark.parametrize(("owner", "repo"), _PLACEHOLDER_NAMES)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_placeholder_names_exception_is_an_error(
        self, mock_gh_cls, owner, repo, async_client
    ):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.side_effect = RuntimeError("boom")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request", json=_pr_body(owner, repo, draft=True)
        )

        assert resp.status_code == 500
        assert "pr_number" not in resp.json()


class TestCreatePRDraft:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_threads_draft_to_service(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": True,
            "pr_number": 7,
            "pr_url": "https://github.com/real-owner/real-repo/pull/7",
            "draft": True,
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "feature/x",
                "base_branch": "main",
                "title": "PR Title",
                "message": "Description",
                "draft": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["draft"] is True
        assert mock_instance.create_pull_request.call_args.kwargs["draft"] is True

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_no_commits_is_benign(self, mock_gh_cls, async_client):
        """An empty-diff 'No commits between' result is benign at the HTTP
        surface too: 200 success, no PR number, no_changes flagged — the loop
        is not hard-failed by a 500 on the response model."""
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": True,
            "no_commits": True,
            "pr_number": None,
            "pr_url": None,
            "branch_name": "fork-user:feature/x",
            "message": "No commits between main and fork-user:feature/x; no pull request opened",
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "feature/x",
                "base_branch": "main",
                "title": "PR Title",
                "message": "Description",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["no_changes"] is True
        assert body["pr_number"] is None
        assert body["pr_url"] is None
        assert body["pr_status"] == "no_commits"


class TestMarkPullRequestReady:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_mark_ready_success(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.mark_pull_request_ready.return_value = {
            "success": True,
            "pr_number": 7,
            "pr_url": "https://github.com/owner/repo/pull/7",
            "draft": False,
            "pr_status": "open",
            "already_ready": False,
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/mark-pull-request-ready",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 7},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["draft"] is False
        assert body["already_ready"] is False

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_mark_ready_not_found(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.mark_pull_request_ready.return_value = {
            "success": False,
            "error": "GitHub API error: Not Found",
            "status_code": 404,
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/mark-pull-request-ready",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 999},
        )
        assert resp.status_code == 404


class TestUpdatePullRequestBody:
    @staticmethod
    def _mock(result):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.update_pull_request_body.return_value = result
        return mock_instance

    @staticmethod
    def _ok(changed):
        return {
            "success": True,
            "pr_number": 7,
            "pr_url": "https://github.com/owner/repo/pull/7",
            "pr_status": "open",
            "changed": changed,
        }

    _PAYLOAD = {"repo_owner": "owner", "repo_name": "repo", "pr_number": 7, "body": "## Fixed"}

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_changed(self, mock_gh_cls, async_client):
        mock_gh_cls.return_value = self._mock(self._ok(True))

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["changed"] is True
        assert body["message"] == "Pull request body updated"
        mock_gh_cls.return_value.update_pull_request_body.assert_called_once_with(
            repo_owner="owner", repo_name="repo", pr_number=7, body="## Fixed"
        )

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_unchanged(self, mock_gh_cls, async_client):
        mock_gh_cls.return_value = self._mock(self._ok(False))

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["changed"] is False
        assert body["message"] == "Pull request body unchanged"

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_not_found(self, mock_gh_cls, async_client):
        mock_gh_cls.return_value = self._mock(
            {"success": False, "error": "GitHub API error: Not Found", "status_code": 404}
        )

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 404

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_closed_pr_conflict(self, mock_gh_cls, async_client):
        mock_gh_cls.return_value = self._mock(
            {"success": False, "error": "pull request #7 is closed, not open", "status_code": 409}
        )

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 409

    @pytest.mark.parametrize(
        "payload",
        [
            {"repo_owner": "owner", "repo_name": "repo", "pr_number": 7, "body": ""},
            {"repo_owner": "owner", "repo_name": "repo", "pr_number": 7, "body": "x", "x": 1},
        ],
        ids=["empty_body", "extra_key"],
    )
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_invalid_payload_422(self, mock_gh_cls, async_client, payload):
        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=payload)
        assert resp.status_code == 422
        mock_gh_cls.return_value.update_pull_request_body.assert_not_called()

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_public_repo_403(self, mock_gh_cls, async_client):
        """A public target is a policy refusal (403), never folded into a 500."""
        from services.repo_visibility_guard import RepoVisibilityRefusedError

        mock_instance = self._mock(None)
        mock_instance.update_pull_request_body.side_effect = RepoVisibilityRefusedError(
            "public_repo_refused", "owner/repo", "owner/repo is public", 403
        )
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 403

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_update_body_no_token_412(self, mock_gh_cls, async_client):
        from services.github_integration import GitHubAuthError

        mock_instance = self._mock(None)
        mock_instance.update_pull_request_body.side_effect = GitHubAuthError(
            "No GitHub token configured."
        )
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/update-pull-request-body", json=self._PAYLOAD)
        assert resp.status_code == 412
        assert resp.json()["error"]["code"] == "HTTP_412"


class TestCreatePREmptyFields:
    async def test_empty_fields_returns_400(self, async_client):
        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": " ",
                "original_repo": " ",
                "fork_owner": "fork",
                "fork_repo": "repo",
                "head_branch": "branch",
                "base_branch": "main",
                "title": "Title",
                "message": "Message",
            },
        )
        assert resp.status_code == 400


class TestCreatePRServiceFailure:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_service_failure_returns_500(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.side_effect = Exception("API failure")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "feature/x",
                "base_branch": "main",
                "title": "PR Title",
                "message": "Description",
            },
        )
        assert resp.status_code == 500


# ===========================================================================
# Validate PR tests
# ===========================================================================


class TestValidatePRSuccess:
    @patch(
        "routers.github_pull_request_operations.validate_pr_comprehensive", new_callable=AsyncMock
    )
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_validate_pr_success(self, mock_gh_cls, mock_validate, async_client):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_files = [
            MagicMock(patch="+print('hi')", filename="a.py", additions=1, deletions=0, changes=1)
        ]
        mock_pr.get_files.return_value = mock_files
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_validate.return_value = [
            PRValidationResult(
                validation_type="security", passed=True, score=90.0, issues=[], suggestions=[]
            ),
            PRValidationResult(
                validation_type="quality", passed=True, score=85.0, issues=[], suggestions=[]
            ),
            PRValidationResult(
                validation_type="performance", passed=True, score=80.0, issues=[], suggestions=[]
            ),
        ]

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security", "quality", "performance"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert body["overall_score"] > 0
        assert body["recommendation"] == "approve"


class TestValidatePRLegacy:
    @patch("routers.github_pull_request_operations._validate_pr_security", new_callable=AsyncMock)
    @patch(
        "routers.github_pull_request_operations._validate_pr_performance", new_callable=AsyncMock
    )
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_legacy_fallback_two_types(self, mock_gh_cls, mock_perf, mock_sec, async_client):
        """Two validation types use legacy path."""
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_sec.return_value = PRValidationResult(
            validation_type="security", passed=True, score=90.0, issues=[], suggestions=[]
        )
        mock_perf.return_value = PRValidationResult(
            validation_type="performance", passed=True, score=80.0, issues=[], suggestions=[]
        )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security", "performance"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert len(body["validations"]) == 2
        assert body["overall_score"] == pytest.approx(85.0)


# ===========================================================================
# Error mapping tests
# ===========================================================================


class TestErrorMapping:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_repo_not_found(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client.get_repo.side_effect = Exception("404 Not Found")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code in (404, 500)

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_repo_forbidden(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client.get_repo.side_effect = Exception("403 Forbidden")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code in (403, 500)

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_generic_error_returns_500(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client.get_repo.side_effect = Exception("Something weird")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code == 500


class TestApplyChangesEmptySetMintsNoRef:
    """An EMPTY change set must not create a branch.

    MEASURED 2026-09-14 on nonagenticai/enterprise_mcp_server: `AA Worker -
    Generate Refactoring` errored at 03:17:23Z and 11 seconds later the branch
    `refactor/break-cycle-aacyc-20260901-143226` appeared with
    `compare main...branch` = ahead 0, commits [], files []. `_create_branch`
    ran before anything was known about `changes`, so a FAILED generator left a
    plausible-looking artefact in a real repo — and `AA Maintenance - Blocked
    Sweep` re-drives that card every 15 minutes, re-minting it on a timer.

    The response shape must stay identical to the committed == 0 case, which
    `AA Activity - Action :: Validate Typed Step` already maps to a typed
    'no_changes' defect. Raising instead would turn a graded outcome into an
    untyped node error."""

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_empty_changes_creates_no_branch(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client()

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = (None, 0)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [],
                "branch_name": "refactor/empty-generator-output",
            },
        )

        # The defect: without the guard this is called and a real ref is minted.
        mock_instance._create_branch.assert_not_called()
        # ...and nothing is committed either.
        mock_instance._apply_changes.assert_not_called()

        assert resp.status_code == 200
        body = resp.json()
        assert body["no_changes"] is True
        assert body["changes_applied"] == 0
        assert body["commit_sha"] is None
        # The message must say plainly that the branch does not exist, so a
        # reader cannot mistake the named branch for one that was created.
        assert "does NOT exist" in body["message"]
