"""No commit or ref update lands on a repository's default branch.

Before this guard, ``apply-changes`` with ``branch_name="main"`` found the ref
already at the base SHA, "reused" it and committed straight onto ``main`` — no
PR, no review, no CI. coding-security's ``commit=true, branch_name="main"``
reaches the same code. There is NO override: an ``allow_default_branch_write``
body flag existed briefly and was removed, because the gateway forwards the body
unchanged and n8n's generic dispatch passes unknown keys through — every caller
held the switch. Every spelling of the ref (``refs/heads/main``, ``heads/main``,
`` main``, ``MAIN``) is refused.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.github_api_models import ApplyChangesRequest
from services.github_integration import (
    DefaultBranchWriteRefusedError,
    GitHubIntegration,
    assert_branch_writable,
    normalize_branch_ref,
)

BASE = "/api/repositories"


def _repo(default: str = "main") -> MagicMock:
    repo = MagicMock()
    repo.default_branch = default
    repo.full_name = "owner/repo"
    repo.html_url = "https://github.com/owner/repo"
    repo.owner = SimpleNamespace(login="owner")
    repo.name = "repo"
    return repo


def _bare_integration() -> GitHubIntegration:
    # The branch helpers use no client state; skip __init__'s token ladder.
    return GitHubIntegration.__new__(GitHubIntegration)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


class TestAssertBranchWritable:
    @pytest.mark.parametrize(
        ("default", "target"),
        [
            ("main", "main"),
            ("master", "master"),
            ("develop", "develop"),  # the repo's REPORTED default
            ("develop", "main"),  # main/master stay protected regardless
            ("main", "someone:main"),  # owner-qualified head ref
        ],
    )
    def test_protected_targets_are_refused(self, default, target):
        with pytest.raises(DefaultBranchWriteRefusedError) as exc:
            assert_branch_writable(_repo(default), target)
        assert exc.value.error_code == "default_branch_write_refused"

    @pytest.mark.parametrize(
        "target",
        [
            "refs/heads/main",
            "heads/main",
            " main",
            "main ",
            "\tmain\n",
            "MAIN",
            "Main",
            "Refs/Heads/Main",
            " refs/heads/main ",
            "owner:refs/heads/main",
            "refs/heads/refs/heads/main",
            "/main",
            "refs/heads/master",
            "heads/MASTER",
        ],
    )
    def test_every_ref_spelling_of_a_protected_branch_is_refused(self, target):
        with pytest.raises(DefaultBranchWriteRefusedError):
            assert_branch_writable(_repo("main"), target)

    @pytest.mark.parametrize("target", ["refs/heads/Develop", " develop", "heads/develop"])
    def test_reported_default_is_normalised_too(self, target):
        with pytest.raises(DefaultBranchWriteRefusedError):
            assert_branch_writable(_repo("develop"), target)

    @pytest.mark.parametrize(
        "target",
        ["aa/fix-1", "feat/x", "code-changes-deadbeef", "mainline", "refs/heads/feat/main-fix"],
    )
    def test_feature_branches_pass(self, target):
        assert_branch_writable(_repo("main"), target)

    def test_normalisation(self):
        assert normalize_branch_ref(" Refs/Heads/Main ") == "main"
        assert normalize_branch_ref("heads/main") == "main"
        assert normalize_branch_ref("feat/Main-Fix") == "feat/main-fix"


class TestNoOverrideExists:
    """The override flag must not come back — in any layer."""

    def test_guard_takes_no_override_parameter(self):
        assert list(inspect.signature(assert_branch_writable).parameters) == [
            "repo",
            "branch_name",
        ]

    @pytest.mark.parametrize(
        "method", ["_apply_changes", "_update_files", "_create_branch", "create_pull_request"]
    )
    def test_no_write_path_accepts_an_override(self, method):
        params = inspect.signature(getattr(GitHubIntegration, method)).parameters
        assert not any("allow" in name for name in params), list(params)

    def test_request_model_has_no_override_field(self):
        assert not any("allow" in name for name in ApplyChangesRequest.model_fields)
        req = ApplyChangesRequest.model_validate(
            {"container_name": "c", "allow_default_branch_write": True}
        )
        assert "allow_default_branch_write" not in req.model_dump()


# ---------------------------------------------------------------------------
# The service chokepoints
# ---------------------------------------------------------------------------


class TestServiceChokepoints:
    def test_apply_changes_refuses_default_branch_before_any_write(self):
        integration = _bare_integration()
        repo = _repo("main")
        with patch.object(GitHubIntegration, "_create_or_update_file") as write:
            with pytest.raises(DefaultBranchWriteRefusedError):
                integration._apply_changes(repo, "main", [{"file_path": "a.py", "content": "x"}])
        write.assert_not_called()

    def test_update_files_refuses_default_branch(self):
        integration = _bare_integration()
        with patch.object(GitHubIntegration, "_create_or_update_file") as write:
            with pytest.raises(DefaultBranchWriteRefusedError):
                integration._update_files(
                    _repo("trunk"), "trunk", [{"file_path": "a.py", "content": "x"}]
                )
        write.assert_not_called()

    def test_apply_changes_refuses_refs_heads_spelling(self):
        integration = _bare_integration()
        with patch.object(GitHubIntegration, "_create_or_update_file") as write:
            with pytest.raises(DefaultBranchWriteRefusedError):
                integration._apply_changes(
                    _repo("main"), "refs/heads/main", [{"file_path": "a.py", "content": "x"}]
                )
        write.assert_not_called()

    def test_apply_changes_feature_branch_unchanged(self):
        integration = _bare_integration()
        with patch.object(GitHubIntegration, "_create_or_update_file", return_value="sha1"):
            assert integration._apply_changes(
                _repo("main"), "aa/fix", [{"file_path": "a.py", "content": "x"}]
            ) == ("sha1", 1)

    def test_force_reset_of_default_branch_is_refused(self):
        """A force-reset is a force-push. base='release' resolves to a SHA other
        than main's, so the reset path would rewrite main — refused, not run."""
        integration = _bare_integration()
        repo = _repo("main")
        main_ref = MagicMock()
        main_ref.object.sha = "main-sha"
        base_ref = SimpleNamespace(object=SimpleNamespace(sha="release-sha"))

        from services import github_integration as gi

        class _Exists(gi.GithubException):  # type: ignore[misc,valid-type]
            status = 422

        repo.create_git_ref.side_effect = _Exists("exists")
        repo.get_git_ref.side_effect = lambda ref: base_ref if ref == "heads/release" else main_ref

        with pytest.raises(DefaultBranchWriteRefusedError):
            integration._create_branch(repo, "main", "release", reset_existing_to_base=True)
        main_ref.edit.assert_not_called()


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


def _workspaces_client(branch: str = "main"):
    client = MagicMock()
    client.get_workspace = AsyncMock(
        return_value={
            "status": "success",
            "message": "ok",
            "container": {
                "container_name": "test-container",
                "repo_url": "https://github.com/owner/repo",
                "branch": branch,
                "status": "running",
                "github_access_token": None,
            },
        }
    )
    return client


def _integration_double(fork_repo):
    inst = MagicMock()
    inst.github_client.get_repo.return_value.private = True  # a private target repo
    fork_repo.private = True  # the resolved WRITE target is private too (#118)
    inst._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
    inst._create_branch.return_value = None
    inst._apply_changes.return_value = ("sha123", 1)
    return inst


class TestApplyChangesRoute:
    @pytest.mark.parametrize(
        ("default", "branch"),
        [
            ("main", "main"),
            ("master", "master"),
            ("develop", "develop"),
            ("main", "x:main"),
            ("main", "refs/heads/main"),
            ("main", "heads/main"),
            ("main", " main"),
            ("main", "MAIN"),
        ],
    )
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_default_branch_is_refused_with_403_and_nothing_written(
        self, mock_gh_cls, mock_get_client, default, branch, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration_double(_repo(default))
        mock_gh_cls.return_value = inst

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
                "branch_name": branch,
            },
        )

        assert resp.status_code == 403, resp.text
        assert "default_branch_write_refused" in resp.text
        inst._create_branch.assert_not_called()
        inst._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_empty_change_set_on_default_branch_is_still_refused(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        mock_gh_cls.return_value = _integration_double(_repo("main"))

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={"container_name": "test-container", "changes": [], "branch_name": "main"},
        )

        assert resp.status_code == 403

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_body_allow_flag_is_ignored_and_still_refused(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """The caller-settable flag that used to bypass the guard does nothing."""
        mock_get_client.return_value = _workspaces_client()
        inst = _integration_double(_repo("main"))
        mock_gh_cls.return_value = inst

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
                "branch_name": "main",
                "allow_default_branch_write": True,
            },
        )

        assert resp.status_code == 403, resp.text
        assert "default_branch_write_refused" in resp.text
        inst._create_branch.assert_not_called()
        inst._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_feature_branch_on_a_pr_head_workspace_still_writes(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """A workspace provisioned on a PR head adds commits to that same head."""
        mock_get_client.return_value = _workspaces_client(branch="aa/fix-1")
        inst = _integration_double(_repo("main"))
        mock_gh_cls.return_value = inst

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
                "branch_name": "aa/fix-1",
            },
        )

        assert resp.status_code == 200, resp.text
        inst._apply_changes.assert_called_once()

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_maps_a_service_refusal_to_403(self, mock_gh_cls, async_client):
        inst = MagicMock()
        inst.github_client.get_repo.return_value.private = True  # a private target repo
        inst.create_pull_request.side_effect = DefaultBranchWriteRefusedError("refused")
        mock_gh_cls.return_value = inst

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "main",
                "base_branch": "main",
                "title": "t",
                "message": "m",
            },
        )

        assert resp.status_code == 403
