"""apply-changes writes card deliverables only — never conveyor process data.

The guard is ONE function (``services.path_guard.assert_commit_paths_allowed``) with
the denylist as ONE module constant. ``test_denylist_constant_is_what_refuses`` is the
mutation control: with ``PROCESS_DATA_DENYLIST`` emptied, ``.aa/x`` is accepted, so the
refusals below demonstrably come from that constant and not from somewhere else.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.path_guard as pg
from services.path_guard import (
    PATH_DENIED_PROCESS_DATA,
    PATH_INVALID,
    PATH_NOT_IN_TARGET_FILES,
    CommitPathRefusedError,
    assert_commit_paths_allowed,
)

BASE = "/api/repositories"


def _code(paths, target_files=None) -> str:
    with pytest.raises(CommitPathRefusedError) as exc:
        assert_commit_paths_allowed(paths, target_files)
    return exc.value.error_code


class TestTargetFiles:
    def test_a_target_file_is_accepted(self):
        assert_commit_paths_allowed(["src/app.py"], ["src/app.py", "README.md"])

    def test_dot_slash_spelling_of_a_target_is_the_same_path(self):
        assert_commit_paths_allowed(["./src/app.py"], ["src/app.py"])

    def test_a_non_target_file_is_refused(self):
        assert _code(["src/other.py"], ["src/app.py"]) == PATH_NOT_IN_TARGET_FILES

    def test_an_empty_target_list_admits_nothing(self):
        assert _code(["src/app.py"], []) == PATH_NOT_IN_TARGET_FILES

    def test_without_target_files_an_ordinary_path_is_accepted(self):
        assert_commit_paths_allowed(["src/app.py", "src/billing/plans/tier.py"])


class TestProcessDataDenylist:
    @pytest.mark.parametrize(
        "path",
        [
            ".aa/x",
            ".aa/state/run.json",
            "pkg/.aa/x",
            "foo_probe/y.py",
            "src/aa_bugfix_probe/queries.py",
            "a.log",
            "logs/deep/run.log",
            ".claude/settings.json",
            "plans/plan.md",
            "traces/t1.json",
            "run-state/x.json",
            "out/step.trace.json",
        ],
    )
    def test_process_data_is_refused_without_target_files(self, path):
        assert _code([path]) == PATH_DENIED_PROCESS_DATA

    @pytest.mark.parametrize(
        "path", [".AA/x", ".Claude/y", "X.LOG", "Plans/x", "pkg/.Aa/x", "out/S.Trace.JSON"]
    )
    def test_matching_is_case_insensitive(self, path):
        with pytest.raises(CommitPathRefusedError) as exc:
            assert_commit_paths_allowed([path])
        assert exc.value.error_code == PATH_DENIED_PROCESS_DATA
        assert exc.value.path == path  # the caller's spelling, not the folded one
        assert repr(path) in str(exc.value)

    @pytest.mark.parametrize(
        "path", [".aa", ".claude", "plans", "traces", "run-state", "./PLANS", "Run-State"]
    )
    def test_bare_root_process_data_name_is_refused(self, path):
        assert _code([path]) == PATH_DENIED_PROCESS_DATA

    @pytest.mark.parametrize("path", ["src/plans", "docs/traces", "plans.md", "tracesx"])
    def test_process_data_names_below_root_or_as_prefix_are_accepted(self, path):
        assert_commit_paths_allowed([path])

    def test_process_data_is_refused_even_when_listed_as_a_target(self):
        """No caller can opt a path back in by naming it."""
        assert _code([".aa/x"], [".aa/x"]) == PATH_DENIED_PROCESS_DATA

    def test_denylist_constant_is_what_refuses(self, monkeypatch):
        """MUTATION CONTROL: empty the constant and `.aa/x` goes through."""
        assert pg.PROCESS_DATA_DENYLIST, "the denylist must never be empty"
        assert _code([".aa/x"]) == PATH_DENIED_PROCESS_DATA
        monkeypatch.setattr(pg, "PROCESS_DATA_DENYLIST", ())
        assert_commit_paths_allowed([".aa/x"])  # would raise if the refusal lived elsewhere


class TestInvalidPaths:
    @pytest.mark.parametrize(
        "path", ["../etc", "../etc/passwd", "src/../../x", "/abs", "/etc/passwd", "C:\\x", ""]
    )
    def test_absolute_and_traversal_paths_are_refused(self, path):
        assert _code([path]) == PATH_INVALID

    def test_rename_source_is_checked_too(self):
        assert _code(["src/new.py", ".aa/old.py"]) == PATH_DENIED_PROCESS_DATA
        assert _code(["src/new.py", "src/old.py"], ["src/new.py"]) == PATH_NOT_IN_TARGET_FILES


# ---------------------------------------------------------------------------
# The HTTP surface: refused BEFORE any GitHub object is even constructed.
# ---------------------------------------------------------------------------


def _workspaces_client():
    client = MagicMock()
    client.get_workspace = AsyncMock(
        return_value={
            "status": "success",
            "message": "ok",
            "container": {
                "container_name": "test-container",
                "repo_url": "https://github.com/owner/repo",
                "branch": "main",
                "status": "running",
                "github_access_token": None,
            },
        }
    )
    return client


def _fork_repo():
    repo = MagicMock()
    repo.default_branch = "main"
    repo.full_name = "owner/repo"
    repo.html_url = "https://github.com/owner/repo"
    repo.owner = SimpleNamespace(login="owner")
    repo.name = "repo"
    repo.private = True  # the resolved WRITE target is private too (#118)
    return repo


class TestApplyChangesRoute:
    @pytest.mark.parametrize(
        ("changes", "target_files", "code"),
        [
            ([{"file_path": ".aa/x", "content": "x"}], None, PATH_DENIED_PROCESS_DATA),
            ([{"file_path": "a.log", "content": "x"}], None, PATH_DENIED_PROCESS_DATA),
            ([{"file_path": "../etc", "content": "x"}], None, PATH_INVALID),
            ([{"file_path": "/abs", "content": "x"}], None, PATH_INVALID),
            ([{"file_path": "src/b.py", "content": "x"}], ["src/a.py"], PATH_NOT_IN_TARGET_FILES),
        ],
    )
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_refused_with_422_and_nothing_touched(
        self, mock_gh_cls, mock_get_client, changes, target_files, code, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        body = {"container_name": "test-container", "changes": changes, "branch_name": "aa/x"}
        if target_files is not None:
            body["target_files"] = target_files

        resp = await async_client.post(f"{BASE}/apply-changes", json=body)

        assert resp.status_code == 422, resp.text
        assert code in resp.text
        mock_gh_cls.assert_not_called()  # not even the fork lookup ran

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_a_target_file_reaches_the_write_path(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = MagicMock()
        inst.github_client.get_repo.return_value.private = True  # a private target repo
        inst._ensure_fork.return_value = SimpleNamespace(repository=_fork_repo())
        inst._apply_changes.return_value = ("sha123", 1)
        mock_gh_cls.return_value = inst

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "src/a.py", "content": "x"}],
                "branch_name": "aa/x",
                "target_files": ["src/a.py"],
            },
        )

        for code in (PATH_INVALID, PATH_DENIED_PROCESS_DATA, PATH_NOT_IN_TARGET_FILES):
            assert code not in resp.text, resp.text
        inst._ensure_fork.assert_called_once()
