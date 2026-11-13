"""Tests for services/github/webhook_handler.py"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.github.webhook_handler import GitHubWebhookHandler


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch):
    """These tests pin behaviour that uses the PLATFORM PAT, which since
    2026-09-22 may act only on ``system_github_owner`` repos; a customer repo is
    skipped/refused (tests/unit/test_platform_pat_customer_guard.py). So the
    fixture owner "owner" is declared the platform org here.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "owner")


class TestGitHubWebhookHandler:
    """Tests for GitHubWebhookHandler."""

    def _make_handler(self):
        handler = GitHubWebhookHandler()
        return handler

    # ------------------------------------------------------------------ #
    # handle_repository_update — delegates the graph refresh, owns no graph
    #
    # ⚠️ The suite this replaces patched out `_get_database_path`,
    # `_handle_workspace_update` AND `_record_update_in_database` — i.e. it
    # mocked away every part that was broken — and then asserted
    # `status == "success"`. It was green the whole time the real path raised
    # `AttributeError: Settings.get_database_manager` on its FIRST line. The
    # tests below call the real method and assert on the real collaborators.
    # ------------------------------------------------------------------ #
    async def test_push_requests_a_rescan_from_coding_analysis(self):
        """The happy path: repo resolves to a container -> re-scan kickoff."""
        handler = self._make_handler()

        seen: dict = {}

        async def _fake_scan(container_name, tenant_id):
            seen["container_name"] = container_name
            seen["tenant_id"] = tenant_id

        with (
            patch(
                "services.github_provisioning._ContainersHTTPAdapter.find_existing_container",
                new_callable=AsyncMock,
                return_value={"container_name": "codegraph-repo-abc123"},
            ),
            patch("services.github_provisioning._trigger_full_scan", _fake_scan),
        ):
            result = await handler.handle_repository_update(
                repo_name="owner/repo",
                branch="main",
                changed_files=["src/app.py"],
                commit_shas=["abc123"],
                tenant_id="tenant-a",
            )

        assert result["status"] == "scan_requested"
        assert result["commits_processed"] == 1
        assert result["files_changed"] == 1
        # The refresh must reach coding-analysis for the RIGHT container+tenant.
        assert seen["container_name"] == "codegraph-repo-abc123"
        assert seen["tenant_id"] == "tenant-a"

    async def test_push_writes_no_sqlite_and_never_reports_plain_success(self):
        """This service must not own a code graph.

        Two properties in one: the handler exposes no sqlite machinery at all,
        and its happy-path status is `scan_requested` — never the bare
        `"success"` the sqlite path used to fabricate.
        """
        handler = self._make_handler()

        for gone in (
            "_get_database_path",
            "_record_update_in_database",
            "_handle_workspace_update",
        ):
            assert not hasattr(handler, gone), f"{gone} must not come back"

        async def _fake_scan(container_name, tenant_id):
            return None

        with (
            patch(
                "services.github_provisioning._ContainersHTTPAdapter.find_existing_container",
                new_callable=AsyncMock,
                return_value={"container_name": "c1"},
            ),
            patch("services.github_provisioning._trigger_full_scan", _fake_scan),
        ):
            result = await handler.handle_repository_update(
                repo_name="owner/repo",
                branch="main",
                changed_files=["src/app.py"],
                commit_shas=["abc123"],
            )
        assert result["status"] != "success"
        assert result["status"] == "scan_requested"

    async def test_unresolvable_repo_is_no_container_not_success(self):
        """A push for a repo nobody provisioned is a no-op, stated as such."""
        handler = self._make_handler()

        with patch(
            "services.github_provisioning._ContainersHTTPAdapter.find_existing_container",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_repository_update(
                repo_name="owner/never-provisioned",
                branch="main",
                changed_files=["src/app.py"],
                commit_shas=["abc123"],
            )

        assert result["status"] == "no_container"
        # Not laundered into a success, and no fabricated counts.
        assert result["status"] != "scan_requested"
        assert result["update_stats"]["added_nodes"] == 0
        assert result["update_stats"]["status"] == "no_container"

    async def test_resolution_failure_is_reported_as_error(self):
        """A failed container lookup must surface, not read as 'nothing to do'."""
        handler = self._make_handler()

        with patch(
            "services.github_provisioning._ContainersHTTPAdapter.find_existing_container",
            new_callable=AsyncMock,
            side_effect=RuntimeError("containers unreachable"),
        ):
            result = await handler.handle_repository_update(
                repo_name="owner/repo",
                branch="main",
                changed_files=[],
                commit_shas=[],
            )

        assert result["status"] == "error"
        assert "containers unreachable" in result["update_stats"]["error"]

    async def test_scan_kickoff_failure_is_reported_as_error(self):
        """coding-analysis refusing the kickoff is an error, not a success."""
        handler = self._make_handler()

        async def _boom(container_name, tenant_id):
            raise RuntimeError("analysis unreachable")

        with (
            patch(
                "services.github_provisioning._ContainersHTTPAdapter.find_existing_container",
                new_callable=AsyncMock,
                return_value={"container_name": "c1"},
            ),
            patch("services.github_provisioning._trigger_full_scan", _boom),
        ):
            result = await handler.handle_repository_update(
                repo_name="owner/repo",
                branch="main",
                changed_files=["a.py"],
                commit_shas=["s1"],
            )

        assert result["status"] == "error"
        assert "analysis unreachable" in result["update_stats"]["error"]

    # ------------------------------------------------------------------ #
    # Process commit changes categorization
    # ------------------------------------------------------------------ #
    async def test_process_commit_changes(self):
        """Test _process_commit_changes categorizes file changes."""
        handler = self._make_handler()

        mock_service = AsyncMock()
        mock_service.get_commit_files = AsyncMock(
            return_value=[
                {"filename": "new.py", "status": "added"},
                {"filename": "old.py", "status": "removed"},
                {"filename": "changed.py", "status": "modified"},
            ]
        )
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github.webhook_handler.get_github_service",
            new_callable=AsyncMock,
            return_value=mock_service,
        ):
            changes = await handler._process_commit_changes("owner/repo", ["sha1"])

        assert len(changes["added"]) == 1
        assert len(changes["deleted"]) == 1
        assert len(changes["modified"]) == 1

    # ------------------------------------------------------------------ #
    # Processing failure
    # ------------------------------------------------------------------ #
    async def test_process_commit_changes_skips_bad_commits(self):
        """Test _process_commit_changes handles errors in individual commits."""
        handler = self._make_handler()

        mock_service = AsyncMock()
        mock_service.get_commit_files = AsyncMock(
            side_effect=[
                KeyError("bad commit"),
                [{"filename": "ok.py", "status": "added"}],
            ]
        )
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github.webhook_handler.get_github_service",
            new_callable=AsyncMock,
            return_value=mock_service,
        ):
            changes = await handler._process_commit_changes("owner/repo", ["bad-sha", "good-sha"])

        # First commit failed but second succeeded
        assert len(changes["added"]) == 1
