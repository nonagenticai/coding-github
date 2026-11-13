"""Unit tests for GitHubIntegrationHub in services/github_integration_core.py.

Verifies that the hub initialises all sub-components and correctly delegates
each operation (clone, sync, PR analysis, webhook handling)
to the appropriate sub-module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch):
    """These tests pin behaviour that uses the PLATFORM PAT, which since
    2026-09-22 may act only on ``system_github_owner`` repos; a customer repo is
    skipped/refused (tests/unit/test_platform_pat_customer_guard.py). So the
    fixture owner "owner" is declared the platform org here.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "owner")


class TestGitHubIntegrationHubInit:
    """Constructor initialises all four sub-components."""

    def test_all_components_initialised(self):
        with (
            patch("services.github_integration_core.GitHubAuthManager") as auth_cls,
            patch("services.github_integration_core.GitHubClient") as client_cls,
            patch("services.github_integration_core.RepositoryOperations") as repo_cls,
            patch("services.github_integration_core.GitHubWebhookHandler") as wh_cls,
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()

            auth_cls.assert_called_once()
            client_cls.assert_called_once()
            repo_cls.assert_called_once()
            wh_cls.assert_called_once()

            assert hub.auth_manager is auth_cls.return_value
            assert hub.github_client is client_cls.return_value
            assert hub.repo_operations is repo_cls.return_value
            assert hub.webhook_handler is wh_cls.return_value


class TestCloneRepository:
    """clone_repository delegates to repo_operations WITHOUT a credential."""

    async def test_delegates_without_a_token_even_if_the_auth_manager_had_one(self):
        """No PAT exists and the hub has no tenant for the App (2026-09-30): the
        clone is always anonymous, never the auth manager's token."""
        mock_auth = MagicMock()
        mock_auth.get_token.return_value = "ghp_test_token"
        mock_repo_ops = MagicMock()
        mock_repo_ops.clone_repository = AsyncMock(
            return_value=MagicMock(success=True, local_path="/tmp/repo")
        )

        with (
            patch("services.github_integration_core.GitHubAuthManager", return_value=mock_auth),
            patch("services.github_integration_core.GitHubClient"),
            patch(
                "services.github_integration_core.RepositoryOperations",
                return_value=mock_repo_ops,
            ),
            patch("services.github_integration_core.GitHubWebhookHandler"),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            result = await hub.clone_repository("https://github.com/owner/repo")

            mock_repo_ops.clone_repository.assert_awaited_once_with(
                "https://github.com/owner/repo", None
            )
            assert result.success is True

    async def test_works_with_none_token(self):
        """When auth_manager returns None, clone still delegates correctly."""
        mock_auth = MagicMock()
        mock_auth.get_token.return_value = None
        mock_repo_ops = MagicMock()
        mock_repo_ops.clone_repository = AsyncMock(
            return_value=MagicMock(success=True, local_path="/tmp/repo")
        )

        with (
            patch("services.github_integration_core.GitHubAuthManager", return_value=mock_auth),
            patch("services.github_integration_core.GitHubClient"),
            patch(
                "services.github_integration_core.RepositoryOperations",
                return_value=mock_repo_ops,
            ),
            patch("services.github_integration_core.GitHubWebhookHandler"),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            result = await hub.clone_repository("https://github.com/owner/repo")

            mock_repo_ops.clone_repository.assert_awaited_once_with(
                "https://github.com/owner/repo", None
            )
            assert result.success is True


class TestSyncRepository:
    """sync_repository delegates to repo_operations."""

    async def test_delegates_to_repo_operations(self):
        mock_repo_ops = MagicMock()
        mock_repo_ops.sync_repository = AsyncMock(
            return_value=MagicMock(success=True, has_updates=True)
        )

        with (
            patch("services.github_integration_core.GitHubAuthManager"),
            patch("services.github_integration_core.GitHubClient"),
            patch(
                "services.github_integration_core.RepositoryOperations",
                return_value=mock_repo_ops,
            ),
            patch("services.github_integration_core.GitHubWebhookHandler"),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            workspace = MagicMock()
            result = await hub.sync_repository(workspace)

            mock_repo_ops.sync_repository.assert_awaited_once_with(workspace)
            assert result.success is True


class TestAnalyzePullRequest:
    """analyze_pull_request and get_repository_info delegate to github_client."""

    async def test_delegates_and_returns_results(self):
        fake_analysis = {
            "pull_request": MagicMock(),
            "files": [],
            "statistics": {"total_files": 2},
            "languages": {},
            "total_additions": 10,
            "total_deletions": 5,
        }
        fake_repo_info = MagicMock(full_name="owner/repo", default_branch="main")

        mock_client = MagicMock()
        mock_client.analyze_pull_request = AsyncMock(return_value=fake_analysis)
        mock_client.get_repository_info = AsyncMock(return_value=fake_repo_info)

        with (
            patch("services.github_integration_core.GitHubAuthManager"),
            patch("services.github_integration_core.GitHubClient", return_value=mock_client),
            patch("services.github_integration_core.RepositoryOperations"),
            patch("services.github_integration_core.GitHubWebhookHandler"),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()

            analysis = await hub.analyze_pull_request("https://github.com/o/r", 1)
            assert analysis["total_additions"] == 10
            assert analysis["total_deletions"] == 5

            info = await hub.get_repository_info("https://github.com/o/r")
            assert info.full_name == "owner/repo"


class TestReadPRFile:
    """read_pr_file delegates to github_client.read_pr_file."""

    async def test_delegates_with_all_args(self):
        mock_client = MagicMock()
        mock_client.read_pr_file = AsyncMock(return_value="file content here")

        with (
            patch("services.github_integration_core.GitHubAuthManager"),
            patch("services.github_integration_core.GitHubClient", return_value=mock_client),
            patch("services.github_integration_core.RepositoryOperations"),
            patch("services.github_integration_core.GitHubWebhookHandler"),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            result = await hub.read_pr_file("https://github.com/o/r", 5, "src/main.py", "head")

            mock_client.read_pr_file.assert_awaited_once_with(
                "https://github.com/o/r", 5, "src/main.py", "head"
            )
            assert result == "file content here"


class TestUnhardenedCommitPushHelperStaysRemoved:
    """Keep-out guard (coding-containers#156).

    ``services.github.github_utils.GitHubUtils`` wrote caller-supplied
    ``file_path`` values into a git workspace unvalidated (``.git/hooks/*`` and
    ``.git/config`` were writable), then ran ``git commit``/``git push`` with the
    repo's hooks live. It was reachable from no route, worker or script, so it
    was deleted rather than hardened. Re-adding it must go through a review that
    adds path confinement and ``core.hooksPath=/dev/null``, not slip back in.
    """

    def test_module_is_gone(self):
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("services.github.github_utils")

    def test_hub_has_no_pr_creation_entrypoint(self):
        from services.github_integration_core import GitHubIntegrationHub

        assert not hasattr(GitHubIntegrationHub, "create_pull_request_with_analysis")

    def test_package_does_not_reexport_it(self):
        import services.github as gh

        assert not hasattr(gh, "GitHubUtils")
        assert "GitHubUtils" not in gh.__all__


class TestHandleRepositoryUpdate:
    """handle_repository_update delegates to webhook_handler."""

    async def test_delegates_to_webhook_handler(self):
        mock_webhook = MagicMock()
        mock_webhook.handle_repository_update = AsyncMock(
            return_value={"status": "processed", "files_affected": 3}
        )

        with (
            patch("services.github_integration_core.GitHubAuthManager"),
            patch("services.github_integration_core.GitHubClient"),
            patch("services.github_integration_core.RepositoryOperations"),
            patch(
                "services.github_integration_core.GitHubWebhookHandler",
                return_value=mock_webhook,
            ),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            result = await hub.handle_repository_update(
                "org/repo", "main", ["a.py", "b.py"], ["sha1"], tenant_id="tenant-a"
            )

            # The tenant must be forwarded: the graph the handler refreshes is
            # resolved by a tenant-scoped container lookup.
            mock_webhook.handle_repository_update.assert_awaited_once_with(
                "org/repo", "main", ["a.py", "b.py"], ["sha1"], tenant_id="tenant-a"
            )
            assert result["status"] == "processed"
