"""Extended tests for services/github/webhook_handler.py — uncovered lines."""

from __future__ import annotations

from pathlib import Path
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


def _make_handler():
    handler = GitHubWebhookHandler()
    return handler


def _mock_github_service(file_infos_by_sha=None, side_effects=None):
    """Create a mock GitHub service context manager.

    Args:
        file_infos_by_sha: list of return values for successive get_commit_files calls.
        side_effects: side_effect list for get_commit_files (takes precedence).
    """
    mock_service = AsyncMock()
    if side_effects is not None:
        mock_service.get_commit_files = AsyncMock(side_effect=side_effects)
    elif file_infos_by_sha is not None:
        mock_service.get_commit_files = AsyncMock(side_effect=file_infos_by_sha)
    else:
        mock_service.get_commit_files = AsyncMock(return_value=[])
    mock_service.__aenter__ = AsyncMock(return_value=mock_service)
    mock_service.__aexit__ = AsyncMock(return_value=False)
    return mock_service


# ===================================================================== #
# Lines 212-216: renamed file handling in _process_commit_changes
# ===================================================================== #


class TestProcessCommitChangesRenamed:
    """Cover the 'renamed' status branch (lines 212-216)."""

    async def test_renamed_file_produces_delete_and_add(self):
        """A renamed file should appear in both 'deleted' (old name) and 'added' (new name)."""
        handler = _make_handler()

        mock_svc = _mock_github_service(
            file_infos_by_sha=[
                [
                    {
                        "filename": "src/new_name.py",
                        "status": "renamed",
                        "additions": 0,
                        "deletions": 0,
                        "previous_filename": "src/old_name.py",
                    }
                ]
            ]
        )

        with patch(
            "services.github.webhook_handler.get_github_service",
            new_callable=AsyncMock,
            return_value=mock_svc,
        ):
            changes = await handler._process_commit_changes("owner/repo", ["sha1"])

        assert Path("src/old_name.py") in changes["deleted"]
        assert Path("src/new_name.py") in changes["added"]

    async def test_renamed_file_with_no_previous_filename(self):
        """Renamed with previous_filename=None should use empty path for delete."""
        handler = _make_handler()

        mock_svc = _mock_github_service(
            file_infos_by_sha=[
                [
                    {
                        "filename": "src/new.py",
                        "status": "renamed",
                        "additions": 1,
                        "deletions": 1,
                        "previous_filename": None,
                    }
                ]
            ]
        )

        with patch(
            "services.github.webhook_handler.get_github_service",
            new_callable=AsyncMock,
            return_value=mock_svc,
        ):
            changes = await handler._process_commit_changes("owner/repo", ["sha1"])

        # old path falls back to Path("")
        assert Path("") in changes["deleted"]
        assert Path("src/new.py") in changes["added"]
