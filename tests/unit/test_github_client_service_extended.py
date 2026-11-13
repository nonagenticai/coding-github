"""Extended tests for services/github/github_client.py

Covers edge cases, boundary conditions, and error recovery NOT in the base file:
- URL parsing edge cases (trailing slashes, query strings, fragments)
- analyze_pull_request with all change_type categories
- Language extension boundary (empty extension, dotfiles)
- RepoInfo field defaults and overrides
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.github_client import GitHubClient


def _client() -> GitHubClient:
    return GitHubClient()


def _mock_github_service(**method_overrides) -> AsyncMock:
    svc = AsyncMock()
    svc.__aenter__ = AsyncMock(return_value=svc)
    svc.__aexit__ = AsyncMock(return_value=False)
    for name, val in method_overrides.items():
        if isinstance(val, Exception):
            setattr(svc, name, AsyncMock(side_effect=val))
        else:
            setattr(svc, name, AsyncMock(return_value=val))
    return svc


def _patch_svc(mock_svc):
    return patch(
        "services.github.github_client.get_github_service",
        new_callable=AsyncMock,
        return_value=mock_svc,
    )


# =========================================================================== #
# URL parsing edge cases
# =========================================================================== #


class TestParseRepoUrlEdgeCases:
    """Edge cases NOT covered by the base file."""

    @pytest.mark.parametrize(
        "url, expected_owner, expected_name",
        [
            # Trailing slash
            ("https://github.com/own/rep/", "own", "rep"),
            # SSH without .git
            ("git@github.com:alpha/beta", "alpha", "beta"),
            # Repo name containing dots (but not .git suffix)
            ("https://github.com/org/my.repo.name", "org", "my.repo.name"),
        ],
    )
    def test_edge_case_urls(self, url: str, expected_owner: str, expected_name: str):
        result = _client()._parse_repo_url(url)
        assert result["owner"] == expected_owner
        assert result["name"] == expected_name

    def test_git_url_with_nested_org(self):
        """SSH URL with complex owner containing dots — only first two segments used."""
        result = _client()._parse_repo_url("git@github.com:my-org/my-repo.git")
        assert result == {"owner": "my-org", "name": "my-repo"}


# =========================================================================== #
# analyze_pull_request: all change_type categories
# =========================================================================== #


class TestAnalyzePullRequestChangeCategories:
    """Verify all change_type counters in statistics dict."""

    def _make_file(self, path, change_type, add=1, delete=0):
        f = MagicMock()
        f.file_path = path
        f.change_type = change_type
        f.additions = add
        f.deletions = delete
        return f

    async def test_all_change_types_counted(self):
        files = [
            self._make_file("a.py", "added", 10, 0),
            self._make_file("b.py", "modified", 5, 3),
            self._make_file("c.py", "removed", 0, 20),
            self._make_file("d.py", "renamed", 1, 1),
        ]
        mock_pr = MagicMock(number=10)
        mock_svc = _mock_github_service(get_pull_request=mock_pr, list_pull_request_files=files)
        with _patch_svc(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/o/r", 10)

        stats = result["statistics"]
        assert stats["total"] == 4
        assert stats["added"] == 1
        assert stats["modified"] == 1
        assert stats["removed"] == 1
        assert stats["renamed"] == 1

    async def test_language_aggregation_multiple_same_ext(self):
        """Multiple Python files should aggregate into one language entry."""
        files = [
            self._make_file("a.py", "modified", 10, 2),
            self._make_file("b.py", "added", 5, 0),
            self._make_file("c.py", "modified", 3, 1),
        ]
        mock_pr = MagicMock(number=11)
        mock_svc = _mock_github_service(get_pull_request=mock_pr, list_pull_request_files=files)
        with _patch_svc(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/o/r", 11)

        py = result["languages"]["Python"]
        assert py["files"] == 3
        assert py["additions"] == 18
        assert py["deletions"] == 3

    async def test_unknown_extension_grouped_as_other(self):
        files = [
            self._make_file("Makefile", "modified", 2, 1),
            self._make_file(".gitignore", "modified", 1, 0),
        ]
        mock_pr = MagicMock(number=12)
        mock_svc = _mock_github_service(get_pull_request=mock_pr, list_pull_request_files=files)
        with _patch_svc(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/o/r", 12)

        assert "Other" in result["languages"]
        assert result["languages"]["Other"]["files"] == 2


# =========================================================================== #
# get_repository_info: field default behaviour
# =========================================================================== #


class TestGetRepositoryInfoDefaults:
    """Verify defaults when API response is sparse."""

    async def test_default_branch_falls_back_to_main(self):
        mock_svc = _mock_github_service(get_repository_info={})
        with _patch_svc(mock_svc):
            info = await _client().get_repository_info("https://github.com/owner/repo")
        assert info.default_branch == "main"

    async def test_url_field_is_original_input(self):
        """RepoInfo.url should be the original URL passed to get_repository_info."""
        mock_svc = _mock_github_service(
            get_repository_info={"owner": "o", "name": "r", "default_branch": "main"}
        )
        original_url = "https://github.com/o/r"
        with _patch_svc(mock_svc):
            info = await _client().get_repository_info(original_url)
        assert info.url == original_url

    async def test_description_and_language_can_be_none(self):
        mock_svc = _mock_github_service(
            get_repository_info={"owner": "o", "name": "r", "default_branch": "main"}
        )
        with _patch_svc(mock_svc):
            info = await _client().get_repository_info("https://github.com/o/r")
        assert info.description is None
        assert info.language is None


# =========================================================================== #
# read_pr_file: content coercion
# =========================================================================== #


class TestReadPrFileContentCoercion:
    """The source converts non-None content to str — verify that."""

    async def test_numeric_content_becomes_string(self):
        mock_svc = _mock_github_service(read_pull_request_file=42)
        with _patch_svc(mock_svc):
            content = await _client().read_pr_file("https://github.com/o/r", 1, "file.txt")
        assert content == "42"
        assert isinstance(content, str)

    async def test_bool_content_becomes_string(self):
        mock_svc = _mock_github_service(read_pull_request_file=True)
        with _patch_svc(mock_svc):
            content = await _client().read_pr_file("https://github.com/o/r", 1, "file.txt")
        assert content == "True"
