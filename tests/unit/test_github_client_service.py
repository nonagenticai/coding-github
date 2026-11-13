"""Tests for services/github/github_client.py

Tests URL parsing, repo info retrieval, PR analysis, PR file reading,
and language extension mapping on the GitHubClient class.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.github_client import GitHubClient

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _client() -> GitHubClient:
    return GitHubClient()


def _mock_github_service(**method_overrides) -> AsyncMock:
    """Build a mock GitHubService that works as an async context manager."""
    svc = AsyncMock()
    svc.__aenter__ = AsyncMock(return_value=svc)
    svc.__aexit__ = AsyncMock(return_value=False)
    for name, val in method_overrides.items():
        setattr(
            svc,
            name,
            AsyncMock(return_value=val)
            if not isinstance(val, Exception)
            else AsyncMock(side_effect=val),
        )
    return svc


def _patch_github_service(mock_svc):
    return patch(
        "services.github.github_client.get_github_service",
        new_callable=AsyncMock,
        return_value=mock_svc,
    )


# =========================================================================== #
# _parse_repo_url
# =========================================================================== #


class TestParseRepoUrl:
    """Verify URL parsing for HTTPS, SSH, .git suffixes, and invalid inputs."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://github.com/owner/repo", {"owner": "owner", "name": "repo"}),
            ("https://github.com/owner/repo.git", {"owner": "owner", "name": "repo"}),
            ("git@github.com:owner/repo.git", {"owner": "owner", "name": "repo"}),
            ("git@github.com:org/my-repo.git", {"owner": "org", "name": "my-repo"}),
            (
                "https://github.com/owner/repo/tree/main/src",
                {"owner": "owner", "name": "repo"},
            ),
            ("https://github.com/UPPER/CASE", {"owner": "UPPER", "name": "CASE"}),
        ],
        ids=[
            "https-plain",
            "https-dot-git",
            "ssh-dot-git",
            "ssh-hyphenated",
            "https-deep-path",
            "https-uppercase",
        ],
    )
    def test_valid_urls(self, url: str, expected: dict[str, str]):
        result = _client()._parse_repo_url(url)
        assert result == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/bad",
            "https://github.com/",
            "not-a-url",
        ],
        ids=["single-segment", "empty-path", "garbage"],
    )
    def test_invalid_urls_raise_value_error(self, url: str):
        with pytest.raises((ValueError, IndexError)):
            _client()._parse_repo_url(url)


# =========================================================================== #
# _get_repo_name_from_url
# =========================================================================== #


class TestGetRepoNameFromUrl:
    """Thin wrapper over _parse_repo_url that returns 'owner/repo' string."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://github.com/owner/repo", "owner/repo"),
            ("git@github.com:org/lib.git", "org/lib"),
        ],
    )
    def test_returns_owner_slash_name(self, url: str, expected: str):
        assert _client()._get_repo_name_from_url(url) == expected


# =========================================================================== #
# _get_language_from_extension
# =========================================================================== #


class TestLanguageExtensionMapping:
    """Test the file-extension-to-language lookup table."""

    @pytest.mark.parametrize(
        "ext, language",
        [
            (".py", "Python"),
            (".js", "JavaScript"),
            (".ts", "TypeScript"),
            (".java", "Java"),
            (".go", "Go"),
            (".rs", "Rust"),
            (".cpp", "C++"),
            (".c", "C"),
            (".cs", "C#"),
            (".rb", "Ruby"),
            (".php", "PHP"),
            (".swift", "Swift"),
            (".kt", "Kotlin"),
            (".scala", "Scala"),
            (".r", "R"),
            (".m", "Objective-C"),
            (".sh", "Shell"),
            (".yml", "YAML"),
            (".yaml", "YAML"),
            (".json", "JSON"),
            (".xml", "XML"),
            (".md", "Markdown"),
            (".rst", "reStructuredText"),
            (".tex", "LaTeX"),
        ],
    )
    def test_known_extensions(self, ext: str, language: str):
        assert _client()._get_language_from_extension(ext) == language

    @pytest.mark.parametrize("ext", [".unknown", ".xyz", "", ".BLAH"])
    def test_unknown_extensions_return_other(self, ext: str):
        assert _client()._get_language_from_extension(ext) == "Other"

    def test_case_insensitive(self):
        assert _client()._get_language_from_extension(".PY") == "Python"
        assert _client()._get_language_from_extension(".Js") == "JavaScript"


# =========================================================================== #
# get_repository_info
# =========================================================================== #


class TestGetRepositoryInfo:
    """Test get_repository_info delegates to GitHubService and builds RepoInfo."""

    async def test_success(self):
        mock_svc = _mock_github_service(
            get_repository_info={
                "owner": "test-owner",
                "name": "test-repo",
                "default_branch": "main",
                "description": "A test repo",
                "language": "Python",
            }
        )
        with _patch_github_service(mock_svc):
            info = await _client().get_repository_info("https://github.com/test-owner/test-repo")
        assert info.owner == "test-owner"
        assert info.name == "test-repo"
        assert info.default_branch == "main"
        assert info.description == "A test repo"
        assert info.language == "Python"

    async def test_uses_parsed_owner_as_fallback(self):
        """When API response is missing owner/name, fall back to URL-parsed values."""
        mock_svc = _mock_github_service(
            get_repository_info={
                "default_branch": "develop",
            }
        )
        with _patch_github_service(mock_svc):
            info = await _client().get_repository_info(
                "https://github.com/fallback-owner/fallback-repo"
            )
        assert info.owner == "fallback-owner"
        assert info.name == "fallback-repo"
        assert info.default_branch == "develop"

    @pytest.mark.parametrize(
        "exc_cls, match",
        [
            (ValueError, "bad data"),
            (KeyError, "missing_key"),
            (OSError, "network down"),
        ],
        ids=["ValueError", "KeyError", "OSError"],
    )
    async def test_error_passthrough(self, exc_cls, match):
        """ValueError/KeyError/OSError from the service are re-raised."""
        mock_svc = _mock_github_service()
        mock_svc.get_repository_info = AsyncMock(side_effect=exc_cls(match))
        with _patch_github_service(mock_svc):
            with pytest.raises(exc_cls):
                await _client().get_repository_info("https://github.com/owner/repo")


# =========================================================================== #
# read_pr_file
# =========================================================================== #


class TestReadPrFile:
    """Test read_pr_file delegates to GitHubService and handles errors."""

    async def test_success_returns_content(self):
        mock_svc = _mock_github_service(read_pull_request_file="file content here")
        with _patch_github_service(mock_svc):
            content = await _client().read_pr_file(
                "https://github.com/owner/repo", 1, "src/main.py"
            )
        assert content == "file content here"
        mock_svc.read_pull_request_file.assert_awaited_once_with(
            repo_name="owner/repo", pr_number=1, filepath="src/main.py", ref="head"
        )

    async def test_custom_ref(self):
        mock_svc = _mock_github_service(read_pull_request_file="base version")
        with _patch_github_service(mock_svc):
            content = await _client().read_pr_file(
                "https://github.com/owner/repo", 5, "README.md", ref="base"
            )
        assert content == "base version"
        mock_svc.read_pull_request_file.assert_awaited_once_with(
            repo_name="owner/repo", pr_number=5, filepath="README.md", ref="base"
        )

    async def test_none_content_returns_empty_string(self):
        mock_svc = _mock_github_service(read_pull_request_file=None)
        with _patch_github_service(mock_svc):
            content = await _client().read_pr_file("https://github.com/owner/repo", 1, "empty.py")
        assert content == ""

    async def test_file_not_found_propagates(self):
        mock_svc = _mock_github_service()
        mock_svc.read_pull_request_file = AsyncMock(side_effect=FileNotFoundError("not found"))
        with _patch_github_service(mock_svc):
            with pytest.raises(FileNotFoundError):
                await _client().read_pr_file("https://github.com/owner/repo", 1, "missing.py")

    @pytest.mark.parametrize(
        "exc_cls, msg",
        [
            (OSError, "connection refused"),
            (ValueError, "invalid ref"),
            (KeyError, "missing"),
        ],
        ids=["OSError", "ValueError", "KeyError"],
    )
    async def test_io_errors_propagate(self, exc_cls, msg):
        mock_svc = _mock_github_service()
        mock_svc.read_pull_request_file = AsyncMock(side_effect=exc_cls(msg))
        with _patch_github_service(mock_svc):
            with pytest.raises(exc_cls):
                await _client().read_pr_file("https://github.com/owner/repo", 1, "file.py")


# =========================================================================== #
# analyze_pull_request
# =========================================================================== #


class TestAnalyzePullRequest:
    """Test PR analysis aggregation: stats, language breakdown, totals."""

    def _make_file(self, path="src/main.py", change_type="modified", add=5, delete=2):
        f = MagicMock()
        f.file_path = path
        f.change_type = change_type
        f.additions = add
        f.deletions = delete
        return f

    async def test_single_file_stats(self):
        mock_pr = MagicMock(number=1, title="fix")
        mock_file = self._make_file()

        mock_svc = _mock_github_service(
            get_pull_request=mock_pr, list_pull_request_files=[mock_file]
        )
        with _patch_github_service(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/owner/repo", 1)

        assert result["total_additions"] == 5
        assert result["total_deletions"] == 2
        assert result["statistics"]["total"] == 1
        assert result["statistics"]["modified"] == 1
        assert result["statistics"]["added"] == 0

    async def test_multi_file_language_breakdown(self):
        files = [
            self._make_file("src/app.py", "modified", 10, 3),
            self._make_file("src/util.py", "added", 20, 0),
            self._make_file("tests/test.js", "modified", 5, 1),
            self._make_file("README.md", "modified", 2, 0),
        ]
        mock_pr = MagicMock(number=2)
        mock_svc = _mock_github_service(get_pull_request=mock_pr, list_pull_request_files=files)
        with _patch_github_service(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/owner/repo", 2)

        assert result["total_additions"] == 37
        assert result["total_deletions"] == 4
        assert result["statistics"]["total"] == 4
        assert result["statistics"]["modified"] == 3
        assert result["statistics"]["added"] == 1

        langs = result["languages"]
        assert langs["Python"]["files"] == 2
        assert langs["Python"]["additions"] == 30
        assert langs["JavaScript"]["files"] == 1
        assert langs["Markdown"]["files"] == 1

    async def test_empty_pr(self):
        mock_pr = MagicMock(number=3)
        mock_svc = _mock_github_service(get_pull_request=mock_pr, list_pull_request_files=[])
        with _patch_github_service(mock_svc):
            result = await _client().analyze_pull_request("https://github.com/owner/repo", 3)

        assert result["total_additions"] == 0
        assert result["total_deletions"] == 0
        assert result["statistics"]["total"] == 0
        assert result["languages"] == {}
