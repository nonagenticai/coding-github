"""Tests for services/github/github_service.py

Covers GitHubService lifecycle, get_repository_info, create_file,
create_pull_request, retry logic, and the get_github_service factory.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.github_service import (
    GITHUB_COMPARE_FILES_CAP,
    CompareFilesTruncatedError,
    GitHubConfig,
    GitHubService,
    get_github_service,
)
from tests.unit.github_rest_fake import rate_limit_overview

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_config(**overrides) -> GitHubConfig:
    defaults = {"token_id": "test-token", "max_retries": 2, "retry_delay": 0.01}
    defaults.update(overrides)
    return GitHubConfig(**defaults)


def _make_service(config=None, *, github=None, skip_rate_limit=True):
    """Build a GitHubService with pre-injected mocks for unit testing."""
    config = config or _make_config()
    mock_token_mgr = AsyncMock()
    mock_token_mgr.retrieve_token = AsyncMock(return_value="ghp_tok")

    with patch(
        "services.github.github_service.get_secure_token_manager",
        return_value=mock_token_mgr,
    ):
        svc = GitHubService(config)

    svc._github = github or MagicMock()
    # MagicMock (not AsyncMock) — only .close() is async (for __aexit__).
    # AsyncMock would make all attrs return coroutines → event loop deadlock.
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    svc._async_client = mock_client

    if skip_rate_limit:
        svc._wait_for_rate_limit = AsyncMock()

    return svc


def _mock_repo(**overrides):
    """Build a MagicMock that looks like a github.Repository.Repository."""
    defaults = {
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "description": "desc",
        "language": "Python",
        "default_branch": "main",
        "stargazers_count": 10,
        "forks_count": 2,
        "private": False,
        "html_url": "https://github.com/owner/test-repo",
        "clone_url": "https://github.com/owner/test-repo.git",
        "ssh_url": "git@github.com:owner/test-repo.git",
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2024, 6, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    repo = MagicMock()
    for k, v in defaults.items():
        setattr(repo, k, v)
    repo.owner.login = "owner"
    return repo


def _rate_limit_ok():
    return rate_limit_overview(remaining=4999, limit=5000)


# =========================================================================== #
# Async context manager lifecycle
# =========================================================================== #


class TestServiceLifecycle:
    async def test_aenter_sets_up_session_aexit_closes(self):
        config = _make_config()
        mock_token_mgr = AsyncMock()
        mock_token_mgr.retrieve_token = AsyncMock(return_value="ghp_valid_token_abc123")

        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        with (
            patch(
                "services.github.github_service.get_secure_token_manager",
                return_value=mock_token_mgr,
            ),
            patch("services.github.github_service.aiohttp") as mock_aiohttp,
            patch("services.github.github_service.HAS_AIOHTTP", True),
            patch("services.github.github_service.HAS_GITHUB", True),
            patch("services.github.github_service.Github"),
        ):
            mock_aiohttp.ClientSession.return_value = mock_session
            service = GitHubService(config)
            result = await service.__aenter__()
            assert result is service
            assert service._async_client is mock_session

            await service.__aexit__(None, None, None)
            mock_session.close.assert_awaited_once()
            assert service._async_client is None

    async def test_aenter_raises_on_missing_token(self):
        config = _make_config()
        mock_token_mgr = AsyncMock()
        mock_token_mgr.retrieve_token = AsyncMock(return_value=None)

        with patch(
            "services.github.github_service.get_secure_token_manager",
            return_value=mock_token_mgr,
        ):
            service = GitHubService(config)
            with pytest.raises(ValueError, match="not found or expired"):
                await service.__aenter__()

    async def test_aexit_is_noop_when_no_client(self):
        """If __aenter__ was never called, __aexit__ should not crash."""
        config = _make_config()
        mock_token_mgr = AsyncMock()
        mock_token_mgr.retrieve_token = AsyncMock(return_value="tok")
        with patch(
            "services.github.github_service.get_secure_token_manager",
            return_value=mock_token_mgr,
        ):
            service = GitHubService(config)
            # _async_client is None by default
            await service.__aexit__(None, None, None)  # should not raise


# =========================================================================== #
# get_repository_info
# =========================================================================== #


class TestGetRepositoryInfo:
    async def test_returns_full_dict(self):
        repo = _mock_repo()
        github = MagicMock()
        github.get_repo.return_value = repo
        github.get_rate_limit.return_value = _rate_limit_ok()

        svc = _make_service(github=github)
        with (
            patch("services.github.github_service.Repository", type(repo)),
            patch("github.Repository.Repository", type(repo)),
        ):
            info = await svc.get_repository_info("owner/test-repo")

        assert info["name"] == "test-repo"
        assert info["language"] == "Python"
        assert info["owner"] == "owner"
        assert info["default_branch"] == "main"
        assert info["stars"] == 10
        assert info["forks"] == 2
        assert info["private"] is False
        assert "created_at" in info
        assert "updated_at" in info

    async def test_none_dates(self):
        repo = _mock_repo(created_at=None, updated_at=None)
        github = MagicMock()
        github.get_repo.return_value = repo
        github.get_rate_limit.return_value = _rate_limit_ok()

        svc = _make_service(github=github)
        with (
            patch("services.github.github_service.Repository", type(repo)),
            patch("github.Repository.Repository", type(repo)),
        ):
            info = await svc.get_repository_info("owner/test-repo")

        assert info["created_at"] is None
        assert info["updated_at"] is None


# =========================================================================== #
# create_file
# =========================================================================== #


class TestCreateFile:
    async def test_delegates_to_repo_create_file(self):
        mock_author = MagicMock()
        mock_author.name = "bot"
        mock_commit = MagicMock(sha="abc123", message="add file", author=mock_author)
        mock_content = MagicMock(html_url="https://github.com/file")

        repo = MagicMock()
        repo.create_file.return_value = {"commit": mock_commit, "content": mock_content}

        github = MagicMock()
        github.get_repo.return_value = repo
        github.get_rate_limit.return_value = _rate_limit_ok()

        svc = _make_service(github=github)
        with (
            patch("services.github.github_service.Repository", type(repo)),
            patch("github.Repository.Repository", type(repo)),
        ):
            result = await svc.create_file("owner/repo", "new.py", "content", "add file")

        assert result["sha"] == "abc123"
        assert result["url"] == "https://github.com/file"
        assert result["commit"]["message"] == "add file"
        assert result["commit"]["author"] == "bot"


# =========================================================================== #
# create_pull_request
# =========================================================================== #


class TestCreatePullRequest:
    async def test_creates_and_returns_pr(self):
        mock_pr_create = MagicMock(number=42)
        mock_pr_get = MagicMock(
            id=42,
            number=42,
            title="feature",
            body="desc",
            html_url="https://github.com/pr/42",
            state="open",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            updated_at=datetime(2024, 1, 1, tzinfo=UTC),
            merged_at=None,
            user=MagicMock(login="author"),
            base=MagicMock(ref="main"),
            head=MagicMock(ref="feature-branch"),
            labels=[],
            requested_reviewers=[],
        )

        repo = MagicMock()
        repo.create_pull.return_value = mock_pr_create
        repo.get_pull.return_value = mock_pr_get

        github = MagicMock()
        github.get_repo.return_value = repo
        github.get_rate_limit.return_value = _rate_limit_ok()

        svc = _make_service(github=github)
        with (
            patch("services.github.github_service.Repository", type(repo)),
            patch("github.Repository.Repository", type(repo)),
        ):
            # Mock get_pull_request to avoid model construction complexity
            mock_pr_result = MagicMock(number=42)
            svc.get_pull_request = AsyncMock(return_value=mock_pr_result)

            pr = await svc.create_pull_request("owner/repo", "feature", "desc", "feature-branch")

        assert pr.number == 42


# =========================================================================== #
# Retry logic
# =========================================================================== #


class TestRetryLogic:
    async def test_exhausts_retries_on_server_error(self):
        """After max_retries server errors, the exception propagates."""
        svc = _make_service()

        class _TestGithubError(Exception):
            pass

        exc = _TestGithubError("Server error")
        exc.status = 500

        class _NeverMatchError(Exception):
            pass

        with (
            patch("services.github.github_service.GithubException", _TestGithubError),
            patch("services.github.github_service.RateLimitExceededException", _NeverMatchError),
        ):
            failing_func = MagicMock(side_effect=exc)
            with pytest.raises(_TestGithubError, match="Server error"):
                await svc._execute_with_retry(failing_func)

    async def test_retries_on_500_then_succeeds(self):
        """A transient 500 followed by success should return the result."""
        svc = _make_service(config=_make_config(max_retries=3))

        class _TestGithubError(Exception):
            pass

        exc = _TestGithubError("transient")
        exc.status = 500

        class _NeverMatchError(Exception):
            pass

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise exc
            return "success"

        with (
            patch("services.github.github_service.GithubException", _TestGithubError),
            patch("services.github.github_service.RateLimitExceededException", _NeverMatchError),
        ):
            func = MagicMock(side_effect=side_effect)
            result = await svc._execute_with_retry(func)

        assert result == "success"
        assert call_count == 3

    async def test_client_error_not_retried(self):
        """A 4xx GithubException should NOT be retried."""
        svc = _make_service()

        class _TestGithubError(Exception):
            pass

        exc = _TestGithubError("Not found")
        exc.status = 404

        class _NeverMatchError(Exception):
            pass

        with (
            patch("services.github.github_service.GithubException", _TestGithubError),
            patch("services.github.github_service.RateLimitExceededException", _NeverMatchError),
        ):
            func = MagicMock(side_effect=exc)
            with pytest.raises(_TestGithubError, match="Not found"):
                await svc._execute_with_retry(func)
            # Should have been called exactly once (no retry for 4xx)
            assert func.call_count == 1


# =========================================================================== #
# get_github_service factory
# =========================================================================== #


class TestGetGitHubServiceFactory:
    async def test_returns_service_instance_only_under_schemathesis(self):
        with (
            patch("services.github.github_service.get_settings") as mock_settings,
            patch.dict("os.environ", {"SCHEMATHESIS_TESTING": "true"}),
        ):
            mock_settings.return_value = MagicMock(spec=["is_production"], is_production=False)
            service = await get_github_service("test-owner/coding-github")
        assert isinstance(service, GitHubService)
        assert service.config.token == "schemathesis_test_token"

    async def test_raises_without_token_even_with_every_legacy_pat_set(self):
        """⛔ No PAT fallback (owner ruling 2026-09-30): a platform repo with every
        legacy PAT variable armed still gets NO credential on this tenantless path."""
        with (
            patch("services.github.github_service.get_settings") as mock_settings,
            patch.dict(
                "os.environ",
                {
                    "SCHEMATHESIS_TESTING": "",
                    "GITHUB_ACCESS_TOKEN": "ghp_env_pat",
                    "GITHUB_TOKEN": "ghp_env_pat",
                    "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat",
                    "CODING_GITHUB_SYSTEM_GITHUB_TOKEN": "ghp_env_pat",
                },
            ),
        ):
            mock_settings.return_value = MagicMock(spec=["is_production"], is_production=True)
            with pytest.raises(ValueError, match="no PAT fallback"):
                await get_github_service("test-owner/coding-github")


# =========================================================================== #
# check_rate_limit
# =========================================================================== #


class TestCheckRateLimit:
    async def test_returns_rate_limit_info(self):
        github = MagicMock()
        github.get_rate_limit.return_value = rate_limit_overview(remaining=4500, limit=5000)
        svc = _make_service(github=github, skip_rate_limit=False)

        info = await svc.check_rate_limit()
        assert info.remaining == 4500
        assert info.limit == 5000
        assert info.used == 500
        assert info.reset_at == datetime(2099, 1, 1, tzinfo=UTC)
        assert info.probe_failed is False

    async def test_returns_conservative_default_on_error(self):
        github = MagicMock()
        github.get_rate_limit.side_effect = Exception("API down")
        svc = _make_service(github=github, skip_rate_limit=False)

        info = await svc.check_rate_limit()
        # Conservative placeholder numbers, flagged so nothing sleeps on them.
        assert info.remaining == 0
        assert info.limit == 5000
        assert info.probe_failed is True

    async def test_raises_when_github_not_initialized(self):
        svc = _make_service(skip_rate_limit=False)
        svc._github = None
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.check_rate_limit()


# =========================================================================== #
# get_repository raises when not initialized
# =========================================================================== #


class TestGetRepository:
    async def test_raises_when_github_not_initialized(self):
        svc = _make_service()
        svc._github = None
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.get_repository("owner/repo")


# =========================================================================== #
# compare_files — exact list or refusal, never a possibly-truncated list
# =========================================================================== #

_MERGE_BASE = "cd" * 20


class _CmpFile:
    def __init__(self, filename: str, status: str = "modified", previous=None) -> None:
        self.filename = filename
        self.status = status
        self.previous_filename = previous


class _LazyFilesComparison:
    """PyGithub's lazy ``Comparison``: ``files`` raises ``files_raises`` on read."""

    def __init__(self, files, files_raises: Exception | None = None) -> None:
        self._files = files
        self._files_raises = files_raises
        self.status = "ahead"
        self.ahead_by = 1
        self.behind_by = 0
        self.merge_base_commit = MagicMock(sha=_MERGE_BASE)

    @property
    def files(self):
        if self._files_raises is not None:
            raise self._files_raises
        return self._files


def _compare_service(comparisons: list) -> tuple[GitHubService, list]:
    svc = _make_service()
    calls: list = []

    def _compare(base, head):
        calls.append((base, head))
        return comparisons.pop(0)

    repo = MagicMock()
    repo.compare = _compare
    svc.get_repository = AsyncMock(return_value=repo)
    return svc, calls


def _files(n: int) -> list[_CmpFile]:
    return [_CmpFile(f"f{i:04d}.py") for i in range(n)]


class TestCompareFiles:
    def test_cap_is_githubs_300(self):
        assert GITHUB_COMPARE_FILES_CAP == 300

    def test_truncated_error_is_not_a_value_error(self):
        """Routers map ValueError to 404; a truncation must never read as not-found."""
        assert not issubclass(CompareFilesTruncatedError, ValueError)

    async def test_299_files_are_returned_in_full(self):
        svc, _ = _compare_service([_LazyFilesComparison(_files(299))])
        out = await svc.compare_files("owner/repo", "abc1234", "main")
        assert out["files_count"] == 299
        assert len(out["files"]) == 299

    async def test_300_files_raise_and_return_nothing(self):
        """⛔ MUTANT: ``>`` instead of ``>=`` (or no check) -> 300 files come back as an
        exact answer although GitHub may have cut them."""
        svc, _ = _compare_service([_LazyFilesComparison(_files(300))])
        with pytest.raises(CompareFilesTruncatedError) as exc:
            await svc.compare_files("owner/repo", "abc1234", "main")
        assert exc.value.count == 300

    async def test_sorted_and_previous_filename_only_on_rename(self):
        files = [
            _CmpFile("z.py", "renamed", previous="y.py"),
            _CmpFile("a.py", "modified", previous="stray.py"),
        ]
        svc, calls = _compare_service([_LazyFilesComparison(files)])
        out = await svc.compare_files("owner/repo", "abc1234", "main")
        assert out["files"] == [
            {"filename": "a.py", "status": "modified", "previous_filename": None},
            {"filename": "z.py", "status": "renamed", "previous_filename": "y.py"},
        ]
        assert out["base_is_ancestor_of_head"] is True
        assert calls == [("abc1234", "main")]

    async def test_lazy_files_fetch_is_retried_inside_the_wrapper(self):
        """⛔ ``_ = comparison.files`` in ``_fetch`` puts the files read inside
        ``_execute_with_retry``. MUTANT: drop it -> the 502 surfaces in ``_shape``,
        where nothing retries it, and this raises instead of succeeding."""
        from github import GithubException

        flaky = GithubException(502, {"message": "Bad Gateway"}, None)
        svc, calls = _compare_service(
            [_LazyFilesComparison([], files_raises=flaky), _LazyFilesComparison(_files(2))]
        )
        out = await svc.compare_files("owner/repo", "abc1234", "main")
        assert out["files_count"] == 2
        assert len(calls) == 2
