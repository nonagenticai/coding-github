"""Extended tests for services/github/github_service.py — covering uncovered lines."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.github_service import (
    GitHubConfig,
    GitHubService,
    RateLimitInfo,
    get_github_service,
)
from tests.unit.github_rest_fake import rate_limit_overview

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = dict(token_id="test-token", max_retries=2, retry_delay=0.01)
    defaults.update(overrides)
    return GitHubConfig(**defaults)


def _make_service(config=None, *, github=None, async_client=True, wait_for_rate_limit=True):
    """Create a GitHubService with mocked internals."""
    config = config or _make_config()
    mock_token_mgr = AsyncMock()
    mock_token_mgr.retrieve_token = AsyncMock(return_value="ghp_tok")
    with patch(
        "services.github.github_service.get_secure_token_manager",
        return_value=mock_token_mgr,
    ):
        svc = GitHubService(config)
    svc._token_manager = mock_token_mgr
    if github is not None:
        svc._github = github
    if async_client:
        # Use MagicMock (not AsyncMock) for _async_client because the production
        # code (aiohttp.ClientSession) has sync attribute access. AsyncMock makes
        # ALL attribute accesses return coroutines, which causes unawaited
        # coroutine accumulation → event loop deadlock when running multiple tests.
        # Only .close() needs to be async (called via `await` in __aexit__).
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        svc._async_client = mock_client
    if wait_for_rate_limit:
        svc._wait_for_rate_limit = AsyncMock()
    return svc


def _mock_github(repo=None):
    """Return a mock Github object with rate-limit plumbing."""
    g = MagicMock()
    g.get_rate_limit.return_value = rate_limit_overview(remaining=4999, limit=5000)
    if repo is not None:
        g.get_repo.return_value = repo
    return g


# ===========================================================================
# Lines 25-27 / 31-33: import fallbacks (HAS_AIOHTTP=False, HAS_GITHUB=False)
# These branches only run at import time when the libraries are absent.
# We test the HAS_GITHUB=False warning path in __init__.
# ===========================================================================


class TestImportFallbacks:
    async def test_init_warns_when_has_github_false(self):
        """Lines 75-78: warning when HAS_GITHUB is False."""
        mock_token_mgr = AsyncMock()
        with (
            patch(
                "services.github.github_service.get_secure_token_manager",
                return_value=mock_token_mgr,
            ),
            patch("services.github.github_service.HAS_GITHUB", False),
            patch("services.github.github_service.logger") as mock_logger,
        ):
            GitHubService(_make_config())
            mock_logger.warning.assert_called_once()
            assert "limited functionality" in mock_logger.warning.call_args[0][0]


# ===========================================================================
# Lines 99-101, 104-115: __aenter__ schemathesis mock token paths
# ===========================================================================


class TestAenterSchemathesis:
    async def test_aenter_schemathesis_token_bypass(self):
        """Lines 99-101: SCHEMATHESIS_TESTING=true generates mock token."""
        config = _make_config()
        mock_token_mgr = AsyncMock()
        mock_token_mgr.retrieve_token = AsyncMock(return_value=None)

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
            patch.dict(
                "os.environ", {"SCHEMATHESIS_TESTING": "true", "ENVIRONMENT": "development"}
            ),
        ):
            mock_aiohttp.ClientSession.return_value = mock_session
            svc = GitHubService(config)
            result = await svc.__aenter__()
            assert result is svc
            assert svc._async_client is mock_session
            # schemathesis branch sets _github = None
            assert svc._github is None
            await svc.__aexit__(None, None, None)

    async def test_aenter_schemathesis_token_starts_with_schemathesis(self):
        """Lines 104-115: token starting with 'schemathesis' creates mock client."""
        config = _make_config()
        mock_token_mgr = AsyncMock()
        mock_token_mgr.retrieve_token = AsyncMock(return_value="schemathesis_mock_token_abc")

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
        ):
            mock_aiohttp.ClientSession.return_value = mock_session
            svc = GitHubService(config)
            result = await svc.__aenter__()
            assert result is svc
            assert svc._github is None
            # Check User-Agent has Schemathesis tag
            call_kwargs = mock_aiohttp.ClientSession.call_args[1]
            assert "Schemathesis" in call_kwargs["headers"]["User-Agent"]
            await svc.__aexit__(None, None, None)


# ===========================================================================
# Lines 137, 147-149: check_rate_limit
# ===========================================================================


class TestCheckRateLimit:
    async def test_check_rate_limit_no_github_client(self):
        """Line 137: raises RuntimeError when _github is None."""
        svc = _make_service(github=None)
        svc._github = None
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.check_rate_limit()

    async def test_check_rate_limit_exception_returns_conservative_defaults(self):
        """On failure, report conservative placeholder numbers, flagged
        ``probe_failed`` so ``_wait_for_rate_limit`` never sleeps on them (D14)."""
        mock_gh = MagicMock()
        mock_gh.get_rate_limit.side_effect = RuntimeError("API error")
        svc = _make_service(github=mock_gh)
        info = await svc.check_rate_limit()
        assert isinstance(info, RateLimitInfo)
        assert info.remaining == 0
        assert info.limit == 5000
        assert info.used == 5000
        assert info.probe_failed is True


# ===========================================================================
# Lines 160-165: _wait_for_rate_limit when remaining < 10
# ===========================================================================


class TestWaitForRateLimit:
    async def test_wait_for_rate_limit_sleeps_when_low(self):
        """Lines 160-165: sleeps when rate limit remaining < 10."""
        config = _make_config()
        mock_gh = MagicMock()
        svc = _make_service(config=config, github=mock_gh, wait_for_rate_limit=False)
        # Force _last_rate_check far in the past so the check runs
        svc._last_rate_check = datetime.min.replace(tzinfo=UTC)

        future_reset = datetime.now(UTC) + timedelta(seconds=2)
        rate_info = RateLimitInfo(remaining=5, reset_at=future_reset, limit=5000, used=4995)
        svc.check_rate_limit = AsyncMock(return_value=rate_info)

        with patch(
            "services.github.github_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await svc._wait_for_rate_limit()
            mock_sleep.assert_awaited_once()
            assert mock_sleep.call_args[0][0] > 0


# ===========================================================================
# Lines 174-180: _execute_with_retry — RateLimitExceededException path
# ===========================================================================


class TestExecuteWithRetry:
    async def test_retry_on_rate_limit_exceeded(self):
        """Lines 174-180: retries on RateLimitExceededException, then succeeds."""
        config = _make_config(max_retries=3)
        svc = _make_service(config=config)

        class _RateExc(Exception):  # noqa: N818
            pass

        exc = _RateExc("rate limited")
        exc.reset_time = datetime.now(UTC).timestamp() + 1

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise exc
            return "ok"

        with (
            patch("services.github.github_service.RateLimitExceededException", _RateExc),
            patch("services.github.github_service.GithubException", Exception),
            patch("services.github.github_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await svc._execute_with_retry(flaky_func)
            assert result == "ok"

    async def test_retry_rate_limit_exhausted_raises(self):
        """Lines 174-175: raises after max_retries on RateLimitExceededException."""
        config = _make_config(max_retries=2)
        svc = _make_service(config=config)

        class _RateExc(Exception):  # noqa: N818
            pass

        exc = _RateExc("rate limited")
        exc.reset_time = datetime.now(UTC).timestamp() + 1

        failing_func = MagicMock(side_effect=exc)

        with (
            patch("services.github.github_service.RateLimitExceededException", _RateExc),
            patch("services.github.github_service.GithubException", Exception),
            patch("services.github.github_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_RateExc):
                await svc._execute_with_retry(failing_func)


# ===========================================================================
# Lines 190, 195: get_repository — not initialised / wrong return type
# ===========================================================================


class TestGetRepository:
    async def test_get_repository_not_initialized(self):
        """Line 190: raises RuntimeError when _github is None."""
        svc = _make_service(github=None)
        svc._github = None
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.get_repository("owner/repo")

    async def test_get_repository_wrong_type(self):
        """Line 195: raises ValueError when result is not Repository."""
        mock_gh = _mock_github()
        mock_gh.get_repo.return_value = "not-a-repo"
        svc = _make_service(github=mock_gh)
        with (
            patch("services.github.github_service.Repository", MagicMock),
            patch("github.Repository.Repository", MagicMock),
        ):
            with pytest.raises(ValueError, match="Expected Repository"):
                await svc.get_repository("owner/repo")


# ===========================================================================
# Lines 220-234: clone_repository
# ===========================================================================


class TestCloneRepository:
    async def test_clone_repository_success(self):
        """Lines 220-231: successful clone."""
        svc = _make_service()
        svc._token_manager.retrieve_token = AsyncMock(return_value="ghp_tok123")

        mock_git_mod = MagicMock()
        with patch.dict("sys.modules", {"git": mock_git_mod}):
            result = await svc.clone_repository(
                "https://github.com/owner/my-repo.git", "/tmp/target"
            )
            assert result == "/tmp/target"
            mock_git_mod.Repo.clone_from.assert_called_once()
            # Verify token was injected into URL
            call_args = mock_git_mod.Repo.clone_from.call_args
            assert "ghp_tok123@" in call_args[0][0]

    async def test_clone_repository_strips_git_suffix(self):
        """Line 224-225: strips .git suffix from repo name."""
        svc = _make_service()
        svc._token_manager.retrieve_token = AsyncMock(return_value="ghp_tok123")
        mock_git_mod = MagicMock()
        with patch.dict("sys.modules", {"git": mock_git_mod}):
            result = await svc.clone_repository(
                "https://github.com/owner/my-repo.git", "/tmp/target"
            )
            assert result == "/tmp/target"

    async def test_clone_repository_failure(self):
        """Lines 232-234: clone failure raises."""
        svc = _make_service()
        svc._token_manager.retrieve_token = AsyncMock(return_value="ghp_tok123")
        mock_git_mod = MagicMock()
        mock_git_mod.Repo.clone_from.side_effect = RuntimeError("clone failed")
        with patch.dict("sys.modules", {"git": mock_git_mod}):
            with pytest.raises(RuntimeError, match="clone failed"):
                await svc.clone_repository("https://github.com/owner/repo", "/tmp/target")


# ===========================================================================
# Lines 238-240: get_pull_request
# ===========================================================================


class TestGetPullRequest:
    async def test_get_pull_request(self):
        """Lines 238-255: get_pull_request returns PullRequest model."""
        mock_pr = MagicMock()
        mock_pr.id = 42
        mock_pr.number = 42
        mock_pr.title = "Test PR"
        mock_pr.body = "description"
        mock_pr.html_url = "https://github.com/pr/42"
        mock_pr.state = "open"
        mock_pr.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_pr.updated_at = datetime(2024, 1, 2, tzinfo=UTC)
        mock_pr.merged_at = None
        mock_pr.user = MagicMock(login="author")
        mock_pr.base = MagicMock(ref="main")
        mock_pr.head = MagicMock(ref="feature")
        mock_pr.labels = []
        mock_pr.requested_reviewers = []

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        # PullRequest stub in conftest doesn't accept labels/reviewers,
        # so we patch it to accept any kwargs.
        mock_pr_cls = MagicMock()
        mock_pr_instance = MagicMock(number=42, title="Test PR")
        mock_pr_cls.return_value = mock_pr_instance

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
            patch("services.github.github_service.PullRequest", mock_pr_cls),
        ):
            pr = await svc.get_pull_request("owner/repo", 42)
            assert pr.number == 42


# ===========================================================================
# Lines 281-302: list_pull_request_files
# ===========================================================================


class TestListPullRequestFiles:
    async def test_list_pull_request_files(self):
        """Lines 281-302: paginated file listing."""
        mock_file = MagicMock()
        mock_file.filename = "src/main.py"
        mock_file.status = "modified"
        mock_file.additions = 10
        mock_file.deletions = 5
        mock_file.patch = "@@ -1,5 +1,10 @@"

        mock_paginated = MagicMock()
        mock_paginated.get_page = MagicMock(side_effect=[[mock_file], []])

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = mock_paginated

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        # PRFileChange stub in conftest uses 'filename' not 'file_path'; patch it.
        mock_prf_cls = MagicMock()
        mock_prf_instance = MagicMock(file_path="src/main.py", additions=10)
        mock_prf_cls.return_value = mock_prf_instance

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
            patch("services.github.github_service.PRFileChange", mock_prf_cls),
        ):
            files = await svc.list_pull_request_files("owner/repo", 1)
            assert len(files) == 1
            assert files[0].file_path == "src/main.py"
            assert files[0].additions == 10


# ===========================================================================
# Lines 308-329: read_pull_request_file
# ===========================================================================


class TestReadPullRequestFile:
    async def test_read_pr_file_head(self):
        """Lines 308-322: read file at head ref with base64 encoding."""
        mock_content = MagicMock()
        mock_content.encoding = "base64"
        mock_content.content = base64.b64encode(b"hello world").decode()

        mock_pr = MagicMock()
        mock_pr.head = MagicMock(sha="head-sha")
        mock_pr.base = MagicMock(sha="base-sha")

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_contents.return_value = mock_content
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.read_pull_request_file("owner/repo", 1, "src/main.py", ref="head")
            assert result == "hello world"

    async def test_read_pr_file_base(self):
        """Lines 312-313: read file at base ref."""
        mock_content = MagicMock()
        mock_content.encoding = "utf-8"
        mock_content.decoded_content = b"base content"

        mock_pr = MagicMock()
        mock_pr.head = MagicMock(sha="head-sha")
        mock_pr.base = MagicMock(sha="base-sha")

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_contents.return_value = mock_content
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.read_pull_request_file("owner/repo", 1, "src/main.py", ref="base")
            assert result == "base content"

    async def test_read_pr_file_custom_ref(self):
        """Lines 314-315: custom ref string."""
        mock_content = MagicMock()
        mock_content.encoding = "base64"
        mock_content.content = base64.b64encode(b"custom").decode()

        mock_pr = MagicMock()
        mock_pr.head = MagicMock(sha="head-sha")
        mock_pr.base = MagicMock(sha="base-sha")

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_contents.return_value = mock_content
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.read_pull_request_file("owner/repo", 1, "f.py", ref="abc123")
            assert result == "custom"

    async def test_read_pr_file_directory_raises(self):
        """Lines 318-319: directory path raises ValueError."""
        mock_pr = MagicMock()
        mock_pr.head = MagicMock(sha="head-sha")
        mock_pr.base = MagicMock(sha="base-sha")

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_contents.return_value = [MagicMock(), MagicMock()]  # list = directory
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            with pytest.raises(ValueError, match="is a directory"):
                await svc.read_pull_request_file("owner/repo", 1, "src/")

    async def test_read_pr_file_404_raises_file_not_found(self):
        """Lines 326-328: GithubException(404) becomes FileNotFoundError."""
        from services.github.github_service import GithubException

        mock_pr = MagicMock()
        mock_pr.head = MagicMock(sha="head-sha")

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_contents.side_effect = GithubException(status=404)
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            with pytest.raises(FileNotFoundError, match="not found"):
                await svc.read_pull_request_file("owner/repo", 1, "missing.py")


# ===========================================================================
# Lines 335-338: add_pull_request_comment
# ===========================================================================


class TestAddPullRequestComment:
    async def test_add_pr_comment(self):
        """Lines 335-344: adds issue comment and returns dict."""
        mock_comment = MagicMock()
        mock_comment.id = 99
        mock_comment.body = "Nice work"
        mock_comment.user = MagicMock(login="reviewer")
        mock_comment.created_at = datetime(2024, 3, 1, tzinfo=UTC)
        mock_comment.updated_at = datetime(2024, 3, 1, tzinfo=UTC)

        mock_pr = MagicMock()
        mock_pr.create_issue_comment.return_value = mock_comment

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.add_pull_request_comment("owner/repo", 1, "Nice work")
            assert result["id"] == 99
            assert result["body"] == "Nice work"
            assert result["author"] == "reviewer"


# ===========================================================================
# Lines 356-361: add_pull_request_review_comment
# ===========================================================================


class TestAddPullRequestReviewComment:
    async def test_add_review_comment(self):
        """Lines 356-369: adds review comment on specific line."""
        mock_review_comment = MagicMock()
        mock_review_comment.id = 200
        mock_review_comment.body = "Fix this"
        mock_review_comment.user = MagicMock(login="reviewer")
        mock_review_comment.created_at = datetime(2024, 3, 1, tzinfo=UTC)
        mock_review_comment.updated_at = datetime(2024, 3, 1, tzinfo=UTC)

        mock_pr = MagicMock()
        mock_pr.create_review_comment.return_value = mock_review_comment

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.add_pull_request_review_comment(
                "owner/repo", 1, "Fix this", "main.py", 42
            )
            assert result["id"] == 200
            assert result["filepath"] == "main.py"
            assert result["line"] == 42


# ===========================================================================
# Lines 373-387: read_file
# ===========================================================================


class TestReadFile:
    async def test_read_file_base64(self):
        """Lines 373-380: read file with base64 encoding."""
        mock_content = MagicMock()
        mock_content.encoding = "base64"
        mock_content.content = base64.b64encode(b"file content").decode()

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.read_file("owner/repo", "main.py")
            assert result == "file content"

    async def test_read_file_non_base64(self):
        """Lines 381-383: read file with non-base64 encoding."""
        mock_content = MagicMock()
        mock_content.encoding = "utf-8"
        mock_content.decoded_content = b"raw content"

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.read_file("owner/repo", "main.py")
            assert result == "raw content"

    async def test_read_file_directory_raises(self):
        """Lines 376-377: directory path raises ValueError."""

        class _GHExc(Exception):  # noqa: N818
            def __init__(self, status):
                self.status = status
                super().__init__(f"status {status}")

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [MagicMock()]
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
            patch("services.github.github_service.GithubException", _GHExc),
        ):
            with pytest.raises(ValueError, match="is a directory"):
                await svc.read_file("owner/repo", "src/")

    async def test_read_file_404_raises(self):
        """Lines 384-387: GithubException(404) raises FileNotFoundError."""

        class _GHExc(Exception):  # noqa: N818
            def __init__(self, status):
                self.status = status
                super().__init__(f"status {status}")

        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = _GHExc(404)
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
            patch("services.github.github_service.GithubException", _GHExc),
        ):
            with pytest.raises(FileNotFoundError, match="not found"):
                await svc.read_file("owner/repo", "missing.py")


# ===========================================================================
# Lines 393-414: list_files
# ===========================================================================


class TestListFiles:
    async def test_list_files_flat(self):
        """Lines 393-414: list files without recursion."""
        file1 = MagicMock(path="main.py", name="main.py", type="file", size=100, sha="aaa")
        file2 = MagicMock(path="utils/", name="utils", type="dir", size=0, sha="bbb")

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [file1, file2]
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            files = await svc.list_files("owner/repo")
            assert len(files) == 2
            assert files[0]["path"] == "main.py"
            assert files[1]["type"] == "dir"

    async def test_list_files_recursive(self):
        """Lines 409-411: recursive listing."""
        file1 = MagicMock(path="main.py", name="main.py", type="file", size=100, sha="a")
        subdir = MagicMock(path="utils", name="utils", type="dir", size=0, sha="b")
        subfile = MagicMock(path="utils/helper.py", name="helper.py", type="file", size=50, sha="c")

        mock_repo = MagicMock()
        # First call returns top-level, second call for "utils" subdir
        mock_repo.get_contents.side_effect = [[file1, subdir], [subfile]]
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            files = await svc.list_files("owner/repo", recursive=True)
            assert len(files) == 3
            paths = [f["path"] for f in files]
            assert "utils/helper.py" in paths

    async def test_list_files_single_item_not_list(self):
        """Lines 397-398: single content item (not a list) is wrapped."""
        single_file = MagicMock()
        single_file.path = "readme.md"
        single_file.name = "readme.md"
        single_file.type = "file"
        single_file.size = 10
        single_file.sha = "x"

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = single_file  # not a list
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            files = await svc.list_files("owner/repo", path="readme.md")
            assert len(files) == 1
            assert files[0]["name"] == "readme.md"


# ===========================================================================
# Lines 453-462: update_file
# ===========================================================================


class TestUpdateFile:
    async def test_update_file(self):
        """Lines 453-470: update file returns commit info."""
        mock_commit = MagicMock(sha="def456", message="update file", author=MagicMock(name="bot"))
        mock_content = MagicMock(html_url="https://github.com/updated")
        mock_result = {"commit": mock_commit, "content": mock_content}

        mock_repo = MagicMock()
        mock_repo.update_file.return_value = mock_result
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.update_file(
                "owner/repo", "main.py", "new content", "update file", sha="oldsha"
            )
            assert result["sha"] == "def456"
            assert result["commit"]["message"] == "update file"


# ===========================================================================
# Lines 476-483: create_branch
# ===========================================================================


class TestCreateBranch:
    async def test_create_branch(self):
        """Lines 476-483: create branch from source."""
        mock_source_ref = MagicMock()
        mock_source_ref.object = MagicMock(sha="abc123")
        mock_new_ref = MagicMock()
        mock_new_ref.ref = "refs/heads/feature"
        mock_new_ref.object = MagicMock(sha="abc123")
        mock_new_ref.url = "https://api.github.com/refs/heads/feature"

        mock_repo = MagicMock()
        mock_repo.get_git_ref.return_value = mock_source_ref
        mock_repo.create_git_ref.return_value = mock_new_ref
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.create_branch("owner/repo", "feature")
            assert result["sha"] == "abc123"


# ===========================================================================
# Lines 487-495: delete_branch
# ===========================================================================


class TestDeleteBranch:
    async def test_delete_branch_success(self):
        """Lines 487-491: successful branch deletion."""
        mock_ref = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_git_ref.return_value = mock_ref
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.delete_branch("owner/repo", "old-branch")
            assert result is True
            mock_ref.delete.assert_called_once()

    async def test_delete_branch_not_found(self):
        """Lines 492-494: branch not found returns False."""

        class _GHExc(Exception):  # noqa: N818
            def __init__(self, status):
                self.status = status
                super().__init__(f"status {status}")

        class _NeverMatchExc(Exception):  # noqa: N818
            pass

        mock_repo = MagicMock()
        mock_repo.get_git_ref.side_effect = _GHExc(404)
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
            patch("services.github.github_service.GithubException", _GHExc),
            patch("services.github.github_service.RateLimitExceededException", _NeverMatchExc),
        ):
            result = await svc.delete_branch("owner/repo", "nonexistent")
            assert result is False


# ===========================================================================
# Lines 501-512: create_webhook
# ===========================================================================


class TestCreateWebhook:
    async def test_create_webhook_with_secret(self):
        """Lines 501-517: create webhook with secret."""
        mock_hook = MagicMock()
        mock_hook.id = 55
        mock_hook.config = {"url": "https://example.com/hook"}
        mock_hook.events = ["push", "pull_request"]
        mock_hook.active = True

        mock_repo = MagicMock()
        mock_repo.create_hook.return_value = mock_hook
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.create_webhook(
                "owner/repo",
                "https://example.com/hook",
                ["push", "pull_request"],
                secret="my-secret",
            )
            assert result["id"] == 55
            assert result["active"] is True

    async def test_create_webhook_without_secret(self):
        """Lines 502-503: config without secret."""
        mock_hook = MagicMock()
        mock_hook.id = 56
        mock_hook.config = {"url": "https://example.com/hook"}
        mock_hook.events = ["push"]
        mock_hook.active = True

        mock_repo = MagicMock()
        mock_repo.create_hook.return_value = mock_hook
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.create_webhook("owner/repo", "https://example.com/hook", ["push"])
            assert result["id"] == 56


# ===========================================================================
# Lines 521-527: verify_webhook_signature
# ===========================================================================


class TestVerifyWebhookSignature:
    async def test_verify_webhook_signature_valid(self):
        """Lines 521-527: valid signature returns True."""
        import hashlib
        import hmac

        svc = _make_service()
        secret = "test-secret"
        payload = b'{"action": "opened"}'
        expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        result = await svc.verify_webhook_signature(payload, expected, secret)
        assert result is True

    async def test_verify_webhook_signature_invalid(self):
        """Lines 521-527: invalid signature returns False."""
        svc = _make_service()
        result = await svc.verify_webhook_signature(b"payload", "sha256=bad", "secret")
        assert result is False


# ===========================================================================
# Lines 533-538: create_issue
# ===========================================================================


class TestCreateIssue:
    async def test_create_issue_with_labels(self):
        """Lines 533-544: create issue with labels."""
        mock_issue = MagicMock()
        mock_issue.number = 10
        mock_issue.title = "Bug"
        mock_issue.state = "open"
        mock_issue.html_url = "https://github.com/issues/10"
        mock_issue.created_at = datetime(2024, 5, 1, tzinfo=UTC)

        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = mock_issue
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.create_issue("owner/repo", "Bug", "description", labels=["bug"])
            assert result["number"] == 10
            assert result["title"] == "Bug"

    async def test_create_issue_without_labels(self):
        """Lines 535-536: no labels branch."""
        mock_issue = MagicMock()
        mock_issue.number = 11
        mock_issue.title = "Feature"
        mock_issue.state = "open"
        mock_issue.html_url = "https://github.com/issues/11"
        mock_issue.created_at = None

        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = mock_issue
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            result = await svc.create_issue("owner/repo", "Feature", "desc")
            assert result["number"] == 11
            assert result["created_at"] is None


# ===========================================================================
# Lines 548-562: get_commit_files
# ===========================================================================


class TestGetCommitFiles:
    async def test_get_commit_files(self):
        """Lines 548-562: list files in a commit."""
        file1 = MagicMock()
        file1.filename = "src/main.py"
        file1.status = "modified"
        file1.additions = 5
        file1.deletions = 2
        file1.changes = 7
        file1.patch = "@@ -1,2 +1,5 @@"

        file2 = MagicMock(spec=[])  # no .patch attribute
        file2.filename = "binary.bin"
        file2.status = "added"
        file2.additions = 0
        file2.deletions = 0
        file2.changes = 0

        mock_commit = MagicMock()
        mock_commit.files = [file1, file2]

        mock_repo = MagicMock()
        mock_repo.get_commit.return_value = mock_commit
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            files = await svc.get_commit_files("owner/repo", "sha123")
            assert len(files) == 2
            assert files[0]["filename"] == "src/main.py"
            assert files[0]["patch"] == "@@ -1,2 +1,5 @@"
            assert files[1]["patch"] is None  # getattr fallback


# ===========================================================================
# Lines 568-594: get_pull_requests
# ===========================================================================


class TestGetPullRequests:
    async def test_get_pull_requests_no_limit(self):
        """Lines 568-594: list PRs without limit."""
        pr1 = MagicMock()
        pr1.number = 1
        pr1.title = "PR 1"
        pr1.state = "open"
        pr1.html_url = "https://github.com/pr/1"
        pr1.user = MagicMock(login="author1")
        pr1.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        pr1.updated_at = datetime(2024, 1, 2, tzinfo=UTC)
        pr1.head = MagicMock(ref="feature-1")
        pr1.base = MagicMock(ref="main")

        mock_prs = [pr1]

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = mock_prs
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            results = await svc.get_pull_requests("owner/repo")
            assert len(results) == 1
            assert results[0]["number"] == 1
            assert results[0]["head_branch"] == "feature-1"

    async def test_get_pull_requests_with_limit(self):
        """Lines 578-579: limit applied."""
        prs = []
        for i in range(5):
            pr = MagicMock()
            pr.number = i + 1
            pr.title = f"PR {i + 1}"
            pr.state = "open"
            pr.html_url = f"https://github.com/pr/{i + 1}"
            pr.user = MagicMock(login="author")
            pr.created_at = datetime(2024, 1, 1, tzinfo=UTC)
            pr.updated_at = datetime(2024, 1, 2, tzinfo=UTC)
            pr.head = MagicMock(ref=f"feature-{i + 1}")
            pr.base = MagicMock(ref="main")
            prs.append(pr)

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = prs
        mock_gh = _mock_github(repo=mock_repo)
        svc = _make_service(github=mock_gh)

        with (
            patch("services.github.github_service.Repository", type(mock_repo)),
            patch("github.Repository.Repository", type(mock_repo)),
        ):
            results = await svc.get_pull_requests("owner/repo", limit=2)
            assert len(results) == 2


# ===========================================================================
# Line 600: create_pr_comment (delegates to add_pull_request_comment)
# ===========================================================================


class TestCreatePRComment:
    async def test_create_pr_comment_delegates(self):
        """Line 600: create_pr_comment delegates to add_pull_request_comment."""
        svc = _make_service()
        svc.add_pull_request_comment = AsyncMock(return_value={"id": 1, "body": "hi"})
        result = await svc.create_pr_comment("owner/repo", 1, "hi")
        assert result["id"] == 1
        svc.add_pull_request_comment.assert_awaited_once_with("owner/repo", 1, "hi")


# ===========================================================================
# Lines 616-618, 620: get_github_service factory — schemathesis & error paths
# ===========================================================================


class TestGetGithubServiceFactory:
    async def test_get_github_service_schemathesis_token(self):
        """Schemathesis testing (non-production) substitutes a fake token."""
        mock_settings = MagicMock()
        # The environment resolves through Settings — a plain MagicMock attribute
        # would be truthy and wrongly assert production here.
        mock_settings.is_production = False

        with (
            patch("services.github.github_service.get_settings", return_value=mock_settings),
            patch.dict("os.environ", {"SCHEMATHESIS_TESTING": "true"}),
            patch(
                "services.github.github_service.get_secure_token_manager", return_value=AsyncMock()
            ),
        ):
            svc = await get_github_service("test-owner/coding-github")
            assert isinstance(svc, GitHubService)
            assert svc.config.token == "schemathesis_test_token"

    async def test_get_github_service_schemathesis_refused_in_production(self):
        """The schemathesis bypass never applies in production."""
        mock_settings = MagicMock()
        mock_settings.is_production = True

        with (
            patch("services.github.github_service.get_settings", return_value=mock_settings),
            patch.dict("os.environ", {"SCHEMATHESIS_TESTING": "true"}),
        ):
            with pytest.raises(ValueError, match="no PAT fallback"):
                await get_github_service("test-owner/coding-github")

    async def test_get_github_service_no_token_raises(self):
        """No credential exists on this tenantless path: always ValueError."""
        mock_settings = MagicMock()
        mock_settings.is_production = True

        with (
            patch("services.github.github_service.get_settings", return_value=mock_settings),
            patch.dict("os.environ", {"SCHEMATHESIS_TESTING": ""}, clear=False),
        ):
            with pytest.raises(ValueError, match="no PAT fallback"):
                await get_github_service("test-owner/coding-github")
