"""Extended tests for GitHubIntegration — covering all previously-uncovered lines.

Lines targeted: 23-28, 159, 181, 213, 216-226, 232, 241-254, 278, 286,
296-301, 330, 342-346, 351-355, 370-373, 406-411, 423, 429, 453, 455, 457,
463, 465, 472, 487-498, 503-505, 510, 512, 523-524, 545-558, 575, 582, 693
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from services.github_integration import GitHubIntegration


# ---------------------------------------------------------------------------
# Mock GithubException reusable across all test classes
# ---------------------------------------------------------------------------
class MockGithubError(Exception):
    """Mock GitHub exception for testing."""

    def __init__(
        self,
        status: int = 500,
        message: str = "GitHub error",
        data: dict | None = None,
    ):
        self.status = status
        self.data = data or {"message": message}
        super().__init__(message)


# Shared decorator stack used on almost every test
_COMMON_PATCHES = [
    patch("services.github_integration.FixValidator"),
    patch("services.github_integration.Github"),
    patch("services.github_integration.Auth"),
    patch("services.github_integration.HAS_GITHUB", True),
]


def _make_integration(mock_auth, mock_github, token="test-token"):
    """Helper to build a GitHubIntegration with mocked client."""
    mock_auth.Token.return_value = MagicMock()
    mock_client = MagicMock()
    mock_github.return_value = mock_client
    integration = GitHubIntegration(github_token=token)
    return integration, mock_client


# ============================================================================
# Lines 23-28: HAS_GITHUB = False import fallback
# ============================================================================
@pytest.mark.unit
class TestImportFallback:
    """Cover the HAS_GITHUB = False branch (lines 23-28)."""

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github", None)
    @patch("services.github_integration.Auth", None)
    @patch("services.github_integration.HAS_GITHUB", False)
    def test_init_no_github_library_with_token(self, _fv):
        """When HAS_GITHUB is False the client stays None even with a token."""
        integration = GitHubIntegration(github_token="tok")
        assert integration.github_client is None

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github", None)
    @patch("services.github_integration.Auth", None)
    @patch("services.github_integration.HAS_GITHUB", False)
    def test_init_no_github_library_without_token(self, _fv):
        with patch.dict(os.environ, {}, clear=True):
            integration = GitHubIntegration(github_token="")
            assert integration.github_client is None


# ============================================================================
# Line 159: user login empty / missing
# ============================================================================
@pytest.mark.unit
class TestEnsureForkUserLogin:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_user_no_login(self, mock_auth, mock_github, _fv):
        """Line 159 — user.login is empty string."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = ""
        mock_client.get_user.return_value = user
        with pytest.raises(ValueError, match="Could not determine GitHub user login"):
            integration._ensure_fork("owner", "repo")


# ============================================================================
# Line 181: _ensure_fork owner repo GithubException non-404/403
# ============================================================================
@pytest.mark.unit
class TestEnsureForkOwnerRepoOtherError:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_owner_repo_500(self, mock_auth, mock_github, _fv):
        """Line 181 — GithubException with status 500 re-raised."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "owner"
        mock_client.get_user.return_value = user
        mock_client.get_repo.side_effect = MockGithubError(status=500, message="Server error")
        with pytest.raises(MockGithubError):
            integration._ensure_fork("owner", "repo")


# ============================================================================
# Line 213: warning on non-404 error when checking for existing fork
# ============================================================================
@pytest.mark.unit
class TestEnsureForkCheckExistingNon404:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_existing_check_non404(self, mock_auth, mock_github, _fv):
        """Line 213 — non-404 error when checking existing fork, continues to create fork."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        fork_repo = MagicMock()
        fork_repo.private = True  # write targets must be private (#118)
        fork_repo.full_name = "forkuser/repo"
        user.create_fork = MagicMock(return_value=fork_repo)
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                # checking forkuser/repo -> 500 error
                raise MockGithubError(status=500, message="Server error")
            # second call: get original repo
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side

        result = integration._ensure_fork("owner", "repo")
        assert result.is_fork is True


# ============================================================================
# Lines 216-226: original repo get failures (404 / 403 / other)
# ============================================================================
@pytest.mark.unit
class TestEnsureForkOriginalRepoErrors:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_original_repo_404(self, mock_auth, mock_github, _fv):
        """Lines 217-220 — original repo not found."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            # Second call (original repo) also 404
            raise MockGithubError(status=404, message="Not found")

        mock_client.get_repo.side_effect = get_repo_side

        with pytest.raises(ValueError, match="not found"):
            integration._ensure_fork("owner", "repo")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_original_repo_403(self, mock_auth, mock_github, _fv):
        """Lines 221-224 — original repo access denied."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            raise MockGithubError(status=403, message="Forbidden")

        mock_client.get_repo.side_effect = get_repo_side

        with pytest.raises(ValueError, match="Access denied"):
            integration._ensure_fork("owner", "repo")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_original_repo_500(self, mock_auth, mock_github, _fv):
        """Lines 225-226 — original repo other error re-raised."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            raise MockGithubError(status=500, message="Server error")

        mock_client.get_repo.side_effect = get_repo_side

        with pytest.raises(MockGithubError):
            integration._ensure_fork("owner", "repo")


# ============================================================================
# Line 232: user.create_fork is None (no attribute)
# ============================================================================
@pytest.mark.unit
class TestEnsureForkNoCreateFork:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_user_no_create_fork_attr(self, mock_auth, mock_github, _fv):
        """Line 232 — user has no create_fork, raises GithubException(403)."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock(spec=[])  # no attributes at all
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side

        # create_fork is None via getattr, so GithubException(403) is raised,
        # then caught by the fork_error handler -> _create_user_repository_from_original
        user.create_repo = MagicMock(return_value=MagicMock(full_name="forkuser/repo"))
        object.__setattr__(
            integration,
            "_create_user_repository_from_original",
            MagicMock(return_value=MagicMock()),
        )

        result = integration._ensure_fork("owner", "repo")
        assert result.is_clone is True


# ============================================================================
# Lines 241-254: fork creation failures (403, 422, other) -> clone fallback
# ============================================================================
@pytest.mark.unit
class TestEnsureForkForkCreationFails:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_fork_fails_403(self, mock_auth, mock_github, _fv):
        """Lines 241-254 — fork fails with 403, falls back to clone."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        user.create_fork = MagicMock(
            side_effect=MockGithubError(status=403, message="Forking disabled")
        )
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side
        cloned_repo = MagicMock()
        object.__setattr__(
            integration,
            "_create_user_repository_from_original",
            MagicMock(return_value=cloned_repo),
        )

        result = integration._ensure_fork("owner", "repo")
        assert result.is_fork is False
        assert result.is_clone is True

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_fork_fails_422(self, mock_auth, mock_github, _fv):
        """Lines 248-249 — fork fails with 422."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        user.create_fork = MagicMock(
            side_effect=MockGithubError(status=422, message="Unprocessable")
        )
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side
        object.__setattr__(
            integration,
            "_create_user_repository_from_original",
            MagicMock(return_value=MagicMock()),
        )

        result = integration._ensure_fork("owner", "repo")
        assert result.is_clone is True

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_fork_fails_other_status(self, mock_auth, mock_github, _fv):
        """Lines 250-251 — fork fails with unexpected status."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        user.create_fork = MagicMock(
            side_effect=MockGithubError(status=500, message="Server error")
        )
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side
        object.__setattr__(
            integration,
            "_create_user_repository_from_original",
            MagicMock(return_value=MagicMock()),
        )

        result = integration._ensure_fork("owner", "repo")
        assert result.is_clone is True


# ============================================================================
# Lines 278, 286: _create_user_repository_from_original edge cases
# ============================================================================
@pytest.mark.unit
class TestCreateUserRepoEdgeCases:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_github_client_none(self, mock_auth, mock_github, _fv):
        """github_client is None during repo name check → typed GitHubAuthError."""
        from services.github_integration import GitHubAuthError

        integration, _ = _make_integration(mock_auth, mock_github)
        integration.github_client = None
        user = MagicMock()
        user.login = "testuser"
        original = MagicMock()
        original.name = "repo"
        with pytest.raises(GitHubAuthError, match="No GitHub token configured"):
            integration._create_user_repository_from_original(original, user)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_name_check_non404_reraise(self, mock_auth, mock_github, _fv):
        """Line 286 — non-404 GithubException re-raised during name check."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "testuser"
        original = MagicMock()
        original.name = "repo"
        mock_client.get_repo.side_effect = MockGithubError(status=500, message="Server error")
        with pytest.raises(MockGithubError):
            integration._create_user_repository_from_original(original, user)


# ============================================================================
# Lines 296-301: _create_user_repository_from_original exception handlers
# ============================================================================
@pytest.mark.unit
class TestCreateUserRepoExceptions:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_github_exception_in_create_repo(self, mock_auth, mock_github, _fv):
        """Lines 296-298 — GithubException during user.create_repo."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "testuser"
        original = MagicMock()
        original.name = "repo"
        original.full_name = "owner/repo"
        original.private = False
        mock_client.get_repo.side_effect = MockGithubError(status=404, message="Not found")
        user.create_repo.side_effect = MockGithubError(status=403, message="Forbidden")
        with pytest.raises(MockGithubError):
            integration._create_user_repository_from_original(original, user)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_unexpected_exception_in_create_repo(self, mock_auth, mock_github, _fv):
        """Lines 299-301 — unexpected exception during user.create_repo."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "testuser"
        original = MagicMock()
        original.name = "repo"
        original.full_name = "owner/repo"
        original.private = False
        mock_client.get_repo.side_effect = MockGithubError(status=404, message="Not found")
        user.create_repo.side_effect = RuntimeError("Unexpected")
        with pytest.raises(RuntimeError, match="Unexpected"):
            integration._create_user_repository_from_original(original, user)


# ============================================================================
# Lines 330, 342-346, 351-355, 370-373: _copy_repository_content edge cases
# ============================================================================
@pytest.mark.unit
class TestCopyRepoContent:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_file_content_is_list_skipped(self, mock_auth, mock_github, _fv):
        """Line 330 — file_content is a list => skip."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = "main"
        source.full_name = "owner/source"
        source.html_url = "https://github.com/owner/source"

        # All essential files return a list (directory)
        source.get_contents.return_value = [MagicMock(), MagicMock()]

        target = MagicMock()
        target.name = "target"
        target.default_branch = "main"
        # README update: get_contents returns 404, so create_file is called
        target.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        integration._copy_repository_content(source, target)
        # create_file should only be called for the README fallback, not for copied files
        # (all source files returned lists and were skipped)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_copy_file_non404_github_error(self, mock_auth, mock_github, _fv):
        """Lines 342-343 — non-404 GithubException during copy (warning, continue)."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = "main"
        source.full_name = "owner/source"
        source.html_url = "https://github.com/owner/source"

        readme_content = MagicMock()
        readme_content.decoded_content = b"# README"

        def get_source_contents(f, ref=None):
            if f == "README.md":
                return readme_content
            raise MockGithubError(status=404, message="Not found")

        source.get_contents.side_effect = get_source_contents

        target = MagicMock()
        target.name = "target"
        target.default_branch = "main"
        # create_file fails with non-path-related error
        target.create_file.side_effect = MockGithubError(status=500, message="Server error")
        target.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        # Should not raise — just warns and continues
        integration._copy_repository_content(source, target)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_copy_file_unicode_decode_error(self, mock_auth, mock_github, _fv):
        """Lines 344-346 — UnicodeDecodeError during copy."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = "main"
        source.full_name = "owner/source"
        source.html_url = "https://github.com/owner/source"

        bad_file = MagicMock()
        bad_file.decoded_content = MagicMock()
        bad_file.decoded_content.decode = MagicMock(
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")
        )

        def get_source_contents(f, ref=None):
            if f == "README.md":
                return bad_file
            raise MockGithubError(status=404, message="Not found")

        source.get_contents.side_effect = get_source_contents

        target = MagicMock()
        target.name = "target"
        target.default_branch = "main"
        target.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        integration._copy_repository_content(source, target)
        # Should not raise

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_readme_update_list_return(self, mock_auth, mock_github, _fv):
        """Lines 351-355 — README get_contents returns a list."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = "main"
        source.full_name = "owner/source"
        source.html_url = "https://github.com/owner/source"
        source.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        target = MagicMock()
        target.name = "target"
        target.default_branch = "main"

        # README get_contents returns a list
        readme_item = MagicMock()
        readme_item.sha = "readme_sha"
        target.get_contents.return_value = [readme_item]

        integration._copy_repository_content(source, target)
        target.update_file.assert_called_once()

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_readme_update_exception(self, mock_auth, mock_github, _fv):
        """Lines 370-371 — exception during README update."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = "main"
        source.full_name = "owner/source"
        source.html_url = "https://github.com/owner/source"
        source.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        target = MagicMock()
        target.name = "target"
        target.default_branch = "main"
        # First get_contents succeeds (single file)
        readme = MagicMock()
        readme.sha = "sha123"
        target.get_contents.return_value = readme
        # update_file raises non-GithubException
        target.update_file.side_effect = RuntimeError("Disk full")

        # Should not raise — catches Exception and logs warning
        integration._copy_repository_content(source, target)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_copy_content_outer_exception(self, mock_auth, mock_github, _fv):
        """Lines 372-373 — outer exception in _copy_repository_content."""
        integration, _ = _make_integration(mock_auth, mock_github)
        source = MagicMock()
        source.default_branch = MagicMock(side_effect=RuntimeError("broken"))
        # Accessing default_branch via property raises

        # Actually we need to make it raise when first accessed:
        type(source).default_branch = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("oops"))
        )

        target = MagicMock()
        # Should not raise — catches outer Exception and logs warning
        integration._copy_repository_content(source, target)


# ============================================================================
# Lines 406-411: _create_branch default branch fallback also fails
# ============================================================================
@pytest.mark.unit
class TestCreateBranchFallbackFails:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_both_base_and_default_branch_not_found(self, mock_auth, mock_github, _fv):
        """Implicit default base (main) missing AND the default branch missing."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.default_branch = "develop"
        repo.get_git_ref.side_effect = MockGithubError(status=404, message="Not found")

        with pytest.raises(ValueError, match="Neither base branch 'main' nor default branch"):
            integration._create_branch(repo, "feature", "main")
        repo.create_git_ref.assert_not_called()


# ============================================================================
# Lines 423, 429: branch already exists with different SHA / re-raise non-422
# ============================================================================
@pytest.mark.unit
class TestCreateBranchExistsDiffSha:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_branch_exists_different_sha(self, mock_auth, mock_github, _fv):
        """Branch exists but its SHA differs from base — the stale reused
        per-drive refactor branch is FORCE-RESET back to base so the re-apply
        starts clean and can produce a real commit (instead of being served
        stale, which would make the next apply a no-op → 'No commits between')."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "base_sha"

        existing_ref = MagicMock()
        existing_ref.object.sha = "different_sha"

        repo.create_git_ref.side_effect = MockGithubError(status=422, message="Exists")

        # First get_git_ref call resolves the base ref; the second (after the
        # 422) resolves the existing branch ref (different sha).
        call_count = [0]

        def get_ref_ordered(ref):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_ref
            return existing_ref

        repo.get_git_ref.side_effect = get_ref_ordered

        result = integration._create_branch(repo, "feature-branch", "main")
        # The stale ref is returned, but only after being force-reset to base.
        assert result == existing_ref
        existing_ref.edit.assert_called_once_with(sha="base_sha", force=True)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_branch_exists_different_sha_preserved_when_no_reset(self, mock_auth, mock_github, _fv):
        """Branch exists ahead of base but reset_existing_to_base=False — the
        commits are PRESERVED (not force-reset). This is the pure PR-open step of
        the architecture-refactoring procedure: a SEPARATE prior apply step
        populated the branch, and resetting it here would wipe that real diff
        → 'No commits between' → the card strands in BLOCKED before QA."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "base_sha"
        existing_ref = MagicMock()
        existing_ref.object.sha = "applied_sha"  # ahead of base (apply step's commit)

        repo.create_git_ref.side_effect = MockGithubError(status=422, message="Exists")

        call_count = [0]

        def get_ref_ordered(ref):
            call_count[0] += 1
            return base_ref if call_count[0] == 1 else existing_ref

        repo.get_git_ref.side_effect = get_ref_ordered

        result = integration._create_branch(
            repo, "feature-branch", "main", reset_existing_to_base=False
        )
        # The applied branch is returned UNTOUCHED — no force-reset.
        assert result == existing_ref
        existing_ref.edit.assert_not_called()

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_branch_exists_same_sha_not_reset(self, mock_auth, mock_github, _fv):
        """A reused branch already at base is left untouched — no force-reset."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "base_sha"
        existing_ref = MagicMock()
        existing_ref.object.sha = "base_sha"  # already at base

        repo.create_git_ref.side_effect = MockGithubError(status=422, message="Exists")

        call_count = [0]

        def get_ref_ordered(ref):
            call_count[0] += 1
            return base_ref if call_count[0] == 1 else existing_ref

        repo.get_git_ref.side_effect = get_ref_ordered

        result = integration._create_branch(repo, "feature-branch", "main")
        assert result == existing_ref
        existing_ref.edit.assert_not_called()

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_create_branch_non422_error_reraise(self, mock_auth, mock_github, _fv):
        """Line 429 — non-422 GithubException during create_git_ref is re-raised."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "base_sha"
        repo.get_git_ref.return_value = base_ref
        repo.create_git_ref.side_effect = MockGithubError(status=500, message="Server error")

        with pytest.raises(MockGithubError):
            integration._create_branch(repo, "feature-branch", "main")


# ============================================================================
# Lines 453, 455, 457: _update_files renamed missing fields
# ============================================================================
@pytest.mark.unit
class TestUpdateFilesRenamedMissing:
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_renamed_missing_old_path(self, mock_auth, mock_github, _fv):
        """Line 453 — renamed without old_file_path."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"file_path": "new.py", "content": "c", "change_type": "renamed"}]
        with pytest.raises(ValueError, match="old_file_path required"):
            integration._update_files(MagicMock(), "branch", changes)

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_renamed_missing_file_path(self, mock_auth, mock_github, _fv):
        """Line 455 — renamed without file_path."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"old_file_path": "old.py", "content": "c", "change_type": "renamed"}]
        with pytest.raises(ValueError, match="file_path is required"):
            integration._update_files(MagicMock(), "branch", changes)

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_renamed_missing_content(self, mock_auth, mock_github, _fv):
        """Line 457 — renamed without content."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"file_path": "new.py", "old_file_path": "old.py", "change_type": "renamed"}]
        with pytest.raises(ValueError, match="content is required"):
            integration._update_files(MagicMock(), "branch", changes)


# ============================================================================
# Lines 463, 465: _update_files added/modified missing fields
# ============================================================================
@pytest.mark.unit
class TestUpdateFilesAddedModifiedMissing:
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_added_missing_file_path(self, mock_auth, mock_github, _fv):
        """Line 463 — added without file_path."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"content": "c", "change_type": "added"}]
        with pytest.raises(ValueError, match="file_path is required"):
            integration._update_files(MagicMock(), "branch", changes)

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_modified_missing_content(self, mock_auth, mock_github, _fv):
        """Line 465 — modified without content."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"file_path": "f.py", "change_type": "modified"}]
        with pytest.raises(ValueError, match="content is required"):
            integration._update_files(MagicMock(), "branch", changes)


# ============================================================================
# Line 472: _update_files unknown type missing both file_path and content
# ============================================================================
@pytest.mark.unit
class TestUpdateFilesUnknownTypeMissing:
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_unknown_type_missing_file_path_and_content(self, mock_auth, mock_github, _fv):
        """Line 472 — unknown change_type with missing file_path and content."""
        integration, _ = _make_integration(mock_auth, mock_github)
        changes = [{"change_type": "weird"}]
        with pytest.raises(ValueError, match="file_path and content are required"):
            integration._update_files(MagicMock(), "branch", changes)


# ============================================================================
# Lines 487-498: _create_or_update_file bytes content fallback encoding
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileBytes:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_bytes_utf8_decode_works(self, mock_auth, mock_github, _fv):
        """Lines 487-488 — bytes content that decodes as UTF-8."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.return_value = {"commit": MagicMock(sha="sha1")}

        result = integration._create_or_update_file(repo, "branch", "f.py", b"hello utf8")
        assert result == "sha1"

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_bytes_latin1_fallback(self, mock_auth, mock_github, _fv):
        """Lines 490-496 — bytes that fail UTF-8 but succeed with latin-1."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.return_value = {"commit": MagicMock(sha="sha1")}

        # \xe9 is valid latin-1 but invalid UTF-8 in isolation
        content = b"caf\xe9"
        result = integration._create_or_update_file(repo, "branch", "f.py", content)
        assert result == "sha1"

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_bytes_all_encodings_fail(self, mock_auth, mock_github, _fv):
        """Lines 495-498 — all fallback encodings fail -> ValueError."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        # Create a bytes subclass that always fails decode
        class BadBytes(bytes):
            def decode(self, encoding="utf-8", errors="strict"):
                raise UnicodeDecodeError(encoding, b"", 0, 1, "invalid")

        bad = BadBytes(b"\xff\xfe")
        with pytest.raises(ValueError, match="Cannot decode file content"):
            integration._create_or_update_file(repo, "branch", "f.py", bad)

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_bytes_empty_after_decode(self, mock_auth, mock_github, _fv):
        """Empty bytes content triggers warning but still proceeds."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.return_value = {"commit": MagicMock(sha="sha1")}

        result = integration._create_or_update_file(repo, "branch", "f.py", b"   ")
        assert result == "sha1"


# ============================================================================
# Lines 503-505: content with invalid UTF-8 characters (encode error)
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileEncodeError:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_content_encode_error_replaced(self, mock_auth, mock_github, _fv):
        """Lines 503-505 — content.encode('utf-8') raises, gets replaced."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.return_value = {"commit": MagicMock(sha="sha1")}

        # Create a string subclass that fails on encode but works otherwise
        class BadStr(str):
            _encode_called = False

            def encode(self, encoding="utf-8", errors="strict"):
                if not BadStr._encode_called and errors == "strict":
                    BadStr._encode_called = True
                    raise UnicodeEncodeError(encoding, "", 0, 1, "surrogates")
                return super().encode(encoding, errors)

            def strip(self):
                return "content"

        content = BadStr("content with \ud800 surrogates")
        result = integration._create_or_update_file(repo, "branch", "f.py", content)
        assert result == "sha1"


# ============================================================================
# Lines 510, 512: get_contents returns empty list / multiple files
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileListReturns:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_get_contents_empty_list(self, mock_auth, mock_github, _fv):
        """Line 510 — get_contents returns empty list -> raises GithubException(404)."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.return_value = []
        repo.create_file.return_value = {"commit": MagicMock(sha="sha1")}

        # Empty list -> GithubException(404) -> falls through to create_file
        result = integration._create_or_update_file(repo, "branch", "f.py", "content")
        assert result == "sha1"

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_get_contents_multiple_files(self, mock_auth, mock_github, _fv):
        """Line 512 — get_contents returns list with > 1 item -> ValueError."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.return_value = [MagicMock(), MagicMock()]

        with pytest.raises(ValueError, match="Multiple files found"):
            integration._create_or_update_file(repo, "branch", "f.py", "content")


# ============================================================================
# 409/422 optimistic-concurrency conflict on update_file (TOCTOU on a shared
# fork branch): a concurrent writer changed the file between our sha read and the
# PUT. The apply must be idempotent — a no-op when the file already holds our
# content, one retry against the fresh sha otherwise — never a 500 that strands
# the architecture_refactoring card in BLOCKED despite a valid PR.
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileConflict:
    @staticmethod
    def _file(content, sha):
        f = MagicMock()
        f.decoded_content.decode.return_value = content
        f.sha = sha
        return f

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_conflict_already_at_target_is_idempotent_noop(self, mock_auth, mock_github, _fv):
        """409 then the file already holds the desired content -> None (no-op), no raise."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        # initial read = old content (differs -> proceeds to update);
        # post-conflict refetch = the desired content (a sibling apply landed it).
        repo.get_contents.side_effect = [
            self._file("old body", "stale_sha"),
            self._file("new body", "fresh_sha"),
        ]
        repo.update_file.side_effect = MockGithubError(
            status=409, message="does not match stale_sha"
        )

        result = integration._create_or_update_file(repo, "branch", "f.py", "new body")
        assert result is None
        assert repo.get_contents.call_count == 2

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_conflict_retries_with_fresh_sha(self, mock_auth, mock_github, _fv):
        """409 then the file still differs -> retry update with the fresh sha -> commit sha."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = [
            self._file("old body", "stale_sha"),
            self._file("other body", "fresh_sha"),
        ]
        repo.update_file.side_effect = [
            MockGithubError(status=409, message="does not match stale_sha"),
            {"commit": MagicMock(sha="retry_sha")},
        ]

        result = integration._create_or_update_file(repo, "branch", "f.py", "new body")
        assert result == "retry_sha"
        # the retry used the refreshed sha, not the stale one
        _, kwargs = repo.update_file.call_args
        assert kwargs["sha"] == "fresh_sha"


# ============================================================================
# Lines 523-524: exception when comparing existing content
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileCompareError:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_compare_existing_content_fails(self, mock_auth, mock_github, _fv):
        """Lines 523-524 — exception when decoding existing content for comparison."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()

        existing_file = MagicMock()
        existing_file.sha = "old_sha"
        existing_file.decoded_content = MagicMock()
        existing_file.decoded_content.decode = MagicMock(
            side_effect=RuntimeError("Decode exploded")
        )
        repo.get_contents.return_value = existing_file
        repo.update_file.return_value = {"commit": MagicMock(sha="updated_sha")}

        result = integration._create_or_update_file(repo, "branch", "f.py", "new content")
        assert result == "updated_sha"
        repo.update_file.assert_called_once()


# ============================================================================
# Lines 545-558: create_file errors in _create_or_update_file
# ============================================================================
@pytest.mark.unit
class TestCreateOrUpdateFileCreateErrors:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_create_file_path_error(self, mock_auth, mock_github, _fv):
        """Lines 545-553 — create_file fails with path-related error."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.side_effect = MockGithubError(
            status=422, message="Invalid path in request"
        )

        with pytest.raises(ValueError, match="Invalid file path or directory conflict"):
            integration._create_or_update_file(repo, "branch", "bad/path", "content")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_create_file_non_path_error(self, mock_auth, mock_github, _fv):
        """Line 554 — create_file fails with non-path error re-raised."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        repo.create_file.side_effect = MockGithubError(status=500, message="Server error")

        with pytest.raises(MockGithubError):
            integration._create_or_update_file(repo, "branch", "f.py", "content")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_get_contents_non404_reraise(self, mock_auth, mock_github, _fv):
        """Line 555 — get_contents raises non-404 GithubException."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=500, message="Server error")

        with pytest.raises(MockGithubError):
            integration._create_or_update_file(repo, "branch", "f.py", "content")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_outer_exception_reraise(self, mock_auth, mock_github, _fv):
        """Lines 556-558 — outer exception catches and re-raises."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = RuntimeError("Unexpected")

        with pytest.raises(RuntimeError, match="Unexpected"):
            integration._create_or_update_file(repo, "branch", "f.py", "content")


# ============================================================================
# Lines 575, 582: _delete_file edge cases
# ============================================================================
@pytest.mark.unit
class TestDeleteFileEdgeCases:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_delete_result_no_sha_attr(self, mock_auth, mock_github, _fv):
        """Line 575 — delete result commit has no sha attr -> returns None."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        existing = MagicMock()
        existing.sha = "file_sha"
        repo.get_contents.return_value = existing

        # delete_file returns a dict where commit has no .sha
        commit_obj = MagicMock(spec=[])  # no sha attribute
        repo.delete_file.return_value = {"commit": commit_obj}

        result = integration._delete_file(repo, "branch", "f.py")
        assert result is None

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_delete_file_404_from_delete_call(self, mock_auth, mock_github, _fv):
        """Line 582 — get_contents succeeds but delete_file raises 404."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        existing = MagicMock()
        existing.sha = "file_sha"
        repo.get_contents.return_value = existing
        repo.delete_file.side_effect = MockGithubError(status=404, message="Not found")

        result = integration._delete_file(repo, "branch", "f.py")
        assert result is None

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_delete_no_commit_key(self, mock_auth, mock_github, _fv):
        """Line 575 — delete result has no 'commit' key -> returns None."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        existing = MagicMock()
        existing.sha = "file_sha"
        repo.get_contents.return_value = existing
        repo.delete_file.return_value = {}

        result = integration._delete_file(repo, "branch", "f.py")
        assert result is None


# ============================================================================
# Line 693 — generate_pr_description final line (return)
# Already covered in original tests but ensure edge case with no resolves
# ============================================================================
@pytest.mark.unit
class TestGeneratePRDescriptionExtended:
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_generate_pr_description_single_resolve_no_context(self, mock_auth, mock_github, _fv):
        integration, _ = _make_integration(mock_auth, mock_github)
        result = integration.generate_pr_description([{"file_path": "a.py", "fix_type": "fix"}])
        assert "a.py" in result
        assert "## Checklist" in result

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_generate_pr_description_missing_keys(self, mock_auth, mock_github, _fv):
        integration, _ = _make_integration(mock_auth, mock_github)
        result = integration.generate_pr_description([{}])
        assert "Unknown file" in result
        assert "Unknown fix" in result


# ============================================================================
# Lines 148-150: create_pull_request generic Exception handler
# ============================================================================
@pytest.mark.unit
class TestCreatePRGenericException:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_create_pr_generic_exception(self, mock_auth, mock_github, _fv):
        """Lines 148-150 — non-GithubException triggers generic handler."""
        integration, _ = _make_integration(mock_auth, mock_github)
        # Raise a RuntimeError which is NOT a MockGithubError
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(side_effect=RuntimeError("Something broke")),
        )
        result = integration.create_pull_request(
            repo_owner="o",
            repo_name="r",
            title="T",
            message="M",
            head_branch="b",
        )
        assert result["success"] is False
        assert "Something broke" in result["error"]


# ============================================================================
# Line 411: _create_branch base branch non-404 GithubException re-raise
# ============================================================================
@pytest.mark.unit
class TestCreateBranchBaseNon404:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_base_branch_500_reraise(self, mock_auth, mock_github, _fv):
        """Line 411 — non-404 GithubException from get_git_ref for base branch."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_git_ref.side_effect = MockGithubError(status=500, message="Server error")
        with pytest.raises(MockGithubError):
            integration._create_branch(repo, "feature", "main")


# ============================================================================
# Line 582: _delete_file non-404 GithubException re-raise
# ============================================================================
@pytest.mark.unit
class TestDeleteFileNon404:
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_delete_file_500_reraise(self, mock_auth, mock_github, _fv):
        """Line 582 — non-404 GithubException from get_contents in _delete_file."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        repo.get_contents.side_effect = MockGithubError(status=500, message="Server error")
        with pytest.raises(MockGithubError):
            integration._delete_file(repo, "branch", "f.py")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_delete_file_non404_from_delete_call(self, mock_auth, mock_github, _fv):
        """Line 582 — non-404 GithubException from delete_file call."""
        integration, _ = _make_integration(mock_auth, mock_github)
        repo = MagicMock()
        existing = MagicMock()
        existing.sha = "sha"
        repo.get_contents.return_value = existing
        repo.delete_file.side_effect = MockGithubError(status=403, message="Forbidden")
        with pytest.raises(MockGithubError):
            integration._delete_file(repo, "branch", "f.py")


# ============================================================================
# Line 693: create_upstream_pull_request with colon in head_branch
# ============================================================================
@pytest.mark.unit
class TestUpstreamPRColonBranch:
    @pytest.mark.asyncio
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("main.get_workspaces_client")
    @patch("services.github_token_manager.get_github_token_manager")
    async def test_upstream_pr_with_colon_branch(
        self,
        mock_token_mgr_fn,
        mock_get_containers,
        mock_auth,
        mock_github,
        _fv,
    ):
        """Line 693 — head_branch already contains colon."""
        from unittest.mock import AsyncMock

        integration, _ = _make_integration(mock_auth, mock_github)

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "u1",
                "user_token_id": "t1",
                "original_repo_owner": "owner",
                "original_repo_name": "repo",
                "system_fork_owner": "sys",
            }
        )
        mock_get_containers.return_value = mock_client

        mock_tm = MagicMock()
        mock_tm.get_user_token = AsyncMock(return_value="tok")
        mock_token_mgr_fn.return_value = mock_tm

        pr = MagicMock()
        pr.number = 99
        pr.html_url = "https://github.com/owner/repo/pull/99"
        mock_gh_instance = MagicMock()
        mock_gh_instance.get_repo.return_value = MagicMock(
            create_pull=MagicMock(return_value=pr), private=True
        )
        mock_github.return_value = mock_gh_instance

        result = await integration.create_upstream_pull_request(
            container_name="ctr",
            title="T",
            message="M",
            head_branch="sys:feature",  # has colon
        )
        assert result["success"] is True
        assert result["pr_number"] == 99


# ============================================================================
# Head-ref pre-check: missing head branch → honest 422 (not opaque GitHub 422)
# ============================================================================
@pytest.mark.unit
class TestUpstreamPRMissingHeadBranch:
    @pytest.mark.asyncio
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("main.get_workspaces_client")
    @patch("services.github_token_manager.get_github_token_manager")
    async def test_upstream_pr_head_branch_missing_on_remote(
        self,
        mock_token_mgr_fn,
        mock_get_containers,
        mock_auth,
        mock_github,
        _fv,
    ):
        """A missing head branch (push never landed) returns a clear 422, and
        create_pull is never called (no opaque GitHub 'head invalid')."""
        from unittest.mock import AsyncMock

        integration, _ = _make_integration(mock_auth, mock_github)

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "u1",
                "user_token_id": "t1",
                "original_repo_owner": "owner",
                "original_repo_name": "repo",
                "system_fork_owner": "sys",
            }
        )
        mock_get_containers.return_value = mock_client

        mock_tm = MagicMock()
        mock_tm.get_user_token = AsyncMock(return_value="tok")
        mock_token_mgr_fn.return_value = mock_tm

        # The head ref lookup 404s → branch is not on the remote.
        repo = MagicMock()
        repo.get_git_ref.side_effect = MockGithubError(status=404, message="Not Found")
        repo.create_pull = MagicMock()
        repo.private = True
        mock_gh_instance = MagicMock()
        mock_gh_instance.get_repo.return_value = repo
        mock_github.return_value = mock_gh_instance

        result = await integration.create_upstream_pull_request(
            container_name="ctr",
            title="T",
            message="M",
            head_branch="aa/chore-ctr",
        )

        assert result["success"] is False
        assert result["status_code"] == 422
        assert "not found" in result["error"].lower()
        assert "push step did not complete" in result["error"]
        repo.create_pull.assert_not_called()


# ============================================================================
# Lines 731-733: create_upstream_pull_request generic Exception handler
# ============================================================================
@pytest.mark.unit
class TestUpstreamPRGenericException:
    @pytest.mark.asyncio
    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("main.get_workspaces_client")
    async def test_upstream_pr_unexpected_error(
        self,
        mock_get_containers,
        mock_auth,
        mock_github,
        _fv,
    ):
        """Lines 731-733 — non-GithubException in create_upstream_pull_request."""
        from unittest.mock import AsyncMock

        integration, _ = _make_integration(mock_auth, mock_github)

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=RuntimeError("Boom"))
        mock_get_containers.return_value = mock_client

        result = await integration.create_upstream_pull_request(
            container_name="ctr",
            title="T",
            message="M",
            head_branch="branch",
        )
        assert result["success"] is False
        assert "Boom" in result["error"]


# ============================================================================
# resolve_github_token precedence + GitHub App same-repo short-circuit
# ============================================================================
from types import SimpleNamespace  # noqa: E402

from services.github_integration import resolve_github_token  # noqa: E402


def _settings_ns():
    """A settings stand-in with no PAT fields (none exist since 2026-09-30)."""
    return SimpleNamespace()


@pytest.mark.unit
class TestResolveGithubToken:
    """Precedence: user token > (owner required) > customer public App | platform App."""

    def test_user_token_wins(self):
        with patch("services.github_integration.settings", _settings_ns()):
            resolved = resolve_github_token("user-tok")
        assert resolved.token == "user-tok"
        assert resolved.is_app is False

    @pytest.mark.parametrize("owner", [None, ""])
    def test_ownerless_request_is_refused_with_every_rung_armed(self, owner):
        """⛔ The env / settings / pool / system / home-App rungs used to answer an
        OWNERLESS request with an org-wide credential, gated by nothing. They are
        gone (the PAT rungs and the pool deleted outright, 2026-09-30): every
        remaining source is armed here and none may answer."""
        from services.tenant_binding import OwnerRequiredError

        home = MagicMock(return_value="ghs_home_org_wide")
        with (
            patch.dict(
                os.environ,
                {
                    "GITHUB_ACCESS_TOKEN": "env-tok",
                    "GITHUB_TOKEN": "env-tok",
                    "CODING_GITHUB_GITHUB_TOKEN": "set-tok",
                    "CODING_GITHUB_SYSTEM_GITHUB_TOKEN": "sys-tok",
                },
                clear=True,
            ),
            patch("services.github_app_auth.app_auth_configured", return_value=True),
            patch("services.github_app_auth.get_installation_token", home),
            pytest.raises(OwnerRequiredError) as ei,
        ):
            resolve_github_token(owner=owner)
        assert (ei.value.code, ei.value.http_status) == ("github_owner_required", 400)
        home.assert_not_called()

    def test_user_token_outranks_a_configured_app_precedence_lock(self):
        """PRECEDENCE LOCK: an explicit user token beats the App, always.

        ``test_user_token_wins`` above only proves this when the App is absent.
        This pins the case that actually matters: the App IS configured and
        installed, and the user token STILL wins — the App is never even
        consulted. PR authorship depends on it (an App identity has no user
        account to attribute commits to), so if anyone ever "fixes" PR authorship
        by promoting the App above the caller's token, this test must fail loudly
        rather than let the change land silently.
        """
        app_configured = MagicMock(return_value=True)
        app_mint = MagicMock(return_value="ghs_app_should_never_be_used")
        app_mint_for = MagicMock(return_value="ghs_app_should_never_be_used")
        with (
            patch(
                "services.github_integration.settings",
                _settings_ns(),
            ),
            patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "env-tok"}, clear=True),
            patch("services.github_app_auth.app_auth_configured", app_configured),
            patch("services.github_app_auth.get_installation_token", app_mint),
            patch("services.github_app_auth.get_installation_token_for", app_mint_for),
        ):
            resolved = resolve_github_token("user-tok", owner="acme", repo="widgets")

        assert resolved.token == "user-tok"
        assert resolved.is_app is False  # must not take the App same-repo path
        app_mint.assert_not_called()
        app_mint_for.assert_not_called()


@pytest.mark.unit
class TestAppAuthSameRepoShortCircuit:
    """is_app_auth makes _ensure_fork operate on the target repo (no fork)."""

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_returns_target_repo(self, mock_auth, mock_github, _fv):
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        target_repo = MagicMock()
        target_repo.private = True  # write targets must be private (#118)
        target_repo.full_name = "nonagenticai/enterprise_mcp_server"
        mock_client.get_repo.return_value = target_repo
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="ghs_app", is_app_auth=True)
        info = integration._ensure_fork("nonagenticai", "enterprise_mcp_server")

        assert info.repository is target_repo
        assert info.is_fork is False
        assert info.is_clone is False
        # Same-repo path must not query the (nonexistent) App bot user.
        mock_client.get_user.assert_not_called()
        mock_client.get_repo.assert_called_once_with("nonagenticai/enterprise_mcp_server")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_404_raises_valueerror(self, mock_auth, mock_github, _fv):
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.get_repo.side_effect = MockGithubError(404, "Not Found")
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="ghs_app", is_app_auth=True)
        with pytest.raises(ValueError, match="not found"):
            integration._ensure_fork("nonagenticai", "missing")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_create_pull_request_uses_plain_branch_head(self, mock_auth, mock_github, _fv):
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        target_repo = MagicMock()
        target_repo.private = True  # write targets must be private (#118)
        target_repo.full_name = "nonagenticai/enterprise_mcp_server"
        # _create_branch path
        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        target_repo.get_git_ref.return_value = base_ref
        target_repo.create_git_ref.return_value = MagicMock()
        created_pr = MagicMock()
        created_pr.number = 7
        created_pr.html_url = "https://github.com/nonagenticai/enterprise_mcp_server/pull/7"
        target_repo.create_pull.return_value = created_pr
        mock_client.get_repo.return_value = target_repo
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="ghs_app", is_app_auth=True)
        result = integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="enterprise_mcp_server",
            title="Bump dep",
            message="body",
            head_branch="nonagenticai:chore/bump",  # owner prefix must be stripped
            base_branch="main",
        )

        assert result["success"] is True
        assert result["pr_number"] == 7
        # head must be the plain branch (no owner: prefix), and get_user never used.
        _, kwargs = target_repo.create_pull.call_args
        assert kwargs["head"] == "chore/bump"
        mock_client.get_user.assert_not_called()


@pytest.mark.unit
class TestCreatePullRequestIdempotent:
    """create_pull_request is idempotent: a re-run of the same upgrade returns
    the existing open PR instead of failing on GitHub's 'already exists' 422."""

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_returns_existing_pr_when_already_exists(self, mock_auth, mock_github, _fv):
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        target_repo = MagicMock()
        target_repo.private = True  # write targets must be private (#118)
        target_repo.full_name = "nonagenticai/enterprise_mcp_server"
        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        target_repo.get_git_ref.return_value = base_ref
        target_repo.create_git_ref.return_value = MagicMock()
        # First open of this head -> base already exists.
        # Real GitHub 422: generic top-level message, reason in errors[].
        target_repo.create_pull.side_effect = MockGithubError(
            422,
            "Validation Failed",
            data={
                "message": "Validation Failed",
                "errors": [
                    {"message": "A pull request already exists for nonagenticai:chore/bump."}
                ],
            },
        )
        existing_pr = MagicMock()
        existing_pr.number = 4
        existing_pr.html_url = "https://github.com/nonagenticai/enterprise_mcp_server/pull/4"
        target_repo.get_pulls.return_value = [existing_pr]
        mock_client.get_repo.return_value = target_repo
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="ghs_app", is_app_auth=True)
        result = integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="enterprise_mcp_server",
            title="Bump dep",
            message="body",
            head_branch="chore/bump",
            base_branch="main",
        )

        assert result["success"] is True
        assert result["already_existed"] is True
        assert result["pr_number"] == 4
        assert result["pr_url"].endswith("/pull/4")
        # Looked up the existing PR with the qualified head filter.
        _, kwargs = target_repo.get_pulls.call_args
        assert kwargs["head"] == "nonagenticai:chore/bump"
        assert kwargs["base"] == "main"
        assert kwargs["state"] == "open"

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_no_commits_between_is_benign(self, mock_auth, mock_github, _fv):
        """An empty-diff apply leaves nothing to open a PR for, so GitHub's
        'No commits between' 422 is benign — create_pull_request returns
        success with a no_commits flag (and no PR) instead of hard-failing the
        AA refactoring loop. Mirrors the 'already exists' idempotency block."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        target_repo = MagicMock()
        target_repo.private = True  # write targets must be private (#118)
        target_repo.full_name = "nonagenticai/enterprise_mcp_server"
        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        target_repo.get_git_ref.return_value = base_ref
        target_repo.create_git_ref.return_value = MagicMock()
        target_repo.create_pull.side_effect = MockGithubError(
            422,
            "Validation Failed",
            data={
                "message": "Validation Failed",
                "errors": [{"message": "No commits between main and chore/bump"}],
            },
        )
        mock_client.get_repo.return_value = target_repo
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="ghs_app", is_app_auth=True)
        result = integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="enterprise_mcp_server",
            title="Bump dep",
            message="body",
            head_branch="chore/bump",
            base_branch="main",
        )

        # Benign: success=True so the loop does not hard-fail, but a clear flag
        # tells the caller no PR was opened and there is no PR number/url.
        assert result["success"] is True
        assert result["no_commits"] is True
        assert result["pr_number"] is None
        assert result.get("pr_url") is None
        assert "no commits between" in result["message"].lower()
        # No existing-PR lookup happens for the no-commits case.
        target_repo.get_pulls.assert_not_called()


# ============================================================================
# Fork-readiness poll + not-ready fallback (replaces the old time.sleep(2))
# ============================================================================
@pytest.mark.unit
class TestForkReadinessPoll:
    """_wait_for_fork_ready and the copy-repo fallback when a fork never readies."""

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_wait_ready_true_when_branch_resolves(self, mock_auth, mock_github, _fv):
        integration, _ = _make_integration(mock_auth, mock_github)
        fork = MagicMock()
        fork.default_branch = "main"
        fork.get_branch = MagicMock(return_value=MagicMock())
        assert integration._wait_for_fork_ready(fork, timeout=5, interval=0.01) is True
        fork.get_branch.assert_called_once_with("main")

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_wait_ready_false_on_timeout(self, mock_auth, mock_github, _fv):
        integration, _ = _make_integration(mock_auth, mock_github)
        fork = MagicMock()
        fork.default_branch = "main"
        fork.full_name = "bot/repo"
        # Always 404 → never ready; timeout=0 so the deadline lapses on first miss.
        fork.get_branch = MagicMock(side_effect=MockGithubError(status=404, message="no ref"))
        assert integration._wait_for_fork_ready(fork, timeout=0.0, interval=0.01) is False

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_falls_back_to_copy_when_fork_never_ready(
        self, mock_auth, mock_github, _fv
    ):
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "forkuser"
        fork_repo = MagicMock()
        fork_repo.private = True  # write targets must be private (#118)
        fork_repo.full_name = "forkuser/repo"
        user.create_fork = MagicMock(return_value=fork_repo)
        mock_client.get_user.return_value = user

        original_repo = MagicMock()
        original_repo.html_url = "https://github.com/owner/repo"

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockGithubError(status=404, message="Not found")
            return original_repo

        mock_client.get_repo.side_effect = get_repo_side
        cloned_repo = MagicMock()
        object.__setattr__(
            integration,
            "_create_user_repository_from_original",
            MagicMock(return_value=cloned_repo),
        )
        # Force the fork to never become push-ready.
        object.__setattr__(integration, "_wait_for_fork_ready", MagicMock(return_value=False))

        result = integration._ensure_fork("owner", "repo")
        assert result.is_fork is False
        assert result.is_clone is True
        assert result.repository is cloned_repo
        integration._create_user_repository_from_original.assert_called_once()


# ============================================================================
# Direct same-repo PR (no fork) when the token has push access — covers org
# repos like nonagenticai/* driven by a member PAT (login != owner).
# ============================================================================
@pytest.mark.unit
class TestDirectPushNoFork:
    """_token_can_push gating + _ensure_fork direct path + same-repo head ref."""

    def test_token_can_push_strict_bool(self):
        repo = MagicMock()
        repo.permissions.push = True
        assert GitHubIntegration._token_can_push(repo) is True

    def test_token_can_push_false_for_truthy_nonbool(self):
        # A MagicMock attribute is truthy but is NOT the bool True — must not
        # be mistaken for push access.
        repo = MagicMock()
        assert GitHubIntegration._token_can_push(repo) is False
        repo2 = MagicMock()
        repo2.permissions.push = False
        assert GitHubIntegration._token_can_push(repo2) is False
        repo3 = MagicMock()
        repo3.permissions = None
        assert GitHubIntegration._token_can_push(repo3) is False

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_ensure_fork_direct_when_push_access_and_not_owner(self, mock_auth, mock_github, _fv):
        """Org repo: login != owner, but token can push → no fork, use repo directly."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "georgiedekker"  # member, not the org
        user.create_fork = MagicMock()
        mock_client.get_user.return_value = user

        target_repo = MagicMock()

        target_repo.private = True  # write targets must be private (#118)
        target_repo.full_name = "nonagenticai/enterprise_mcp_server"
        target_repo.permissions.push = True

        call_count = [0]

        def get_repo_side(name):
            call_count[0] += 1
            if call_count[0] == 1:
                # existing-fork lookup under user.login → none
                raise MockGithubError(status=404, message="Not found")
            return target_repo

        mock_client.get_repo.side_effect = get_repo_side

        result = integration._ensure_fork("nonagenticai", "enterprise_mcp_server")

        assert result.is_fork is False
        assert result.is_clone is False
        assert result.repository is target_repo
        user.create_fork.assert_not_called()  # the whole point: no fork

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_same_repo_uses_plain_branch(self, _fv, mock_github, mock_auth):
        """A same-repo PR (is_fork=False/is_clone=False) must use a plain branch head."""
        from services.github_integration import GitHubRepoInfo

        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        target_repo = MagicMock()

        target_repo.private = True  # write targets must be private (#118)
        target_repo.create_pull.return_value = MagicMock(
            number=11, html_url="https://github.com/nonagenticai/x/pull/11", draft=False
        )
        mock_client.get_repo.return_value = target_repo

        integration = GitHubIntegration(github_token="ghp_member")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=target_repo, is_fork=False, is_clone=False, original_url=None
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())
        object.__setattr__(integration, "_update_files", MagicMock())

        result = integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="x",
            title="Bump dep",
            message="body",
            head_branch="chore/bump",
            base_branch="main",
        )

        assert result["success"] is True
        _, kwargs = target_repo.create_pull.call_args
        assert kwargs["head"] == "chore/bump"  # no "georgiedekker:" prefix
        mock_client.get_user.assert_not_called()  # same-repo path never needs login

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_pure_open_preserves_applied_branch(self, _fv, mock_github, mock_auth):
        """The pure PR-open step (resolves=None) must NOT force-reset the head
        branch: a SEPARATE prior apply step committed the real diff there.
        create_pull_request threads reset_existing_to_base=False; with resolves
        provided it threads True (this call re-applies)."""
        from services.github_integration import GitHubRepoInfo

        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        target_repo = MagicMock()

        target_repo.private = True  # write targets must be private (#118)
        target_repo.create_pull.return_value = MagicMock(
            number=21, html_url="https://github.com/nonagenticai/x/pull/21", draft=True
        )
        mock_client.get_repo.return_value = target_repo

        integration = GitHubIntegration(github_token="ghp_member")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=target_repo, is_fork=False, is_clone=False, original_url=None
                )
            ),
        )
        cb = MagicMock()
        object.__setattr__(integration, "_create_branch", cb)
        object.__setattr__(integration, "_update_files", MagicMock())

        # Pure PR-open: no resolves -> reset_existing_to_base must be False.
        integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="x",
            title="Break cycle",
            message="body",
            head_branch="refactor/break-cycle-abc",
            base_branch="aa/fixture",
        )
        assert cb.call_args.kwargs.get("reset_existing_to_base") is False

        # Apply-in-PR: resolves present -> reset_existing_to_base must be True.
        cb.reset_mock()
        integration.create_pull_request(
            repo_owner="nonagenticai",
            repo_name="x",
            title="Bump dep",
            message="body",
            head_branch="chore/bump",
            base_branch="main",
            resolves=[{"file_path": "p.txt", "content": "x"}],
        )
        assert cb.call_args.kwargs.get("reset_existing_to_base") is True

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_fork_uses_login_prefixed_head(self, _fv, mock_github, mock_auth):
        """A fork PR keeps the cross-repo login:branch head."""
        from services.github_integration import GitHubRepoInfo

        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        user = MagicMock()
        user.login = "botuser"
        mock_client.get_user.return_value = user

        target_repo = MagicMock()

        target_repo.private = True  # write targets must be private (#118)
        target_repo.create_pull.return_value = MagicMock(
            number=12, html_url="https://github.com/someorg/y/pull/12", draft=False
        )
        mock_client.get_repo.return_value = target_repo

        integration = GitHubIntegration(github_token="ghp_bot")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=target_repo,
                    is_fork=True,
                    is_clone=False,
                    original_url="https://github.com/someorg/y",
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())
        object.__setattr__(integration, "_update_files", MagicMock())

        result = integration.create_pull_request(
            repo_owner="someorg",
            repo_name="y",
            title="Bump dep",
            message="body",
            head_branch="chore/bump",
            base_branch="main",
        )

        assert result["success"] is True
        _, kwargs = target_repo.create_pull.call_args
        assert kwargs["head"] == "botuser:chore/bump"


@pytest.mark.unit
class TestRedirectCanonicalDirect:
    """A moved/renamed target redirects the user-namespace lookup back to the
    canonical repo; with push access that must be a DIRECT same-repo PR, not a
    clone (else create_pull_request builds a wrong login:branch head)."""

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_redirect_to_canonical_with_push_is_direct(self, mock_auth, mock_github, _fv):
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "georgiedekker"
        user.create_fork = MagicMock()
        mock_client.get_user.return_value = user

        # get_repo("georgiedekker/enterprise_mcp_server") follows the 301 and
        # returns the canonical org repo.
        canonical = MagicMock()
        canonical.full_name = "nonagenticai/enterprise_mcp_server"
        canonical.fork = False
        canonical.parent = None
        canonical.permissions.push = True
        mock_client.get_repo.return_value = canonical

        result = integration._ensure_fork("nonagenticai", "enterprise_mcp_server")

        assert result.is_fork is False
        assert result.is_clone is False
        assert result.repository is canonical
        user.create_fork.assert_not_called()

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.HAS_GITHUB", True)
    def test_redirect_to_canonical_without_push_not_direct(self, mock_auth, mock_github, _fv):
        """full_name matches but no push → must NOT short-circuit to direct."""
        integration, mock_client = _make_integration(mock_auth, mock_github)
        user = MagicMock()
        user.login = "someuser"
        mock_client.get_user.return_value = user

        canonical = MagicMock()
        canonical.full_name = "org/repo"
        canonical.fork = False
        canonical.parent = None
        canonical.permissions.push = False  # read-only
        mock_client.get_repo.return_value = canonical

        result = integration._ensure_fork("org", "repo")
        # Falls through to the existing "user repository (not a fork)" clone path.
        assert (result.is_fork, result.is_clone) == (False, True)
