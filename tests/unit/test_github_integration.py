"""
Comprehensive unit tests for GitHubIntegration service module.

Tests cover:
- GitHub client initialization
- Pull request creation
- Fork management
- Branch creation
- File operations (create, update, delete, rename)
- Error handling for GitHub API errors
- PR description generation
- Upstream PR creation
- Fix validation
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Module imports with conditional handling
try:
    from services.github_integration import (
        BaseRefNotFoundError,
        GitHubIntegration,
        GitHubRepoInfo,
    )
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


# Create a mock GithubException for tests
class MockGithubError(Exception):
    """Mock GitHub exception for testing."""

    def __init__(self, status: int = 500, message: str = "GitHub error", data: dict | None = None):
        self.status = status
        self.data = data or {"message": message}
        super().__init__(message)


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = MagicMock()
    user = MagicMock()
    user.login = "testuser"
    user.create_fork = MagicMock()
    user.create_repo = MagicMock()
    client.get_user = MagicMock(return_value=user)
    client.get_repo = MagicMock()
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_repository():
    """Create a mock GitHub repository."""
    repo = MagicMock()
    repo.name = "test-repo"
    repo.full_name = "owner/test-repo"
    repo.html_url = "https://github.com/owner/test-repo"
    repo.default_branch = "main"
    repo.private = True  # write targets must be private (#118)
    repo.fork = False
    repo.parent = None
    repo.create_pull = MagicMock()
    repo.get_git_ref = MagicMock()
    repo.create_git_ref = MagicMock()
    repo.get_contents = MagicMock()
    repo.create_file = MagicMock()
    repo.update_file = MagicMock()
    repo.delete_file = MagicMock()
    return repo


@pytest.fixture
def mock_pull_request():
    """Create a mock pull request."""
    pr = MagicMock()
    pr.number = 42
    pr.html_url = "https://github.com/owner/test-repo/pull/42"
    pr.title = "Test PR"
    return pr


@pytest.fixture
def mock_fix_validator():
    """Create a mock fix validator."""
    validator = MagicMock()
    validator.validate_fix = MagicMock(
        return_value={
            "valid": True,
            "message": "Validation passed",
            "suggestions": [],
        }
    )
    return validator


@pytest.fixture
def sample_resolves():
    """Create sample resolves for testing."""
    return [
        {
            "file_path": "src/main.py",
            "content": "def fixed_function():\n    pass",
            "change_type": "modified",
            "fix_type": "Bug fix",
        },
        {
            "file_path": "src/utils.py",
            "content": "def helper():\n    return True",
            "change_type": "added",
            "fix_type": "New feature",
        },
    ]


# ============================================================================
# GitHubIntegration Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestGitHubIntegrationInit:
    """Tests for GitHubIntegration initialization."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_init_with_token(self, mock_validator, mock_github, mock_auth):
        """Test initialization with provided token."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")

        mock_auth.Token.assert_called_once_with("test-token")
        mock_github.assert_called_once()
        assert integration.github_client is mock_github.return_value

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "env-token"})
    def test_init_with_no_owner_never_uses_the_env_token(
        self, mock_validator, mock_github, mock_auth
    ):
        """⛔ No token and no target owner is refused: it used to fall through to
        the env PAT / home installation token with no tenant or repo gate."""
        from services.tenant_binding import OwnerRequiredError

        with pytest.raises(OwnerRequiredError):
            GitHubIntegration()

        mock_auth.Token.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_init_with_codebase_path(self, mock_validator, mock_github, mock_auth):
        """Test initialization with custom codebase path."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token", codebase_path="/custom/path")

        assert integration.codebase_path == "/custom/path"
        # FixValidator is now a TODO stub (inter-service call to coding-analysis)

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_init_github_client_exception(self, mock_validator, mock_github, mock_auth):
        """Test initialization handles GitHub client exception."""
        mock_auth.Token.side_effect = Exception("Auth failed")

        integration = GitHubIntegration(github_token="bad-token")

        assert integration.github_client is None

    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.Github", None)
    @patch("services.github_integration.Auth", None)
    @patch("services.github_integration.HAS_GITHUB", False)
    def test_init_without_github_library(self, mock_validator):
        """Test initialization when GitHub library is not available.

        When HAS_GITHUB is False, Auth and Github are set to None at module level,
        so trying to use them raises an error and github_client stays None.
        """
        integration = GitHubIntegration(github_token="test-token")

        # Since Auth is None, trying to call Auth.Token will fail
        # and github_client will remain None
        assert integration.github_client is None

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.FixValidator")
    def test_init_without_token(self, mock_validator):
        """No caller token and no owner: construction is refused (no ladder)."""
        from types import SimpleNamespace

        from services.tenant_binding import OwnerRequiredError

        empty_settings = SimpleNamespace()  # no PAT fields exist (2026-09-30)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("services.github_integration.settings", empty_settings),
            patch("services.github_app_auth.app_auth_configured", return_value=False),
            pytest.raises(OwnerRequiredError),
        ):
            GitHubIntegration(github_token=None)


# ============================================================================
# Pull Request Creation Tests
# ============================================================================


@pytest.mark.unit
class TestCreatePullRequest:
    """Tests for pull request creation."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_success(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
    ):
        """Test successful PR creation."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        user.create_fork = MagicMock(return_value=mock_repository)
        mock_client.get_user.return_value = user

        mock_repository.create_pull.return_value = mock_pull_request
        mock_repository.get_git_ref.return_value = MagicMock(object=MagicMock(sha="abc123"))
        mock_repository.create_git_ref.return_value = MagicMock()
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=mock_repository,
                    is_fork=True,
                    is_clone=False,
                    original_url="https://github.com/owner/test-repo",
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())
        object.__setattr__(integration, "_update_files", MagicMock())

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
            base_branch="main",
        )

        assert result["success"] is True
        assert result["pr_number"] == 42

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_draft(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
    ):
        """draft=True is threaded into create_pull and reflected in the result."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        mock_client.get_user.return_value = user
        mock_pull_request.draft = True
        mock_repository.create_pull.return_value = mock_pull_request
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=mock_repository,
                    is_fork=True,
                    is_clone=False,
                    original_url=None,
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
            draft=True,
        )

        assert result["success"] is True
        assert result["draft"] is True
        assert mock_repository.create_pull.call_args.kwargs["draft"] is True

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_mark_pull_request_ready_flips_draft(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A draft PR is flipped to ready via the GraphQL mutation."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        pr = MagicMock()
        pr.number = 42
        pr.draft = True
        pr.node_id = "PR_node_abc"
        pr.html_url = "https://github.com/owner/test-repo/pull/42"
        pr.state = "open"
        mock_repository.get_pull.return_value = pr
        mock_client.get_repo.return_value = mock_repository
        mock_client.requester.graphql_query.return_value = (
            {},
            {
                "data": {
                    "markPullRequestReadyForReview": {
                        "pullRequest": {
                            "number": 42,
                            "isDraft": False,
                            "url": "https://github.com/owner/test-repo/pull/42",
                            "state": "open",
                        }
                    }
                }
            },
        )

        integration = GitHubIntegration(github_token="test-token")
        result = integration.mark_pull_request_ready("owner", "test-repo", 42)

        assert result["success"] is True
        assert result["draft"] is False
        assert result["already_ready"] is False
        assert mock_client.requester.graphql_query.call_args.args[1] == {"id": "PR_node_abc"}

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_mark_pull_request_ready_idempotent(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A PR that is already ready returns already_ready without a mutation."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        pr = MagicMock()
        pr.number = 42
        pr.draft = False
        pr.node_id = "PR_node_abc"
        pr.html_url = "https://github.com/owner/test-repo/pull/42"
        pr.state = "open"
        mock_repository.get_pull.return_value = pr
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration.mark_pull_request_ready("owner", "test-repo", 42)

        assert result["success"] is True
        assert result["already_ready"] is True
        assert result["draft"] is False
        mock_client.requester.graphql_query.assert_not_called()

    @staticmethod
    def _open_pr(body: str | None = "old body", state: str = "open") -> MagicMock:
        pr = MagicMock()
        pr.number = 42
        pr.body = body
        pr.html_url = "https://github.com/owner/test-repo/pull/42"
        pr.state = state
        return pr

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_pull_request_body_edits(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A different body is written with exactly one edit call."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        pr = self._open_pr()
        mock_repository.get_pull.return_value = pr
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration.update_pull_request_body("owner", "test-repo", 42, "## Fixed\n- a")

        assert result["success"] is True
        assert result["changed"] is True
        assert result["pr_number"] == 42
        assert result["pr_status"] == "open"
        pr.edit.assert_called_once_with(body="## Fixed\n- a")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_pull_request_body_idempotent(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """An identical body returns changed=False without calling edit."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        pr = self._open_pr(body="same")
        mock_repository.get_pull.return_value = pr
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration.update_pull_request_body("owner", "test-repo", 42, "same")

        assert result["success"] is True
        assert result["changed"] is False
        pr.edit.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_pull_request_body_refuses_closed_pr(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A PR that is not open is refused with 409 and never edited."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        pr = self._open_pr(state="closed")
        mock_repository.get_pull.return_value = pr
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration.update_pull_request_body("owner", "test-repo", 42, "new")

        assert result["success"] is False
        assert result["status_code"] == 409
        assert result["error"] == "pull request #42 is closed, not open"
        pr.edit.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_pull_request_body_refuses_public_repo(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A public repo raises RepoVisibilityRefusedError before the PR is touched."""
        from services.repo_visibility_guard import RepoVisibilityRefusedError

        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        mock_repository.private = False
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        with pytest.raises(RepoVisibilityRefusedError):
            integration.update_pull_request_body("owner", "test-repo", 42, "new")

        mock_repository.get_pull.assert_not_called()
        mock_repository.get_pull.return_value.edit.assert_not_called()

    @patch("services.github_integration.GithubException", MockGithubError)
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_pull_request_body_github_error(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """A GitHub API error is returned as success=False with its status_code."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client
        mock_repository.get_pull.side_effect = MockGithubError(404, "Not Found")
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration.update_pull_request_body("owner", "test-repo", 42, "new")

        assert result["success"] is False
        assert result["status_code"] == 404

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_with_resolves(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
        sample_resolves,
    ):
        """Test PR creation with file changes."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        mock_client.get_user.return_value = user
        mock_repository.create_pull.return_value = mock_pull_request
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=mock_repository,
                    is_fork=True,
                    is_clone=False,
                    original_url=None,
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())
        object.__setattr__(integration, "_update_files", MagicMock())

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
            resolves=sample_resolves,
        )

        assert result["success"] is True
        integration._update_files.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_create_pr_github_error(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test PR creation handles GitHub API error."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                side_effect=MockGithubError(
                    status=403,
                    message="Forbidden",
                    data={"message": "API rate limit exceeded"},
                )
            ),
        )

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False
        assert "API rate limit exceeded" in result["error"]

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_unexpected_error(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test PR creation handles unexpected error."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration, "_ensure_fork", MagicMock(side_effect=Exception("Unexpected error"))
        )

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False
        assert "Unexpected error" in result["error"]

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_no_client(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test PR creation fails without client."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        integration.github_client = None
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=mock_repository,
                    is_fork=True,
                    is_clone=False,
                    original_url=None,
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_pr_with_colon_in_head_branch(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
    ):
        """Test PR creation with colon in head branch (user:branch format)."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        mock_client.get_user.return_value = user
        mock_repository.create_pull.return_value = mock_pull_request
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration,
            "_ensure_fork",
            MagicMock(
                return_value=GitHubRepoInfo(
                    repository=mock_repository,
                    is_fork=True,
                    is_clone=False,
                    original_url=None,
                )
            ),
        )
        object.__setattr__(integration, "_create_branch", MagicMock())

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="test-repo",
            title="Test PR",
            message="Test message",
            head_branch="forkuser:feature-branch",
        )

        assert result["success"] is True


# ============================================================================
# Fork Management Tests
# ============================================================================


@pytest.mark.unit
class TestForkManagement:
    """Tests for fork management."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_ensure_fork_user_owns_repo(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test _ensure_fork when user owns the repository."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "owner"
        mock_client.get_user.return_value = user
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration._ensure_fork("owner", "test-repo")

        assert result.is_fork is False
        assert result.is_clone is False

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_ensure_fork_existing_fork(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test _ensure_fork with existing fork."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        # Set up fork
        fork_repo = MagicMock()
        fork_repo.fork = True
        fork_repo.full_name = "forkuser/test-repo"
        parent = MagicMock()
        parent.full_name = "owner/test-repo"
        fork_repo.parent = parent
        mock_client.get_repo.return_value = fork_repo

        integration = GitHubIntegration(github_token="test-token")
        result = integration._ensure_fork("owner", "test-repo")

        assert result.is_fork is True

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_ensure_fork_creates_new_fork(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test _ensure_fork creates new fork when none exists."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "forkuser"
        user.create_fork = MagicMock(return_value=mock_repository)
        mock_client.get_user.return_value = user

        # First call raises 404 (no fork), second returns original
        mock_client.get_repo.side_effect = [
            MockGithubError(status=404, message="Not found"),
            mock_repository,
        ]

        integration = GitHubIntegration(github_token="test-token")
        result = integration._ensure_fork("owner", "test-repo")

        assert result.is_fork is True
        user.create_fork.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_ensure_fork_repo_not_found(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test _ensure_fork handles repo not found."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "owner"
        mock_client.get_user.return_value = user
        mock_client.get_repo.side_effect = MockGithubError(status=404, message="Not found")

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="not found"):
            integration._ensure_fork("owner", "nonexistent-repo")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_ensure_fork_access_denied(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test _ensure_fork handles access denied."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "owner"
        mock_client.get_user.return_value = user
        mock_client.get_repo.side_effect = MockGithubError(status=403, message="Forbidden")

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="Access denied"):
            integration._ensure_fork("owner", "private-repo")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_ensure_fork_no_client(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """_ensure_fork raises typed GitHubAuthError when no token is available."""
        from services.github_integration import GitHubAuthError

        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        integration.github_client = None

        with pytest.raises(GitHubAuthError, match="No GitHub token configured"):
            integration._ensure_fork("owner", "test-repo")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_ensure_fork_user_repo_not_fork(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test _ensure_fork when user has repo but it's not a fork."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "forkuser"
        mock_client.get_user.return_value = user

        # User's repo is not a fork
        mock_repository.fork = False
        mock_client.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = integration._ensure_fork("owner", "test-repo")

        assert result.is_fork is False
        assert result.is_clone is True


# ============================================================================
# Branch Creation Tests
# ============================================================================


@pytest.mark.unit
class TestBranchCreation:
    """Tests for branch creation."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_branch_success(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test successful branch creation."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        mock_repository.get_git_ref.return_value = base_ref

        new_ref = MagicMock()
        mock_repository.create_git_ref.return_value = new_ref

        integration = GitHubIntegration(github_token="test-token")
        result = integration._create_branch(mock_repository, "feature-branch", "main")

        assert result == new_ref
        mock_repository.create_git_ref.assert_called_once_with(
            ref="refs/heads/feature-branch", sha="abc123"
        )

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_branch_with_colon(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test branch creation extracts name from user:branch format."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        mock_repository.get_git_ref.return_value = base_ref

        new_ref = MagicMock()
        mock_repository.create_git_ref.return_value = new_ref

        integration = GitHubIntegration(github_token="test-token")
        integration._create_branch(mock_repository, "user:feature-branch", "main")

        mock_repository.create_git_ref.assert_called_once_with(
            ref="refs/heads/feature-branch", sha="abc123"
        )

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_branch_invalid_name(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test branch creation rejects invalid names."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="Invalid branch name"):
            integration._create_branch(mock_repository, "invalid branch!@#", "main")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_create_branch_already_exists(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test branch creation handles existing branch."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        base_ref = MagicMock()
        base_ref.object.sha = "abc123"
        mock_repository.get_git_ref.return_value = base_ref

        mock_repository.create_git_ref.side_effect = MockGithubError(
            status=422, message="Reference already exists"
        )

        existing_ref = MagicMock()
        existing_ref.object.sha = "abc123"
        mock_repository.get_git_ref.return_value = existing_ref

        integration = GitHubIntegration(github_token="test-token")
        result = integration._create_branch(mock_repository, "feature-branch", "main")

        assert result is existing_ref
        assert result.object.sha == "abc123"

    @staticmethod
    def _integration():
        return GitHubIntegration(github_token="test-token")

    @staticmethod
    def _repo_with_heads(mock_repository, heads: dict[str, str], default: str = "main"):
        """Repo whose only branch heads are ``heads`` (name -> sha)."""

        def get_ref(ref):
            name = ref.removeprefix("heads/")
            if name not in heads:
                raise MockGithubError(status=404, message="Not found")
            return SimpleNamespace(object=SimpleNamespace(sha=heads[name]))

        mock_repository.get_git_ref.side_effect = get_ref
        mock_repository.default_branch = default
        mock_repository.full_name = "o/r"
        mock_repository.create_git_ref.return_value = MagicMock()
        return mock_repository

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_implicit_main_base_falls_back_to_default_branch(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        """fr defaults `branch` to "main"; on a master-default repo that must still work."""
        repo = self._repo_with_heads(mock_repository, {"master": "def456"}, default="master")

        self._integration()._create_branch(repo, "feature-branch", "main")

        repo.create_git_ref.assert_called_once_with(ref="refs/heads/feature-branch", sha="def456")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_explicit_missing_base_raises_instead_of_using_default(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        """The defect: an explicit base that does not exist was silently replaced
        by the default branch head, so the fix branch started from the wrong code."""
        repo = self._repo_with_heads(mock_repository, {"main": "def456"})
        repo.get_commit.side_effect = AssertionError("a non-hex name is not a commit")

        with pytest.raises(BaseRefNotFoundError, match="release-9"):
            self._integration()._create_branch(repo, "feature-branch", "release-9")

        repo.create_git_ref.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_commit_sha_base_cuts_branch_from_that_commit(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        sha = "c" * 40  # low-entropy: a real-looking SHA trips detect-secrets
        repo = self._repo_with_heads(mock_repository, {"main": "def456"})
        repo.get_commit.return_value = SimpleNamespace(sha=sha)

        self._integration()._create_branch(repo, "feature-branch", sha)

        repo.get_commit.assert_called_once_with(sha)
        repo.create_git_ref.assert_called_once_with(ref="refs/heads/feature-branch", sha=sha)

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_abbreviated_sha_base_is_expanded_to_the_full_commit(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        full = "abcdef0" + "1" * 33
        repo = self._repo_with_heads(mock_repository, {"main": "def456"})
        repo.get_commit.return_value = SimpleNamespace(sha=full)

        self._integration()._create_branch(repo, "feature-branch", "abcdef0")

        repo.create_git_ref.assert_called_once_with(ref="refs/heads/feature-branch", sha=full)

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_unknown_sha_base_raises_instead_of_using_default(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        repo = self._repo_with_heads(mock_repository, {"main": "def456"})
        repo.get_commit.side_effect = MockGithubError(status=422, message="No commit found")

        with pytest.raises(BaseRefNotFoundError, match="neither a branch nor a commit"):
            self._integration()._create_branch(repo, "feature-branch", "deadbeef")

        repo.create_git_ref.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch("services.github_integration.GithubException", MockGithubError)
    def test_branch_named_like_a_sha_resolves_as_the_branch(
        self, mock_validator, mock_github, mock_auth, mock_repository
    ):
        repo = self._repo_with_heads(mock_repository, {"cafe1234": "branchsha"})

        self._integration()._create_branch(repo, "feature-branch", "cafe1234")

        repo.get_commit.assert_not_called()
        repo.create_git_ref.assert_called_once_with(
            ref="refs/heads/feature-branch", sha="branchsha"
        )


# ============================================================================
# File Operations Tests
# ============================================================================


@pytest.mark.unit
class TestFileOperations:
    """Tests for file operations."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_modified(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test updating modified files."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "file_path": "src/main.py",
                "content": "new content",
                "change_type": "modified",
            }
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_create_or_update_file", MagicMock(return_value="sha123"))

        result = integration._update_files(mock_repository, "feature-branch", changes)

        assert result == "sha123"
        integration._create_or_update_file.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_apply_changes_counts_committed_files(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """_apply_changes returns (last_commit_sha, committed_count); a no-op
        write (helper returns None) is NOT counted."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {"file_path": "a.py", "content": "x", "change_type": "modified"},
            {"file_path": "b.py", "content": "y", "change_type": "modified"},
        ]

        integration = GitHubIntegration(github_token="test-token")
        # First file commits, second is a no-op (None).
        object.__setattr__(
            integration,
            "_create_or_update_file",
            MagicMock(side_effect=["sha_a", None]),
        )

        sha, committed = integration._apply_changes(mock_repository, "feature-branch", changes)
        assert sha == "sha_a"
        assert committed == 1

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_apply_changes_all_noop_zero_commits(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """When every requested change is an identical-content no-op, the
        per-file helper returns None for all → (last_commit_sha, committed) is
        ('', 0): the zero-commit case the router must surface as no_changes
        rather than a fabricated SUCCESS. The legacy _update_files wrapper
        still returns the empty string for backward compatibility."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {"file_path": "a.py", "content": "same", "change_type": "modified"},
            {"file_path": "b.py", "content": "same", "change_type": "modified"},
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_create_or_update_file", MagicMock(return_value=None))

        sha, committed = integration._apply_changes(mock_repository, "feature-branch", changes)
        assert sha == ""
        assert committed == 0
        # Backward-compatible wrapper still returns just the (empty) sha string.
        assert integration._update_files(mock_repository, "feature-branch", changes) == ""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_added(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test adding new files."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "file_path": "src/new_file.py",
                "content": "new content",
                "change_type": "added",
            }
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_create_or_update_file", MagicMock(return_value="sha456"))

        result = integration._update_files(mock_repository, "feature-branch", changes)

        assert result == "sha456"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_deleted(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test deleting files."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "file_path": "src/old_file.py",
                "change_type": "deleted",
            }
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_delete_file", MagicMock(return_value="sha789"))

        result = integration._update_files(mock_repository, "feature-branch", changes)

        assert result == "sha789"
        integration._delete_file.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_renamed(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test renaming files."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "file_path": "src/new_name.py",
                "old_file_path": "src/old_name.py",
                "content": "file content",
                "change_type": "renamed",
            }
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_rename_file", MagicMock(return_value="shaabc"))

        result = integration._update_files(mock_repository, "feature-branch", changes)

        assert result == "shaabc"
        integration._rename_file.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_missing_required_fields(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test update_files raises error for missing required fields."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "change_type": "deleted",
                # Missing file_path
            }
        ]

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="file_path is required"):
            integration._update_files(mock_repository, "feature-branch", changes)

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_or_update_file_create_new(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test creating a new file."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_repository.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        create_result = {"commit": MagicMock(sha="new_sha")}
        mock_repository.create_file.return_value = create_result

        integration = GitHubIntegration(github_token="test-token")

        with patch(
            "services.github_integration.GithubException",
            MockGithubError,
        ):
            result = integration._create_or_update_file(
                mock_repository, "feature-branch", "src/new.py", "content"
            )

        assert result == "new_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_or_update_file_update_existing(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test updating an existing file."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        existing_file = MagicMock()
        existing_file.sha = "old_sha"
        existing_file.decoded_content = b"old content"
        mock_repository.get_contents.return_value = existing_file

        update_result = {"commit": MagicMock(sha="updated_sha")}
        mock_repository.update_file.return_value = update_result

        integration = GitHubIntegration(github_token="test-token")
        result = integration._create_or_update_file(
            mock_repository, "feature-branch", "src/existing.py", "new content"
        )

        assert result == "updated_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_or_update_file_no_changes(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test update file skips when content unchanged."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        existing_file = MagicMock()
        existing_file.sha = "old_sha"
        existing_file.decoded_content = b"same content"
        mock_repository.get_contents.return_value = existing_file

        integration = GitHubIntegration(github_token="test-token")
        result = integration._create_or_update_file(
            mock_repository, "feature-branch", "src/file.py", "same content"
        )

        assert result is None
        mock_repository.update_file.assert_not_called()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_or_update_file_bytes_content(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test handling bytes content."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_repository.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        create_result = {"commit": MagicMock(sha="new_sha")}
        mock_repository.create_file.return_value = create_result

        integration = GitHubIntegration(github_token="test-token")

        with patch(
            "services.github_integration.GithubException",
            MockGithubError,
        ):
            result = integration._create_or_update_file(
                mock_repository,
                "feature-branch",
                "src/file.py",
                "bytes content",
            )

        assert result == "new_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_delete_file_success(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test deleting a file."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        existing_file = MagicMock()
        existing_file.sha = "file_sha"
        mock_repository.get_contents.return_value = existing_file

        delete_result = {"commit": MagicMock(sha="delete_sha")}
        mock_repository.delete_file.return_value = delete_result

        integration = GitHubIntegration(github_token="test-token")
        result = integration._delete_file(mock_repository, "feature-branch", "src/old.py")

        assert result == "delete_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_delete_file_not_found(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test deleting a file that doesn't exist."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_repository.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        integration = GitHubIntegration(github_token="test-token")
        result = integration._delete_file(mock_repository, "feature-branch", "nonexistent.py")

        assert result is None

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_rename_file_success(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test renaming a file."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration, "_create_or_update_file", MagicMock(return_value="create_sha")
        )
        object.__setattr__(integration, "_delete_file", MagicMock(return_value="delete_sha"))

        result = integration._rename_file(
            mock_repository,
            "feature-branch",
            "src/old.py",
            "src/new.py",
            "content",
        )

        assert result == "delete_sha"
        integration._create_or_update_file.assert_called_once()
        integration._delete_file.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_rename_file_without_content(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test renaming file fetches content when not provided."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        old_file = MagicMock()
        old_file.decoded_content = b"file content"
        mock_repository.get_contents.return_value = old_file

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(
            integration, "_create_or_update_file", MagicMock(return_value="create_sha")
        )
        object.__setattr__(integration, "_delete_file", MagicMock(return_value="delete_sha"))

        integration._rename_file(
            mock_repository,
            "feature-branch",
            "src/old.py",
            "src/new.py",
            None,  # No content provided
        )

        mock_repository.get_contents.assert_called_with("src/old.py", ref="feature-branch")


# ============================================================================
# PR Description Generation Tests
# ============================================================================


@pytest.mark.unit
class TestPRDescriptionGeneration:
    """Tests for PR description generation."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_generate_pr_description(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        sample_resolves,
    ):
        """Test generating PR description."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        result = integration.generate_pr_description(sample_resolves)

        assert "## Code Updates" in result
        assert "## Changes Made" in result
        assert "src/main.py" in result
        assert "Bug fix" in result

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_generate_pr_description_with_context(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        sample_resolves,
    ):
        """Test generating PR description with issue context."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        result = integration.generate_pr_description(
            sample_resolves, issue_description="Fix null pointer exception"
        )

        assert "## Context" in result
        assert "Fix null pointer exception" in result

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_generate_pr_description_empty_resolves(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test generating PR description with no resolves."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        result = integration.generate_pr_description([])

        assert "## Code Updates" in result
        assert "## Testing" in result


# ============================================================================
# Fix Validation Tests
# ============================================================================


@pytest.mark.unit
class TestFixValidation:
    """Tests for fix validation (graceful stub — coding-analysis not yet available)."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    def test_validate_fix_returns_valid_stub(
        self,
        mock_github,
        mock_auth,
    ):
        """Test fix validation returns graceful stub response."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        result = integration.validate_fix({"file_path": "test.py", "content": "pass"})

        assert result["valid"] is True
        assert isinstance(result["warnings"], list)
        assert len(result["warnings"]) == 1
        assert "unavailable" in result["warnings"][0].lower()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    def test_validate_fix_with_any_input(
        self,
        mock_github,
        mock_auth,
    ):
        """Test fix validation always returns valid since it's a stub."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        integration = GitHubIntegration(github_token="test-token")
        result = integration.validate_fix({"anything": "goes"})

        assert result["valid"] is True


# ============================================================================
# Upstream PR Creation Tests
# ============================================================================


@pytest.mark.unit
class TestUpstreamPRCreation:
    """Tests for upstream PR creation (dual-token workflow).

    The create_upstream_pull_request method uses get_workspaces_client() from main
    and get_github_token_manager from services.github_token_manager.
    """

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    @patch("services.github_token_manager.get_github_token_manager")
    async def test_create_upstream_pr_success(
        self,
        mock_token_manager,
        mock_get_containers,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
    ):
        """Test successful upstream PR creation."""
        mock_auth.Token.return_value = MagicMock()
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance

        # Mock containers client returning container info dict
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "test-user",
                "user_token_id": "token-123",
                "original_repo_owner": "owner",
                "original_repo_name": "repo",
                "system_fork_owner": "nonagenticai",
            }
        )
        mock_get_containers.return_value = mock_client

        mock_token_mgr = MagicMock()
        mock_token_mgr.get_user_token = AsyncMock(return_value="user-github-token")
        mock_token_manager.return_value = mock_token_mgr

        mock_repository.create_pull.return_value = mock_pull_request
        mock_repository.private = True  # upstream target is the user's PRIVATE repo
        mock_github_instance.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="test-container",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is True
        assert result["pr_number"] == 42

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    @patch("services.secret_resolver.resolve_git_token")
    async def test_create_upstream_pr_byo_secret_same_repo(
        self,
        mock_resolve,
        mock_get_containers,
        mock_github,
        mock_auth,
        mock_repository,
        mock_pull_request,
    ):
        """BYO flow: no system fork → same-repo PR, token from secrets-service.

        ``system_fork_owner`` is None (no fork is created when we hold the user's
        token), so the head ref falls back to the repo owner and the token is
        resolved from secrets-service via ``secret_config_id`` + tenant.
        """
        mock_auth.Token.return_value = MagicMock()
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance

        mock_client = AsyncMock()
        # coding-containers wraps the model in a {"status",...,"container":{...}}
        # envelope — the metadata must be read from inside "container".
        mock_client.get_workspace = AsyncMock(
            return_value={
                "status": "success",
                "message": "Container details retrieved",
                "container": {
                    "user_id": "test-user",
                    "user_token_id": "placeholder",
                    "original_repo_owner": "owner",
                    "original_repo_name": "repo",
                    "system_fork_owner": None,
                    "secret_config_id": "sec-1",
                },
            }
        )
        mock_get_containers.return_value = mock_client
        # Token resolves from secrets-service (the in-memory store is bypassed).
        mock_resolve.return_value = "user-github-token"

        mock_repository.create_pull.return_value = mock_pull_request
        mock_repository.private = True  # upstream target is the user's PRIVATE repo
        mock_github_instance.get_repo.return_value = mock_repository

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="test-container",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
            base_branch="main",
            tenant_id="tenant-xyz",
        )

        assert result["success"] is True
        assert result["pr_number"] == 42
        # Token was resolved from secrets-service with the forwarded tenant.
        mock_resolve.assert_awaited_once_with("sec-1", "tenant-xyz")
        # Tenant was forwarded to the container lookup (else 404 cross-tenant).
        #
        # Asserted case-insensitively and by CONTENT, because these headers are now
        # built by services.outbound.optional_sibling_headers rather than hand-rolled
        # here. That helper emits coding-client's canonical `X-Tenant-Id` spelling and
        # ALSO attaches this service's own licence when CODING_SERVICE_TOKEN is set —
        # required since coding-containers stopped trusting a bare tenant header
        # (TRUST_TENANT_HEADER=false), which was 401ing the whole PR path.
        # Pinning the exact dict froze the old hand-rolled spelling and would fail on a
        # purely cosmetic difference while saying nothing about the credential.
        assert mock_client.get_workspace.await_args.args[0] == "test-container"
        sent = mock_client.get_workspace.await_args.kwargs.get("headers") or {}
        lowered = {k.lower(): v for k, v in sent.items()}
        assert lowered.get("x-tenant-id") == "tenant-xyz"
        # Same-repo PR: head ref falls back to the repo owner (no fork).
        _, pull_kwargs = mock_repository.create_pull.call_args
        assert pull_kwargs["head"] == "owner:feature-branch"

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    async def test_create_upstream_pr_container_not_found(
        self,
        mock_get_containers,
        mock_github,
        mock_auth,
    ):
        """Test upstream PR creation with missing container."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        from coding_client.clients import ServiceError

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=ServiceError(404, "Not found"))
        mock_get_containers.return_value = mock_client

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="nonexistent",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    async def test_create_upstream_pr_missing_metadata(
        self,
        mock_get_containers,
        mock_github,
        mock_auth,
    ):
        """Missing repository metadata (owner/name) fails the PR.

        The token is now resolved from secrets-service (not the dead in-memory
        store), so the hard requirement is the repository identity; ``user_id`` /
        ``user_token_id`` are no longer load-bearing for the metadata gate.
        """
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "user-1",
                "user_token_id": "token-1",
                "original_repo_owner": None,
                "original_repo_name": None,
                "secret_config_id": "sec-1",
            }
        )
        mock_get_containers.return_value = mock_client

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="test-container",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False
        assert "missing repository metadata" in result["error"]

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    @patch("services.github_token_manager.get_github_token_manager")
    async def test_create_upstream_pr_token_not_found(
        self,
        mock_token_manager,
        mock_get_containers,
        mock_github,
        mock_auth,
    ):
        """Test upstream PR creation when user token not found."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "test-user",
                "user_token_id": "token-123",
                "original_repo_owner": "owner",
                "original_repo_name": "repo",
                "system_fork_owner": "nonagenticai",
            }
        )
        mock_get_containers.return_value = mock_client

        mock_token_mgr = MagicMock()
        mock_token_mgr.get_user_token = AsyncMock(return_value=None)
        mock_token_manager.return_value = mock_token_mgr

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="test-container",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False
        # owner "owner" is NOT the platform org. With no BYO / user token the
        # ONLY remaining credential is the tenant-bound public App (2026-09-22),
        # and with no tenant that is refused — naming the repo and a machine
        # code — rather than falling back to any platform token.
        assert result["error_code"] == "tenant_required"
        assert result["status_code"] == 403
        assert "owner/repo" in result["error"]

    @pytest.mark.asyncio
    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("main.get_workspaces_client")
    @patch("services.github_token_manager.get_github_token_manager")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    async def test_create_upstream_pr_github_error(
        self,
        mock_token_manager,
        mock_get_containers,
        mock_github_cls,
        mock_auth,
    ):
        """Test upstream PR creation handles GitHub API error."""
        mock_auth.Token.return_value = MagicMock()

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "user_id": "test-user",
                "user_token_id": "token-123",
                "original_repo_owner": "owner",
                "original_repo_name": "repo",
                "system_fork_owner": "nonagenticai",
            }
        )
        mock_get_containers.return_value = mock_client

        mock_token_mgr = MagicMock()
        mock_token_mgr.get_user_token = AsyncMock(return_value="user-token")
        mock_token_manager.return_value = mock_token_mgr

        mock_github_instance = MagicMock()
        mock_github_instance.get_repo.side_effect = MockGithubError(status=404, message="Not found")
        mock_github_cls.return_value = mock_github_instance

        integration = GitHubIntegration(github_token="test-token")
        result = await integration.create_upstream_pull_request(
            container_name="test-container",
            title="Test PR",
            message="Test message",
            head_branch="feature-branch",
        )

        assert result["success"] is False
        assert "GitHub API error" in result["error"]


# ============================================================================
# Clone Repository Creation Tests
# ============================================================================


@pytest.mark.unit
class TestCloneRepositoryCreation:
    """Tests for creating clone repositories when forking fails."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_create_user_repository_from_original(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test creating user repository from original."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        new_repo = MagicMock()
        new_repo.full_name = "testuser/test-repo"
        new_repo.private = True  # fallback repo must be confirmed private (#118)
        user.create_repo.return_value = new_repo
        mock_client.get_repo.side_effect = MockGithubError(status=404, message="Not found")

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_copy_repository_content", MagicMock())

        integration._create_user_repository_from_original(mock_repository, user)

        user.create_repo.assert_called_once()
        integration._copy_repository_content.assert_called_once()

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_create_user_repository_name_conflict(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test creating repository handles name conflicts."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "testuser"
        new_repo = MagicMock()
        new_repo.full_name = "testuser/test-repo-clone-1"
        new_repo.private = True  # fallback repo must be confirmed private (#118)
        user.create_repo.return_value = new_repo

        # First name exists, second doesn't
        call_count = [0]

        def get_repo_side_effect(name):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock()  # Exists
            raise MockGithubError(status=404, message="Not found")

        mock_client.get_repo.side_effect = get_repo_side_effect

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_copy_repository_content", MagicMock())

        integration._create_user_repository_from_original(mock_repository, user)

        # Should use incremented name
        assert user.create_repo.called


# ============================================================================
# Content Copy Tests
# ============================================================================


@pytest.mark.unit
class TestContentCopy:
    """Tests for copying repository content."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_copy_repository_content(
        self,
        mock_validator,
        mock_github,
        mock_auth,
    ):
        """Test copying repository content."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        source_repo = MagicMock()
        source_repo.default_branch = "main"
        source_repo.full_name = "owner/source"
        source_repo.html_url = "https://github.com/owner/source"

        readme_content = MagicMock()
        readme_content.decoded_content = b"# Test Repo"

        def get_contents_side_effect(file_name, ref=None):
            if file_name == "README.md":
                return readme_content
            raise MockGithubError(status=404, message="Not found")

        source_repo.get_contents.side_effect = get_contents_side_effect

        target_repo = MagicMock()
        target_repo.name = "target"
        target_repo.default_branch = "main"
        target_repo.get_contents.side_effect = MockGithubError(status=404, message="Not found")

        integration = GitHubIntegration(github_token="test-token")
        integration._copy_repository_content(source_repo, target_repo)

        # Should try to copy files
        target_repo.create_file.assert_called()


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_create_or_update_file_empty_content(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test handling empty content (warning but still creates)."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        mock_repository.get_contents.side_effect = MockGithubError(status=404, message="Not found")
        create_result = {"commit": MagicMock(sha="new_sha")}
        mock_repository.create_file.return_value = create_result

        integration = GitHubIntegration(github_token="test-token")

        with patch(
            "services.github_integration.GithubException",
            MockGithubError,
        ):
            result = integration._create_or_update_file(
                mock_repository, "feature-branch", "empty.py", "   "
            )

        # Should still create the file
        assert result == "new_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_get_contents_returns_list(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test handling when get_contents returns a list."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        # Return a list with one item
        file_content = MagicMock()
        file_content.sha = "file_sha"
        file_content.decoded_content = b"content"
        mock_repository.get_contents.return_value = [file_content]

        update_result = {"commit": MagicMock(sha="updated_sha")}
        mock_repository.update_file.return_value = update_result

        integration = GitHubIntegration(github_token="test-token")
        result = integration._create_or_update_file(
            mock_repository, "feature-branch", "file.py", "new content"
        )

        assert result == "updated_sha"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_delete_file_is_directory(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test deleting raises error for directories."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        # Return a list (indicating directory)
        mock_repository.get_contents.return_value = [MagicMock(), MagicMock()]

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="Cannot delete directory"):
            integration._delete_file(mock_repository, "feature-branch", "src/")

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_rename_file_directory_error(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test renaming raises error for directories."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        # Return a list (indicating directory)
        mock_repository.get_contents.return_value = [MagicMock(), MagicMock()]

        integration = GitHubIntegration(github_token="test-token")

        with pytest.raises(ValueError, match="Cannot rename directory"):
            integration._rename_file(mock_repository, "feature-branch", "src/", "new_src/", None)

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    def test_update_files_unknown_change_type(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test update_files handles unknown change type."""
        mock_auth.Token.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        changes = [
            {
                "file_path": "src/file.py",
                "content": "content",
                "change_type": "unknown_type",
            }
        ]

        integration = GitHubIntegration(github_token="test-token")
        object.__setattr__(integration, "_create_or_update_file", MagicMock(return_value="sha123"))

        # Should fall through to create_or_update
        result = integration._update_files(mock_repository, "feature-branch", changes)

        assert result == "sha123"

    @patch("services.github_integration.HAS_GITHUB", True)
    @patch("services.github_integration.Auth")
    @patch("services.github_integration.Github")
    @patch("services.github_integration.FixValidator")
    @patch(
        "services.github_integration.GithubException",
        MockGithubError,
    )
    def test_ensure_fork_fork_wrong_parent(
        self,
        mock_validator,
        mock_github,
        mock_auth,
        mock_repository,
    ):
        """Test _ensure_fork when fork exists but for different repo."""
        mock_auth.Token.return_value = MagicMock()
        mock_client = MagicMock()
        mock_github.return_value = mock_client

        user = MagicMock()
        user.login = "forkuser"
        user.create_fork = MagicMock(return_value=mock_repository)
        mock_client.get_user.return_value = user

        # Fork exists but for different parent
        fork_repo = MagicMock()
        fork_repo.fork = True
        fork_repo.full_name = "forkuser/test-repo"
        wrong_parent = MagicMock()
        wrong_parent.full_name = "other-owner/test-repo"
        fork_repo.parent = wrong_parent

        def get_repo_side_effect(name):
            if "forkuser" in name:
                return fork_repo
            return mock_repository

        mock_client.get_repo.side_effect = get_repo_side_effect

        integration = GitHubIntegration(github_token="test-token")
        result = integration._ensure_fork("owner", "test-repo")

        # Should use existing repo as clone since fork parent doesn't match
        assert result.is_clone is True


# ============================================================================
# GitHubRepoInfo Tests
# ============================================================================


@pytest.mark.unit
class TestGitHubRepoInfo:
    """Tests for GitHubRepoInfo named tuple."""

    def test_github_repo_info_creation(self, mock_repository):
        """Test creating GitHubRepoInfo."""
        info = GitHubRepoInfo(
            repository=mock_repository,
            is_fork=True,
            is_clone=False,
            original_url="https://github.com/owner/repo",
        )

        assert info.repository == mock_repository
        assert info.is_fork is True
        assert info.is_clone is False
        assert info.original_url == "https://github.com/owner/repo"

    def test_github_repo_info_with_none_url(self, mock_repository):
        """Test GitHubRepoInfo with None original_url."""
        info = GitHubRepoInfo(
            repository=mock_repository,
            is_fork=False,
            is_clone=False,
            original_url=None,
        )

        assert info.original_url is None
