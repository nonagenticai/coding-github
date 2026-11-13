"""Unit tests for CoreGitHubValidationService in services/github_validation.py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.github_validation import CoreGitHubValidationService


# Lightweight request stub that mirrors what the validation service expects
@dataclass
class _RepoRequest:
    repo_url: str
    access_token: str | None = None
    branch: str = "main"
    pr_number: int | None = None


# Mock GithubException used by the service
class _GithubException(Exception):  # noqa: N818
    def __init__(self, status: int, data: dict | None = None):
        self.status = status
        self.data = data or {"message": f"HTTP {status}"}
        super().__init__(str(data))


def _env_settings(environment: str):
    """Stub of the settings object the validation service resolves the env from.

    The environment is configured as CODING_GITHUB_ENVIRONMENT, so patching the
    bare ``ENVIRONMENT`` var (as these tests used to) no longer selects anything —
    it never did in production either, which was the bug. Patch the accessor.
    """
    normalised = environment.strip().lower()
    return SimpleNamespace(
        environment=environment,
        is_production=normalised == "production",
        is_dev_or_test=normalised in ("development", "dev", "testing", "test"),
    )


# ---------------------------------------------------------------------------
# 1. test_detect_test_mode_env_flags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env_var,env_value",
    [
        ("PYTEST_CURRENT_TEST", "tests/unit/test_foo.py::test_bar"),
        ("TESTING", "true"),
        ("TEST_MODE", "1"),
        ("CI", "yes"),
        ("SCHEMATHESIS_TEST", "true"),
        ("SCHEMATHESIS_TESTING", "on"),
    ],
)
def test_detect_test_mode_env_flags(env_var, env_value):
    """Each environment flag should cause test mode to be detected."""
    env = {env_var: env_value}
    with patch.dict(os.environ, env, clear=False):
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True, f"{env_var}={env_value} should enable test mode"


# ---------------------------------------------------------------------------
# 2. test_feature_flag_test_mode
# ---------------------------------------------------------------------------
def test_feature_flag_test_mode():
    """Feature flag 'enable_github_test_mode' should enable test mode."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={"enable_github_test_mode": True},
    ):
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True


# ---------------------------------------------------------------------------
# 3. test_production_bypass_denied
# ---------------------------------------------------------------------------
def test_production_bypass_denied():
    """Bypass flag should be rejected when the environment is production."""
    # Must remove PYTEST_CURRENT_TEST or it will trigger test mode before bypass check
    remove_keys = [
        "PYTEST_CURRENT_TEST",
        "TESTING",
        "TEST_MODE",
        "CI",
        "SCHEMATHESIS_TEST",
        "SCHEMATHESIS_TESTING",
    ]
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch(
            "services.github_validation.get_settings",
            return_value=_env_settings("production"),
        ),
        patch.dict(os.environ, {}, clear=False),
    ):
        for k in remove_keys:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        assert svc.test_mode is False


# ---------------------------------------------------------------------------
# 4. test_dev_bypass_allowed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_val", ["development", "testing", "test", "dev"])
def test_dev_bypass_allowed(env_val):
    """Bypass flag should be accepted in dev/test environments."""
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch(
            "services.github_validation.get_settings",
            return_value=_env_settings(env_val),
        ),
    ):
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True


# ---------------------------------------------------------------------------
# 5. test_is_test_token_positive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token",
    [
        "test_token_abc",
        "fake_token_xyz",
        "mock_token_123",
        "dummy_token_456",
        "github_test_token_789",
        "ghp_test_something",
        "test_anything",
        "ghp_123456_foo",
        "1234567890abcdef_bar",
        "schemathesis_run",
    ],
)
def test_is_test_token_positive(token):
    """Known test token patterns should be identified."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        assert svc._is_test_token(token) is True, f"Token '{token}' should be recognized as test"


# ---------------------------------------------------------------------------
# 6. test_is_test_token_negative
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token",
    [
        "ghp_abcdefRealTokenXYZQWERTYUIOPLK",
        "gho_SomeOAuthAppTokenHere",
        "github_pat_ABCxyzRealWorldProdKey",
        None,
        "",
    ],
)
def test_is_test_token_negative(token):
    """Real-looking tokens and None/empty should not match."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        assert svc._is_test_token(token) is False, f"Token '{token}' should NOT be a test token"


# ---------------------------------------------------------------------------
# 7. test_schemathesis_token
# ---------------------------------------------------------------------------
def test_schemathesis_token():
    """When SCHEMATHESIS_TESTING is set, tokens starting with 'schemathesis' are accepted."""
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={},
        ),
        patch.dict(os.environ, {"SCHEMATHESIS_TESTING": "true"}, clear=False),
    ):
        svc = CoreGitHubValidationService()
        assert svc._is_test_token("schemathesis_generated_abc") is True


# ---------------------------------------------------------------------------
# 8. test_parse_repo_url_valid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/owner/repo", ("owner", "repo")),
        ("https://github.com/owner/repo.git", ("owner", "repo")),
        ("https://github.com/my-org/my-repo.git", ("my-org", "my-repo")),
    ],
)
def test_parse_repo_url_valid(url, expected):
    """Valid GitHub URLs should be correctly parsed to (owner, repo)."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        assert svc._parse_repo_url(url) == expected


# ---------------------------------------------------------------------------
# 9. test_parse_repo_url_invalid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",
        "not-a-url",
    ],
)
def test_parse_repo_url_invalid(url):
    """Invalid/non-GitHub URLs should raise ValueError."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        with pytest.raises(ValueError):
            svc._parse_repo_url(url)


# ---------------------------------------------------------------------------
# 10. test_parse_github_pr_url_valid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/owner/repo/pull/123", ("owner", "repo", 123)),
        ("https://github.com/owner/repo/pull/1/files", ("owner", "repo", 1)),
        ("https://github.com/owner/repo/pull/42/commits", ("owner", "repo", 42)),
    ],
)
def test_parse_github_pr_url_valid(url, expected):
    """PR URLs should be parsed to (owner, repo, number)."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        assert svc.parse_github_pr_url(url) == expected


# ---------------------------------------------------------------------------
# 11. test_parse_github_pr_url_invalid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/owner/repo/pull/1",
        "https://github.com/owner/repo/issues/42",
        "https://github.com/owner/repo",
        "https://github.com/owner/repo/pull/abc",
    ],
)
def test_parse_github_pr_url_invalid(url):
    """Bad PR URLs should raise ValueError."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={},
    ):
        svc = CoreGitHubValidationService()
        with pytest.raises(ValueError):
            svc.parse_github_pr_url(url)


# ---------------------------------------------------------------------------
# 12. test_public_repo_no_token
# ---------------------------------------------------------------------------
async def test_public_repo_no_token():
    """Public repo should validate without a token using unauthenticated Github client."""
    mock_repo = MagicMock()
    mock_repo.private = False
    mock_repo.full_name = "owner/repo"

    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={},
        ),
        patch.dict(os.environ, {}, clear=False),
        patch("services.github_validation.Github", return_value=mock_g),
    ):
        # Remove env vars that would trigger test_mode
        for key in (
            "PYTEST_CURRENT_TEST",
            "TESTING",
            "TEST_MODE",
            "CI",
            "SCHEMATHESIS_TEST",
            "SCHEMATHESIS_TESTING",
        ):
            os.environ.pop(key, None)

        svc = CoreGitHubValidationService()
        svc.test_mode = False  # force non-test mode

        request = _RepoRequest(
            repo_url="https://github.com/owner/repo",
            access_token=None,
        )
        # Should not raise
        await svc.validate_repository_access(request)
        mock_g.get_repo.assert_called_once_with("owner/repo")


# ---------------------------------------------------------------------------
# 13. test_private_repo_with_token
# ---------------------------------------------------------------------------
async def test_private_repo_with_token():
    """Private repo should validate successfully with a real token."""
    mock_repo = MagicMock()
    mock_repo.full_name = "owner/private-repo"
    mock_branch = MagicMock()

    mock_user = MagicMock()
    mock_user.login = "testuser"

    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_repo.get_branch.return_value = mock_branch
    mock_g.close = MagicMock()

    mock_auth_cls = MagicMock()

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={},
        ),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
    ):
        mock_auth.Token = mock_auth_cls

        svc = CoreGitHubValidationService()
        svc.test_mode = False

        request = _RepoRequest(
            repo_url="https://github.com/owner/private-repo",
            access_token="ghp_realTokenValue123456789ABCDEF",
            branch="main",
        )
        await svc.validate_repository_access(request)
        mock_g.get_repo.assert_called_once_with("owner/private-repo")


# ---------------------------------------------------------------------------
# 14. test_github_api_404
# ---------------------------------------------------------------------------
async def test_github_api_404():
    """A 404 from the GitHub API should be mapped to NotFoundException."""

    # The service catches GithubException and re-raises as NotFoundException.
    # We need to make sure the GithubException type used in the isinstance/except
    # check matches what we raise, so we patch it at module level.
    exc = _GithubException(404, {"message": "Not Found"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    # We also need NotFoundException to accept keyword args. Monkey-patch it.
    class _NotFoundException(Exception):  # noqa: N818
        def __init__(self, *args, **kwargs):
            super().__init__(kwargs.get("message", ""))

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={},
        ),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
    ):
        mock_auth.Token = MagicMock()

        svc = CoreGitHubValidationService()
        svc.test_mode = False

        request = _RepoRequest(
            repo_url="https://github.com/owner/missing-repo",
            access_token="ghp_realTokenValueXYZQWERTY",
        )
        with pytest.raises(_NotFoundException):
            await svc.validate_repository_access(request)


# ---------------------------------------------------------------------------
# 15. test_malformed_repo_url
# ---------------------------------------------------------------------------
async def test_malformed_repo_url():
    """Malformed repo URLs should map to ValidationException."""

    class _ValidationException(Exception):  # noqa: N818
        def __init__(self, *args, **kwargs):
            super().__init__(kwargs.get("message", ""))

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={},
        ),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        request = _RepoRequest(
            repo_url="not-a-valid-url",
            access_token="ghp_realTokenValueXYZQWERTY",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(request)
