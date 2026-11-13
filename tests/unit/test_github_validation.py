"""Extended unit tests for CoreGitHubValidationService — covers uncovered branches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.github_app_auth import GitHubAppMintError, GitHubAppNotInstalledError
from services.github_validation import CoreGitHubValidationService, get_github_validation_service
from tests.conftest import PLATFORM_TEST_TENANT


@pytest.fixture(autouse=True)
def _repo_owner_is_the_platform_org(monkeypatch):
    """These tests pin the PLATFORM credential ladder (explicit → env → internal
    App → anonymous) and its mechanics. Since 2026-09-22 that ladder serves ONLY
    ``system_github_owner``; a customer owner goes through the tenant-bound public
    App instead (tests/unit/test_tenant_repo_token.py). So the fixture owner used
    throughout this module is declared the platform org here.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")


# Lightweight request stub matching service expectations
@dataclass
class _RepoRequest:
    repo_url: str
    access_token: str | None = None
    branch: str = "main"
    pr_number: int | None = None


class _GithubException(Exception):  # noqa: N818
    def __init__(self, status: int, data: dict | None = None):
        self.status = status
        self.data = data or {"message": f"HTTP {status}"}
        super().__init__(str(data))


class _NotFoundException(Exception):  # noqa: N818
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get("message", ""))


class _ValidationException(Exception):  # noqa: N818
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get("message", ""))


def _env_settings(environment: str):
    """Stub of the settings object the validation service resolves the env from.

    The environment is configured as CODING_GITHUB_ENVIRONMENT, so patching the
    bare ``ENVIRONMENT`` var (as these tests used to) selects nothing — which was
    precisely the production bug. Patch the accessor instead.
    """
    normalised = environment.strip().lower()
    return SimpleNamespace(
        environment=environment,
        is_production=normalised == "production",
        is_dev_or_test=normalised in ("development", "dev", "testing", "test"),
    )


# Helper to clear test-mode env vars so we can test production-like paths
_TEST_ENV_KEYS = [
    "PYTEST_CURRENT_TEST",
    "TESTING",
    "TEST_MODE",
    "CI",
    "SCHEMATHESIS_TEST",
    "SCHEMATHESIS_TESTING",
]


def _make_svc(feature_flags=None, clear_test_env=True):
    """Create a CoreGitHubValidationService with controlled env and flags."""
    ff = feature_flags or {}
    svc = None
    with patch("services.github_validation.load_feature_flags", return_value=ff):
        if clear_test_env:
            for k in _TEST_ENV_KEYS:
                os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
    if clear_test_env:
        svc.test_mode = False
    return svc


# ---- Patches shared across many tests ----
def _common_patches():
    """Return context managers for GithubException, NotFoundException, ValidationException."""
    return (
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
    )


# ===========================================================================
# get_github_validation_service factory (line 703)
# ===========================================================================
def test_get_github_validation_service_factory():
    """Factory should return a CoreGitHubValidationService instance."""
    svc = get_github_validation_service()
    assert isinstance(svc, CoreGitHubValidationService)


# ===========================================================================
# __init__ when HAS_GITHUB is False (lines 21-25, 41)
# ===========================================================================
def test_init_without_github_library():
    """When HAS_GITHUB is False the service should still initialize."""
    with patch("services.github_validation.HAS_GITHUB", False):
        svc = _make_svc()
        # Service should work even without PyGithub
        assert svc is not None


# ===========================================================================
# _detect_test_mode — truthy env values on non-CI vars (lines 67-72)
# ===========================================================================
@pytest.mark.parametrize("env_var", ["TESTING", "TEST_MODE", "SCHEMATHESIS_TEST"])
def test_detect_test_mode_truthy_values(env_var):
    """Non-CI/PYTEST env vars with truthy values should enable test mode."""
    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch.dict(os.environ, {env_var: "yes"}, clear=False),
    ):
        for k in _TEST_ENV_KEYS:
            if k != env_var:
                os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True


def test_detect_test_mode_non_truthy_value():
    """A non-truthy, non-CI env var value should NOT enable test mode from that var."""
    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch.dict(os.environ, {"TESTING": "maybe"}, clear=False),
    ):
        for k in _TEST_ENV_KEYS:
            if k != "TESTING":
                os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        # "maybe" is not in ("true","1","yes","on") and TESTING is not PYTEST_CURRENT_TEST/CI
        assert svc.test_mode is False


# ===========================================================================
# _detect_test_mode — feature flag enable_github_test_mode (lines 80-81)
# ===========================================================================
def test_detect_test_mode_feature_flag():
    """Feature flag enable_github_test_mode should enable test mode."""
    with patch(
        "services.github_validation.load_feature_flags",
        return_value={"enable_github_test_mode": True},
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True


# ===========================================================================
# _detect_test_mode — bypass in dev env (lines 94-98)
# ===========================================================================
@pytest.mark.parametrize("env", ["development", "testing", "test", "dev"])
def test_detect_test_mode_bypass_dev_env(env):
    """github_bypass_validation in dev/test env should enable test mode."""
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch("services.github_validation.get_settings", return_value=_env_settings(env)),
        patch.dict(os.environ, {}, clear=False),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        assert svc.test_mode is True


# ===========================================================================
# _should_bypass_validation — bypass flag in dev env (lines 143-156)
# ===========================================================================
def test_should_bypass_validation_flag_dev_env():
    """Bypass flag in a dev environment should return True."""
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch("services.github_validation.get_settings", return_value=_env_settings("development")),
        patch.dict(os.environ, {}, clear=False),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False  # isolate the bypass-flag path
        svc.feature_flags = {"github_bypass_validation": True}
        req = _RepoRequest(repo_url="https://github.com/nonagenticai/repo")
        assert svc._should_bypass_validation(req) is True


def test_should_bypass_validation_flag_prod_env():
    """Bypass flag in production should NOT bypass validation."""
    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch("services.github_validation.get_settings", return_value=_env_settings("production")),
        patch.dict(os.environ, {}, clear=False),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc.feature_flags = {"github_bypass_validation": True}
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realProductionTokenXYZ",
        )
        # Should NOT bypass because environment is production
        assert svc._should_bypass_validation(req) is False


# ===========================================================================
# _should_bypass_validation — test mode (lines 160-161)
# ===========================================================================
def test_should_bypass_validation_test_mode():
    """When test_mode is True, bypass validation should return True."""
    svc = _make_svc(clear_test_env=False)
    svc.test_mode = True
    svc.feature_flags = {}
    req = _RepoRequest(repo_url="https://github.com/nonagenticai/repo")
    assert svc._should_bypass_validation(req) is True


# ===========================================================================
# _should_bypass_validation — test token (lines 165-166)
# ===========================================================================
def test_should_bypass_validation_test_token():
    """When access_token is a test token, bypass should return True."""
    svc = _make_svc()
    svc.feature_flags = {}
    req = _RepoRequest(
        repo_url="https://github.com/nonagenticai/repo",
        access_token="test_token_abc",
    )
    assert svc._should_bypass_validation(req) is True


# ===========================================================================
# validate_repository_access — bypass with valid URL (lines 229-234)
# ===========================================================================
async def test_validate_repo_bypass_valid_url():
    """In bypass mode, a valid URL should pass and return early."""
    svc = _make_svc()
    svc.test_mode = True  # trigger bypass
    req = _RepoRequest(repo_url="https://github.com/nonagenticai/repo")
    # Should not raise
    await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — bypass with bad URL, but contains github.com (lines 235-240)
# ===========================================================================
async def test_validate_repo_bypass_bad_url_with_github_com():
    """In bypass mode, a bad URL that contains 'github.com' should succeed."""
    svc = _make_svc()
    svc.test_mode = True
    req = _RepoRequest(repo_url="https://github.com/only-owner")
    # _parse_repo_url raises ValueError because < 2 path parts, but it contains github.com
    await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


async def test_validate_repo_bypass_bad_url_no_github():
    """In bypass mode, a bad URL without 'github.com' should raise ValueError."""
    svc = _make_svc()
    svc.test_mode = True
    req = _RepoRequest(repo_url="https://gitlab.com/owner/repo")
    with pytest.raises(ValueError, match="github.com"):
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — URL too short (lines 248-252)
# ===========================================================================
async def test_validate_repo_url_too_short():
    """A very short URL should raise ValidationException."""
    with patch("services.github_validation.ValidationException", _ValidationException):
        svc = _make_svc()
        req = _RepoRequest(repo_url="http://x", access_token="ghp_realTokenXYZ123456789")
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — special test data (lines 259-260)
# ===========================================================================
@pytest.mark.parametrize("url", ["0/0", "0", "/0/0", "//0"])
async def test_validate_repo_special_test_data(url):
    """Special test data like '0/0' should raise ValidationException."""
    with patch("services.github_validation.ValidationException", _ValidationException):
        svc = _make_svc()
        req = _RepoRequest(repo_url=url, access_token="ghp_realTokenXYZ123456789")
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — invalid hostname (line 277)
# ===========================================================================
async def test_validate_repo_invalid_hostname():
    """Non-github.com hostname should raise ValidationException."""
    with patch("services.github_validation.ValidationException", _ValidationException):
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://gitlab.com/owner/repo",
            access_token="ghp_realTokenXYZ123456789",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — too few path parts (line 285)
# ===========================================================================
async def test_validate_repo_too_few_path_parts():
    """URL with only owner and no repo should raise ValidationException."""
    with patch("services.github_validation.ValidationException", _ValidationException):
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/owner",
            access_token="ghp_realTokenXYZ123456789",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — test token in non-test mode (lines 302-308)
# ===========================================================================
async def test_validate_repo_test_token_non_test_mode():
    """Test token outside test mode should warn but continue to GitHub API call."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_repo.get_branch.return_value = MagicMock()
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        # Override bypass to force through to the inner test-token check (lines 301-308)
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_abc",
            branch="main",
        )
        # Bypass is disabled, so it proceeds to validation.
        # test_token_abc IS a test token but test_mode is False, so it warns (line 308)
        # and continues to GitHub API call
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        mock_g.get_repo.assert_called_once_with("nonagenticai/repo")


# ===========================================================================
# validate_repository_access — test token in test mode returns early (lines 302-306)
# ===========================================================================
async def test_validate_repo_test_token_in_test_mode_returns_early():
    """Test token in test mode should return early without calling GitHub API."""
    mock_g = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc(clear_test_env=False)
        svc.test_mode = False  # don't bypass
        svc.feature_flags = {}
        # But test token should still be detected...
        # We need the validation to proceed to the authenticated code path.
        # Let's set test_mode to False but has_valid_token and _is_test_token both True
        # and self.test_mode = True for the inner check at line 303
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_xyz",
            branch="main",
        )
        # _should_bypass_validation returns True because test_token -> bypass
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        # Because bypass happened, Github().get_repo should NOT be called
        mock_g.get_repo.assert_not_called()


# ===========================================================================
# validate_repository_access — token fallback to env (lines 329-331, 336-339)
# ===========================================================================
async def test_validate_repo_explicit_token_is_used_with_env_pat_armed():
    """The caller's explicit token drives the probe; the armed env PAT is not a
    rung (the env fallback this test once covered was removed 2026-09-30)."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.return_value = MagicMock()
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch.dict(
            os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envFallbackToken123456789"}, clear=False
        ),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        mock_g.get_repo.assert_called_once_with("nonagenticai/repo")


async def test_validate_repo_no_token_no_env_fallback():
    """When token is empty and no env fallback, go to public repo path."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.private = False
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        mock_g.get_repo.assert_called_once_with("nonagenticai/repo")


# ===========================================================================
# validate_repository_access — skip auth test for test token (line 348)
# ===========================================================================
async def test_validate_repo_skip_auth_for_test_token():
    """When token is a test token, authentication test should be skipped."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.return_value = MagicMock()
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        # Use test_token pattern but NOT in _should_bypass_validation's test_token check
        # because bypass is checked first. We need to NOT bypass, but still reach line 348.
        # _should_bypass_validation returns True for test tokens via _is_test_token, so
        # we can't easily reach line 348 without bypassing.
        # Let's override _should_bypass_validation to return False but keep _is_test_token working.
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_skip_auth",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        # g.get_user() should NOT have been called because test token skips auth
        mock_g.get_user.assert_not_called()
        mock_g.get_repo.assert_called_once_with("nonagenticai/repo")


# ===========================================================================
# validate_repository_access — branch fallback to default (lines 363-389)
# ===========================================================================
async def test_validate_repo_branch_fallback_main_to_default():
    """When 'main' branch doesn't exist, should fallback to repo's default branch."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.default_branch = "master"

    branch_exc = _GithubException(404, {"message": "Branch not found"})
    default_branch_mock = MagicMock()

    def get_branch_side_effect(branch_name):
        if branch_name == "main":
            raise branch_exc
        return default_branch_mock

    mock_repo.get_branch.side_effect = get_branch_side_effect
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)
        assert req.branch == "master"


async def test_validate_repo_branch_not_found_non_main():
    """When a non-'main' branch doesn't exist, should raise ValueError."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    branch_exc = _GithubException(404, {"message": "Branch not found"})
    mock_repo.get_branch.side_effect = branch_exc
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="feature-xyz",
        )
        with pytest.raises(ValueError, match="Branch 'feature-xyz' not found"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


async def test_validate_repo_branch_fallback_default_also_fails():
    """When 'main' AND default branch both fail, should raise ValueError."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.default_branch = "master"
    branch_exc = _GithubException(404, {"message": "Branch not found"})
    mock_repo.get_branch.side_effect = branch_exc  # always fails
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        with pytest.raises(ValueError, match="Neither 'main' nor default branch"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


async def test_validate_repo_branch_non_404_error():
    """A non-404 branch error should re-raise through the outer handler."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    branch_exc = _GithubException(500, {"message": "Server error"})
    mock_repo.get_branch.side_effect = branch_exc
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_g = MagicMock()
    mock_g.get_user.return_value = mock_user
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        # The inner except re-raises, then outer except catches status 500 -> ValueError
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — 401 with test token in test mode (lines 400-416)
# ===========================================================================
async def test_validate_repo_401_test_token_test_mode():
    """401 with test token in test mode should be gracefully handled."""
    exc = _GithubException(401, {"message": "Bad credentials"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = True
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_abc",
            branch="main",
        )
        # Should return without raising in test mode
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


async def test_validate_repo_401_test_token_schemathesis_mode():
    """401 with test token in SCHEMATHESIS_TESTING mode should succeed."""
    exc = _GithubException(401, {"message": "Bad credentials"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch.dict(os.environ, {"SCHEMATHESIS_TESTING": "true"}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            if k != "SCHEMATHESIS_TESTING":
                os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_abc",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


async def test_validate_repo_401_test_token_non_test_mode_raises():
    """401 with test token outside test mode should raise ValidationException."""
    exc = _GithubException(401, {"message": "Bad credentials"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="test_token_abc",
            branch="main",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — 401 with real token (lines 424-428)
# ===========================================================================
async def test_validate_repo_401_real_token():
    """401 with a real token should raise ValidationException with INVALID_TOKEN."""
    exc = _GithubException(401, {"message": "Bad credentials"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — 403 rate limit (lines 429-435)
# ===========================================================================
async def test_validate_repo_403_rate_limit():
    """403 with rate limit message should raise ValidationException."""
    exc = _GithubException(403, {"message": "API rate limit exceeded"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — 403 forbidden (lines 436-439)
# ===========================================================================
async def test_validate_repo_403_forbidden():
    """403 without rate limit message should raise ValidationException."""
    exc = _GithubException(403, {"message": "Resource not accessible"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — other GithubException (line 440)
# ===========================================================================
async def test_validate_repo_other_github_error():
    """Other GithubException status codes should raise ValueError."""
    exc = _GithubException(500, {"message": "Internal Server Error"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789ABCDEF",
            branch="main",
        )
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — private repo no token (line 450)
# ===========================================================================
async def test_validate_repo_private_no_token():
    """Private repo without token should raise ValueError."""
    mock_repo = MagicMock()
    mock_repo.private = True
    mock_repo.full_name = "nonagenticai/repo"
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
        )
        with pytest.raises(ValueError, match="private"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — public repo 404 (lines 456-464)
# ===========================================================================
async def test_validate_repo_public_404():
    """Public repo 404 should raise NotFoundException."""
    exc = _GithubException(404, {"message": "Not Found"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
        )
        with pytest.raises(_NotFoundException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — public repo 403 rate limit (lines 465-471)
# ===========================================================================
async def test_validate_repo_public_403_rate_limit():
    """Public repo 403 with rate limit should raise ValidationException."""
    exc = _GithubException(403, {"message": "API rate limit exceeded"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — public repo 403 non-rate-limit (lines 472-475)
# ===========================================================================
async def test_validate_repo_public_403_non_rate_limit():
    """Public repo 403 without rate limit should raise ValidationException."""
    exc = _GithubException(403, {"message": "Access denied"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_repository_access — public repo other error (lines 476-479)
# ===========================================================================
async def test_validate_repo_public_other_error():
    """Public repo other GithubException should raise ValidationException."""
    exc = _GithubException(500, {"message": "Server error"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.ValidationException", _ValidationException),
    ):
        for k in _TEST_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = CoreGitHubValidationService()
        svc.test_mode = False

        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# get_branch_head_commit — success with token (lines 495-554)
# ===========================================================================
async def test_get_branch_head_commit_with_token():
    """Should return commit SHA for a valid branch with token."""
    mock_branch = MagicMock()
    mock_branch.commit.sha = "abc123def456"
    mock_branch.name = "main"
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.return_value = mock_branch
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        result = await svc.get_branch_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            branch="main",
            access_token="ghp_realTokenXYZ123456789",
        )
        assert result == "abc123def456"


# ===========================================================================
# get_branch_head_commit — success without token (lines 515-518)
# ===========================================================================
async def test_get_branch_head_commit_no_token():
    """Should use unauthenticated client when no token is provided."""
    mock_branch = MagicMock()
    mock_branch.commit.sha = "abc123def456"
    mock_branch.name = "main"
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.return_value = mock_branch
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
    ):
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = _make_svc()
        result = await svc.get_branch_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            branch="main",
            access_token=None,
        )
        assert result == "abc123def456"


# ===========================================================================
# get_branch_head_commit — an env PAT is NEVER used (no PAT rung exists)
# ===========================================================================
async def test_get_branch_head_commit_ignores_an_env_pat():
    """A PAT in the environment is ignored: no App => the probe is anonymous."""
    mock_branch = MagicMock()
    mock_branch.commit.sha = "envfallback123"
    mock_branch.name = "main"
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.return_value = mock_branch
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envToken"}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        result = await svc.get_branch_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            branch="main",
            access_token="",
        )
        assert result == "envfallback123"
        mock_auth.Token.assert_not_called()


# ===========================================================================
# get_branch_head_commit — branch fallback (lines 526-550)
# ===========================================================================
async def test_get_branch_head_commit_branch_fallback():
    """When 'main' branch not found, should try default branch."""
    branch_exc = _GithubException(404, {"message": "Branch not found"})
    default_branch_mock = MagicMock()
    default_branch_mock.commit.sha = "default123"
    default_branch_mock.name = "master"

    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.default_branch = "master"

    def get_branch_side_effect(name):
        if name == "main":
            raise branch_exc
        return default_branch_mock

    mock_repo.get_branch.side_effect = get_branch_side_effect
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        result = await svc.get_branch_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            branch="main",
            access_token="ghp_token123",
        )
        assert result == "default123"


# ===========================================================================
# get_branch_head_commit — non-main branch 404 re-raises (lines 538-540)
# ===========================================================================
async def test_get_branch_head_commit_non_main_404():
    """Non-main branch 404 should raise ValueError."""
    branch_exc = _GithubException(404, {"message": "Branch not found"})
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.side_effect = branch_exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="Branch 'feature' not found"):
            await svc.get_branch_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                branch="feature",
                access_token="ghp_token123",
            )


# ===========================================================================
# get_branch_head_commit — 403 rate limit (lines 560-565)
# ===========================================================================
async def test_get_branch_head_commit_403_rate_limit():
    """403 with rate limit should raise ValueError."""
    exc = _GithubException(403, {"message": "API rate limit exceeded"})
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="rate limit"):
            await svc.get_branch_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                branch="main",
                access_token="ghp_token123",
            )


# ===========================================================================
# get_branch_head_commit — 403 non-rate-limit (line 566)
# ===========================================================================
async def test_get_branch_head_commit_403_non_rate_limit():
    """403 without rate limit should raise ValueError with API error message."""
    exc = _GithubException(403, {"message": "Access forbidden"})
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.get_branch_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                branch="main",
                access_token="ghp_token123",
            )


# ===========================================================================
# get_branch_head_commit — other error (line 567)
# ===========================================================================
async def test_get_branch_head_commit_other_error():
    """Other GithubException should raise ValueError."""
    exc = _GithubException(500, {"message": "Server error"})
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/repo"
    mock_repo.get_branch.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.get_branch_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                branch="main",
                access_token="ghp_token123",
            )


# ===========================================================================
# get_branch_head_commit — repo 404 outer except (lines 556-559)
# ===========================================================================
async def test_get_branch_head_commit_repo_404():
    """Repo 404 should raise ValueError."""
    exc = _GithubException(404, {"message": "Not Found"})
    mock_g = MagicMock()
    mock_g.get_repo.side_effect = exc
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="not found"):
            await svc.get_branch_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                branch="main",
                access_token="ghp_token123",
            )


# ===========================================================================
# validate_pull_request_access — success (lines 577-620)
# ===========================================================================
async def test_validate_pr_access_success():
    """Should validate PR access successfully."""
    mock_pr = MagicMock()
    mock_pr.state = "open"
    mock_pr.title = "Test PR"
    mock_pr.user.login = "testuser"
    mock_pr.head.ref = "feature"
    mock_pr.base.ref = "main"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789",
            pr_number=42,
        )
        await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)
        mock_repo.get_pull.assert_called_once_with(42)


# ===========================================================================
# validate_pull_request_access — closed PR warning (lines 607-611)
# ===========================================================================
async def test_validate_pr_access_closed():
    """Closed PR should log warning but succeed."""
    mock_pr = MagicMock()
    mock_pr.state = "closed"
    mock_pr.title = "Old PR"
    mock_pr.user.login = "testuser"
    mock_pr.head.ref = "feature"
    mock_pr.base.ref = "main"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realTokenXYZ123456789",
            pr_number=10,
        )
        # Should not raise even though PR is closed
        await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — no pr_number (line 579)
# ===========================================================================
async def test_validate_pr_access_no_pr_number():
    """Missing PR number should raise ValueError."""
    svc = _make_svc()
    req = _RepoRequest(
        repo_url="https://github.com/nonagenticai/repo",
        access_token="ghp_realTokenXYZ123456789",
        pr_number=None,
    )
    with pytest.raises(ValueError, match="PR number is required"):
        await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — no token (lines 598-600)
# ===========================================================================
async def test_validate_pr_access_no_token():
    """PR access without token should use unauthenticated client."""
    mock_pr = MagicMock()
    mock_pr.state = "open"
    mock_pr.title = "Test PR"
    mock_pr.user.login = "testuser"
    mock_pr.head.ref = "feature"
    mock_pr.base.ref = "main"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
    ):
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token=None,
            pr_number=5,
        )
        await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — an env PAT is NEVER used
# ===========================================================================
async def test_validate_pr_access_ignores_an_env_pat():
    """A PAT in the environment is ignored when the request token is empty."""
    mock_pr = MagicMock()
    mock_pr.state = "open"
    mock_pr.title = "Test PR"
    mock_pr.user.login = "testuser"
    mock_pr.head.ref = "feature"
    mock_pr.base.ref = "main"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envTokenPR"}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="",
            pr_number=5,
        )
        await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)
        mock_auth.Token.assert_not_called()


# ===========================================================================
# validate_pull_request_access — 404 (lines 627-630)
# ===========================================================================
async def test_validate_pr_access_404():
    """PR 404 should raise ValueError."""
    exc = _GithubException(404, {"message": "Not Found"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realToken123",
            pr_number=999,
        )
        with pytest.raises(ValueError, match="Pull request #999 not found"):
            await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — 403 rate limit (lines 631-636)
# ===========================================================================
async def test_validate_pr_access_403_rate_limit():
    """PR 403 rate limit should raise ValueError."""
    exc = _GithubException(403, {"message": "API rate limit exceeded"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realToken123",
            pr_number=42,
        )
        with pytest.raises(ValueError, match="rate limit"):
            await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — 403 non-rate-limit (line 637)
# ===========================================================================
async def test_validate_pr_access_403_non_rate_limit():
    """PR 403 without rate limit should raise ValueError."""
    exc = _GithubException(403, {"message": "Forbidden"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realToken123",
            pr_number=42,
        )
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# validate_pull_request_access — other error (line 638)
# ===========================================================================
async def test_validate_pr_access_other_error():
    """PR other GithubException should raise ValueError."""
    exc = _GithubException(500, {"message": "Server error"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        req = _RepoRequest(
            repo_url="https://github.com/nonagenticai/repo",
            access_token="ghp_realToken123",
            pr_number=42,
        )
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.validate_pull_request_access(req, tenant_id=PLATFORM_TEST_TENANT)


# ===========================================================================
# get_pull_request_head_commit — success (lines 650-677)
# ===========================================================================
async def test_get_pr_head_commit_success():
    """Should return head SHA for a valid PR."""
    mock_pr = MagicMock()
    mock_pr.head.sha = "pr_head_sha_123"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        result = await svc.get_pull_request_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            pr_number=42,
            access_token="ghp_token123",
        )
        assert result == "pr_head_sha_123"


# ===========================================================================
# get_pull_request_head_commit — no token (lines 668-670)
# ===========================================================================
async def test_get_pr_head_commit_no_token():
    """Should use unauthenticated client when no token."""
    mock_pr = MagicMock()
    mock_pr.head.sha = "pr_sha_no_token"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.GithubException", _GithubException),
    ):
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        svc = _make_svc()
        result = await svc.get_pull_request_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            pr_number=42,
            access_token=None,
        )
        assert result == "pr_sha_no_token"


# ===========================================================================
# get_pull_request_head_commit — an env PAT is NEVER used
# ===========================================================================
async def test_get_pr_head_commit_ignores_an_env_pat():
    """A PAT in the environment is ignored: the probe is anonymous."""
    mock_pr = MagicMock()
    mock_pr.head.sha = "pr_sha_env"

    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envPRToken"}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        result = await svc.get_pull_request_head_commit(
            "https://github.com/nonagenticai/repo",
            tenant_id=PLATFORM_TEST_TENANT,
            pr_number=42,
            access_token="",
        )
        assert result == "pr_sha_env"
        mock_auth.Token.assert_not_called()


# ===========================================================================
# get_pull_request_head_commit — 404 (lines 680-683)
# ===========================================================================
async def test_get_pr_head_commit_404():
    """PR 404 should raise ValueError."""
    exc = _GithubException(404, {"message": "Not Found"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="Pull request #99 not found"):
            await svc.get_pull_request_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                pr_number=99,
                access_token="ghp_token123",
            )


# ===========================================================================
# get_pull_request_head_commit — 403 rate limit (lines 684-689)
# ===========================================================================
async def test_get_pr_head_commit_403_rate_limit():
    """PR 403 rate limit should raise ValueError."""
    exc = _GithubException(403, {"message": "API rate limit exceeded"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="rate limit"):
            await svc.get_pull_request_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                pr_number=42,
                access_token="ghp_token123",
            )


# ===========================================================================
# get_pull_request_head_commit — 403 non-rate-limit (line 690)
# ===========================================================================
async def test_get_pr_head_commit_403_non_rate_limit():
    """PR 403 without rate limit should raise ValueError."""
    exc = _GithubException(403, {"message": "Forbidden"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.get_pull_request_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                pr_number=42,
                access_token="ghp_token123",
            )


# ===========================================================================
# get_pull_request_head_commit — other error (line 691)
# ===========================================================================
async def test_get_pr_head_commit_other_error():
    """PR other GithubException should raise ValueError."""
    exc = _GithubException(500, {"message": "Server error"})
    mock_repo = MagicMock()
    mock_repo.get_pull.side_effect = exc
    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.close = MagicMock()

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()
        with pytest.raises(ValueError, match="GitHub API error"):
            await svc.get_pull_request_head_commit(
                "https://github.com/nonagenticai/repo",
                tenant_id=PLATFORM_TEST_TENANT,
                pr_number=42,
                access_token="ghp_token123",
            )


# ===========================================================================
# Repository pre-flight token resolution: explicit -> env -> GitHub App -> None
#
# THE DEFECT THESE PIN. ``validate_repository_access`` resolved a token from
# ``request.access_token`` ALONE, so a tokenless request always fell to the
# ANONYMOUS PyGithub client and api.github.com answered 404 for every PRIVATE
# repository — ``Repository '... not found or is private' not found``, 80 times
# on one repo in 48h from 2026-09-10 14:33Z, while a working GitHub App
# credential sat mounted in the same pod that this path never consulted.
#
# ⚠️ WHY THE DOUBLE BELOW IS ASYMMETRIC — ON TWO AXES.
#
# (1) ANONYMOUS vs AUTHENTICATED. A mock that answers the SAME for both cannot
#     fail when the App rung is deleted — it would pass under both hypotheses.
#     So for a private repo the ANONYMOUS client 404s and the AUTHENTICATED
#     client succeeds, exactly as measured in the pod.
#
# (2) PAT vs INSTALLATION TOKEN — the axis the fix actually turns on, and the
#     one this double used to be SYMMETRIC on (``authed.get_user.return_value =
#     user`` for every credential). That made the headline test unable to fail
#     for the REAL failure: a ``ghs_…`` App installation token CANNOT call
#     ``GET /user``; api.github.com answers ``403 Resource not accessible by
#     integration``. Measured in the pod against a token minted by this repo's
#     own ``get_installation_token_for``::
#
#         minted: len=390 prefix=ghs_
#         get_user  RAISED GithubException 403 "Resource not accessible by integration"
#         get_repo  OK  private=True
#
#     So the double now 403s on ``get_user`` for a ``ghs_`` token and succeeds
#     for a PAT. The headline test therefore passes ONLY because of the
#     ``resolved.is_app`` skip in ``validate_repository_access``; delete that
#     skip and it reddens.
#
# ⚠️ HOW THE DOUBLE LEARNS THE TOKEN. ``Github(auth=...)`` receives whatever
# ``Auth.Token(...)`` returned, so the tests below install a PASS-THROUGH
# ``Auth.Token`` (``_passthrough_auth``) that returns the token string itself.
# Without it ``auth=`` is an opaque sentinel and the double is blind to the very
# distinction it exists to make.
# ===========================================================================
_PRIVATE_REPO = "nonagenticai/enterprise_mcp_server"
_PRIVATE_REPO_URL = "https://github.com/nonagenticai/enterprise_mcp_server"
_PUBLIC_REPO_URL = "https://github.com/nonagenticai/Hello-World"


def _passthrough_auth(mock_auth):
    """Make ``Auth.Token(t)`` return ``t`` so the ``Github`` double sees the token.

    ``assert_called_once_with`` on ``mock_auth.Token`` is unaffected.
    """
    mock_auth.Token = MagicMock(side_effect=lambda token: token)
    return mock_auth


def _differential_github(*, private: bool, revoked: bool = False):
    """Return ``(github_class_double, authed_client, anon_client)``.

    ``private=True`` makes the anonymous client raise 404 (GitHub's answer for a
    private repo with no credential) while the authenticated client returns the
    repo. ``private=False`` is the public control: both clients succeed.

    ``revoked=True`` models a credential GitHub no longer accepts: BOTH
    ``get_user`` and ``get_repo`` on the authenticated client answer
    ``401 Bad credentials``, while the anonymous client is unaffected.

    Regardless of ``revoked``, a ``ghs_``-prefixed (App installation) token
    always 403s on ``get_user`` and always succeeds on ``get_repo`` — that is
    GitHub's real behaviour for an integration, not a failure mode.
    """
    repo = MagicMock()
    repo.full_name = _PRIVATE_REPO if private else "nonagenticai/Hello-World"
    repo.private = private
    repo.default_branch = "main"
    repo.get_branch.return_value = MagicMock()

    user = MagicMock()
    user.login = "coding-github[bot]"

    # The credential the authenticated client was constructed with, recorded by
    # the factory below. ``get_user``'s answer depends on it — that is the whole
    # point of this double.
    state: dict[str, str] = {"token": ""}

    def _get_user():
        if state["token"].startswith("ghs_"):
            raise _GithubException(403, {"message": "Resource not accessible by integration"})
        if revoked:
            raise _GithubException(401, {"message": "Bad credentials"})
        return user

    def _get_repo(_full_name):
        if revoked and not state["token"].startswith("ghs_"):
            raise _GithubException(401, {"message": "Bad credentials"})
        return repo

    authed = MagicMock()
    authed.get_user.side_effect = _get_user
    authed.get_repo.side_effect = _get_repo
    authed.close = MagicMock()

    anon = MagicMock()
    if private:
        anon.get_repo.side_effect = _GithubException(404, {"message": "Not Found"})
    else:
        anon.get_repo.return_value = repo
    anon.close = MagicMock()

    def _factory(*_args, **kwargs):
        auth_obj = kwargs.get("auth")
        if auth_obj is None:
            return anon
        state["token"] = auth_obj if isinstance(auth_obj, str) else ""
        return authed

    return MagicMock(side_effect=_factory), authed, anon


def _authed_client_was_built(github_cls) -> bool:
    """True when at least one PyGithub client was constructed WITH ``auth=``."""
    return any(c.kwargs.get("auth") is not None for c in github_cls.call_args_list)


async def test_app_installation_token_authenticates_private_repo_preflight():
    """THE FIX. No explicit token, no env token, App configured => the App token
    is minted and the AUTHENTICATED client is used, so a PRIVATE repo validates.

    Kills TWO mutants:

    * remove the App rung — this request builds the anonymous client, whose
      ``get_repo`` 404s, and the call raises NotFoundException;
    * remove the ``resolved.is_app`` ``get_user()`` skip — the double 403s on
      ``get_user`` exactly as api.github.com does for an integration, so
      ``get_repo`` is never reached and ``get_user`` IS called.
    """
    github_cls, authed, anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mint.assert_called_once_with("nonagenticai", "enterprise_mcp_server")
    mock_auth.Token.assert_called_once_with("ghs_appInstallationToken")
    assert _authed_client_was_built(github_cls)
    # ⛔ The authentication probe must be SKIPPED for an installation token.
    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with(_PRIVATE_REPO)
    anon.get_repo.assert_not_called()


async def test_without_the_app_rung_the_private_repo_404_returns():
    """THE MUTANT CONTROL for the test above.

    ``app_auth_configured()`` false is exactly what deleting step (c) does to
    this call site: no credential is obtainable, the anonymous client is built,
    and the private repo 404s with the live discriminator message. Same double,
    same request as the test above — so that test cannot be passing vacuously.
    """
    github_cls, authed, _anon = _differential_github(private=True)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_app_auth.app_auth_configured", return_value=False),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_NotFoundException, match="not found or is private"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    assert not _authed_client_was_built(github_cls)
    mock_auth.Token.assert_not_called()
    authed.get_repo.assert_not_called()


async def test_anonymous_client_still_validates_public_repo_when_app_unconfigured():
    """POSITIVE CONTROL — the anonymous path must NOT have become auth-mandatory.

    App unconfigured and no token anywhere => the unauthenticated client is
    built (no ``auth=`` kwarg, ``Auth.Token`` never called) and a PUBLIC repo
    still validates. If this reddens, the fallback was turned into a hard
    requirement and every public tokenless request breaks.
    """
    github_cls, authed, anon = _differential_github(private=False)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=False),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    assert not _authed_client_was_built(github_cls)
    mock_auth.Token.assert_not_called()
    anon.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    authed.get_repo.assert_not_called()


async def test_explicit_access_token_wins_over_env_and_app():
    """PRECEDENCE rung 1. An explicit request token is used verbatim and never
    triggers an App mint — a caller's own credential is the caller's choice of
    identity."""
    github_cls, authed, _anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envToken000"}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(
            repo_url=_PRIVATE_REPO_URL,
            access_token="ghp_explicitCallerToken123",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_called_once_with("ghp_explicitCallerToken123")
    mint.assert_not_called()
    authed.get_repo.assert_called_once_with(_PRIVATE_REPO)


async def test_explicit_installation_token_also_skips_the_get_user_probe():
    """An EXPLICIT caller token that is itself an installation token must skip the
    `get_user()` probe.

    `is_app` is keyed on the CREDENTIAL, not on which rung answered, because GitHub
    403s "Resource not accessible by integration" for the credential no matter how it
    arrived. Without this case, mutating `_is_installation_token(explicit)` to False
    survives the whole suite (124/124) while a caller-supplied `ghs_` token reverts to
    the ORIGINAL OUTAGE SHAPE: get_user 403 -> ValidationException/ACCESS_FORBIDDEN ->
    HTTP 400, with get_repo never reached. Found by an adversarial audit as the one
    surviving mutation that mattered.
    """
    github_cls, authed, _anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(
            repo_url=_PRIVATE_REPO_URL,
            access_token="ghs_callerSuppliedInstallationToken",
            branch="main",
        )
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    # the caller's credential is used verbatim and no mint happens (rung 1 wins)
    mock_auth.Token.assert_called_once_with("ghs_callerSuppliedInstallationToken")
    mint.assert_not_called()
    # THE POINT: the probe is skipped because the CREDENTIAL is an installation token
    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with(_PRIVATE_REPO)


async def test_an_env_pat_is_ignored_and_the_app_token_is_used():
    """There is NO env rung (owner ruling 2026-09-30). With a PAT sitting in
    ``GITHUB_ACCESS_TOKEN`` and the App configured, the App mints and the PAT is
    never handed to GitHub. MUTANT: restore the env rung above the App and
    ``Auth.Token`` receives the PAT and ``mint`` is never called."""
    github_cls, authed, _anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_envTokenMustNotBeUsed"}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_called_once_with("ghs_appInstallationToken")
    mint.assert_called_once_with("nonagenticai", "enterprise_mcp_server")
    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with(_PRIVATE_REPO)


@pytest.mark.parametrize(
    "mint_error",
    [
        GitHubAppMintError("GitHub refused the mint"),
        GitHubAppNotInstalledError("App is not installed on this owner"),
    ],
    ids=["mint_failed", "not_installed"],
)
async def test_app_mint_failure_is_non_fatal_and_keeps_the_clean_404(mint_error):
    """NON-FATAL MINT. A failing App mint must leave the caller's error EXACTLY as
    it was — the clean ``NotFoundException`` "not found or is private".

    ⚠️ This is the specific regression to look for: if the App rung raised
    through, a routine private-repo 404 would surface as a 500 (or a raw
    ``GitHubAppAuthError``), the live discriminator between the anonymous and
    authenticated 404 messages would be destroyed, and the diagnosis that found
    this defect would stop being reproducible.
    """
    github_cls, authed, anon = _differential_github(private=True)
    mint = MagicMock(side_effect=mint_error)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_NotFoundException, match="not found or is private") as excinfo:
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    # Not a GitHubAppAuthError, and not an unhandled 500-shaped failure.
    assert not isinstance(excinfo.value, GitHubAppMintError)
    mint.assert_called_once_with("nonagenticai", "enterprise_mcp_server")
    assert not _authed_client_was_built(github_cls)
    anon.get_repo.assert_called_once_with(_PRIVATE_REPO)
    authed.get_repo.assert_not_called()


async def test_unexpected_app_error_is_non_fatal_and_keeps_the_clean_404():
    """NON-FATAL MINT, unexpected-exception arm. An error that is NOT a
    ``GitHubAppAuthError`` subclass (a bad settings double, a broken import, a
    TypeError inside PyGithub) must also degrade to anonymous, never escalate.
    """
    github_cls, _authed, anon = _differential_github(private=True)
    mint = MagicMock(side_effect=RuntimeError("private key is not a PEM"))

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_NotFoundException, match="not found or is private"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    assert not _authed_client_was_built(github_cls)
    anon.get_repo.assert_called_once_with(_PRIVATE_REPO)


async def test_test_token_behaviour_unchanged_by_token_resolution():
    """TEST TOKEN unchanged. A test token is still used verbatim (never replaced
    by an App token, even with the App fully configured) and still skips the
    ``get_user`` authentication probe."""
    github_cls, authed, _anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()
        svc.test_mode = False
        # Bypass would return early on a test token; disable it to reach the
        # resolution + client-selection code this test is about.
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token="test_token_abc", branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_called_once_with("test_token_abc")
    mint.assert_not_called()
    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with(_PRIVATE_REPO)


async def test_test_token_in_test_mode_still_never_calls_github():
    """TEST TOKEN unchanged, bypass arm. In test mode a test token still returns
    before any client is constructed — token resolution must not have pulled the
    GitHub call ahead of the bypass."""
    github_cls, _authed, _anon = _differential_github(private=True)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
    ):
        mock_auth.Token = MagicMock()
        svc = _make_svc(clear_test_env=False)
        svc.test_mode = True

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token="test_token_xyz", branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    github_cls.assert_not_called()
    mint.assert_not_called()


# ===========================================================================
# The get_user() authentication probe vs a GitHub App INSTALLATION token
#
# THE DEFECT THESE PIN. The first cut of the App rung was LATENT-BROKEN: every
# site ran ``g.get_user()`` as an "authentication test" before ``g.get_repo()``,
# and an App installation token cannot call ``GET /user`` — api.github.com
# answers ``403 Resource not accessible by integration``. Traced end to end in
# the pod: ``github_token_source=app`` -> ``ValidationException`` /
# ``ACCESS_FORBIDDEN`` -> **HTTP 400**, with ``get_repo`` call count **0**. So
# the rung that was supposed to fix a private-repo 404 could only ever DEGRADE
# the error. The fix mirrors ``github_integration``'s ``is_app_auth`` branch.
# ===========================================================================


async def test_the_double_403s_get_user_for_an_installation_token():
    """INSTRUMENT CONTROL. Proves the double is asymmetric on the axis that
    matters, so ``get_user.assert_not_called()`` above is not passing vacuously
    against a mock that would have answered happily either way.

    Mirrors the live measurement: a ``ghs_`` token 403s on ``get_user`` with
    GitHub's exact message and still succeeds on ``get_repo``.
    """
    github_cls, authed, _anon = _differential_github(private=True)

    # Build the authenticated client the way the service does: Github(auth=<token>).
    client = github_cls(auth="ghs_appInstallationToken", timeout=10, retry=None)
    assert client is authed

    with pytest.raises(_GithubException) as excinfo:
        client.get_user()
    assert excinfo.value.status == 403
    assert "not accessible by integration" in str(excinfo.value.data["message"])

    # ...and the repository read the probe was blocking works fine.
    assert client.get_repo(_PRIVATE_REPO).private is True


async def test_installation_token_skips_the_get_user_probe_on_a_public_repo():
    """The skip is a property of the CREDENTIAL, not of the repo's visibility.

    Same App rung, PUBLIC repo: still no ``get_user``, still an authenticated
    ``get_repo``. Without the skip this 403s and returns ACCESS_FORBIDDEN for a
    repository anyone can read.
    """
    github_cls, authed, anon = _differential_github(private=False)
    mint = MagicMock(return_value="ghs_appInstallationToken")

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_app_auth.get_installation_token_for", mint),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    anon.get_repo.assert_not_called()


async def test_an_installation_token_in_the_env_is_ignored_too():
    """Even a ``ghs_`` token pasted into ``GITHUB_ACCESS_TOKEN`` is never used:
    with no App configured the probe is anonymous and a PRIVATE repo 404s."""
    github_cls, authed, anon = _differential_github(private=True)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=False),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghs_installationTokenInEnv"}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_NotFoundException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_not_called()
    authed.get_repo.assert_not_called()
    anon.get_repo.assert_called_once_with(_PRIVATE_REPO)


# ===========================================================================
# A PLATFORM credential that GitHub rejects must DEGRADE, not escalate
#
# THE REGRESSION THESE PIN. Site 1 (``validate_repository_access``) had NO env
# rung before the token ladder: a tokenless request probed ANONYMOUSLY, so a
# PUBLIC repo returned 200 even with a broken platform credential, because
# nothing consulted it. With the ladder the same request authenticates (now:
# the GitHub App — there is no env PAT rung any more), ``get_repo`` 401s, and it becomes INVALID_TOKEN /
# HTTP 400 — a NEW failure on requests that never asked for a credential.
#
# ⛔ The boundary: an EXPLICIT caller token still surfaces its own error. A
# caller who passes a bad token must hear about it.
# ===========================================================================


async def test_revoked_app_token_degrades_to_an_anonymous_public_probe():
    """Rejected App-minted platform token + PUBLIC repo + tokenless request =>
    still validates, ANONYMOUSLY, exactly as it did before this path resolved a
    token at all. A WARNING names the degrade.

    Kills the mutant "escalate a rejected platform credential": without the
    degrade this raises ValidationException/INVALID_TOKEN.
    """
    github_cls, authed, anon = _differential_github(private=False, revoked=True)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_validation.logger") as mock_logger,
        patch(
            "services.github_app_auth.get_installation_token_for",
            return_value="app_revokedPlatformCred000",
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="main")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_called_once_with("app_revokedPlatformCred000")
    authed.get_user.assert_not_called()
    authed.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    # The anonymous re-probe is what actually answered.
    anon.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    warnings = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("github_token_degraded" in w for w in warnings)


async def test_revoked_app_token_on_a_private_repo_keeps_the_clean_404():
    """DEGRADE, never invent success. The same revoked env credential against a
    PRIVATE repo must land on the anonymous path's clean 404 — the live
    diagnostic discriminator — not on a 400 and not on a pass."""
    github_cls, _authed, anon = _differential_github(private=True, revoked=True)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch(
            "services.github_app_auth.get_installation_token_for",
            return_value="app_revokedPlatformCred000",
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PRIVATE_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_NotFoundException, match="not found or is private"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    anon.get_repo.assert_called_once_with(_PRIVATE_REPO)


async def test_revoked_explicit_token_still_errors_and_never_degrades():
    """THE BOUNDARY of the degrade. A caller-supplied token that GitHub rejects
    must still surface as INVALID_TOKEN — never be quietly replaced by an
    anonymous probe that succeeds and hides the caller's broken credential.

    Kills the mutant "degrade for every source": drop the ``source`` check and
    this call returns success and ``anon.get_repo`` is hit.
    """
    github_cls, authed, anon = _differential_github(private=False, revoked=True)

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=False),
        patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": ""}, clear=False),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()
        svc._should_bypass_validation = lambda req: False

        req = _RepoRequest(
            repo_url=_PUBLIC_REPO_URL,
            access_token="ghp_revokedCallerCredential000",
            branch="main",
        )
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    mock_auth.Token.assert_called_once_with("ghp_revokedCallerCredential000")
    authed.get_user.assert_called_once_with()
    anon.get_repo.assert_not_called()


async def test_rate_limited_app_token_does_not_degrade_to_anonymous():
    """A 403 RATE LIMIT is not a credential error, and the anonymous quota is
    LOWER — degrading it would make the outcome worse and lose the actionable
    RATE_LIMIT_EXCEEDED code. Pins the carve-out.
    """
    github_cls, authed, anon = _differential_github(private=False)
    authed.get_repo.side_effect = _GithubException(403, {"message": "API rate limit exceeded"})

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch(
            "services.github_app_auth.get_installation_token_for",
            return_value="app_rateLimitedPlatformCred000",
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="main")
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    anon.get_repo.assert_not_called()


# ===========================================================================
# The degrade must not swallow the BRANCH probe (follow-up to PR #88)
#
# THE DEFECT THESE PIN. ``_may_degrade_to_anonymous`` was written for ONE
# failure mode — GitHub REJECTING a credential the caller never supplied — but
# the ``except GithubException`` that consults it also wraps the
# ``repo.get_branch(...)`` block, and the retry it authorises,
# ``_validate_anonymously``, probes the REPOSITORY and checks NO BRANCH. So
# with an env token on a PUBLIC repo, a non-rate-limit 403 out of
# ``get_branch`` degraded, the anonymous repository probe succeeded, and
# ``validate_repository_access`` RETURNED SUCCESS for a branch nothing had
# validated — where before PR #88 it raised ACCESS_FORBIDDEN. A caller was
# being told a branch exists on the strength of a check that never ran.
#
# ⚠️ WHY THE DOUBLE IS A PUBLIC REPO. On a private repo the anonymous retry
# 404s, so the false success cannot appear and the test would pass under both
# hypotheses. Public is the only visibility on which the degrade can invent a
# success, which is exactly why the defect was invisible.
# ===========================================================================
_PLATFORM_APP_TOKEN = "app_platformToken0000"
_NOT_RATE_LIMIT_403 = {"message": "Resource not accessible by personal access token"}
_RATE_LIMIT_403 = {"message": "API rate limit exceeded"}


def _branch_probe_github(*, branch_exc=None, user_exc=None, repo_exc=None):
    """``(github_class_double, authed, anon)`` for a PUBLIC repo.

    ``get_user`` and ``get_repo`` succeed on the authenticated client unless
    ``user_exc`` is given; ``get_branch`` raises ``branch_exc`` when given. The
    anonymous client always succeeds — so if the degrade fires, the call returns
    SUCCESS and the branch was never checked.
    """
    repo = MagicMock()
    repo.full_name = "nonagenticai/Hello-World"
    repo.private = False
    repo.default_branch = "main"
    if branch_exc is not None:
        repo.get_branch.side_effect = branch_exc
    else:
        repo.get_branch.return_value = MagicMock()

    user = MagicMock()
    user.login = "octocat"

    authed = MagicMock()
    if user_exc is not None:
        authed.get_user.side_effect = user_exc
    else:
        authed.get_user.return_value = user
    if repo_exc is not None:
        authed.get_repo.side_effect = repo_exc
    else:
        authed.get_repo.return_value = repo
    authed.close = MagicMock()

    anon = MagicMock()
    anon.get_repo.return_value = repo
    anon.close = MagicMock()

    def _factory(*_args, **kwargs):
        return anon if kwargs.get("auth") is None else authed

    return MagicMock(side_effect=_factory), authed, anon


async def test_a_non_rate_limit_403_from_the_branch_probe_surfaces_and_never_degrades():
    """THE FIX. Platform (App) token, PUBLIC repo, ``get_branch`` 403s for a reason that is
    not a rate limit => ACCESS_FORBIDDEN surfaces and the anonymous retry never
    runs.

    Kills mutant (iii) "re-widen the degrade to swallow the branch probe": drop
    the phase check and this call RETURNS SUCCESS — ``anon.get_repo`` answers for
    the repository, no branch is validated, and the caller is told a branch
    exists that nothing looked at.
    """
    github_cls, authed, anon = _branch_probe_github(
        branch_exc=_GithubException(403, _NOT_RATE_LIMIT_403)
    )

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_validation.logger") as mock_logger,
        patch(
            "services.github_app_auth.get_installation_token_for", return_value=_PLATFORM_APP_TOKEN
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="release-1.2")
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    # The branch probe really was the thing that failed...
    authed.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    # ...and the degrade did NOT run in its place.
    anon.get_repo.assert_not_called()
    warnings = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("github_token_not_degraded" in w for w in warnings)
    assert not any("github_token_degraded" in w for w in warnings)


async def test_the_same_403_before_the_branch_probe_still_degrades_to_anonymous():
    """THE DIFFERENTIAL CONTROL for the test above, and the boundary of the
    narrowing.

    Same status, same message, same env credential, same PUBLIC repo — only the
    PHASE differs: the 403 comes out of the ``get_repo`` access probe (the App
    token skips ``get_user``) rather than out of ``get_branch``. Here the degrade is the whole point of PR #88 and
    must still fire, so the call SUCCEEDS anonymously.

    Without this test, "make ``_may_degrade_to_anonymous`` always return False"
    would look like a valid narrowing on the 403 path instead of the removal of
    the feature it is.
    """
    github_cls, authed, anon = _branch_probe_github(
        repo_exc=_GithubException(403, _NOT_RATE_LIMIT_403)
    )

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch("services.github_validation.logger") as mock_logger,
        patch(
            "services.github_app_auth.get_installation_token_for", return_value=_PLATFORM_APP_TOKEN
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="release-1.2")
        await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    authed.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    anon.get_repo.assert_called_once_with("nonagenticai/Hello-World")
    warnings = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("github_token_degraded" in w for w in warnings)


async def test_a_rate_limit_403_from_the_branch_probe_keeps_rate_limit_exceeded():
    """THE CARVE-OUT, on the branch probe. A 403 rate limit is not a credential
    error and the anonymous quota is LOWER, so it must never degrade — it keeps
    the actionable ``RATE_LIMIT_EXCEEDED`` code. Narrowing the degrade must not
    have moved this decision either.
    """
    github_cls, _authed, anon = _branch_probe_github(
        branch_exc=_GithubException(403, _RATE_LIMIT_403)
    )

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch(
            "services.github_app_auth.get_installation_token_for", return_value=_PLATFORM_APP_TOKEN
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="release-1.2")
        with pytest.raises(_ValidationException, match="rate limit"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    anon.get_repo.assert_not_called()


async def test_a_401_from_the_branch_probe_also_surfaces_rather_than_degrading():
    """The narrowing is keyed on the PHASE, not on the status code. A 401 raised
    by ``get_branch`` (a platform token revoked between ``get_repo`` and the
    branch read) is still a branch probe that did not complete, so it surfaces as
    INVALID_TOKEN instead of an anonymous success.
    """
    github_cls, _authed, anon = _branch_probe_github(
        branch_exc=_GithubException(401, {"message": "Bad credentials"})
    )

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch(
            "services.github_app_auth.get_installation_token_for", return_value=_PLATFORM_APP_TOKEN
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="release-1.2")
        with pytest.raises(_ValidationException):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    anon.get_repo.assert_not_called()


async def test_a_404_from_the_branch_probe_still_reports_the_missing_branch():
    """POSITIVE CONTROL — the branch probe's own 404 handling is untouched.

    A missing branch must still be the specific ``ValueError`` naming the branch,
    not a NotFoundException about the repository and not a degrade. If this
    reddens, the phase marker was set somewhere that swallowed the 404 path.
    """
    github_cls, _authed, anon = _branch_probe_github(
        branch_exc=_GithubException(404, {"message": "Branch not found"})
    )

    with (
        patch("services.github_validation.Github", github_cls),
        patch("services.github_validation.Auth") as mock_auth,
        patch("services.github_validation.GithubException", _GithubException),
        patch("services.github_validation.NotFoundException", _NotFoundException),
        patch("services.github_validation.ValidationException", _ValidationException),
        patch("services.github_app_auth.app_auth_configured", return_value=True),
        patch(
            "services.github_app_auth.get_installation_token_for", return_value=_PLATFORM_APP_TOKEN
        ),
    ):
        _passthrough_auth(mock_auth)
        svc = _make_svc()

        req = _RepoRequest(repo_url=_PUBLIC_REPO_URL, access_token=None, branch="release-1.2")
        with pytest.raises(ValueError, match="Branch 'release-1.2' not found"):
            await svc.validate_repository_access(req, tenant_id=PLATFORM_TEST_TENANT)

    anon.get_repo.assert_not_called()
