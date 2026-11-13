"""Extended unit tests for GitHubTokenManager.

Covers: store_user_token metadata construction, check_token_rotation_needed,
cleanup_expired_tokens delegation, system op with missing token,
and the get_github_token_manager singleton.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from services.github_token_manager import GitHubTokenManager, get_github_token_manager


def _settings(**overrides):
    defaults = {
        # A stale PAT field on the settings object must be ignored entirely
        # (the system-token branch was removed 2026-09-30).
        "system_github_token": "ghp_system_tok",
        "system_github_owner": "nonagenticai",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def mgr():
    """Create a GitHubTokenManager with mocked dependencies."""
    with (
        patch(
            "services.github_token_manager.get_settings",
            return_value=_settings(),
        ),
        patch(
            "services.github_token_manager.SecureTokenManager",
        ) as mock_secure_cls,
    ):
        mock_secure = mock_secure_cls.return_value
        mock_secure.store_token = AsyncMock(return_value="stored-id")
        mock_secure.retrieve_token = AsyncMock(return_value="ghp_user_tok")
        mock_secure.check_rotation_needed = AsyncMock(return_value=[])
        mock_secure.cleanup_expired_tokens = AsyncMock(return_value=0)
        manager = GitHubTokenManager()
        manager._secure_mock = mock_secure
        yield manager


@pytest.fixture
def mgr_no_system_token():
    """Manager with no system token configured."""
    with (
        patch(
            "services.github_token_manager.get_settings",
            return_value=_settings(system_github_token=""),
        ),
        patch("services.github_token_manager.SecureTokenManager"),
    ):
        yield GitHubTokenManager()


# ---------------------------------------------------------------------------
# store_user_token
# ---------------------------------------------------------------------------


class TestStoreUserToken:
    async def test_store_without_container_id(self, mgr):
        """token_id should be 'user-{user_id}' when no container_id."""
        result = await mgr.store_user_token(user_id="alice", token="ghp_abc")
        assert result == "stored-id"
        call_args = mgr._secure_mock.store_token.call_args
        metadata_arg = call_args[0][1]
        assert metadata_arg.token_id == "user-alice"

    async def test_store_with_container_id(self, mgr):
        """token_id includes container_id when provided."""
        await mgr.store_user_token(user_id="bob", token="ghp_xyz", container_id="ctr-99")
        call_args = mgr._secure_mock.store_token.call_args
        metadata_arg = call_args[0][1]
        assert metadata_arg.token_id == "user-bob-ctr-99"

    async def test_store_metadata_fields_populated(self, mgr):
        """Metadata carries correct scope, rotation days, and nested metadata dict."""
        await mgr.store_user_token(
            user_id="carol", token="ghp_tok", container_id="ctr-5", rotation_days=60
        )
        call_args = mgr._secure_mock.store_token.call_args
        token_arg = call_args[0][0]
        metadata_arg = call_args[0][1]

        assert token_arg == "ghp_tok"

        from services.github.secure_token_manager import TokenScope

        assert metadata_arg.scope == TokenScope.READ_WRITE
        assert metadata_arg.rotation_interval_days == 60
        assert metadata_arg.metadata["user_id"] == "carol"
        assert metadata_arg.metadata["container_id"] == "ctr-5"
        assert metadata_arg.metadata["token_type"] == "user_github_token"
        assert "created_at" in metadata_arg.metadata

    async def test_store_returns_whatever_secure_manager_returns(self, mgr):
        """Return value proxies from secure_manager.store_token."""
        mgr._secure_mock.store_token.return_value = "custom-id-42"
        result = await mgr.store_user_token(user_id="dave", token="ghp_d")
        assert result == "custom-id-42"


# ---------------------------------------------------------------------------
# check_token_rotation_needed
# ---------------------------------------------------------------------------


class TestCheckTokenRotation:
    async def test_delegates_to_secure_manager(self, mgr):
        """Proxies to secure_manager.check_rotation_needed."""
        mgr._secure_mock.check_rotation_needed.return_value = ["tok-1", "tok-2"]
        result = await mgr.check_token_rotation_needed()
        assert result == ["tok-1", "tok-2"]
        mgr._secure_mock.check_rotation_needed.assert_awaited_once()

    async def test_returns_empty_when_none_needed(self, mgr):
        """Returns empty list when nothing needs rotation."""
        result = await mgr.check_token_rotation_needed()
        assert result == []


# ---------------------------------------------------------------------------
# cleanup_expired_tokens
# ---------------------------------------------------------------------------


class TestCleanupExpiredTokens:
    async def test_returns_zero_when_nothing_removed(self, mgr):
        """Returns 0 when nothing was cleaned up."""
        result = await mgr.cleanup_expired_tokens()
        assert result == 0

    async def test_returns_count_when_removed(self, mgr):
        """Returns count > 0 when tokens were removed."""
        mgr._secure_mock.cleanup_expired_tokens.return_value = 3
        result = await mgr.cleanup_expired_tokens()
        assert result == 3

    async def test_delegates_to_secure_manager(self, mgr):
        """Calls through to secure_manager.cleanup_expired_tokens."""
        mgr._secure_mock.cleanup_expired_tokens.return_value = 5
        await mgr.cleanup_expired_tokens()
        mgr._secure_mock.cleanup_expired_tokens.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_token_for_operation edge cases
# ---------------------------------------------------------------------------


class TestGetTokenForOperationEdgeCases:
    @pytest.mark.parametrize("op", ["provision", "validate_private", "create_upstream_pr"])
    def test_user_op_with_explicit_none_raises(self, mgr, op):
        """Explicitly passing None as user_token raises ValueError."""
        with pytest.raises(ValueError, match="requires user token"):
            mgr.get_token_for_operation(op, user_token=None)

    @pytest.mark.parametrize("op", ["fork", "clone", "commit", "push", "github_actions"])
    def test_system_op_raises_when_no_system_token(self, mgr_no_system_token, op):
        """System operations raise when system token is empty."""
        with pytest.raises(ValueError):
            mgr_no_system_token.get_token_for_operation(op)

    def test_unknown_op_raises_when_no_system_token(self, mgr_no_system_token):
        """Unknown operations raise when system token is empty."""
        with pytest.raises(ValueError):
            mgr_no_system_token.get_token_for_operation("unknown_op")


# ---------------------------------------------------------------------------
# get_github_token_manager singleton
# ---------------------------------------------------------------------------


class TestGetGitHubTokenManagerSingleton:
    def test_creates_and_reuses_instance(self):
        """get_github_token_manager returns the same singleton on repeated calls."""
        import services.github_token_manager as mod

        mod._token_manager = None
        with (
            patch(
                "services.github_token_manager.get_settings",
                return_value=_settings(),
            ),
            patch("services.github_token_manager.SecureTokenManager"),
        ):
            result = get_github_token_manager()
            assert isinstance(result, GitHubTokenManager)
            result2 = get_github_token_manager()
            assert result2 is result
        mod._token_manager = None

    def test_reuses_existing_instance(self):
        """If _token_manager is already set, returns it directly."""
        import services.github_token_manager as mod

        sentinel = object()
        mod._token_manager = sentinel
        try:
            assert get_github_token_manager() is sentinel
        finally:
            mod._token_manager = None
