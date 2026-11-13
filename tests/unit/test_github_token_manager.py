"""Unit tests for GitHubTokenManager.

Verifies BYO user token store/retrieve via SecureTokenManager and
operation-based token selection. There is NO system/platform PAT.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from services.github_token_manager import GitHubTokenManager


def _settings(**overrides):
    defaults = {
        # A stale PAT field on the settings object must be ignored entirely.
        "system_github_token": "ghp_system_tok",
        "system_github_owner": "nonagenticai",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestGitHubTokenManager:
    @pytest.fixture
    def mgr(self):
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

    # -- System token --

    def test_get_system_owner(self, mgr):
        """System owner is returned from settings."""
        assert mgr.get_system_owner() == "nonagenticai"

    def test_no_system_token_exists(self, mgr):
        """There is no platform PAT: no accessor, no stored copy."""
        assert not hasattr(mgr, "get_system_token")
        assert not hasattr(mgr, "system_token")

    # -- User token retrieval --

    async def test_user_token_with_container_id(self, mgr):
        """User token lookup includes container_id in the key."""
        result = await mgr.get_user_token("user-1", container_id="ctr-1")
        assert result == "ghp_user_tok"
        mgr._secure_mock.retrieve_token.assert_awaited_once_with("user-user-1-ctr-1")

    async def test_user_token_without_container_id(self, mgr):
        """User token lookup without container_id uses just user_id."""
        result = await mgr.get_user_token("user-1")
        assert result == "ghp_user_tok"
        mgr._secure_mock.retrieve_token.assert_awaited_once_with("user-user-1")

    async def test_user_token_not_found_returns_none(self, mgr):
        """Missing user token returns None."""
        mgr._secure_mock.retrieve_token.return_value = None
        result = await mgr.get_user_token("user-missing")
        assert result is None

    async def test_user_token_decrypt_error_returns_none(self, mgr):
        """Decryption errors return None (caught internally)."""
        mgr._secure_mock.retrieve_token.side_effect = Exception("decrypt failed")
        result = await mgr.get_user_token("user-1")
        assert result is None

    # -- Operation-based token selection --

    @pytest.mark.parametrize("op", ["provision", "validate_private", "create_upstream_pr"])
    def test_user_operation_returns_user_token(self, mgr, op):
        """User operations return the provided user_token."""
        result = mgr.get_token_for_operation(op, user_token="ghp_user_abc")
        assert result == "ghp_user_abc"

    @pytest.mark.parametrize("op", ["provision", "validate_private", "create_upstream_pr"])
    def test_user_operation_without_token_raises(self, mgr, op):
        """User operations without user_token raise ValueError."""
        with pytest.raises(ValueError, match="requires user token"):
            mgr.get_token_for_operation(op)

    @pytest.mark.parametrize(
        "op", ["fork", "clone", "commit", "push", "github_actions", "some_weird_operation"]
    )
    def test_non_user_operation_is_refused_never_a_system_token(self, mgr, op):
        """Former system-token operations (and unknown ones) are refused: no PAT."""
        with pytest.raises(ValueError, match="no PAT credential"):
            mgr.get_token_for_operation(op)
        with pytest.raises(ValueError, match="no PAT credential"):
            mgr.get_token_for_operation(op, user_token="ghp_user_abc")

    # -- Custom settings --

    def test_custom_settings_applied(self):
        """Custom settings override the system owner."""
        custom = _settings(system_github_owner="custom-org")
        with (
            patch("services.github_token_manager.get_settings", return_value=custom),
            patch("services.github_token_manager.SecureTokenManager"),
        ):
            manager = GitHubTokenManager()
            assert manager.get_system_owner() == "custom-org"
