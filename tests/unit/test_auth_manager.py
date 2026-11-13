"""Tests for services/github/auth_manager.py"""

from __future__ import annotations

from services.github.auth_manager import GitHubAuthManager


class TestGitHubAuthManager:
    """Tests for GitHubAuthManager."""

    def test_token_parsing_and_validation(self):
        """Test validate_token_format with various token types."""
        mgr = GitHubAuthManager()

        # Classic personal access token
        assert mgr.validate_token_format("ghp_abcdefghijklmnopqrstuvwxyz123456") is True

        # Fine-grained token
        assert mgr.validate_token_format("github_pat_someLongTokenValue") is True

        # GitHub App installation token
        assert mgr.validate_token_format("ghs_appInstallToken12345") is True

        # Legacy 40-char hex token
        assert mgr.validate_token_format("a" * 40) is True

        # Invalid tokens
        assert mgr.validate_token_format("short") is False
        assert mgr.validate_token_format("") is False
        assert mgr.validate_token_format(None) is False

    def test_auth_object_creation_and_url_building(self):
        """Test build_authenticated_url for HTTPS and SSH URLs."""
        mgr = GitHubAuthManager()
        # The manager holds no credential of its own; the helpers act on a token
        # handed to them explicitly.
        mgr.github_token = "test-token"

        # HTTPS URL with token present
        auth_url = mgr.build_authenticated_url("https://github.com/owner/repo.git")
        assert "test-token@" in auth_url
        assert auth_url.startswith("https://")

        # SSH URL should be returned unchanged
        ssh_url = "git@github.com:owner/repo.git"
        assert mgr.build_authenticated_url(ssh_url) == ssh_url

        # URL stripping and checking
        assert mgr.is_authenticated_url(auth_url) is True
        stripped = mgr.strip_auth_from_url(auth_url)
        assert "test-token" not in stripped

        # mask_token_for_logging
        masked = mgr.mask_token_for_logging("ghp_abcdefghijk")
        assert masked.startswith("ghp_")
        assert masked.endswith("hijk")
        assert "..." in masked

    def test_missing_credentials_handling(self, monkeypatch):
        """A fresh manager has NO token, even with every legacy PAT var set.

        There is no platform personal access token (owner ruling 2026-09-30); the
        manager must not pick one up from the environment or settings.
        """
        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_ENV_PAT_must_not_be_read")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_ENV_PAT_must_not_be_read")
        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_TOKEN", "ghp_ENV_PAT_must_not_be_read")
        mgr = GitHubAuthManager()

        assert mgr.has_token() is False
        assert mgr.get_token() is None
        assert mgr.validate_token_format() is False

        # build_authenticated_url returns original URL when no token
        url = "https://github.com/owner/repo.git"
        assert mgr.build_authenticated_url(url) == url

        # mask_token_for_logging with no token
        assert mgr.mask_token_for_logging() == "None"
