"""Extended tests for services/github/auth_manager.py — covering uncovered lines.

Uncovered lines targeted:
  53-55  — build_authenticated_url: ValueError/AttributeError during URL parsing
  85     — mask_token_for_logging: token shorter than 8 chars
  94-95  — is_authenticated_url: Exception during URL parsing
  105    — strip_auth_from_url: URL with port in netloc
  107-109 — strip_auth_from_url: Exception during URL parsing
"""

from __future__ import annotations

from unittest.mock import patch

from services.github.auth_manager import GitHubAuthManager


class TestBuildAuthenticatedUrlParseError:
    def test_build_authenticated_url_malformed_url(self):
        """Lines 53-55: ValueError/AttributeError during urlparse => return original URL."""
        mgr = GitHubAuthManager()
        # Force a failure in urlparse by patching it
        with patch("services.github.auth_manager.urlparse", side_effect=ValueError("bad url")):
            result = mgr.build_authenticated_url("https://github.com/owner/repo")
            assert result == "https://github.com/owner/repo"

    def test_build_authenticated_url_attribute_error(self):
        """Lines 53-55: AttributeError => return original URL."""
        mgr = GitHubAuthManager()
        with patch("services.github.auth_manager.urlparse", side_effect=AttributeError("no attr")):
            result = mgr.build_authenticated_url("https://github.com/owner/repo")
            assert result == "https://github.com/owner/repo"


class TestMaskTokenShort:
    def test_mask_token_short_token(self):
        """Line 85: token shorter than 8 chars => returns '***'."""
        mgr = GitHubAuthManager()
        assert mgr.mask_token_for_logging("abc") == "***"
        assert mgr.mask_token_for_logging("1234567") == "***"

    def test_mask_token_exactly_8_chars(self):
        """Token of exactly 8 chars: should show first 4 and last 4."""
        mgr = GitHubAuthManager()
        result = mgr.mask_token_for_logging("12345678")
        assert result == "1234...5678"


class TestIsAuthenticatedUrlException:
    def test_is_authenticated_url_exception(self):
        """Lines 94-95: Exception during parsing => return False."""
        mgr = GitHubAuthManager()
        with patch("services.github.auth_manager.urlparse", side_effect=Exception("boom")):
            assert mgr.is_authenticated_url("https://user:pass@github.com/o/r") is False


class TestStripAuthFromUrlWithPort:
    def test_strip_auth_url_with_port(self):
        """Line 105: URL has username and port."""
        mgr = GitHubAuthManager()
        url = "https://token123@github.com:8443/owner/repo"
        result = mgr.strip_auth_from_url(url)
        assert "token123" not in result
        assert "8443" in result
        assert "/owner/repo" in result

    def test_strip_auth_url_no_port(self):
        """URL with username but no explicit port."""
        mgr = GitHubAuthManager()
        url = "https://token123@github.com/owner/repo"
        result = mgr.strip_auth_from_url(url)
        assert "token123" not in result
        assert "github.com" in result
        assert "/owner/repo" in result


class TestStripAuthFromUrlException:
    def test_strip_auth_exception(self):
        """Lines 108-109: Exception during parsing => return original URL."""
        mgr = GitHubAuthManager()
        with patch("services.github.auth_manager.urlparse", side_effect=Exception("broken")):
            result = mgr.strip_auth_from_url("https://token@github.com/o/r")
            assert result == "https://token@github.com/o/r"


class TestStripAuthNoCredentials:
    def test_strip_auth_no_credentials(self):
        """Line 107: URL without credentials => return as-is."""
        mgr = GitHubAuthManager()
        url = "https://github.com/owner/repo"
        assert mgr.strip_auth_from_url(url) == url
