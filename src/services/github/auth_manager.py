"""Authentication and token management module.

This module provides GitHub authentication and token management including:
- GitHub token validation and management
- Authentication URL building
- Token-based API access control
- Secure credential handling
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class GitHubAuthManager:
    """Manages GitHub authentication and token operations."""

    def __init__(self) -> None:
        """Initialize GitHub authentication manager.

        ⛔ Holds NO credential: there is no platform personal access token
        (owner ruling 2026-09-30). Platform GitHub access is the GitHub App via
        ``services.github_integration.resolve_github_token``; the helpers below
        only operate on a token passed to them explicitly.
        """
        self.github_token: str | None = None

    def has_token(self) -> bool:
        """Check if GitHub token is available."""
        return bool(self.github_token)

    def get_token(self) -> str | None:
        """Get GitHub token for API access."""
        return self.github_token

    def build_authenticated_url(self, repo_url: str) -> str:
        """Build authenticated clone URL with token if available."""
        if not self.github_token:
            logger.info("No GitHub token available, using public clone")
            return repo_url

        if repo_url.startswith("git@"):
            # SSH URLs don't use token authentication
            logger.info("SSH URL detected, token authentication not supported")
            return repo_url

        # Convert to authenticated HTTPS URL
        try:
            parsed = urlparse(repo_url)
            authenticated_url = f"https://{self.github_token}@{parsed.netloc}{parsed.path}"
            logger.info("Using authenticated clone URL")
            return authenticated_url
        except (ValueError, AttributeError) as e:
            logger.warning("Failed to build authenticated URL: %s", e)
            return repo_url

    def validate_token_format(self, token: str | None = None) -> bool:
        """Validate GitHub token format."""
        check_token = token or self.github_token
        if not check_token:
            return False

        # GitHub personal access tokens should start with 'ghp_' (classic)
        # or 'github_pat_' (fine-grained)
        # GitHub app tokens start with 'ghs_'
        # OAuth tokens can vary but are typically 40 characters
        valid_prefixes = ["ghp_", "github_pat_", "ghs_"]

        if any(check_token.startswith(prefix) for prefix in valid_prefixes):
            return True

        # Legacy tokens are typically 40 character hex strings
        if len(check_token) == 40 and all(c in "0123456789abcdef" for c in check_token.lower()):
            return True

        return False

    def mask_token_for_logging(self, token: str | None = None) -> str:
        """Mask token for safe logging."""
        check_token = token or self.github_token
        if not check_token:
            return "None"

        if len(check_token) < 8:
            return "***"

        return f"{check_token[:4]}...{check_token[-4:]}"

    def is_authenticated_url(self, url: str) -> bool:
        """Check if URL contains authentication credentials."""
        try:
            parsed = urlparse(url)
            return bool(parsed.username or "@" in parsed.netloc)
        except Exception:
            return False

    def strip_auth_from_url(self, url: str) -> str:
        """Remove authentication credentials from URL for safe logging."""
        try:
            parsed = urlparse(url)
            if parsed.username:
                # Rebuild URL without credentials
                netloc = parsed.hostname or ""
                if parsed.port:
                    netloc += f":{parsed.port}"
                return f"{parsed.scheme}://{netloc}{parsed.path}"
            return url
        except Exception:
            return url
