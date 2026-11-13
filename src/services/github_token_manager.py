"""GitHub Token Manager for a tenant's OWN (BYO) GitHub tokens.

It stores and retrieves user-supplied tokens via the encrypted SecureTokenManager.
⛔ There is NO platform/system personal access token (owner ruling 2026-09-30):
platform GitHub access is the internal GitHub App only
(:func:`services.github_integration.resolve_github_token`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from config.settings import get_settings
from services.github.secure_token_manager import (
    SecureTokenManager,
    SecureTokenMetadata,
    TokenScope,
)

logger = logging.getLogger(__name__)


class GitHubTokenManager:
    """Manages GitHub tokens for multi-account workflow.

    Handles:
    - User tokens for private repo access (provisioning, PR creation)
    - Token storage via encrypted SecureTokenManager
    - Token lifecycle (rotation, expiration)
    """

    def __init__(self):
        """Initialize token manager with settings and secure storage."""
        self.settings = get_settings()
        self.system_owner = self.settings.system_github_owner
        self.secure_manager = SecureTokenManager()

        logger.info(f"GitHubTokenManager initialized with system owner: {self.system_owner}")

    def get_system_owner(self) -> str:
        """Get system GitHub account/organization name.

        Returns:
            System owner name (e.g., "nonagenticai")
        """
        return self.system_owner

    async def store_user_token(
        self,
        user_id: str,
        token: str,
        container_id: str | None = None,
        rotation_days: int = 90,
    ) -> str:
        """Store user's GitHub token with encryption.

        Args:
            user_id: User identifier (from auth context)
            token: User's GitHub personal access token
            container_id: Optional container ID for reference
            rotation_days: Days until token rotation recommended (default: 90)

        Returns:
            token_id: Reference ID for retrieving the token

        Example:
            token_id = await manager.store_user_token(
                user_id="user-123",
                token="ghp_...",
                container_id="container-456"
            )
        """
        # Create unique token ID
        token_id = f"user-{user_id}"
        if container_id:
            token_id = f"{token_id}-{container_id}"

        # Create metadata
        metadata = SecureTokenMetadata(
            token_id=token_id,
            scope=TokenScope.READ_WRITE,
            expires_at=datetime.now() + timedelta(days=rotation_days),
            rotation_interval_days=rotation_days,
            metadata={
                "user_id": user_id,
                "container_id": container_id,
                "token_type": "user_github_token",
                "created_at": datetime.now().isoformat(),
            },
        )

        # Store encrypted
        stored_id = await self.secure_manager.store_token(token, metadata)

        logger.info(
            f"Stored user token for user_id={user_id}, container_id={container_id}, token_id={stored_id}"
        )

        return stored_id

    async def get_user_token(self, user_id: str, container_id: str | None = None) -> str | None:
        """Retrieve user's GitHub token from secure storage.

        Args:
            user_id: User identifier
            container_id: Optional container ID for lookup

        Returns:
            Decrypted user GitHub token or None if not found

        Example:
            token = await manager.get_user_token(
                user_id="user-123",
                container_id="container-456"
            )
        """
        # Construct token ID
        token_id = f"user-{user_id}"
        if container_id:
            token_id = f"{token_id}-{container_id}"

        try:
            token = await self.secure_manager.retrieve_token(token_id)
            if token:
                logger.debug(f"Retrieved user token for token_id={token_id}")
            else:
                logger.warning(f"User token not found for token_id={token_id}")
            return token
        except Exception as e:
            logger.error(f"Error retrieving user token: {e}", exc_info=True)
            return None

    async def check_token_rotation_needed(self) -> list[str]:
        """Check which tokens need rotation.

        Returns:
            List of token_ids that need rotation
        """
        return await self.secure_manager.check_rotation_needed()

    async def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage.

        Returns:
            Number of tokens that were removed
        """
        removed = await self.secure_manager.cleanup_expired_tokens()
        if removed:
            logger.info(f"Cleaned up {removed} expired tokens")
        return removed

    def get_token_for_operation(self, operation: str, user_token: str | None = None) -> str:
        """Return the caller's own token for a user operation.

        ⛔ There is no system-token branch any more: operations that used the
        platform PAT (fork / clone / commit / push / github_actions) must go
        through the GitHub App, so any non-user operation is refused.

        Raises:
            ValueError: If the operation is not a user operation, or no user
                token was provided.
        """
        user_operations = ["provision", "validate_private", "create_upstream_pr"]
        if operation not in user_operations:
            raise ValueError(
                f"Operation '{operation}' has no PAT credential; platform GitHub access "
                "is the GitHub App only"
            )
        if not user_token:
            raise ValueError(f"Operation '{operation}' requires user token but none provided")
        logger.debug(f"Using user token for operation: {operation}")
        return user_token


# Global instance
_token_manager: GitHubTokenManager | None = None


def get_github_token_manager() -> GitHubTokenManager:
    """Get or create global GitHubTokenManager instance.

    Returns:
        Singleton GitHubTokenManager instance
    """
    global _token_manager
    if _token_manager is None:
        _token_manager = GitHubTokenManager()
    return _token_manager
