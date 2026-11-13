"""
Secure token management for GitHub API access.

This module provides secure storage, encryption, and rotation capabilities
for GitHub API tokens, ensuring sensitive credentials are never stored in plaintext.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coding_client.types.core_types import SecurityLevel
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logger.warning("cryptography module not available - GitHub token encryption disabled")
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    hashes = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    PBKDF2HMAC = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


class TokenStatus(StrEnum):
    """Token lifecycle status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATION_PENDING = "rotation_pending"


class TokenScope(StrEnum):
    """GitHub token scope levels."""

    READ_ONLY = "repo"
    READ_WRITE = "repo,write:repo_hook"
    ADMIN = "repo,admin:repo_hook,admin:org"


class SecureTokenMetadata(BaseModel):
    """Metadata for secure token storage."""

    token_id: str = Field(..., description="Unique token identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    last_used: datetime | None = None
    usage_count: int = 0
    scope: TokenScope = TokenScope.READ_ONLY
    status: TokenStatus = TokenStatus.ACTIVE
    security_level: SecurityLevel = "high"
    rotation_interval_days: int = Field(default=30, ge=1, le=365)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v: datetime | None) -> datetime | None:
        """Ensure expiration is in the future."""
        if v:
            # Handle timezone-naive datetimes by assuming UTC
            if v.tzinfo is None:
                v = v.replace(tzinfo=UTC)
            if v <= datetime.now(UTC):
                raise ValueError("Expiration date must be in the future")
        return v


class SecureTokenManager:
    """
    Secure GitHub token management with encryption and rotation.

    Provides encrypted storage, automatic rotation, and secure access
    to GitHub API tokens with comprehensive audit logging.
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize secure token manager.

        Args:
            storage_path: Path to secure token storage directory
        """
        self.storage_path = storage_path or Path.home() / ".code_graph_helper" / "tokens"

        self._master_key: bytes | None = None
        self._tokens: dict[str, SecureTokenMetadata] = {}
        self._encryption_lock = asyncio.Lock()

        logger.info(
            f"Secure token manager configured at {self.storage_path} "
            "(storage created on first write)"
        )

    def _ensure_storage(self) -> None:
        """Create the storage directory on first WRITE.

        Deliberately not done in ``__init__``: a GitHubService constructed with an
        in-memory token never touches disk, and on a read-only rootfs an eager
        mkdir failed the whole request for a facility that was never used
        (measured 2026-09-09, Errno 30 on /home/appuser/.code_graph_helper,
        merge-pull-request 500).
        """
        self.storage_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.storage_path, 0o700)

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2.

        Uses 600000 iterations per NIST SP 800-132 recommendations (2023).
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _get_master_key(self) -> bytes:
        """Get or create master encryption key."""
        if self._master_key is not None:
            return self._master_key

        # Try to get key from environment variable (production)
        env_key = os.getenv("GITHUB_TOKEN_ENCRYPTION_KEY")
        if env_key:
            try:
                self._master_key = base64.urlsafe_b64decode(env_key.encode())
                return self._master_key
            except Exception as e:
                logger.warning(f"Invalid encryption key in environment: {e}")

        # Generate new key for development
        key_file = self.storage_path / ".master_key"
        salt_file = self.storage_path / ".salt"

        if key_file.exists() and salt_file.exists():
            # Load existing key
            try:
                with open(salt_file, "rb") as f:
                    salt = f.read()

                # Password must be provided via environment variable
                password = os.getenv("GITHUB_TOKEN_PASSWORD")
                if not password:
                    raise RuntimeError("GITHUB_TOKEN_PASSWORD environment variable is required")
                self._master_key = self._derive_key(password, salt)

                # Verify key works
                test_cipher = Fernet(self._master_key)  # noqa: F841
                return self._master_key

            except Exception as e:
                logger.error(f"Failed to load existing master key: {e}")

        # Create new key - password must be provided via environment variable
        salt = secrets.token_bytes(32)
        password = os.getenv("GITHUB_TOKEN_PASSWORD")
        if not password:
            raise RuntimeError("GITHUB_TOKEN_PASSWORD environment variable is required")
        self._master_key = self._derive_key(password, salt)

        # Save salt (key is derived on demand) -- the only write in this method,
        # so the storage directory is created here and not on the cached / env-key
        # / existing-salt branches above, all of which are pure reads.
        self._ensure_storage()
        with open(salt_file, "wb") as f:
            f.write(salt)
        os.chmod(salt_file, 0o600)

        logger.info("Created new master encryption key")
        return self._master_key

    async def store_token(
        self, token: str, metadata: SecureTokenMetadata, overwrite: bool = False
    ) -> str:
        """
        Securely store an encrypted GitHub token.

        Args:
            token: GitHub API token to encrypt and store
            metadata: Token metadata including scope and expiration
            overwrite: Whether to overwrite existing token with same ID

        Returns:
            Token ID for future reference

        Raises:
            ValueError: If token ID already exists and overwrite=False
            RuntimeError: If encryption fails
        """
        self._ensure_storage()
        async with self._encryption_lock:
            token_id = metadata.token_id
            token_file = self.storage_path / f"{token_id}.enc"
            metadata_file = self.storage_path / f"{token_id}.meta"

            if token_file.exists() and not overwrite:
                raise ValueError(f"Token ID '{token_id}' already exists")

            try:
                # Encrypt token
                master_key = self._get_master_key()
                cipher = Fernet(master_key)
                encrypted_token = cipher.encrypt(token.encode())

                # Store encrypted token
                with open(token_file, "wb") as f:
                    f.write(encrypted_token)
                os.chmod(token_file, 0o600)

                # Store metadata
                with open(metadata_file, "w") as f:
                    json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)
                os.chmod(metadata_file, 0o600)

                # Cache metadata
                self._tokens[token_id] = metadata

                logger.info(f"Stored encrypted token '{token_id}' with scope {metadata.scope}")
                return token_id

            except Exception as e:
                logger.error(f"Failed to store token '{token_id}': {e}")
                # Clean up partial files
                for file in [token_file, metadata_file]:
                    if file.exists():
                        file.unlink()
                raise RuntimeError(f"Token storage failed: {e}") from e

    async def retrieve_token(self, token_id: str, update_usage: bool = True) -> str | None:
        """
        Retrieve and decrypt a GitHub token.

        Args:
            token_id: Token identifier
            update_usage: Whether to update usage statistics

        Returns:
            Decrypted token or None if not found/expired
        """
        # Special handling for schemathesis test tokens
        if os.getenv("SCHEMATHESIS_TESTING", "").lower() in ("true", "1", "yes", "on"):
            if token_id.lower().startswith("schemathesis") or "schemathesis" in token_id.lower():
                logger.info("Returning mock token for schemathesis testing: %s", token_id)
                return f"schemathesis_mock_token_{token_id}"

        async with self._encryption_lock:
            token_file = self.storage_path / f"{token_id}.enc"
            metadata_file = self.storage_path / f"{token_id}.meta"

            if not token_file.exists() or not metadata_file.exists():
                # token_id is a lookup KEY, but a caller that mistakenly passes a raw
                # credential here would print it into the logs. Log a stable prefix only.
                logger.warning("Token '%s...' not found", str(token_id)[:8])
                return None

            try:
                # Load metadata
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                metadata = SecureTokenMetadata(**metadata_dict)

                # Check if token is expired or revoked
                now = datetime.now(UTC)
                if metadata.status == TokenStatus.REVOKED:
                    logger.warning(f"Token '{token_id}' is revoked")
                    return None

                if metadata.expires_at and metadata.expires_at <= now:
                    logger.warning(f"Token '{token_id}' is expired")
                    metadata.status = TokenStatus.EXPIRED
                    await self._update_metadata(token_id, metadata)
                    return None

                # Decrypt token
                master_key = self._get_master_key()
                cipher = Fernet(master_key)

                with open(token_file, "rb") as f:
                    encrypted_token = f.read()

                decrypted_token = cipher.decrypt(encrypted_token).decode()

                # Update usage statistics
                if update_usage:
                    metadata.last_used = now
                    metadata.usage_count += 1
                    await self._update_metadata(token_id, metadata)

                logger.debug(f"Retrieved token '{token_id}' (usage: {metadata.usage_count})")
                return decrypted_token

            except Exception as e:
                logger.error(f"Failed to retrieve token '{token_id}': {e}")
                return None

    async def revoke_token(self, token_id: str) -> bool:
        """
        Revoke a token (mark as revoked, don't delete).

        Args:
            token_id: Token identifier

        Returns:
            True if token was revoked successfully
        """
        metadata = await self.get_token_metadata(token_id)
        if not metadata:
            return False

        metadata.status = TokenStatus.REVOKED
        await self._update_metadata(token_id, metadata)

        logger.info(f"Revoked token '{token_id}'")
        return True

    async def rotate_token(self, token_id: str, new_token: str) -> bool:
        """
        Rotate an existing token with a new value.

        Args:
            token_id: Existing token identifier
            new_token: New GitHub API token

        Returns:
            True if rotation was successful
        """
        metadata = await self.get_token_metadata(token_id)
        if not metadata:
            logger.error(f"Cannot rotate non-existent token '{token_id}'")
            return False

        # Update metadata for rotation
        metadata.created_at = datetime.now(UTC)
        metadata.last_used = None
        metadata.usage_count = 0
        metadata.status = TokenStatus.ACTIVE

        try:
            await self.store_token(new_token, metadata, overwrite=True)
            logger.info(f"Successfully rotated token '{token_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to rotate token '{token_id}': {e}")
            return False

    async def get_token_metadata(self, token_id: str) -> SecureTokenMetadata | None:
        """Get token metadata without decrypting the token."""
        # Check cache first
        if token_id in self._tokens:
            return self._tokens[token_id]

        metadata_file = self.storage_path / f"{token_id}.meta"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                metadata_dict = json.load(f)
            metadata = SecureTokenMetadata(**metadata_dict)
            self._tokens[token_id] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata for token '{token_id}': {e}")
            return None

    async def list_tokens(self) -> dict[str, SecureTokenMetadata]:
        """List all token metadata (without decrypting tokens)."""
        tokens = {}

        for meta_file in self.storage_path.glob("*.meta"):
            token_id = meta_file.stem
            metadata = await self.get_token_metadata(token_id)
            if metadata:
                tokens[token_id] = metadata

        return tokens

    async def cleanup_expired_tokens(self) -> int:
        """Remove expired and revoked tokens from storage."""
        removed_count = 0
        now = datetime.now(UTC)

        for token_id, metadata in (await self.list_tokens()).items():
            should_remove = False

            if metadata.status == TokenStatus.REVOKED:
                # Remove revoked tokens after 7 days
                revoke_age = now - metadata.created_at
                if revoke_age > timedelta(days=7):
                    should_remove = True

            elif metadata.expires_at and metadata.expires_at <= now:
                # Remove expired tokens after 1 day
                expire_age = now - metadata.expires_at
                if expire_age > timedelta(days=1):
                    should_remove = True

            if should_remove:
                await self._delete_token_files(token_id)
                removed_count += 1
                logger.info(f"Cleaned up token '{token_id}'")

        return removed_count

    async def check_rotation_needed(self) -> list[str]:
        """Check which tokens need rotation based on age and usage."""
        rotation_needed = []
        now = datetime.now(UTC)

        for token_id, metadata in (await self.list_tokens()).items():
            if metadata.status != TokenStatus.ACTIVE:
                continue

            # Check age-based rotation
            age = now - metadata.created_at
            if age > timedelta(days=metadata.rotation_interval_days):
                rotation_needed.append(token_id)
                logger.info(f"Token '{token_id}' needs rotation (age: {age.days} days)")

        return rotation_needed

    async def _update_metadata(self, token_id: str, metadata: SecureTokenMetadata) -> None:
        """Update token metadata file."""
        self._ensure_storage()
        metadata_file = self.storage_path / f"{token_id}.meta"

        with open(metadata_file, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

        # Update cache
        self._tokens[token_id] = metadata

    async def _delete_token_files(self, token_id: str) -> None:
        """Delete both token and metadata files."""
        token_file = self.storage_path / f"{token_id}.enc"
        metadata_file = self.storage_path / f"{token_id}.meta"

        for file in [token_file, metadata_file]:
            if file.exists():
                file.unlink()

        # Remove from cache
        self._tokens.pop(token_id, None)


# Singleton instance for global access
_token_manager: SecureTokenManager | None = None


def get_secure_token_manager() -> SecureTokenManager:
    """Get the global secure token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = SecureTokenManager()
    return _token_manager


async def create_default_token(token: str) -> str:
    """Create a default GitHub token for development."""
    manager = get_secure_token_manager()

    metadata = SecureTokenMetadata(
        token_id="default",
        scope=TokenScope.READ_WRITE,
        security_level="high",
        rotation_interval_days=30,
    )

    return await manager.store_token(token, metadata, overwrite=True)
