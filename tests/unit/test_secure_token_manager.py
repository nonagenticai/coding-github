"""Tests for services/github/secure_token_manager.py

Verifies encryption/decryption round-trips, token lifecycle (store, retrieve,
revoke, rotate, expire), file permission enforcement, and error handling.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from services.github.secure_token_manager import (
    SecureTokenManager,
    SecureTokenMetadata,
    TokenScope,
    TokenStatus,
)


@pytest.fixture
def token_manager(tmp_path):
    """Create a token manager with a temp storage directory."""
    with patch.dict(os.environ, {"GITHUB_TOKEN_PASSWORD": "test-password-123"}):
        mgr = SecureTokenManager(storage_path=tmp_path / "tokens")
        yield mgr


@pytest.fixture
def sample_metadata():
    """Create sample token metadata."""
    return SecureTokenMetadata(
        token_id="test-token-1",
        scope=TokenScope.READ_ONLY,
        security_level="high",
        rotation_interval_days=30,
        expires_at=datetime.now(UTC) + timedelta(days=60),
    )


class TestStoreToken:
    """Token storage creates encrypted files and rejects duplicates."""

    async def test_store_writes_encrypted_file_and_metadata(self, token_manager, sample_metadata):
        """Stored token is encrypted on disk -- plaintext must not appear in .enc file."""
        plaintext = "ghp_abcdefghijklmnop"
        token_id = await token_manager.store_token(plaintext, sample_metadata)

        assert token_id == "test-token-1"
        enc_file = token_manager.storage_path / "test-token-1.enc"
        meta_file = token_manager.storage_path / "test-token-1.meta"
        assert enc_file.exists()
        assert meta_file.exists()

        # Encrypted content must NOT contain the plaintext
        enc_bytes = enc_file.read_bytes()
        assert plaintext.encode() not in enc_bytes

        # Metadata JSON must contain the token_id and scope
        with open(meta_file) as f:
            meta_data = json.load(f)
        assert meta_data["token_id"] == "test-token-1"
        assert meta_data["scope"] == TokenScope.READ_ONLY

    async def test_store_duplicate_rejected(self, token_manager, sample_metadata):
        """Storing the same token_id twice without overwrite raises ValueError."""
        await token_manager.store_token("ghp_first", sample_metadata)
        with pytest.raises(ValueError, match="already exists"):
            await token_manager.store_token("ghp_second", sample_metadata)

    async def test_store_duplicate_with_overwrite_succeeds(self, token_manager, sample_metadata):
        """Storing the same token_id with overwrite=True replaces the value."""
        await token_manager.store_token("ghp_first", sample_metadata)
        await token_manager.store_token("ghp_second", sample_metadata, overwrite=True)
        result = await token_manager.retrieve_token("test-token-1")
        assert result == "ghp_second"

    async def test_store_sets_secure_file_permissions(self, token_manager, sample_metadata):
        """Encrypted and metadata files should have 0o600 permissions."""
        await token_manager.store_token("ghp_perms", sample_metadata)
        enc_file = token_manager.storage_path / "test-token-1.enc"
        meta_file = token_manager.storage_path / "test-token-1.meta"
        assert oct(enc_file.stat().st_mode)[-3:] == "600"
        assert oct(meta_file.stat().st_mode)[-3:] == "600"


class TestRetrieveToken:
    """Token retrieval decrypts correctly and handles missing/expired/revoked tokens."""

    async def test_roundtrip_decrypt(self, token_manager, sample_metadata):
        """Store and retrieve returns the exact plaintext."""
        plaintext = "ghp_secret_token_value"
        await token_manager.store_token(plaintext, sample_metadata)
        result = await token_manager.retrieve_token("test-token-1")
        assert result == plaintext

    @pytest.mark.parametrize(
        "plaintext",
        [
            "ghp_short",
            "ghp_" + "x" * 200,  # long token
            "ghp_special_!@#$%^&*()",
            "ghp_unicode_\u00e9\u00e8\u00ea",
        ],
        ids=["short", "long", "special-chars", "unicode"],
    )
    async def test_roundtrip_various_tokens(self, token_manager, plaintext):
        """Encryption/decryption preserves various token formats."""
        meta = SecureTokenMetadata(
            token_id="varied-token",
            scope=TokenScope.READ_ONLY,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token(plaintext, meta)
        result = await token_manager.retrieve_token("varied-token")
        assert result == plaintext

    async def test_retrieve_nonexistent_returns_none(self, token_manager):
        """Retrieving a token that was never stored returns None."""
        result = await token_manager.retrieve_token("does-not-exist")
        assert result is None

    async def test_retrieve_updates_usage_count(self, token_manager, sample_metadata):
        """Retrieving a token increments usage_count in metadata."""
        await token_manager.store_token("ghp_usage", sample_metadata)
        await token_manager.retrieve_token("test-token-1")
        await token_manager.retrieve_token("test-token-1")
        meta = await token_manager.get_token_metadata("test-token-1")
        assert meta.usage_count == 2
        assert meta.last_used is not None

    async def test_retrieve_without_usage_update(self, token_manager, sample_metadata):
        """Retrieving with update_usage=False does not change metadata."""
        await token_manager.store_token("ghp_noupdate", sample_metadata)
        await token_manager.retrieve_token("test-token-1", update_usage=False)
        meta = await token_manager.get_token_metadata("test-token-1")
        assert meta.usage_count == 0

    async def test_expired_token_returns_none(self, token_manager):
        """Expired tokens cannot be retrieved and get their status updated."""
        metadata = SecureTokenMetadata(
            token_id="expiring-token",
            scope=TokenScope.READ_ONLY,
            security_level="high",
            rotation_interval_days=1,
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        await token_manager.store_token("ghp_expiring", metadata)

        # Manually backdate expiry in the metadata file
        meta_file = token_manager.storage_path / "expiring-token.meta"
        with open(meta_file) as f:
            data = json.load(f)
        data["expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        with open(meta_file, "w") as f:
            json.dump(data, f)
        token_manager._tokens.pop("expiring-token", None)

        result = await token_manager.retrieve_token("expiring-token")
        assert result is None


class TestRevokeToken:
    """Token revocation marks the token and blocks retrieval."""

    async def test_revoke_blocks_retrieval(self, token_manager, sample_metadata):
        """Revoked token returns None on retrieval."""
        await token_manager.store_token("ghp_to_revoke", sample_metadata)
        success = await token_manager.revoke_token("test-token-1")
        assert success is True

        result = await token_manager.retrieve_token("test-token-1")
        assert result is None

    async def test_revoke_updates_status_in_metadata(self, token_manager, sample_metadata):
        """Revocation sets status to REVOKED in persisted metadata."""
        await token_manager.store_token("ghp_rev", sample_metadata)
        await token_manager.revoke_token("test-token-1")

        meta_file = token_manager.storage_path / "test-token-1.meta"
        with open(meta_file) as f:
            data = json.load(f)
        assert data["status"] == TokenStatus.REVOKED

    async def test_revoke_nonexistent_returns_false(self, token_manager):
        """Revoking a non-existent token returns False."""
        result = await token_manager.revoke_token("no-such-token")
        assert result is False


class TestRotateToken:
    """Token rotation replaces old token with new encrypted value."""

    async def test_rotate_replaces_token_value(self, token_manager, sample_metadata):
        """After rotation, retrieving returns the new token."""
        await token_manager.store_token("ghp_old_token", sample_metadata)
        success = await token_manager.rotate_token("test-token-1", "ghp_new_token")
        assert success is True

        result = await token_manager.retrieve_token("test-token-1")
        assert result == "ghp_new_token"

    async def test_rotate_resets_usage_count(self, token_manager, sample_metadata):
        """Rotation resets usage_count and last_used."""
        await token_manager.store_token("ghp_old", sample_metadata)
        await token_manager.retrieve_token("test-token-1")  # increment usage
        await token_manager.rotate_token("test-token-1", "ghp_rotated")

        meta = await token_manager.get_token_metadata("test-token-1")
        assert meta.usage_count == 0
        assert meta.status == TokenStatus.ACTIVE

    async def test_rotate_nonexistent_returns_false(self, token_manager):
        """Rotating a non-existent token returns False."""
        result = await token_manager.rotate_token("missing-token", "ghp_new")
        assert result is False

    async def test_rotate_storage_failure_returns_false(self, token_manager, sample_metadata):
        """If store_token fails during rotation, returns False."""
        await token_manager.store_token("ghp_old", sample_metadata)
        with patch.object(
            token_manager,
            "store_token",
            side_effect=RuntimeError("disk full"),
        ):
            result = await token_manager.rotate_token("test-token-1", "ghp_new")
            assert result is False


class TestListAndCleanupTokens:
    """Token listing and cleanup of expired/revoked tokens."""

    async def test_list_tokens_empty(self, token_manager):
        """Empty storage returns empty dict."""
        tokens = await token_manager.list_tokens()
        assert tokens == {}

    async def test_list_tokens_returns_all_stored(self, token_manager):
        """All stored tokens appear in the listing."""
        for i in range(3):
            meta = SecureTokenMetadata(
                token_id=f"tok-{i}",
                scope=TokenScope.READ_ONLY,
                expires_at=datetime.now(UTC) + timedelta(days=60),
            )
            await token_manager.store_token(f"ghp_val_{i}", meta)

        tokens = await token_manager.list_tokens()
        assert len(tokens) == 3
        assert all(f"tok-{i}" in tokens for i in range(3))

    async def test_cleanup_removes_old_revoked_tokens(self, token_manager):
        """Revoked tokens older than 7 days are cleaned up."""
        meta = SecureTokenMetadata(
            token_id="old-revoked",
            scope=TokenScope.READ_ONLY,
            status=TokenStatus.REVOKED,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_revoked", meta)

        # Backdate created_at to 10 days ago
        meta_file = token_manager.storage_path / "old-revoked.meta"
        with open(meta_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        data["status"] = "revoked"
        with open(meta_file, "w") as f:
            json.dump(data, f)
        token_manager._tokens.pop("old-revoked", None)

        removed = await token_manager.cleanup_expired_tokens()
        assert removed == 1
        assert not (token_manager.storage_path / "old-revoked.enc").exists()

    async def test_cleanup_keeps_recent_revoked(self, token_manager):
        """Recently revoked tokens (< 7 days) are kept."""
        meta = SecureTokenMetadata(
            token_id="recent-revoked",
            scope=TokenScope.READ_ONLY,
            status=TokenStatus.REVOKED,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_rr", meta)

        meta_file = token_manager.storage_path / "recent-revoked.meta"
        with open(meta_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        data["status"] = "revoked"
        with open(meta_file, "w") as f:
            json.dump(data, f)
        token_manager._tokens.pop("recent-revoked", None)

        removed = await token_manager.cleanup_expired_tokens()
        assert removed == 0
        assert (token_manager.storage_path / "recent-revoked.enc").exists()

    async def test_cleanup_keeps_active_tokens(self, token_manager):
        """Active non-expired tokens are not cleaned up."""
        meta = SecureTokenMetadata(
            token_id="still-good",
            scope=TokenScope.READ_ONLY,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_active", meta)
        removed = await token_manager.cleanup_expired_tokens()
        assert removed == 0


class TestCheckRotationNeeded:
    """Rotation checks identify tokens past their rotation interval."""

    async def test_old_token_flagged_for_rotation(self, token_manager):
        """Token older than rotation_interval_days is flagged."""
        meta = SecureTokenMetadata(
            token_id="old-active",
            scope=TokenScope.READ_ONLY,
            rotation_interval_days=7,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_old_active", meta)

        meta_file = token_manager.storage_path / "old-active.meta"
        with open(meta_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        with open(meta_file, "w") as f:
            json.dump(data, f)
        token_manager._tokens.pop("old-active", None)

        needed = await token_manager.check_rotation_needed()
        assert "old-active" in needed

    async def test_fresh_token_not_flagged(self, token_manager):
        """Recently created token should not need rotation."""
        meta = SecureTokenMetadata(
            token_id="fresh",
            scope=TokenScope.READ_ONLY,
            rotation_interval_days=30,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_fresh", meta)
        needed = await token_manager.check_rotation_needed()
        assert "fresh" not in needed

    async def test_rotation_skips_non_active_tokens(self, token_manager):
        """Non-active tokens are not considered for rotation."""
        meta = SecureTokenMetadata(
            token_id="revoked-skip",
            scope=TokenScope.READ_ONLY,
            status=TokenStatus.REVOKED,
            rotation_interval_days=1,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        await token_manager.store_token("ghp_skip", meta)

        meta_file = token_manager.storage_path / "revoked-skip.meta"
        with open(meta_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        data["status"] = "revoked"
        with open(meta_file, "w") as f:
            json.dump(data, f)
        token_manager._tokens.pop("revoked-skip", None)

        needed = await token_manager.check_rotation_needed()
        assert "revoked-skip" not in needed


class TestMetadataManagement:
    """Metadata loading from cache and file."""

    async def test_metadata_loaded_from_file_when_not_cached(self, token_manager, sample_metadata):
        """Metadata not in cache is loaded from the .meta file."""
        await token_manager.store_token("ghp_test", sample_metadata)
        token_manager._tokens.clear()

        meta = await token_manager.get_token_metadata("test-token-1")
        assert meta.token_id == "test-token-1"
        assert "test-token-1" in token_manager._tokens

    async def test_metadata_missing_file_returns_none(self, token_manager):
        """Non-existent metadata file returns None."""
        result = await token_manager.get_token_metadata("ghost")
        assert result is None

    async def test_metadata_corrupted_file_returns_none(self, token_manager):
        """Corrupted metadata file returns None gracefully."""
        # Storage is created lazily on first write, so this test -- which writes a
        # file directly rather than going through store_token -- must create it.
        token_manager.storage_path.mkdir(parents=True, exist_ok=True)
        bad_file = token_manager.storage_path / "corrupt.meta"
        bad_file.write_text("{invalid json!!")
        result = await token_manager.get_token_metadata("corrupt")
        assert result is None


class TestDeleteTokenFiles:
    """File deletion removes .enc, .meta and cache entry."""

    async def test_delete_removes_files_and_cache(self, token_manager, sample_metadata):
        """_delete_token_files removes all artifacts."""
        await token_manager.store_token("ghp_del", sample_metadata)
        assert "test-token-1" in token_manager._tokens

        await token_manager._delete_token_files("test-token-1")

        assert not (token_manager.storage_path / "test-token-1.enc").exists()
        assert not (token_manager.storage_path / "test-token-1.meta").exists()
        assert "test-token-1" not in token_manager._tokens

    async def test_delete_nonexistent_does_not_raise(self, token_manager):
        """Deleting non-existent files does not raise."""
        await token_manager._delete_token_files("nonexistent-token")


class TestLazyStorageCreation:
    """Construction must do no filesystem I/O; storage appears on first write.

    The pod runs on a read-only root filesystem. An eager ``mkdir`` in
    ``__init__`` raised ``OSError(30)`` and failed every merge request, even
    though the merge path carries its token in memory and never reads the store.
    """

    def test_init_performs_no_filesystem_io(self, tmp_path, monkeypatch):
        """__init__ must not mkdir/chmod -- a read-only rootfs must not break it.

        Mutant that turns this red: moving mkdir/chmod back into __init__.
        """

        def _readonly_mkdir(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        def _readonly_chmod(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(Path, "mkdir", _readonly_mkdir)
        monkeypatch.setattr(os, "chmod", _readonly_chmod)

        storage = tmp_path / "nope"
        mgr = SecureTokenManager(storage_path=storage)

        assert mgr.storage_path == storage
        assert not storage.exists()

    async def test_store_token_creates_storage_lazily(self, tmp_path):
        """store_token creates the storage directory on demand and writes the token.

        Mutant that turns this red: removing _ensure_storage() from store_token.
        """
        from cryptography.fernet import Fernet

        storage = tmp_path / "created_on_demand"

        mgr = SecureTokenManager(storage_path=storage)
        # Seed the master key so _get_master_key returns from cache and never
        # reaches its salt-file write. Without this, _get_master_key's own
        # _ensure_storage() would create the directory and the mutant
        # (store_token missing _ensure_storage) would survive.
        mgr._master_key = Fernet.generate_key()
        assert not storage.exists()

        metadata = SecureTokenMetadata(
            token_id="lazy-token",
            scope=TokenScope.READ_ONLY,
            security_level="high",
            rotation_interval_days=30,
            expires_at=datetime.now(UTC) + timedelta(days=60),
        )
        token_id = await mgr.store_token("ghp_lazy_value", metadata)

        assert token_id == "lazy-token"
        assert storage.is_dir()
        assert (storage / "lazy-token.enc").exists()
