"""Extended tests for services/github/secure_token_manager.py

Covers edge cases: cryptography import fallback, master key derivation paths,
environment variable handling, schemathesis mock tokens, error cleanup,
and module-level singleton/factory functions.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

import services.github.secure_token_manager as stm_mod
from services.github.secure_token_manager import (
    CRYPTOGRAPHY_AVAILABLE,
    SecureTokenManager,
    SecureTokenMetadata,
    TokenScope,
    TokenStatus,
    create_default_token,
    get_secure_token_manager,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_OriginalMeta = SecureTokenMetadata


class _PermissiveMetadata(_OriginalMeta):
    """Subclass that allows past expires_at (skips future-date validation)."""

    @classmethod
    def validate_expires_at(cls, v):
        if v and v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        return v

    model_config = {**_OriginalMeta.model_config, "validate_default": False}


_PermissiveMetadata.model_rebuild()


@contextmanager
def _allow_past_expiry():
    """Patch SecureTokenMetadata in the module so past expires_at is accepted."""
    with patch.object(stm_mod, "SecureTokenMetadata", _PermissiveMetadata):
        yield


def _backdate_meta_file(path, overrides: dict):
    """Read a .meta JSON file, apply overrides, and write back."""
    with open(path) as f:
        data = json.load(f)
    data.update(overrides)
    with open(path, "w") as f:
        json.dump(data, f)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def token_manager(tmp_path):
    """Create a token manager with a temp storage directory."""
    with patch.dict(os.environ, {"GITHUB_TOKEN_PASSWORD": "test-password-123"}):
        mgr = SecureTokenManager(storage_path=tmp_path / "tokens")
        yield mgr


@pytest.fixture
def sample_metadata():
    """Create sample token metadata with future expiry."""
    return SecureTokenMetadata(
        token_id="test-token-1",
        scope=TokenScope.READ_ONLY,
        security_level="high",
        rotation_interval_days=30,
        expires_at=datetime.now(UTC) + timedelta(days=60),
    )


# ------------------------------------------------------------------ #
# Cryptography import fallback
# ------------------------------------------------------------------ #


class TestCryptographyImportFallback:
    """Verify the CRYPTOGRAPHY_AVAILABLE flag and import fallback behaviour."""

    def test_cryptography_available_flag_true(self):
        """When cryptography is installed, CRYPTOGRAPHY_AVAILABLE should be True."""
        assert CRYPTOGRAPHY_AVAILABLE is True

    def test_import_error_branch(self):
        """Simulate cryptography import failure to cover the fallback path."""
        import importlib

        orig_modules = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("cryptography"):
                orig_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _fake_import(name, *args, **kwargs):
            if name.startswith("cryptography"):
                raise ImportError("test: no cryptography")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=_fake_import):
                importlib.reload(stm_mod)
                assert stm_mod.CRYPTOGRAPHY_AVAILABLE is False
                assert stm_mod.Fernet is None
                assert stm_mod.hashes is None
                assert stm_mod.PBKDF2HMAC is None
        finally:
            sys.modules.update(orig_modules)
            importlib.reload(stm_mod)


# ------------------------------------------------------------------ #
# SecureTokenMetadata validation
# ------------------------------------------------------------------ #


class TestSecureTokenMetadataValidation:
    def test_timezone_naive_expiry_gets_utc(self):
        """A naive datetime should be coerced to UTC by the validator."""
        future_naive = datetime.now().replace(tzinfo=None) + timedelta(days=10)
        meta = SecureTokenMetadata(
            token_id="tz-test",
            expires_at=future_naive,
        )
        assert meta.expires_at is not None
        assert meta.expires_at.tzinfo is not None

    def test_past_expiry_raises_validation_error(self):
        """Expiration date in the past should raise a validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            SecureTokenMetadata(
                token_id="past-test",
                expires_at=datetime.now(UTC) - timedelta(days=1),
            )


# ------------------------------------------------------------------ #
# Master key derivation paths
# ------------------------------------------------------------------ #


class TestGetMasterKeyFromEnv:
    """Test _get_master_key with GITHUB_TOKEN_ENCRYPTION_KEY env var."""

    def test_valid_env_key_used(self, tmp_path):
        """Valid base64 Fernet key in env var is decoded and used."""
        import base64

        from cryptography.fernet import Fernet

        raw_key = Fernet.generate_key()
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN_ENCRYPTION_KEY": raw_key.decode(),
                "GITHUB_TOKEN_PASSWORD": "unused",
            },
        ):
            mgr = SecureTokenManager(storage_path=tmp_path / "tok1")
            key = mgr._get_master_key()
            assert key == base64.urlsafe_b64decode(raw_key)

    def test_invalid_env_key_falls_through(self, tmp_path):
        """Invalid base64 in env var falls through to password-based derivation."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN_ENCRYPTION_KEY": "!!!not-base64!!!",
                "GITHUB_TOKEN_PASSWORD": "fallback-pw",
            },
        ):
            mgr = SecureTokenManager(storage_path=tmp_path / "tok2")
            key = mgr._get_master_key()
            assert key is not None


class TestGetMasterKeyFromFiles:
    """Test key loading from existing salt files."""

    def test_load_existing_key_files(self, tmp_path):
        """When .salt exists, key is derived from same password."""
        sp = tmp_path / "tok3"
        env = {"GITHUB_TOKEN_PASSWORD": "my-password"}
        with patch.dict(os.environ, env):
            mgr1 = SecureTokenManager(storage_path=sp)
            key1 = mgr1._get_master_key()
            (sp / ".master_key").write_bytes(b"placeholder")
            mgr2 = SecureTokenManager(storage_path=sp)
            key2 = mgr2._get_master_key()
            assert key1 == key2

    def test_no_password_raises_runtime_error(self, tmp_path):
        """Missing GITHUB_TOKEN_PASSWORD env var raises RuntimeError."""
        sp = tmp_path / "tok4"
        with patch.dict(os.environ, {"GITHUB_TOKEN_PASSWORD": "pw123"}):
            mgr = SecureTokenManager(storage_path=sp)
            mgr._get_master_key()
            (sp / ".master_key").write_bytes(b"placeholder")

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN_ENCRYPTION_KEY", None)
            os.environ.pop("GITHUB_TOKEN_PASSWORD", None)
            mgr2 = SecureTokenManager(storage_path=sp)
            with pytest.raises(RuntimeError, match="GITHUB_TOKEN_PASSWORD"):
                mgr2._get_master_key()


class TestGetMasterKeyNoPassword:
    """No key files and no password env var."""

    def test_raises_runtime_error(self, tmp_path):
        """No existing key files and no GITHUB_TOKEN_PASSWORD -> RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN_ENCRYPTION_KEY", None)
            os.environ.pop("GITHUB_TOKEN_PASSWORD", None)
            mgr = SecureTokenManager(storage_path=tmp_path / "tok6")
            with pytest.raises(RuntimeError, match="GITHUB_TOKEN_PASSWORD"):
                mgr._get_master_key()


# ------------------------------------------------------------------ #
# Store token error cleanup
# ------------------------------------------------------------------ #


class TestStoreTokenErrorCleanup:
    """Partial files are cleaned up when encryption fails."""

    async def test_encryption_failure_cleans_up(self, token_manager, sample_metadata):
        """When encryption fails, no partial .enc or .meta files remain."""
        with patch.object(token_manager, "_get_master_key", side_effect=Exception("boom")):
            with pytest.raises(RuntimeError, match="Token storage failed"):
                await token_manager.store_token("ghp_test", sample_metadata)

        enc = token_manager.storage_path / "test-token-1.enc"
        meta = token_manager.storage_path / "test-token-1.meta"
        assert not enc.exists()
        assert not meta.exists()

    async def test_json_dump_failure_cleans_up_enc_file(self, token_manager, sample_metadata):
        """If json.dump fails after .enc is written, cleanup unlinks both files."""
        with patch(
            "services.github.secure_token_manager.json.dump", side_effect=OSError("disk full")
        ):
            with pytest.raises(RuntimeError, match="Token storage failed"):
                await token_manager.store_token("ghp_test", sample_metadata)

        enc_path = token_manager.storage_path / f"{sample_metadata.token_id}.enc"
        meta_path = token_manager.storage_path / f"{sample_metadata.token_id}.meta"
        assert not enc_path.exists()
        assert not meta_path.exists()


# ------------------------------------------------------------------ #
# Schemathesis mock token path
# ------------------------------------------------------------------ #


class TestRetrieveTokenSchemathesis:
    """SCHEMATHESIS_TESTING env var returns mock tokens."""

    @pytest.mark.parametrize("env_val", ["true", "on", "1", "yes"])
    async def test_schemathesis_env_returns_mock(self, token_manager, env_val):
        """Various truthy SCHEMATHESIS_TESTING values return a mock token."""
        with patch.dict(os.environ, {"SCHEMATHESIS_TESTING": env_val}):
            result = await token_manager.retrieve_token("schemathesis-token-1")
            assert result == "schemathesis_mock_token_schemathesis-token-1"

    async def test_schemathesis_non_matching_id_goes_normal(self, token_manager):
        """Non-schemathesis token IDs go through the normal retrieval path."""
        with patch.dict(os.environ, {"SCHEMATHESIS_TESTING": "true"}):
            result = await token_manager.retrieve_token("regular-token")
            assert result is None


# ------------------------------------------------------------------ #
# Expired token status update
# ------------------------------------------------------------------ #


class TestRetrieveTokenExpired:
    """Expired tokens have their status updated to EXPIRED."""

    async def test_expired_token_updates_status_to_expired(self, token_manager):
        """Expired token sets status to EXPIRED and returns None."""
        meta = SecureTokenMetadata(
            token_id="expire-me",
            scope=TokenScope.READ_ONLY,
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        await token_manager.store_token("ghp_exp", meta)
        token_manager._tokens.pop("expire-me", None)

        meta_file = token_manager.storage_path / "expire-me.meta"
        _backdate_meta_file(
            meta_file, {"expires_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat()}
        )

        with _allow_past_expiry():
            result = await token_manager.retrieve_token("expire-me")
            assert result is None

        with open(meta_file) as f:
            data = json.load(f)
        assert data["status"] == TokenStatus.EXPIRED


# ------------------------------------------------------------------ #
# Cleanup expired tokens -- old expired
# ------------------------------------------------------------------ #


class TestCleanupExpiredOldToken:
    """Expired tokens older than 1 day are removed."""

    async def test_cleanup_expired_old_token(self, token_manager):
        meta = SecureTokenMetadata(
            token_id="old-expired",
            scope=TokenScope.READ_ONLY,
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        await token_manager.store_token("ghp_expired", meta)

        meta_file = token_manager.storage_path / "old-expired.meta"
        _backdate_meta_file(
            meta_file,
            {"expires_at": (datetime.now(UTC) - timedelta(days=3)).isoformat()},
        )
        token_manager._tokens.pop("old-expired", None)

        with _allow_past_expiry():
            removed = await token_manager.cleanup_expired_tokens()
        assert removed == 1


# ------------------------------------------------------------------ #
# Module-level functions
# ------------------------------------------------------------------ #


class TestModuleLevelFunctions:
    """get_secure_token_manager singleton and create_default_token."""

    def test_singleton_returns_same_instance(self, tmp_path):
        """get_secure_token_manager returns the same instance on repeated calls."""
        stm_mod._token_manager = None
        with patch.dict(os.environ, {"GITHUB_TOKEN_PASSWORD": "pw"}):
            mgr1 = get_secure_token_manager()
            mgr2 = get_secure_token_manager()
            assert mgr1 is mgr2
            stm_mod._token_manager = None

    async def test_create_default_token_stores_and_retrieves(self, tmp_path):
        """create_default_token stores a token with id='default' that can be retrieved."""
        with patch.dict(os.environ, {"GITHUB_TOKEN_PASSWORD": "pw"}):
            stm_mod._token_manager = SecureTokenManager(storage_path=tmp_path / "def_tok")
            try:
                token_id = await create_default_token("ghp_default_value")
                assert token_id == "default"

                mgr = stm_mod._token_manager
                val = await mgr.retrieve_token("default")
                assert val == "ghp_default_value"

                meta = await mgr.get_token_metadata("default")
                assert meta.scope == TokenScope.READ_WRITE
            finally:
                stm_mod._token_manager = None
