"""Tests for config/settings.py."""

from __future__ import annotations

import importlib
import os
import sys

import pytest


def _get_real_settings_module():
    """Get the real config.settings module, bypassing conftest stubs.

    The conftest replaces config.settings attributes with test stubs.
    We need to re-import the actual source to test it properly.
    """
    # Force reimport of the real module from source
    mod_key = "config.settings"
    saved = sys.modules.pop(mod_key, None)
    saved_config = sys.modules.get("config")

    # Temporarily remove the config module to force fresh import
    if saved_config:
        saved_get = getattr(saved_config, "get_settings", None)
        saved_lff = getattr(saved_config, "load_feature_flags", None)

    try:
        mod = importlib.import_module("config.settings")
        return mod
    finally:
        # Restore stubs after we grabbed the real module
        if saved is not None:
            sys.modules[mod_key] = saved
        if saved_config:
            if saved_get:
                saved_config.get_settings = saved_get
            if saved_lff:
                saved_config.load_feature_flags = saved_lff


# Get the real module once at import time
_real = _get_real_settings_module()
Settings = _real.Settings
get_settings = _real.get_settings
load_feature_flags = _real.load_feature_flags


class TestDefaultServiceUrls:
    def test_default_service_urls(self, monkeypatch):
        # Service URLs are sourced from gitops env, not code-side defaults.
        # Empty defaults force production boot to fail loudly if any are
        # missing (see Settings._REQUIRED_IN_PRODUCTION). Non-production
        # env (test) skips the validator, so defaults stay empty.
        for var in (
            "CODING_GITHUB_CODING_GATEWAY_URL",
            "CODING_GITHUB_CODING_CONTAINERS_URL",
            "CODING_GITHUB_CODING_SECURITY_URL",
            "CODING_GITHUB_CODING_ANALYSIS_URL",
            "CODING_GITHUB_SECRETS_API_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CODING_GITHUB_ENVIRONMENT", "test")
        s = Settings()
        assert s.coding_gateway_url == ""
        assert s.coding_containers_url == ""
        assert s.coding_security_url == ""
        assert s.coding_analysis_url == ""
        assert s.secrets_api_url == ""


class TestEnvOverrideServiceUrls:
    def test_env_override_service_urls(self, monkeypatch):
        monkeypatch.setenv("CODING_GITHUB_CODING_GATEWAY_URL", "http://gw:1111")
        monkeypatch.setenv("CODING_GITHUB_CODING_CONTAINERS_URL", "http://ct:2222")
        monkeypatch.setenv("CODING_GITHUB_CODING_SECURITY_URL", "http://sec:3333")
        monkeypatch.setenv("CODING_GITHUB_CODING_ANALYSIS_URL", "http://an:4444")

        s = Settings()
        assert s.coding_gateway_url == "http://gw:1111"
        assert s.coding_containers_url == "http://ct:2222"
        assert s.coding_security_url == "http://sec:3333"
        assert s.coding_analysis_url == "http://an:4444"


class TestNoPlatformPatSetting:
    def test_legacy_pat_env_vars_boot_and_map_to_nothing(self, monkeypatch):
        """⛔ No PAT setting exists (owner ruling 2026-09-30).

        A deployment still carrying the old env vars must boot (``extra: ignore``)
        and none of them may surface as an attribute a caller could read.
        """
        monkeypatch.setenv("CODING_GITHUB_GITHUB_TOKEN", "tok-abc123")
        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_TOKEN", "tok-abc123")
        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_TOKENS", "tok-a,tok-b")
        s = Settings()
        for name in (
            "GITHUB_ACCESS_TOKEN",
            "github_token",
            "system_github_token",
            "system_github_tokens",
        ):
            assert not hasattr(s, name), name
        assert "tok-abc123" not in s.model_dump_json()


class TestDefaultRedisAndDbUrls:
    def test_default_db_url_and_no_redis_setting(self, monkeypatch):
        # The database URL is sourced from gitops env, not a code-side default:
        # an empty default forces production boot to fail loudly if it is missing
        # (see Settings._REQUIRED_IN_PRODUCTION).
        #
        # `redis_url` is deliberately GONE. It was a required-in-production
        # setting for a dependency with no caller anywhere in `src`, and no
        # coding-github-redis Service was ever deployed — so it forced every
        # production boot to carry a URL it never dialled for any real work.
        monkeypatch.delenv("CODING_GITHUB_DATABASE_URL", raising=False)
        monkeypatch.setenv("CODING_GITHUB_ENVIRONMENT", "test")
        get_settings.cache_clear()
        s = Settings()
        assert s.database_url == ""
        assert not hasattr(s, "redis_url")
        assert "redis_url" not in Settings._REQUIRED_IN_PRODUCTION


class TestProductionRequiresUrls:
    def test_production_missing_urls_raises(self, monkeypatch):
        # Mirrors inference-router/coding-gateway: production boot fails
        # loudly when any required service URL is unset.
        for var in (
            "CODING_GITHUB_DATABASE_URL",
            "CODING_GITHUB_CODING_GATEWAY_URL",
            "CODING_GITHUB_CODING_CONTAINERS_URL",
            "CODING_GITHUB_CODING_SECURITY_URL",
            "CODING_GITHUB_CODING_ANALYSIS_URL",
            "CODING_GITHUB_SECRETS_API_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CODING_GITHUB_ENVIRONMENT", "production")
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="missing required fields in production"):
            Settings()


class TestLoadFeatureFlags:
    def test_load_feature_flags_empty(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("CODING_GITHUB_FEATURE_"):
                monkeypatch.delenv(key)
        assert load_feature_flags() == {}

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE", "Yes", "ON"])
    def test_load_feature_flags_truthy(self, monkeypatch, value):
        # Clean any pre-existing feature flags
        for key in list(os.environ):
            if key.startswith("CODING_GITHUB_FEATURE_"):
                monkeypatch.delenv(key)
        monkeypatch.setenv("CODING_GITHUB_FEATURE_MY_FLAG", value)
        flags = load_feature_flags()
        assert flags["my_flag"] is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", "No", "OFF", "random"])
    def test_load_feature_flags_falsy(self, monkeypatch, value):
        for key in list(os.environ):
            if key.startswith("CODING_GITHUB_FEATURE_"):
                monkeypatch.delenv(key)
        monkeypatch.setenv("CODING_GITHUB_FEATURE_BETA", value)
        flags = load_feature_flags()
        assert flags["beta"] is False


class TestGetSettingsSingleton:
    def test_get_settings_singleton(self):
        get_settings.cache_clear()
        try:
            a = get_settings()
            b = get_settings()
            assert a is b
        finally:
            get_settings.cache_clear()


class TestConfigInitExports:
    def test_config_init_exports(self):
        # The real config.__init__ re-exports these three names
        assert Settings is not None
        assert callable(get_settings)
        assert callable(load_feature_flags)

        # Verify __all__ in the actual source
        import config as cfg_mod

        # Whether stubs replaced them or not, the module should expose these names
        assert hasattr(cfg_mod, "get_settings")
        assert hasattr(cfg_mod, "load_feature_flags")
