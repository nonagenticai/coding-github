"""The ``environment`` setting must resolve through ``Settings``, never ``os.getenv``.

``Settings`` carries ``env_prefix="CODING_GITHUB_"`` (see ``src/config/settings.py``),
so the environment is configured as ``CODING_GITHUB_ENVIRONMENT``. Four sites used to
read the *bare* ``ENVIRONMENT`` name directly off the process environment:

    src/services/github/github_service.py:98, :627
    src/services/github_validation.py:86, :143

The live Deployment (``gitops-production/clusters/strato/services/coding-github/``)
sets **neither** spelling — verified against the running pod and the
``coding-github-env`` secret on 2026-07-29 — so every one of those reads returned
``""`` and ``is_production`` was ``False`` *in production*.

This is the same class as #58, where the startup banner read ``ENVIRONMENT`` /
``SERVICE_NAME`` unprefixed and therefore lied. The fix there was to read the
value off the settings object; these tests pin that invariant so a fifth site
cannot drift back.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import Settings, get_settings

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# The env-var name that must only ever be reached through ``Settings``.
_GUARDED_NAME = "ENVIRONMENT"


# ---------------------------------------------------------------------------
# 1. The invariant: no source file may read ENVIRONMENT off os.environ.
# ---------------------------------------------------------------------------
def _bare_environment_reads(tree: ast.AST) -> list[int]:
    """Return line numbers of any bare ``ENVIRONMENT`` process-env read.

    Catches all three spellings:
        os.getenv("ENVIRONMENT")
        os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"]

    Prefixed names (``CODING_GITHUB_ENVIRONMENT``) and unrelated names are fine,
    as is anything reached via the ``Settings`` object.
    """
    hits: list[int] = []

    for node in ast.walk(tree):
        # os.getenv("ENVIRONMENT") / os.environ.get("ENVIRONMENT")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("getenv", "get") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == _GUARDED_NAME:
                    hits.append(node.lineno)

        # os.environ["ENVIRONMENT"]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == _GUARDED_NAME:
                hits.append(node.lineno)

    return hits


def test_no_source_file_reads_environment_off_os_environ():
    """Reintroducing ``os.getenv("ENVIRONMENT")`` anywhere in src/ must fail here.

    This is the guard that would have caught the original drift: four independent
    sites each grew their own bare read because nothing pinned the invariant.
    """
    offenders: dict[str, list[int]] = {}

    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = _bare_environment_reads(tree)
        if lines:
            offenders[str(path.relative_to(SRC_ROOT))] = lines

    assert offenders == {}, (
        "These files read ENVIRONMENT off the process environment. Settings carries "
        "env_prefix='CODING_GITHUB_', so the bare name is never set in production and "
        "the read silently returns ''. Use get_settings().environment / .is_production "
        f"instead. Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# 2. is_production must be True under the configuration the deployment ACTUALLY
#    sets — which notably does not include any ENVIRONMENT variable at all.
# ---------------------------------------------------------------------------

# Exactly what the live pod gets: the five service URLs + the database URL from
# the coding-github-env secret. Reproduced from
# clusters/strato/services/coding-github/deployment.yaml and the live secret's
# key list. Note the deliberate absence of ENVIRONMENT / CODING_GITHUB_ENVIRONMENT.
_LIVE_DEPLOYMENT_ENV = {
    "CODING_GITHUB_DATABASE_URL": "postgresql+psycopg://u:p@postgres:5432/coding_github",
    "CODING_GITHUB_CODING_GATEWAY_URL": "http://coding-gateway.coding-gateway.svc.cluster.local:8687",
    "CODING_GITHUB_CODING_CONTAINERS_URL": "http://coding-containers.coding-containers.svc.cluster.local:8691",
    "CODING_GITHUB_CODING_SECURITY_URL": "http://coding-security.coding-security.svc.cluster.local:8693",
    "CODING_GITHUB_CODING_ANALYSIS_URL": "http://coding-analysis.coding-analysis.svc.cluster.local:8688",
    # Not a secret: a plain in-cluster Service URL. detect-secrets' "Secret Keyword"
    # heuristic fires on the SECRETS_API substring in the key name, not on the value.
    "CODING_GITHUB_SECRETS_API_URL": "http://secrets.secrets.svc.cluster.local:8543",  # pragma: allowlist secret
}


def _settings_from(env: dict[str, str]) -> Settings:
    """Build a Settings from a pristine environment containing only ``env``.

    ``clear=True`` matters: the repo conftest exports CODING_GITHUB_ENVIRONMENT=test
    process-wide, which would otherwise mask exactly the bug under test. ``_env_file=None``
    stops pydantic-settings reading a developer's local ``.env``.
    """
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


def test_is_production_true_under_the_live_deployment_env():
    """The deployment sets no ENVIRONMENT var; the Settings default must carry it."""
    settings = _settings_from(_LIVE_DEPLOYMENT_ENV)

    assert settings.environment == "production"
    assert settings.is_production is True


def test_is_production_true_when_deployment_sets_the_prefixed_name_explicitly():
    """And it must still hold once the Deployment declares the value explicitly."""
    settings = _settings_from({**_LIVE_DEPLOYMENT_ENV, "CODING_GITHUB_ENVIRONMENT": "production"})

    assert settings.is_production is True


def test_is_production_ignores_the_unprefixed_name():
    """A bare ENVIRONMENT=development must NOT be able to de-escalate production.

    Nothing in the deployment sets the bare name, so honouring it would only ever
    be an injection vector, never a legitimate configuration path.
    """
    settings = _settings_from({**_LIVE_DEPLOYMENT_ENV, "ENVIRONMENT": "development"})

    assert settings.is_production is True


@pytest.mark.parametrize("env_val", ["development", "dev", "testing", "test"])
def test_is_production_false_and_dev_or_test_true_for_dev_environments(env_val):
    settings = _settings_from({"CODING_GITHUB_ENVIRONMENT": env_val})

    assert settings.is_production is False
    assert settings.is_dev_or_test is True


@pytest.mark.parametrize("env_val", ["production", "PRODUCTION", " Production "])
def test_is_production_is_case_and_whitespace_insensitive(env_val):
    settings = _settings_from({**_LIVE_DEPLOYMENT_ENV, "CODING_GITHUB_ENVIRONMENT": env_val})

    assert settings.is_production is True
    assert settings.is_dev_or_test is False


def test_staging_is_neither_production_nor_dev():
    """staging must not unlock dev bypasses, and must not claim to be production."""
    settings = _settings_from({"CODING_GITHUB_ENVIRONMENT": "staging"})

    assert settings.is_production is False
    assert settings.is_dev_or_test is False


# ---------------------------------------------------------------------------
# 3. The call sites must honour the prefixed setting.
# ---------------------------------------------------------------------------
def _prefixed_settings(environment: str, **overrides) -> Settings:
    """A Settings pinned to ``environment`` via the prefixed name only."""
    env = {
        **_LIVE_DEPLOYMENT_ENV,
        "CODING_GITHUB_ENVIRONMENT": environment,
        **{f"CODING_GITHUB_{k.upper()}": v for k, v in overrides.items()},
    }
    return _settings_from(env)


@pytest.mark.parametrize("env_val", ["development", "testing", "test", "dev"])
def test_validation_bypass_honours_prefixed_dev_environment(env_val):
    """github_bypass_validation in a *prefixed* dev env must enable test mode.

    On main this fails: the site read the bare ENVIRONMENT, which is unset, so the
    dev check never matched however the service was configured.
    """
    from services.github_validation import CoreGitHubValidationService

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch(
            "services.github_validation.get_settings",
            return_value=_prefixed_settings(env_val),
        ),
        patch.dict(os.environ, {}, clear=False),
    ):
        for key in (
            "PYTEST_CURRENT_TEST",
            "TESTING",
            "TEST_MODE",
            "CI",
            "SCHEMATHESIS_TEST",
            "SCHEMATHESIS_TESTING",
            "ENVIRONMENT",
        ):
            os.environ.pop(key, None)

        assert CoreGitHubValidationService().test_mode is True


def test_validation_bypass_denied_under_the_live_production_config():
    """And it must stay denied when configured exactly as production is."""
    from services.github_validation import CoreGitHubValidationService

    with (
        patch(
            "services.github_validation.load_feature_flags",
            return_value={"github_bypass_validation": True},
        ),
        patch(
            "services.github_validation.get_settings",
            return_value=_settings_from(_LIVE_DEPLOYMENT_ENV),
        ),
        patch.dict(os.environ, {}, clear=False),
    ):
        for key in (
            "PYTEST_CURRENT_TEST",
            "TESTING",
            "TEST_MODE",
            "CI",
            "SCHEMATHESIS_TEST",
            "SCHEMATHESIS_TESTING",
        ):
            os.environ.pop(key, None)

        assert CoreGitHubValidationService().test_mode is False


async def test_schemathesis_token_bypass_is_refused_under_the_live_production_config():
    """The latent guard: SCHEMATHESIS_TESTING must not mint a mock token in production.

    This is the one site where the fix changes reachable behaviour. On main
    ``is_production`` is False in production, so a stray SCHEMATHESIS_TESTING=true
    would hand out ``schemathesis_test_token``. With the setting resolved properly
    the guard is armed and the request is refused.
    """
    from services.github import github_service as gh

    production = _settings_from(_LIVE_DEPLOYMENT_ENV)  # no PAT setting exists at all

    with (
        patch.object(gh, "get_settings", return_value=production),
        patch.dict(os.environ, {"SCHEMATHESIS_TESTING": "true"}, clear=False),
    ):
        os.environ.pop("ENVIRONMENT", None)

        with pytest.raises(ValueError, match="No GitHub credential"):
            await gh.get_github_service("test-owner/coding-github")


async def test_schemathesis_token_bypass_still_works_in_a_dev_environment():
    """The bypass must remain available where it is legitimate."""
    from services.github import github_service as gh

    development = _prefixed_settings("development")

    with (
        patch.object(gh, "get_settings", return_value=development),
        patch.dict(os.environ, {"SCHEMATHESIS_TESTING": "true"}, clear=False),
    ):
        service = await gh.get_github_service("test-owner/coding-github")

    assert service.config.token == "schemathesis_test_token"


def test_get_settings_is_the_single_resolution_point():
    """Both modules must resolve the environment through the same accessor."""
    from services import github_validation as gv
    from services.github import github_service as gh

    assert gh.get_settings is get_settings
    assert gv.get_settings is get_settings
