"""⛔ There is NO GitHub personal-access-token fallback anywhere in coding-github.

Owner ruling 2026-09-30: GitHub access is ONLY through the GitHub Apps — the
internal App for ``system_github_owner`` repos (platform tenants only) and the
public App through the fr tenant binding. A tenant's own BYO ``user_token`` /
``secret_config_id`` is unchanged and is not a platform credential.

Two independent instruments, so neither can hide a regression the other misses:

(a) STRUCTURAL — ``src/`` must not name any of the removed PAT sources. This is
    what catches a re-added rung on a path no behavioural test drives.
(b) BEHAVIOURAL — with every legacy PAT variable armed in the environment and no
    App configured, platform-repo token resolution (the read-only probe in
    ``github_validation`` and the PR-actions route resolver) yields NO token.
    Each has a positive control (App configured -> App token) so "no token" can
    never mean "this path never reaches a credential rung at all".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import PLATFORM_TEST_TENANT

SRC = Path(__file__).resolve().parents[2] / "src"
PLATFORM = "nonagenticai"
LEGACY_PAT = "ghp_LEGACY_PAT_must_never_be_used"  # pragma: allowlist secret
LEGACY_PAT_VARS = (
    "GITHUB_ACCESS_TOKEN",
    "GITHUB_TOKEN",
    "CODING_GITHUB_GITHUB_TOKEN",
    "CODING_GITHUB_SYSTEM_GITHUB_TOKEN",
    "CODING_GITHUB_SYSTEM_GITHUB_TOKENS",
)

# Each entry: (label, compiled pattern). Case-sensitive except where noted.
FORBIDDEN = (
    ("GITHUB_ACCESS_TOKEN", re.compile(r"GITHUB_ACCESS_TOKEN")),
    ("system_github_token", re.compile(r"system_github_token", re.IGNORECASE)),
    ("github_token_pool", re.compile(r"github_token_pool")),
    ("getenv GITHUB_TOKEN", re.compile(r"getenv\(\s*[\"']GITHUB_TOKEN[\"']")),
    ("environ.get GITHUB_TOKEN", re.compile(r"environ\.get\(\s*[\"']GITHUB_TOKEN[\"']")),
    ("environ[GITHUB_TOKEN]", re.compile(r"environ\[\s*[\"']GITHUB_TOKEN[\"']\s*\]")),
    ("settings.github_token", re.compile(r"settings\.github_token\b")),
    ("CODING_GITHUB_GITHUB_TOKEN", re.compile(r"CODING_GITHUB_GITHUB_TOKEN")),
)


def _violations(text: str) -> list[str]:
    return [label for label, pat in FORBIDDEN if pat.search(text)]


# ---------------------------------------------------------------------------
# (a) structural
# ---------------------------------------------------------------------------
class TestSourceNamesNoPatSource:
    def test_the_scanner_can_fail(self):
        """Positive control: every pattern fires on the shape it forbids, and does
        NOT fire on the legitimate look-alikes that must stay (BYO container
        token, secure-token-manager key vars)."""
        bad = (
            'x = os.getenv("GITHUB_ACCESS_TOKEN")\n'
            "settings.system_github_token\n"
            "SYSTEM_GITHUB_TOKEN\n"
            "from services.github_token_pool import get_token_pool\n"
            'os.getenv("GITHUB_TOKEN")\n'
            "os.environ.get('GITHUB_TOKEN')\n"
            'os.environ["GITHUB_TOKEN"]\n'
            "settings.github_token\n"
            'os.getenv("CODING_GITHUB_GITHUB_TOKEN")\n'
        )
        assert sorted(_violations(bad)) == sorted(label for label, _ in FORBIDDEN)
        fine = (
            'info.get("github_access_token")\n'
            'os.getenv("GITHUB_TOKEN_ENCRYPTION_KEY")\n'
            'os.getenv("GITHUB_TOKEN_PASSWORD")\n'
            'error_code="GITHUB_TOKEN_REQUIRED"\n'
        )
        assert _violations(fine) == []

    def test_src_names_no_removed_pat_source(self):
        files = sorted(SRC.rglob("*.py"))
        # Guard against a vacuous pass from a wrong path / empty tree.
        assert len(files) > 50, f"only {len(files)} files under {SRC}"
        hits = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            for label in _violations(text):
                hits.append(f"{path.relative_to(SRC)}: {label}")
        assert hits == [], "PAT fallback source re-introduced:\n" + "\n".join(hits)

    def test_deleted_pat_modules_stay_deleted(self):
        assert not (SRC / "services" / "github_token_pool.py").exists()
        assert not (SRC / "services" / "github_container_provisioner.py").exists()


# ---------------------------------------------------------------------------
# (b) behavioural
# ---------------------------------------------------------------------------
@pytest.fixture
def armed(monkeypatch):
    """Platform org set, every legacy PAT variable armed, App NOT configured."""
    import services.github_app_auth as ga
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", PLATFORM)
    for var in LEGACY_PAT_VARS:
        monkeypatch.setenv(var, LEGACY_PAT)
    monkeypatch.setattr(ga, "app_auth_configured", lambda: False)
    ga.reset_cache()
    yield
    ga.reset_cache()


class TestValidationProbeHasNoPatRung:
    def test_platform_repo_platform_tenant_gets_no_token_without_the_app(self, armed):
        from services.github_validation import _resolve_github_token

        resolved = _resolve_github_token(None, PLATFORM, "coding-github", PLATFORM_TEST_TENANT)
        assert resolved.token is None
        assert resolved.is_app is False

    def test_positive_control_app_configured_gives_the_app_token(self, armed, internal_app):
        from services.github_validation import _resolve_github_token

        resolved = _resolve_github_token(None, PLATFORM, "coding-github", PLATFORM_TEST_TENANT)
        assert resolved.token == internal_app
        assert resolved.is_app is True


class TestPrActionsResolverHasNoPatRung:
    async def test_platform_repo_platform_tenant_gets_no_token_without_the_app(self, armed):
        from routers.pr_actions import _platform_pr_actions_token

        token = await _platform_pr_actions_token(
            PLATFORM, "coding-github", PLATFORM_TEST_TENANT, "test_no_pat_fallback"
        )
        assert token is None

    async def test_positive_control_app_configured_gives_the_app_token(self, armed, internal_app):
        from routers.pr_actions import _platform_pr_actions_token

        token = await _platform_pr_actions_token(
            PLATFORM, "coding-github", PLATFORM_TEST_TENANT, "test_no_pat_fallback"
        )
        assert token == internal_app


class TestNoCarrierHoldsAPat:
    def test_settings_expose_no_pat(self, armed):
        from config.settings import Settings

        s = Settings()
        for name in ("GITHUB_ACCESS_TOKEN", "github_token", "system_github_token"):
            assert not hasattr(s, name), name
        assert LEGACY_PAT not in s.model_dump_json()

    def test_auth_manager_and_workflow_trigger_hold_no_token(self, armed):
        from services.github.auth_manager import GitHubAuthManager
        from services.github_actions_trigger import GitHubActionsWorkflowTrigger

        assert GitHubAuthManager().get_token() is None
        trigger = GitHubActionsWorkflowTrigger()
        assert trigger.github_token is None
        assert trigger.github_client is None


class TestRepoRouteHelperHasNoPatRung:
    """``routers.github_issue_operations._github`` is the ONE credential helper
    behind every tenant-facing repo route landed after this guard was written:
    get-issue and the issue ops, get-repository, get-repository-file, get-branch,
    compare, compare/files, list-pull-request-files, list-workflow-runs and the
    CI-wait (registration AND the evaluator's ``fetch_snapshot``). Armed with
    every legacy PAT variable and no App, it must yield NO service at all."""

    async def test_platform_repo_platform_tenant_gets_no_service_without_the_app(self, armed):
        from fastapi import HTTPException

        from routers.github_issue_operations import _github
        from services.github_integration import GitHubAuthError

        with pytest.raises((GitHubAuthError, HTTPException)) as exc:
            async with _github(PLATFORM, "coding-github", PLATFORM_TEST_TENANT, None):
                pytest.fail("a GitHubService was opened with no App configured")
        assert LEGACY_PAT not in str(exc.value)

    async def test_positive_control_app_configured_gives_the_app_token(self, armed, internal_app):
        from routers.github_issue_operations import _github

        async with _github(PLATFORM, "coding-github", PLATFORM_TEST_TENANT, None) as service:
            assert service.config.token == internal_app


class TestPrAuthoringIdentityHasNoPatRung:
    """``GitHubIntegration`` with no explicit token (update-pull-request-body, the
    PR-create / merge family) resolves through the App ladder only."""

    def test_platform_repo_raises_without_the_app(self, armed):
        from services.github_integration import GitHubAuthError, GitHubIntegration

        with pytest.raises(GitHubAuthError):
            GitHubIntegration(
                target_owner=PLATFORM, target_repo="coding-github", tenant_id=PLATFORM_TEST_TENANT
            )

    def test_positive_control_app_configured_gives_the_app_token(self, armed, internal_app):
        from services.github_integration import GitHubIntegration

        gi = GitHubIntegration(
            target_owner=PLATFORM, target_repo="coding-github", tenant_id=PLATFORM_TEST_TENANT
        )
        assert gi.github_token == internal_app
        assert gi.token_rung == "app"


class TestCloneTokenHasNoPatRung:
    """The internal clone-token mint (push facts, #142) is App-or-nothing."""

    async def test_platform_clone_token_is_none_without_the_app(self, armed, monkeypatch):
        import routers.github_internal as gint
        from services.github_app_auth import GitHubAppNotConfiguredError

        def _unconfigured(*_a, **_k):
            raise GitHubAppNotConfiguredError("GitHub App ID not configured.")

        monkeypatch.setattr(gint, "get_scoped_clone_token", _unconfigured)
        token, _exp, source, reason = await gint._platform_clone_token(
            PLATFORM, "coding-github", "clone"
        )
        assert (token, source, reason) == (None, "none", "not_configured")
