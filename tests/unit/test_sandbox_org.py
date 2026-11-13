"""Platform-owned SANDBOX orgs (``CODING_GITHUB_SANDBOX_GITHUB_OWNERS``).

A sandbox org (``conveyor-coding``) is owned by the platform but the internal App
cannot be installed there, so its repos are served by the PUBLIC App. The access
matrix must be the PLATFORM one — platform tenants + the per-tenant repo
allowlist, everyone else ``platform_repo_forbidden`` — and fr's customer binding
must never be consulted for it. Unset setting ⇒ the org is an ordinary customer
owner, exactly as today.

External boundaries are mocked exactly as in ``test_tenant_repo_token.py``: fr
(``httpx.get``), GitHub's installation lookup (PyGithub ``GithubIntegration``) and
the repo-scoped mint (``httpx.post``).
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import services.github_app_auth as ga
import services.github_integration as gi
import services.tenant_binding as tb
from config.settings import (
    get_settings,
    parse_platform_repo_allowlist,
    parse_sandbox_github_owners,
)
from services.github_app_auth import RepoPushMetadata

PLATFORM = "nonagenticai"
SANDBOX = "conveyor-coding"
FIXTURE = "aa-bugfix-capstone-fixture"
OTHER = "other-sandbox-repo"
EXEC_TENANT = "0195f48b-d000-7000-8000-000000000003"  # platform tenant
DEMO_TENANT = "0195f48b-d000-7000-8000-000000000103"  # allowlisted, NOT platform
CUSTOMER_TENANT = "11111111-1111-1111-1111-111111111111"
PUBLIC_INSTALLATION = 555
PUBLIC_APP_ID = "777000"
INTERNAL_APP_ID = "2333879"
SANDBOX_TOKEN = "ghs_public_sandbox_one_repo"
FR_TOKEN = "fr-binding-secret"  # pragma: allowlist secret
INTERNAL_TOKEN = "s3cret-internal-token"  # pragma: allowlist secret
CLONE_TOKEN_URL = "/api/internal/github/clone-token"


def _resp(status: int, payload=None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


class _World:
    def __init__(self) -> None:
        # fr WOULD bind the customer tenant to conveyor-coding: the sandbox path
        # must refuse it anyway, without ever asking fr.
        self.bindings = {
            (CUSTOMER_TENANT, SANDBOX): PUBLIC_INSTALLATION,
            (EXEC_TENANT, SANDBOX): None,
        }
        self.github_installation: int | None = PUBLIC_INSTALLATION
        # GitHub's ``account`` on the installation covering the repo, and the
        # owner of the repository GitHub lists in the minted token's response.
        self.installation_account: dict | None = {"login": SANDBOX, "type": "Organization"}
        self.minted_repo_owner = SANDBOX
        self.fr_get = MagicMock(side_effect=self._fr_answer)
        self.integration = MagicMock()
        self.integration.get_repo_installation.side_effect = self._gh_answer
        self.integration.get_access_token.return_value = SimpleNamespace(token="ghs_ORG_WIDE")
        self.mock_gi = MagicMock(return_value=self.integration)
        self.mock_auth = MagicMock()
        self.mock_auth.AppAuth.return_value.create_jwt.return_value = "app-jwt"
        self.post = MagicMock(side_effect=self._mint_answer)
        self.app_settings = SimpleNamespace(
            github_app_id=INTERNAL_APP_ID,
            github_app_installation_id="99",
            github_app_private_key="INTERNAL-KEY",  # pragma: allowlist secret
            github_app_private_key_path="",
            public_app_id=PUBLIC_APP_ID,
            public_app_private_key="PUBLIC-KEY",  # pragma: allowlist secret
            public_app_private_key_path="",
        )
        self.tb_settings = SimpleNamespace(fr_url="http://fr.test", fr_binding_token=FR_TOKEN)

    def _fr_answer(self, url, params=None, headers=None, timeout=None):
        key = (params["tenant_id"], params["owner"].lower())
        return _resp(200, {"installation_id": self.bindings.get(key)})

    def _gh_answer(self, owner, repo):
        if self.github_installation is None:
            from github import UnknownObjectException

            raise UnknownObjectException(404, {"message": "Not Found"}, {})
        raw = {"id": self.github_installation}
        if self.installation_account is not None:
            raw["account"] = dict(self.installation_account)
        return SimpleNamespace(id=self.github_installation, raw_data=raw)

    def _mint_answer(self, url, json=None, headers=None, timeout=None):
        repos = [
            {"name": r, "owner": {"login": self.minted_repo_owner}} for r in json["repositories"]
        ]
        return _resp(
            201,
            {"token": SANDBOX_TOKEN, "expires_at": "2999-01-01T00:00:00Z", "repositories": repos},
        )

    def enter(self, stack: ExitStack) -> _World:
        for p in (
            patch.object(ga, "get_settings", return_value=self.app_settings),
            patch.object(ga, "Auth", self.mock_auth),
            patch.object(ga, "GithubIntegration", self.mock_gi),
            patch.object(ga, "HAS_GITHUB", True),
            patch.object(ga.httpx, "post", self.post),
            patch.object(tb, "get_settings", return_value=self.tb_settings),
            patch.object(tb.httpx, "get", self.fr_get),
        ):
            stack.enter_context(p)
        return self

    def assert_public_one_repo_mint(self, repo: str, permissions: dict[str, str]) -> None:
        assert self.post.call_count == 1
        assert self.post.call_args.args[0].endswith(
            f"/app/installations/{PUBLIC_INSTALLATION}/access_tokens"
        )
        assert self.post.call_args.kwargs["json"] == {
            "permissions": permissions,
            "repositories": [repo],
        }
        app_ids = {c.args[0] for c in self.mock_auth.AppAuth.call_args_list if c.args}
        assert PUBLIC_APP_ID in app_ids and INTERNAL_APP_ID not in app_ids, app_ids
        self.integration.get_access_token.assert_not_called()  # no org-wide token
        assert ga._token_by_id == {}


def _settings_objects():
    """``gi.settings`` (import-time) and ``get_settings()`` are one object in prod;
    other suites clear the lru_cache, so patch both to reach every reader."""
    real = get_settings()
    return (real,) if gi.settings is real else (real, gi.settings)


def _patch_all(monkeypatch, name, value):
    for obj in _settings_objects():
        monkeypatch.setattr(obj, name, value)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _patch_all(monkeypatch, "system_github_owner", PLATFORM)
    _patch_all(monkeypatch, "_sandbox_owners", frozenset())
    _patch_all(monkeypatch, "_platform_repo_grants", {})
    _patch_all(monkeypatch, "_platform_repo_read_only_grants", {})
    ga.reset_cache()
    tb.reset_binding_cache()
    yield
    ga.reset_cache()
    tb.reset_binding_cache()


@pytest.fixture
def world():
    with ExitStack() as stack:
        yield _World().enter(stack)


@pytest.fixture
def sandbox(monkeypatch):
    """CODING_GITHUB_SANDBOX_GITHUB_OWNERS=conveyor-coding + demo allowlisted on the fixture."""
    real = get_settings()
    owners = parse_sandbox_github_owners(SANDBOX, platform_owner=PLATFORM)
    grants = parse_platform_repo_allowlist(
        f'{{"{DEMO_TENANT}": ["{SANDBOX}/{FIXTURE}"]}}',
        platform_owner=PLATFORM,
        platform_tenants=real.platform_tenant_ids,
        sandbox_owners=owners,
    )
    _patch_all(monkeypatch, "_sandbox_owners", owners)
    _patch_all(monkeypatch, "_platform_repo_grants", grants)
    assert EXEC_TENANT in real.platform_tenant_ids
    return owners


# ---------------------------------------------------------------------------
# Unset ⇒ exactly today's customer path
# ---------------------------------------------------------------------------
class TestUnsetIsToday:
    def test_not_sandbox_owned(self):
        assert gi.is_sandbox_owned(SANDBOX) is False
        assert gi.is_platform_owned(SANDBOX) is False

    def test_customer_path_consults_fr(self, world, enforce_platform_tenants):
        # The customer tenant fr bound to conveyor-coding gets its token, as today.
        tok = tb.resolve_customer_repo_token(SANDBOX, FIXTURE, CUSTOMER_TENANT, tb.PERMS_CLONE)
        assert tok.token == SANDBOX_TOKEN
        world.fr_get.assert_called_once()
        # A platform tenant with no fr binding is not_installed, as today.
        world.fr_get.reset_mock()
        with pytest.raises(tb.NotInstalledForTenantError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        world.fr_get.assert_called_once()


# ---------------------------------------------------------------------------
# Set ⇒ the platform access matrix, served by the PUBLIC App, fr never called
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("sandbox", "enforce_platform_tenants")
class TestSandboxMatrix:
    def test_owner_predicates(self):
        assert gi.is_sandbox_owned("Conveyor-Coding") is True
        assert gi.is_sandbox_owned(PLATFORM) is False
        assert gi.is_sandbox_owned("acme-corp") is False
        assert gi.is_sandbox_owned(None) is False

    def test_platform_tenant_gets_public_one_repo_token(self, world):
        tok = tb.resolve_customer_repo_token(SANDBOX, OTHER, EXEC_TENANT, tb.PERMS_AUTHORING)
        assert tok.token == SANDBOX_TOKEN
        world.assert_public_one_repo_mint(OTHER, tb.PERMS_AUTHORING)
        world.fr_get.assert_not_called()

    def test_allowlisted_tenant_gets_only_its_repo(self, world):
        tok = tb.resolve_customer_repo_token(SANDBOX, FIXTURE, DEMO_TENANT, tb.PERMS_CLONE)
        assert tok.token == SANDBOX_TOKEN
        world.assert_public_one_repo_mint(FIXTURE, tb.PERMS_CLONE)
        world.post.reset_mock()
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.resolve_customer_repo_token(SANDBOX, OTHER, DEMO_TENANT, tb.PERMS_CLONE)
        world.post.assert_not_called()
        world.fr_get.assert_not_called()

    def test_customer_with_fr_binding_is_forbidden_and_fr_not_called(self, world, caplog):
        with caplog.at_level("WARNING"), pytest.raises(tb.PlatformRepoForbiddenError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, CUSTOMER_TENANT, tb.PERMS_CLONE)
        world.fr_get.assert_not_called()
        world.post.assert_not_called()
        world.integration.get_repo_installation.assert_not_called()
        assert "platform_repo_tenant_gate decision=refused" in caplog.text

    def test_tenantless_is_forbidden(self, world):
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, None, tb.PERMS_CLONE)
        world.fr_get.assert_not_called()

    def test_public_app_not_installed(self, world):
        world.github_installation = None
        with pytest.raises(tb.NotInstalledForTenantError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        world.post.assert_not_called()
        world.fr_get.assert_not_called()

    def test_public_app_not_configured(self, world):
        world.app_settings.public_app_id = ""
        with pytest.raises(tb.PublicAppNotConfiguredError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        world.post.assert_not_called()

    def test_internal_app_grant_never_applies_to_sandbox(self, world):
        # The allowlist entry is for the PUBLIC App; the internal-App grant is refused.
        assert tb.platform_repo_grant(DEMO_TENANT, SANDBOX, FIXTURE) == "refuse"
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.mint_allowlisted_platform_token(
                SANDBOX, FIXTURE, tb.PERMS_CLONE, tenant_id=DEMO_TENANT
            )

    def test_resolve_github_token_routes_sandbox(self, world, monkeypatch):
        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_must_not_leak")
        r = gi.resolve_github_token(owner=SANDBOX, repo=FIXTURE, tenant_id=EXEC_TENANT)
        assert (r.token, r.is_app, r.rung) == (SANDBOX_TOKEN, True, "public_app")
        with pytest.raises(tb.PlatformRepoForbiddenError):
            gi.resolve_github_token(owner=SANDBOX, repo=FIXTURE, tenant_id=CUSTOMER_TENANT)
        world.fr_get.assert_not_called()

    def test_resolve_tenant_repo_token_routes_sandbox(self, world):
        assert (
            tb.resolve_tenant_repo_token(SANDBOX, FIXTURE, DEMO_TENANT, tb.PERMS_PR_READ)
            == SANDBOX_TOKEN
        )
        world.fr_get.assert_not_called()


@pytest.mark.usefixtures("sandbox")
class TestSandboxObserve:
    def test_observe_keeps_today_for_a_would_refuse(self, world, caplog):
        assert tb.platform_tenant_enforced() is False
        with caplog.at_level("WARNING"):
            tok = tb.resolve_customer_repo_token(SANDBOX, FIXTURE, CUSTOMER_TENANT, tb.PERMS_CLONE)
        assert tok.token == SANDBOX_TOKEN  # today's customer answer, unchanged
        world.fr_get.assert_called_once()
        assert "platform_repo_tenant_gate decision=would_refuse" in caplog.text

    def test_observe_still_serves_platform_tenant_without_fr(self, world):
        tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        world.fr_get.assert_not_called()


# ---------------------------------------------------------------------------
# clone-token route
# ---------------------------------------------------------------------------
@pytest.fixture
def clone_env(world):
    from routers import github_internal

    private_flag = MagicMock(return_value=True)
    internal_mint = MagicMock()
    with (
        patch.object(
            github_internal,
            "get_settings",
            return_value=SimpleNamespace(internal_callers={"coding-containers": INTERNAL_TOKEN}),
        ),
        patch.object(github_internal, "get_repo_visibility", MagicMock(return_value="private")),
        patch.object(
            github_internal,
            "get_repo_push_metadata",
            MagicMock(side_effect=lambda o, r, t: RepoPushMetadata(private_flag(o, r, t), "main")),
        ),
        patch.object(github_internal, "get_scoped_clone_token", internal_mint),
    ):
        yield world, private_flag, internal_mint


async def _clone(client, repo, tenant, scope="clone"):
    body = {"repo_url": f"https://github.com/{SANDBOX}/{repo}", "scope": scope}
    if tenant:
        body["tenant_id"] = tenant
    return await client.post(
        CLONE_TOKEN_URL, json=body, headers={"X-Internal-Token": INTERNAL_TOKEN}
    )


@pytest.mark.unit
@pytest.mark.usefixtures("sandbox", "enforce_platform_tenants")
class TestSandboxCloneToken:
    async def test_platform_tenant_push_token_requires_private(self, async_client, clone_env):
        w, private_flag, internal_mint = clone_env
        r = await _clone(async_client, OTHER, EXEC_TENANT, scope="push")
        assert r.status_code == 200, r.text
        assert (r.json()["token"], r.json()["source"], r.json()["reason"]) == (
            SANDBOX_TOKEN,
            "app",
            None,
        )
        w.assert_public_one_repo_mint(OTHER, {"contents": "write"})
        private_flag.assert_called_once()  # push guard ran on the sandbox repo
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()

    async def test_public_sandbox_repo_push_refused(self, async_client, clone_env):
        w, private_flag, _ = clone_env
        private_flag.return_value = False
        r = await _clone(async_client, OTHER, EXEC_TENANT, scope="push")
        assert r.status_code == 403, r.text

    async def test_allowlisted_only_its_repo(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        ok = await _clone(async_client, FIXTURE, DEMO_TENANT)
        assert ok.json()["token"] == SANDBOX_TOKEN
        no = await _clone(async_client, OTHER, DEMO_TENANT)
        assert (no.json()["token"], no.json()["reason"]) == (None, "platform_repo_forbidden")
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()

    async def test_customer_with_binding_forbidden(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(async_client, FIXTURE, CUSTOMER_TENANT)
        assert (r.json()["token"], r.json()["reason"]) == (None, "platform_repo_forbidden")
        w.fr_get.assert_not_called()
        w.post.assert_not_called()
        internal_mint.assert_not_called()

    async def test_tenantless_forbidden(self, async_client, clone_env):
        w, _, _ = clone_env
        r = await _clone(async_client, FIXTURE, None)
        assert (r.json()["token"], r.json()["reason"]) == (None, "platform_repo_forbidden")
        w.fr_get.assert_not_called()

    async def test_public_app_not_installed(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        w.github_installation = None
        r = await _clone(async_client, FIXTURE, EXEC_TENANT)
        assert (r.json()["token"], r.json()["reason"]) == (None, "not_installed")
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()

    async def test_public_app_not_configured(self, async_client, clone_env):
        w, _, _ = clone_env
        w.app_settings.public_app_id = ""
        r = await _clone(async_client, FIXTURE, EXEC_TENANT)
        assert (r.json()["token"], r.json()["reason"]) == (None, "not_configured")


# ---------------------------------------------------------------------------
# Settings parsing
# ---------------------------------------------------------------------------
class TestSandboxSettings:
    def test_owner_list_is_lowercased(self):
        assert parse_sandbox_github_owners(
            " Conveyor-Coding , ,other-sbx", platform_owner=PLATFORM
        ) == frozenset({SANDBOX, "other-sbx"})
        assert parse_sandbox_github_owners("", platform_owner=PLATFORM) == frozenset()

    @pytest.mark.parametrize("raw", ["NonAgenticAI", "bad/name", "sp ace", ".."])
    def test_bad_owner_fails_boot(self, raw):
        with pytest.raises(RuntimeError, match="CODING_GITHUB_SANDBOX_GITHUB_OWNERS"):
            parse_sandbox_github_owners(raw, platform_owner=PLATFORM)

    def test_allowlist_accepts_sandbox_only_when_configured(self):
        raw = f'{{"{DEMO_TENANT}": ["Conveyor-Coding/{FIXTURE}"]}}'
        with pytest.raises(RuntimeError, match="CODING_GITHUB_PLATFORM_REPO_ALLOWLIST"):
            parse_platform_repo_allowlist(
                raw, platform_owner=PLATFORM, platform_tenants=frozenset()
            )
        grants = parse_platform_repo_allowlist(
            raw,
            platform_owner=PLATFORM,
            platform_tenants=frozenset(),
            sandbox_owners=frozenset({SANDBOX}),
        )
        assert grants == {DEMO_TENANT: frozenset({f"{SANDBOX}/{FIXTURE}"})}

    def test_allowlist_rejects_trailing_newline_in_name(self):
        # `$` matched before a trailing newline; the name regex must anchor with \Z.
        with pytest.raises(RuntimeError, match="is not 'owner/repo'"):
            parse_platform_repo_allowlist(
                f'{{"{DEMO_TENANT}": ["{SANDBOX}/{FIXTURE}\\n"]}}',
                platform_owner=PLATFORM,
                platform_tenants=frozenset(),
                sandbox_owners=frozenset({SANDBOX}),
            )

    def test_allowlist_still_rejects_customer_and_platform_tenant(self):
        with pytest.raises(RuntimeError, match="not owned"):
            parse_platform_repo_allowlist(
                f'{{"{DEMO_TENANT}": ["acme-corp/{FIXTURE}"]}}',
                platform_owner=PLATFORM,
                platform_tenants=frozenset(),
                sandbox_owners=frozenset({SANDBOX}),
            )
        with pytest.raises(RuntimeError, match="platform tenant"):
            parse_platform_repo_allowlist(
                f'{{"{EXEC_TENANT}": ["{SANDBOX}/{FIXTURE}"]}}',
                platform_owner=PLATFORM,
                platform_tenants=frozenset({EXEC_TENANT}),
                sandbox_owners=frozenset({SANDBOX}),
            )

    def test_settings_from_env(self, monkeypatch):
        from config.settings import Settings

        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_OWNER", PLATFORM)
        monkeypatch.setenv("CODING_GITHUB_SANDBOX_GITHUB_OWNERS", "Conveyor-Coding")
        monkeypatch.setenv(
            "CODING_GITHUB_PLATFORM_REPO_ALLOWLIST", f'{{"{DEMO_TENANT}": ["{SANDBOX}/{FIXTURE}"]}}'
        )
        s = Settings()
        assert s.sandbox_owners == frozenset({SANDBOX})
        assert s.platform_repo_grants == {DEMO_TENANT: frozenset({f"{SANDBOX}/{FIXTURE}"})}
        monkeypatch.delenv("CODING_GITHUB_SANDBOX_GITHUB_OWNERS")
        with pytest.raises(RuntimeError, match="CODING_GITHUB_PLATFORM_REPO_ALLOWLIST"):
            Settings()
        monkeypatch.setenv("CODING_GITHUB_SANDBOX_GITHUB_OWNERS", PLATFORM)
        with pytest.raises(RuntimeError, match="CODING_GITHUB_SANDBOX_GITHUB_OWNERS"):
            Settings()


# ---------------------------------------------------------------------------
# ⛔ The installation / minted repo must BE the sandbox org's (review finding 1)
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("sandbox", "enforce_platform_tenants")
class TestSandboxInstallationMustBeTheSandboxOrgs:
    """GitHub follows a transferred/renamed repo, so ``GET /repos/conveyor-coding/x/
    installation`` can answer with ANOTHER account's installation. A platform tenant
    must then get nothing — never a public-App token on a customer's installation."""

    @pytest.mark.parametrize(
        "account",
        [
            {"login": "acme-corp", "type": "Organization"},  # customer org's install
            {"login": SANDBOX, "type": "User"},  # a user account, not the org
            {"login": "acme-corp"},  # no type
            None,  # GitHub omitted the account: fail closed
        ],
        ids=["customer-org", "user-account", "no-type", "no-account"],
    )
    def test_foreign_installation_refused_before_mint(self, world, caplog, account):
        world.installation_account = account
        with caplog.at_level("ERROR"), pytest.raises(tb.NotInstalledForTenantError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_AUTHORING)
        world.post.assert_not_called()  # no token was ever minted
        world.fr_get.assert_not_called()  # and no fall-through to the customer path
        assert "decision=sandbox_installation_mismatch" in caplog.text

    def test_account_login_compared_case_insensitively(self, world):
        world.installation_account = {"login": "Conveyor-Coding", "type": "Organization"}
        tok = tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        assert tok.token == SANDBOX_TOKEN

    def test_minted_repo_owned_by_customer_refused_and_discarded(self, world, caplog):
        world.minted_repo_owner = "acme-corp"
        with caplog.at_level("ERROR"), pytest.raises(tb.NotInstalledForTenantError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE)
        assert world.post.call_count == 1
        assert ga._SCOPED_CACHES[ga.PUBLIC_APP] == {}  # never cached ⇒ never served
        assert "OWNER MISMATCH" in caplog.text
        # And the next call re-checks instead of serving it.
        world.minted_repo_owner = SANDBOX
        assert (
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, EXEC_TENANT, tb.PERMS_CLONE).token
            == SANDBOX_TOKEN
        )

    def test_resolve_github_token_refuses_foreign_installation(self, world):
        world.installation_account = {"login": "acme-corp", "type": "Organization"}
        with pytest.raises(tb.NotInstalledForTenantError):
            gi.resolve_github_token(owner=SANDBOX, repo=FIXTURE, tenant_id=EXEC_TENANT)
        world.post.assert_not_called()

    async def test_clone_token_route_says_not_installed(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        w.installation_account = {"login": "acme-corp", "type": "Organization"}
        r = await _clone(async_client, FIXTURE, EXEC_TENANT, scope="push")
        assert (r.json()["token"], r.json()["reason"]) == (None, "not_installed")
        w.post.assert_not_called()
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()


# ---------------------------------------------------------------------------
# ⛔ Tenantless is refused under OBSERVE too (review finding 2)
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("sandbox")
class TestSandboxTenantlessUnderObserve:
    """Under enforce, the tenant gate would refuse a tenantless call anyway, so only
    OBSERVE can tell whether the unconditional tenantless guard is present: without
    it a would-refuse falls through to the customer path (``tenant_required``)."""

    @pytest.mark.parametrize("tenant", [None, "", "   "])
    def test_tenantless_forbidden_even_in_observe(self, world, caplog, tenant):
        assert tb.platform_tenant_enforced() is False
        with caplog.at_level("WARNING"), pytest.raises(tb.PlatformRepoForbiddenError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, tenant, tb.PERMS_CLONE)
        assert "decision=refused tenant=none" in caplog.text
        assert "would_refuse" not in caplog.text
        world.fr_get.assert_not_called()
        world.post.assert_not_called()
        world.integration.get_repo_installation.assert_not_called()

    async def test_clone_token_route_tenantless_forbidden_in_observe(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(async_client, FIXTURE, None)
        assert (r.json()["token"], r.json()["reason"]) == (None, "platform_repo_forbidden")
        w.post.assert_not_called()
        w.fr_get.assert_not_called()
        internal_mint.assert_not_called()


# ---------------------------------------------------------------------------
# ⛔ Stored project / user tokens never author on a sandbox repo (review finding 3)
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("sandbox")
class TestSandboxSkipsStoredTokens:
    def _container(self, **kw):
        from routers.github_pull_request_operations import _ResolvedContainer

        info = {
            "container_name": "sbx-c0",
            "repo_url": f"https://github.com/{SANDBOX}/{FIXTURE}",
            "branch": "main",
        }
        info.update(kw)
        return _ResolvedContainer(info)

    @pytest.mark.parametrize(
        "kw",
        [
            {"secret_config_id": "01a00427-2708-7322-8e31-9950972e9702"},
            {"github_access_token": "ghp_inline"},
        ],
        ids=["stored-project-token", "inline-legacy-token"],
    )
    async def test_effective_token_is_none_for_sandbox(self, monkeypatch, kw):
        import routers.github_pull_request_operations as ops
        import services.secret_resolver as sr

        async def _must_not_run(secret_config_id, tenant_id):
            raise AssertionError("secrets-service must not be consulted for a sandbox repo")

        monkeypatch.setattr(sr, "resolve_git_token", _must_not_run)
        token = await ops._effective_github_token(
            self._container(**kw), EXEC_TENANT, owner="Conveyor-Coding"
        )
        assert token is None, "None is what routes GitHubIntegration to the App"

    async def test_customer_owner_still_uses_stored_token(self, monkeypatch):
        """Control: the same call for a non-sandbox owner DOES return the BYO token."""
        import routers.github_pull_request_operations as ops
        import services.secret_resolver as sr

        async def _byo(secret_config_id, tenant_id):
            return "ghp_byo"

        monkeypatch.setattr(sr, "resolve_git_token", _byo)
        token = await ops._effective_github_token(
            self._container(secret_config_id="x"), CUSTOMER_TENANT, owner="acme-corp"
        )
        assert token == "ghp_byo"

    @pytest.mark.parametrize(
        ("owner", "expect_app"),
        [(SANDBOX, True), ("acme-corp", False)],
        ids=["sandbox", "customer"],
    )
    async def test_upstream_pr_uses_app_not_stored_token(self, monkeypatch, owner, expect_app):
        import main
        import services.github_token_manager as gtm
        import services.secret_resolver as sr

        container = {
            "container": {
                "user_id": "u1",
                "original_repo_owner": owner,
                "original_repo_name": FIXTURE,
                "secret_config_id": "01a00427-2708-7322-8e31-9950972e9702",
            }
        }
        client = MagicMock()

        async def _get_workspace(name, headers=None):
            return container

        client.get_workspace = _get_workspace
        monkeypatch.setattr(main, "get_workspaces_client", lambda: client)
        stored_calls: list[str] = []

        async def _stored(secret_config_id, tenant_id):
            stored_calls.append("project")
            return "ghp_stored_project_token"

        monkeypatch.setattr(sr, "resolve_git_token", _stored)
        mgr = MagicMock()

        async def _user_token(**kw):
            stored_calls.append("user")
            return "ghp_stored_user_token"

        mgr.get_user_token = _user_token
        monkeypatch.setattr(gtm, "get_github_token_manager", lambda: mgr)
        resolved: list[tuple] = []

        def _resolve(**kw):
            resolved.append((kw["owner"], kw["tenant_id"]))
            return gi.ResolvedToken("ghs_app_sandbox", True, "public_app")

        monkeypatch.setattr(gi, "resolve_github_token", _resolve)
        auth = MagicMock()
        monkeypatch.setattr(gi, "Auth", auth)
        gh = MagicMock()
        gh.return_value.get_repo.side_effect = RuntimeError("stop after token choice")
        monkeypatch.setattr(gi, "Github", gh)

        await gi.GitHubIntegration.create_upstream_pull_request(
            "sbx-c0", "t", "m", "feature", tenant_id=EXEC_TENANT
        )
        used = auth.Token.call_args.args[0]
        if expect_app:
            assert used == "ghs_app_sandbox"
            assert resolved == [(owner, EXEC_TENANT)]
            assert stored_calls == []
        else:  # control: a customer owner's stored token still wins
            assert used == "ghp_stored_project_token"
            assert resolved == []


# ---------------------------------------------------------------------------
# A ``:read`` grant on a SANDBOX repo: the public-App mint refuses write perms.
# ---------------------------------------------------------------------------
@pytest.fixture
def sandbox_read_only(monkeypatch):
    from config.settings import parse_platform_repo_allowlist_modes

    real = get_settings()
    owners = parse_sandbox_github_owners(SANDBOX, platform_owner=PLATFORM)
    grants, read_only = parse_platform_repo_allowlist_modes(
        f'{{"{DEMO_TENANT}": ["{SANDBOX}/{FIXTURE}:read"]}}',
        platform_owner=PLATFORM,
        platform_tenants=real.platform_tenant_ids,
        sandbox_owners=owners,
    )
    _patch_all(monkeypatch, "_sandbox_owners", owners)
    _patch_all(monkeypatch, "_platform_repo_grants", grants)
    _patch_all(monkeypatch, "_platform_repo_read_only_grants", read_only)
    return read_only


@pytest.mark.unit
@pytest.mark.usefixtures("sandbox_read_only", "enforce_platform_tenants")
class TestSandboxReadOnlyGrant:
    def test_clone_permissions_are_minted(self, world):
        tok = tb.resolve_customer_repo_token(SANDBOX, FIXTURE, DEMO_TENANT, tb.PERMS_CLONE)
        assert tok.token == SANDBOX_TOKEN
        world.assert_public_one_repo_mint(FIXTURE, tb.PERMS_CLONE)

    @pytest.mark.parametrize("perms", [tb.PERMS_AUTHORING, tb.PERMS_ISSUES, tb.PERMS_MERGE])
    def test_write_permissions_refused_before_github(self, world, perms):
        """MUTANT: drop the chokepoint call in resolve_sandbox_repo_token -> minted."""
        with pytest.raises(tb.PlatformRepoReadOnlyError):
            tb.resolve_customer_repo_token(SANDBOX, FIXTURE, DEMO_TENANT, perms)
        world.post.assert_not_called()
        world.fr_get.assert_not_called()

    async def test_clone_token_push_is_403(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(async_client, FIXTURE, DEMO_TENANT, scope="push")
        assert r.status_code == 403, r.text
        assert "platform_repo_read_only" in r.text
        w.post.assert_not_called()
        internal_mint.assert_not_called()

    async def test_clone_token_push_refused_at_the_mint_without_the_route_check(
        self, async_client, clone_env, monkeypatch
    ):
        from routers import github_internal

        monkeypatch.setattr(github_internal, "is_read_only_grant", lambda *a: False)
        w, _, _ = clone_env
        r = await _clone(async_client, FIXTURE, DEMO_TENANT, scope="push")
        assert r.status_code == 403, r.text
        assert "platform_repo_read_only" in r.text
        w.post.assert_not_called()
