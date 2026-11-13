"""Tenant-bound public-App tokens for CUSTOMER repositories.

The invariant (plan: public-github-app-usable.md): a token for a repo outside
``system_github_owner`` is issued ONLY via the PUBLIC App, ONLY for an
installation fr has bound to the REQUESTING tenant, ONLY if GitHub confirms that
installation covers the repo. Never a PAT fallback.

Everything external is mocked at the HTTP / PyGithub boundary:
* fr's binding lookup  -> ``httpx.get`` (answers per (tenant, owner))
* GitHub's installation lookup -> the PUBLIC App's ``GithubIntegration``
* the repo-scoped mint -> ``httpx.post``
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import services.github_app_auth as ga
import services.github_integration as gi
import services.tenant_binding as tb
from services.github_app_auth import RepoPushMetadata

PLATFORM = "nonagenticai"
CUSTOMER = "acme-corp"
REPO = "widgets"
TENANT_A = "11111111-1111-1111-1111-111111111111"
TENANT_B = "22222222-2222-2222-2222-222222222222"
EXEC_TENANT = "0195f48b-d000-7000-8000-000000000003"  # fr EXECUTION (platform tenant)
BOUND_ID = 4242
FR_TOKEN = "fr-binding-secret"  # pragma: allowlist secret


def _resp(status: int, payload=None, *, text: str | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    if text is not None:
        r.json.side_effect = ValueError("not json")
    else:
        r.json.return_value = payload
    return r


class _World:
    """One configured estate: fr bindings, GitHub's truth, and the mint."""

    def __init__(self, bindings=None, github_installation=BOUND_ID):
        # (tenant, owner.lower()) -> installation id
        self.bindings = {(TENANT_A, CUSTOMER): BOUND_ID} if bindings is None else bindings
        self.github_installation = github_installation
        self.fr_get = MagicMock(side_effect=self._fr_answer)
        self.integration = MagicMock()
        self.integration.get_repo_installation.side_effect = self._gh_answer
        self.mock_gi = MagicMock(return_value=self.integration)
        self.mock_auth = MagicMock()
        self.mock_auth.AppAuth.return_value.create_jwt.return_value = "public-app-jwt"
        self.post = MagicMock(
            return_value=_resp(
                201, {"token": "ghs_tenant_scoped", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        self.app_settings = SimpleNamespace(
            github_app_id="2333879",
            github_app_installation_id="99",
            github_app_private_key="INTERNAL-KEY",  # pragma: allowlist secret
            github_app_private_key_path="",
            public_app_id="777000",
            public_app_private_key="PUBLIC-KEY",  # pragma: allowlist secret
            public_app_private_key_path="",
        )
        self.tb_settings = SimpleNamespace(fr_url="http://fr.test", fr_binding_token=FR_TOKEN)

    def _fr_answer(self, url, params=None, headers=None, timeout=None):
        assert url == "http://fr.test/api/internal/github-app/binding"
        assert headers["X-Internal-Token"] == FR_TOKEN
        key = (params["tenant_id"], params["owner"].lower())
        return _resp(200, {"installation_id": self.bindings.get(key), "account_login": CUSTOMER})

    def _gh_answer(self, owner, repo):
        if self.github_installation is None:
            from github import UnknownObjectException

            raise UnknownObjectException(404, {"message": "Not Found"}, {})
        return SimpleNamespace(id=self.github_installation)

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


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(gi.settings, "system_github_owner", PLATFORM)
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
def armed_pats(monkeypatch):
    """Every former PAT source armed in the environment.

    No PAT rung exists any more (owner ruling 2026-09-30); these prove none of
    the legacy variables is ever turned into a credential, for any repo.
    """
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_ENV_PAT_must_not_leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_ENV_PAT_must_not_leak")
    monkeypatch.setenv("CODING_GITHUB_GITHUB_TOKEN", "ghp_SETTINGS_PAT_must_not_leak")
    monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_TOKEN", "ghp_SYSTEM_PAT_must_not_leak")
    return {
        "ghp_ENV_PAT_must_not_leak",
        "ghp_SETTINGS_PAT_must_not_leak",
        "ghp_SYSTEM_PAT_must_not_leak",
    }


# ---------------------------------------------------------------------------
# Platform owner: unchanged internal path, fr never consulted
# ---------------------------------------------------------------------------
class TestPlatformOwnerUnchanged:
    def test_platform_owner_uses_internal_app_and_never_asks_fr(self, world, monkeypatch):
        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: f"ghs_internal_{o}")

        token = tb.resolve_tenant_repo_token(
            PLATFORM, "coding-github", EXEC_TENANT, tb.PERMS_AUTHORING
        )

        assert token == "ghs_internal_nonagenticai"
        world.fr_get.assert_not_called()
        world.post.assert_not_called()

    def test_platform_owner_still_fails_closed_without_the_internal_app(
        self, world, monkeypatch, armed_pats
    ):
        monkeypatch.setattr(ga, "app_auth_configured", lambda: False)
        with pytest.raises(gi.GitHubAuthError) as ei:
            tb.resolve_tenant_repo_token(PLATFORM, "coding-github", EXEC_TENANT, tb.PERMS_AUTHORING)
        assert not isinstance(ei.value, tb.TenantRepoTokenError)
        world.fr_get.assert_not_called()


# ---------------------------------------------------------------------------
# Platform owner: ONLY platform tenants (CODING_GITHUB_PLATFORM_TENANTS)
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("enforce_platform_tenants")
class TestPlatformOwnerRefusedToCustomerTenants:
    @pytest.fixture
    def internal_app(self, monkeypatch):
        minted: list[tuple[str, str]] = []

        def _mint(o, r):
            minted.append((o, r))
            return f"ghs_internal_{o}"

        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", _mint)
        return minted

    @pytest.mark.parametrize("tenant", [TENANT_A, None, "", "not-a-uuid"])
    def test_non_platform_tenant_is_refused_before_the_internal_app(
        self, world, internal_app, armed_pats, tenant
    ):
        with pytest.raises(tb.PlatformRepoForbiddenError) as ei:
            tb.resolve_tenant_repo_token(PLATFORM, "coding-github", tenant, tb.PERMS_AUTHORING)
        assert ei.value.code == "platform_repo_forbidden"
        assert ei.value.http_status == 403
        assert (ei.value.owner, ei.value.repo) == (PLATFORM, "coding-github")
        assert internal_app == []
        world.fr_get.assert_not_called()

    def test_resolve_github_token_refuses_a_customer_tenant_on_a_platform_repo(
        self, internal_app, armed_pats
    ):
        """The shared ladder: a tenant is present and NOT a platform tenant."""
        with pytest.raises(tb.PlatformRepoForbiddenError):
            gi.resolve_github_token(owner=PLATFORM, repo="coding-github", tenant_id=TENANT_A)
        assert internal_app == []

    def test_resolve_github_token_serves_a_platform_tenant(self, internal_app):
        resolved = gi.resolve_github_token(
            owner=PLATFORM, repo="coding-github", tenant_id=EXEC_TENANT.upper()
        )
        assert resolved.token == "ghs_internal_nonagenticai"
        assert internal_app == [(PLATFORM, "coding-github")]

    def test_resolve_github_token_without_tenant_context_is_refused(self, internal_app):
        """⛔ H1: ``tenant_id=None`` on a platform repo used to skip the gate and
        get the org installation token. It is refused now (0 live callers,
        measured 2026-09-28). Kills the mutant "restore the None skip"."""
        with pytest.raises(tb.PlatformRepoForbiddenError) as ei:
            gi.resolve_github_token(owner=PLATFORM, repo="coding-github")
        assert ei.value.code == "platform_repo_forbidden"
        assert internal_app == []

    def test_explicit_byo_token_is_the_callers_choice_and_unchanged(self, internal_app):
        resolved = gi.resolve_github_token(
            user_token="ghp_callers_own", owner=PLATFORM, repo="coding-github", tenant_id=TENANT_A
        )
        assert (resolved.token, resolved.rung) == ("ghp_callers_own", "user")
        assert internal_app == []

    def test_validation_probe_refuses_a_customer_tenant_on_a_platform_repo(
        self, internal_app, armed_pats
    ):
        from services.github_validation import _resolve_github_token as probe_token

        with pytest.raises(tb.PlatformRepoForbiddenError):
            probe_token(None, PLATFORM, "coding-github", TENANT_A)
        assert internal_app == []
        # Platform tenant: unchanged (env PAT rung first on this read-only probe).
        assert probe_token(None, PLATFORM, "coding-github", EXEC_TENANT).token is not None

    def test_error_maps_to_403_with_owner_and_repo(self):
        exc = tb.github_auth_http_exception(
            tb.PlatformRepoForbiddenError("x", owner=PLATFORM, repo="coding-github")
        )
        assert exc.status_code == 403
        assert "platform_repo_forbidden" in str(exc.detail)
        assert PLATFORM in str(exc.detail) and "coding-github" in str(exc.detail)


# ---------------------------------------------------------------------------
# Customer owner: tenant-bound public App
# ---------------------------------------------------------------------------
class TestCustomerOwnerBoundTenant:
    def test_bound_tenant_gets_public_app_repo_scoped_mint(self, world):
        token = tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)

        assert token == "ghs_tenant_scoped"
        # The PUBLIC App identity signed the JWT, not the internal one.
        world.mock_auth.AppAuth.assert_called_once_with("777000", "PUBLIC-KEY")
        world.integration.get_repo_installation.assert_called_once_with(CUSTOMER, REPO)
        world.post.assert_called_once()
        url = world.post.call_args.args[0]
        kwargs = world.post.call_args.kwargs
        assert url == f"https://api.github.com/app/installations/{BOUND_ID}/access_tokens"
        assert kwargs["json"] == {
            "permissions": {"contents": "write", "pull_requests": "write"},
            "repositories": [REPO],
        }
        assert kwargs["headers"]["Authorization"] == "Bearer public-app-jwt"

    def test_owner_is_matched_case_insensitively_in_the_binding_cache(self, world):
        tb.resolve_tenant_repo_token("ACME-Corp", REPO, TENANT_A, tb.PERMS_CLONE)
        tb.resolve_tenant_repo_token("acme-corp", REPO, TENANT_A, tb.PERMS_CLONE)
        assert world.fr_get.call_count == 1  # cached under (tenant, owner.lower())

    def test_public_token_never_enters_the_internal_caches(self, world):
        tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_CLONE)
        assert ga._token_by_id == {}
        assert ga._scoped_token_by_repo == {}
        assert list(ga._public_scoped_token_by_repo) == [
            (BOUND_ID, CUSTOMER, REPO, "contents:read")
        ]


class TestCustomerOwnerRefusals:
    def test_tenant_b_cannot_use_tenant_a_binding(self, world):
        """THE TENANT CHECK (fr side): tenant B has no binding for A's owner."""
        with pytest.raises(tb.NotInstalledForTenantError) as ei:
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_B, tb.PERMS_AUTHORING)
        assert ei.value.code == "github_app_not_installed"
        assert CUSTOMER in str(ei.value)
        world.post.assert_not_called()
        # And fr was asked about tenant B specifically.
        assert world.fr_get.call_args.kwargs["params"]["tenant_id"] == TENANT_B

    def test_tenant_a_cached_binding_is_never_served_to_tenant_b(self, world):
        """CROSS-TENANT CACHE POISONING. Tenant A's binding for the owner is
        resolved (and cached) first; tenant B then asks for the SAME owner/repo
        inside the TTL. B must be asked about at fr on its own key and refused,
        never handed A's cached installation. A cache keyed on owner alone would
        answer B from A's entry and mint B a token for A's installation.
        """
        assert tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_CLONE)
        assert world.fr_get.call_count == 1

        with pytest.raises(tb.NotInstalledForTenantError):
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_B, tb.PERMS_CLONE)
        assert asyncio.run(tb.alookup_binding(TENANT_B, CUSTOMER)) is None

        assert world.fr_get.call_count == 2  # B went to fr; the async twin hit B's entry
        assert world.fr_get.call_args.kwargs["params"]["tenant_id"] == TENANT_B
        world.post.assert_called_once()  # only A's mint ever happened

    def test_binding_exists_but_github_installation_differs_is_refused(self):
        """THE TENANT CHECK (GitHub side): the repo is covered by an installation
        that is NOT the one bound to this tenant (e.g. bound to another tenant)."""
        with ExitStack() as stack:
            w = _World(github_installation=BOUND_ID + 1).enter(stack)
            with pytest.raises(tb.NotInstalledForTenantError):
                tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
            w.post.assert_not_called()

    def test_repo_not_covered_by_any_installation_is_refused(self):
        with ExitStack() as stack:
            w = _World(github_installation=None).enter(stack)
            with pytest.raises(tb.NotInstalledForTenantError):
                tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
            w.post.assert_not_called()

    def test_no_tenant_is_tenant_required_and_fr_is_not_asked(self, world):
        with pytest.raises(tb.TenantRequiredError) as ei:
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, None, tb.PERMS_AUTHORING)
        assert ei.value.http_status == 403
        world.fr_get.assert_not_called()
        world.post.assert_not_called()

    def test_public_app_unconfigured_is_refused_not_downgraded(self, world):
        world.app_settings.public_app_id = ""
        with pytest.raises(tb.PublicAppNotConfiguredError):
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
        world.post.assert_not_called()

    def test_github_refusing_the_mint_is_refused(self, world):
        world.post.return_value = _resp(422, {"message": "permissions not granted"})
        with pytest.raises(tb.PublicAppMintFailedError):
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)


class TestFrFailsClosed:
    @pytest.mark.parametrize(
        "answer",
        [
            _resp(500, {"error": "boom"}),
            _resp(403, {"error": "bad token"}),
            _resp(200, None, text="<html>"),
            _resp(200, {"unexpected": True}),
            _resp(200, {"installation_id": "4242"}),
            _resp(200, {"installation_id": True}),
            _resp(200, {"installation_id": 0}),
        ],
        ids=["5xx", "403", "non_json", "shape", "string_id", "bool_id", "zero_id"],
    )
    def test_fr_error_is_refused_never_unbound_or_bound(self, world, answer):
        world.fr_get.side_effect = None
        world.fr_get.return_value = answer
        with pytest.raises(tb.TenantBindingUnavailableError) as ei:
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
        assert ei.value.http_status == 503
        world.post.assert_not_called()
        world.integration.get_repo_installation.assert_not_called()

    def test_fr_unreachable_is_refused_and_not_cached(self, world):
        world.fr_get.side_effect = httpx.ConnectError("fr down")
        with pytest.raises(tb.TenantBindingUnavailableError):
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
        # fr recovers: the failure must not have been cached as "unbound".
        world.fr_get.side_effect = world._fr_answer
        assert tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_CLONE)

    def test_fr_unconfigured_is_refused(self, world):
        world.tb_settings.fr_url = ""
        with pytest.raises(tb.TenantBindingUnavailableError):
            tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, tb.PERMS_AUTHORING)
        world.post.assert_not_called()

    def test_async_lookup_fails_closed_too(self, world):
        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **k):
                raise httpx.ReadTimeout("slow")

        with patch.object(tb.httpx, "AsyncClient", _Client):
            with pytest.raises(tb.TenantBindingUnavailableError):
                asyncio.run(tb.alookup_binding(TENANT_A, CUSTOMER))

    def test_binding_cache_expires(self, world, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(tb.time, "monotonic", lambda: clock[0])
        tb.lookup_binding(TENANT_A, CUSTOMER)
        tb.lookup_binding(TENANT_A, CUSTOMER)
        assert world.fr_get.call_count == 1
        clock[0] += tb.BINDING_TTL_SECONDS + 1
        tb.lookup_binding(TENANT_A, CUSTOMER)
        assert world.fr_get.call_count == 2


# ---------------------------------------------------------------------------
# No PAT fallback for a customer owner — anywhere
# ---------------------------------------------------------------------------
class TestNoPatFallbackForCustomerOwner:
    def test_resolve_github_token_never_returns_a_pat_for_a_customer(self, world, armed_pats):
        world.bindings.clear()  # tenant A has NO binding now
        with pytest.raises(tb.NotInstalledForTenantError):
            gi.resolve_github_token(owner=CUSTOMER, repo=REPO, tenant_id=TENANT_A)
        with pytest.raises(tb.TenantRequiredError):
            gi.resolve_github_token(owner=CUSTOMER, repo=REPO)

    def test_resolve_github_token_customer_success_is_the_public_app(self, world, armed_pats):
        rt = gi.resolve_github_token(owner=CUSTOMER, repo=REPO, tenant_id=TENANT_A)
        assert rt.token == "ghs_tenant_scoped"
        assert rt.token not in armed_pats
        assert rt.is_app is True
        assert rt.rung == "public_app"

    def test_github_integration_refuses_instead_of_using_a_pat(self, world, armed_pats):
        world.bindings.clear()
        with pytest.raises(tb.NotInstalledForTenantError):
            gi.GitHubIntegration(target_owner=CUSTOMER, target_repo=REPO, tenant_id=TENANT_A)

    def test_validation_probe_goes_anonymous_not_env_pat(self, world, armed_pats):
        from services.github_validation import _resolve_github_token

        world.bindings.clear()
        resolved = _resolve_github_token(None, CUSTOMER, REPO, TENANT_A)
        assert resolved.token is None  # anonymous probe, NOT the env PAT
        resolved = _resolve_github_token(None, CUSTOMER, REPO, None)
        assert resolved.token is None

    def test_validation_probe_fr_error_raises_not_anonymous(self, world, armed_pats):
        from services.github_validation import _resolve_github_token

        world.fr_get.side_effect = httpx.ConnectError("fr down")
        with pytest.raises(tb.TenantBindingUnavailableError):
            _resolve_github_token(None, CUSTOMER, REPO, TENANT_A)

    def test_validation_probe_bound_tenant_gets_read_only_public_token(self, world, armed_pats):
        from services.github_validation import _resolve_github_token

        resolved = _resolve_github_token(None, CUSTOMER, REPO, TENANT_A)
        assert resolved.token == "ghs_tenant_scoped"
        assert resolved.is_app is True
        assert world.post.call_args.kwargs["json"]["permissions"] == {"contents": "read"}


# ---------------------------------------------------------------------------
# Route mapping: a refusal is a clear HTTP error naming the owner
# ---------------------------------------------------------------------------
class TestRouteErrorMapping:
    def test_issue_route_is_409_github_app_not_installed_for_unbound_tenant(self, world):
        from fastapi import HTTPException

        import routers.github_issue_operations as mod
        from schemas.github_api_models import CreateIssueRequest

        req = CreateIssueRequest(owner=CUSTOMER, repo=REPO, title="t", body="b")
        with pytest.raises(HTTPException) as ei:
            asyncio.run(mod.create_issue(req, tenant_id=TENANT_B))
        assert ei.value.status_code == 409
        assert ei.value.detail["error_code"] == "github_app_not_installed"
        assert ei.value.detail["details"]["owner"] == CUSTOMER
        world.post.assert_not_called()

    def test_generic_auth_error_keeps_the_historical_412(self):
        e = tb.github_auth_http_exception(gi.GitHubAuthError("platform app down"))
        assert e.status_code == 412


# ---------------------------------------------------------------------------
# clone-token: tenant in the BODY; tenant_required / not_installed / success
# ---------------------------------------------------------------------------
CLONE_TOKEN_URL = "/api/internal/github/clone-token"
INTERNAL_TOKEN = "s3cret-internal-token"  # pragma: allowlist secret


@pytest.fixture
def clone_env(world):
    from routers import github_internal

    visibility = MagicMock(return_value="private")
    internal_mint = MagicMock()
    with (
        patch.object(
            github_internal,
            "get_settings",
            return_value=SimpleNamespace(internal_callers={"coding-containers": INTERNAL_TOKEN}),
        ),
        patch.object(github_internal, "get_repo_visibility", visibility),
        # push scope requires GitHub to CONFIRM private (#118); these repos are private.
        patch.object(
            github_internal,
            "get_repo_push_metadata",
            MagicMock(return_value=RepoPushMetadata(True, "main")),
        ),
        patch.object(github_internal, "get_scoped_clone_token", internal_mint),
    ):
        yield world, visibility, internal_mint


async def _clone(client, body):
    return await client.post(
        CLONE_TOKEN_URL, json=body, headers={"X-Internal-Token": INTERNAL_TOKEN}
    )


@pytest.mark.unit
class TestCloneTokenCustomer:
    async def test_customer_without_tenant_is_tenant_required(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(async_client, {"repo_url": f"https://github.com/{CUSTOMER}/{REPO}"})
        assert r.status_code == 200
        body = r.json()
        assert body["token"] is None
        assert body["source"] == "none"
        assert body["reason"] == "tenant_required"
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()

    async def test_customer_unbound_tenant_is_not_installed(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{CUSTOMER}/{REPO}", "tenant_id": TENANT_B},
        )
        assert r.status_code == 200
        assert r.json()["reason"] == "not_installed"
        assert r.json()["token"] is None
        internal_mint.assert_not_called()
        w.post.assert_not_called()

    async def test_customer_bound_tenant_gets_public_scoped_token(self, async_client, clone_env):
        w, visibility, internal_mint = clone_env
        r = await _clone(
            async_client,
            {
                "repo_url": f"https://github.com/{CUSTOMER}/{REPO}",
                "tenant_id": TENANT_A,
                "scope": "pr-read",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["token"] == "ghs_tenant_scoped"
        assert body["source"] == "app"
        assert body["reason"] is None
        assert w.post.call_args.kwargs["json"] == {
            "permissions": {"contents": "read", "pull_requests": "read"},
            "repositories": [REPO],
        }
        internal_mint.assert_not_called()
        visibility.assert_called_once_with(CUSTOMER, REPO, "ghs_tenant_scoped")

    async def test_customer_fr_down_is_503_not_a_tokenless_200(self, async_client, clone_env):
        w, _, _ = clone_env
        w.fr_get.side_effect = httpx.ConnectError("fr down")
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{CUSTOMER}/{REPO}", "tenant_id": TENANT_A},
        )
        assert r.status_code == 503
        w.post.assert_not_called()

    async def test_platform_owner_path_is_unchanged(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        r = await _clone(async_client, {"repo_url": f"https://github.com/{PLATFORM}/coding-github"})
        assert r.status_code == 200
        assert r.json()["token"] == "ghs_internal_clone"
        internal_mint.assert_called_once_with(PLATFORM, "coding-github", scope="clone")
        w.fr_get.assert_not_called()

    @pytest.mark.parametrize("scope", ["clone", "pr-read", "push"])
    async def test_platform_owner_for_a_customer_tenant_is_forbidden(
        self, async_client, clone_env, scope, enforce_platform_tenants
    ):
        w, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        r = await _clone(
            async_client,
            {
                "repo_url": f"https://github.com/{PLATFORM}/coding-github",
                "tenant_id": TENANT_A,
                "scope": scope,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert (body["token"], body["source"], body["reason"]) == (
            None,
            "none",
            "platform_repo_forbidden",
        )
        internal_mint.assert_not_called()
        w.fr_get.assert_not_called()

    async def test_platform_owner_for_a_platform_tenant_is_served(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/coding-github", "tenant_id": EXEC_TENANT},
        )
        assert r.json()["token"] == "ghs_internal_clone"
        assert r.json()["reason"] is None
        internal_mint.assert_called_once_with(PLATFORM, "coding-github", scope="clone")


# ---------------------------------------------------------------------------
# OBSERVE mode (CODING_GITHUB_PLATFORM_TENANT_ENFORCE=false, the default)
# ---------------------------------------------------------------------------
def _gate_lines(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("platform_repo_tenant_gate ")
    ]


@pytest.mark.unit
class TestPlatformGateObserveMode:
    @pytest.fixture
    def internal_app(self, monkeypatch):
        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: f"ghs_internal_{o}")

    def test_default_is_observe(self):
        from config.settings import Settings

        assert Settings().platform_tenant_enforce is False

    @pytest.mark.parametrize("tenant", [TENANT_A, "not-a-uuid"])
    def test_strict_resolver_does_not_raise_and_logs_would_refuse(
        self, world, internal_app, caplog, tenant
    ):
        caplog.set_level("WARNING", logger="services.tenant_binding")
        token = tb.resolve_tenant_repo_token(PLATFORM, "coding-github", tenant, tb.PERMS_AUTHORING)
        assert token == "ghs_internal_nonagenticai"
        # The tenant is forwarded into the ladder (so no tenantless path into it
        # exists), which re-gates: under OBSERVE that is a second line.
        assert _gate_lines(caplog) == [
            "platform_repo_tenant_gate decision=would_refuse "
            f"tenant={tenant} owner={PLATFORM} repo=coding-github "
            f"caller_path={site} caller=-"
            for site in ("resolve_tenant_repo_token", "resolve_github_token")
        ]

    @pytest.mark.parametrize("tenant", [None, "", "  "])
    def test_no_tenant_is_refused_even_in_observe_mode(self, world, internal_app, caplog, tenant):
        """⛔ OBSERVE never lets a TENANTLESS request through: it cannot be a
        platform tenant and every rung past the gate is the org identity."""
        caplog.set_level("WARNING", logger="services.tenant_binding")
        assert tb.platform_tenant_enforced() is False  # observe, really
        for call in (
            lambda: tb.resolve_tenant_repo_token(
                PLATFORM, "coding-github", tenant, tb.PERMS_AUTHORING
            ),
            lambda: gi.resolve_github_token(owner=PLATFORM, repo="coding-github", tenant_id=tenant),
        ):
            with pytest.raises(tb.PlatformRepoForbiddenError) as ei:
                call()
            assert (ei.value.code, ei.value.http_status) == ("platform_repo_forbidden", 403)
        assert [line.split(" caller_path=")[0] for line in _gate_lines(caplog)] == [
            f"platform_repo_tenant_gate decision=refused tenant=none owner={PLATFORM} "
            "repo=coding-github"
        ] * 2

    def test_ladder_does_not_raise_and_logs_would_refuse(self, internal_app, caplog):
        caplog.set_level("WARNING", logger="services.tenant_binding")
        resolved = gi.resolve_github_token(owner=PLATFORM, repo="coding-github", tenant_id=TENANT_A)
        assert resolved.token == "ghs_internal_nonagenticai"
        (line,) = _gate_lines(caplog)
        assert "decision=would_refuse" in line and "caller_path=resolve_github_token" in line
        assert f"tenant={TENANT_A}" in line

    def test_platform_tenant_logs_nothing(self, internal_app, caplog):
        caplog.set_level("DEBUG", logger="services.tenant_binding")
        gi.resolve_github_token(owner=PLATFORM, repo="coding-github", tenant_id=EXEC_TENANT)
        assert _gate_lines(caplog) == []

    def test_enforce_logs_refused(self, internal_app, caplog, enforce_platform_tenants):
        caplog.set_level("WARNING", logger="services.tenant_binding")
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.check_platform_repo_tenant(PLATFORM, "coding-github", None, site="x")
        assert _gate_lines(caplog) == [
            "platform_repo_tenant_gate decision=refused tenant=none "
            f"owner={PLATFORM} repo=coding-github caller_path=x caller=-"
        ]

    def test_validation_probe_does_not_raise(self, internal_app, armed_pats, caplog):
        from services.github_validation import _resolve_github_token as probe_token

        caplog.set_level("WARNING", logger="services.tenant_binding")
        assert probe_token(None, PLATFORM, "coding-github", TENANT_A).token is not None
        (line,) = _gate_lines(caplog)
        assert "decision=would_refuse" in line and "caller_path=validation_probe" in line

    async def test_clone_token_still_mints_and_logs(self, async_client, clone_env, caplog):
        _, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        caplog.set_level("WARNING", logger="services.tenant_binding")
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/coding-github", "tenant_id": TENANT_A},
        )
        assert r.json()["token"] == "ghs_internal_clone"
        assert r.json()["reason"] is None
        internal_mint.assert_called_once()
        (line,) = _gate_lines(caplog)
        assert "decision=would_refuse" in line and "caller_path=clone_token" in line
        assert line.endswith("caller=coding-containers")

    async def test_clone_token_without_tenant_mints_and_logs_tenant_none(
        self, async_client, clone_env, caplog
    ):
        """A tenantless platform request is GATED too (observe: logged, minted)."""
        _, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        caplog.set_level("WARNING", logger="services.tenant_binding")
        r = await _clone(async_client, {"repo_url": f"https://github.com/{PLATFORM}/coding-github"})
        assert r.json()["token"] == "ghs_internal_clone"
        internal_mint.assert_called_once()
        assert _gate_lines(caplog) == [
            "platform_repo_tenant_gate decision=would_refuse tenant=none "
            f"owner={PLATFORM} repo=coding-github caller_path=clone_token "
            "caller=coding-containers"
        ]


@pytest.mark.unit
@pytest.mark.usefixtures("enforce_platform_tenants")
class TestCloneTokenPlatformEnforced:
    async def test_without_tenant_is_forbidden(self, async_client, clone_env, caplog):
        _, _, internal_mint = clone_env
        caplog.set_level("WARNING", logger="services.tenant_binding")
        r = await _clone(async_client, {"repo_url": f"https://github.com/{PLATFORM}/coding-github"})
        assert r.status_code == 200
        body = r.json()
        assert (body["token"], body["source"], body["reason"]) == (
            None,
            "none",
            "platform_repo_forbidden",
        )
        internal_mint.assert_not_called()
        (line,) = _gate_lines(caplog)
        assert "decision=refused tenant=none" in line

    async def test_platform_tenant_is_still_minted(self, async_client, clone_env):
        _, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_clone", "2999-01-01T00:00:00Z")
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/coding-github", "tenant_id": EXEC_TENANT},
        )
        assert r.json()["token"] == "ghs_internal_clone"
        assert r.json()["reason"] is None
        internal_mint.assert_called_once_with(PLATFORM, "coding-github", scope="clone")


@pytest.mark.unit
@pytest.mark.usefixtures("enforce_platform_tenants")
class TestUpstreamPrForwardsTheTenant:
    async def test_platform_repo_upstream_pr_is_refused_for_a_customer_tenant(self, monkeypatch):
        """``create_upstream_pull_request`` must pass ``tenant_id`` to the ladder.

        MUTANT: drop ``tenant_id=tenant_id`` from its platform-owned
        ``resolve_github_token`` call -> the internal App mints and this passes 403.
        """
        from unittest.mock import AsyncMock

        minted: list[tuple[str, str]] = []
        monkeypatch.setattr(gi.settings, "system_github_owner", PLATFORM)
        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(
            ga, "get_installation_token_for", lambda o, r: minted.append((o, r)) or "ghs_i"
        )
        client = AsyncMock()
        client.get_workspace = AsyncMock(
            return_value={
                "user_id": "u",
                "original_repo_owner": PLATFORM,
                "original_repo_name": "coding-github",
                "system_fork_owner": PLATFORM,
            }
        )
        with patch("main.get_workspaces_client", return_value=client):
            integration = gi.GitHubIntegration(github_token="unused")
            result = await integration.create_upstream_pull_request(
                container_name="c",
                title="t",
                message="m",
                head_branch="b",
                tenant_id=TENANT_A,
            )
        assert result["success"] is False
        assert result["status_code"] == 403
        assert result["error_code"] == "platform_repo_forbidden"
        assert minted == []


# ---------------------------------------------------------------------------
# CODING_GITHUB_PLATFORM_TENANTS parsing
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPlatformTenantsSetting:
    def test_default_is_the_five_fr_internal_tenants(self):
        from config.settings import Settings

        ids = Settings().platform_tenant_ids
        # ⛔ LITERALS, deliberately — do not re-derive this with the same
        # comprehension `settings.py` uses. A pin that shares the producer's
        # shape is not a pin: when this read
        #     frozenset(f"…{i}" for i in range(1, 6))
        # a repo-wide `range(1, 6)` -> `range(1, 7)` edit changed the allowlist
        # AND the expectation together, and this test stayed green while a
        # sixth tenant silently gained access to every platform repo.
        # The authoritative list is DECLARED in
        # gitops-production clusters/strato/services/coding-github/deployment.yaml
        # (`CODING_GITHUB_PLATFORM_TENANTS`); these literals must equal it.
        assert ids == frozenset(
            {
                "0195f48b-d000-7000-8000-000000000001",
                "0195f48b-d000-7000-8000-000000000002",
                "0195f48b-d000-7000-8000-000000000003",
                "0195f48b-d000-7000-8000-000000000004",
                "0195f48b-d000-7000-8000-000000000005",
            }
        )
        assert len(ids) == 5
        assert EXEC_TENANT in ids

    def test_env_list_is_normalised_to_lowercase(self, monkeypatch):
        from config.settings import Settings

        monkeypatch.setenv(
            "CODING_GITHUB_PLATFORM_TENANTS", " 0195F48B-D000-7000-8000-000000000003 , ,"
        )
        assert Settings().platform_tenant_ids == frozenset({EXEC_TENANT})

    @pytest.mark.parametrize("raw", ["", " ", ",,", "exec", f"{EXEC_TENANT},nope"])
    def test_empty_or_invalid_is_a_startup_error_naming_the_key(self, monkeypatch, raw):
        from config.settings import Settings

        monkeypatch.setenv("CODING_GITHUB_PLATFORM_TENANTS", raw)
        with pytest.raises(RuntimeError, match="CODING_GITHUB_PLATFORM_TENANTS"):
            Settings()

    def test_membership_uses_the_configured_list(self, monkeypatch):
        from config.settings import Settings

        assert tb.is_platform_tenant(EXEC_TENANT)
        assert not tb.is_platform_tenant(TENANT_A)
        assert not tb.is_platform_tenant(None)
        assert not tb.is_platform_tenant("")

        monkeypatch.setenv("CODING_GITHUB_PLATFORM_TENANTS", TENANT_A)
        configured = Settings()
        monkeypatch.setattr(tb._settings_module, "get_settings", lambda: configured)
        assert tb.is_platform_tenant(TENANT_A)
        assert tb.is_platform_tenant(TENANT_A.upper())
        assert not tb.is_platform_tenant(EXEC_TENANT)


# ---------------------------------------------------------------------------
# push scope: the ONLY write scope, contents:write on ONE repo, same tenant gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPushScope:
    async def test_push_mints_contents_write_repo_scoped_for_bound_tenant(
        self, async_client, clone_env
    ):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {
                "repo_url": f"https://github.com/{CUSTOMER}/{REPO}",
                "tenant_id": TENANT_A,
                "scope": "push",
            },
        )
        assert r.status_code == 200
        assert r.json()["token"] == "ghs_tenant_scoped"
        assert w.post.call_args.kwargs["json"] == {
            "permissions": {"contents": "write"},
            "repositories": [REPO],
        }
        # Went through the tenant gate: fr asked, GitHub's installation compared.
        assert w.fr_get.call_args.kwargs["params"]["tenant_id"] == TENANT_A
        w.integration.get_repo_installation.assert_called_once_with(CUSTOMER, REPO)
        internal_mint.assert_not_called()

    async def test_push_for_unbound_tenant_is_refused(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {
                "repo_url": f"https://github.com/{CUSTOMER}/{REPO}",
                "tenant_id": TENANT_B,
                "scope": "push",
            },
        )
        assert r.status_code == 200
        assert r.json()["token"] is None
        assert r.json()["reason"] == "not_installed"
        w.post.assert_not_called()
        internal_mint.assert_not_called()

    async def test_push_without_tenant_is_tenant_required(self, async_client, clone_env):
        w, _, _ = clone_env
        r = await _clone(
            async_client, {"repo_url": f"https://github.com/{CUSTOMER}/{REPO}", "scope": "push"}
        )
        assert r.json()["reason"] == "tenant_required"
        w.post.assert_not_called()

    def test_cache_key_separates_push_from_clone_and_pr_read(self, world):
        world.post.side_effect = [
            _resp(201, {"token": "ghs_read", "expires_at": "2999-01-01T00:00:00Z"}),
            _resp(201, {"token": "ghs_write", "expires_at": "2999-01-01T00:00:00Z"}),
            _resp(201, {"token": "ghs_prread", "expires_at": "2999-01-01T00:00:00Z"}),
        ]
        perms = ga._SCOPE_PERMISSIONS
        clone = tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, dict(perms["clone"]))
        push = tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, dict(perms["push"]))
        prr = tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, dict(perms["pr-read"]))
        # Repeat reads are served from cache and never cross scopes.
        assert tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, dict(perms["clone"])) == (
            "ghs_read"
        )
        assert tb.resolve_tenant_repo_token(CUSTOMER, REPO, TENANT_A, dict(perms["push"])) == (
            "ghs_write"
        )
        assert (clone, push, prr) == ("ghs_read", "ghs_write", "ghs_prread")
        assert world.post.call_count == 3
        assert set(ga._public_scoped_token_by_repo) == {
            (BOUND_ID, CUSTOMER, REPO, "contents:read"),
            (BOUND_ID, CUSTOMER, REPO, "contents:write"),
            (BOUND_ID, CUSTOMER, REPO, "contents:read,pull_requests:read"),
        }

    async def test_platform_push_uses_the_internal_app(self, async_client, clone_env):
        w, _, internal_mint = clone_env
        internal_mint.return_value = ga.ScopedToken("ghs_internal_push", "2999-01-01T00:00:00Z")
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/coding-github", "scope": "push"},
        )
        assert r.json()["token"] == "ghs_internal_push"
        internal_mint.assert_called_once_with(PLATFORM, "coding-github", scope="push")
        w.fr_get.assert_not_called()

    def test_internal_push_mint_is_contents_write_and_cached_apart(self, world):
        world.post.side_effect = [
            _resp(201, {"token": "ghs_i_read", "expires_at": "2999-01-01T00:00:00Z"}),
            _resp(201, {"token": "ghs_i_write", "expires_at": "2999-01-01T00:00:00Z"}),
        ]
        world.integration.get_repo_installation.side_effect = None
        world.integration.get_repo_installation.return_value = SimpleNamespace(id=99)
        read = ga.get_scoped_clone_token(PLATFORM, "coding-github", scope="clone")
        write = ga.get_scoped_clone_token(PLATFORM, "coding-github", scope="push")
        assert (read.token, write.token) == ("ghs_i_read", "ghs_i_write")
        assert world.post.call_args.kwargs["json"] == {
            "permissions": {"contents": "write"},
            "repositories": ["coding-github"],
        }
        assert ga.get_scoped_clone_token(PLATFORM, "coding-github", scope="clone").token == (
            "ghs_i_read"
        )
        assert ga._token_by_id == {}  # never the PR-opening cache

    @pytest.mark.parametrize(
        "scope", ["pull_requests:write", "workflows", "write", "admin", "contents:write"]
    )
    async def test_no_other_write_scope_is_grantable(self, async_client, clone_env, scope):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {
                "repo_url": f"https://github.com/{CUSTOMER}/{REPO}",
                "tenant_id": TENANT_A,
                "scope": scope,
            },
        )
        assert r.status_code == 422
        w.post.assert_not_called()
        internal_mint.assert_not_called()


# ---------------------------------------------------------------------------
# CODING_GITHUB_PLATFORM_REPO_ALLOWLIST: a NON-platform tenant on EXACTLY the
# repos it is granted, served ONLY by a one-repo internal-App token.
# ---------------------------------------------------------------------------
DEMO_TENANT = "0195f48b-d000-7000-8000-000000000103"  # 'demo' — NOT a platform tenant
FIXTURE = "aa-bugfix-capstone-fixture"
DEMO_TOKEN = "ghs_demo_fixture_only"
ALLOWLIST_ENV = "CODING_GITHUB_PLATFORM_REPO_ALLOWLIST"


@pytest.fixture
def demo_allowlist(monkeypatch):
    """The live Settings carries {demo: {nonagenticai/<fixture>}} (as boot would parse it)."""
    from config.settings import parse_platform_repo_allowlist

    real = tb._settings_module.get_settings()
    grants = parse_platform_repo_allowlist(
        f'{{"{DEMO_TENANT}": ["{PLATFORM}/{FIXTURE}"]}}',
        platform_owner=PLATFORM,
        platform_tenants=real.platform_tenant_ids,
    )
    monkeypatch.setattr(real, "_platform_repo_grants", grants)
    return grants


@pytest.fixture
def scoped_world(world, demo_allowlist, armed_pats):
    """The world, with a recognisable scoped token and every PAT/org rung armed."""
    world.post.return_value = _resp(
        201, {"token": DEMO_TOKEN, "expires_at": "2999-01-01T00:00:00Z"}
    )
    world.integration.get_repo_installation.side_effect = None
    world.integration.get_repo_installation.return_value = SimpleNamespace(id=99)
    world.integration.get_access_token.return_value = SimpleNamespace(token="ghs_ORG_WIDE")
    return world


def _assert_repo_scoped_only(w, permissions):
    """The mint was a one-repo POST and the org-wide token cache was never touched."""
    assert w.post.call_count == 1
    assert w.post.call_args.args[0].endswith("/app/installations/99/access_tokens")
    assert w.post.call_args.kwargs["json"] == {
        "permissions": permissions,
        "repositories": [FIXTURE],
    }
    assert ga._token_by_id == {}
    w.integration.get_access_token.assert_not_called()


@pytest.mark.usefixtures("enforce_platform_tenants")
class TestPlatformRepoAllowlist:
    def test_verdicts(self, demo_allowlist):
        assert tb.platform_repo_grant(DEMO_TENANT, PLATFORM, FIXTURE) == "scoped"
        assert tb.platform_repo_grant(DEMO_TENANT.upper(), "NonAgenticAI", FIXTURE.upper()) == (
            "scoped"
        )
        assert tb.platform_repo_grant(EXEC_TENANT, PLATFORM, "coding-github") == "platform"
        for tenant, owner, repo in [
            (DEMO_TENANT, PLATFORM, "coding-github"),
            (DEMO_TENANT, PLATFORM, f"{FIXTURE}-2"),
            (DEMO_TENANT, PLATFORM, "aa-bugfix-capstone"),
            (DEMO_TENANT, PLATFORM, None),
            (DEMO_TENANT, CUSTOMER, FIXTURE),
            (TENANT_A, PLATFORM, FIXTURE),
            (None, PLATFORM, FIXTURE),
            ("", PLATFORM, FIXTURE),
            ("not-a-uuid", PLATFORM, FIXTURE),
        ]:
            assert tb.platform_repo_grant(tenant, owner, repo) == "refuse", (tenant, owner, repo)

    def test_resolve_github_token_demo_gets_one_repo_token(self, scoped_world, armed_pats):
        resolved = gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=DEMO_TENANT)
        assert (resolved.token, resolved.is_app, resolved.rung) == (
            DEMO_TOKEN,
            True,
            "app_repo_scoped",
        )
        assert resolved.token not in armed_pats
        _assert_repo_scoped_only(scoped_world, dict(tb.PERMS_AUTHORING))

    def test_resolve_tenant_repo_token_demo_gets_one_repo_token(self, scoped_world):
        token = tb.resolve_tenant_repo_token(PLATFORM, FIXTURE, DEMO_TENANT, tb.PERMS_REPO_READ)
        assert token == DEMO_TOKEN
        _assert_repo_scoped_only(scoped_world, {"contents": "read"})

    def test_validation_probe_demo_gets_one_repo_token_not_env_pat(self, scoped_world):
        from services.github_validation import _resolve_github_token as probe_token

        resolved = probe_token(None, PLATFORM, FIXTURE, DEMO_TENANT)
        assert (resolved.token, resolved.is_app) == (DEMO_TOKEN, True)
        _assert_repo_scoped_only(scoped_world, {"contents": "read"})

    def test_repo_case_variant_is_still_the_granted_repo(self, scoped_world):
        resolved = gi.resolve_github_token(
            owner="NonAgenticAI", repo=FIXTURE, tenant_id=DEMO_TENANT
        )
        assert resolved.token == DEMO_TOKEN
        _assert_repo_scoped_only(scoped_world, dict(tb.PERMS_AUTHORING))

    @pytest.mark.parametrize("repo", ["coding-github", f"{FIXTURE}-2", "aa-bugfix-capstone"])
    def test_demo_on_any_other_platform_repo_is_403(self, scoped_world, repo):
        from services.github_validation import _resolve_github_token as probe_token

        for call in (
            lambda: gi.resolve_github_token(owner=PLATFORM, repo=repo, tenant_id=DEMO_TENANT),
            lambda: tb.resolve_tenant_repo_token(PLATFORM, repo, DEMO_TENANT, tb.PERMS_REPO_READ),
            lambda: probe_token(None, PLATFORM, repo, DEMO_TENANT),
        ):
            with pytest.raises(tb.PlatformRepoForbiddenError) as ei:
                call()
            assert ei.value.code == "platform_repo_forbidden"
        scoped_world.post.assert_not_called()
        scoped_world.integration.get_access_token.assert_not_called()

    def test_second_non_platform_tenant_cannot_use_demos_grant(self, scoped_world):
        with pytest.raises(tb.PlatformRepoForbiddenError):
            gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=TENANT_A)
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.mint_allowlisted_platform_token(
                PLATFORM, FIXTURE, tb.PERMS_REPO_READ, tenant_id=TENANT_A
            )
        scoped_world.post.assert_not_called()

    def test_platform_tenant_is_unchanged_org_installation(self, scoped_world):
        resolved = gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=EXEC_TENANT)
        assert (resolved.token, resolved.rung) == ("ghs_ORG_WIDE", "app")
        scoped_world.post.assert_not_called()

    @pytest.mark.parametrize(
        "failure",
        ["refused", "unreachable", "not_installed"],
    )
    def test_scoped_mint_failure_raises_never_falls_through(
        self, scoped_world, armed_pats, failure
    ):
        if failure == "refused":
            scoped_world.post.return_value = _resp(422, {"message": "no"})
        elif failure == "unreachable":
            scoped_world.post.side_effect = httpx.ConnectError("down")
        else:
            from github import UnknownObjectException

            scoped_world.integration.get_repo_installation.side_effect = UnknownObjectException(
                404, {"message": "Not Found"}, {}
            )
        from services.github_validation import _resolve_github_token as probe_token

        for call in (
            lambda: gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=DEMO_TENANT),
            lambda: tb.resolve_tenant_repo_token(
                PLATFORM, FIXTURE, DEMO_TENANT, tb.PERMS_REPO_READ
            ),
            lambda: probe_token(None, PLATFORM, FIXTURE, DEMO_TENANT),
        ):
            with pytest.raises(tb.PlatformRepoScopedMintFailedError) as ei:
                call()
            assert (ei.value.code, ei.value.http_status) == (
                "platform_repo_scoped_mint_failed",
                502,
            )
        assert ga._token_by_id == {}
        scoped_world.integration.get_access_token.assert_not_called()

    def test_observe_mode_still_serves_only_the_scoped_token(self, scoped_world, monkeypatch):
        monkeypatch.setattr(tb._settings_module.get_settings(), "platform_tenant_enforce", False)
        resolved = gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=DEMO_TENANT)
        assert resolved.token == DEMO_TOKEN
        _assert_repo_scoped_only(scoped_world, dict(tb.PERMS_AUTHORING))

    def test_no_allowlist_means_no_grant(self, scoped_world, monkeypatch):
        monkeypatch.setattr(tb._settings_module.get_settings(), "_platform_repo_grants", {})
        with pytest.raises(tb.PlatformRepoForbiddenError):
            gi.resolve_github_token(owner=PLATFORM, repo=FIXTURE, tenant_id=DEMO_TENANT)
        scoped_world.post.assert_not_called()


@pytest.mark.unit
@pytest.mark.usefixtures("enforce_platform_tenants")
class TestCloneTokenAllowlist:
    async def test_demo_fixture_clone_is_one_repo_internal_token(
        self, async_client, clone_env, scoped_world
    ):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/{FIXTURE}", "tenant_id": DEMO_TENANT},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["token"], body["source"], body["reason"]) == (DEMO_TOKEN, "app", None)
        internal_mint.assert_not_called()
        _assert_repo_scoped_only(w, {"contents": "read"})

    async def test_demo_other_platform_repo_is_forbidden(
        self, async_client, clone_env, scoped_world
    ):
        w, _, internal_mint = clone_env
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/coding-github", "tenant_id": DEMO_TENANT},
        )
        assert r.json()["reason"] == "platform_repo_forbidden"
        assert r.json()["token"] is None
        internal_mint.assert_not_called()
        w.post.assert_not_called()

    async def test_demo_fixture_mint_failure_is_502_not_a_tokenless_200(
        self, async_client, clone_env, scoped_world
    ):
        w, _, internal_mint = clone_env
        w.post.return_value = _resp(422, {"message": "no"})
        r = await _clone(
            async_client,
            {"repo_url": f"https://github.com/{PLATFORM}/{FIXTURE}", "tenant_id": DEMO_TENANT},
        )
        assert r.status_code == 502, r.text
        assert "platform_repo_scoped_mint_failed" in r.text
        internal_mint.assert_not_called()
        assert ga._token_by_id == {}


class TestPlatformRepoAllowlistSetting:
    def _settings(self, monkeypatch, raw):
        from config.settings import Settings

        monkeypatch.setenv(ALLOWLIST_ENV, raw)
        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_OWNER", PLATFORM)
        return Settings()

    def test_unset_is_no_grants(self, monkeypatch):
        from config.settings import Settings

        monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
        assert Settings().platform_repo_grants == {}

    def test_valid_is_canonical_lowercase(self, monkeypatch):
        s = self._settings(
            monkeypatch, f'{{"{DEMO_TENANT.upper()}": ["NonAgenticAI/AA-Bugfix-Capstone-Fixture"]}}'
        )
        assert s.platform_repo_grants == {DEMO_TENANT: frozenset({f"{PLATFORM}/{FIXTURE}"})}

    @pytest.mark.parametrize(
        "raw",
        [
            "{not json",
            f'["{PLATFORM}/{FIXTURE}"]',
            f'{{"demo": ["{PLATFORM}/{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": ["{CUSTOMER}/{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": ["{PLATFORM}evil/{FIXTURE}"]}}',
            f'{{"{EXEC_TENANT}": ["{PLATFORM}/{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": ["{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": ["{PLATFORM}/{FIXTURE}/x"]}}',
            f'{{"{DEMO_TENANT}": ["{PLATFORM}/"]}}',
            f'{{"{DEMO_TENANT}": ["{PLATFORM}/*"]}}',
            f'{{"{DEMO_TENANT}": ["{PLATFORM}/.."]}}',
            f'{{"{DEMO_TENANT}": [" {PLATFORM}/{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": "{PLATFORM}/{FIXTURE}"}}',
            f'{{"{DEMO_TENANT}": [42]}}',
            f'{{"{DEMO_TENANT}": [], "{DEMO_TENANT}": ["{PLATFORM}/{FIXTURE}"]}}',
            f'{{"{DEMO_TENANT}": [], "{DEMO_TENANT.upper()}": ["{PLATFORM}/{FIXTURE}"]}}',
        ],
    )
    def test_bad_value_fails_boot_naming_the_key(self, monkeypatch, raw):
        with pytest.raises(RuntimeError, match=ALLOWLIST_ENV):
            self._settings(monkeypatch, raw)


# ---------------------------------------------------------------------------
# Tenantless / ownerless requests never get an org credential (H1-H3), and the
# allowlist grant is exact: no ".git" / "owner/repo" forms, whitespace stripped.
# ---------------------------------------------------------------------------
@pytest.fixture
def org_rungs(monkeypatch):
    """Every ORG-wide rung armed: the home + per-owner installation tokens."""
    calls: list[str] = []

    def _home():
        calls.append("home")
        return "ghs_HOME_INSTALLATION_ORG_WIDE"

    def _for(o, r):
        calls.append(f"for:{o}/{r}")
        return "ghs_PER_OWNER_ORG_WIDE"

    monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
    monkeypatch.setattr(ga, "get_installation_token", _home)
    monkeypatch.setattr(ga, "get_installation_token_for", _for)
    return calls


class TestNoOrgCredentialWithoutTenantOrOwner:
    @pytest.mark.parametrize("owner", [None, "", "   "])
    def test_ownerless_resolution_raises_and_mints_nothing(self, org_rungs, armed_pats, owner):
        """H2: ``owner=None`` used to return the HOME installation token, ungated.
        Kills the mutant "restore the ownerless home-token / PAT rungs"."""
        for tenant in (None, EXEC_TENANT):
            with pytest.raises(tb.OwnerRequiredError, match="owner is required"):
                gi.resolve_github_token(owner=owner, repo="coding-github", tenant_id=tenant)
        assert org_rungs == []

    def test_no_arg_github_integration_cannot_be_constructed(self, org_rungs, armed_pats):
        with pytest.raises(gi.GitHubAuthError):
            gi.GitHubIntegration()
        assert org_rungs == []

    def test_platform_tenant_with_owner_is_unchanged(self, org_rungs):
        resolved = gi.resolve_github_token(owner=PLATFORM, repo="x", tenant_id=EXEC_TENANT)
        assert (resolved.token, resolved.rung) == ("ghs_PER_OWNER_ORG_WIDE", "app")

    @pytest.mark.parametrize("tenant", [None, "", "  "])
    def test_validation_probe_refuses_a_tenantless_platform_repo(
        self, org_rungs, armed_pats, caplog, tenant
    ):
        """H3: the read-only probe used to skip the gate for ``tenant_id=None`` and
        hand out the env PAT / org App. Kills the mutant "restore the None skip"."""
        from services.github_validation import _resolve_github_token as probe_token

        caplog.set_level("WARNING", logger="services.tenant_binding")
        with pytest.raises(tb.PlatformRepoForbiddenError) as ei:
            probe_token(None, PLATFORM, "coding-github", tenant)
        assert (ei.value.code, ei.value.http_status) == ("platform_repo_forbidden", 403)
        assert org_rungs == []
        (line,) = _gate_lines(caplog)
        assert "decision=refused tenant=none" in line and "caller_path=validation_probe" in line

    def test_validation_probe_platform_tenant_still_gets_a_credential(self, org_rungs, armed_pats):
        """A platform tenant gets the internal App token — never an env PAT, even
        with every legacy PAT variable armed (the env-first rung was removed)."""
        from services.github_validation import _resolve_github_token as probe_token

        resolved = probe_token(None, PLATFORM, "coding-github", EXEC_TENANT)
        assert resolved.token == "ghs_PER_OWNER_ORG_WIDE"
        assert resolved.token not in armed_pats

    def test_validation_probe_with_no_owner_gets_no_credential(self, org_rungs, armed_pats):
        from services.github_validation import _resolve_github_token as probe_token

        assert probe_token(None, "", "coding-github", None).token is None
        assert org_rungs == []


@pytest.mark.usefixtures("enforce_platform_tenants")
class TestAllowlistGrantIsExact:
    @pytest.mark.parametrize(
        "repo",
        [
            f"{FIXTURE}.git",
            f"{FIXTURE}.GIT",
            f"{PLATFORM}/{FIXTURE}",
            f"{PLATFORM}/{FIXTURE}.git",
            f"{FIXTURE}/",
        ],
    )
    def test_git_suffix_and_owner_repo_forms_are_refused(self, scoped_world, repo):
        """Kills the mutant "accept a .git suffix / owner/repo form" in the grant."""
        assert tb.platform_repo_grant(DEMO_TENANT, PLATFORM, repo) == "refuse", repr(repo)
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.mint_allowlisted_platform_token(
                PLATFORM, repo, tb.PERMS_REPO_READ, tenant_id=DEMO_TENANT
            )
        with pytest.raises(tb.PlatformRepoForbiddenError):
            gi.resolve_github_token(owner=PLATFORM, repo=repo, tenant_id=DEMO_TENANT)
        scoped_world.post.assert_not_called()

    def test_surrounding_whitespace_is_stripped_before_grant_and_mint(self, scoped_world):
        """The name that was authorised is exactly the name minted for. Kills the
        mutant "drop the strip in mint_allowlisted_platform_token": the POST
        then carries ``repositories=[' fixture ']``."""
        assert tb.platform_repo_grant(DEMO_TENANT, f" {PLATFORM} ", f"\t{FIXTURE} ") == "scoped"
        token = tb.mint_allowlisted_platform_token(
            f" {PLATFORM} ", f" {FIXTURE}\t", tb.PERMS_REPO_READ, tenant_id=f" {DEMO_TENANT} "
        )
        assert token.token == DEMO_TOKEN
        _assert_repo_scoped_only(scoped_world, {"contents": "read"})
        installation_lookup = scoped_world.integration.get_repo_installation.call_args
        assert installation_lookup.args == (PLATFORM, FIXTURE)


# ---------------------------------------------------------------------------
# READ-ONLY grants: ``"owner/repo:read"`` in CODING_GITHUB_PLATFORM_REPO_ALLOWLIST.
# The tenant may CLONE / READ the repo; every write (push, PR create, issue and
# label writes, merge, PR actions …) is refused 403 ``platform_repo_read_only``.
# ---------------------------------------------------------------------------
RO_TENANT = "5f868d42-03f8-51f0-a93c-3e83070dc196"  # local conveyor tenant (NOT platform)
RO_REPO = "fr"
FULL_REPO = "ems"
RO_TOKEN = "ghs_read_only_one_repo"
_WRITE_SETS = {
    "authoring": tb.PERMS_AUTHORING,
    "issues": tb.PERMS_ISSUES,
    "pr_actions": tb.PERMS_PR_ACTIONS,
    "merge": tb.PERMS_MERGE,
    "push": {"contents": "write"},
}
_READ_SETS = {
    "clone": tb.PERMS_CLONE,
    "pr_read": tb.PERMS_PR_READ,
    "pr_checks": tb.PERMS_PR_CHECKS,
    "actions_read": tb.PERMS_ACTIONS_READ,
}


@pytest.fixture
def ro_allowlist(monkeypatch):
    """{conveyor: [nonagenticai/ems (FULL), nonagenticai/fr:read]} as boot parses it."""
    from config.settings import parse_platform_repo_allowlist_modes

    real = tb._settings_module.get_settings()
    grants, read_only = parse_platform_repo_allowlist_modes(
        f'{{"{RO_TENANT}": ["{PLATFORM}/{FULL_REPO}", "{PLATFORM}/{RO_REPO}:read"]}}',
        platform_owner=PLATFORM,
        platform_tenants=real.platform_tenant_ids,
    )
    monkeypatch.setattr(real, "_platform_repo_grants", grants)
    monkeypatch.setattr(real, "_platform_repo_read_only_grants", read_only)
    return grants, read_only


@pytest.fixture
def ro_world(world, ro_allowlist, armed_pats):
    world.post.return_value = _resp(201, {"token": RO_TOKEN, "expires_at": "2999-01-01T00:00:00Z"})
    world.integration.get_repo_installation.side_effect = None
    world.integration.get_repo_installation.return_value = SimpleNamespace(id=99)
    world.integration.get_access_token.return_value = SimpleNamespace(token="ghs_ORG_WIDE")
    return world


def _assert_one_repo_mint(w, repo, permissions):
    assert w.post.call_count == 1
    assert w.post.call_args.kwargs["json"] == {"permissions": permissions, "repositories": [repo]}
    assert ga._token_by_id == {}
    w.integration.get_access_token.assert_not_called()


class TestReadOnlyGrantSetting:
    def _settings(self, monkeypatch, raw):
        from config.settings import Settings

        monkeypatch.setenv(ALLOWLIST_ENV, raw)
        monkeypatch.setenv("CODING_GITHUB_SYSTEM_GITHUB_OWNER", PLATFORM)
        return Settings()

    def test_read_suffix_is_a_read_only_grant(self, monkeypatch):
        s = self._settings(
            monkeypatch,
            f'{{"{RO_TENANT}": ["{PLATFORM}/ems", "NonAgenticAI/FR:read", '
            f'"{PLATFORM}/n8n-nodes-fr:read"]}}',
        )
        assert s.platform_repo_grants == {
            RO_TENANT: frozenset({f"{PLATFORM}/ems", f"{PLATFORM}/fr", f"{PLATFORM}/n8n-nodes-fr"})
        }
        assert s.platform_repo_read_only_grants == {
            RO_TENANT: frozenset({f"{PLATFORM}/fr", f"{PLATFORM}/n8n-nodes-fr"})
        }

    def test_full_grants_only_have_no_read_only_entries(self, monkeypatch):
        s = self._settings(monkeypatch, f'{{"{RO_TENANT}": ["{PLATFORM}/ems"]}}')
        assert s.platform_repo_read_only_grants == {}

    @pytest.mark.parametrize(
        "entries",
        [
            # both forms of one repo: ambiguous
            f'["{PLATFORM}/fr", "{PLATFORM}/fr:read"]',
            f'["{PLATFORM}/fr:read", "NonAgenticAI/FR"]',
            # the same repo twice in one form
            f'["{PLATFORM}/fr:read", "{PLATFORM}/FR:read"]',
            # only ':read' is a known suffix
            f'["{PLATFORM}/fr:write"]',
            f'["{PLATFORM}/fr:READ"]',
            f'["{PLATFORM}/fr:read:read"]',
            f'["{PLATFORM}/fr:"]',
            f'["{PLATFORM}:read/fr"]',
            # the suffix does not relax the owner / shape rules
            f'["{CUSTOMER}/fr:read"]',
            f'["{PLATFORM}/:read"]',
            '["fr:read"]',
            f'["{PLATFORM}/..:read"]',
        ],
    )
    def test_bad_read_only_entry_fails_boot(self, monkeypatch, entries):
        with pytest.raises(RuntimeError, match=ALLOWLIST_ENV):
            self._settings(monkeypatch, f'{{"{RO_TENANT}": {entries}}}')


@pytest.mark.usefixtures("enforce_platform_tenants")
class TestReadOnlyGrantMint:
    def test_verdicts(self, ro_allowlist):
        assert tb.platform_repo_grant(RO_TENANT, PLATFORM, RO_REPO) == "scoped"
        assert tb.is_read_only_grant(RO_TENANT, PLATFORM, RO_REPO)
        assert tb.is_read_only_grant(RO_TENANT.upper(), "NonAgenticAI", "FR")
        assert not tb.is_read_only_grant(RO_TENANT, PLATFORM, FULL_REPO)
        assert not tb.is_read_only_grant(EXEC_TENANT, PLATFORM, RO_REPO)
        assert not tb.is_read_only_grant(TENANT_A, PLATFORM, RO_REPO)
        assert not tb.is_read_only_grant(None, PLATFORM, RO_REPO)

    @pytest.mark.parametrize("name", sorted(_READ_SETS))
    def test_read_permission_sets_mint_one_repo_token(self, ro_world, name):
        tok = tb.mint_allowlisted_platform_token(
            PLATFORM, RO_REPO, _READ_SETS[name], tenant_id=RO_TENANT
        )
        assert tok.token == RO_TOKEN
        _assert_one_repo_mint(ro_world, RO_REPO, dict(_READ_SETS[name]))

    @pytest.mark.parametrize("name", sorted(_WRITE_SETS))
    def test_write_permission_sets_refused_before_github(self, ro_world, name):
        """MUTANT: drop the chokepoint call in mint_allowlisted_platform_token -> minted."""
        with pytest.raises(tb.PlatformRepoReadOnlyError) as ei:
            tb.mint_allowlisted_platform_token(
                PLATFORM, RO_REPO, _WRITE_SETS[name], tenant_id=RO_TENANT
            )
        assert (ei.value.code, ei.value.http_status) == ("platform_repo_read_only", 403)
        ro_world.post.assert_not_called()
        ro_world.integration.get_repo_installation.assert_not_called()

    def test_pr_authoring_ladder_default_is_refused(self, ro_world):
        """resolve_github_token with no permissions = PERMS_AUTHORING (the PR-create path)."""
        with pytest.raises(tb.PlatformRepoReadOnlyError):
            gi.resolve_github_token(owner=PLATFORM, repo=RO_REPO, tenant_id=RO_TENANT)
        with pytest.raises(tb.PlatformRepoReadOnlyError):
            tb.resolve_tenant_repo_token(PLATFORM, RO_REPO, RO_TENANT, tb.PERMS_ISSUES)
        ro_world.post.assert_not_called()

    def test_read_ladders_still_serve_the_read_only_repo(self, ro_world):
        from services.github_validation import _resolve_github_token as probe_token

        assert tb.resolve_tenant_repo_token(PLATFORM, RO_REPO, RO_TENANT, tb.PERMS_REPO_READ) == (
            RO_TOKEN
        )
        assert probe_token(None, PLATFORM, RO_REPO, RO_TENANT).token == RO_TOKEN

    def test_full_grant_in_the_same_tenant_still_authors(self, ro_world):
        resolved = gi.resolve_github_token(owner=PLATFORM, repo=FULL_REPO, tenant_id=RO_TENANT)
        assert (resolved.token, resolved.rung) == (RO_TOKEN, "app_repo_scoped")
        _assert_one_repo_mint(ro_world, FULL_REPO, dict(tb.PERMS_AUTHORING))

    def test_unlisted_repo_is_still_forbidden(self, ro_world):
        with pytest.raises(tb.PlatformRepoForbiddenError):
            tb.mint_allowlisted_platform_token(
                PLATFORM, "coding-github", tb.PERMS_CLONE, tenant_id=RO_TENANT
            )
        ro_world.post.assert_not_called()


@pytest.mark.unit
@pytest.mark.usefixtures("enforce_platform_tenants")
class TestReadOnlyGrantRoutes:
    async def _clone(self, client, repo, scope):
        return await _clone(
            client,
            {
                "repo_url": f"https://github.com/{PLATFORM}/{repo}",
                "tenant_id": RO_TENANT,
                "scope": scope,
            },
        )

    @pytest.mark.parametrize(
        ("scope", "perms"),
        [
            ("clone", {"contents": "read"}),
            ("pr-read", {"contents": "read", "pull_requests": "read"}),
        ],
    )
    async def test_read_scopes_mint_read_only(
        self, async_client, clone_env, ro_world, scope, perms
    ):
        _, _, internal_mint = clone_env
        r = await self._clone(async_client, RO_REPO, scope)
        assert r.status_code == 200, r.text
        assert (r.json()["token"], r.json()["source"], r.json()["reason"]) == (
            RO_TOKEN,
            "app",
            None,
        )
        internal_mint.assert_not_called()
        _assert_one_repo_mint(ro_world, RO_REPO, perms)

    async def test_push_scope_is_403_read_only(self, async_client, clone_env, ro_world):
        _, _, internal_mint = clone_env
        r = await self._clone(async_client, RO_REPO, "push")
        assert r.status_code == 403, r.text
        assert "platform_repo_read_only" in r.text
        assert "scope='push' is refused" in r.text  # the explicit route check, not the mint
        internal_mint.assert_not_called()
        ro_world.post.assert_not_called()

    async def test_push_refused_at_the_mint_even_without_the_route_check(
        self, async_client, clone_env, ro_world, monkeypatch
    ):
        """The route check removed (a future route forgetting it): the mint still refuses."""
        from routers import github_internal

        monkeypatch.setattr(github_internal, "is_read_only_grant", lambda *a: False)
        r = await self._clone(async_client, RO_REPO, "push")
        assert r.status_code == 403, r.text
        assert "platform_repo_read_only" in r.text
        ro_world.post.assert_not_called()

    async def test_full_grant_still_pushes(self, async_client, clone_env, ro_world):
        r = await self._clone(async_client, FULL_REPO, "push")
        assert r.status_code == 200, r.text
        assert r.json()["token"] == RO_TOKEN
        _assert_one_repo_mint(ro_world, FULL_REPO, {"contents": "write"})

    async def test_unlisted_repo_still_platform_repo_forbidden(
        self, async_client, clone_env, ro_world
    ):
        r = await self._clone(async_client, "coding-github", "clone")
        assert (r.json()["token"], r.json()["reason"]) == (None, "platform_repo_forbidden")
        ro_world.post.assert_not_called()

    async def test_create_pull_request_is_403_read_only(self, async_client, ro_world):
        from coding_client.testing import override_tenant_auth

        from main import app

        undo = override_tenant_auth(app, tenant_id=RO_TENANT)
        try:
            r = await async_client.post(
                "/api/repositories/create-pull-request",
                json={
                    "fork_owner": PLATFORM,
                    "fork_repo": RO_REPO,
                    "original_owner": PLATFORM,
                    "original_repo": RO_REPO,
                    "head_branch": "feat/x",
                    "base_branch": "main",
                    "title": "t",
                    "message": "m",
                },
            )
        finally:
            undo()
        assert r.status_code == 403, r.text
        assert r.json()["error"]["details"]["error_code"] == "platform_repo_read_only"
        ro_world.post.assert_not_called()

    async def test_create_issue_is_403_read_only(self, async_client, ro_world):
        from coding_client.testing import override_tenant_auth

        from main import app

        undo = override_tenant_auth(app, tenant_id=RO_TENANT)
        try:
            r = await async_client.post(
                "/api/repositories/create-issue",
                json={"owner": PLATFORM, "repo": RO_REPO, "title": "t"},
            )
        finally:
            undo()
        assert r.status_code == 403, r.text
        assert r.json()["error"]["details"]["error_code"] == "platform_repo_read_only"
        ro_world.post.assert_not_called()
