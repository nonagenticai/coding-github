"""The authoring identity for a platform-owned repo is the GitHub App.

THE DEFECT THIS PINS (measured 2026-08-15, n8n executions 531834/531862/531895).

``drive-it-flow1-demo.sh`` bakes whatever PAT the operator exported into a fresh
per-project secret at intake. On 2026-08-15 that PAT resolved fine — rung 'byo',
secret_config_id=01a00427… — and simply lacked ``contents: write`` on
``nonagenticai/enterprise_mcp_server``. ``POST /git/refs`` answered 403
*"Resource not accessible by personal access token"*. No branch, no commit, no
PR. The run that produced PR #85 the previous evening differed in exactly one
field: which secret_config_id had been baked in.

Two independent silences made it a three-session diagnosis:

1. ``_effective_github_token`` returned the BYO PAT for our OWN repo, and
   ``GitHubIntegration.__init__`` only consults the ladder when the token is
   ``None`` — so the App was never reached, for a repo it has written to since
   June (PRs #78-#83, author ``app/code-graph-helper-app``).
2. When the App rung DID fail, resolution continued to env / pool / system PAT.
   GitHub then names the credential TYPE it ended up with and never the reason a
   weaker one was chosen.

So the guarantees are: our own repos are authored by the App regardless of what
a drive supplies, and a platform-owned repo whose App cannot mint FAILS CLOSED
naming the rung rather than substituting a PAT.
"""

from __future__ import annotations

import pytest

from tests.conftest import PLATFORM_TEST_TENANT


class TestPlatformOwnedRepoUsesTheApp:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        import services.github_app_auth as ga
        import services.github_integration as gi

        # tests/conftest.py pins the org to "test-owner"; pin it to the real
        # production value so these read as the incident they encode.
        monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")
        ga.reset_cache()
        yield
        ga.reset_cache()

    def test_is_platform_owned_matches_the_configured_org_case_insensitively(self) -> None:
        import services.github_integration as gi

        assert gi.is_platform_owned("nonagenticai") is True
        assert gi.is_platform_owned("NonAgenticAI") is True
        assert gi.is_platform_owned("  nonagenticai  ") is True
        assert gi.is_platform_owned("some-customer") is False
        assert gi.is_platform_owned(None) is False
        assert gi.is_platform_owned("") is False

    def test_own_repo_fails_closed_naming_the_rung_instead_of_using_a_pat(
        self, monkeypatch
    ) -> None:
        """THE DEFECT. A platform PAT must never silently stand in for the App."""
        import services.github_app_auth as ga
        import services.github_integration as gi

        # Every lower rung is armed and MUST NOT be reached.
        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "github_pat_underprivileged")
        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)

        def _app_cannot_mint(owner: str, repo: str) -> str:
            raise ga.GitHubAppAuthError("installation not found")

        monkeypatch.setattr(ga, "get_installation_token_for", _app_cannot_mint)

        with pytest.raises(gi.GitHubAuthError) as excinfo:
            gi.resolve_github_token(
                owner="nonagenticai", repo="enterprise_mcp_server", tenant_id=PLATFORM_TEST_TENANT
            )

        message = str(excinfo.value)
        # The rung must be NAMED — that is the whole point of the change.
        assert "app" in message
        assert "nonagenticai/enterprise_mcp_server" in message
        # And the PAT must not have leaked into the message or the result.
        assert "github_pat_underprivileged" not in message
        assert "system-pat" not in message

    def test_own_repo_resolves_to_the_app_and_reports_the_rung(self, monkeypatch) -> None:
        import services.github_app_auth as ga
        import services.github_integration as gi

        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: "ghs_app_token")

        rt = gi.resolve_github_token(
            owner="nonagenticai", repo="enterprise_mcp_server", tenant_id=PLATFORM_TEST_TENANT
        )
        assert rt.token == "ghs_app_token"
        assert rt.is_app is True
        assert rt.rung == "app"

    def test_third_party_owner_never_falls_through_to_a_pat(self, monkeypatch) -> None:
        """INVERTED 2026-09-22 (public-github-app-usable). This test used to pin
        that a customer owner falls through to the env PAT when the App cannot
        mint. That fallback IS the defect the tenant-bound public App removes: a
        customer repo gets a token ONLY from the public App, only for a tenant-
        bound installation. Every lower rung is armed here and none may answer.
        """
        import services.github_app_auth as ga
        import services.github_integration as gi
        import services.tenant_binding as tb

        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "env-token")
        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: "ghs_internal_app")
        monkeypatch.setattr(tb, "lookup_binding", lambda tenant, owner: None)

        with pytest.raises(tb.TenantRequiredError):
            gi.resolve_github_token(owner="some-customer", repo="their-repo")
        with pytest.raises(tb.NotInstalledForTenantError):
            gi.resolve_github_token(owner="some-customer", repo="their-repo", tenant_id="t-1")

    def test_explicit_user_token_is_labelled_and_still_wins(self, monkeypatch) -> None:
        import services.github_integration as gi

        rt = gi.resolve_github_token("byo-token", owner="some-customer", repo="r")
        assert rt.token == "byo-token"
        assert rt.rung == "user"


class TestUpstreamPrPathAlsoUsesTheApp:
    """create_upstream_pull_request resolves its token INDEPENDENTLY.

    It never called resolve_github_token, so App-first in the apply-changes
    router does not reach it. Left alone, a drive on a platform-owned repo would
    have the App create the branch and then die here with "User token not found"
    the moment no project PAT was stored — which is precisely what removing the
    required GH_DEMO_PAT produces. The branch would exist and the PR never would.
    """

    @pytest.fixture(autouse=True)
    def _pin_org(self, monkeypatch):
        import services.github_integration as gi

        monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")

    def test_own_repo_resolves_the_app_rather_than_requiring_a_project_pat(
        self, monkeypatch
    ) -> None:
        import services.github_app_auth as ga
        import services.github_integration as gi

        monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
        monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: "ghs_app_token")

        resolved = gi.resolve_github_token(
            owner="nonagenticai", repo="enterprise_mcp_server", tenant_id=PLATFORM_TEST_TENANT
        )
        assert resolved.token == "ghs_app_token"
        assert resolved.rung == "app"

    def test_the_source_routes_platform_owned_repos_through_the_app(self) -> None:
        """Structural pin: the upstream-PR path must branch on is_platform_owned.

        A behavioural test would need the whole containers-client + PyGithub
        stack stood up; this asserts the branch exists and that the PAT lookup
        is on the ELSE side, which is the property that actually broke.
        """
        import inspect

        import services.github_integration as gi

        src = inspect.getsource(gi.GitHubIntegration.create_upstream_pull_request)
        assert "is_platform_owned(original_repo_owner)" in src
        app_at = src.index("is_platform_owned(original_repo_owner)")
        byo_at = src.index("resolve_git_token(secret_config_id")
        assert app_at < byo_at, "the App branch must be evaluated before the BYO lookup"
        assert "rung=" in src, "the failure message must name the rung it tried"


class TestEffectiveTokenRefusesToBypassTheAppForOwnRepos:
    """``_effective_github_token`` is where the bypass actually happened."""

    @pytest.fixture(autouse=True)
    def _pin_org(self, monkeypatch):
        import services.github_integration as gi

        monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")

    def _container(self, **kw):
        from routers.github_pull_request_operations import _ResolvedContainer

        info = {
            "container_name": "tour-1842030083498312926-c0",
            "repo_url": "https://github.com/nonagenticai/enterprise_mcp_server",
            "branch": "main",
        }
        info.update(kw)
        return _ResolvedContainer(info)

    @pytest.mark.asyncio
    async def test_own_repo_ignores_a_stored_project_pat(self, monkeypatch) -> None:
        """The exact 2026-08-15 shape: a resolvable but under-privileged BYO PAT."""
        import routers.github_pull_request_operations as ops
        import services.secret_resolver as sr

        async def _resolves_fine(secret_config_id, tenant_id):  # pragma: no cover - guarded below
            raise AssertionError(
                "secrets-service must not even be consulted for a platform-owned repo"
            )

        monkeypatch.setattr(sr, "resolve_git_token", _resolves_fine)

        container = self._container(secret_config_id="01a00427-2708-7322-8e31-9950972e9702")
        token = await ops._effective_github_token(container, "tenant-1", owner="nonagenticai")
        assert token is None, "None is what lets GitHubIntegration reach the App rung"

    @pytest.mark.asyncio
    async def test_own_repo_ignores_an_inline_legacy_token_too(self) -> None:
        import routers.github_pull_request_operations as ops

        container = self._container(github_access_token="ghp_legacy_inline")
        token = await ops._effective_github_token(container, "tenant-1", owner="nonagenticai")
        assert token is None

    def test_write_precheck_refuses_a_credential_without_push(self) -> None:
        """The customer-repo half of the same defect, caught before the 403."""
        from types import SimpleNamespace

        import routers.github_pull_request_operations as ops
        from services.github_integration import GitHubAuthError

        repo = SimpleNamespace(
            full_name="some-customer/their-repo",
            permissions=SimpleNamespace(push=False),
        )
        with pytest.raises(GitHubAuthError) as excinfo:
            ops._assert_can_write(repo, "byo")
        message = str(excinfo.value)
        assert "byo" in message
        assert "some-customer/their-repo" in message
        assert "push permission" in message

    def test_the_write_precheck_is_actually_wired_into_apply_changes(self) -> None:
        """NON-VACUITY. Every other precheck test passes with the call site DELETED.

        Measured by mutation: removing the `_assert_can_write(...)` call from
        apply_changes left all 14 tests in this file green. A guard whose only
        tests call it directly certifies the FUNCTION, never the SYSTEM — and an
        unwired guard is indistinguishable from a working one.

        So this asserts the wiring itself: the call exists in apply_changes,
        it runs BEFORE the branch is created (after that, the 403 has already
        happened), and it is passed the is_app flag rather than defaulting.
        """
        import inspect

        import routers.github_pull_request_operations as ops

        src = inspect.getsource(ops.apply_code_changes)
        assert "_assert_can_write(" in src, "the precheck is not wired into apply_changes"
        assert "is_app=" in src, "the App exemption must be passed explicitly, not defaulted"
        guard_at = src.index("_assert_can_write(")
        branch_at = src.index("_create_branch(")
        assert guard_at < branch_at, (
            "the precheck must run BEFORE _create_branch — after it, the 403 it "
            "exists to pre-empt has already been returned by GitHub"
        )

    def test_write_precheck_never_fires_for_an_app_token(self) -> None:
        """THE MASS FALSE-BLOCK THIS AVOIDS.

        Measured live 2026-08-15 against installation 131797507
        (code-graph-helper-app: contents:write, repository_selection all, 208
        repos including the target, author of PRs #78-#83 on it). The raw
        permissions block returned to its installation token was

            {'admin': False, 'maintain': False, 'push': False,
             'triage': False, 'pull': False}

        with pull:False coming back from the request that had just successfully
        READ the repo. permissions is user-shaped and GitHub does not populate
        it for App tokens. Enforcing it would refuse every App write.
        """
        from types import SimpleNamespace

        import routers.github_pull_request_operations as ops

        repo = SimpleNamespace(
            full_name="nonagenticai/enterprise_mcp_server",
            permissions=SimpleNamespace(
                admin=False, maintain=False, push=False, triage=False, pull=False
            ),
        )
        ops._assert_can_write(repo, "app", is_app=True)  # must not raise

        # ...and the same repo shape WOULD be refused for a PAT rung, so the
        # exemption is scoped to App tokens rather than disabling the guard.
        from services.github_integration import GitHubAuthError

        with pytest.raises(GitHubAuthError):
            ops._assert_can_write(repo, "byo", is_app=False)

    def test_write_precheck_allows_a_credential_with_push(self) -> None:
        from types import SimpleNamespace

        import routers.github_pull_request_operations as ops

        repo = SimpleNamespace(full_name="x/y", permissions=SimpleNamespace(push=True))
        ops._assert_can_write(repo, "app")  # must not raise

    def test_write_precheck_stays_silent_when_capability_is_unknown(self) -> None:
        """An unknown capability must not manufacture a refusal.

        A missing ``.permissions`` is not evidence of "cannot write"; treating it
        as such would convert a read gap into a mass false-block, which is the
        failure mode a guard on an unmeasured base rate always has.
        """
        from types import SimpleNamespace

        import routers.github_pull_request_operations as ops

        ops._assert_can_write(SimpleNamespace(full_name="x/y"), "app")
        ops._assert_can_write(SimpleNamespace(permissions=SimpleNamespace()), "app")
        ops._assert_can_write(object(), None)

    @pytest.mark.asyncio
    async def test_customer_repo_still_uses_its_byo_token(self, monkeypatch) -> None:
        """R1 stands: the App cannot reach a customer's private repo."""
        import routers.github_pull_request_operations as ops
        import services.secret_resolver as sr

        async def _resolve(secret_config_id, tenant_id):
            assert secret_config_id == "secret-123"
            return "customer-pat"

        monkeypatch.setattr(sr, "resolve_git_token", _resolve)

        container = self._container(
            repo_url="https://github.com/some-customer/their-repo",
            secret_config_id="secret-123",
        )
        token = await ops._effective_github_token(container, "tenant-1", owner="some-customer")
        assert token == "customer-pat"
