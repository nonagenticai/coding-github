"""The merge route authenticates with the GitHub App, not the empty env PAT.

THE DEFECT THIS PINS (measured 2026-09-09 against production).

``POST /api/repositories/merge-pull-request`` had been invoked exactly TWICE in
production -- n8n executions 901393 (2026-09-08T17:48:33Z) and 927884
(2026-09-09T00:49:05Z), both targeting ``nonagenticai/enterprise_mcp_server#140``
-- and both returned **400 in under 3 ms** with *"No GitHub access token
configured for this service"*. The request never reached GitHub.

``_github()`` read ``settings.GITHUB_ACCESS_TOKEN or settings.github_token``,
both of which resolve to env ``CODING_GITHUB_GITHUB_TOKEN`` -- **empty** in the
live ``coding-github-env`` Secret. Meanwhile the App credential was mounted and
correct the whole time (Secret ``coding-github-github-app``, app id 2333879,
installation 131797507, ``contents:write`` + ``pull_requests:write`` granted
2026-06-05) and was ALREADY authoring this service's PRs through
``resolve_github_token``. Only this one helper looked in the wrong place.

⚠️ TWO defects were stacked, and fixing the first alone turns the 400 into a 404.
``_github()`` passed the raw token as ``GitHubConfig.token_id``, which is a KEY
into ``SecureTokenManager``'s on-disk store -- so a populated token made
``retrieve_token`` search for a file named after the secret, miss, return None
and raise, surfacing as ``not_found("Pull request")`` on a route whose real
problem was authentication. Case (b) below is the one that catches that layer.

Each test names the mutant that must turn it red. A test that passes with the
defect restored is not testing this.
"""

from __future__ import annotations

import pytest


class _Head:
    sha = "deadbeef"


class _PR:
    def __init__(self) -> None:
        self.merged = False
        self.mergeable = True
        self.title = "t"
        self.head = _Head()

    def merge(self, **kwargs):
        return type("R", (), {"merged": True, "sha": "abc123", "message": "ok"})()

    def get_commits(self):
        return type("C", (), {"reversed": [self]})()


class _Repo:
    private = True  # a private target repo (a public one is refused before merge)

    def get_pull(self, _n):
        return _PR()


class _CapturingGithub:
    """Stands in for PyGithub, recording the credential it was handed."""

    tokens: list[str] = []

    def __init__(self, token, base_url=None, **kwargs):
        type(self).tokens.append(token)

    def get_repo(self, _full_name):
        return _Repo()


@pytest.fixture(autouse=True)
def _app_ladder(monkeypatch, platform_tenant):
    """Arm the App rung and DISARM every PAT rung, as production has it."""
    import services.github_app_auth as ga
    import services.github_integration as gi
    from services.github.github_service import GitHubService

    monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")
    # The production reality this pins: the env PAT is EMPTY.
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
    monkeypatch.setattr(ga, "get_installation_token_for", lambda o, r: "ghs_test")

    _CapturingGithub.tokens = []
    monkeypatch.setattr("services.github.github_service.Github", _CapturingGithub)
    monkeypatch.setattr("services.github.github_service.HAS_GITHUB", True)

    # ``get_repository`` gates on ``isinstance(result, github.Repository.Repository)``,
    # which no stub can satisfy. Patch at that boundary ONLY: everything these tests
    # assert on -- which credential ``__aenter__`` resolves and hands to PyGithub -- has
    # already happened by the time this is reached.
    async def _repo(self, _full_name):
        return _Repo()

    monkeypatch.setattr(GitHubService, "get_repository", _repo)

    ga.reset_cache()
    yield GitHubService
    ga.reset_cache()


async def _merge(async_client):
    return await async_client.post(
        "/api/repositories/merge-pull-request",
        json={
            "owner": "nonagenticai",
            "repo": "enterprise_mcp_server",
            "pr_number": 140,
            "require_checks": False,
        },
    )


class TestMergeUsesTheApp:
    async def test_merge_succeeds_on_the_app_token_with_the_env_pat_empty(
        self, async_client
    ) -> None:
        """CASE (a). Green with an EMPTY env PAT is the whole proof.

        MUTANT: restore ``_github()`` to ``settings.GITHUB_ACCESS_TOKEN or
        settings.github_token`` -> 400 "No GitHub access token configured".
        """
        resp = await _merge(async_client)

        assert resp.status_code == 200, resp.text
        assert resp.json()["merged"] is True
        # The credential actually handed to PyGithub was the App's, not a PAT.
        assert _CapturingGithub.tokens == ["ghs_test"]

    async def test_the_resolved_token_bypasses_the_on_disk_token_store(
        self, async_client, monkeypatch
    ) -> None:
        """CASE (b). The SECOND, masked defect.

        ``retrieve_token`` must never be consulted for an already-resolved
        credential -- there is no ``<token>.enc`` file to find.

        MUTANT: drop the ``GitHubConfig.token`` passthrough so ``__aenter__``
        falls back to ``retrieve_token(token_id)`` -> that lookup misses and the
        route answers 404, not 200.
        """
        from services.github.secure_token_manager import SecureTokenManager

        calls: list[str] = []

        async def _spy(self, token_id):
            calls.append(token_id)
            return None

        monkeypatch.setattr(SecureTokenManager, "retrieve_token", _spy)

        resp = await _merge(async_client)

        assert resp.status_code == 200, resp.text
        assert calls == [], f"the on-disk token store was consulted: {calls}"

    async def test_the_app_is_asked_for_an_installation_on_the_target_owner(
        self, async_client, monkeypatch
    ) -> None:
        """CASE (c). Per-owner minting, not the home installation.

        MUTANT: call ``get_installation_token()`` (home) instead of
        ``get_installation_token_for(owner, repo)`` -> nothing is recorded here.
        """
        import services.github_app_auth as ga

        seen: list[tuple[str, str]] = []

        def _mint_for(owner: str, repo: str) -> str:
            seen.append((owner, repo))
            return "ghs_test"

        monkeypatch.setattr(ga, "get_installation_token_for", _mint_for)

        resp = await _merge(async_client)

        assert resp.status_code == 200, resp.text
        assert seen == [("nonagenticai", "enterprise_mcp_server")]

    async def test_a_platform_repo_whose_app_cannot_mint_fails_closed_as_412(
        self, async_client, monkeypatch
    ) -> None:
        """CASE (d). The failure must stay LOUD and must not be a 404.

        A 404 would say "no such pull request" about a PR that exists perfectly
        well; the real problem is that we could not authenticate to reach it.

        MUTANT: swallow ``GitHubAuthError`` and fall through to the ``ValueError``
        clause -> 404, and the operator debugs the wrong subsystem. A silent
        downgrade to a PAT rung would return 200 here, which is worse.
        """
        import services.github_app_auth as ga

        def _cannot_mint(owner: str, repo: str) -> str:
            raise ga.GitHubAppAuthError("installation not found")

        monkeypatch.setattr(ga, "get_installation_token_for", _cannot_mint)

        resp = await _merge(async_client)

        assert resp.status_code == 412, resp.text
        # This app wraps HTTPException in an {"error": {...}} envelope, so the message
        # is not at the bare "detail" key. Read both shapes rather than pinning one.
        body = resp.json()
        detail = body.get("detail") or body.get("error", {}).get("message", "")
        # The rung that failed must be NAMED.
        assert "app" in detail
        assert "nonagenticai/enterprise_mcp_server" in detail
        assert _CapturingGithub.tokens == [], "a PAT was substituted for the App"


class TestPlatformRepoIsForbiddenToCustomerTenants:
    async def test_a_customer_tenant_gets_403_and_the_app_is_never_asked(
        self, async_client, monkeypatch, enforce_platform_tenants
    ) -> None:
        """A licensed CUSTOMER tenant must not drive the internal App on our org.

        MUTANT: drop the membership check in ``check_platform_repo_tenant`` -> 200.
        """
        from coding_client.testing import override_tenant_auth

        import services.github_app_auth as ga
        from main import app

        asked: list[tuple[str, str]] = []
        monkeypatch.setattr(
            ga, "get_installation_token_for", lambda o, r: asked.append((o, r)) or "ghs_test"
        )
        undo = override_tenant_auth(app, tenant_id="11111111-1111-1111-1111-111111111111")
        try:
            response = await _merge(async_client)
        finally:
            undo()

        assert response.status_code == 403, response.text
        details = response.json()["error"]["details"]
        assert details == {
            "owner": "nonagenticai",
            "repo": "enterprise_mcp_server",
            "error_code": "platform_repo_forbidden",
        }
        assert asked == []
        assert _CapturingGithub.tokens == []
