"""Unit tests for the internal clone-token endpoint (``/api/internal``).

Covers the shared-secret guard (the endpoint vends credentials, so this IS the
authz boundary — see routers/github_internal.py for why require_tenant cannot
be used), repo-URL parsing, and the public/private + app/none answer matrix.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from routers import github_internal
from services.github_app_auth import (
    GitHubAppAuthError,
    GitHubAppMintError,
    GitHubAppNotConfiguredError,
    GitHubAppNotInstalledError,
    ScopedToken,
)

#: Push-scope repository facts (additive fields): absent from every non-push answer.
_NO_PUSH_FACTS = {
    "private": None,
    "default_branch": None,
    "branch_head": None,
    "default_branch_head": None,
}


@pytest.fixture(autouse=True)
def _acme_is_the_platform_org(monkeypatch):
    """These tests pin the PLATFORM (internal App) clone-token path. Since
    2026-09-22 that path serves only ``system_github_owner``; customer owners are
    tenant-bound and covered in tests/unit/test_tenant_repo_token.py. So the
    fixture owner "acme" is declared the platform org here.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "acme")


INTERNAL_TOKEN = "s3cret-internal-token"  # pragma: allowlist secret
CLONE_TOKEN_URL = "/api/internal/github/clone-token"


def _settings(internal_service_token=INTERNAL_TOKEN) -> SimpleNamespace:
    callers = {"coding-containers": internal_service_token} if internal_service_token else {}
    return SimpleNamespace(internal_callers=callers)


@pytest.fixture
def patched_settings():
    with patch.object(github_internal, "get_settings", return_value=_settings()):
        yield


@pytest.fixture
def mock_github():
    """Patch both GitHub-egress helpers the router calls."""
    scoped = MagicMock(return_value=ScopedToken("ghs_scoped", "2999-01-01T00:00:00Z"))
    visibility = MagicMock(return_value="private")
    with (
        patch.object(github_internal, "get_scoped_clone_token", scoped),
        patch.object(github_internal, "get_repo_visibility", visibility),
    ):
        yield scoped, visibility


# ---------------------------------------------------------------------------
# AUTHZ — the shared secret is the only real gate on this endpoint
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestInternalTokenGuard:
    async def test_403_without_header(self, async_client, patched_settings, mock_github):
        r = await async_client.post(
            CLONE_TOKEN_URL, json={"repo_url": "https://github.com/acme/widgets"}
        )
        assert r.status_code == 403

    async def test_403_with_wrong_token(self, async_client, patched_settings, mock_github):
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": "wrong"},
        )
        assert r.status_code == 403

    async def test_403_when_secret_unconfigured(self, async_client, mock_github):
        """Fail-closed: no configured secret ⇒ the endpoint vends nothing to anyone."""
        with patch.object(
            github_internal, "get_settings", return_value=_settings(internal_service_token="")
        ):
            r = await async_client.post(
                CLONE_TOKEN_URL,
                json={"repo_url": "https://github.com/acme/widgets"},
                headers={"X-Internal-Token": ""},
            )
        assert r.status_code == 403

    async def test_403_does_not_mint_anything(self, async_client, patched_settings, mock_github):
        scoped, _ = mock_github
        await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": "wrong"},
        )
        scoped.assert_not_called()

    async def test_200_with_correct_token(self, async_client, patched_settings, mock_github):
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Response matrix
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCloneTokenEndpoint:
    async def test_private_repo_gets_scoped_app_token(
        self, async_client, patched_settings, mock_github
    ):
        scoped, visibility = mock_github
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 200
        assert r.json() == _NO_PUSH_FACTS | {
            "token": "ghs_scoped",
            "expires_at": "2999-01-01T00:00:00Z",
            "visibility": "private",
            "source": "app",
            "reason": None,
        }
        # No scope in the body ⇒ the backward-compatible contents:read clone scope.
        scoped.assert_called_once_with("acme", "widgets", scope="clone")
        # Visibility is resolved WITH the minted token (a private repo is
        # invisible to an unauthenticated probe).
        visibility.assert_called_once_with("acme", "widgets", "ghs_scoped")

    async def test_app_not_installed_returns_null_token_not_an_error(
        self, async_client, patched_settings, mock_github, caplog
    ):
        """A public repo needs no credential — 200 with source=none, caller decides."""
        scoped, visibility = mock_github
        scoped.side_effect = GitHubAppNotInstalledError(
            "GitHub App is not installed on acme/widgets"
        )
        visibility.return_value = "public"

        with caplog.at_level(logging.DEBUG):
            r = await async_client.post(
                CLONE_TOKEN_URL,
                json={"repo_url": "https://github.com/acme/widgets"},
                headers={"X-Internal-Token": INTERNAL_TOKEN},
            )
        assert r.status_code == 200
        assert r.json() == _NO_PUSH_FACTS | {
            "token": None,
            "expires_at": None,
            "visibility": "public",
            "source": "none",
            "reason": "not_installed",
        }
        visibility.assert_called_once_with("acme", "widgets", None)
        # A routine miss must never be logged at ERROR — that is what makes a real
        # mint failure findable.
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    async def test_mint_failure_is_reason_mint_failed_not_not_installed(
        self, async_client, patched_settings, mock_github, caplog
    ):
        """The distinction this field exists for: a FAILED mint is not an absent App.

        Both used to come back as ``source="none"`` with nothing else to go on, so
        a GitHub outage was indistinguishable from "that owner isn't a customer".
        """
        scoped, visibility = mock_github
        scoped.side_effect = GitHubAppMintError(
            "GitHub refused a scoped clone token for acme/widgets (HTTP 500)."
        )
        visibility.return_value = "private"

        with caplog.at_level(logging.DEBUG):
            r = await async_client.post(
                CLONE_TOKEN_URL,
                json={"repo_url": "https://github.com/acme/widgets"},
                headers={"X-Internal-Token": INTERNAL_TOKEN},
            )
        assert r.status_code == 200
        assert r.json()["reason"] == "mint_failed"
        assert r.json()["token"] is None
        assert r.json()["source"] == "none"
        # ...and it is LOUD, unlike the routine miss above.
        assert any(rec.levelno >= logging.ERROR for rec in caplog.records)

    async def test_unconfigured_app_is_reason_not_configured(
        self, async_client, patched_settings, mock_github
    ):
        scoped, visibility = mock_github
        scoped.side_effect = GitHubAppNotConfiguredError("No GitHub App private key configured")
        visibility.return_value = "public"

        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 200
        assert r.json()["reason"] == "not_configured"

    async def test_bare_auth_error_fails_loud_rather_than_reading_as_absent(
        self, async_client, patched_settings, mock_github
    ):
        """An unclassified App error must NOT be reported as a routine miss."""
        scoped, visibility = mock_github
        scoped.side_effect = GitHubAppAuthError("something new and unclassified")
        visibility.return_value = "private"

        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.json()["reason"] == "mint_failed"

    async def test_successful_mint_carries_no_reason(
        self, async_client, patched_settings, mock_github
    ):
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 200
        assert r.json()["token"] == "ghs_scoped"
        assert r.json()["reason"] is None

    async def test_unresolvable_visibility_is_502(
        self, async_client, patched_settings, mock_github
    ):
        _, visibility = mock_github
        visibility.side_effect = GitHubAppAuthError("api.github.com unreachable")
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 502

    async def test_invalid_repo_url_is_400(self, async_client, patched_settings, mock_github):
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme"},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 400

    async def test_missing_repo_url_is_422(self, async_client, patched_settings, mock_github):
        r = await async_client.post(
            CLONE_TOKEN_URL, json={}, headers={"X-Internal-Token": INTERNAL_TOKEN}
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# scope — "clone" (default) | "pr-read"; nothing else is grantable
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCloneTokenScope:
    async def test_pr_read_scope_is_threaded_to_the_mint(
        self, async_client, patched_settings, mock_github, caplog
    ):
        scoped, _ = mock_github
        with caplog.at_level(logging.INFO, logger=github_internal.logger.name):
            r = await async_client.post(
                CLONE_TOKEN_URL,
                json={"repo_url": "https://github.com/acme/widgets", "scope": "pr-read"},
                headers={"X-Internal-Token": INTERNAL_TOKEN},
            )
        assert r.status_code == 200
        # Response shape is unchanged by the new field (push facts are additive, None here).
        body = r.json()
        assert set(body) == {"token", "expires_at", "visibility", "source", "reason"} | set(
            _NO_PUSH_FACTS
        )
        assert {k: body[k] for k in _NO_PUSH_FACTS} == _NO_PUSH_FACTS
        scoped.assert_called_once_with("acme", "widgets", scope="pr-read")
        served = [m for m in caplog.messages if "Clone token request served" in m]
        assert len(served) == 1
        assert "scope=pr-read" in served[0]
        assert "ghs_scoped" not in caplog.text  # never log the credential

    async def test_default_scope_is_logged_as_clone(
        self, async_client, patched_settings, mock_github, caplog
    ):
        with caplog.at_level(logging.INFO, logger=github_internal.logger.name):
            await async_client.post(
                CLONE_TOKEN_URL,
                json={"repo_url": "https://github.com/acme/widgets"},
                headers={"X-Internal-Token": INTERNAL_TOKEN},
            )
        assert any(
            "Clone token request served" in m and "scope=clone" in m for m in caplog.messages
        )

    @pytest.mark.parametrize("scope", ["admin", "contents:write", "pull_requests:write", ""])
    async def test_unknown_scope_is_422_and_mints_nothing(
        self, async_client, patched_settings, mock_github, scope
    ):
        scoped, _ = mock_github
        r = await async_client.post(
            CLONE_TOKEN_URL,
            json={"repo_url": "https://github.com/acme/widgets", "scope": scope},
            headers={"X-Internal-Token": INTERNAL_TOKEN},
        )
        assert r.status_code == 422
        scoped.assert_not_called()


# ---------------------------------------------------------------------------
# parse_repo_url
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestParseRepoUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme/widgets",
            "https://github.com/acme/widgets.git",
            "https://github.com/acme/widgets/",
            "git@github.com:acme/widgets.git",
            "  https://github.com/acme/widgets  ",
        ],
    )
    def test_accepts_supported_url_forms(self, url):
        assert github_internal.parse_repo_url(url) == ("acme", "widgets")

    @pytest.mark.parametrize("url", ["https://github.com/acme", "not-a-url", "", "/"])
    def test_rejects_malformed_urls(self, url):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            github_internal.parse_repo_url(url)
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# THE TENANT IS NOT PART OF THIS ROUTE'S CONTRACT
#
# coding-containers#101 (2026-08-16): every PAT-less container provision 403'd
# because the caller sent `X-Tenant-Id` alongside its own licence. The tenant
# middleware refuses on DISAGREEMENT, so a header this route never reads became a
# hard block -- 403 TENANT_ASSERTION_NOT_PERMITTED, raised in middleware before
# this module ran.
#
# ⚠️ WHY THESE ARE STRUCTURAL AND NOT BEHAVIOURAL. `tests/conftest.py` sets
# ENABLE_AUTH=false, so the app under test has NO Auth/Tenant middleware at all.
# A test that POSTed without a tenant and asserted 200 would pass for the wrong
# reason -- the middleware that does the refusing is simply absent -- and would
# keep passing if someone bolted `Depends(require_tenant)` onto the route. So
# these assert the ROUTE CONTRACT directly. The behavioural half was measured
# in-pod against the live service (403 with the header, 200 without) and is
# recorded in the PR body; it cannot be reproduced in this suite.
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRouteDeclaresNoTenant:
    """The route must never grow a tenant dependency or parameter."""

    def _clone_token_route(self):
        for route in github_internal.router.routes:
            if getattr(route, "path", None) == "/github/clone-token":
                return route
        raise AssertionError("clone-token route not found on the router")

    def test_handler_takes_no_tenant_parameter(self):
        """`mint_clone_token(request, http_request)` and nothing else.

        A `tenant_id: str = Depends(require_tenant)` parameter added here would
        reintroduce exactly the coupling that produced the 403.
        """
        import inspect

        params = list(inspect.signature(github_internal.mint_clone_token).parameters)
        # ``http_request`` is the raw starlette Request, read only for the
        # authenticated caller name the guard stashes on request.state.
        assert params == ["request", "http_request"], (
            f"clone-token handler grew parameters {params!r}; the tenant is not part of "
            "this route's contract -- see the module docstring, point 3"
        )

    def test_route_dependencies_do_not_include_a_tenant_resolver(self):
        """The only declared dependency is the shared-secret guard."""
        route = self._clone_token_route()
        names = [
            getattr(d.dependency, "__name__", repr(d.dependency))
            for d in getattr(route, "dependencies", [])
        ]
        assert names == ["require_internal_token"], (
            f"clone-token dependencies are {names!r}; expected exactly ['require_internal_token']"
        )
        assert not any("tenant" in n.lower() for n in names)

    def test_no_route_on_this_router_resolves_a_tenant(self):
        """Covers routes added later, not just clone-token."""
        offenders = {}
        for route in github_internal.router.routes:
            names = [
                getattr(d.dependency, "__name__", repr(d.dependency))
                for d in getattr(route, "dependencies", [])
            ]
            bad = [n for n in names if "tenant" in n.lower()]
            if bad:
                offenders[getattr(route, "path", "?")] = bad
        assert not offenders, (
            f"/api/internal routes must not resolve a tenant, found: {offenders!r}. "
            "The shared secret is the authz boundary; a tenant assertion here can only 403."
        )

    def test_module_docstring_records_the_middleware_trap(self):
        """The trap is invisible in the code, so the docstring IS the mechanism.

        Without this, the next reader sees `Depends(require_internal_token)`, concludes
        the tenant plays no part, and cannot explain a 403 raised upstream of the route.
        """
        doc = github_internal.__doc__ or ""
        assert "TENANT_ASSERTION_NOT_PERMITTED" in doc
        assert "globally" in doc
        assert "OMITS" in doc or "omits" in doc
