"""Sibling calls must carry this service's licence; GitHub calls must not change.

coding-github sends two very different credentials and the tests here exist to
keep them apart:

* to **github.com** — a GitHub App JWT / installation token / user PAT. Not
  touched by this change and not touched by these tests.
* to **siblings** (coding-analysis, coding-security, coding-containers) — a bare
  ``X-Tenant-Id`` until now, accepted only because those callees still trust the
  header from the pod CIDR. Under ``TRUST_TENANT_HEADER=false`` every one of
  them 401s.

A 401 here is silent by construction. ``_trigger_full_scan`` /
``_trigger_dependency_scan`` log-and-swallow; ``_analyze_security_vulnerabilities``
is wrapped in ``except Exception -> return []``, so a refused scan is reported to
the push-event pipeline as *"no vulnerabilities"*. Nothing in production would
turn red — the results would just quietly become false.

Assertions are on what reached the wire (``httpx.MockTransport``), and always on
both halves at once: ``Authorization`` carries **our** identity and
``X-Tenant-Id`` still carries the **customer's**. Folding the licence into
``X-Tenant-Id`` would make the callee resolve the tenant from our token and
collapse every customer into ``svc-coding-github``'s own tenant.

The last class covers the ``"anonymous"`` defect on the secrets-service path:
that literal is not a null, it is a valid partition key, so the fetch *succeeded*
against a partition no registry issued.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from coding_client.clients import CodingWorkspacesClient
from coding_client.clients.coding_security import CodingSecurityClient

from services import github_provisioning, secret_resolver
from services.outbound import optional_sibling_headers, secrets_service_headers

TOKEN = "svc-github-token-xyz"
TENANT = "0195f48b-d000-7000-8000-000000000001"


# ---------------------------------------------------------------------------
# Transport plumbing
# ---------------------------------------------------------------------------


def _install_async_transport(monkeypatch, responder=None) -> list[httpx.Request]:
    """Force every ``httpx.AsyncClient`` built anywhere onto a MockTransport.

    The call sites under test build their client inline (``async with
    httpx.AsyncClient(...)``) or inside the coding-client SDK, so there is no
    client object to hand a transport to. Wrapping the constructor keeps the
    real client — and its real header merging — in the path.
    """
    seen: list[httpx.Request] = []
    real_cls = httpx.AsyncClient
    responder = responder or (lambda _r: httpx.Response(200, json={}))

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responder(request)

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return seen


def _assert_vouches_for(request: httpx.Request, tenant: str = TENANT) -> None:
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["x-tenant-id"] == tenant


def _container_payload(**overrides) -> dict:
    """A complete container body — the adapter parses it into a ``ContainerInfo``."""
    return {
        "container_name": "codegraph-r",
        "api_endpoint": "http://containers:8691/api/containers/codegraph-r",
        "codebase_path": "/workspace",
        "repo_url": "https://github.com/o/r",
        "branch": "main",
        "status": "running",
        "tenant_id": TENANT,
        **overrides,
    }


@pytest.fixture
def _service_urls(monkeypatch):
    """Point the sibling/secrets URLs somewhere the MockTransport can answer."""
    from config.settings import get_settings

    monkeypatch.setenv("CODING_GITHUB_CODING_ANALYSIS_URL", "http://analysis:8688")
    monkeypatch.setenv("CODING_GITHUB_CODING_SECURITY_URL", "http://security:8693")
    monkeypatch.setenv("CODING_GITHUB_CODING_CONTAINERS_URL", "http://containers:8691")
    monkeypatch.setenv("CODING_GITHUB_SECRETS_API_URL", "http://secrets:8080")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# services/outbound.py — the single decision point
# ---------------------------------------------------------------------------


class TestOutboundHelpers:
    def test_the_token_is_read_per_call_not_frozen_at_import(self, monkeypatch):
        """A rotated secret must take effect without restarting the pod."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "first")
        assert optional_sibling_headers(TENANT)["Authorization"] == "Bearer first"
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "second")
        assert optional_sibling_headers(TENANT)["Authorization"] == "Bearer second"

    def test_an_absent_tenant_is_omitted_and_never_becomes_a_literal(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        headers = optional_sibling_headers(None)
        assert "X-Tenant-Id" not in headers
        # Identity is still proven — that is not a claim about whose data we want.
        assert headers["Authorization"] == f"Bearer {TOKEN}"

    def test_an_empty_token_env_var_is_not_sent_as_a_bare_bearer(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "")
        assert "Authorization" not in optional_sibling_headers(TENANT)

    def test_the_secrets_service_never_receives_the_coding_licence(self, monkeypatch):
        """secrets-service cannot validate a coding licence and has no need to see it."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        headers = secrets_service_headers(TENANT)
        assert headers == {"X-Tenant-Id": TENANT}


# ---------------------------------------------------------------------------
# services/github_provisioning.py — the fire-and-forget provisioning scans
# ---------------------------------------------------------------------------


class TestProvisioningScans:
    async def test_full_scan_kickoff_vouches_for_the_customer_tenant(
        self, monkeypatch, _service_urls
    ):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(monkeypatch)

        await github_provisioning._trigger_full_scan("codegraph-repo", TENANT)

        _assert_vouches_for(seen[0])

    async def test_dependency_scan_kickoff_vouches_for_the_customer_tenant(
        self, monkeypatch, _service_urls
    ):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(monkeypatch)

        await github_provisioning._trigger_dependency_scan("codegraph-repo", TENANT)

        _assert_vouches_for(seen[0])

    async def test_no_token_means_no_authorization_header_at_all(self, monkeypatch, _service_urls):
        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        seen = _install_async_transport(monkeypatch)

        await github_provisioning._trigger_full_scan("codegraph-repo", TENANT)

        assert "authorization" not in seen[0].headers
        assert seen[0].headers["x-tenant-id"] == TENANT

    async def test_an_unresolvable_tenant_still_names_the_caller(self, monkeypatch, _service_urls):
        """No tenant and no DEFAULT_TENANT_ID: refuse to invent one, but say who we are.

        The callee will reject the scan for want of a tenant — which is correct
        — and its log will attribute the rejection to svc-coding-github instead
        of to an anonymous pod IP.
        """
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(github_provisioning, "DEFAULT_TENANT_ID", None)
        seen = _install_async_transport(monkeypatch)

        await github_provisioning._trigger_full_scan("codegraph-repo", None)

        assert "x-tenant-id" not in seen[0].headers
        assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"


# ---------------------------------------------------------------------------
# services/github_provisioning.py — the coding-containers adapter
# ---------------------------------------------------------------------------


class TestContainersAdapter:
    """The six container calls all route through ``_ContainersHTTPAdapter``."""

    @staticmethod
    def _adapter_with_client(monkeypatch, responder=None):
        seen = _install_async_transport(monkeypatch, responder)
        client = CodingWorkspacesClient(base_url="http://containers:8691")
        return github_provisioning._ContainersHTTPAdapter(), client, seen

    async def test_find_existing_container_vouches_for_the_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(
            monkeypatch, lambda _r: httpx.Response(200, json={"containers": []})
        )
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.find_existing_container("https://github.com/o/r", tenant_id=TENANT)

        _assert_vouches_for(seen[0])

    async def test_find_container_by_name_vouches_for_the_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(
            monkeypatch,
            lambda _r: httpx.Response(200, json={"container": _container_payload()}),
        )
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.find_container_by_name("codegraph-r", tenant_id=TENANT)

        _assert_vouches_for(seen[0])

    async def test_remove_container_vouches_for_the_customer_tenant(self, monkeypatch):
        """A DELETE going out unscoped is the worst of the six."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(monkeypatch)
        container = type("C", (), {"container_name": "c"})()
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.remove_container(container, tenant_id=TENANT)

        _assert_vouches_for(seen[0])

    async def test_remove_container_by_name_vouches_for_the_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(
            monkeypatch,
            lambda _r: httpx.Response(200, json={"container": _container_payload()}),
        )
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.remove_container_by_name("codegraph-r", tenant_id=TENANT)

        # The ownership pre-check GET and the DELETE must BOTH be credentialed.
        assert len(seen) >= 2
        for request in seen:
            _assert_vouches_for(request)

    async def test_create_container_vouches_for_the_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(
            monkeypatch,
            lambda _r: httpx.Response(200, json={"container": _container_payload()}),
        )
        request = type("R", (), {"repo_url": "https://github.com/o/r", "branch": "main"})()
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.create_container(request, tenant_id=TENANT)

        _assert_vouches_for(seen[0])

    async def test_list_containers_vouches_for_the_customer_tenant(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        adapter, client, seen = self._adapter_with_client(
            monkeypatch, lambda _r: httpx.Response(200, json={"containers": []})
        )
        with patch("main.get_workspaces_client", return_value=client):
            await adapter.list_containers(tenant_id=TENANT)

        _assert_vouches_for(seen[0])


# ---------------------------------------------------------------------------
# routers/github_webhooks.py — the detached push-event scan
# ---------------------------------------------------------------------------


class TestWebhookVulnerabilityScan:
    async def test_the_push_event_scan_vouches_for_the_configured_tenant(self, monkeypatch):
        """A refused scan here returns ``[]`` — "no vulnerabilities" — not an error."""
        from routers import github_webhooks

        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        monkeypatch.setattr(github_webhooks, "DEFAULT_TENANT_ID", TENANT)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"vulnerabilities_by_type": {}})
        )
        client = CodingSecurityClient(base_url="http://security:8693")

        with patch("main.get_security_client", return_value=client):
            await github_webhooks._analyze_security_vulnerabilities(["a.py"])

        assert seen, "the scan never left the process"
        _assert_vouches_for(seen[0])

    async def test_no_token_means_no_authorization_header_at_all(self, monkeypatch):
        from routers import github_webhooks

        monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)
        monkeypatch.setattr(github_webhooks, "DEFAULT_TENANT_ID", TENANT)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"vulnerabilities_by_type": {}})
        )
        client = CodingSecurityClient(base_url="http://security:8693")

        with patch("main.get_security_client", return_value=client):
            await github_webhooks._analyze_security_vulnerabilities(["a.py"])

        assert seen
        assert "authorization" not in seen[0].headers


# ---------------------------------------------------------------------------
# The "anonymous" defect on the secrets-service paths
# ---------------------------------------------------------------------------


class TestSecretsServiceNeverInventsATenant:
    """``tenant_id or request.user_id or "anonymous"`` fabricated a tenant twice over.

    ``"anonymous"`` is not a null. It is a valid partition key, so an
    unattributable fetch SUCCEEDED against a partition no registry issued, and
    downstream nothing could tell it apart from a real tenant named that. A user
    id is no better: it is an identifier from a different namespace entirely.
    """

    async def test_provisioning_omits_the_tenant_rather_than_inventing_one(
        self, monkeypatch, _service_urls
    ):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {}})
        )
        service = github_provisioning.GitHubProvisioningService.__new__(
            github_provisioning.GitHubProvisioningService
        )
        from config.settings import get_settings

        service.settings = get_settings()
        request = type("R", (), {"secret_config_id": "cfg-1", "user_id": "user-42"})()

        await service._get_environment_variables(request, tenant_id=None)

        assert seen, "the secrets fetch never left the process"
        assert "x-tenant-id" not in seen[0].headers, (
            "an absent tenant must be omitted, never replaced by a user id or a literal"
        )

    async def test_provisioning_never_substitutes_the_user_id_for_a_tenant(
        self, monkeypatch, _service_urls
    ):
        """Explicit: ``user-42`` must not appear as a tenant anywhere on the wire."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {}})
        )
        service = github_provisioning.GitHubProvisioningService.__new__(
            github_provisioning.GitHubProvisioningService
        )
        from config.settings import get_settings

        service.settings = get_settings()
        request = type("R", (), {"secret_config_id": "cfg-1", "user_id": "user-42"})()

        await service._get_environment_variables(request, tenant_id=None)

        assert "user-42" not in dict(seen[0].headers).values()
        assert "anonymous" not in dict(seen[0].headers).values()

    async def test_provisioning_still_sends_a_real_tenant(self, monkeypatch, _service_urls):
        """The fix must not stop a legitimately-scoped fetch from being scoped."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {"GIT_TOKEN": "t"}})
        )
        service = github_provisioning.GitHubProvisioningService.__new__(
            github_provisioning.GitHubProvisioningService
        )
        from config.settings import get_settings

        service.settings = get_settings()
        request = type("R", (), {"secret_config_id": "cfg-1", "user_id": "user-42"})()

        result = await service._get_environment_variables(request, tenant_id=TENANT)

        assert seen[0].headers["x-tenant-id"] == TENANT
        assert result == {"GIT_TOKEN": "t"}

    async def test_secret_resolver_omits_the_tenant_rather_than_inventing_one(
        self, monkeypatch, _service_urls
    ):
        """Same defect, same fix, in the upstream-PR token resolver."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {}})
        )

        await secret_resolver.resolve_git_token("cfg-1", None)

        assert seen
        assert "x-tenant-id" not in seen[0].headers
        assert "anonymous" not in dict(seen[0].headers).values()

    async def test_secret_resolver_still_sends_a_real_tenant(self, monkeypatch, _service_urls):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {"GIT_TOKEN": "t"}})
        )

        token = await secret_resolver.resolve_git_token("cfg-1", TENANT)

        assert seen[0].headers["x-tenant-id"] == TENANT
        assert token == "t"

    async def test_the_secrets_service_is_not_handed_the_coding_licence(
        self, monkeypatch, _service_urls
    ):
        """It cannot validate one, and a fleet credential should not travel there."""
        monkeypatch.setenv("CODING_SERVICE_TOKEN", TOKEN)
        seen = _install_async_transport(
            monkeypatch, lambda _r: httpx.Response(200, json={"secrets": {}})
        )

        await secret_resolver.resolve_git_token("cfg-1", TENANT)

        assert "authorization" not in seen[0].headers
