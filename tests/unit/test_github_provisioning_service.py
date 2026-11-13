"""Unit tests for GitHubProvisioningService and _ContainersHTTPAdapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github_provisioning import (
    GitHubProvisioningService,
    _ContainersHTTPAdapter,
    _scan_headers,
)

# ---------------------------------------------------------------------------
# _scan_headers — tenant resolution for the fire-and-forget provisioning scans
#
# coding-client 1.9.0 (#36) changed DEFAULT_TENANT_ID from `str` (it always had
# the built-in "n8n-dev" fallback) to `str | None`. This service does not set
# DEFAULT_TENANT_ID on its Deployment, so None is the LIVE case — and the old
# `tenant_id or DEFAULT_TENANT_ID` would have put a literal None into a header
# value. These cover both halves so the None path cannot silently regress into
# inventing a tenant again.
# ---------------------------------------------------------------------------


def test_scan_headers_uses_caller_tenant():
    """An explicit caller tenant is sent as-is, regardless of the default."""
    with patch("services.github_provisioning.DEFAULT_TENANT_ID", "some-default"):
        assert _scan_headers("caller-tenant") == {"X-Tenant-Id": "caller-tenant"}


def test_scan_headers_falls_back_to_configured_default():
    """With no caller tenant, an explicitly CONFIGURED default is used."""
    with patch("services.github_provisioning.DEFAULT_TENANT_ID", "configured-default"):
        assert _scan_headers(None) == {"X-Tenant-Id": "configured-default"}


def test_scan_headers_omits_header_when_no_tenant_is_known():
    """Unset DEFAULT_TENANT_ID must omit the header, never send None.

    This is the case that is live in production today. Inventing a tenant here
    is precisely what coding-client #36 exists to stop, so the scan is left to
    be refused by the callee rather than silently attributed to a fallback.
    """
    with patch("services.github_provisioning.DEFAULT_TENANT_ID", None):
        headers = _scan_headers(None)

    assert headers == {}
    # The regression this guards against is a header whose VALUE is None.
    assert None not in headers.values()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides):
    """Build a fake GitHubRepoRequest with sensible defaults."""
    defaults = {
        "repo_url": "https://github.com/test-org/test-repo",
        "branch": "main",
        "access_token": "ghp_fake",
        "container_name": "",
        "force_rebuild": False,
        "provision_type": "repository",
        "pr_number": None,
        "user_id": "user-1",
        "description": "",
        "secret_config_id": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_container_info(**overrides):
    defaults = {
        "container_name": "codegraph-test_repo",
        "api_endpoint": "http://localhost:8691",
        "status": "running",
        "repo_url": "https://github.com/test-org/test-repo",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _ContainersHTTPAdapter tests
# ---------------------------------------------------------------------------


class TestContainersHTTPAdapter:
    """Tests for the thin HTTP adapter wrapping containers client."""

    @pytest.fixture
    def adapter(self):
        return _ContainersHTTPAdapter()

    # 1
    async def test_adapter_find_existing_container_success(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "ctr-1",
                    "api_endpoint": "",
                    "codebase_path": "/tmp/repos/r",
                    "status": "running",
                    "repo_url": "https://github.com/o/r",
                    "branch": "main",
                }
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container("https://github.com/o/r")
        assert result.container_name == "ctr-1"
        assert result.status == "running"
        assert result.repo_url == "https://github.com/o/r"

    # 1b — regression: a container for a DIFFERENT repo must NOT be returned
    # (the old code returned containers[0] unconditionally, handing provision a
    # stale/unrelated container and never creating one for the requested repo).
    async def test_adapter_find_existing_no_repo_match_returns_none(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "other",
                    "repo_url": "https://github.com/o/other",
                    "status": "running",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container("https://github.com/o/r")
        assert result is None

    # 1c — picks the container matching the requested repo, not the first in the
    # list, and forwards the tenant as an X-Tenant-Id header (repo_url is matched
    # ignoring a trailing slash / .git suffix).
    async def test_adapter_find_existing_matches_repo_and_forwards_tenant(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "other",
                    "repo_url": "https://github.com/o/other",
                    "status": "running",
                },
                {
                    "container_name": "want",
                    "api_endpoint": "",
                    "codebase_path": "/app/code",
                    "repo_url": "https://github.com/o/r.git",
                    "branch": "main",
                    "status": "running",
                    # Labelled with the requesting tenant: this test is about
                    # repo matching + header forwarding, and ownership is now
                    # fail-closed (an unlabelled container matches no tenant).
                    "tenant_id": "tenant-9",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result.container_name == "want"
        headers = mock_client.list_workspaces.call_args.kwargs.get("headers")
        assert headers == {"X-Tenant-Id": "tenant-9"}

    # 1d — a same-repo container owned by a DIFFERENT tenant must be skipped
    # (defence for the admin-lists-all-tenants case).
    async def test_adapter_find_existing_skips_cross_tenant(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "theirs",
                    "repo_url": "https://github.com/o/r",
                    "tenant_id": "tenant-X",
                    "status": "running",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result is None

    # 2
    async def test_adapter_find_existing_empty_list(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(return_value=[])
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container("https://github.com/o/r")
        assert result is None

    # 3
    async def test_adapter_find_existing_service_error(self, adapter):
        from coding_client.clients import ServiceError

        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(side_effect=ServiceError(500, "boom"))
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container("https://github.com/o/r")
        assert result is None

    # 3b — fr's dedicated quality workspaces (``q-`` + 12 hex) are owned by
    # their creator and deleted/re-cloned freely, so repo-url reuse must never
    # hand one out: only a dedicated one present -> None.
    async def test_adapter_find_existing_skips_dedicated_workspace(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "q-0123456789ab",
                    "repo_url": "https://github.com/o/r",
                    "status": "running",
                    "tenant_id": "tenant-9",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result is None

    # 3c — dedicated + shared for the same repo, in either order -> the shared one.
    @pytest.mark.parametrize("dedicated_first", [True, False])
    async def test_adapter_find_existing_prefers_shared_over_dedicated(
        self, adapter, dedicated_first
    ):
        dedicated = {
            "container_name": "q-0123456789ab",
            "api_endpoint": "",
            "codebase_path": "/app/code",
            "repo_url": "https://github.com/o/r",
            "branch": "main",
            "status": "running",
            "tenant_id": "tenant-9",
        }
        shared = {**dedicated, "container_name": "codegraph-r"}
        listing = [dedicated, shared] if dedicated_first else [shared, dedicated]
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(return_value=listing)
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result.container_name == "codegraph-r"

    # 3d — the reserved pattern is STRICT: near-misses are ordinary shared
    # workspaces and are still reused.
    @pytest.mark.parametrize(
        "name", ["q-tool", "q-0123456789abc", "q-0123456789AB", "xq-0123456789ab"]
    )
    async def test_adapter_find_existing_near_miss_names_still_reused(self, adapter, name):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": name,
                    "api_endpoint": "",
                    "codebase_path": "/app/code",
                    "repo_url": "https://github.com/o/r",
                    "branch": "main",
                    "status": "running",
                    "tenant_id": "tenant-9",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result is not None
        assert result.container_name == name

    # 4
    async def test_adapter_find_by_name_404(self, adapter):
        from coding_client.clients import ServiceError

        err = ServiceError(404, "not found")
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=err)
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name("missing")
        assert result is None

    # 5
    async def test_adapter_remove_container_success(self, adapter):
        mock_client = AsyncMock()
        mock_client.delete_workspace = AsyncMock()
        container = SimpleNamespace(container_name="ctr-1")
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.remove_container(container)
        assert result is True
        mock_client.delete_workspace.assert_awaited_once_with("ctr-1", headers=None)

    # 5b — remove_container forwards X-Tenant-Id when the caller has a tenant,
    # so the delete is never issued unscoped on a tenant-scoped request.
    async def test_adapter_remove_container_forwards_tenant_header(self, adapter):
        mock_client = AsyncMock()
        mock_client.delete_workspace = AsyncMock()
        container = SimpleNamespace(container_name="ctr-1")
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.remove_container(container, tenant_id="tenant-A")
        assert result is True
        mock_client.delete_workspace.assert_awaited_once_with(
            "ctr-1", headers={"X-Tenant-Id": "tenant-A"}
        )

    # 6
    async def test_adapter_remove_container_404(self, adapter):
        from coding_client.clients import ServiceError

        err = ServiceError(404, "not found")
        mock_client = AsyncMock()
        mock_client.delete_workspace = AsyncMock(side_effect=err)
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="not found"):
                await adapter.remove_container_by_name("missing")

    # 7
    async def test_adapter_remove_by_name_success(self, adapter):
        mock_client = AsyncMock()
        mock_client.delete_workspace = AsyncMock()
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.remove_container_by_name("ctr-1")
        assert result is True
        # No tenant supplied -> unscoped (admin/legacy) delete, no ownership probe.
        mock_client.delete_workspace.assert_awaited_once_with("ctr-1", headers=None)

    # 7a — SECURITY (cross-tenant): tenant A must NOT delete tenant B's
    # identically-named container on force_rebuild. remove_container_by_name
    # first looks the container up scoped to the caller's tenant; a container
    # owned by another tenant reads as "not found" and delete is never issued.
    async def test_adapter_remove_by_name_rejects_other_tenant(self, adapter):
        mock_client = AsyncMock()
        # The name resolves, but the container is owned by tenant-B.
        mock_client.get_workspace = AsyncMock(
            return_value={
                "tenant_id": "tenant-B",
                "container": {
                    "container_name": "codegraph-shared_repo",
                    "api_endpoint": "http://x",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "https://github.com/other/shared_repo",
                    "branch": "main",
                    "tenant_id": "tenant-B",
                },
            }
        )
        mock_client.delete_workspace = AsyncMock()
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="not found"):
                await adapter.remove_container_by_name(
                    "codegraph-shared_repo", tenant_id="tenant-A"
                )
        # Critical: tenant B's container was never deleted.
        mock_client.delete_workspace.assert_not_awaited()

    # 7b — the owning tenant CAN delete, and the X-Tenant-Id header is forwarded.
    async def test_adapter_remove_by_name_owner_deletes_with_header(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "tenant_id": "tenant-A",
                "container": {
                    "container_name": "codegraph-shared_repo",
                    "api_endpoint": "http://x",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "https://github.com/a/shared_repo",
                    "branch": "main",
                    "tenant_id": "tenant-A",
                },
            }
        )
        mock_client.delete_workspace = AsyncMock()
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.remove_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is True
        mock_client.delete_workspace.assert_awaited_once_with(
            "codegraph-shared_repo", headers={"X-Tenant-Id": "tenant-A"}
        )

    # 4b — SECURITY (cross-tenant): find_container_by_name must return None for
    # another tenant's container instead of leaking its status/repo_url.
    async def test_adapter_find_by_name_rejects_other_tenant(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "tenant_id": "tenant-B",
                "container": {
                    "container_name": "codegraph-shared_repo",
                    "api_endpoint": "http://x",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "https://github.com/other/shared_repo",
                    "branch": "main",
                    "tenant_id": "tenant-B",
                },
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is None
        # And the scoping header is forwarded downstream.
        assert mock_client.get_workspace.call_args.kwargs.get("headers") == {
            "X-Tenant-Id": "tenant-A"
        }

    # 4c — find_container_by_name returns the container for its owning tenant.
    async def test_adapter_find_by_name_returns_owned_container(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "tenant_id": "tenant-A",
                "container": {
                    "container_name": "codegraph-shared_repo",
                    "api_endpoint": "http://x",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "https://github.com/a/shared_repo",
                    "branch": "main",
                    "tenant_id": "tenant-A",
                },
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is not None
        assert result.container_name == "codegraph-shared_repo"

    # ------------------------------------------------------------------
    # SECURITY — fail-closed ownership for UNLABELLED containers.
    #
    # The ownership predicate used to accept ``tenant_id in (None, "", caller)``,
    # so a container carrying no tenant label belonged to whoever asked: any
    # tenant could read or delete it just by guessing its (repo-name-derived)
    # name. Unlabelled containers are a live upstream state, not only legacy
    # data — coding-containers' registry-MISS path rebuilds ContainerInfo from
    # the Deployment and deliberately leaves ``tenant_id`` unset. A tenant-scoped
    # request must DENY when ownership cannot be positively verified.
    # ------------------------------------------------------------------

    @staticmethod
    def _envelope(**container_overrides):
        """Build a coding-containers GET /api/containers/{name} envelope."""
        container = {
            "container_name": "codegraph-shared_repo",
            "api_endpoint": "http://x",
            "codebase_path": "/app/code",
            "status": "running",
            "repo_url": "https://github.com/other/shared_repo",
            "branch": "main",
        }
        container.update(container_overrides)
        return {"container": container}

    # 4d — an UNLABELLED container (no tenant_id key anywhere) must not be
    # readable by an arbitrary tenant.
    async def test_adapter_find_by_name_rejects_unlabelled_container(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(return_value=self._envelope())
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is None

    # 4e — same for an EMPTY-STRING tenant label on the container.
    async def test_adapter_find_by_name_rejects_empty_tenant_label(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={"tenant_id": "", **self._envelope(tenant_id="")}
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is None

    # 7c — an UNLABELLED container must not be deletable by an arbitrary tenant:
    # the ownership probe reads as "not found" and DELETE is never issued.
    async def test_adapter_remove_by_name_rejects_unlabelled_container(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(return_value=self._envelope())
        mock_client.delete_workspace = AsyncMock()
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="not found"):
                await adapter.remove_container_by_name(
                    "codegraph-shared_repo", tenant_id="tenant-A"
                )
        mock_client.delete_workspace.assert_not_awaited()

    # 7d — control: a container labelled with the caller's tenant is still
    # deletable (proves 7c denies on ownership, not on the envelope shape).
    async def test_adapter_remove_by_name_owner_still_deletes_same_envelope(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(return_value=self._envelope(tenant_id="tenant-A"))
        mock_client.delete_workspace = AsyncMock()
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.remove_container_by_name(
                "codegraph-shared_repo", tenant_id="tenant-A"
            )
        assert result is True
        mock_client.delete_workspace.assert_awaited_once_with(
            "codegraph-shared_repo", headers={"X-Tenant-Id": "tenant-A"}
        )

    # 1e — find_existing_container must also fail closed on an unlabelled
    # same-repo container: force_rebuild feeds this straight into a delete.
    async def test_adapter_find_existing_rejects_unlabelled_container(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "unlabelled",
                    "repo_url": "https://github.com/o/r",
                    "status": "running",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_existing_container(
                "https://github.com/o/r", tenant_id="tenant-9"
            )
        assert result is None

    # 9b — list_containers re-verifies ownership client-side and drops both
    # cross-tenant AND unlabelled containers. Both routers fall back to a
    # by-name scan over this list when find_container_by_name returns None,
    # so an unfiltered list re-opens the by-name hole.
    async def test_adapter_list_containers_drops_unlabelled_and_cross_tenant(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value={
                "containers": [
                    {
                        "container_name": "mine",
                        "api_endpoint": "",
                        "codebase_path": "/app/code",
                        "repo_url": "https://github.com/o/mine",
                        "branch": "main",
                        "status": "running",
                        "tenant_id": "tenant-A",
                    },
                    {
                        "container_name": "unlabelled",
                        "api_endpoint": "",
                        "codebase_path": "/app/code",
                        "repo_url": "https://github.com/o/unlabelled",
                        "branch": "main",
                        "status": "running",
                    },
                    {
                        "container_name": "theirs",
                        "api_endpoint": "",
                        "codebase_path": "/app/code",
                        "repo_url": "https://github.com/o/theirs",
                        "branch": "main",
                        "status": "running",
                        "tenant_id": "tenant-B",
                    },
                ]
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers(tenant_id="tenant-A")
        assert [c.container_name for c in result] == ["mine"]

    # 9c — control: with no tenant (admin / internal, unscoped) the list is not
    # filtered, so the unlabelled-legacy view stays reachable off the
    # tenant-scoped path.
    async def test_adapter_list_containers_unscoped_keeps_everything(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value={
                "containers": [
                    {
                        "container_name": "unlabelled",
                        "api_endpoint": "",
                        "codebase_path": "/app/code",
                        "repo_url": "https://github.com/o/unlabelled",
                        "branch": "main",
                        "status": "running",
                    },
                ]
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers()
        assert [c.container_name for c in result] == ["unlabelled"]

    # 8
    async def test_adapter_create_payload_never_carries_exact_name(self, adapter):
        """The /provision model (with exact_name) goes through the adapter; the
        coding-containers create schema is extra="forbid", so exact_name must
        not reach it, whether True or False."""
        from schemas.github_api_models import ProvisionRepositoryRequest

        for flag in (True, False):
            mock_client = AsyncMock()
            mock_client.create_workspace = AsyncMock(
                return_value={
                    "container_name": "q-0123456789ab",
                    "api_endpoint": "",
                    "codebase_path": "/app/code",
                    "repo_url": "https://github.com/o/r",
                    "branch": "main",
                    "status": "running",
                }
            )
            request = ProvisionRepositoryRequest(
                repo_url="https://github.com/o/r",
                branch="main",
                container_name="q-0123456789ab",
                exact_name=flag,
            )
            with patch("main.get_workspaces_client", return_value=mock_client):
                await adapter.create_container(request, env_vars={})
            payload = {
                k: v
                for k, v in mock_client.create_workspace.call_args.kwargs.items()
                if k != "headers"
            }
            assert "exact_name" not in payload
            assert payload["container_name"] == "q-0123456789ab"
            # Contract: the forwarded body must validate against the shared
            # extra="forbid" create schema, so ANY leaked field fails here.
            from coding_client.models.container import GitHubRepoRequest

            GitHubRepoRequest(**payload)

    async def test_adapter_create_container_payload(self, adapter):
        from coding_client.models.container import GitHubRepoRequest

        mock_client = AsyncMock()
        mock_client.create_workspace = AsyncMock(
            return_value={
                "container_name": "ctr-new",
                "api_endpoint": "",
                "codebase_path": "/app/code",
                "repo_url": "https://github.com/o/r",
                "branch": "dev",
                "status": "running",
            }
        )
        request = GitHubRepoRequest(
            repo_url="https://github.com/o/r",
            branch="dev",
            access_token="tok-123",
            container_name="my-ctr",
            description="migration coverage",
            pr_number=7,
            provision_type="pull_request",
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.create_container(request, env_vars={"FOO": "bar"})
        # CodingWorkspacesClient.create_workspace(headers=..., **workspace) sends
        # the workspace kwargs as the JSON body; ``headers`` is the only non-body
        # kwarg, so everything else IS the payload.
        payload = {
            k: v for k, v in mock_client.create_workspace.call_args.kwargs.items() if k != "headers"
        }
        assert payload["repo_url"] == "https://github.com/o/r"
        assert payload["branch"] == "dev"
        assert payload["access_token"] == "tok-123"
        assert payload["container_name"] == "my-ctr"
        assert payload["description"] == "migration coverage"
        assert payload["pr_number"] == 7
        assert payload["provision_type"] == "pull_request"
        assert "env_vars" not in payload
        # The route is no longer hand-built here — the typed client owns
        # POST /api/workspaces/ — so assert the typed call, not a URL string.
        mock_client.create_workspace.assert_awaited_once()
        assert result.container_name == "ctr-new"

    # 9
    async def test_adapter_list_containers_with_filter(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value=[
                {
                    "container_name": "ctr-1",
                    "api_endpoint": "",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "r",
                    "branch": "main",
                    # Owned by the requesting tenant: this test is about the
                    # header-vs-query-param scoping, and the list is now
                    # additionally re-verified client-side (fail closed).
                    "tenant_id": "tenant-42",
                },
            ]
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers(tenant_id="tenant-42")
        # coding-containers scopes by the X-Tenant-Id header, not a query param.
        headers = mock_client.list_workspaces.call_args.kwargs.get("headers")
        assert headers == {"X-Tenant-Id": "tenant-42"}
        assert len(result) == 1

    # 10 — create_container forwards the request tenant as an X-Tenant-Id header
    # so the new container is labelled with the caller's tenant.
    async def test_adapter_create_container_forwards_tenant(self, adapter):
        mock_client = AsyncMock()
        mock_client.create_workspace = AsyncMock(
            return_value={
                "container_name": "ctr-new",
                "api_endpoint": "",
                "codebase_path": "/app/code",
                "repo_url": "https://github.com/o/r",
                "branch": "main",
                "status": "running",
            }
        )
        request = SimpleNamespace(
            repo_url="https://github.com/o/r", branch="main", access_token="t", container_name="c"
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            await adapter.create_container(request, tenant_id="tenant-7")
        headers = mock_client.create_workspace.call_args.kwargs.get("headers")
        assert headers == {"X-Tenant-Id": "tenant-7"}


# ---------------------------------------------------------------------------
# GitHubProvisioningService tests
# ---------------------------------------------------------------------------


class TestGitHubProvisioningService:
    """Tests for the provisioning orchestration layer."""

    @pytest.fixture
    def docker_svc(self):
        svc = AsyncMock()
        svc.find_existing_container = AsyncMock(return_value=None)
        # A bare AsyncMock attribute returns a truthy MagicMock; default the
        # by-name lookup to "no such workspace" so force_rebuild sees nothing.
        svc.find_container_by_name = AsyncMock(return_value=None)
        svc.remove_container = AsyncMock(return_value=True)
        svc.remove_container_by_name = AsyncMock(return_value=True)
        svc.create_container = AsyncMock(return_value=_make_container_info())
        svc.list_containers = AsyncMock(return_value=[])
        return svc

    @pytest.fixture
    def github_svc(self):
        svc = AsyncMock()
        svc.validate_repository_access = AsyncMock()
        svc.validate_pull_request_access = AsyncMock()
        svc.get_branch_head_commit = AsyncMock(return_value="abc123")
        svc.get_pull_request_head_commit = AsyncMock(return_value="def456")
        return svc

    @pytest.fixture
    def provisioner(self, docker_svc, github_svc):
        return GitHubProvisioningService(
            docker_service=docker_svc,
            github_service=github_svc,
        )

    # 10
    async def test_provision_existing_container_reuse(self, provisioner, docker_svc):
        existing = _make_container_info()
        docker_svc.find_existing_container.return_value = existing
        request = _make_request(force_rebuild=False)

        result = await provisioner.provision_repository(request)
        assert result["container_info"] is existing
        assert result["commit_sha"] == "abc123"
        docker_svc.create_container.assert_not_awaited()

    # 11
    async def test_provision_force_rebuild(self, provisioner, docker_svc):
        existing = _make_container_info()
        docker_svc.find_existing_container.return_value = existing
        request = _make_request(force_rebuild=True)

        result = await provisioner.provision_repository(request)
        docker_svc.remove_container.assert_awaited_once_with(existing, tenant_id=None)
        docker_svc.create_container.assert_awaited_once()
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"
        assert result["commit_sha"] == "abc123"

    # 11b — SECURITY: force_rebuild threads the caller's tenant into the
    # by-name delete so it can only remove THIS tenant's container.
    async def test_provision_force_rebuild_threads_tenant_to_delete(self, provisioner, docker_svc):
        docker_svc.find_existing_container.return_value = None  # take the by-name branch
        docker_svc.find_container_by_name.return_value = _make_container_info()
        request = _make_request(force_rebuild=True)
        await provisioner.provision_repository(request, tenant_id="tenant-A")
        docker_svc.find_container_by_name.assert_awaited_once_with(
            "codegraph-test_repo", tenant_id="tenant-A"
        )
        docker_svc.remove_container_by_name.assert_awaited_once()
        call = docker_svc.remove_container_by_name.call_args
        assert call.kwargs.get("tenant_id") == "tenant-A"

    # 11b2 — #136: force_rebuild WITHOUT exact_name deleted the workspace NAMED
    # container_name / codegraph-<repo> without checking its repo. A stale or
    # mistyped name (or two owners' repos sharing a repo name) destroyed another
    # repo's workspace. The named workspace must hold THIS repo or nothing goes.
    @pytest.mark.parametrize("container_name", ["q-0123456789ab", ""])
    async def test_force_rebuild_named_workspace_other_repo_is_409_and_touches_nothing(
        self, provisioner, docker_svc, container_name
    ):
        from services.github_provisioning import WorkspaceRepoMismatchError

        expected = container_name or "codegraph-test_repo"
        other = _make_container_info(
            container_name=expected,
            repo_url="https://github.com/other-org/test-repo",
        )
        docker_svc.find_container_by_name.return_value = other
        request = _make_request(container_name=container_name, force_rebuild=True)

        with pytest.raises(WorkspaceRepoMismatchError) as exc:
            await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert exc.value.code == "workspace_repo_mismatch"
        assert exc.value.http_status == 409
        docker_svc.find_container_by_name.assert_awaited_once_with(expected, tenant_id="tenant-A")
        docker_svc.remove_container_by_name.assert_not_awaited()
        docker_svc.remove_container.assert_not_awaited()
        docker_svc.create_container.assert_not_awaited()

    # The refusal must be ATOMIC: this repo's own workspace (found by repo) must
    # not be removed before the named workspace is found to hold another repo.
    async def test_force_rebuild_named_mismatch_does_not_remove_own_existing_first(
        self, provisioner, docker_svc
    ):
        from services.github_provisioning import WorkspaceRepoMismatchError

        own = _make_container_info(container_name="codegraph-test_repo")
        docker_svc.find_existing_container.return_value = own
        docker_svc.find_container_by_name.return_value = _make_container_info(
            container_name="q-x", repo_url="https://github.com/other-org/other-repo"
        )
        request = _make_request(container_name="q-x", force_rebuild=True)

        with pytest.raises(WorkspaceRepoMismatchError) as exc:
            await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert exc.value.http_status == 409
        docker_svc.find_container_by_name.assert_awaited_once_with("q-x", tenant_id="tenant-A")
        docker_svc.remove_container.assert_not_awaited()
        docker_svc.remove_container_by_name.assert_not_awaited()
        docker_svc.create_container.assert_not_awaited()

    # When the named workspace IS ``existing`` (same name), it is removed once.
    async def test_force_rebuild_named_is_existing_removed_once(self, provisioner, docker_svc):
        own = _make_container_info(container_name="codegraph-test_repo")
        # Two DISTINCT objects for the same workspace: the two lookups are separate HTTP calls, so
        # the de-dup must compare names, never object identity (an `is` check would remove twice).
        same_by_name = _make_container_info(container_name="codegraph-test_repo")
        docker_svc.find_existing_container.return_value = own
        docker_svc.find_container_by_name.return_value = same_by_name
        request = _make_request(force_rebuild=True)

        await provisioner.provision_repository(request, tenant_id="tenant-A")

        docker_svc.remove_container.assert_awaited_once_with(own, tenant_id="tenant-A")
        docker_svc.remove_container_by_name.assert_not_awaited()
        docker_svc.create_container.assert_awaited_once()

    async def test_force_rebuild_named_workspace_same_repo_variant_is_removed_then_created(
        self, provisioner, docker_svc
    ):
        named = _make_container_info(
            container_name="q-0123456789ab",
            repo_url="https://github.com/Test-Org/test-repo.git",
        )
        docker_svc.find_container_by_name.return_value = named
        request = _make_request(container_name="q-0123456789ab", force_rebuild=True)

        result = await provisioner.provision_repository(request, tenant_id="tenant-A")

        docker_svc.remove_container_by_name.assert_awaited_once_with(
            "q-0123456789ab", tenant_id="tenant-A"
        )
        docker_svc.create_container.assert_awaited_once()
        assert result["container_info"] is docker_svc.create_container.return_value

    async def test_force_rebuild_no_workspace_of_that_name_creates_without_removal(
        self, provisioner, docker_svc
    ):
        docker_svc.find_container_by_name.return_value = None
        request = _make_request(container_name="q-0123456789ab", force_rebuild=True)

        result = await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert "error" not in result
        docker_svc.remove_container_by_name.assert_not_awaited()
        docker_svc.remove_container.assert_not_awaited()
        docker_svc.create_container.assert_awaited_once()

    @pytest.mark.parametrize("repo_url", [None, ""])
    async def test_force_rebuild_named_workspace_without_repo_url_is_refused(
        self, provisioner, docker_svc, repo_url
    ):
        from services.github_provisioning import WorkspaceRepoMismatchError

        unattributed = _make_container_info(container_name="q-0123456789ab", repo_url=repo_url)
        docker_svc.find_container_by_name.return_value = unattributed
        request = _make_request(container_name="q-0123456789ab", force_rebuild=True)

        with pytest.raises(WorkspaceRepoMismatchError) as exc:
            await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert exc.value.http_status == 409
        assert "no recorded repository" in str(exc.value)
        docker_svc.remove_container_by_name.assert_not_awaited()
        docker_svc.remove_container.assert_not_awaited()
        docker_svc.create_container.assert_not_awaited()

    # 11c — exact_name: a SHARED same-repo workspace must NOT be handed back for
    # a dedicated name; the by-name lookup returns the dedicated one if it exists.
    async def test_provision_exact_name_reuses_only_the_named_workspace(
        self, provisioner, docker_svc
    ):
        shared = _make_container_info(container_name="codegraph-test_repo")
        dedicated = _make_container_info(container_name="q-0123456789ab")
        docker_svc.find_existing_container.return_value = shared
        docker_svc.find_container_by_name = AsyncMock(return_value=dedicated)
        request = _make_request(container_name="q-0123456789ab", exact_name=True)

        result = await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert result["container_info"] is dedicated
        docker_svc.find_container_by_name.assert_awaited_once_with(
            "q-0123456789ab", tenant_id="tenant-A"
        )
        docker_svc.find_existing_container.assert_not_awaited()
        docker_svc.create_container.assert_not_awaited()

    # 11d — exact_name with no workspace of that name: create it, never fall
    # back to the shared same-repo workspace.
    async def test_provision_exact_name_missing_creates_instead_of_reusing_shared(
        self, provisioner, docker_svc
    ):
        docker_svc.find_existing_container.return_value = _make_container_info()
        docker_svc.find_container_by_name = AsyncMock(return_value=None)
        request = _make_request(container_name="q-0123456789ab", exact_name=True)

        await provisioner.provision_repository(request, tenant_id="tenant-A")

        docker_svc.find_existing_container.assert_not_awaited()
        docker_svc.create_container.assert_awaited_once()

    # 11e — exact_name without a container_name has no name to match: keep the
    # repo-url path.
    async def test_provision_exact_name_without_container_name_uses_repo_url(
        self, provisioner, docker_svc
    ):
        existing = _make_container_info()
        docker_svc.find_existing_container.return_value = existing
        docker_svc.find_container_by_name = AsyncMock(return_value=None)
        request = _make_request(container_name="", exact_name=True)

        result = await provisioner.provision_repository(request)

        assert result["container_info"] is existing
        docker_svc.find_container_by_name.assert_not_awaited()

    # 11f — exact_name=False (explicit) is the unchanged repo-url reuse path.
    async def test_provision_exact_name_false_keeps_repo_url_reuse(self, provisioner, docker_svc):
        existing = _make_container_info()
        docker_svc.find_existing_container.return_value = existing
        docker_svc.find_container_by_name = AsyncMock(return_value=None)
        request = _make_request(container_name="q-0123456789ab", exact_name=False)

        result = await provisioner.provision_repository(request)

        assert result["container_info"] is existing
        docker_svc.find_container_by_name.assert_not_awaited()

    # 11g — exact_name only reuses a workspace that holds the REQUESTED repo.
    # An auditor got tenant A's OTHER-repo workspace back as a success for a
    # test-repo request, carrying test-repo's commit_sha, with nothing created.
    async def test_provision_exact_name_reuses_when_repo_matches(self, provisioner, docker_svc):
        named = _make_container_info(container_name="q-0123456789ab")
        docker_svc.find_container_by_name = AsyncMock(return_value=named)
        request = _make_request(container_name="q-0123456789ab", exact_name=True)

        result = await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert result["container_info"] is named
        docker_svc.create_container.assert_not_awaited()

    @pytest.mark.parametrize("force_rebuild", [False, True])
    async def test_provision_exact_name_repo_mismatch_is_409_and_touches_nothing(
        self, provisioner, docker_svc, force_rebuild
    ):
        from services.github_provisioning import WorkspaceRepoMismatchError
        from services.tenant_binding import tenant_error_http_exception

        other = _make_container_info(
            container_name="q-0123456789ab",
            repo_url="https://github.com/test-org/OTHER-repo",
        )
        docker_svc.find_container_by_name = AsyncMock(return_value=other)
        request = _make_request(
            container_name="q-0123456789ab", exact_name=True, force_rebuild=force_rebuild
        )

        with pytest.raises(WorkspaceRepoMismatchError) as exc:
            await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert exc.value.code == "workspace_repo_mismatch"
        assert exc.value.http_status == 409
        # Both repos are named, so the caller can see what the name is bound to.
        assert "OTHER-repo" in str(exc.value)
        assert "https://github.com/test-org/test-repo" in str(exc.value)
        # The route's existing TenantRepoTokenError mapping yields the 409.
        assert tenant_error_http_exception(exc.value).status_code == 409
        # Nothing created, nothing deleted — not even under force_rebuild.
        docker_svc.create_container.assert_not_awaited()
        docker_svc.remove_container.assert_not_awaited()
        docker_svc.remove_container_by_name.assert_not_awaited()

    @pytest.mark.parametrize("repo_url", [None, ""])
    async def test_provision_exact_name_workspace_without_repo_url_is_refused(
        self, provisioner, docker_svc, repo_url
    ):
        from services.github_provisioning import WorkspaceRepoMismatchError

        unattributed = _make_container_info(container_name="q-0123456789ab", repo_url=repo_url)
        docker_svc.find_container_by_name = AsyncMock(return_value=unattributed)
        request = _make_request(container_name="q-0123456789ab", exact_name=True)

        with pytest.raises(WorkspaceRepoMismatchError) as exc:
            await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert exc.value.http_status == 409
        assert "no recorded repository" in str(exc.value)
        docker_svc.create_container.assert_not_awaited()

    @pytest.mark.parametrize(
        "held",
        [
            "https://github.com/test-org/test-repo.git",
            "https://github.com/test-org/test-repo/",
            "https://github.com/Test-Org/Test-Repo",
            "https://github.com/TEST-ORG/test-repo.git/",
        ],
    )
    async def test_provision_exact_name_url_variants_still_reuse(
        self, provisioner, docker_svc, held
    ):
        named = _make_container_info(container_name="q-0123456789ab", repo_url=held)
        docker_svc.find_container_by_name = AsyncMock(return_value=named)
        request = _make_request(container_name="q-0123456789ab", exact_name=True)

        result = await provisioner.provision_repository(request, tenant_id="tenant-A")

        assert result["container_info"] is named
        docker_svc.create_container.assert_not_awaited()

    # 12
    async def test_provision_validation_failure_shortcircuit(self, provisioner, github_svc):
        github_svc.validate_repository_access.side_effect = ValueError("bad repo")
        request = _make_request()

        result = await provisioner.provision_repository(request)
        assert result["error_type"] == "validation"
        assert "bad repo" in result["error"]

    # 13
    async def test_provision_create_path(self, provisioner, docker_svc):
        request = _make_request()
        result = await provisioner.provision_repository(request)
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"
        assert result["container_info"].status == "running"
        assert result["commit_sha"] == "abc123"
        docker_svc.create_container.assert_awaited_once()

    # 14
    async def test_list_containers_delegation(self, provisioner, docker_svc):
        docker_svc.list_containers.return_value = [_make_container_info()]
        result = await provisioner.list_containers(tenant_id="t-1")
        docker_svc.list_containers.assert_awaited_once_with(tenant_id="t-1")
        assert len(result) == 1


class TestAutoIndexOnProvision:
    """Verify fire-and-forget full scan trigger after successful provisioning."""

    @pytest.fixture
    def provisioner(self):
        docker_svc = AsyncMock()
        docker_svc.find_existing_container = AsyncMock(return_value=None)
        docker_svc.remove_container_by_name = AsyncMock(return_value=True)
        docker_svc.create_container = AsyncMock(return_value=_make_container_info())
        github_svc = AsyncMock()
        github_svc.validate_repository_access = AsyncMock()
        github_svc.get_branch_head_commit = AsyncMock(return_value="abc123")
        return GitHubProvisioningService(
            docker_service=docker_svc,
            github_service=github_svc,
        )

    async def test_auto_index_fires_when_enabled(self, provisioner, monkeypatch):
        monkeypatch.setenv("AUTO_INDEX_ON_PROVISION", "true")
        trigger = AsyncMock()
        with patch("services.github_provisioning._trigger_full_scan", trigger):
            await provisioner.provision_repository(_make_request(), tenant_id="tenant-a")

        # asyncio.create_task schedules the coroutine; give the loop one tick to run it.
        import asyncio

        await asyncio.sleep(0)
        trigger.assert_awaited_once_with("codegraph-test_repo", "tenant-a")

    async def test_auto_index_skipped_when_disabled(self, provisioner, monkeypatch):
        monkeypatch.setenv("AUTO_INDEX_ON_PROVISION", "false")
        trigger = AsyncMock()
        with patch("services.github_provisioning._trigger_full_scan", trigger):
            await provisioner.provision_repository(_make_request())

        import asyncio

        await asyncio.sleep(0)
        trigger.assert_not_awaited()

    async def test_trigger_full_scan_swallows_http_errors(self, monkeypatch):
        import httpx

        from services.github_provisioning import _trigger_full_scan

        monkeypatch.setenv("ANALYSIS_URL", "http://coding-analysis:8688")

        class _FailClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                raise httpx.ConnectError("boom")

        with patch("httpx.AsyncClient", return_value=_FailClient()):
            # Must not raise.
            await _trigger_full_scan("codegraph-test_repo", "tenant-a")

    @pytest.mark.parametrize(
        ("trigger_name", "url_env", "url"),
        [
            ("_trigger_full_scan", "ANALYSIS_URL", "http://coding-analysis:8688"),
            ("_trigger_dependency_scan", "SECURITY_URL", "http://coding-security:8693"),
        ],
    )
    @pytest.mark.parametrize("tenant_id", ["tenant-a", None])
    async def test_scan_kickoff_always_sends_tenant_header(
        self, monkeypatch, trigger_name, url_env, url, tenant_id
    ):
        """Both kickoffs carry X-Tenant-Id even when the caller's tenant is unknown.

        With no header these calls were only accepted because coding-analysis /
        coding-security fall back to trusting any RFC1918 source IP, which covers
        the whole pod CIDR. Captured off the real httpx call, so dropping the
        header fails this test.

        ⚠️ The ``tenant_id=None`` leg needs a CONFIGURED DEFAULT_TENANT_ID to mean
        anything, and that precondition is now stated rather than assumed. Before
        coding-client 1.9.0 it was supplied invisibly by the library's built-in
        "n8n-dev" fallback, so this test was really pinning that literal. #36
        removed the fallback; without the patch below DEFAULT_TENANT_ID is None
        here, and "always sends the header" would only be satisfiable by putting
        a None into a header value — the exact defect #36 exists to prevent. The
        unset case is covered separately by
        ``test_scan_headers_omits_header_when_no_tenant_is_known``.
        """
        import services.github_provisioning as gp

        monkeypatch.setattr(gp, "DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")
        monkeypatch.setenv(url_env, url)
        captured: dict[str, Any] = {}

        class _CaptureClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                captured.update(kw)
                resp = MagicMock()
                resp.status_code = 202
                return resp

        with patch("httpx.AsyncClient", return_value=_CaptureClient()):
            await getattr(gp, trigger_name)("codegraph-test_repo", tenant_id)

        headers = captured["headers"]
        assert headers["X-Tenant-Id"] == (tenant_id or gp.DEFAULT_TENANT_ID)
        assert headers["X-Tenant-Id"].strip()

    async def test_dependency_scan_fires_when_enabled(self, provisioner, monkeypatch):
        monkeypatch.setenv("AUTO_INDEX_ON_PROVISION", "true")
        scan = AsyncMock()
        dep = AsyncMock()
        with (
            patch("services.github_provisioning._trigger_full_scan", scan),
            patch("services.github_provisioning._trigger_dependency_scan", dep),
        ):
            await provisioner.provision_repository(_make_request(), tenant_id="tenant-a")

        import asyncio

        await asyncio.sleep(0)
        scan.assert_awaited_once_with("codegraph-test_repo", "tenant-a")
        dep.assert_awaited_once_with("codegraph-test_repo", "tenant-a")

    async def test_dependency_scan_skipped_when_disabled(self, provisioner, monkeypatch):
        monkeypatch.setenv("AUTO_INDEX_ON_PROVISION", "false")
        scan = AsyncMock()
        dep = AsyncMock()
        with (
            patch("services.github_provisioning._trigger_full_scan", scan),
            patch("services.github_provisioning._trigger_dependency_scan", dep),
        ):
            await provisioner.provision_repository(_make_request())

        import asyncio

        await asyncio.sleep(0)
        scan.assert_not_awaited()
        dep.assert_not_awaited()

    async def test_trigger_dependency_scan_swallows_http_errors(self, monkeypatch):
        import httpx

        from services.github_provisioning import _trigger_dependency_scan

        class _FailClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                raise httpx.ConnectError("boom")

        with patch("httpx.AsyncClient", return_value=_FailClient()):
            # Must not raise.
            await _trigger_dependency_scan("codegraph-test_repo", "tenant-a")


class TestSecretsConsumer:
    """Verify _get_environment_variables resolves secret_config_id via secrets-service."""

    @pytest.fixture
    def provisioner(self):
        docker_svc = AsyncMock()
        github_svc = AsyncMock()
        return GitHubProvisioningService(
            docker_service=docker_svc,
            github_service=github_svc,
        )

    async def test_missing_secret_config_id_returns_none(self, provisioner):
        req = _make_request()
        req.secret_config_id = None
        # No httpx mock needed — the early-return short-circuits before any call.
        result = await provisioner._get_environment_variables(req)
        assert result is None

    async def test_happy_path_returns_combined_env_and_secrets(self, provisioner):
        """The happy path is an AUTHENTICATED caller, so supply the tenant.

        This used to pass no ``tenant_id`` at all and assert that
        ``req.user_id`` came back out in ``X-Tenant-Id`` — it was pinning the
        ``tenant_id or request.user_id or "anonymous"`` fallback in place. A
        user id is an identifier from a different namespace than a tenant, so
        that assertion described a cross-namespace substitution, not a happy
        path. The tenant is now supplied explicitly (which is what
        ``require_tenant`` yields in production) and ``user_id`` is set to a
        DIFFERENT value so the assertion proves the tenant wins rather than
        merely that *something* reached the header.
        """
        req = _make_request()
        req.secret_config_id = "sec_abc123"
        req.user_id = "user-42"

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                import httpx

                assert "sec_abc123" in url
                assert headers["X-Tenant-Id"] == "tenant-a"
                return httpx.Response(
                    200,
                    json={
                        "secret_id": "sec_abc123",
                        "user_id": "tenant-a",
                        "identifier": "project:x:github_token",
                        "env_vars": {"FOO": "bar"},
                        "secrets": {"GIT_TOKEN": "ghp_secret"},
                    },
                    request=httpx.Request("GET", url),
                )

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            result = await provisioner._get_environment_variables(req, tenant_id="tenant-a")

        assert result == {"FOO": "bar", "GIT_TOKEN": "ghp_secret"}

    async def test_404_returns_none(self, provisioner):
        req = _make_request()
        req.secret_config_id = "sec_missing"

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                import httpx

                return httpx.Response(
                    404, json={"detail": "not found"}, request=httpx.Request("GET", url)
                )

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            result = await provisioner._get_environment_variables(req)

        assert result is None  # caller will continue without token

    async def test_http_error_returns_none(self, provisioner):
        import httpx

        req = _make_request()
        req.secret_config_id = "sec_abc123"

        class _FailClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **kw):
                raise httpx.ConnectError("boom")

        with patch("httpx.AsyncClient", return_value=_FailClient()):
            result = await provisioner._get_environment_variables(req)

        assert result is None


# ---------------------------------------------------------------------------
# ProvisionRepositoryRequest — the exact_name field on the provision body
# ---------------------------------------------------------------------------


def test_provision_request_exact_name_defaults_false_and_accepts_true():
    from schemas.github_api_models import ProvisionRepositoryRequest

    base = {"repo_url": "https://github.com/o/r", "branch": "main"}
    assert ProvisionRepositoryRequest(**base).exact_name is False
    req = ProvisionRepositoryRequest(**base, container_name="q-0123456789ab", exact_name=True)
    assert req.exact_name is True
    assert req.container_name == "q-0123456789ab"
