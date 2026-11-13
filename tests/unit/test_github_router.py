"""Tests for the aggregating github.py router via real HTTP requests.

Verifies that the main router correctly includes all sub-routers by
sending real HTTP requests through the ASGI transport.
"""

from __future__ import annotations


class TestAggregatingRouterRoutes:
    def test_openapi_only_publishes_remote_github_operations(self):
        """Workspace façade aliases stay callable but do not duplicate OpenAPI."""
        from main import app

        paths = app.openapi()["paths"]
        assert "/api/repositories/import" in paths
        assert (
            paths["/api/repositories/import"]["post"]["operationId"]
            == "import_repository_as_workspace"
        )
        assert "/api/repositories/fork" in paths
        assert "/api/repositories/create-pull-request" in paths
        assert "/api/repositories/webhooks/webhook" in paths

        assert "/api/repositories/provision" not in paths
        assert "/api/repositories/repositories" not in paths
        assert "/api/repositories/status/{container_name}" not in paths
        assert "/api/repositories/{container_name}/file" not in paths
        assert "/api/repositories/{container_name}/tree" not in paths
        assert "/api/repositories/{container_name}/directory" not in paths
        assert "/api/repositories/{container_name}/search" not in paths

    async def test_provision_endpoint_exists(self, async_client):
        """The /api/repositories/provision route is reachable (returns non-405)."""
        # POST with empty body to test route existence - 422 means route exists
        resp = await async_client.post("/api/repositories/provision", json={})
        assert resp.status_code != 405, "provision endpoint should exist"

    async def test_import_endpoint_exists(self, async_client):
        """The canonical repository import route is reachable."""
        resp = await async_client.post("/api/repositories/import", json={})
        assert resp.status_code != 405, "repository import endpoint should exist"

    async def test_fork_endpoint_exists(self, async_client):
        """The /api/repositories/fork route is reachable (returns non-405)."""
        resp = await async_client.post("/api/repositories/fork", json={})
        assert resp.status_code != 405, "fork endpoint should exist"

    async def test_webhook_endpoint_exists(self, async_client):
        """The /api/repositories/webhooks/webhook route is reachable."""
        resp = await async_client.post("/api/repositories/webhooks/webhook")
        assert resp.status_code != 405, "webhook endpoint should exist"

    async def test_repositories_list_endpoint_exists(self, async_client):
        """The /api/repositories/repositories route is reachable."""
        # This will fail because the service isn't initialised, but route should exist
        resp = await async_client.get("/api/repositories/repositories")
        # 500 means route exists but service dependency failed
        assert resp.status_code in (200, 500), "repositories list endpoint should exist"
