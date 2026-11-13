"""Extended tests for repository management endpoints via real HTTP requests.

Uses app.dependency_overrides for FastAPI Depends() injection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from coding_client.models.container import ContainerStatus
from fastapi import HTTPException

from services.github_provisioning import get_github_provisioning_service

BASE = "/api/repositories"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container(**overrides):
    defaults = {
        "container_name": "test-container",
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "status": ContainerStatus("running"),
        "api_endpoint": "http://localhost:8700",
        "code_graph_id": "cg-1",
        "description": "Test repo",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_provision_service(provision_result=None, containers=None, find_result=None):
    svc = AsyncMock()
    if provision_result is not None:
        svc.provision_repository = AsyncMock(return_value=provision_result)
    if containers is not None:
        svc.list_containers = AsyncMock(return_value=containers)
    svc.docker_service = AsyncMock()
    svc.docker_service.find_container_by_name = AsyncMock(return_value=find_result)
    return svc


def _override_service(async_client, svc):
    app = async_client._transport.app
    app.dependency_overrides[get_github_provisioning_service] = lambda: svc
    return app


def _clear_overrides(async_client):
    async_client._transport.app.dependency_overrides.clear()


# ===========================================================================
# Provision — invalid repo_url
# ===========================================================================


class TestProvisionInvalidRepoUrl:
    async def test_empty_repo_url_returns_422(self, async_client):
        svc = _make_provision_service()
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "", "branch": "main"},
            )
            # Pydantic validates repo_url pattern at model level, returns 422
            assert resp.status_code == 422
        finally:
            _clear_overrides(async_client)

    async def test_non_github_url_returns_422(self, async_client):
        svc = _make_provision_service()
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "ftp://not-github.com/foo/bar", "branch": "main"},
            )
            # Pydantic validates repo_url pattern at model level, returns 422
            assert resp.status_code == 422
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Provision — ValueError
# ===========================================================================


class TestProvisionValueError:
    async def test_value_error_returns_400(self, async_client):
        svc = _make_provision_service()
        svc.provision_repository = AsyncMock(side_effect=ValueError("bad input data"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 400
            data = resp.json()
            msg = data.get("error", {}).get("message", data.get("message", ""))
            assert "bad input data" in msg
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Provision — HTTPException re-raised
# ===========================================================================


class TestProvisionHTTPException:
    async def test_http_exception_passthrough(self, async_client):
        svc = _make_provision_service()
        svc.provision_repository = AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Forbidden by upstream")
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 403
            data = resp.json()
            msg = data.get("error", {}).get("message", data.get("detail", ""))
            assert msg == "Forbidden by upstream"
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# List repositories — error paths
# ===========================================================================


class TestListReposErrors:
    async def test_os_error_returns_503(self, async_client):
        svc = _make_provision_service()
        svc.list_containers = AsyncMock(side_effect=OSError("disk full"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/repositories")
            assert resp.status_code == 503
        finally:
            _clear_overrides(async_client)

    async def test_generic_exception_returns_500(self, async_client):
        svc = _make_provision_service()
        svc.list_containers = AsyncMock(side_effect=RuntimeError("unexpected"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/repositories")
            assert resp.status_code == 500
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Status — error paths
# ===========================================================================


class TestStatusErrors:
    async def test_runtime_error_returns_404(self, async_client):
        svc = _make_provision_service()
        svc.docker_service.find_container_by_name = AsyncMock(
            side_effect=RuntimeError("client not init")
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/status/test-ctr")
            assert resp.status_code == 404
        finally:
            _clear_overrides(async_client)

    async def test_generic_exception_returns_500(self, async_client):
        svc = _make_provision_service()
        svc.docker_service.find_container_by_name = AsyncMock(
            side_effect=TypeError("something weird")
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/status/test-ctr")
            assert resp.status_code == 500
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Create upstream PR — unexpected exception
# ===========================================================================


class TestCreateUpstreamPRError:
    @patch("routers.github_repository_management.GitHubIntegration")
    async def test_unexpected_error_returns_500(self, mock_cls, async_client):
        mock_cls.create_upstream_pull_request = AsyncMock(
            side_effect=RuntimeError("connection reset")
        )
        resp = await async_client.post(
            f"{BASE}/create-upstream-pr",
            json={
                "container_name": "test-ctr",
                "title": "My PR",
                "message": "Some changes",
                "head_branch": "feature-branch",
                "base_branch": "main",
            },
        )
        assert resp.status_code == 500
        data = resp.json()
        msg = data.get("error", {}).get("message", data.get("message", ""))
        assert "connection reset" in msg


# ===========================================================================
# Provision — container_info type handling
# ===========================================================================


class TestProvisionContainerInfoTypes:
    async def test_container_info_invalid_type_returns_500(self, async_client):
        svc = _make_provision_service(
            provision_result={"container_info": "just-a-string", "commit_sha": "abc123"}
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 500
        finally:
            _clear_overrides(async_client)

    async def test_container_info_bad_dict_returns_500(self, async_client):
        svc = _make_provision_service(
            provision_result={"container_info": {"unexpected_key": "value"}, "commit_sha": "abc"}
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 500
        finally:
            _clear_overrides(async_client)
