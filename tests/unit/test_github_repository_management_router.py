"""Tests for repository management endpoints via real HTTP requests.

Covers: provision, list, status, create-upstream-pr.
Uses app.dependency_overrides for FastAPI Depends() injection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from coding_client.models.container import ContainerStatus

from schemas.github_api_models import ContainerInfo
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
    """Override the FastAPI dependency for get_github_provisioning_service."""
    app = async_client._transport.app
    app.dependency_overrides[get_github_provisioning_service] = lambda: svc
    return app


def _clear_overrides(async_client):
    app = async_client._transport.app
    app.dependency_overrides.clear()


# ===========================================================================
# Provision tests
# ===========================================================================


class TestProvisionSuccess:
    async def test_provision_success(self, async_client):
        container_info = ContainerInfo(
            container_name="test-ctr",
            api_endpoint="http://localhost:8700",
            status=ContainerStatus("running"),
            repo_url="https://github.com/owner/repo",
            codebase_path="/workspace",
            branch="main",
        )
        svc = _make_provision_service(
            provision_result={"container_info": container_info, "commit_sha": "abc123"}
        )
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["commit_sha"] == "abc123"
            assert body["container"]["container_name"] == "test-ctr"
        finally:
            _clear_overrides(async_client)


class TestProvisionInvalidToken:
    async def test_invalid_token_returns_400(self, async_client):
        svc = _make_provision_service()
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={
                    "repo_url": "https://github.com/owner/repo",
                    "branch": "main",
                    "access_token": "invalid",
                },
            )
            assert resp.status_code == 400
        finally:
            _clear_overrides(async_client)


class TestProvisionMalformedUrl:
    async def test_malformed_url_returns_422(self, async_client):
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


class TestProvisionTimeout:
    async def test_timeout_returns_408(self, async_client):
        svc = _make_provision_service()
        svc.provision_repository = AsyncMock(side_effect=TimeoutError("timed out"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 408
        finally:
            _clear_overrides(async_client)


class TestProvisionNotFound:
    async def test_not_found_returns_404(self, async_client):
        from coding_client.exceptions import NotFoundException

        svc = _make_provision_service()
        svc.provision_repository = AsyncMock(side_effect=NotFoundException(message="Repo gone"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 404
        finally:
            _clear_overrides(async_client)


class TestProvisionInternalError:
    async def test_internal_error_returns_500(self, async_client):
        svc = _make_provision_service()
        svc.provision_repository = AsyncMock(side_effect=RuntimeError("boom"))
        _override_service(async_client, svc)
        try:
            resp = await async_client.post(
                f"{BASE}/provision",
                json={"repo_url": "https://github.com/owner/repo", "branch": "main"},
            )
            assert resp.status_code == 500
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# List repositories tests
# ===========================================================================


class TestListReposEmpty:
    async def test_list_repos_empty(self, async_client):
        svc = _make_provision_service(containers=[])
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/repositories")
            assert resp.status_code == 200
            body = resp.json()
            assert body["repositories"] == []
            assert "0 repository" in body["message"]
        finally:
            _clear_overrides(async_client)


class TestListReposNonEmpty:
    async def test_list_repos_returns_containers(self, async_client):
        c = _make_container()
        svc = _make_provision_service(containers=[c])
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/repositories")
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["repositories"]) == 1
            assert body["repositories"][0]["container_name"] == "test-container"
        finally:
            _clear_overrides(async_client)


class TestListReposFilter:
    async def test_containers_without_repo_url_excluded(self, async_client):
        c_with = _make_container(repo_url="https://github.com/o/r")
        c_without = _make_container(container_name="no-repo", repo_url="")
        svc = _make_provision_service(containers=[c_with, c_without])
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/repositories")
            assert resp.status_code == 200
            assert len(resp.json()["repositories"]) == 1
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Status tests
# ===========================================================================


class TestStatusDirectFind:
    async def test_status_direct_find(self, async_client):
        c = _make_container()
        svc = _make_provision_service(find_result=c)
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/status/test-container")
            assert resp.status_code == 200
            body = resp.json()
            assert body["container_name"] == "test-container"
            assert body["status"] == "running"
        finally:
            _clear_overrides(async_client)


class TestStatusFallbackList:
    async def test_status_fallback_list(self, async_client):
        c = _make_container(container_name="my-ctr")
        svc = _make_provision_service(find_result=None, containers=[c])
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/status/my-ctr")
            assert resp.status_code == 200
            assert resp.json()["container_name"] == "my-ctr"
        finally:
            _clear_overrides(async_client)


class TestStatusNotFound:
    async def test_status_not_found_returns_404(self, async_client):
        svc = _make_provision_service(find_result=None, containers=[])
        _override_service(async_client, svc)
        try:
            resp = await async_client.get(f"{BASE}/status/ghost")
            assert resp.status_code == 404
        finally:
            _clear_overrides(async_client)


# ===========================================================================
# Create upstream PR tests
# ===========================================================================


class TestCreateUpstreamPRSuccess:
    @patch("routers.github_repository_management.GitHubIntegration")
    async def test_success(self, mock_cls, async_client):
        mock_cls.create_upstream_pull_request = AsyncMock(
            return_value={
                "success": True,
                "pr_number": 42,
                "pr_url": "https://github.com/owner/repo/pull/42",
                "message": "PR created",
                "original_repo": "owner/repo",
                "system_fork": "nonagenticai",
                "branch_name": "feature-branch",
            }
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
        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["pr_number"] == 42
        # ⛔ No GitHubIntegration is constructed: a no-arg construct resolved an
        # OWNERLESS credential (home installation / PAT). The tenant is forwarded.
        mock_cls.assert_not_called()
        kwargs = mock_cls.create_upstream_pull_request.await_args.kwargs
        assert kwargs["container_name"] == "test-ctr"
        assert kwargs["tenant_id"]


class TestCreateUpstreamPRFailure:
    @patch("routers.github_repository_management.GitHubIntegration")
    async def test_failure_returns_error(self, mock_cls, async_client):
        mock_cls.create_upstream_pull_request = AsyncMock(
            return_value={
                "success": False,
                "error": "Branch not found",
                "status_code": 404,
            }
        )
        resp = await async_client.post(
            f"{BASE}/create-upstream-pr",
            json={
                "container_name": "test-ctr",
                "title": "My PR",
                "message": "Some changes",
                "head_branch": "ghost-branch",
                "base_branch": "main",
            },
        )
        assert resp.status_code == 404
