"""Extended unit tests for GitHubProvisioningService — covers uncovered lines."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github_provisioning import (
    GitHubProvisioningService,
    _ContainersHTTPAdapter,
    get_github_provisioning_service,
)

# ---------------------------------------------------------------------------
# Helpers (mirrors existing test helpers)
# ---------------------------------------------------------------------------


def _make_request(**overrides):
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
# _ContainersHTTPAdapter – uncovered branches
# ---------------------------------------------------------------------------


class TestContainersHTTPAdapterExtended:
    @pytest.fixture
    def adapter(self):
        return _ContainersHTTPAdapter()

    # Line 51: find_container_by_name returns dict -> wraps in ContainerInfo
    async def test_find_by_name_returns_dict(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "container_name": "ctr-x",
                "api_endpoint": "http://x",
                "codebase_path": "/app/code",
                "status": "running",
                "repo_url": "https://github.com/o/r",
                "branch": "main",
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name("ctr-x")
        assert result.container_name == "ctr-x"
        assert result.api_endpoint == "http://x"
        assert result.status == "running"
        assert result.repo_url == "https://github.com/o/r"

    # Line 51: find_container_by_name returns non-dict (already ContainerInfo)
    async def test_find_by_name_returns_object(self, adapter):
        obj = _make_container_info(container_name="ctr-obj")
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(return_value=obj)
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name("ctr-obj")
        assert result is obj

    # Regression: find_container_by_name unwraps coding-containers status envelope
    # (GET /api/containers/{name} returns {status, message, ..., container: {...}}).
    # Previously the envelope was fed straight into ContainerInfo and raised
    # "6 validation errors for ContainerInfo" on the fork path.
    async def test_find_by_name_unwraps_envelope(self, adapter):
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(
            return_value={
                "status": "success",
                "message": "Container 'ctr-x' details retrieved",
                "timestamp": "2026-05-20T00:00:00Z",
                "request_id": "req_envelope",
                "tenant_id": None,
                "container": {
                    "container_name": "ctr-x",
                    "api_endpoint": "http://x",
                    "codebase_path": "/app/code",
                    "status": "running",
                    "repo_url": "https://github.com/o/r",
                    "branch": "main",
                },
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.find_container_by_name("ctr-x")
        assert result.container_name == "ctr-x"
        assert result.repo_url == "https://github.com/o/r"

    # Line 55: find_container_by_name re-raises non-404 ServiceError
    async def test_find_by_name_reraises_non_404(self, adapter):
        from coding_client.clients import ServiceError

        err = ServiceError(500, "server error")
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=err)
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(ServiceError):
                await adapter.find_container_by_name("bad")

    # Line 77: remove_container_by_name re-raises non-404 ServiceError
    async def test_remove_by_name_reraises_non_404(self, adapter):
        from coding_client.clients import ServiceError

        err = ServiceError(500, "internal")
        mock_client = AsyncMock()
        mock_client.delete_workspace = AsyncMock(side_effect=err)
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(ServiceError):
                await adapter.remove_container_by_name("fail")

    # Lines 111-113: list_containers returns non-list -> returns []
    async def test_list_containers_non_list_response(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(return_value={"unexpected": "dict"})
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers()
        assert result == []

    # Regression: list_containers unwraps coding-containers envelope
    # ({status, message, ..., containers: [...], total}).
    async def test_list_containers_unwraps_envelope(self, adapter):
        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(
            return_value={
                "status": "success",
                "message": "Found 1 containers",
                "containers": [
                    {
                        "container_name": "ctr-1",
                        "api_endpoint": "http://x",
                        "codebase_path": "/app/code",
                        "status": "running",
                        "repo_url": "https://github.com/o/r",
                        "branch": "main",
                    },
                ],
                "total": 1,
            }
        )
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers()
        assert len(result) == 1
        assert result[0].container_name == "ctr-1"

    # Lines 111-113: list_containers ServiceError -> returns []
    async def test_list_containers_service_error(self, adapter):
        from coding_client.clients import ServiceError

        mock_client = AsyncMock()
        mock_client.list_workspaces = AsyncMock(side_effect=ServiceError(500, "boom"))
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await adapter.list_containers()
        assert result == []


# ---------------------------------------------------------------------------
# GitHubProvisioningService – uncovered branches
# ---------------------------------------------------------------------------


class TestGitHubProvisioningServiceExtended:
    @pytest.fixture
    def docker_svc(self):
        svc = AsyncMock()
        svc.find_existing_container = AsyncMock(return_value=None)
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

    # Line 160: github_service is None -> RuntimeError
    async def test_provision_no_github_service_raises_runtime_error(self, docker_svc):
        prov = GitHubProvisioningService(docker_service=docker_svc, github_service=None)
        result = await prov.provision_repository(_make_request())
        assert result["error_type"] == "runtime"
        assert "not initialized" in result["error"]

    # Lines 162-169: PR provision type with pr_number
    async def test_provision_pull_request_type(self, provisioner, github_svc, docker_svc):
        request = _make_request(provision_type="pull_request", pr_number=42)
        result = await provisioner.provision_repository(request)
        github_svc.validate_pull_request_access.assert_awaited_once_with(request, tenant_id=None)
        github_svc.get_pull_request_head_commit.assert_awaited_once()
        assert result["commit_sha"] == "def456"
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"

    # Lines 167-169: PR commit SHA retrieval fails with ValueError
    async def test_provision_pr_commit_sha_value_error(self, provisioner, github_svc):
        github_svc.get_pull_request_head_commit.side_effect = ValueError("no PR sha")
        request = _make_request(provision_type="pull_request", pr_number=42)
        result = await provisioner.provision_repository(request)
        assert result["commit_sha"] is None
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"

    # Lines 176-178: branch commit SHA retrieval fails with ValueError
    async def test_provision_branch_commit_sha_value_error(self, provisioner, github_svc):
        github_svc.get_branch_head_commit.side_effect = ValueError("no branch sha")
        request = _make_request()
        result = await provisioner.provision_repository(request)
        assert result["commit_sha"] is None
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].status == "running"

    # KUBERNETES_IN_CLUSTER must not divert provisioning: the old branch
    # imported a function that exists nowhere (ImportError). The flag is now
    # ignored and the coding-containers adapter is always used.
    async def test_provision_in_cluster_flag_uses_container_adapter(self, provisioner, docker_svc):
        request = _make_request()
        with patch.dict("os.environ", {"KUBERNETES_IN_CLUSTER": "true"}):
            result = await provisioner.provision_repository(request)
        docker_svc.create_container.assert_awaited_once()
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["commit_sha"] == "abc123"

    # Line 191: custom container_name in request
    async def test_provision_custom_container_name(self, provisioner, docker_svc):
        request = _make_request(container_name="my-custom-name")
        docker_svc.find_existing_container.return_value = None
        result = await provisioner.provision_repository(request)
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"
        assert result["commit_sha"] == "abc123"

    # Line 193: PR provision type generates pr-specific container name
    async def test_provision_pr_container_name(self, provisioner, docker_svc):
        request = _make_request(
            provision_type="pull_request",
            pr_number=99,
            container_name="",
            force_rebuild=True,
        )
        docker_svc.find_existing_container.return_value = None
        # #136: force_rebuild removes by name only a workspace that holds this repo.
        docker_svc.find_container_by_name.return_value = _make_container_info(
            container_name="codegraph-test_repo-pr-99"
        )
        await provisioner.provision_repository(request)
        # The expected name should be derived from PR naming convention
        docker_svc.remove_container_by_name.assert_awaited_once()
        call_args = docker_svc.remove_container_by_name.call_args
        assert "pr-99" in call_args[0][0]

    # Lines 222-223: force_rebuild, remove_container_by_name raises RuntimeError (container not found)
    async def test_provision_force_rebuild_container_not_found_by_name(
        self, provisioner, docker_svc
    ):
        docker_svc.find_existing_container.return_value = None
        # Found by the #136 repo check, then gone by the time of the delete.
        docker_svc.find_container_by_name.return_value = _make_container_info()
        docker_svc.remove_container_by_name.side_effect = RuntimeError("Container x not found")
        request = _make_request(force_rebuild=True)
        # Should not fail - the RuntimeError from remove_container_by_name is caught
        result = await provisioner.provision_repository(request)
        assert result["container_info"].container_name == "codegraph-test_repo"
        assert result["container_info"].api_endpoint == "http://localhost:8691"
        assert result["commit_sha"] == "abc123"

    # Lines 247-251: NotFoundException is re-raised
    async def test_provision_not_found_exception_reraised(self, provisioner, github_svc):
        from coding_client.exceptions.api_exceptions import NotFoundException

        github_svc.validate_repository_access.side_effect = NotFoundException("repo gone")
        request = _make_request()
        with pytest.raises(NotFoundException):
            await provisioner.provision_repository(request)

    # Lines 252-263: RuntimeError during provisioning
    async def test_provision_runtime_error(self, provisioner, docker_svc, github_svc):
        docker_svc.create_container.side_effect = RuntimeError("docker exploded")
        request = _make_request()
        result = await provisioner.provision_repository(request)
        assert result["error_type"] == "runtime"
        assert "docker exploded" in result["error"]

    # Lines 264-275: Unexpected Exception during provisioning
    async def test_provision_unexpected_exception(self, provisioner, docker_svc, github_svc):
        docker_svc.create_container.side_effect = OSError("disk full")
        request = _make_request()
        result = await provisioner.provision_repository(request)
        assert result["error_type"] == "system"
        assert "disk full" in result["error"]

    # Lines 281, 286-289: _extract_repo_name edge cases
    async def test_extract_repo_name_with_git_suffix(self, provisioner):
        name = provisioner._extract_repo_name("https://github.com/org/my-repo.git")
        assert name == "my_repo"

    async def test_extract_repo_name_single_segment(self, provisioner):
        name = provisioner._extract_repo_name("single")
        assert name == "unknown_repo"

    async def test_extract_repo_name_trailing_slash(self, provisioner):
        name = provisioner._extract_repo_name("https://github.com/org/cool-repo/")
        assert name == "cool_repo"

    async def test_extract_repo_name_value_error(self, provisioner):
        """Lines 287-289: ValueError during extraction returns 'unknown_repo'."""

        class BadStr(str):
            """String subclass that raises ValueError on endswith."""

            def endswith(self, suffix, *args, **kwargs):
                raise ValueError("bad value during endswith")

        name = provisioner._extract_repo_name(BadStr("https://github.com/org/repo"))
        assert name == "unknown_repo"

    # Lines 317-344: remove_container tests
    async def test_remove_container_success(self, provisioner, docker_svc):
        container = _make_container_info(container_name="ctr-to-remove")
        docker_svc.list_containers.return_value = [container]
        docker_svc.remove_container.return_value = True
        result = await provisioner.remove_container("ctr-to-remove", tenant_id="t-1")
        assert result["status"] == "success"
        assert result["container_name"] == "ctr-to-remove"

    async def test_remove_container_not_found(self, provisioner, docker_svc):
        from coding_client.exceptions.api_exceptions import NotFoundException

        docker_svc.list_containers.return_value = []
        with pytest.raises(NotFoundException):
            await provisioner.remove_container("missing-ctr")

    async def test_remove_container_failure(self, provisioner, docker_svc):
        container = _make_container_info(container_name="ctr-fail")
        docker_svc.list_containers.return_value = [container]
        docker_svc.remove_container.return_value = False
        result = await provisioner.remove_container("ctr-fail")
        assert result["status"] == "error"

    async def test_remove_container_value_error(self, provisioner, docker_svc):
        docker_svc.list_containers.side_effect = ValueError("bad tenant")
        with pytest.raises(ValueError, match="bad tenant"):
            await provisioner.remove_container("ctr", tenant_id="bad")

    async def test_remove_container_runtime_error(self, provisioner, docker_svc):
        container = _make_container_info(container_name="ctr-err")
        docker_svc.list_containers.return_value = [container]
        docker_svc.remove_container.side_effect = RuntimeError("docker down")
        with pytest.raises(RuntimeError, match="docker down"):
            await provisioner.remove_container("ctr-err")

    # Lines 360-376: _get_environment_variables
    async def test_get_env_vars_no_secret_config(self, provisioner):
        request = _make_request(secret_config_id=None)
        result = await provisioner._get_environment_variables(request)
        assert result is None

    async def test_get_env_vars_with_secret_config_id(self, provisioner):
        request = _make_request(secret_config_id="secret-123")
        result = await provisioner._get_environment_variables(request)
        # Currently returns None because SecretsAPIService is not yet available
        assert result is None

    async def test_get_env_vars_runtime_error(self, provisioner):
        request = _make_request(secret_config_id="secret-fail")
        with patch("config.settings.get_settings", side_effect=RuntimeError("no secrets")):
            result = await provisioner._get_environment_variables(request)
        assert result is None

    # Lines 381-383: get_github_provisioning_service factory
    async def test_get_github_provisioning_service_factory(self):
        mock_validation_svc = MagicMock()
        with patch(
            "services.github_validation.get_github_validation_service",
            return_value=mock_validation_svc,
        ):
            svc = get_github_provisioning_service()
        assert isinstance(svc, GitHubProvisioningService)
        assert svc.github_service is mock_validation_svc
