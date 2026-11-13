"""Optional ``toolchain`` on POST /api/repositories/import (and hidden /provision).

coding-containers#184 added ``toolchain: "python" | "python+rust"`` (default
"python") to its workspace create schema. The import request carries the same
optional field and forwards it. R3 (strictly additive): an omitted toolchain must
leave the body sent to coding-containers byte-identical to the pre-toolchain one,
so the key is not sent at all.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from coding_client.models.container import ContainerStatus, GitHubRepoRequest

from schemas.github_api_models import ContainerInfo, ProvisionRepositoryRequest
from services.github_provisioning import (
    _ContainersHTTPAdapter,
    get_github_provisioning_service,
)

BASE = "/api/repositories"
REQUEST = {
    "repo_url": "https://github.com/o/r",
    "branch": "main",
    "container_name": "q-0123456789ab",
    "access_token": "tok-123",
}
# What coding-containers received before this change for REQUEST: the same
# exclude_none dump, minus the stripped exact_name. Pinned as a literal.
SNAPSHOT = {
    "repo_url": "https://github.com/o/r",
    "branch": "main",
    "container_name": "q-0123456789ab",
    "access_token": "tok-123",
    "force_rebuild": False,
    "provision_type": "branch",
}


async def _forwarded_payload(request) -> dict:
    client = AsyncMock()
    client.create_workspace = AsyncMock(
        return_value={
            "container_name": "q-0123456789ab",
            "api_endpoint": "",
            "codebase_path": "/app/code",
            "repo_url": "https://github.com/o/r",
            "branch": "main",
            "status": "running",
        }
    )
    with patch("main.get_workspaces_client", return_value=client):
        await _ContainersHTTPAdapter().create_container(request)
    return {k: v for k, v in client.create_workspace.call_args.kwargs.items() if k != "headers"}


class TestForwardedPayload:
    async def test_omitted_toolchain_payload_is_unchanged(self):
        payload = await _forwarded_payload(ProvisionRepositoryRequest(**REQUEST))
        assert "toolchain" not in payload
        assert payload == SNAPSHOT
        # Byte-identical to the pre-toolchain shared contract's own dump.
        legacy = GitHubRepoRequest(**REQUEST).model_dump(exclude_none=True)
        assert json.dumps(payload) == json.dumps(legacy)

    @pytest.mark.parametrize("toolchain", ["python+rust", "python"])
    async def test_explicit_toolchain_is_passed_through(self, toolchain):
        payload = await _forwarded_payload(
            ProvisionRepositoryRequest(**REQUEST, toolchain=toolchain)
        )
        assert payload == {**SNAPSHOT, "toolchain": toolchain}

    def test_unknown_toolchain_is_rejected_by_the_model(self):
        with pytest.raises(ValueError):
            ProvisionRepositoryRequest(**REQUEST, toolchain="rust")


@pytest.fixture
def provision_service(async_client):
    container = ContainerInfo(
        container_name="q-0123456789ab",
        api_endpoint="http://localhost:8700",
        status=ContainerStatus("running"),
        repo_url="https://github.com/o/r",
        codebase_path="/workspace",
        branch="main",
    )
    svc = AsyncMock()
    svc.provision_repository = AsyncMock(
        return_value={"container_info": container, "commit_sha": "abc123"}
    )
    app = async_client._transport.app
    app.dependency_overrides[get_github_provisioning_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_github_provisioning_service, None)


class TestImportRoute:
    @pytest.mark.parametrize("path", ["/import", "/provision"])
    async def test_python_rust_reaches_the_service(self, async_client, provision_service, path):
        body = {"repo_url": "https://github.com/o/r", "branch": "main", "toolchain": "python+rust"}
        resp = await async_client.post(f"{BASE}{path}", json=body)
        assert resp.status_code == 201, resp.text
        request = provision_service.provision_repository.call_args.args[0]
        assert request.toolchain == "python+rust"

    async def test_omitted_toolchain_reaches_the_service_as_none(
        self, async_client, provision_service
    ):
        resp = await async_client.post(
            f"{BASE}/import", json={"repo_url": "https://github.com/o/r", "branch": "main"}
        )
        assert resp.status_code == 201, resp.text
        request = provision_service.provision_repository.call_args.args[0]
        assert request.toolchain is None

    @pytest.mark.parametrize("bad", ["rust", "PYTHON", "", 3])
    @pytest.mark.parametrize("path", ["/import", "/provision"])
    async def test_unknown_toolchain_is_422(self, async_client, provision_service, path, bad):
        body = {"repo_url": "https://github.com/o/r", "branch": "main", "toolchain": bad}
        resp = await async_client.post(f"{BASE}{path}", json=body)
        assert resp.status_code == 422, resp.text
        provision_service.provision_repository.assert_not_awaited()


def test_openapi_adds_only_the_optional_toolchain_property():
    from main import app

    app.openapi_schema = None
    spec = app.openapi()
    ref = spec["paths"]["/api/repositories/import"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    schema = spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    expected = set(GitHubRepoRequest.model_json_schema()["properties"]) | {
        "exact_name",
        "toolchain",
    }
    assert set(schema["properties"]) == expected
    assert "toolchain" not in schema.get("required", [])
    enums = [opt.get("enum") for opt in schema["properties"]["toolchain"]["anyOf"]]
    assert ["python", "python+rust"] in enums


# ---------------------------------------------------------------------------
# Reuse with a different toolchain: 409, never a silent reuse or rebuild.
# ---------------------------------------------------------------------------


def _workspace(toolchain=None, name="codegraph-r"):
    from services.github_provisioning import _WorkspaceInfo

    raw = {
        "container_name": name,
        "api_endpoint": "",
        "codebase_path": "/workspace",
        "repo_url": "https://github.com/o/r",
        "branch": "main",
        "status": "running",
    }
    if toolchain is not None:
        raw["toolchain"] = toolchain
    return _WorkspaceInfo(**raw)


@pytest.fixture
def provisioner():
    from services.github_provisioning import GitHubProvisioningService

    docker = AsyncMock()
    docker.find_existing_container = AsyncMock(return_value=None)
    docker.find_container_by_name = AsyncMock(return_value=None)
    docker.remove_container = AsyncMock(return_value=True)
    docker.remove_container_by_name = AsyncMock(return_value=True)
    docker.create_container = AsyncMock(return_value=_workspace("python+rust"))
    github = AsyncMock()
    github.get_branch_head_commit = AsyncMock(return_value="abc123")
    return GitHubProvisioningService(docker_service=docker, github_service=github)


def _req(**kw):
    return ProvisionRepositoryRequest(repo_url="https://github.com/o/r", branch="main", **kw)


class TestReuseToolchainMismatch:
    @pytest.mark.parametrize(
        ("have", "want"),
        [("python", "python+rust"), (None, "python+rust"), ("python+rust", "python")],
    )
    async def test_mismatch_is_409_and_touches_nothing(self, provisioner, have, want):
        from services.github_provisioning import WorkspaceToolchainMismatchError

        provisioner.docker_service.find_existing_container.return_value = _workspace(have)
        with pytest.raises(WorkspaceToolchainMismatchError) as exc:
            await provisioner.provision_repository(_req(toolchain=want))
        assert exc.value.http_status == 409
        assert "force_rebuild" in str(exc.value)
        provisioner.docker_service.remove_container.assert_not_awaited()
        provisioner.docker_service.remove_container_by_name.assert_not_awaited()
        provisioner.docker_service.create_container.assert_not_awaited()

    async def test_exact_name_reuse_mismatch_is_409(self, provisioner):
        from services.github_provisioning import WorkspaceToolchainMismatchError

        provisioner.docker_service.find_container_by_name.return_value = _workspace(
            "python", name="q-0123456789ab"
        )
        with pytest.raises(WorkspaceToolchainMismatchError):
            await provisioner.provision_repository(
                _req(container_name="q-0123456789ab", exact_name=True, toolchain="python+rust")
            )
        provisioner.docker_service.create_container.assert_not_awaited()

    @pytest.mark.parametrize(("have", "want"), [("python+rust", "python+rust"), (None, "python")])
    async def test_matching_toolchain_reuses(self, provisioner, have, want):
        existing = _workspace(have)
        provisioner.docker_service.find_existing_container.return_value = existing
        result = await provisioner.provision_repository(_req(toolchain=want))
        assert result["container_info"] is existing
        provisioner.docker_service.create_container.assert_not_awaited()

    @pytest.mark.parametrize("have", ["python", "python+rust", None])
    async def test_omitted_toolchain_reuses_unchanged(self, provisioner, have):
        existing = _workspace(have)
        provisioner.docker_service.find_existing_container.return_value = existing
        result = await provisioner.provision_repository(_req())
        assert result["container_info"] is existing
        provisioner.docker_service.create_container.assert_not_awaited()

    async def test_force_rebuild_with_mismatch_rebuilds(self, provisioner):
        existing = _workspace("python")
        provisioner.docker_service.find_existing_container.return_value = existing
        await provisioner.provision_repository(_req(toolchain="python+rust", force_rebuild=True))
        provisioner.docker_service.remove_container.assert_awaited_once()
        provisioner.docker_service.create_container.assert_awaited_once()


class TestAdapterKeepsToolchain:
    """coding-client's ContainerInfo drops unknown fields; the finders must not."""

    async def test_find_existing_container_keeps_toolchain(self):
        client = AsyncMock()
        client.list_workspaces = AsyncMock(
            return_value={"containers": [_workspace("python+rust").model_dump(mode="json")]}
        )
        with patch("main.get_workspaces_client", return_value=client):
            found = await _ContainersHTTPAdapter().find_existing_container("https://github.com/o/r")
        assert found.toolchain == "python+rust"

    async def test_find_container_by_name_keeps_toolchain(self):
        client = AsyncMock()
        client.get_workspace = AsyncMock(
            return_value={"container": _workspace("python+rust").model_dump(mode="json")}
        )
        with patch("main.get_workspaces_client", return_value=client):
            found = await _ContainersHTTPAdapter().find_container_by_name("codegraph-r")
        assert found.toolchain == "python+rust"


class TestMismatchRoute:
    async def test_route_maps_mismatch_to_409_envelope(self, async_client):
        from services.github_provisioning import WorkspaceToolchainMismatchError

        svc = AsyncMock()
        svc.provision_repository = AsyncMock(
            side_effect=WorkspaceToolchainMismatchError(
                "toolchain mismatch; pass force_rebuild=true", owner="o", repo="r"
            )
        )
        app = async_client._transport.app
        app.dependency_overrides[get_github_provisioning_service] = lambda: svc
        try:
            resp = await async_client.post(
                f"{BASE}/import",
                json={
                    "repo_url": "https://github.com/o/r",
                    "branch": "main",
                    "toolchain": "python+rust",
                },
            )
        finally:
            app.dependency_overrides.pop(get_github_provisioning_service, None)
        assert resp.status_code == 409, resp.text
        assert "workspace_toolchain_mismatch" in resp.text
        assert "force_rebuild" in resp.text

    async def test_reused_workspace_response_does_not_leak_toolchain(
        self, async_client, provision_service
    ):
        provision_service.provision_repository.return_value = {
            "container_info": _workspace("python+rust"),
            "commit_sha": "abc123",
        }
        resp = await async_client.post(
            f"{BASE}/import", json={"repo_url": "https://github.com/o/r", "branch": "main"}
        )
        assert resp.status_code == 201, resp.text
        assert "toolchain" not in resp.json()["container"]
