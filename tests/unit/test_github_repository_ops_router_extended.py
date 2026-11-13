"""Extended tests for repository file-ops endpoints via real HTTP requests.

Tests /{container_name}/file, /tree, /directory, /search through ASGI transport.
All endpoints are under /api/repositories/ prefix (mounted via github.py aggregator).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from coding_client.clients import CodingWorkspacesClient

BASE = "/api/repositories"


def _mock_client(get_return=None, post_return=None):
    """Create a mock ``CodingWorkspacesClient`` with the typed methods stubbed.

    ``get_return`` feeds ``list_files`` / ``get_workspace``; ``post_return``
    feeds ``exec``.
    """
    client = MagicMock()
    client.get_workspace = AsyncMock(return_value=get_return)
    client.list_files = AsyncMock(return_value=get_return)
    client.exec = AsyncMock(return_value=post_return)
    return client


# ===========================================================================
# _get_container_by_name — ServiceError paths
# ===========================================================================


class TestGetContainerServiceError:
    async def test_service_error_404_returns_404(self, async_client):
        """ServiceError with status_code=404 propagates as 404."""
        from coding_client.clients import ServiceError

        err = ServiceError(404, "not found")

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=err)

        with patch("main.get_workspaces_client", return_value=mock_client):
            resp = await async_client.get(f"{BASE}/ghost-ctr/file", params={"path": "README.md"})
        assert resp.status_code == 404

    async def test_service_error_500_returns_500(self, async_client):
        """ServiceError with status_code=500 propagates as 500."""
        from coding_client.clients import ServiceError

        err = ServiceError(500, "server error")

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=err)

        with patch("main.get_workspaces_client", return_value=mock_client):
            resp = await async_client.get(f"{BASE}/bad-ctr/file", params={"path": "README.md"})
        assert resp.status_code == 500


# ===========================================================================
# _get_container_by_name — RuntimeError (uninitialised client)
# ===========================================================================


class TestGetContainerRuntimeError:
    async def test_runtime_error_returns_404(self, async_client):
        """RuntimeError from uninitialised client returns 404."""
        with patch("main.get_workspaces_client", side_effect=RuntimeError("not init")):
            resp = await async_client.get(f"{BASE}/test-ctr/file", params={"path": "README.md"})
        assert resp.status_code == 404


# ===========================================================================
# _get_container_by_name — generic Exception
# ===========================================================================


class TestGetContainerGenericException:
    async def test_generic_exception_returns_500(self, async_client):
        """Generic exception returns 500."""
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=TypeError("unexpected"))

        with patch("main.get_workspaces_client", return_value=mock_client):
            resp = await async_client.get(f"{BASE}/ctr/file", params={"path": "README.md"})
        assert resp.status_code == 500


# ===========================================================================
# read_repository_file — wired to coding-containers
# ===========================================================================


class TestReadFileWired:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_read_file_empty_content(self, mock_get_client, mock_get, async_client):
        """Read empty file returns 0 lines."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace/empty.txt",
                "type": "file",
                "content": "",
                "entries": None,
            }
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/empty.txt"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == 0
        assert data["lines_total"] == 0


# ===========================================================================
# get_repository_tree — wired to exec
# ===========================================================================


class TestGetTreeWired:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_tree_empty_workspace(self, mock_get_client, mock_get, async_client):
        """Tree with only workspace root returns empty children."""
        mock_get_client.return_value = _mock_client(
            post_return={"exit_code": 0, "stdout": "/workspace\n", "stderr": "", "duration": 0.1}
        )
        resp = await async_client.get(f"{BASE}/test-ctr/tree")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tree"]["type"] == "directory"


# ===========================================================================
# get_repository_tree — generic Exception path
# ===========================================================================


class TestGetTreeGenericException:
    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_tree_generic_exception_returns_500(self, mock_get, async_client):
        """Non-HTTPException in get_repository_tree returns 500."""
        mock_get.side_effect = TypeError("tree exploded")
        resp = await async_client.get(f"{BASE}/test-ctr/tree")
        assert resp.status_code == 500
        assert (
            "unexpected error"
            in resp.json().get("error", {}).get("message", resp.json().get("message", "")).lower()
        )


# ===========================================================================
# list_directory_contents — wired to coding-containers
# ===========================================================================


class TestListDirectoryWired:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_directory_filters_dot_entries(self, mock_get_client, mock_get, async_client):
        """Dot entries (. and ..) are filtered out."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace",
                "type": "directory",
                "content": None,
                "entries": [
                    {
                        "permissions": "drwxr-xr-x",
                        "size": "4096",
                        "modified": "Mar 10",
                        "name": ".",
                        "type": "directory",
                    },
                    {
                        "permissions": "drwxr-xr-x",
                        "size": "4096",
                        "modified": "Mar 10",
                        "name": "..",
                        "type": "directory",
                    },
                    {
                        "permissions": "-rw-r--r--",
                        "size": "100",
                        "modified": "Mar 10",
                        "name": "main.py",
                        "type": "file",
                    },
                ],
            }
        )
        resp = await async_client.get(f"{BASE}/test-ctr/directory", params={"path": "/workspace"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "main.py"


# ===========================================================================
# list_directory_contents — generic Exception path
# ===========================================================================


class TestListDirectoryGenericException:
    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_directory_generic_exception_returns_500(self, mock_get, async_client):
        mock_get.side_effect = TypeError("dir broken")
        resp = await async_client.get(f"{BASE}/test-ctr/directory", params={"path": "src"})
        assert resp.status_code == 500
        assert (
            "unexpected error"
            in resp.json().get("error", {}).get("message", resp.json().get("message", "")).lower()
        )


# ===========================================================================
# search_repository — inner Exception from _perform_search
# ===========================================================================


class TestSearchInnerException:
    @patch("routers.github_repository_ops._perform_search", new_callable=AsyncMock)
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    async def test_search_inner_exception_returns_500(self, mock_get, mock_search, async_client):
        """Non-HTTPException from _perform_search returns 500."""
        mock_search.side_effect = RuntimeError("search engine crashed")
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "content", "pattern": "TODO"},
        )
        assert resp.status_code == 500
        assert (
            "search operation failed"
            in resp.json().get("error", {}).get("message", resp.json().get("message", "")).lower()
        )


# ===========================================================================
# search_repository — outer generic Exception
# ===========================================================================


class TestSearchOuterException:
    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_search_outer_exception_returns_500(self, mock_get, async_client):
        """Non-HTTPException from _get_container_by_name returns 500."""
        mock_get.side_effect = TypeError("container lookup failed")
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "files", "pattern": "*.py"},
        )
        assert resp.status_code == 500
        assert (
            "unexpected error"
            in resp.json().get("error", {}).get("message", resp.json().get("message", "")).lower()
        )


# ===========================================================================
# search_repository — successful return via mock of _perform_search
# ===========================================================================


class TestSearchSuccess:
    @patch("routers.github_repository_ops._perform_search", new_callable=AsyncMock)
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    async def test_search_returns_results(self, mock_get, mock_search, async_client):
        """Successful search returns results in response body."""
        mock_search.return_value = ([{"file": "main.py", "line": 10}], False)
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "content", "pattern": "TODO"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["total_found"] == 1
        assert body["truncated"] is False
        assert body["search_type"] == "content"
        assert body["pattern"] == "TODO"

    @patch("routers.github_repository_ops._perform_search", new_callable=AsyncMock)
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    async def test_search_reports_truncation(self, mock_get, mock_search, async_client):
        """A capped search reports truncated=True instead of silently dropping matches."""
        mock_search.return_value = ([{"file": "main.py", "line": 10}], True)
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "content", "pattern": "TODO", "max_results": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["truncated"] is True
        assert "truncated" in body["message"].lower()


# ===========================================================================
# Request shape on the wire — the query key MUST be ``path``, never ``file_path``
# ===========================================================================


class TestWorkspacesRequestShape:
    """Guards the exact defect class this migration exists to remove.

    coding-realtime sent ``file_path=`` to a route whose query parameter is
    ``path``. FastAPI drops unknown query keys silently, so the route fell back
    to listing ``/workspace`` and every file read returned 200 with
    ``content: None`` — a bug no mock-level assertion could catch, because the
    mock happily accepts any kwarg name.

    These tests therefore assert on what reaches the WIRE: a real
    ``CodingWorkspacesClient`` over ``httpx.MockTransport``. The responder
    deliberately answers a request that does NOT carry ``path`` with the
    directory-listing payload, i.e. the exact silent-fallback shape, so a wrong
    key name fails here instead of in production.
    """

    @staticmethod
    def _client_on_mock_transport(monkeypatch):
        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            path = request.url.path
            if path.endswith("/files"):
                if request.url.params.get("path") == "README.md":
                    return httpx.Response(
                        200,
                        json={
                            "path": "README.md",
                            "type": "file",
                            "content": "hi",
                            "entries": None,
                        },
                    )
                # Unknown/missing query key -> the silent 200-with-no-content
                # fallback that hid the coding-realtime defect.
                return httpx.Response(
                    200,
                    json={
                        "path": "/workspace",
                        "type": "directory",
                        "content": None,
                        "entries": [],
                    },
                )
            if path.endswith("/exec"):
                return httpx.Response(
                    200, json={"exit_code": 0, "stdout": "/workspace\n", "stderr": ""}
                )
            return httpx.Response(200, json={"container_name": "test-ctr"})

        real_cls = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda *a, **k: real_cls(*a, **{**k, "transport": httpx.MockTransport(_handler)}),
        )
        return CodingWorkspacesClient(base_url="http://containers:8691"), seen

    @staticmethod
    def _only(seen, suffix):
        matches = [r for r in seen if r.url.path.endswith(suffix)]
        assert matches, f"no request to *{suffix}; saw {[r.url.path for r in seen]}"
        return matches[0]

    async def test_read_file_sends_path_query_key(self, monkeypatch, async_client):
        """read_repository_file -> GET /api/workspaces/{n}/files?path=..."""
        client, seen = self._client_on_mock_transport(monkeypatch)
        with patch("main.get_workspaces_client", return_value=client):
            resp = await async_client.get(f"{BASE}/test-ctr/file", params={"path": "README.md"})

        assert resp.status_code == 200
        # Real content came back -> the callee matched the query key.
        assert resp.json()["content"] == "hi"

        request = self._only(seen, "/api/workspaces/test-ctr/files")
        assert request.url.params.get("path") == "README.md"
        assert "file_path" not in request.url.params

    async def test_list_directory_sends_path_query_key(self, monkeypatch, async_client):
        """list_directory_contents -> GET /api/workspaces/{n}/files?path=..."""
        client, seen = self._client_on_mock_transport(monkeypatch)
        with patch("main.get_workspaces_client", return_value=client):
            resp = await async_client.get(
                f"{BASE}/test-ctr/directory", params={"path": "/workspace"}
            )

        assert resp.status_code == 200
        request = self._only(seen, "/api/workspaces/test-ctr/files")
        assert request.url.params.get("path") == "/workspace"
        assert "file_path" not in request.url.params

    async def test_container_lookup_hits_the_bare_workspace_route(self, monkeypatch, async_client):
        """_get_container_by_name -> GET /api/workspaces/{n} (no extra segment)."""
        client, seen = self._client_on_mock_transport(monkeypatch)
        with patch("main.get_workspaces_client", return_value=client):
            resp = await async_client.get(f"{BASE}/test-ctr/file", params={"path": "README.md"})

        assert resp.status_code == 200
        assert any(r.url.path == "/api/workspaces/test-ctr" for r in seen), [
            r.url.path for r in seen
        ]

    async def test_tree_exec_body_shape(self, monkeypatch, async_client):
        """get_repository_tree -> POST /api/workspaces/{n}/exec with the exec body."""
        client, seen = self._client_on_mock_transport(monkeypatch)
        with patch("main.get_workspaces_client", return_value=client):
            resp = await async_client.get(f"{BASE}/test-ctr/tree")

        assert resp.status_code == 200
        request = self._only(seen, "/api/workspaces/test-ctr/exec")
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["command"][0] == "find"
        assert body["workdir"] == "/workspace"
        assert body["timeout"] == 30
