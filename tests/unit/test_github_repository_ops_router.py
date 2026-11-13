"""Tests for repository file-ops endpoints via real HTTP requests.

Covers: file read, tree, directory listing, search through ASGI transport.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

BASE = "/api/repositories"


# ---------------------------------------------------------------------------
# Helper: mock containers HTTP client
# ---------------------------------------------------------------------------


def _mock_client(get_return=None, post_return=None):
    """Create a MagicMock standing in for ``CodingWorkspacesClient``.

    ``get_return`` feeds the typed read methods (``list_files`` /
    ``get_workspace``); ``post_return`` feeds ``exec``. The router no longer
    builds ``/api/workspaces/...`` URLs, so there is nothing to stub on
    ``.get`` / ``.post`` any more.
    """
    client = MagicMock()
    client.get_workspace = AsyncMock(return_value=get_return)
    client.list_files = AsyncMock(return_value=get_return)
    client.exec = AsyncMock(return_value=post_return)
    return client


# ===========================================================================
# read_repository_file
# ===========================================================================


class TestReadFile:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_read_file_success(self, mock_get_client, mock_get_ctr, async_client):
        """Read file returns content with metadata."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace/README.md",
                "type": "file",
                "content": "hello\nworld",
                "entries": None,
            }
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/README.md"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "hello\nworld"
        assert data["file_path"] == "/workspace/README.md"
        assert data["size"] == 11
        assert data["lines_total"] == 2
        assert data["encoding"] == "utf-8"

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_read_file_is_directory(self, mock_get_client, mock_get_ctr, async_client):
        """Reading a directory path returns 400."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace/src",
                "type": "directory",
                "content": None,
                "entries": [],
            }
        )
        resp = await async_client.get(f"{BASE}/test-ctr/file", params={"path": "/workspace/src"})
        assert resp.status_code == 400

    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_read_file_container_missing(self, mock_get, async_client):
        """Container not found returns 404."""
        mock_get.side_effect = HTTPException(status_code=404, detail="Container 'ghost' not found")
        resp = await async_client.get(f"{BASE}/ghost/file", params={"path": "README.md"})
        assert resp.status_code == 404

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_read_file_single_line(self, mock_get_client, mock_get_ctr, async_client):
        """Single-line file has lines_total == 1."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace/VERSION",
                "type": "file",
                "content": "1.0.0",
                "entries": None,
            }
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/VERSION"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["lines_total"] == 1
        assert data["lines_returned"] == 1
        assert data["size"] == 5

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_read_file_containers_unavailable(
        self, mock_get_client, mock_get_ctr, async_client
    ):
        """Containers service unavailable returns 503."""
        mock_get_client.side_effect = HTTPException(
            status_code=503, detail="Containers service not available"
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/README.md"}
        )
        assert resp.status_code == 503


# ===========================================================================
# list_directory_contents
# ===========================================================================


class TestListDirectory:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_list_directory_success(self, mock_get_client, mock_get_ctr, async_client):
        """List directory returns items."""
        mock_get_client.return_value = _mock_client(
            get_return={
                "path": "/workspace/src",
                "type": "directory",
                "content": None,
                "entries": [
                    {
                        "permissions": "-rw-r--r--",
                        "size": "1024",
                        "modified": "Mar 10 14:30",
                        "name": "main.py",
                        "type": "file",
                    },
                    {
                        "permissions": "drwxr-xr-x",
                        "size": "4096",
                        "modified": "Mar 10 14:30",
                        "name": "utils",
                        "type": "directory",
                    },
                ],
            }
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/directory", params={"path": "/workspace/src"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "main.py"

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_list_directory_recursive(self, mock_get_client, mock_get_ctr, async_client):
        """Recursive listing uses find exec."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace/src/main.py\n/workspace/src/utils\n/workspace/src/utils/helper.py\n",
                "stderr": "",
                "duration": 0.1,
            }
        )
        resp = await async_client.get(
            f"{BASE}/test-ctr/directory",
            params={"path": "/workspace/src", "recursive": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_list_directory_container_missing(self, mock_get, async_client):
        """Container not found returns 404."""
        mock_get.side_effect = HTTPException(status_code=404, detail="Container 'ghost' not found")
        resp = await async_client.get(f"{BASE}/ghost/directory", params={"path": "/workspace/src"})
        assert resp.status_code == 404


# ===========================================================================
# get_repository_tree
# ===========================================================================


class TestGetTree:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_tree_success(self, mock_get_client, mock_get_ctr, async_client):
        """Tree returns nested structure."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace\n/workspace/src\n/workspace/src/main.py\n/workspace/README.md\n",
                "stderr": "",
                "duration": 0.1,
            }
        )
        resp = await async_client.get(f"{BASE}/test-ctr/tree")
        assert resp.status_code == 200
        data = resp.json()
        assert data["container_name"] == "test-ctr"
        assert data["tree"]["type"] == "directory"
        assert "src" in data["tree"]["children"]

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_tree_custom_depth(self, mock_get_client, mock_get_ctr, async_client):
        """Tree respects max_depth parameter."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace\n/workspace/src\n",
                "stderr": "",
                "duration": 0.05,
            }
        )
        resp = await async_client.get(f"{BASE}/test-ctr/tree", params={"max_depth": "1"})
        assert resp.status_code == 200

    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_tree_container_missing(self, mock_get, async_client):
        """Container not found returns 404."""
        mock_get.side_effect = HTTPException(status_code=404, detail="Container 'ghost' not found")
        resp = await async_client.get(f"{BASE}/ghost/tree")
        assert resp.status_code == 404


# ===========================================================================
# search_repository
# ===========================================================================


class TestSearch:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_content_search(self, mock_get_client, mock_get_ctr, async_client):
        """Content search returns grep results."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace/src/main.py:10:# TODO: fix this\n/workspace/src/utils.py:5:# TODO: refactor\n",
                "stderr": "",
                "duration": 0.2,
            }
        )
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "content", "pattern": "TODO"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 2
        assert data["search_type"] == "content"

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_file_search(self, mock_get_client, mock_get_ctr, async_client):
        """File search returns find results."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace/src/main.py\n/workspace/tests/test_main.py\n",
                "stderr": "",
                "duration": 0.1,
            }
        )
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "files", "pattern": "main.py"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 2
        assert data["search_type"] == "files"

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_search_no_matches(self, mock_get_client, mock_get_ctr, async_client):
        """grep no matches (exit code 1) returns empty results."""
        mock_get_client.return_value = _mock_client(
            post_return={"exit_code": 1, "stdout": "", "stderr": "", "duration": 0.1}
        )
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={"search_type": "content", "pattern": "nonexistent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 0

    @patch("routers.github_repository_ops._get_container_by_name", new_callable=AsyncMock)
    async def test_search_container_missing(self, mock_get, async_client):
        """Container not found returns 404."""
        mock_get.side_effect = HTTPException(status_code=404, detail="Container 'ghost' not found")
        resp = await async_client.post(
            f"{BASE}/ghost/search",
            json={"search_type": "content", "pattern": "hello"},
        )
        assert resp.status_code == 404

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    @patch("routers.github_repository_ops._get_client")
    async def test_content_search_with_file_pattern(
        self, mock_get_client, mock_get_ctr, async_client
    ):
        """Content search passes file_pattern to grep --include."""
        mock_get_client.return_value = _mock_client(
            post_return={
                "exit_code": 0,
                "stdout": "/workspace/src/main.py:1:import os\n",
                "stderr": "",
                "duration": 0.1,
            }
        )
        resp = await async_client.post(
            f"{BASE}/test-ctr/search",
            json={
                "search_type": "content",
                "pattern": "import os",
                "file_pattern": "*.py",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 1
        assert data["pattern"] == "import os"


# ===========================================================================
# Container runtime errors
# ===========================================================================


class TestContainerRuntimeError:
    async def test_runtime_error_returns_503_or_404(self, async_client):
        """RuntimeError from uninitialised client returns error status."""
        with patch("main.get_workspaces_client", side_effect=RuntimeError("not init")):
            resp = await async_client.get(f"{BASE}/test-ctr/file", params={"path": "README.md"})
        # RuntimeError in _get_container_by_name → 404 (container not found)
        # RuntimeError in _get_client → 503 (service unavailable)
        assert resp.status_code in (404, 503)


# ===========================================================================
# Path validation
# ===========================================================================


class TestPathValidation:
    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    async def test_path_traversal_rejected(self, mock_get_ctr, async_client):
        """Paths with .. are rejected with 400."""
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/../etc/passwd"}
        )
        assert resp.status_code == 400

    @patch(
        "routers.github_repository_ops._get_container_by_name",
        new_callable=AsyncMock,
        return_value={"id": "c1"},
    )
    async def test_dangerous_chars_rejected(self, mock_get_ctr, async_client):
        """Paths with shell metacharacters are rejected with 400."""
        resp = await async_client.get(
            f"{BASE}/test-ctr/file", params={"path": "/workspace/file;rm -rf /"}
        )
        assert resp.status_code == 400
