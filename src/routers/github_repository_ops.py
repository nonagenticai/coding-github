"""
GitHub Repository File Operations Endpoints

Split from main github.py to reduce file size.
Handles file reading, tree operations, directory listing, and search.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (  # TODO: verify coding-client import
    internal_error,
    not_found,
)
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from schemas.github_api_models import (
    DirectoryItem,
    DirectoryListResponse,
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error422Response,
    Error500Response,
    FileContentResponse,
    RepositoryTreeResponse,
    ResponseStatus,
    SearchRequest,
    SearchResponse,
)
from services.github_provisioning import (
    GitHubProvisioningService,
    get_github_provisioning_service,
)
from services.outbound import optional_sibling_headers

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_path(path: str) -> str:
    """Validate and sanitize a file/directory path for container operations."""
    if not path:
        raise HTTPException(status_code=400, detail="Path must not be empty")
    dangerous = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\n", "\r", "\x00"]
    if any(c in path for c in dangerous):
        raise HTTPException(status_code=400, detail="Path contains forbidden characters")
    if ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return path


def _get_client() -> Any:
    """Get the workspaces HTTP client, raising 503 if unavailable."""
    from main import get_workspaces_client

    try:
        return get_workspaces_client()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail="Containers service not available",
        ) from e


def _build_tree(paths: list[str], base_path: str, dir_paths: set[str] | None = None) -> Any:
    """Build a TreeNode tree from a list of file paths.

    Args:
        paths: All paths from find output.
        base_path: Base path to strip (e.g. "/workspace").
        dir_paths: Explicit set of directory paths. If None, infers from
            parent relationships (may misclassify leaf dirs at max depth).
    """
    from schemas.github_api_models import TreeNode

    if dir_paths is None:
        # Fallback: infer directories from parent relationships
        path_set = set(paths)
        dir_paths = set()
        for p in paths:
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            if parent and parent in path_set:
                dir_paths.add(parent)
    dir_paths.add(base_path.rstrip("/"))

    root = TreeNode(type="directory", name=".", children={})

    for p in sorted(paths):
        rel = p.removeprefix(base_path).lstrip("/")
        if not rel:
            continue
        parts = rel.split("/")
        node = root
        for i, part in enumerate(parts):
            if node.children is None:
                node.children = {}
            if i == len(parts) - 1:
                is_dir = p in dir_paths
                node.children[part] = TreeNode(
                    type="directory" if is_dir else "file",
                    name=part,
                    children={} if is_dir else None,
                )
            else:
                if part not in node.children:
                    node.children[part] = TreeNode(type="directory", name=part, children={})
                node = node.children[part]
    return root


async def _get_container_by_name(
    container_name: str,
    service: GitHubProvisioningService,
    tenant_id: str | None = None,
) -> Any:
    """
    Get container by name or ID via coding-containers HTTP API.

    Args:
        container_name: Container name or ID to search for
        service: GitHub provisioning service instance (unused, kept for API compat)
        tenant_id: Tenant identity to forward to the workspace owner

    Returns:
        Container dict if found

    Raises:
        HTTPException: If container not found or service error
    """
    try:
        from coding_client.clients import ServiceError

        from main import get_workspaces_client

        client = get_workspaces_client()
        return await client.get_workspace(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
        )
    except ServiceError as e:
        if e.status_code == 404:
            raise not_found("Container", container_name) from e
        raise internal_error(f"Containers service error: {e}") from e
    except RuntimeError as e:
        # Containers client not initialised → treat as container not found
        raise not_found("Container", container_name) from e
    except Exception as e:
        raise internal_error(f"Failed to retrieve container: {str(e)}") from e


async def _perform_search(
    container: Any,
    request: SearchRequest,
    container_name: str,
    tenant_id: str | None = None,
) -> tuple[list[Any], bool]:
    """
    Perform the actual search operation based on search type.

    Args:
        container: Container object
        request: Search request parameters
        container_name: Name of the container to search in

    Returns:
        Tuple of (search results, truncated) where ``truncated`` is True when
        more matches existed than ``max_results`` and were dropped.

    Raises:
        HTTPException: If search fails or invalid search type
    """
    client = _get_client()

    # Validate pattern
    if "\x00" in request.pattern or "\n" in request.pattern:
        raise HTTPException(status_code=400, detail="Search pattern contains forbidden characters")
    if request.file_pattern and ("\x00" in request.file_pattern or "\n" in request.file_pattern):
        raise HTTPException(status_code=400, detail="File pattern contains forbidden characters")

    if request.search_type == "files":
        command = ["find", "/workspace", "-name", request.pattern, "-type", "f"]
        data = await client.exec(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
            command=command,
            workdir="/workspace",
            timeout=30,
        )
        exit_code = data.get("exit_code", -1)
        if exit_code != 0:
            raise internal_error(f"Search failed: {data.get('stderr', '')}")

        results: list[dict[str, Any]] = []
        truncated = False
        stdout = data.get("stdout", "")
        for line in stdout.strip().splitlines():
            if not line:
                continue
            if len(results) >= request.max_results:
                # More matches exist than max_results — report truncation
                # instead of silently dropping the remainder.
                truncated = True
                break
            results.append(
                {
                    "file": line.removeprefix("/workspace/"),
                    "line": 0,
                    "content": line.split("/")[-1],
                    "match_pattern": request.pattern,
                }
            )
        return results, truncated

    elif request.search_type == "content":
        command: list[str] = ["grep", "-rn"]
        if not request.case_sensitive:
            command.append("-i")
        if request.file_pattern:
            command.extend(["--include", request.file_pattern])
        command.extend(["--", request.pattern, "/workspace"])

        data = await client.exec(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
            command=command,
            workdir="/workspace",
            timeout=30,
        )
        exit_code = data.get("exit_code", -1)
        # grep exit_code=1 means no matches — not an error
        if exit_code == 1:
            return [], False
        if exit_code > 1:
            raise internal_error(f"Search failed: {data.get('stderr', '')}")

        results = []
        truncated = False
        stdout = data.get("stdout", "")
        for line in stdout.strip().splitlines():
            if not line:
                continue
            if len(results) >= request.max_results:
                # More matches exist than max_results — report truncation
                # instead of silently dropping the remainder.
                truncated = True
                break
            parts = line.split(":", 2)
            if len(parts) >= 3:
                results.append(
                    {
                        "file": parts[0].removeprefix("/workspace/"),
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "content": parts[2],
                        "match_pattern": request.pattern,
                    }
                )
        return results, truncated

    else:
        raise HTTPException(status_code=400, detail=f"Invalid search type: {request.search_type}")


@router.get(
    "/{container_name}/file",
    operation_id="read_repository_file",
    response_model=FileContentResponse,
    summary="Read Repository File",
    include_in_schema=False,
    description="""
    Read a specific file from the containerized repository.

    **Usage:**
    ```
    GET /api/repositories/{container_name}/file?path=src/main.py
    ```

    **Parameters:**
    - `container_name` - Container ID or name from provisioning
    - `path` - File path within the repository (e.g., "README.md", "src/app.py")

    **Response includes:**
    - `content` - UTF-8 file content
    - `encoding` - Content encoding ("utf-8")
    - `size` - File size in bytes
    - `path` - Requested file path

    **Use for:**
    - Reading configuration files
    - Viewing source code
    - Extracting documentation
    - Analyzing file contents
    """,
    responses={
        200: {"description": "File content retrieved successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        404: {
            "model": Error404Response,
            "description": "Container or file not found",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def read_repository_file(
    container_name: str = Path(..., description="Container name or ID"),
    path: str = Query(..., description="File path to read"),
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> FileContentResponse:
    """Read file content from containerized repository."""
    try:
        # Get container by ID or name
        await _get_container_by_name(container_name, service, tenant_id)

        safe_path = _validate_path(path)
        client = _get_client()
        data = await client.list_files(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
            path=safe_path,
        )
        if data is None or "type" not in data:
            raise not_found("File", safe_path)
        if data.get("type") != "file":
            raise HTTPException(
                status_code=400, detail="Path is a directory, use the directory endpoint"
            )
        content = data.get("content", "")
        size = len(content.encode("utf-8"))
        lines = content.splitlines()
        return FileContentResponse(
            status=ResponseStatus.SUCCESS,
            message="File content retrieved successfully",
            content=content,
            file_path=safe_path,
            size=size,
            modified_time=0,
            lines_total=len(lines),
            lines_returned=len(lines),
            encoding="utf-8",
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        from coding_client.clients import ServiceError

        if isinstance(e, ServiceError):
            if getattr(e, "status_code", 500) == 404:
                raise not_found("File", path) from e
            raise internal_error(f"Containers service error: {e}") from e
        raise internal_error(f"Unexpected error reading file: {str(e)}") from e


@router.get(
    "/{container_name}/tree",
    operation_id="get_repository_tree",
    response_model=RepositoryTreeResponse,
    summary="Get Repository Tree",
    include_in_schema=False,
    description="""
    Get the complete directory tree structure of the repository.

    **Usage:**
    ```
    GET /api/repositories/{container_name}/tree?max_depth=3
    ```

    **Parameters:**
    - `container_name` - Container ID or name
    - `max_depth` - Maximum tree depth (default: 3, max: 10)

    **Response includes:**
    - Hierarchical tree structure
    - File counts by type
    - Directory organization
    - Total file statistics

    **Perfect for:**
    - Understanding project structure
    - Navigation interfaces
    - File discovery
    - Project analysis
    """,
    responses={
        200: {"description": "Repository tree retrieved successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {"model": Error404Response, "description": "Container not found"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def get_repository_tree(
    container_name: str,
    max_depth: int = Query(3, ge=1, le=10, description="Maximum tree depth"),
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> RepositoryTreeResponse:
    """Get repository directory tree structure."""
    try:
        # Get container by ID or name with proper error handling
        await _get_container_by_name(container_name, service, tenant_id)

        client = _get_client()
        # Get all paths
        data = await client.exec(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
            command=["find", "/workspace", "-maxdepth", str(max_depth)],
            workdir="/workspace",
            timeout=30,
        )
        if data.get("exit_code", -1) != 0:
            raise internal_error(f"Tree retrieval failed: {data.get('stderr', '')}")
        stdout = data.get("stdout", "")
        paths = [line for line in stdout.strip().splitlines() if line]

        # Get explicit directory list for accurate type classification
        dir_data = await client.exec(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
            command=["find", "/workspace", "-maxdepth", str(max_depth), "-type", "d"],
            workdir="/workspace",
            timeout=30,
        )
        explicit_dirs: set[str] = set()
        if dir_data.get("exit_code", 0) == 0:
            explicit_dirs = {
                line for line in dir_data.get("stdout", "").strip().splitlines() if line
            }
        tree = _build_tree(paths, "/workspace", dir_paths=explicit_dirs)
        return RepositoryTreeResponse(
            status=ResponseStatus.SUCCESS,
            message="Repository tree retrieved successfully",
            tree=tree,
            container_name=container_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        from coding_client.clients import ServiceError

        if isinstance(e, ServiceError):
            if getattr(e, "status_code", 500) == 404:
                raise not_found("Container", container_name) from e
            raise internal_error(f"Containers service error: {e}") from e
        raise internal_error(f"Unexpected error getting repository tree: {str(e)}") from e


@router.get(
    "/{container_name}/directory",
    operation_id="list_directory_contents",
    response_model=DirectoryListResponse,
    summary="List Directory Contents",
    include_in_schema=False,
    description="""
    List files and subdirectories in a specific path.

    **Usage:**
    ```
    GET /api/repositories/{container_name}/directory?path=src&recursive=false
    ```

    **Parameters:**
    - `container_name` - Container ID or name
    - `path` - Directory path (empty for root)
    - `recursive` - List all subdirectories (default: false)

    **Response includes:**
    - Array of files and directories
    - File types, sizes, permissions
    - Path information for navigation

    **Use for:**
    - Directory browsing
    - File discovery
    - Building navigation interfaces
    - Finding specific file types
    """,
    responses={
        200: {"description": "Directory contents listed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid request parameters",
        },
        404: {
            "model": Error404Response,
            "description": "Container or directory not found",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def list_directory_contents(
    container_name: str,
    path: str = Query("", description="Directory path to List[Any]"),
    recursive: bool = Query(False, description="List recursively"),
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> DirectoryListResponse:
    """List directory contents from container."""
    try:
        # Get container by ID or name with proper error handling
        await _get_container_by_name(container_name, service, tenant_id)

        safe_path = _validate_path(path) if path else "/workspace"
        client = _get_client()

        if not recursive:
            data = await client.list_files(
                container_name,
                headers=optional_sibling_headers(tenant_id) or None,
                path=safe_path,
            )
            if data is None or data.get("type") != "directory":
                raise HTTPException(status_code=400, detail="Path is not a directory")
            items: list[DirectoryItem] = []
            for entry in data.get("entries", []):
                name = entry.get("name", "")
                if name in (".", ".."):
                    continue
                item_path = f"{safe_path.rstrip('/')}/{name}" if safe_path != "/" else f"/{name}"
                entry_type = entry.get("type", "file")
                size = None
                if entry_type == "file":
                    try:
                        size = int(entry.get("size", 0))
                    except (ValueError, TypeError):
                        size = None
                items.append(
                    DirectoryItem(
                        name=name,
                        path=item_path,
                        type=entry_type,
                        size=size,
                        permissions=entry.get("permissions"),
                    )
                )
        else:
            data = await client.exec(
                container_name,
                headers=optional_sibling_headers(tenant_id) or None,
                command=["find", safe_path, "-mindepth", "1"],
                workdir="/workspace",
                timeout=30,
            )
            if data.get("exit_code", -1) != 0:
                raise internal_error(f"Directory listing failed: {data.get('stderr', '')}")
            stdout = data.get("stdout", "")
            all_lines = [line for line in stdout.strip().splitlines() if line]
            line_set = set(all_lines)
            items = []
            for line in sorted(all_lines):
                # Determine if directory: another path starts with this path + "/"
                is_dir = any(other.startswith(line + "/") for other in line_set if other != line)
                rel = line.removeprefix(safe_path).lstrip("/")
                if not rel:
                    continue
                name = rel.split("/")[-1]
                items.append(
                    DirectoryItem(
                        name=name,
                        path=line,
                        type="directory" if is_dir else "file",
                        size=None,
                        permissions=None,
                    )
                )

        return DirectoryListResponse(
            status=ResponseStatus.SUCCESS,
            message="Directory contents retrieved successfully",
            path=safe_path,
            items=items,
            total=len(items),
        )

    except HTTPException:
        raise
    except Exception as e:
        from coding_client.clients import ServiceError

        if isinstance(e, ServiceError):
            if getattr(e, "status_code", 500) == 404:
                raise not_found("Directory", path or "/workspace") from e
            raise internal_error(f"Containers service error: {e}") from e
        raise internal_error(f"Unexpected error listing directory: {str(e)}") from e


@router.post(
    "/{container_name}/search",
    operation_id="search_repository",
    response_model=SearchResponse,
    summary="Search Repository",
    include_in_schema=False,
    description="""
    Search for files or content patterns in the containerized repository.

    **Search for files by name:**
    ```json
    {
        "search_type": "files",
        "pattern": "*.py",
        "max_results": 50
    }
    ```

    **Search for content:**
    ```json
    {
        "search_type": "content",
        "pattern": "TODO",
        "file_pattern": "*.py",
        "case_sensitive": false,
        "max_results": 100
    }
    ```

    **Search types:**
    - `files` - Find files by name pattern (supports wildcards)
    - `content` - Search text content within files (grep-like)

    **Response includes:**
    - Matching files or content locations
    - File paths and line numbers
    - Context around matches
    - Total results found

    **Perfect for:**
    - Finding specific files or functions
    - Locating TODOs and comments
    - Code pattern analysis
    - Security auditing
    """,
    responses={
        200: {"description": "Search completed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid search parameters",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {"model": Error404Response, "description": "Container not found"},
        422: {"model": Error422Response, "description": "Validation error"},
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def search_repository(
    container_name: str,
    request: SearchRequest,
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> SearchResponse:
    """Search files and content in containerized repository."""
    try:
        # Get container by ID or name with proper error handling
        container = await _get_container_by_name(container_name, service, tenant_id)

        try:
            results, truncated = await _perform_search(
                container,
                request,
                container_name,
                tenant_id,
            )
        except HTTPException:
            # Re-raise HTTP exceptions (like 400 for invalid search_type)
            raise
        except Exception as e:
            raise internal_error(f"Search operation failed: {str(e)}") from e

        if truncated:
            logger.warning(
                "Search in %s capped at max_results=%d; results truncated",
                container_name,
                request.max_results,
            )

        message = f"Found {len(results)} results"
        if truncated:
            message += f" (truncated at max_results={request.max_results}; more matches exist)"
        return SearchResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            results=results,
            total_found=len(results),
            truncated=truncated,
            search_type=request.search_type,
            pattern=request.pattern,
            request_id=f"search-{container_name}-{request.search_type}",
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        # Catch any other unexpected exceptions
        raise internal_error(f"Unexpected error during search: {str(e)}") from e


# Vulture whitelist for FastAPI route handlers (not directly called)
_vulture_whitelist = [
    "read_repository_file",
    "get_repository_tree",
    "list_directory_contents",
    "search_repository",
]
