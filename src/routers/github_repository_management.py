"""
GitHub Repository Management Endpoints

Handles repository provisioning, listing, and status operations.
Split from main github.py to reduce file size and improve organization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from coding_client.auth.tenant import require_tenant
from coding_client.exceptions import NotFoundException
from coding_client.models.responses import (  # TODO: verify coding-client import
    bad_request,
    error_response,
    internal_error,
    not_found,
)
from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field

from schemas.github_api_models import (
    ContainerInfo,
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error408Response,
    Error409Response,
    Error429Response,
    Error500Response,
    ListRepositoriesResponse,
    ProvisionRepositoryRequest,
    ProvisionResponse,
    RepositoryInfo,
    ValidationErrorResponse,
)
from services.andon import require_andon_released
from services.github_integration import GitHubIntegration  # TODO: verify
from services.github_provisioning import (  # TODO: verify
    GitHubProvisioningService,
    get_github_provisioning_service,
)
from services.tenant_binding import TenantRepoTokenError, tenant_error_http_exception
from utils.github_error_handling import (
    map_provision_error_to_http_status,
    validate_provision_result,
)
from utils.github_utils import validate_github_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/import",
    operation_id="import_repository_as_workspace",
    response_model=ProvisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import GitHub Repository as Workspace",
    description=(
        "Validate GitHub access, resolve repository credentials and commit metadata, "
        "then create or rebuild the repository's canonical workspace."
    ),
)
@router.post(
    "/provision",
    operation_id="provision_repository",
    response_model=ProvisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Provision Repository",
    include_in_schema=False,
    description="""
    Create a containerized analysis environment from any GitHub repository.

    **Basic Usage:**
    ```json
    {
        "repo_url": "https://github.com/georgiedekker/enterprise_mcp_server",
        "branch": "main"
    }
    ```

    **With Options:**
    ```json
    {
        "repo_url": "https://github.com/georgiedekker/enterprise_mcp_server",
        "branch": "main",
        "container_name": "enterprise-mcp-server",
        "access_token": "ghp_xxxx",
        "provision_type": "branch",
        "force_rebuild": false
    }
    ```

    **What it creates:**
    - Isolated Docker container with repository code
    - Code graph database for analysis
    - REST API endpoint for queries
    - Analysis-ready environment

    **Response includes:**
    - `container_name` - for API operations
    - `api_endpoint` - direct container access
    - `code_graph_id` - for graph queries
    - `commit_sha` - analyzed commit reference

    **Typical provisioning time:** 1-10 minutes depending on repository size.
    """,
    responses={
        200: {
            "model": ProvisionResponse,
            "description": "Request processed successfully",
        },
        201: {
            "model": ProvisionResponse,
            "description": "Repository successfully provisioned",
        },
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
        404: {
            "model": Error404Response,
            "description": "Resource not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        408: {
            "model": Error408Response,
            "description": "Request timeout - provisioning took too long",
        },
        409: {
            "model": Error409Response,
            "description": "Resource conflict",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error - invalid request body",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error",
        },
    },
)
async def provision_repository(
    request: ProvisionRepositoryRequest,
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> ProvisionResponse:
    """Import a GitHub repository into a canonical analysis workspace."""
    logger.info(
        "Received provisioning request: repo_url=%s, branch=%s",
        request.repo_url,
        request.branch,
    )

    # Validate access token if provided
    if request.access_token and not validate_github_token(request.access_token):
        raise bad_request("Invalid or expired access token")

    # Validate required fields to prevent timeouts
    if not request.repo_url or not isinstance(request.repo_url, str):
        raise bad_request("Invalid repository URL")

    # Quick validation to prevent timeouts on malformed requests
    if not request.repo_url.startswith(("https://github.com/", "http://github.com/")):
        raise bad_request("Repository URL must be a valid GitHub URL")

    try:
        # Call the service with timeout to prevent hanging on malformed requests
        provision_result = await asyncio.wait_for(
            service.provision_repository(request, tenant_id=tenant_id),
            timeout=600.0,  # 10 minute timeout for large repos and dependency installation
        )

        # Validate and extract container info from the result
        result = validate_provision_result(provision_result)
        container_info = result["container_info"]
        commit_sha = result["commit_sha"]

        # Accept container_info as Dict[str, Any] or ContainerInfo object
        if isinstance(container_info, ContainerInfo):
            pass  # already correct type
        elif isinstance(container_info, dict):
            try:
                container_info = ContainerInfo(**container_info)
            except Exception as e:
                raise internal_error(f"Invalid container information format: {str(e)}") from e
        else:
            raise internal_error("Invalid container information type returned from service")

        return ProvisionResponse(
            message=("Repository successfully provisioned with code graph initialized"),
            container=container_info,
            commit_sha=commit_sha,
            request_id=str(hash(f"{request.repo_url}_{request.branch}")),
        )

    except TimeoutError as e:
        raise error_response(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            message="Repository provisioning timed out. Please try again with a valid repository.",
            error_code="TIMEOUT",
        ) from e
    except TenantRepoTokenError as e:
        raise tenant_error_http_exception(e) from e
    except NotFoundException as e:
        # Handle repository not found errors with proper 404 status
        raise not_found(
            "Repository",
            str(
                e.detail["message"]
                if hasattr(e, "detail") and isinstance(e.detail, dict)
                else str(e)
            ),
        ) from e
    except ValueError as e:
        raise bad_request(f"Invalid request: {str(e)}") from e
    except HTTPException as e:
        # Log and re-raise HTTPExceptions with their original status codes
        logger.error(
            "GITHUB_DEBUG: HTTPException caught: status=%s, detail=%s",
            e.status_code,
            e.detail,
        )
        raise
    except Exception as e:
        # Use the error mapping utility for consistent error handling
        logger.error("Provisioning failed for repo %s: %s", request.repo_url, str(e))
        raise map_provision_error_to_http_status(str(e)) from e


@router.get(
    "/",
    operation_id="list_repositories_root",
    response_model=ListRepositoriesResponse,
    summary="List Repositories (root)",
    include_in_schema=False,
)
@router.get(
    "/repositories",
    operation_id="list_repositories",
    response_model=ListRepositoriesResponse,
    summary="List Repositories",
    include_in_schema=False,
    description="""
    Get all provisioned GitHub repositories and their status.

    **Usage:**
    ```bash
    GET /api/repositories/repositories
    ```

    **Response includes:**
    - Repository details (URL, branch, description)
    - Container status (`running`, `building`, `stopped`, `error`)
    - API endpoints for direct access
    - Code graph IDs for queries

    **Status meanings:**
    - `running` - Ready for analysis
    - `building` - Still provisioning (check back later)
    - `stopped` - Paused (can be restarted)
    - `error` - Failed (needs troubleshooting)

    **Use container_name from response with other endpoints for analysis operations.**
    """,
    responses={
        200: {
            "model": ListRepositoriesResponse,
            "description": "Successfully retrieved repository list",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "No repositories found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Error retrieving repositories",
        },
    },
)
async def list_repositories(
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> ListRepositoriesResponse:
    """List all provisioned repositories with their current status."""
    try:
        # Get containers from the service
        containers = await service.list_containers(tenant_id=tenant_id)

        # Convert to repository info format
        repositories: list[Any] = []
        for container in containers:
            if container.repo_url:  # Only include containers with GitHub repos
                repo_info = RepositoryInfo(
                    container_name=container.container_name,
                    repo_url=container.repo_url,
                    branch=container.branch,
                    status=container.status,
                    api_endpoint=container.api_endpoint,
                    internal_endpoint=getattr(
                        container, "internal_endpoint", container.api_endpoint
                    ),
                    public_endpoint=getattr(container, "public_endpoint", container.api_endpoint),
                    code_graph_id=container.code_graph_id,
                    description=container.description,
                )
                repositories.append(repo_info)

        return ListRepositoriesResponse(
            message=f"Successfully retrieved {len(repositories)} repository containers",
            repositories=repositories,
            request_id=str(hash("list_repositories")),
        )

    except OSError as e:
        # Handle I/O related errors
        raise error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=f"Service temporarily unavailable: {str(e)}",
            error_code="SERVICE_UNAVAILABLE",
        ) from e
    except Exception as e:
        logger.exception("Failed to list repositories")
        raise internal_error(f"Failed to list repositories: {str(e)}") from e


@router.get(
    "/status/{container_name}",
    operation_id="get_provisioning_status",
    response_model=dict[str, Any],
    summary="Repository Status",
    include_in_schema=False,
    description="""
    Get real-time status and details for a specific repository container.

    **Usage:**
    ```bash
    GET /api/repositories/status/my-container-name
    ```

    **Accepts:** Container name from the system registry

    **Response includes:**
    - Current status (`running`, `building`, `stopped`, `error`)
    - API endpoint (if running)
    - Repository details (URL, branch, description)
    - Code graph ID for queries

    **Use this to:**
    - Monitor provisioning progress
    - Check if container is ready for analysis
    - Get connection details for direct API access
    - Troubleshoot container issues
    """,
    responses={
        200: {
            "model": dict[str, Any],
            "description": "Successfully retrieved container status",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Container not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Error retrieving status",
        },
    },
)
async def get_provisioning_status(
    container_name: str = Path(..., description="Container name to check status for"),
    service: GitHubProvisioningService = Depends(get_github_provisioning_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Get detailed status information for a specific container."""
    try:
        # Try to find container by name
        target_container = await service.docker_service.find_container_by_name(
            container_name, tenant_id=tenant_id
        )

        # Fallback to searching all containers if not found directly
        if not target_container:
            containers = await service.list_containers(tenant_id=tenant_id)
            for container in containers:
                if container_name == container.container_name:
                    target_container = container
                    break

        if not target_container:
            raise not_found("Container", container_name)

        return {
            "container_name": target_container.container_name,
            "status": (
                target_container.status.value
                if hasattr(target_container.status, "value")
                else str(target_container.status)
            ),
            "repo_url": target_container.repo_url,
            "branch": target_container.branch,
            "api_endpoint": target_container.api_endpoint,
            "internal_endpoint": getattr(
                target_container, "internal_endpoint", target_container.api_endpoint
            ),
            "public_endpoint": getattr(
                target_container, "public_endpoint", target_container.api_endpoint
            ),
            "code_graph_id": target_container.code_graph_id,
            "description": target_container.description,
            "message": f"Container is {target_container.status}",
        }

    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except RuntimeError:
        # Containers client not initialised → treat as container not found
        raise not_found("Container", container_name)
    except Exception as e:
        raise internal_error(f"Failed to get status: {str(e)}") from e


# Request/Response models for upstream PR creation
class CreateUpstreamPRRequest(BaseModel):
    """Request model for creating upstream PR from system fork to user's repo."""

    container_name: str = Field(
        ...,
        description="Container name containing the repository metadata",
        min_length=1,
        json_schema_extra={"example": "codegraph-myrepo-abc123"},
    )
    title: str = Field(
        ...,
        description="Pull request title",
        min_length=1,
        max_length=256,
        json_schema_extra={"example": "Feature: Add new capability"},
    )
    message: str = Field(
        ...,
        description="Pull request body/description",
        min_length=1,
        json_schema_extra={"example": "This PR adds a new feature..."},
    )
    head_branch: str = Field(
        ...,
        description="Source branch name (optionally with owner prefix like 'nonagenticai:feature-branch')",
        min_length=1,
        json_schema_extra={"example": "feature-branch"},
    )
    base_branch: str = Field(
        default="main",
        description="Target branch on user's repository",
        json_schema_extra={"example": "main"},
    )


class CreateUpstreamPRResponse(BaseModel):
    """Response model for upstream PR creation."""

    success: bool = Field(..., description="Whether the PR was created successfully")
    pr_number: int | None = Field(None, description="Pull request number if successful")
    pr_url: str | None = Field(None, description="Pull request URL if successful")
    message: str = Field(..., description="Success or error message")
    original_repo: str | None = Field(None, description="Original repository (user's repo)")
    system_fork: str | None = Field(None, description="System fork owner")
    branch_name: str | None = Field(None, description="Branch name used")
    error: str | None = Field(None, description="Error details if unsuccessful")
    status_code: int | None = Field(None, description="HTTP status code for errors")


@router.post(
    "/create-upstream-pr",
    operation_id="create_upstream_pull_request",
    dependencies=[Depends(require_andon_released)],
    response_model=CreateUpstreamPRResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Upstream Pull Request",
    description="""
    Create a pull request from the system fork back to the user's original private repository.

    **Use Case:**
    In the dual-token workflow, after work is completed on the system fork (@nonagenticai),
    this endpoint creates a PR back to the user's private repository.

    **Workflow:**
    1. User provisions their private repo (using their GitHub token)
    2. System creates fork under @nonagenticai account
    3. Work is performed on the system fork
    4. This endpoint creates PR from system fork -> user's private repo

    **Example:**
    ```json
    {
        "container_name": "codegraph-myrepo-abc123",
        "title": "Feature: Add authentication system",
        "message": "This PR adds JWT-based authentication...",
        "head_branch": "feature/auth",
        "base_branch": "main"
    }
    ```

    **Authentication:**
    Uses the user's encrypted GitHub token stored during provisioning.
    """,
    responses={
        201: {
            "description": "Pull request created successfully",
            "model": CreateUpstreamPRResponse,
        },
        400: {"model": Error400Response, "description": "Invalid request parameters"},
        404: {"model": Error404Response, "description": "Container not found"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def create_upstream_pull_request(
    request: CreateUpstreamPRRequest,
    tenant_id: str = Depends(require_tenant),
) -> CreateUpstreamPRResponse:
    """Create a pull request from system fork to user's private repository."""
    try:
        # No GitHubIntegration instance: the method resolves its own credential
        # WITH this tenant once it knows the repo owner. A no-arg construct used
        # to resolve an ownerless (home installation / PAT) credential up front,
        # ungated and unused.
        result = await GitHubIntegration.create_upstream_pull_request(
            container_name=request.container_name,
            title=request.title,
            message=request.message,
            head_branch=request.head_branch,
            base_branch=request.base_branch,
            tenant_id=tenant_id,
        )

        # Return response
        if result.get("success"):
            return CreateUpstreamPRResponse(
                success=True,
                pr_number=result.get("pr_number"),
                pr_url=result.get("pr_url"),
                message=result.get("message", "Pull request created successfully"),
                original_repo=result.get("original_repo"),
                system_fork=result.get("system_fork"),
                branch_name=result.get("branch_name"),
            )
        else:
            # Return error response with appropriate status code
            http_status = result.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
            error_msg = result.get("error", "Failed to create pull request")

            raise error_response(
                status_code=http_status,
                message=error_msg,
                error_code=result.get("error_code") or "PR_CREATION_FAILED",
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating upstream PR: {e}", exc_info=True)
        raise internal_error(f"Failed to create upstream PR: {str(e)}") from e


# Vulture whitelist for FastAPI route handlers (not directly called)
_vulture_whitelist = [
    "provision_repository",
    "list_repositories",
    "get_provisioning_status",
    "create_upstream_pull_request",
]
