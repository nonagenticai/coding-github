"""
GitHub API Integration Utilities

Common utilities and helper functions for GitHub API operations.
Split from main github.py to improve code organization and reusability.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.models.responses import (  # TODO: verify coding-client import
    internal_error,
    not_found,
)

from services.github_provisioning import GitHubProvisioningService

logger = logging.getLogger(__name__)


async def get_container_by_name_or_id(
    container_identifier: str, service: GitHubProvisioningService, tenant_id: str
) -> Any:
    """
    Get container by name or ID with comprehensive error handling.

    Args:
        container_identifier: Container name or ID to search for
        service: GitHub provisioning service instance
        tenant_id: Tenant ID for container filtering

    Returns:
        Container object if found

    Raises:
        HTTPException: If container not found or service error
    """
    try:
        # Try to find container by name first
        container = await service.docker_service.find_container_by_name(
            container_identifier, tenant_id=tenant_id
        )

        # Fallback to searching all containers if not found directly
        if not container:
            containers = await service.list_containers(tenant_id=tenant_id)
            for c in containers:
                if container_identifier in [c.container_name, getattr(c, "id", None)]:
                    container = c
                    break

    except Exception as e:
        raise internal_error(f"Failed to retrieve containers: {str(e)}") from e

    if not container:
        raise not_found("Container", container_identifier)

    return container


def validate_github_repository_url(repo_url: str) -> tuple[bool, str]:
    """
    Validate GitHub repository URL format and extract components.

    Args:
        repo_url: Repository URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not repo_url or not isinstance(repo_url, str):
        return False, "Repository URL must be a non-empty string"

    repo_url = repo_url.strip()

    if not repo_url.startswith(("https://github.com/", "http://github.com/")):
        return (
            False,
            "Repository URL must be a valid GitHub URL (https://github.com/...)",
        )

    # Check if URL has proper format: https://github.com/owner/repo
    url_parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
    path_parts = url_parts.split("/")

    if len(path_parts) < 2:
        return False, "Repository URL must include owner and repository name"

    owner, repo = path_parts[0], path_parts[1]

    if not owner or not repo:
        return False, "Repository URL must have valid owner and repository name"

    # Remove .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]

    if not repo:
        return False, "Repository name cannot be empty"

    return True, ""


def extract_repo_info_from_url(repo_url: str) -> dict[str, str]:
    """
    Extract repository owner and name from GitHub URL.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Dictionary with 'owner' and 'repo' keys

    Raises:
        ValueError: If URL format is invalid
    """
    is_valid, error_msg = validate_github_repository_url(repo_url)
    if not is_valid:
        raise ValueError(error_msg)

    # Clean up the URL
    clean_url = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
    path_parts = clean_url.split("/")

    owner = path_parts[0]
    repo = path_parts[1]

    # Remove .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]

    return {"owner": owner, "repo": repo}


def format_container_status_response(container: Any) -> dict[str, Any]:
    """
    Format container information into a standardized status response.

    Args:
        container: Container object

    Returns:
        Formatted status dictionary
    """
    return {
        "container_name": getattr(container, "container_name", "unknown"),
        "status": (
            container.status.value if hasattr(container.status, "value") else str(container.status)
        ),
        "repo_url": getattr(container, "repo_url", None),
        "branch": getattr(container, "branch", None),
        "api_endpoint": getattr(container, "api_endpoint", None),
        "internal_endpoint": getattr(
            container,
            "internal_endpoint",
            getattr(container, "api_endpoint", None),
        ),
        "public_endpoint": getattr(
            container,
            "public_endpoint",
            getattr(container, "api_endpoint", None),
        ),
        "code_graph_id": getattr(container, "code_graph_id", None),
        "description": getattr(container, "description", None),
        "message": f"Container is {container.status}",
    }


def validate_pr_creation_request(
    request_data: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Validate pull request creation request data.

    Args:
        request_data: PR creation request data

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_fields = [
        "original_owner",
        "original_repo",
        "fork_owner",
        "fork_repo",
        "head_branch",
        "base_branch",
        "title",
        "message",
    ]

    # Check for missing required fields
    for field in required_fields:
        if field not in request_data:
            errors.append(f"Missing required field: {field}")
        elif not request_data[field] or not str(request_data[field]).strip():
            errors.append(f"Field '{field}' cannot be empty")

    # Additional validation for specific fields
    if "title" in request_data and len(str(request_data["title"])) > 200:
        errors.append("PR title cannot exceed 200 characters")

    if "message" in request_data and len(str(request_data["message"])) > 10000:
        errors.append("PR message cannot exceed 10,000 characters")

    # Validate branch names don't contain dangerous characters
    dangerous_chars = [";", "&", "|", "`", "$", "&&", "||"]
    for branch_field in ["head_branch", "base_branch"]:
        if branch_field in request_data:
            branch_name = str(request_data[branch_field])
            for char in dangerous_chars:
                if char in branch_name:
                    errors.append(f"Branch name '{branch_name}' contains invalid character: {char}")
                    break

    return len(errors) == 0, errors


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize data for safe logging by masking sensitive information.

    Args:
        data: Data dictionary to sanitize

    Returns:
        Sanitized data dictionary
    """
    if not isinstance(data, dict):
        return data

    sanitized = data.copy()
    sensitive_fields = ["access_token", "token", "password", "secret", "authorization"]

    for field in sensitive_fields:
        if field in sanitized and sanitized[field]:
            # Mask all but first and last 4 characters
            value = str(sanitized[field])
            if len(value) > 8:
                sanitized[field] = f"{value[:4]}***{value[-4:]}"
            else:
                sanitized[field] = "***masked***"

    return sanitized


def calculate_validation_score(issues: list[Any]) -> float:
    """
    Calculate a validation score based on issues found.

    Args:
        issues: List of validation issues

    Returns:
        Validation score (0-100)
    """
    if not issues:
        return 100.0

    # Weight issues by severity
    severity_weights = {"critical": 30, "high": 15, "medium": 10, "low": 5}

    total_penalty = 0
    for issue in issues:
        severity = getattr(issue, "severity", "medium")
        penalty = severity_weights.get(severity, 10)
        total_penalty += penalty

    # Cap the penalty to ensure score doesn't go below 0
    score = max(0, 100 - total_penalty)
    return float(score)


def determine_pr_recommendation(score: float, has_critical_issues: bool) -> str:
    """
    Determine PR recommendation based on validation score and issues.

    Args:
        score: Validation score (0-100)
        has_critical_issues: Whether critical issues were found

    Returns:
        Recommendation string: "approve", "needs_work", or "reject"
    """
    if has_critical_issues:
        return "reject"

    if score >= 80:
        return "approve"
    elif score >= 60:
        return "needs_work"
    else:
        return "reject"


def format_github_error_message(error: Exception) -> str:
    """
    Format GitHub API error messages for user-friendly display.

    Args:
        error: Exception from GitHub API

    Returns:
        User-friendly error message
    """
    error_msg = str(error).lower()

    if "404" in error_msg or "not found" in error_msg:
        return "Repository or resource not found. Please check the URL and permissions."

    if "403" in error_msg or "forbidden" in error_msg:
        return "Access denied. Please check your GitHub token permissions."

    if "401" in error_msg or "unauthorized" in error_msg:
        return "Authentication failed. Please check your GitHub token."

    if "rate limit" in error_msg:
        return "GitHub API rate limit exceeded. Please try again later."

    if "timeout" in error_msg:
        return (
            "Request timed out. The repository may be too large or GitHub is experiencing issues."
        )

    # Return a generic message for other errors
    return f"GitHub API error: {str(error)}"


def validate_file_changes(changes: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """
    Validate file changes for apply-changes operations.

    Args:
        changes: List of file changes

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not changes:
        errors.append("At least one file change is required")
        return False, errors

    if len(changes) > 50:
        errors.append("Too many file changes (maximum 50 allowed)")

    for i, change in enumerate(changes):
        if not isinstance(change, dict):
            errors.append(f"Change {i + 1}: Must be an object with file_path and content")
            continue

        if "file_path" not in change or not change["file_path"]:
            errors.append(f"Change {i + 1}: Missing or empty file_path")

        if "content" not in change:
            errors.append(f"Change {i + 1}: Missing content field")

        # Validate file path
        file_path = change.get("file_path", "")
        from pathlib import PurePosixPath

        try:
            normalized = PurePosixPath(file_path)
            if normalized.is_absolute() or ".." in normalized.parts or "\x00" in file_path:
                errors.append(
                    f"Change {i + 1}: Invalid file path '{file_path}' (no parent directory access)"
                )
        except (ValueError, TypeError):  # fmt: skip
            errors.append(f"Change {i + 1}: Invalid file path '{file_path}'")

        # Check content size (limit to 1MB)
        content = change.get("content", "")
        if len(str(content)) > 1024 * 1024:
            errors.append(f"Change {i + 1}: File content too large (maximum 1MB)")

    return len(errors) == 0, errors


# Utility class for GitHub operation context
class GitHubOperationContext:
    """
    Context manager for GitHub operations with consistent error handling.
    """

    def __init__(self, operation_name: str, container_name: str | None = None):
        self.operation_name = operation_name
        self.container_name = container_name
        self.start_time = None

    def __enter__(self):
        logger.info(f"Starting GitHub operation: {self.operation_name}")
        if self.container_name:
            logger.info(f"Container: {self.container_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"GitHub operation failed: {self.operation_name} - {exc_val}")
        else:
            logger.info(f"GitHub operation completed: {self.operation_name}")
        return False  # Don't suppress exceptions
