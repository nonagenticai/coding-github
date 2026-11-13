"""
GitHub Authentication and Validation Utilities

Provides centralized authentication, token validation, and security utilities for GitHub operations.
Split from main github.py to improve organization and reusability.
"""

from __future__ import annotations

import logging

from utils.github_utils import validate_github_token

logger = logging.getLogger(__name__)


def validate_access_token(token: str | None) -> bool:
    """
    Validate a GitHub access token.

    Args:
        token: GitHub access token to validate

    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        return False

    return validate_github_token(token)


def validate_repository_url(repo_url: str) -> bool:
    """
    Validate that a repository URL is a valid GitHub URL.

    Args:
        repo_url: Repository URL to validate

    Returns:
        bool: True if URL is valid GitHub URL, False otherwise
    """
    if not repo_url or not isinstance(repo_url, str):
        return False

    return repo_url.startswith(("https://github.com/", "http://github.com/"))


def sanitize_container_name(name: str) -> str:
    """
    Sanitize a container name to ensure it's safe for Docker usage.

    Args:
        name: Container name to sanitize

    Returns:
        str: Sanitized container name
    """
    if not name:
        return ""

    # Remove invalid characters and convert to lowercase
    sanitized = "".join(c for c in name.lower() if c.isalnum() or c in "-_")

    # Ensure it doesn't start with a dash or underscore
    while sanitized and sanitized[0] in "-_":
        sanitized = sanitized[1:]

    # Limit length to reasonable Docker container name length
    return sanitized[:63] if sanitized else ""


def validate_github_credentials(token: str | None = None) -> dict:
    """
    Validate GitHub credentials and return validation result.

    Args:
        token: Optional GitHub access token

    Returns:
        dict: Validation result with status and details
    """
    result = {
        "valid": False,
        "token_provided": bool(token),
        "token_valid": False,
        "message": "",
    }

    if not token:
        result["message"] = "No GitHub access token provided"
        return result

    if validate_github_token(token):
        result["valid"] = True
        result["token_valid"] = True
        result["message"] = "GitHub credentials are valid"
    else:
        result["message"] = "Invalid or expired GitHub access token"

    return result


def check_repository_access_permissions(repo_url: str, token: str | None = None) -> dict:
    """
    Check if we have appropriate permissions to access a repository.

    Args:
        repo_url: GitHub repository URL
        token: Optional GitHub access token

    Returns:
        dict: Access check result
    """
    result = {
        "accessible": False,
        "public_repo": False,
        "token_required": False,
        "message": "",
    }

    if not validate_repository_url(repo_url):
        result["message"] = "Invalid GitHub repository URL"
        return result

    # For now, assume public repos are accessible without tokens
    # and private repos require valid tokens
    if token and validate_github_token(token):
        result["accessible"] = True
        result["message"] = "Repository accessible with provided token"
    else:
        # Assume public access for repositories without authentication
        result["accessible"] = True
        result["public_repo"] = True
        result["message"] = "Attempting public repository access"

        if not token:
            result["token_required"] = True
            result["message"] += " (token may be required for private repositories)"

    return result


def generate_secure_container_name(repo_url: str, branch: str = "main") -> str:
    """
    Generate a secure, unique container name based on repository and branch.

    Args:
        repo_url: GitHub repository URL
        branch: Git branch name

    Returns:
        str: Generated container name
    """
    if not repo_url:
        return ""

    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    # Combine repo name and branch
    base_name = f"{repo_name}-{branch}"

    # Sanitize the result
    return sanitize_container_name(base_name)


# Security validation utilities
def is_safe_branch_name(branch_name: str) -> bool:
    """
    Check if a branch name is safe to use (no injection attacks).

    Args:
        branch_name: Git branch name to validate

    Returns:
        bool: True if branch name is safe, False otherwise
    """
    if not branch_name or not isinstance(branch_name, str):
        return False

    # Check for potentially dangerous characters
    dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "{", "}", "[", "]", "<", ">"]

    for char in dangerous_chars:
        if char in branch_name:
            return False

    return True


def validate_pr_request_fields(fields: dict) -> dict:
    """
    Validate fields for PR creation requests.

    Args:
        fields: Dictionary of PR request fields

    Returns:
        dict: Validation result
    """
    result = {"valid": True, "errors": [], "warnings": []}

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
        if field not in fields or not fields[field] or not str(fields[field]).strip():
            result["valid"] = False
            result["errors"].append(f"Missing or empty required field: {field}")

    # Validate branch names
    for branch_field in ["head_branch", "base_branch"]:
        if branch_field in fields and fields[branch_field]:
            if not is_safe_branch_name(fields[branch_field]):
                result["valid"] = False
                result["errors"].append(
                    f"Unsafe branch name in {branch_field}: {fields[branch_field]}"
                )

    # Check for reasonable field lengths
    max_lengths = {
        "title": 200,
        "message": 10000,
        "original_owner": 100,
        "original_repo": 100,
        "fork_owner": 100,
        "fork_repo": 100,
        "head_branch": 100,
        "base_branch": 100,
    }

    for field, max_length in max_lengths.items():
        if field in fields and fields[field] and len(str(fields[field])) > max_length:
            result["warnings"].append(
                f"Field {field} is longer than recommended ({max_length} chars)"
            )

    return result


def mask_sensitive_data(data: dict) -> dict:
    """
    Mask sensitive data in request/response for logging.

    Args:
        data: Dictionary that may contain sensitive data

    Returns:
        dict: Dictionary with sensitive fields masked
    """
    if not isinstance(data, dict):
        return data

    sensitive_fields = ["access_token", "token", "password", "secret", "key"]
    masked_data = data.copy()

    for field in sensitive_fields:
        if field in masked_data and masked_data[field]:
            # Show only first 4 and last 4 characters
            token_value = str(masked_data[field])
            if len(token_value) > 8:
                masked_data[field] = f"{token_value[:4]}...{token_value[-4:]}"
            else:
                masked_data[field] = "***"

    return masked_data


# Utility functions for validation results
def create_validation_error(message: str, field: str | None = None) -> dict:
    """
    Create a standardized validation error response.

    Args:
        message: Error message
        field: Optional field name that caused the error

    Returns:
        dict: Standardized error response
    """
    error = {
        "error": "validation_error",
        "message": message,
        "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp in real implementation
    }

    if field:
        error["field"] = field

    return error


def create_auth_error(message: str) -> dict:
    """
    Create a standardized authentication error response.

    Args:
        message: Error message

    Returns:
        dict: Standardized auth error response
    """
    return {
        "error": "authentication_error",
        "message": message,
        "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp in real implementation
    }
