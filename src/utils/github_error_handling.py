"""
GitHub Error Handling Utilities

Extracted error handling logic to reduce complexity in main endpoint file.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status


def map_provision_error_to_http_status(
    error_message: str, error_type: str = "unknown"
) -> HTTPException:
    """
    Map provisioning error messages to appropriate HTTP status codes.

    Args:
        error_message: The error message to analyze
        error_type: The error type if known

    Returns:
        HTTPException with appropriate status code
    """
    # Map error types to status codes
    if error_type == "validation":
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {error_message}",
        )

    if error_type in ["runtime", "system"]:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {error_message}",
        )

    # Fallback: analyze error message for keywords
    error_lower = error_message.lower()

    # Check for network/connectivity errors first
    network_error_keywords = [
        "connection refused",
        "connection timeout",
        "connection error",
        "network is unreachable",
        "no route to host",
        "name resolution",
        "dns",
        "max retries exceeded",
        "connection aborted",
        "httpsconnectionpool",
        "failed to establish a new connection",
        "[errno 111] connection refused",
        "newconnectionerror",
    ]
    if any(keyword in error_lower for keyword in network_error_keywords):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Network connectivity issue: {error_message}",
        )

    # Check for 404 errors (more specific)
    not_found_keywords = [
        "not found",
        "404",
        "repository.*not found",
        "repository.*not found or is private",
    ]
    if any(keyword in error_lower for keyword in not_found_keywords):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message,
        )

    # Check for other client errors (400)
    client_error_keywords = [
        "invalid",
        "validation",
        "format",
        "supported",
        "url",
        "expired",
        "token",
        "authentication",
        "authorization",
        "access",
        "permission",
        "400",
        "bad_request",
    ]

    if any(keyword in error_lower for keyword in client_error_keywords):
        # Extract clean error message if it contains status code prefix
        detail = error_message
        if "server error:" in error_lower and "400:" in error_message:
            parts = error_message.split("400:", 1)
            if len(parts) > 1:
                detail = parts[1].strip()
        elif "400" in error_message:
            parts = error_message.split("400", 1)
            if len(parts) > 1 and parts[1].strip().startswith(":"):
                detail = parts[1][1:].strip()

        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    # Default to 500 for unknown errors
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Provisioning error: {error_message}",
    )


def validate_provision_result(provision_result: Any) -> dict[str, Any]:
    """
    Validate and extract container info from provision result.

    Args:
        provision_result: The result from provisioning service

    Returns:
        Dictionary with container_info and commit_sha

    Raises:
        HTTPException if validation fails
    """
    if isinstance(provision_result, dict):
        container_info = provision_result.get("container_info")
        commit_sha = provision_result.get("commit_sha")

        if not container_info:
            error_message = provision_result.get(
                "error", "No container information returned from service"
            )
            error_type = provision_result.get("error_type", "unknown")
            raise map_provision_error_to_http_status(error_message, error_type)

        return {"container_info": container_info, "commit_sha": commit_sha}
    else:
        # Handle backward compatibility
        if not provision_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No container information returned from service",
            )
        return {"container_info": provision_result, "commit_sha": None}
