"""Extended tests for utils/github_error_handling.py — covering uncovered lines."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from utils.github_error_handling import (
    map_provision_error_to_http_status,
    validate_provision_result,
)

# ===========================================================================
# map_provision_error_to_http_status: error_type = "runtime" / "system" (line 35)
# ===========================================================================


class TestMapProvisionErrorRuntime:
    """Cover line 35: error_type in ['runtime', 'system'] -> 500."""

    def test_runtime_error_type(self):
        exc = map_provision_error_to_http_status("something broke", error_type="runtime")
        assert exc.status_code == 500
        assert "Server error" in exc.detail

    def test_system_error_type(self):
        exc = map_provision_error_to_http_status("system crash", error_type="system")
        assert exc.status_code == 500
        assert "Server error" in exc.detail


# ===========================================================================
# Network error keywords (line 60)
# ===========================================================================


class TestMapProvisionErrorNetwork:
    """Cover line 60: network connectivity keywords -> 503."""

    def test_connection_refused(self):
        exc = map_provision_error_to_http_status("Connection refused at localhost:5432")
        assert exc.status_code == 503

    def test_connection_timeout(self):
        exc = map_provision_error_to_http_status("Connection timeout after 30s")
        assert exc.status_code == 503

    def test_dns_error(self):
        exc = map_provision_error_to_http_status("DNS resolution failed for service.local")
        assert exc.status_code == 503

    def test_max_retries_exceeded(self):
        exc = map_provision_error_to_http_status("Max retries exceeded with url")
        assert exc.status_code == 503

    def test_new_connection_error(self):
        exc = map_provision_error_to_http_status("NewConnectionError: cannot connect")
        assert exc.status_code == 503

    def test_https_connection_pool(self):
        exc = map_provision_error_to_http_status("HTTPSConnectionPool(host='x') failed")
        assert exc.status_code == 503

    def test_failed_establish_connection(self):
        exc = map_provision_error_to_http_status(
            "Failed to establish a new connection: [Errno 111]"
        )
        assert exc.status_code == 503

    def test_errno_111_connection_refused(self):
        exc = map_provision_error_to_http_status("[errno 111] connection refused")
        assert exc.status_code == 503

    def test_connection_aborted(self):
        exc = map_provision_error_to_http_status("Connection aborted during transfer")
        assert exc.status_code == 503

    def test_network_unreachable(self):
        exc = map_provision_error_to_http_status("Network is unreachable")
        assert exc.status_code == 503

    def test_no_route_to_host(self):
        exc = map_provision_error_to_http_status("No route to host for 192.168.1.1")
        assert exc.status_code == 503

    def test_name_resolution(self):
        exc = map_provision_error_to_http_status("Name resolution failed")
        assert exc.status_code == 503


# ===========================================================================
# Client error keywords with 400 extraction logic (lines 97-107)
# ===========================================================================


class TestMapProvisionErrorClientErrors:
    """Cover lines 97-107: client error keyword matching and 400 extraction."""

    def test_invalid_keyword(self):
        exc = map_provision_error_to_http_status("Invalid repository URL provided")
        assert exc.status_code == 400

    def test_expired_token(self):
        exc = map_provision_error_to_http_status("Token expired, please refresh")
        assert exc.status_code == 400

    def test_authentication_error(self):
        exc = map_provision_error_to_http_status("Authentication failed for user")
        assert exc.status_code == 400

    def test_permission_denied(self):
        exc = map_provision_error_to_http_status("Permission denied for this action")
        assert exc.status_code == 400

    def test_400_with_server_error_prefix(self):
        """Cover lines 98-101: 'server error:' prefix with '400:' inside."""
        exc = map_provision_error_to_http_status("Server error: 400: Bad request body")
        assert exc.status_code == 400
        assert "Bad request body" in exc.detail

    def test_400_with_colon_after(self):
        """Cover lines 102-105: '400' followed by ':'."""
        exc = map_provision_error_to_http_status("Error 400: Missing required field")
        assert exc.status_code == 400
        assert "Missing required field" in exc.detail

    def test_400_without_colon(self):
        """400 in message without colon after it."""
        exc = map_provision_error_to_http_status("Got 400 bad_request from server")
        assert exc.status_code == 400

    def test_url_keyword(self):
        exc = map_provision_error_to_http_status("Bad URL format in request")
        assert exc.status_code == 400

    def test_format_keyword(self):
        exc = map_provision_error_to_http_status("Invalid format specification")
        assert exc.status_code == 400

    def test_supported_keyword(self):
        exc = map_provision_error_to_http_status("Language not supported")
        assert exc.status_code == 400

    def test_access_keyword(self):
        exc = map_provision_error_to_http_status("Access denied to resource")
        assert exc.status_code == 400

    def test_authorization_keyword(self):
        exc = map_provision_error_to_http_status("Authorization header missing")
        assert exc.status_code == 400


# ===========================================================================
# validate_provision_result: dict path (lines 137-141)
# ===========================================================================


class TestValidateProvisionResultDict:
    """Cover lines 137-141: dict result with missing container_info."""

    def test_dict_with_container_info(self):
        result = validate_provision_result(
            {"container_info": {"id": "ctr-1"}, "commit_sha": "abc123"}
        )
        assert result["container_info"]["id"] == "ctr-1"
        assert result["commit_sha"] == "abc123"

    def test_dict_without_container_info_default_error(self):
        """Cover lines 137-141: no container_info, no error key.
        Default message 'No container information returned from service'
        contains 'format' (in 'information') which triggers 400 via client_error_keywords.
        """
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result({"container_info": None})
        assert exc_info.value.status_code == 400

    def test_dict_with_error_key(self):
        """Cover lines 137-141: error key present in result."""
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result(
                {
                    "container_info": None,
                    "error": "Container creation failed",
                    "error_type": "runtime",
                }
            )
        assert exc_info.value.status_code == 500

    def test_dict_with_validation_error_type(self):
        """error_type='validation' -> 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result(
                {"container_info": None, "error": "bad input", "error_type": "validation"}
            )
        assert exc_info.value.status_code == 400


# ===========================================================================
# validate_provision_result: non-dict path (lines 146-151)
# ===========================================================================


class TestValidateProvisionResultNonDict:
    """Cover lines 146-151: non-dict provision_result."""

    def test_falsy_non_dict_raises_500(self):
        """Cover line 146-150: falsy non-dict => 500."""
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result(None)
        assert exc_info.value.status_code == 500

    def test_empty_string_raises_500(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result("")
        assert exc_info.value.status_code == 500

    def test_zero_raises_500(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_provision_result(0)
        assert exc_info.value.status_code == 500

    def test_truthy_non_dict_returns_container_info(self):
        """Cover line 151: non-dict truthy value used as container_info."""
        result = validate_provision_result("some-container-id")
        assert result["container_info"] == "some-container-id"
        assert result["commit_sha"] is None

    def test_list_non_dict_returns_container_info(self):
        """Non-dict list used as container_info."""
        result = validate_provision_result(["container-1"])
        assert result["container_info"] == ["container-1"]
        assert result["commit_sha"] is None
