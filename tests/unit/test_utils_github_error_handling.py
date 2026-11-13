"""Tests for utils/github_error_handling.py."""

from utils.github_error_handling import map_provision_error_to_http_status


class TestGitHubErrorHandling:
    """Tests for map_provision_error_to_http_status and friends."""

    def test_map_provision_error_validation(self):
        """error_type='validation' should produce HTTP 400."""
        exc = map_provision_error_to_http_status("bad input", error_type="validation")
        assert exc.status_code == 400
        assert "Validation error" in exc.detail

    def test_map_provision_error_unknown(self):
        """Unknown error with no keyword match should produce HTTP 500."""
        exc = map_provision_error_to_http_status("something went wrong", error_type="unknown")
        assert exc.status_code == 500

    def test_error_envelope_shape(self):
        """Returned HTTPException should have status_code and detail."""
        exc = map_provision_error_to_http_status("not found resource", error_type="unknown")
        assert hasattr(exc, "status_code")
        assert hasattr(exc, "detail")
        # "not found" keyword => 404
        assert exc.status_code == 404
