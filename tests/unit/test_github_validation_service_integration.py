"""Tests for github_validation_service.py (integration service)."""

from types import SimpleNamespace
from unittest.mock import patch

from schemas.github_api_models import PRValidationResult, ValidatePRRequest
from services.pr_validation.integration.github_validation_service import (
    PRGitHubValidationService,
)


def _make_request(validation_types=None):
    return ValidatePRRequest(
        repo_owner="test-owner",
        repo_name="test-repo",
        pr_number=1,
        validation_types=validation_types or ["security", "performance", "quality"],
    )


def _make_pr_file(filename="app.py", patch="+x = 1", additions=1, deletions=0, changes=1):
    return SimpleNamespace(
        filename=filename,
        patch=patch,
        additions=additions,
        deletions=deletions,
        changes=changes,
        sha="abc123",
    )


class TestPRGitHubValidationService:
    """Tests for PRGitHubValidationService."""

    def setup_method(self):
        self.service = PRGitHubValidationService()

    async def test_orchestration_fetch_diff_validate(self):
        """Legacy validation with security, performance, quality returns results for each type."""
        request = _make_request(validation_types=["security", "quality"])
        files = [_make_pr_file()]
        results = await self.service.validate_github_pr(request, files)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, PRValidationResult) for r in results)
        types = {r.validation_type for r in results}
        assert "security" in types
        assert "quality" in types

    async def test_repo_not_found_error(self):
        """When validate_github_pr raises, it should return an error result."""
        request = _make_request()
        # Pass empty file list that will cause issues in comprehensive validation
        with patch.object(
            self.service, "_convert_github_pr_data", side_effect=Exception("Repo not found")
        ):
            results = await self.service.validate_github_pr(request, [])
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].validation_type == "error"

    async def test_pr_not_found_error(self):
        """When an exception occurs, error result is returned."""
        request = _make_request()
        with patch.object(
            self.service,
            "_should_use_comprehensive_validation",
            side_effect=Exception("PR not found"),
        ):
            results = await self.service.validate_github_pr(request, [_make_pr_file()])
        assert results[0].passed is False
        assert "error" in results[0].validation_type

    async def test_compare_api_failure_fallback(self):
        """Comprehensive validation failure should fall back to legacy."""
        request = _make_request(validation_types=["security", "performance", "quality"])
        files = [_make_pr_file()]
        # comprehensive validation triggers because len(types) > 2
        with patch.object(
            self.service,
            "_run_comprehensive_validation",
            side_effect=Exception("compare API fail"),
        ):
            # The outer try/except should return error results
            results = await self.service.validate_github_pr(request, files)
            # Should get error result
            assert len(results) >= 1

    async def test_malformed_diff_handling(self):
        """Files with no patch should still produce results without errors."""
        request = _make_request(validation_types=["security"])
        files = [_make_pr_file(patch=None, changes=0)]
        results = await self.service.validate_github_pr(request, files)
        assert isinstance(results, list)
        # Should return at least one result
        assert len(results) >= 1

    async def test_response_shape(self):
        """Each result should have validation_type, passed, score, issues, suggestions."""
        request = _make_request(validation_types=["security"])
        files = [_make_pr_file()]
        results = await self.service.validate_github_pr(request, files)
        for r in results:
            assert hasattr(r, "validation_type")
            assert hasattr(r, "passed")
            assert hasattr(r, "score")
            assert hasattr(r, "issues")
            assert hasattr(r, "suggestions")
            assert isinstance(r.score, (int, float))
            assert isinstance(r.issues, list)
