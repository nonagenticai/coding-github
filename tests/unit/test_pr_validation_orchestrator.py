"""Tests for pr_validation_orchestrator.py."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from schemas.github_api_models import PRValidationResult


def _make_pr_file(filename="app.py", patch="+x = 1", additions=1, deletions=0, changes=1):
    return SimpleNamespace(
        filename=filename,
        patch=patch,
        additions=additions,
        deletions=deletions,
        changes=changes,
        sha="abc123",
    )


def _make_validation_result(validation_type="security", passed=True, score=90.0, issues=None):
    return PRValidationResult(
        validation_type=validation_type,
        passed=passed,
        score=score,
        issues=issues or [],
        suggestions=[],
    )


class TestPRValidationOrchestrator:
    """Tests for PRValidationOrchestrator."""

    def _get_orchestrator(self):
        """Import and create orchestrator with mocked dependencies."""
        # Remove the conftest stub so the real module can be imported
        key = "services.pr_validation.pr_validation_orchestrator"
        if key in sys.modules:
            existing = sys.modules[key]
            # Only remove if it's the conftest stub (no PRValidationOrchestrator)
            if not hasattr(existing, "PRValidationOrchestrator"):
                del sys.modules[key]

        from services.pr_validation.pr_validation_orchestrator import PRValidationOrchestrator

        orch = PRValidationOrchestrator()
        return orch

    async def test_phase1_fast_validation(self):
        """Phase 1 fast validation should run and return results."""
        orch = self._get_orchestrator()
        fast_results = [
            _make_validation_result("security", True, 90),
            _make_validation_result("quality", True, 85),
        ]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        orch.workflow_trigger.trigger_pr_comprehensive_validation = AsyncMock(
            return_value={"workflow_id": "wf-1"}
        )

        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
            pr_metadata={"head": {"sha": "abc123"}},
        )
        assert result["fast_validation"]["passed"] is True
        assert result["fast_validation"]["score"] == 87.5

    async def test_phase2_github_actions_triggered(self):
        """When fast validation passes, comprehensive validation should be triggered."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("security", True, 95)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        orch.workflow_trigger.trigger_pr_comprehensive_validation = AsyncMock(
            return_value={"workflow_id": "wf-2"}
        )

        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
            pr_metadata={"head": {"sha": "abc123"}},
        )
        assert result["comprehensive_validation"]["status"] == "triggered"
        orch.workflow_trigger.trigger_pr_comprehensive_validation.assert_called_once()

    async def test_orchestration_order_fast_before_comprehensive(self):
        """Fast validation must complete before comprehensive is triggered."""
        orch = self._get_orchestrator()
        call_order = []

        async def mock_fast(*args, **kwargs):
            call_order.append("fast")
            return [_make_validation_result("security", True, 80)]

        async def mock_trigger(*args, **kwargs):
            call_order.append("comprehensive")
            return {"workflow_id": "wf-3"}

        orch.fast_validator.validate_github_pr = mock_fast
        orch.workflow_trigger.trigger_pr_comprehensive_validation = mock_trigger

        await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
            pr_metadata={"head": {"sha": "abc"}},
        )
        assert call_order == ["fast", "comprehensive"]

    async def test_timeout_handling_trigger_failure(self):
        """If comprehensive trigger fails, fast result is still returned."""
        orch = self._get_orchestrator()
        orch.fast_validator.validate_github_pr = AsyncMock(
            return_value=[_make_validation_result("security", True, 90)]
        )
        orch.workflow_trigger.trigger_pr_comprehensive_validation = AsyncMock(
            side_effect=TimeoutError("Trigger timeout")
        )

        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
            pr_metadata={"head": {"sha": "abc"}},
        )
        assert result["fast_validation"]["passed"] is True
        assert result["comprehensive_validation"]["status"] == "trigger_failed"

    async def test_error_aggregation_fast_fails(self):
        """If fast validation fails entirely, status reflects the error."""
        orch = self._get_orchestrator()
        orch.fast_validator.validate_github_pr = AsyncMock(
            side_effect=RuntimeError("DB connection lost")
        )

        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
        )
        assert result["status"] == "error"
        assert "DB connection lost" in result["error"]
        assert result["fast_validation"]["passed"] is False

    async def test_result_shape(self):
        """Result dict should have validation_id, status, fast_validation, comprehensive_validation."""
        orch = self._get_orchestrator()
        orch.fast_validator.validate_github_pr = AsyncMock(
            return_value=[_make_validation_result("security", True, 100)]
        )
        orch.workflow_trigger.trigger_pr_comprehensive_validation = AsyncMock(
            return_value={"workflow_id": "wf-4"}
        )

        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            1,
            [_make_pr_file()],
            pr_metadata={"head": {"sha": "abc"}},
        )
        assert "validation_id" in result
        assert "status" in result
        assert "fast_validation" in result
        assert "comprehensive_validation" in result
        assert result["validation_id"].startswith("pr-val-")
