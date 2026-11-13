"""Extended tests for pr_validation_orchestrator.py — covering uncovered lines."""

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


class TestPRValidationOrchestratorExtended:
    """Extended tests for uncovered lines in PRValidationOrchestrator."""

    def _get_orchestrator(self):
        key = "services.pr_validation.pr_validation_orchestrator"
        if key in sys.modules:
            existing = sys.modules[key]
            if not hasattr(existing, "PRValidationOrchestrator"):
                del sys.modules[key]

        from services.pr_validation.pr_validation_orchestrator import PRValidationOrchestrator

        orch = PRValidationOrchestrator()
        return orch

    # --- Lines 129-132: SHA extraction from pr_files when metadata has no head.sha ---

    async def test_sha_extracted_from_file_sha_attribute(self):
        """When pr_metadata has no head.sha, SHA should be extracted from file objects."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("security", True, 90)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        trigger_mock = AsyncMock(return_value={"workflow_id": "wf-sha"})
        orch.workflow_trigger.trigger_pr_comprehensive_validation = trigger_mock

        # pr_metadata without head.sha, but pr_files have sha attributes
        pr_file = _make_pr_file()
        result = await orch.validate_pr_with_actions(
            "owner",
            "repo",
            42,
            [pr_file],
            pr_metadata={"head": {"ref": "feature"}},  # no sha key
        )

        assert result["status"] == "fast_validation_passed"
        # The trigger should have been called with the file sha
        call_kwargs = trigger_mock.call_args
        assert call_kwargs[1]["pr_sha"] == "abc123" or call_kwargs.kwargs.get("pr_sha") == "abc123"

    async def test_sha_extracted_iterates_files_until_found(self):
        """Should iterate pr_files and use the first file that has a sha attribute."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("quality", True, 88)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        trigger_mock = AsyncMock(return_value={"workflow_id": "wf-iter"})
        orch.workflow_trigger.trigger_pr_comprehensive_validation = trigger_mock

        # First file has no sha, second has sha
        file_no_sha = SimpleNamespace(
            filename="a.py", patch="+a", additions=1, deletions=0, changes=1
        )
        file_with_sha = _make_pr_file(filename="b.py")
        file_with_sha.sha = "sha-from-b"

        result = await orch.validate_pr_with_actions(
            "owner", "repo", 10, [file_no_sha, file_with_sha], pr_metadata={}
        )
        assert result["comprehensive_validation"]["status"] == "triggered"

    # --- Lines 135-136: Fallback SHA when neither metadata nor files provide one ---

    async def test_sha_fallback_when_no_sha_available(self):
        """When no SHA is available from metadata or files, a fallback should be used."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("security", True, 95)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        trigger_mock = AsyncMock(return_value={"workflow_id": "wf-fallback"})
        orch.workflow_trigger.trigger_pr_comprehensive_validation = trigger_mock

        # No pr_metadata, empty pr_files
        result = await orch.validate_pr_with_actions("owner", "repo", 7, [], pr_metadata=None)

        assert result["comprehensive_validation"]["status"] == "triggered"
        call_kwargs = trigger_mock.call_args
        # Fallback sha should be "pr-7-head"
        pr_sha = call_kwargs.kwargs.get("pr_sha") or call_kwargs[1].get("pr_sha")
        assert pr_sha == "pr-7-head"

    async def test_sha_fallback_with_no_metadata_no_files(self):
        """Fallback SHA is 'pr-<number>-head' when metadata is None and files list is empty."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("quality", True, 100)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        trigger_mock = AsyncMock(return_value={"workflow_id": "wf-fb2"})
        orch.workflow_trigger.trigger_pr_comprehensive_validation = trigger_mock

        await orch.validate_pr_with_actions("o", "r", 99, [])
        pr_sha = trigger_mock.call_args.kwargs.get("pr_sha") or trigger_mock.call_args[1].get(
            "pr_sha"
        )
        assert pr_sha == "pr-99-head"

    # --- Lines 202-205: Fast validation failed branch ---

    async def test_fast_validation_failed_returns_not_triggered(self):
        """When fast validation fails, comprehensive validation should not be triggered."""
        orch = self._get_orchestrator()
        fast_results = [
            _make_validation_result("security", False, 20),
            _make_validation_result("quality", True, 80),
        ]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)
        trigger_mock = AsyncMock()
        orch.workflow_trigger.trigger_pr_comprehensive_validation = trigger_mock

        result = await orch.validate_pr_with_actions("owner", "repo", 5, [_make_pr_file()])

        assert result["status"] == "fast_validation_failed"
        assert result["comprehensive_validation"]["status"] == "not_triggered"
        assert result["comprehensive_validation"]["reason"] == "fast_validation_failed"
        trigger_mock.assert_not_called()

    # --- Line 242: get_validation_status ---

    async def test_comprehensive_not_requested_returns_not_requested(self):
        """When trigger_comprehensive=False, status should be 'not_requested'."""
        orch = self._get_orchestrator()
        fast_results = [_make_validation_result("security", True, 95)]
        orch.fast_validator.validate_github_pr = AsyncMock(return_value=fast_results)

        result = await orch.validate_pr_with_actions(
            "owner", "repo", 3, [_make_pr_file()], trigger_comprehensive=False
        )

        assert result["status"] == "not_requested"
        assert result["comprehensive_validation"]["reason"] == "not_requested"

    def test_get_validation_status_returns_none_for_unknown(self):
        """get_validation_status should return None for unknown validation IDs."""
        orch = self._get_orchestrator()
        assert orch.get_validation_status("nonexistent-id") is None

    def test_get_validation_status_returns_stored_validation(self):
        """get_validation_status should return stored validation data."""
        orch = self._get_orchestrator()
        orch.validations["test-id"] = {"pr_number": 1, "status": "completed"}
        result = orch.get_validation_status("test-id")
        assert result is not None
        assert result["pr_number"] == 1

    # --- Lines 259-274: update_comprehensive_results ---

    async def test_update_comprehensive_results_not_found(self):
        """Updating results for a non-existent validation_id should return False."""
        orch = self._get_orchestrator()
        result = await orch.update_comprehensive_results("nonexistent", {"score": 95})
        assert result is False

    async def test_update_comprehensive_results_success(self):
        """Updating results for an existing validation_id should return True."""
        orch = self._get_orchestrator()
        orch.validations["val-1"] = {
            "pr_number": 1,
            "comprehensive_validation": {"status": "triggered"},
        }
        result = await orch.update_comprehensive_results("val-1", {"score": 95, "passed": True})
        assert result is True
        assert orch.validations["val-1"]["comprehensive_validation"]["status"] == "completed"
        assert orch.validations["val-1"]["comprehensive_validation"]["results"]["score"] == 95

    async def test_update_comprehensive_results_exception_returns_false(self):
        """If an exception occurs during update, should return False."""
        orch = self._get_orchestrator()
        # Make validations dict raise on __setitem__ for the comprehensive_validation key
        orch.validations["val-err"] = MagicMockThatRaises()
        result = await orch.update_comprehensive_results("val-err", {"score": 50})
        assert result is False

    # --- Lines 289-291: get_pr_validation_orchestrator singleton ---

    def test_get_pr_validation_orchestrator_returns_instance(self):
        """get_pr_validation_orchestrator should return a PRValidationOrchestrator."""
        # Reset singleton
        import services.pr_validation.pr_validation_orchestrator as orch_module
        from services.pr_validation.pr_validation_orchestrator import (
            PRValidationOrchestrator,
            get_pr_validation_orchestrator,
        )

        orch_module._orchestrator = None

        instance = get_pr_validation_orchestrator()
        assert isinstance(instance, PRValidationOrchestrator)

    def test_get_pr_validation_orchestrator_returns_same_instance(self):
        """get_pr_validation_orchestrator should return the same singleton instance."""
        import services.pr_validation.pr_validation_orchestrator as orch_module
        from services.pr_validation.pr_validation_orchestrator import (
            get_pr_validation_orchestrator,
        )

        orch_module._orchestrator = None

        instance1 = get_pr_validation_orchestrator()
        instance2 = get_pr_validation_orchestrator()
        assert instance1 is instance2


class MagicMockThatRaises(dict):
    """A dict subclass that raises on item assignment to simulate errors."""

    def __setitem__(self, key, value):
        raise RuntimeError("Simulated storage error")
