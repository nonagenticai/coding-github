"""Extended tests for utils/github_validation_helpers.py – targeting uncovered lines.

Uncovered lines:
  19-32    – validate_and_get_container_for_tenant (happy path, 404, other error, RuntimeError)
  38       – basic_security_check when file.patch is None
  69-80    – basic_security_check detecting hardcoded credentials
  98-121   – validate_pr_security PythonCodeProcessor path (import + analyze)
  127-128  – validate_pr_security AttributeError/TypeError/ValueError handling
  157      – validate_pr_performance skip file with no patch
  201      – validate_pr_quality skip file with no patch
  223      – validate_pr_quality commented-code detection
  243      – validate_pr_quality no suggestions → default suggestion
  259      – validate_readme_changes with no patch
  288-290  – validate_readme_changes unclosed code block detection
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from schemas.github_api_models import AnalysisIssue, PRValidationResult
from utils.github_validation_helpers import (
    basic_security_check,
    validate_and_get_container_for_tenant,
    validate_pr_performance,
    validate_pr_quality,
    validate_pr_security,
    validate_readme_changes,
)


def _make_file(filename="app.py", patch="+x = 1"):
    return SimpleNamespace(filename=filename, patch=patch)


# ---------------------------------------------------------------------------
# validate_and_get_container_for_tenant  (lines 19-32)
# ---------------------------------------------------------------------------
class TestValidateAndGetContainerForTenant:
    """Cover the three branches in validate_and_get_container_for_tenant."""

    async def test_happy_path_returns_container(self):
        """Lines 24-26: Successful container retrieval."""
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(return_value={"name": "my-ctr"})
        with patch("main.get_workspaces_client", return_value=mock_client):
            result = await validate_and_get_container_for_tenant("my-ctr", "tenant-1")
        assert result == {"name": "my-ctr"}
        assert mock_client.get_workspace.await_args.args[0] == "my-ctr"
        headers = mock_client.get_workspace.await_args.kwargs["headers"]
        assert {key.lower(): value for key, value in headers.items()}["x-tenant-id"] == "tenant-1"

    async def test_404_raises_value_error(self):
        """Lines 28-29: ServiceError with status_code 404 → ValueError."""
        from coding_client.clients import ServiceError

        class ServiceError404(ServiceError):  # noqa: N818
            def __init__(self):
                super().__init__(404, "not found")

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=ServiceError404())
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(ValueError, match="not found"):
                await validate_and_get_container_for_tenant("missing-ctr", "tenant-1")

    async def test_other_service_error_raises_runtime_error(self):
        """Line 30: ServiceError with non-404 status → RuntimeError."""
        from coding_client.clients import ServiceError

        class ServiceError500(ServiceError):  # noqa: N818
            def __init__(self):
                super().__init__(500, "internal")

        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=ServiceError500())
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Containers service error"):
                await validate_and_get_container_for_tenant("bad-ctr", "tenant-1")

    async def test_runtime_error_reraised(self):
        """Lines 31-32: RuntimeError is re-raised directly."""
        mock_client = AsyncMock()
        mock_client.get_workspace = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("main.get_workspaces_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="boom"):
                await validate_and_get_container_for_tenant("ctr", "tenant-1")


# ---------------------------------------------------------------------------
# basic_security_check  (lines 38, 69-80)
# ---------------------------------------------------------------------------
class TestBasicSecurityCheck:
    """Cover basic_security_check edge cases."""

    def test_none_patch_returns_early(self):
        """Line 38: file.patch is None → early return, no issues."""
        file = _make_file(patch=None)
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert issues == []
        assert suggestions == []

    def test_detects_hardcoded_password(self):
        """Lines 69-80: Hardcoded password= triggers critical issue."""
        file = _make_file(patch='+db_conn = connect(password="secret123")')
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 1
        assert any(i.severity == "critical" for i in issues)
        assert any("credentials" in i.message.lower() for i in issues)
        assert any("environment variables" in s.lower() for s in suggestions)

    def test_detects_hardcoded_api_key(self):
        """Lines 69-80: api_key= triggers critical issue."""
        file = _make_file(patch='+client = Client(api_key="AKIAIOSFODNN7EXAMPLE")')
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 1
        assert any(i.rule == "pr-security-credentials" for i in issues)

    def test_detects_hardcoded_secret(self):
        """Lines 69-80: secret= triggers critical issue."""
        file = _make_file(patch='+config = {"secret=mysecretvalue"}')
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 1

    def test_detects_hardcoded_token(self):
        """Lines 69-80: token= triggers critical issue."""
        file = _make_file(patch='+headers = {"Authorization": token="abc123"}')
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 1

    def test_detects_eval_and_exec(self):
        """Lines 46-62: eval() and exec() trigger security issues."""
        file = _make_file(patch="+result = eval(user_input)\n+exec(code)")
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 2
        assert all(i.severity == "high" for i in issues)
        assert all(i.rule == "pr-security-injection" for i in issues)

    def test_non_added_lines_ignored(self):
        """Lines starting with '-' or context lines should not trigger issues."""
        file = _make_file(patch="-result = eval(user_input)\n password=secret")
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        basic_security_check(file, issues, suggestions)
        assert issues == []


# ---------------------------------------------------------------------------
# validate_pr_security  (lines 98-121, 126-128)
# ---------------------------------------------------------------------------
class TestValidatePrSecurityExtended:
    """Cover analyzer import path and error-handling branches."""

    async def test_security_check_needs_no_foreign_module(self):
        """D16 class: ``validate_pr_security`` used to try
        ``from processing.analyzers.python_analyzer import ...`` — a module that only
        exists in another service, so it resolved only under a different sys.path and
        raised ImportError on every file in prod. The check is now the basic one,
        directly, and must not import anything named ``processing``."""
        before = {k for k in sys.modules if k == "processing" or k.startswith("processing.")}
        files = [_make_file(patch="+result = eval(user_input)")]
        result = await validate_pr_security(files)
        assert isinstance(result, PRValidationResult)
        assert any(i.rule == "pr-security-injection" for i in result.issues)
        after = {k for k in sys.modules if k == "processing" or k.startswith("processing.")}
        assert after == before

    async def test_no_patch_file_skipped(self):
        """File with no patch should be skipped in security validation."""
        files = [_make_file(patch=None)]
        result = await validate_pr_security(files)
        assert isinstance(result, PRValidationResult)
        assert result.passed is True

    async def test_score_degrades_with_critical_issues(self):
        """Score calculation: critical issues reduce score by 30 each."""
        # Two files each with hardcoded credentials (critical severity)
        files = [
            _make_file(patch='+db = connect(password="x")'),
            _make_file(filename="b.py", patch='+api = connect(password="y")'),
        ]
        result = await validate_pr_security(files)
        critical_count = len([i for i in result.issues if i.severity == "critical"])
        assert critical_count >= 2
        assert result.passed is False

    async def test_default_suggestion_when_none(self):
        """Lines 138-139: Default suggestion when none added."""
        files = [_make_file(patch="+x = 1")]
        result = await validate_pr_security(files)
        assert "Consider adding security-focused unit tests" in result.suggestions


# ---------------------------------------------------------------------------
# validate_pr_performance  (line 157)
# ---------------------------------------------------------------------------
class TestValidatePrPerformanceExtended:
    """Cover performance validation edge cases."""

    async def test_file_with_no_patch_skipped(self):
        """Line 157: File with no patch should be skipped."""
        files = [_make_file(patch=None), _make_file(patch="+x = 1")]
        result = await validate_pr_performance(files)
        assert isinstance(result, PRValidationResult)
        assert result.validation_type == "performance"
        assert result.passed is True

    async def test_sleep_detection(self):
        """sleep() in added lines triggers warning."""
        files = [_make_file(patch="+time.sleep(5)\n+sleep(1)")]
        result = await validate_pr_performance(files)
        assert len(result.issues) >= 1
        assert all(i.severity == "warning" for i in result.issues)

    async def test_many_sleeps_lower_score(self):
        """Multiple sleep warnings degrade score."""
        lines = "\n".join(f"+time.sleep({i})" for i in range(5))
        files = [_make_file(patch=lines)]
        result = await validate_pr_performance(files)
        assert result.score <= 60
        assert result.passed is False

    async def test_clean_files_pass(self):
        """Files without sleep should pass with score 100."""
        files = [_make_file(patch="+x = 1\n+y = 2")]
        result = await validate_pr_performance(files)
        assert result.score == 100
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_pr_quality  (lines 201, 223, 243)
# ---------------------------------------------------------------------------
class TestValidatePrQualityExtended:
    """Cover quality validation edge cases."""

    async def test_file_with_no_patch_skipped(self):
        """Line 201: File with no patch should be skipped."""
        files = [_make_file(patch=None)]
        result = await validate_pr_quality(files)
        assert isinstance(result, PRValidationResult)
        assert result.validation_type == "quality"
        assert result.passed is True

    async def test_fixme_detected(self):
        """FIXME in added lines triggers quality issue."""
        files = [_make_file(patch="+# FIXME: needs refactoring")]
        result = await validate_pr_quality(files)
        assert len(result.issues) >= 1
        assert any(
            "unfinished" in i.message.lower()
            or "todo" in i.message.lower()
            or "fixme" in i.message.lower()
            for i in result.issues
        )

    async def test_commented_code_detected(self):
        """Line 222-233: Commented-out code triggers info-level issue.

        The check is: line.strip().startswith('+ #') and len(line.strip()) > 10
        The line from the patch starts with '+', so line.strip() is e.g. '+ # some long commented code'
        """
        # The patch line is "+    # some old code that was left behind"
        # line.strip() => "+ # some old code that was left behind" → starts with "+ #", len > 10
        # Wait, let me re-read the code...
        # line = "+    # some old code that was left behind"
        # line.startswith("+") is True (first check at line 206)
        # then line 222: line.strip().startswith("+ #") and len(line.strip()) > 10
        # line.strip() = "+    # some old code that was left behind".strip() = "+ # some old code..."
        # Hmm, that doesn't start with "+ #" because after strip the leading spaces are gone
        # but the '+' and spaces after it... Let me think again.
        # Actually: line = "+    # some old code" → line.strip() = "# some old code"... no.
        # line.strip() removes leading/trailing whitespace from the whole line.
        # line = "+    # some old code" → strip() = "+    # some old code"... no, strip removes
        # leading spaces too. So "+    # some old code".strip() = "+    # some old code"
        # Wait no. "  +    # some old code".strip() = "+    # some old code"
        # But our lines don't have leading spaces (they're patch lines starting with +).
        # So line = "+    # some old code" → line.strip() = "+    # some old code"
        # That does NOT start with "+ #" because there are spaces between + and #.
        #
        # Actually re-reading line 222: `line.strip().startswith("+ #")`
        # For this to match, after stripping, the line must start with "+ #"
        # So the original line (from patch) must be something like:
        # "+ # commented code here that is long enough"
        # where line.strip() = "+ # commented code here that is long enough"
        # That starts with "+ #" ✓ and len > 10 ✓
        files = [_make_file(patch="+ # this is some old commented out code that should be removed")]
        result = await validate_pr_quality(files)
        commented_issues = [i for i in result.issues if i.rule == "pr-quality-commented-code"]
        assert len(commented_issues) >= 1
        assert commented_issues[0].severity == "info"

    async def test_no_issues_gets_default_suggestion(self):
        """Line 242-243: When no suggestions are added, default is used."""
        files = [_make_file(patch="+clean_code = 42")]
        result = await validate_pr_quality(files)
        assert "Consider adding unit tests for new functionality" in result.suggestions

    async def test_todo_adds_specific_suggestion(self):
        """TODO should add file-specific suggestion."""
        files = [_make_file(filename="handler.py", patch="+# TODO: implement caching")]
        result = await validate_pr_quality(files)
        assert any("handler.py" in s for s in result.suggestions)

    async def test_many_warnings_lower_score(self):
        """Multiple TODO/FIXME warnings degrade score."""
        lines = "\n".join(f"+# TODO: fix item {i}" for i in range(5))
        files = [_make_file(patch=lines)]
        result = await validate_pr_quality(files)
        assert result.score <= 60

    async def test_info_issues_degrade_score_slightly(self):
        """Info-level issues (commented code) degrade score by 2 each."""
        lines = "\n".join(f"+ # old code line number {i} that was commented out" for i in range(10))
        files = [_make_file(patch=lines)]
        result = await validate_pr_quality(files)
        assert result.score < 100


# ---------------------------------------------------------------------------
# validate_readme_changes  (lines 259, 288-290)
# ---------------------------------------------------------------------------
class TestValidateReadmeChangesExtended:
    """Cover README validation edge cases."""

    async def test_no_patch_returns_empty(self):
        """Line 258-259: File with no patch returns empty issues list."""
        file = _make_file(filename="README.md", patch=None)
        issues = await validate_readme_changes(file)
        assert issues == []

    async def test_unclosed_code_block_detected(self):
        """Lines 288-290: Single ``` on an added line triggers unclosed code block issue."""
        file = _make_file(
            filename="README.md",
            patch="+```python\n+print('hello')\n+some more text",
        )
        issues = await validate_readme_changes(file)
        # The first added line "+```python" → after removing '+': "```python"
        # "```" appears once → backtick_count=1, 1%2!=0 → issue
        code_block_issues = [i for i in issues if i.rule == "pr-doc-markdown-syntax"]
        assert len(code_block_issues) >= 1
        assert any("code block" in i.message.lower() for i in code_block_issues)

    async def test_closed_code_block_no_issue(self):
        """Properly closed code blocks (even count of ```) should not trigger."""
        file = _make_file(
            filename="README.md",
            patch="+```python```\n+normal text",
        )
        issues = await validate_readme_changes(file)
        code_block_issues = [i for i in issues if i.rule == "pr-doc-markdown-syntax"]
        assert len(code_block_issues) == 0

    async def test_removed_usage_section_flagged(self):
        """Removing ## Usage section should be flagged."""
        file = _make_file(
            filename="README.md",
            patch="-## Usage\n-Use the tool\n+New content",
        )
        issues = await validate_readme_changes(file)
        assert any("Usage" in i.message for i in issues)

    async def test_removed_license_section_flagged(self):
        """Removing ## License section should be flagged."""
        file = _make_file(
            filename="README.md",
            patch="-## License\n-MIT\n+New content",
        )
        issues = await validate_readme_changes(file)
        assert any("License" in i.message for i in issues)

    async def test_replaced_section_not_flagged(self):
        """Removing and re-adding ## Installation should NOT flag."""
        file = _make_file(
            filename="README.md",
            patch="-## Installation\n-Old steps\n+## Installation\n+New steps",
        )
        issues = await validate_readme_changes(file)
        assert not any("Installation" in i.message for i in issues)

    async def test_no_issues_for_clean_readme(self):
        """Clean README changes should produce no issues."""
        file = _make_file(
            filename="README.md",
            patch="+## New Section\n+Some great docs",
        )
        issues = await validate_readme_changes(file)
        assert issues == []
