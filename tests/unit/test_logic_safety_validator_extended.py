"""Edge-case and boundary tests for LogicSafetyValidator."""

from __future__ import annotations

import ast
from unittest.mock import MagicMock, patch

import pytest
from coding_client.models.pr_validation import ChangeType, FileChange, FileType

from services.pr_validation.analyzers.semantic.logic_safety_validator import (
    LogicSafetyValidator,
)


def _fc(**kwargs):
    """Shortcut to build a FileChange with sensible defaults."""
    defaults = dict(
        filename="src/app.py",
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=1,
        deletions=0,
        patch="",
    )
    defaults.update(kwargs)
    return FileChange(**defaults)


@pytest.fixture
def validator():
    return LogicSafetyValidator()


# ---------------------------------------------------------------------------
# validate_logic_safety — exception handler (lines 191-193)
# ---------------------------------------------------------------------------


class TestValidateLogicSafetyException:
    """When internal analysis raises, a validation_error issue is produced."""

    @pytest.mark.asyncio
    async def test_exception_produces_validation_error_issue(self, validator):
        fc = _fc(filename="app.py", file_type=FileType.PYTHON, patch="+ x = 1")
        with patch.object(
            validator,
            "_analyze_patch_safety",
            side_effect=RuntimeError("crash"),
        ):
            issues = await validator.validate_logic_safety(fc)
        error_issues = [i for i in issues if i.type == "validation_error"]
        assert len(error_issues) == 1
        assert "Logic safety validation failed" in error_issues[0].message
        assert error_issues[0].severity == "medium"


# ---------------------------------------------------------------------------
# _assess_inversion_severity — boundary cases
# ---------------------------------------------------------------------------


class TestAssessInversionSeverityBoundary:
    """Edge cases in severity assessment."""

    @pytest.mark.parametrize(
        "old_pattern,expected",
        [
            (r">=", "medium"),
            (r"<=", "medium"),
            (r"\bFalse\b", "medium"),
            (r"\btrue\b", "medium"),
            (r"\bfalse\b", "medium"),
        ],
    )
    def test_comparison_and_boolean_patterns(self, validator, old_pattern, expected):
        result = validator._assess_inversion_severity(old_pattern, "", "value = 42")
        assert result == expected

    def test_negation_removal_returns_medium(self, validator):
        result = validator._assess_inversion_severity(r"\bnot\s+", r"", "x = compute()")
        assert result == "medium"

    def test_equality_in_if_context_returns_high(self, validator):
        # The source checks for "i" in context_lower and == pattern
        result = validator._assess_inversion_severity(r"==", r"!=", "if status == ready:")
        assert result == "high"


# ---------------------------------------------------------------------------
# _analyze_ast_safety — None AST (line 394)
# ---------------------------------------------------------------------------


class TestAnalyzeAstSafetyNullAst:
    """Return empty when AST parsing fails."""

    @pytest.mark.asyncio
    async def test_before_ast_none_returns_empty(self, validator):
        fc = _fc(
            filename="a.py",
            file_type=FileType.PYTHON,
            before_content="invalid {{{{",
            after_content="x = 1",
        )
        with patch.object(
            validator.parser,
            "parse_with_fallback",
            side_effect=[(None, None), (MagicMock(), None)],
        ):
            issues = await validator._analyze_ast_safety(fc)
        assert issues == []

    @pytest.mark.asyncio
    async def test_after_ast_none_returns_empty(self, validator):
        fc = _fc(
            filename="a.py",
            file_type=FileType.PYTHON,
            before_content="x = 1",
            after_content="invalid {{{{",
        )
        with patch.object(
            validator.parser,
            "parse_with_fallback",
            side_effect=[(MagicMock(), None), (None, None)],
        ):
            issues = await validator._analyze_ast_safety(fc)
        assert issues == []


# ---------------------------------------------------------------------------
# _analyze_ast_safety — exception handler (lines 408-409)
# ---------------------------------------------------------------------------


class TestAnalyzeAstSafetyException:
    """AST parse crash is caught gracefully."""

    @pytest.mark.asyncio
    async def test_exception_returns_list(self, validator):
        fc = _fc(
            filename="a.py",
            file_type=FileType.PYTHON,
            before_content="x = 1",
            after_content="y = 2",
        )
        with patch.object(
            validator.parser,
            "parse_with_fallback",
            side_effect=RuntimeError("AST crash"),
        ):
            issues = await validator._analyze_ast_safety(fc)
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# _analyze_exception_handling_changes — catch-all detection (line 466)
# ---------------------------------------------------------------------------


class TestCatchAllExceptionDetection:
    """Bare except triggers catch_all_exception issue."""

    def test_bare_except_detected(self, validator):
        before_ast = ast.parse("x = 1")
        after_ast = ast.parse("try:\n    x = 1\nexcept:\n    pass\n")
        issues = validator._analyze_exception_handling_changes(before_ast, after_ast, "test.py")
        catch_all = [i for i in issues if i.type == "catch_all_exception"]
        assert len(catch_all) >= 1
        assert catch_all[0].severity == "medium"

    def test_specific_except_not_flagged_as_catch_all(self, validator):
        before_ast = ast.parse("x = 1")
        after_ast = ast.parse("try:\n    x = 1\nexcept ValueError:\n    pass\n")
        issues = validator._analyze_exception_handling_changes(before_ast, after_ast, "test.py")
        catch_all = [i for i in issues if i.type == "catch_all_exception"]
        assert len(catch_all) == 0


# ---------------------------------------------------------------------------
# _analyze_security_changes / _analyze_error_handling_changes — no patch
# ---------------------------------------------------------------------------


class TestNoPatchReturnsEmpty:
    """Empty patch produces no issues from these analyzers."""

    def test_security_changes_empty_patch(self, validator):
        fc = _fc(filename="a.py", file_type=FileType.PYTHON, patch="")
        assert validator._analyze_security_changes(fc) == []

    def test_error_handling_changes_empty_patch(self, validator):
        fc = _fc(filename="a.py", file_type=FileType.PYTHON, patch="")
        assert validator._analyze_error_handling_changes(fc) == []


# ---------------------------------------------------------------------------
# _detect_logic_inversions — similarity threshold
# ---------------------------------------------------------------------------


class TestSimilarityThreshold:
    """Lines too different (similarity < 0.6) produce no violations."""

    def test_completely_different_lines_no_violation(self, validator):
        violations = validator._detect_logic_inversions(
            "completely different line one here",
            "another totally unrelated line there",
            1,
            1,
        )
        assert violations == []

    def test_very_similar_lines_produce_violation(self, validator):
        violations = validator._detect_logic_inversions(
            "    if x and y:",
            "    if x or y:",
            5,
            5,
        )
        assert len(violations) >= 1
        assert violations[0].violation_type == "LOGIC_INVERSION"


# ---------------------------------------------------------------------------
# _check_dangerous_patterns — combined security + error patterns
# ---------------------------------------------------------------------------


class TestCheckDangerousPatterns:
    """Security and error handling patterns in added/removed lines."""

    def test_security_pattern_in_removed_line(self, validator):
        removed = [(10, "def authenticate(user):")]
        added = [(10, "def process(user):")]
        issues = validator._check_dangerous_patterns(added, removed, "auth.py")
        sec = [i for i in issues if i.type == "security_sensitive_change"]
        assert len(sec) >= 1

    def test_error_pattern_in_added_line(self, validator):
        removed = []
        added = [(5, "    try:")]
        issues = validator._check_dangerous_patterns(added, removed, "handler.py")
        err = [i for i in issues if i.type == "error_handling_change"]
        assert len(err) >= 1


# ---------------------------------------------------------------------------
# _extract_control_flow_nodes / _extract_exception_nodes
# ---------------------------------------------------------------------------


class TestASTNodeExtraction:
    """Extract control flow and exception nodes from real ASTs."""

    def test_control_flow_nodes(self, validator):
        code = "if True:\n    for x in [1]:\n        while False:\n            break\n        continue\n    return 0\n"
        tree = ast.parse(code)
        nodes = validator._extract_control_flow_nodes(tree)
        types = {n["type"] for n in nodes}
        assert "If" in types
        assert "For" in types
        assert "While" in types
        assert "Break" in types
        assert "Continue" in types
        assert "Return" in types

    def test_exception_nodes(self, validator):
        code = "try:\n    raise ValueError('bad')\nexcept ValueError:\n    pass\n"
        tree = ast.parse(code)
        nodes = validator._extract_exception_nodes(tree)
        types = {n["type"] for n in nodes}
        assert "ExceptHandler" in types
        assert "Raise" in types
        # Check that the except handler is not a catch-all
        handler = next(n for n in nodes if n["type"] == "ExceptHandler")
        assert handler["is_catch_all"] is False
        assert handler["exception_type"] == "ValueError"


# ---------------------------------------------------------------------------
# _violation_to_issue
# ---------------------------------------------------------------------------


class TestViolationToIssue:
    """Convert SafetyViolation to ValidationIssue."""

    def test_conversion(self, validator):
        from services.pr_validation.analyzers.semantic.logic_safety_validator import (
            SafetyViolation,
        )

        violation = SafetyViolation(
            violation_type="LOGIC_INVERSION",
            severity="high",
            line_number=42,
            before_code="if x and y:",
            after_code="if x or y:",
            explanation="AND changed to OR",
            recommendation="Check this change",
            confidence=0.9,
        )
        issue = validator._violation_to_issue(violation, "app.py")
        assert issue.type == "logic_inversion"
        assert issue.severity == "high"
        assert issue.file == "app.py"
        assert issue.line == 42
        assert issue.rule == "logic_safety_validator"
