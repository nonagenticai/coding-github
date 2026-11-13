"""Core tests for LogicSafetyValidator — real patches, real detection."""

from __future__ import annotations

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
# _is_validatable_file
# ---------------------------------------------------------------------------


class TestIsValidatableFile:
    """Only Python/JS/TS non-binary files with patches are validatable."""

    @pytest.mark.parametrize(
        "file_type,expected",
        [
            (FileType.PYTHON, True),
            (FileType.JAVASCRIPT, True),
            (FileType.TYPESCRIPT, True),
            (FileType.YAML, False),
            (FileType.JSON, False),
            (FileType.MARKDOWN, False),
            (FileType.UNKNOWN, False),
        ],
    )
    def test_file_type_filtering(self, validator, file_type, expected):
        fc = _fc(filename="f.py", file_type=file_type, patch="+ x = 1")
        assert validator._is_validatable_file(fc) == expected

    def test_binary_file_rejected(self, validator):
        fc = _fc(filename="f.py", file_type=FileType.PYTHON, patch="+x", binary=True)
        assert validator._is_validatable_file(fc) is False

    def test_no_patch_no_content_rejected(self, validator):
        fc = _fc(filename="f.py", file_type=FileType.PYTHON, patch="")
        assert validator._is_validatable_file(fc) is False

    def test_content_without_patch_accepted(self, validator):
        fc = _fc(
            filename="f.py",
            file_type=FileType.PYTHON,
            patch="",
            before_content="x = 1",
            after_content="x = 2",
        )
        assert validator._is_validatable_file(fc) is True


# ---------------------------------------------------------------------------
# Logic inversion detection — _detect_logic_inversions
# ---------------------------------------------------------------------------


class TestLogicInversionDetection:
    """Detect dangerous logic inversions between removed/added lines."""

    @pytest.mark.asyncio
    async def test_and_to_or_inversion(self, validator):
        patch = (
            "@@ -1,3 +1,3 @@\n"
            " def check(a, b):\n"
            "-    if a and b:\n"
            "+    if a or b:\n"
            "         return True\n"
        )
        fc = _fc(filename="logic.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        inversion_issues = [i for i in issues if i.type == "logic_inversion"]
        assert len(inversion_issues) >= 1
        assert any("AND" in i.message or "OR" in i.message for i in inversion_issues)

    @pytest.mark.asyncio
    async def test_equality_to_inequality_inversion(self, validator):
        patch = (
            "@@ -5,3 +5,3 @@\n"
            " def validate(status):\n"
            "-    if status == 'active':\n"
            "+    if status != 'active':\n"
            "         return True\n"
        )
        fc = _fc(filename="validate.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        inversion_issues = [i for i in issues if i.type == "logic_inversion"]
        assert len(inversion_issues) >= 1

    @pytest.mark.asyncio
    async def test_true_to_false_inversion(self, validator):
        patch = "@@ -1,3 +1,3 @@\n def get_flag():\n-    return True\n+    return False\n"
        fc = _fc(filename="flags.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        inversion_issues = [i for i in issues if i.type == "logic_inversion"]
        assert len(inversion_issues) >= 1

    @pytest.mark.asyncio
    async def test_no_inversion_on_unrelated_changes(self, validator):
        patch = "@@ -1,3 +1,3 @@\n def greet():\n-    print('hello')\n+    print('goodbye')\n"
        fc = _fc(filename="greet.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        inversion_issues = [i for i in issues if i.type == "logic_inversion"]
        assert len(inversion_issues) == 0


# ---------------------------------------------------------------------------
# Security-sensitive changes — _analyze_security_changes
# ---------------------------------------------------------------------------


class TestSecurityChanges:
    """Detect modifications to security-sensitive code."""

    @pytest.mark.asyncio
    async def test_security_keyword_in_added_line(self, validator):
        patch = "@@ -1 +1,2 @@\n old\n+def authenticate(user):\n"
        fc = _fc(filename="auth.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        sec_issues = [i for i in issues if i.type == "security_code_change"]
        assert len(sec_issues) >= 1

    @pytest.mark.asyncio
    async def test_security_keyword_in_removed_line(self, validator):
        patch = "@@ -1,2 +1 @@\n-def validate_token(tok):\n remaining\n"
        fc = _fc(filename="auth.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        sec_issues = [i for i in issues if i.type == "security_code_change"]
        assert len(sec_issues) >= 1
        # Removed security code should be high severity
        assert any(i.severity == "high" for i in sec_issues)

    @pytest.mark.asyncio
    async def test_no_security_issues_in_normal_code(self, validator):
        patch = "@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        fc = _fc(filename="math.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        sec_issues = [i for i in issues if i.type == "security_code_change"]
        assert len(sec_issues) == 0


# ---------------------------------------------------------------------------
# Error handling changes — _analyze_error_handling_changes
# ---------------------------------------------------------------------------


class TestErrorHandlingChanges:
    """Detect modifications to error handling code."""

    @pytest.mark.asyncio
    async def test_try_except_added(self, validator):
        patch = "@@ -1 +1,4 @@\n old\n+try:\n+    do_thing()\n+except ValueError:\n+    handle()\n"
        fc = _fc(filename="handler.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        err_issues = [i for i in issues if i.type == "error_handling_change"]
        assert len(err_issues) >= 1

    @pytest.mark.asyncio
    async def test_try_except_removed(self, validator):
        patch = "@@ -1,4 +1 @@\n-try:\n-    do_thing()\n-except ValueError:\n-    handle()\n+do_thing()\n"
        fc = _fc(filename="handler.py", file_type=FileType.PYTHON, patch=patch)
        issues = await validator.validate_logic_safety(fc)
        err_issues = [i for i in issues if i.type == "error_handling_change"]
        assert len(err_issues) >= 1


# ---------------------------------------------------------------------------
# _assess_inversion_severity
# ---------------------------------------------------------------------------


class TestAssessInversionSeverity:
    """Severity depends on context (security keywords, operator type)."""

    def test_security_context_returns_high(self, validator):
        result = validator._assess_inversion_severity(
            r"==", r"!=", "if authenticate(user) == True:"
        )
        assert result == "high"

    def test_boolean_operator_returns_high(self, validator):
        result = validator._assess_inversion_severity(r"\band\b", r"\bor\b", "x = compute()")
        assert result == "high"

    def test_comparison_operator_returns_medium(self, validator):
        result = validator._assess_inversion_severity(r"<", r">", "if count < limit:")
        assert result == "medium"

    def test_boolean_literal_returns_medium(self, validator):
        result = validator._assess_inversion_severity(r"\bTrue\b", r"\bFalse\b", "return value")
        assert result == "medium"


# ---------------------------------------------------------------------------
# AST-based analysis — control flow and exception handling
# ---------------------------------------------------------------------------


class TestASTAnalysis:
    """AST-based detection of control flow and exception changes."""

    @pytest.mark.asyncio
    async def test_control_flow_change_detected(self, validator):
        before = "def f():\n    return 1\n"
        after = "def f():\n    if True:\n        return 1\n    return 0\n"
        fc = _fc(
            filename="flow.py",
            file_type=FileType.PYTHON,
            patch="+if True:",
            before_content=before,
            after_content=after,
        )
        issues = await validator.validate_logic_safety(fc)
        flow_issues = [i for i in issues if i.type == "control_flow_change"]
        assert len(flow_issues) >= 1

    @pytest.mark.asyncio
    async def test_exception_handler_removed(self, validator):
        before = "def f():\n    try:\n        x()\n    except ValueError:\n        pass\n"
        after = "def f():\n    x()\n"
        fc = _fc(
            filename="exc.py",
            file_type=FileType.PYTHON,
            patch="-try:\n-    x()\n-except ValueError:\n-    pass\n+x()",
            before_content=before,
            after_content=after,
        )
        issues = await validator.validate_logic_safety(fc)
        exc_issues = [i for i in issues if i.type == "exception_handling_removed"]
        assert len(exc_issues) >= 1
        assert exc_issues[0].severity == "high"

    @pytest.mark.asyncio
    async def test_bare_except_detected(self, validator):
        before = "def f():\n    pass\n"
        after = "def f():\n    try:\n        x()\n    except:\n        pass\n"
        fc = _fc(
            filename="bare.py",
            file_type=FileType.PYTHON,
            patch="+try:\n+    x()\n+except:\n+    pass",
            before_content=before,
            after_content=after,
        )
        issues = await validator.validate_logic_safety(fc)
        catch_all = [i for i in issues if i.type == "catch_all_exception"]
        assert len(catch_all) >= 1


# ---------------------------------------------------------------------------
# validate_logic_safety — full pipeline
# ---------------------------------------------------------------------------


class TestValidateLogicSafetyPipeline:
    """End-to-end validate_logic_safety with realistic patches."""

    @pytest.mark.asyncio
    async def test_non_validatable_file_returns_empty(self, validator):
        fc = _fc(filename="readme.md", file_type=FileType.MARKDOWN, patch="+ stuff")
        issues = await validator.validate_logic_safety(fc)
        assert issues == []

    @pytest.mark.asyncio
    async def test_complex_patch_with_multiple_issue_types(self, validator):
        patch = (
            "@@ -1,6 +1,6 @@\n"
            " def process(user):\n"
            "-    if authenticate(user) and authorized(user):\n"
            "+    if authenticate(user) or authorized(user):\n"
            "-        try:\n"
            "-            handle()\n"
            "-        except ValueError:\n"
            "-            rollback()\n"
            "+        handle()\n"
        )
        fc = _fc(
            filename="service.py",
            file_type=FileType.PYTHON,
            patch=patch,
        )
        issues = await validator.validate_logic_safety(fc)
        issue_types = {i.type for i in issues}
        # Should detect the and->or inversion and security/error changes
        assert len(issues) >= 2
        assert (
            "security_sensitive_change" in issue_types
            or "logic_inversion" in issue_types
            or "error_handling_change" in issue_types
        )
