"""Unit tests for MaintainabilityAssessor.

Verifies: complexity detection, nesting depth, parameter count thresholds,
empty/non-code file handling, and baseline clean code produces no issues.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.context.maintainability_assessor import (
    MaintainabilityAssessor,
)


@pytest.fixture
def assessor():
    return MaintainabilityAssessor()


@pytest.fixture
def codebase_context():
    return CodebaseContext(
        total_files=10,
        total_lines=5000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


class TestBaselineBehavior:
    """Simple, clean code should produce zero relevant issues."""

    async def test_no_issues_for_clean_code(self, assessor, codebase_context):
        simple_code = (
            '"""Module docstring."""\n'
            "\n\n"
            "def add(a, b):\n"
            '    """Add two numbers."""\n'
            "    return a + b\n"
        )
        changes = [
            FileChange(
                filename="src/math_utils.py",
                file_type=FileType.PYTHON,
                after_content=simple_code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        issues = await assessor.assess_maintainability(changes, codebase_context)
        complexity_issues = [
            i for i in issues if "complexity" in i.type or "long_functions" in i.type
        ]
        assert complexity_issues == []

    async def test_empty_content_produces_no_issues(self, assessor, codebase_context):
        """Files with no content (None) produce no issues."""
        changes = [
            FileChange(
                filename="src/empty.py",
                file_type=FileType.PYTHON,
                after_content=None,
                before_content=None,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        issues = await assessor.assess_maintainability(changes, codebase_context)
        assert issues == []

    async def test_non_code_file_type_skipped(self, assessor, codebase_context):
        """UNKNOWN file type is not analysed."""
        changes = [
            FileChange(
                filename="data.csv",
                file_type=FileType.UNKNOWN,
                after_content="col1,col2\n1,2\n3,4",
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        issues = await assessor.assess_maintainability(changes, codebase_context)
        assert issues == []


class TestComplexityDetection:
    """High cyclomatic complexity triggers warnings."""

    async def test_many_branches_trigger_complexity_issue(self, assessor, codebase_context):
        """Function with many if-branches triggers high_cyclomatic_complexity."""
        branches = "\n".join([f"    if x == {i}:\n        return {i}" for i in range(20)])
        complex_code = f"def branchy(x):\n{branches}\n"
        changes = [
            FileChange(
                filename="src/complex.py",
                file_type=FileType.PYTHON,
                after_content=complex_code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        with (
            patch.object(assessor, "_detect_code_smells", new_callable=AsyncMock, return_value=[]),
            patch.object(
                assessor, "_detect_anti_patterns", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(
                assessor,
                "_analyze_documentation_quality",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                assessor, "_analyze_naming_conventions", new_callable=AsyncMock, return_value=[]
            ),
        ):
            issues = await assessor.assess_maintainability(changes, codebase_context)
        issue_types = [i.type for i in issues]
        assert "high_cyclomatic_complexity" in issue_types


class TestNestingDepth:
    """Deep nesting triggers warnings."""

    async def test_deep_nesting_detected(self, assessor, codebase_context):
        nested_code = (
            "def deep():\n"
            "    if True:\n"
            "        for i in range(10):\n"
            "            while True:\n"
            "                if True:\n"
            "                    for j in range(5):\n"
            "                        pass\n"
        )
        changes = [
            FileChange(
                filename="src/nested.py",
                file_type=FileType.PYTHON,
                after_content=nested_code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        with (
            patch.object(assessor, "_detect_code_smells", new_callable=AsyncMock, return_value=[]),
            patch.object(
                assessor, "_detect_anti_patterns", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(
                assessor,
                "_analyze_documentation_quality",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                assessor, "_analyze_naming_conventions", new_callable=AsyncMock, return_value=[]
            ),
        ):
            issues = await assessor.assess_maintainability(changes, codebase_context)
        issue_types = [i.type for i in issues]
        assert "deep_nesting" in issue_types


class TestParameterCount:
    """Functions with too many parameters are flagged."""

    async def test_many_parameters_flagged(self, assessor, codebase_context):
        many_params_code = (
            "def many_params(a, b, c, d, e, f, g, h, i, j, k):\n"
            '    """Lots of params."""\n'
            "    return a + b + c + d + e + f + g + h + i + j + k\n"
        )
        changes = [
            FileChange(
                filename="src/params.py",
                file_type=FileType.PYTHON,
                after_content=many_params_code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            )
        ]
        issues = await assessor.assess_maintainability(changes, codebase_context)
        issue_types = [i.type for i in issues]
        assert "long_parameter_list" in issue_types
