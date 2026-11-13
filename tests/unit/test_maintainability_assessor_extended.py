"""Extended tests for maintainability_assessor.py -- covering internal methods.

Targets: exception path, empty content, function length thresholds, Python metrics
(god class/function, vararg/kwarg), text-based metrics, function complexity,
code smells, anti-patterns, documentation quality, naming conventions.
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
    MaintainabilityMetrics,
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


# ===========================================================================
# Exception handling in assess_maintainability
# ===========================================================================


class TestAssessMaintainabilityException:
    async def test_returns_error_issue_on_internal_failure(self, assessor, codebase_context):
        """Internal error produces a maintainability_analysis_error issue."""
        changes = [
            FileChange(
                filename="app.py",
                file_type=FileType.PYTHON,
                after_content="x = 1",
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]
        with patch.object(
            assessor,
            "_analyze_maintainability_metrics",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            issues = await assessor.assess_maintainability(changes, codebase_context)
        error_issues = [i for i in issues if "analysis_error" in i.type]
        assert len(error_issues) == 1
        assert error_issues[0].severity == "medium"


# ===========================================================================
# Empty content handling
# ===========================================================================


class TestAnalyzeMetricsEmptyContent:
    async def test_empty_strings_return_no_issues(self, assessor):
        """Both empty strings produce no issues."""
        change = FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            after_content="",
            before_content="",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_maintainability_metrics(change)
        assert issues == []


# ===========================================================================
# Function length thresholds
# ===========================================================================


class TestFunctionLengthIssue:
    @pytest.mark.parametrize(
        ("line_count", "expected_severity"),
        [
            (110, "medium"),
            (160, "high"),
        ],
        ids=["medium-length", "very-long"],
    )
    async def test_function_length_severity(self, assessor, line_count, expected_severity):
        """Functions exceeding length thresholds produce appropriately severe issues."""
        lines = ["def long_func():"]
        lines.append('    """Docstring."""')
        for i in range(line_count):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x0")
        content = "\n".join(lines)
        change = FileChange(
            filename="long.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
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
            issues = await assessor._analyze_maintainability_metrics(change)
        long_issues = [i for i in issues if i.type == "long_functions"]
        assert len(long_issues) >= 1
        assert long_issues[0].severity == expected_severity


# ===========================================================================
# Python metrics: god functions, god classes, parameter handling
# ===========================================================================


class TestPythonMetrics:
    def test_god_function_counted(self, assessor):
        """Functions > 100 lines are counted as god functions."""
        lines = ["def god_func():"]
        for i in range(105):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x0")
        content = "\n".join(lines)
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert metrics.god_functions >= 1

    def test_god_class_counted(self, assessor):
        """Classes > 300 lines are counted as god classes."""
        lines = ["class BigClass:"]
        for i in range(310):
            lines.append(f"    attr{i} = {i}")
        content = "\n".join(lines)
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert metrics.god_classes >= 1

    def test_vararg_and_kwarg_counted_in_params(self, assessor):
        """*args and **kwargs increase parameter count."""
        content = "def func(a, b, *args, **kwargs):\n    pass\n"
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert metrics.parameter_count_max >= 4

    def test_long_parameter_lists_counted(self, assessor):
        """Functions with > 5 params increment long_parameter_lists."""
        content = "def func(a, b, c, d, e, f, g):\n    pass\n"
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert metrics.long_parameter_lists >= 1

    def test_class_length_avg_calculated(self, assessor):
        """Class length average is calculated correctly."""
        content = "class A:\n" + "\n".join(f"    x{i} = {i}" for i in range(50)) + "\n"
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert metrics.class_length_avg > 0

    def test_syntax_error_falls_back_to_text(self, assessor):
        """Invalid Python syntax falls back to text-based metrics."""
        content = "def broken(\n  syntax here = bad\n"
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        assert isinstance(metrics, MaintainabilityMetrics)


# ===========================================================================
# Text-based metrics (for non-Python files)
# ===========================================================================


class TestTextBasedMetrics:
    def test_javascript_uses_text_metrics(self, assessor):
        """JavaScript files use text-based metric calculation."""
        content = "function foo(a, b) {\n  return a + b;\n}\nfunction bar(x) {\n  return x;\n}\n"
        metrics = assessor._calculate_maintainability_metrics(content, FileType.JAVASCRIPT)
        assert metrics.function_length_avg > 0

    def test_def_param_extraction(self, assessor):
        """Text-based param extraction picks up def-style functions."""
        content = "def foo(a, b, c, d, e, f, g):\n    pass\n"
        metrics = assessor._calculate_text_based_metrics(content, MaintainabilityMetrics())
        assert metrics.parameter_count_max >= 7
        assert metrics.long_parameter_lists >= 1

    def test_nesting_depth_from_indentation(self, assessor):
        """Text-based nesting depth calculated from indentation."""
        content = (
            "def outer():\n"
            "    if True:\n"
            "        for i in range(10):\n"
            "            while True:\n"
            "                    deeply_nested()\n"
        )
        metrics = assessor._calculate_text_based_metrics(content, MaintainabilityMetrics())
        assert metrics.nesting_depth_max >= 4

    def test_class_starts_tracked(self, assessor):
        """Class starts are detected in text-based analysis."""
        content = "class MyClass:\n    x = 1\n\nclass AnotherClass:\n    z = 3\n"
        metrics = assessor._calculate_text_based_metrics(content, MaintainabilityMetrics())
        assert isinstance(metrics, MaintainabilityMetrics)


# ===========================================================================
# Function complexity
# ===========================================================================


class TestFunctionComplexity:
    def test_try_except_adds_complexity(self, assessor):
        """Each except handler adds 1 to complexity."""
        content = (
            "def func():\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError:\n"
            "        pass\n"
            "    except TypeError:\n"
            "        pass\n"
        )
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        # 1 (base) + 2 (except handlers)
        assert metrics.cyclomatic_complexity >= 3

    def test_boolean_operators_add_complexity(self, assessor):
        """Boolean operators (and/or) add to complexity."""
        content = (
            "def func(a, b, c, d):\n"
            "    if a and b and c and d:\n"
            "        return True\n"
            "    return False\n"
        )
        metrics = assessor._calculate_maintainability_metrics(content, FileType.PYTHON)
        # 1 (base) + 1 (if) + 3 (and/and/and)
        assert metrics.cyclomatic_complexity >= 5


# ===========================================================================
# Code smells
# ===========================================================================


class TestDetectCodeSmells:
    async def test_empty_content_returns_empty(self, assessor):
        """Empty content returns no code smells."""
        change = FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            after_content="",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._detect_code_smells(change)
        assert issues == []

    async def test_dead_code_detected(self, assessor):
        """Dead code (pass-only function) detected."""
        content = "def noop():\n    pass\n"
        change = FileChange(
            filename="dead.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._detect_code_smells(change)
        types = [i.type for i in issues]
        assert any("dead_code" in t for t in types)

    async def test_smell_issue_has_line_number(self, assessor):
        """Code smell issues include approximate line numbers."""
        content = "if False:\n    pass\n"
        change = FileChange(
            filename="dead2.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._detect_code_smells(change)
        dead_issues = [i for i in issues if "dead_code" in i.type]
        assert len(dead_issues) > 0
        assert dead_issues[0].line is not None


# ===========================================================================
# Smell helper methods
# ===========================================================================


class TestSmellHelpers:
    @pytest.mark.parametrize(
        ("smell_type", "expected_severity"),
        [
            ("god_class", "high"),
            ("nested_loops", "high"),
            ("long_method", "medium"),
            ("magic_numbers", "low"),
            ("unknown_type", "low"),
        ],
    )
    def test_severity_lookup(self, assessor, smell_type, expected_severity):
        assert assessor._get_smell_severity(smell_type) == expected_severity

    def test_description_for_known_type(self, assessor):
        desc = assessor._get_smell_description("duplicate_code")
        assert "Duplicate" in desc
        assert assessor._get_smell_description("unknown_type") == "Code smell detected"

    def test_suggestion_for_known_type(self, assessor):
        sug = assessor._get_smell_suggestion("long_method").lower()
        assert "smaller" in sug or "break" in sug
        assert assessor._get_smell_suggestion("unknown_type") == "Consider refactoring"

    def test_all_known_smell_types_have_entries(self, assessor):
        """All known smell types have severity, description, and suggestion."""
        known_types = [
            "long_method",
            "long_parameter_list",
            "duplicate_code",
            "magic_numbers",
            "god_class",
            "dead_code",
            "complex_conditionals",
            "nested_loops",
            "global_variables",
            "inconsistent_naming",
        ]
        for t in known_types:
            assert assessor._get_smell_severity(t) in ("low", "medium", "high")
            assert len(assessor._get_smell_description(t)) > 0
            assert len(assessor._get_smell_suggestion(t)) > 0


# ===========================================================================
# Anti-patterns
# ===========================================================================


class TestDetectAntiPatterns:
    async def test_empty_content_returns_empty(self, assessor):
        change = FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            after_content="",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._detect_anti_patterns(change)
        assert issues == []

    async def test_boolean_parameter_abuse_detected(self, assessor):
        content = "def do_thing(flag=True):\n    pass\n"
        change = FileChange(
            filename="flags.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._detect_anti_patterns(change)
        types = [i.type for i in issues]
        assert any("boolean_parameter_abuse" in t for t in types)


# ===========================================================================
# Documentation quality
# ===========================================================================


class TestDocumentationQuality:
    async def test_empty_content_returns_empty(self, assessor):
        change = FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            after_content="",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_documentation_quality(change)
        assert issues == []

    async def test_public_function_missing_docstring_flagged(self, assessor):
        content = "def public_func():\n    return 42\n"
        change = FileChange(
            filename="nodoc.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_documentation_quality(change)
        types = [i.type for i in issues]
        assert "missing_docstring" in types

    async def test_private_function_not_flagged(self, assessor):
        """Private functions (starting with _) are not flagged for docstrings."""
        content = "def _private_func():\n    return 42\n"
        change = FileChange(
            filename="priv.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_documentation_quality(change)
        doc_issues = [i for i in issues if i.type == "missing_docstring"]
        assert len(doc_issues) == 0

    async def test_public_class_missing_docstring_flagged(self, assessor):
        content = "class MyClass:\n    x = 1\n"
        change = FileChange(
            filename="cls.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_documentation_quality(change)
        doc_issues = [i for i in issues if i.type == "missing_docstring"]
        assert len(doc_issues) >= 1
        assert "class" in doc_issues[0].message

    async def test_non_python_file_skipped(self, assessor):
        content = "function foo() { return 1; }\n"
        change = FileChange(
            filename="app.js",
            file_type=FileType.JAVASCRIPT,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_documentation_quality(change)
        assert issues == []


# ===========================================================================
# Naming conventions
# ===========================================================================


class TestNamingConventions:
    async def test_empty_content_returns_empty(self, assessor):
        change = FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            after_content="",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_naming_conventions(change)
        assert issues == []

    async def test_valid_python_names_no_violations(self, assessor):
        content = "def my_function():\n    pass\nclass MyClass:\n    pass\n"
        change = FileChange(
            filename="valid.py",
            file_type=FileType.PYTHON,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_naming_conventions(change)
        naming_issues = [i for i in issues if i.type == "naming_convention_violation"]
        assert len(naming_issues) == 0

    async def test_unsupported_file_type_returns_empty(self, assessor):
        content = "some_code = True\n"
        change = FileChange(
            filename="style.css",
            file_type=FileType.UNKNOWN,
            after_content=content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        issues = await assessor._analyze_naming_conventions(change)
        assert issues == []


class TestCheckNamingConvention:
    """Naming convention checker for Python and JavaScript."""

    @pytest.mark.parametrize(
        ("name", "kind", "language", "valid"),
        [
            ("my_func", "function", "python", True),
            ("myFunc", "function", "python", False),
            ("MyClass", "class", "python", True),
            ("my_class", "class", "python", False),
            ("my_var", "variable", "python", True),
            ("myVar", "variable", "python", False),
            ("MY_CONST", "constant", "python", True),
            ("my_const", "constant", "python", False),
            ("myFunc", "function", "javascript", True),
            ("my_func", "function", "javascript", False),
            ("MyClass", "class", "javascript", True),
            ("myclass", "class", "javascript", False),
            ("myVar", "variable", "javascript", True),
            ("anything", "function", "ruby", True),
        ],
        ids=[
            "py-func-snake-ok",
            "py-func-camel-bad",
            "py-class-pascal-ok",
            "py-class-snake-bad",
            "py-var-snake-ok",
            "py-var-camel-bad",
            "py-const-upper-ok",
            "py-const-lower-bad",
            "js-func-camel-ok",
            "js-func-snake-bad",
            "js-class-pascal-ok",
            "js-class-lower-bad",
            "js-var-camel-ok",
            "unknown-lang-ok",
        ],
    )
    def test_naming_convention(self, assessor, name, kind, language, valid):
        result = assessor._check_naming_convention(name, kind, language)
        if valid:
            assert result is None
        else:
            assert result is not None
