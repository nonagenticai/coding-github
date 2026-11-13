"""Extended tests for test_coverage_correlator.py — covering uncovered lines."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.test_coverage_correlator import (
    CoverageGap,
    TestCoverageCorrelator,
    TestFile,
)


@pytest.fixture
def codebase_ctx():
    return CodebaseContext(
        total_files=10,
        total_lines=5000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


@pytest.fixture
def correlator(codebase_ctx):
    return TestCoverageCorrelator(codebase_context=codebase_ctx)


# ---------------------------------------------------------------------------
# Lines 186-188: Exception in correlate_test_coverage
# ---------------------------------------------------------------------------
async def test_correlate_test_coverage_exception(correlator, codebase_ctx):
    """When an exception occurs during correlation, a fallback result should be returned."""
    # Pass a change that will cause an error in _identify_test_files
    bad_change = FileChange(
        filename="src/broken.py",
        file_type=FileType.PYTHON,
        after_content="def hello():\n    return 1\n",
        additions=2,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )

    # Monkey-patch _identify_test_files to raise
    original = correlator._identify_test_files
    correlator._identify_test_files = lambda changes: (_ for _ in ()).throw(
        RuntimeError("Correlation error")
    )

    result = await correlator.correlate_test_coverage([bad_change], codebase_ctx)
    assert result.coverage_gaps == []
    assert result.coverage_score == 0.0
    assert any("Error" in r for r in result.test_recommendations)

    correlator._identify_test_files = original


# ---------------------------------------------------------------------------
# Lines 231-232: _analyze_test_file returns None for no content
# ---------------------------------------------------------------------------
async def test_analyze_test_file_no_content(correlator, codebase_ctx):
    """Test file with no content should return None from _analyze_test_file."""
    change = FileChange(
        filename="tests/test_empty.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content=None,
        additions=0,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )
    result = await correlator._analyze_test_file(change)
    assert result is None


# ---------------------------------------------------------------------------
# Line 240: _determine_test_type with integration/e2e
# ---------------------------------------------------------------------------
def test_determine_test_type_integration(correlator):
    """Integration files without test_ prefix should be classified as integration."""
    # Note: test_ in filename matches unit first; use a filename without test_ prefix
    assert correlator._determine_test_type("integration/api_check.py") == "integration"


def test_determine_test_type_e2e(correlator):
    """E2e files without test_ prefix should be classified as integration."""
    assert correlator._determine_test_type("e2e/flow_check.py") == "integration"


def test_determine_test_type_spec(correlator):
    """Spec files should be classified as unit."""
    assert correlator._determine_test_type("test.spec.js") == "unit"


def test_determine_test_type_unknown(correlator):
    """Unrecognizable test file names should be classified as unknown."""
    assert correlator._determine_test_type("random_file.py") == "unknown"


# ---------------------------------------------------------------------------
# Lines 253-257: _analyze_javascript_test_file
# ---------------------------------------------------------------------------
async def test_analyze_javascript_test_file(correlator, codebase_ctx):
    """JavaScript test files should have describe blocks and test functions extracted."""
    js_test_code = (
        "import { render } from '@testing-library/react';\n"
        "import { Calculator } from './Calculator';\n"
        "\n"
        "describe('Calculator', () => {\n"
        "  it('should add two numbers', () => {\n"
        "    expect(add(1, 2)).toBe(3);\n"
        "  });\n"
        "  test('should subtract numbers', () => {\n"
        "    expect(subtract(3, 1)).toBe(2);\n"
        "  });\n"
        "});\n"
    )
    changes = [
        FileChange(
            filename="tests/Calculator.test.js",
            file_type=FileType.JAVASCRIPT,
            after_content=js_test_code,
            additions=12,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    assert len(result.existing_tests) > 0
    test_file = result.existing_tests[0]
    assert "Calculator" in test_file.test_classes
    assert len(test_file.test_functions) >= 2


# ---------------------------------------------------------------------------
# Lines 270-275: _detect_test_framework various frameworks
# ---------------------------------------------------------------------------
def test_detect_framework_pytest(correlator):
    """Should detect pytest framework."""
    content = "import pytest\ndef test_something(): pass"
    assert correlator._detect_test_framework(content) == "pytest"


def test_detect_framework_unittest(correlator):
    """Should detect unittest framework."""
    content = "import unittest\nclass TestFoo(unittest.TestCase): pass"
    assert correlator._detect_test_framework(content) == "unittest"


def test_detect_framework_jest(correlator):
    """Should detect jest framework."""
    content = "describe('test', () => { it('works', () => { expect(1).toBe(1); }); });"
    assert correlator._detect_test_framework(content) == "jest"


def test_detect_framework_none(correlator):
    """Should return None when no framework detected."""
    content = "print('hello world')"
    assert correlator._detect_test_framework(content) is None


# ---------------------------------------------------------------------------
# Line 283: SyntaxError in _analyze_python_test_file
# ---------------------------------------------------------------------------
async def test_analyze_python_test_file_syntax_error(correlator, codebase_ctx):
    """Python test files with syntax errors should be handled gracefully."""
    bad_python = "def test_something(\n    this is not valid python\n"
    change = FileChange(
        filename="tests/test_broken.py",
        file_type=FileType.PYTHON,
        after_content=bad_python,
        additions=2,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )
    # Should not raise
    test_file = await correlator._analyze_test_file(change)
    # The TestFile should still be created, just without parsed functions
    assert test_file is not None
    assert test_file.test_functions == []


# ---------------------------------------------------------------------------
# Lines 301-313: _analyze_python_test_file with class test methods
# ---------------------------------------------------------------------------
async def test_analyze_python_test_file_with_classes(correlator, codebase_ctx):
    """Python test files with test classes should have methods extracted."""
    content = (
        "import pytest\n"
        "\n"
        "class TestCalculator:\n"
        "    def test_add(self):\n"
        "        assert 1 + 1 == 2\n"
        "\n"
        "    def test_subtract(self):\n"
        "        assert 3 - 1 == 2\n"
    )
    change = FileChange(
        filename="tests/test_calc.py",
        file_type=FileType.PYTHON,
        after_content=content,
        additions=8,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )
    result = await correlator.correlate_test_coverage([change], codebase_ctx)
    assert len(result.existing_tests) > 0
    test_file = result.existing_tests[0]
    assert "TestCalculator" in test_file.test_classes
    # Methods should be prefixed with class name
    assert any("TestCalculator.test_add" in f for f in test_file.test_functions)


# ---------------------------------------------------------------------------
# Lines 318-338: _analyze_javascript_test_file patterns
# ---------------------------------------------------------------------------
async def test_analyze_javascript_test_file_with_it_blocks(correlator, codebase_ctx):
    """JavaScript test files should extract it() and test() blocks."""
    js_content = (
        "describe('UserService', () => {\n"
        "  it('should create user', () => {\n"
        "    expect(createUser()).toBeDefined();\n"
        "  });\n"
        "  test('should delete user', () => {\n"
        "    expect(deleteUser()).toBe(true);\n"
        "  });\n"
        "});\n"
    )
    change = FileChange(
        filename="tests/user.test.ts",
        file_type=FileType.TYPESCRIPT,
        after_content=js_content,
        additions=8,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )
    test_file = await correlator._analyze_test_file(change)
    assert test_file is not None
    assert "UserService" in test_file.test_classes
    assert "should create user" in test_file.test_functions
    assert "should delete user" in test_file.test_functions


# ---------------------------------------------------------------------------
# Line 347: _find_tested_modules with javascript imports
# ---------------------------------------------------------------------------
def test_find_tested_modules_javascript(correlator):
    """JavaScript import patterns should extract tested modules."""
    content = (
        "import { Calculator } from '../Calculator';\n"
        "const utils = require('../utils');\n"
        "import { render } from 'jest';\n"
    )
    modules = correlator._find_tested_modules(content, FileType.JAVASCRIPT)
    assert "../Calculator" in modules
    assert "../utils" in modules
    # jest should be filtered out
    assert "jest" not in modules


# ---------------------------------------------------------------------------
# Line 379: _infer_test_target_from_name patterns
# ---------------------------------------------------------------------------
def test_infer_test_target_should_pattern(correlator):
    """should_ pattern should be recognized."""
    target = correlator._infer_test_target_from_name("test_should_add_numbers")
    assert target is not None


def test_infer_test_target_success_pattern(correlator):
    """_success pattern should be recognized."""
    target = correlator._infer_test_target_from_name("test_create_user_success")
    assert target is not None


def test_infer_test_target_failure_pattern(correlator):
    """_failure pattern should be recognized."""
    target = correlator._infer_test_target_from_name("test_delete_user_failure")
    assert target is not None


def test_infer_test_target_error_pattern(correlator):
    """_error pattern should be recognized."""
    target = correlator._infer_test_target_from_name("test_parse_data_error")
    assert target is not None


def test_infer_test_target_simple_name(correlator):
    """Simple names with underscores should return first segment."""
    target = correlator._infer_test_target_from_name("test_calculate_sum")
    assert target is not None


def test_infer_test_target_empty_returns_none(correlator):
    """Empty test name after cleaning should return None."""
    target = correlator._infer_test_target_from_name("test_")
    assert target is None


# ---------------------------------------------------------------------------
# Line 421: _analyze_file_coverage_gaps with no content
# ---------------------------------------------------------------------------
async def test_analyze_file_coverage_gaps_no_content(correlator):
    """Source file with no content should produce no gaps."""
    source = FileChange(
        filename="src/empty.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content=None,
        additions=0,
        change_type=ChangeType.MODIFIED,
        deletions=0,
        patch="",
    )
    gaps = await correlator._analyze_file_coverage_gaps(source, set(), set())
    assert gaps == []


# ---------------------------------------------------------------------------
# Lines 443-447: _analyze_javascript_coverage_gaps
# ---------------------------------------------------------------------------
async def test_analyze_javascript_coverage_gaps(correlator, codebase_ctx):
    """JavaScript source files should have exported functions checked for gaps."""
    js_source = (
        "export function calculateTotal(items) {\n"
        "    return items.reduce((sum, item) => sum + item.price, 0);\n"
        "}\n"
        "\n"
        "export const formatCurrency = (amount) => {\n"
        "    return `$${amount.toFixed(2)}`;\n"
        "}\n"
    )
    changes = [
        FileChange(
            filename="src/utils.js",
            file_type=FileType.JAVASCRIPT,
            after_content=js_source,
            additions=7,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    gap_names = [g.symbol_name for g in result.coverage_gaps]
    assert "calculateTotal" in gap_names
    assert "formatCurrency" in gap_names


# ---------------------------------------------------------------------------
# Lines 479-493: _analyze_python_coverage_gaps with classes
# ---------------------------------------------------------------------------
async def test_analyze_python_coverage_gaps_with_class(correlator, codebase_ctx):
    """Public classes without test coverage should be flagged as gaps."""
    source = (
        "class UserService:\n"
        "    def create_user(self, name):\n"
        "        return {'name': name}\n"
        "\n"
        "    def delete_user(self, user_id):\n"
        "        return True\n"
    )
    changes = [
        FileChange(
            filename="src/user_service.py",
            file_type=FileType.PYTHON,
            after_content=source,
            additions=6,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    gap_types = [(g.symbol_name, g.symbol_type) for g in result.coverage_gaps]
    # UserService should be a class gap
    assert ("UserService", "class") in gap_types


async def test_analyze_python_coverage_gaps_private_ignored(correlator, codebase_ctx):
    """Private functions should not be flagged as coverage gaps."""
    source = (
        "def _private_helper():\n"
        "    return 42\n"
        "\n"
        "def public_function():\n"
        "    return _private_helper()\n"
    )
    changes = [
        FileChange(
            filename="src/helpers.py",
            file_type=FileType.PYTHON,
            after_content=source,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    gap_names = [g.symbol_name for g in result.coverage_gaps]
    assert "_private_helper" not in gap_names
    assert "public_function" in gap_names


# ---------------------------------------------------------------------------
# Lines 503-533: _analyze_javascript_coverage_gaps with exports patterns
# ---------------------------------------------------------------------------
async def test_analyze_javascript_coverage_gaps_module_exports(correlator, codebase_ctx):
    """module.exports and exports.X patterns should be detected."""
    js_source = (
        "exports.helper = function() { return 1; };\n"
        "module.exports.main = function() { return 2; };\n"
    )
    changes = [
        FileChange(
            filename="src/lib.js",
            file_type=FileType.JAVASCRIPT,
            after_content=js_source,
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    gap_names = [g.symbol_name for g in result.coverage_gaps]
    assert "helper" in gap_names
    assert "main" in gap_names


# ---------------------------------------------------------------------------
# Line 557: _assess_function_test_priority with security code
# ---------------------------------------------------------------------------
def test_assess_function_priority_security(correlator):
    """Functions with security-sensitive code should have high priority."""
    import ast

    source = (
        "def authenticate(password):\n"
        "    if password == 'secret':\n"
        "        return True\n"
        "    return False\n"
    )
    tree = ast.parse(source)
    func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    priority = correlator._assess_function_test_priority(func_node, source)
    assert priority == "high"


def test_assess_function_priority_error_handling(correlator):
    """Functions with error handling should have medium priority."""
    import ast

    source = (
        "def process(data):\n"
        "    try:\n"
        "        return data[0]\n"
        "    except IndexError:\n"
        "        raise ValueError('empty')\n"
    )
    tree = ast.parse(source)
    func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    priority = correlator._assess_function_test_priority(func_node, source)
    assert priority in ("medium", "high")


def test_assess_function_priority_simple(correlator):
    """Simple functions should have low priority."""
    import ast

    source = "def add(a, b):\n    return a + b\n"
    tree = ast.parse(source)
    func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    priority = correlator._assess_function_test_priority(func_node, source)
    assert priority == "low"


# ---------------------------------------------------------------------------
# Line 574: _generate_test_recommendations with large changes
# ---------------------------------------------------------------------------
def test_generate_test_recommendations_large_changes(correlator):
    """Source files with many changes should generate specific recommendations."""
    source_files = [
        FileChange(
            filename="src/big_module.py",
            file_type=FileType.PYTHON,
            additions=60,
            deletions=10,
            change_type=ChangeType.MODIFIED,
            patch="",
        ),
    ]
    gaps = [
        CoverageGap(
            file_path="src/big_module.py",
            symbol_name="big_function",
            symbol_type="function",
            line_number=10,
            severity="high",
            reason="Not covered",
        ),
    ]
    recs = correlator._generate_test_recommendations(source_files, gaps)
    assert any("big_module.py" in r for r in recs)
    assert any("High priority" in r for r in recs)


# ---------------------------------------------------------------------------
# Lines 586-587: _generate_test_recommendations with medium gaps
# ---------------------------------------------------------------------------
def test_generate_test_recommendations_medium_gaps(correlator):
    """Medium priority gaps should produce medium priority recommendations."""
    source_files = []
    gaps = [
        CoverageGap(
            file_path="src/utils.py",
            symbol_name="helper",
            symbol_type="function",
            line_number=5,
            severity="medium",
            reason="Not covered",
        ),
    ]
    recs = correlator._generate_test_recommendations(source_files, gaps)
    assert any("Medium priority" in r for r in recs)


# ---------------------------------------------------------------------------
# Line 626: _calculate_coverage_score capped at 1.0
# ---------------------------------------------------------------------------
def test_coverage_score_capped_at_one(correlator):
    """Coverage score should not exceed 1.0."""
    source_files = [
        FileChange(
            filename="src/simple.py",
            file_type=FileType.PYTHON,
            after_content="def a():\n    pass\n",
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    # Many coverage targets
    test_files = [
        TestFile(
            file_path="tests/test_simple.py",
            test_type="unit",
            coverage_targets={"a", "b", "c", "d", "e"},
        ),
    ]
    score = correlator._calculate_coverage_score(source_files, test_files)
    assert score <= 1.0


# ---------------------------------------------------------------------------
# Lines 687-690: _suggest_test_file_name for different extensions
# ---------------------------------------------------------------------------
def test_suggest_test_file_python(correlator):
    """Python files should get test_ prefix."""
    assert correlator._suggest_test_file_name("src/calculator.py") == "test_calculator.py"


def test_suggest_test_file_javascript(correlator):
    """JavaScript files should get .test suffix."""
    assert correlator._suggest_test_file_name("src/utils.js") == "utils.test.js"


def test_suggest_test_file_typescript(correlator):
    """TypeScript files should get .test suffix."""
    assert correlator._suggest_test_file_name("src/api.ts") == "api.test.ts"


def test_suggest_test_file_other(correlator):
    """Other file types should get test_ prefix."""
    assert correlator._suggest_test_file_name("src/config.yaml") == "test_config.yaml"


# ---------------------------------------------------------------------------
# Line 698: _is_public_class
# ---------------------------------------------------------------------------
def test_is_public_class(correlator):
    """Public classes should return True."""
    assert correlator._is_public_class("MyClass") is True


def test_is_private_class(correlator):
    """Private classes (starting with _) should return False."""
    assert correlator._is_public_class("_InternalClass") is False


# ---------------------------------------------------------------------------
# _find_orphaned_tests
# ---------------------------------------------------------------------------
async def test_find_orphaned_tests(correlator, codebase_ctx):
    """Test files without matching source files should be flagged as orphaned."""
    test_code = (
        "import pytest\n"
        "from deleted_module import old_func\n"
        "\n"
        "def test_old_func():\n"
        "    assert old_func() is not None\n"
    )
    source_code = "def something():\n    return True\n"
    changes = [
        FileChange(
            filename="tests/test_deleted_module.py",
            file_type=FileType.PYTHON,
            after_content=test_code,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="src/something.py",
            file_type=FileType.PYTHON,
            after_content=source_code,
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # test_deleted_module -> deleted_module, which is not in source files
    assert len(result.orphaned_tests) > 0


# ---------------------------------------------------------------------------
# _calculate_coverage_impact with gaps
# ---------------------------------------------------------------------------
async def test_coverage_impact_negative(correlator, codebase_ctx):
    """New code without test coverage should produce negative impact."""
    source = "def new_func():\n    return 42\n"
    changes = [
        FileChange(
            filename="src/new_module.py",
            file_type=FileType.PYTHON,
            after_content=source,
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # Should have negative or zero impact
    assert result.impact_on_coverage <= 0.0


# ---------------------------------------------------------------------------
# Python file with SyntaxError in coverage gap analysis
# ---------------------------------------------------------------------------
async def test_python_coverage_gaps_syntax_error(correlator, codebase_ctx):
    """Python files with syntax errors in coverage analysis should be handled gracefully."""
    bad_source = "def broken(\n    pass\n"
    changes = [
        FileChange(
            filename="src/broken.py",
            file_type=FileType.PYTHON,
            after_content=bad_source,
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # Should still have a module-level gap at minimum
    assert any(g.symbol_type == "module" for g in result.coverage_gaps)


# ---------------------------------------------------------------------------
# _get_module_name
# ---------------------------------------------------------------------------
def test_get_module_name(correlator):
    """Module name should be extracted from file path."""
    assert correlator._get_module_name("src/services/calculator.py") == "calculator"
    assert correlator._get_module_name("tests/test_utils.py") == "test_utils"


# ---------------------------------------------------------------------------
# Lines 231-232: _analyze_existing_tests error in individual test file
# ---------------------------------------------------------------------------
async def test_analyze_existing_tests_individual_error(correlator, codebase_ctx):
    """Error analyzing one test file should not prevent others from being analyzed."""
    good_test = "import pytest\ndef test_good():\n    assert True\n"
    changes = [
        FileChange(
            filename="tests/test_good.py",
            file_type=FileType.PYTHON,
            after_content=good_test,
            additions=3,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="tests/test_bad.py",
            file_type=FileType.PYTHON,
            after_content="some content",
            additions=1,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]

    # Monkey-patch _analyze_test_file to raise for the bad file
    original = correlator._analyze_test_file

    async def patched_analyze(change):
        if "bad" in change.filename:
            raise RuntimeError("Analysis failed")
        return await original(change)

    correlator._analyze_test_file = patched_analyze

    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # At least the good test file should be analyzed
    assert len(result.existing_tests) >= 1

    correlator._analyze_test_file = original


# ---------------------------------------------------------------------------
# Line 271: _determine_test_type with integration path and no test_ prefix
# ---------------------------------------------------------------------------
def test_determine_test_type_integration_path_only(correlator):
    """Integration path without test_ prefix should classify as integration."""
    assert correlator._determine_test_type("integration_check.py") == "integration"


# ---------------------------------------------------------------------------
# _find_tested_modules with python imports filtering test frameworks
# ---------------------------------------------------------------------------
def test_find_tested_modules_filters_frameworks(correlator):
    """Test framework imports should be filtered out."""
    content = (
        "import pytest\n"
        "from unittest import mock\n"
        "from myapp.core import handler\n"
        "import myapp.utils\n"
    )
    modules = correlator._find_tested_modules(content, FileType.PYTHON)
    assert "myapp.core" in modules
    assert "myapp.utils" in modules
    assert "pytest" not in modules
    assert "unittest" not in modules
