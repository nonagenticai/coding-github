"""Extended tests for PerformanceAnalyzer — targeting uncovered lines."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.context.performance_analyzer import (
    PerformanceAnalyzer,
    PRValidationPerformanceMetrics,
)


@pytest.fixture
def analyzer():
    return PerformanceAnalyzer()


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


# ---------------------------------------------------------------------------
# Lines 311-313: exception handler in analyze_performance_impact
# ---------------------------------------------------------------------------
async def test_analyze_performance_impact_exception_handler(analyzer, codebase_ctx):
    """When an internal method raises, the top-level handler catches it and
    returns a performance_analysis_error issue (lines 311-313)."""
    change = FileChange(
        filename="src/boom.py",
        file_type=FileType.PYTHON,
        after_content="x = 1\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    with patch.object(analyzer, "_analyze_performance_patterns", side_effect=RuntimeError("boom")):
        issues = await analyzer.analyze_performance_impact([change], codebase_ctx)

    assert len(issues) == 1
    assert issues[0].type == "performance_analysis_error"
    assert issues[0].rule == "performance_analyzer_error"
    assert "boom" in issues[0].message


# ---------------------------------------------------------------------------
# Line 340: empty content early return in _analyze_performance_patterns
# ---------------------------------------------------------------------------
async def test_analyze_performance_patterns_empty_content(analyzer):
    """When both before_content and after_content are empty strings,
    _analyze_performance_patterns returns early (line 340)."""
    change = FileChange(
        filename="src/empty.py",
        file_type=FileType.PYTHON,
        before_content="",
        after_content="",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_performance_patterns(change)
    assert issues == []


# ---------------------------------------------------------------------------
# Lines 373-378: multi-line pattern matching with line number calculation
# ---------------------------------------------------------------------------
def test_find_pattern_matches_multiline(analyzer):
    """_find_pattern_matches takes the multi-line branch when the pattern
    string contains an actual newline character, computing the line number
    from preceding newlines (lines 373-378)."""
    from services.pr_validation.analyzers.context.performance_analyzer import (
        PerformancePattern,
    )

    # Pattern with a real newline (not a raw-string \\n)
    pattern = PerformancePattern(
        pattern_id="test_multiline",
        name="Test Multi-line",
        pattern="for .*:\n.*for .*:",  # actual newline in pattern string
        severity="medium",
        description="test",
        recommendation="test",
        complexity_impact="O(n^2)",
        file_types=["python"],
    )
    content = "x = 1\ny = 2\nfor i in range(10):\n    for j in range(10):\n        pass\n"
    matches = analyzer._find_pattern_matches(content, pattern, "test.py")
    assert len(matches) == 1
    assert matches[0]["line_number"] == 3
    assert matches[0]["pattern_id"] == "test_multiline"


# ---------------------------------------------------------------------------
# Lines 484-486: AST SyntaxError fallback to text-based complexity estimation
# ---------------------------------------------------------------------------
def test_calculate_complexity_metrics_syntax_error_fallback(analyzer):
    """When ast.parse raises SyntaxError, _calculate_complexity_metrics
    falls back to _estimate_cyclomatic_complexity (lines 484-486)."""
    invalid_python = "def foo(:\n    if x and y:\n        for i in z:\n            pass\n"
    metrics = analyzer._calculate_complexity_metrics(invalid_python)
    # _estimate_cyclomatic_complexity should count keywords
    assert metrics.cyclomatic_complexity >= 1


# ---------------------------------------------------------------------------
# Lines 574, 576, 578: cyclomatic complexity for Try/ExceptHandler/BoolOp
# ---------------------------------------------------------------------------
def test_cyclomatic_complexity_try_except_boolop(analyzer):
    """_calculate_cyclomatic_complexity counts Try handlers, ExceptHandler,
    and BoolOp operands (lines 574, 576, 578)."""
    import ast

    code = (
        "def foo():\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"
        "        pass\n"
        "    except TypeError:\n"
        "        pass\n"
        "    if a and b or c:\n"
        "        pass\n"
    )
    tree = ast.parse(code)
    complexity = analyzer._calculate_cyclomatic_complexity(tree)
    # Base=1, Try with 2 handlers=+2, 2 ExceptHandlers=+2,
    # if=+1, 'and'=BoolOp(+1), 'or'=BoolOp(+1)
    # Total should be >= 5
    assert complexity >= 5


# ---------------------------------------------------------------------------
# Lines 584-600: _estimate_cyclomatic_complexity text-based analysis
# ---------------------------------------------------------------------------
def test_estimate_cyclomatic_complexity(analyzer):
    """_estimate_cyclomatic_complexity counts keyword occurrences (lines 584-600)."""
    content = (
        "if x:\n"
        "    pass\n"
        "elif y:\n"
        "    pass\n"
        "while z:\n"
        "    for i in items:\n"
        "        try:\n"
        "            something()\n"
        "        except:\n"
        "            pass\n"
        "if a and b or c:\n"
        "    pass\n"
    )
    result = analyzer._estimate_cyclomatic_complexity(content)
    # Base 1 + if(2) + elif(1) + while(1) + for(1) + try(1) + except(1) + and(1) + or(1)
    assert result >= 9


# ---------------------------------------------------------------------------
# Lines 610, 615-618: _count_recursive_calls with function name extraction
# ---------------------------------------------------------------------------
def test_count_recursive_calls(analyzer):
    """_count_recursive_calls finds functions that call themselves (lines 610, 615-618)."""
    import ast

    code = (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
        "\n"
        "async def async_recurse(items):\n"
        "    if not items:\n"
        "        return\n"
        "    await async_recurse(items[1:])\n"
    )
    tree = ast.parse(code)
    count = analyzer._count_recursive_calls(tree)
    assert count == 2


def test_count_recursive_calls_no_recursion(analyzer):
    """Non-recursive functions should return 0."""
    import ast

    code = "def greet(name):\n    return f'Hello, {name}'\n"
    tree = ast.parse(code)
    count = analyzer._count_recursive_calls(tree)
    assert count == 0


# ---------------------------------------------------------------------------
# Lines 628, 630, 634, 636: _determine_algorithmic_complexity branches
# ---------------------------------------------------------------------------
def test_determine_complexity_n_cubed(analyzer):
    """nested_loop_count >= 3 -> O(n^3) (line 628)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=3)
    result = analyzer._determine_algorithmic_complexity("x = 1\n", metrics)
    assert result == "O(n^3)"


def test_determine_complexity_n_squared(analyzer):
    """nested_loop_count >= 2 -> O(n^2) (line 630)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=2)
    result = analyzer._determine_algorithmic_complexity("x = 1\n", metrics)
    assert result == "O(n^2)"


def test_determine_complexity_n_log_n(analyzer):
    """sorted() without for-loop -> O(n log n) (line 634)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=0)
    content = "result = sorted(items)\n"
    result = analyzer._determine_algorithmic_complexity(content, metrics)
    assert result == "O(n log n)"


def test_determine_complexity_log_n(analyzer):
    """bisect without loops or sort -> O(log n) (line 636)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=0)
    content = "pos = bisect(items, val)\n"
    result = analyzer._determine_algorithmic_complexity(content, metrics)
    assert result == "O(log n)"


def test_determine_complexity_o_n(analyzer):
    """A for-loop with nested_loop_count=0 -> O(n) (line 632 implicit, covers branch)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=0)
    content = "for x in items:\n    process(x)\n"
    result = analyzer._determine_algorithmic_complexity(content, metrics)
    assert result == "O(n)"


def test_determine_complexity_o1(analyzer):
    """No loops, no sort, no binary search -> O(1)."""
    metrics = PRValidationPerformanceMetrics(nested_loop_count=0)
    content = "x = 1\n"
    result = analyzer._determine_algorithmic_complexity(content, metrics)
    assert result == "O(1)"


# ---------------------------------------------------------------------------
# Line 646: _analyze_resource_usage empty content return
# ---------------------------------------------------------------------------
async def test_analyze_resource_usage_empty_content(analyzer):
    """_analyze_resource_usage returns early on empty content (line 646)."""
    change = FileChange(
        filename="src/empty.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content="",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_resource_usage(change)
    assert issues == []


async def test_analyze_resource_usage_no_content(analyzer):
    """_analyze_resource_usage returns early when both contents are None."""
    change = FileChange(
        filename="src/none.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content=None,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_resource_usage(change)
    assert issues == []


# ---------------------------------------------------------------------------
# Line 683: _analyze_caching_opportunities empty content return
# ---------------------------------------------------------------------------
async def test_analyze_caching_opportunities_empty_content(analyzer):
    """_analyze_caching_opportunities returns early on empty content (line 683)."""
    change = FileChange(
        filename="src/empty.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content="",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_caching_opportunities(change)
    assert issues == []


async def test_analyze_caching_opportunities_no_content(analyzer):
    """_analyze_caching_opportunities returns early when both contents are None."""
    change = FileChange(
        filename="src/none.py",
        file_type=FileType.PYTHON,
        before_content=None,
        after_content=None,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_caching_opportunities(change)
    assert issues == []


# ---------------------------------------------------------------------------
# Additional coverage: _analyze_caching_opportunities with HTTP request
# ---------------------------------------------------------------------------
async def test_analyze_caching_opportunities_with_http(analyzer):
    """Detects caching opportunity for HTTP requests."""
    code = "response = requests.get('https://api.example.com/data')\n"
    change = FileChange(
        filename="src/client.py",
        file_type=FileType.PYTHON,
        after_content=code,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_caching_opportunities(change)
    caching_issues = [i for i in issues if i.type == "caching_opportunity"]
    assert len(caching_issues) >= 1


# ---------------------------------------------------------------------------
# Additional: full integration path with syntax-error code to exercise
# the SyntaxError fallback through analyze_performance_impact
# ---------------------------------------------------------------------------
async def test_full_path_with_syntax_error_code(analyzer, codebase_ctx):
    """Invalid Python in both before and after still completes analysis
    via the SyntaxError fallback path."""
    before = "def foo(:\n    if x:\n        pass\n"
    after = "def foo(:\n    if x:\n        pass\n    if y and z:\n        pass\n"
    change = FileChange(
        filename="src/broken.py",
        file_type=FileType.PYTHON,
        before_content=before,
        after_content=after,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer.analyze_performance_impact([change], codebase_ctx)
    # Should not crash; may produce complexity issues
    assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# Additional: _analyze_resource_usage detects memory patterns
# ---------------------------------------------------------------------------
async def test_analyze_resource_usage_detects_read_all(analyzer):
    """Detects .read() as memory concern."""
    code = "data = f.read()\n"
    change = FileChange(
        filename="src/io.py",
        file_type=FileType.PYTHON,
        after_content=code,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    issues = await analyzer._analyze_resource_usage(change)
    assert any(i.type == "memory_usage_concern" for i in issues)
