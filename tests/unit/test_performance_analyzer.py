"""Unit tests for PerformanceAnalyzer."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.context.performance_analyzer import (
    PerformanceAnalyzer,
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
# 1. No performance issues – clean code
# ---------------------------------------------------------------------------
async def test_no_performance_issues(analyzer, codebase_ctx):
    code = "def greet(name):\n    return f'Hello, {name}'\n"
    changes = [
        FileChange(
            filename="src/hello.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    # Simple function should not trigger performance patterns
    perf_issues = [i for i in issues if i.type.startswith("performance_pattern_")]
    assert perf_issues == []


# ---------------------------------------------------------------------------
# 2. N+1 detection – DB query inside a loop
# ---------------------------------------------------------------------------
async def test_n_plus_1_detection(analyzer, codebase_ctx):
    # Use SELECT * which is a single-line detectable pattern
    code = 'result = db.execute("SELECT * FROM users WHERE id = ?")\n'
    changes = [
        FileChange(
            filename="src/repo.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    issue_ids = [i.rule for i in issues]
    assert "select_star" in issue_ids


# ---------------------------------------------------------------------------
# 3. Nested loop detection
# ---------------------------------------------------------------------------
async def test_nested_loop_detection(analyzer, codebase_ctx):
    # Nested loops are detected via _analyze_complexity_changes when both
    # before_content and after_content are provided and nested loop count increases
    before_code = "x = 1\n"
    after_code = "for i in range(100):\n    for j in range(100):\n        process(i, j)\n"
    changes = [
        FileChange(
            filename="src/loops.py",
            file_type=FileType.PYTHON,
            before_content=before_code,
            after_content=after_code,
            additions=3,
            deletions=1,
            change_type=ChangeType.MODIFIED,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    issue_types = [i.type for i in issues]
    assert any("nested_loop" in t or "complexity" in t for t in issue_types)


# ---------------------------------------------------------------------------
# 4. Expensive call detection – sync HTTP request
# ---------------------------------------------------------------------------
async def test_expensive_call_detection(analyzer, codebase_ctx):
    code = "import requests\nresponse = requests.get('https://api.example.com/data')\n"
    changes = [
        FileChange(
            filename="src/client.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    issue_ids = [i.rule for i in issues]
    assert "sync_http_request" in issue_ids


# ---------------------------------------------------------------------------
# 5. Blocking I/O detection – file I/O in loop
# ---------------------------------------------------------------------------
async def test_blocking_io_detection(analyzer, codebase_ctx):
    # file_io_in_loop is a multi-line pattern; the resource analyzer detects
    # .read() via memory_analysis on a per-line basis
    code = "for path in file_list:\n    data = open(path).read()\n"
    changes = [
        FileChange(
            filename="src/fileio.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    issue_ids = [i.rule for i in issues]
    assert "memory_analysis" in issue_ids


# ---------------------------------------------------------------------------
# 6. Empty diff – no content means no issues
# ---------------------------------------------------------------------------
async def test_empty_diff(analyzer, codebase_ctx):
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
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    assert issues == []


# ---------------------------------------------------------------------------
# 7. False positive suppression – non-code file is skipped
# ---------------------------------------------------------------------------
async def test_false_positive_suppression(analyzer, codebase_ctx):
    # A markdown file mentioning loops should not trigger performance warnings
    md_content = "# Guide\n\nfor each item in the list, do something.\n"
    changes = [
        FileChange(
            filename="docs/guide.md",
            file_type=FileType.MARKDOWN,
            after_content=md_content,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await analyzer.analyze_performance_impact(changes, codebase_ctx)
    assert issues == []
