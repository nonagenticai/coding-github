"""Unit tests for TestCoverageCorrelator."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.test_coverage_correlator import (
    TestCoverageCorrelator,
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
# 1. Source with matching test – correlates test to source
# ---------------------------------------------------------------------------
async def test_source_with_matching_test(correlator, codebase_ctx):
    source_code = "def calculate(x, y):\n    return x + y\n"
    test_code = (
        "import pytest\n"
        "from calculator import calculate\n"
        "\n"
        "def test_calculate():\n"
        "    assert calculate(1, 2) == 3\n"
    )
    changes = [
        FileChange(
            filename="src/calculator.py",
            file_type=FileType.PYTHON,
            after_content=source_code,
            additions=2,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="tests/test_calculator.py",
            file_type=FileType.PYTHON,
            after_content=test_code,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # Should detect test file and analyze it
    assert len(result.existing_tests) > 0
    # The test should cover 'calculator' module
    tested_modules = set()
    for tf in result.existing_tests:
        tested_modules.update(tf.tested_modules)
    assert "calculator" in tested_modules


# ---------------------------------------------------------------------------
# 2. Source without test – coverage gap detected
# ---------------------------------------------------------------------------
async def test_source_without_test(correlator, codebase_ctx):
    source_code = (
        "def process_data(data):\n"
        "    if not data:\n"
        "        raise ValueError('empty')\n"
        "    return [d * 2 for d in data]\n"
    )
    changes = [
        FileChange(
            filename="src/processor.py",
            file_type=FileType.PYTHON,
            after_content=source_code,
            additions=4,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # No test files → coverage gaps should exist
    assert len(result.coverage_gaps) > 0
    # Missing test file should be suggested
    assert len(result.missing_test_files) > 0


# ---------------------------------------------------------------------------
# 3. Test-only change – no coverage gaps, existing tests analysed
# ---------------------------------------------------------------------------
async def test_test_only_change(correlator, codebase_ctx):
    test_code = (
        "import pytest\n"
        "from mymodule import do_stuff\n"
        "\n"
        "def test_do_stuff():\n"
        "    assert do_stuff() is not None\n"
    )
    changes = [
        FileChange(
            filename="tests/test_mymodule.py",
            file_type=FileType.PYTHON,
            after_content=test_code,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # Only test file changed → no source coverage gaps
    assert result.coverage_gaps == []
    assert result.missing_test_files == []


# ---------------------------------------------------------------------------
# 4. Partial coverage – some functions tested, some not
# ---------------------------------------------------------------------------
async def test_partial_coverage(correlator, codebase_ctx):
    source_code = "def func_a():\n    return 1\n\ndef func_b():\n    return 2\n"
    test_code = (
        "import pytest\nfrom mylib import func_a\n\ndef test_func_a():\n    assert func_a() == 1\n"
    )
    changes = [
        FileChange(
            filename="src/mylib.py",
            file_type=FileType.PYTHON,
            after_content=source_code,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="tests/test_mylib.py",
            file_type=FileType.PYTHON,
            after_content=test_code,
            additions=5,
            change_type=ChangeType.MODIFIED,
            deletions=0,
            patch="",
        ),
    ]
    result = await correlator.correlate_test_coverage(changes, codebase_ctx)
    # func_b should appear as a coverage gap
    gap_symbols = [g.symbol_name for g in result.coverage_gaps]
    assert "func_b" in gap_symbols


# ---------------------------------------------------------------------------
# 5. Empty repo metadata – no changes, perfect score
# ---------------------------------------------------------------------------
async def test_empty_repo_metadata(correlator, codebase_ctx):
    result = await correlator.correlate_test_coverage([], codebase_ctx)
    assert result.coverage_gaps == []
    assert result.coverage_score == 1.0
    assert result.missing_test_files == []
