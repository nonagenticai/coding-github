"""Unit tests for PRChangeDetector."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.pr_change_detector import (
    PRChangeDetector,
)


def _fc(**kwargs):
    """Shortcut to build a FileChange with sensible defaults."""
    defaults = dict(
        filename="src/app.py",
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=5,
        deletions=2,
        patch="",
    )
    defaults.update(kwargs)
    return FileChange(**defaults)


@pytest.fixture
def detector():
    return PRChangeDetector()


# ---------------------------------------------------------------------------
# 1. Detect added files
# ---------------------------------------------------------------------------
def test_detect_added_files(detector):
    changes = [
        _fc(
            filename="src/new_feature.py",
            change_type=ChangeType.ADDED,
            additions=50,
            deletions=0,
        ),
    ]
    patterns = detector.detect_change_patterns(changes)
    # New file with many additions → feature addition
    assert len(patterns["feature_additions"]) > 0


# ---------------------------------------------------------------------------
# 2. Detect modified files – refactoring pattern
# ---------------------------------------------------------------------------
def test_detect_modified_files_refactoring(detector):
    changes = [
        _fc(
            filename="src/service.py",
            additions=20,
            deletions=18,
            patch="+new line\n-old line\n",
        ),
    ]
    patterns = detector.detect_change_patterns(changes)
    # High deletion-to-addition ratio → refactoring
    assert len(patterns["refactoring_indicators"]) > 0


# ---------------------------------------------------------------------------
# 3. Detect deleted files – bug fix keywords in patch
# ---------------------------------------------------------------------------
def test_detect_deleted_files_bug_fix(detector):
    changes = [
        _fc(
            filename="src/buggy.py",
            additions=3,
            deletions=3,
            patch="+    # fix: corrected off-by-one error\n-    idx = len(arr)\n+    idx = len(arr) - 1\n",
        ),
    ]
    patterns = detector.detect_change_patterns(changes)
    assert len(patterns["bug_fixes"]) > 0


# ---------------------------------------------------------------------------
# 4. Detect renamed / dependency files
# ---------------------------------------------------------------------------
def test_detect_renamed_dependency_file(detector):
    changes = [
        _fc(
            filename="requirements.txt",
            file_type=FileType.UNKNOWN,
            additions=1,
            deletions=1,
        ),
    ]
    patterns = detector.detect_change_patterns(changes)
    assert len(patterns["dependency_changes"]) > 0


# ---------------------------------------------------------------------------
# 5. Categorization – metadata extraction
# ---------------------------------------------------------------------------
def test_categorization_metadata(detector):
    changes = [
        _fc(
            filename="src/app.py",
            additions=100,
            deletions=20,
        ),
        _fc(
            filename="tests/test_app.py",
            additions=50,
            deletions=5,
        ),
        _fc(
            filename="README.md",
            file_type=FileType.MARKDOWN,
            additions=10,
            deletions=2,
        ),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert metadata["total_files_changed"] == 3
    assert metadata["total_additions"] == 160
    assert metadata["total_deletions"] == 27
    # Largest change should be app.py with 120 total
    assert metadata["largest_changes"][0]["filename"] == "src/app.py"
