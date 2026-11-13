"""Unit tests for ImpactScorer."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.impact_scorer import (
    ImpactScorer,
)


def _fc(filename="src/main.py", **kw):
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 10)
    kw.setdefault("deletions", 5)
    kw.setdefault("patch", "@@ -1,5 +1,10 @@")
    return FileChange(filename=filename, **kw)


@pytest.fixture
def scorer():
    return ImpactScorer()


# ---------------------------------------------------------------------------
# 1. Score by file types – critical files boost score
# ---------------------------------------------------------------------------
def test_score_by_file_types(scorer):
    changes = [
        _fc(filename="src/main.py", additions=10, deletions=5),
        _fc(filename="src/config.py", additions=3, deletions=1),
    ]
    result = scorer.analyze_dependency_impacts(
        changes=changes,
        affected_files={"src/main.py", "src/config.py", "src/utils.py"},
        affected_functions={"main", "configure"},
        affected_classes={"AppConfig"},
        affected_modules={"main", "config"},
        circular_dependencies=[],
        missing_dependencies=[],
    )
    assert result.impact_score > 0
    # Critical files (main, config) should trigger recommendations
    assert any("critical" in r.lower() or "Critical" in r for r in result.recommendations)


# ---------------------------------------------------------------------------
# 2. Score by change size – large changes produce higher complexity
# ---------------------------------------------------------------------------
def test_score_by_change_size(scorer):
    changes = [
        _fc(filename="big.py", additions=200, deletions=50),
    ]
    complexity = scorer.calculate_complexity_score(changes)
    assert complexity["complexity_level"] in ("complex", "very_complex")
    assert complexity["total_lines_changed"] == 250


# ---------------------------------------------------------------------------
# 3. Combined scoring – circular deps + missing deps amplify score
# ---------------------------------------------------------------------------
def test_combined_scoring(scorer):
    changes = [
        _fc(filename="src/a.py", additions=5, deletions=2),
    ]
    result = scorer.analyze_dependency_impacts(
        changes=changes,
        affected_files={"src/a.py", "src/b.py"},
        affected_functions={"func_a"},
        affected_classes=set(),
        affected_modules={"a"},
        circular_dependencies=[("src/a.py", "src/b.py"), ("src/b.py", "src/a.py")],
        missing_dependencies=["src/a.py -> missing_module"],
    )
    # Score should be higher due to circular + missing deps
    assert result.impact_score > 0
    assert len(result.circular_dependencies) == 2
    assert len(result.missing_dependencies) == 1


# ---------------------------------------------------------------------------
# 4. Empty PR – no changes means minimal impact
# ---------------------------------------------------------------------------
def test_empty_pr(scorer):
    result = scorer.analyze_dependency_impacts(
        changes=[],
        affected_files=set(),
        affected_functions=set(),
        affected_classes=set(),
        affected_modules=set(),
        circular_dependencies=[],
        missing_dependencies=[],
    )
    assert result.impact_score == 0.0
    assert result.affected_files == []
    assert result.recommendations == []
