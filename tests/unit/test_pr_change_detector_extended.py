"""Extended tests for PRChangeDetector targeting uncovered lines.

Uncovered: 52, 64, 91, 116, 120, 150-162, 165-166, 207-208, 212-221, 224-226,
           255, 265, 275, 341-349, 447, 453
"""

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
# Line 52: analyze_test_coverage_impact — test file in changes
# ---------------------------------------------------------------------------
def test_test_coverage_impact_test_file_detected(detector):
    """Test files in changes appear in tests_affected."""
    changes = [
        _fc(filename="tests/test_utils.py", additions=10, deletions=0),
    ]
    result = detector.analyze_test_coverage_impact(changes, [])
    assert "tests/test_utils.py" in result.tests_affected


# ---------------------------------------------------------------------------
# Line 64: analyze_test_coverage_impact — affected files generate test suggestions
# ---------------------------------------------------------------------------
def test_test_coverage_impact_affected_files_suggest_tests(detector):
    """Non-test affected files generate test suggestions without duplicates."""
    changes = []
    affected_files = ["src/service.py", "src/utils.py"]
    result = detector.analyze_test_coverage_impact(changes, affected_files)
    assert len(result.new_tests_needed) >= 2
    # No duplicates
    assert len(result.new_tests_needed) == len(set(result.new_tests_needed))


def test_test_coverage_impact_affected_files_no_dup_with_changes(detector):
    """When a change already suggests a test, affected_files don't duplicate it."""
    changes = [
        _fc(filename="src/service.py", additions=5, deletions=0),
    ]
    affected_files = ["src/service.py"]
    result = detector.analyze_test_coverage_impact(changes, affected_files)
    # test_service.py should appear only once
    count = sum(1 for t in result.new_tests_needed if "test_service" in t)
    assert count == 1


# ---------------------------------------------------------------------------
# Line 91: analyze_documentation_impact — doc file in changes
# ---------------------------------------------------------------------------
def test_documentation_impact_doc_file_detected(detector):
    """Documentation files in changes appear in docs_affected."""
    changes = [
        _fc(filename="README.md", file_type=FileType.MARKDOWN, additions=3, deletions=0),
    ]
    result = detector.analyze_documentation_impact(changes, [])
    assert "README.md" in result.docs_affected


# ---------------------------------------------------------------------------
# Lines 116, 120: analyze_documentation_impact — code changes trigger outdated docs and new docs needed
# ---------------------------------------------------------------------------
def test_documentation_impact_code_changes_suggest_docs(detector):
    """Code changes with net additions generate new_docs_needed entries."""
    changes = [
        _fc(filename="src/new_feature.py", additions=20, deletions=2),
    ]
    result = detector.analyze_documentation_impact(changes, [])
    assert len(result.outdated_docs) > 0
    assert len(result.new_docs_needed) > 0
    assert result.doc_completeness_score < 1.0


def test_documentation_impact_code_changes_no_net_addition(detector):
    """Code changes with deletions >= additions don't trigger new_docs_needed."""
    changes = [
        _fc(filename="src/cleanup.py", additions=5, deletions=10),
    ]
    result = detector.analyze_documentation_impact(changes, [])
    assert len(result.new_docs_needed) == 0


# ---------------------------------------------------------------------------
# Lines 150-162: analyze_performance_impact — performance patterns in patches
# ---------------------------------------------------------------------------
def test_performance_impact_loop_detected(detector):
    """Loop patterns in patches trigger performance concerns."""
    changes = [
        _fc(
            filename="src/process.py",
            additions=2,
            deletions=0,
            patch="+for item in collection:\n+    process(item)\n",
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "medium"
    assert len(result["concerns"]) > 0
    assert len(result["recommendations"]) > 0


def test_performance_impact_while_loop(detector):
    """While loop patterns trigger performance concerns."""
    changes = [
        _fc(
            filename="src/worker.py",
            additions=2,
            deletions=0,
            patch="+while not done:\n+    work()\n",
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "medium"


def test_performance_impact_js_map_filter_reduce(detector):
    """JS array methods .map(), .filter(), .reduce() trigger concerns."""
    changes = [
        _fc(
            filename="src/transform.js",
            file_type=FileType.JAVASCRIPT,
            additions=3,
            deletions=0,
            patch="+const result = data.map(x => x * 2)\n+const filtered = data.filter(x => x > 0)\n+const sum = data.reduce((a, b) => a + b)\n",
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "medium"
    assert len(result["concerns"]) >= 3


def test_performance_impact_sql_query(detector):
    """SQL SELECT patterns trigger concerns."""
    changes = [
        _fc(
            filename="src/db.py",
            additions=1,
            deletions=0,
            patch='+query = "SELECT * FROM users WHERE active = 1"\n',
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "medium"


def test_performance_impact_async_await(detector):
    """Async/await patterns trigger concerns."""
    changes = [
        _fc(
            filename="src/async_handler.py",
            additions=2,
            deletions=0,
            patch="+async def handle():\n+    result = await fetch_data()\n",
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "medium"


def test_performance_impact_no_patch(detector):
    """Changes without patches don't trigger concerns."""
    changes = [
        _fc(filename="src/clean.py", additions=5, deletions=0),
    ]
    result = detector.analyze_performance_impact(changes)
    assert result["risk_level"] == "low"
    assert len(result["concerns"]) == 0


# ---------------------------------------------------------------------------
# Lines 165-166: analyze_performance_impact — recommendations generated
# ---------------------------------------------------------------------------
def test_performance_impact_recommendations(detector):
    """When concerns exist, recommendations are generated."""
    changes = [
        _fc(
            filename="src/heavy.py",
            additions=2,
            deletions=0,
            patch="+for i in range(1000):\n+    process(i)\n",
        ),
    ]
    result = detector.analyze_performance_impact(changes)
    assert "performance testing" in result["recommendations"][0].lower()
    assert "algorithmic complexity" in result["recommendations"][1].lower()


# ---------------------------------------------------------------------------
# Lines 207-208: analyze_security_impact — security-sensitive filenames
# ---------------------------------------------------------------------------
def test_security_impact_sensitive_filename(detector):
    """Files with auth/security in name trigger high risk."""
    changes = [
        _fc(filename="src/auth/login.py", additions=5, deletions=0),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] == "high"
    assert any("Security-sensitive file" in c for c in result["concerns"])


def test_security_impact_security_filename(detector):
    """Files with 'security' in name trigger high risk."""
    changes = [
        _fc(filename="src/security/middleware.py", additions=3, deletions=0),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] == "high"


def test_security_impact_permission_filename(detector):
    """Files with 'permission' in name trigger high risk."""
    changes = [
        _fc(filename="src/permissions.py", additions=3, deletions=0),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# Lines 212-221: analyze_security_impact — security patterns in patches
# ---------------------------------------------------------------------------
def test_security_impact_password_in_patch(detector):
    """Patch containing 'password' triggers security concern."""
    changes = [
        _fc(
            filename="src/user.py",
            additions=1,
            deletions=0,
            patch="+password = request.form['password']\n",
        ),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] in ("medium", "high")
    assert len(result["concerns"]) > 0


def test_security_impact_eval_in_patch(detector):
    """Patch containing 'eval' triggers security concern."""
    changes = [
        _fc(
            filename="src/dynamic.py",
            additions=1,
            deletions=0,
            patch="+result = eval(user_input)\n",
        ),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] in ("medium", "high")


def test_security_impact_deleted_line(detector):
    """Deleted lines with security patterns also trigger concerns."""
    changes = [
        _fc(
            filename="src/cleanup.py",
            additions=0,
            deletions=1,
            patch="-secret_key = 'abc123'\n",
        ),
    ]
    result = detector.analyze_security_impact(changes)
    assert len(result["concerns"]) > 0


# ---------------------------------------------------------------------------
# Lines 224-226: analyze_security_impact — recommendations
# ---------------------------------------------------------------------------
def test_security_impact_recommendations(detector):
    """When security concerns exist, recommendations are generated."""
    changes = [
        _fc(
            filename="src/auth_handler.py",
            additions=5,
            deletions=0,
            patch="+token = generate_token(secret)\n",
        ),
    ]
    result = detector.analyze_security_impact(changes)
    assert len(result["recommendations"]) >= 3
    rec_text = " ".join(result["recommendations"]).lower()
    assert "security review" in rec_text
    assert "security testing" in rec_text
    assert "injection" in rec_text


def test_security_impact_no_concerns(detector):
    """Clean changes with no security indicators produce low risk."""
    changes = [
        _fc(
            filename="src/math_utils.py",
            additions=5,
            deletions=0,
            patch="+def add(a, b):\n+    return a + b\n",
        ),
    ]
    result = detector.analyze_security_impact(changes)
    assert result["risk_level"] == "low"
    assert len(result["concerns"]) == 0
    assert len(result["recommendations"]) == 0


# ---------------------------------------------------------------------------
# Line 255: detect_change_patterns — API file detection
# ---------------------------------------------------------------------------
def test_detect_patterns_api_file(detector):
    """Files with api-related names are detected."""
    changes = [
        _fc(filename="src/api/endpoints.py", additions=5, deletions=0),
    ]
    patterns = detector.detect_change_patterns(changes)
    assert len(patterns["api_changes"]) > 0
    assert patterns["api_changes"][0]["impact"] == "high"


# ---------------------------------------------------------------------------
# Line 265: detect_change_patterns — database file detection
# ---------------------------------------------------------------------------
def test_detect_patterns_database_file(detector):
    """Files with db-related names are detected."""
    changes = [
        _fc(filename="src/migrations/001_initial.py", additions=20, deletions=0),
    ]
    patterns = detector.detect_change_patterns(changes)
    assert len(patterns["database_changes"]) > 0


# ---------------------------------------------------------------------------
# Line 275: detect_change_patterns — config file detection
# ---------------------------------------------------------------------------
def test_detect_patterns_config_file(detector):
    """Configuration files are detected."""
    changes = [
        _fc(filename="config/settings.yaml", file_type=FileType.YAML, additions=3, deletions=0),
    ]
    patterns = detector.detect_change_patterns(changes)
    assert len(patterns["config_changes"]) > 0


# ---------------------------------------------------------------------------
# Lines 341-349: _suggest_test_file — JS/TS files
# ---------------------------------------------------------------------------
def test_suggest_test_file_javascript(detector):
    """JS file gets .test.js suggestion."""
    result = detector._suggest_test_file("src/app.js")
    assert result == "app.test.js"


def test_suggest_test_file_typescript(detector):
    """TS file gets .test.ts suggestion."""
    result = detector._suggest_test_file("src/service.ts")
    assert result == "service.test.ts"


def test_suggest_test_file_tsx(detector):
    """TSX file gets .test.tsx suggestion."""
    result = detector._suggest_test_file("src/component.tsx")
    assert result == "component.test.tsx"


def test_suggest_test_file_jsx(detector):
    """JSX file gets .test.jsx suggestion."""
    result = detector._suggest_test_file("src/widget.jsx")
    assert result == "widget.test.jsx"


def test_suggest_test_file_python(detector):
    """Python file gets test_ prefix suggestion."""
    result = detector._suggest_test_file("src/utils.py")
    assert result == "test_utils.py"


def test_suggest_test_file_unknown(detector):
    """Unknown file extension returns None."""
    result = detector._suggest_test_file("src/data.csv")
    assert result is None


# ---------------------------------------------------------------------------
# Lines 447, 453: extract_change_metadata — change distribution categories
# ---------------------------------------------------------------------------
def test_extract_metadata_small_change(detector):
    """Changes < 10 lines are categorized as 'small'."""
    changes = [
        _fc(filename="src/tiny.py", additions=3, deletions=1),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert "small" in metadata["change_distribution"]


def test_extract_metadata_medium_change(detector):
    """Changes 10-49 lines are categorized as 'medium'."""
    changes = [
        _fc(filename="src/mid.py", additions=20, deletions=10),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert "medium" in metadata["change_distribution"]


def test_extract_metadata_large_change(detector):
    """Changes 50-199 lines are categorized as 'large'."""
    changes = [
        _fc(filename="src/big.py", additions=100, deletions=50),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert "large" in metadata["change_distribution"]


def test_extract_metadata_very_large_change(detector):
    """Changes >= 200 lines are categorized as 'very_large'."""
    changes = [
        _fc(filename="src/huge.py", additions=150, deletions=100),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert "very_large" in metadata["change_distribution"]


def test_extract_metadata_multiple_file_types(detector):
    """file_types dict counts by type."""
    changes = [
        _fc(filename="a.py", additions=5, deletions=0),
        _fc(filename="b.py", additions=3, deletions=0),
        _fc(filename="c.js", file_type=FileType.JAVASCRIPT, additions=10, deletions=0),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert metadata["file_types"]["python"] == 2
    assert metadata["file_types"]["javascript"] == 1


def test_extract_metadata_largest_changes_sorted(detector):
    """largest_changes are sorted by total changes descending."""
    changes = [
        _fc(filename="small.py", additions=2, deletions=1),
        _fc(filename="big.py", additions=50, deletions=30),
        _fc(filename="mid.py", additions=10, deletions=5),
    ]
    metadata = detector.extract_change_metadata(changes)
    assert metadata["largest_changes"][0]["filename"] == "big.py"
    assert metadata["largest_changes"][0]["total"] == 80


# ---------------------------------------------------------------------------
# _is_refactoring_change edge cases
# ---------------------------------------------------------------------------
def test_is_refactoring_no_additions(detector):
    """Zero additions -> not refactoring."""
    change = _fc(filename="x.py", additions=0, deletions=5)
    assert not detector._is_refactoring_change(change)


def test_is_refactoring_low_ratio(detector):
    """Low deletion/addition ratio -> not refactoring."""
    change = _fc(filename="x.py", additions=20, deletions=5)
    assert not detector._is_refactoring_change(change)


# ---------------------------------------------------------------------------
# _is_feature_addition edge cases
# ---------------------------------------------------------------------------
def test_is_feature_addition_true(detector):
    """Additions > 2*deletions and > 10 -> feature addition."""
    change = _fc(filename="x.py", additions=30, deletions=5)
    assert detector._is_feature_addition(change)


def test_is_feature_addition_false_low_additions(detector):
    """Additions <= 10 -> not a feature addition."""
    change = _fc(filename="x.py", additions=8, deletions=0)
    assert not detector._is_feature_addition(change)


# ---------------------------------------------------------------------------
# _is_bug_fix edge cases
# ---------------------------------------------------------------------------
def test_is_bug_fix_no_patch(detector):
    """No patch content -> not a bug fix."""
    change = _fc(filename="x.py", additions=3, deletions=1)
    assert not detector._is_bug_fix(change)


def test_is_bug_fix_no_keywords(detector):
    """Patch without bug keywords -> not a bug fix."""
    change = _fc(
        filename="x.py",
        additions=1,
        deletions=0,
        patch="+new_feature = True\n",
    )
    assert not detector._is_bug_fix(change)


# ---------------------------------------------------------------------------
# _is_dependency_file
# ---------------------------------------------------------------------------
def test_is_dependency_file_various(detector):
    """Various dependency file names are detected."""
    assert detector._is_dependency_file("requirements.txt")
    assert detector._is_dependency_file("package.json")
    assert detector._is_dependency_file("pyproject.toml")
    assert detector._is_dependency_file("Cargo.toml")
    assert not detector._is_dependency_file("src/main.py")


# ---------------------------------------------------------------------------
# _is_database_file
# ---------------------------------------------------------------------------
def test_is_database_file_various(detector):
    """Various database file patterns are detected."""
    assert detector._is_database_file("models/user.py")
    assert detector._is_database_file("migrations/001.sql")
    assert detector._is_database_file("schema.py")
    assert not detector._is_database_file("src/utils.py")
