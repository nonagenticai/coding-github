"""Extended unit tests for ImpactScorer — covering uncovered lines."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import ChangeType, FileChange, FileType

from services.pr_validation.analyzers.impact.impact_scorer import (
    ImpactResult,
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
# _calculate_impact_score: circular_dependencies > 5 → score *= 1.5  (line 139)
# ---------------------------------------------------------------------------
class TestCalculateImpactScoreMultipliers:
    def test_circular_deps_gt_5_multiplier(self, scorer):
        """More than 5 circular deps triggers the 1.5× multiplier."""
        circ = [("a", "b")] * 6
        score = scorer._calculate_impact_score(
            affected_files={"f1"},
            affected_functions=set(),
            affected_classes=set(),
            affected_modules=set(),
            circular_dependencies=circ,
            missing_dependencies=[],
        )
        # Base: 1*0.1 + 6*0.3 = 1.9 → *1.5 = 2.85
        assert score == pytest.approx(2.85, abs=0.01)

    def test_missing_deps_gt_10_multiplier(self, scorer):
        """More than 10 missing deps triggers the 1.3× multiplier."""
        missing = [f"dep_{i}" for i in range(11)]
        score = scorer._calculate_impact_score(
            affected_files=set(),
            affected_functions=set(),
            affected_classes=set(),
            affected_modules=set(),
            circular_dependencies=[],
            missing_dependencies=missing,
        )
        # Base: 11*0.15 = 1.65 → *1.3 = 2.145
        assert score == pytest.approx(2.145, abs=0.01)

    def test_both_multipliers_applied(self, scorer):
        """Both multipliers stack when both thresholds exceeded."""
        circ = [("a", "b")] * 7
        missing = [f"dep_{i}" for i in range(12)]
        score = scorer._calculate_impact_score(
            affected_files=set(),
            affected_functions=set(),
            affected_classes=set(),
            affected_modules=set(),
            circular_dependencies=circ,
            missing_dependencies=missing,
        )
        # Base: 7*0.3 + 12*0.15 = 2.1 + 1.8 = 3.9 → *1.5 → *1.3 = 7.605
        assert score == pytest.approx(7.605, abs=0.01)

    def test_score_capped_at_10(self, scorer):
        """Score is capped at 10.0 even with huge inputs."""
        circ = [("a", "b")] * 50
        missing = [f"dep_{i}" for i in range(50)]
        score = scorer._calculate_impact_score(
            affected_files=set(f"f{i}" for i in range(100)),
            affected_functions=set(f"fn{i}" for i in range(100)),
            affected_classes=set(f"C{i}" for i in range(100)),
            affected_modules=set(f"m{i}" for i in range(100)),
            circular_dependencies=circ,
            missing_dependencies=missing,
        )
        assert score == 10.0


# ---------------------------------------------------------------------------
# calculate_risk_level: lines 157-164
# ---------------------------------------------------------------------------
class TestCalculateRiskLevel:
    def test_low(self, scorer):
        assert scorer.calculate_risk_level(0.0) == "low"
        assert scorer.calculate_risk_level(2.0) == "low"

    def test_medium(self, scorer):
        assert scorer.calculate_risk_level(2.1) == "medium"
        assert scorer.calculate_risk_level(5.0) == "medium"

    def test_high(self, scorer):
        assert scorer.calculate_risk_level(5.1) == "high"
        assert scorer.calculate_risk_level(8.0) == "high"

    def test_critical(self, scorer):
        assert scorer.calculate_risk_level(8.1) == "critical"
        assert scorer.calculate_risk_level(10.0) == "critical"


# ---------------------------------------------------------------------------
# calculate_complexity_score: lines 184, 186, 188 (complexity branches)
# ---------------------------------------------------------------------------
class TestCalculateComplexityScore:
    def test_simple_complexity(self, scorer):
        changes = [
            _fc(filename="a.py", additions=5, deletions=3),
        ]
        result = scorer.calculate_complexity_score(changes)
        assert result["complexity_level"] == "simple"
        assert result["average_change_size"] == 8

    def test_moderate_complexity(self, scorer):
        changes = [
            _fc(filename="a.py", additions=20, deletions=15),
        ]
        result = scorer.calculate_complexity_score(changes)
        assert result["complexity_level"] == "moderate"

    def test_complex_complexity(self, scorer):
        changes = [
            _fc(filename="a.py", additions=50, deletions=30),
        ]
        result = scorer.calculate_complexity_score(changes)
        assert result["complexity_level"] == "complex"

    def test_very_complex_complexity(self, scorer):
        changes = [
            _fc(filename="a.py", additions=200, deletions=50),
        ]
        result = scorer.calculate_complexity_score(changes)
        assert result["complexity_level"] == "very_complex"

    def test_empty_changes(self, scorer):
        result = scorer.calculate_complexity_score([])
        assert result["complexity_level"] == "simple"
        assert result["average_change_size"] == 0
        assert result["largest_change"] == 0
        assert result["smallest_change"] == 0
        assert result["median_change"] == 0

    def test_multiple_files_metrics(self, scorer):
        changes = [
            _fc(filename="a.py", additions=10, deletions=0),
            _fc(filename="b.py", additions=50, deletions=0),
            _fc(filename="c.py", additions=30, deletions=0),
        ]
        result = scorer.calculate_complexity_score(changes)
        assert result["total_files_changed"] == 3
        assert result["total_lines_changed"] == 90
        assert result["largest_change"] == 50
        assert result["smallest_change"] == 10


# ---------------------------------------------------------------------------
# calculate_test_impact_score: lines 224-243
# ---------------------------------------------------------------------------
class TestCalculateTestImpactScore:
    def test_low_impact(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=["test_a.py"],
            new_tests_needed=[],
            coverage_change=0.0,
        )
        assert result["impact_level"] == "low"
        assert result["test_impact_score"] == pytest.approx(0.1, abs=0.01)

    def test_medium_impact(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=["t1", "t2", "t3", "t4", "t5"],
            new_tests_needed=["n1", "n2", "n3", "n4", "n5"],
            coverage_change=0.0,
        )
        # score = 5*0.1 + 5*0.2 = 0.5 + 1.0 = 1.5 → medium
        assert result["impact_level"] == "medium"

    def test_high_impact(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=["t1"] * 10,
            new_tests_needed=["n1"] * 10,
            coverage_change=-0.05,
        )
        # score = 10*0.1 + 10*0.2 + 0.05*2 = 1.0 + 2.0 + 0.1 = 3.1 → high
        assert result["impact_level"] == "high"

    def test_critical_impact(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=["t1"] * 20,
            new_tests_needed=["n1"] * 15,
            coverage_change=-0.5,
        )
        # score = 20*0.1 + 15*0.2 + 0.5*2 = 2.0 + 3.0 + 1.0 = 6.0 → critical
        assert result["impact_level"] == "critical"

    def test_coverage_decrease_penalty(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=[],
            new_tests_needed=[],
            coverage_change=-0.3,
        )
        # score = 0 + 0 + 0.3*2 = 0.6 → low
        assert result["test_impact_score"] == pytest.approx(0.6, abs=0.01)

    def test_positive_coverage_no_penalty(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=[],
            new_tests_needed=[],
            coverage_change=0.5,
        )
        assert result["test_impact_score"] == 0.0

    def test_score_capped_at_10(self, scorer):
        result = scorer.calculate_test_impact_score(
            tests_affected=["t"] * 100,
            new_tests_needed=["n"] * 100,
            coverage_change=-5.0,
        )
        assert result["test_impact_score"] == 10.0

    def test_result_keys(self, scorer):
        result = scorer.calculate_test_impact_score(["a"], ["b"], -0.1)
        assert "test_impact_score" in result
        assert "impact_level" in result
        assert "tests_affected_count" in result
        assert "new_tests_needed_count" in result
        assert "coverage_change" in result
        assert "recommendations" in result


# ---------------------------------------------------------------------------
# _generate_file_recommendations: lines 259, 263
# ---------------------------------------------------------------------------
class TestGenerateFileRecommendations:
    def test_more_than_20_files(self, scorer):
        files = {f"file_{i}.py" for i in range(25)}
        recs = scorer._generate_file_recommendations(files)
        assert any("smaller PRs" in r for r in recs)

    def test_more_than_10_files(self, scorer):
        files = {f"file_{i}.py" for i in range(15)}
        recs = scorer._generate_file_recommendations(files)
        assert any("comprehensive testing" in r for r in recs)

    def test_few_files_no_recommendation(self, scorer):
        files = {"a.py", "b.py"}
        recs = scorer._generate_file_recommendations(files)
        # Only critical-file recommendations possible
        assert not any("smaller PRs" in r for r in recs)
        assert not any("comprehensive testing" in r for r in recs)


# ---------------------------------------------------------------------------
# _generate_function_recommendations: lines 281, 285
# ---------------------------------------------------------------------------
class TestGenerateFunctionRecommendations:
    def test_more_than_50_functions(self, scorer):
        funcs = {f"func_{i}" for i in range(55)}
        recs = scorer._generate_function_recommendations(funcs)
        assert any("architectural review" in r for r in recs)

    def test_more_than_20_functions(self, scorer):
        funcs = {f"func_{i}" for i in range(25)}
        recs = scorer._generate_function_recommendations(funcs)
        assert any("documentation updates" in r for r in recs)

    def test_few_functions_no_recommendation(self, scorer):
        funcs = {"a", "b"}
        recs = scorer._generate_function_recommendations(funcs)
        assert recs == []


# ---------------------------------------------------------------------------
# _generate_dependency_recommendations: lines 301, 306, 318, 323
# ---------------------------------------------------------------------------
class TestGenerateDependencyRecommendations:
    def test_circular_deps_gt_10(self, scorer):
        circ = [("a", "b")] * 11
        recs = scorer._generate_dependency_recommendations(circ, [])
        assert any("immediate refactoring" in r for r in recs)

    def test_circular_deps_gt_5(self, scorer):
        circ = [("a", "b")] * 7
        recs = scorer._generate_dependency_recommendations(circ, [])
        assert any("consider refactoring" in r for r in recs)

    def test_circular_deps_few(self, scorer):
        circ = [("a", "b"), ("c", "d")]
        recs = scorer._generate_dependency_recommendations(circ, [])
        assert any("monitor" in r.lower() for r in recs)

    def test_missing_deps_gt_15(self, scorer):
        missing = [f"dep_{i}" for i in range(20)]
        recs = scorer._generate_dependency_recommendations([], missing)
        assert any("properly declared" in r for r in recs)

    def test_missing_deps_gt_5(self, scorer):
        missing = [f"dep_{i}" for i in range(8)]
        recs = scorer._generate_dependency_recommendations([], missing)
        assert any("verify dependency" in r for r in recs)

    def test_missing_deps_few(self, scorer):
        missing = ["dep_1", "dep_2"]
        recs = scorer._generate_dependency_recommendations([], missing)
        assert any("import statements" in r for r in recs)

    def test_no_issues(self, scorer):
        recs = scorer._generate_dependency_recommendations([], [])
        assert recs == []


# ---------------------------------------------------------------------------
# _generate_test_recommendations: lines 340-357
# ---------------------------------------------------------------------------
class TestGenerateTestRecommendations:
    def test_new_tests_needed(self, scorer):
        recs = scorer._generate_test_recommendations([], ["test_new.py"], 0.0)
        assert any("Create" in r for r in recs)

    def test_coverage_decreased(self, scorer):
        recs = scorer._generate_test_recommendations([], [], -0.15)
        assert any("Coverage decreased" in r for r in recs)

    def test_many_tests_affected(self, scorer):
        tests = [f"test_{i}.py" for i in range(12)]
        recs = scorer._generate_test_recommendations(tests, [], 0.0)
        assert any("still pass" in r for r in recs)

    def test_no_issues(self, scorer):
        recs = scorer._generate_test_recommendations([], [], 0.0)
        assert recs == []

    def test_coverage_slightly_decreased_no_rec(self, scorer):
        """Coverage decrease <= 0.1 should not trigger recommendation."""
        recs = scorer._generate_test_recommendations([], [], -0.05)
        assert not any("Coverage decreased" in r for r in recs)


# ---------------------------------------------------------------------------
# generate_summary_metrics: line 388
# ---------------------------------------------------------------------------
class TestGenerateSummaryMetrics:
    def test_summary_metrics_structure(self, scorer):
        result = ImpactResult(
            affected_files=["a.py", "b.py"],
            affected_functions=["func_a"],
            affected_classes=["ClassA"],
            affected_modules=["mod_a"],
            impact_score=3.5,
            circular_dependencies=[("a", "b")],
            missing_dependencies=["dep1"],
            recommendations=["rec1"],
        )
        metrics = scorer.generate_summary_metrics(result)
        assert metrics["overall_impact_score"] == 3.5
        assert metrics["risk_level"] == "medium"
        assert metrics["affected_components"]["files"] == 2
        assert metrics["affected_components"]["functions"] == 1
        assert metrics["affected_components"]["classes"] == 1
        assert metrics["affected_components"]["modules"] == 1
        assert metrics["dependency_issues"]["circular_dependencies"] == 1
        assert metrics["dependency_issues"]["missing_dependencies"] == 1
        assert metrics["recommendation_count"] == 1
        assert "critical_issues" in metrics


# ---------------------------------------------------------------------------
# _count_critical_issues: lines 407-423
# ---------------------------------------------------------------------------
class TestCountCriticalIssues:
    def test_no_critical_issues(self, scorer):
        result = ImpactResult(
            affected_files=["a.py"],
            affected_functions=["f"],
            affected_classes=[],
            affected_modules=[],
            impact_score=1.0,
            circular_dependencies=[],
            missing_dependencies=[],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) == 0

    def test_circular_deps_critical(self, scorer):
        result = ImpactResult(
            affected_files=[],
            affected_functions=[],
            affected_classes=[],
            affected_modules=[],
            impact_score=5.0,
            circular_dependencies=[("a", "b")] * 11,
            missing_dependencies=[],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) >= 1

    def test_missing_deps_critical(self, scorer):
        result = ImpactResult(
            affected_files=[],
            affected_functions=[],
            affected_classes=[],
            affected_modules=[],
            impact_score=5.0,
            circular_dependencies=[],
            missing_dependencies=[f"d{i}" for i in range(16)],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) >= 1

    def test_many_files_critical(self, scorer):
        result = ImpactResult(
            affected_files=[f"f{i}" for i in range(51)],
            affected_functions=[],
            affected_classes=[],
            affected_modules=[],
            impact_score=5.0,
            circular_dependencies=[],
            missing_dependencies=[],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) >= 1

    def test_many_functions_critical(self, scorer):
        result = ImpactResult(
            affected_files=[],
            affected_functions=[f"fn{i}" for i in range(101)],
            affected_classes=[],
            affected_modules=[],
            impact_score=5.0,
            circular_dependencies=[],
            missing_dependencies=[],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) >= 1

    def test_all_critical_issues(self, scorer):
        result = ImpactResult(
            affected_files=[f"f{i}" for i in range(51)],
            affected_functions=[f"fn{i}" for i in range(101)],
            affected_classes=[],
            affected_modules=[],
            impact_score=10.0,
            circular_dependencies=[("a", "b")] * 11,
            missing_dependencies=[f"d{i}" for i in range(16)],
            recommendations=[],
        )
        assert scorer._count_critical_issues(result) == 4
