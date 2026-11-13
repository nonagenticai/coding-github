"""Core tests for RiskCalculator — compute real risk scores from real inputs."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    DiffAnalysisResult,
    DiffStats,
    FileChange,
    FileType,
    ImpactAssessment,
    PRAnalysisContext,
    PRDiff,
    RiskLevel,
    ValidationIssue,
)

from services.pr_validation.scoring.risk_calculator import (
    RiskCalculator,
    RiskDimension,
    RiskScore,
)


def _fc(filename="src/main.py", **kw):
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 10)
    kw.setdefault("deletions", 5)
    kw.setdefault("patch", "@@ -1,5 +1,10 @@")
    return FileChange(filename=filename, **kw)


def _diff_stats(files=None):
    if files:
        return DiffStats.from_files(files)
    return DiffStats(
        total_files=0,
        total_additions=0,
        total_deletions=0,
        files_added=0,
        files_deleted=0,
        files_modified=0,
        files_renamed=0,
    )


def _ctx(files=None, diff_analysis=None, impact_assessment=None, semantic_analysis=None):
    f = files or []
    return PRAnalysisContext(
        pr_diff=PRDiff(
            base_sha="abc123",
            head_sha="def456",
            files_changed=f,
            stats=_diff_stats(f),
            metadata={},
        ),
        codebase_context=CodebaseContext(
            total_files=100,
            total_lines=10000,
            main_languages=["python"],
            test_framework="pytest",
            documentation_format="markdown",
            dependencies=[],
            architecture_patterns=[],
        ),
        diff_analysis=diff_analysis,
        impact_assessment=impact_assessment,
        semantic_analysis=semantic_analysis,
    )


# ---------------------------------------------------------------------------
# RiskScore.risk_level property — threshold boundaries
# ---------------------------------------------------------------------------


class TestRiskScoreThresholds:
    """Verify risk level boundaries: LOW < 3, MEDIUM < 6, HIGH < 8, CRITICAL >= 8."""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, RiskLevel.LOW),
            (2.9, RiskLevel.LOW),
            (3.0, RiskLevel.MEDIUM),
            (5.9, RiskLevel.MEDIUM),
            (6.0, RiskLevel.HIGH),
            (7.9, RiskLevel.HIGH),
            (8.0, RiskLevel.CRITICAL),
            (10.0, RiskLevel.CRITICAL),
        ],
    )
    def test_risk_level_boundaries(self, score, expected):
        assert RiskScore(overall_score=score).risk_level == expected


# ---------------------------------------------------------------------------
# calculate_overall_risk — integration with real inputs
# ---------------------------------------------------------------------------


class TestCalculateOverallRisk:
    """End-to-end risk calculation."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_clean_pr_low_risk(self):
        ctx = _ctx(files=[_fc("app.py", additions=5, deletions=2)])
        score = self.calc.calculate_overall_risk(ctx, [])
        assert score.overall_score < 3.0
        assert score.risk_level == RiskLevel.LOW

    def test_many_security_issues_raise_security_dimension(self):
        ctx = _ctx(files=[_fc("auth.py", additions=50, deletions=30)])
        issues = [
            ValidationIssue(
                type="security_injection", severity="critical", message="SQLi", file="auth.py"
            ),
            ValidationIssue(type="security_xss", severity="high", message="XSS", file="auth.py"),
            ValidationIssue(type="security_csrf", severity="high", message="CSRF", file="auth.py"),
        ]
        score = self.calc.calculate_overall_risk(ctx, issues)
        assert score.dimension_scores[RiskDimension.SECURITY] > 4.0
        assert score.overall_score > 1.0

    def test_missing_analysis_reduces_confidence(self):
        ctx = _ctx()  # no diff_analysis, semantic_analysis, impact_assessment
        score = self.calc.calculate_overall_risk(ctx, [])
        assert score.confidence < 1.0
        # missing diff (-0.2), semantic (-0.15), impact (-0.15) = 0.5
        assert score.confidence <= 0.55

    def test_dimension_scores_capped_at_10(self):
        ctx = _ctx(files=[_fc("auth.py", additions=100, deletions=100)])
        issues = [
            ValidationIssue(
                type="security_vuln", severity="critical", message=f"v{i}", file="auth.py"
            )
            for i in range(20)
        ]
        score = self.calc.calculate_overall_risk(ctx, issues)
        assert score.dimension_scores[RiskDimension.SECURITY] <= 10.0

    def test_confidence_never_below_zero(self):
        ctx = _ctx(files=[_fc(f"f{i}.py") for i in range(40)])
        issues = [
            ValidationIssue(
                type="error_analysis", severity="low", message=f"err{i}", file=f"f{i}.py"
            )
            for i in range(20)
        ]
        score = self.calc.calculate_overall_risk(ctx, issues)
        assert score.confidence >= 0.0


# ---------------------------------------------------------------------------
# Individual dimension calculations
# ---------------------------------------------------------------------------


class TestSecurityRisk:
    """Security risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_security_files_add_risk(self):
        ctx = _ctx(
            files=[
                _fc("src/auth/handler.py"),
                _fc("src/token_service.py"),
            ]
        )
        score = self.calc._calculate_security_risk(ctx, [])
        # Two security-sensitive filenames
        assert score >= 1.0

    def test_security_issues_weighted_by_severity(self):
        ctx = _ctx(files=[_fc("app.py")])
        issues = [
            ValidationIssue(
                type="security_sqli", severity="critical", message="sql", file="app.py"
            ),
        ]
        score = self.calc._calculate_security_risk(ctx, issues)
        assert score >= 4.0  # critical weight = 4.0


class TestPerformanceRisk:
    """Performance risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_loop_in_patch_adds_risk(self):
        ctx = _ctx(files=[_fc("app.py", patch="+ for item in items:\n+   process(item)")])
        score = self.calc._calculate_performance_risk(ctx, [])
        assert score >= 0.5

    def test_db_query_in_patch_adds_risk(self):
        ctx = _ctx(files=[_fc("db.py", patch="+ result = db.query('SELECT * FROM users')")])
        score = self.calc._calculate_performance_risk(ctx, [])
        assert score >= 0.8

    def test_diff_analysis_complexity_contributes(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=5.0,
            requires_careful_review=False,
        )
        ctx = _ctx(diff_analysis=diff)
        score = self.calc._calculate_performance_risk(ctx, [])
        assert score >= 1.5  # 5.0 * 0.3


class TestMaintainabilityRisk:
    """Maintainability risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_large_change_over_500_lines(self):
        ctx = _ctx(files=[_fc("big.py", additions=400, deletions=200)])
        score = self.calc._calculate_maintainability_risk(ctx, [])
        assert score >= 2.0

    def test_many_files_over_20(self):
        files = [_fc(f"f{i}.py") for i in range(25)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_maintainability_risk(ctx, [])
        assert score >= 1.5

    def test_between_10_and_20_files(self):
        files = [_fc(f"f{i}.py") for i in range(15)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_maintainability_risk(ctx, [])
        assert score >= 0.8


class TestReliabilityRisk:
    """Reliability risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_logic_issues_weighted_1_5x(self):
        ctx = _ctx(files=[_fc("app.py")])
        issues = [
            ValidationIssue(type="logic_error", severity="high", message="bug", file="app.py")
        ]
        score = self.calc._calculate_reliability_risk(ctx, issues)
        assert score >= 3.75  # 2.5 * 1.5

    def test_removed_error_handling_increases_risk(self):
        ctx = _ctx(
            files=[
                _fc(
                    "app.py",
                    patch="-    try:\n-        do()\n-    except:\n-        handle()\n+    do()",
                )
            ]
        )
        score = self.calc._calculate_reliability_risk(ctx, [])
        assert score >= 1.0  # "try:" and "except:" each +0.5


class TestCompatibilityRisk:
    """Compatibility risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_breaking_change_weighted_1_2x(self):
        ctx = _ctx(files=[_fc("api.py")])
        issues = [
            ValidationIssue(type="breaking_change", severity="high", message="break", file="api.py")
        ]
        score = self.calc._calculate_compatibility_risk(ctx, issues)
        assert score >= 3.0  # 2.5 * 1.2

    def test_added_defs_increase_risk(self):
        ctx = _ctx(
            files=[
                _fc(
                    "api.py",
                    patch="+def new_func():\n+    pass\n+class NewClass:\n+    pass",
                )
            ]
        )
        score = self.calc._calculate_compatibility_risk(ctx, [])
        assert score >= 0.6  # 2 * 0.3


class TestComplexityRisk:
    """Complexity risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_change_complexity_from_diff(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=4.0,
            requires_careful_review=False,
        )
        ctx = _ctx(diff_analysis=diff)
        score = self.calc._calculate_complexity_risk(ctx, [])
        assert score >= 4.0

    def test_complete_rewrites_pattern(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=["CONTAINS_COMPLETE_REWRITES"],
            risk_factors=[],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        ctx = _ctx(diff_analysis=diff)
        score = self.calc._calculate_complexity_risk(ctx, [])
        assert score >= 2.0


class TestImpactScopeRisk:
    """Impact scope risk dimension."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_over_50_files(self):
        files = [_fc(f"f{i}.py") for i in range(55)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 5.0

    def test_dependency_impacts(self):
        from coding_client.models.pr_validation import DocumentationImpact, TestImpact

        impact = ImpactAssessment(
            dependency_impacts=[f"dep{i}" for i in range(20)],
            test_coverage_impact=TestImpact(
                tests_affected=[],
                coverage_change=0.0,
                new_tests_needed=[],
                obsolete_tests=[],
            ),
            documentation_impact=DocumentationImpact(
                docs_affected=[],
                outdated_docs=[],
                new_docs_needed=[],
                doc_completeness_score=0.0,
            ),
            performance_impact={},
            security_impact={},
        )
        ctx = _ctx(impact_assessment=impact)
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 2.0  # 20 * 0.1


# ---------------------------------------------------------------------------
# explain_risk_score
# ---------------------------------------------------------------------------


class TestExplainRiskScore:
    """Risk score explanations include correct recommendations."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_critical_risk_has_4_recommendations(self):
        score = RiskScore(overall_score=9.0, dimension_scores={RiskDimension.SECURITY: 9.5})
        explanation = self.calc.explain_risk_score(score)
        assert explanation["overall_risk"]["level"] == "CRITICAL"
        assert len(explanation["recommendations"]) == 4

    def test_high_risk_has_3_recommendations(self):
        score = RiskScore(overall_score=7.0, dimension_scores={RiskDimension.PERFORMANCE: 7.5})
        explanation = self.calc.explain_risk_score(score)
        assert explanation["overall_risk"]["level"] == "HIGH"
        assert len(explanation["recommendations"]) == 3

    def test_low_risk_has_0_recommendations(self):
        score = RiskScore(overall_score=1.0, dimension_scores={})
        explanation = self.calc.explain_risk_score(score)
        assert len(explanation["recommendations"]) == 0

    def test_dimension_breakdown_contribution(self):
        score = RiskScore(overall_score=2.5, dimension_scores={RiskDimension.SECURITY: 10.0})
        explanation = self.calc.explain_risk_score(score)
        entry = explanation["dimension_breakdown"]["security"]
        assert entry["score"] == 10.0
        assert entry["weight"] == 0.25
        assert entry["contribution"] == 2.5

    @pytest.mark.parametrize(
        "score_val,expected",
        [(9.0, "critical"), (7.0, "high"), (5.0, "medium"), (1.0, "low")],
    )
    def test_score_to_level(self, score_val, expected):
        assert self.calc._score_to_level(score_val) == expected
