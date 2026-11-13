"""Core tests for PRValidationRecommendationGenerator — generate real recommendations."""

from __future__ import annotations

from coding_client.models.pr_validation import (
    ChangeSeverity,
    ChangeType,
    CodebaseContext,
    DiffAnalysisResult,
    DiffStats,
    DocumentationImpact,
    FileChange,
    FileChangeAnalysis,
    FileType,
    ImpactAssessment,
    PRAnalysisContext,
    PRDiff,
    Recommendation,
    RiskLevel,
    TestImpact,
    ValidationIssue,
    ValidationResult,
)

from services.pr_validation.scoring.recommendation_generator import (
    PRValidationRecommendationGenerator,
)
from services.pr_validation.scoring.risk_calculator import RiskDimension, RiskScore


def _fc(filename="src/main.py", **kw):
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 10)
    kw.setdefault("deletions", 5)
    kw.setdefault("patch", "@@ -1,5 +1,10 @@")
    return FileChange(filename=filename, **kw)


def _result(issues=None, risk_level=RiskLevel.LOW, score=90.0):
    return ValidationResult(
        validation_type="comprehensive",
        passed=True,
        score=score,
        risk_level=risk_level,
        issues=issues or [],
        recommendations=[],
    )


def _codebase_context():
    return CodebaseContext(
        total_files=100,
        total_lines=10000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


def _pr_diff(files=None):
    files = files or []
    return PRDiff(
        base_sha="aaa",
        head_sha="bbb",
        files_changed=files,
        stats=DiffStats.from_files(files),
        metadata={},
    )


def _diff_analysis(**kw):
    kw.setdefault("file_analyses", [])
    kw.setdefault("overall_patterns", [])
    kw.setdefault("risk_factors", [])
    kw.setdefault("change_complexity", 0.0)
    kw.setdefault("requires_careful_review", False)
    return DiffAnalysisResult(**kw)


def _test_impact(**kw):
    kw.setdefault("tests_affected", [])
    kw.setdefault("coverage_change", 0.0)
    kw.setdefault("new_tests_needed", [])
    kw.setdefault("obsolete_tests", [])
    return TestImpact(**kw)


def _impact_assessment(**kw):
    kw.setdefault("dependency_impacts", [])
    kw.setdefault(
        "test_coverage_impact",
        _test_impact(),
    )
    kw.setdefault(
        "documentation_impact",
        DocumentationImpact(
            docs_affected=[], outdated_docs=[], new_docs_needed=[], doc_completeness_score=0.0
        ),
    )
    kw.setdefault("performance_impact", {})
    kw.setdefault("security_impact", {})
    return ImpactAssessment(**kw)


def _file_change_analysis(filename="core.py", **kw):
    kw.setdefault("change_severity", ChangeSeverity.MINOR)
    kw.setdefault("change_patterns", [])
    kw.setdefault("risk_indicators", [])
    kw.setdefault("line_change_ratio", 0.1)
    kw.setdefault("content_similarity", 0.9)
    kw.setdefault("file_type", FileType.PYTHON)
    return FileChangeAnalysis(filename=filename, **kw)


def _ctx(files=None, diff_analysis=None, impact_assessment=None):
    return PRAnalysisContext(
        pr_diff=_pr_diff(files),
        codebase_context=_codebase_context(),
        diff_analysis=diff_analysis,
        impact_assessment=impact_assessment,
    )


# ---------------------------------------------------------------------------
# No issues / clean PR
# ---------------------------------------------------------------------------


class TestCleanPR:
    """Clean PRs with no issues produce minimal recommendations."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_no_issues_no_risk_score(self):
        recs = self.gen.generate_recommendations(_result(), _ctx())
        assert isinstance(recs, list)
        assert len(recs) == 0

    def test_low_signal_pr(self):
        recs = self.gen.generate_recommendations(
            _result(),
            _ctx(
                files=[
                    _fc(filename="readme.md", file_type=FileType.MARKDOWN, additions=1, deletions=0)
                ]
            ),
        )
        assert len(recs) <= 1


# ---------------------------------------------------------------------------
# Template-based recommendations from issue types
# ---------------------------------------------------------------------------


class TestTemplateRecommendations:
    """Issue types trigger matching recommendation templates."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_security_issues_trigger_security_rec(self):
        recs = self.gen.generate_recommendations(
            _result(
                issues=[
                    ValidationIssue(
                        type="security_pattern_xss", severity="high", message="XSS", file="v.py"
                    ),
                ],
                risk_level=RiskLevel.HIGH,
            ),
            _ctx(),
        )
        sec_recs = [r for r in recs if "security" in r.type.lower()]
        assert len(sec_recs) >= 1

    def test_breaking_change_triggers_architecture_rec(self):
        recs = self.gen.generate_recommendations(
            _result(
                issues=[
                    ValidationIssue(
                        type="breaking_change_api",
                        severity="high",
                        message="API break",
                        file="api.py",
                    ),
                ],
                risk_level=RiskLevel.HIGH,
            ),
            _ctx(),
        )
        assert any("breaking" in r.type.lower() or "architecture" in r.type.lower() for r in recs)

    def test_test_coverage_gap_triggers_testing_rec(self):
        recs = self.gen.generate_recommendations(
            _result(
                issues=[
                    ValidationIssue(
                        type="test_coverage_gap",
                        severity="medium",
                        message="No tests",
                        file="app.py",
                    ),
                ]
            ),
            _ctx(
                impact_assessment=_impact_assessment(
                    test_coverage_impact=_test_impact(coverage_change=-5.0),
                )
            ),
        )
        test_recs = [r for r in recs if "test" in r.type.lower()]
        assert len(test_recs) >= 1

    def test_hardcoded_secret_triggers_security_rec(self):
        recs = self.gen.generate_recommendations(
            _result(
                issues=[
                    ValidationIssue(
                        type="hardcoded_secret",
                        severity="critical",
                        message="Secret",
                        file="cfg.py",
                    ),
                ]
            ),
            _ctx(),
        )
        sec = [r for r in recs if "security" in r.type.lower()]
        assert len(sec) >= 1


# ---------------------------------------------------------------------------
# Risk-based recommendations
# ---------------------------------------------------------------------------


class TestRiskBasedRecommendations:
    """Risk scores drive risk-based recommendations."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_critical_risk_produces_mitigation_rec(self):
        risk = RiskScore(
            overall_score=9.0,
            dimension_scores={RiskDimension.SECURITY: 9.5},
            confidence=0.8,
        )
        recs = self.gen.generate_recommendations(
            _result(risk_level=RiskLevel.CRITICAL),
            _ctx(),
            risk_score=risk,
        )
        assert any(r.type == "critical_risk_mitigation" for r in recs)

    def test_high_security_dimension_produces_expert_review(self):
        risk = RiskScore(
            overall_score=7.5,
            dimension_scores={RiskDimension.SECURITY: 8.0},
            confidence=0.7,
        )
        recs = self.gen.generate_recommendations(
            _result(risk_level=RiskLevel.HIGH),
            _ctx(),
            risk_score=risk,
        )
        assert any(r.type == "security_expert_review" for r in recs)

    def test_high_performance_dimension_produces_perf_testing(self):
        risk = RiskScore(
            overall_score=5.0,
            dimension_scores={RiskDimension.PERFORMANCE: 7.5},
            confidence=0.9,
        )
        recs = self.gen.generate_recommendations(
            _result(risk_level=RiskLevel.MEDIUM),
            _ctx(),
            risk_score=risk,
        )
        perf = [r for r in recs if r.type == "performance_testing"]
        assert len(perf) >= 1
        assert perf[0].automated is True

    def test_high_reliability_dimension(self):
        risk = RiskScore(
            overall_score=6.0,
            dimension_scores={RiskDimension.RELIABILITY: 8.0},
            confidence=0.9,
        )
        recs = self.gen.generate_recommendations(
            _result(risk_level=RiskLevel.HIGH),
            _ctx(),
            risk_score=risk,
        )
        assert any(r.type == "reliability_verification" for r in recs)

    def test_low_confidence_produces_manual_review(self):
        risk = RiskScore(overall_score=3.0, dimension_scores={}, confidence=0.3)
        recs = self.gen.generate_recommendations(
            _result(risk_level=RiskLevel.LOW),
            _ctx(),
            risk_score=risk,
        )
        assert any(r.type == "manual_review_required" for r in recs)


# ---------------------------------------------------------------------------
# Custom recommendations
# ---------------------------------------------------------------------------


class TestCustomRecommendations:
    """Custom recommendations from specific patterns."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_logic_issues_trigger_logic_safety_review(self):
        recs = self.gen.generate_recommendations(
            _result(
                issues=[
                    ValidationIssue(
                        type="logic_inversion", severity="high", message="inv", file="a.py"
                    ),
                    ValidationIssue(
                        type="logic_change", severity="medium", message="chg", file="b.py"
                    ),
                ]
            ),
            _ctx(),
        )
        assert any(r.type == "logic_safety_review" for r in recs)
        logic_rec = next(r for r in recs if r.type == "logic_safety_review")
        assert "2 logic changes" in logic_rec.action

    def test_complete_rewrite_triggers_review(self):
        diff = _diff_analysis(
            file_analyses=[
                _file_change_analysis(
                    filename="core.py", change_severity=ChangeSeverity.COMPLETE_REWRITE
                ),
            ],
        )
        recs = self.gen.generate_recommendations(_result(), _ctx(diff_analysis=diff))
        assert any(r.type == "complete_rewrite_review" for r in recs)

    def test_obsolete_tests_trigger_cleanup(self):
        impact = _impact_assessment(
            test_coverage_impact=_test_impact(obsolete_tests=["test_old_func", "test_removed"]),
        )
        recs = self.gen.generate_recommendations(_result(), _ctx(impact_assessment=impact))
        assert any(r.type == "obsolete_tests_cleanup" for r in recs)


# ---------------------------------------------------------------------------
# Prioritization and deduplication
# ---------------------------------------------------------------------------


class TestPrioritizationAndDedup:
    """Recommendations sorted by priority and deduplicated."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_prioritize_sorts_critical_first(self):
        recs = [
            Recommendation(type="low", priority="low", description="", action="", automated=False),
            Recommendation(
                type="crit", priority="critical", description="", action="", automated=False
            ),
            Recommendation(
                type="med", priority="medium", description="", action="", automated=False
            ),
            Recommendation(type="hi", priority="high", description="", action="", automated=False),
        ]
        sorted_recs = self.gen._prioritize_recommendations(recs)
        priorities = [r.priority for r in sorted_recs]
        assert priorities == ["critical", "high", "medium", "low"]

    def test_deduplicate_removes_same_type_action(self):
        r1 = Recommendation(
            type="sec", priority="high", description="A", action="Do X to fix it", automated=False
        )
        r2 = Recommendation(
            type="sec", priority="high", description="B", action="Do X to fix it", automated=False
        )
        r3 = Recommendation(
            type="perf", priority="medium", description="C", action="Run bench", automated=True
        )
        result = self.gen._deduplicate_recommendations([r1, r2, r3])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# generate_summary_recommendations
# ---------------------------------------------------------------------------


class TestSummaryRecommendations:
    """Summary generation from recommendation lists."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_empty_summary(self):
        summary = self.gen.generate_summary_recommendations([])
        assert summary["total_recommendations"] == 0
        assert summary["automated_count"] == 0

    def test_summary_counts(self):
        recs = [
            Recommendation(
                type="security_fix",
                priority="critical",
                description="S",
                action="Fix",
                automated=False,
            ),
            Recommendation(
                type="performance_opt",
                priority="medium",
                description="P",
                action="Opt",
                automated=True,
            ),
            Recommendation(
                type="test_update", priority="low", description="T", action="Add", automated=False
            ),
        ]
        summary = self.gen.generate_summary_recommendations(recs)
        assert summary["total_recommendations"] == 3
        assert summary["automated_count"] == 1
        assert summary["by_priority"]["critical"] == 1
        assert summary["by_priority"]["medium"] == 1
        assert summary["by_priority"]["low"] == 1
        assert "Security" in summary["by_category"]
        assert "Performance" in summary["by_category"]
        assert "Testing" in summary["by_category"]

    def test_unknown_category_becomes_other(self):
        recs = [
            Recommendation(
                type="xyz_unknown", priority="low", description="", action="", automated=False
            ),
        ]
        summary = self.gen.generate_summary_recommendations(recs)
        assert "Other" in summary["by_category"]
