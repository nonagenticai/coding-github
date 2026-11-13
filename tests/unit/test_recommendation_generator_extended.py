"""Edge-case and boundary tests for PRValidationRecommendationGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    DiffAnalysisResult,
    DiffStats,
    DocumentationImpact,
    FileChange,
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
    RecommendationCategory,
    RecommendationContext,
    RecommendationPriority,
    RecommendationTemplate,
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
    kw.setdefault("test_coverage_impact", _test_impact())
    kw.setdefault(
        "documentation_impact",
        DocumentationImpact(
            docs_affected=[], outdated_docs=[], new_docs_needed=[], doc_completeness_score=0.0
        ),
    )
    kw.setdefault("performance_impact", {})
    kw.setdefault("security_impact", {})
    return ImpactAssessment(**kw)


def _ctx(files=None, diff_analysis=None, impact_assessment=None):
    return PRAnalysisContext(
        pr_diff=_pr_diff(files),
        codebase_context=_codebase_context(),
        diff_analysis=diff_analysis,
        impact_assessment=impact_assessment,
    )


# ---------------------------------------------------------------------------
# Exception handling in generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendationsException:
    """Internal error returns error recommendation."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_exception_returns_error_rec(self):
        original = self.gen._build_recommendation_context
        self.gen._build_recommendation_context = MagicMock(
            side_effect=RuntimeError("Context build error"),
        )
        try:
            recs = self.gen.generate_recommendations(_result(), _ctx())
        finally:
            self.gen._build_recommendation_context = original

        assert len(recs) == 1
        assert recs[0].type == "recommendation_error"
        assert recs[0].priority == "medium"


# ---------------------------------------------------------------------------
# _build_recommendation_context
# ---------------------------------------------------------------------------


class TestBuildRecommendationContext:
    """Context building from validation results."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_counts_critical_and_high_issues(self):
        result = _result(
            issues=[
                ValidationIssue(type="sec", severity="critical", message="Bad", file="a.py"),
                ValidationIssue(type="sec2", severity="critical", message="Worse", file="b.py"),
                ValidationIssue(type="perf", severity="high", message="Slow", file="c.py"),
            ],
            risk_level=RiskLevel.CRITICAL,
        )
        ctx = self.gen._build_recommendation_context(result, _ctx(), None)
        assert ctx.critical_issues == 2
        assert ctx.high_severity_issues == 1

    def test_captures_diff_analysis_complexity(self):
        diff = _diff_analysis(change_complexity=6.5)
        result = _result()
        rec_ctx = self.gen._build_recommendation_context(result, _ctx(diff_analysis=diff), None)
        assert rec_ctx.change_complexity == 6.5

    def test_captures_test_coverage_change(self):
        impact = _impact_assessment(
            test_coverage_impact=_test_impact(coverage_change=-3.0),
        )
        result = _result()
        rec_ctx = self.gen._build_recommendation_context(
            result, _ctx(impact_assessment=impact), None
        )
        assert rec_ctx.test_coverage_change == -3.0

    def test_documentation_updated_flag(self):
        result = _result()
        rec_ctx = self.gen._build_recommendation_context(
            result,
            _ctx(files=[_fc(filename="docs/guide.md", file_type=FileType.MARKDOWN)]),
            None,
        )
        assert rec_ctx.documentation_updated is True


# ---------------------------------------------------------------------------
# Template conditions
# ---------------------------------------------------------------------------


class TestTemplateConditions:
    """Various template condition checks."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_multiple_components_affected_met(self):
        ctx = RecommendationContext(total_files_changed=10, change_complexity=6.0)
        # integration_testing template (index 8) has conditions: multiple_components_affected, high_change_complexity
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[8],
            ctx,
            [],
            None,
        )
        assert met is True

    def test_high_change_complexity_not_met(self):
        ctx = RecommendationContext(total_files_changed=10, change_complexity=3.0)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[8],
            ctx,
            [],
            None,
        )
        assert met is False

    def test_large_pr_condition(self):
        ctx = RecommendationContext(total_files_changed=20, change_complexity=8.0)
        # large_pr_breakdown (index 12) has conditions: large_pr, high_complexity
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[12],
            ctx,
            [],
            None,
        )
        assert met is True

    def test_large_pr_condition_not_met_low_files(self):
        ctx = RecommendationContext(total_files_changed=5, change_complexity=8.0)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[12],
            ctx,
            [],
            None,
        )
        assert met is False

    def test_high_risk_level_condition(self):
        ctx = RecommendationContext(risk_level=RiskLevel.CRITICAL)
        # staged_deployment (index 13) has conditions: high_risk_level, critical_system_changes
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[13],
            ctx,
            [],
            None,
        )
        assert met is True

    def test_low_risk_level_fails_condition(self):
        ctx = RecommendationContext(risk_level=RiskLevel.LOW)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[13],
            ctx,
            [],
            None,
        )
        assert met is False

    def test_new_features_condition_not_met(self):
        ctx = RecommendationContext(total_files_changed=1)
        # documentation_update (index 9) has conditions: api_changes, new_features
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[9],
            ctx,
            ["api_endpoint"],
            None,
        )
        assert met is False


# ---------------------------------------------------------------------------
# Template with risk_dimensions but no issue_types (elif branch)
# ---------------------------------------------------------------------------


class TestTemplateRiskDimensionsBranch:
    """Template matching via risk_dimensions when issue_types is empty."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_risk_dimensions_only_template(self):
        custom = RecommendationTemplate(
            template_id="custom_risk_only",
            category=RecommendationCategory.PERFORMANCE,
            title_template="Custom",
            description_template="Test risk dimension",
            action_template="Fix in {file_count} files",
            priority=RecommendationPriority.HIGH,
            conditions=[],
            issue_types=[],
            risk_dimensions=[RiskDimension.PERFORMANCE],
        )
        self.gen.recommendation_templates.append(custom)
        try:
            risk = RiskScore(
                overall_score=7.0,
                dimension_scores={RiskDimension.PERFORMANCE: 6.0},
                confidence=0.9,
            )
            recs = self.gen.generate_recommendations(
                _result(risk_level=RiskLevel.HIGH),
                _ctx(),
                risk_score=risk,
            )
            assert any(r.type == "custom_risk_only" for r in recs)
        finally:
            self.gen.recommendation_templates.pop()


# ---------------------------------------------------------------------------
# Summary — top_priorities
# ---------------------------------------------------------------------------


class TestSummaryTopPriorities:
    """Top priorities limited to 5 critical/high recs."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_top_priorities_limited_to_5(self):
        recs = [
            Recommendation(
                type=f"sec_{i}",
                priority="critical",
                description=f"S{i}",
                action=f"Fix{i}",
                automated=False,
            )
            for i in range(8)
        ]
        summary = self.gen.generate_summary_recommendations(recs)
        assert len(summary["top_priorities"]) == 5

    def test_top_priorities_only_critical_and_high(self):
        recs = [
            Recommendation(
                type="crit", priority="critical", description="C", action="A", automated=False
            ),
            Recommendation(
                type="hi", priority="high", description="H", action="B", automated=False
            ),
            Recommendation(
                type="med", priority="medium", description="M", action="C", automated=False
            ),
            Recommendation(type="lo", priority="low", description="L", action="D", automated=False),
        ]
        summary = self.gen.generate_summary_recommendations(recs)
        assert len(summary["top_priorities"]) == 2
        assert all(tp["priority"] in ["critical", "high"] for tp in summary["top_priorities"])


# ---------------------------------------------------------------------------
# _create_recommendation_from_template
# ---------------------------------------------------------------------------


class TestCreateRecommendationFromTemplate:
    """Verify recommendation creation from template with issue details."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_issue_details_included_in_action(self):
        template = self.gen.recommendation_templates[1]  # security_vulnerability_fix
        issues = [
            ValidationIssue(
                type="security_pattern_sqli", severity="high", message="SQLi", file="db.py"
            ),
            ValidationIssue(
                type="security_pattern_sqli", severity="high", message="SQLi2", file="db2.py"
            ),
        ]
        ctx = RecommendationContext(total_files_changed=2, change_complexity=3.0)
        rec = self.gen._create_recommendation_from_template(template, issues, ctx)
        assert rec.type == "security_vulnerability_fix"
        assert "security_pattern_sqli" in rec.action
        assert rec.priority == "high"
