"""Core tests for PRValidationEngine — config, scoring, risk, and pipeline."""

from __future__ import annotations

from unittest.mock import patch

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
    SemanticAnalysisResult,
    ValidationIssue,
)

from services.pr_validation.core.pr_validation_engine import (
    PRValidationConfig,
    PRValidationEngine,
    ValidationStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pr_diff(files=None, metadata=None):
    files = files or []
    return PRDiff(
        base_sha="aaa",
        head_sha="bbb",
        files_changed=files,
        stats=DiffStats.from_files(files),
        metadata=metadata or {},
    )


def _py_file(name="src/foo.py", additions=5, deletions=2, patch="@@ -1 +1 @@\n-old\n+new"):
    return FileChange(
        filename=name,
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=additions,
        deletions=deletions,
        patch=patch,
        before_content="old",
        after_content="new",
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


def _diff_analysis(**kw):
    kw.setdefault("file_analyses", [])
    kw.setdefault("overall_patterns", [])
    kw.setdefault("risk_factors", [])
    kw.setdefault("change_complexity", 0.0)
    kw.setdefault("requires_careful_review", False)
    return DiffAnalysisResult(**kw)


def _impact_assessment(**kw):
    from coding_client.models.pr_validation import DocumentationImpact, TestImpact

    kw.setdefault("dependency_impacts", [])
    kw.setdefault(
        "test_coverage_impact",
        TestImpact(tests_affected=[], coverage_change=0.0, new_tests_needed=[], obsolete_tests=[]),
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


def _semantic_analysis(**kw):
    kw.setdefault("breaking_changes", [])
    kw.setdefault("logic_safety_issues", [])
    kw.setdefault("api_surface_changes", {})
    kw.setdefault("complexity_changes", {})
    return SemanticAnalysisResult(**kw)


# ---------------------------------------------------------------------------
# PRValidationConfig
# ---------------------------------------------------------------------------


class TestPRValidationConfig:
    """Config defaults and custom values."""

    def test_default_thresholds(self):
        cfg = PRValidationConfig()
        assert cfg.risk_thresholds["low"] == 2.0
        assert cfg.risk_thresholds["critical"] == 10.0

    def test_custom_config(self):
        cfg = PRValidationConfig(
            enable_semantic_analysis=False,
            overall_timeout=600,
            analysis_intensity=5,
        )
        assert cfg.enable_semantic_analysis is False
        assert cfg.overall_timeout == 600
        assert cfg.analysis_intensity == 5


# ---------------------------------------------------------------------------
# ValidationStats
# ---------------------------------------------------------------------------


class TestValidationStats:
    """Stats initialization."""

    def test_default_stats(self):
        stats = ValidationStats()
        assert stats.analyzer_times == {}
        assert stats.errors_encountered == []

    def test_custom_stats(self):
        stats = ValidationStats(total_files_analyzed=10, total_analysis_time=5.0)
        assert stats.total_files_analyzed == 10


# ---------------------------------------------------------------------------
# _calculate_validation_score
# ---------------------------------------------------------------------------


class TestCalculateValidationScore:
    """Score deductions from issues and risk level."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    @pytest.mark.parametrize(
        "severity,deduction",
        [("critical", 3.0), ("high", 2.0), ("medium", 1.0), ("low", 0.5)],
    )
    def test_severity_deductions(self, severity, deduction):
        issues = [ValidationIssue(type="t", severity=severity, message="m", file="f")]
        score = self.engine._calculate_validation_score(issues, RiskLevel.LOW)
        assert score == pytest.approx(10.0 - deduction)

    def test_unknown_severity_deducts_0_5(self):
        issues = [ValidationIssue(type="t", severity="info", message="m", file="f")]
        score = self.engine._calculate_validation_score(issues, RiskLevel.LOW)
        assert score == pytest.approx(9.5)

    def test_critical_risk_penalty(self):
        score = self.engine._calculate_validation_score([], RiskLevel.CRITICAL)
        assert score == 8.0

    def test_high_risk_penalty(self):
        score = self.engine._calculate_validation_score([], RiskLevel.HIGH)
        assert score == 9.0

    def test_score_never_below_zero(self):
        issues = [
            ValidationIssue(type="t", severity="critical", message="m", file="f") for _ in range(10)
        ]
        score = self.engine._calculate_validation_score(issues, RiskLevel.CRITICAL)
        assert score == 0.0

    def test_mixed_severity_issues(self):
        issues = [
            ValidationIssue(type="a", severity="critical", message="", file=""),
            ValidationIssue(type="b", severity="high", message="", file=""),
            ValidationIssue(type="c", severity="medium", message="", file=""),
            ValidationIssue(type="d", severity="low", message="", file=""),
        ]
        score = self.engine._calculate_validation_score(issues, RiskLevel.LOW)
        # 10 - 3.0 - 2.0 - 1.0 - 0.5 = 3.5
        assert score == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# _determine_validation_pass
# ---------------------------------------------------------------------------


class TestDetermineValidationPass:
    """Pass/fail determination."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_critical_issue_fails(self):
        issues = [ValidationIssue(type="t", severity="critical", message="m", file="f")]
        assert self.engine._determine_validation_pass(8.0, RiskLevel.LOW, issues) is False

    def test_critical_risk_level_fails(self):
        assert self.engine._determine_validation_pass(8.0, RiskLevel.CRITICAL, []) is False

    def test_low_score_fails(self):
        assert self.engine._determine_validation_pass(4.9, RiskLevel.LOW, []) is False

    def test_good_score_passes(self):
        assert self.engine._determine_validation_pass(7.0, RiskLevel.LOW, []) is True


# ---------------------------------------------------------------------------
# _calculate_risk_level
# ---------------------------------------------------------------------------


class TestCalculateRiskLevel:
    """Risk level from issues + context."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_no_issues_low_risk(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        risk = self.engine._calculate_risk_level([], ctx)
        assert risk == RiskLevel.LOW

    def test_critical_issues_drive_critical_risk(self):
        self.engine.config.risk_thresholds = {
            "low": 1.0,
            "medium": 3.0,
            "high": 5.0,
            "critical": 8.0,
        }
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        issues = [
            ValidationIssue(type="t", severity="critical", message="m", file="f") for _ in range(2)
        ]
        # 2 * 4.0 = 8.0 >= critical threshold
        risk = self.engine._calculate_risk_level(issues, ctx)
        assert risk == RiskLevel.CRITICAL

    def test_many_files_add_risk(self):
        files = [_py_file(name=f"f{i}.py") for i in range(25)]
        ctx = PRAnalysisContext(pr_diff=_pr_diff(files), codebase_context=_codebase_context())
        self.engine.config.risk_thresholds = {
            "low": 1.0,
            "medium": 1.5,
            "high": 5.0,
            "critical": 8.0,
        }
        risk = self.engine._calculate_risk_level([], ctx)
        # 25 files > 20 => +2.0 risk
        assert risk == RiskLevel.MEDIUM

    def test_diff_analysis_complexity_added(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_context(),
            diff_analysis=_diff_analysis(change_complexity=5.0),
        )
        self.engine.config.risk_thresholds = {
            "low": 1.0,
            "medium": 3.0,
            "high": 6.0,
            "critical": 9.0,
        }
        risk = self.engine._calculate_risk_level([], ctx)
        assert risk == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# _calculate_confidence
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    """Confidence reduction from missing analysis."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_full_analysis_high_confidence(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_context(),
            diff_analysis=_diff_analysis(),
            impact_assessment=_impact_assessment(),
        )
        self.engine.stats.errors_encountered = []
        conf = self.engine._calculate_confidence([], ctx)
        assert conf == pytest.approx(0.9)

    def test_missing_both_reduces_by_0_2(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_context(),
        )
        self.engine.stats.errors_encountered = []
        conf = self.engine._calculate_confidence([], ctx)
        assert conf == pytest.approx(0.7)

    def test_errors_reduce_confidence(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_context(),
            diff_analysis=_diff_analysis(),
            impact_assessment=_impact_assessment(),
        )
        self.engine.stats.errors_encountered = ["err1", "err2"]
        conf = self.engine._calculate_confidence([], ctx)
        assert conf == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    """Recommendation generation from issues."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_logic_inversion_recommendation(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        issues = [
            ValidationIssue(type="logic_inversion", severity="high", message="inv", file="a.py")
        ]
        recs = self.engine._generate_recommendations(issues, ctx)
        logic_recs = [r for r in recs if r.type == "logic_safety"]
        assert len(logic_recs) == 1

    def test_complexity_recommendation(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        issues = [
            ValidationIssue(type="complexity_increase", severity="medium", message="c", file="a.py")
        ]
        recs = self.engine._generate_recommendations(issues, ctx)
        maint_recs = [r for r in recs if r.type == "maintainability"]
        assert len(maint_recs) == 1

    def test_high_change_complexity_recommendation(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_context(),
            diff_analysis=_diff_analysis(change_complexity=6.0),
        )
        recs = self.engine._generate_recommendations([], ctx)
        complexity_recs = [r for r in recs if r.type == "complexity"]
        assert len(complexity_recs) == 1

    def test_no_recommendations_for_benign_issues(self):
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        issues = [ValidationIssue(type="typo", severity="low", message="t", file="a.py")]
        recs = self.engine._generate_recommendations(issues, ctx)
        assert len(recs) == 0


# ---------------------------------------------------------------------------
# _create_timeout_result / _create_error_result
# ---------------------------------------------------------------------------


class TestSpecialResults:
    """Timeout and error result creation."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_timeout_result(self):
        result = self.engine._create_timeout_result()
        assert result.passed is False
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.issues[0].type == "validation_timeout"

    def test_error_result(self):
        result = self.engine._create_error_result("something broke")
        assert result.passed is False
        assert "something broke" in result.issues[0].message
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# _estimate_review_effort
# ---------------------------------------------------------------------------


class TestEstimateReviewEffort:
    """Review effort estimation."""

    def setup_method(self):
        self.engine = PRValidationEngine()

    def test_low_effort(self):
        from services.pr_validation.core.change_categorizer import (
            ChangeCategory,
            ChangeScope,
            PRChangeProfile,
        )

        profile = PRChangeProfile(
            primary_category=ChangeCategory.BUG_FIX,
            secondary_categories=[],
            overall_scope=ChangeScope.FILE_LEVEL,
            change_types={},
            risk_factors=[],
            complexity_score=2.0,
        )
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        effort = self.engine._estimate_review_effort(profile, ctx)
        assert effort == "LOW"

    def test_medium_effort(self):
        from services.pr_validation.core.change_categorizer import (
            ChangeCategory,
            ChangeScope,
            PRChangeProfile,
        )

        profile = PRChangeProfile(
            primary_category=ChangeCategory.FEATURE_ADDITION,
            secondary_categories=[],
            overall_scope=ChangeScope.MODULE_LEVEL,
            change_types={},
            risk_factors=[],
            complexity_score=5.0,
        )
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()] * 12), codebase_context=_codebase_context()
        )
        effort = self.engine._estimate_review_effort(profile, ctx)
        assert effort == "MEDIUM"

    def test_high_effort(self):
        from services.pr_validation.core.change_categorizer import (
            ChangeCategory,
            ChangeScope,
            PRChangeProfile,
        )

        profile = PRChangeProfile(
            primary_category=ChangeCategory.REFACTORING,
            secondary_categories=[],
            overall_scope=ChangeScope.PROJECT_LEVEL,
            change_types={},
            risk_factors=[],
            complexity_score=8.0,
        )
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()] * 25), codebase_context=_codebase_context()
        )
        effort = self.engine._estimate_review_effort(profile, ctx)
        assert effort == "HIGH"


# ---------------------------------------------------------------------------
# Pipeline — _execute_validation_pipeline with mocked analyzers
# ---------------------------------------------------------------------------


class TestExecuteValidationPipeline:
    """Pipeline produces correct ValidationResult from mocked analyzers."""

    @pytest.mark.asyncio
    async def test_basic_pipeline(self):
        engine = PRValidationEngine()
        py_file = _py_file()
        ctx = PRAnalysisContext(pr_diff=_pr_diff([py_file]), codebase_context=_codebase_context())

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "diff_analysis":
                return _diff_analysis(change_complexity=1.0)
            if name == "ast_analysis":
                return _semantic_analysis()
            if name == "breaking_change_detection":
                return []
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            result = await engine._execute_validation_pipeline(ctx)

        assert result.validation_type == "comprehensive_pr_validation"
        assert isinstance(result.score, float)
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_pipeline_with_issues(self):
        engine = PRValidationEngine()
        py_file = _py_file()
        ctx = PRAnalysisContext(pr_diff=_pr_diff([py_file]), codebase_context=_codebase_context())

        logic_issue = ValidationIssue(
            type="logic_inversion", severity="high", message="inv", file="src/foo.py"
        )
        sec_issue = ValidationIssue(
            type="security_pattern_sqli", severity="critical", message="sqli", file="src/foo.py"
        )

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "diff_analysis":
                return _diff_analysis(change_complexity=1.0)
            if name == "ast_analysis":
                return _semantic_analysis()
            if name == "breaking_change_detection":
                return []
            if name == "logic_safety":
                return [logic_issue]
            if name == "security_analysis":
                return [sec_issue]
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            result = await engine._execute_validation_pipeline(ctx)

        issue_types = {i.type for i in result.issues}
        assert "logic_inversion" in issue_types
        assert "security_pattern_sqli" in issue_types
        # Critical issue means passed=False
        assert result.passed is False
