"""Edge-case and boundary tests for RiskCalculator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
    TestImpact,
    ValidationIssue,
)

from services.pr_validation.scoring.risk_calculator import (
    RiskCalculationConfig,
    RiskCalculator,
    RiskDimension,
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
# _validate_config — weight normalization
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Weights that do not sum to 1.0 get normalized."""

    def test_normalizes_weights(self):
        cfg = RiskCalculationConfig(
            dimension_weights={d: 0.5 for d in RiskDimension},
        )
        calc = RiskCalculator(config=cfg)
        total = sum(calc.config.dimension_weights.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# calculate_overall_risk — exception path
# ---------------------------------------------------------------------------


class TestCalculateOverallRiskException:
    """Internal error returns 5.0 risk with 0 confidence."""

    def test_returns_default_on_internal_error(self):
        calc = RiskCalculator()
        ctx = _ctx()
        with patch.object(calc, "_calculate_security_risk", side_effect=RuntimeError("boom")):
            score = calc.calculate_overall_risk(ctx, [])
        assert score.overall_score == 5.0
        assert score.confidence == 0.0
        assert any("Risk calculation error" in f for f in score.risk_factors)


# ---------------------------------------------------------------------------
# _identify_risk_factors — various triggers
# ---------------------------------------------------------------------------


class TestIdentifyRiskFactors:
    """Risk factor identification from issues and diff analysis."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_critical_issues_flagged(self):
        issues = [
            ValidationIssue(type="sec", severity="critical", message="bad", file="a.py"),
        ]
        factors = self.calc._identify_risk_factors(_ctx(), issues)
        assert any("critical issues" in f.lower() for f in factors)

    def test_many_high_issues_flagged(self):
        issues = [
            ValidationIssue(type="issue", severity="high", message=f"h{i}", file="a.py")
            for i in range(7)
        ]
        factors = self.calc._identify_risk_factors(_ctx(), issues)
        assert any("Many high-severity" in f for f in factors)

    def test_dangerous_logic_changes(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=["DANGEROUS_LOGIC_CHANGES"],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        factors = self.calc._identify_risk_factors(_ctx(diff_analysis=diff), [])
        assert "Dangerous logic changes detected" in factors

    def test_security_sensitive_file(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=["SECURITY_SENSITIVE_FILE_MODIFIED"],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        factors = self.calc._identify_risk_factors(_ctx(diff_analysis=diff), [])
        assert "Security-sensitive files modified" in factors

    def test_requires_careful_review(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=0.0,
            requires_careful_review=True,
        )
        factors = self.calc._identify_risk_factors(_ctx(diff_analysis=diff), [])
        assert "Changes require careful review" in factors

    def test_high_change_complexity(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=8.5,
            requires_careful_review=False,
        )
        factors = self.calc._identify_risk_factors(_ctx(diff_analysis=diff), [])
        assert "High change complexity detected" in factors

    def test_large_pr(self):
        files = [_fc(f"f{i}.py") for i in range(25)]
        factors = self.calc._identify_risk_factors(_ctx(files=files), [])
        assert "Large PR with many file changes" in factors


# ---------------------------------------------------------------------------
# _identify_mitigating_factors
# ---------------------------------------------------------------------------


class TestIdentifyMitigatingFactors:
    """Mitigating factors from test updates, docs, etc."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_test_updates_mitigating(self):
        impact = ImpactAssessment(
            dependency_impacts=[],
            test_coverage_impact=TestImpact(
                tests_affected=["test_foo"],
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
        factors = self.calc._identify_mitigating_factors(_ctx(impact_assessment=impact), [])
        assert "Changes include test updates" in factors

    def test_coverage_maintained(self):
        impact = ImpactAssessment(
            dependency_impacts=[],
            test_coverage_impact=TestImpact(
                tests_affected=["t"],
                coverage_change=5.0,
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
        factors = self.calc._identify_mitigating_factors(_ctx(impact_assessment=impact), [])
        assert "Test coverage maintained or updated" in factors

    def test_documentation_changes(self):
        factors = self.calc._identify_mitigating_factors(
            _ctx(files=[_fc("README.md", file_type=FileType.MARKDOWN)]),
            [],
        )
        assert "Documentation updated alongside code changes" in factors

    def test_single_file_analysis(self):
        diff = DiffAnalysisResult(
            file_analyses=[
                FileChangeAnalysis(
                    filename="a.py",
                    change_severity=ChangeSeverity.MINOR,
                    change_patterns=[],
                    risk_indicators=[],
                    line_change_ratio=0.1,
                    content_similarity=0.9,
                    file_type=FileType.PYTHON,
                )
            ],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        factors = self.calc._identify_mitigating_factors(_ctx(diff_analysis=diff), [])
        assert "Single file change - limited scope" in factors

    def test_all_minor_changes(self):
        minor = SimpleNamespace(value="MINOR")
        diff = DiffAnalysisResult(
            file_analyses=[
                SimpleNamespace(filename="a.py", change_severity=minor),
                SimpleNamespace(filename="b.py", change_severity=minor),
            ],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        factors = self.calc._identify_mitigating_factors(_ctx(diff_analysis=diff), [])
        assert "All changes are minor modifications" in factors

    def test_no_critical_or_high_issues(self):
        issues = [ValidationIssue(type="style", severity="low", message="s", file="a.py")]
        factors = self.calc._identify_mitigating_factors(_ctx(), issues)
        assert "No critical or high-severity issues found" in factors


# ---------------------------------------------------------------------------
# _calculate_confidence — edge cases
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    """Confidence calculation edge cases."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_high_complexity_reduces_confidence(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=9.0,
            requires_careful_review=False,
        )
        ctx = _ctx(diff_analysis=diff)
        confidence = self.calc._calculate_confidence(ctx, [])
        # base 1.0 - 0.15 (no semantic) - 0.15 (no impact) - 0.1 (high complexity)
        assert confidence <= 0.65

    def test_many_files_reduces_confidence(self):
        files = [_fc(f"f{i}.py") for i in range(35)]
        ctx = _ctx(files=files)
        confidence = self.calc._calculate_confidence(ctx, [])
        # base 1.0 - 0.2 (no diff) - 0.15 (no semantic) - 0.15 (no impact) - 0.1 (>30 files)
        assert confidence <= 0.45

    def test_error_issues_reduce_confidence(self):
        issues = [
            ValidationIssue(type="error_in_analysis", severity="low", message="e", file="a.py")
            for _ in range(5)
        ]
        ctx = _ctx()
        confidence = self.calc._calculate_confidence(ctx, issues)
        # 5 error issues * 0.1 = 0.5 reduction
        assert confidence <= 0.2


# ---------------------------------------------------------------------------
# Performance risk — multiple patterns
# ---------------------------------------------------------------------------


class TestPerformanceRiskPatterns:
    """Multiple performance-sensitive patterns in patches."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_complexity_type_issues_counted(self):
        ctx = _ctx(files=[_fc("app.py")])
        issues = [
            ValidationIssue(type="complexity_increase", severity="high", message="c", file="app.py")
        ]
        score = self.calc._calculate_performance_risk(ctx, issues)
        assert score >= 2.5

    def test_while_loop_in_patch(self):
        ctx = _ctx(files=[_fc("app.py", patch="+while True:\n+    process()")])
        score = self.calc._calculate_performance_risk(ctx, [])
        assert score >= 0.5


# ---------------------------------------------------------------------------
# Impact scope — change volume
# ---------------------------------------------------------------------------


class TestImpactScopeVolume:
    """Change volume thresholds."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_over_2000_changes(self):
        ctx = _ctx(files=[_fc("big.py", additions=1500, deletions=600)])
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 2.0

    def test_between_1000_and_2000_changes(self):
        ctx = _ctx(files=[_fc("med.py", additions=700, deletions=400)])
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 1.0

    def test_file_count_between_10_and_20(self):
        files = [_fc(f"f{i}.py") for i in range(15)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 1.5

    def test_file_count_between_20_and_50(self):
        files = [_fc(f"f{i}.py") for i in range(25)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_impact_scope_risk(ctx, [])
        assert score >= 3.0


# ---------------------------------------------------------------------------
# Complexity risk — MULTIPLE_COMPLETE_REWRITES
# ---------------------------------------------------------------------------


class TestComplexityRiskPatterns:
    """Complexity risk from rewrite patterns."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_multiple_complete_rewrites(self):
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=["MULTIPLE_COMPLETE_REWRITES"],
            risk_factors=[],
            change_complexity=0.0,
            requires_careful_review=False,
        )
        ctx = _ctx(diff_analysis=diff)
        score = self.calc._calculate_complexity_risk(ctx, [])
        assert score >= 3.0

    def test_complexity_issues_half_weight(self):
        ctx = _ctx(files=[_fc("app.py")])
        issues = [
            ValidationIssue(type="complexity_increase", severity="high", message="c", file="app.py")
        ]
        score = self.calc._calculate_complexity_risk(ctx, issues)
        assert score >= 1.25  # 2.5 * 0.5
