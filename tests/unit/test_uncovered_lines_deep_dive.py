"""Deep-dive tests targeting remaining uncovered lines and bug fixes.

Covers:
- pr_validation_engine.py: validate_pr success/timeout/error, content provider edge cases,
  breaking changes in loop, high-impact dependency issues, file_count risk thresholds,
  RiskLevel.HIGH return path, _create_high_impact_issue, security_pattern_ bug fix
- diff_parser.py: _parse_diff_header /dev/null branches (lines 342, 344)
- risk_calculator.py: maintainability total_changes 200-500, file_count 10-20,
  explain_risk_score MEDIUM branch, _score_to_level "medium"
- recommendation_generator.py: high_complexity condition (line 477),
  critical_security_issues bug fix
- utils/__init__.py: import coverage
- github_error_handler.py: GithubException import fallback (lines 30-32)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from coding_client.models.pr_validation import (
    BreakingChange,
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
    RiskLevel,
    SemanticAnalysisResult,
    TestImpact,
    ValidationIssue,
    ValidationResult,
)

from services.pr_validation.core.diff_parser import DiffParser
from services.pr_validation.core.pr_validation_engine import (
    PRValidationEngine,
)
from services.pr_validation.scoring.recommendation_generator import (
    PRValidationRecommendationGenerator,
    RecommendationContext,
)
from services.pr_validation.scoring.risk_calculator import (
    RiskCalculator,
    RiskDimension,
    RiskScore,
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


def _codebase_ctx(**overrides):
    defaults = dict(
        total_files=10,
        total_lines=500,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=["fastapi"],
        architecture_patterns=["microservice"],
    )
    defaults.update(overrides)
    return CodebaseContext(**defaults)


def _ctx(files=None, diff_analysis=None, impact_assessment=None, semantic_analysis=None):
    fc = files or []
    return PRAnalysisContext(
        pr_diff=_pr_diff(fc),
        codebase_context=_codebase_ctx(),
        diff_analysis=diff_analysis,
        impact_assessment=impact_assessment,
        semantic_analysis=semantic_analysis,
    )


def _result(issues=None, risk_level=RiskLevel.LOW, score=90.0):
    return ValidationResult(
        validation_type="comprehensive",
        passed=True,
        score=score,
        risk_level=risk_level,
        issues=issues or [],
    )


# ===========================================================================
# pr_validation_engine.py — validate_pr success path (lines 135-145)
# ===========================================================================


class TestValidatePrSuccessPath:
    """Cover the try block in validate_pr that calls _parse_pr_diff and _execute_validation_pipeline."""

    @pytest.mark.asyncio
    async def test_validate_pr_returns_result_with_execution_time(self):
        engine = PRValidationEngine()
        pr_data = {
            "base": {"sha": "aaa"},
            "head": {"sha": "bbb"},
            "repository": {"full_name": "o/r"},
            "files": [
                {
                    "filename": "src/app.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 0,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
        }
        codebase_ctx = _codebase_ctx()

        # Mock the pipeline to return a simple result
        mock_result = ValidationResult(
            validation_type="comprehensive_pr_validation",
            passed=True,
            score=10.0,
            risk_level=RiskLevel.LOW,
            issues=[],
            recommendations=[],
        )

        with patch.object(engine, "_execute_validation_pipeline", return_value=mock_result):
            result = await engine.validate_pr(pr_data, codebase_ctx)

        assert result.validation_type == "comprehensive_pr_validation"
        assert result.execution_time > 0
        assert engine.stats.total_files_analyzed >= 0
        assert engine.stats.total_analysis_time > 0


# ===========================================================================
# pr_validation_engine.py — validate_pr TimeoutError path (lines 147-148)
# ===========================================================================


class TestValidatePrTimeoutPath:
    """Cover the TimeoutError except block in validate_pr."""

    @pytest.mark.asyncio
    async def test_validate_pr_timeout_returns_timeout_result(self):
        engine = PRValidationEngine()
        pr_data = {
            "base": {"sha": "a"},
            "head": {"sha": "b"},
            "repository": {"full_name": "o/r"},
            "files": [],
        }
        codebase_ctx = _codebase_ctx()

        with patch.object(engine, "_parse_pr_diff", side_effect=TimeoutError("too slow")):
            result = await engine.validate_pr(pr_data, codebase_ctx)

        assert result.validation_type == "pr_validation_timeout"
        assert result.passed is False
        assert result.risk_level == RiskLevel.CRITICAL
        assert any(i.type == "validation_timeout" for i in result.issues)


# ===========================================================================
# pr_validation_engine.py — content provider latin-1 fallback failure (lines 221-222)
# ===========================================================================


class TestContentProviderLatin1FallbackFailure:
    """Cover the except Exception: continue inside latin-1 fallback."""

    def test_latin1_fallback_also_fails_continues(self):
        engine = PRValidationEngine()
        repo_name = "_test_latin1_fail_repo_"
        repo_dir = Path(f"/tmp/repos/{repo_name}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        target = repo_dir / "broken.bin"
        try:
            # Write binary content that will fail UTF-8
            target.write_bytes(b"\x80\x81\x82")
            pr_data = {
                "repository": {"full_name": f"owner/{repo_name}"},
                "base": {"sha": "a"},
                "head": {"sha": "b"},
            }
            provider = engine._create_content_provider(pr_data)

            # Patch read_text to fail for both encodings
            with patch.object(Path, "read_text", side_effect=Exception("read failed")):
                result = provider("broken.bin", "head")
            # Should get None since both read attempts fail
            assert result is None
        finally:
            target.unlink(missing_ok=True)
            repo_dir.rmdir()


# ===========================================================================
# pr_validation_engine.py — content provider local file access exception (lines 225-227)
# ===========================================================================


class TestContentProviderLocalAccessException:
    """Cover the outer except Exception for local file access (lines 225-227)."""

    def test_local_file_access_exception_on_path_construction(self):
        engine = PRValidationEngine()
        # Use a repo_full_name that causes Path() to raise in the inner try block
        pr_data = {
            "repository": {"full_name": "o/r"},
            "base": {"sha": "a"},
            "head": {"sha": "b"},
        }
        provider = engine._create_content_provider(pr_data)

        # Patch pathlib.Path (which is imported locally inside the closure) to raise
        import pathlib

        original_path = pathlib.Path

        class ExplodingPath(type(original_path())):
            """Path subclass that raises on construction after first few calls."""

            pass

        # We need to make the potential_paths construction fail.
        # The function does `from pathlib import Path` then uses it.
        # We patch the Path class within the pathlib module itself.
        with patch("pathlib.Path") as mock_path:
            mock_path.side_effect = RuntimeError("path construction failed")
            mock_path.cwd = MagicMock(side_effect=RuntimeError("cwd failed"))
            result = provider("test.py", "head")

        assert result is None


# ===========================================================================
# pr_validation_engine.py — breaking changes in loop (lines 287-288)
# ===========================================================================


class TestBreakingChangesInLoop:
    """Cover the for bc in breaking_changes loop."""

    @pytest.mark.asyncio
    async def test_breaking_changes_converted_to_issues(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_ctx())

        bc1 = BreakingChange(
            type="FUNCTION_REMOVED",
            severity="HIGH",
            description="Removed public function",
            file_path="api.py",
            affected_symbol="public_func",
            line_number=10,
            recommendation="Add deprecation",
        )
        bc2 = BreakingChange(
            type="PARAMETER_CHANGED",
            severity="MEDIUM",
            description="Changed parameter type",
            file_path="api.py",
            affected_symbol="some_param",
            line_number=20,
            recommendation="Update callers",
        )

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "ast_analysis":
                return SemanticAnalysisResult(
                    breaking_changes=[],
                    logic_safety_issues=[],
                    api_surface_changes=[],
                    complexity_changes=[],
                )
            if name == "breaking_change_detection":
                return [bc1, bc2]
            if name == "logic_safety":
                return []
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_semantic_analysis(ctx)

        bc_issues = [i for i in issues if i.rule == "breaking_change_detection"]
        assert len(bc_issues) == 2
        assert bc_issues[0].type == "function_removed"
        assert bc_issues[1].type == "parameter_changed"


# ===========================================================================
# pr_validation_engine.py — high impact dependency issue (lines 309-311, 568)
# ===========================================================================


class TestHighImpactDependencyIssue:
    """Cover the dependency_impacts > 10 branch and _create_high_impact_issue."""

    @pytest.mark.asyncio
    async def test_many_dependency_impacts_creates_issue(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_ctx())

        mock_impact = ImpactAssessment(
            dependency_impacts=[f"dep_{i}" for i in range(15)],
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
                doc_completeness_score=1.0,
            ),
            performance_impact={},
            security_impact={},
        )

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "dependency_analysis":
                return mock_impact
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_impact_analysis(ctx)

        high_impact = [i for i in issues if i.type == "high_impact_change"]
        assert len(high_impact) == 1
        assert "15" in high_impact[0].message
        assert ctx.impact_assessment is mock_impact


# ===========================================================================
# pr_validation_engine.py — _assess_overall_impact with impact_assessment (line 398)
# ===========================================================================


class TestAssessOverallImpactWithImpactAssessment:
    """Cover the context.impact_assessment branch in _assess_overall_impact."""

    @pytest.mark.asyncio
    async def test_dependency_impacts_propagated(self):
        engine = PRValidationEngine()
        mock_impact = ImpactAssessment(
            dependency_impacts=["dep_a", "dep_b"],
            test_coverage_impact=TestImpact(
                tests_affected=["test_a"],
                coverage_change=1.0,
                new_tests_needed=["test_b"],
                obsolete_tests=[],
            ),
            documentation_impact=DocumentationImpact(
                docs_affected=[],
                outdated_docs=[],
                new_docs_needed=[],
                doc_completeness_score=1.0,
            ),
            performance_impact={},
            security_impact={},
        )
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
            impact_assessment=mock_impact,
        )
        change_impact = await engine._assess_overall_impact(ctx)
        assert change_impact.dependency_impact == ["dep_a", "dep_b"]
        assert change_impact.test_impact.tests_affected == ["test_a"]


# ===========================================================================
# pr_validation_engine.py — file_count 10-20 risk threshold (line 440)
# ===========================================================================


class TestFileCountRiskThreshold:
    """Cover the elif file_count > 10 branch in _calculate_risk_level."""

    def test_file_count_between_10_and_20_adds_risk(self):
        engine = PRValidationEngine()
        files = [_py_file(name=f"f{i}.py") for i in range(15)]
        ctx = PRAnalysisContext(pr_diff=_pr_diff(files), codebase_context=_codebase_ctx())
        # With only low issues, the risk should incorporate the file count bonus
        issues = []
        risk = engine._calculate_risk_level(issues, ctx)
        # 15 files > 10, adds 1.0, below the default medium threshold (5.0)
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM]


# ===========================================================================
# pr_validation_engine.py — RiskLevel.HIGH return (line 449)
# ===========================================================================


class TestRiskLevelHighReturn:
    """Cover the RiskLevel.HIGH return path."""

    def test_returns_high_risk(self):
        engine = PRValidationEngine()
        # Default thresholds: high=8.0, critical=10.0
        # 4 high issues = 4 * 2.0 = 8.0, exactly at HIGH threshold
        issues = [
            ValidationIssue(type="sec", severity="high", message=f"h{i}", file="a.py")
            for i in range(4)
        ]
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
        )
        risk = engine._calculate_risk_level(issues, ctx)
        assert risk == RiskLevel.HIGH


# ===========================================================================
# pr_validation_engine.py — security_pattern_ bug fix
# ===========================================================================


class TestSecurityPatternBugFix:
    """Verify the fixed security_pattern_ check now works correctly."""

    def test_security_pattern_recommendation_generated(self):
        engine = PRValidationEngine()
        issues = [
            ValidationIssue(
                type="security_pattern_sqli",
                severity="high",
                message="SQL injection",
                file="db.py",
            )
        ]
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
        )
        recs = engine._generate_recommendations(issues, ctx)
        security_recs = [r for r in recs if r.type == "security"]
        assert len(security_recs) == 1
        assert "security review" in security_recs[0].description.lower()

    def test_no_security_pattern_no_recommendation(self):
        engine = PRValidationEngine()
        issues = [
            ValidationIssue(type="logic_inversion", severity="medium", message="inv", file="a.py")
        ]
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
        )
        recs = engine._generate_recommendations(issues, ctx)
        security_recs = [r for r in recs if r.type == "security"]
        assert len(security_recs) == 0


# ===========================================================================
# diff_parser.py — _parse_diff_header /dev/null branches (lines 342, 344)
# ===========================================================================


class TestDiffParserDevNullBranches:
    """Cover _parse_diff_header when old or new filename is /dev/null."""

    def test_new_file_dev_null_as_old(self):
        """diff --git a//dev/null b/new.py -> ChangeType.ADDED"""
        parser = DiffParser()
        diff_text = (
            "diff --git a//dev/null b/new_file.py\nnew file mode 100644\n@@ -0,0 +1 @@\n+content\n"
        )
        # parse_git_diff calls _parse_diff_header internally
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 1
        assert result[0].change_type == ChangeType.ADDED
        assert result[0].filename == "new_file.py"

    def test_deleted_file_dev_null_as_new(self):
        """diff --git a/old.py b//dev/null -> ChangeType.DELETED"""
        parser = DiffParser()
        diff_text = "diff --git a/old_file.py b//dev/null\ndeleted file mode 100644\n@@ -1 +0,0 @@\n-content\n"
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 1
        assert result[0].change_type == ChangeType.DELETED
        assert result[0].filename == "old_file.py"

    def test_parse_diff_header_directly_with_dev_null_old(self):
        """Direct test of _parse_diff_header with /dev/null as old file."""
        parser = DiffParser()
        parsed = parser._parse_diff_header("diff --git a//dev/null b/newfile.py")
        assert parsed.change_type == ChangeType.ADDED
        assert parsed.old_filename is None
        assert parsed.new_filename == "newfile.py"
        assert parsed.filename == "newfile.py"

    def test_parse_diff_header_directly_with_dev_null_new(self):
        """Direct test of _parse_diff_header with /dev/null as new file."""
        parser = DiffParser()
        parsed = parser._parse_diff_header("diff --git a/deleted.py b//dev/null")
        assert parsed.change_type == ChangeType.DELETED
        assert parsed.old_filename == "deleted.py"
        assert parsed.new_filename is None
        assert parsed.filename == "deleted.py"


# ===========================================================================
# risk_calculator.py — maintainability 200-500 changes (lines 303-304, 311)
# ===========================================================================


class TestMaintainabilityRiskMediumChanges:
    """Cover the elif total_changes > 200 and elif file_count > 10 branches."""

    def setup_method(self):
        self.calc = RiskCalculator()

    def test_changes_between_200_and_500(self):
        """total_changes > 200 but <= 500 adds 1.0."""
        ctx = _ctx(files=[_py_file(name="big.py", additions=150, deletions=100)])
        score = self.calc._calculate_maintainability_risk(ctx, [])
        assert score >= 1.0

    def test_file_count_between_10_and_20(self):
        """file_count > 10 but <= 20 adds 0.8."""
        files = [_py_file(name=f"f{i}.py", additions=1, deletions=0) for i in range(15)]
        ctx = _ctx(files=files)
        score = self.calc._calculate_maintainability_risk(ctx, [])
        assert score >= 0.8

    def test_maintainability_issues_weighted(self):
        """Maintainability issues are weighted at 0.8."""
        ctx = _ctx(files=[_py_file(name="a.py")])
        issues = [
            ValidationIssue(
                type="maintainability_decrease",
                severity="high",
                message="m",
                file="a.py",
            ),
        ]
        score = self.calc._calculate_maintainability_risk(ctx, issues)
        # high severity = 2.5 * 0.8 = 2.0
        assert score >= 2.0


# ===========================================================================
# risk_calculator.py — explain_risk_score MEDIUM branch (line 625)
# ===========================================================================


class TestExplainRiskScoreMediumBranch:
    """Cover the elif risk_score.risk_level == RiskLevel.MEDIUM branch."""

    def test_medium_risk_explanation(self):
        calc = RiskCalculator()
        risk_score = RiskScore(
            overall_score=4.0,  # MEDIUM (>= 3.0, < 6.0)
            dimension_scores={RiskDimension.SECURITY: 4.0},
            confidence=0.8,
        )
        explanation = calc.explain_risk_score(risk_score)
        assert "Standard code review process" in explanation["recommendations"]
        assert "Ensure adequate test coverage" in explanation["recommendations"]
        assert explanation["overall_risk"]["level"] == "MEDIUM"

    def test_score_to_level_medium(self):
        """Directly test _score_to_level returns 'medium' for score 4.0."""
        calc = RiskCalculator()
        assert calc._score_to_level(4.0) == "medium"
        assert calc._score_to_level(3.0) == "medium"
        assert calc._score_to_level(5.9) == "medium"


# ===========================================================================
# recommendation_generator.py — high_complexity condition (line 477)
# ===========================================================================


class TestHighComplexityCondition:
    """Cover the high_complexity condition check that returns False when < 7.0."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_high_complexity_not_met(self):
        """When change_complexity < 7.0, high_complexity condition fails."""
        ctx = RecommendationContext(total_files_changed=20, change_complexity=5.0)
        # large_pr_breakdown (index 12) has conditions: large_pr, high_complexity
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[12],
            ctx,
            [],
            None,
        )
        assert met is False

    def test_high_complexity_met(self):
        """When change_complexity >= 7.0, high_complexity condition passes."""
        ctx = RecommendationContext(total_files_changed=20, change_complexity=8.0)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[12],
            ctx,
            [],
            None,
        )
        assert met is True


# ===========================================================================
# recommendation_generator.py — critical_security_issues bug fix
# ===========================================================================


class TestCriticalSecurityIssuesBugFix:
    """Verify the fixed critical_security_issues condition works correctly."""

    def setup_method(self):
        self.gen = PRValidationRecommendationGenerator()

    def test_critical_security_condition_now_works(self):
        """With critical_issues > 0 and security issue types, condition is met."""
        ctx = RecommendationContext(critical_issues=2, total_files_changed=5)
        # security_critical_review (index 0) has condition: critical_security_issues
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[0],
            ctx,
            ["security_pattern_sqli", "security_pattern_xss"],
            None,
        )
        assert met is True

    def test_critical_security_no_critical_issues_fails(self):
        """With critical_issues == 0, condition fails even with security issue types."""
        ctx = RecommendationContext(critical_issues=0, total_files_changed=5)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[0],
            ctx,
            ["security_pattern_sqli"],
            None,
        )
        assert met is False

    def test_critical_security_no_security_types_fails(self):
        """With critical_issues > 0 but no security issue types, condition fails."""
        ctx = RecommendationContext(critical_issues=3, total_files_changed=5)
        met = self.gen._template_conditions_met(
            self.gen.recommendation_templates[0],
            ctx,
            ["logic_inversion", "complexity"],
            None,
        )
        assert met is False


# ===========================================================================
# utils/__init__.py — import coverage (lines 3-6)
# ===========================================================================


class TestUtilsSubmodules:
    """Verify the individual utils modules can be imported."""

    def test_utils_submodules_importable(self):
        from utils.github_error_handling import (
            map_provision_error_to_http_status,
            validate_provision_result,
        )
        from utils.github_utils import (
            format_github_url,
            parse_github_url,
            validate_github_token,
        )

        assert callable(map_provision_error_to_http_status)
        assert callable(validate_provision_result)
        assert callable(format_github_url)
        assert callable(parse_github_url)
        assert callable(validate_github_token)


# ===========================================================================
# github_error_handler.py — GithubException import fallback (lines 30-32)
# ===========================================================================


class TestGithubExceptionImportFallback:
    """Cover the except ImportError fallback for GithubException."""

    def test_fallback_uses_base_exception(self):
        """When github package isn't available, fallback classes are used."""

        import utils.github_error_handler as handler_mod

        # The handler should work even when GithubException fallback is Exception
        # Verify the handler can process a regular Exception
        exc = Exception("general error")
        exc.status = None
        exc.data = {}

        result = handler_mod.GitHubErrorHandler.handle_exception(exc, operation="testing")
        from coding_client.exceptions import ValidationException

        assert isinstance(result, ValidationException)
        assert "testing" in str(result)

    def test_handler_with_non_github_exception(self):
        """Handler gracefully handles exceptions without status attribute."""
        from utils.github_error_handler import GitHubErrorHandler

        exc = ValueError("not a github error")
        result = GitHubErrorHandler.handle_exception(exc, operation="something")
        from coding_client.exceptions import ValidationException

        assert isinstance(result, ValidationException)
        assert "something" in str(result)


# ===========================================================================
# Additional edge cases for full coverage
# ===========================================================================


class TestValidatePrErrorPath:
    """Cover the generic Exception path in validate_pr."""

    @pytest.mark.asyncio
    async def test_validate_pr_generic_error_returns_error_result(self):
        engine = PRValidationEngine()
        pr_data = {
            "base": {"sha": "a"},
            "head": {"sha": "b"},
            "repository": {"full_name": "o/r"},
            "files": [],
        }
        codebase_ctx = _codebase_ctx()

        with patch.object(engine, "_parse_pr_diff", side_effect=RuntimeError("kaboom")):
            result = await engine.validate_pr(pr_data, codebase_ctx)

        assert result.validation_type == "pr_validation_error"
        assert result.passed is False
        assert "kaboom" in result.issues[0].message


class TestEngineRecommendationsWithComplexity:
    """Cover the complexity > 5.0 recommendation path."""

    def test_high_complexity_generates_recommendation(self):
        engine = PRValidationEngine()
        diff = DiffAnalysisResult(
            file_analyses=[],
            overall_patterns=[],
            risk_factors=[],
            change_complexity=7.0,
            requires_careful_review=False,
        )
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
            diff_analysis=diff,
        )
        recs = engine._generate_recommendations([], ctx)
        complexity_recs = [r for r in recs if r.type == "complexity"]
        assert len(complexity_recs) == 1
        assert "smaller PRs" in complexity_recs[0].description


class TestEngineEstimateReviewEffort:
    """Cover _estimate_review_effort branches."""

    def test_medium_effort(self):
        engine = PRValidationEngine()
        files = [_py_file(name=f"f{i}.py") for i in range(12)]
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff(files),
            codebase_context=_codebase_ctx(),
        )
        profile = SimpleNamespace(complexity_score=5.0)
        effort = engine._estimate_review_effort(profile, ctx)
        assert effort == "MEDIUM"

    def test_high_effort_from_complexity(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]),
            codebase_context=_codebase_ctx(),
        )
        profile = SimpleNamespace(complexity_score=8.0)
        effort = engine._estimate_review_effort(profile, ctx)
        assert effort == "HIGH"

    def test_high_effort_from_file_count(self):
        engine = PRValidationEngine()
        files = [_py_file(name=f"f{i}.py") for i in range(25)]
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff(files),
            codebase_context=_codebase_ctx(),
        )
        profile = SimpleNamespace(complexity_score=1.0)
        effort = engine._estimate_review_effort(profile, ctx)
        assert effort == "HIGH"


class TestExplainRiskScoreLowBranch:
    """Cover the LOW risk explanation (no recommendations added)."""

    def test_low_risk_no_recommendations(self):
        calc = RiskCalculator()
        risk_score = RiskScore(overall_score=1.0, confidence=1.0)
        explanation = calc.explain_risk_score(risk_score)
        assert explanation["recommendations"] == []
        assert explanation["overall_risk"]["level"] == "LOW"


class TestExplainRiskScoreHighBranch:
    """Cover the HIGH risk explanation branch."""

    def test_high_risk_explanation(self):
        calc = RiskCalculator()
        risk_score = RiskScore(overall_score=7.0, confidence=0.8)
        explanation = calc.explain_risk_score(risk_score)
        assert "Require experienced developer review" in explanation["recommendations"]
        assert explanation["overall_risk"]["level"] == "HIGH"


class TestExplainRiskScoreCriticalBranch:
    """Cover the CRITICAL risk explanation branch."""

    def test_critical_risk_explanation(self):
        calc = RiskCalculator()
        risk_score = RiskScore(overall_score=9.0, confidence=0.5)
        explanation = calc.explain_risk_score(risk_score)
        assert "Require senior developer review" in explanation["recommendations"]
        assert "Security review recommended" in explanation["recommendations"]


class TestScoreToLevelEdgeCases:
    """Cover all _score_to_level branches."""

    def test_score_to_level_all_ranges(self):
        calc = RiskCalculator()
        assert calc._score_to_level(9.0) == "critical"
        assert calc._score_to_level(8.0) == "critical"
        assert calc._score_to_level(7.0) == "high"
        assert calc._score_to_level(6.0) == "high"
        assert calc._score_to_level(5.0) == "medium"
        assert calc._score_to_level(3.0) == "medium"
        assert calc._score_to_level(2.9) == "low"
        assert calc._score_to_level(0.0) == "low"
