"""
Main PR validation engine for orchestrating comprehensive PR analysis.

This module coordinates all validation components to provide a complete
assessment of pull request changes including diff analysis, semantic analysis,
impact assessment, and context-aware validation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from coding_client.models.pr_validation import (
    APICompatibility,
    ChangeImpact,
    CodebaseContext,
    DocumentationImpact,
    PRAnalysisContext,
    PRDiff,
    Recommendation,
    RiskLevel,
    TestImpact,
    ValidationIssue,
    ValidationResult,
)

from ..analyzers.context.maintainability_assessor import MaintainabilityAssessor
from ..analyzers.context.performance_analyzer import PerformanceAnalyzer
from ..analyzers.context.security_regressor import SecurityRegressionDetector
from ..analyzers.diff.git_diff_analyzer import GitDiffAnalyzer
from ..analyzers.impact.dependency_analyzer import PRImpactAnalyzer
from ..analyzers.impact.documentation_checker import DocumentationSyncChecker
from ..analyzers.impact.test_coverage_correlator import TestCoverageCorrelator
from ..analyzers.semantic.ast_diff_analyzer import ASTDiffAnalyzer
from ..analyzers.semantic.breaking_change_detector import BreakingChangeDetector
from ..analyzers.semantic.logic_safety_validator import LogicSafetyValidator
from .change_categorizer import ChangeCategorizer
from .diff_parser import DiffParser
from .file_classifier import PRFileTypeClassifier

logger = logging.getLogger(__name__)


@dataclass
class PRValidationConfig:
    """Configuration for PR validation."""

    enable_diff_analysis: bool = True
    enable_semantic_analysis: bool = True
    enable_impact_assessment: bool = True
    enable_security_analysis: bool = True
    enable_performance_analysis: bool = True
    enable_maintainability_analysis: bool = True
    overall_timeout: int = 300
    analyzer_timeout: int = 60
    risk_thresholds: dict[str, float] | None = None
    analysis_intensity: int = 3

    def __post_init__(self) -> None:
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                "low": 2.0,
                "medium": 5.0,
                "high": 8.0,
                "critical": 10.0,
            }


@dataclass
class ValidationStats:
    """Statistics about the validation process."""

    total_files_analyzed: int = 0
    total_analysis_time: float = 0.0
    analyzer_times: dict[str, float] | None = None
    memory_peak: int = 0
    errors_encountered: list[str] | None = None

    def __post_init__(self) -> None:
        if self.analyzer_times is None:
            self.analyzer_times = {}
        if self.errors_encountered is None:
            self.errors_encountered = []


class PRValidationEngine:
    """
    Main orchestration engine for PR validation.

    This engine coordinates all validation components to provide comprehensive
    analysis of pull request changes including:
    - Diff analysis and change categorization
    - Semantic analysis and breaking change detection
    - Impact assessment across dependencies and tests
    - Security regression detection
    - Performance impact analysis
    - Maintainability assessment
    """

    def __init__(self, config: PRValidationConfig | None = None):
        self.config = config or PRValidationConfig()
        self.diff_parser = DiffParser()
        self.file_classifier = PRFileTypeClassifier()
        self.change_categorizer = ChangeCategorizer()
        self.diff_analyzer = GitDiffAnalyzer()
        self.ast_analyzer = ASTDiffAnalyzer()
        self.breaking_change_detector = BreakingChangeDetector()
        self.logic_safety_validator = LogicSafetyValidator()
        self.dependency_analyzer = PRImpactAnalyzer()
        self.test_coverage_correlator = TestCoverageCorrelator()
        self.documentation_checker = DocumentationSyncChecker()
        self.security_detector = SecurityRegressionDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.maintainability_assessor = MaintainabilityAssessor()
        self.stats = ValidationStats()

    async def validate_pr(
        self, pr_data: dict[str, Any], codebase_context: CodebaseContext
    ) -> ValidationResult:
        """
        Main entry point for PR validation.

        Args:
            pr_data: PR data including files, metadata, etc.
            codebase_context: Context about the codebase

        Returns:
            ValidationResult with comprehensive analysis
        """
        start_time = time.time()
        try:
            pr_diff = await self._parse_pr_diff(pr_data)
            self.stats.total_files_analyzed = len(pr_diff.files_changed)
            analysis_context = PRAnalysisContext(
                pr_diff=pr_diff,
                codebase_context=codebase_context,
                validation_config=self.config.__dict__,
            )
            validation_result = await self._execute_validation_pipeline(analysis_context)
            self.stats.total_analysis_time = time.time() - start_time
            validation_result.execution_time = self.stats.total_analysis_time
            logger.info(f"PR validation completed in {self.stats.total_analysis_time:.2f}s")
            return validation_result
        except TimeoutError:
            logger.error(f"PR validation timed out after {self.config.overall_timeout}s")
            return self._create_timeout_result()
        except Exception as e:
            logger.error(f"PR validation failed: {e}")
            return self._create_error_result(str(e))

    async def _parse_pr_diff(self, pr_data: dict[str, Any]) -> PRDiff:
        """Parse PR data into structured diff format."""
        base_sha = pr_data.get("base", {}).get("sha", "unknown")
        head_sha = pr_data.get("head", {}).get("sha", "unknown")
        files_data = pr_data.get("files", [])
        parsed_diffs = self.diff_parser.parse_github_diff(files_data)
        content_provider = self._create_content_provider(pr_data)
        file_changes = self.diff_parser.create_file_changes(parsed_diffs, content_provider)
        for file_change in file_changes:
            if file_change.file_type is None:
                file_change.file_type = self.file_classifier.classify_file(
                    file_change.filename, file_change.after_content
                )
        pr_diff = self.diff_parser.create_pr_diff(
            file_changes,
            base_sha,
            head_sha,
            metadata={
                "pr_number": pr_data.get("number"),
                "title": pr_data.get("title"),
                "description": pr_data.get("body"),
                "author": pr_data.get("user", {}).get("login"),
                "created_at": pr_data.get("created_at"),
                "updated_at": pr_data.get("updated_at"),
            },
        )
        return pr_diff

    def _create_content_provider(self, pr_data: dict[str, Any]):
        """Create a content provider function for fetching file contents."""

        def content_provider(filename: str, ref: str) -> str | None:
            """Fetch file content from GitHub API or local git repository."""
            try:
                repo_full_name = pr_data.get("repository", {}).get("full_name")
                base_sha = pr_data.get("base", {}).get("sha")
                head_sha = pr_data.get("head", {}).get("sha")
                target_sha = head_sha if ref == "head" else base_sha
                if not repo_full_name or not target_sha:
                    logger.warning(f"Missing repository info for content fetch: {filename}")
                    return None
                if hasattr(self, "github_service") and self.github_service:
                    try:
                        content = self.github_service.get_file_content(
                            repo_full_name, filename, target_sha
                        )
                        if content:
                            return str(content)
                    except Exception as e:
                        logger.warning(f"GitHub API content fetch failed for {filename}: {e}")
                try:
                    from pathlib import Path

                    potential_paths = [
                        Path(f"/tmp/repos/{repo_full_name.split('/')[-1]}") / filename,
                        Path(f"/tmp/repos/{repo_full_name}") / filename,
                        Path.cwd() / filename,
                    ]
                    for file_path in potential_paths:
                        if file_path.exists() and file_path.is_file():
                            try:
                                content = file_path.read_text(encoding="utf-8")
                                logger.debug(f"Found local content for {filename}")
                                return content
                            except UnicodeDecodeError:
                                try:
                                    content = file_path.read_text(encoding="latin-1")
                                    return content
                                except Exception:
                                    continue
                    logger.info(f"File not found locally: {filename}")
                    return None
                except Exception as e:
                    logger.warning(f"Local file access failed for {filename}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Content provider error for {filename}: {e}")
                return None

        return content_provider

    async def _execute_validation_pipeline(self, context: PRAnalysisContext) -> ValidationResult:
        """Execute the complete validation pipeline."""
        issues = []
        recommendations = []
        if self.config.enable_diff_analysis:
            diff_result = await self._run_analyzer_with_timeout(
                "diff_analysis", self.diff_analyzer.analyze_diff, context.pr_diff
            )
            if diff_result:
                context.diff_analysis = diff_result
        if self.config.enable_semantic_analysis:
            semantic_issues = await self._run_semantic_analysis(context)
            issues.extend(semantic_issues)
        if self.config.enable_impact_assessment:
            impact_issues = await self._run_impact_analysis(context)
            issues.extend(impact_issues)
        context_issues = await self._run_context_analysis(context)
        issues.extend(context_issues)
        change_impact = await self._assess_overall_impact(context)
        risk_level = self._calculate_risk_level(issues, context)
        recommendations = self._generate_recommendations(issues, context)
        score = self._calculate_validation_score(issues, risk_level)
        passed = self._determine_validation_pass(score, risk_level, issues)
        return ValidationResult(
            validation_type="comprehensive_pr_validation",
            passed=passed,
            score=score,
            risk_level=risk_level,
            issues=issues,
            recommendations=recommendations,
            change_impact=change_impact,
            confidence=self._calculate_confidence(issues, context),
        )

    async def _run_semantic_analysis(self, context: PRAnalysisContext) -> list:
        """Run semantic analysis phase."""
        issues = []
        for file_change in context.pr_diff.files_changed:
            if file_change.file_type.value in ["python", "javascript", "typescript"]:
                result = await self._run_analyzer_with_timeout(
                    "ast_analysis",
                    self.ast_analyzer.analyze_semantic_changes,
                    file_change,
                )
                if result:
                    issues.extend(result.logic_safety_issues)
        for file_change in context.pr_diff.files_changed:
            breaking_changes = await self._run_analyzer_with_timeout(
                "breaking_change_detection",
                self.breaking_change_detector.detect_breaking_changes,
                file_change,
            )
            if breaking_changes:
                for bc in breaking_changes:
                    issues.append(self._breaking_change_to_issue(bc))
        for file_change in context.pr_diff.files_changed:
            logic_issues = await self._run_analyzer_with_timeout(
                "logic_safety",
                self.logic_safety_validator.validate_logic_safety,
                file_change,
            )
            if logic_issues:
                issues.extend(logic_issues)
        return issues

    async def _run_impact_analysis(self, context: PRAnalysisContext) -> list:
        """Run impact analysis phase."""
        issues = []
        impact_assessment = await self._run_analyzer_with_timeout(
            "dependency_analysis",
            self.dependency_analyzer.assess_impact,
            context.pr_diff.files_changed,
            context.codebase_context,
        )
        if impact_assessment:
            context.impact_assessment = impact_assessment
            if len(impact_assessment.dependency_impacts) > 10:
                issues.append(
                    self._create_high_impact_issue(
                        "Many files affected by changes",
                        len(impact_assessment.dependency_impacts),
                    )
                )
        test_result = await self._run_analyzer_with_timeout(
            "test_coverage",
            self.test_coverage_correlator.correlate_test_coverage,
            context.pr_diff.files_changed,
            context.codebase_context,
        )
        if test_result and test_result.coverage_gaps:
            issues.extend(self._coverage_gaps_to_issues(test_result.coverage_gaps))
        doc_result = await self._run_analyzer_with_timeout(
            "documentation_sync",
            self.documentation_checker.check_documentation_sync,
            context.pr_diff.files_changed,
            context.codebase_context,
        )
        if doc_result and doc_result.outdated_docs:
            issues.extend(self._doc_issues_to_validation_issues(doc_result.outdated_docs))
        return issues

    async def _run_context_analysis(self, context: PRAnalysisContext) -> list:
        """Run context-aware analysis phase."""
        issues = []
        if self.config.enable_security_analysis:
            security_issues = await self._run_analyzer_with_timeout(
                "security_analysis",
                self.security_detector.detect_security_regressions,
                context.pr_diff.files_changed,
                context.codebase_context,
            )
            if security_issues:
                issues.extend(security_issues)
        if self.config.enable_performance_analysis:
            performance_issues = await self._run_analyzer_with_timeout(
                "performance_analysis",
                self.performance_analyzer.analyze_performance_impact,
                context.pr_diff.files_changed,
                context.codebase_context,
            )
            if performance_issues:
                issues.extend(performance_issues)
        if self.config.enable_maintainability_analysis:
            maintainability_issues = await self._run_analyzer_with_timeout(
                "maintainability_analysis",
                self.maintainability_assessor.assess_maintainability,
                context.pr_diff.files_changed,
                context.codebase_context,
            )
            if maintainability_issues:
                issues.extend(maintainability_issues)
        return issues

    async def _run_analyzer_with_timeout(self, analyzer_name: str, analyzer_func, *args, **kwargs):
        """Run an analyzer function with timeout and error handling."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                analyzer_func(*args, **kwargs), timeout=self.config.analyzer_timeout
            )
            elapsed_time = time.time() - start_time
            if self.stats.analyzer_times is not None:
                self.stats.analyzer_times[analyzer_name] = elapsed_time
            logger.debug(f"Analyzer {analyzer_name} completed in {elapsed_time:.2f}s")
            return result
        except TimeoutError:
            logger.warning(
                f"Analyzer {analyzer_name} timed out after {self.config.analyzer_timeout}s"
            )
            if self.stats.errors_encountered is not None:
                self.stats.errors_encountered.append(f"{analyzer_name}_timeout")
            return None
        except Exception as e:
            logger.error(f"Analyzer {analyzer_name} failed: {e}")
            if self.stats.errors_encountered is not None:
                self.stats.errors_encountered.append(f"{analyzer_name}_error: {str(e)}")
            return None

    async def _assess_overall_impact(self, context: PRAnalysisContext) -> ChangeImpact:
        """Assess the overall impact of changes."""
        affected_functions: list[str] = []
        affected_classes: list[str] = []
        dependency_impact: list[str] = []
        if context.impact_assessment:
            dependency_impact = context.impact_assessment.dependency_impacts
        change_profile = self.change_categorizer.categorize_pr_changes(
            context.pr_diff.files_changed, context.pr_diff.metadata
        )
        return ChangeImpact(
            affected_functions=affected_functions,
            affected_classes=affected_classes,
            dependency_impact=dependency_impact,
            test_impact=(
                context.impact_assessment.test_coverage_impact
                if context.impact_assessment
                else TestImpact([], 0.0, [], [])
            ),
            documentation_impact=(
                context.impact_assessment.documentation_impact
                if context.impact_assessment
                else DocumentationImpact([], [], [], 0.0)
            ),
            api_compatibility=APICompatibility(True, [], [], "PATCH"),
            estimated_effort=self._estimate_review_effort(change_profile, context),
        )

    def _calculate_risk_level(self, issues: list, context: PRAnalysisContext) -> RiskLevel:
        """Calculate overall risk level based on issues and context."""
        severity_counts = {
            "critical": len([i for i in issues if i.severity == "critical"]),
            "high": len([i for i in issues if i.severity == "high"]),
            "medium": len([i for i in issues if i.severity == "medium"]),
            "low": len([i for i in issues if i.severity == "low"]),
        }
        risk_score = (
            severity_counts["critical"] * 4.0
            + severity_counts["high"] * 2.0
            + severity_counts["medium"] * 1.0
            + severity_counts["low"] * 0.5
        )
        if context.diff_analysis:
            risk_score += context.diff_analysis.change_complexity
        file_count = len(context.pr_diff.files_changed)
        if file_count > 20:
            risk_score += 2.0
        elif file_count > 10:
            risk_score += 1.0
        thresholds = self.config.risk_thresholds or {
            "critical": 8.0,
            "high": 6.0,
            "medium": 4.0,
        }
        if risk_score >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_recommendations(self, issues: list, context: PRAnalysisContext) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        issue_types: dict[str, list[Any]] = {}
        for issue in issues:
            issue_type = issue.type
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        if any("security_pattern_" in issue_type for issue_type in issue_types):
            recommendations.append(
                self._create_recommendation(
                    "security",
                    "high",
                    "Conduct security review of changes",
                    "Review all security-related changes and consider penetration testing",
                )
            )
        if any("logic_inversion" in issue_type for issue_type in issue_types):
            recommendations.append(
                self._create_recommendation(
                    "logic_safety",
                    "high",
                    "Carefully review logic changes",
                    "Verify that logical operator changes are intentional and correct",
                )
            )
        if any("complexity" in issue_type for issue_type in issue_types):
            recommendations.append(
                self._create_recommendation(
                    "maintainability",
                    "medium",
                    "Consider refactoring complex code",
                    "Break down complex functions and classes for better maintainability",
                )
            )
        change_complexity = context.diff_analysis.change_complexity if context.diff_analysis else 0
        if change_complexity > 5.0:
            recommendations.append(
                self._create_recommendation(
                    "complexity",
                    "medium",
                    "Consider breaking changes into smaller PRs",
                    "Large complex changes are harder to review and more likely to introduce issues",
                )
            )
        return recommendations

    def _calculate_validation_score(self, issues: list, risk_level: RiskLevel) -> float:
        """Calculate overall validation score (0-10)."""
        base_score = 10.0
        for issue in issues:
            severity_deductions = {
                "critical": 3.0,
                "high": 2.0,
                "medium": 1.0,
                "low": 0.5,
            }
            deduction = severity_deductions.get(issue.severity, 0.5)
            base_score -= deduction
        if risk_level == RiskLevel.CRITICAL:
            base_score -= 2.0
        elif risk_level == RiskLevel.HIGH:
            base_score -= 1.0
        return max(base_score, 0.0)

    def _determine_validation_pass(self, score: float, risk_level: RiskLevel, issues: list) -> bool:
        """Determine if validation passes."""
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            return False
        if risk_level == RiskLevel.CRITICAL:
            return False
        if score < 5.0:
            return False
        return True

    def _calculate_confidence(self, issues: list, context: PRAnalysisContext) -> float:
        """Calculate confidence in the validation results."""
        base_confidence = 0.9
        error_count = len(self.stats.errors_encountered or [])
        confidence = base_confidence - error_count * 0.1
        if not context.diff_analysis:
            confidence -= 0.1
        if not context.impact_assessment:
            confidence -= 0.1
        return max(confidence, 0.0)

    def _estimate_review_effort(self, change_profile, context: PRAnalysisContext) -> str:
        """Estimate the effort required for reviewing changes."""
        complexity_score = change_profile.complexity_score
        file_count = len(context.pr_diff.files_changed)
        if complexity_score > 7.0 or file_count > 20:
            return "HIGH"
        elif complexity_score > 4.0 or file_count > 10:
            return "MEDIUM"
        else:
            return "LOW"

    def _breaking_change_to_issue(self, breaking_change):
        """Convert BreakingChange to ValidationIssue."""
        return ValidationIssue(
            type=breaking_change.type.lower(),
            severity=breaking_change.severity.lower(),
            message=breaking_change.description,
            file=breaking_change.file_path,
            line=breaking_change.line_number,
            rule="breaking_change_detection",
            suggestion=breaking_change.recommendation,
        )

    def _create_high_impact_issue(self, description: str, count: int):
        """Create a high impact validation issue."""
        return ValidationIssue(
            type="high_impact_change",
            severity="medium",
            message=f"{description}: {count} items affected",
            file="multiple",
            rule="impact_analysis",
            suggestion="Consider the scope of changes and ensure adequate testing",
        )

    def _coverage_gaps_to_issues(self, coverage_gaps: list) -> list:
        """Convert coverage gaps to validation issues."""
        issues = []
        for gap in coverage_gaps[:10]:
            issues.append(
                ValidationIssue(
                    type="test_coverage_gap",
                    severity=gap.severity,
                    message=f"Missing test coverage: {gap.symbol_name}",
                    file=gap.file_path,
                    line=gap.line_number,
                    rule="test_coverage",
                    suggestion=gap.suggested_action,
                )
            )
        return issues

    def _doc_issues_to_validation_issues(self, doc_issues: list[str]) -> list:
        """Convert documentation issues to validation issues."""
        issues = []
        for doc_issue in doc_issues[:5]:
            issues.append(
                ValidationIssue(
                    type="documentation_sync",
                    severity="low",
                    message=f"Documentation may need update: {doc_issue}",
                    file="documentation",
                    rule="documentation_sync",
                    suggestion="Review and update documentation to match code changes",
                )
            )
        return issues

    def _create_recommendation(self, category: str, priority: str, description: str, action: str):
        """Create a recommendation object."""
        return Recommendation(
            type=category,
            priority=priority,
            description=description,
            action=action,
            automated=False,
        )

    def _create_timeout_result(self) -> ValidationResult:
        """Create a validation result for timeout scenarios."""
        return ValidationResult(
            validation_type="pr_validation_timeout",
            passed=False,
            score=0.0,
            risk_level=RiskLevel.CRITICAL,
            issues=[
                ValidationIssue(
                    type="validation_timeout",
                    severity="critical",
                    message="PR validation timed out",
                    file="system",
                    rule="timeout",
                    suggestion="Reduce PR size or increase timeout limits",
                )
            ],
            recommendations=[],
            confidence=0.0,
        )

    def _create_error_result(self, error_message: str) -> ValidationResult:
        """Create a validation result for error scenarios."""
        return ValidationResult(
            validation_type="pr_validation_error",
            passed=False,
            score=0.0,
            risk_level=RiskLevel.CRITICAL,
            issues=[
                ValidationIssue(
                    type="validation_error",
                    severity="critical",
                    message=f"Validation error: {error_message}",
                    file="system",
                    rule="system_error",
                    suggestion="Check system logs and retry validation",
                )
            ],
            recommendations=[],
            confidence=0.0,
        )
