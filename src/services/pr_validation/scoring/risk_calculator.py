"""
Risk calculator for PR validation.

This module provides sophisticated risk calculation and scoring mechanisms
for pull request validation, combining multiple dimensions of analysis
into comprehensive risk assessments.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from coding_client.models.pr_validation import (
    PRAnalysisContext,
    RiskLevel,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


class RiskDimension(Enum):
    """Different dimensions of risk assessment."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    COMPATIBILITY = "compatibility"
    COMPLEXITY = "complexity"
    IMPACT_SCOPE = "impact_scope"


@dataclass
class RiskScore:
    """Represents a risk score with detailed breakdown."""

    overall_score: float
    dimension_scores: dict[RiskDimension, float] = field(default_factory=dict)
    confidence: float = 1.0
    risk_factors: list[str] = field(default_factory=list)
    mitigating_factors: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> RiskLevel:
        """Convert score to risk level."""
        if self.overall_score >= 8.0:
            return RiskLevel.CRITICAL
        elif self.overall_score >= 6.0:
            return RiskLevel.HIGH
        elif self.overall_score >= 3.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class RiskCalculationConfig:
    """Configuration for risk calculation."""

    # Dimension weights (should sum to 1.0)
    dimension_weights: dict[RiskDimension, float] = field(
        default_factory=lambda: {
            RiskDimension.SECURITY: 0.25,
            RiskDimension.PERFORMANCE: 0.15,
            RiskDimension.MAINTAINABILITY: 0.15,
            RiskDimension.RELIABILITY: 0.20,
            RiskDimension.COMPATIBILITY: 0.15,
            RiskDimension.COMPLEXITY: 0.05,
            RiskDimension.IMPACT_SCOPE: 0.05,
        }
    )

    # Issue severity weights
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "critical": 4.0,
            "high": 2.5,
            "medium": 1.5,
            "low": 0.5,
        }
    )

    # Change type risk multipliers
    change_type_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "COMPLETE_REWRITE": 2.0,
            "MAJOR": 1.5,
            "MODERATE": 1.0,
            "MINOR": 0.5,
        }
    )

    # File type risk factors
    file_type_risks: dict[str, float] = field(
        default_factory=lambda: {
            "python": 1.0,
            "javascript": 1.1,
            "typescript": 1.0,
            "yaml": 1.2,  # Config files are riskier
            "json": 1.2,
            "dockerfile": 1.3,
            "unknown": 0.8,
        }
    )


class RiskCalculator:
    """
    Calculates comprehensive risk scores for pull requests.

    This calculator combines multiple analysis dimensions to provide
    a nuanced risk assessment that considers:
    - Security vulnerabilities and regressions
    - Performance impact and degradation
    - Maintainability and technical debt
    - Reliability and error-prone patterns
    - API compatibility and breaking changes
    - Change complexity and scope
    """

    def __init__(self, config: RiskCalculationConfig | None = None) -> None:
        self.config = config or RiskCalculationConfig()

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the risk calculation configuration."""
        # Check that dimension weights sum to approximately 1.0
        total_weight = sum(self.config.dimension_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Dimension weights sum to {total_weight}, not 1.0")
            # Normalize weights
            for dimension in self.config.dimension_weights:
                self.config.dimension_weights[dimension] /= total_weight

    def calculate_overall_risk(
        self, analysis_context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> RiskScore:
        """
        Calculate overall risk score from analysis context and issues.

        Args:
            analysis_context: Complete analysis context
            issues: List of validation issues found

        Returns:
            RiskScore with detailed breakdown
        """
        try:
            # Calculate dimension scores
            dimension_scores = {}

            dimension_scores[RiskDimension.SECURITY] = self._calculate_security_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.PERFORMANCE] = self._calculate_performance_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.MAINTAINABILITY] = self._calculate_maintainability_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.RELIABILITY] = self._calculate_reliability_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.COMPATIBILITY] = self._calculate_compatibility_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.COMPLEXITY] = self._calculate_complexity_risk(
                analysis_context, issues
            )

            dimension_scores[RiskDimension.IMPACT_SCOPE] = self._calculate_impact_scope_risk(
                analysis_context, issues
            )

            # Calculate weighted overall score
            overall_score = 0.0
            for dimension, score in dimension_scores.items():
                weight = self.config.dimension_weights.get(dimension, 0.0)
                overall_score += score * weight

            # Identify risk and mitigating factors
            risk_factors = self._identify_risk_factors(analysis_context, issues)
            mitigating_factors = self._identify_mitigating_factors(analysis_context, issues)

            # Calculate confidence
            confidence = self._calculate_confidence(analysis_context, issues)

            return RiskScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                confidence=confidence,
                risk_factors=risk_factors,
                mitigating_factors=mitigating_factors,
            )

        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return RiskScore(
                overall_score=5.0,  # Default to medium risk
                confidence=0.0,
                risk_factors=[f"Risk calculation error: {str(e)}"],
            )

    def _calculate_security_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate security risk dimension score."""
        base_risk = 0.0

        # Count security-related issues
        security_issues = [i for i in issues if "security" in i.type.lower()]

        for issue in security_issues:
            severity_weight = self.config.severity_weights.get(issue.severity, 1.0)
            base_risk += severity_weight

        # Adjust for security-sensitive files
        security_files = 0
        for file_change in context.pr_diff.files_changed:
            filename_lower = file_change.filename.lower()
            if any(
                keyword in filename_lower
                for keyword in [
                    "auth",
                    "security",
                    "password",
                    "token",
                    "permission",
                ]
            ):
                security_files += 1

        if security_files > 0:
            base_risk += security_files * 0.5

        # Cap at 10.0
        return min(base_risk, 10.0)

    def _calculate_performance_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate performance risk dimension score."""
        base_risk = 0.0

        # Count performance-related issues
        performance_issues = [
            i for i in issues if "performance" in i.type.lower() or "complexity" in i.type.lower()
        ]

        for issue in performance_issues:
            severity_weight = self.config.severity_weights.get(issue.severity, 1.0)
            base_risk += severity_weight

        # Adjust for algorithmic complexity changes
        if context.diff_analysis:
            base_risk += context.diff_analysis.change_complexity * 0.3

        # Look for performance-sensitive patterns
        for file_change in context.pr_diff.files_changed:
            if file_change.patch:
                # Check for loops, database queries, etc.
                if "for " in file_change.patch or "while " in file_change.patch:
                    base_risk += 0.5
                if any(
                    db_word in file_change.patch.lower()
                    for db_word in ["query", "select", "database", "db."]
                ):
                    base_risk += 0.8

        return min(base_risk, 10.0)

    def _calculate_maintainability_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate maintainability risk dimension score."""
        base_risk = 0.0

        # Count maintainability-related issues
        maintainability_issues = [
            i
            for i in issues
            if any(
                keyword in i.type.lower()
                for keyword in [
                    "maintainability",
                    "complexity",
                    "code_smell",
                    "documentation",
                ]
            )
        ]

        for issue in maintainability_issues:
            severity_weight = self.config.severity_weights.get(issue.severity, 1.0)
            base_risk += severity_weight * 0.8  # Slightly lower weight than security

        # Adjust for change characteristics
        total_changes = sum(f.additions + f.deletions for f in context.pr_diff.files_changed)
        if total_changes > 500:
            base_risk += 2.0
        elif total_changes > 200:
            base_risk += 1.0

        # Adjust for file count
        file_count = len(context.pr_diff.files_changed)
        if file_count > 20:
            base_risk += 1.5
        elif file_count > 10:
            base_risk += 0.8

        return min(base_risk, 10.0)

    def _calculate_reliability_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate reliability risk dimension score."""
        base_risk = 0.0

        # Count reliability-related issues
        reliability_issues = [
            i
            for i in issues
            if any(
                keyword in i.type.lower()
                for keyword in [
                    "logic",
                    "error",
                    "exception",
                    "safety",
                    "validation",
                ]
            )
        ]

        for issue in reliability_issues:
            severity_weight = self.config.severity_weights.get(issue.severity, 1.0)
            # Logic and safety issues are critical for reliability
            if "logic" in issue.type.lower() or "safety" in issue.type.lower():
                base_risk += severity_weight * 1.5
            else:
                base_risk += severity_weight

        # Check for error handling changes
        for file_change in context.pr_diff.files_changed:
            if file_change.patch:
                removed_lines = [
                    line for line in file_change.patch.split("\n") if line.startswith("-")
                ]
                for line in removed_lines:
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "try:",
                            "except:",
                            "catch",
                            "error",
                            "validation",
                        ]
                    ):
                        base_risk += 0.5

        return min(base_risk, 10.0)

    def _calculate_compatibility_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate compatibility risk dimension score."""
        base_risk = 0.0

        # Count compatibility-related issues
        compatibility_issues = [
            i
            for i in issues
            if any(
                keyword in i.type.lower()
                for keyword in [
                    "breaking",
                    "compatibility",
                    "api",
                    "signature",
                ]
            )
        ]

        for issue in compatibility_issues:
            severity_weight = self.config.severity_weights.get(issue.severity, 1.0)
            base_risk += severity_weight * 1.2  # Breaking changes are serious

        # Check for public API changes
        for file_change in context.pr_diff.files_changed:
            # Look for function/class definitions in additions
            if file_change.patch:
                added_lines = [
                    line for line in file_change.patch.split("\n") if line.startswith("+")
                ]
                for line in added_lines:
                    # Public function/class changes
                    if any(
                        pattern in line
                        for pattern in [
                            "def ",
                            "class ",
                            "function ",
                            "export ",
                        ]
                    ):
                        base_risk += 0.3

        return min(base_risk, 10.0)

    def _calculate_complexity_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate complexity risk dimension score."""
        base_risk = 0.0

        # Use diff analysis complexity if available
        if context.diff_analysis:
            base_risk += context.diff_analysis.change_complexity

        # Count complexity-related issues
        complexity_issues = [i for i in issues if "complexity" in i.type.lower()]
        for issue in complexity_issues:
            base_risk += self.config.severity_weights.get(issue.severity, 1.0) * 0.5

        # Adjust for change patterns
        if context.diff_analysis:
            if "CONTAINS_COMPLETE_REWRITES" in context.diff_analysis.overall_patterns:
                base_risk += 2.0
            if "MULTIPLE_COMPLETE_REWRITES" in context.diff_analysis.overall_patterns:
                base_risk += 3.0

        return min(base_risk, 10.0)

    def _calculate_impact_scope_risk(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate impact scope risk dimension score."""
        base_risk = 0.0

        # File count impact
        file_count = len(context.pr_diff.files_changed)
        if file_count > 50:
            base_risk += 5.0
        elif file_count > 20:
            base_risk += 3.0
        elif file_count > 10:
            base_risk += 1.5

        # Dependency impact
        if context.impact_assessment:
            dependency_count = len(context.impact_assessment.dependency_impacts)
            base_risk += min(dependency_count * 0.1, 3.0)

        # Change volume impact
        total_changes = sum(f.additions + f.deletions for f in context.pr_diff.files_changed)
        if total_changes > 2000:
            base_risk += 2.0
        elif total_changes > 1000:
            base_risk += 1.0

        return min(base_risk, 10.0)

    def _identify_risk_factors(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> list[str]:
        """Identify specific risk factors."""
        risk_factors = []

        # Issue-based risk factors
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            risk_factors.append(f"{len(critical_issues)} critical issues found")

        high_issues = [i for i in issues if i.severity == "high"]
        if len(high_issues) > 5:
            risk_factors.append(f"Many high-severity issues ({len(high_issues)})")

        # Pattern-based risk factors
        if context.diff_analysis:
            if "DANGEROUS_LOGIC_CHANGES" in context.diff_analysis.risk_factors:
                risk_factors.append("Dangerous logic changes detected")

            if "SECURITY_SENSITIVE_FILE_MODIFIED" in context.diff_analysis.risk_factors:
                risk_factors.append("Security-sensitive files modified")

            if context.diff_analysis.requires_careful_review:
                risk_factors.append("Changes require careful review")

        # Scope-based risk factors
        large_pr_threshold = 20
        if len(context.pr_diff.files_changed) > large_pr_threshold:
            risk_factors.append("Large PR with many file changes")

        # Complexity-based risk factors
        if context.diff_analysis and context.diff_analysis.change_complexity > 7.0:
            risk_factors.append("High change complexity detected")

        return risk_factors

    def _identify_mitigating_factors(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> list[str]:
        """Identify factors that mitigate risk."""
        mitigating_factors = []

        # Test-related mitigations
        if context.impact_assessment:
            test_impact = context.impact_assessment.test_coverage_impact
            if test_impact and len(test_impact.tests_affected) > 0:
                mitigating_factors.append("Changes include test updates")

            if test_impact and test_impact.coverage_change >= 0:
                mitigating_factors.append("Test coverage maintained or updated")

        # Documentation mitigations
        doc_changes = [
            f
            for f in context.pr_diff.files_changed
            if f.filename.lower().endswith((".md", ".rst", ".txt"))
        ]
        if doc_changes:
            mitigating_factors.append("Documentation updated alongside code changes")

        # Change characteristics
        if context.diff_analysis:
            if len(context.diff_analysis.file_analyses) == 1:
                mitigating_factors.append("Single file change - limited scope")

            minor_changes = [
                a for a in context.diff_analysis.file_analyses if a.change_severity.value == "MINOR"
            ]
            if len(minor_changes) == len(context.diff_analysis.file_analyses):
                mitigating_factors.append("All changes are minor modifications")

        # Issue characteristics
        if not any(i.severity in ["critical", "high"] for i in issues):
            mitigating_factors.append("No critical or high-severity issues found")

        return mitigating_factors

    def _calculate_confidence(
        self, context: PRAnalysisContext, issues: list[ValidationIssue]
    ) -> float:
        """Calculate confidence in the risk assessment."""
        base_confidence = 1.0

        # Reduce confidence for missing analysis
        if not context.diff_analysis:
            base_confidence -= 0.2

        if not context.semantic_analysis:
            base_confidence -= 0.15

        if not context.impact_assessment:
            base_confidence -= 0.15

        # Reduce confidence for analysis errors
        analysis_errors = [i for i in issues if "error" in i.type.lower()]
        base_confidence -= len(analysis_errors) * 0.1

        # Reduce confidence for large, complex changes (harder to analyze accurately)
        if len(context.pr_diff.files_changed) > 30:
            base_confidence -= 0.1

        if context.diff_analysis and context.diff_analysis.change_complexity > 8.0:
            base_confidence -= 0.1

        return max(base_confidence, 0.0)

    def explain_risk_score(self, risk_score: RiskScore) -> dict[str, Any]:
        """Generate a detailed explanation of the risk score."""
        explanation: dict[str, Any] = {
            "overall_risk": {
                "score": risk_score.overall_score,
                "level": risk_score.risk_level.value,
                "confidence": risk_score.confidence,
            },
            "dimension_breakdown": {},
            "key_risk_factors": risk_score.risk_factors,
            "mitigating_factors": risk_score.mitigating_factors,
            "recommendations": [],
        }

        # Add dimension breakdown
        for dimension, score in risk_score.dimension_scores.items():
            weight = self.config.dimension_weights.get(dimension, 0.0)
            contribution = score * weight

            explanation["dimension_breakdown"][dimension.value] = {
                "score": score,
                "weight": weight,
                "contribution": contribution,
                "level": self._score_to_level(score),
            }

        # Add recommendations based on risk level
        if risk_score.risk_level == RiskLevel.CRITICAL:
            explanation["recommendations"].extend(
                [
                    "Require senior developer review",
                    "Consider breaking into smaller changes",
                    "Extensive testing required",
                    "Security review recommended",
                ]
            )
        elif risk_score.risk_level == RiskLevel.HIGH:
            explanation["recommendations"].extend(
                [
                    "Require experienced developer review",
                    "Additional testing recommended",
                    "Consider impact on dependent systems",
                ]
            )
        elif risk_score.risk_level == RiskLevel.MEDIUM:
            explanation["recommendations"].extend(
                [
                    "Standard code review process",
                    "Ensure adequate test coverage",
                    "Monitor for issues after deployment",
                ]
            )

        return explanation

    def _score_to_level(self, score: float) -> str:
        """Convert a numeric score to a risk level string."""
        if score >= 8.0:
            return "critical"
        elif score >= 6.0:
            return "high"
        elif score >= 3.0:
            return "medium"
        else:
            return "low"
