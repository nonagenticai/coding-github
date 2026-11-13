"""
Recommendation generator for PR validation.

This module generates actionable recommendations based on validation results,
providing specific guidance for addressing issues and updating code quality.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from coding_client.models.pr_validation import (
    ChangeSeverity,
    PRAnalysisContext,
    Recommendation,
    RiskLevel,
    ValidationIssue,
    ValidationResult,
)

from services.pr_validation.scoring.risk_calculator import (
    RiskDimension,
    RiskScore,
)

logger = logging.getLogger(__name__)


class RecommendationCategory(Enum):
    """Categories of recommendations."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    PROCESS = "process"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RecommendationTemplate:
    """Template for generating recommendations."""

    template_id: str
    category: RecommendationCategory
    title_template: str
    description_template: str
    action_template: str
    priority: RecommendationPriority
    conditions: list[str] = field(default_factory=list)
    issue_types: list[str] = field(default_factory=list)
    risk_dimensions: list[RiskDimension] = field(default_factory=list)
    automated: bool = False


@dataclass
class RecommendationContext:
    """Context for generating recommendations."""

    total_files_changed: int = 0
    total_issues: int = 0
    critical_issues: int = 0
    high_severity_issues: int = 0
    change_complexity: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    affected_components: list[str] = field(default_factory=list)
    test_coverage_change: float = 0.0
    documentation_updated: bool = False


class PRValidationRecommendationGenerator:
    """
    Generates actionable recommendations based on validation results.

    This generator analyzes validation results and provides specific,
    prioritized recommendations for addressing issues and updating
    code quality, security, and maintainability.
    """

    def __init__(self):
        self.recommendation_templates = self._initialize_recommendation_templates()
        self.priority_weights = {
            RecommendationPriority.CRITICAL: 10.0,
            RecommendationPriority.HIGH: 5.0,
            RecommendationPriority.MEDIUM: 2.0,
            RecommendationPriority.LOW: 1.0,
        }

    def _initialize_recommendation_templates(
        self,
    ) -> list[RecommendationTemplate]:
        """Initialize recommendation templates."""
        return [
            # Security recommendations
            RecommendationTemplate(
                template_id="security_critical_review",
                category=RecommendationCategory.SECURITY,
                title_template="Require security expert review",
                description_template="Critical security issues detected require expert review",
                action_template="Schedule security review with senior security engineer before merge",
                priority=RecommendationPriority.CRITICAL,
                conditions=["critical_security_issues"],
                issue_types=[
                    "security_pattern_",
                    "command_injection",
                    "sql_injection",
                ],
            ),
            RecommendationTemplate(
                template_id="security_vulnerability_fix",
                category=RecommendationCategory.SECURITY,
                title_template="Address security vulnerabilities",
                description_template="Security vulnerabilities found in code changes",
                action_template="Resolve security issues before merging: {issue_details}",
                priority=RecommendationPriority.HIGH,
                issue_types=[
                    "security_pattern_",
                    "hardcoded_secret",
                    "weak_crypto",
                ],
                risk_dimensions=[RiskDimension.SECURITY],
            ),
            RecommendationTemplate(
                template_id="security_control_restoration",
                category=RecommendationCategory.SECURITY,
                title_template="Restore removed security controls",
                description_template="Security controls appear to have been removed",
                action_template="Verify security control removal is intentional and implement alternatives",
                priority=RecommendationPriority.HIGH,
                issue_types=["security_control_removal"],
            ),
            # Performance recommendations
            RecommendationTemplate(
                template_id="performance_optimization_required",
                category=RecommendationCategory.PERFORMANCE,
                title_template="Optimize performance-critical code",
                description_template="Performance issues detected that may impact system performance",
                action_template="Review and optimize algorithmic complexity and resource usage",
                priority=RecommendationPriority.HIGH,
                issue_types=[
                    "performance_pattern_",
                    "complexity_increase",
                    "nested_loop",
                ],
                risk_dimensions=[RiskDimension.PERFORMANCE],
            ),
            RecommendationTemplate(
                template_id="caching_implementation",
                category=RecommendationCategory.PERFORMANCE,
                title_template="Implement caching for expensive operations",
                description_template="Repeated expensive operations detected",
                action_template="Add caching layer for database queries and API calls",
                priority=RecommendationPriority.MEDIUM,
                issue_types=["caching_opportunity", "db_query_in_loop"],
            ),
            # Maintainability recommendations
            RecommendationTemplate(
                template_id="complexity_reduction",
                category=RecommendationCategory.MAINTAINABILITY,
                title_template="Reduce code complexity",
                description_template="High complexity code detected that may be hard to maintain",
                action_template="Break down complex functions and classes into smaller, focused units",
                priority=RecommendationPriority.MEDIUM,
                issue_types=[
                    "high_cyclomatic_complexity",
                    "long_functions",
                    "god_class",
                ],
                risk_dimensions=[
                    RiskDimension.MAINTAINABILITY,
                    RiskDimension.COMPLEXITY,
                ],
            ),
            RecommendationTemplate(
                template_id="code_smell_cleanup",
                category=RecommendationCategory.MAINTAINABILITY,
                title_template="Address code smells",
                description_template="Code smells detected that impact maintainability",
                action_template="Refactor code to eliminate identified anti-patterns",
                priority=RecommendationPriority.LOW,
                issue_types=["code_smell_", "anti_pattern_"],
            ),
            # Testing recommendations
            RecommendationTemplate(
                template_id="test_coverage_improvement",
                category=RecommendationCategory.TESTING,
                title_template="Update test coverage",
                description_template="Code changes lack adequate test coverage",
                action_template="Add unit tests for new functionality and edge cases",
                priority=RecommendationPriority.MEDIUM,
                issue_types=["test_coverage_gap", "missing_tests"],
                conditions=["low_test_coverage"],
            ),
            RecommendationTemplate(
                template_id="integration_testing",
                category=RecommendationCategory.TESTING,
                title_template="Add integration tests",
                description_template="Complex changes affecting multiple components",
                action_template="Create integration tests to verify component interactions",
                priority=RecommendationPriority.MEDIUM,
                conditions=[
                    "multiple_components_affected",
                    "high_change_complexity",
                ],
            ),
            # Documentation recommendations
            RecommendationTemplate(
                template_id="documentation_update",
                category=RecommendationCategory.DOCUMENTATION,
                title_template="Update documentation",
                description_template="Code changes require documentation updates",
                action_template="Update relevant documentation, API docs, and README files",
                priority=RecommendationPriority.LOW,
                issue_types=["documentation_sync", "missing_docstring"],
                conditions=["api_changes", "new_features"],
            ),
            # Architecture recommendations
            RecommendationTemplate(
                template_id="breaking_change_migration",
                category=RecommendationCategory.ARCHITECTURE,
                title_template="Plan breaking change migration",
                description_template="Breaking changes detected that affect API compatibility",
                action_template="Create migration guide and deprecation timeline for breaking changes",
                priority=RecommendationPriority.HIGH,
                issue_types=["breaking_change_", "api_compatibility"],
                risk_dimensions=[RiskDimension.COMPATIBILITY],
            ),
            RecommendationTemplate(
                template_id="dependency_impact_review",
                category=RecommendationCategory.ARCHITECTURE,
                title_template="Review dependency impact",
                description_template="Changes affect many dependent components",
                action_template="Coordinate with teams owning dependent components",
                priority=RecommendationPriority.MEDIUM,
                conditions=["high_dependency_impact"],
            ),
            # Process recommendations
            RecommendationTemplate(
                template_id="large_pr_breakdown",
                category=RecommendationCategory.PROCESS,
                title_template="Consider breaking down large PR",
                description_template="PR is large and complex, making review difficult",
                action_template="Split into smaller, focused PRs for easier review",
                priority=RecommendationPriority.MEDIUM,
                conditions=["large_pr", "high_complexity"],
            ),
            RecommendationTemplate(
                template_id="staged_deployment",
                category=RecommendationCategory.PROCESS,
                title_template="Plan staged deployment",
                description_template="High-risk changes require careful deployment",
                action_template="Deploy incrementally with monitoring and rollback plan",
                priority=RecommendationPriority.HIGH,
                conditions=["high_risk_level", "critical_system_changes"],
            ),
        ]

    def generate_recommendations(
        self,
        validation_result: ValidationResult,
        analysis_context: PRAnalysisContext,
        risk_score: RiskScore | None = None,
    ) -> list[Recommendation]:
        """
        Generate actionable recommendations based on validation results.

        Args:
            validation_result: Complete validation result
            analysis_context: Analysis context
            risk_score: Optional risk score for additional context

        Returns:
            List of prioritized recommendations
        """
        try:
            # Build recommendation context
            context = self._build_recommendation_context(
                validation_result, analysis_context, risk_score
            )

            # Generate recommendations from templates
            recommendations = []

            # Template-based recommendations
            template_recs = self._generate_template_recommendations(
                validation_result.issues, context, risk_score
            )
            recommendations.extend(template_recs)

            # Custom recommendations based on specific patterns
            custom_recs = self._generate_custom_recommendations(
                validation_result, analysis_context, context
            )
            recommendations.extend(custom_recs)

            # Risk-based recommendations
            if risk_score:
                risk_recs = self._generate_risk_based_recommendations(risk_score, context)
                recommendations.extend(risk_recs)

            # Prioritize and deduplicate
            recommendations = self._prioritize_recommendations(recommendations)
            recommendations = self._deduplicate_recommendations(recommendations)

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [
                Recommendation(
                    type="recommendation_error",
                    priority="medium",
                    description="Failed to generate recommendations",
                    action="Manual review recommended",
                    automated=False,
                )
            ]

    def _build_recommendation_context(
        self,
        validation_result: ValidationResult,
        analysis_context: PRAnalysisContext,
        risk_score: RiskScore | None,
    ) -> RecommendationContext:
        """Build context for recommendation generation."""
        context = RecommendationContext()

        # Basic metrics
        context.total_files_changed = len(analysis_context.pr_diff.files_changed)
        context.total_issues = len(validation_result.issues)
        context.risk_level = validation_result.risk_level

        # Issue severity breakdown
        for issue in validation_result.issues:
            if issue.severity == "critical":
                context.critical_issues += 1
            elif issue.severity == "high":
                context.high_severity_issues += 1

        # Change complexity
        if analysis_context.diff_analysis:
            context.change_complexity = analysis_context.diff_analysis.change_complexity

        # Test coverage
        if (
            analysis_context.impact_assessment
            and analysis_context.impact_assessment.test_coverage_impact
        ):
            test_impact = analysis_context.impact_assessment.test_coverage_impact
            context.test_coverage_change = test_impact.coverage_change

        # Documentation updates
        doc_files = [
            f
            for f in analysis_context.pr_diff.files_changed
            if f.filename.lower().endswith((".md", ".rst", ".txt"))
        ]
        context.documentation_updated = len(doc_files) > 0

        return context

    def _generate_template_recommendations(
        self,
        issues: list[ValidationIssue],
        context: RecommendationContext,
        risk_score: RiskScore | None,
    ) -> list[Recommendation]:
        """Generate recommendations based on templates."""
        recommendations = []

        # Group issues by type for easier template matching
        issue_types = [issue.type for issue in issues]
        Counter(issue_types)

        for template in self.recommendation_templates:
            # Check if template conditions are met
            if not self._template_conditions_met(template, context, issue_types, risk_score):
                continue

            # Check if relevant issue types are present
            if template.issue_types:
                matching_issues = [
                    issue
                    for issue in issues
                    if any(issue_pattern in issue.type for issue_pattern in template.issue_types)
                ]
                if not matching_issues:
                    continue

                # Generate recommendation with issue-specific details
                recommendation = self._create_recommendation_from_template(
                    template, matching_issues, context
                )
                recommendations.append(recommendation)

            # Check risk dimension conditions
            elif template.risk_dimensions and risk_score:
                high_risk_dimensions = [
                    dim
                    for dim, score in risk_score.dimension_scores.items()
                    if score >= 5.0 and dim in template.risk_dimensions
                ]
                if high_risk_dimensions:
                    recommendation = self._create_recommendation_from_template(
                        template, [], context, high_risk_dimensions
                    )
                    recommendations.append(recommendation)

        return recommendations

    def _template_conditions_met(
        self,
        template: RecommendationTemplate,
        context: RecommendationContext,
        issue_types: list[str],
        risk_score: RiskScore | None,
    ) -> bool:
        """Check if template conditions are met."""
        if not template.conditions:
            return True

        for condition in template.conditions:
            if condition == "critical_security_issues":
                security_critical = context.critical_issues > 0 and any(
                    "security" in issue_type for issue_type in issue_types
                )
                if not security_critical:
                    return False

            elif condition == "low_test_coverage":
                if context.test_coverage_change >= 0:  # Coverage maintained or updated
                    return False

            elif condition == "multiple_components_affected":
                if context.total_files_changed < 5:
                    return False

            elif condition == "high_change_complexity":
                if context.change_complexity < 5.0:
                    return False

            elif condition == "api_changes":
                if not any(
                    "breaking" in issue_type or "api" in issue_type for issue_type in issue_types
                ):
                    return False

            elif condition == "new_features":
                # Basic heuristic: many additions suggest new features
                if context.total_files_changed < 3:
                    return False

            elif condition == "high_dependency_impact":
                # Would need dependency impact data
                pass

            elif condition == "large_pr":
                if context.total_files_changed < 15:
                    return False

            elif condition == "high_complexity":
                if context.change_complexity < 7.0:
                    return False

            elif condition == "high_risk_level":
                if context.risk_level not in [
                    RiskLevel.HIGH,
                    RiskLevel.CRITICAL,
                ]:
                    return False

            elif condition == "critical_system_changes":
                # Would need system context
                pass

        return True

    def _create_recommendation_from_template(
        self,
        template: RecommendationTemplate,
        matching_issues: list[ValidationIssue],
        context: RecommendationContext,
        risk_dimensions: list[RiskDimension] | None = None,
    ) -> Recommendation:
        """Create a recommendation from a template."""
        # Build issue details for action template
        issue_details = ""
        if matching_issues:
            issue_summary = {}
            for issue in matching_issues[:5]:  # Limit to top 5 issues
                if issue.type not in issue_summary:
                    issue_summary[issue.type] = 0
                issue_summary[issue.type] += 1

            details = [f"{count} {issue_type}" for issue_type, count in issue_summary.items()]
            issue_details = ", ".join(details)

        # Format action template
        action = template.action_template.format(
            issue_details=issue_details,
            file_count=context.total_files_changed,
            complexity=context.change_complexity,
        )

        return Recommendation(
            type=template.template_id,
            priority=template.priority.value,
            description=template.description_template,
            action=action,
            automated=template.automated,
        )

    def _generate_custom_recommendations(
        self,
        validation_result: ValidationResult,
        analysis_context: PRAnalysisContext,
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """Generate custom recommendations based on specific patterns."""
        recommendations = []

        # Logic safety specific recommendations
        logic_issues = [i for i in validation_result.issues if "logic" in i.type]
        if logic_issues:
            recommendations.append(
                Recommendation(
                    type="logic_safety_review",
                    priority="high",
                    description="Logic changes detected that require careful verification",
                    action=f"Manually verify {len(logic_issues)} logic changes for correctness",
                    automated=False,
                )
            )

        # Complete rewrite recommendations
        if analysis_context.diff_analysis:
            complete_rewrites = [
                f
                for f in analysis_context.diff_analysis.file_analyses
                if f.change_severity == ChangeSeverity.COMPLETE_REWRITE
            ]
            if complete_rewrites:
                recommendations.append(
                    Recommendation(
                        type="complete_rewrite_review",
                        priority="high",
                        description=f"{len(complete_rewrites)} files completely rewritten",
                        action="Verify rewritten files maintain expected functionality and contracts",
                        automated=False,
                    )
                )

        # Test impact recommendations
        if (
            analysis_context.impact_assessment
            and analysis_context.impact_assessment.test_coverage_impact
        ):
            test_impact = analysis_context.impact_assessment.test_coverage_impact
            if test_impact.obsolete_tests:
                recommendations.append(
                    Recommendation(
                        type="obsolete_tests_cleanup",
                        priority="low",
                        description=f"{len(test_impact.obsolete_tests)} obsolete tests detected",
                        action="Remove or update tests that no longer have corresponding code",
                        automated=False,
                    )
                )

        return recommendations

    def _generate_risk_based_recommendations(
        self, risk_score: RiskScore, context: RecommendationContext
    ) -> list[Recommendation]:
        """Generate recommendations based on risk score analysis."""
        recommendations = []

        # Overall risk level recommendations
        if risk_score.risk_level == RiskLevel.CRITICAL:
            recommendations.append(
                Recommendation(
                    type="critical_risk_mitigation",
                    priority="critical",
                    description="Critical risk level requires immediate attention",
                    action="Require senior developer approval and extensive testing before merge",
                    automated=False,
                )
            )

        # Dimension-specific recommendations
        for dimension, score in risk_score.dimension_scores.items():
            if score >= 7.0:  # High risk in this dimension
                if dimension == RiskDimension.SECURITY:
                    recommendations.append(
                        Recommendation(
                            type="security_expert_review",
                            priority="high",
                            description=f"High security risk (score: {score:.1f})",
                            action="Require security team review before deployment",
                            automated=False,
                        )
                    )

                elif dimension == RiskDimension.PERFORMANCE:
                    recommendations.append(
                        Recommendation(
                            type="performance_testing",
                            priority="medium",
                            description=f"High performance risk (score: {score:.1f})",
                            action="Run performance benchmarks and load testing",
                            automated=True,
                        )
                    )

                elif dimension == RiskDimension.RELIABILITY:
                    recommendations.append(
                        Recommendation(
                            type="reliability_verification",
                            priority="high",
                            description=f"High reliability risk (score: {score:.1f})",
                            action="Increase test coverage and add error handling verification",
                            automated=False,
                        )
                    )

        # Confidence-based recommendations
        if risk_score.confidence < 0.5:
            recommendations.append(
                Recommendation(
                    type="manual_review_required",
                    priority="medium",
                    description=f"Low confidence in automated analysis ({risk_score.confidence:.1%})",
                    action="Increase manual code review focus due to analysis limitations",
                    automated=False,
                )
            )

        return recommendations

    def _prioritize_recommendations(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Sort recommendations by priority and impact."""
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        return sorted(
            recommendations,
            key=lambda r: priority_order.get(r.priority, 0),
            reverse=True,
        )

    def _deduplicate_recommendations(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique_recommendations = []

        for rec in recommendations:
            # Create a basic hash based on type and action
            rec_hash = f"{rec.type}:{rec.action[:50]}"
            if rec_hash not in seen:
                seen.add(rec_hash)
                unique_recommendations.append(rec)

        return unique_recommendations

    def generate_summary_recommendations(
        self, recommendations: list[Recommendation]
    ) -> dict[str, Any]:
        """Generate a summary of recommendations."""
        summary: dict[str, Any] = {
            "total_recommendations": len(recommendations),
            "by_priority": {},
            "by_category": {},
            "automated_count": 0,
            "top_priorities": [],
        }

        # Count by priority
        for rec in recommendations:
            priority = rec.priority
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1

            if rec.automated:
                summary["automated_count"] += 1

        # Count by category (based on type)
        category_mapping = {
            "security": "Security",
            "performance": "Performance",
            "maintainability": "Maintainability",
            "test": "Testing",
            "documentation": "Documentation",
            "architecture": "Architecture",
            "process": "Process",
        }

        for rec in recommendations:
            category = "Other"
            for key, cat in category_mapping.items():
                if key in rec.type.lower():
                    category = cat
                    break
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

        # Top priority recommendations
        critical_and_high = [r for r in recommendations if r.priority in ["critical", "high"]]
        summary["top_priorities"] = [
            {
                "type": rec.type,
                "priority": rec.priority,
                "description": rec.description,
                "action": rec.action,
            }
            for rec in critical_and_high[:5]
        ]

        return summary
