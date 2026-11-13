"""
Impact scoring and metrics calculation module.

This module handles the calculation of impact scores and metrics
for dependency analysis and change assessment.
"""

import logging
from dataclasses import dataclass
from typing import Any

from coding_client.models.pr_validation import FileChange

logger = logging.getLogger(__name__)


@dataclass
class ImpactResult:
    """Result of dependency impact analysis."""

    affected_files: list[str]
    affected_functions: list[str]
    affected_classes: list[str]
    affected_modules: list[str]
    impact_score: float
    circular_dependencies: list[tuple[str, str]]
    missing_dependencies: list[str]
    recommendations: list[str]


class ImpactScorer:
    """
    Calculates impact scores and generates recommendations.

    This scorer provides functionality for:
    - Impact score calculation based on various factors
    - Risk assessment and categorization
    - Recommendation generation
    - Metrics aggregation and analysis
    """

    def __init__(self):
        # Scoring weights for different impact factors
        self.scoring_weights = {
            "files": 0.1,
            "functions": 0.05,
            "classes": 0.1,
            "modules": 0.08,
            "dependencies": 0.15,
            "test_coverage": 0.2,
            "circular_deps": 0.3,
            "security_risk": 0.4,
            "performance_risk": 0.2,
        }

    def analyze_dependency_impacts(
        self,
        changes: list[FileChange],
        affected_files: set[str],
        affected_functions: set[str],
        affected_classes: set[str],
        affected_modules: set[str],
        circular_dependencies: list[tuple[str, str]],
        missing_dependencies: list[str],
    ) -> ImpactResult:
        """
        Analyze the impact of changes on dependencies.

        Args:
            changes: List of file changes
            affected_files: Set of affected file paths
            affected_functions: Set of affected function names
            affected_classes: Set of affected class names
            affected_modules: Set of affected module names
            circular_dependencies: List of circular dependency pairs
            missing_dependencies: List of missing dependency identifiers

        Returns:
            ImpactResult with comprehensive analysis
        """
        recommendations: list[str] = []

        # Calculate impact score
        impact_score = self._calculate_impact_score(
            affected_files,
            affected_functions,
            affected_classes,
            affected_modules,
            circular_dependencies,
            missing_dependencies,
        )

        # Generate recommendations based on analysis
        recommendations.extend(self._generate_file_recommendations(affected_files))
        recommendations.extend(self._generate_function_recommendations(affected_functions))
        recommendations.extend(
            self._generate_dependency_recommendations(circular_dependencies, missing_dependencies)
        )

        return ImpactResult(
            affected_files=list(affected_files),
            affected_functions=list(affected_functions),
            affected_classes=list(affected_classes),
            affected_modules=list(affected_modules),
            impact_score=impact_score,
            circular_dependencies=circular_dependencies,
            missing_dependencies=missing_dependencies,
            recommendations=recommendations,
        )

    def _calculate_impact_score(
        self,
        affected_files: set[str],
        affected_functions: set[str],
        affected_classes: set[str],
        affected_modules: set[str],
        circular_dependencies: list[tuple[str, str]],
        missing_dependencies: list[str],
    ) -> float:
        """Calculate an impact score based on the scope of changes."""
        score = 0.0

        # Base score from number of affected components
        score += len(affected_files) * self.scoring_weights["files"]
        score += len(affected_functions) * self.scoring_weights["functions"]
        score += len(affected_classes) * self.scoring_weights["classes"]
        score += len(affected_modules) * self.scoring_weights["modules"]

        # Penalty for circular dependencies
        if circular_dependencies:
            score += len(circular_dependencies) * self.scoring_weights["circular_deps"]

        # Penalty for missing dependencies
        if missing_dependencies:
            score += len(missing_dependencies) * self.scoring_weights["dependencies"]

        # Apply multipliers for critical issues
        if len(circular_dependencies) > 5:
            score *= 1.5  # Significant circular dependency problem

        if len(missing_dependencies) > 10:
            score *= 1.3  # Many missing dependencies

        # Cap the score at 10.0
        return min(score, 10.0)

    def calculate_risk_level(self, impact_score: float) -> str:
        """
        Calculate risk level based on impact score.

        Args:
            impact_score: The calculated impact score

        Returns:
            Risk level as a string
        """
        if impact_score <= 2.0:
            return "low"
        elif impact_score <= 5.0:
            return "medium"
        elif impact_score <= 8.0:
            return "high"
        else:
            return "critical"

    def calculate_complexity_score(self, changes: list[FileChange]) -> dict[str, Any]:
        """
        Calculate complexity score based on changes.

        Args:
            changes: List of file changes

        Returns:
            Dictionary containing complexity metrics
        """
        total_lines_changed = sum(change.additions + change.deletions for change in changes)
        total_files = len(changes)

        # Calculate average change size
        avg_change_size = total_lines_changed / total_files if total_files > 0 else 0

        # Categorize complexity
        if avg_change_size <= 10:
            complexity_level = "simple"
        elif avg_change_size <= 50:
            complexity_level = "moderate"
        elif avg_change_size <= 100:
            complexity_level = "complex"
        else:
            complexity_level = "very_complex"

        # Calculate distribution of change sizes
        change_sizes = [change.additions + change.deletions for change in changes]
        change_sizes.sort()

        return {
            "complexity_level": complexity_level,
            "average_change_size": avg_change_size,
            "total_lines_changed": total_lines_changed,
            "total_files_changed": total_files,
            "largest_change": max(change_sizes) if change_sizes else 0,
            "smallest_change": min(change_sizes) if change_sizes else 0,
            "median_change": (change_sizes[len(change_sizes) // 2] if change_sizes else 0),
        }

    def calculate_test_impact_score(
        self,
        tests_affected: list[str],
        new_tests_needed: list[str],
        coverage_change: float,
    ) -> dict[str, Any]:
        """
        Calculate test impact score.

        Args:
            tests_affected: List of affected test files
            new_tests_needed: List of new tests that should be created
            coverage_change: Estimated coverage change percentage

        Returns:
            Dictionary containing test impact metrics
        """
        # Base score from affected tests
        score = len(tests_affected) * 0.1

        # Penalty for missing tests
        score += len(new_tests_needed) * 0.2

        # Factor in coverage change
        if coverage_change < 0:
            score += abs(coverage_change) * 2  # Penalize coverage decrease

        # Determine impact level
        if score <= 1.0:
            impact_level = "low"
        elif score <= 3.0:
            impact_level = "medium"
        elif score <= 5.0:
            impact_level = "high"
        else:
            impact_level = "critical"

        return {
            "test_impact_score": min(score, 10.0),
            "impact_level": impact_level,
            "tests_affected_count": len(tests_affected),
            "new_tests_needed_count": len(new_tests_needed),
            "coverage_change": coverage_change,
            "recommendations": self._generate_test_recommendations(
                tests_affected, new_tests_needed, coverage_change
            ),
        }

    def _generate_file_recommendations(self, affected_files: set[str]) -> list[str]:
        """Generate recommendations based on affected files."""
        recommendations = []

        if len(affected_files) > 20:
            recommendations.append(
                "Large number of affected files detected - consider breaking changes into smaller PRs"
            )
        elif len(affected_files) > 10:
            recommendations.append(
                "Moderate number of files affected - ensure comprehensive testing"
            )

        # Check for critical file patterns
        critical_files = [f for f in affected_files if self._is_critical_file(f)]
        if critical_files:
            recommendations.append(
                f"Critical files affected ({len(critical_files)} files) - require extra review and testing"
            )

        return recommendations

    def _generate_function_recommendations(self, affected_functions: set[str]) -> list[str]:
        """Generate recommendations based on affected functions."""
        recommendations = []

        if len(affected_functions) > 50:
            recommendations.append(
                "Very large number of functions affected - consider architectural review"
            )
        elif len(affected_functions) > 20:
            recommendations.append(
                "Many functions affected - ensure comprehensive testing and documentation updates"
            )

        return recommendations

    def _generate_dependency_recommendations(
        self,
        circular_dependencies: list[tuple[str, str]],
        missing_dependencies: list[str],
    ) -> list[str]:
        """Generate recommendations based on dependency issues."""
        recommendations = []

        if circular_dependencies:
            if len(circular_dependencies) > 10:
                recommendations.append(
                    f"Critical: {len(circular_dependencies)} circular dependencies detected - "
                    "immediate refactoring required to break cycles"
                )
            elif len(circular_dependencies) > 5:
                recommendations.append(
                    f"Warning: {len(circular_dependencies)} circular dependencies detected - "
                    "consider refactoring to improve maintainability"
                )
            else:
                recommendations.append(
                    f"Minor circular dependencies detected ({len(circular_dependencies)}) - "
                    "monitor and consider refactoring"
                )

        if missing_dependencies:
            if len(missing_dependencies) > 15:
                recommendations.append(
                    f"Critical: {len(missing_dependencies)} missing dependencies - "
                    "ensure all dependencies are properly declared and accessible"
                )
            elif len(missing_dependencies) > 5:
                recommendations.append(
                    f"Warning: {len(missing_dependencies)} missing dependencies - verify dependency declarations"
                )
            else:
                recommendations.append(
                    f"Minor missing dependencies detected ({len(missing_dependencies)}) - verify import statements"
                )

        return recommendations

    def _generate_test_recommendations(
        self,
        tests_affected: list[str],
        new_tests_needed: list[str],
        coverage_change: float,
    ) -> list[str]:
        """Generate test-related recommendations."""
        recommendations = []

        if new_tests_needed:
            recommendations.append(
                f"Create {len(new_tests_needed)} new test files for adequate coverage"
            )

        if coverage_change < -0.1:
            recommendations.append(
                f"Coverage decreased by {abs(coverage_change):.1%} - add tests to maintain coverage"
            )

        if len(tests_affected) > 10:
            recommendations.append(
                "Many test files affected - verify all tests still pass and are relevant"
            )

        return recommendations

    def _is_critical_file(self, file_path: str) -> bool:
        """Check if a file is considered critical based on its path."""
        critical_patterns = [
            "main",
            "app",
            "server",
            "index",
            "init",
            "config",
            "settings",
            "auth",
            "security",
            "database",
            "model",
        ]

        file_path_lower = file_path.lower()
        return any(pattern in file_path_lower for pattern in critical_patterns)

    def generate_summary_metrics(self, impact_result: ImpactResult) -> dict[str, Any]:
        """
        Generate summary metrics for the impact analysis.

        Args:
            impact_result: The impact analysis result

        Returns:
            Dictionary containing summary metrics
        """
        return {
            "overall_impact_score": impact_result.impact_score,
            "risk_level": self.calculate_risk_level(impact_result.impact_score),
            "affected_components": {
                "files": len(impact_result.affected_files),
                "functions": len(impact_result.affected_functions),
                "classes": len(impact_result.affected_classes),
                "modules": len(impact_result.affected_modules),
            },
            "dependency_issues": {
                "circular_dependencies": len(impact_result.circular_dependencies),
                "missing_dependencies": len(impact_result.missing_dependencies),
            },
            "recommendation_count": len(impact_result.recommendations),
            "critical_issues": self._count_critical_issues(impact_result),
        }

    def _count_critical_issues(self, impact_result: ImpactResult) -> int:
        """Count the number of critical issues in the impact result."""
        critical_count = 0

        # Count critical dependency issues
        if len(impact_result.circular_dependencies) > 10:
            critical_count += 1

        if len(impact_result.missing_dependencies) > 15:
            critical_count += 1

        # Count critical scale issues
        if len(impact_result.affected_files) > 50:
            critical_count += 1

        if len(impact_result.affected_functions) > 100:
            critical_count += 1

        return critical_count
