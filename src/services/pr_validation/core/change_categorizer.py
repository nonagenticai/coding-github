"""
Change categorizer for PR validation.

This module categorizes and classifies different types of changes
to provide better context for validation and risk assessment.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from coding_client.models.pr_validation import (
    FileChange,
    FileType,
)

from services.pr_validation.core.file_classifier import PRFileTypeClassifier

logger = logging.getLogger(__name__)


class ChangeCategory(Enum):
    """Categories of changes for better classification."""

    FEATURE_ADDITION = "FEATURE_ADDITION"
    BUG_FIX = "BUG_FIX"
    REFACTORING = "REFACTORING"
    DOCUMENTATION = "DOCUMENTATION"
    CONFIGURATION = "CONFIGURATION"
    DEPENDENCY_UPDATE = "DEPENDENCY_UPDATE"
    SECURITY_FIX = "SECURITY_FIX"
    PERFORMANCE_IMPROVEMENT = "PERFORMANCE_IMPROVEMENT"
    TEST_UPDATE = "TEST_UPDATE"
    BUILD_SYSTEM = "BUILD_SYSTEM"
    CODE_CLEANUP = "CODE_CLEANUP"
    API_CHANGE = "API_CHANGE"
    DATABASE_SCHEMA = "DATABASE_SCHEMA"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    UNKNOWN = "UNKNOWN"


class ChangeScope(Enum):
    """Scope of changes."""

    FILE_LEVEL = "FILE_LEVEL"
    MODULE_LEVEL = "MODULE_LEVEL"
    PACKAGE_LEVEL = "PACKAGE_LEVEL"
    PROJECT_LEVEL = "PROJECT_LEVEL"
    ECOSYSTEM_LEVEL = "ECOSYSTEM_LEVEL"


@dataclass
class ChangeClassification:
    """Classification result for a change."""

    category: ChangeCategory
    scope: ChangeScope
    confidence: float
    indicators: list[str]
    impact_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    breaking_potential: float  # 0.0 to 1.0


@dataclass
class PRChangeProfile:
    """Overall change profile for a PR."""

    primary_category: ChangeCategory
    secondary_categories: list[ChangeCategory]
    overall_scope: ChangeScope
    change_types: dict[ChangeCategory, int]
    risk_factors: list[str]
    complexity_score: float


class ChangeCategorizer:
    """
    Categorizes changes in pull requests to understand their nature and impact.

    This categorizer helps identify:
    - What type of change is being made (feature, bug resolve, refactoring, etc.)
    - The scope of the change (file, module, project level)
    - The potential impact and risk level
    """

    def __init__(self):
        self.file_classifier = PRFileTypeClassifier()

        # Patterns for different change categories
        self.category_patterns = {
            ChangeCategory.FEATURE_ADDITION: {
                "commit_patterns": [
                    r"add",
                    r"implement",
                    r"feature",
                    r"new",
                    r"introduce",
                    r"support",
                    r"enable",
                    r"augment",
                ],
                "file_patterns": [
                    r"def\s+\w+",
                    r"function\s+\w+",
                    r"class\s+\w+",
                    r"interface\s+\w+",
                    r"module\.exports",
                ],
                "change_indicators": [
                    "new_files_added",
                    "new_functions_added",
                    "new_classes_added",
                    "new_endpoints_added",
                    "new_features_enabled",
                ],
            },
            ChangeCategory.BUG_FIX: {
                "commit_patterns": [
                    r"resolve",
                    r"bug",
                    r"issue",
                    r"problem",
                    r"error",
                    r"correct",
                    r"resolve",
                    r"patch",
                ],
                "file_patterns": [
                    r"if\s+.*error",
                    r"except\s+",
                    r"catch\s*\(",
                    r"null\s+check",
                    r"undefined\s+check",
                ],
                "change_indicators": [
                    "error_handling_improved",
                    "condition_fixed",
                    "validation_added",
                    "edge_case_handled",
                ],
            },
            ChangeCategory.REFACTORING: {
                "commit_patterns": [
                    r"refactor",
                    r"cleanup",
                    r"reorganize",
                    r"restructure",
                    r"streamline",
                    r"optimize",
                    r"update",
                ],
                "file_patterns": [
                    r"extract\s+method",
                    r"rename\s+",
                    r"move\s+",
                    r"split\s+",
                    r"merge\s+",
                ],
                "change_indicators": [
                    "code_moved",
                    "methods_extracted",
                    "variables_renamed",
                    "structure_improved",
                ],
            },
            ChangeCategory.SECURITY_FIX: {
                "commit_patterns": [
                    r"security",
                    r"vulnerability",
                    r"exploit",
                    r"cve",
                    r"auth",
                    r"permission",
                    r"sanitize",
                    r"escape",
                ],
                "file_patterns": [
                    r"validate\s*\(",
                    r"sanitize\s*\(",
                    r"escape\s*\(",
                    r"auth",
                    r"permission",
                    r"role",
                    r"token",
                ],
                "change_indicators": [
                    "input_validation_added",
                    "authentication_improved",
                    "authorization_fixed",
                    "injection_prevented",
                ],
            },
            ChangeCategory.PERFORMANCE_IMPROVEMENT: {
                "commit_patterns": [
                    r"performance",
                    r"optimize",
                    r"speed",
                    r"faster",
                    r"cache",
                    r"efficiency",
                    r"memory",
                ],
                "file_patterns": [
                    r"cache",
                    r"memoize",
                    r"index",
                    r"batch",
                    r"async",
                    r"await",
                    r"parallel",
                ],
                "change_indicators": [
                    "caching_added",
                    "algorithms_optimized",
                    "queries_optimized",
                    "async_processing_added",
                ],
            },
        }

        # File type specific categorization
        self.file_type_categories = {
            FileType.MARKDOWN: ChangeCategory.DOCUMENTATION,
            FileType.YAML: ChangeCategory.CONFIGURATION,
            FileType.JSON: ChangeCategory.CONFIGURATION,
        }

        # Scope indicators
        self.scope_indicators = {
            ChangeScope.FILE_LEVEL: [
                "single_file_change",
                "local_function_change",
                "variable_rename",
            ],
            ChangeScope.MODULE_LEVEL: [
                "module_interface_change",
                "class_restructure",
                "api_change",
            ],
            ChangeScope.PACKAGE_LEVEL: [
                "cross_module_changes",
                "package_structure_change",
                "imports_reorganized",
            ],
            ChangeScope.PROJECT_LEVEL: [
                "build_system_change",
                "dependency_changes",
                "config_overhaul",
            ],
            ChangeScope.ECOSYSTEM_LEVEL: [
                "breaking_api_changes",
                "major_version_bump",
                "architecture_change",
            ],
        }

    def categorize_file_change(self, file_change: FileChange) -> ChangeClassification:
        """
        Categorize a single file change.

        Args:
            file_change: The file change to categorize

        Returns:
            ChangeClassification with category and metadata
        """
        # Start with file type based categorization
        category = self.file_type_categories.get(file_change.file_type, ChangeCategory.UNKNOWN)
        confidence = 0.5 if category != ChangeCategory.UNKNOWN else 0.1
        indicators = []

        # Analyze content changes if available
        if file_change.patch and category == ChangeCategory.UNKNOWN:
            content_analysis = self._analyze_content_changes(file_change)
            category = content_analysis["category"]
            confidence = content_analysis["confidence"]
            indicators.extend(content_analysis["indicators"])

        # Determine scope
        scope = self._determine_change_scope(file_change)

        # Assess impact level
        impact_level = self._assess_impact_level(file_change, category, scope)

        # Calculate breaking potential
        breaking_potential = self._calculate_breaking_potential(file_change, category)

        return ChangeClassification(
            category=category,
            scope=scope,
            confidence=confidence,
            indicators=indicators,
            impact_level=impact_level,
            breaking_potential=breaking_potential,
        )

    def categorize_pr_changes(
        self,
        file_changes: list[FileChange],
        pr_metadata: dict | None = None,
    ) -> PRChangeProfile:
        """
        Categorize all changes in a PR to create an overall profile.

        Args:
            file_changes: List of file changes
            pr_metadata: Optional PR metadata (title, description, etc.)

        Returns:
            PRChangeProfile with overall categorization
        """
        # Categorize individual files
        file_classifications = []
        for file_change in file_changes:
            classification = self.categorize_file_change(file_change)
            file_classifications.append(classification)

        # Analyze PR metadata if available
        metadata_category = ChangeCategory.UNKNOWN
        if pr_metadata:
            metadata_category = self._analyze_pr_metadata(pr_metadata)

        # Aggregate categories
        category_counts: dict[ChangeCategory, int] = {}
        for classification in file_classifications:
            category = classification.category
            category_counts[category] = category_counts.get(category, 0) + 1

        # Add metadata category if it's more specific
        if metadata_category != ChangeCategory.UNKNOWN:
            category_counts[metadata_category] = category_counts.get(metadata_category, 0) + 1

        # Determine primary and secondary categories
        if category_counts:
            primary_category = max(category_counts, key=lambda x: category_counts[x])
            secondary_categories = [
                cat
                for cat, count in category_counts.items()
                if cat != primary_category and count > 0
            ]
        else:
            primary_category = ChangeCategory.UNKNOWN
            secondary_categories = []

        # Determine overall scope
        scopes = [classification.scope for classification in file_classifications]
        overall_scope = self._determine_overall_scope(scopes)

        # Identify risk factors
        risk_factors = self._identify_pr_risk_factors(file_classifications, category_counts)

        # Calculate complexity score
        complexity_score = self._calculate_pr_complexity(file_classifications, category_counts)

        return PRChangeProfile(
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            overall_scope=overall_scope,
            change_types=category_counts,
            risk_factors=risk_factors,
            complexity_score=complexity_score,
        )

    def _analyze_content_changes(self, file_change: FileChange) -> dict:
        """Analyze the content of changes to determine category."""
        patch_lines = file_change.patch.split("\n") if file_change.patch else []
        added_lines = [
            line[1:] for line in patch_lines if line.startswith("+") and not line.startswith("+++")
        ]
        removed_lines = [
            line[1:] for line in patch_lines if line.startswith("-") and not line.startswith("---")
        ]

        category_scores = {}
        indicators = []

        # Score against each category
        for category, patterns in self.category_patterns.items():
            score = 0

            # Check file patterns
            all_lines = added_lines + removed_lines
            for pattern in patterns.get("file_patterns", []):
                matches = sum(1 for line in all_lines if re.search(pattern, line, re.IGNORECASE))
                score += matches

            if score > 0:
                category_scores[category] = score

        # Additional specific checks
        indicators.extend(
            self._detect_specific_change_indicators(added_lines, removed_lines, file_change)
        )

        # Determine best category
        if category_scores:
            best_category = max(category_scores, key=lambda x: category_scores[x])
            confidence = min(category_scores[best_category] / 10.0, 1.0)
            return {
                "category": best_category,
                "confidence": confidence,
                "indicators": indicators,
            }

        return {
            "category": ChangeCategory.UNKNOWN,
            "confidence": 0.1,
            "indicators": indicators,
        }

    def _detect_specific_change_indicators(
        self,
        added_lines: list[str],
        removed_lines: list[str],
        file_change: FileChange,
    ) -> list[str]:
        """Detect specific indicators of change types."""
        indicators = []

        # New function/class additions
        for line in added_lines:
            if re.search(r"async\s+def\s+\w+\s*\(", line):
                indicators.append("async_function_added")
            elif re.search(r"def\s+\w+\s*\(", line):
                indicators.append("new_function_added")
            elif re.search(r"class\s+\w+", line):
                indicators.append("new_class_added")

        # Error handling updates
        error_handling_added = any(
            re.search(r"(try|except|catch|throw)", line, re.IGNORECASE) for line in added_lines
        )
        if error_handling_added:
            indicators.append("error_handling_improved")

        # Validation additions
        validation_added = any(
            re.search(r"(validate|check|verify)", line, re.IGNORECASE) for line in added_lines
        )
        if validation_added:
            indicators.append("validation_added")

        # Import changes
        import_changes = any(
            re.search(r"(import|require|from)", line) for line in added_lines + removed_lines
        )
        if import_changes:
            indicators.append("imports_modified")

        # Configuration changes
        if file_change.file_type in [FileType.YAML, FileType.JSON]:
            if added_lines or removed_lines:
                indicators.append("configuration_modified")

        # Test changes
        if self.file_classifier._is_test_file(file_change.filename):
            indicators.append("tests_modified")

        return indicators

    def _determine_change_scope(self, file_change: FileChange) -> ChangeScope:
        """Determine the scope of a file change."""
        filename = file_change.filename.lower()

        # Project-level files
        project_level_files = [
            "setup.py",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "dockerfile",
            "docker-compose.yml",
            "makefile",
            "cmake",
        ]

        if any(proj_file in filename for proj_file in project_level_files):
            return ChangeScope.PROJECT_LEVEL

        # Configuration that affects multiple modules
        if file_change.file_type in [FileType.YAML, FileType.JSON] and "config" in filename:
            return ChangeScope.PACKAGE_LEVEL

        # API or interface changes
        if "api" in filename or "interface" in filename:
            return ChangeScope.MODULE_LEVEL

        # Large changes might affect module level
        if file_change.total_changes > 100:
            return ChangeScope.MODULE_LEVEL

        return ChangeScope.FILE_LEVEL

    def _assess_impact_level(
        self,
        file_change: FileChange,
        category: ChangeCategory,
        scope: ChangeScope,
    ) -> str:
        """Assess the impact level of a change."""
        # Base impact from scope
        scope_impact = {
            ChangeScope.FILE_LEVEL: 1,
            ChangeScope.MODULE_LEVEL: 2,
            ChangeScope.PACKAGE_LEVEL: 3,
            ChangeScope.PROJECT_LEVEL: 4,
            ChangeScope.ECOSYSTEM_LEVEL: 5,
        }

        # Category impact modifiers
        category_impact = {
            ChangeCategory.SECURITY_FIX: 3,
            ChangeCategory.API_CHANGE: 3,
            ChangeCategory.DATABASE_SCHEMA: 3,
            ChangeCategory.DEPENDENCY_UPDATE: 2,
            ChangeCategory.CONFIGURATION: 2,
            ChangeCategory.FEATURE_ADDITION: 2,
            ChangeCategory.BUG_FIX: 1,
            ChangeCategory.REFACTORING: 1,
            ChangeCategory.DOCUMENTATION: 0,
            ChangeCategory.TEST_UPDATE: 0,
        }

        base_score = scope_impact.get(scope, 1)
        category_modifier = category_impact.get(category, 1)

        # Change size impact
        size_impact = 0
        if file_change.total_changes > 500:
            size_impact = 2
        elif file_change.total_changes > 100:
            size_impact = 1

        total_score = base_score + category_modifier + size_impact

        if total_score >= 7:
            return "CRITICAL"
        elif total_score >= 5:
            return "HIGH"
        elif total_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_breaking_potential(
        self, file_change: FileChange, category: ChangeCategory
    ) -> float:
        """Calculate the potential for breaking changes."""
        base_potential = 0.0

        # Category-based potential
        category_potential = {
            ChangeCategory.API_CHANGE: 0.8,
            ChangeCategory.SECURITY_FIX: 0.6,
            ChangeCategory.DEPENDENCY_UPDATE: 0.5,
            ChangeCategory.DATABASE_SCHEMA: 0.7,
            ChangeCategory.CONFIGURATION: 0.4,
            ChangeCategory.REFACTORING: 0.3,
            ChangeCategory.FEATURE_ADDITION: 0.2,
            ChangeCategory.BUG_FIX: 0.1,
            ChangeCategory.DOCUMENTATION: 0.0,
            ChangeCategory.TEST_UPDATE: 0.0,
        }

        base_potential = category_potential.get(category, 0.1)

        # File-specific adjustments
        filename = file_change.filename.lower()
        if any(keyword in filename for keyword in ["api", "interface", "public", "export"]):
            base_potential += 0.2

        # Change size adjustment
        if file_change.total_changes > 200:
            base_potential += 0.1

        return min(base_potential, 1.0)

    def _analyze_pr_metadata(self, metadata: dict) -> ChangeCategory:
        """Analyze PR title and description for category hints."""
        text = ""
        if metadata.get("title"):
            text += metadata["title"] + " "
        if metadata.get("description"):
            text += metadata["description"]

        text = text.lower()

        # Check against category patterns
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns.get("commit_patterns", []):
                if re.search(pattern, text):
                    score += 1
            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=lambda x: category_scores[x])

        return ChangeCategory.UNKNOWN

    def _determine_overall_scope(self, scopes: list[ChangeScope]) -> ChangeScope:
        """Determine overall scope from individual file scopes."""
        if not scopes:
            return ChangeScope.FILE_LEVEL

        # Return the highest scope
        scope_levels = {
            ChangeScope.FILE_LEVEL: 1,
            ChangeScope.MODULE_LEVEL: 2,
            ChangeScope.PACKAGE_LEVEL: 3,
            ChangeScope.PROJECT_LEVEL: 4,
            ChangeScope.ECOSYSTEM_LEVEL: 5,
        }

        max_scope = max(scopes, key=lambda s: scope_levels.get(s, 1))

        # If multiple files are at project level, it might be ecosystem level
        project_level_count = sum(1 for s in scopes if s == ChangeScope.PROJECT_LEVEL)
        if project_level_count > 2:
            return ChangeScope.ECOSYSTEM_LEVEL

        return max_scope

    def _identify_pr_risk_factors(
        self,
        classifications: list[ChangeClassification],
        category_counts: dict[ChangeCategory, int],
    ) -> list[str]:
        """Identify risk factors for the overall PR."""
        risks = []

        # High breaking potential
        avg_breaking_potential = (
            sum(c.breaking_potential for c in classifications) / len(classifications)
            if classifications
            else 0.0
        )
        if avg_breaking_potential > 0.5:
            risks.append("HIGH_BREAKING_POTENTIAL")

        # Multiple high-impact changes
        high_impact_count = sum(
            1 for c in classifications if c.impact_level in ["HIGH", "CRITICAL"]
        )
        if high_impact_count > 3:
            risks.append("MULTIPLE_HIGH_IMPACT_CHANGES")

        # Mixed change types (might indicate lack of focus)
        if len(category_counts) > 4:
            risks.append("MIXED_CHANGE_TYPES")

        # Security changes mixed with other changes
        if ChangeCategory.SECURITY_FIX in category_counts and len(category_counts) > 1:
            risks.append("SECURITY_MIXED_WITH_OTHER_CHANGES")

        # Low confidence classifications
        low_confidence_count = sum(1 for c in classifications if c.confidence < 0.5)
        if low_confidence_count > len(classifications) * 0.3:
            risks.append("UNCLEAR_CHANGE_PURPOSE")

        return risks

    def _calculate_pr_complexity(
        self,
        classifications: list[ChangeClassification],
        category_counts: dict[ChangeCategory, int],
    ) -> float:
        """Calculate overall PR complexity score."""
        if not classifications:
            return 0.0

        complexity = 0.0

        # Base complexity from number of files
        complexity += len(classifications) * 0.1

        # Complexity from change types
        complexity += len(category_counts) * 0.2

        # Complexity from impact levels
        impact_weights = {
            "LOW": 0.1,
            "MEDIUM": 0.3,
            "HIGH": 0.6,
            "CRITICAL": 1.0,
        }
        for classification in classifications:
            complexity += impact_weights.get(classification.impact_level, 0.1)

        # Complexity from breaking potential
        total_breaking_potential = sum(c.breaking_potential for c in classifications)
        complexity += total_breaking_potential * 0.5

        # Complexity from scope diversity
        scopes = set(c.scope for c in classifications)
        complexity += len(scopes) * 0.3

        return min(complexity, 10.0)
