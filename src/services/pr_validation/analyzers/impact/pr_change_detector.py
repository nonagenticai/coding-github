"""
PR change detection and parsing module.

This module handles the detection and analysis of PR changes,
including content analysis, change type classification, and pattern recognition.
"""

import logging
import re
from pathlib import Path
from typing import Any

from coding_client.models.pr_validation import (
    DocumentationImpact,
    FileChange,
    FileType,
    TestImpact,
)

from utils.file_classifier import CoreFileTypeClassifier

logger = logging.getLogger(__name__)


class PRChangeDetector:
    """
    Detects and analyzes changes in PR files.

    This detector provides functionality for:
    - Change type classification
    - Content analysis and pattern detection
    - Test coverage impact analysis
    - Documentation impact assessment
    - Performance and security impact detection
    """

    def __init__(self):
        self.file_classifier = CoreFileTypeClassifier()

    def analyze_test_coverage_impact(
        self, changes: list[FileChange], affected_files: list[str]
    ) -> TestImpact:
        """Analyze impact on test coverage."""
        tests_affected: list[str] = []
        new_tests_needed: list[str] = []
        obsolete_tests: list[str] = []
        coverage_change = 0.0

        for change in changes:
            # Check if this is a test file
            if self._is_test_file(change.filename):
                tests_affected.append(change.filename)
            else:
                # For non-test files, suggest test files that might need updates
                potential_test_file = self._suggest_test_file(change.filename)
                if potential_test_file:
                    new_tests_needed.append(potential_test_file)

        # Check affected files for test implications
        for file_path in affected_files:
            if not self._is_test_file(file_path):
                test_file = self._suggest_test_file(file_path)
                if test_file and test_file not in new_tests_needed:
                    new_tests_needed.append(test_file)

        # Calculate coverage change (streamlined)
        # In a real implementation, this would integrate with coverage tools
        if len(new_tests_needed) > 0:
            # Assume coverage decreases
            coverage_change = -0.05 * len(new_tests_needed)

        return TestImpact(
            tests_affected=tests_affected,
            coverage_change=coverage_change,
            new_tests_needed=new_tests_needed,
            obsolete_tests=obsolete_tests,
        )

    def analyze_documentation_impact(
        self, changes: list[FileChange], _affected_files: list[str]
    ) -> DocumentationImpact:
        """Analyze impact on documentation."""
        docs_affected = []
        outdated_docs = []
        new_docs_needed = []
        doc_completeness_score = 1.0

        # Check for documentation files in changes
        for change in changes:
            if self._is_documentation_file(change.filename):
                docs_affected.append(change.filename)

        # Check for code changes that might need documentation updates
        code_changes = [
            change
            for change in changes
            if change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
        ]

        if code_changes:
            # Suggest documentation that might need updates
            potential_docs = [
                "README.md",
                "CHANGELOG.md",
                "docs/api.md",
                "docs/user_guide.md",
            ]

            for doc in potential_docs:
                if doc not in docs_affected:
                    outdated_docs.append(doc)

        # Check for new API additions that need documentation
        for change in code_changes:
            if change.additions > change.deletions:  # Net addition
                new_docs_needed.append(f"Documentation for changes in {change.filename}")

        # Calculate documentation completeness
        if new_docs_needed:
            doc_completeness_score = max(0.0, 1.0 - len(new_docs_needed) * 0.1)

        return DocumentationImpact(
            docs_affected=docs_affected,
            outdated_docs=outdated_docs,
            new_docs_needed=new_docs_needed,
            doc_completeness_score=doc_completeness_score,
        )

    def analyze_performance_impact(self, changes: list[FileChange]) -> dict[str, Any]:
        """Analyze potential performance impact."""
        impact: dict[str, Any] = {
            "risk_level": "low",
            "concerns": [],
            "recommendations": [],
        }

        performance_indicators = [
            r"for\s+.*\s+in\s+.*:",  # Loops
            r"while\s+.*:",  # While loops
            r"\.map\s*\(",  # JavaScript map
            r"\.filter\s*\(",  # JavaScript filter
            r"\.reduce\s*\(",  # JavaScript reduce
            r"SELECT\s+.*\s+FROM",  # SQL queries
            r"async\s+def",  # Async functions
            r"await\s+",  # Await calls
        ]

        for change in changes:
            if change.patch:
                added_lines = [
                    line[1:]
                    for line in change.patch.split("\n")
                    if line.startswith("+") and not line.startswith("+++")
                ]

                for line in added_lines:
                    for pattern in performance_indicators:
                        if re.search(pattern, line, re.IGNORECASE):
                            impact["concerns"].append(
                                f"Performance-sensitive code added: {line.strip()}"
                            )
                            impact["risk_level"] = "medium"

        if impact["concerns"]:
            impact["recommendations"].append("Consider performance testing for added code")
            impact["recommendations"].append("Review algorithmic complexity of changes")

        return impact

    def analyze_security_impact(self, changes: list[FileChange]) -> dict[str, Any]:
        """Analyze potential security impact."""
        impact: dict[str, Any] = {
            "risk_level": "low",
            "concerns": [],
            "recommendations": [],
        }

        security_indicators = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"auth",
            r"sql",
            r"query",
            r"execute",
            r"eval",
            r"exec",
            r"input",
            r"request",
            r"param",
            r"user_input",
            r"file",
            r"path",
            r"directory",
            r"upload",
            r"permission",
            r"role",
            r"access",
            r"privilege",
        ]

        for change in changes:
            # Check filename for security-sensitive areas
            filename_lower = change.filename.lower()
            if any(indicator in filename_lower for indicator in ["auth", "security", "permission"]):
                impact["concerns"].append(f"Security-sensitive file modified: {change.filename}")
                impact["risk_level"] = "high"

            # Check content for security patterns
            if change.patch:
                for line in change.patch.split("\n"):
                    if line.startswith(("+", "-")):
                        line_content = line[1:].lower()
                        for indicator in security_indicators:
                            if indicator in line_content:
                                impact["concerns"].append(
                                    f"Security-sensitive code changed: {line.strip()}"
                                )
                                impact["risk_level"] = "medium"
                                break

        if impact["concerns"]:
            impact["recommendations"].append("Conduct security review of changes")
            impact["recommendations"].append("Consider security testing")
            impact["recommendations"].append("Review for injection vulnerabilities")

        return impact

    def detect_change_patterns(self, changes: list[FileChange]) -> dict[str, Any]:
        """
        Detect patterns in the changes that might indicate specific types of modifications.

        Args:
            changes: List of file changes to analyze

        Returns:
            Dictionary containing detected patterns and their implications
        """
        patterns = {
            "api_changes": [],
            "database_changes": [],
            "config_changes": [],
            "dependency_changes": [],
            "refactoring_indicators": [],
            "feature_additions": [],
            "bug_fixes": [],
        }

        for change in changes:
            filename = change.filename.lower()

            # Detect API changes
            if self._is_api_file(filename):
                patterns["api_changes"].append(
                    {
                        "file": change.filename,
                        "type": "API endpoint modification",
                        "impact": "high",
                    }
                )

            # Detect database changes
            if self._is_database_file(filename):
                patterns["database_changes"].append(
                    {
                        "file": change.filename,
                        "type": "Database schema or migration",
                        "impact": "high",
                    }
                )

            # Detect configuration changes
            if self._is_config_file(filename):
                patterns["config_changes"].append(
                    {
                        "file": change.filename,
                        "type": "Configuration modification",
                        "impact": "medium",
                    }
                )

            # Detect dependency changes
            if self._is_dependency_file(filename):
                patterns["dependency_changes"].append(
                    {
                        "file": change.filename,
                        "type": "Dependency modification",
                        "impact": "medium",
                    }
                )

            # Detect refactoring patterns
            if self._is_refactoring_change(change):
                patterns["refactoring_indicators"].append(
                    {
                        "file": change.filename,
                        "type": "Code refactoring",
                        "impact": "low",
                    }
                )

            # Detect feature additions
            if self._is_feature_addition(change):
                patterns["feature_additions"].append(
                    {"file": change.filename, "type": "New feature", "impact": "medium"}
                )

            # Detect bug fixes
            if self._is_bug_fix(change):
                patterns["bug_fixes"].append(
                    {"file": change.filename, "type": "Bug fix", "impact": "low"}
                )

        return patterns

    def _is_test_file(self, filename: str) -> bool:
        """Check if a file is a test file using the file classifier."""
        return self.file_classifier.get_file_type(filename) == "test"

    def _is_documentation_file(self, filename: str) -> bool:
        """Check if a file is a documentation file using the file classifier."""
        return self.file_classifier.get_file_type(filename) == "documentation"

    def _suggest_test_file(self, file_path: str) -> str | None:
        """Suggest a test file name for a given source file."""
        path = Path(file_path)

        # Python test patterns
        if path.suffix == ".py":
            # Try different test patterns
            patterns = [
                f"test_{path.stem}.py",
                f"{path.stem}_test.py",
                f"tests/test_{path.stem}.py",
                f"test/test_{path.stem}.py",
            ]
            return patterns[0]  # Return the first pattern as suggestion

        # JavaScript/TypeScript test patterns
        elif path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            patterns = [
                f"{path.stem}.test{path.suffix}",
                f"{path.stem}.spec{path.suffix}",
                f"__tests__/{path.stem}.test{path.suffix}",
            ]
            return patterns[0]

        return None

    def _is_api_file(self, filename: str) -> bool:
        """Check if a file is likely an API endpoint file."""
        api_indicators = ["api", "endpoint", "route", "controller", "handler", "view"]
        return any(indicator in filename for indicator in api_indicators)

    def _is_database_file(self, filename: str) -> bool:
        """Check if a file is related to database operations."""
        db_indicators = ["migration", "model", "schema", "database", "db", "sql"]
        return any(indicator in filename for indicator in db_indicators)

    def _is_config_file(self, filename: str) -> bool:
        """Check if a file is a configuration file."""
        config_indicators = [
            "config",
            "settings",
            "env",
            ".ini",
            ".conf",
            ".yaml",
            ".yml",
            ".toml",
        ]
        return any(indicator in filename for indicator in config_indicators)

    def _is_dependency_file(self, filename: str) -> bool:
        """Check if a file manages dependencies."""
        dependency_files = [
            "requirements.txt",
            "package.json",
            "package-lock.json",
            "pipfile",
            "pipfile.lock",
            "poetry.lock",
            "pyproject.toml",
            "cargo.toml",
            "cargo.lock",
            "go.mod",
            "go.sum",
        ]
        return filename.lower() in dependency_files

    def _is_refactoring_change(self, change: FileChange) -> bool:
        """Detect if a change appears to be refactoring."""
        # Heuristic: high deletion to addition ratio might indicate refactoring
        if change.additions > 0 and change.deletions > 0:
            ratio = change.deletions / change.additions
            # If deletions are high relative to additions, might be refactoring
            return ratio > 0.5
        return False

    def _is_feature_addition(self, change: FileChange) -> bool:
        """Detect if a change appears to be a new feature."""
        # Heuristic: high additions with few deletions might indicate new feature
        if change.additions > change.deletions * 2 and change.additions > 10:
            return True
        return False

    def _is_bug_fix(self, change: FileChange) -> bool:
        """Detect if a change appears to be a bug fix."""
        # Check for bug fix keywords in commit messages or file content
        bug_keywords = ["fix", "bug", "error", "issue", "problem", "correct"]

        if change.patch:
            patch_content = change.patch.lower()
            return any(keyword in patch_content for keyword in bug_keywords)

        return False

    def extract_change_metadata(self, changes: list[FileChange]) -> dict[str, Any]:
        """
        Extract metadata about the changes for better analysis.

        Args:
            changes: List of file changes

        Returns:
            Dictionary containing change metadata
        """
        metadata = {
            "total_files_changed": len(changes),
            "total_additions": sum(change.additions for change in changes),
            "total_deletions": sum(change.deletions for change in changes),
            "file_types": {},
            "change_distribution": {},
            "largest_changes": [],
        }

        # Analyze file types
        for change in changes:
            file_type = change.file_type.value if change.file_type else "unknown"
            metadata["file_types"][file_type] = metadata["file_types"].get(file_type, 0) + 1

        # Analyze change distribution
        for change in changes:
            change_size = change.additions + change.deletions
            if change_size < 10:
                size_category = "small"
            elif change_size < 50:
                size_category = "medium"
            elif change_size < 200:
                size_category = "large"
            else:
                size_category = "very_large"

            metadata["change_distribution"][size_category] = (
                metadata["change_distribution"].get(size_category, 0) + 1
            )

        # Find largest changes
        sorted_changes = sorted(changes, key=lambda c: c.additions + c.deletions, reverse=True)
        metadata["largest_changes"] = [
            {
                "filename": change.filename,
                "additions": change.additions,
                "deletions": change.deletions,
                "total": change.additions + change.deletions,
            }
            for change in sorted_changes[:5]
        ]

        return metadata
