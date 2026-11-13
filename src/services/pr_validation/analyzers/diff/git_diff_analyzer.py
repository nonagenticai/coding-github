"""
Git diff analyzer for PR validation.

This module provides comprehensive diff analysis to understand change patterns,
detect problematic modifications, and assess the impact of changes.
"""

import difflib
import logging
import re

from coding_client.models.pr_validation import (
    ChangeSeverity,
    DiffAnalysisResult,
    FileChange,
    FileChangeAnalysis,
    FileType,
    PRDiff,
)

from utils.file_classifier import CoreFileTypeClassifier

logger = logging.getLogger(__name__)


class GitDiffAnalyzer:
    """
    Analyzes git diffs to understand change patterns and impact.

    This analyzer focuses on detecting:
    - Complete file rewrites vs incremental changes
    - Logic inversions and dangerous changes
    - High-impact file modifications
    - Change patterns that indicate risk
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.rewrite_threshold = self.config.get("rewrite_threshold", 0.7)
        self.major_change_threshold = self.config.get("major_change_threshold", 0.4)
        self.file_classifier = CoreFileTypeClassifier()

        # High-impact files that require careful review
        self.high_impact_files = {
            "readme.md",
            "readme.rst",
            "readme.txt",
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "requirements.txt",
            "requirements-dev.txt",
            "poetry.lock",
            "dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            ".github/workflows/",
            "Jenkinsfile",
            ".gitlab-ci.yml",
            "makefile",
            "cmake",
            "build.gradle",
            "pom.xml",
        }

        # Security-sensitive file patterns
        self.security_sensitive_patterns = [
            r"auth",
            r"security",
            r"password",
            r"token",
            r"secret",
            r"key",
            r"credential",
            r"cert",
            r"ssl",
            r"tls",
            r"oauth",
            r"jwt",
            r"encryption",
            r"crypto",
            r"hash",
            r"salt",
        ]

    async def analyze_diff(self, pr_diff: PRDiff) -> DiffAnalysisResult:
        """
        Analyze the complete PR diff to understand change patterns.

        Args:
            pr_diff: The PR diff to analyze

        Returns:
            DiffAnalysisResult with comprehensive analysis
        """
        logger.info(f"Analyzing diff with {len(pr_diff.files_changed)} files")

        file_analyses = []
        overall_patterns = []
        risk_factors = []

        # Analyze each file change
        for file_change in pr_diff.files_changed:
            try:
                analysis = await self._analyze_file_change(file_change)
                file_analyses.append(analysis)

                # Collect patterns and risks
                overall_patterns.extend(analysis.change_patterns)
                risk_factors.extend(analysis.risk_indicators)

            except Exception as e:
                logger.error(f"Error analyzing file {file_change.filename}: {e}")
                # Create a minimal analysis for failed files
                file_analyses.append(
                    FileChangeAnalysis(
                        filename=file_change.filename,
                        change_severity=ChangeSeverity.UNKNOWN,
                        change_patterns=["ANALYSIS_FAILED"],
                        risk_indicators=[f"ANALYSIS_ERROR: {str(e)[:100]}"],
                        line_change_ratio=0.0,
                        content_similarity=0.0,
                        file_type=file_change.file_type,
                    )
                )

        # Analyze overall PR patterns
        pr_patterns = self._analyze_pr_patterns(file_analyses)
        overall_patterns.extend(pr_patterns)

        # Calculate change complexity
        change_complexity = self._calculate_change_complexity(file_analyses)

        # Determine if careful review is required
        requires_careful_review = self._requires_careful_review(file_analyses, overall_patterns)

        return DiffAnalysisResult(
            file_analyses=file_analyses,
            overall_patterns=list(set(overall_patterns)),  # Deduplicate
            risk_factors=list(set(risk_factors)),  # Deduplicate
            change_complexity=change_complexity,
            requires_careful_review=requires_careful_review,
        )

    async def _analyze_file_change(self, file_change: FileChange) -> FileChangeAnalysis:
        """
        Analyze a single file change for patterns and risks.

        Args:
            file_change: The file change to analyze

        Returns:
            FileChangeAnalysis with detailed analysis
        """
        # Calculate basic metrics
        before_lines = (
            len(file_change.before_content.split("\n")) if file_change.before_content else 0
        )
        after_lines = len(file_change.after_content.split("\n")) if file_change.after_content else 0
        total_changes = file_change.additions + file_change.deletions
        max_lines = max(before_lines, after_lines, 1)  # Avoid division by zero

        line_change_ratio = total_changes / max_lines
        content_similarity = self._calculate_content_similarity(
            file_change.before_content, file_change.after_content
        )

        # Determine change severity
        change_severity = self._determine_change_severity(
            line_change_ratio, content_similarity, file_change
        )

        # Detect change patterns
        change_patterns = self._detect_change_patterns(file_change)

        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(
            file_change, change_severity, change_patterns
        )

        return FileChangeAnalysis(
            filename=file_change.filename,
            change_severity=change_severity,
            change_patterns=change_patterns,
            risk_indicators=risk_indicators,
            line_change_ratio=line_change_ratio,
            content_similarity=content_similarity,
            file_type=file_change.file_type,
        )

    def _calculate_content_similarity(
        self, before_content: str | None, after_content: str | None
    ) -> float:
        """
        Calculate similarity between before and after content.

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not before_content and not after_content:
            return 1.0
        if not before_content or not after_content:
            return 0.0

        # Use difflib's sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, before_content, after_content)
        return matcher.ratio()

    def _determine_change_severity(
        self,
        line_change_ratio: float,
        content_similarity: float,
        file_change: FileChange,
    ) -> ChangeSeverity:
        """
        Determine the severity of changes to a file.

        Args:
            line_change_ratio: Ratio of changed lines to total lines
            content_similarity: Content similarity score
            file_change: The file change object

        Returns:
            ChangeSeverity enum value
        """
        # Validate inputs
        line_change_ratio = max(0.0, min(10.0, line_change_ratio or 0.0))
        content_similarity = max(0.0, min(1.0, content_similarity or 0.0))

        deletions = getattr(file_change, "deletions", 0) or 0
        additions = getattr(file_change, "additions", 0) or 0

        # Complete rewrite detection
        if (
            line_change_ratio > self.rewrite_threshold
            or content_similarity < 0.3
            or (deletions > 50 and additions > 0 and deletions > additions * 2)
        ):
            return ChangeSeverity.COMPLETE_REWRITE

        # Major change detection
        elif line_change_ratio > self.major_change_threshold or content_similarity < 0.6:
            return ChangeSeverity.MAJOR

        # Moderate change
        elif line_change_ratio > 0.1 or file_change.additions + file_change.deletions > 20:
            return ChangeSeverity.MODERATE

        # Minor change
        else:
            return ChangeSeverity.MINOR

    def _detect_change_patterns(self, file_change: FileChange) -> list[str]:
        """
        Detect specific patterns in file changes.

        Args:
            file_change: The file change to analyze

        Returns:
            List of detected change patterns
        """
        patterns: list[str] = []

        if not file_change.patch:
            return patterns

        try:
            patch_lines = file_change.patch.split("\n")
            added_lines = [
                line[1:]
                for line in patch_lines
                if line.startswith("+") and not line.startswith("+++")
            ]
            removed_lines = [
                line[1:]
                for line in patch_lines
                if line.startswith("-") and not line.startswith("---")
            ]

            # Pattern detection
            if self._has_import_changes(added_lines, removed_lines):
                patterns.append("IMPORT_CHANGES")

            if self._has_function_signature_changes(patch_lines):
                patterns.append("FUNCTION_SIGNATURE_CHANGES")

            if self._has_logic_inversions(added_lines, removed_lines):
                patterns.append("LOGIC_INVERSIONS")

            if self._has_error_handling_changes(added_lines, removed_lines):
                patterns.append("ERROR_HANDLING_CHANGES")

            if self._has_security_sensitive_changes(added_lines, removed_lines):
                patterns.append("SECURITY_SENSITIVE_CHANGES")

            if self._has_configuration_changes(added_lines, removed_lines, file_change.filename):
                patterns.append("CONFIGURATION_CHANGES")

            if self._has_test_changes(added_lines, removed_lines, file_change.filename):
                patterns.append("TEST_CHANGES")

            if self._has_documentation_changes(added_lines, removed_lines, file_change.filename):
                patterns.append("DOCUMENTATION_CHANGES")

        except Exception as e:
            logger.error(f"Error detecting patterns in {file_change.filename}: {e}")
            patterns.append("PATTERN_DETECTION_ERROR")

        return patterns

    def _has_import_changes(self, added_lines: list[str], removed_lines: list[str]) -> bool:
        """Detect import/require statement changes."""
        import_patterns = [
            r"^\s*(import|from)\s+\w+",  # Python imports
            r'^\s*require\s*\([\'"]',  # Node.js requires
            r'^\s*import\s+.*from\s+[\'"]',  # ES6 imports
        ]

        all_lines = added_lines + removed_lines
        for line in all_lines:
            for pattern in import_patterns:
                if re.search(pattern, line):
                    return True
        return False

    def _has_function_signature_changes(self, patch_lines: list[str]) -> bool:
        """Detect function signature modifications."""
        function_patterns = [
            r"[-+]\s*def\s+\w+\s*\(",  # Python functions
            r"[-+]\s*function\s+\w+\s*\(",  # JavaScript functions
            r"[-+]\s*\w+\s*\([^)]*\)\s*{",  # C-style functions
        ]

        for line in patch_lines:
            for pattern in function_patterns:
                if re.search(pattern, line):
                    return True
        return False

    def _has_logic_inversions(self, added_lines: list[str], removed_lines: list[str]) -> bool:
        """
        Detect potentially dangerous logic inversions.

        This is critical for security - changes like 'and' to 'or' in authentication
        logic can introduce serious vulnerabilities.
        """
        # Look for condition flips
        inversion_patterns = [
            (r"<", r">"),
            (r">", r"<"),
            (r"<=", r">="),
            (r">=", r"<="),
            (r"==", r"!="),
            (r"!=", r"=="),
            (r"\band\b", r"\bor\b"),
            (r"\bor\b", r"\band\b"),
            (r"True", r"False"),
            (r"False", r"True"),
            (r"&&", r"\|\|"),
            (r"\|\|", r"&&"),
        ]

        for removed_line in removed_lines:
            for added_line in added_lines:
                for old_pattern, new_pattern in inversion_patterns:
                    if (
                        re.search(old_pattern, removed_line)
                        and re.search(new_pattern, added_line)
                        and self._lines_are_similar(removed_line, added_line, 0.7)
                    ):
                        return True

        return False

    def _has_error_handling_changes(self, added_lines: list[str], removed_lines: list[str]) -> bool:
        """Detect changes to error handling logic."""
        error_patterns = [
            r"try:",
            r"except:",
            r"finally:",
            r"raise\s+",  # Python
            r"try\s*{",
            r"catch\s*\(",
            r"throw\s+",  # JavaScript
            r"if\s+.*error",
            r"if\s+.*fail",
            r"if\s+.*exception",
        ]

        all_lines = added_lines + removed_lines
        for line in all_lines:
            for pattern in error_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return True
        return False

    def _has_security_sensitive_changes(
        self, added_lines: list[str], removed_lines: list[str]
    ) -> bool:
        """Detect security-sensitive changes."""
        security_patterns = [
            r"password",
            r"token",
            r"secret",
            r"key",
            r"auth",
            r"permission",
            r"privilege",
            r"access",
            r"role",
            r"encrypt",
            r"decrypt",
            r"hash",
            r"salt",
            r"crypto",
            r"validate",
            r"sanitize",
            r"escape",
            r"sql",
            r"query",
        ]

        all_lines = added_lines + removed_lines
        for line in all_lines:
            for pattern in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return True
        return False

    def _has_configuration_changes(
        self, added_lines: list[str], removed_lines: list[str], filename: str
    ) -> bool:
        """Detect configuration changes."""
        if not self.file_classifier._is_config_file(filename):
            return False

        config_patterns = [
            r"^\s*\w+\s*[:=]",  # Key-value pairs
            r"^\s*-\s+\w+",  # YAML list items
            r"^\s*\[.*\]",  # Section headers
        ]

        all_lines = added_lines + removed_lines
        for line in all_lines:
            for pattern in config_patterns:
                if re.search(pattern, line):
                    return True
        return False

    def _has_test_changes(
        self, added_lines: list[str], removed_lines: list[str], filename: str
    ) -> bool:
        """Detect test-related changes."""
        if not self.file_classifier._is_test_file(filename):
            return False

        test_patterns = [
            r"def\s+test_",
            r"function\s+test",
            r"it\s*\(",
            r"describe\s*\(",
            r"assert",
            r"expect\s*\(",
            r"@pytest",
            r"@test",
            r"setUp",
            r"tearDown",
        ]

        all_lines = added_lines + removed_lines
        for line in all_lines:
            for pattern in test_patterns:
                if re.search(pattern, line):
                    return True
        return False

    def _has_documentation_changes(
        self, added_lines: list[str], removed_lines: list[str], filename: str
    ) -> bool:
        """Detect documentation changes."""
        if not self.file_classifier._is_documentation_file(filename):
            return False

        # For docs, any significant change counts
        return len(added_lines) > 0 or len(removed_lines) > 0

    def _lines_are_similar(self, line1: str, line2: str, threshold: float = 0.7) -> bool:
        """Check if two lines are similar enough to be considered related."""
        if not line1.strip() or not line2.strip():
            return False

        matcher = difflib.SequenceMatcher(None, line1.strip(), line2.strip())
        return matcher.ratio() >= threshold

    def _identify_risk_indicators(
        self,
        file_change: FileChange,
        change_severity: ChangeSeverity,
        change_patterns: list[str],
    ) -> list[str]:
        """
        Identify specific risk indicators for the file change.

        Args:
            file_change: The file change
            change_severity: Calculated change severity
            change_patterns: Detected change patterns

        Returns:
            List of risk indicator strings
        """
        risks = []

        # High-impact file changes
        filename_lower = file_change.filename.lower()
        if any(high_impact_file in filename_lower for high_impact_file in self.high_impact_files):
            if change_severity in [
                ChangeSeverity.MAJOR,
                ChangeSeverity.COMPLETE_REWRITE,
            ]:
                risks.append("HIGH_IMPACT_FILE_MAJOR_CHANGE")

        # Security-sensitive files
        if any(pattern in filename_lower for pattern in self.security_sensitive_patterns):
            risks.append("SECURITY_SENSITIVE_FILE_MODIFIED")

        # Configuration files
        if file_change.filename.endswith((".yaml", ".yml", ".json", ".con", ".ini", ".toml")):
            if change_severity in [
                ChangeSeverity.MAJOR,
                ChangeSeverity.COMPLETE_REWRITE,
            ]:
                risks.append("CONFIG_FILE_MAJOR_CHANGE")

        # Pattern-based risks
        if "LOGIC_INVERSIONS" in change_patterns:
            risks.append("DANGEROUS_LOGIC_CHANGES")

        if "FUNCTION_SIGNATURE_CHANGES" in change_patterns:
            risks.append("POTENTIAL_BREAKING_CHANGES")

        if "SECURITY_SENSITIVE_CHANGES" in change_patterns:
            risks.append("SECURITY_CHANGES_DETECTED")

        if "ERROR_HANDLING_CHANGES" in change_patterns:
            risks.append("ERROR_HANDLING_MODIFIED")

        # Large change risks
        if file_change.total_changes > 500:
            risks.append("VERY_LARGE_CHANGE")

        # Binary file changes
        if file_change.binary:
            risks.append("BINARY_FILE_CHANGE")

        # Complete rewrites of critical files
        if change_severity == ChangeSeverity.COMPLETE_REWRITE and filename_lower in [
            "readme.md",
            "setup.py",
            "package.json",
        ]:
            risks.append("CRITICAL_FILE_COMPLETE_REWRITE")

        return risks

    def _analyze_pr_patterns(self, file_analyses: list[FileChangeAnalysis]) -> list[str]:
        """
        Analyze patterns across the entire PR.

        Args:
            file_analyses: List of individual file analyses

        Returns:
            List of PR-level patterns
        """
        patterns = []

        # Count different types of changes
        rewrite_count = sum(
            1
            for analysis in file_analyses
            if analysis.change_severity == ChangeSeverity.COMPLETE_REWRITE
        )
        major_count = sum(
            1 for analysis in file_analyses if analysis.change_severity == ChangeSeverity.MAJOR
        )

        # PR-level pattern detection
        if rewrite_count > 0:
            patterns.append("CONTAINS_COMPLETE_REWRITES")
            if rewrite_count > 3:
                patterns.append("MULTIPLE_COMPLETE_REWRITES")

        if major_count > 5:
            patterns.append("MANY_MAJOR_CHANGES")

        # Check for coordinated changes
        security_changes = sum(
            1
            for analysis in file_analyses
            if "SECURITY_SENSITIVE_CHANGES" in analysis.change_patterns
        )
        if security_changes > 1:
            patterns.append("COORDINATED_SECURITY_CHANGES")

        config_changes = sum(
            1 for analysis in file_analyses if "CONFIGURATION_CHANGES" in analysis.change_patterns
        )
        if config_changes > 2:
            patterns.append("MULTIPLE_CONFIG_CHANGES")

        # Check for missing test changes
        code_changes = sum(
            1
            for analysis in file_analyses
            if analysis.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
        )
        test_changes = sum(
            1 for analysis in file_analyses if "TEST_CHANGES" in analysis.change_patterns
        )

        if code_changes > 0 and test_changes == 0:
            patterns.append("CODE_CHANGES_WITHOUT_TESTS")

        return patterns

    def _calculate_change_complexity(self, file_analyses: list[FileChangeAnalysis]) -> float:
        """
        Calculate overall change complexity score.

        Args:
            file_analyses: List of file analyses

        Returns:
            Complexity score between 0.0 and 10.0
        """
        if not file_analyses:
            return 0.0

        complexity_score = 0.0

        # Base complexity from number of files
        complexity_score += len(file_analyses) * 0.1

        # Add complexity based on change severity
        severity_weights = {
            ChangeSeverity.MINOR: 0.1,
            ChangeSeverity.MODERATE: 0.3,
            ChangeSeverity.MAJOR: 0.7,
            ChangeSeverity.COMPLETE_REWRITE: 1.5,
            ChangeSeverity.SECURITY_ISSUE: 2.0,
            ChangeSeverity.UNKNOWN: 0.5,
        }

        for analysis in file_analyses:
            complexity_score += severity_weights.get(analysis.change_severity, 0.5)

            # Add complexity for risk indicators
            complexity_score += len(analysis.risk_indicators) * 0.2

            # Add complexity for change patterns
            complexity_score += len(analysis.change_patterns) * 0.1

        # Cap at maximum complexity
        return min(complexity_score, 10.0)

    def _requires_careful_review(
        self,
        file_analyses: list[FileChangeAnalysis],
        overall_patterns: list[str],
    ) -> bool:
        """
        Determine if the PR requires careful review.

        Args:
            file_analyses: List of file analyses
            overall_patterns: Overall PR patterns

        Returns:
            True if careful review is required
        """
        # High-risk patterns that always require review
        high_risk_patterns = [
            "CONTAINS_COMPLETE_REWRITES",
            "DANGEROUS_LOGIC_CHANGES",
            "COORDINATED_SECURITY_CHANGES",
            "MULTIPLE_COMPLETE_REWRITES",
        ]

        if any(pattern in overall_patterns for pattern in high_risk_patterns):
            return True

        # High-risk files
        high_risk_files = sum(1 for analysis in file_analyses if analysis.is_high_risk)
        if high_risk_files > 0:
            return True

        # Large number of major changes
        major_changes = sum(
            1
            for analysis in file_analyses
            if analysis.change_severity in [ChangeSeverity.MAJOR, ChangeSeverity.COMPLETE_REWRITE]
        )
        if major_changes > 3:
            return True

        # Complex changes
        total_complexity = sum(
            len(analysis.risk_indicators) + len(analysis.change_patterns)
            for analysis in file_analyses
        )
        if total_complexity > 10:
            return True

        return False
