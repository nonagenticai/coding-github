"""
Test coverage correlator for PR validation.

This module analyzes the relationship between code changes and test coverage,
identifying gaps and providing recommendations for test updates.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from coding_client.models.pr_validation import (
    CodebaseContext,
    FileChange,
    FileType,
)

from utils.file_classifier import CoreFileTypeClassifier

logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    """Represents a test file and its characteristics."""

    file_path: str
    test_type: str  # 'unit', 'integration', 'e2e', 'unknown'
    tested_modules: set[str] = field(default_factory=set)
    test_functions: list[str] = field(default_factory=list)
    test_classes: list[str] = field(default_factory=list)
    coverage_targets: set[str] = field(default_factory=set)
    framework: str | None = None  # 'pytest', 'unittest', 'jest', etc.


@dataclass
class CoverageGap:
    """Represents a gap in test coverage."""

    file_path: str
    symbol_name: str
    symbol_type: str  # 'function', 'class', 'method'
    line_number: int
    severity: str  # 'high', 'medium', 'low'
    reason: str
    suggested_test: str | None = None


@dataclass
class TestCorrelationResult:
    """Result of test coverage correlation analysis."""

    coverage_gaps: list[CoverageGap]
    existing_tests: list[TestFile]
    test_recommendations: list[str]
    coverage_score: float
    impact_on_coverage: float
    missing_test_files: list[str]
    orphaned_tests: list[str]  # Tests that no longer test anything


class TestCoverageCorrelator:
    """
    Correlates code changes with test coverage to identify gaps and updates.

    This correlator provides insights into:
    - Which changes are covered by tests
    - Missing test coverage for new/modified code
    - Orphaned tests that no longer test anything
    - Test recommendations for changes
    - Impact on overall test coverage
    """

    def __init__(self, codebase_context: CodebaseContext | None = None):
        self.codebase_context = codebase_context
        self.file_classifier = CoreFileTypeClassifier()

        # Test framework patterns
        self.test_frameworks = {
            "pytest": [
                r"import\s+pytest",
                r"def\s+test_\w+",
                r"@pytest\.",
                r"assert\s+",
            ],
            "unittest": [
                r"import\s+unittest",
                r"class\s+\w*Test\w*\s*\(\s*unittest\.TestCase\s*\)",
                r"def\s+test_\w+",
                r"self\.assert\w+",
            ],
            "jest": [
                r"describe\s*\(",
                r"it\s*\(",
                r"test\s*\(",
                r"expect\s*\(",
            ],
            "mocha": [
                r"describe\s*\(",
                r"it\s*\(",
                r"expect\s*\(",
                r"should\.",
            ],
        }

        # Patterns for identifying test targets
        self.import_patterns = {
            "python": [
                r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
                r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            ],
            "javascript": [
                r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
                r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            ],
        }

        # Complexity indicators that require more testing
        self.complexity_indicators = [
            r"if\s+.*:",
            r"elif\s+.*:",
            r"else\s*:",
            r"for\s+.*:",
            r"while\s+.*:",
            r"try\s*:",
            r"except\s*:",
            r"finally\s*:",
            r"with\s+.*:",
            r"async\s+de",
            r"await\s+",
        ]

    async def correlate_test_coverage(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> TestCorrelationResult:
        """
        Correlate code changes with test coverage.

        Args:
            changes: List of file changes
            codebase_context: Context about the codebase

        Returns:
            TestCorrelationResult with coverage analysis
        """
        self.codebase_context = codebase_context

        try:
            # Identify test files in changes
            test_files = self._identify_test_files(changes)

            # Identify source files that need testing
            source_files = self._identify_source_files(changes)

            # Analyze existing test coverage
            existing_tests = await self._analyze_existing_tests(test_files)

            # Find coverage gaps
            coverage_gaps = await self._find_coverage_gaps(source_files, existing_tests)

            # Generate test recommendations
            recommendations = self._generate_test_recommendations(source_files, coverage_gaps)

            # Calculate coverage scores
            coverage_score = self._calculate_coverage_score(source_files, existing_tests)
            impact_on_coverage = self._calculate_coverage_impact(changes, coverage_gaps)

            # Find missing test files
            missing_test_files = self._find_missing_test_files(source_files)

            # Find orphaned tests
            orphaned_tests = self._find_orphaned_tests(test_files, source_files)

            return TestCorrelationResult(
                coverage_gaps=coverage_gaps,
                existing_tests=existing_tests,
                test_recommendations=recommendations,
                coverage_score=coverage_score,
                impact_on_coverage=impact_on_coverage,
                missing_test_files=missing_test_files,
                orphaned_tests=orphaned_tests,
            )

        except Exception as e:
            logger.error(f"Error in test coverage correlation: {e}")
            return TestCorrelationResult(
                coverage_gaps=[],
                existing_tests=[],
                test_recommendations=[f"Error in analysis: {str(e)}"],
                coverage_score=0.0,
                impact_on_coverage=0.0,
                missing_test_files=[],
                orphaned_tests=[],
            )

    def _identify_test_files(self, changes: list[FileChange]) -> list[FileChange]:
        """Identify test files from the list of changes."""
        test_files = []

        for change in changes:
            if self.file_classifier._is_test_file(change.filename):
                test_files.append(change)

        return test_files

    def _identify_source_files(self, changes: list[FileChange]) -> list[FileChange]:
        """Identify source files (non-test) from the list of changes."""
        source_files = []

        for change in changes:
            if (
                not self.file_classifier._is_test_file(change.filename)
                and change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
                and not change.binary
            ):
                source_files.append(change)

        return source_files

    async def _analyze_existing_tests(self, test_files: list[FileChange]) -> list[TestFile]:
        """Analyze existing test files to understand what they test."""
        analyzed_tests = []

        for test_file_change in test_files:
            try:
                test_file = await self._analyze_test_file(test_file_change)
                if test_file:
                    analyzed_tests.append(test_file)
            except Exception as e:
                logger.error(f"Error analyzing test file {test_file_change.filename}: {e}")

        return analyzed_tests

    async def _analyze_test_file(self, test_file_change: FileChange) -> TestFile | None:
        """Analyze a single test file."""
        content = test_file_change.after_content or test_file_change.before_content
        if not content:
            return None

        test_file = TestFile(
            file_path=test_file_change.filename,
            test_type=self._determine_test_type(test_file_change.filename),
        )

        # Detect test framework
        test_file.framework = self._detect_test_framework(content)

        # Extract test functions and classes
        if test_file_change.file_type == FileType.PYTHON:
            await self._analyze_python_test_file(test_file, content)
        elif test_file_change.file_type in [
            FileType.JAVASCRIPT,
            FileType.TYPESCRIPT,
        ]:
            await self._analyze_javascript_test_file(test_file, content)

        # Find what modules this test file imports/tests
        test_file.tested_modules = self._find_tested_modules(content, test_file_change.file_type)

        return test_file

    def _determine_test_type(self, filename: str) -> str:
        """Determine the type of test based on filename and path."""
        filename_lower = filename.lower()

        if "unit" in filename_lower or "test_" in filename_lower:
            return "unit"
        elif "integration" in filename_lower or "e2e" in filename_lower:
            return "integration"
        elif "spec" in filename_lower:
            return "unit"
        else:
            return "unknown"

    def _detect_test_framework(self, content: str) -> str | None:
        """Detect the test framework used in the file."""
        for framework, patterns in self.test_frameworks.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return framework
        return None

    async def _analyze_python_test_file(self, test_file: TestFile, content: str) -> None:
        """Analyze Python test file using AST."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_") or "test" in node.name.lower():
                        test_file.test_functions.append(node.name)

                        # Try to infer what this test covers
                        target = self._infer_test_target_from_name(node.name)
                        if target:
                            test_file.coverage_targets.add(target)

                elif isinstance(node, ast.ClassDef):
                    if "test" in node.name.lower():
                        test_file.test_classes.append(node.name)

                        # Analyze methods in test class
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                                test_file.test_functions.append(f"{node.name}.{item.name}")
                                target = self._infer_test_target_from_name(item.name)
                                if target:
                                    test_file.coverage_targets.add(target)

        except SyntaxError:
            logger.warning(f"Could not parse Python test file: {test_file.file_path}")

    async def _analyze_javascript_test_file(self, test_file: TestFile, content: str) -> None:
        """Analyze JavaScript/TypeScript test file using regex patterns."""
        # Extract describe blocks
        describe_pattern = r'describe\s*\(\s*[\'"]([^\'"]*)[\'"]'
        describes = re.findall(describe_pattern, content)

        for describe in describes:
            test_file.test_classes.append(describe)
            test_file.coverage_targets.add(describe)

        # Extract test functions
        test_patterns = [
            r'it\s*\(\s*[\'"]([^\'"]*)[\'"]',
            r'test\s*\(\s*[\'"]([^\'"]*)[\'"]',
        ]

        for pattern in test_patterns:
            tests = re.findall(pattern, content)
            test_file.test_functions.extend(tests)

            for test in tests:
                target = self._infer_test_target_from_name(test)
                if target:
                    test_file.coverage_targets.add(target)

    def _find_tested_modules(self, content: str, file_type: FileType) -> set[str]:
        """Find what modules this test file imports and potentially tests."""
        tested_modules = set()

        if file_type == FileType.PYTHON:
            patterns = self.import_patterns["python"]
        else:
            patterns = self.import_patterns["javascript"]

        for line in content.split("\n"):
            for pattern in patterns:
                match = re.search(pattern, line.strip())
                if match:
                    module = match.group(1)
                    # Filter out test frameworks and utilities
                    if not any(
                        fw in module.lower() for fw in ["pytest", "unittest", "jest", "mocha"]
                    ):
                        tested_modules.add(module)

        return tested_modules

    def _infer_test_target_from_name(self, test_name: str) -> str | None:
        """Infer what function/class a test is targeting from its name."""
        # Remove test_ prefix and common test suffixes
        cleaned_name = test_name.replace("test_", "").replace("_test", "")

        # Handle common patterns
        patterns = [
            r"should_(.+)",
            r"when_(.+)",
            r"(.+)_success",
            r"(.+)_failure",
            r"(.+)_error",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned_name)
            if match:
                return match.group(1)

        # If no pattern matches, return the cleaned name if it looks like a function
        if cleaned_name and "_" in cleaned_name:
            return cleaned_name.split("_")[0]

        return cleaned_name if cleaned_name else None

    async def _find_coverage_gaps(
        self, source_files: list[FileChange], existing_tests: list[TestFile]
    ) -> list[CoverageGap]:
        """Find gaps in test coverage for source files."""
        coverage_gaps = []

        # Create a map of what's being tested
        tested_symbols = set()
        tested_modules = set()

        for test_file in existing_tests:
            tested_symbols.update(test_file.coverage_targets)
            tested_modules.update(test_file.tested_modules)

        # Analyze each source file for coverage gaps
        for source_file in source_files:
            gaps = await self._analyze_file_coverage_gaps(
                source_file, tested_symbols, tested_modules
            )
            coverage_gaps.extend(gaps)

        return coverage_gaps

    async def _analyze_file_coverage_gaps(
        self,
        source_file: FileChange,
        tested_symbols: set[str],
        tested_modules: set[str],
    ) -> list[CoverageGap]:
        """Analyze coverage gaps for a single source file."""
        gaps: list[CoverageGap] = []
        content = source_file.after_content or source_file.before_content

        if not content:
            return gaps

        # Check if the module itself is tested
        module_name = self._get_module_name(source_file.filename)
        if module_name not in tested_modules:
            gaps.append(
                CoverageGap(
                    file_path=source_file.filename,
                    symbol_name=module_name,
                    symbol_type="module",
                    line_number=1,
                    severity="medium",
                    reason="Module not imported by any test file",
                    suggested_test=self._suggest_test_file_name(source_file.filename),
                )
            )

        # Analyze specific functions and classes
        if source_file.file_type == FileType.PYTHON:
            gaps.extend(
                await self._analyze_python_coverage_gaps(source_file, content, tested_symbols)
            )
        elif source_file.file_type in [
            FileType.JAVASCRIPT,
            FileType.TYPESCRIPT,
        ]:
            gaps.extend(
                await self._analyze_javascript_coverage_gaps(source_file, content, tested_symbols)
            )

        return gaps

    async def _analyze_python_coverage_gaps(
        self, source_file: FileChange, content: str, tested_symbols: set[str]
    ) -> list[CoverageGap]:
        """Analyze Python file for coverage gaps."""
        gaps = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if self._is_public_function(node.name) and node.name not in tested_symbols:
                        severity = self._assess_function_test_priority(node, content)
                        gaps.append(
                            CoverageGap(
                                file_path=source_file.filename,
                                symbol_name=node.name,
                                symbol_type="function",
                                line_number=node.lineno,
                                severity=severity,
                                reason="Public function not covered by tests",
                                suggested_test=f"test_{node.name}",
                            )
                        )

                elif isinstance(node, ast.ClassDef):
                    if self._is_public_class(node.name) and node.name not in tested_symbols:
                        gaps.append(
                            CoverageGap(
                                file_path=source_file.filename,
                                symbol_name=node.name,
                                symbol_type="class",
                                line_number=node.lineno,
                                severity="high",
                                reason="Public class not covered by tests",
                                suggested_test=f"Test{node.name}",
                            )
                        )

        except SyntaxError:
            logger.warning(
                f"Could not parse Python file for coverage analysis: {source_file.filename}"
            )

        return gaps

    async def _analyze_javascript_coverage_gaps(
        self, source_file: FileChange, content: str, tested_symbols: set[str]
    ) -> list[CoverageGap]:
        """Analyze JavaScript/TypeScript file for coverage gaps."""
        gaps = []

        # Find exported functions
        export_patterns = [
            r"export\s+function\s+(\w+)",
            r"export\s+const\s+(\w+)\s*=",
            r"exports\.(\w+)\s*=",
            r"module\.exports\.(\w+)\s*=",
        ]

        line_number = 1
        for line in content.split("\n"):
            for pattern in export_patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    if func_name not in tested_symbols:
                        gaps.append(
                            CoverageGap(
                                file_path=source_file.filename,
                                symbol_name=func_name,
                                symbol_type="function",
                                line_number=line_number,
                                severity="medium",
                                reason="Exported function not covered by tests",
                                suggested_test=f"test {func_name}",
                            )
                        )
            line_number += 1

        return gaps

    def _assess_function_test_priority(self, node: ast.FunctionDef, content: str) -> str:
        """Assess the testing priority of a function based on complexity."""
        # Count complexity indicators in function
        func_lines = content.split("\n")[node.lineno - 1 : node.end_lineno]
        func_content = "\n".join(func_lines)

        complexity_count = 0
        for pattern in self.complexity_indicators:
            complexity_count += len(re.findall(pattern, func_content))

        # Check for error handling
        has_error_handling = any(
            keyword in func_content.lower() for keyword in ["try", "except", "raise", "error"]
        )

        # Check for security-sensitive code
        has_security_code = any(
            keyword in func_content.lower()
            for keyword in ["password", "auth", "token", "permission"]
        )

        if has_security_code or complexity_count > 5:
            return "high"
        elif has_error_handling or complexity_count > 2:
            return "medium"
        else:
            return "low"

    def _generate_test_recommendations(
        self, source_files: list[FileChange], coverage_gaps: list[CoverageGap]
    ) -> list[str]:
        """Generate test recommendations based on analysis."""
        recommendations = []

        # Group gaps by severity
        high_priority_gaps = [gap for gap in coverage_gaps if gap.severity == "high"]
        medium_priority_gaps = [gap for gap in coverage_gaps if gap.severity == "medium"]

        if high_priority_gaps:
            recommendations.append(
                f"High priority: Add tests for {len(high_priority_gaps)} critical functions/classes"
            )

        if medium_priority_gaps:
            recommendations.append(
                f"Medium priority: Add tests for {len(medium_priority_gaps)} functions with moderate complexity"
            )

        # Check for files with many changes but no tests
        for source_file in source_files:
            if source_file.total_changes > 50:
                test_file = self._suggest_test_file_name(source_file.filename)
                recommendations.append(
                    f"Large changes in {source_file.filename} - ensure {test_file} exists and is comprehensive"
                )

        # General recommendations
        if coverage_gaps:
            recommendations.append(
                "Consider using test coverage tools to track and update coverage"
            )
            recommendations.append("Add integration tests for components with complex interactions")

        return recommendations

    def _calculate_coverage_score(
        self, source_files: list[FileChange], existing_tests: list[TestFile]
    ) -> float:
        """Calculate an estimated coverage score."""
        if not source_files:
            return 1.0

        total_symbols = 0
        covered_symbols = 0

        # This is a streamlined calculation
        # In a real implementation, you'd integrate with actual coverage tools

        for source_file in source_files:
            # Estimate symbols in file based on content
            content = source_file.after_content or source_file.before_content or ""
            estimated_functions = len(re.findall(r"def\s+\w+|function\s+\w+", content))
            estimated_classes = len(re.findall(r"class\s+\w+", content))

            total_symbols += estimated_functions + estimated_classes

        # Count covered symbols
        for test_file in existing_tests:
            covered_symbols += len(test_file.coverage_targets)

        if total_symbols == 0:
            return 1.0

        return min(covered_symbols / total_symbols, 1.0)

    def _calculate_coverage_impact(
        self, changes: list[FileChange], coverage_gaps: list[CoverageGap]
    ) -> float:
        """Calculate the impact of changes on coverage."""
        # Negative impact for uncovered new code
        impact = 0.0

        for change in changes:
            if change.additions > 0:  # New code added
                file_gaps = [gap for gap in coverage_gaps if gap.file_path == change.filename]
                if file_gaps:
                    # Assume coverage decreases proportionally to uncovered additions
                    impact -= len(file_gaps) * 0.1

        return max(impact, -1.0)  # Cap at -1.0

    def _find_missing_test_files(self, source_files: list[FileChange]) -> list[str]:
        """Find source files that don't have corresponding test files."""
        missing_tests = []

        for source_file in source_files:
            suggested_test = self._suggest_test_file_name(source_file.filename)
            if suggested_test:
                missing_tests.append(suggested_test)

        return missing_tests

    def _find_orphaned_tests(
        self, test_files: list[FileChange], source_files: list[FileChange]
    ) -> list[str]:
        """Find test files that may no longer test anything."""
        orphaned = []

        source_file_names = {self._get_module_name(sf.filename) for sf in source_files}

        for test_file in test_files:
            # This is a streamlined check
            # In practice, you'd do more sophisticated analysis
            test_name = Path(test_file.filename).stem
            potential_target = test_name.replace("test_", "").replace("_test", "")

            if potential_target not in source_file_names:
                orphaned.append(test_file.filename)

        return orphaned

    def _get_module_name(self, file_path: str) -> str:
        """Get module name from file path."""
        path = Path(file_path)
        return path.stem

    def _suggest_test_file_name(self, source_file: str) -> str:
        """Suggest a test file name for a source file."""
        path = Path(source_file)

        if path.suffix == ".py":
            return f"test_{path.stem}.py"
        elif path.suffix in [".js", ".ts"]:
            return f"{path.stem}.test{path.suffix}"
        else:
            return f"test_{path.stem}{path.suffix}"

    def _is_public_function(self, name: str) -> bool:
        """Check if function is public (not private)."""
        return not name.startswith("_")

    def _is_public_class(self, name: str) -> bool:
        """Check if class is public (not private)."""
        return not name.startswith("_")
