"""
Maintainability assessor for PR validation.

This module assesses the impact of code changes on maintainability,
identifying patterns that could make code harder to understand, modify, or extend.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

from coding_client.models.pr_validation import (
    CodebaseContext,
    FileChange,
    FileType,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


@dataclass
class MaintainabilityMetrics:
    """Represents maintainability metrics for code."""

    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    function_length_avg: float = 0.0
    class_length_avg: float = 0.0
    nesting_depth_max: int = 0
    parameter_count_max: int = 0
    duplicate_code_blocks: int = 0
    magic_numbers: int = 0
    long_parameter_lists: int = 0
    god_classes: int = 0
    god_functions: int = 0


@dataclass
class MaintainabilityIssue:
    """Represents a maintainability issue."""

    issue_type: str
    severity: str
    file_path: str
    line_number: int
    symbol_name: str
    metric_value: float
    threshold: float
    description: str
    recommendation: str
    technical_debt_score: float


@dataclass
class CodeSmell:
    """Represents a code smell detection."""

    smell_type: str
    severity: str
    description: str
    location: tuple[int, int]
    affected_code: str
    refactoring_suggestion: str


class MaintainabilityAssessor:
    """
    Assesses code maintainability and identifies technical debt.

    This assessor identifies:
    - Code complexity issues
    - Code smells and anti-patterns
    - Documentation gaps
    - Design principle violations
    - Readability issues
    - Technical debt accumulation
    """

    def __init__(self):
        self.complexity_thresholds = {
            "cyclomatic_complexity": {"low": 10, "medium": 15, "high": 20},
            "cognitive_complexity": {"low": 15, "medium": 25, "high": 30},
            "function_length": {"low": 50, "medium": 100, "high": 150},
            "class_length": {"low": 200, "medium": 400, "high": 600},
            "parameter_count": {"low": 5, "medium": 7, "high": 10},
            "nesting_depth": {"low": 3, "medium": 4, "high": 5},
        }
        self.code_smells = self._initialize_code_smells()
        self.anti_patterns = self._initialize_anti_patterns()

    def _initialize_code_smells(self) -> dict[str, list[str]]:
        """Initialize code smell patterns."""
        return {
            "long_method": ["def\\s+\\w+.*:\\s*\\n(?:\\s*.*\\n){50,}"],
            "long_parameter_list": [
                "def\\s+\\w+\\s*\\([^)]{100,}\\)",
                "function\\s+\\w+\\s*\\([^)]{100,}\\)",
            ],
            "duplicate_code": ["(.*\\n){3,}\\1{3,}"],
            "magic_numbers": [
                "\\b(?<![\\w.])((?!0|1|2|10|100|1000)\\d{2,}|(?!0\\.0|1\\.0)\\d+\\.\\d+)(?![\\w.])\\b"
            ],
            "god_class": ["class\\s+\\w+.*:\\s*\\n(?:\\s*.*\\n){200,}"],
            "dead_code": [
                "def\\s+\\w+.*:\\s*\\n\\s*pass\\s*\\n",
                "if\\s+False\\s*:",
                "if\\s+0\\s*:",
            ],
            "complex_conditionals": [
                "if\\s+[^:]{50,}:",
                "if\\s+.*\\sand\\s+.*\\sand\\s+.*\\sand\\s+.*:",
            ],
            "nested_loops": ["for\\s+.*:\\s*\\n\\s*for\\s+.*:\\s*\\n\\s*for\\s+.*:"],
            "global_variables": ["global\\s+\\w+", "var\\s+\\w+\\s*=.*;\\s*$"],
            "inconsistent_naming": [
                "def\\s+([a-z]+[A-Z]\\w*|[A-Z]\\w*[a-z]\\w*)\\s*\\(",
                "class\\s+([a-z]\\w*)\\s*[:(]",
            ],
        }

    def _initialize_anti_patterns(self) -> dict[str, dict[str, Any]]:
        """Initialize anti-pattern definitions."""
        return {
            "singleton_abuse": {
                "pattern": "class\\s+\\w+.*:\\s*\\n.*_instance\\s*=.*\\n.*__new__",
                "severity": "medium",
                "description": "Potential singleton anti-pattern",
                "recommendation": "Consider dependency injection or other patterns",
            },
            "god_object": {
                "pattern": "class\\s+(\\w*Manager\\w*|\\w*Controller\\w*|\\w*Handler\\w*).*:\\s*\\n(?:\\s*.*\\n){100,}",
                "severity": "high",
                "description": "God object anti-pattern (Manager/Controller/Handler with many responsibilities)",
                "recommendation": "Split into smaller, focused classes following Single Responsibility Principle",
            },
            "copy_paste_code": {
                "pattern": "(def\\s+\\w+.*:\\s*\\n(?:\\s*.*\\n){5,})\\s*\\1",
                "severity": "medium",
                "description": "Potential copy-paste code duplication",
                "recommendation": "Extract common functionality into shared methods",
            },
            "boolean_parameter_abuse": {
                "pattern": "def\\s+\\w+.*\\bflag\\b.*:",
                "severity": "low",
                "description": "Boolean parameter may indicate multiple responsibilities",
                "recommendation": "Consider splitting into separate methods",
            },
            "primitive_obsession": {
                "pattern": "def\\s+\\w+.*str.*str.*str.*:",
                "severity": "low",
                "description": "Many primitive parameters may indicate missing domain objects",
                "recommendation": "Consider creating value objects for related data",
            },
        }

    async def assess_maintainability(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> list[ValidationIssue]:
        """
        Assess maintainability impact of code changes.

        Args:
            changes: List of file changes to analyze
            codebase_context: Context about the codebase

        Returns:
            List of ValidationIssue objects for maintainability concerns
        """
        issues = []
        try:
            for change in changes:
                if self._should_analyze_file(change):
                    metrics_issues = await self._analyze_maintainability_metrics(change)
                    issues.extend(metrics_issues)
                    smell_issues = await self._detect_code_smells(change)
                    issues.extend(smell_issues)
                    pattern_issues = await self._detect_anti_patterns(change)
                    issues.extend(pattern_issues)
                    doc_issues = await self._analyze_documentation_quality(change)
                    issues.extend(doc_issues)
                    naming_issues = await self._analyze_naming_conventions(change)
                    issues.extend(naming_issues)
        except Exception as e:
            logger.error(f"Error in maintainability assessment: {e}")
            issues.append(
                ValidationIssue(
                    type="maintainability_analysis_error",
                    severity="medium",
                    message=f"Maintainability analysis failed: {str(e)[:100]}",
                    file="system",
                    rule="maintainability_assessor_error",
                    suggestion="Manual code review recommended",
                )
            )
        return issues

    def _should_analyze_file(self, change: FileChange) -> bool:
        """Determine if a file should be analyzed for maintainability."""
        return (
            change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
            and (not change.binary)
            and bool(change.before_content or change.after_content)
        )

    async def _analyze_maintainability_metrics(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze maintainability metrics for the file."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""
        if not content:
            return issues
        metrics = self._calculate_maintainability_metrics(content, change.file_type)
        if (
            metrics.cyclomatic_complexity
            > self.complexity_thresholds["cyclomatic_complexity"]["medium"]
        ):
            severity = (
                "high"
                if metrics.cyclomatic_complexity
                > self.complexity_thresholds["cyclomatic_complexity"]["high"]
                else "medium"
            )
            issues.append(
                ValidationIssue(
                    type="high_cyclomatic_complexity",
                    severity=severity,
                    message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                    file=change.filename,
                    rule="complexity_metrics",
                    suggestion="Consider breaking down complex functions into smaller, focused functions",
                )
            )
        if metrics.function_length_avg > self.complexity_thresholds["function_length"]["medium"]:
            severity = (
                "high"
                if metrics.function_length_avg
                > self.complexity_thresholds["function_length"]["high"]
                else "medium"
            )
            issues.append(
                ValidationIssue(
                    type="long_functions",
                    severity=severity,
                    message=f"Average function length: {metrics.function_length_avg:.1f} lines",
                    file=change.filename,
                    rule="function_length",
                    suggestion="Consider breaking down long functions for better readability",
                )
            )
        if metrics.parameter_count_max > self.complexity_thresholds["parameter_count"]["medium"]:
            severity = (
                "high"
                if metrics.parameter_count_max
                > self.complexity_thresholds["parameter_count"]["high"]
                else "medium"
            )
            issues.append(
                ValidationIssue(
                    type="long_parameter_list",
                    severity=severity,
                    message=f"Function with {metrics.parameter_count_max} parameters",
                    file=change.filename,
                    rule="parameter_count",
                    suggestion="Consider using parameter objects or reducing parameter count",
                )
            )
        if metrics.nesting_depth_max > self.complexity_thresholds["nesting_depth"]["medium"]:
            severity = (
                "high"
                if metrics.nesting_depth_max > self.complexity_thresholds["nesting_depth"]["high"]
                else "medium"
            )
            issues.append(
                ValidationIssue(
                    type="deep_nesting",
                    severity=severity,
                    message=f"Maximum nesting depth: {metrics.nesting_depth_max}",
                    file=change.filename,
                    rule="nesting_depth",
                    suggestion="Consider extracting nested logic into separate functions",
                )
            )
        if metrics.comment_ratio < 0.1:
            issues.append(
                ValidationIssue(
                    type="insufficient_documentation",
                    severity="low",
                    message=f"Low comment ratio: {metrics.comment_ratio:.2%}",
                    file=change.filename,
                    rule="documentation_ratio",
                    suggestion="Consider adding more comments to explain complex logic",
                )
            )
        return issues

    def _calculate_maintainability_metrics(
        self, content: str, file_type: FileType
    ) -> MaintainabilityMetrics:
        """Calculate maintainability metrics for code content."""
        metrics = MaintainabilityMetrics()
        lines = content.split("\n")
        metrics.lines_of_code = len(
            [line for line in lines if line.strip() and (not line.strip().startswith("#"))]
        )
        comment_lines = len(
            [
                line
                for line in lines
                if line.strip().startswith("#") or line.strip().startswith("//")
            ]
        )
        metrics.comment_ratio = comment_lines / len(lines) if lines else 0
        for line in lines:
            metrics.magic_numbers += len(re.findall(self.code_smells["magic_numbers"][0], line))
        if file_type == FileType.PYTHON:
            try:
                tree = ast.parse(content)
                metrics = self._calculate_python_metrics(tree, metrics)
            except SyntaxError:
                metrics = self._calculate_text_based_metrics(content, metrics)
        else:
            metrics = self._calculate_text_based_metrics(content, metrics)
        return metrics

    def _calculate_python_metrics(
        self, tree: ast.AST, metrics: MaintainabilityMetrics
    ) -> MaintainabilityMetrics:
        """Calculate Python-specific metrics using AST."""
        function_lengths = []
        class_lengths = []
        parameter_counts = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "end_lineno") and node.end_lineno:
                    func_length = node.end_lineno - node.lineno
                    function_lengths.append(func_length)
                    if func_length > 100:
                        metrics.god_functions += 1
                param_count = len(node.args.args)
                if node.args.vararg:
                    param_count += 1
                if node.args.kwarg:
                    param_count += 1
                parameter_counts.append(param_count)
                if param_count > 5:
                    metrics.long_parameter_lists += 1
                complexity = self._calculate_function_complexity(node)
                metrics.cyclomatic_complexity = max(metrics.cyclomatic_complexity, complexity)
                depth = self._calculate_nesting_depth(node)
                metrics.nesting_depth_max = max(metrics.nesting_depth_max, depth)
            elif isinstance(node, ast.ClassDef):
                if hasattr(node, "end_lineno") and node.end_lineno:
                    class_length = node.end_lineno - node.lineno
                    class_lengths.append(class_length)
                    if class_length > 300:
                        metrics.god_classes += 1
        if function_lengths:
            metrics.function_length_avg = sum(function_lengths) / len(function_lengths)
        if class_lengths:
            metrics.class_length_avg = sum(class_lengths) / len(class_lengths)
        if parameter_counts:
            metrics.parameter_count_max = max(parameter_counts)
        return metrics

    def _calculate_text_based_metrics(
        self, content: str, metrics: MaintainabilityMetrics
    ) -> MaintainabilityMetrics:
        """Calculate metrics using text-based analysis."""
        lines = content.split("\n")
        function_starts: list[int] = []
        class_starts: list[int] = []
        for i, line in enumerate(lines):
            if re.match("\\s*def\\s+\\w+", line) or re.match("\\s*function\\s+\\w+", line):
                function_starts.append(i)
            elif re.match("\\s*class\\s+\\w+", line):
                class_starts.append(i)
        if function_starts:
            if len(function_starts) > 1:
                avg_length = (function_starts[-1] - function_starts[0]) / (len(function_starts) - 1)
                metrics.function_length_avg = avg_length
        for line in lines:
            func_match = re.search("def\\s+\\w+\\s*\\(([^)]*)\\)", line)
            if func_match:
                params = func_match.group(1).split(",") if func_match.group(1).strip() else []
                param_count = len([p for p in params if p.strip()])
                metrics.parameter_count_max = max(metrics.parameter_count_max, param_count)
                if param_count > 5:
                    metrics.long_parameter_lists += 1
        for line in lines:
            indent_level = (len(line) - len(line.lstrip())) // 4
            metrics.nesting_depth_max = max(metrics.nesting_depth_max, indent_level)
        return metrics

    def _calculate_function_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in an AST node."""
        max_depth = 0

        def traverse(node, current_depth) -> None:
            """Recursively traverse AST nodes to find maximum nesting depth.

            Args:
                node: Current AST node being processed
                current_depth: Current nesting depth level
            """
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            if isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.AsyncFor,
                    ast.AsyncWith,
                ),
            ):
                current_depth += 1
            for child in ast.iter_child_nodes(node):
                traverse(child, current_depth)

        traverse(node, 0)
        return max_depth

    async def _detect_code_smells(self, change: FileChange) -> list[ValidationIssue]:
        """Detect code smells in the file."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""
        if not content:
            return issues
        for smell_type, patterns in self.code_smells.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    severity = self._get_smell_severity(smell_type)
                    description = self._get_smell_description(smell_type)
                    suggestion = self._get_smell_suggestion(smell_type)
                    issues.append(
                        ValidationIssue(
                            type=f"code_smell_{smell_type}",
                            severity=severity,
                            message=f"Code smell detected: {description}",
                            file=change.filename,
                            line=line_num,
                            rule=f"code_smell_{smell_type}",
                            suggestion=suggestion,
                        )
                    )
        return issues

    def _get_smell_severity(self, smell_type: str) -> str:
        """Get severity for a code smell type."""
        severity_map = {
            "long_method": "medium",
            "long_parameter_list": "medium",
            "duplicate_code": "medium",
            "magic_numbers": "low",
            "god_class": "high",
            "dead_code": "low",
            "complex_conditionals": "medium",
            "nested_loops": "high",
            "global_variables": "medium",
            "inconsistent_naming": "low",
        }
        return severity_map.get(smell_type, "low")

    def _get_smell_description(self, smell_type: str) -> str:
        """Get description for a code smell type."""
        descriptions = {
            "long_method": "Method is too long and may be doing too much",
            "long_parameter_list": "Function has too many parameters",
            "duplicate_code": "Duplicate code blocks detected",
            "magic_numbers": "Magic numbers should be replaced with named constants",
            "god_class": "Class is too large and may have too many responsibilities",
            "dead_code": "Dead or unreachable code detected",
            "complex_conditionals": "Complex conditional statements are hard to understand",
            "nested_loops": "Deeply nested loops can be hard to understand and maintain",
            "global_variables": "Global variables can make code harder to reason about",
            "inconsistent_naming": "Inconsistent naming conventions",
        }
        return descriptions.get(smell_type, "Code smell detected")

    def _get_smell_suggestion(self, smell_type: str) -> str:
        """Get suggestion for a code smell type."""
        suggestions = {
            "long_method": "Break down into smaller, focused methods",
            "long_parameter_list": "Consider using parameter objects or reducing parameters",
            "duplicate_code": "Extract common code into shared methods",
            "magic_numbers": "Replace with named constants or configuration",
            "god_class": "Split into smaller classes following Single Responsibility Principle",
            "dead_code": "Remove unused code to update maintainability",
            "complex_conditionals": "Streamline conditions or extract into separate methods",
            "nested_loops": "Consider refactoring to reduce nesting depth",
            "global_variables": "Use dependency injection or pass as parameters",
            "inconsistent_naming": "Follow consistent naming conventions",
        }
        return suggestions.get(smell_type, "Consider refactoring")

    async def _detect_anti_patterns(self, change: FileChange) -> list[ValidationIssue]:
        """Detect anti-patterns in the code."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""
        if not content:
            return issues
        for pattern_name, pattern_info in self.anti_patterns.items():
            matches = re.finditer(pattern_info["pattern"], content, re.MULTILINE | re.DOTALL)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    ValidationIssue(
                        type=f"anti_pattern_{pattern_name}",
                        severity=pattern_info["severity"],
                        message=f"Anti-pattern detected: {pattern_info['description']}",
                        file=change.filename,
                        line=line_num,
                        rule=f"anti_pattern_{pattern_name}",
                        suggestion=pattern_info["recommendation"],
                    )
                )
        return issues

    async def _analyze_documentation_quality(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze documentation quality in the code."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""
        if not content:
            return issues
        lines = content.split("\n")
        if change.file_type == FileType.PYTHON:
            function_pattern = "^\\s*def\\s+(\\w+)"
            class_pattern = "^\\s*class\\s+(\\w+)"
            for i, line in enumerate(lines):
                func_match = re.search(function_pattern, line)
                class_match = re.search(class_pattern, line)
                if func_match or class_match:
                    symbol_name = (
                        func_match.group(1)
                        if func_match
                        else class_match.group(1)
                        if class_match
                        else "unknown"
                    )
                    symbol_type = "function" if func_match else "class"
                    has_docstring = False
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line:
                            if next_line.startswith('"""') or next_line.startswith("'''"):
                                has_docstring = True
                            break
                    if not has_docstring and (not symbol_name.startswith("_")):
                        issues.append(
                            ValidationIssue(
                                type="missing_docstring",
                                severity="low",
                                message=f"Public {symbol_type} '{symbol_name}' missing docstring",
                                file=change.filename,
                                line=i + 1,
                                rule="documentation_quality",
                                suggestion=f"Add docstring to explain the purpose and usage of {symbol_name}",
                            )
                        )
        return issues

    async def _analyze_naming_conventions(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze naming conventions in the code."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""
        if not content:
            return issues
        lines = content.split("\n")
        naming_patterns = {
            "python": {
                "function": "def\\s+(\\w+)\\s*\\(",
                "class": "class\\s+(\\w+)\\s*[:(]",
                "variable": "^\\s*(\\w+)\\s*=",
                "constant": "^([A-Z_]\\w*)\\s*=",
            },
            "javascript": {
                "function": "function\\s+(\\w+)\\s*\\(",
                "class": "class\\s+(\\w+)\\s*[{]",
                "variable": "(let|const|var)\\s+(\\w+)\\s*=",
            },
        }
        language = "python" if change.file_type == FileType.PYTHON else "javascript"
        if language not in naming_patterns:
            return issues
        patterns = naming_patterns[language]
        for line_num, line in enumerate(lines, 1):
            for symbol_type, pattern in patterns.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    symbol_name = match.group(1) if symbol_type != "variable" else match.group(2)
                    violation = self._check_naming_convention(symbol_name, symbol_type, language)
                    if violation:
                        issues.append(
                            ValidationIssue(
                                type="naming_convention_violation",
                                severity="low",
                                message=f"Naming convention violation: {symbol_type} '{symbol_name}' {violation}",
                                file=change.filename,
                                line=line_num,
                                rule="naming_conventions",
                                suggestion=f"Follow {language} naming conventions for {symbol_type}s",
                            )
                        )
        return issues

    def _check_naming_convention(self, name: str, symbol_type: str, language: str) -> str | None:
        """Check if a name follows naming conventions."""
        if language == "python":
            if symbol_type == "function":
                if not re.match("^[a-z_][a-z0-9_]*$", name):
                    return "should use snake_case"
            elif symbol_type == "class":
                if not re.match("^[A-Z][a-zA-Z0-9]*$", name):
                    return "should use PascalCase"
            elif symbol_type == "variable":
                if not re.match("^[a-z_][a-z0-9_]*$", name):
                    return "should use snake_case"
            elif symbol_type == "constant":
                if not re.match("^[A-Z_][A-Z0-9_]*$", name):
                    return "should use UPPER_SNAKE_CASE"
        elif language == "javascript":
            if symbol_type in ["function", "variable"]:
                if not re.match("^[a-z][a-zA-Z0-9]*$", name):
                    return "should use camelCase"
            elif symbol_type == "class":
                if not re.match("^[A-Z][a-zA-Z0-9]*$", name):
                    return "should use PascalCase"
        return None
