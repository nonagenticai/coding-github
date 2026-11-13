"""
Performance impact analyzer for PR validation.

This module analyzes code changes for potential performance impacts,
identifying patterns that could affect system performance and scalability.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from coding_client.models.pr_validation import (
    CodebaseContext,
    FileChange,
    FileType,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformancePattern:
    """Represents a performance pattern to detect."""

    pattern_id: str
    name: str
    pattern: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    recommendation: str
    complexity_impact: str  # 'O(1)', 'O(n)', 'O(n^2)', etc.
    file_types: list[str] = field(default_factory=list)


@dataclass
class PerformanceIssue:
    """Represents a detected performance issue."""

    issue_type: str
    severity: str
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    recommendation: str
    complexity_before: str
    complexity_after: str
    confidence: float


@dataclass
class PRValidationPerformanceMetrics:
    """Represents performance metrics for analysis."""

    cyclomatic_complexity: int = 0
    nested_loop_count: int = 0
    recursive_call_count: int = 0
    database_query_count: int = 0
    network_call_count: int = 0
    file_io_count: int = 0
    memory_allocation_patterns: list[str] = field(default_factory=list)
    algorithmic_complexity: str = "O(1)"


class PerformanceAnalyzer:
    """
    Analyzes code changes for performance impact.

    This analyzer identifies:
    - Algorithmic complexity increases
    - Inefficient loops and nested operations
    - Database query performance issues
    - Memory usage patterns
    - I/O intensive operations
    - Caching opportunities
    - Concurrency issues
    """

    def __init__(self):
        self.performance_patterns = self._initialize_performance_patterns()
        self.complexity_patterns = self._initialize_complexity_patterns()
        self.inefficient_patterns = self._initialize_inefficient_patterns()

    def _initialize_performance_patterns(self) -> list[PerformancePattern]:
        """Initialize performance pattern definitions."""
        return [
            # Loop inefficiencies
            PerformancePattern(
                pattern_id="nested_loops_1",
                name="Nested Loops - O(n^2)",
                pattern=r"for\s+.*:\s*\n.*for\s+.*:",
                severity="medium",
                description="Nested loops can cause O(n^2) complexity",
                recommendation="Consider algorithm optimization or data structure changes",
                complexity_impact="O(n^2)",
                file_types=["python", "javascript", "typescript"],
            ),
            PerformancePattern(
                pattern_id="nested_loops_2",
                name="Triple Nested Loops - O(n^3)",
                pattern=r"for\s+.*:\s*\n.*for\s+.*:\s*\n.*for\s+.*:",
                severity="high",
                description="Triple nested loops cause O(n^3) complexity",
                recommendation="Urgently optimize algorithm - consider hash maps or preprocessing",
                complexity_impact="O(n^3)",
                file_types=["python", "javascript", "typescript"],
            ),
            # Database inefficiencies
            PerformancePattern(
                pattern_id="db_query_in_loop",
                name="Database Query in Loop",
                pattern=r"for\s+.*:.*\n.*\.(query|execute|find|get)\s*\(",
                severity="high",
                description="Database query inside loop causes N+1 problem",
                recommendation="Use batch queries or eager loading",
                complexity_impact="O(n)",
                file_types=["python", "javascript", "typescript"],
            ),
            PerformancePattern(
                pattern_id="select_star",
                name="SELECT * Query",
                pattern=r"SELECT\s+\*\s+FROM",
                severity="medium",
                description="SELECT * queries fetch unnecessary data",
                recommendation="Specify only required columns",
                complexity_impact="O(1)",
                file_types=["python", "javascript", "typescript"],
            ),
            # Memory inefficiencies
            PerformancePattern(
                pattern_id="large_list_creation",
                name="Large List Creation",
                pattern=r"\[.*for\s+.*\s+in\s+.*\].*len\(",
                severity="medium",
                description="Large list comprehension may consume excessive memory",
                recommendation="Consider using generators or itertools",
                complexity_impact="O(n)",
                file_types=["python"],
            ),
            PerformancePattern(
                pattern_id="string_concatenation_loop",
                name="String Concatenation in Loop",
                pattern=r'for\s+.*:.*\n.*\+=.*[\'"]',
                severity="medium",
                description="String concatenation in loop is inefficient",
                recommendation="Use list join or string builder pattern",
                complexity_impact="O(n^2)",
                file_types=["python", "javascript", "typescript"],
            ),
            # I/O inefficiencies
            PerformancePattern(
                pattern_id="file_io_in_loop",
                name="File I/O in Loop",
                pattern=r"for\s+.*:.*\n.*(open\s*\(|\.read\s*\(|\.write\s*\()",
                severity="high",
                description="File I/O operations inside loop",
                recommendation="Batch file operations or use streaming",
                complexity_impact="O(n)",
                file_types=["python", "javascript", "typescript"],
            ),
            # Inefficient algorithms
            PerformancePattern(
                pattern_id="linear_search",
                name="Linear Search Pattern",
                pattern=r"for\s+.*\s+in\s+.*:\s*\n.*if\s+.*==",
                severity="low",
                description="Linear search pattern detected",
                recommendation="Consider using hash maps or binary search",
                complexity_impact="O(n)",
                file_types=["python", "javascript", "typescript"],
            ),
            # Synchronous blocking operations
            PerformancePattern(
                pattern_id="sync_http_request",
                name="Synchronous HTTP Request",
                pattern=r"requests\.(get|post|put|delete)\s*\(",
                severity="medium",
                description="Synchronous HTTP request can block execution",
                recommendation="Use async HTTP client for non-blocking requests",
                complexity_impact="O(1)",
                file_types=["python"],
            ),
            # Inefficient data structures
            PerformancePattern(
                pattern_id="list_membership_check",
                name="List Membership Check",
                pattern=r"if\s+.*\s+in\s+\[.*\]|if\s+.*\s+in\s+list\(",
                severity="low",
                description="List membership check is O(n)",
                recommendation="Use set for membership testing",
                complexity_impact="O(n)",
                file_types=["python"],
            ),
            # Regex inefficiencies
            PerformancePattern(
                pattern_id="regex_compilation_loop",
                name="Regex Compilation in Loop",
                pattern=r"for\s+.*:.*\n.*re\.(search|match|findall|sub)\s*\(",
                severity="medium",
                description="Regex compilation inside loop",
                recommendation="Compile regex patterns outside the loop",
                complexity_impact="O(n)",
                file_types=["python"],
            ),
        ]

    def _initialize_complexity_patterns(self) -> dict[str, list[str]]:
        """Initialize algorithmic complexity patterns."""
        return {
            "O(1)": [
                r"dict\[.*\]",
                r"set\[.*\]",
                r"\.get\s*\(",
                r"\.add\s*\(",
                r"\.pop\s*\(",
                r"\.append\s*\(",
            ],
            "O(log n)": [
                r"binary_search",
                r"bisect",
                r"heappush",
                r"heappop",
                r"\.sort\s*\(.*key=",
                r"sorted\s*\(.*key=",
            ],
            "O(n)": [
                r"for\s+.*\s+in\s+",
                r"while\s+.*:",
                r"\.map\s*\(",
                r"\.filter\s*\(",
                r"\.reduce\s*\(",
                r"list\s*\(.*\)",
                r"sum\s*\(",
                r"max\s*\(",
                r"min\s*\(",
            ],
            "O(n log n)": [
                r"\.sort\s*\(\s*\)",
                r"sorted\s*\(",
                r"merge_sort",
                r"quick_sort",
                r"heap_sort",
            ],
            "O(n^2)": [
                r"for\s+.*:\s*\n.*for\s+.*:",
                r"bubble_sort",
                r"selection_sort",
                r"insertion_sort",
            ],
            "O(n^3)": [r"for\s+.*:\s*\n.*for\s+.*:\s*\n.*for\s+.*:"],
        }

    def _initialize_inefficient_patterns(self) -> dict[str, list[str]]:
        """Initialize inefficient code patterns."""
        return {
            "memory_inefficient": [
                # List instead of generator
                r"\[.*for\s+.*\s+in\s+range\(.*\)\]",
                r"\.copy\s*\(\).*for\s+.*:",  # Copying in loop
                r"global\s+\w+.*=.*\[.*\]",  # Global large data structures
            ],
            "cpu_inefficient": [
                r"time\.sleep\s*\(.*\).*for\s+.*:",  # Sleep in loop
                r"eval\s*\(.*\).*for\s+.*:",  # Eval in loop
                r"compile\s*\(.*\).*for\s+.*:",  # Compilation in loop
            ],
            "io_inefficient": [
                r"open\s*\(.*\).*for\s+.*:",  # File opening in loop
                r"\.write\s*\(.*\).*for\s+.*:",  # Multiple writes
                r"print\s*\(.*\).*for\s+.*:",  # Print in tight loop
            ],
        }

    async def analyze_performance_impact(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> list[ValidationIssue]:
        """
        Analyze performance impact of code changes.

        Args:
            changes: List of file changes to analyze
            codebase_context: Context about the codebase

        Returns:
            List of ValidationIssue objects for performance concerns
        """
        issues = []

        try:
            for change in changes:
                if self._should_analyze_file(change):
                    # Analyze performance patterns
                    pattern_issues = await self._analyze_performance_patterns(change)
                    issues.extend(pattern_issues)

                    # Analyze complexity changes
                    complexity_issues = await self._analyze_complexity_changes(change)
                    issues.extend(complexity_issues)

                    # Analyze resource usage
                    resource_issues = await self._analyze_resource_usage(change)
                    issues.extend(resource_issues)

                    # Analyze caching opportunities
                    caching_issues = await self._analyze_caching_opportunities(change)
                    issues.extend(caching_issues)

        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            issues.append(
                ValidationIssue(
                    type="performance_analysis_error",
                    severity="medium",
                    message=f"Performance analysis failed: {str(e)[:100]}",
                    file="system",
                    rule="performance_analyzer_error",
                    suggestion="Manual performance review recommended",
                )
            )

        return issues

    def _should_analyze_file(self, change: FileChange) -> bool:
        """Determine if a file should be analyzed for performance issues."""
        return (
            change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
            and not change.binary
            and bool(change.before_content or change.after_content)
        )

    async def _analyze_performance_patterns(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze file for known performance anti-patterns."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""

        if not content:
            return issues

        file_type = (
            change.file_type.value if hasattr(change.file_type, "value") else str(change.file_type)
        )

        # Check each performance pattern
        for pattern in self.performance_patterns:
            if file_type in pattern.file_types or not pattern.file_types:
                matches = self._find_pattern_matches(content, pattern, change.filename)
                for match in matches:
                    issues.append(
                        ValidationIssue(
                            type=f"performance_pattern_{pattern.pattern_id}",
                            severity=pattern.severity,
                            message=f"{pattern.name}: {pattern.description}",
                            file=change.filename,
                            line=match["line_number"],
                            rule=pattern.pattern_id,
                            suggestion=pattern.recommendation,
                        )
                    )

        return issues

    def _find_pattern_matches(
        self, content: str, pattern: PerformancePattern, filename: str
    ) -> list[dict[str, Any]]:
        """Find matches for a performance pattern in content."""
        matches = []

        # For multi-line patterns, analyze the whole content
        if "\n" in pattern.pattern:
            match = re.search(pattern.pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                # Find approximate line number
                pattern_start = content.find(match.group())
                line_num = content[:pattern_start].count("\n") + 1
                matches.append(
                    {
                        "line_number": line_num,
                        "line_content": "Multi-line pattern match",
                        "pattern_id": pattern.pattern_id,
                    }
                )
        else:
            # Single line pattern
            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern.pattern, line, re.IGNORECASE):
                    matches.append(
                        {
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "pattern_id": pattern.pattern_id,
                        }
                    )

        return matches

    async def _analyze_complexity_changes(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze changes in algorithmic complexity."""
        issues: list[ValidationIssue] = []

        if not change.before_content or not change.after_content:
            return issues

        # Calculate complexity metrics for before and after
        before_metrics = self._calculate_complexity_metrics(change.before_content)
        after_metrics = self._calculate_complexity_metrics(change.after_content)

        # Check for complexity increases
        if after_metrics.cyclomatic_complexity > before_metrics.cyclomatic_complexity * 1.5:
            issues.append(
                ValidationIssue(
                    type="complexity_increase",
                    severity="medium",
                    message=f"Cyclomatic complexity increased from {before_metrics.cyclomatic_complexity} to {after_metrics.cyclomatic_complexity}",
                    file=change.filename,
                    rule="complexity_analysis",
                    suggestion="Consider refactoring to reduce complexity",
                )
            )

        if after_metrics.nested_loop_count > before_metrics.nested_loop_count:
            issues.append(
                ValidationIssue(
                    type="nested_loop_increase",
                    severity="high",
                    message=f"Nested loops increased from {before_metrics.nested_loop_count} to {after_metrics.nested_loop_count}",
                    file=change.filename,
                    rule="nested_loop_analysis",
                    suggestion="Review algorithm for potential optimization",
                )
            )

        # Check for algorithmic complexity degradation
        complexity_order = [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n^2)",
            "O(n^3)",
            "O(2^n)",
        ]

        if (
            before_metrics.algorithmic_complexity in complexity_order
            and after_metrics.algorithmic_complexity in complexity_order
        ):
            before_idx = complexity_order.index(before_metrics.algorithmic_complexity)
            after_idx = complexity_order.index(after_metrics.algorithmic_complexity)

            if after_idx > before_idx:
                issues.append(
                    ValidationIssue(
                        type="algorithmic_complexity_degradation",
                        severity="high",
                        message=f"Algorithmic complexity degraded from {before_metrics.algorithmic_complexity} to {after_metrics.algorithmic_complexity}",
                        file=change.filename,
                        rule="algorithmic_complexity_analysis",
                        suggestion="Consider optimizing the algorithm to maintain or update complexity",
                    )
                )

        return issues

    def _calculate_complexity_metrics(self, content: str) -> PRValidationPerformanceMetrics:
        """Calculate performance metrics for code content."""
        metrics = PRValidationPerformanceMetrics()

        # Basic metrics from text analysis
        metrics.nested_loop_count = self._count_nested_loops(content)
        metrics.database_query_count = self._count_database_queries(content)
        metrics.network_call_count = self._count_network_calls(content)
        metrics.file_io_count = self._count_file_io_operations(content)

        # Try AST analysis for more accurate metrics (Python only)
        try:
            if content.strip():
                tree = ast.parse(content)
                metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
                metrics.recursive_call_count = self._count_recursive_calls(tree)
        except SyntaxError:
            # Fall back to text-based analysis
            metrics.cyclomatic_complexity = self._estimate_cyclomatic_complexity(content)

        # Determine overall algorithmic complexity
        metrics.algorithmic_complexity = self._determine_algorithmic_complexity(content, metrics)

        return metrics

    def _count_nested_loops(self, content: str) -> int:
        """Count nested loops in content."""
        # Look for nested for/while patterns
        nested_patterns = [
            r"for\s+.*:\s*\n.*for\s+.*:",
            r"while\s+.*:\s*\n.*while\s+.*:",
            r"for\s+.*:\s*\n.*while\s+.*:",
            r"while\s+.*:\s*\n.*for\s+.*:",
        ]

        count = 0
        for pattern in nested_patterns:
            count += len(re.findall(pattern, content, re.MULTILINE | re.DOTALL))

        return count

    def _count_database_queries(self, content: str) -> int:
        """Count database query operations."""
        db_patterns = [
            r"\.query\s*\(",
            r"\.execute\s*\(",
            r"\.find\s*\(",
            r"SELECT\s+.*FROM",
            r"INSERT\s+INTO",
            r"UPDATE\s+.*SET",
            r"DELETE\s+FROM",
            r"cursor\.",
            r"session\.query",
            r"db\.",
        ]

        count = 0
        for pattern in db_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _count_network_calls(self, content: str) -> int:
        """Count network/HTTP operations."""
        network_patterns = [
            r"requests\.(get|post|put|delete)",
            r"urllib\.request",
            r"fetch\s*\(",
            r"axios\.(get|post|put|delete)",
            r"http\.request",
            r"socket\.",
        ]

        count = 0
        for pattern in network_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _count_file_io_operations(self, content: str) -> int:
        """Count file I/O operations."""
        io_patterns = [
            r"open\s*\(",
            r"\.read\s*\(",
            r"\.write\s*\(",
            r"\.readline\s*\(",
            r"\.readlines\s*\(",
            r"fs\.readFile",
            r"fs\.writeFile",
        ]

        count = 0
        for pattern in io_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity from AST."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _estimate_cyclomatic_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity from text."""
        complexity = 1

        complexity_patterns = [
            r"\bif\b",
            r"\belif\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\btry\b",
            r"\bexcept\b",
            r"\band\b",
            r"\bor\b",
        ]

        for pattern in complexity_patterns:
            complexity += len(re.findall(pattern, content, re.IGNORECASE))

        return complexity

    def _count_recursive_calls(self, tree: ast.AST) -> int:
        """Count recursive function calls."""
        recursive_calls = 0

        # Extract function names
        function_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.add(node.name)

        # Look for calls to these functions within their own definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id == node.name:
                            recursive_calls += 1

        return recursive_calls

    def _determine_algorithmic_complexity(
        self, content: str, metrics: PRValidationPerformanceMetrics
    ) -> str:
        """Determine overall algorithmic complexity."""
        # Basic heuristics based on patterns
        if metrics.nested_loop_count >= 3:
            return "O(n^3)"
        elif metrics.nested_loop_count >= 2:
            return "O(n^2)"
        elif metrics.nested_loop_count >= 1 or re.search(r"for\s+.*\s+in\s+", content):
            return "O(n)"
        elif re.search(r"\.sort\s*\(\s*\)|sorted\s*\(", content):
            return "O(n log n)"
        elif re.search(r"binary_search|bisect", content):
            return "O(log n)"
        else:
            return "O(1)"

    async def _analyze_resource_usage(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze resource usage patterns."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""

        if not content:
            return issues

        # Check for memory-intensive patterns
        memory_patterns = [
            (
                r"\[.*for\s+.*\s+in\s+range\(\d{4,}\)\]",
                "Large list comprehension",
                "medium",
            ),
            (r"\.read\s*\(\s*\)", "Reading entire file into memory", "low"),
            (r"global\s+\w+.*=.*\[.*\]", "Global list variable", "low"),
        ]

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, description, severity in memory_patterns:
                if re.search(pattern, line):
                    issues.append(
                        ValidationIssue(
                            type="memory_usage_concern",
                            severity=severity,
                            message=f"Memory usage concern: {description}",
                            file=change.filename,
                            line=line_num,
                            rule="memory_analysis",
                            suggestion=f"Consider memory efficiency: {line.strip()}",
                        )
                    )

        return issues

    async def _analyze_caching_opportunities(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze opportunities for caching."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""

        if not content:
            return issues

        # Look for repeated expensive operations
        caching_opportunities = [
            (
                r"def\s+(\w+).*:.*\n.*expensive_operation.*\n.*return",
                "Expensive function without caching",
            ),
            (
                r"requests\.(get|post)\s*\([^)]*\)",
                "HTTP request without caching",
            ),
            (r"\.query\s*\([^)]*\).*for\s+.*:", "Database query in loop"),
        ]

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, description in caching_opportunities:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(
                        ValidationIssue(
                            type="caching_opportunity",
                            severity="low",
                            message=f"Caching opportunity: {description}",
                            file=change.filename,
                            line=line_num,
                            rule="caching_analysis",
                            suggestion="Consider adding caching for repeated expensive operations",
                        )
                    )

        return issues
