"""
Logic safety validator for detecting dangerous code changes.

This module validates changes for logic safety, focusing on detecting
potentially dangerous modifications that could introduce bugs or security issues.
"""

import ast
import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any

from coding_client.models.pr_validation import (
    FileChange,
    FileType,
    ValidationIssue,
)

from services.pr_validation.analyzers.semantic.ast_diff_analyzer import (
    SafeASTParser,
)

logger = logging.getLogger(__name__)


@dataclass
class LogicPattern:
    """Represents a logic pattern that could be dangerous when changed."""

    pattern_type: str
    before_pattern: str
    after_pattern: str
    context: str
    risk_level: str
    description: str


@dataclass
class SafetyViolation:
    """Represents a safety violation found in code changes."""

    violation_type: str
    severity: str
    line_number: int
    before_code: str
    after_code: str
    explanation: str
    recommendation: str
    confidence: float


class LogicSafetyValidator:
    """
    Validates logic changes for safety issues.

    This validator focuses on detecting:
    - Logic inversions (AND ↔ OR, < ↔ >, == ↔ !=)
    - Condition negations
    - Error handling changes
    - Security control modifications
    - Flow control changes
    """

    def __init__(self):
        self.parser = SafeASTParser()

        # Dangerous logic inversion patterns
        self.inversion_patterns = [
            # Boolean operators
            (r"\band\b", r"\bor\b", "Boolean AND changed to OR"),
            (r"\bor\b", r"\band\b", "Boolean OR changed to AND"),
            (r"&&", r"\|\|", "AND operator changed to OR"),
            (r"\|\|", r"&&", "OR operator changed to AND"),
            # Comparison operators
            (r"==", r"!=", "Equality changed to inequality"),
            (r"!=", r"==", "Inequality changed to equality"),
            (r"<", r">", "Less-than changed to greater-than"),
            (r">", r"<", "Greater-than changed to less-than"),
            (
                r"<=",
                r">=",
                "Less-than-or-equal changed to greater-than-or-equal",
            ),
            (
                r">=",
                r"<=",
                "Greater-than-or-equal changed to less-than-or-equal",
            ),
            # Boolean literals
            (r"\bTrue\b", r"\bFalse\b", "True changed to False"),
            (r"\bFalse\b", r"\bTrue\b", "False changed to True"),
            (r"\btrue\b", r"\bfalse\b", "true changed to false (JS)"),
            (r"\bfalse\b", r"\btrue\b", "false changed to true (JS)"),
            # Negation patterns
            (r"\bnot\s+", r"", "Negation operator removed"),
            (r"", r"\bnot\s+", "Negation operator added"),
            (r"!", r"", "Negation operator removed (JS)"),
            (r"", r"!", "Negation operator added (JS)"),
        ]

        # Security-sensitive function patterns
        self.security_patterns = [
            r"authenticate",
            r"authorize",
            r"permission",
            r"access",
            r"validate",
            r"sanitize",
            r"escape",
            r"filter",
            r"encrypt",
            r"decrypt",
            r"hash",
            r"verify",
            r"login",
            r"logout",
            r"session",
            r"token",
            r"password",
            r"secret",
            r"key",
            r"credential",
        ]

        # Error handling patterns
        self.error_patterns = [
            r"try\s*:",
            r"except\s*:",
            r"finally\s*:",
            r"raise\s+",
            r"try\s*\{",
            r"catch\s*\(",
            r"throw\s+",
            r"finally\s*\{",
            r"if\s+.*error",
            r"if\s+.*fail",
            r"if\s+.*exception",
        ]

        # Flow control patterns
        self.flow_patterns = [
            r"return\s+",
            r"break\s*;?",
            r"continue\s*;?",
            r"yield\s+",
            r"await\s+",
            r"async\s+de",
            r"if\s+.*:",
            r"elif\s+.*:",
            r"else\s*:",
            r"while\s+.*:",
            r"for\s+.*:",
        ]

    async def validate_logic_safety(self, file_change: FileChange) -> list[ValidationIssue]:
        """
        Validate a file change for logic safety issues.

        Args:
            file_change: The file change to validate

        Returns:
            List of ValidationIssue objects
        """
        if not self._is_validatable_file(file_change):
            return []

        issues = []

        try:
            # Analyze patch for dangerous patterns
            if file_change.patch:
                patch_issues = self._analyze_patch_safety(file_change.patch, file_change.filename)
                issues.extend(patch_issues)

            # Analyze AST changes for code-level issues
            if file_change.file_type == FileType.PYTHON:
                ast_issues = await self._analyze_ast_safety(file_change)
                issues.extend(ast_issues)

            # Analyze security-sensitive changes
            security_issues = self._analyze_security_changes(file_change)
            issues.extend(security_issues)

            # Analyze error handling changes
            error_issues = self._analyze_error_handling_changes(file_change)
            issues.extend(error_issues)

        except Exception as e:
            logger.error(f"Error in logic safety validation for {file_change.filename}: {e}")
            issues.append(
                ValidationIssue(
                    type="validation_error",
                    severity="medium",
                    message=f"Logic safety validation failed: {str(e)[:100]}",
                    file=file_change.filename,
                    rule="logic_safety_error",
                    suggestion="Manual review recommended for safety-critical changes",
                )
            )

        return issues

    def _is_validatable_file(self, file_change: FileChange) -> bool:
        """Check if file can be validated for logic safety."""
        return bool(
            file_change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
            and not file_change.binary
            and (file_change.patch or (file_change.before_content and file_change.after_content))
        )

    def _analyze_patch_safety(self, patch: str, filename: str) -> list[ValidationIssue]:
        """Analyze patch for dangerous logic changes."""
        issues = []

        patch_lines = patch.split("\n")
        added_lines = []
        removed_lines = []

        current_line_number = 0

        for i, line in enumerate(patch_lines):
            # Track line numbers from hunk headers
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line_number = int(match.group(1))
                continue

            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append((current_line_number, line[1:]))
                current_line_number += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines.append((current_line_number, line[1:]))
            elif line.startswith(" "):
                current_line_number += 1

        # Check for dangerous logic inversions
        for removed_line_num, removed_line in removed_lines:
            for added_line_num, added_line in added_lines:
                violations = self._detect_logic_inversions(
                    removed_line, added_line, removed_line_num, added_line_num
                )
                for violation in violations:
                    issues.append(self._violation_to_issue(violation, filename))

        # Check for dangerous pattern modifications
        pattern_issues = self._check_dangerous_patterns(added_lines, removed_lines, filename)
        issues.extend(pattern_issues)

        return issues

    def _detect_logic_inversions(
        self,
        removed_line: str,
        added_line: str,
        removed_line_num: int,
        added_line_num: int,
    ) -> list[SafetyViolation]:
        """Detect dangerous logic inversions between two lines."""
        violations: list[SafetyViolation] = []

        # Check if lines are similar enough to be considered related
        similarity = difflib.SequenceMatcher(None, removed_line.strip(), added_line.strip()).ratio()
        if similarity < 0.6:  # Lines too different to be related
            return violations

        # Check each inversion pattern
        for old_pattern, new_pattern, description in self.inversion_patterns:
            if re.search(old_pattern, removed_line) and re.search(new_pattern, added_line):
                # Verify the inversion by checking if replacing the pattern makes lines more similar
                removed_normalized = re.sub(old_pattern, "PATTERN", removed_line)
                added_normalized = re.sub(new_pattern, "PATTERN", added_line)
                normalized_similarity = difflib.SequenceMatcher(
                    None, removed_normalized, added_normalized
                ).ratio()

                if normalized_similarity > 0.8:  # High similarity after pattern replacement
                    confidence = min(similarity + 0.2, 1.0)
                    severity = self._assess_inversion_severity(
                        old_pattern, new_pattern, removed_line
                    )

                    violation = SafetyViolation(
                        violation_type="LOGIC_INVERSION",
                        severity=severity,
                        line_number=added_line_num,
                        before_code=removed_line.strip(),
                        after_code=added_line.strip(),
                        explanation=f"Dangerous logic inversion detected: {description}",
                        recommendation=f"Verify that changing '{old_pattern}' to '{new_pattern}' is intentional and correct",
                        confidence=confidence,
                    )
                    violations.append(violation)

        return violations

    def _assess_inversion_severity(self, old_pattern: str, new_pattern: str, context: str) -> str:
        """Assess the severity of a logic inversion."""
        # High severity for security-sensitive contexts
        context_lower = context.lower()
        for security_pattern in self.security_patterns:
            if re.search(security_pattern, context_lower):
                return "high"

        # High severity for critical boolean operations
        if old_pattern in [r"\band\b", r"\bor\b", r"&&", r"\|\|"]:
            return "high"

        # High severity for equality/inequality in conditions
        if "i" in context_lower and old_pattern in [r"==", r"!="]:
            return "high"

        # Medium severity for other comparison operators
        if old_pattern in [r"<", r">", r"<=", r">="]:
            return "medium"

        # Medium severity for boolean literals
        if old_pattern in [
            r"\bTrue\b",
            r"\bFalse\b",
            r"\btrue\b",
            r"\bfalse\b",
        ]:
            return "medium"

        return "medium"

    def _check_dangerous_patterns(
        self,
        added_lines: list[tuple[int, str]],
        removed_lines: list[tuple[int, str]],
        filename: str,
    ) -> list[ValidationIssue]:
        """Check for dangerous pattern modifications."""
        issues = []

        # Check for security function modifications
        for line_num, line in added_lines + removed_lines:
            line_lower = line.lower()
            for security_pattern in self.security_patterns:
                if re.search(security_pattern, line_lower):
                    # This is a security-sensitive line change
                    issues.append(
                        ValidationIssue(
                            type="security_sensitive_change",
                            severity="medium",
                            message=f"Security-sensitive function '{security_pattern}' modified",
                            file=filename,
                            line=line_num,
                            rule="security_function_safety",
                            suggestion=f"Carefully review changes to security-sensitive code: {line.strip()}",
                        )
                    )
                    break

        # Check for error handling modifications
        for line_num, line in added_lines + removed_lines:
            for error_pattern in self.error_patterns:
                if re.search(error_pattern, line, re.IGNORECASE):
                    issues.append(
                        ValidationIssue(
                            type="error_handling_change",
                            severity="medium",
                            message="Error handling logic modified",
                            file=filename,
                            line=line_num,
                            rule="error_handling_safety",
                            suggestion=f"Review error handling changes: {line.strip()}",
                        )
                    )
                    break

        return issues

    async def _analyze_ast_safety(self, file_change: FileChange) -> list[ValidationIssue]:
        """Analyze AST changes for safety issues."""
        issues: list[ValidationIssue] = []

        try:
            # Parse ASTs
            before_ast, _ = self.parser.parse_with_fallback(
                file_change.before_content or "",
                f"{file_change.filename}:before",
            )
            after_ast, _ = self.parser.parse_with_fallback(
                file_change.after_content or "",
                f"{file_change.filename}:after",
            )

            if not before_ast or not after_ast:
                return issues

            # Check for dangerous control flow changes
            control_flow_issues = self._analyze_control_flow_changes(
                before_ast, after_ast, file_change.filename
            )
            issues.extend(control_flow_issues)

            # Check for exception handling changes
            exception_issues = self._analyze_exception_handling_changes(
                before_ast, after_ast, file_change.filename
            )
            issues.extend(exception_issues)

        except Exception as e:
            logger.error(f"Error in AST safety analysis for {file_change.filename}: {e}")

        return issues

    def _analyze_control_flow_changes(
        self, before_ast: ast.AST, after_ast: ast.AST, filename: str
    ) -> list[ValidationIssue]:
        """Analyze control flow changes for safety issues."""
        issues = []

        # Extract control flow nodes
        before_flows = self._extract_control_flow_nodes(before_ast)
        after_flows = self._extract_control_flow_nodes(after_ast)

        # Check for significant changes in control flow
        if len(before_flows) != len(after_flows):
            diff = abs(len(after_flows) - len(before_flows))
            issues.append(
                ValidationIssue(
                    type="control_flow_change",
                    severity="medium",
                    message=f"Significant control flow changes detected ({diff} flow changes)",
                    file=filename,
                    rule="control_flow_safety",
                    suggestion="Review control flow changes for correctness and safety",
                )
            )

        return issues

    def _analyze_exception_handling_changes(
        self, before_ast: ast.AST, after_ast: ast.AST, filename: str
    ) -> list[ValidationIssue]:
        """Analyze exception handling changes."""
        issues = []

        # Extract exception handling nodes
        before_exceptions = self._extract_exception_nodes(before_ast)
        after_exceptions = self._extract_exception_nodes(after_ast)

        # Check for removed exception handling
        if len(before_exceptions) > len(after_exceptions):
            removed_count = len(before_exceptions) - len(after_exceptions)
            issues.append(
                ValidationIssue(
                    type="exception_handling_removed",
                    severity="high",
                    message=f"{removed_count} exception handler(s) removed",
                    file=filename,
                    rule="exception_safety",
                    suggestion="Ensure exception handling removal doesn't introduce unhandled errors",
                )
            )

        # Check for catch-all exception handlers
        for node_info in after_exceptions:
            if node_info.get("is_catch_all", False):
                issues.append(
                    ValidationIssue(
                        type="catch_all_exception",
                        severity="medium",
                        message="Catch-all exception handler detected",
                        file=filename,
                        line=node_info.get("line", 0),
                        rule="exception_specificity",
                        suggestion="Consider catching specific exceptions instead of using bare except",
                    )
                )

        return issues

    def _extract_control_flow_nodes(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract control flow nodes from AST."""
        nodes = []

        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Break,
                    ast.Continue,
                    ast.Return,
                ),
            ):
                nodes.append(
                    {
                        "type": type(node).__name__,
                        "line": getattr(node, "lineno", 0),
                        "col": getattr(node, "col_offset", 0),
                    }
                )

        return nodes

    def _extract_exception_nodes(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract exception handling nodes from AST."""
        nodes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    is_catch_all = handler.type is None
                    nodes.append(
                        {
                            "type": "ExceptHandler",
                            "line": getattr(handler, "lineno", 0),
                            "exception_type": (ast.unparse(handler.type) if handler.type else None),
                            "is_catch_all": is_catch_all,
                        }
                    )
            elif isinstance(node, ast.Raise):
                nodes.append(
                    {
                        "type": "Raise",
                        "line": getattr(node, "lineno", 0),
                        "exception": (ast.unparse(node.exc) if node.exc else None),
                    }
                )

        return nodes

    def _analyze_security_changes(self, file_change: FileChange) -> list[ValidationIssue]:
        """Analyze security-sensitive changes."""
        issues: list[ValidationIssue] = []

        if not file_change.patch:
            return issues

        patch_lines = file_change.patch.split("\n")

        for i, line in enumerate(patch_lines):
            if line.startswith(("+", "-")):
                line_content = line[1:].lower()

                # Check for security-sensitive keywords
                for pattern in self.security_patterns:
                    if re.search(pattern, line_content):
                        severity = "high" if line.startswith("-") else "medium"
                        action = "removed" if line.startswith("-") else "added"

                        issues.append(
                            ValidationIssue(
                                type="security_code_change",
                                severity=severity,
                                message=f"Security-sensitive code {action}: contains '{pattern}'",
                                file=file_change.filename,
                                line=i + 1,
                                rule="security_change_detection",
                                suggestion="Carefully review security implications of this change",
                            )
                        )
                        break

        return issues

    def _analyze_error_handling_changes(self, file_change: FileChange) -> list[ValidationIssue]:
        """Analyze error handling changes."""
        issues: list[ValidationIssue] = []

        if not file_change.patch:
            return issues

        patch_lines = file_change.patch.split("\n")

        for i, line in enumerate(patch_lines):
            if line.startswith(("+", "-")):
                line_content = line[1:]

                # Check for error handling patterns
                for pattern in self.error_patterns:
                    if re.search(pattern, line_content, re.IGNORECASE):
                        severity = "medium"
                        action = "removed" if line.startswith("-") else "added"

                        issues.append(
                            ValidationIssue(
                                type="error_handling_change",
                                severity=severity,
                                message=f"Error handling {action}",
                                file=file_change.filename,
                                line=i + 1,
                                rule="error_handling_detection",
                                suggestion="Review error handling changes for correctness",
                            )
                        )
                        break

        return issues

    def _violation_to_issue(self, violation: SafetyViolation, filename: str) -> ValidationIssue:
        """Convert SafetyViolation to ValidationIssue."""
        return ValidationIssue(
            type=violation.violation_type.lower(),
            severity=violation.severity,
            message=violation.explanation,
            file=filename,
            line=violation.line_number,
            rule="logic_safety_validator",
            suggestion=violation.recommendation,
        )
