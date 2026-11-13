"""
Breaking change detector for API compatibility analysis.

This module detects breaking changes in code that could affect
external consumers or API compatibility.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

from coding_client.models.pr_validation import (
    BreakingChange,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.semantic.ast_diff_analyzer import (
    SafeASTParser,
)

logger = logging.getLogger(__name__)


@dataclass
class APIElement:
    """Represents an API element (function, class, etc.)."""

    name: str
    type: str  # 'function', 'class', 'variable', 'constant'
    signature: str | None
    location: tuple[int, int]  # (line, column)
    visibility: str  # 'public', 'private', 'protected'
    metadata: dict[str, Any]


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue."""

    issue_type: str
    severity: str
    element_name: str
    description: str
    before_signature: str | None
    after_signature: str | None
    recommendation: str
    breaking_score: float  # 0.0 to 1.0


class BreakingChangeDetector:
    """
    Detects breaking changes in code by comparing API surfaces.

    This detector identifies:
    - Removed public functions/classes
    - Modified function signatures
    - Changed class hierarchies
    - Removed public constants/variables
    - Type annotation changes
    """

    def __init__(self):
        self.parser = SafeASTParser()

        # Public API prefixes that are considered part of the API
        self.public_api_prefixes = [
            "__init__",
            "__call__",
            "__enter__",
            "__exit__",
            "__str__",
            "__repr__",
        ]

        # Breaking change severity weights
        self.severity_weights = {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "LOW": 0.2,
        }

        # Language-specific patterns
        self.language_handlers = {
            FileType.PYTHON: self._analyze_python_breaking_changes,
            FileType.JAVASCRIPT: self._analyze_javascript_breaking_changes,
            FileType.TYPESCRIPT: self._analyze_typescript_breaking_changes,
        }

    async def detect_breaking_changes(self, file_change: FileChange) -> list[BreakingChange]:
        """
        Detect breaking changes in a file change.

        Args:
            file_change: The file change to analyze

        Returns:
            List of BreakingChange objects
        """
        if not self._is_analyzable_file(file_change):
            return []

        # Use language-specific handler
        handler = self.language_handlers.get(file_change.file_type)
        if handler:
            return await handler(file_change)

        return []

    def _is_analyzable_file(self, file_change: FileChange) -> bool:
        """Check if file can be analyzed for breaking changes."""
        return bool(
            file_change.file_type in self.language_handlers
            and not file_change.binary
            and file_change.before_content
            and file_change.after_content
        )

    async def _analyze_python_breaking_changes(
        self, file_change: FileChange
    ) -> list[BreakingChange]:
        """Analyze breaking changes in Python files."""
        try:
            # Ensure content is not None
            before_content = file_change.before_content or ""
            after_content = file_change.after_content or ""

            # Parse ASTs
            before_ast, before_errors = self.parser.parse_with_fallback(
                before_content, f"{file_change.filename}:before"
            )
            after_ast, after_errors = self.parser.parse_with_fallback(
                after_content, f"{file_change.filename}:after"
            )

            if not before_ast or not after_ast:
                return self._handle_parse_failure(
                    file_change.filename, before_errors + after_errors
                )

            # Extract API surfaces
            before_api = self._extract_python_api(before_ast)
            after_api = self._extract_python_api(after_ast)

            # Compare APIs
            compatibility_issues = self._compare_python_apis(before_api, after_api)

            # Convert to breaking changes
            breaking_changes = []
            for issue in compatibility_issues:
                breaking_change = BreakingChange(
                    type=issue.issue_type,
                    severity=issue.severity,
                    description=issue.description,
                    file_path=file_change.filename,
                    affected_symbol=issue.element_name,
                    recommendation=issue.recommendation,
                    confidence=issue.breaking_score,
                )
                breaking_changes.append(breaking_change)

            return breaking_changes

        except Exception as e:
            logger.error(f"Error analyzing Python breaking changes in {file_change.filename}: {e}")
            return [
                BreakingChange(
                    type="ANALYSIS_ERROR",
                    severity="MEDIUM",
                    description=f"Breaking change analysis failed: {str(e)[:100]}",
                    file_path=file_change.filename,
                    affected_symbol="",
                    recommendation="Manual review required for API compatibility",
                )
            ]

    def _extract_python_api(self, tree: ast.AST) -> dict[str, APIElement]:
        """Extract public API elements from Python AST."""
        api_elements = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_public_python_function(node.name):
                    element = self._create_python_function_element(node)
                    api_elements[element.name] = element

            elif isinstance(node, ast.AsyncFunctionDef):
                if self._is_public_python_function(node.name):
                    element = self._create_python_function_element(node, is_async=True)
                    api_elements[element.name] = element

            elif isinstance(node, ast.ClassDef):
                if self._is_public_python_class(node.name):
                    element = self._create_python_class_element(node)
                    api_elements[element.name] = element

            elif isinstance(node, ast.Assign):
                # Check for module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and self._is_public_python_constant(target.id):
                        element = self._create_python_constant_element(target, node)
                        api_elements[element.name] = element

        return api_elements

    def _create_python_function_element(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool = False,
    ) -> APIElement:
        """Create APIElement for Python function."""
        signature = self._extract_python_function_signature(node, is_async)

        metadata = {
            "args": [arg.arg for arg in node.args.args],
            "defaults": len(node.args.defaults),
            "varargs": node.args.vararg.arg if node.args.vararg else None,
            "kwargs": node.args.kwarg.arg if node.args.kwarg else None,
            "decorators": [self._ast_to_string(dec) for dec in node.decorator_list],
            "return_annotation": (self._ast_to_string(node.returns) if node.returns else None),
            "is_async": is_async,
        }

        return APIElement(
            name=node.name,
            type="function",
            signature=signature,
            location=(node.lineno, node.col_offset),
            visibility="public",
            metadata=metadata,
        )

    def _create_python_class_element(self, node: ast.ClassDef) -> APIElement:
        """Create APIElement for Python class."""
        bases = [self._ast_to_string(base) for base in node.bases]

        # Extract public methods
        methods = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_public_python_function(item.name):
                    method_sig = self._extract_python_function_signature(
                        item, isinstance(item, ast.AsyncFunctionDef)
                    )
                    methods[item.name] = method_sig

        metadata = {
            "bases": bases,
            "methods": methods,
            "decorators": [self._ast_to_string(dec) for dec in node.decorator_list],
        }

        return APIElement(
            name=node.name,
            type="class",
            signature=f"class {node.name}({', '.join(bases)})",
            location=(node.lineno, node.col_offset),
            visibility="public",
            metadata=metadata,
        )

    def _create_python_constant_element(
        self, target: ast.Name, assign_node: ast.Assign
    ) -> APIElement:
        """Create APIElement for Python module-level constant."""
        value_str = self._ast_to_string(assign_node.value)

        metadata = {
            "value": value_str,
            "type_annotation": None,  # Could be augmented to detect type annotations
        }

        return APIElement(
            name=target.id,
            type="constant",
            signature=f"{target.id} = {value_str}",
            location=(assign_node.lineno, assign_node.col_offset),
            visibility="public",
            metadata=metadata,
        )

    def _extract_python_function_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool = False,
    ) -> str:
        """Extract function signature string."""
        parts = []

        if is_async:
            parts.append("async")

        parts.append("de")
        parts.append(node.name)

        # Build arguments
        args = []

        # Regular arguments
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._ast_to_string(arg.annotation)}"
            args.append(arg_str)

        # Add defaults
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                idx = len(node.args.args) - num_defaults + i
                if 0 <= idx < len(args):
                    args[idx] += f" = {self._ast_to_string(default)}"

        # *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {self._ast_to_string(node.args.vararg.annotation)}"
            args.append(vararg_str)

        # **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {self._ast_to_string(node.args.kwarg.annotation)}"
            args.append(kwarg_str)

        signature = f"({', '.join(args)})"

        # Return annotation
        if node.returns:
            signature += f" -> {self._ast_to_string(node.returns)}"

        return f"{' '.join(parts)}{signature}"

    def _compare_python_apis(
        self,
        before_api: dict[str, APIElement],
        after_api: dict[str, APIElement],
    ) -> list[CompatibilityIssue]:
        """Compare Python API surfaces and find compatibility issues."""
        issues = []

        before_names = set(before_api.keys())
        after_names = set(after_api.keys())

        # Removed elements
        removed_names = before_names - after_names
        for name in removed_names:
            element = before_api[name]
            issues.append(
                CompatibilityIssue(
                    issue_type=f"{element.type.upper()}_REMOVED",
                    severity="HIGH",
                    element_name=name,
                    description=f"Public {element.type} '{name}' was removed",
                    before_signature=element.signature,
                    after_signature=None,
                    recommendation="Consider deprecating instead of removing, or provide migration path",
                    breaking_score=0.9,
                )
            )

        # Modified elements
        common_names = before_names & after_names
        for name in common_names:
            before_element = before_api[name]
            after_element = after_api[name]

            if before_element.type != after_element.type:
                issues.append(
                    CompatibilityIssue(
                        issue_type="ELEMENT_TYPE_CHANGED",
                        severity="HIGH",
                        element_name=name,
                        description=f"Element '{name}' changed from {before_element.type} to {after_element.type}",
                        before_signature=before_element.signature,
                        after_signature=after_element.signature,
                        recommendation="This is a breaking change that requires careful migration",
                        breaking_score=1.0,
                    )
                )

            elif before_element.signature != after_element.signature:
                if before_element.type == "function":
                    function_issues = self._analyze_function_signature_changes(
                        before_element, after_element
                    )
                    issues.extend(function_issues)
                elif before_element.type == "class":
                    class_issues = self._analyze_class_changes(before_element, after_element)
                    issues.extend(class_issues)

        return issues

    def _analyze_function_signature_changes(
        self, before: APIElement, after: APIElement
    ) -> list[CompatibilityIssue]:
        """Analyze function signature changes for breaking changes."""
        issues = []

        before_meta = before.metadata
        after_meta = after.metadata

        # Check argument changes
        before_args = before_meta.get("args", [])
        after_args = after_meta.get("args", [])

        # Removed arguments
        if len(after_args) < len(before_args):
            issues.append(
                CompatibilityIssue(
                    issue_type="FUNCTION_ARGS_REMOVED",
                    severity="HIGH",
                    element_name=before.name,
                    description=f"Function '{before.name}' has fewer arguments ({len(after_args)} vs {len(before_args)})",
                    before_signature=before.signature,
                    after_signature=after.signature,
                    recommendation="Consider making removed arguments optional or provide overloaded versions",
                    breaking_score=0.8,
                )
            )

        # Added required arguments
        elif len(after_args) > len(before_args):
            before_defaults = before_meta.get("defaults", 0)
            after_defaults = after_meta.get("defaults", 0)

            new_args = len(after_args) - len(before_args)
            new_defaults = after_defaults - before_defaults

            if new_args > new_defaults:
                issues.append(
                    CompatibilityIssue(
                        issue_type="FUNCTION_REQUIRED_ARGS_ADDED",
                        severity="HIGH",
                        element_name=before.name,
                        description=f"Function '{before.name}' added {new_args - new_defaults} required arguments",
                        before_signature=before.signature,
                        after_signature=after.signature,
                        recommendation="Make new arguments optional with default values",
                        breaking_score=0.9,
                    )
                )

        # Check argument reordering
        common_len = min(len(before_args), len(after_args))
        for i in range(common_len):
            if before_args[i] != after_args[i]:
                issues.append(
                    CompatibilityIssue(
                        issue_type="FUNCTION_ARGS_REORDERED",
                        severity="HIGH",
                        element_name=before.name,
                        description=f"Function '{before.name}' arguments reordered at position {i}",
                        before_signature=before.signature,
                        after_signature=after.signature,
                        recommendation="Avoid reordering arguments; add new arguments at the end",
                        breaking_score=0.7,
                    )
                )
                break

        # Check return type changes (if annotated)
        before_return = before_meta.get("return_annotation")
        after_return = after_meta.get("return_annotation")

        if before_return and after_return and before_return != after_return:
            issues.append(
                CompatibilityIssue(
                    issue_type="FUNCTION_RETURN_TYPE_CHANGED",
                    severity="MEDIUM",
                    element_name=before.name,
                    description=f"Function '{before.name}' return type changed from {before_return} to {after_return}",
                    before_signature=before.signature,
                    after_signature=after.signature,
                    recommendation="Ensure return type changes are backward compatible",
                    breaking_score=0.5,
                )
            )

        # Check async/sync changes
        before_async = before_meta.get("is_async", False)
        after_async = after_meta.get("is_async", False)

        if before_async != after_async:
            change_type = "sync to async" if after_async else "async to sync"
            issues.append(
                CompatibilityIssue(
                    issue_type="FUNCTION_ASYNC_CHANGED",
                    severity="HIGH",
                    element_name=before.name,
                    description=f"Function '{before.name}' changed from {change_type}",
                    before_signature=before.signature,
                    after_signature=after.signature,
                    recommendation="This is a breaking change requiring caller updates",
                    breaking_score=0.9,
                )
            )

        return issues

    def _analyze_class_changes(
        self, before: APIElement, after: APIElement
    ) -> list[CompatibilityIssue]:
        """Analyze class changes for breaking changes."""
        issues = []

        before_meta = before.metadata
        after_meta = after.metadata

        # Check base class changes
        before_bases = set(before_meta.get("bases", []))
        after_bases = set(after_meta.get("bases", []))

        removed_bases = before_bases - after_bases
        if removed_bases:
            issues.append(
                CompatibilityIssue(
                    issue_type="CLASS_INHERITANCE_REMOVED",
                    severity="HIGH",
                    element_name=before.name,
                    description=f"Class '{before.name}' no longer inherits from: {', '.join(removed_bases)}",
                    before_signature=before.signature,
                    after_signature=after.signature,
                    recommendation="Removing inheritance can break isinstance() checks and method availability",
                    breaking_score=0.8,
                )
            )

        # Check method changes
        before_methods = before_meta.get("methods", {})
        after_methods = after_meta.get("methods", {})

        removed_methods = set(before_methods.keys()) - set(after_methods.keys())
        for method_name in removed_methods:
            issues.append(
                CompatibilityIssue(
                    issue_type="CLASS_METHOD_REMOVED",
                    severity="HIGH",
                    element_name=f"{before.name}.{method_name}",
                    description=f"Public method '{method_name}' removed from class '{before.name}'",
                    before_signature=before_methods[method_name],
                    after_signature=None,
                    recommendation="Consider deprecating methods instead of removing them",
                    breaking_score=0.8,
                )
            )

        # Check modified methods
        for method_name in set(before_methods.keys()) & set(after_methods.keys()):
            if before_methods[method_name] != after_methods[method_name]:
                issues.append(
                    CompatibilityIssue(
                        issue_type="CLASS_METHOD_SIGNATURE_CHANGED",
                        severity="MEDIUM",
                        element_name=f"{before.name}.{method_name}",
                        description=f"Method '{method_name}' signature changed in class '{before.name}'",
                        before_signature=before_methods[method_name],
                        after_signature=after_methods[method_name],
                        recommendation="Ensure method signature changes are backward compatible",
                        breaking_score=0.6,
                    )
                )

        return issues

    async def _analyze_javascript_breaking_changes(
        self, file_change: FileChange
    ) -> list[BreakingChange]:
        """Analyze breaking changes in JavaScript files."""
        # Streamlined JavaScript analysis - could be augmented with proper JS parser
        breaking_changes = []

        if file_change.patch:
            # Look for function signature changes in exports
            patch_lines = file_change.patch.split("\n")

            for line in patch_lines:
                if line.startswith("-") and "export" in line and "function" in line:
                    # Exported function removed or changed
                    func_match = re.search(r"export.*function\s+(\w+)", line)
                    if func_match:
                        func_name = func_match.group(1)
                        breaking_changes.append(
                            BreakingChange(
                                type="EXPORTED_FUNCTION_CHANGED",
                                severity="HIGH",
                                description=f"Exported function '{func_name}' was modified or removed",
                                file_path=file_change.filename,
                                affected_symbol=func_name,
                                recommendation="Ensure exported function changes maintain backward compatibility",
                            )
                        )

        return breaking_changes

    async def _analyze_typescript_breaking_changes(
        self, file_change: FileChange
    ) -> list[BreakingChange]:
        """Analyze breaking changes in TypeScript files."""
        # Streamlined TypeScript analysis - could be augmented with proper TS parser
        breaking_changes = []

        if file_change.patch:
            # Look for interface and type changes
            patch_lines = file_change.patch.split("\n")

            for line in patch_lines:
                if line.startswith("-") and re.search(r"(interface|type)\s+\w+", line):
                    # Interface or type definition changed
                    type_match = re.search(r"(interface|type)\s+(\w+)", line)
                    if type_match:
                        type_kind = type_match.group(1)
                        type_name = type_match.group(2)
                        breaking_changes.append(
                            BreakingChange(
                                type=f"{type_kind.upper()}_CHANGED",
                                severity="HIGH",
                                description=f"TypeScript {type_kind} '{type_name}' was modified",
                                file_path=file_change.filename,
                                affected_symbol=type_name,
                                recommendation=f"Ensure {type_kind} changes maintain backward compatibility",
                            )
                        )

        return breaking_changes

    def _is_public_python_function(self, name: str) -> bool:
        """Check if Python function is public."""
        return not name.startswith("_") or name in self.public_api_prefixes

    def _is_public_python_class(self, name: str) -> bool:
        """Check if Python class is public."""
        return not name.startswith("_")

    def _is_public_python_constant(self, name: str) -> bool:
        """Check if Python constant is public."""
        return not name.startswith("_") and name.isupper()

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        try:
            return ast.unparse(node)
        except (AttributeError, ValueError):  # fmt: skip
            # Fallback for older Python versions or complex nodes
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Attribute):
                return f"{self._ast_to_string(node.value)}.{node.attr}"
            else:
                return f"<{type(node).__name__}>"

    def _handle_parse_failure(self, filename: str, errors: list[str]) -> list[BreakingChange]:
        """Handle AST parsing failures."""
        return [
            BreakingChange(
                type="PARSE_FAILURE",
                severity="MEDIUM",
                description=f"Could not parse file for breaking change analysis: {'; '.join(errors[:3])}",
                file_path=filename,
                affected_symbol="",
                recommendation="Check file syntax and review changes manually for API compatibility",
            )
        ]
