"""
AST diff analyzer for semantic analysis of code changes.

This module provides deep semantic analysis of code changes by comparing
Abstract Syntax Trees (ASTs) before and after modifications.
"""

import ast
import difflib
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from coding_client.models.pr_validation import (
    BreakingChange,
    FileChange,
    FileType,
    SemanticAnalysisResult,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """Represents an AST node with metadata.

    Attributes:
        node_type (str): Class attribute node_type.
        name (str): Class attribute name.
        lineno (int): Class attribute lineno.
        col_offset (int): Class attribute col_offset.
        end_lineno (Optional[int]): Class attribute end_lineno.
        end_col_offset (Optional[int]): Class attribute end_col_offset.
        signature (Optional[str]): Class attribute signature.
        attributes (Dict[str, Any]): Class attribute attributes.

    Example:
        >>> obj = ASTNode()"""

    node_type: str
    name: str
    lineno: int
    col_offset: int
    end_lineno: int | None = None
    end_col_offset: int | None = None
    signature: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ASTDiffResult:
    """Result of AST comparison.

    Attributes:
        added_nodes (List[ASTNode]): Class attribute added_nodes.
        removed_nodes (List[ASTNode]): Class attribute removed_nodes.
        modified_nodes (List[Tuple[ASTNode, ASTNode]]): Class attribute modified_nodes.
        unchanged_nodes (List[ASTNode]): Class attribute unchanged_nodes.
        complexity_change (float): Class attribute complexity_change.
        structural_changes (List[str]): Class attribute structural_changes.

    Example:
        >>> obj = ASTDiffResult()"""

    added_nodes: list[ASTNode]
    removed_nodes: list[ASTNode]
    modified_nodes: list[tuple[ASTNode, ASTNode]]  # (before, after)
    unchanged_nodes: list[ASTNode]
    complexity_change: float
    structural_changes: list[str]


class SafeASTParser:
    """Safe AST parser with fallback strategies.

    Attributes:
        supported_python_versions (List): Instance attribute supported_python_versions.
    """

    def __init__(self) -> None:
        self.supported_python_versions = [
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
        ]

    def parse_with_fallback(
        self, content: str, filename: str = "<unknown>"
    ) -> tuple[ast.AST | None, list[str]]:
        """Parse AST with version fallbacks and error collection.

        Args:
            content (str): The content parameter.
            filename (str, optional): The filename parameter. Defaults to '<unknown>'.

        Returns:
            Tuple[Optional[ast.AST], List[str]]: The tuple[optional[ast.ast], list[str]] result.

        Example:
            >>> parse_with_fallback("content_value", "example")"""
        errors = []

        if not content or not content.strip():
            return None, ["Empty content"]

        # Try encoding detection first
        try:
            if isinstance(content, bytes):
                import chardet

                detected = chardet.detect(content)
                encoding = detected.get("encoding", "utf-8")
                content = content.decode(encoding, errors="replace")
        except Exception as e:
            errors.append(f"Encoding detection failed: {e}")

        # Try current Python version first
        try:
            return ast.parse(content, filename=filename), errors
        except SyntaxError as e:
            errors.append(f"Syntax error in Python {sys.version_info[:2]}: {e}")
        except Exception as e:
            errors.append(f"Parse error: {e}")

        # Try older Python syntax modes
        for version in self.supported_python_versions:
            if version >= sys.version_info[:2]:
                continue

            try:
                # Use compile with older version flags
                compile(content, filename, "exec")
                tree = ast.parse(content, filename=filename)
                errors.append(f"Parsed successfully with Python {version} compatibility")
                return tree, errors
            except Exception as e:
                errors.append(f"Failed with Python {version}: {e}")

        # Last resort: try to parse partial content
        try:
            lines = content.split("\n")
            for i in range(len(lines), 0, -10):  # Try removing 10 lines at a time
                partial_content = "\n".join(lines[:i])
                try:
                    tree = ast.parse(partial_content, filename=filename)
                    errors.append(f"Partial parse successful ({i}/{len(lines)} lines)")
                    return tree, errors
                except SyntaxError:
                    continue
        except Exception as e:
            errors.append(f"Partial parse failed: {e}")

        return None, errors


class ASTDiffAnalyzer:
    """Analyzes semantic changes by comparing ASTs.

    This analyzer provides deep understanding of code changes including:
    - Function and class modifications
    - Structural changes to code organization
    - Complexity changes
    - Logic flow modifications

    Attributes:
        parser (SafeASTParser): Instance attribute parser.
        important_nodes (Set): Instance attribute important_nodes.
        complexity_nodes (Set): Instance attribute complexity_nodes."""

    def __init__(self) -> None:
        self.parser = SafeASTParser()

        # Node types we care about for semantic analysis
        self.important_nodes = {
            "FunctionDef",
            "AsyncFunctionDef",
            "ClassDef",
            "Import",
            "ImportFrom",
            "If",
            "For",
            "While",
            "Try",
            "With",
            "Return",
            "Yield",
            "Assert",
            "Raise",
            "Global",
            "Nonlocal",
        }

        # Complexity indicators
        self.complexity_nodes = {
            "If",
            "For",
            "While",
            "Try",
            "ExceptHandler",
            "With",
            "AsyncWith",
            "ListComp",
            "SetComp",
            "DictComp",
            "GeneratorExp",
            "Lambda",
        }

    async def analyze_semantic_changes(self, file_change: FileChange) -> SemanticAnalysisResult:
        """Analyze semantic changes in a file.

        Args:
            file_change (FileChange): The file_change parameter.

        Returns:
            SemanticAnalysisResult:     SemanticAnalysisResult with detailed analysis.

        Example:
            >>> await analyze_semantic_changes("file_change_value")"""
        if not self._is_analyzable_file(file_change):
            return SemanticAnalysisResult(
                breaking_changes=[],
                logic_safety_issues=[],
                api_surface_changes={},
                complexity_changes={},
            )

        try:
            # Parse ASTs
            before_ast, before_errors = self.parser.parse_with_fallback(
                file_change.before_content or "",
                f"{file_change.filename}:before",
            )
            after_ast, after_errors = self.parser.parse_with_fallback(
                file_change.after_content or "",
                f"{file_change.filename}:after",
            )

            # Log parsing issues
            if before_errors:
                logger.warning(
                    f"Before content parsing issues for {file_change.filename}: {before_errors}"
                )
            if after_errors:
                logger.warning(
                    f"After content parsing issues for {file_change.filename}: {after_errors}"
                )

            # Handle parsing failures
            if not before_ast and not after_ast:
                return self._create_parse_failure_result(file_change.filename)

            # Perform AST diff analysis
            ast_diff = self._compare_asts(before_ast, after_ast)

            # Analyze changes for breaking changes
            breaking_changes = self._detect_breaking_changes(ast_diff, file_change.filename)

            # Analyze logic safety
            logic_safety_issues = self._analyze_logic_safety(ast_diff, file_change)

            # Analyze API surface changes
            api_surface_changes = self._analyze_api_surface_changes(before_ast, after_ast)

            # Calculate complexity changes
            complexity_changes = self._calculate_complexity_changes(before_ast, after_ast)

            return SemanticAnalysisResult(
                breaking_changes=breaking_changes,
                logic_safety_issues=logic_safety_issues,
                api_surface_changes=api_surface_changes,
                complexity_changes=complexity_changes,
            )

        except Exception as e:
            logger.error(f"Error in semantic analysis for {file_change.filename}: {e}")
            return SemanticAnalysisResult(
                breaking_changes=[
                    BreakingChange(
                        type="ANALYSIS_ERROR",
                        severity="MEDIUM",
                        description=f"Semantic analysis failed: {str(e)[:100]}",
                        file_path=file_change.filename,
                        affected_symbol="",
                        recommendation="Review file manually for semantic changes",
                    )
                ],
                logic_safety_issues=[],
                api_surface_changes={},
                complexity_changes={},
            )

    def _is_analyzable_file(self, file_change: FileChange) -> bool:
        """Check if file can be analyzed semantically."""
        # Currently supports Python files
        return file_change.file_type == FileType.PYTHON and not file_change.binary

    def _compare_asts(self, before_ast: ast.AST | None, after_ast: ast.AST | None) -> ASTDiffResult:
        """Compare two ASTs and identify differences."""
        if not before_ast:
            # New file
            after_nodes = self._extract_nodes(after_ast) if after_ast else []
            return ASTDiffResult(
                added_nodes=after_nodes,
                removed_nodes=[],
                modified_nodes=[],
                unchanged_nodes=[],
                complexity_change=(self._calculate_ast_complexity(after_ast) if after_ast else 0.0),
                structural_changes=["NEW_FILE"],
            )

        if not after_ast:
            # Deleted file
            before_nodes = self._extract_nodes(before_ast)
            return ASTDiffResult(
                added_nodes=[],
                removed_nodes=before_nodes,
                modified_nodes=[],
                unchanged_nodes=[],
                complexity_change=-self._calculate_ast_complexity(before_ast),
                structural_changes=["FILE_DELETED"],
            )

        # Compare existing files
        before_node_dict = {self._node_key(node): node for node in self._extract_nodes(before_ast)}
        after_node_dict = {self._node_key(node): node for node in self._extract_nodes(after_ast)}

        # Find differences
        before_keys = set(before_node_dict.keys())
        after_keys = set(after_node_dict.keys())

        added_keys = after_keys - before_keys
        removed_keys = before_keys - after_keys
        common_keys = before_keys & after_keys

        added_nodes = [after_node_dict[key] for key in added_keys]
        removed_nodes = [before_node_dict[key] for key in removed_keys]

        # Check for modifications in common nodes
        modified_nodes = []
        unchanged_nodes = []

        for key in common_keys:
            before_node = before_node_dict[key]
            after_node = after_node_dict[key]

            if self._nodes_are_different(before_node, after_node):
                modified_nodes.append((before_node, after_node))
            else:
                unchanged_nodes.append(before_node)

        # Calculate complexity change
        before_complexity = self._calculate_ast_complexity(before_ast)
        after_complexity = self._calculate_ast_complexity(after_ast)
        complexity_change = after_complexity - before_complexity

        # Identify structural changes
        structural_changes = self._identify_structural_changes(
            added_nodes, removed_nodes, modified_nodes
        )

        return ASTDiffResult(
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            modified_nodes=modified_nodes,
            unchanged_nodes=unchanged_nodes,
            complexity_change=complexity_change,
            structural_changes=structural_changes,
        )

    def _extract_nodes(self, tree: ast.AST) -> list[ASTNode]:
        """Extract important nodes from AST."""
        nodes = []

        for node in ast.walk(tree):
            if type(node).__name__ in self.important_nodes:
                ast_node = self._create_ast_node(node)
                nodes.append(ast_node)

        return nodes

    def _create_ast_node(self, node: ast.AST) -> ASTNode:
        """Create ASTNode from ast node."""
        node_type = type(node).__name__
        name = ""
        signature = None
        attributes: dict[str, Any] = {}

        # Extract name based on node type
        if hasattr(node, "name"):
            name = str(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            for alias in node.names:
                names.append(alias.name)
            name = ", ".join(names)
            if isinstance(node, ast.ImportFrom) and node.module:
                name = f"from {node.module} import {name}"

        # Extract signature for functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = self._extract_function_signature(node)
            attributes["args"] = [arg.arg for arg in node.args.args]
            attributes["decorators"] = [self._ast_to_string(dec) for dec in node.decorator_list]
            attributes["is_async"] = isinstance(node, ast.AsyncFunctionDef)

        # Extract attributes for classes
        elif isinstance(node, ast.ClassDef):
            attributes["bases"] = [self._ast_to_string(base) for base in node.bases]
            attributes["decorators"] = [self._ast_to_string(dec) for dec in node.decorator_list]

        return ASTNode(
            node_type=node_type,
            name=name,
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            signature=signature,
            attributes=attributes,
        )

    def _extract_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature as string."""
        parts = []

        if isinstance(node, ast.AsyncFunctionDef):
            parts.append("async")

        parts.append("de")
        parts.append(node.name)

        # Arguments
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._ast_to_string(arg.annotation)}"
            args.append(arg_str)

        # Defaults
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                idx = len(args) - num_defaults + i
                if idx >= 0 and idx < len(args):
                    args[idx] += f" = {self._ast_to_string(default)}"

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        signature = f"({', '.join(args)})"

        # Return annotation
        if node.returns:
            signature += f" -> {self._ast_to_string(node.returns)}"

        return f"{' '.join(parts)}{signature}"

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

    def _node_key(self, node: ASTNode) -> str:
        """Generate a unique key for an AST node."""
        return f"{node.node_type}:{node.name}:{node.signature or ''}"

    def _nodes_are_different(self, before: ASTNode, after: ASTNode) -> bool:
        """Check if two nodes are different."""
        # Compare signatures for functions
        if before.signature and after.signature:
            return before.signature != after.signature

        # Compare attributes
        if before.attributes != after.attributes:
            return True

        # If we can't determine difference, assume they're the same
        return False

    def _calculate_ast_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of AST."""
        if not tree:
            return 0.0

        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            node_type = type(node).__name__
            if node_type in self.complexity_nodes:
                if node_type == "If":
                    complexity += 1
                    # Add for elif branches
                    complexity += len(getattr(node, "orelse", []))
                elif node_type in ["For", "While"]:
                    complexity += 1
                elif node_type == "Try":
                    complexity += len(getattr(node, "handlers", []))
                else:
                    complexity += 1

        return float(complexity)

    def _identify_structural_changes(
        self,
        added_nodes: list[ASTNode],
        removed_nodes: list[ASTNode],
        modified_nodes: list[tuple[ASTNode, ASTNode]],
    ) -> list[str]:
        """Identify types of structural changes."""
        changes = []

        # Check for new functions/classes
        for node in added_nodes:
            if node.node_type in ["FunctionDef", "AsyncFunctionDef"]:
                changes.append("NEW_FUNCTION_ADDED")
            elif node.node_type == "ClassDef":
                changes.append("NEW_CLASS_ADDED")
            elif node.node_type in ["Import", "ImportFrom"]:
                changes.append("NEW_IMPORT_ADDED")

        # Check for removed functions/classes
        for node in removed_nodes:
            if node.node_type in ["FunctionDef", "AsyncFunctionDef"]:
                changes.append("FUNCTION_REMOVED")
            elif node.node_type == "ClassDef":
                changes.append("CLASS_REMOVED")
            elif node.node_type in ["Import", "ImportFrom"]:
                changes.append("IMPORT_REMOVED")

        # Check for modifications
        for before, after in modified_nodes:
            if before.node_type in ["FunctionDef", "AsyncFunctionDef"]:
                changes.append("FUNCTION_MODIFIED")
            elif before.node_type == "ClassDef":
                changes.append("CLASS_MODIFIED")

        return list(set(changes))  # Remove duplicates

    def _detect_breaking_changes(
        self, ast_diff: ASTDiffResult, filename: str
    ) -> list[BreakingChange]:
        """Detect potential breaking changes from AST diff."""
        breaking_changes = []

        # Removed public functions/classes
        for node in ast_diff.removed_nodes:
            if self._is_public_symbol(node.name):
                if node.node_type in ["FunctionDef", "AsyncFunctionDef"]:
                    breaking_changes.append(
                        BreakingChange(
                            type="FUNCTION_REMOVED",
                            severity="HIGH",
                            description=f"Public function '{node.name}' was removed",
                            file_path=filename,
                            affected_symbol=node.name,
                            recommendation="Consider deprecating instead of removing, or provide migration path",
                            line_number=node.lineno,
                        )
                    )
                elif node.node_type == "ClassDef":
                    breaking_changes.append(
                        BreakingChange(
                            type="CLASS_REMOVED",
                            severity="HIGH",
                            description=f"Public class '{node.name}' was removed",
                            file_path=filename,
                            affected_symbol=node.name,
                            recommendation="Consider deprecating instead of removing, or provide migration path",
                            line_number=node.lineno,
                        )
                    )

        # Modified public function signatures
        for before, after in ast_diff.modified_nodes:
            if self._is_public_symbol(before.name) and before.node_type in [
                "FunctionDef",
                "AsyncFunctionDef",
            ]:
                if before.signature != after.signature:
                    breaking_changes.append(
                        BreakingChange(
                            type="FUNCTION_SIGNATURE_CHANGED",
                            severity="HIGH",
                            description=f"Public function '{before.name}' signature changed",
                            file_path=filename,
                            affected_symbol=before.name,
                            recommendation="Ensure signature changes are backward compatible or provide overloads",
                            line_number=after.lineno,
                        )
                    )

        return breaking_changes

    def _analyze_logic_safety(
        self, ast_diff: ASTDiffResult, file_change: FileChange
    ) -> list[ValidationIssue]:
        """Analyze changes for logic safety issues."""
        issues = []

        # Look for dangerous condition changes in the patch
        if file_change.patch:
            issues.extend(
                self._detect_condition_inversions(file_change.patch, file_change.filename)
            )

        # Look for error handling changes
        for before, after in ast_diff.modified_nodes:
            if before.node_type in ["Try", "ExceptHandler"]:
                issues.append(
                    ValidationIssue(
                        type="error_handling_change",
                        severity="medium",
                        message=f"Error handling logic modified in {before.name or 'unnamed block'}",
                        file=file_change.filename,
                        line=after.lineno,
                        rule="logic_safety_check",
                        suggestion="Review error handling changes carefully",
                    )
                )

        return issues

    def _detect_condition_inversions(self, patch: str, filename: str) -> list[ValidationIssue]:
        """Detect dangerous condition inversions in patch."""
        issues = []

        patch_lines = patch.split("\n")
        added_lines = [
            (i, line[1:])
            for i, line in enumerate(patch_lines)
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed_lines = [
            (i, line[1:])
            for i, line in enumerate(patch_lines)
            if line.startswith("-") and not line.startswith("---")
        ]

        # Look for logical operator inversions
        dangerous_patterns = [
            (r"\band\b", r"\bor\b"),
            (r"\bor\b", r"\band\b"),
            (r"==", r"!="),
            (r"!=", r"=="),
            (r"<", r">"),
            (r">", r"<"),
            (r"<=", r">="),
            (r">=", r"<="),
            (r"\bTrue\b", r"\bFalse\b"),
            (r"\bFalse\b", r"\bTrue\b"),
        ]

        for removed_idx, removed_line in removed_lines:
            for added_idx, added_line in added_lines:
                # Check if lines are similar except for logical operators
                for old_pattern, new_pattern in dangerous_patterns:
                    if re.search(old_pattern, removed_line) and re.search(new_pattern, added_line):
                        # Calculate similarity without the changed operator
                        removed_normalized = re.sub(old_pattern, "OP", removed_line)
                        added_normalized = re.sub(new_pattern, "OP", added_line)

                        similarity = difflib.SequenceMatcher(
                            None, removed_normalized, added_normalized
                        ).ratio()

                        if similarity > 0.8:  # Lines are very similar except for the operator
                            issues.append(
                                ValidationIssue(
                                    type="logic_inversion",
                                    severity="high",
                                    message=f"Potential dangerous logic inversion: '{old_pattern}' changed to '{new_pattern}'",
                                    file=filename,
                                    line=added_idx + 1,
                                    rule="logic_safety_inversion",
                                    suggestion=f"Verify that changing '{old_pattern}' to '{new_pattern}' is intentional and correct",
                                )
                            )

        return issues

    def _analyze_api_surface_changes(
        self, before_ast: ast.AST | None, after_ast: ast.AST | None
    ) -> dict[str, Any]:
        """Analyze changes to the public API surface."""
        changes: dict[str, list[str]] = {
            "functions_added": [],
            "functions_removed": [],
            "functions_modified": [],
            "classes_added": [],
            "classes_removed": [],
            "classes_modified": [],
        }

        if not before_ast and not after_ast:
            return changes

        before_api = (
            self._extract_api_surface(before_ast)
            if before_ast
            else {"functions": {}, "classes": {}}
        )
        after_api = (
            self._extract_api_surface(after_ast) if after_ast else {"functions": {}, "classes": {}}
        )

        # Analyze function changes
        before_funcs = set(before_api["functions"].keys())
        after_funcs = set(after_api["functions"].keys())

        changes["functions_added"] = list(after_funcs - before_funcs)
        changes["functions_removed"] = list(before_funcs - after_funcs)

        for func_name in before_funcs & after_funcs:
            if before_api["functions"][func_name] != after_api["functions"][func_name]:
                changes["functions_modified"].append(func_name)

        # Analyze class changes
        before_classes = set(before_api["classes"].keys())
        after_classes = set(after_api["classes"].keys())

        changes["classes_added"] = list(after_classes - before_classes)
        changes["classes_removed"] = list(before_classes - after_classes)

        for class_name in before_classes & after_classes:
            if before_api["classes"][class_name] != after_api["classes"][class_name]:
                changes["classes_modified"].append(class_name)

        return changes

    def _extract_api_surface(self, tree: ast.AST) -> dict[str, dict]:
        """Extract the public API surface from an AST."""
        functions = {}
        classes = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_public_symbol(node.name):
                    functions[node.name] = self._extract_function_signature(node)

            elif isinstance(node, ast.ClassDef):
                if self._is_public_symbol(node.name):
                    class_info: dict[str, Any] = {
                        "bases": [self._ast_to_string(base) for base in node.bases],
                        "methods": {},
                    }

                    # Extract public methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if self._is_public_symbol(item.name):
                                class_info["methods"][item.name] = self._extract_function_signature(
                                    item
                                )

                    classes[node.name] = class_info

        return {"functions": functions, "classes": classes}

    def _is_public_symbol(self, name: str) -> bool:
        """Check if a symbol is part of the public API."""
        if not name:
            return False

        # Python convention: names starting with _ are private
        if name.startswith("_") and not name.startswith("__"):
            return False

        # Special methods are considered public
        if name.startswith("__") and name.endswith("__"):
            return True

        return True

    def _calculate_complexity_changes(
        self, before_ast: ast.AST | None, after_ast: ast.AST | None
    ) -> dict[str, float]:
        """Calculate complexity changes."""
        before_complexity = self._calculate_ast_complexity(before_ast) if before_ast else 0.0
        after_complexity = self._calculate_ast_complexity(after_ast) if after_ast else 0.0

        return {
            "before": before_complexity,
            "after": after_complexity,
            "change": after_complexity - before_complexity,
            "relative_change": (after_complexity - before_complexity) / max(before_complexity, 1.0),
        }

    def _create_parse_failure_result(self, filename: str) -> SemanticAnalysisResult:
        """Create result for parsing failures."""
        return SemanticAnalysisResult(
            breaking_changes=[
                BreakingChange(
                    type="PARSE_FAILURE",
                    severity="MEDIUM",
                    description="Could not parse file for semantic analysis",
                    file_path=filename,
                    affected_symbol="",
                    recommendation="Check file syntax and encoding",
                )
            ],
            logic_safety_issues=[],
            api_surface_changes={},
            complexity_changes={},
        )
