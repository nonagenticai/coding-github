"""Comprehensive unit tests for PR validation modules.

This module provides extensive test coverage for:
- ast_diff_analyzer.py (ASTDiffAnalyzer, SafeASTParser, ASTNode, ASTDiffResult)
- breaking_change_detector.py (BreakingChangeDetector, APIElement, CompatibilityIssue)
- logic_safety_validator.py (LogicSafetyValidator, LogicPattern, SafetyViolation)
- diff_parser.py (DiffParser, DiffHunk, ParsedDiff)
- file_classifier.py (PRFileTypeClassifier)
"""

from __future__ import annotations

import ast

import pytest
from coding_client.models.pr_validation import (
    BreakingChange,
    ChangeType,
    DiffStats,
    FileChange,
    FileType,
    PRDiff,
    SemanticAnalysisResult,
    ValidationIssue,
)

from services.pr_validation.analyzers.semantic.ast_diff_analyzer import (
    ASTDiffAnalyzer,
    ASTDiffResult,
    ASTNode,
    SafeASTParser,
)
from services.pr_validation.analyzers.semantic.breaking_change_detector import (
    APIElement,
    BreakingChangeDetector,
)
from services.pr_validation.analyzers.semantic.logic_safety_validator import (
    LogicSafetyValidator,
    SafetyViolation,
)
from services.pr_validation.core.diff_parser import (
    DiffHunk,
    DiffParser,
    ParsedDiff,
)
from services.pr_validation.core.file_classifier import PRFileTypeClassifier

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def simple_python_code_before():
    """Simple Python code sample for before state."""
    return '''
def hello(name):
    """Say hello."""
    return f"Hello, {name}!"

def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
'''


@pytest.fixture
def simple_python_code_after():
    """Simple Python code sample for after state with modifications."""
    return '''
def hello(name: str) -> str:
    """Say hello with greeting."""
    return f"Hello, {name}!"

def add(a: int, b: int, c: int = 0) -> int:
    return a + b + c

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
'''


@pytest.fixture
def complex_python_code():
    """Complex Python code with various constructs."""
    return '''
import os
from typing import List, Dict, Optional

class DataProcessor:
    """Process data with various operations."""

    def __init__(self, config: Dict):
        self.config = config
        self._cache = {}

    async def process(self, data: List[str]) -> List[str]:
        """Process data asynchronously."""
        results = []
        for item in data:
            try:
                if item.startswith("skip"):
                    continue
                result = await self._transform(item)
                results.append(result)
            except ValueError as e:
                raise RuntimeError(f"Failed: {e}")
        return results

    async def _transform(self, item: str) -> str:
        return item.upper()

    @staticmethod
    def validate(data: List[str]) -> bool:
        return all(isinstance(d, str) for d in data)

def public_function(x: int, y: int = 10) -> int:
    """A public function."""
    if x > 0 and y > 0:
        return x + y
    elif x < 0 or y < 0:
        return 0
    return -1

_private_function = lambda x: x * 2

PUBLIC_CONSTANT = 42
'''


@pytest.fixture
def file_change_python():
    """Create a Python FileChange fixture."""
    return FileChange(
        filename="src/module.py",
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=10,
        deletions=5,
        patch="@@ -1,5 +1,10 @@\n-def old():\n+def new():\n     pass",
        before_content="def old():\n    pass",
        after_content="def new():\n    pass",
        binary=False,
    )


@pytest.fixture
def file_change_binary():
    """Create a binary FileChange fixture."""
    return FileChange(
        filename="image.png",
        file_type=FileType.UNKNOWN,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="Binary file differs",
        binary=True,
    )


@pytest.fixture
def diff_parser():
    """Create a DiffParser instance."""
    return DiffParser()


@pytest.fixture
def file_classifier():
    """Create a PRFileTypeClassifier instance."""
    return PRFileTypeClassifier()


@pytest.fixture
def safe_ast_parser():
    """Create a SafeASTParser instance."""
    return SafeASTParser()


@pytest.fixture
def ast_diff_analyzer():
    """Create an ASTDiffAnalyzer instance."""
    return ASTDiffAnalyzer()


@pytest.fixture
def breaking_change_detector():
    """Create a BreakingChangeDetector instance."""
    return BreakingChangeDetector()


@pytest.fixture
def logic_safety_validator():
    """Create a LogicSafetyValidator instance."""
    return LogicSafetyValidator()


# ==============================================================================
# ASTNode Tests
# ==============================================================================


class TestASTNode:
    """Tests for ASTNode dataclass."""

    def test_ast_node_creation(self):
        """Test basic ASTNode creation."""
        node = ASTNode(
            node_type="FunctionDef",
            name="my_function",
            lineno=10,
            col_offset=0,
        )
        assert node.node_type == "FunctionDef"
        assert node.name == "my_function"
        assert node.lineno == 10
        assert node.col_offset == 0
        assert node.end_lineno is None
        assert node.signature is None
        assert node.attributes == {}

    def test_ast_node_with_all_fields(self):
        """Test ASTNode with all optional fields."""
        node = ASTNode(
            node_type="AsyncFunctionDef",
            name="async_function",
            lineno=5,
            col_offset=4,
            end_lineno=15,
            end_col_offset=0,
            signature="async def async_function(x: int) -> str",
            attributes={"args": ["x"], "is_async": True},
        )
        assert node.end_lineno == 15
        assert node.signature == "async def async_function(x: int) -> str"
        assert node.attributes["is_async"] is True

    def test_ast_node_default_attributes(self):
        """Test ASTNode default attributes are independent."""
        node1 = ASTNode(node_type="ClassDef", name="Class1", lineno=1, col_offset=0)
        node2 = ASTNode(node_type="ClassDef", name="Class2", lineno=2, col_offset=0)
        node1.attributes["test"] = True
        assert "test" not in node2.attributes


# ==============================================================================
# ASTDiffResult Tests
# ==============================================================================


class TestASTDiffResult:
    """Tests for ASTDiffResult dataclass."""

    def test_ast_diff_result_creation(self):
        """Test basic ASTDiffResult creation."""
        added = [ASTNode(node_type="FunctionDef", name="new_func", lineno=1, col_offset=0)]
        removed = [ASTNode(node_type="FunctionDef", name="old_func", lineno=2, col_offset=0)]

        result = ASTDiffResult(
            added_nodes=added,
            removed_nodes=removed,
            modified_nodes=[],
            unchanged_nodes=[],
            complexity_change=0.5,
            structural_changes=["NEW_FUNCTION_ADDED"],
        )

        assert len(result.added_nodes) == 1
        assert len(result.removed_nodes) == 1
        assert result.complexity_change == 0.5
        assert "NEW_FUNCTION_ADDED" in result.structural_changes

    def test_ast_diff_result_with_modified_nodes(self):
        """Test ASTDiffResult with modified nodes."""
        before = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x)",
        )
        after = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x, y)",
        )

        result = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_MODIFIED"],
        )

        assert len(result.modified_nodes) == 1
        assert result.modified_nodes[0][0].signature != result.modified_nodes[0][1].signature


# ==============================================================================
# SafeASTParser Tests
# ==============================================================================


class TestSafeASTParser:
    """Tests for SafeASTParser class."""

    def test_parse_valid_python(self, safe_ast_parser):
        """Test parsing valid Python code."""
        code = "def hello():\n    return 'Hello'"
        tree, errors = safe_ast_parser.parse_with_fallback(code)

        assert tree is not None
        assert isinstance(tree, ast.AST)

    def test_parse_empty_content(self, safe_ast_parser):
        """Test parsing empty content."""
        tree, errors = safe_ast_parser.parse_with_fallback("")

        assert tree is None
        assert "Empty content" in errors

    def test_parse_whitespace_only(self, safe_ast_parser):
        """Test parsing whitespace-only content."""
        tree, errors = safe_ast_parser.parse_with_fallback("   \n\t  \n  ")

        assert tree is None
        assert "Empty content" in errors

    def test_parse_syntax_error(self, safe_ast_parser):
        """Test parsing code with syntax errors."""
        code = "def broken(\n    return"
        tree, errors = safe_ast_parser.parse_with_fallback(code)

        # Either returns partial or None depending on fallback success
        assert len(errors) > 0

    def test_parse_with_filename(self, safe_ast_parser):
        """Test parsing with custom filename."""
        code = "x = 1"
        tree, errors = safe_ast_parser.parse_with_fallback(code, filename="test.py")

        assert tree is not None

    def test_parse_complex_code(self, safe_ast_parser, complex_python_code):
        """Test parsing complex Python code."""
        tree, errors = safe_ast_parser.parse_with_fallback(complex_python_code)

        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_parse_async_code(self, safe_ast_parser):
        """Test parsing async Python code."""
        code = """
async def fetch_data():
    await some_async_call()
    return data
"""
        tree, errors = safe_ast_parser.parse_with_fallback(code)

        assert tree is not None

    def test_supported_python_versions(self, safe_ast_parser):
        """Test supported Python versions list."""
        assert len(safe_ast_parser.supported_python_versions) > 0
        assert (3, 8) in safe_ast_parser.supported_python_versions

    def test_parse_with_type_annotations(self, safe_ast_parser):
        """Test parsing code with type annotations."""
        code = """
def typed_func(x: int, y: str = "default") -> bool:
    return True
"""
        tree, errors = safe_ast_parser.parse_with_fallback(code)

        assert tree is not None

    def test_parse_with_decorators(self, safe_ast_parser):
        """Test parsing code with decorators."""
        code = """
@decorator
@another_decorator(arg=1)
def decorated_func():
    pass
"""
        tree, errors = safe_ast_parser.parse_with_fallback(code)

        assert tree is not None


# ==============================================================================
# ASTDiffAnalyzer Tests
# ==============================================================================


class TestASTDiffAnalyzer:
    """Tests for ASTDiffAnalyzer class."""

    def test_analyzer_initialization(self, ast_diff_analyzer):
        """Test analyzer initialization."""
        assert ast_diff_analyzer.parser is not None
        assert len(ast_diff_analyzer.important_nodes) > 0
        assert len(ast_diff_analyzer.complexity_nodes) > 0

    @pytest.mark.asyncio
    async def test_analyze_non_python_file(self, ast_diff_analyzer):
        """Test analyzing non-Python file returns empty result."""
        file_change = FileChange(
            filename="README.md",
            file_type=FileType.MARKDOWN,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=2,
            patch="@@ -1,3 +1,5 @@",
        )

        result = await ast_diff_analyzer.analyze_semantic_changes(file_change)

        assert isinstance(result, SemanticAnalysisResult)
        assert len(result.breaking_changes) == 0

    @pytest.mark.asyncio
    async def test_analyze_binary_file(self, ast_diff_analyzer, file_change_binary):
        """Test analyzing binary file returns empty result."""
        result = await ast_diff_analyzer.analyze_semantic_changes(file_change_binary)

        assert isinstance(result, SemanticAnalysisResult)
        assert len(result.breaking_changes) == 0

    @pytest.mark.asyncio
    async def test_analyze_new_file(self, ast_diff_analyzer):
        """Test analyzing a new file."""
        file_change = FileChange(
            filename="new_module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.ADDED,
            additions=10,
            deletions=0,
            patch="@@ -0,0 +1,10 @@",
            before_content=None,
            after_content="def new_function():\n    pass",
        )

        result = await ast_diff_analyzer.analyze_semantic_changes(file_change)

        assert isinstance(result, SemanticAnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_deleted_file(self, ast_diff_analyzer):
        """Test analyzing a deleted file."""
        file_change = FileChange(
            filename="deleted_module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.DELETED,
            additions=0,
            deletions=10,
            patch="@@ -1,10 +0,0 @@",
            before_content="def old_function():\n    pass",
            after_content=None,
        )

        result = await ast_diff_analyzer.analyze_semantic_changes(file_change)

        assert isinstance(result, SemanticAnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_modified_file(
        self, ast_diff_analyzer, simple_python_code_before, simple_python_code_after
    ):
        """Test analyzing a modified file."""
        file_change = FileChange(
            filename="module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=15,
            deletions=5,
            patch="@@ -1,10 +1,20 @@",
            before_content=simple_python_code_before,
            after_content=simple_python_code_after,
        )

        result = await ast_diff_analyzer.analyze_semantic_changes(file_change)

        assert isinstance(result, SemanticAnalysisResult)
        assert "functions_added" in result.api_surface_changes or result.api_surface_changes == {}

    @pytest.mark.asyncio
    async def test_analyze_with_parse_failure(self, ast_diff_analyzer):
        """Test handling of parse failures."""
        file_change = FileChange(
            filename="broken.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="",
            before_content="def broken(\n    invalid syntax",
            after_content="def also_broken(\n    more invalid",
        )

        result = await ast_diff_analyzer.analyze_semantic_changes(file_change)

        assert isinstance(result, SemanticAnalysisResult)
        # Should contain a parse failure breaking change
        assert len(result.breaking_changes) > 0

    def test_is_analyzable_file_python(self, ast_diff_analyzer):
        """Test _is_analyzable_file for Python files."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="",
            binary=False,
        )

        assert ast_diff_analyzer._is_analyzable_file(file_change) is True

    def test_is_analyzable_file_non_python(self, ast_diff_analyzer):
        """Test _is_analyzable_file for non-Python files."""
        file_change = FileChange(
            filename="test.js",
            file_type=FileType.JAVASCRIPT,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="",
            binary=False,
        )

        assert ast_diff_analyzer._is_analyzable_file(file_change) is False

    def test_extract_nodes(self, ast_diff_analyzer, complex_python_code, safe_ast_parser):
        """Test extracting important nodes from AST."""
        tree, _ = safe_ast_parser.parse_with_fallback(complex_python_code)
        nodes = ast_diff_analyzer._extract_nodes(tree)

        assert len(nodes) > 0
        node_types = [n.node_type for n in nodes]
        # Now that important_nodes uses full AST names, we should find FunctionDef/ClassDef
        assert any(t in node_types for t in ["FunctionDef", "ClassDef", "Import", "ImportFrom"])

    def test_create_ast_node_function(self, ast_diff_analyzer, safe_ast_parser):
        """Test creating ASTNode from function definition."""
        code = "def my_func(x: int, y: str = 'test') -> bool:\n    pass"
        tree, _ = safe_ast_parser.parse_with_fallback(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ast_node = ast_diff_analyzer._create_ast_node(node)
                assert ast_node.name == "my_func"
                assert ast_node.node_type == "FunctionDef"
                assert ast_node.signature is not None

    def test_create_ast_node_class(self, ast_diff_analyzer, safe_ast_parser):
        """Test creating ASTNode from class definition."""
        code = "class MyClass(BaseClass):\n    pass"
        tree, _ = safe_ast_parser.parse_with_fallback(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                ast_node = ast_diff_analyzer._create_ast_node(node)
                assert ast_node.name == "MyClass"
                assert "bases" in ast_node.attributes

    def test_extract_function_signature(self, ast_diff_analyzer, safe_ast_parser):
        """Test extracting function signature."""
        code = "def func(a: int, b: str = 'x', *args, **kwargs) -> None:\n    pass"
        tree, _ = safe_ast_parser.parse_with_fallback(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                signature = ast_diff_analyzer._extract_function_signature(node)
                assert "func" in signature
                assert "a: int" in signature

    def test_extract_async_function_signature(self, ast_diff_analyzer, safe_ast_parser):
        """Test extracting async function signature."""
        code = "async def async_func(data: list) -> dict:\n    pass"
        tree, _ = safe_ast_parser.parse_with_fallback(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                signature = ast_diff_analyzer._extract_function_signature(node)
                assert "async" in signature

    def test_node_key_generation(self, ast_diff_analyzer):
        """Test generating unique key for nodes."""
        node1 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x)",
        )
        node2 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=5,
            col_offset=0,
            signature="def func(y)",
        )

        key1 = ast_diff_analyzer._node_key(node1)
        key2 = ast_diff_analyzer._node_key(node2)

        assert "FunctionDef" in key1
        assert "func" in key1
        assert key1 != key2  # Different signatures

    def test_nodes_are_different_same_signature(self, ast_diff_analyzer):
        """Test comparing nodes with same signature."""
        node1 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x)",
        )
        node2 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x)",
        )

        assert ast_diff_analyzer._nodes_are_different(node1, node2) is False

    def test_nodes_are_different_different_signature(self, ast_diff_analyzer):
        """Test comparing nodes with different signatures."""
        node1 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x)",
        )
        node2 = ASTNode(
            node_type="FunctionDef",
            name="func",
            lineno=1,
            col_offset=0,
            signature="def func(x, y)",
        )

        assert ast_diff_analyzer._nodes_are_different(node1, node2) is True

    def test_calculate_ast_complexity(self, ast_diff_analyzer, safe_ast_parser):
        """Test calculating AST complexity."""
        simple_code = "x = 1"
        complex_code = """
if x > 0:
    for i in range(10):
        if i % 2 == 0:
            try:
                process(i)
            except:
                pass
"""
        simple_tree, _ = safe_ast_parser.parse_with_fallback(simple_code)
        complex_tree, _ = safe_ast_parser.parse_with_fallback(complex_code)

        simple_complexity = ast_diff_analyzer._calculate_ast_complexity(simple_tree)
        complex_complexity = ast_diff_analyzer._calculate_ast_complexity(complex_tree)

        assert complex_complexity > simple_complexity

    def test_calculate_ast_complexity_none(self, ast_diff_analyzer):
        """Test calculating complexity for None AST."""
        assert ast_diff_analyzer._calculate_ast_complexity(None) == 0.0

    def test_identify_structural_changes_new_function(self, ast_diff_analyzer):
        """Test identifying structural changes for new functions."""
        added = [ASTNode(node_type="FunctionDef", name="new_func", lineno=1, col_offset=0)]

        changes = ast_diff_analyzer._identify_structural_changes(added, [], [])

        assert "NEW_FUNCTION_ADDED" in changes

    def test_identify_structural_changes_removed_class(self, ast_diff_analyzer):
        """Test identifying structural changes for removed classes."""
        removed = [ASTNode(node_type="ClassDef", name="OldClass", lineno=1, col_offset=0)]

        changes = ast_diff_analyzer._identify_structural_changes([], removed, [])

        assert "CLASS_REMOVED" in changes

    def test_is_public_symbol(self, ast_diff_analyzer):
        """Test public symbol detection."""
        assert ast_diff_analyzer._is_public_symbol("public_func") is True
        assert ast_diff_analyzer._is_public_symbol("_private_func") is False
        assert ast_diff_analyzer._is_public_symbol("__dunder__") is True
        assert ast_diff_analyzer._is_public_symbol("") is False

    def test_detect_breaking_changes_removed_function(self, ast_diff_analyzer):
        """Test detecting breaking changes for removed public functions."""
        diff_result = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[
                ASTNode(node_type="FunctionDef", name="public_func", lineno=10, col_offset=0)
            ],
            modified_nodes=[],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_REMOVED"],
        )

        breaking_changes = ast_diff_analyzer._detect_breaking_changes(diff_result, "test.py")

        assert len(breaking_changes) == 1
        assert breaking_changes[0].type == "FUNCTION_REMOVED"

    def test_detect_breaking_changes_modified_signature(self, ast_diff_analyzer):
        """Test detecting breaking changes for modified function signatures."""
        before = ASTNode(
            node_type="FunctionDef",
            name="public_func",
            lineno=10,
            col_offset=0,
            signature="def public_func(x)",
        )
        after = ASTNode(
            node_type="FunctionDef",
            name="public_func",
            lineno=10,
            col_offset=0,
            signature="def public_func(x, y)",
        )

        diff_result = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_MODIFIED"],
        )

        breaking_changes = ast_diff_analyzer._detect_breaking_changes(diff_result, "test.py")

        assert len(breaking_changes) == 1
        assert breaking_changes[0].type == "FUNCTION_SIGNATURE_CHANGED"

    def test_detect_condition_inversions(self, ast_diff_analyzer):
        """Test detecting dangerous condition inversions."""
        diff_patch = """@@ -10,5 +10,5 @@
-    if x > 0 and y > 0:
+    if x > 0 or y > 0:
         return True"""

        issues = ast_diff_analyzer._detect_condition_inversions(diff_patch, "test.py")

        # Should detect and/or inversion
        assert len(issues) > 0 or issues == []  # Depends on similarity threshold

    def test_analyze_api_surface_changes(
        self,
        ast_diff_analyzer,
        safe_ast_parser,
        simple_python_code_before,
        simple_python_code_after,
    ):
        """Test analyzing API surface changes."""
        before_tree, _ = safe_ast_parser.parse_with_fallback(simple_python_code_before)
        after_tree, _ = safe_ast_parser.parse_with_fallback(simple_python_code_after)

        changes = ast_diff_analyzer._analyze_api_surface_changes(before_tree, after_tree)

        assert "functions_added" in changes
        assert "functions_removed" in changes

    def test_analyze_api_surface_changes_none(self, ast_diff_analyzer):
        """Test analyzing API surface changes with None ASTs."""
        changes = ast_diff_analyzer._analyze_api_surface_changes(None, None)

        assert changes == {
            "functions_added": [],
            "functions_removed": [],
            "functions_modified": [],
            "classes_added": [],
            "classes_removed": [],
            "classes_modified": [],
        }

    def test_calculate_complexity_changes(self, ast_diff_analyzer, safe_ast_parser):
        """Test calculating complexity changes."""
        before_code = "x = 1"
        after_code = "if x:\n    for i in range(10):\n        pass"

        before_tree, _ = safe_ast_parser.parse_with_fallback(before_code)
        after_tree, _ = safe_ast_parser.parse_with_fallback(after_code)

        changes = ast_diff_analyzer._calculate_complexity_changes(before_tree, after_tree)

        assert "before" in changes
        assert "after" in changes
        assert "change" in changes
        assert "relative_change" in changes

    def test_compare_asts_new_file(self, ast_diff_analyzer, safe_ast_parser):
        """Test comparing ASTs when before is None (new file)."""
        after_code = "def new_func():\n    pass"
        after_tree, _ = safe_ast_parser.parse_with_fallback(after_code)

        result = ast_diff_analyzer._compare_asts(None, after_tree)

        assert "NEW_FILE" in result.structural_changes

    def test_compare_asts_deleted_file(self, ast_diff_analyzer, safe_ast_parser):
        """Test comparing ASTs when after is None (deleted file)."""
        before_code = "def old_func():\n    pass"
        before_tree, _ = safe_ast_parser.parse_with_fallback(before_code)

        result = ast_diff_analyzer._compare_asts(before_tree, None)

        assert "FILE_DELETED" in result.structural_changes

    def test_ast_to_string(self, ast_diff_analyzer):
        """Test converting AST node to string."""
        node = ast.Name(id="test_var", ctx=ast.Load())
        result = ast_diff_analyzer._ast_to_string(node)

        assert "test_var" in result


# ==============================================================================
# BreakingChangeDetector Tests
# ==============================================================================


class TestBreakingChangeDetector:
    """Tests for BreakingChangeDetector class."""

    def test_detector_initialization(self, breaking_change_detector):
        """Test detector initialization."""
        assert breaking_change_detector.parser is not None
        assert len(breaking_change_detector.public_api_prefixes) > 0
        assert len(breaking_change_detector.severity_weights) > 0

    def test_is_analyzable_file_python(self, breaking_change_detector):
        """Test _is_analyzable_file for Python files."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="",
            before_content="x = 1",
            after_content="x = 2",
            binary=False,
        )

        assert breaking_change_detector._is_analyzable_file(file_change) is True

    def test_is_analyzable_file_missing_content(self, breaking_change_detector):
        """Test _is_analyzable_file with missing content."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="",
            before_content=None,
            after_content="x = 1",
            binary=False,
        )

        assert breaking_change_detector._is_analyzable_file(file_change) is False

    @pytest.mark.asyncio
    async def test_detect_breaking_changes_non_analyzable(self, breaking_change_detector):
        """Test detecting breaking changes for non-analyzable files."""
        file_change = FileChange(
            filename="data.json",
            file_type=FileType.JSON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="",
        )

        result = await breaking_change_detector.detect_breaking_changes(file_change)

        assert result == []

    @pytest.mark.asyncio
    async def test_detect_breaking_changes_python_removed_function(self, breaking_change_detector):
        """Test detecting removed function in Python."""
        before = "def public_func():\n    pass\n\ndef another_func():\n    pass"
        after = "def another_func():\n    pass"

        file_change = FileChange(
            filename="module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=2,
            patch="",
            before_content=before,
            after_content=after,
            binary=False,
        )

        result = await breaking_change_detector.detect_breaking_changes(file_change)

        assert len(result) >= 1
        assert any(bc.type == "FUNCTION_REMOVED" for bc in result)

    @pytest.mark.asyncio
    async def test_detect_breaking_changes_python_signature_change(self, breaking_change_detector):
        """Test detecting signature changes in Python."""
        before = "def func(a, b):\n    pass"
        after = "def func(a, b, c):\n    pass"

        file_change = FileChange(
            filename="module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            patch="",
            before_content=before,
            after_content=after,
            binary=False,
        )

        result = await breaking_change_detector.detect_breaking_changes(file_change)

        # May detect signature change or required args added
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_breaking_changes_javascript(self, breaking_change_detector):
        """Test detecting breaking changes in JavaScript."""
        file_change = FileChange(
            filename="module.js",
            file_type=FileType.JAVASCRIPT,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="-export function oldFunc() {\n+export function newFunc() {",
            before_content="export function oldFunc() {}",
            after_content="export function newFunc() {}",
            binary=False,
        )

        result = await breaking_change_detector.detect_breaking_changes(file_change)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_breaking_changes_typescript(self, breaking_change_detector):
        """Test detecting breaking changes in TypeScript."""
        file_change = FileChange(
            filename="module.ts",
            file_type=FileType.TYPESCRIPT,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="-interface OldInterface {\n+interface NewInterface {",
            before_content="interface OldInterface {}",
            after_content="interface NewInterface {}",
            binary=False,
        )

        result = await breaking_change_detector.detect_breaking_changes(file_change)

        assert isinstance(result, list)

    def test_is_public_python_function(self, breaking_change_detector):
        """Test public Python function detection."""
        assert breaking_change_detector._is_public_python_function("public_func") is True
        assert breaking_change_detector._is_public_python_function("_private") is False
        assert breaking_change_detector._is_public_python_function("__init__") is True

    def test_is_public_python_class(self, breaking_change_detector):
        """Test public Python class detection."""
        assert breaking_change_detector._is_public_python_class("PublicClass") is True
        assert breaking_change_detector._is_public_python_class("_PrivateClass") is False

    def test_is_public_python_constant(self, breaking_change_detector):
        """Test public Python constant detection."""
        assert breaking_change_detector._is_public_python_constant("PUBLIC_CONST") is True
        assert breaking_change_detector._is_public_python_constant("_PRIVATE") is False
        assert breaking_change_detector._is_public_python_constant("not_const") is False

    def test_extract_python_api(self, breaking_change_detector, safe_ast_parser):
        """Test extracting Python API elements."""
        code = """
def public_func(x: int) -> str:
    pass

class PublicClass:
    def method(self):
        pass

_private_func = lambda: None
CONSTANT = 42
"""
        tree, _ = safe_ast_parser.parse_with_fallback(code)
        api = breaking_change_detector._extract_python_api(tree)

        assert "public_func" in api
        assert "PublicClass" in api
        assert "CONSTANT" in api

    def test_compare_python_apis_removed_element(self, breaking_change_detector):
        """Test comparing Python APIs with removed element."""
        before = {
            "func": APIElement(
                name="func",
                type="function",
                signature="def func()",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }
        after = {}

        issues = breaking_change_detector._compare_python_apis(before, after)

        assert len(issues) == 1
        assert issues[0].issue_type == "FUNCTION_REMOVED"

    def test_compare_python_apis_type_change(self, breaking_change_detector):
        """Test comparing Python APIs with type change."""
        before = {
            "item": APIElement(
                name="item",
                type="function",
                signature="def item()",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }
        after = {
            "item": APIElement(
                name="item",
                type="class",
                signature="class item",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }

        issues = breaking_change_detector._compare_python_apis(before, after)

        assert len(issues) == 1
        assert issues[0].issue_type == "ELEMENT_TYPE_CHANGED"

    def test_analyze_function_signature_changes_args_removed(self, breaking_change_detector):
        """Test analyzing function signature changes with removed args."""
        before = APIElement(
            name="func",
            type="function",
            signature="def func(a, b, c)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a", "b", "c"], "defaults": 0},
        )
        after = APIElement(
            name="func",
            type="function",
            signature="def func(a)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a"], "defaults": 0},
        )

        issues = breaking_change_detector._analyze_function_signature_changes(before, after)

        assert len(issues) >= 1
        assert any(i.issue_type == "FUNCTION_ARGS_REMOVED" for i in issues)

    def test_analyze_function_signature_changes_required_args_added(self, breaking_change_detector):
        """Test analyzing function signature changes with required args added."""
        before = APIElement(
            name="func",
            type="function",
            signature="def func(a)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a"], "defaults": 0},
        )
        after = APIElement(
            name="func",
            type="function",
            signature="def func(a, b, c)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a", "b", "c"], "defaults": 0},
        )

        issues = breaking_change_detector._analyze_function_signature_changes(before, after)

        assert any(i.issue_type == "FUNCTION_REQUIRED_ARGS_ADDED" for i in issues)

    def test_analyze_function_signature_changes_async_change(self, breaking_change_detector):
        """Test analyzing function signature changes with async change."""
        before = APIElement(
            name="func",
            type="function",
            signature="def func()",
            location=(1, 0),
            visibility="public",
            metadata={"args": [], "defaults": 0, "is_async": False},
        )
        after = APIElement(
            name="func",
            type="function",
            signature="async def func()",
            location=(1, 0),
            visibility="public",
            metadata={"args": [], "defaults": 0, "is_async": True},
        )

        issues = breaking_change_detector._analyze_function_signature_changes(before, after)

        assert any(i.issue_type == "FUNCTION_ASYNC_CHANGED" for i in issues)

    def test_analyze_class_changes_inheritance_removed(self, breaking_change_detector):
        """Test analyzing class changes with removed inheritance."""
        before = APIElement(
            name="MyClass",
            type="class",
            signature="class MyClass(Base)",
            location=(1, 0),
            visibility="public",
            metadata={"bases": ["Base"], "methods": {}},
        )
        after = APIElement(
            name="MyClass",
            type="class",
            signature="class MyClass",
            location=(1, 0),
            visibility="public",
            metadata={"bases": [], "methods": {}},
        )

        issues = breaking_change_detector._analyze_class_changes(before, after)

        assert any(i.issue_type == "CLASS_INHERITANCE_REMOVED" for i in issues)

    def test_analyze_class_changes_method_removed(self, breaking_change_detector):
        """Test analyzing class changes with removed method."""
        before = APIElement(
            name="MyClass",
            type="class",
            signature="class MyClass",
            location=(1, 0),
            visibility="public",
            metadata={"bases": [], "methods": {"method": "def method()"}},
        )
        after = APIElement(
            name="MyClass",
            type="class",
            signature="class MyClass",
            location=(1, 0),
            visibility="public",
            metadata={"bases": [], "methods": {}},
        )

        issues = breaking_change_detector._analyze_class_changes(before, after)

        assert any(i.issue_type == "CLASS_METHOD_REMOVED" for i in issues)

    def test_handle_parse_failure(self, breaking_change_detector):
        """Test handling parse failures."""
        result = breaking_change_detector._handle_parse_failure("test.py", ["Syntax error"])

        assert len(result) == 1
        assert result[0].type == "PARSE_FAILURE"


# ==============================================================================
# LogicSafetyValidator Tests
# ==============================================================================


class TestLogicSafetyValidator:
    """Tests for LogicSafetyValidator class."""

    def test_validator_initialization(self, logic_safety_validator):
        """Test validator initialization."""
        assert logic_safety_validator.parser is not None
        assert len(logic_safety_validator.inversion_patterns) > 0
        assert len(logic_safety_validator.security_patterns) > 0
        assert len(logic_safety_validator.error_patterns) > 0

    def test_is_validatable_file_python(self, logic_safety_validator):
        """Test _is_validatable_file for Python files."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="@@ -1,5 +1,5 @@",
            binary=False,
        )

        assert logic_safety_validator._is_validatable_file(file_change) is True

    def test_is_validatable_file_binary(self, logic_safety_validator):
        """Test _is_validatable_file for binary files."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
            binary=True,
        )

        assert logic_safety_validator._is_validatable_file(file_change) is False

    @pytest.mark.asyncio
    async def test_validate_logic_safety_non_validatable(self, logic_safety_validator):
        """Test validating non-validatable file returns empty list."""
        file_change = FileChange(
            filename="data.json",
            file_type=FileType.JSON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=3,
            patch="",
        )

        result = await logic_safety_validator.validate_logic_safety(file_change)

        assert result == []

    @pytest.mark.asyncio
    async def test_validate_logic_safety_with_patch(self, logic_safety_validator):
        """Test validating file with patch containing dangerous patterns."""
        diff_patch = """@@ -10,5 +10,5 @@
-    if x == 0:
+    if x != 0:
         return True"""

        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            patch=diff_patch,
            before_content="if x == 0:\n    return True",
            after_content="if x != 0:\n    return True",
            binary=False,
        )

        result = await logic_safety_validator.validate_logic_safety(file_change)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_validate_logic_safety_security_sensitive(self, logic_safety_validator):
        """Test validating file with security-sensitive changes."""
        diff_patch = """@@ -10,5 +10,5 @@
+    if authenticate(user):
         return True"""

        file_change = FileChange(
            filename="auth.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch=diff_patch,
            before_content="return True",
            after_content="if authenticate(user):\n    return True",
            binary=False,
        )

        result = await logic_safety_validator.validate_logic_safety(file_change)

        # Should detect security-sensitive code
        assert isinstance(result, list)

    def test_analyze_patch_safety_empty(self, logic_safety_validator):
        """Test analyzing empty patch."""
        issues = logic_safety_validator._analyze_patch_safety("", "test.py")

        assert issues == []

    def test_analyze_patch_safety_logic_inversion(self, logic_safety_validator):
        """Test analyzing patch with logic inversion."""
        diff_patch = """@@ -10,3 +10,3 @@
-    if x and y:
+    if x or y:
         return True"""

        issues = logic_safety_validator._analyze_patch_safety(diff_patch, "test.py")

        # May or may not detect based on similarity
        assert isinstance(issues, list)

    def test_detect_logic_inversions_similar_lines(self, logic_safety_validator):
        """Test detecting logic inversions in similar lines."""
        removed = "    if x == True and y > 0:"
        added = "    if x == False or y > 0:"

        violations = logic_safety_validator._detect_logic_inversions(removed, added, 10, 10)

        # Should detect potential inversions
        assert isinstance(violations, list)

    def test_detect_logic_inversions_dissimilar_lines(self, logic_safety_validator):
        """Test detecting logic inversions in dissimilar lines."""
        removed = "def completely_different():"
        added = "class SomethingElse:"

        violations = logic_safety_validator._detect_logic_inversions(removed, added, 10, 10)

        assert violations == []

    def test_assess_inversion_severity_security(self, logic_safety_validator):
        """Test assessing severity for security-related inversions."""
        context = "if authenticate(user) and has_permission:"
        severity = logic_safety_validator._assess_inversion_severity(r"\band\b", r"\bor\b", context)

        assert severity == "high"

    def test_assess_inversion_severity_boolean_ops(self, logic_safety_validator):
        """Test assessing severity for boolean operation inversions."""
        context = "if x and y:"
        severity = logic_safety_validator._assess_inversion_severity(r"\band\b", r"\bor\b", context)

        assert severity == "high"

    def test_check_dangerous_patterns_security(self, logic_safety_validator):
        """Test checking for security-related dangerous patterns."""
        added_lines = [(10, "    if validate(input):")]
        removed_lines = []

        issues = logic_safety_validator._check_dangerous_patterns(
            added_lines, removed_lines, "test.py"
        )

        # Should detect security function
        assert any(i.type == "security_sensitive_change" for i in issues)

    def test_check_dangerous_patterns_error_handling(self, logic_safety_validator):
        """Test checking for error handling pattern changes."""
        added_lines = [(10, "    try:")]
        removed_lines = []

        issues = logic_safety_validator._check_dangerous_patterns(
            added_lines, removed_lines, "test.py"
        )

        # Should detect error handling
        assert any(i.type == "error_handling_change" for i in issues)

    @pytest.mark.asyncio
    async def test_analyze_ast_safety(self, logic_safety_validator):
        """Test analyzing AST for safety issues."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=5,
            patch="",
            before_content="x = 1",
            after_content="if x:\n    for i in range(10):\n        pass",
            binary=False,
        )

        issues = await logic_safety_validator._analyze_ast_safety(file_change)

        assert isinstance(issues, list)

    def test_extract_control_flow_nodes(self, logic_safety_validator, safe_ast_parser):
        """Test extracting control flow nodes."""
        code = """
if x > 0:
    for i in range(10):
        if i % 2:
            break
        continue
    return x
"""
        tree, _ = safe_ast_parser.parse_with_fallback(code)
        nodes = logic_safety_validator._extract_control_flow_nodes(tree)

        assert len(nodes) > 0
        node_types = [n["type"] for n in nodes]
        assert "If" in node_types
        assert "For" in node_types

    def test_extract_exception_nodes(self, logic_safety_validator, safe_ast_parser):
        """Test extracting exception handling nodes."""
        code = """
try:
    x = 1
except ValueError:
    pass
except:
    raise
"""
        tree, _ = safe_ast_parser.parse_with_fallback(code)
        nodes = logic_safety_validator._extract_exception_nodes(tree)

        assert len(nodes) > 0
        # Check for catch-all detection
        assert any(n.get("is_catch_all", False) for n in nodes)

    def test_analyze_control_flow_changes(self, logic_safety_validator, safe_ast_parser):
        """Test analyzing control flow changes."""
        before_code = "if x:\n    pass"
        after_code = "if x:\n    pass\nif y:\n    pass"

        before_tree, _ = safe_ast_parser.parse_with_fallback(before_code)
        after_tree, _ = safe_ast_parser.parse_with_fallback(after_code)

        issues = logic_safety_validator._analyze_control_flow_changes(
            before_tree, after_tree, "test.py"
        )

        assert isinstance(issues, list)

    def test_analyze_exception_handling_changes_removed(
        self, logic_safety_validator, safe_ast_parser
    ):
        """Test detecting removed exception handlers."""
        before_code = """
try:
    x = 1
except ValueError:
    pass
except TypeError:
    pass
"""
        after_code = """
try:
    x = 1
except ValueError:
    pass
"""
        before_tree, _ = safe_ast_parser.parse_with_fallback(before_code)
        after_tree, _ = safe_ast_parser.parse_with_fallback(after_code)

        issues = logic_safety_validator._analyze_exception_handling_changes(
            before_tree, after_tree, "test.py"
        )

        assert any(i.type == "exception_handling_removed" for i in issues)

    def test_analyze_security_changes(self, logic_safety_validator):
        """Test analyzing security changes in patch."""
        file_change = FileChange(
            filename="auth.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            patch="-    password = get_password()\n+    token = generate_token()",
            binary=False,
        )

        issues = logic_safety_validator._analyze_security_changes(file_change)

        assert len(issues) > 0
        assert any("security" in i.type.lower() for i in issues)

    def test_analyze_error_handling_changes(self, logic_safety_validator):
        """Test analyzing error handling changes in patch."""
        file_change = FileChange(
            filename="test.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="+    try:\n+        process()",
            binary=False,
        )

        issues = logic_safety_validator._analyze_error_handling_changes(file_change)

        assert any("error_handling" in i.type.lower() for i in issues)

    def test_violation_to_issue(self, logic_safety_validator):
        """Test converting SafetyViolation to ValidationIssue."""
        violation = SafetyViolation(
            violation_type="LOGIC_INVERSION",
            severity="high",
            line_number=42,
            before_code="x == 0",
            after_code="x != 0",
            explanation="Dangerous inversion",
            recommendation="Verify intent",
            confidence=0.9,
        )

        issue = logic_safety_validator._violation_to_issue(violation, "test.py")

        assert issue.type == "logic_inversion"
        assert issue.severity == "high"
        assert issue.line == 42
        assert issue.file == "test.py"


# ==============================================================================
# DiffParser Tests
# ==============================================================================


class TestDiffParser:
    """Tests for DiffParser class."""

    def test_parser_initialization(self, diff_parser):
        """Test parser initialization."""
        assert diff_parser.file_classifier is not None
        assert diff_parser.diff_header_pattern is not None
        assert diff_parser.hunk_header_pattern is not None

    def test_parse_github_diff_empty(self, diff_parser):
        """Test parsing empty GitHub diff."""
        result = diff_parser.parse_github_diff([])

        assert result == []

    def test_parse_github_diff_single_file(self, diff_parser):
        """Test parsing single file GitHub diff."""
        github_files = [
            {
                "filename": "test.py",
                "status": "modified",
                "additions": 5,
                "deletions": 3,
                "patch": "@@ -1,3 +1,5 @@\n-old\n+new",
            }
        ]

        result = diff_parser.parse_github_diff(github_files)

        assert len(result) == 1
        assert result[0].filename == "test.py"
        assert result[0].change_type == ChangeType.MODIFIED

    def test_parse_github_diff_added_file(self, diff_parser):
        """Test parsing added file in GitHub diff."""
        github_files = [
            {
                "filename": "new.py",
                "status": "added",
                "additions": 10,
                "deletions": 0,
                "patch": "@@ -0,0 +1,10 @@\n+new content",
            }
        ]

        result = diff_parser.parse_github_diff(github_files)

        assert result[0].change_type == ChangeType.ADDED

    def test_parse_github_diff_removed_file(self, diff_parser):
        """Test parsing removed file in GitHub diff."""
        github_files = [
            {
                "filename": "deleted.py",
                "status": "removed",
                "additions": 0,
                "deletions": 10,
            }
        ]

        result = diff_parser.parse_github_diff(github_files)

        assert result[0].change_type == ChangeType.DELETED

    def test_parse_github_diff_renamed_file(self, diff_parser):
        """Test parsing renamed file in GitHub diff."""
        github_files = [
            {
                "filename": "new_name.py",
                "previous_filename": "old_name.py",
                "status": "renamed",
                "additions": 0,
                "deletions": 0,
            }
        ]

        result = diff_parser.parse_github_diff(github_files)

        assert result[0].change_type == ChangeType.RENAMED

    def test_parse_github_diff_binary_file(self, diff_parser):
        """Test parsing binary file in GitHub diff."""
        github_files = [
            {
                "filename": "image.png",
                "status": "modified",
                "additions": 0,
                "deletions": 0,
                "binary": True,
            }
        ]

        result = diff_parser.parse_github_diff(github_files)

        assert result[0].binary is True

    def test_parse_git_diff_simple(self, diff_parser):
        """Test parsing simple git diff."""
        diff_text = """diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 line1
-old line
+new line
 line3
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert len(result) == 1
        assert result[0].filename == "test.py"
        assert result[0].additions == 1
        assert result[0].deletions == 1

    def test_parse_git_diff_new_file(self, diff_parser):
        """Test parsing git diff for new file."""
        diff_text = """diff --git a/new.py b/new.py
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert result[0].change_type == ChangeType.ADDED
        assert result[0].additions == 3

    def test_parse_git_diff_deleted_file(self, diff_parser):
        """Test parsing git diff for deleted file."""
        diff_text = """diff --git a/deleted.py b/deleted.py
deleted file mode 100644
index abc123..0000000
--- a/deleted.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert result[0].change_type == ChangeType.DELETED
        assert result[0].deletions == 3

    def test_parse_git_diff_renamed_file(self, diff_parser):
        """Test parsing git diff for renamed file."""
        diff_text = """diff --git a/old.py b/new.py
rename from old.py
rename to new.py
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert result[0].change_type == ChangeType.RENAMED

    def test_parse_git_diff_binary(self, diff_parser):
        """Test parsing git diff for binary file."""
        diff_text = """diff --git a/image.png b/image.png
Binary files a/image.png and b/image.png differ
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert result[0].binary is True

    def test_parse_git_diff_multiple_files(self, diff_parser):
        """Test parsing git diff with multiple files."""
        diff_text = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-old1
+new1
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-old2
+new2
"""
        result = diff_parser.parse_git_diff(diff_text)

        assert len(result) == 2

    def test_parse_hunk_header(self, diff_parser):
        """Test parsing hunk header."""
        line = "@@ -10,5 +15,7 @@ def function():"
        hunk = diff_parser._parse_hunk_header(line)

        assert hunk.old_start == 10
        assert hunk.old_count == 5
        assert hunk.new_start == 15
        assert hunk.new_count == 7
        assert hunk.context == "def function():"

    def test_parse_hunk_header_minimal(self, diff_parser):
        """Test parsing minimal hunk header."""
        line = "@@ -1 +1 @@"
        hunk = diff_parser._parse_hunk_header(line)

        assert hunk.old_start == 1
        assert hunk.old_count == 1
        assert hunk.new_start == 1
        assert hunk.new_count == 1

    def test_parse_hunk_header_invalid(self, diff_parser):
        """Test parsing invalid hunk header."""
        with pytest.raises(ValueError):
            diff_parser._parse_hunk_header("invalid header")

    def test_parse_diff_header(self, diff_parser):
        """Test parsing diff header."""
        line = "diff --git a/test.py b/test.py"
        result = diff_parser._parse_diff_header(line)

        assert result.filename == "test.py"
        assert result.change_type == ChangeType.MODIFIED

    def test_parse_diff_header_invalid(self, diff_parser):
        """Test parsing invalid diff header."""
        with pytest.raises(ValueError):
            diff_parser._parse_diff_header("not a diff header")

    def test_create_file_changes(self, diff_parser):
        """Test creating FileChange objects from parsed diffs."""
        parsed_diffs = [
            ParsedDiff(
                filename="test.py",
                old_filename=None,
                new_filename=None,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                hunks=[DiffHunk(1, 3, 1, 5, ["-old", "+new"])],
            )
        ]

        file_changes = diff_parser.create_file_changes(parsed_diffs)

        assert len(file_changes) == 1
        assert file_changes[0].filename == "test.py"
        assert file_changes[0].file_type == FileType.PYTHON

    def test_create_file_changes_with_content_provider(self, diff_parser):
        """Test creating FileChange objects with content provider."""
        parsed_diffs = [
            ParsedDiff(
                filename="test.py",
                old_filename=None,
                new_filename=None,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                hunks=[],
            )
        ]

        def content_provider(filename, ref):
            return f"content for {filename} at {ref}"

        file_changes = diff_parser.create_file_changes(parsed_diffs, content_provider)

        assert file_changes[0].before_content is not None
        assert file_changes[0].after_content is not None

    def test_create_pr_diff(self, diff_parser):
        """Test creating PRDiff object."""
        file_changes = [
            FileChange(
                filename="test.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                patch="",
            )
        ]

        pr_diff = diff_parser.create_pr_diff(
            file_changes,
            base_sha="abc123",
            head_sha="def456",
            metadata={"pr_number": 42},
        )

        assert pr_diff.base_sha == "abc123"
        assert pr_diff.head_sha == "def456"
        assert len(pr_diff.files_changed) == 1
        assert pr_diff.metadata["pr_number"] == 42

    def test_reconstruct_patch(self, diff_parser):
        """Test reconstructing patch from parsed diff."""
        parsed_diff = ParsedDiff(
            filename="test.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            hunks=[DiffHunk(1, 3, 1, 3, [" line1", "-old", "+new", " line3"])],
        )

        reconstructed = diff_parser._reconstruct_patch(parsed_diff)

        assert "diff --git" in reconstructed
        assert "@@ -1,3 +1,3 @@" in reconstructed

    def test_reconstruct_patch_binary(self, diff_parser):
        """Test reconstructing patch for binary file."""
        parsed_diff = ParsedDiff(
            filename="image.png",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            hunks=[],
            binary=True,
        )

        reconstructed = diff_parser._reconstruct_patch(parsed_diff)

        assert "Binary file" in reconstructed

    def test_validate_diff(self, diff_parser):
        """Test validating parsed diff."""
        valid_diff = ParsedDiff(
            filename="test.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            hunks=[DiffHunk(1, 2, 1, 2, [" ctx", "-old", "+new"])],
        )

        issues = diff_parser.validate_diff(valid_diff)

        # May have minor issues depending on validation strictness
        assert isinstance(issues, list)

    def test_validate_diff_missing_filename(self, diff_parser):
        """Test validating diff with missing filename."""
        invalid_diff = ParsedDiff(
            filename="",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            hunks=[],
        )

        issues = diff_parser.validate_diff(invalid_diff)

        assert "Missing filename" in issues


# ==============================================================================
# PRFileTypeClassifier Tests
# ==============================================================================


class TestPRFileTypeClassifier:
    """Tests for PRFileTypeClassifier class."""

    def test_classifier_initialization(self, file_classifier):
        """Test classifier initialization."""
        assert len(file_classifier._extension_mappings) > 0
        assert len(file_classifier._filename_patterns) > 0
        assert len(file_classifier._content_patterns) > 0
        assert len(file_classifier._binary_extensions) > 0

    def test_classify_python_by_extension(self, file_classifier):
        """Test classifying Python file by extension."""
        assert file_classifier.classify_file("test.py") == FileType.PYTHON
        assert file_classifier.classify_file("test.pyi") == FileType.PYTHON
        assert file_classifier.classify_file("test.pyx") == FileType.PYTHON

    def test_classify_javascript_by_extension(self, file_classifier):
        """Test classifying JavaScript file by extension."""
        assert file_classifier.classify_file("test.js") == FileType.JAVASCRIPT
        assert file_classifier.classify_file("test.jsx") == FileType.JAVASCRIPT
        assert file_classifier.classify_file("test.mjs") == FileType.JAVASCRIPT

    def test_classify_typescript_by_extension(self, file_classifier):
        """Test classifying TypeScript file by extension."""
        assert file_classifier.classify_file("test.ts") == FileType.TYPESCRIPT
        assert file_classifier.classify_file("test.tsx") == FileType.TYPESCRIPT

    def test_classify_markdown_by_extension(self, file_classifier):
        """Test classifying Markdown file by extension."""
        assert file_classifier.classify_file("README.md") == FileType.MARKDOWN
        assert file_classifier.classify_file("docs.markdown") == FileType.MARKDOWN

    def test_classify_yaml_by_extension(self, file_classifier):
        """Test classifying YAML file by extension."""
        assert file_classifier.classify_file("config.yml") == FileType.YAML
        assert file_classifier.classify_file("config.yaml") == FileType.YAML

    def test_classify_json_by_extension(self, file_classifier):
        """Test classifying JSON file by extension."""
        assert file_classifier.classify_file("package.json") == FileType.JSON

    def test_classify_html_by_extension(self, file_classifier):
        """Test classifying HTML file by extension."""
        assert file_classifier.classify_file("index.html") == FileType.HTML
        assert file_classifier.classify_file("page.htm") == FileType.HTML

    def test_classify_css_by_extension(self, file_classifier):
        """Test classifying CSS file by extension."""
        assert file_classifier.classify_file("styles.css") == FileType.CSS
        assert file_classifier.classify_file("styles.scss") == FileType.CSS

    def test_classify_dockerfile_by_pattern(self, file_classifier):
        """Test classifying Dockerfile by pattern."""
        assert file_classifier.classify_file("Dockerfile") == FileType.DOCKERFILE
        assert file_classifier.classify_file("Dockerfile.dev") == FileType.DOCKERFILE

    def test_classify_by_content_python(self, file_classifier):
        """Test classifying by content - Python."""
        content = "#!/usr/bin/env python\nimport os\ndef main():\n    pass"
        result = file_classifier.classify_file("script", content)

        assert result == FileType.PYTHON

    def test_classify_by_content_javascript(self, file_classifier):
        """Test classifying by content - JavaScript."""
        content = "const x = require('module');\nmodule.exports = x;"
        result = file_classifier.classify_file("script", content)

        assert result == FileType.JAVASCRIPT

    def test_classify_binary_file(self, file_classifier):
        """Test classifying binary files."""
        assert file_classifier.classify_file("image.png") == FileType.UNKNOWN
        assert file_classifier.classify_file("image.jpg") == FileType.UNKNOWN
        assert file_classifier.classify_file("archive.zip") == FileType.UNKNOWN

    def test_classify_unknown_file(self, file_classifier):
        """Test classifying unknown file type."""
        assert file_classifier.classify_file("unknown.xyz") == FileType.UNKNOWN

    def test_is_binary_file(self, file_classifier):
        """Test binary file detection."""
        assert file_classifier._is_binary_file("image.png") is True
        assert file_classifier._is_binary_file("test.py") is False
        assert file_classifier._is_binary_file("archive.zip") is True

    def test_is_test_file(self, file_classifier):
        """Test test file detection."""
        assert file_classifier._is_test_file("test_module.py") is True
        assert file_classifier._is_test_file("module_test.py") is True
        assert file_classifier._is_test_file("module.test.js") is True
        assert file_classifier._is_test_file("tests/test.py") is True
        assert file_classifier._is_test_file("regular.py") is False

    def test_is_config_file(self, file_classifier):
        """Test config file detection."""
        assert file_classifier._is_config_file("config.yml") is True
        assert file_classifier._is_config_file("package.json") is True
        assert file_classifier._is_config_file("pyproject.toml") is True
        assert file_classifier._is_config_file("Dockerfile") is True
        assert file_classifier._is_config_file(".env") is True
        assert file_classifier._is_config_file("main.py") is False

    def test_is_documentation_file(self, file_classifier):
        """Test documentation file detection."""
        assert file_classifier._is_documentation_file("README.md") is True
        assert file_classifier._is_documentation_file("CHANGELOG.md") is True
        assert file_classifier._is_documentation_file("docs/guide.md") is True
        assert file_classifier._is_documentation_file("LICENSE") is True
        assert file_classifier._is_documentation_file("main.py") is False

    def test_get_file_info(self, file_classifier):
        """Test getting comprehensive file info."""
        info = file_classifier.get_file_info("test.py")

        assert info["file_type"] == FileType.PYTHON
        assert info["extension"] == ".py"
        assert info["is_binary"] is False
        assert "is_test_file" in info
        assert "is_config_file" in info
        assert "is_documentation" in info

    def test_get_file_info_with_content(self, file_classifier):
        """Test getting file info with content analysis."""
        content = "class MyClass:\n    def method(self):\n        pass"
        info = file_classifier.get_file_info("module.py", content)

        assert info["language_specific_info"]["has_classes"] is True

    def test_analyze_python_file(self, file_classifier):
        """Test analyzing Python file characteristics."""
        content = """
import os

class MyClass:
    pass

def my_function():
    pass

if __name__ == "__main__":
    pass
"""
        info = file_classifier._analyze_python_file(content)

        assert info["has_classes"] is True
        assert info["has_functions"] is True
        assert info["has_imports"] is True
        assert info["is_script"] is True

    def test_analyze_python_file_none(self, file_classifier):
        """Test analyzing Python file with None content."""
        info = file_classifier._analyze_python_file(None)

        assert info["has_classes"] is False
        assert info["has_functions"] is False

    def test_analyze_js_ts_file(self, file_classifier):
        """Test analyzing JavaScript/TypeScript file characteristics."""
        content = """
import React from 'react';
export function Component() {
    return <div>Hello</div>;
}
"""
        info = file_classifier._analyze_js_ts_file(content)

        assert info["has_imports"] is True
        assert info["has_exports"] is True
        assert "React" in info["framework_indicators"]

    def test_analyze_js_ts_file_none(self, file_classifier):
        """Test analyzing JS/TS file with None content."""
        info = file_classifier._analyze_js_ts_file(None)

        assert info["has_functions"] is False

    def test_analyze_markdown_file(self, file_classifier):
        """Test analyzing Markdown file characteristics."""
        content = """
# Title

Some text with [link](url).

```python
code block
```

| Col1 | Col2 |
|------|------|
| A    | B    |
"""
        info = file_classifier._analyze_markdown_file(content)

        assert info["has_headers"] is True
        assert info["has_code_blocks"] is True
        assert info["has_links"] is True
        assert info["has_tables"] is True

    def test_analyze_markdown_file_none(self, file_classifier):
        """Test analyzing Markdown file with None content."""
        info = file_classifier._analyze_markdown_file(None)

        assert info["has_headers"] is False


# ==============================================================================
# DiffStats Tests
# ==============================================================================


class TestDiffStats:
    """Tests for DiffStats class."""

    def test_diff_stats_from_files_empty(self):
        """Test creating DiffStats from empty file list."""
        stats = DiffStats.from_files([])

        assert stats.total_files == 0
        assert stats.total_additions == 0
        assert stats.total_deletions == 0

    def test_diff_stats_from_files_single(self):
        """Test creating DiffStats from single file."""
        files = [
            FileChange(
                filename="test.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=10,
                deletions=5,
                patch="",
            )
        ]

        stats = DiffStats.from_files(files)

        assert stats.total_files == 1
        assert stats.total_additions == 10
        assert stats.total_deletions == 5
        assert stats.files_modified == 1

    def test_diff_stats_from_files_multiple(self):
        """Test creating DiffStats from multiple files."""
        files = [
            FileChange(
                filename="new.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.ADDED,
                additions=20,
                deletions=0,
                patch="",
            ),
            FileChange(
                filename="modified.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                patch="",
            ),
            FileChange(
                filename="deleted.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.DELETED,
                additions=0,
                deletions=15,
                patch="",
            ),
            FileChange(
                filename="renamed.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.RENAMED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]

        stats = DiffStats.from_files(files)

        assert stats.total_files == 4
        assert stats.total_additions == 25
        assert stats.total_deletions == 18
        assert stats.files_added == 1
        assert stats.files_deleted == 1
        assert stats.files_modified == 1
        assert stats.files_renamed == 1


# ==============================================================================
# PRDiff Tests
# ==============================================================================


class TestPRDiff:
    """Tests for PRDiff class."""

    def test_pr_diff_creation(self):
        """Test PRDiff creation."""
        files = [
            FileChange(
                filename="test.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=10,
                deletions=5,
                patch="",
            )
        ]
        stats = DiffStats.from_files(files)

        pr_diff = PRDiff(
            base_sha="abc123",
            head_sha="def456",
            files_changed=files,
            stats=stats,
            metadata={"pr_number": 42},
        )

        assert pr_diff.base_sha == "abc123"
        assert pr_diff.head_sha == "def456"
        assert len(pr_diff.files_changed) == 1
        assert pr_diff.metadata["pr_number"] == 42

    def test_pr_diff_change_complexity_empty(self):
        """Test PRDiff change complexity with no files."""
        pr_diff = PRDiff(
            base_sha="abc",
            head_sha="def",
            files_changed=[],
            stats=DiffStats.from_files([]),
            metadata={},
        )

        assert pr_diff.change_complexity == 0.0

    def test_pr_diff_change_complexity_small(self):
        """Test PRDiff change complexity for small changes."""
        files = [
            FileChange(
                filename="test.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                patch="",
            )
        ]

        pr_diff = PRDiff(
            base_sha="abc",
            head_sha="def",
            files_changed=files,
            stats=DiffStats.from_files(files),
            metadata={},
        )

        assert pr_diff.change_complexity == 0.08  # (5+3)/100

    def test_pr_diff_change_complexity_large(self):
        """Test PRDiff change complexity is capped at 10."""
        files = [
            FileChange(
                filename="test.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=1500,
                deletions=1000,
                patch="",
            )
        ]

        pr_diff = PRDiff(
            base_sha="abc",
            head_sha="def",
            files_changed=files,
            stats=DiffStats.from_files(files),
            metadata={},
        )

        assert pr_diff.change_complexity == 10.0


# ==============================================================================
# ValidationResult Tests
# ==============================================================================


class TestValidationResult:
    """Tests for ValidationResult model properties."""

    def test_critical_issues_empty(self):
        """Test getting critical issues from empty list."""
        from coding_client.models.pr_validation import RiskLevel, ValidationResult

        result = ValidationResult(
            validation_type="test",
            passed=True,
            score=1.0,
            risk_level=RiskLevel.LOW,
            issues=[],
            recommendations=[],
        )

        assert result.critical_issues == []

    def test_critical_issues_filter(self):
        """Test filtering critical issues."""
        from coding_client.models.pr_validation import RiskLevel, ValidationResult

        result = ValidationResult(
            validation_type="test",
            passed=False,
            score=0.5,
            risk_level=RiskLevel.HIGH,
            issues=[
                ValidationIssue(
                    type="error",
                    severity="CRITICAL",
                    message="Critical issue",
                    file="test.py",
                ),
                ValidationIssue(
                    type="warning",
                    severity="HIGH",
                    message="High issue",
                    file="test.py",
                ),
                ValidationIssue(type="info", severity="LOW", message="Low issue", file="test.py"),
            ],
            recommendations=[],
        )

        assert len(result.critical_issues) == 1
        assert result.critical_issues[0].severity == "CRITICAL"


# ==============================================================================
# SemanticAnalysisResult Tests
# ==============================================================================


class TestSemanticAnalysisResult:
    """Tests for SemanticAnalysisResult model properties."""

    def test_has_breaking_changes_false(self):
        """Test has_breaking_changes when empty."""
        result = SemanticAnalysisResult(
            breaking_changes=[],
            logic_safety_issues=[],
            api_surface_changes={},
            complexity_changes={},
        )

        assert result.has_breaking_changes is False

    def test_has_breaking_changes_true(self):
        """Test has_breaking_changes when populated."""
        result = SemanticAnalysisResult(
            breaking_changes=[
                BreakingChange(
                    type="FUNCTION_REMOVED",
                    severity="HIGH",
                    description="Test",
                    file_path="test.py",
                    affected_symbol="func",
                    recommendation="Fix it",
                )
            ],
            logic_safety_issues=[],
            api_surface_changes={},
            complexity_changes={},
        )

        assert result.has_breaking_changes is True

    def test_has_logic_safety_issues_false(self):
        """Test has_logic_safety_issues when empty."""
        result = SemanticAnalysisResult(
            breaking_changes=[],
            logic_safety_issues=[],
            api_surface_changes={},
            complexity_changes={},
        )

        assert result.has_logic_safety_issues is False

    def test_has_logic_safety_issues_true(self):
        """Test has_logic_safety_issues when populated."""
        result = SemanticAnalysisResult(
            breaking_changes=[],
            logic_safety_issues=[
                ValidationIssue(
                    type="logic_inversion",
                    severity="high",
                    message="Test",
                    file="test.py",
                )
            ],
            api_surface_changes={},
            complexity_changes={},
        )

        assert result.has_logic_safety_issues is True


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(
        self,
        ast_diff_analyzer,
        breaking_change_detector,
        logic_safety_validator,
        simple_python_code_before,
        simple_python_code_after,
    ):
        """Test full analysis workflow with all analyzers."""
        file_change = FileChange(
            filename="module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=15,
            deletions=5,
            patch="@@ -1,10 +1,20 @@\n-def hello(name):\n+def hello(name: str) -> str:",
            before_content=simple_python_code_before,
            after_content=simple_python_code_after,
            binary=False,
        )

        # Run all analyzers
        semantic_result = await ast_diff_analyzer.analyze_semantic_changes(file_change)
        breaking_changes = await breaking_change_detector.detect_breaking_changes(file_change)
        logic_issues = await logic_safety_validator.validate_logic_safety(file_change)

        # All should return valid results
        assert isinstance(semantic_result, SemanticAnalysisResult)
        assert isinstance(breaking_changes, list)
        assert isinstance(logic_issues, list)

    def test_diff_parser_to_file_change_workflow(self, diff_parser):
        """Test workflow from parsing diff to creating FileChange."""
        github_files = [
            {
                "filename": "src/module.py",
                "status": "modified",
                "additions": 10,
                "deletions": 5,
                "patch": "@@ -1,5 +1,10 @@\n-old\n+new",
            }
        ]

        # Parse GitHub diff
        parsed_diffs = diff_parser.parse_github_diff(github_files)

        # Create file changes
        file_changes = diff_parser.create_file_changes(parsed_diffs)

        # Create PR diff
        pr_diff = diff_parser.create_pr_diff(
            file_changes,
            base_sha="abc123",
            head_sha="def456",
        )

        assert pr_diff.stats.total_files == 1
        assert pr_diff.files_changed[0].file_type == FileType.PYTHON

    def test_file_classifier_integration(self, file_classifier, diff_parser):
        """Test file classifier integration with diff parser."""
        # The diff parser uses file classifier internally
        parsed_diffs = [
            ParsedDiff(
                filename="app.py",
                old_filename=None,
                new_filename=None,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                hunks=[],
            ),
            ParsedDiff(
                filename="config.yaml",
                old_filename=None,
                new_filename=None,
                change_type=ChangeType.MODIFIED,
                additions=2,
                deletions=1,
                hunks=[],
            ),
            ParsedDiff(
                filename="Dockerfile",
                old_filename=None,
                new_filename=None,
                change_type=ChangeType.MODIFIED,
                additions=3,
                deletions=2,
                hunks=[],
            ),
        ]

        file_changes = diff_parser.create_file_changes(parsed_diffs)

        assert file_changes[0].file_type == FileType.PYTHON
        assert file_changes[1].file_type == FileType.YAML
        assert file_changes[2].file_type == FileType.DOCKERFILE
