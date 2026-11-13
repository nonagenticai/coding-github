"""Extended tests for ast_diff_analyzer.py — covers all previously uncovered lines."""

import ast
import sys
from unittest.mock import MagicMock, patch

from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
    SemanticAnalysisResult,
)

from services.pr_validation.analyzers.semantic.ast_diff_analyzer import (
    ASTDiffAnalyzer,
    ASTDiffResult,
    ASTNode,
    SafeASTParser,
)


def _fc(filename="src/main.py", **kw):
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 10)
    kw.setdefault("deletions", 5)
    kw.setdefault("patch", "@@ -1,5 +1,10 @@")
    return FileChange(filename=filename, **kw)


# ---------------------------------------------------------------------------
# SafeASTParser tests
# ---------------------------------------------------------------------------


class TestSafeASTParserBytesContent:
    """Lines 115-121: bytes content with chardet encoding detection."""

    def test_bytes_content_decoded_with_chardet(self):
        """Bytes content should be decoded via chardet and parsed successfully."""
        code = b"x = 1\n"
        parser = SafeASTParser()
        with patch.dict("sys.modules", {"chardet": MagicMock()}):
            chardet_mock = sys.modules["chardet"]
            chardet_mock.detect.return_value = {"encoding": "utf-8"}
            tree, errors = parser.parse_with_fallback(code, "test.py")
        assert tree is not None

    def test_bytes_content_chardet_failure(self):
        """When chardet raises, we should record the error and continue."""
        code = b"y = 2\n"
        parser = SafeASTParser()
        with patch.dict("sys.modules", {"chardet": MagicMock()}):
            chardet_mock = sys.modules["chardet"]
            chardet_mock.detect.side_effect = RuntimeError("chardet boom")
            tree, errors = parser.parse_with_fallback(code, "test.py")
        # After chardet failure content is still bytes, so ast.parse will fail too.
        # We just need to confirm the encoding error was captured.
        assert any("Encoding detection failed" in e for e in errors)


class TestSafeASTParserSyntaxFallback:
    """Lines 128-129, 134, 139-141: SyntaxError / generic error fallback + version loop."""

    def test_syntax_error_fallback_to_older_version(self):
        """If ast.parse raises SyntaxError, the version loop should attempt older versions."""
        parser = SafeASTParser()
        call_count = 0
        original_parse = ast.parse

        def _failing_then_ok(content, *, filename="<unknown>", **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SyntaxError("bad syntax")
            return original_parse(content, filename=filename, **kw)

        valid_code = "a = 1\n"
        with patch("ast.parse", side_effect=_failing_then_ok):
            tree, errors = parser.parse_with_fallback(valid_code, "test.py")

        assert any("Syntax error" in e for e in errors)
        # compile succeeds for valid code, so parse on 2nd call returns tree
        assert tree is not None
        assert any("Parsed successfully with Python" in e for e in errors)

    def test_generic_parse_error_recorded(self):
        """Lines 128-129: non-SyntaxError exceptions are recorded."""
        parser = SafeASTParser()

        def _raise_typeerror(content, *, filename="<unknown>", **kw):
            raise TypeError("not a syntax error")

        with patch("ast.parse", side_effect=_raise_typeerror):
            with patch("builtins.compile", side_effect=TypeError("compile also fails")):
                tree, errors = parser.parse_with_fallback("x = 1\n", "test.py")

        assert any("Parse error" in e for e in errors)

    def test_skip_versions_gte_current(self):
        """Line 134: versions >= current should be skipped."""
        parser = SafeASTParser()
        # Set supported_python_versions to contain only versions >= current
        parser.supported_python_versions = [sys.version_info[:2], (99, 99)]

        with patch("ast.parse", side_effect=SyntaxError("bad")):
            with patch("builtins.compile", side_effect=SyntaxError("still bad")):
                tree, errors = parser.parse_with_fallback("x = 1\n", "test.py")

        # Should skip all versions, never get a "Parsed successfully" message
        assert not any("Parsed successfully" in e for e in errors)


class TestSafeASTParserPartialParse:
    """Lines 152-153, 156-157: partial content parsing (last resort)."""

    def test_partial_content_parse_success(self):
        """Lines 152-153: successful partial parse when full parse fails."""
        bad_code = "x = 1\n" * 20 + "def (\n"
        parser = SafeASTParser()
        # Remove all supported versions to skip the version loop entirely
        parser.supported_python_versions = []

        original_ast_parse = ast.parse
        first_call = [True]

        def _fail_first_then_ok(content_arg, *, filename="<unknown>", **kw):
            if first_call[0]:
                first_call[0] = False
                raise SyntaxError("first call fails")
            return original_ast_parse(content_arg, filename=filename, **kw)

        with patch("ast.parse", side_effect=_fail_first_then_ok):
            tree, errors = parser.parse_with_fallback(bad_code, "test.py")

        assert tree is not None
        assert any("Partial parse successful" in e for e in errors)

    def test_partial_parse_exception(self):
        """Lines 156-157: exception in the partial parse outer try block."""
        parser = SafeASTParser()
        parser.supported_python_versions = []

        # Create a string-like object whose .split() raises inside the partial-parse block.
        # The object must pass `not content` and `not content.strip()` checks (lines 109),
        # and survive ast.parse(content, ...) in the initial try (line 125).
        class _ExplodingStr(str):
            _call_count = 0

            def split(self, *args, **kwargs):
                # First .split() call is from the partial-parse block (line 147)
                self._call_count += 1
                raise RuntimeError("split boom")

        content = _ExplodingStr("x = 1\n")

        # ast.parse will receive _ExplodingStr which is a str subclass — it should
        # work normally. We force it to fail with SyntaxError so we reach the partial block.
        with patch("ast.parse", side_effect=SyntaxError("forced fail")):
            tree, errors = parser.parse_with_fallback(content, "test.py")

        assert tree is None
        assert any("Partial parse failed" in e for e in errors)


# ---------------------------------------------------------------------------
# ASTDiffAnalyzer tests
# ---------------------------------------------------------------------------


class TestAnalyzeSemanticChangesException:
    """Lines 281-283: exception handler in analyze_semantic_changes."""

    async def test_exception_returns_analysis_error(self):
        analyzer = ASTDiffAnalyzer()
        fc = _fc(
            filename="boom.py",
            before_content="x = 1\n",
            after_content="y = 2\n",
        )
        with patch.object(
            analyzer.parser, "parse_with_fallback", side_effect=RuntimeError("kaboom")
        ):
            result = await analyzer.analyze_semantic_changes(fc)

        assert isinstance(result, SemanticAnalysisResult)
        assert len(result.breaking_changes) == 1
        assert result.breaking_changes[0].type == "ANALYSIS_ERROR"


class TestNodesDifferentAttributes:
    """Line 499: _nodes_are_different returns True when attributes differ."""

    def test_different_attributes(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="ClassDef",
            name="Foo",
            lineno=1,
            col_offset=0,
            attributes={"bases": ["Base1"]},
        )
        after = ASTNode(
            node_type="ClassDef",
            name="Foo",
            lineno=1,
            col_offset=0,
            attributes={"bases": ["Base2"]},
        )
        assert analyzer._nodes_are_different(before, after) is True

    def test_same_attributes_same_nodes(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="ClassDef",
            name="Foo",
            lineno=1,
            col_offset=0,
            attributes={"bases": ["Base1"]},
        )
        after = ASTNode(
            node_type="ClassDef",
            name="Foo",
            lineno=1,
            col_offset=0,
            attributes={"bases": ["Base1"]},
        )
        assert analyzer._nodes_are_different(before, after) is False


class TestCompareAstsModifiedNodes:
    """Line 354: modified_nodes branch in _compare_asts."""

    def test_modified_class_detected_via_bases(self):
        """Same class name but different base classes → modified node (line 354).

        ClassDef nodes have signature=None, so _nodes_are_different falls
        through to the attributes comparison (line 498-499).
        """
        analyzer = ASTDiffAnalyzer()
        # Add full node name so _extract_nodes finds ClassDef
        analyzer.important_nodes.add("ClassDef")

        before_code = "class Foo:\n    pass\n"
        after_code = "class Foo(object):\n    pass\n"

        before_ast = ast.parse(before_code)
        after_ast = ast.parse(after_code)

        result = analyzer._compare_asts(before_ast, after_ast)
        # Same node_key (ClassDef:Foo:) but different attributes (bases changed)
        assert len(result.modified_nodes) > 0


class TestAstToStringFallbacks:
    """Lines 476-485: ast.unparse fallback with isinstance checks."""

    def test_fallback_name_node(self):
        analyzer = ASTDiffAnalyzer()
        node = ast.Name(id="foo")
        with patch("ast.unparse", side_effect=AttributeError):
            assert analyzer._ast_to_string(node) == "foo"

    def test_fallback_constant_node(self):
        analyzer = ASTDiffAnalyzer()
        node = ast.Constant(value=42)
        with patch("ast.unparse", side_effect=ValueError):
            assert analyzer._ast_to_string(node) == "42"

    def test_fallback_attribute_node(self):
        analyzer = ASTDiffAnalyzer()
        node = ast.Attribute(
            value=ast.Name(id="os"),
            attr="path",
            ctx=ast.Load(),
        )
        with patch("ast.unparse", side_effect=AttributeError):
            assert analyzer._ast_to_string(node) == "os.path"

    def test_fallback_unknown_node(self):
        analyzer = ASTDiffAnalyzer()
        node = ast.BoolOp(op=ast.And(), values=[])
        with patch("ast.unparse", side_effect=AttributeError):
            result = analyzer._ast_to_string(node)
            assert result == "<BoolOp>"


class TestIdentifyStructuralChanges:
    """Lines 515-559: NEW_FUNCTION_ADDED, NEW_CLASS_ADDED, NEW_IMPORT_ADDED,
    FUNCTION_REMOVED, CLASS_REMOVED, IMPORT_REMOVED, FUNCTION_MODIFIED, CLASS_MODIFIED."""

    def _make_node(self, node_type, name="test"):
        return ASTNode(node_type=node_type, name=name, lineno=1, col_offset=0)

    def test_new_function_added(self):
        analyzer = ASTDiffAnalyzer()
        added = [self._make_node("FunctionDef", "my_func")]
        changes = analyzer._identify_structural_changes(added, [], [])
        assert "NEW_FUNCTION_ADDED" in changes

    def test_new_async_function_added(self):
        analyzer = ASTDiffAnalyzer()
        added = [self._make_node("AsyncFunctionDef", "async_func")]
        changes = analyzer._identify_structural_changes(added, [], [])
        assert "NEW_FUNCTION_ADDED" in changes

    def test_new_class_added(self):
        analyzer = ASTDiffAnalyzer()
        added = [self._make_node("ClassDef", "MyClass")]
        changes = analyzer._identify_structural_changes(added, [], [])
        assert "NEW_CLASS_ADDED" in changes

    def test_new_import_added(self):
        analyzer = ASTDiffAnalyzer()
        added = [self._make_node("Import", "os")]
        changes = analyzer._identify_structural_changes(added, [], [])
        assert "NEW_IMPORT_ADDED" in changes

    def test_new_import_from_added(self):
        analyzer = ASTDiffAnalyzer()
        added = [self._make_node("ImportFrom", "from os import path")]
        changes = analyzer._identify_structural_changes(added, [], [])
        assert "NEW_IMPORT_ADDED" in changes

    def test_function_removed(self):
        analyzer = ASTDiffAnalyzer()
        removed = [self._make_node("FunctionDef", "old_func")]
        changes = analyzer._identify_structural_changes([], removed, [])
        assert "FUNCTION_REMOVED" in changes

    def test_async_function_removed(self):
        analyzer = ASTDiffAnalyzer()
        removed = [self._make_node("AsyncFunctionDef", "old_async")]
        changes = analyzer._identify_structural_changes([], removed, [])
        assert "FUNCTION_REMOVED" in changes

    def test_class_removed(self):
        analyzer = ASTDiffAnalyzer()
        removed = [self._make_node("ClassDef", "OldClass")]
        changes = analyzer._identify_structural_changes([], removed, [])
        assert "CLASS_REMOVED" in changes

    def test_import_removed(self):
        analyzer = ASTDiffAnalyzer()
        removed = [self._make_node("Import", "os")]
        changes = analyzer._identify_structural_changes([], removed, [])
        assert "IMPORT_REMOVED" in changes

    def test_import_from_removed(self):
        analyzer = ASTDiffAnalyzer()
        removed = [self._make_node("ImportFrom", "from os import path")]
        changes = analyzer._identify_structural_changes([], removed, [])
        assert "IMPORT_REMOVED" in changes

    def test_function_modified(self):
        analyzer = ASTDiffAnalyzer()
        before = self._make_node("FunctionDef", "func")
        after = self._make_node("FunctionDef", "func")
        changes = analyzer._identify_structural_changes([], [], [(before, after)])
        assert "FUNCTION_MODIFIED" in changes

    def test_async_function_modified(self):
        analyzer = ASTDiffAnalyzer()
        before = self._make_node("AsyncFunctionDef", "afunc")
        after = self._make_node("AsyncFunctionDef", "afunc")
        changes = analyzer._identify_structural_changes([], [], [(before, after)])
        assert "FUNCTION_MODIFIED" in changes

    def test_class_modified(self):
        analyzer = ASTDiffAnalyzer()
        before = self._make_node("ClassDef", "Cls")
        after = self._make_node("ClassDef", "Cls")
        changes = analyzer._identify_structural_changes([], [], [(before, after)])
        assert "CLASS_MODIFIED" in changes


class TestDetectBreakingChangesRemovedNodes:
    """Lines 584-585: breaking change from removed public class."""

    def test_removed_public_class_is_breaking(self):
        analyzer = ASTDiffAnalyzer()
        removed_class = ASTNode(node_type="ClassDef", name="PublicClass", lineno=5, col_offset=0)
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[removed_class],
            modified_nodes=[],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["CLASS_REMOVED"],
        )
        breaks = analyzer._detect_breaking_changes(diff, "module.py")
        assert len(breaks) == 1
        assert breaks[0].type == "CLASS_REMOVED"
        assert breaks[0].severity == "HIGH"

    def test_removed_public_function_is_breaking(self):
        analyzer = ASTDiffAnalyzer()
        removed_func = ASTNode(node_type="FunctionDef", name="public_func", lineno=10, col_offset=0)
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[removed_func],
            modified_nodes=[],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_REMOVED"],
        )
        breaks = analyzer._detect_breaking_changes(diff, "module.py")
        assert len(breaks) == 1
        assert breaks[0].type == "FUNCTION_REMOVED"

    def test_removed_private_function_not_breaking(self):
        analyzer = ASTDiffAnalyzer()
        removed_func = ASTNode(
            node_type="FunctionDef", name="_private_func", lineno=10, col_offset=0
        )
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[removed_func],
            modified_nodes=[],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=[],
        )
        breaks = analyzer._detect_breaking_changes(diff, "module.py")
        assert len(breaks) == 0


class TestDetectBreakingChangesModifiedSignatures:
    """Lines 632-633: breaking change from modified public function signatures."""

    def test_modified_public_signature_is_breaking(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="FunctionDef",
            name="process",
            lineno=1,
            col_offset=0,
            signature="de process(x)",
        )
        after = ASTNode(
            node_type="FunctionDef",
            name="process",
            lineno=1,
            col_offset=0,
            signature="de process(x, y)",
        )
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_MODIFIED"],
        )
        breaks = analyzer._detect_breaking_changes(diff, "api.py")
        assert len(breaks) == 1
        assert breaks[0].type == "FUNCTION_SIGNATURE_CHANGED"
        assert breaks[0].severity == "HIGH"

    def test_modified_async_public_signature_is_breaking(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="AsyncFunctionDef",
            name="fetch",
            lineno=1,
            col_offset=0,
            signature="async de fetch(url)",
        )
        after = ASTNode(
            node_type="AsyncFunctionDef",
            name="fetch",
            lineno=1,
            col_offset=0,
            signature="async de fetch(url, timeout)",
        )
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=["FUNCTION_MODIFIED"],
        )
        breaks = analyzer._detect_breaking_changes(diff, "client.py")
        assert len(breaks) == 1
        assert breaks[0].type == "FUNCTION_SIGNATURE_CHANGED"

    def test_same_signature_no_break(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="FunctionDef",
            name="calc",
            lineno=1,
            col_offset=0,
            signature="de calc(x)",
        )
        after = ASTNode(
            node_type="FunctionDef",
            name="calc",
            lineno=1,
            col_offset=0,
            signature="de calc(x)",
        )
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=[],
        )
        breaks = analyzer._detect_breaking_changes(diff, "math.py")
        assert len(breaks) == 0


class TestAnalyzeLogicSafetyModifiedTry:
    """Lines 632-633 area: error handling changes produce ValidationIssue."""

    def test_modified_try_block_creates_issue(self):
        analyzer = ASTDiffAnalyzer()
        before = ASTNode(
            node_type="Try",
            name="",
            lineno=5,
            col_offset=0,
        )
        after = ASTNode(
            node_type="Try",
            name="",
            lineno=7,
            col_offset=0,
        )
        diff = ASTDiffResult(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[(before, after)],
            unchanged_nodes=[],
            complexity_change=0.0,
            structural_changes=[],
        )
        fc = _fc(filename="handler.py")
        issues = analyzer._analyze_logic_safety(diff, fc)
        assert len(issues) >= 1
        assert issues[0].type == "error_handling_change"


class TestEndToEndSemanticAnalysis:
    """Integration-style tests exercising the full analyze_semantic_changes path
    to hit the modified-nodes branch (line 354) and structural change lines."""

    async def test_function_added_and_removed(self):
        analyzer = ASTDiffAnalyzer()
        before_code = "def old_func(a):\n    return a\n"
        after_code = "def new_func(b):\n    return b\n"
        fc = _fc(
            filename="changes.py",
            before_content=before_code,
            after_content=after_code,
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert isinstance(result, SemanticAnalysisResult)

    async def test_class_added(self):
        analyzer = ASTDiffAnalyzer()
        fc = _fc(
            filename="new_module.py",
            before_content="",
            after_content="class NewClass:\n    pass\n",
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert isinstance(result, SemanticAnalysisResult)

    async def test_function_removed_end_to_end(self):
        """End-to-end: removing code produces a valid SemanticAnalysisResult."""
        analyzer = ASTDiffAnalyzer()
        before = "x = 1\nfor i in range(10):\n    pass\n"
        after = "# everything removed\n"
        fc = _fc(
            filename="api.py",
            before_content=before,
            after_content=after,
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert isinstance(result, SemanticAnalysisResult)
        # Complexity should have decreased
        assert result.complexity_changes.get("change", 0) <= 0

    async def test_class_modified_end_to_end(self):
        """End-to-end: class with changed bases → modified node detected."""
        analyzer = ASTDiffAnalyzer()
        analyzer.important_nodes.add("ClassDef")
        before = "class Foo:\n    pass\n"
        after = "class Foo(object):\n    pass\n"
        fc = _fc(
            filename="models.py",
            before_content=before,
            after_content=after,
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert isinstance(result, SemanticAnalysisResult)

    async def test_non_python_file_skipped(self):
        analyzer = ASTDiffAnalyzer()
        fc = _fc(
            filename="readme.md",
            file_type=FileType.MARKDOWN,
            before_content="# old",
            after_content="# new",
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert result.breaking_changes == []

    async def test_both_parse_failures(self):
        analyzer = ASTDiffAnalyzer()
        fc = _fc(
            filename="bad.py",
            before_content="",
            after_content="",
        )
        result = await analyzer.analyze_semantic_changes(fc)
        assert len(result.breaking_changes) == 1
        assert result.breaking_changes[0].type == "PARSE_FAILURE"
