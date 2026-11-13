"""Extended tests for breaking_change_detector.py covering uncovered lines.

Target lines: 111, 140, 167-169, 191-193, 294, 320-323, 327-330, 394-396,
              454-466, 473, 556-557, 649-658
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.semantic.breaking_change_detector import (
    APIElement,
    BreakingChangeDetector,
)


def _fc(filename="src/app.py", **kw):
    """Shortcut to build a FileChange with sensible defaults."""
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 1)
    kw.setdefault("deletions", 0)
    kw.setdefault("patch", "")
    return FileChange(filename=filename, **kw)


@pytest.fixture
def detector():
    return BreakingChangeDetector()


# -- Line 111: return [] for file_type not in language_handlers ------


class TestDetectBreakingChangesReturnEmpty:
    """Line 111: handler not found for file_type -> return []."""

    async def test_yaml_file_returns_empty(self, detector):
        """YAML has no language handler, so detect_breaking_changes returns []
        even though _is_analyzable_file would already filter it out.
        We need file_type in language_handlers AND not binary AND content present
        for _is_analyzable_file to pass.  YAML is not in language_handlers,
        so _is_analyzable_file returns False and we get [] from line 104.

        To hit line 111 specifically we need _is_analyzable_file True but no handler.
        That's impossible with the current code since _is_analyzable_file checks
        file_type in self.language_handlers.  So we patch _is_analyzable_file.
        """
        fc = _fc(
            "data.yaml",
            file_type=FileType.YAML,
            before_content="key: old",
            after_content="key: new",
        )
        with patch.object(detector, "_is_analyzable_file", return_value=True):
            # handler = self.language_handlers.get(FileType.YAML) -> None -> line 111
            result = await detector.detect_breaking_changes(fc)
        assert result == []


# -- Line 140: _is_analyzable_file returns False for binary files ------


class TestIsAnalyzableFileBinary:
    """Line 140: _is_analyzable_file returns handle_parse_failure path."""

    async def test_binary_python_file_not_analyzable(self, detector):
        """binary=True means _is_analyzable_file returns False."""
        fc = _fc(
            "module.py",
            before_content="x = 1",
            after_content="x = 2",
            binary=True,
        )
        assert detector._is_analyzable_file(fc) is False

    async def test_no_before_content_not_analyzable(self, detector):
        fc = _fc(
            "module.py",
            before_content=None,
            after_content="x = 2",
        )
        assert detector._is_analyzable_file(fc) is False

    async def test_no_after_content_not_analyzable(self, detector):
        fc = _fc(
            "module.py",
            before_content="x = 1",
            after_content=None,
        )
        assert detector._is_analyzable_file(fc) is False

    async def test_parse_failure_returns_breaking_change(self, detector):
        """Line 140: when before_ast or after_ast is None -> _handle_parse_failure."""
        fc = _fc(
            "broken.py",
            before_content="def foo(:\n    pass",  # syntax error
            after_content="def foo():\n    pass",
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert len(result) >= 1
        assert result[0].type == "PARSE_FAILURE"


# -- Lines 167-169: exception handler in _analyze_python_breaking_changes --


class TestAnalyzePythonBreakingChangesException:
    """Lines 167-169: exception during analysis returns ANALYSIS_ERROR."""

    async def test_exception_returns_analysis_error(self, detector):
        fc = _fc(
            "module.py",
            before_content="def foo():\n    pass",
            after_content="def foo():\n    pass",
            binary=False,
        )
        with patch.object(detector, "_extract_python_api", side_effect=RuntimeError("boom")):
            result = await detector._analyze_python_breaking_changes(fc)

        assert len(result) == 1
        assert result[0].type == "ANALYSIS_ERROR"
        assert result[0].severity == "MEDIUM"
        assert "boom" in result[0].description


# -- Lines 191-193: AsyncFunctionDef extraction in _extract_python_api ------


class TestExtractAsyncFunction:
    """Lines 191-193: AsyncFunctionDef branch in _extract_python_api."""

    def test_async_function_extracted(self, detector):
        code = "async def fetch_data(url: str) -> dict:\n    pass"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)

        assert "fetch_data" in api
        el = api["fetch_data"]
        assert el.type == "function"
        assert el.metadata["is_async"] is True
        assert "url" in el.metadata["args"]

    def test_private_async_function_not_extracted(self, detector):
        code = "async def _internal():\n    pass"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        assert "_internal" not in api

    def test_dunder_async_function_extracted(self, detector):
        """__init__-like async dunder methods should be public."""
        code = "class Foo:\n    async def __call__(self):\n        pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        # __call__ is walked via ast.walk, so it shows as a standalone function
        assert "__call__" in api


# -- Line 294: removed element detection in _compare_python_apis ------


class TestCompareRemovedElement:
    """Line 294: removed element generates FUNCTION_REMOVED / CONSTANT_REMOVED etc."""

    def test_removed_constant(self, detector):
        before = {
            "MAX_SIZE": APIElement(
                name="MAX_SIZE",
                type="constant",
                signature="MAX_SIZE = 100",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }
        after = {}
        issues = detector._compare_python_apis(before, after)
        assert len(issues) == 1
        assert issues[0].issue_type == "CONSTANT_REMOVED"
        assert issues[0].severity == "HIGH"
        assert issues[0].breaking_score == 0.9

    def test_removed_class(self, detector):
        before = {
            "Widget": APIElement(
                name="Widget",
                type="class",
                signature="class Widget()",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }
        after = {}
        issues = detector._compare_python_apis(before, after)
        assert len(issues) == 1
        assert issues[0].issue_type == "CLASS_REMOVED"


# -- Lines 320-323: type change detection (function->class) ----------


class TestTypeChangeDetection:
    """Lines 320-323 (via 374-386): ELEMENT_TYPE_CHANGED when type changes."""

    def test_function_to_class(self, detector):
        before = {
            "Item": APIElement(
                name="Item",
                type="function",
                signature="def Item()",
                location=(1, 0),
                visibility="public",
                metadata={"args": [], "defaults": 0},
            )
        }
        after = {
            "Item": APIElement(
                name="Item",
                type="class",
                signature="class Item()",
                location=(1, 0),
                visibility="public",
                metadata={"bases": [], "methods": {}},
            )
        }
        issues = detector._compare_python_apis(before, after)
        assert any(i.issue_type == "ELEMENT_TYPE_CHANGED" for i in issues)
        type_issue = next(i for i in issues if i.issue_type == "ELEMENT_TYPE_CHANGED")
        assert type_issue.breaking_score == 1.0
        assert type_issue.severity == "HIGH"

    def test_constant_to_function(self, detector):
        before = {
            "CONFIG": APIElement(
                name="CONFIG",
                type="constant",
                signature="CONFIG = {}",
                location=(1, 0),
                visibility="public",
                metadata={},
            )
        }
        after = {
            "CONFIG": APIElement(
                name="CONFIG",
                type="function",
                signature="def CONFIG()",
                location=(1, 0),
                visibility="public",
                metadata={"args": [], "defaults": 0},
            )
        }
        issues = detector._compare_python_apis(before, after)
        assert any(i.issue_type == "ELEMENT_TYPE_CHANGED" for i in issues)


# -- Lines 327-330: class signature change -> _analyze_class_changes ---


class TestClassSignatureChange:
    """Lines 327-330 (via 394-396): class signature changed triggers _analyze_class_changes."""

    def test_class_with_changed_signature_calls_analyze_class_changes(self, detector):
        before = {
            "MyClass": APIElement(
                name="MyClass",
                type="class",
                signature="class MyClass(Base)",
                location=(1, 0),
                visibility="public",
                metadata={"bases": ["Base"], "methods": {"run": "def run(self)"}},
            )
        }
        after = {
            "MyClass": APIElement(
                name="MyClass",
                type="class",
                signature="class MyClass(NewBase)",
                location=(1, 0),
                visibility="public",
                metadata={"bases": ["NewBase"], "methods": {"run": "def run(self)"}},
            )
        }
        issues = detector._compare_python_apis(before, after)
        assert any(i.issue_type == "CLASS_INHERITANCE_REMOVED" for i in issues)


# -- Lines 394-396: _analyze_function_signature_changes for removed args --


class TestFunctionArgsRemoved:
    """Lines 394-396: function signature change with fewer args -> FUNCTION_ARGS_REMOVED."""

    def test_args_removed_issue(self, detector):
        before = APIElement(
            name="process",
            type="function",
            signature="def process(a, b, c)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a", "b", "c"], "defaults": 0},
        )
        after = APIElement(
            name="process",
            type="function",
            signature="def process(a)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a"], "defaults": 0},
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert any(i.issue_type == "FUNCTION_ARGS_REMOVED" for i in issues)
        removed = next(i for i in issues if i.issue_type == "FUNCTION_ARGS_REMOVED")
        assert removed.breaking_score == 0.8
        assert "2 vs 3" in removed.description or "1 vs 3" in removed.description


# -- Lines 454-466: argument reordering detection ---------------------


class TestArgumentReordering:
    """Lines 454-466: FUNCTION_ARGS_REORDERED when positional args swap."""

    def test_args_reordered(self, detector):
        before = APIElement(
            name="draw",
            type="function",
            signature="def draw(x, y, color)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["x", "y", "color"], "defaults": 0},
        )
        after = APIElement(
            name="draw",
            type="function",
            signature="def draw(color, x, y)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["color", "x", "y"], "defaults": 0},
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert any(i.issue_type == "FUNCTION_ARGS_REORDERED" for i in issues)
        reorder = next(i for i in issues if i.issue_type == "FUNCTION_ARGS_REORDERED")
        assert reorder.breaking_score == 0.7
        assert "position 0" in reorder.description

    def test_first_arg_renamed(self, detector):
        """Renaming an arg (same count) also triggers FUNCTION_ARGS_REORDERED."""
        before = APIElement(
            name="send",
            type="function",
            signature="def send(msg)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["msg"], "defaults": 0},
        )
        after = APIElement(
            name="send",
            type="function",
            signature="def send(payload)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["payload"], "defaults": 0},
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert any(i.issue_type == "FUNCTION_ARGS_REORDERED" for i in issues)

    def test_no_reorder_when_same(self, detector):
        """No reorder issue when args are identical."""
        before = APIElement(
            name="f",
            type="function",
            signature="def f(a, b)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a", "b"], "defaults": 0, "is_async": False},
        )
        after = APIElement(
            name="f",
            type="function",
            signature="def f(a, b)",
            location=(1, 0),
            visibility="public",
            metadata={"args": ["a", "b"], "defaults": 0, "is_async": False},
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert not any(i.issue_type == "FUNCTION_ARGS_REORDERED" for i in issues)


# -- Line 473: return type change detection ---------------------------


class TestReturnTypeChange:
    """Line 473: FUNCTION_RETURN_TYPE_CHANGED when return annotations differ."""

    def test_return_type_changed(self, detector):
        before = APIElement(
            name="compute",
            type="function",
            signature="def compute() -> int",
            location=(1, 0),
            visibility="public",
            metadata={
                "args": [],
                "defaults": 0,
                "is_async": False,
                "return_annotation": "int",
            },
        )
        after = APIElement(
            name="compute",
            type="function",
            signature="def compute() -> float",
            location=(1, 0),
            visibility="public",
            metadata={
                "args": [],
                "defaults": 0,
                "is_async": False,
                "return_annotation": "float",
            },
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert any(i.issue_type == "FUNCTION_RETURN_TYPE_CHANGED" for i in issues)
        ret = next(i for i in issues if i.issue_type == "FUNCTION_RETURN_TYPE_CHANGED")
        assert ret.severity == "MEDIUM"
        assert ret.breaking_score == 0.5
        assert "int" in ret.description and "float" in ret.description

    def test_return_type_not_flagged_when_none(self, detector):
        """No issue when before or after return_annotation is None."""
        before = APIElement(
            name="f",
            type="function",
            signature="def f()",
            location=(1, 0),
            visibility="public",
            metadata={
                "args": [],
                "defaults": 0,
                "is_async": False,
                "return_annotation": None,
            },
        )
        after = APIElement(
            name="f",
            type="function",
            signature="def f() -> str",
            location=(1, 0),
            visibility="public",
            metadata={
                "args": [],
                "defaults": 0,
                "is_async": False,
                "return_annotation": "str",
            },
        )
        issues = detector._analyze_function_signature_changes(before, after)
        assert not any(i.issue_type == "FUNCTION_RETURN_TYPE_CHANGED" for i in issues)


# -- Lines 556-557: CLASS_METHOD_SIGNATURE_CHANGED detection ----------


class TestClassMethodSignatureChanged:
    """Lines 556-557: method signature changes in a class."""

    def test_method_signature_changed(self, detector):
        before = APIElement(
            name="Service",
            type="class",
            signature="class Service()",
            location=(1, 0),
            visibility="public",
            metadata={
                "bases": [],
                "methods": {
                    "run": "def run(self, timeout: int)",
                    "stop": "def stop(self)",
                },
            },
        )
        after = APIElement(
            name="Service",
            type="class",
            signature="class Service()",
            location=(1, 0),
            visibility="public",
            metadata={
                "bases": [],
                "methods": {
                    "run": "def run(self, timeout: float, retries: int)",
                    "stop": "def stop(self)",
                },
            },
        )
        issues = detector._analyze_class_changes(before, after)
        assert any(i.issue_type == "CLASS_METHOD_SIGNATURE_CHANGED" for i in issues)
        changed = next(i for i in issues if i.issue_type == "CLASS_METHOD_SIGNATURE_CHANGED")
        assert changed.severity == "MEDIUM"
        assert changed.breaking_score == 0.6
        assert "Service.run" == changed.element_name

    def test_multiple_methods_changed(self, detector):
        before = APIElement(
            name="Repo",
            type="class",
            signature="class Repo()",
            location=(1, 0),
            visibility="public",
            metadata={
                "bases": [],
                "methods": {
                    "get": "def get(self, id)",
                    "put": "def put(self, id, data)",
                },
            },
        )
        after = APIElement(
            name="Repo",
            type="class",
            signature="class Repo()",
            location=(1, 0),
            visibility="public",
            metadata={
                "bases": [],
                "methods": {
                    "get": "def get(self, id, fields)",
                    "put": "def put(self, id, data, validate)",
                },
            },
        )
        issues = detector._analyze_class_changes(before, after)
        sig_issues = [i for i in issues if i.issue_type == "CLASS_METHOD_SIGNATURE_CHANGED"]
        assert len(sig_issues) == 2


# -- Lines 649-658: _ast_to_string fallback and _handle_parse_failure --


class TestAstToStringFallback:
    """Lines 649-658: fallback branches in _ast_to_string."""

    def test_ast_name_fallback(self, detector):
        """When ast.unparse raises, fallback returns node.id for Name."""
        node = ast.Name(id="foo")
        with patch("ast.unparse", side_effect=AttributeError):
            result = detector._ast_to_string(node)
        assert result == "foo"

    def test_ast_constant_fallback(self, detector):
        """Fallback returns repr(value) for Constant."""
        node = ast.Constant(value=42)
        with patch("ast.unparse", side_effect=ValueError):
            result = detector._ast_to_string(node)
        assert result == "42"

    def test_ast_attribute_fallback(self, detector):
        """Fallback for Attribute recursively calls _ast_to_string."""
        node = ast.Attribute(
            value=ast.Name(id="os"),
            attr="path",
        )
        with patch("ast.unparse", side_effect=AttributeError):
            result = detector._ast_to_string(node)
        assert result == "os.path"

    def test_ast_unknown_fallback(self, detector):
        """Fallback for unknown node type returns <TypeName>."""
        node = ast.BoolOp(op=ast.And(), values=[])
        with patch("ast.unparse", side_effect=AttributeError):
            result = detector._ast_to_string(node)
        assert result == "<BoolOp>"

    def test_handle_parse_failure(self, detector):
        """_handle_parse_failure returns a PARSE_FAILURE BreakingChange."""
        result = detector._handle_parse_failure("mod.py", ["err1", "err2", "err3", "err4"])
        assert len(result) == 1
        bc = result[0]
        assert bc.type == "PARSE_FAILURE"
        assert bc.severity == "MEDIUM"
        assert bc.file_path == "mod.py"
        # Only first 3 errors included
        assert "err1" in bc.description
        assert "err3" in bc.description


# -- Integration-level: full flow hitting multiple uncovered lines ------


class TestFullFlowIntegration:
    """End-to-end tests through detect_breaking_changes exercising the uncovered paths."""

    async def test_async_function_removed(self, detector):
        """Removing an async function is detected (exercises lines 191-193, 294)."""
        before = "async def fetch(url: str) -> dict:\n    pass\n\ndef sync_helper():\n    pass\n"
        after = "def sync_helper():\n    pass\n"
        fc = _fc(
            "api.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert any(bc.type == "FUNCTION_REMOVED" for bc in result)
        removed = next(bc for bc in result if bc.type == "FUNCTION_REMOVED")
        assert removed.affected_symbol == "fetch"

    async def test_function_to_class_full_flow(self, detector):
        """Function -> class detected through full flow (lines 320-323)."""
        before = "def Widget():\n    pass\n"
        after = "class Widget:\n    def __init__(self):\n        pass\n"
        fc = _fc(
            "widgets.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert any(bc.type == "ELEMENT_TYPE_CHANGED" for bc in result)

    async def test_class_signature_change_full_flow(self, detector):
        """Class base class change detected through full flow (lines 327-330)."""
        before = "class Handler(BaseHandler):\n    def handle(self, req):\n        pass\n"
        after = "class Handler(NewBaseHandler):\n    def handle(self, req):\n        pass\n"
        fc = _fc(
            "handlers.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert any(bc.type == "CLASS_INHERITANCE_REMOVED" for bc in result)

    async def test_arg_reorder_full_flow(self, detector):
        """Arg reordering detected through full flow (lines 454-466)."""
        before = "def render(width, height, color):\n    pass\n"
        after = "def render(color, width, height):\n    pass\n"
        fc = _fc(
            "draw.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert any(bc.type == "FUNCTION_ARGS_REORDERED" for bc in result)

    async def test_return_type_change_full_flow(self, detector):
        """Return type change detected through full flow (line 473)."""
        before = "def compute(x: int) -> int:\n    return x\n"
        after = "def compute(x: int) -> str:\n    return str(x)\n"
        fc = _fc(
            "calc.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert any(bc.type == "FUNCTION_RETURN_TYPE_CHANGED" for bc in result)

    async def test_class_method_signature_change_full_flow(self, detector):
        """Class method signature change detected through full flow (lines 556-557).

        _compare_python_apis only calls _analyze_class_changes when the class
        signature string differs (line 388).  The signature is built from bases,
        so we must also change a base class to trigger this path.
        """
        before = "class Cache(BaseCache):\n    def get(self, key: str) -> str:\n        pass\n"
        after = (
            "class Cache(NewBaseCache):\n"
            "    def get(self, key: str, default: str = None) -> str:\n"
            "        pass\n"
        )
        fc = _fc(
            "cache.py",
            before_content=before,
            after_content=after,
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        # The class has changed bases AND a method signature
        assert any(bc.type == "CLASS_METHOD_SIGNATURE_CHANGED" for bc in result)

    async def test_both_parse_failures(self, detector):
        """Both before and after have syntax errors -> PARSE_FAILURE (line 140)."""
        fc = _fc(
            "bad.py",
            before_content="def (\n",
            after_content="class [\n",
            binary=False,
        )
        result = await detector.detect_breaking_changes(fc)
        assert len(result) >= 1
        assert result[0].type == "PARSE_FAILURE"


# -- Edge case: _extract_python_function_signature with *args/**kwargs --


class TestExtractSignatureEdgeCases:
    """Covers _extract_python_function_signature for vararg/kwarg annotations
    (lines 320-323 and 327-330 in the signature builder)."""

    def test_vararg_with_annotation(self, detector):
        code = "def f(*args: int):\n    pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        assert "f" in api
        sig = api["f"].signature
        assert "*args: int" in sig

    def test_kwarg_with_annotation(self, detector):
        code = "def f(**kwargs: str):\n    pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        assert "f" in api
        sig = api["f"].signature
        assert "**kwargs: str" in sig

    def test_vararg_and_kwarg_together(self, detector):
        code = "def f(a, *args: int, **kwargs: str):\n    pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        sig = api["f"].signature
        assert "*args: int" in sig
        assert "**kwargs: str" in sig

    def test_vararg_without_annotation(self, detector):
        code = "def f(*args):\n    pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        sig = api["f"].signature
        assert "*args" in sig
        # no colon
        assert "*args:" not in sig

    def test_kwarg_without_annotation(self, detector):
        code = "def f(**kwargs):\n    pass\n"
        tree = ast.parse(code)
        api = detector._extract_python_api(tree)
        sig = api["f"].signature
        assert "**kwargs" in sig
        assert "**kwargs:" not in sig
