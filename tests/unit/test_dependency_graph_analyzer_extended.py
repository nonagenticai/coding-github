"""Extended tests for DependencyGraphAnalyzer – covering uncovered lines."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import ChangeType, FileChange, FileType
from coding_client.models.static_analysis import StaticAnalysisChangeType

from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraph,
    DependencyGraphAnalyzer,
    DependencyNode,
)


@pytest.fixture
def analyzer():
    return DependencyGraphAnalyzer()


# ------------------------------------------------------------------ #
# build_dependency_graph – dependent-file processing (lines 138-149)
# ------------------------------------------------------------------ #
async def test_build_graph_processes_dependent_files(analyzer):
    """When a file A is already in the import graph and imports module B,
    changing B should cause A to be analysed as a dependent file."""
    # Pre-populate the import graph so _find_dependent_files returns something
    analyzer.dependency_graph.imports["src/app/routes.py"] = {"app.models"}

    # Change the module that routes.py imports
    changes = [
        FileChange(
            filename="src/app/models.py",
            file_type=FileType.PYTHON,
            change_type=StaticAnalysisChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="",
            after_content="class User:\n    pass\n",
        ),
    ]

    graph = await analyzer.build_dependency_graph(changes)

    # The dependent file should have been picked up and analysed
    assert "src/app/routes.py" in graph.imports


async def test_build_graph_dependent_file_gets_mock_change(analyzer):
    """Dependent files should be wrapped in a FileChange with correct file_type
    and analysed for their own dependencies."""
    # Seed the imports so that "lib/helpers.js" looks like it imports "lib/utils"
    analyzer.dependency_graph.imports["lib/helpers.js"] = {"lib/utils"}

    changes = [
        FileChange(
            filename="lib/utils.js",
            file_type=FileType.JAVASCRIPT,
            change_type=StaticAnalysisChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="",
            after_content="export const add = (a, b) => a + b;\n",
        ),
    ]

    graph = await analyzer.build_dependency_graph(changes)
    # helpers.js should have been processed as a dependent
    assert "lib/helpers.js" in graph.imports


# ------------------------------------------------------------------ #
# _analyze_file_dependencies – unknown extension early-return (line 161)
# ------------------------------------------------------------------ #
async def test_unknown_extension_returns_early(analyzer):
    """Files with unsupported extensions should be silently skipped."""
    changes = [
        FileChange(
            filename="data/image.png",
            file_type=FileType.UNKNOWN,
            after_content="binary content",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        ),
    ]
    graph = await analyzer.build_dependency_graph(changes)
    assert "data/image.png" not in graph.imports


# ------------------------------------------------------------------ #
# _extract_imports – line continuation (lines 187-188)
# ------------------------------------------------------------------ #
async def test_extract_imports_line_continuation(analyzer):
    content = "from mypackage import (\n    ClassA,\n    ClassB,\n)\n"
    imports = analyzer._extract_imports(content, analyzer.python_import_patterns)
    assert "mypackage" in imports


async def test_extract_imports_backslash_continuation(analyzer):
    content = "import os, \\\n    sys\n"
    imports = analyzer._extract_imports(content, analyzer.python_import_patterns)
    assert len(imports) >= 1


# ------------------------------------------------------------------ #
# _extract_imports – comment skipping (lines 200, 204)
# ------------------------------------------------------------------ #
async def test_extract_imports_skips_hash_comments(analyzer):
    content = "# import os\nimport sys\n"
    imports = analyzer._extract_imports(content, analyzer.python_import_patterns)
    assert "sys" in imports
    assert "os" not in imports


async def test_extract_imports_skips_slash_slash_comments(analyzer):
    content = '// import foo from "bar";\nimport baz from "qux";\n'
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    assert "bar" not in imports


async def test_extract_imports_skips_block_comment_start(analyzer):
    content = '/* import foo from "bar"; */\nimport baz from "qux";\n'
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    # "bar" should be skipped (block comment line), "qux" should be found
    assert "qux" in imports


async def test_extract_imports_skips_multiline_block_comment_start(analyzer):
    """A line with /* in the middle (not at start) without */ should be
    skipped by the multi-line comment check at line 203-204."""
    content = 'var x = 1; /* start of comment\nimport baz from "qux";\n'
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    # The first line has /* mid-line without */ → skipped by line 204
    # "qux" should still be found on the next line
    assert "qux" in imports


# ------------------------------------------------------------------ #
# _extract_imports – comma-separated imports (lines 221-226)
# ------------------------------------------------------------------ #
async def test_extract_imports_comma_separated(analyzer):
    content = "import os, sys, json\n"
    imports = analyzer._extract_imports(content, analyzer.python_import_patterns)
    assert "os" in imports
    assert "sys" in imports
    assert "json" in imports


async def test_extract_imports_comma_separated_skips_relative(analyzer):
    """Modules starting with '.' in comma-separated list are excluded."""
    content = "import os, .relative_mod\n"
    imports = analyzer._extract_imports(content, analyzer.python_import_patterns)
    assert "os" in imports
    assert ".relative_mod" not in imports


# ------------------------------------------------------------------ #
# _analyze_python_dependencies – SyntaxError (lines 238-240)
# ------------------------------------------------------------------ #
async def test_python_ast_syntax_error(analyzer):
    """Invalid Python should not crash but log a warning and skip."""
    changes = [
        FileChange(
            filename="bad.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="",
            after_content="def broken(:\n    pass\n",
        ),
    ]
    graph = await analyzer.build_dependency_graph(changes)
    # Should still have imports parsed via regex even though AST failed
    assert "bad.py" in graph.imports


# ------------------------------------------------------------------ #
# _classify_file_type (lines 301-323)
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "path, expected",
    [
        ("app.py", FileType.PYTHON),
        ("index.js", FileType.JAVASCRIPT),
        ("Component.jsx", FileType.JAVASCRIPT),
        ("app.ts", FileType.TYPESCRIPT),
        ("Component.tsx", FileType.TYPESCRIPT),
        ("README.md", FileType.MARKDOWN),
        ("config.yml", FileType.YAML),
        ("config.yaml", FileType.YAML),
        ("data.json", FileType.JSON),
        ("index.html", FileType.HTML),
        ("page.htm", FileType.HTML),
        ("style.css", FileType.CSS),
        ("Dockerfile", FileType.DOCKERFILE),
        ("Dockerfile.dev", FileType.DOCKERFILE),
        ("Dockerfile.prod", FileType.DOCKERFILE),
        ("image.png", FileType.UNKNOWN),
        ("archive.tar.gz", FileType.UNKNOWN),
    ],
)
def test_classify_file_type(analyzer, path, expected):
    assert analyzer._classify_file_type(path) == expected


# ------------------------------------------------------------------ #
# _file_path_to_module_name (lines 335, 348, 354-356, 362-393)
# ------------------------------------------------------------------ #
def test_module_name_init_file(analyzer):
    """__init__.py should use parent directory name."""
    result = analyzer._file_path_to_module_name("src/mypackage/__init__.py")
    assert "mypackage" in result


def test_module_name_index_file(analyzer):
    """index.js should use parent directory name."""
    result = analyzer._file_path_to_module_name("lib/components/index.js")
    assert "components" in result


def test_module_name_mod_file(analyzer):
    """mod.rs should use parent directory name."""
    result = analyzer._file_path_to_module_name("src/utils/mod.rs")
    assert "utils" in result


def test_module_name_python_with_src(analyzer):
    """Python files under src/ should produce dotted module names."""
    result = analyzer._file_path_to_module_name("src/myapp/models/user.py")
    assert result == "myapp.models.user"


def test_module_name_python_fallback(analyzer):
    """Python files without standard src dir use full path fallback."""
    result = analyzer._file_path_to_module_name("custom/path/module.py")
    assert "module" in result
    assert "." in result  # Should be dotted


def test_module_name_js_node_modules(analyzer):
    """node_modules paths should return the package name."""
    result = analyzer._file_path_to_module_name("node_modules/lodash/index.js")
    # index.js → parent name "lodash", but node_modules handling takes priority
    assert "lodash" in result


def test_module_name_js_node_modules_regular(analyzer):
    """node_modules with a regular file name."""
    result = analyzer._file_path_to_module_name("node_modules/express/lib/router.js")
    assert result == "express"


def test_module_name_js_regular(analyzer):
    """Regular JS files use slash-separated paths."""
    result = analyzer._file_path_to_module_name("lib/utils/helpers.js")
    assert "lib/utils/helpers" == result


def test_module_name_go_with_src(analyzer):
    """Go files under src/ should produce slash-separated import paths."""
    result = analyzer._file_path_to_module_name("src/github.com/user/pkg/main.go")
    assert "github.com/user/pkg/main.go" == result


def test_module_name_go_without_src(analyzer):
    """Go files without src/ return str(path)."""
    result = analyzer._file_path_to_module_name("cmd/server/main.go")
    assert "cmd" in result or "main" in result


def test_module_name_rust_with_src(analyzer):
    """Rust files under src/ should produce ::-separated module paths."""
    result = analyzer._file_path_to_module_name("src/utils/parser.rs")
    assert result == "utils::parser"


def test_module_name_rust_without_src(analyzer):
    """Rust files without src/ return just the stem."""
    result = analyzer._file_path_to_module_name("standalone.rs")
    assert result == "standalone"


def test_module_name_java_with_src(analyzer):
    """Java files under src/ should produce dotted namespace."""
    result = analyzer._file_path_to_module_name("src/main/java/com/example/App.java")
    # Should start after 'java' since that's one of the src_dirs checked
    assert "App" in result


def test_module_name_java_without_src(analyzer):
    """Java files without matching src dirs return just the name."""
    result = analyzer._file_path_to_module_name("MyClass.java")
    assert result == "MyClass"


def test_module_name_csharp(analyzer):
    """C# files with standard src dir."""
    result = analyzer._file_path_to_module_name("src/Controllers/HomeController.cs")
    assert "HomeController" in result


def test_module_name_unknown_extension(analyzer):
    """Unknown extensions return just the stem."""
    result = analyzer._file_path_to_module_name("readme.txt")
    assert result == "readme"


# ------------------------------------------------------------------ #
# _search_for_imports – various match branches (lines 411-447)
# ------------------------------------------------------------------ #
def test_search_imports_direct_match(analyzer):
    """Direct module name match."""
    analyzer.dependency_graph.imports["a.py"] = {"mylib"}
    result = analyzer._search_for_imports("mylib")
    assert "a.py" in result


def test_search_imports_parent_module(analyzer):
    """imported_module starts with module_name + '.'"""
    analyzer.dependency_graph.imports["a.py"] = {"mylib.submodule"}
    result = analyzer._search_for_imports("mylib")
    assert "a.py" in result


def test_search_imports_submodule(analyzer):
    """module_name starts with imported_module + '.'"""
    analyzer.dependency_graph.imports["a.py"] = {"mylib"}
    result = analyzer._search_for_imports("mylib.submodule")
    assert "a.py" in result


def test_search_imports_base_module_match(analyzer):
    """Base module name of a dotted module_name matches within imported parts."""
    analyzer.dependency_graph.imports["a.py"] = {"pkg.utils"}
    result = analyzer._search_for_imports("other.utils")
    assert "a.py" in result


def test_search_imports_js_slash_match(analyzer):
    """JavaScript-style imports with / separators."""
    analyzer.dependency_graph.imports["a.js"] = {"lib/utils/helpers"}
    result = analyzer._search_for_imports("lib/utils")
    assert "a.js" in result


def test_search_imports_rust_coloncolon_match(analyzer):
    """Rust-style imports with :: separators – direct match."""
    analyzer.dependency_graph.imports["a.rs"] = {"crate::utils::parser"}
    result = analyzer._search_for_imports("crate::utils::parser")
    assert "a.rs" in result


def test_search_imports_rust_coloncolon_normalized_match(analyzer):
    """Rust :: vs dot notation – normalization match (lines 445-447).

    When module_name uses dot notation and imported_module uses ::,
    they should match after normalizing :: to dot.
    """
    # imported_module uses ::, module_name uses dots.
    # Direct match at line 403 won't hit (different strings).
    # Parent/sub checks won't match.  Dot-based check at line 425
    # splits "std::collections" by "." → ["std::collections"] which
    # doesn't contain "collections".  No "/" present.
    # At line 443 "::" in imported_module → True, and
    # "std.collections" == "std.collections" after replacement → match.
    analyzer.dependency_graph.imports["b.rs"] = {"std::collections"}
    result = analyzer._search_for_imports("std.collections")
    assert "b.rs" in result


def test_search_imports_no_match(analyzer):
    """No imports match."""
    analyzer.dependency_graph.imports["a.py"] = {"unrelated"}
    result = analyzer._search_for_imports("mylib")
    assert len(result) == 0


def test_search_imports_empty_graph(analyzer):
    """Empty import graph returns nothing."""
    result = analyzer._search_for_imports("anything")
    assert len(result) == 0


# ------------------------------------------------------------------ #
# get_dependency_graph / reset_graph (lines 461, 465)
# ------------------------------------------------------------------ #
def test_get_dependency_graph(analyzer):
    graph = analyzer.get_dependency_graph()
    assert isinstance(graph, DependencyGraph)
    assert graph is analyzer.dependency_graph


def test_reset_graph(analyzer):
    analyzer.dependency_graph.imports["foo.py"] = {"bar"}
    analyzer.reset_graph()
    graph = analyzer.get_dependency_graph()
    assert graph.imports == {}
    assert graph.nodes == {}
    assert graph.modules == {}


# ------------------------------------------------------------------ #
# Multi-language import patterns
# ------------------------------------------------------------------ #
async def test_typescript_type_import(analyzer):
    content = 'import type { Foo } from "bar";\n'
    imports = analyzer._extract_imports(content, analyzer.typescript_import_patterns)
    assert "bar" in imports


async def test_go_import(analyzer):
    content = 'import "fmt"\n'
    imports = analyzer._extract_imports(content, analyzer.go_import_patterns)
    assert "fmt" in imports


async def test_rust_use_import(analyzer):
    content = "use std::collections::HashMap;\n"
    imports = analyzer._extract_imports(content, analyzer.rust_import_patterns)
    assert "std::collections::HashMap" in imports


async def test_rust_extern_crate(analyzer):
    content = "extern crate serde;\n"
    imports = analyzer._extract_imports(content, analyzer.rust_import_patterns)
    assert "serde" in imports


async def test_java_import(analyzer):
    content = "import java.util.List;\n"
    imports = analyzer._extract_imports(content, analyzer.java_import_patterns)
    assert "java.util.List" in imports


async def test_java_static_import(analyzer):
    content = "import static org.junit.Assert.assertEquals;\n"
    imports = analyzer._extract_imports(content, analyzer.java_import_patterns)
    assert "org.junit.Assert.assertEquals" in imports


async def test_csharp_using(analyzer):
    content = "using System.Collections.Generic;\n"
    imports = analyzer._extract_imports(content, analyzer.csharp_import_patterns)
    assert "System.Collections.Generic" in imports


async def test_csharp_using_static(analyzer):
    content = "using static System.Math;\n"
    imports = analyzer._extract_imports(content, analyzer.csharp_import_patterns)
    assert "System.Math" in imports


async def test_js_require_import(analyzer):
    content = "const express = require('express');\n"
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    assert "express" in imports


async def test_js_dynamic_import(analyzer):
    content = "import('lodash');\n"
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    assert "lodash" in imports


async def test_js_reexport(analyzer):
    content = 'export { default } from "utils";\n'
    imports = analyzer._extract_imports(content, analyzer.javascript_import_patterns)
    assert "utils" in imports


# ------------------------------------------------------------------ #
# _analyze_file_dependencies – file with before_content only
# ------------------------------------------------------------------ #
async def test_analyze_uses_before_content_when_no_after(analyzer):
    """When after_content is None, before_content should be used."""
    changes = [
        FileChange(
            filename="old.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
            before_content="import os\n",
            after_content=None,
        ),
    ]
    graph = await analyzer.build_dependency_graph(changes)
    assert "os" in graph.imports.get("old.py", set())


async def test_analyze_file_no_content(analyzer):
    """File with no content should still be processed without error."""
    changes = [
        FileChange(
            filename="empty.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
            before_content=None,
            after_content=None,
        ),
    ]
    graph = await analyzer.build_dependency_graph(changes)
    # No imports recorded because there's no content
    assert graph.imports.get("empty.py") is None or graph.imports.get("empty.py") == set()


# ------------------------------------------------------------------ #
# DependencyNode / DependencyGraph dataclass basics
# ------------------------------------------------------------------ #
def test_dependency_node_defaults():
    node = DependencyNode(name="foo", file_path="a.py", node_type="function")
    assert node.dependencies == set()
    assert node.dependents == set()
    assert node.metadata == {}


def test_dependency_graph_defaults():
    graph = DependencyGraph()
    assert graph.nodes == {}
    assert graph.modules == {}
    assert graph.imports == {}
