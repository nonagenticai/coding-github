"""Unit tests for DependencyGraphAnalyzer."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraphAnalyzer,
)


@pytest.fixture
def graph_analyzer():
    return DependencyGraphAnalyzer()


# ---------------------------------------------------------------------------
# 1. Graph traversal – build graph and verify nodes/imports
# ---------------------------------------------------------------------------
async def test_graph_traversal(graph_analyzer):
    code = (
        "from myapp.services import UserService\n"
        "\n"
        "class UserController:\n"
        "    def get_user(self):\n"
        "        return UserService().find()\n"
    )
    changes = [
        FileChange(
            filename="src/controllers/user.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    graph = await graph_analyzer.build_dependency_graph(changes)
    # The file should have its imports recorded
    imports = graph.imports.get("src/controllers/user.py", set())
    assert "myapp.services" in imports
    # AST-based analysis should detect the class and function
    assert any(node.name == "UserController" for node in graph.nodes.values())
    assert any(node.name == "get_user" for node in graph.nodes.values())


# ---------------------------------------------------------------------------
# 2. Impacted modules – verify module symbols are recorded
# ---------------------------------------------------------------------------
async def test_impacted_modules(graph_analyzer):
    code_a = "def helper():\n    return 42\n"
    code_b = "from utils import helper\n\ndef main():\n    return helper()\n"
    changes = [
        FileChange(
            filename="src/utils.py",
            file_type=FileType.PYTHON,
            after_content=code_a,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="src/main.py",
            file_type=FileType.PYTHON,
            after_content=code_b,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        ),
    ]
    graph = await graph_analyzer.build_dependency_graph(changes)
    # utils.py should have helper in its module symbols
    assert "helper" in graph.modules.get("src/utils.py", set())
    # main.py imports utils
    assert "utils" in graph.imports.get("src/main.py", set())


# ---------------------------------------------------------------------------
# 3. Circular dependency – graph records mutual imports
# ---------------------------------------------------------------------------
async def test_circular_dependency_recorded(graph_analyzer):
    code_a = "from module_b import func_b\ndef func_a():\n    return func_b()\n"
    code_b = "from module_a import func_a\ndef func_b():\n    return func_a()\n"
    changes = [
        FileChange(
            filename="src/module_a.py",
            file_type=FileType.PYTHON,
            after_content=code_a,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="src/module_b.py",
            file_type=FileType.PYTHON,
            after_content=code_b,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        ),
    ]
    graph = await graph_analyzer.build_dependency_graph(changes)
    # Both files should have imports of each other
    assert "module_b" in graph.imports.get("src/module_a.py", set())
    assert "module_a" in graph.imports.get("src/module_b.py", set())


# ---------------------------------------------------------------------------
# 4. Empty graph – no changes produces empty graph
# ---------------------------------------------------------------------------
async def test_empty_graph(graph_analyzer):
    graph = await graph_analyzer.build_dependency_graph([])
    assert graph.nodes == {}
    assert graph.imports == {}
    assert graph.modules == {}
