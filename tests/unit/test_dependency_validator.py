"""Unit tests for DependencyValidator."""

from __future__ import annotations

from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraph,
    DependencyNode,
)
from services.pr_validation.analyzers.impact.dependency_validator import (
    DependencyValidator,
)


def _make_graph(
    imports: dict[str, set[str]] | None = None,
    modules: dict[str, set[str]] | None = None,
    nodes: dict[str, DependencyNode] | None = None,
) -> DependencyGraph:
    """Helper to build a DependencyGraph with given data."""
    graph = DependencyGraph()
    graph.imports = imports or {}
    graph.modules = modules or {}
    graph.nodes = nodes or {}
    return graph


# ---------------------------------------------------------------------------
# 1. Safe version bump – no circular deps, no missing deps
# ---------------------------------------------------------------------------
def test_safe_version_bump():
    graph = _make_graph(
        imports={
            "src/main.py": {"os", "json"},
            "src/utils.py": {"logging"},
        },
    )
    validator = DependencyValidator(graph)
    circular = validator.detect_circular_dependencies()
    assert circular == []
    missing = validator.validate_dependency_chains({"src/main.py"})
    assert missing == []


# ---------------------------------------------------------------------------
# 2. Unsafe dependency change – circular dependency detected
# ---------------------------------------------------------------------------
def test_unsafe_dependency_change():
    # Create two modules that import each other
    graph = _make_graph(
        imports={
            "src/a.py": {"b"},
            "src/b.py": {"a"},
        },
        modules={
            "src/a.py": {"func_a"},
            "src/b.py": {"func_b"},
        },
    )
    validator = DependencyValidator(graph)
    circular = validator.detect_circular_dependencies()
    # Circular dependency should be detected between a and b
    assert len(circular) > 0


# ---------------------------------------------------------------------------
# 3. Missing lockfile warning – missing internal dependency
# ---------------------------------------------------------------------------
def test_missing_dependency_warning():
    # Create a node in src/main.py that depends on "mylib_func", which
    # doesn't exist in graph.nodes or graph.modules.
    node = DependencyNode(
        name="main_func",
        file_path="src/main.py",
        node_type="function",
        dependencies={"mylib_func"},
    )
    graph = _make_graph(
        imports={"src/main.py": set()},
        modules={},
        nodes={"src/main.py::main_func": node},
    )
    validator = DependencyValidator(graph)
    missing = validator.validate_dependency_chains({"src/main.py"})
    # mylib_func is an undefined reference
    assert any("mylib_func" in dep for dep in missing)


# ---------------------------------------------------------------------------
# 4. Malformed manifest – empty graph produces clean report
# ---------------------------------------------------------------------------
def test_malformed_manifest_empty_graph():
    graph = _make_graph()
    validator = DependencyValidator(graph)
    report = validator.generate_dependency_report()
    assert report["circular_dependencies"]["count"] == 0
    assert report["overall_health_score"] == 100.0
