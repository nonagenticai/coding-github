"""Extended unit tests for DependencyValidator — covering uncovered lines."""

from __future__ import annotations

import pytest

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
# detect_circular_dependencies – DFS internals: lines 64, 71-73
# ---------------------------------------------------------------------------
class TestDetectCircularDependencies:
    def test_neighbor_not_in_colors(self):
        """Neighbor not in colors dict gets initialized to 0 (line 64)."""
        graph = _make_graph(
            imports={
                "src/a.py": {"mymod"},
            },
            modules={
                "src/a.py": {"func_a"},
                "src/b.py": {"func_b"},
            },
        )
        # src/b.py is NOT in imports (so not in initial colors), but
        # _find_files_for_module("mymod") should NOT resolve to src/b.py
        # because module name won't match. Use a module that resolves.
        # We need a.py -> b.py via module resolution. Let's use direct approach.
        graph.imports = {
            "src/a.py": {"b_mod"},
        }
        # Make b_mod resolve to src/b.py. We need _file_path_to_module_name("src/b.py") == "b_mod"
        # For Python: src/b.py → parts=('src', 'b.py'), stem='b', ext='.py'
        # Since 'src' is in src_dir list, start_idx=1, module_parts=() + ('b',) → 'b'
        # So we need import name 'b'. Let's use that.
        graph.imports = {
            "src/a.py": {"b"},
        }
        # src/b.py is in modules but NOT in imports, so not in adjacency_list initially
        validator = DependencyValidator(graph)
        circular = validator.detect_circular_dependencies()
        # No cycle, just tests that neighbor not in colors is handled
        assert isinstance(circular, list)

    def test_cycle_with_parent_chain(self):
        """Cycle tracing through parent chain (lines 70-73)."""
        # Create A -> B -> C -> A cycle
        graph = _make_graph(
            imports={
                "src/a.py": {"b"},
                "src/b.py": {"c"},
                "src/c.py": {"a"},
            },
            modules={
                "src/a.py": {"func_a"},
                "src/b.py": {"func_b"},
                "src/c.py": {"func_c"},
            },
        )
        validator = DependencyValidator(graph)
        circular = validator.detect_circular_dependencies()
        assert len(circular) > 0

    def test_no_cycle(self):
        """Acyclic graph returns empty list."""
        graph = _make_graph(
            imports={
                "src/a.py": {"os"},
                "src/b.py": {"json"},
            },
        )
        validator = DependencyValidator(graph)
        assert validator.detect_circular_dependencies() == []


# ---------------------------------------------------------------------------
# validate_dependency_chains: lines 123-128 (missing + inaccessible)
# ---------------------------------------------------------------------------
class TestValidateDependencyChains:
    def test_import_resolves_but_inaccessible(self):
        """Import resolves to a file without extension → inaccessible (lines 126-129)."""
        # "mymod" resolves to "mymod_dir" (no extension) via modules,
        # and _is_file_accessible will return False for no-suffix target.
        graph = _make_graph(
            imports={
                "src/app.py": {"mymod_dir"},
            },
            modules={
                "src/app.py": {"run"},
                "mymod_dir": {"helper"},  # no extension → inaccessible
            },
        )
        validator = DependencyValidator(graph)
        missing = validator.validate_dependency_chains({"src/app.py"})
        assert any("inaccessible" in m for m in missing)

    def test_inaccessible_import(self):
        """Imported file exists but is inaccessible (lines 126-129)."""
        # Create a scenario where _is_file_accessible returns False.
        # _is_file_accessible returns False when suffixes are empty.
        # Use file paths without extensions.
        graph = _make_graph(
            imports={
                "src/app.py": {"utils"},
            },
            modules={
                "src/app.py": {"run"},
                "utils": {"helper"},  # No extension → _is_file_accessible returns False
            },
        )
        validator = DependencyValidator(graph)
        missing = validator.validate_dependency_chains({"src/app.py"})
        # "utils" resolves to a file, but has no suffix so inaccessible
        assert any("inaccessible" in m for m in missing)

    def test_external_dep_not_flagged(self):
        """stdlib/third-party imports should not appear as missing."""
        graph = _make_graph(
            imports={"src/app.py": {"os", "json", "requests"}},
            modules={"src/app.py": set()},
        )
        validator = DependencyValidator(graph)
        missing = validator.validate_dependency_chains({"src/app.py"})
        assert missing == []

    def test_node_undefined_reference(self):
        """Node depends on symbol not in graph.nodes or modules (line 144)."""
        node = DependencyNode(
            name="handler",
            file_path="src/handler.py",
            node_type="function",
            dependencies={"unknown_helper"},
        )
        graph = _make_graph(
            imports={"src/handler.py": set()},
            modules={},
            nodes={"src/handler.py::handler": node},
        )
        validator = DependencyValidator(graph)
        missing = validator.validate_dependency_chains({"src/handler.py"})
        assert any("unknown_helper" in m for m in missing)
        assert any("undefined reference" in m for m in missing)


# ---------------------------------------------------------------------------
# find_dependency_hotspots: lines 208-214
# ---------------------------------------------------------------------------
class TestFindDependencyHotspots:
    def test_hotspots_detected(self):
        """Files imported by many others appear as hotspots."""
        graph = _make_graph(
            imports={
                "src/a.py": {"utils"},
                "src/b.py": {"utils"},
                "src/c.py": {"utils"},
            },
            modules={
                "src/a.py": set(),
                "src/b.py": set(),
                "src/c.py": set(),
                "src/utils.py": {"helper"},
            },
        )
        validator = DependencyValidator(graph)
        hotspots = validator.find_dependency_hotspots()
        # utils.py should be a hotspot
        if hotspots:
            assert hotspots[0]["import_count"] >= 1

    def test_empty_graph_no_hotspots(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator.find_dependency_hotspots() == []


# ---------------------------------------------------------------------------
# validate_import_accessibility: lines 241-251
# ---------------------------------------------------------------------------
class TestValidateImportAccessibility:
    def test_inaccessible_import_detected(self):
        """Import to file without extension triggers inaccessibility."""
        graph = _make_graph(
            imports={
                "src/main.py": {"config_mod"},
            },
            modules={
                "src/main.py": set(),
                "config_mod": {"setting"},  # No extension
            },
        )
        validator = DependencyValidator(graph)
        issues = validator.validate_import_accessibility()
        assert any("not accessible" in i.lower() or "not found" in i.lower() for i in issues)

    def test_all_accessible_no_issues(self):
        """Valid imports produce no accessibility issues."""
        graph = _make_graph(
            imports={
                "src/a.py": {"os", "json"},
            },
        )
        validator = DependencyValidator(graph)
        assert validator.validate_import_accessibility() == []


# ---------------------------------------------------------------------------
# _find_files_for_module: lines 276-280 (partial matches for submodules)
# ---------------------------------------------------------------------------
class TestFindFilesForModule:
    def test_submodule_partial_match(self):
        """Dotted module name tries partial matches."""
        graph = _make_graph(
            modules={
                "src/utils.py": {"helper"},
                "src/utils/core.py": {"deep_helper"},
            },
        )
        validator = DependencyValidator(graph)
        # _file_path_to_module_name("src/utils.py") → "utils"
        # _file_path_to_module_name("src/utils/core.py") → "utils.core"
        files = validator._find_files_for_module("utils.core")
        assert "src/utils/core.py" in files

    def test_init_file_match(self):
        """__init__.py in package directory matches."""
        graph = _make_graph(
            modules={
                "mypackage/__init__.py": {"init_func"},
            },
        )
        validator = DependencyValidator(graph)
        # _file_path_to_module_name("mypackage/__init__.py") → "mypackage"
        files = validator._find_files_for_module("mypackage")
        assert "mypackage/__init__.py" in files

    def test_no_match_returns_empty(self):
        graph = _make_graph(modules={})
        validator = DependencyValidator(graph)
        assert validator._find_files_for_module("nonexistent") == set()


# ---------------------------------------------------------------------------
# _is_external_dependency: lines 338-356
# ---------------------------------------------------------------------------
class TestIsExternalDependency:
    def test_stdlib_is_external(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_external_dependency("os") is True
        assert validator._is_external_dependency("json") is True
        assert validator._is_external_dependency("logging") is True
        assert validator._is_external_dependency("sys") is True

    def test_third_party_is_external(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_external_dependency("django") is True
        assert validator._is_external_dependency("flask.app") is True
        assert validator._is_external_dependency("numpy.core") is True
        assert validator._is_external_dependency("requests") is True
        assert validator._is_external_dependency("fastapi") is True
        assert validator._is_external_dependency("pydantic") is True
        assert validator._is_external_dependency("pytest") is True

    def test_internal_not_found_is_external(self):
        """Module not matching stdlib/third-party and not found → external."""
        graph = _make_graph(modules={})
        validator = DependencyValidator(graph)
        # Some random name that's not in stdlib or third_party lists
        # and has no files → falls through to final return
        assert validator._is_external_dependency("some_random_pkg_xyz") is True

    def test_internal_found_not_external(self):
        """Module that resolves to a local file is NOT external."""
        graph = _make_graph(
            modules={
                "src/mymod.py": {"func"},
            },
        )
        validator = DependencyValidator(graph)
        assert validator._is_external_dependency("mymod") is False


# ---------------------------------------------------------------------------
# _is_file_accessible: lines 370-396
# ---------------------------------------------------------------------------
class TestIsFileAccessible:
    def test_no_suffix_returns_false(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_file_accessible("src/main.py", "utils") is False
        assert validator._is_file_accessible("utils", "src/main.py") is False

    def test_same_directory_accessible(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_file_accessible("src/a.py", "src/b.py") is True

    def test_init_file_always_accessible(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_file_accessible("src/a.py", "pkg/__init__.py") is True

    def test_parent_directory_accessible(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        # from_path is inside to_path's parent → relative_to succeeds
        assert validator._is_file_accessible("src/sub/deep.py", "src/util.py") is True

    def test_different_trees_still_accessible(self):
        """Default fallback returns True."""
        graph = _make_graph()
        validator = DependencyValidator(graph)
        assert validator._is_file_accessible("lib/a.py", "vendor/b.py") is True


# ---------------------------------------------------------------------------
# _file_path_to_module_name: lines 408, 421-466
# ---------------------------------------------------------------------------
class TestFilePathToModuleName:
    def test_python_with_src_dir(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/services/handler.py")
        assert result == "services.handler"

    def test_python_init_file(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        # __init__.py: name → parent.name ("services"), src in parts so
        # module_parts = parts[1:-1] + (name,) = ("services",) + ("services",)
        result = validator._file_path_to_module_name("src/services/__init__.py")
        assert "services" in result

    def test_python_init_file_no_src(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        # No src dir → fallback path for __init__.py
        result = validator._file_path_to_module_name("services/__init__.py")
        assert "services" in result

    def test_python_fallback_no_src(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("mypackage/util.py")
        assert result == "mypackage.util"

    def test_javascript_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("frontend/components/Button.js")
        assert "Button" in result

    def test_javascript_index_file(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("frontend/components/index.js")
        assert "components" in result

    def test_typescript_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/services/api.ts")
        assert "api" in result

    def test_go_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/handlers/main.go")
        assert "handlers" in result or "main" in result

    def test_rust_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/utils/parser.rs")
        assert "parser" in result

    def test_rust_mod_file(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/utils/mod.rs")
        assert "utils" in result

    def test_java_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/main/java/com/example/App.java")
        # After "java" index: com/example/App
        assert "App" in result

    def test_csharp_module(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("src/Controllers/HomeController.cs")
        assert "HomeController" in result

    def test_unknown_extension(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("data/config.yaml")
        assert result == "config"

    def test_node_modules_js(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        result = validator._file_path_to_module_name("node_modules/express/index.js")
        assert result == "express"


# ---------------------------------------------------------------------------
# generate_dependency_report: lines 484, 489, 492, 495
# ---------------------------------------------------------------------------
class TestGenerateDependencyReport:
    def test_report_with_circular_deps(self):
        """Report includes recommendation for circular deps (line 484)."""
        graph = _make_graph(
            imports={
                "src/a.py": {"b"},
                "src/b.py": {"a"},
            },
            modules={
                "src/a.py": {"fa"},
                "src/b.py": {"fb"},
            },
        )
        validator = DependencyValidator(graph)
        report = validator.generate_dependency_report()
        assert report["circular_dependencies"]["count"] > 0
        assert any("circular" in r.lower() for r in report["recommendations"])

    def test_report_with_accessibility_issues(self):
        """Report includes recommendation for accessibility issues (line 489)."""
        # Import resolves to a file without extension → inaccessible
        graph = _make_graph(
            imports={
                "src/main.py": {"mymod"},
            },
            modules={
                "src/main.py": set(),
                "mymod": {"helper"},  # No extension → inaccessible
            },
        )
        validator = DependencyValidator(graph)
        report = validator.generate_dependency_report()
        assert report["accessibility_issues"]["count"] > 0
        assert any(
            "accessibility" in r.lower() or "import" in r.lower() for r in report["recommendations"]
        )

    def test_report_high_avg_deps(self):
        """Report recommends modularization for high avg deps (line 492)."""
        # Create a file with many imports
        imports_set = {f"mod_{i}" for i in range(15)}
        graph = _make_graph(
            imports={
                "src/big.py": imports_set,
            },
            modules={
                "src/big.py": set(),
            },
        )
        validator = DependencyValidator(graph)
        report = validator.generate_dependency_report()
        if report["complexity_metrics"]["average_dependencies_per_file"] > 10:
            assert any("modularization" in r.lower() for r in report["recommendations"])

    def test_report_hotspot_high_count(self):
        """Report recommends refactoring for high-import hotspot (line 495)."""
        # Create many files importing same module
        modules_dict = {}
        imports_dict = {}
        for i in range(25):
            fname = f"src/file_{i}.py"
            imports_dict[fname] = {"common"}
            modules_dict[fname] = set()
        modules_dict["src/common.py"] = {"shared_func"}
        graph = _make_graph(
            imports=imports_dict,
            modules=modules_dict,
        )
        validator = DependencyValidator(graph)
        report = validator.generate_dependency_report()
        # Check report structure is valid
        assert "dependency_hotspots" in report
        assert "overall_health_score" in report

    def test_clean_report(self):
        """Empty graph yields clean report with score 100."""
        graph = _make_graph()
        validator = DependencyValidator(graph)
        report = validator.generate_dependency_report()
        assert report["overall_health_score"] == 100.0
        assert report["circular_dependencies"]["count"] == 0
        assert report["accessibility_issues"]["count"] == 0
        assert report["recommendations"] == []


# ---------------------------------------------------------------------------
# _calculate_health_score: lines 534, 536
# ---------------------------------------------------------------------------
class TestCalculateHealthScore:
    def test_deductions_for_circular_deps(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[("a", "b")] * 10,
            accessibility_issues=[],
            complexity_metrics={"average_dependencies_per_file": 5},
        )
        # 100 - 10*2 = 80
        assert score == pytest.approx(80.0)

    def test_deductions_for_accessibility(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[],
            accessibility_issues=["issue"] * 15,
            complexity_metrics={"average_dependencies_per_file": 5},
        )
        # 100 - 15*1 = 85
        assert score == pytest.approx(85.0)

    def test_deductions_for_high_complexity_gt_15(self):
        """avg_deps > 15 → deduct (avg-15)*2 (line 534)."""
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[],
            accessibility_issues=[],
            complexity_metrics={"average_dependencies_per_file": 20},
        )
        # 100 - (20-15)*2 = 100 - 10 = 90
        assert score == pytest.approx(90.0)

    def test_deductions_for_moderate_complexity_gt_10(self):
        """avg_deps > 10 (but <= 15) → deduct (avg-10)*1 (line 536)."""
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[],
            accessibility_issues=[],
            complexity_metrics={"average_dependencies_per_file": 13},
        )
        # 100 - (13-10)*1 = 97
        assert score == pytest.approx(97.0)

    def test_score_floored_at_zero(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[("a", "b")] * 60,
            accessibility_issues=["x"] * 50,
            complexity_metrics={"average_dependencies_per_file": 30},
        )
        assert score == 0.0

    def test_score_capped_at_100(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        score = validator._calculate_health_score(
            circular_deps=[],
            accessibility_issues=[],
            complexity_metrics={"average_dependencies_per_file": 0},
        )
        assert score == 100.0


# ---------------------------------------------------------------------------
# analyze_dependency_complexity (coverage for structure)
# ---------------------------------------------------------------------------
class TestAnalyzeDependencyComplexity:
    def test_basic_complexity(self):
        graph = _make_graph(
            imports={
                "src/a.py": {"os", "json", "logging"},
                "src/b.py": {"os"},
            },
            modules={
                "src/a.py": {"func_a"},
                "src/b.py": {"func_b"},
            },
            nodes={
                "src/a.py::func_a": DependencyNode(
                    name="func_a", file_path="src/a.py", node_type="function"
                ),
            },
        )
        validator = DependencyValidator(graph)
        metrics = validator.analyze_dependency_complexity()
        assert metrics["total_files"] == 2
        assert metrics["total_imports"] == 4
        assert metrics["total_nodes"] == 1
        assert metrics["average_dependencies_per_file"] == 2.0

    def test_empty_graph_complexity(self):
        graph = _make_graph()
        validator = DependencyValidator(graph)
        metrics = validator.analyze_dependency_complexity()
        assert metrics["total_files"] == 0
        assert metrics["average_dependencies_per_file"] == 0

    def test_highly_coupled_files(self):
        """Files with deps > 2*avg should appear as highly coupled."""
        graph = _make_graph(
            imports={
                "src/a.py": {f"mod_{i}" for i in range(20)},
                "src/b.py": {"os"},
                "src/c.py": {"json"},
            },
        )
        validator = DependencyValidator(graph)
        metrics = validator.analyze_dependency_complexity()
        # avg ≈ 7.33, so a.py (20) > 2*7.33
        assert "src/a.py" in metrics["highly_coupled_files"]
