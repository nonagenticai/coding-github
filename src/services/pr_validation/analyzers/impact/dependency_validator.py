"""
Dependency validation utilities and helpers.

This module provides utilities for validating dependencies,
detecting circular imports, and ensuring dependency integrity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraph,
)

logger = logging.getLogger(__name__)


class DependencyValidator:
    """
    Validates dependencies and detects issues in the dependency graph.

    This validator provides functionality for:
    - Circular dependency detection using DFS algorithms
    - Missing dependency validation
    - Dependency chain verification
    - External dependency identification
    - File accessibility checks
    """

    def __init__(self, dependency_graph: DependencyGraph):
        self.dependency_graph = dependency_graph

    def detect_circular_dependencies(self) -> list[tuple[str, str]]:
        """
        Detect circular dependencies using DFS with color marking.

        Returns:
            List of tuples representing circular dependency edges (from_module, to_module)
        """
        circular_dependencies: list[tuple[str, str]] = []

        # Build adjacency list for module-level dependencies
        adjacency_list: dict[str, set[str]] = {}
        for file_path, imports in self.dependency_graph.imports.items():
            adjacency_list[file_path] = set()
            for import_module in imports:
                # Convert import module name to file path
                imported_files = self._find_files_for_module(import_module)
                adjacency_list[file_path].update(imported_files)

        # DFS with color marking (white=0, gray=1, black=2)
        colors: dict[str, int] = {file: 0 for file in adjacency_list}
        parent: dict[str, str | None] = {file: None for file in adjacency_list}

        def dfs(node: str) -> None:
            """Perform DFS to detect cycles."""
            colors[node] = 1  # Gray - currently being processed

            for neighbor in adjacency_list.get(node, set()):
                if neighbor not in colors:
                    colors[neighbor] = 0

                if colors[neighbor] == 1:  # Found a back edge (cycle)
                    # Trace back the cycle
                    cycle_path = [neighbor, node]
                    current = node
                    while parent.get(current) and parent[current] != neighbor:  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                        current = parent[current]  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                        if current:
                            cycle_path.append(current)

                    # Add cycle edges to result
                    for i in range(len(cycle_path) - 1):
                        circular_dependencies.append((cycle_path[i], cycle_path[i + 1]))

                elif colors[neighbor] == 0:
                    parent[neighbor] = node
                    dfs(neighbor)

            colors[node] = 2  # Black - finished processing

        # Run DFS from all unvisited nodes
        for node in list(adjacency_list.keys()):
            if node in colors and colors[node] == 0:
                dfs(node)

        # Remove duplicates while preserving order
        seen = set()
        unique_dependencies = []
        for dep in circular_dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_dependencies.append(dep)

        return unique_dependencies

    def validate_dependency_chains(self, affected_files: set[str]) -> list[str]:
        """
        Validate dependency chains and identify missing dependencies.

        Args:
            affected_files: Set of files affected by changes

        Returns:
            List of missing dependency identifiers
        """
        missing_dependencies: list[str] = []

        # Check each affected file's dependencies
        for file_path in affected_files:
            imports = self.dependency_graph.imports.get(file_path, set())

            for import_module in imports:
                # Check if the imported module exists in our dependency graph
                imported_files = self._find_files_for_module(import_module)

                if not imported_files:
                    # Check if it's an external dependency
                    if not self._is_external_dependency(import_module):
                        missing_dependencies.append(f"{file_path} -> {import_module}")
                else:
                    # Validate that the imported files are accessible
                    for imported_file in imported_files:
                        if not self._is_file_accessible(file_path, imported_file):
                            missing_dependencies.append(
                                f"{file_path} -> {imported_file} (inaccessible)"
                            )

        # Check for function/class level dependencies
        for node_id, node in self.dependency_graph.nodes.items():
            if node.file_path in affected_files:
                for dep in node.dependencies:
                    # Check if dependency exists
                    if dep not in self.dependency_graph.nodes:
                        # Check if it's a module-level import
                        module_exists = any(
                            dep in self.dependency_graph.modules.get(module, set())
                            for module in self.dependency_graph.modules
                        )
                        if not module_exists:
                            missing_dependencies.append(f"{node_id} -> {dep} (undefined reference)")

        return missing_dependencies

    def analyze_dependency_complexity(self) -> dict[str, Any]:
        """
        Analyze the complexity of the dependency graph.

        Returns:
            Dictionary containing complexity metrics
        """
        total_files = len(self.dependency_graph.imports)
        total_imports = sum(len(imports) for imports in self.dependency_graph.imports.values())
        total_nodes = len(self.dependency_graph.nodes)

        # Calculate average dependencies per file
        avg_dependencies = total_imports / total_files if total_files > 0 else 0

        # Find files with highest dependency counts
        files_by_deps = [
            (file_path, len(imports))
            for file_path, imports in self.dependency_graph.imports.items()
        ]
        files_by_deps.sort(key=lambda x: x[1], reverse=True)

        # Calculate dependency distribution
        dependency_counts = [len(imports) for imports in self.dependency_graph.imports.values()]
        dependency_counts.sort()

        median_deps = dependency_counts[len(dependency_counts) // 2] if dependency_counts else 0
        max_deps = max(dependency_counts) if dependency_counts else 0
        min_deps = min(dependency_counts) if dependency_counts else 0

        # Identify highly coupled files (files with many dependencies or dependents)
        highly_coupled_files = [
            file_path
            for file_path, dep_count in files_by_deps[:10]
            if dep_count > avg_dependencies * 2
        ]

        return {
            "total_files": total_files,
            "total_imports": total_imports,
            "total_nodes": total_nodes,
            "average_dependencies_per_file": avg_dependencies,
            "median_dependencies": median_deps,
            "max_dependencies": max_deps,
            "min_dependencies": min_deps,
            "highly_coupled_files": highly_coupled_files,
            "top_dependent_files": files_by_deps[:5],
        }

    def find_dependency_hotspots(self) -> list[dict[str, Any]]:
        """
        Find dependency hotspots - files that are imported by many other files.

        Returns:
            List of hotspot information dictionaries
        """
        # Count how many files import each module
        import_counts: dict[str, int] = {}
        importers: dict[str, list[str]] = {}

        for file_path, imports in self.dependency_graph.imports.items():
            for import_module in imports:
                imported_files = self._find_files_for_module(import_module)
                for imported_file in imported_files:
                    import_counts[imported_file] = import_counts.get(imported_file, 0) + 1
                    if imported_file not in importers:
                        importers[imported_file] = []
                    importers[imported_file].append(file_path)

        # Sort by import count to find hotspots
        hotspots = [
            {
                "file": file_path,
                "import_count": count,
                "importers": importers.get(file_path, [])[:10],  # Limit to top 10 importers
                "total_importers": len(importers.get(file_path, [])),
            }
            for file_path, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[
                :15
            ]  # Top 15 hotspots
        ]

        return hotspots

    def validate_import_accessibility(self) -> list[str]:
        """
        Validate that all imports are accessible based on file structure.

        Returns:
            List of accessibility issues
        """
        accessibility_issues: list[str] = []

        for file_path, imports in self.dependency_graph.imports.items():
            for import_module in imports:
                imported_files = self._find_files_for_module(import_module)

                if imported_files:
                    for imported_file in imported_files:
                        if not self._is_file_accessible(file_path, imported_file):
                            accessibility_issues.append(
                                f"Import not accessible: {file_path} -> {imported_file}"
                            )
                elif not self._is_external_dependency(import_module):
                    accessibility_issues.append(f"Import not found: {file_path} -> {import_module}")

        return accessibility_issues

    def _find_files_for_module(self, module_name: str) -> set[str]:
        """
        Find file paths that correspond to a module name.

        Args:
            module_name: The module name to search for

        Returns:
            Set of file paths that could represent this module
        """
        possible_files: set[str] = set()

        # Direct module name match
        for file_path in self.dependency_graph.modules:
            if self._file_path_to_module_name(file_path) == module_name:
                possible_files.add(file_path)

        # Check for partial matches (submodules)
        module_parts = module_name.split(".")
        if len(module_parts) > 1:
            # Try to find parent modules
            for i in range(len(module_parts)):
                partial_module = ".".join(module_parts[: i + 1])
                for file_path in self.dependency_graph.modules:
                    if self._file_path_to_module_name(file_path) == partial_module:
                        possible_files.add(file_path)

        # Check for __init__.py files in package directories
        if module_name.replace(".", "/") in str(self.dependency_graph.modules):
            init_file = f"{module_name.replace('.', '/')}/__init__.py"
            if init_file in self.dependency_graph.modules:
                possible_files.add(init_file)

        return possible_files

    def _is_external_dependency(self, module_name: str) -> bool:
        """
        Check if a module is an external dependency (not part of the codebase).

        Args:
            module_name: The module name to check

        Returns:
            True if the module is external, False otherwise
        """
        # Common standard library modules
        stdlib_modules = {
            "os",
            "sys",
            "re",
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "typing",
            "pathlib",
            "logging",
            "unittest",
            "asyncio",
            "math",
            "random",
            "time",
            "copy",
            "pickle",
            "subprocess",
            "threading",
            "queue",
            "socket",
            "http",
            "urllib",
            "email",
            "csv",
            "xml",
            "html",
        }

        # Check if it's a standard library module
        root_module = module_name.split(".")[0]
        if root_module in stdlib_modules:
            return True

        # Check for third-party indicators
        third_party_indicators = [
            "django",
            "flask",
            "numpy",
            "pandas",
            "requests",
            "pytest",
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "celery",
            "redis",
        ]

        if any(module_name.startswith(pkg) for pkg in third_party_indicators):
            return True

        # If no local file found for this module, assume it's external
        return len(self._find_files_for_module(module_name)) == 0

    def _is_file_accessible(self, from_file: str, to_file: str) -> bool:
        """
        Check if a file import is accessible based on project structure.

        Args:
            from_file: The file doing the import
            to_file: The file being imported

        Returns:
            True if the import is valid, False otherwise
        """
        # Basic accessibility check - can be enhanced with more rules
        from_path = Path(from_file)
        to_path = Path(to_file)

        # Check if both files exist in the project
        if not from_path.suffix or not to_path.suffix:
            return False

        # Check for relative import validity
        if from_path.parent == to_path.parent:
            return True  # Same directory

        # Check if to_file is in a parent or accessible directory
        # This is a simplified check - real implementation would consider
        # Python path, package structure, etc.
        try:
            # If to_path is in a parent directory of from_path
            from_path.relative_to(to_path.parent)
            return True
        except ValueError:
            pass

        # Check if it's a proper package import
        if "__init__.py" in str(to_path):
            return True

        # Default to accessible for now (can be made stricter)
        return True

    def _file_path_to_module_name(self, file_path: str) -> str:
        """Convert file path to potential module name."""
        path = Path(file_path)
        ext = path.suffix.lower()

        # Remove extension
        name = path.stem

        # Handle index/init files
        if name in ["__init__", "index", "mod"]:
            name = path.parent.name

        # Python module conversion
        if ext == ".py":
            parts = path.parts
            # Look for common Python source directories
            for src_dir in ["src", "lib", "app"]:
                if src_dir in parts:
                    start_idx = parts.index(src_dir) + 1
                    module_parts = parts[start_idx:-1] + (name,)
                    return ".".join(module_parts)

            # Fallback: use full path as module
            return ".".join(path.parts[:-1]) + f".{name}"

        # JavaScript/TypeScript module conversion
        elif ext in [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]:
            # Handle node_modules
            if "node_modules" in path.parts:
                idx = path.parts.index("node_modules")
                if idx + 1 < len(path.parts):
                    return path.parts[idx + 1]  # Package name

            # Return relative path style
            return "/".join(path.parts[:-1]) + f"/{name}"

        # Go module conversion
        elif ext == ".go":
            # Go uses full import paths
            parts = path.parts
            if "src" in parts:
                start_idx = parts.index("src") + 1
                return "/".join(parts[start_idx:])
            return str(path)

        # Rust module conversion
        elif ext == ".rs":
            # Convert to Rust module path notation
            parts = path.parts
            if "src" in parts:
                start_idx = parts.index("src") + 1
                module_parts = parts[start_idx:-1] + (name,)
                return "::".join(module_parts)
            return name

        # Java/C# namespace conversion
        elif ext in [".java", ".cs"]:
            # These typically use package/namespace declarations
            # For now, use dot notation
            parts = path.parts
            for src_dir in ["src", "main", "java", "csharp"]:
                if src_dir in parts:
                    start_idx = parts.index(src_dir) + 1
                    if start_idx < len(parts):
                        namespace_parts = parts[start_idx:-1] + (name,)
                        return ".".join(namespace_parts)
            return name

        return name

    def generate_dependency_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive dependency validation report.

        Returns:
            Dictionary containing validation results and recommendations
        """
        circular_deps = self.detect_circular_dependencies()
        complexity_metrics = self.analyze_dependency_complexity()
        hotspots = self.find_dependency_hotspots()
        accessibility_issues = self.validate_import_accessibility()

        # Generate recommendations based on findings
        recommendations = []

        if circular_deps:
            recommendations.append(
                f"Address {len(circular_deps)} circular dependencies to improve maintainability"
            )

        if accessibility_issues:
            recommendations.append(f"Fix {len(accessibility_issues)} import accessibility issues")

        if complexity_metrics["average_dependencies_per_file"] > 10:
            recommendations.append("High average dependency count - consider modularization")

        if len(hotspots) > 0 and hotspots[0]["import_count"] > 20:
            recommendations.append(
                f"High dependency hotspot detected: {hotspots[0]['file']} - consider refactoring"
            )

        return {
            "circular_dependencies": {
                "count": len(circular_deps),
                "dependencies": circular_deps[:10],  # Limit for readability
            },
            "complexity_metrics": complexity_metrics,
            "dependency_hotspots": hotspots[:5],  # Top 5 hotspots
            "accessibility_issues": {
                "count": len(accessibility_issues),
                "issues": accessibility_issues[:10],  # Limit for readability
            },
            "recommendations": recommendations,
            "overall_health_score": self._calculate_health_score(
                circular_deps, accessibility_issues, complexity_metrics
            ),
        }

    def _calculate_health_score(
        self,
        circular_deps: list[tuple[str, str]],
        accessibility_issues: list[str],
        complexity_metrics: dict[str, Any],
    ) -> float:
        """Calculate an overall dependency health score (0-100)."""
        score = 100.0

        # Deduct for circular dependencies
        score -= len(circular_deps) * 2

        # Deduct for accessibility issues
        score -= len(accessibility_issues) * 1

        # Deduct for high complexity
        avg_deps = complexity_metrics["average_dependencies_per_file"]
        if avg_deps > 15:
            score -= (avg_deps - 15) * 2
        elif avg_deps > 10:
            score -= (avg_deps - 10) * 1

        # Ensure score is within bounds
        return max(0.0, min(100.0, score))
