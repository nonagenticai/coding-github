"""
Dependency graph analyzer for building and analyzing code dependencies.

This module handles the construction and analysis of dependency graphs,
including import detection, AST parsing, and dependency relationship mapping.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coding_client.models.pr_validation import FileChange, FileType
from coding_client.models.static_analysis import (
    StaticAnalysisChangeType,  # TODO: verify import source
)

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Represents a dependency node in the codebase."""

    name: str
    file_path: str
    node_type: str  # 'function', 'class', 'module', 'variable'
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """Represents the dependency graph of the codebase."""

    nodes: dict[str, DependencyNode] = field(default_factory=dict)
    modules: dict[str, set[str]] = field(default_factory=dict)  # module -> {symbols}
    imports: dict[str, set[str]] = field(default_factory=dict)  # file -> {imported_modules}


class DependencyGraphAnalyzer:
    """
    Builds and analyzes dependency graphs for codebase files.

    This analyzer provides functionality for:
    - Parsing file imports and dependencies
    - Building comprehensive dependency graphs
    - AST analysis for Python files
    - Multi-language import pattern detection
    """

    def __init__(self):
        self.dependency_graph = DependencyGraph()

        # Import patterns for different languages
        self.python_import_patterns = [
            r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_.,\s]*)",
            r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+",
            r"^\s*from\s+\.+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+",  # Relative imports
        ]

        self.javascript_import_patterns = [
            r'^\s*import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'^\s*import\s+[\'"]([^\'"]+)[\'"]',  # Side-effect imports
            r'^\s*const\s+.*\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'^\s*var\s+.*\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'^\s*let\s+.*\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'^\s*import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',  # Dynamic imports
            r'^\s*export\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',  # Re-exports
        ]

        self.typescript_import_patterns = self.javascript_import_patterns + [
            r'^\s*import\s+type\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',  # Type imports
            r'^\s*import\s+{[^}]*}\s+from\s+[\'"]([^\'"]+)[\'"]',  # Named imports
        ]

        self.go_import_patterns = [
            r'^\s*import\s+"([^"]+)"',
            r'^\s*import\s+\(\s*"([^"]+)"',  # Multi-line imports
        ]

        self.rust_import_patterns = [
            r"^\s*use\s+([a-zA-Z_][a-zA-Z0-9_:]*)",
            r"^\s*extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]

        self.java_import_patterns = [
            r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_.]*);",
            r"^\s*import\s+static\s+([a-zA-Z_][a-zA-Z0-9_.]*);",
        ]

        self.csharp_import_patterns = [
            r"^\s*using\s+([a-zA-Z_][a-zA-Z0-9_.]*);",
            r"^\s*using\s+static\s+([a-zA-Z_][a-zA-Z0-9_.]*);",
        ]

        # File extension patterns for dependency detection
        self.extension_patterns = {
            ".py": self.python_import_patterns,
            ".js": self.javascript_import_patterns,
            ".jsx": self.javascript_import_patterns,
            ".mjs": self.javascript_import_patterns,
            ".cjs": self.javascript_import_patterns,
            ".ts": self.typescript_import_patterns,
            ".tsx": self.typescript_import_patterns,
            ".go": self.go_import_patterns,
            ".rs": self.rust_import_patterns,
            ".java": self.java_import_patterns,
            ".cs": self.csharp_import_patterns,
        }

    async def build_dependency_graph(self, changes: list[FileChange]) -> DependencyGraph:
        """
        Build dependency graph for the changed files and their dependencies.

        Args:
            changes: List of file changes to analyze

        Returns:
            Complete dependency graph
        """
        processed_files = set()

        # Start with changed files
        for change in changes:
            if change.filename not in processed_files:
                await self._analyze_file_dependencies(change)
                processed_files.add(change.filename)

        # Find files that import the changed files
        dependent_files = self._find_dependent_files(changes)

        # Analyze dependent files for impact
        for file_path in dependent_files:
            if file_path not in processed_files:
                # Create a mock FileChange for analysis
                mock_change = FileChange(
                    filename=file_path,
                    file_type=self._classify_file_type(file_path),
                    change_type=StaticAnalysisChangeType.MODIFIED,
                    additions=0,
                    deletions=0,
                    patch="",
                )
                await self._analyze_file_dependencies(mock_change)
                processed_files.add(file_path)

        return self.dependency_graph

    async def _analyze_file_dependencies(self, file_change: FileChange) -> None:
        """Analyze dependencies for a single file."""
        file_path = file_change.filename
        file_ext = Path(file_path).suffix.lower()

        # Get import patterns for this file type
        import_patterns = self.extension_patterns.get(file_ext, [])
        if not import_patterns:
            return

        # Analyze imports in the file content
        content = file_change.after_content or file_change.before_content or ""
        if content:
            imports = self._extract_imports(content, import_patterns)
            self.dependency_graph.imports[file_path] = imports

        # For Python files, do deeper AST analysis
        if file_change.file_type == FileType.PYTHON and content:
            await self._analyze_python_dependencies(file_path, content)

    def _extract_imports(self, content: str, patterns: list[str]) -> set[str]:
        """Extract imports from file content using regex patterns."""
        imports = set()

        # Handle multi-line imports by joining continuation lines
        lines = content.split("\n")
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            # Check for line continuation
            while i < len(lines) - 1 and (
                line.endswith("\\") or line.endswith(",") or line.count("(") > line.count(")")
            ):
                i += 1
                line = line.rstrip("\\") + " " + lines[i].strip()
            processed_lines.append(line)
            i += 1

        for line in processed_lines:
            # Skip comments
            line_stripped = line.strip()
            if (
                line_stripped.startswith("#")
                or line_stripped.startswith("//")
                or line_stripped.startswith("/*")
            ):
                continue

            # Skip lines inside multi-line comments
            if "/*" in line_stripped and "*/" not in line_stripped:
                continue

            for pattern in patterns:
                match = re.search(pattern, line_stripped)
                if match:
                    if "from" in pattern and len(match.groups()) >= 1:
                        # from X import Y pattern
                        module = match.group(1).strip()
                        # Handle relative imports (remove leading dots)
                        module = module.lstrip(".")
                        if module:
                            imports.add(module)
                    else:
                        # import X pattern or require pattern
                        module_str = match.group(1)
                        # Handle multiple imports on one line
                        if "," in module_str:
                            modules = module_str.split(",")
                            for module in modules:
                                module_clean = module.strip()
                                # Clean up module names
                                if module_clean and not module_clean.startswith("."):
                                    imports.add(module_clean)
                        else:
                            module_clean = module_str.strip()
                            if module_clean:
                                imports.add(module_clean)

        return imports

    async def _analyze_python_dependencies(self, file_path: str, content: str) -> None:
        """Analyze Python-specific dependencies using AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Could not parse Python file for dependency analysis: {file_path}")
            return

        # Extract function and class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = f"{file_path}:{node.name}"
                func_node = DependencyNode(
                    name=node.name,
                    file_path=file_path,
                    node_type="function",
                    metadata={
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                    },
                )
                self.dependency_graph.nodes[func_name] = func_node

                # Add to module symbols
                if file_path not in self.dependency_graph.modules:
                    self.dependency_graph.modules[file_path] = set()
                self.dependency_graph.modules[file_path].add(node.name)

            elif isinstance(node, ast.ClassDef):
                class_name = f"{file_path}:{node.name}"
                class_node = DependencyNode(
                    name=node.name,
                    file_path=file_path,
                    node_type="class",
                    metadata={
                        "lineno": node.lineno,
                        "bases": [ast.unparse(base) for base in node.bases],
                    },
                )
                self.dependency_graph.nodes[class_name] = class_node

                # Add to module symbols
                if file_path not in self.dependency_graph.modules:
                    self.dependency_graph.modules[file_path] = set()
                self.dependency_graph.modules[file_path].add(node.name)

    def _find_dependent_files(self, changes: list[FileChange]) -> set[str]:
        """Find files that depend on the changed files."""
        dependent_files = set()
        changed_files = {change.filename for change in changes}

        # Look for potential imports based on filename patterns
        for change in changes:
            file_path = change.filename
            module_name = self._file_path_to_module_name(file_path)

            # Find files that might import this module
            potential_dependents = self._search_for_imports(module_name)
            dependent_files.update(potential_dependents)

        # Remove self-references
        dependent_files -= changed_files

        return dependent_files

    def _classify_file_type(self, file_path: str) -> FileType:
        """Classify a file path to its FileType enum value."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".py":
            return FileType.PYTHON
        elif ext in [".js", ".jsx"]:
            return FileType.JAVASCRIPT
        elif ext in [".ts", ".tsx"]:
            return FileType.TYPESCRIPT
        elif ext == ".md":
            return FileType.MARKDOWN
        elif ext in [".yml", ".yaml"]:
            return FileType.YAML
        elif ext == ".json":
            return FileType.JSON
        elif ext in [".html", ".htm"]:
            return FileType.HTML
        elif ext == ".css":
            return FileType.CSS
        elif path.name.lower() in ["dockerfile", "dockerfile.dev", "dockerfile.prod"]:
            return FileType.DOCKERFILE
        else:
            return FileType.UNKNOWN

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

    def _search_for_imports(self, module_name: str) -> set[str]:
        """Search for files that import the given module."""
        dependent_files: set[str] = set()

        # Use dependency graph if already built
        if self.dependency_graph.imports:
            for file_path, imports in self.dependency_graph.imports.items():
                # Check if module is imported directly
                if module_name in imports:
                    dependent_files.add(file_path)
                    continue

                # Check for partial imports (e.g., 'app.models' imports 'app.models.user')
                for imported_module in imports:
                    # Check if module_name is a parent module
                    if imported_module.startswith(module_name + "."):
                        dependent_files.add(file_path)
                        break

                    # Check if module_name is a submodule
                    if module_name.startswith(imported_module + "."):
                        dependent_files.add(file_path)
                        break

                    # Handle relative imports for Python
                    if "." in module_name:
                        module_parts = module_name.split(".")
                        module_base = module_parts[-1]

                        # Check if the base module name matches
                        if module_base in imported_module.split("."):
                            dependent_files.add(file_path)
                            break

                    # Handle JavaScript-style imports (with / separators)
                    if "/" in module_name or "/" in imported_module:
                        # Normalize both to use forward slashes
                        normalized_module = module_name.replace(".", "/")
                        normalized_import = imported_module.replace(".", "/")

                        if (
                            normalized_module in normalized_import
                            or normalized_import in normalized_module
                        ):
                            dependent_files.add(file_path)
                            break

                    # Handle Rust-style imports (with :: separators)
                    if "::" in module_name or "::" in imported_module:
                        # Check for Rust module matches
                        if module_name.replace("::", ".") == imported_module.replace("::", "."):
                            dependent_files.add(file_path)
                            break

        # Log results for debugging
        if dependent_files:
            logger.debug(
                f"Found {len(dependent_files)} files importing {module_name}: {list(dependent_files)[:5]}..."
            )
        else:
            logger.debug(f"No files found importing {module_name}")

        return dependent_files

    def get_dependency_graph(self) -> DependencyGraph:
        """Get the current dependency graph."""
        return self.dependency_graph

    def reset_graph(self) -> None:
        """Reset the dependency graph to start fresh."""
        self.dependency_graph = DependencyGraph()
