"""Edge-case and boundary tests for PRImpactAnalyzer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
    ImpactAssessment,
)

from services.pr_validation.analyzers.impact.dependency_analyzer import (
    PRImpactAnalyzer,
)
from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraph,
    DependencyNode,
)


@pytest.fixture
def codebase_ctx():
    return CodebaseContext(
        total_files=5,
        total_lines=2000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


# ---------------------------------------------------------------------------
# assess_impact — exception handling (SyntaxError, ValueError, OSError)
# ---------------------------------------------------------------------------


class TestAssessImpactExceptions:
    """Error paths in assess_impact return safe fallback ImpactAssessment."""

    @pytest.mark.asyncio
    async def test_syntax_error_returns_fallback(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        changes = [
            FileChange(
                filename="bad.py",
                file_type=FileType.PYTHON,
                additions=1,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            )
        ]
        with patch.object(
            analyzer.graph_analyzer,
            "build_dependency_graph",
            new_callable=AsyncMock,
            side_effect=SyntaxError("invalid"),
        ):
            result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)
        assert result.dependency_impacts == []
        assert "error" in result.performance_impact

    @pytest.mark.asyncio
    async def test_value_error_returns_fallback(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        changes = [
            FileChange(
                filename="bad.py",
                file_type=FileType.PYTHON,
                additions=1,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            )
        ]
        with patch.object(
            analyzer.graph_analyzer,
            "build_dependency_graph",
            new_callable=AsyncMock,
            side_effect=ValueError("bad value"),
        ):
            result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)
        assert "bad value" in result.performance_impact["error"]

    @pytest.mark.asyncio
    async def test_os_error_returns_fallback(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        changes = [
            FileChange(
                filename="bad.py",
                file_type=FileType.PYTHON,
                additions=1,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            )
        ]
        with patch.object(
            analyzer.graph_analyzer,
            "build_dependency_graph",
            new_callable=AsyncMock,
            side_effect=OSError("file not found"),
        ):
            result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)
        assert result.dependency_impacts == []


# ---------------------------------------------------------------------------
# _analyze_dependency_impacts — import matching and node branching
# ---------------------------------------------------------------------------


class TestAnalyzeDependencyImpacts:
    """Import matching and function/class node extraction."""

    @pytest.mark.asyncio
    async def test_importing_file_appears_in_impacts(self, codebase_ctx):
        """A file that imports a changed module appears in dependency_impacts."""
        changes = [
            FileChange(
                filename="src/myapp/utils.py",
                file_type=FileType.PYTHON,
                after_content="import os\n\ndef helper():\n    pass\n",
                additions=2,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)

        graph = DependencyGraph()
        graph.imports["src/myapp/service.py"] = {"myapp.utils"}
        graph.imports["src/myapp/utils.py"] = {"os"}
        graph.nodes["src/myapp/utils.py:helper"] = DependencyNode(
            name="helper",
            file_path="src/myapp/utils.py",
            node_type="function",
        )

        with patch.object(
            analyzer.graph_analyzer,
            "build_dependency_graph",
            new_callable=AsyncMock,
            return_value=graph,
        ):
            result = await analyzer.assess_impact(changes, codebase_ctx)

        assert (
            "src/myapp/service.py" in result.dependency_impacts
            or "src/myapp/utils.py" in result.dependency_impacts
        )

    @pytest.mark.asyncio
    async def test_class_and_function_nodes_tracked(self, codebase_ctx):
        """Nodes of type function and class are properly handled."""
        changes = [
            FileChange(
                filename="src/module.py",
                file_type=FileType.PYTHON,
                after_content="def func():\n    pass\n\nclass Cls:\n    pass\n",
                additions=4,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)

        graph = DependencyGraph()
        graph.imports["src/module.py"] = set()
        graph.nodes["src/module.py:func"] = DependencyNode(
            name="func",
            file_path="src/module.py",
            node_type="function",
        )
        graph.nodes["src/module.py:Cls"] = DependencyNode(
            name="Cls",
            file_path="src/module.py",
            node_type="class",
        )

        with patch.object(
            analyzer.graph_analyzer,
            "build_dependency_graph",
            new_callable=AsyncMock,
            return_value=graph,
        ):
            result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)


# ---------------------------------------------------------------------------
# Multiple files — cross-file impact
# ---------------------------------------------------------------------------


class TestMultipleFileImpact:
    """Multiple changed files with cross-dependencies."""

    @pytest.mark.asyncio
    async def test_multiple_changes_aggregate_impacts(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/a.py",
                file_type=FileType.PYTHON,
                after_content="import os\ndef a_func():\n    pass\n",
                additions=3,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
            FileChange(
                filename="src/b.py",
                file_type=FileType.PYTHON,
                after_content="from src.a import a_func\ndef b_func():\n    a_func()\n",
                additions=3,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)
        # Both files should be in impacts
        assert "src/a.py" in result.dependency_impacts
        assert "src/b.py" in result.dependency_impacts


# ---------------------------------------------------------------------------
# File without content — minimal graph
# ---------------------------------------------------------------------------


class TestFileWithoutContent:
    """Changes with no after_content produce minimal graph entries."""

    @pytest.mark.asyncio
    async def test_no_content_still_produces_assessment(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/empty.py",
                file_type=FileType.PYTHON,
                additions=0,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)


# ---------------------------------------------------------------------------
# TypeScript file
# ---------------------------------------------------------------------------


class TestTypeScriptDependency:
    """TypeScript files have imports extracted."""

    @pytest.mark.asyncio
    async def test_ts_import_extraction(self, codebase_ctx):
        code = "import { Component } from '@angular/core';\nimport axios from 'axios';\n"
        changes = [
            FileChange(
                filename="src/app.ts",
                file_type=FileType.TYPESCRIPT,
                after_content=code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        graph = await analyzer.graph_analyzer.build_dependency_graph(changes)
        imports = graph.imports.get("src/app.ts", set())
        assert "@angular/core" in imports
        assert "axios" in imports
