"""Core tests for PRImpactAnalyzer — real dependency extraction and impact assessment."""

from __future__ import annotations

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
# Python dependency extraction — build graph from imports
# ---------------------------------------------------------------------------


class TestPythonDependencyExtraction:
    """Build dependency graph from Python source code."""

    @pytest.mark.asyncio
    async def test_extracts_standard_and_third_party_imports(self, codebase_ctx):
        code = (
            "import os\n"
            "from pathlib import Path\n"
            "from myapp.models import User\n"
            "\n"
            "def get_user():\n"
            "    return User()\n"
        )
        changes = [
            FileChange(
                filename="src/service.py",
                file_type=FileType.PYTHON,
                after_content=code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        graph = await analyzer.graph_analyzer.build_dependency_graph(changes)
        imports = graph.imports.get("src/service.py", set())
        assert "os" in imports
        assert "pathlib" in imports
        assert "myapp.models" in imports

    @pytest.mark.asyncio
    async def test_extracts_function_and_class_nodes(self, codebase_ctx):
        code = "def helper():\n    pass\n\nclass Service:\n    pass\n"
        changes = [
            FileChange(
                filename="src/svc.py",
                file_type=FileType.PYTHON,
                after_content=code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        graph = await analyzer.graph_analyzer.build_dependency_graph(changes)
        node_types = {n.node_type for n in graph.nodes.values() if n.file_path == "src/svc.py"}
        assert "function" in node_types
        assert "class" in node_types


# ---------------------------------------------------------------------------
# JS dependency extraction
# ---------------------------------------------------------------------------


class TestJSDependencyExtraction:
    """Build dependency graph from JavaScript source code."""

    @pytest.mark.asyncio
    async def test_extracts_import_and_require(self, codebase_ctx):
        code = (
            "import React from 'react';\n"
            "import { useState } from 'react';\n"
            "const lodash = require('lodash');\n"
        )
        changes = [
            FileChange(
                filename="src/app.js",
                file_type=FileType.JAVASCRIPT,
                after_content=code,
                change_type=ChangeType.MODIFIED,
                additions=0,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        graph = await analyzer.graph_analyzer.build_dependency_graph(changes)
        imports = graph.imports.get("src/app.js", set())
        assert "react" in imports
        assert "lodash" in imports


# ---------------------------------------------------------------------------
# assess_impact — full pipeline
# ---------------------------------------------------------------------------


class TestAssessImpact:
    """Full impact assessment pipeline."""

    @pytest.mark.asyncio
    async def test_returns_impact_assessment(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/util.py",
                file_type=FileType.PYTHON,
                before_content="x = 1\n",
                after_content="import json\nx = 2\n",
                additions=1,
                deletions=1,
                change_type=ChangeType.MODIFIED,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        result = await analyzer.assess_impact(changes, codebase_ctx)
        assert isinstance(result, ImpactAssessment)
        assert isinstance(result.dependency_impacts, list)
        assert hasattr(result, "test_coverage_impact")
        assert hasattr(result, "documentation_impact")

    @pytest.mark.asyncio
    async def test_changed_file_appears_in_impacts(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/core.py",
                file_type=FileType.PYTHON,
                after_content="def main():\n    pass\n",
                additions=2,
                change_type=ChangeType.MODIFIED,
                deletions=0,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        result = await analyzer.assess_impact(changes, codebase_ctx)
        assert "src/core.py" in result.dependency_impacts


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """Empty change lists produce empty graphs."""

    @pytest.mark.asyncio
    async def test_empty_changes_empty_graph(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        graph = await analyzer.graph_analyzer.build_dependency_graph([])
        assert graph.nodes == {}
        assert graph.imports == {}


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """get_dependency_graph, get_change_patterns, get_change_metadata, reset."""

    def test_get_dependency_graph(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        result = analyzer.get_dependency_graph()
        assert isinstance(result, DependencyGraph)

    def test_get_change_patterns(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/api/routes.py",
                file_type=FileType.PYTHON,
                additions=10,
                deletions=2,
                patch="+def new_route():\n",
                change_type=ChangeType.MODIFIED,
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        patterns = analyzer.get_change_patterns(changes)
        assert isinstance(patterns, dict)
        assert "api_changes" in patterns

    def test_get_change_metadata(self, codebase_ctx):
        changes = [
            FileChange(
                filename="src/utils.py",
                file_type=FileType.PYTHON,
                additions=5,
                deletions=1,
                change_type=ChangeType.MODIFIED,
                patch="",
            ),
        ]
        analyzer = PRImpactAnalyzer(codebase_ctx)
        metadata = analyzer.get_change_metadata(changes)
        assert metadata["total_files_changed"] == 1
        assert metadata["total_additions"] == 5
        assert metadata["total_deletions"] == 1

    def test_reset_analyzer(self, codebase_ctx):
        from services.pr_validation.analyzers.impact.dependency_validator import DependencyValidator

        analyzer = PRImpactAnalyzer(codebase_ctx)
        analyzer.dependency_validator = DependencyValidator(DependencyGraph())
        assert analyzer.dependency_validator is not None

        analyzer.reset_analyzer()
        assert analyzer.dependency_validator is None
        new_graph = analyzer.graph_analyzer.get_dependency_graph()
        assert new_graph.nodes == {}


# ---------------------------------------------------------------------------
# get_dependency_report
# ---------------------------------------------------------------------------


class TestGetDependencyReport:
    """Report generation with and without validator."""

    def test_without_validator_returns_error(self, codebase_ctx):
        analyzer = PRImpactAnalyzer(codebase_ctx)
        report = analyzer.get_dependency_report()
        assert "error" in report
        assert "not initialized" in report["error"]

    def test_with_validator_returns_report(self, codebase_ctx):
        from services.pr_validation.analyzers.impact.dependency_validator import DependencyValidator

        analyzer = PRImpactAnalyzer(codebase_ctx)
        analyzer.dependency_validator = DependencyValidator(DependencyGraph())
        report = analyzer.get_dependency_report()
        assert isinstance(report, dict)
        assert "error" not in report
