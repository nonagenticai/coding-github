"""
Dependency impact analyzer for PR validation.

This module analyzes how changes affect dependencies and what depends on changed code,
providing insights into the ripple effects of modifications.

This is the main orchestration module that uses the modular components:
- DependencyGraphAnalyzer: builds and analyzes dependency graphs
- PRChangeDetector: detects and analyzes PR changes
- ImpactScorer: calculates impact scores and metrics
- DependencyValidator: validates dependencies and detects issues
"""

import logging

from coding_client.models.pr_validation import (
    CodebaseContext,
    DocumentationImpact,
    FileChange,
    ImpactAssessment,
    TestImpact,
)

from services.pr_validation.analyzers.impact.dependency_graph_analyzer import (
    DependencyGraph,
    DependencyGraphAnalyzer,
    DependencyNode,
)
from services.pr_validation.analyzers.impact.dependency_validator import (
    DependencyValidator,
)
from services.pr_validation.analyzers.impact.impact_scorer import (
    ImpactResult,
    ImpactScorer,
)
from services.pr_validation.analyzers.impact.pr_change_detector import (
    PRChangeDetector,
)

logger = logging.getLogger(__name__)


class PRImpactAnalyzer:
    """
    Analyzes dependency impacts of code changes.

    This analyzer provides understanding of:
    - What code depends on the changed files/functions
    - How changes ripple through the codebase
    - Missing test coverage for dependencies
    - Circular dependency issues
    - Import/dependency changes
    """

    def __init__(self, codebase_context: CodebaseContext | None = None):
        self.codebase_context = codebase_context

        # Initialize modular components
        self.graph_analyzer = DependencyGraphAnalyzer()
        self.change_detector = PRChangeDetector()
        self.impact_scorer = ImpactScorer()
        self.dependency_validator = None  # Will be initialized after building graph

    async def assess_impact(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> ImpactAssessment:
        """
        Assess the impact of changes across the codebase.

        Args:
            changes: List of file changes
            codebase_context: Context about the codebase

        Returns:
            ImpactAssessment with comprehensive impact analysis
        """
        self.codebase_context = codebase_context

        try:
            # Build dependency graph for changed files
            dependency_graph = await self.graph_analyzer.build_dependency_graph(changes)

            # Initialize validator with the built graph
            self.dependency_validator = DependencyValidator(dependency_graph)

            # Analyze dependency impacts using modular components
            dependency_impacts = await self._analyze_dependency_impacts(changes, dependency_graph)

            # Analyze test coverage impact
            test_coverage_impact = self.change_detector.analyze_test_coverage_impact(
                changes, dependency_impacts.affected_files
            )

            # Analyze documentation impact
            documentation_impact = self.change_detector.analyze_documentation_impact(
                changes, dependency_impacts.affected_files
            )

            # Analyze performance impact
            performance_impact = self.change_detector.analyze_performance_impact(changes)

            # Analyze security impact
            security_impact = self.change_detector.analyze_security_impact(changes)

            return ImpactAssessment(
                dependency_impacts=dependency_impacts.affected_files,
                test_coverage_impact=test_coverage_impact,
                documentation_impact=documentation_impact,
                performance_impact=performance_impact,
                security_impact=security_impact,
            )

        except (SyntaxError, ValueError, OSError) as e:
            logger.error(f"Error in dependency impact assessment: {e}")
            return ImpactAssessment(
                dependency_impacts=[],
                test_coverage_impact=TestImpact([], 0.0, [], []),
                documentation_impact=DocumentationImpact([], [], [], 0.0),
                performance_impact={"error": str(e)},
                security_impact={"error": str(e)},
            )

    async def _analyze_dependency_impacts(
        self, changes: list[FileChange], dependency_graph: DependencyGraph
    ) -> ImpactResult:
        """Analyze dependency impacts using the modular components."""
        # Find affected components
        affected_files = set()
        affected_functions = set()
        affected_classes = set()
        affected_modules = set()

        for change in changes:
            file_path = change.filename
            affected_files.add(file_path)

            # Find files that import this file
            for import_file, imports in dependency_graph.imports.items():
                module_name = self.graph_analyzer._file_path_to_module_name(file_path)
                if module_name in imports or any(module_name in imp for imp in imports):
                    affected_files.add(import_file)

            # Add functions and classes from this file
            for _, node in dependency_graph.nodes.items():
                if node.file_path == file_path:
                    if node.node_type == "function":
                        affected_functions.add(node.name)
                    elif node.node_type == "class":
                        affected_classes.add(node.name)

            # Add the module itself
            affected_modules.add(file_path)

        # Detect dependency issues using validator
        assert self.dependency_validator is not None, "Dependency validator must be initialized"
        circular_dependencies = self.dependency_validator.detect_circular_dependencies()
        missing_dependencies = self.dependency_validator.validate_dependency_chains(affected_files)

        # Calculate impact and generate recommendations using scorer
        return self.impact_scorer.analyze_dependency_impacts(
            changes,
            affected_files,
            affected_functions,
            affected_classes,
            affected_modules,
            circular_dependencies,
            missing_dependencies,
        )

    def get_dependency_graph(self) -> DependencyGraph | None:
        """Get the current dependency graph if available."""
        return self.graph_analyzer.get_dependency_graph()

    def get_dependency_report(self) -> dict:
        """Get a comprehensive dependency validation report."""
        if self.dependency_validator:
            return self.dependency_validator.generate_dependency_report()
        return {"error": "Dependency validator not initialized"}

    def get_change_patterns(self, changes: list[FileChange]) -> dict:
        """Get detected change patterns from the PR changes."""
        return self.change_detector.detect_change_patterns(changes)

    def get_change_metadata(self, changes: list[FileChange]) -> dict:
        """Get metadata about the changes."""
        return self.change_detector.extract_change_metadata(changes)

    def reset_analyzer(self) -> None:
        """Reset the analyzer to start fresh."""
        self.graph_analyzer.reset_graph()
        self.dependency_validator = None


# Re-export for backward compatibility
__all__ = [
    "PRImpactAnalyzer",
    "DependencyNode",
    "DependencyGraph",
    "ImpactResult",
]
