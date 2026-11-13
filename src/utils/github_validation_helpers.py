"""
GitHub PR Validation Helper Functions

Extracted from github.py to reduce file size and improve maintainability.
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.github_api_models import AnalysisIssue, PRValidationResult

logger = logging.getLogger(__name__)


async def validate_and_get_container_for_tenant(container_name: str, tenant_id: str) -> Any:
    """Validate container belongs to tenant and return its ContainerInfo as a DICT.

    coding-containers answers ``{"status", "message", "container": {<ContainerInfo>}}``
    (a plain dict, never a model), so the envelope is unwrapped here; an already-flat
    payload is returned as is. Read fields with ``.get`` (see
    :func:`container_codebase_path`) — attribute access on this dict was the 500
    ``'dict' object has no attribute 'codebase_path'`` on /process-validation.
    """
    from coding_client.clients import ServiceError

    from main import get_workspaces_client
    from services.outbound import optional_sibling_headers

    try:
        client = get_workspaces_client()
        container = await client.get_workspace(
            container_name,
            headers=optional_sibling_headers(tenant_id) or None,
        )
        if isinstance(container, dict) and isinstance(container.get("container"), dict):
            return container["container"]
        return container
    except ServiceError as e:
        if e.status_code == 404:
            raise ValueError(f"Container {container_name} not found") from e
        raise RuntimeError(f"Containers service error ({e.status_code}): {e}") from e
    except RuntimeError:
        raise


def container_codebase_path(container_info: Any) -> str | None:
    """``codebase_path`` off a container record, whether a dict or a model."""
    if isinstance(container_info, dict):
        value = container_info.get("codebase_path")
    else:
        value = getattr(container_info, "codebase_path", None)
    return value if isinstance(value, str) else None


def basic_security_check(file, issues: list[AnalysisIssue], suggestions: list[str]) -> None:
    """Basic security pattern detection as fallback."""
    if not file.patch:
        return

    patch_lines = file.patch.split("\n")
    for i, line in enumerate(patch_lines):
        if line.startswith("+"):  # Added line
            line_content = line[1:].strip().lower()

            # Check for common security anti-patterns
            if any(
                pattern in line_content for pattern in ["eval(", "exec(", "input(", "raw_input("]
            ):
                issues.append(
                    AnalysisIssue(
                        type="security",
                        severity="high",
                        message="Potential code injection vulnerability detected",
                        file=file.filename,
                        line=i + 1,
                        column=1,
                        rule="pr-security-injection",
                    )
                )
                suggestions.append(
                    f"Review {file.filename} line {i + 1}: Avoid using dangerous functions like eval/exec"
                )

            # Check for hardcoded credentials
            if any(
                pattern in line_content
                for pattern in ["password=", "api_key=", "secret=", "token="]
            ):
                issues.append(
                    AnalysisIssue(
                        type="security",
                        severity="critical",
                        message="Potential hardcoded credentials detected",
                        file=file.filename,
                        line=i + 1,
                        column=1,
                        rule="pr-security-credentials",
                    )
                )
                suggestions.append(f"Use environment variables for secrets in {file.filename}")


async def validate_pr_security(pr_files) -> PRValidationResult:
    """Validate PR for security issues with comprehensive error handling."""
    issues: list[AnalysisIssue] = []
    suggestions: list[str] = []

    # Analyze each changed file for security issues. (This used to try
    # ``from processing.analyzers.python_analyzer import PythonCodeProcessor`` first:
    # a module from another service that resolves under no sys.path this service
    # runs with, so it raised ImportError for every file and always ended up here.)
    for file in pr_files:
        if file.patch:  # File has changes
            basic_security_check(file, issues, suggestions)

    # Calculate score based on issues
    critical_issues = len([i for i in issues if i.severity == "critical"])
    high_issues = len([i for i in issues if i.severity == "high"])

    score = max(0, 100 - (critical_issues * 30 + high_issues * 15))
    passed = critical_issues == 0 and high_issues <= 2

    if not suggestions:
        suggestions.append("Consider adding security-focused unit tests")

    return PRValidationResult(
        validation_type="security",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
    )


async def validate_pr_performance(pr_files) -> PRValidationResult:
    """Validate PR for performance issues."""
    issues: list[AnalysisIssue] = []

    # Analyze each file for performance issues
    for file in pr_files:
        if not file.patch:
            continue

        # Check for common performance anti-patterns
        patch_lines = file.patch.split("\n")
        for i, line in enumerate(patch_lines):
            if line.startswith("+"):  # Added line
                line_content = line[1:].strip().lower()

                # Check for inefficient patterns
                if "sleep(" in line_content or "time.sleep(" in line_content:
                    issues.append(
                        AnalysisIssue(
                            type="performance",
                            severity="warning",
                            message="Sleep statement detected - may impact performance",
                            file=file.filename,
                            line=i + 1,
                            column=1,
                            rule="pr-performance-sleep",
                        )
                    )

    # Calculate score based on issues
    warning_issues = len([i for i in issues if i.severity == "warning"])
    score = max(0, 100 - (warning_issues * 10))
    passed = score >= 70

    return PRValidationResult(
        validation_type="performance",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=["Consider profiling critical paths for performance impact"],
    )


async def validate_pr_quality(pr_files) -> PRValidationResult:
    """Validate PR for code quality issues."""
    issues: list[AnalysisIssue] = []
    suggestions: list[str] = []

    # Analyze each file for quality issues
    for file in pr_files:
        if not file.patch:
            continue

        # Check for TODOs or FIXMEs in new code
        patch_lines = file.patch.split("\n")
        for i, line in enumerate(patch_lines):
            if line.startswith("+"):  # Added line
                if "TODO" in line or "FIXME" in line:
                    issues.append(
                        AnalysisIssue(
                            type="quality",
                            severity="warning",
                            message="Unfinished code marker detected",
                            file=file.filename,
                            line=i + 1,
                            column=1,
                            rule="pr-quality-todo",
                        )
                    )
                    suggestions.append(f"Address TODO/FIXME in {file.filename} before merging")

                # Check for commented code
                if line.strip().startswith("+ #") and len(line.strip()) > 10:
                    issues.append(
                        AnalysisIssue(
                            type="quality",
                            severity="info",
                            message="Commented code detected",
                            file=file.filename,
                            line=i + 1,
                            column=1,
                            rule="pr-quality-commented-code",
                        )
                    )

    # Calculate score based on issues
    warning_issues = len([i for i in issues if i.severity == "warning"])
    info_issues = len([i for i in issues if i.severity == "info"])

    score = max(0, 100 - (warning_issues * 10 + info_issues * 2))
    passed = score >= 70

    if not suggestions:
        suggestions.append("Consider adding unit tests for new functionality")

    return PRValidationResult(
        validation_type="quality",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
    )


async def validate_readme_changes(file) -> list[AnalysisIssue]:
    """Validate README changes for quality."""
    issues: list[AnalysisIssue] = []

    if not file.patch:
        return issues

    # Check for basic README quality
    patch_lines = file.patch.split("\n")
    added_lines = [line[1:] for line in patch_lines if line.startswith("+")]
    removed_lines = [line[1:] for line in patch_lines if line.startswith("-")]

    # Check if important sections were removed
    important_sections = ["## Installation", "## Usage", "## License"]
    for section in important_sections:
        if any(section in line for line in removed_lines):
            if not any(section in line for line in added_lines):
                issues.append(
                    AnalysisIssue(
                        type="documentation",
                        severity="warning",
                        message=f"Important section '{section}' was removed",
                        file=file.filename,
                        line=1,
                        column=1,
                        rule="pr-doc-section-removed",
                    )
                )

    # Check for broken markdown
    for i, line in enumerate(added_lines):
        # Check for unclosed code blocks
        if "```" in line:
            # Simple check - count backticks
            backtick_count = line.count("```")
            if backtick_count % 2 != 0:
                issues.append(
                    AnalysisIssue(
                        type="documentation",
                        severity="error",
                        message="Unclosed code block detected",
                        file=file.filename,
                        line=i + 1,
                        column=1,
                        rule="pr-doc-markdown-syntax",
                    )
                )

    return issues
