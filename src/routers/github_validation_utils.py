"""
GitHub Pull Request Validation Utilities

Helper functions for validating pull requests for security, performance, and quality.
Split from github_pull_request_operations.py to reduce file size.
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.github_api_models import AnalysisIssue, PRValidationResult

logger = logging.getLogger(__name__)


async def validate_pr_security(pr_files) -> PRValidationResult:
    """Validate PR for security issues with comprehensive error handling."""
    issues: list[AnalysisIssue] = []
    suggestions: list[str] = []

    # Analyze each changed file for security issues
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


def basic_security_check(file, issues: list[Any], suggestions: list[Any]) -> None:
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
                        rule="code-injection-risk",
                    )
                )

            if "password" in line_content and ("=" in line_content or ":" in line_content):
                issues.append(
                    AnalysisIssue(
                        type="security",
                        severity="medium",
                        message="Potential hardcoded credential detected",
                        file=file.filename,
                        line=i + 1,
                        column=1,
                        rule="hardcoded-credential",
                    )
                )

    if not suggestions:
        suggestions.append("Use static analysis tools for comprehensive security scanning")


async def validate_pr_performance(pr_files) -> PRValidationResult:
    """Validate PR for performance issues with augmented pattern detection."""
    issues: list[AnalysisIssue] = []
    suggestions: list[str] = []

    # Performance checks on changed files
    for file in pr_files:
        if file.patch and file.filename.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
            # Check for common performance anti-patterns
            patch_lines = file.patch.split("\n")
            for i, line in enumerate(patch_lines):
                if line.startswith("+"):  # Added line
                    line_content = line[1:].strip()

                    # Check for inefficient patterns
                    if "for i in range(len(" in line_content:
                        issues.append(
                            AnalysisIssue(
                                type="performance",
                                severity="medium",
                                message=(
                                    "Inefficient loop pattern detected - consider direct iteration"
                                ),
                                line=i + 1,
                                file=file.filename,
                                column=1,
                                rule="inefficient-loop",
                            )
                        )

                    if "time.sleep(" in line_content or "Thread.sleep(" in line_content:
                        issues.append(
                            AnalysisIssue(
                                type="performance",
                                severity="high",
                                message="Blocking sleep call in added code",
                                line=i + 1,
                                file=file.filename,
                                column=1,
                                rule="blocking-sleep",
                            )
                        )

                    # Additional performance patterns
                    if ".append(" in line_content and "for " in line_content:
                        issues.append(
                            AnalysisIssue(
                                type="performance",
                                severity="low",
                                message="Consider list comprehension for better performance",
                                line=i + 1,
                                file=file.filename,
                                column=1,
                                rule="list-comprehension-opportunity",
                            )
                        )

                    if "+" in line_content and "str(" in line_content and "for " in line_content:
                        issues.append(
                            AnalysisIssue(
                                type="performance",
                                severity="medium",
                                message="String concatenation in loop - consider join() method",
                                line=i + 1,
                                file=file.filename,
                                column=1,
                                rule="inefficient-string-concat",
                            )
                        )

    score = max(0, 100 - len(issues) * 10)
    passed = len([i for i in issues if i.severity == "high"]) == 0

    if not suggestions:
        suggestions.append("Consider performance testing for critical code paths")
        suggestions.append("Use profiling tools to identify bottlenecks")

    return PRValidationResult(
        validation_type="performance",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
    )


async def validate_pr_quality(pr_files) -> PRValidationResult:
    """Validate PR for code quality."""
    issues: list[AnalysisIssue] = []
    suggestions: list[str] = []

    # Basic quality checks
    total_additions = sum(file.additions for file in pr_files)
    sum(file.deletions for file in pr_files)

    # Check if PR is too large
    if total_additions > 500:
        issues.append(
            AnalysisIssue(
                type="quality",
                severity="medium",
                message=f"Large PR with {total_additions} additions - consider splitting",
                line=1,
                column=1,
                file="pull_request",
                rule="large-pr",
            )
        )

    # Check for files with many changes
    for file in pr_files:
        if file.changes > 100:
            issues.append(
                AnalysisIssue(
                    type="quality",
                    severity="low",
                    message=(f"File {file.filename} has many changes ({file.changes})"),
                    file=file.filename,
                    line=1,
                    column=1,
                    rule="large-file-change",
                )
            )

        # README-specific validation
        if file.filename.lower() in [
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]:
            readme_issues = await validate_readme_changes(file)
            issues.extend(readme_issues)

    score = max(0, 100 - len(issues) * 5)
    passed = len(issues) <= 3

    suggestions.append("Ensure PR has adequate test coverage")
    suggestions.append("Consider adding documentation for significant changes")

    return PRValidationResult(
        validation_type="quality",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
    )


async def validate_readme_changes(file) -> list[AnalysisIssue]:
    """Validate README file changes for content preservation."""
    issues: list[AnalysisIssue] = []

    if not file.patch:
        return issues

    patch_lines = file.patch.split("\n")

    # Count additions vs deletions
    additions = len(
        [line for line in patch_lines if line.startswith("+") and not line.startswith("+++")]
    )
    deletions = len(
        [line for line in patch_lines if line.startswith("-") and not line.startswith("---")]
    )

    # Flag complete replacement - if deletions significantly exceed additions
    if deletions > 50 and deletions > additions * 2:
        issues.append(
            AnalysisIssue(
                type="quality",
                severity="high",
                message=(
                    f"README appears to be completely replaced ({deletions} deletions vs "
                    f"{additions} additions) - verify important content is preserved"
                ),
                file=file.filename,
                line=1,
                column=1,
                rule="readme-complete-replacement",
            )
        )

    # Check for removal of common important sections
    deleted_content = []
    for line in patch_lines:
        if line.startswith("-") and not line.startswith("---"):
            deleted_content.append(line[1:].lower())

    deleted_text = " ".join(deleted_content)
    important_sections = [
        "installation",
        "getting started",
        "usage",
        "api",
        "configuration",
        "examples",
        "license",
        "contributing",
    ]

    removed_sections = []
    for section in important_sections:
        if section in deleted_text and not any(
            f"+{section}" in line.lower() for line in patch_lines
        ):
            removed_sections.append(section)

    if removed_sections:
        issues.append(
            AnalysisIssue(
                type="quality",
                severity="medium",
                message=(
                    f"README changes may remove important sections: {', '.join(removed_sections)}"
                ),
                file=file.filename,
                line=1,
                column=1,
                rule="readme-section-removal",
            )
        )

    # Check for complete content replacement (high deletion ratio)
    if deletions > 20 and deletions / max(additions, 1) > 3:
        issues.append(
            AnalysisIssue(
                type="quality",
                severity="medium",
                message=(
                    "README has high deletion-to-addition ratio - consider preserving useful existing content"
                ),
                file=file.filename,
                line=1,
                column=1,
                rule="readme-high-deletion-ratio",
            )
        )

    return issues
