"""
GitHub PR validation integration service.

This module integrates the comprehensive PR validation system with the existing
GitHub API endpoint, providing comprehensive validation while maintaining backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from coding_client.models.pr_validation import (
    CodebaseContext,
    PRAnalysisContext,
    ValidationIssue,
    ValidationResult,
)

from schemas.github_api_models import AnalysisIssue, PRValidationResult, ValidatePRRequest
from services.pr_validation.core import PRValidationConfig, PRValidationEngine
from services.pr_validation.scoring.recommendation_generator import (
    PRValidationRecommendationGenerator,
)
from services.pr_validation.scoring.risk_calculator import RiskCalculator

logger = logging.getLogger(__name__)

#: The GitHub "PR file" fields the diff engine reads. Listed explicitly rather than
#: read off ``File.raw_data``, which may lazily complete the object over the network.
_PR_FILE_FIELDS = (
    "filename",
    "status",
    "additions",
    "deletions",
    "changes",
    "patch",
    "previous_filename",
    "sha",
    "blob_url",
    "raw_url",
)


def github_file_to_dict(file: Any) -> dict[str, Any]:
    """Normalise one PR file to the dict shape ``DiffParser.parse_github_diff`` reads.

    Callers hand this service PyGithub ``File`` objects (``pr.get_files()``), which
    expose attributes, not ``.get``. The diff engine is dict-based, so passing the
    objects straight through failed every comprehensive validation in prod with
    ``'File' object has no attribute 'get'`` (D15). Dicts pass through unchanged.
    """
    if isinstance(file, dict):
        return file
    return {field: getattr(file, field, None) for field in _PR_FILE_FIELDS}


@dataclass
class GitHubPRValidationData:
    """Structured GitHub PR data for validation."""

    repo_owner: str
    repo_name: str
    pr_number: int
    files: list[Any]
    base_sha: str
    head_sha: str
    pr_title: str
    pr_body: str
    author: str
    created_at: str
    updated_at: str


class PRGitHubValidationService:
    """
    Service that integrates comprehensive PR validation with GitHub API.

    This service acts as a bridge between the existing GitHub endpoint
    and the new comprehensive PR validation system, providing advanced
    analysis while maintaining API compatibility.
    """

    def __init__(self):
        self.validation_engine = PRValidationEngine()
        self.risk_calculator = RiskCalculator()
        self.recommendation_generator = PRValidationRecommendationGenerator()
        self.github_config = PRValidationConfig(
            enable_diff_analysis=True,
            enable_semantic_analysis=True,
            enable_impact_assessment=True,
            enable_security_analysis=True,
            enable_performance_analysis=True,
            enable_maintainability_analysis=True,
            overall_timeout=120,
            analyzer_timeout=30,
            analysis_intensity=3,
        )

    async def validate_github_pr(
        self,
        request: ValidatePRRequest,
        pr_files: list[Any],
        pr_metadata: dict[str, Any] | None = None,
    ) -> list[PRValidationResult]:
        """
        Validate a GitHub PR using the comprehensive validation system.

        Args:
            request: GitHub PR validation request
            pr_files: GitHub PR files from the API
            pr_metadata: Optional PR metadata from GitHub API (title, body, author, etc.)

        Returns:
            List of PRValidationResult objects for each validation type
        """
        try:
            pr_data = self._convert_github_pr_data(request, pr_files, pr_metadata)
            codebase_context = CodebaseContext(
                total_files=len(pr_files),
                total_lines=sum(f.changes for f in pr_files),
                main_languages=self._detect_languages(pr_files),
                test_framework=None,
                documentation_format=None,
                dependencies=[],
                architecture_patterns=[],
            )
            validation_types = request.validation_types or [
                "security",
                "performance",
                "quality",
            ]
            if self._should_use_comprehensive_validation(validation_types):
                return await self._run_comprehensive_validation(
                    pr_data, codebase_context, validation_types
                )
            else:
                return await self._run_legacy_validation(pr_data, validation_types, pr_metadata)
        except Exception as e:
            logger.error(f"GitHub PR validation failed: {e}")
            return [
                PRValidationResult(
                    validation_type="error",
                    passed=False,
                    score=0.0,
                    issues=[
                        AnalysisIssue(
                            type="validation_error",
                            severity="critical",
                            message=f"Validation failed: {str(e)[:100]}",
                            file="system",
                            line=1,
                            column=1,
                            rule="github_validation_error",
                        )
                    ],
                    suggestions=["Manual review required due to validation error"],
                )
            ]

    def _convert_github_pr_data(
        self,
        request: ValidatePRRequest,
        pr_files: list[Any],
        pr_metadata: dict[str, Any] | None = None,
    ) -> GitHubPRValidationData:
        """Convert GitHub API data to internal format.

        Args:
            request: The validation request containing repo and PR information
            pr_files: List of file objects from GitHub API containing PR changes
            pr_metadata: Optional PR metadata dict from GitHub API

        Returns:
            GitHubPRValidationData object with extracted PR information
        """
        from datetime import datetime

        current_time = datetime.now(UTC).isoformat()
        base_sha = "main"
        head_sha = f"pr-{request.pr_number}"
        pr_title = f"Pull Request #{request.pr_number}"
        pr_body = f"Validation requested for {request.repo_owner}/{request.repo_name} PR #{request.pr_number}"
        author = request.repo_owner
        created_at = current_time
        updated_at = current_time
        if pr_metadata and isinstance(pr_metadata, dict):
            if "base" in pr_metadata and isinstance(pr_metadata["base"], dict):
                base_sha = pr_metadata["base"].get("sha", base_sha)
                if not base_sha and "ref" in pr_metadata["base"]:
                    base_sha = pr_metadata["base"]["ref"]
            if "head" in pr_metadata and isinstance(pr_metadata["head"], dict):
                head_sha = pr_metadata["head"].get("sha", head_sha)
                if not head_sha and "ref" in pr_metadata["head"]:
                    head_sha = pr_metadata["head"]["ref"]
            pr_title = pr_metadata.get("title", pr_title)
            pr_body = pr_metadata.get("body", pr_body) or pr_body
            if "user" in pr_metadata and isinstance(pr_metadata["user"], dict):
                author = pr_metadata["user"].get("login", author)
            elif "author" in pr_metadata:
                author = pr_metadata.get("author", author)
            created_at = pr_metadata.get("created_at", created_at)
            updated_at = pr_metadata.get("updated_at", updated_at)
        elif pr_files:
            for file in pr_files:
                if hasattr(file, "sha"):
                    head_sha = str(file.sha)[:7]
                    break
                elif hasattr(file, "blob_url") and isinstance(file.blob_url, str):
                    parts = file.blob_url.split("/")
                    if len(parts) > 6 and parts[5] == "blob":
                        head_sha = parts[6][:7]
                        break
            if hasattr(pr_files[0], "last_modified"):
                try:
                    updated_at = str(pr_files[0].last_modified)
                except Exception:
                    pass
        return GitHubPRValidationData(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name,
            pr_number=request.pr_number,
            files=pr_files,
            base_sha=base_sha,
            head_sha=head_sha,
            pr_title=pr_title,
            pr_body=pr_body,
            author=author,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _detect_languages(self, pr_files: list[Any]) -> list[str]:
        """Detect programming languages from PR files."""
        languages = set()
        for file in pr_files:
            filename = file.filename.lower()
            if filename.endswith(".py"):
                languages.add("python")
            elif filename.endswith((".js", ".jsx")):
                languages.add("javascript")
            elif filename.endswith((".ts", ".tsx")):
                languages.add("typescript")
            elif filename.endswith(".java"):
                languages.add("java")
            elif filename.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
                languages.add("c++")
        return list(languages)

    def _should_use_comprehensive_validation(self, validation_types: list[str]) -> bool:
        """Determine if comprehensive validation should be used."""
        return (
            len(validation_types) > 2
            or "comprehensive" in validation_types
            or "advanced" in validation_types
        )

    async def _run_comprehensive_validation(
        self,
        pr_data: GitHubPRValidationData,
        codebase_context: CodebaseContext,
        validation_types: list[str],
    ) -> list[PRValidationResult]:
        """Run comprehensive PR validation using the full engine."""
        try:
            pr_api_data = {
                "base": {"sha": pr_data.base_sha},
                "head": {"sha": pr_data.head_sha},
                "files": [github_file_to_dict(f) for f in pr_data.files],
                "number": pr_data.pr_number,
                "title": pr_data.pr_title,
                "body": pr_data.pr_body,
                "user": {"login": pr_data.author},
                "created_at": pr_data.created_at,
                "updated_at": pr_data.updated_at,
            }
            validation_result = await self.validation_engine.validate_pr(
                pr_api_data, codebase_context
            )
            if (
                validation_result.change_impact
                and validation_result.change_impact.affected_functions
            ):
                pass
            # Scoring context only; built by the engine's own parser so its shape
            # always matches coding_client's PRDiff. (This was a bare
            # ``from models.pr_validation import PRDiff`` — a module that exists in
            # no installed package — so every comprehensive run died here with
            # ``No module named 'models'`` and fell back to legacy: D16.)
            empty_pr_diff = self.validation_engine.diff_parser.create_pr_diff(
                [], pr_data.base_sha, pr_data.head_sha, metadata={}
            )
            analysis_context = PRAnalysisContext(
                pr_diff=empty_pr_diff,
                codebase_context=codebase_context,
                validation_config=self.github_config.__dict__,
            )
            risk_score = self.risk_calculator.calculate_overall_risk(
                analysis_context, validation_result.issues
            )
            recommendations = self.recommendation_generator.generate_recommendations(
                validation_result, analysis_context, risk_score
            )
            return self._convert_comprehensive_results(
                validation_result, risk_score, recommendations, validation_types
            )
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            return await self._run_legacy_validation(pr_data, validation_types, None)

    async def _run_legacy_validation(
        self,
        pr_data: GitHubPRValidationData,
        validation_types: list[str],
        pr_metadata: dict[str, Any] | None = None,
    ) -> list[PRValidationResult]:
        """Run legacy validation for backward compatibility."""
        results = []
        for validation_type in validation_types:
            if validation_type == "security":
                result = await self._legacy_security_validation(pr_data.files)
            elif validation_type == "performance":
                result = await self._legacy_performance_validation(pr_data.files)
            elif validation_type == "quality":
                result = await self._legacy_quality_validation(pr_data.files)
            else:
                continue
            result.validation_type = validation_type
            results.append(result)
        return results

    async def _legacy_security_validation(self, pr_files: list[Any]) -> PRValidationResult:
        """Legacy security validation using pattern matching."""
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        for file in pr_files:
            if file.patch:
                patch_lines = file.patch.split("\n")
                for i, line in enumerate(patch_lines):
                    if line.startswith("+"):
                        line_content = line[1:].strip().lower()
                        if any(
                            pattern in line_content for pattern in ["eval(", "exec(", "system("]
                        ):
                            issues.append(
                                AnalysisIssue(
                                    type="security",
                                    severity="high",
                                    message="Potentially dangerous function detected",
                                    file=file.filename,
                                    line=i + 1,
                                    column=1,
                                    rule="dangerous-function",
                                )
                            )
                        if "password" in line_content and "=" in line_content:
                            issues.append(
                                AnalysisIssue(
                                    type="security",
                                    severity="medium",
                                    message="Potential hardcoded credential",
                                    file=file.filename,
                                    line=i + 1,
                                    column=1,
                                    rule="hardcoded-secret",
                                )
                            )
        score = max(0, 100 - len(issues) * 15)
        passed = len([i for i in issues if i.severity in ["critical", "high"]]) == 0
        if not suggestions:
            suggestions.append("Consider using static security analysis tools")
        return PRValidationResult(
            validation_type="security",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    async def _legacy_performance_validation(self, pr_files: list[Any]) -> PRValidationResult:
        """Legacy performance validation using pattern matching."""
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        for file in pr_files:
            if file.patch and file.filename.endswith((".py", ".js", ".ts")):
                patch_lines = file.patch.split("\n")
                for i, line in enumerate(patch_lines):
                    if line.startswith("+"):
                        line_content = line[1:].strip()
                        if "for i in range(len(" in line_content:
                            issues.append(
                                AnalysisIssue(
                                    type="performance",
                                    severity="medium",
                                    message="Inefficient loop pattern",
                                    file=file.filename,
                                    line=i + 1,
                                    column=1,
                                    rule="inefficient-loop",
                                )
                            )
                        if any(
                            pattern in line_content for pattern in ["time.sleep(", "Thread.sleep("]
                        ):
                            issues.append(
                                AnalysisIssue(
                                    type="performance",
                                    severity="high",
                                    message="Blocking operation in code",
                                    file=file.filename,
                                    line=i + 1,
                                    column=1,
                                    rule="blocking-operation",
                                )
                            )
        score = max(0, 100 - len(issues) * 10)
        passed = len([i for i in issues if i.severity == "high"]) == 0
        if not suggestions:
            suggestions.append("Consider performance testing for critical paths")
        return PRValidationResult(
            validation_type="performance",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    async def _legacy_quality_validation(self, pr_files: list[Any]) -> PRValidationResult:
        """Legacy quality validation with comprehensive README checking."""
        issues: list[AnalysisIssue] = []
        suggestions: list[str] = []
        total_additions = sum(file.additions for file in pr_files)
        if total_additions > 500:
            issues.append(
                AnalysisIssue(
                    type="quality",
                    severity="medium",
                    message=f"Large PR with {total_additions} additions",
                    file="pull_request",
                    line=1,
                    column=1,
                    rule="large-pr",
                )
            )
        for file in pr_files:
            if file.filename.lower() in [
                "readme.md",
                "readme.rst",
                "readme.txt",
                "readme",
            ]:
                readme_issues = await self._validate_readme_changes_comprehensive(file)
                issues.extend(readme_issues)
            if file.changes > 100:
                issues.append(
                    AnalysisIssue(
                        type="quality",
                        severity="low",
                        message=f"File has many changes ({file.changes})",
                        file=file.filename,
                        line=1,
                        column=1,
                        rule="large-file-change",
                    )
                )
        score = max(0, 100 - len(issues) * 5)
        passed = len(issues) <= 3
        suggestions.append("Ensure adequate test coverage")
        suggestions.append("Consider documentation updates")
        return PRValidationResult(
            validation_type="quality",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    async def _validate_readme_changes_comprehensive(self, file) -> list[AnalysisIssue]:
        """Comprehensive README validation that detects complete replacements."""
        issues: list[AnalysisIssue] = []
        if not file.patch:
            return issues
        patch_lines = file.patch.split("\n")
        additions = len(
            [line for line in patch_lines if line.startswith("+") and (not line.startswith("+++"))]
        )
        deletions = len(
            [line for line in patch_lines if line.startswith("-") and (not line.startswith("---"))]
        )
        if deletions > 20 and deletions > additions * 1.5:
            issues.append(
                AnalysisIssue(
                    type="quality",
                    severity="high",
                    message=f"README appears to be completely replaced ({deletions} deletions vs {additions} additions) - verify important content is preserved",
                    file=file.filename,
                    line=1,
                    column=1,
                    rule="readme-complete-replacement",
                )
            )
        deleted_content = []
        for line in patch_lines:
            if line.startswith("-") and (not line.startswith("---")):
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
            if section in deleted_text and (
                not any(f"+{section}" in line.lower() for line in patch_lines)
            ):
                removed_sections.append(section)
        if removed_sections:
            issues.append(
                AnalysisIssue(
                    type="quality",
                    severity="medium",
                    message=f"README changes may remove important sections: {', '.join(removed_sections)}",
                    file=file.filename,
                    line=1,
                    column=1,
                    rule="readme-section-removal",
                )
            )
        if deletions > 50:
            added_lines = [
                line[1:]
                for line in patch_lines
                if line.startswith("+") and (not line.startswith("+++"))
            ]
            added_text = " ".join(added_lines).lower()
            generic_indicators = [
                "lorem ipsum",
                "example project",
                "template",
                "placeholder",
                "your project name",
            ]
            if any(indicator in added_text for indicator in generic_indicators):
                issues.append(
                    AnalysisIssue(
                        type="quality",
                        severity="high",
                        message="README replacement contains generic/template content - ensure it's customized for this project",
                        file=file.filename,
                        line=1,
                        column=1,
                        rule="readme-generic-content",
                    )
                )
        return issues

    def _convert_comprehensive_results(
        self,
        validation_result: ValidationResult,
        risk_score,
        recommendations,
        validation_types: list[str],
    ) -> list[PRValidationResult]:
        """Convert comprehensive validation results to GitHub API format."""
        results = []
        issues_by_type: dict[str, list[ValidationIssue]] = {}
        for issue in validation_result.issues:
            issue_type = self._categorize_issue_type(issue.type)
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        for validation_type in validation_types:
            if validation_type in ["comprehensive", "advanced"]:
                continue
            type_issues = issues_by_type.get(validation_type, [])
            converted_issues = []
            for issue in type_issues:
                # Handle both ValidationIssue and CodeIssue types
                severity = (
                    str(issue.severity.value)
                    if hasattr(issue.severity, "value")
                    else str(issue.severity)
                )
                message = getattr(issue, "message", getattr(issue, "description", str(issue)))
                file_path = getattr(issue, "file", getattr(issue, "file_path", "unknown"))
                line_num = getattr(issue, "line", getattr(issue, "line_number", 1)) or 1
                rule_val = getattr(issue, "rule", None) or f"{validation_type}-check"

                converted_issues.append(
                    AnalysisIssue(
                        type=validation_type,
                        severity=severity,
                        message=message,
                        file=file_path,
                        line=line_num,
                        column=1,
                        rule=rule_val,
                    )
                )
            type_score = self._calculate_type_score(validation_type, risk_score, type_issues)
            passed = self._determine_type_passed(validation_type, type_issues, type_score)
            type_suggestions = self._get_type_suggestions(validation_type, recommendations)
            results.append(
                PRValidationResult(
                    validation_type=validation_type,
                    passed=passed,
                    score=type_score,
                    issues=converted_issues,
                    suggestions=type_suggestions,
                )
            )
        return results

    def _categorize_issue_type(self, issue_type: str) -> str:
        """Categorize issue type for grouping."""
        if any(
            keyword in issue_type.lower() for keyword in ["security", "auth", "crypto", "injection"]
        ):
            return "security"
        elif any(
            keyword in issue_type.lower()
            for keyword in ["performance", "complexity", "loop", "cache"]
        ):
            return "performance"
        else:
            return "quality"

    def _calculate_type_score(self, validation_type: str, risk_score, issues: list) -> float:
        """Calculate score for a specific validation type."""
        if not risk_score:
            critical_count = len([i for i in issues if i.severity == "critical"])
            high_count = len([i for i in issues if i.severity == "high"])
            return max(0, 100 - (critical_count * 30 + high_count * 15))
        dimension_map = {
            "security": "SECURITY",
            "performance": "PERFORMANCE",
            "quality": "MAINTAINABILITY",
        }
        dimension_name = dimension_map.get(validation_type, "MAINTAINABILITY")
        dimension_risk = 0
        for dim, score in risk_score.dimension_scores.items():
            if dim.value == dimension_name:
                dimension_risk = score
                break
        return max(0, 100 - dimension_risk * 10)

    def _determine_type_passed(self, validation_type: str, issues: list, score: float) -> bool:
        """Determine if validation type passed."""
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        if validation_type == "security":
            return critical_issues == 0 and high_issues <= 1
        elif validation_type == "performance":
            return high_issues == 0 and score >= 70
        else:
            return score >= 60 and critical_issues == 0

    def _get_type_suggestions(self, validation_type: str, recommendations) -> list[str]:
        """Get suggestions for a specific validation type."""
        if not recommendations:
            return [f"Consider {validation_type}-focused review"]
        type_suggestions = []
        for rec in recommendations[:3]:
            if validation_type in rec.type.lower() or rec.priority in [
                "critical",
                "high",
            ]:
                type_suggestions.append(rec.action)
        if not type_suggestions:
            type_suggestions.append(f"Review {validation_type} aspects of the changes")
        return type_suggestions


async def validate_pr_comprehensive(
    request: ValidatePRRequest,
    pr_files: list[Any],
    pr_metadata: dict[str, Any] | None = None,
) -> list[PRValidationResult]:
    """
    Integration function for the existing GitHub endpoint.

    This function can be called from the existing validate_pull_request
    endpoint to provide comprehensive validation capabilities.

    Args:
        request: ValidatePRRequest object with repo and PR details
        pr_files: List of PR file objects from GitHub API
        pr_metadata: Optional PR metadata containing title, body, author, etc.

    Returns:
        List of PRValidationResult objects for each validation type
    """
    service = PRGitHubValidationService()
    return await service.validate_github_pr(request, pr_files, pr_metadata)
