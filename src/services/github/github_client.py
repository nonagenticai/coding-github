"""GitHub API client management module.

This module provides centralized GitHub API client operations including:
- GitHub service initialization and management
- API rate limiting and error handling
- Repository information retrieval
- Pull request analysis and management
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from coding_client.models.github import GitHubRepoInfo as RepoInfo
from coding_client.models.github import PRFileChange as FileChange
from coding_client.models.github import (
    PullRequest,
)

from services.github.github_service import (
    get_github_service,
)

logger = logging.getLogger(__name__)


# TypedDict definitions for type safety
class PRAnalysisDict(TypedDict):
    """Analysis results for pull request data."""

    pull_request: PullRequest
    files: list[FileChange]
    statistics: dict[str, int]
    languages: dict[str, dict[str, int]]
    total_additions: int
    total_deletions: int


class RepositoryInfoDict(TypedDict):
    """Repository information from GitHub API."""

    owner: str
    name: str
    default_branch: str
    description: str | None
    language: str | None


class GitHubClient:
    """Centralized GitHub API client operations."""

    def __init__(self) -> None:
        """Initialize GitHub client."""
        pass

    async def get_repository_info(self, repo_url: str) -> RepoInfo:
        """Get repository metadata and structure from GitHub API"""
        try:
            logger.info("Fetching repository information for %s", repo_url)
            repo_info = self._parse_repo_url(repo_url)
            repo_name = f"{repo_info['owner']}/{repo_info['name']}"

            # Get actual repository information from GitHub API
            async with await get_github_service(repo_name) as github_service:
                logger.debug("Calling GitHub API for repository info: %s", repo_name)
                repo_data = await github_service.get_repository_info(repo_name)

                repo_info_obj = RepoInfo(
                    owner=repo_data.get("owner", repo_info["owner"]),
                    name=repo_data.get("name", repo_info["name"]),
                    url=repo_url,
                    default_branch=repo_data.get("default_branch", "main"),
                    description=repo_data.get("description"),
                    language=repo_data.get("language"),
                )

                logger.info("Successfully retrieved repository info for %s", repo_name)
                return repo_info_obj

        except (ValueError, KeyError, OSError) as e:
            logger.error("Failed to get repository info for %s: %s", repo_url, e)
            raise

    async def read_pr_file(
        self, repo_url: str, pr_number: int, filepath: str, ref: str = "head"
    ) -> str:
        """
        Read file content from a pull request.

        This replaces the placeholder implementation with full GitHub API integration.
        """
        try:
            repo_name = self._get_repo_name_from_url(repo_url)

            async with await get_github_service(repo_name) as github_service:
                content_data = await github_service.read_pull_request_file(
                    repo_name=repo_name,
                    pr_number=pr_number,
                    filepath=filepath,
                    ref=ref,
                )

                # Ensure content is a string
                content = str(content_data) if content_data is not None else ""

                logger.info("Successfully read file %s from PR #%d", filepath, pr_number)
                return content

        except FileNotFoundError:
            logger.warning("File %s not found in PR #%d", filepath, pr_number)
            raise
        except (OSError, ValueError, KeyError) as e:
            logger.error("Error reading PR file: %s", e)
            raise

    async def analyze_pull_request(self, repo_url: str, pr_number: int) -> PRAnalysisDict:
        """Analyze a pull request and its changes."""
        repo_name = self._get_repo_name_from_url(repo_url)

        async with await get_github_service(repo_name) as github_service:
            # Get PR details
            pr = await github_service.get_pull_request(repo_name, pr_number)

            # Get changed files
            files = await github_service.list_pull_request_files(repo_name, pr_number)

            # Categorize files
            file_stats = {
                "total": len(files),
                "added": sum(1 for f in files if f.change_type == "added"),
                "modified": sum(1 for f in files if f.change_type == "modified"),
                "removed": sum(1 for f in files if f.change_type == "removed"),
                "renamed": sum(1 for f in files if f.change_type == "renamed"),
            }

            # Language breakdown
            language_changes = {}
            for file in files:
                ext = Path(file.file_path).suffix
                language = self._get_language_from_extension(ext)
                if language not in language_changes:
                    language_changes[language] = {
                        "files": 0,
                        "additions": 0,
                        "deletions": 0,
                    }
                language_changes[language]["files"] += 1
                language_changes[language]["additions"] += file.additions
                language_changes[language]["deletions"] += file.deletions

            return {
                "pull_request": pr,
                "files": files,
                "statistics": file_stats,
                "languages": language_changes,
                "total_additions": sum(f.additions for f in files),
                "total_deletions": sum(f.deletions for f in files),
            }

    def _get_language_from_extension(self, ext: str) -> str:
        """Map file extension to language."""
        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".r": "R",
            ".m": "Objective-C",
            ".sh": "Shell",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".json": "JSON",
            ".xml": "XML",
            ".md": "Markdown",
            ".rst": "reStructuredText",
            ".tex": "LaTeX",
        }
        return language_map.get(ext.lower(), "Other")

    def _parse_repo_url(self, repo_url: str) -> dict[str, str]:
        """Parse GitHub repository URL"""
        from urllib.parse import urlparse

        try:
            # Handle both https and git URLs
            if repo_url.startswith("git@"):
                # git@github.com:owner/repo.git
                parts = repo_url.replace("git@github.com:", "").replace(".git", "").split("/")
            else:
                # https://github.com/owner/repo or https://github.com/owner/repo.git
                parsed = urlparse(repo_url)
                path = parsed.path.strip("/").replace(".git", "")
                parts = path.split("/")

            if len(parts) < 2:
                raise ValueError(f"Invalid repository URL format: {repo_url}")

            return {"owner": parts[0], "name": parts[1]}

        except (ValueError, IndexError) as e:
            logger.error("Failed to parse repository URL %s: %s", repo_url, e)
            raise

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """Extract owner/repo format from repository URL."""
        repo_info = self._parse_repo_url(repo_url)
        return f"{repo_info['owner']}/{repo_info['name']}"
