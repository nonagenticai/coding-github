"""GitHub integration module for repository operations and pull request management.

This module provides centralized GitHub operations including:
- Repository cloning and synchronization
- Pull request analysis
- File reading from pull requests
- Repository update handling from webhooks

This is now a facade that delegates to focused sub-modules for better organization.
"""

from __future__ import annotations

import logging

from coding_client.models.github import (
    CloneResult,
    PullRequest,
    RepoWorkspace,
    SyncResult,
)
from coding_client.models.github import GitHubRepoInfo as RepoInfo
from coding_client.models.github import PRFileChange as FileChange

from services.github import (
    GitHubAuthManager,
    GitHubClient,
    GitHubWebhookHandler,
    RepositoryOperations,
)
from services.github.github_client import PRAnalysisDict
from services.github.webhook_handler import RepositoryUpdateResponseDict

logger = logging.getLogger(__name__)


class GitHubIntegrationHub:
    """Centralizes all GitHub operations using focused sub-modules."""

    def __init__(self) -> None:
        """Initialize GitHub integration hub with component modules."""
        self.auth_manager = GitHubAuthManager()
        self.github_client = GitHubClient()
        self.repo_operations = RepositoryOperations()
        self.webhook_handler = GitHubWebhookHandler()

    # Repository Operations
    async def clone_repository(self, repo_url: str) -> CloneResult:
        """Clone repository to local workspace, WITHOUT a credential.

        ⛔ There is no platform PAT to attach (owner ruling 2026-09-30) and this
        path carries no tenant for the GitHub App, so every repo clones
        anonymously (public repos only) — as it did while the PAT was empty.
        """
        return await self.repo_operations.clone_repository(repo_url, None)

    async def sync_repository(self, workspace: RepoWorkspace) -> SyncResult:
        """Fetch latest changes from GitHub"""
        return await self.repo_operations.sync_repository(workspace)

    # GitHub API Client Operations
    async def get_repository_info(self, repo_url: str) -> RepoInfo:
        """Get repository metadata and structure from GitHub API"""
        return await self.github_client.get_repository_info(repo_url)

    async def read_pr_file(
        self, repo_url: str, pr_number: int, filepath: str, ref: str = "head"
    ) -> str:
        """Read file content from a pull request."""
        return await self.github_client.read_pr_file(repo_url, pr_number, filepath, ref)

    async def analyze_pull_request(self, repo_url: str, pr_number: int) -> PRAnalysisDict:
        """Analyze a pull request and its changes."""
        return await self.github_client.analyze_pull_request(repo_url, pr_number)

    # Webhook Handling
    async def handle_repository_update(
        self,
        repo_name: str,
        branch: str,
        changed_files: list[str],
        commit_shas: list[str],
        tenant_id: str | None = None,
    ) -> RepositoryUpdateResponseDict:
        """Handle repository updates from webhook events.

        ``tenant_id`` is threaded through because the container lookup that
        resolves the repo's code graph is tenant-scoped. A GitHub webhook does
        not carry one; see ``services.github.webhook_handler``.
        """
        return await self.webhook_handler.handle_repository_update(
            repo_name, branch, changed_files, commit_shas, tenant_id=tenant_id
        )


# Export the main classes and functions for backward compatibility
__all__ = [
    "GitHubIntegrationHub",
    "CloneResult",
    "SyncResult",
    "RepoInfo",
    "FileChange",
    "PullRequest",
    "RepoWorkspace",
    "PRAnalysisDict",
    "RepositoryUpdateResponseDict",
]
