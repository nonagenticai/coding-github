"""Repository operations and cloning module.

This module provides repository-specific operations including:
- Repository cloning and workspace management
- Repository synchronization and updates
- Local repository operations
- Workspace cleanup and maintenance
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

from coding_client.models.github import (
    CloneResult,
    RepoWorkspace,
    SyncResult,
)

from config import get_settings

# ⛔ Shared redaction helper (the second clone path it was written for, the dead
# ``services.github_container_provisioner``, was deleted 2026-09-30). It lives
# in ``utils`` so any clone path can reach it: ``utils`` imports
# nothing from ``services``, so this leaf module does not have to depend on a
# top-level service module (which would invert the import direction and risk a
# cycle through ``services.github.__init__``). ⚠️ Read its module docstring —
# a TRUNCATED credential is a known, unclosed residual.
from utils.credential_redaction import redact_credentials

settings = get_settings()
logger = logging.getLogger(__name__)


class RepositoryOperations:
    """Handles repository cloning, syncing, and local operations."""

    def __init__(self) -> None:
        """Initialize repository operations."""
        # Use a secure default workspace path
        default_workspace = os.path.join(tempfile.gettempdir(), "code_graph_workspaces")
        self.workspace_base = getattr(settings, "WORKSPACE_BASE_PATH", default_workspace)

        # Ensure workspace base directory exists with secure permissions
        workspace_path = Path(self.workspace_base)
        workspace_path.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 700 (owner only)
        os.chmod(self.workspace_base, 0o700)

    async def clone_repository(self, repo_url: str, github_token: str | None = None) -> CloneResult:
        """Clone repository to local workspace.

        ⚠️ THIS is the clone path with a production caller
        (``services.github_integration_core.GitHubIntegrationHub
        .clone_repository``), which is why the redaction below matters more than
        anywhere else (the dormant ``github_container_provisioner`` clone path
        was deleted on 2026-09-30).
        """
        try:
            # Defence in depth: callers pass a PLAIN url here (the authenticated
            # one is the separate ``clone_url`` built below), but a future caller
            # handing in userinfo must not be able to log it.
            logger.info("Cloning repository: %s", redact_credentials(repo_url))

            # Parse repo info
            repo_info = self._parse_repo_url(repo_url)
            repo_id = f"{repo_info['owner']}-{repo_info['name']}"
            local_path = os.path.join(self.workspace_base, repo_id)

            # Ensure directory doesn't exist or clean it
            if os.path.exists(local_path):
                await self._remove_directory(local_path)

            # Clone with authentication if token available
            clone_url = self._build_authenticated_url(repo_url, github_token)

            # Execute git clone
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                clone_url,
                local_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await process.communicate()

            if process.returncode != 0:
                # ⛔ CREDENTIAL IN BOTH A LOG AND A RETURN VALUE. ``clone_url``
                # is ``https://<token>@github.com/o/r`` and git echoes the WHOLE
                # url back in its own stderr (``fatal: repository
                # 'https://<token>@github.com/o/r/' not found``). This used to
                # log that verbatim AND hand it back as ``CloneResult.error``.
                # ⚠️ The returned copy is the worse of the two: a log is at
                # least append-only and local, whereas ``.error`` is a
                # caller-visible field that gets persisted, echoed into a card,
                # or rendered in a UI. Redact BEFORE either use.
                error_msg = redact_credentials(
                    stderr.decode() if stderr else "Unknown git clone error", github_token
                )
                logger.error("Git clone failed: %s", error_msg)
                return CloneResult(
                    success=False,
                    repo_id=repo_id,
                    local_path=local_path,
                    error=error_msg,
                )

            logger.info("Successfully cloned %s to %s", redact_credentials(repo_url), local_path)
            return CloneResult(
                success=True,
                repo_id=repo_id,
                local_path=local_path,
                error=None,
            )

        except (OSError, ValueError, RuntimeError) as e:
            # Same two exits, same redaction: this handler also both logs and
            # RETURNS the message.
            error_msg = redact_credentials(str(e), github_token)
            logger.error("Clone repository failed: %s", error_msg)
            return CloneResult(success=False, repo_id="", local_path="", error=error_msg)

    async def sync_repository(self, workspace: RepoWorkspace) -> SyncResult:
        """Fetch latest changes from GitHub"""
        try:
            logger.info("Syncing repository at %s", workspace.local_path)

            # Change to repo directory and fetch
            process = await asyncio.create_subprocess_exec(
                "git",
                "fetch",
                "origin",
                cwd=workspace.local_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                # ⛔ THE CLONE CREDENTIAL OUTLIVES THE CLONE. ``git clone
                # https://<token>@github.com/o/r`` writes that url, credential
                # included, into ``.git/config`` as ``remote.origin.url``. So a
                # failing ``git fetch origin`` in this workspace echoes the token
                # back even though nothing here ever held it — which is why this
                # is redacted with NO literal secret to match on: the URL pass of
                # ``redact_credentials`` is the one that fires here.
                error_msg = redact_credentials(stderr.decode() if stderr else "Git fetch failed")
                logger.error("Git fetch failed: %s", error_msg)
                return SyncResult(
                    success=False,
                    error=error_msg,
                    commits_behind=0,
                    has_updates=False,
                )

            # Check for updates (compare with origin/main or origin/master)
            for branch in ["main", "master"]:
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "rev-list",
                    "--count",
                    f"HEAD..origin/{branch}",
                    cwd=workspace.local_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    commits_behind = int(stdout.decode().strip()) if stdout.decode().strip() else 0

                    # If there are updates, pull them
                    if commits_behind > 0:
                        logger.info("Found %d commits behind, pulling updates", commits_behind)
                        pull_process = await asyncio.create_subprocess_exec(
                            "git",
                            "pull",
                            "origin",
                            branch,
                            cwd=workspace.local_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        await pull_process.communicate()

                        if pull_process.returncode != 0:
                            logger.warning("Git pull failed, but fetch succeeded")

                    return SyncResult(
                        success=True,
                        commits_behind=commits_behind,
                        has_updates=commits_behind > 0,
                        error=None,
                    )

            # If we get here, no main/master branch found
            return SyncResult(success=True, commits_behind=0, has_updates=False, error=None)

        except (OSError, RuntimeError) as e:
            error_msg = redact_credentials(str(e))
            logger.error("Repository sync failed: %s", error_msg)
            return SyncResult(
                success=False,
                error=error_msg,
                commits_behind=0,
                has_updates=False,
            )

    async def checkout_branch(self, workspace_path: Path, branch: str) -> None:
        """Checkout the specified branch if needed."""
        if branch and branch not in {"main", "master"}:
            try:
                logger.info("Switching to branch: %s", branch)
                checkout_process = await asyncio.create_subprocess_exec(
                    "git",
                    "checkout",
                    branch,
                    cwd=str(workspace_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await checkout_process.communicate()
                if checkout_process.returncode != 0:
                    logger.warning(
                        "Failed to checkout branch %s: %s",
                        branch,
                        redact_credentials(stderr.decode() if stderr else "Unknown error"),
                    )
                else:
                    logger.info("Successfully switched to branch: %s", branch)
            except (OSError, RuntimeError) as e:
                # A git OSError/RuntimeError carries the argv, and this workspace's
                # ``.git/config`` holds the credential, so ``e`` is URL-shaped.
                logger.warning(
                    "Error checking out branch %s: %s", branch, redact_credentials(str(e))
                )

    async def create_and_checkout_branch(self, workspace: RepoWorkspace, branch_name: str) -> None:
        """Create and checkout a new branch locally."""
        process = await asyncio.create_subprocess_exec(
            "git",
            "checkout",
            "-b",
            branch_name,
            cwd=workspace.local_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()
        if process.returncode != 0:
            # Raised, not logged — and a raised message travels further. Same
            # ``.git/config``-persisted credential as in ``sync_repository``.
            raise RuntimeError(f"Failed to create branch: {redact_credentials(stderr.decode())}")

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

    def _build_authenticated_url(self, repo_url: str, github_token: str | None = None) -> str:
        """Build authenticated clone URL"""
        if not github_token:
            logger.info("No GitHub token available, using public clone")
            return repo_url

        if repo_url.startswith("git@"):
            # SSH URLs don't use token authentication
            return repo_url

        # Convert to authenticated HTTPS URL
        try:
            from urllib.parse import urlparse

            parsed = urlparse(repo_url)
            authenticated_url = f"https://{github_token}@{parsed.netloc}{parsed.path}"
            logger.info("Using authenticated clone URL")
            return authenticated_url
        except (ValueError, AttributeError) as e:
            logger.warning(
                "Failed to build authenticated URL: %s", redact_credentials(str(e), github_token)
            )
            return repo_url

    async def _remove_directory(self, directory_path: str) -> None:
        """Safely remove directory and all contents"""
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
                logger.info("Removed existing directory: %s", directory_path)
        except (OSError, PermissionError) as e:
            logger.warning("Failed to remove directory %s: %s", directory_path, e)
            # Continue anyway, git clone might handle it

    async def get_changed_files_from_git(self, workspace_path: Path) -> list[str]:
        """Get list of changed files from git diff."""
        try:
            git_status_process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--name-only",
                "HEAD~1",
                "HEAD",
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await git_status_process.communicate()

            if git_status_process.returncode == 0 and stdout:
                changed_files = stdout.decode().strip().split("\n")
                return [f for f in changed_files if f]

            return []

        except (OSError, RuntimeError) as e:
            logger.warning("Failed to get changed files from git: %s", e)
            return []

    def get_workspace_path(self, repo_id: str) -> Path:
        """Get workspace path for a repository."""
        return Path(self.workspace_base) / repo_id

    def workspace_exists(self, repo_id: str) -> bool:
        """Check if workspace exists for a repository."""
        workspace_path = self.get_workspace_path(repo_id)
        return workspace_path.exists() and workspace_path.is_dir()
