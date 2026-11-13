"""Push-webhook handling: refresh the code graph coding-analysis owns.

⚠️ THIS SERVICE DOES NOT OWN A CODE GRAPH — do not reintroduce one here.

This module used to maintain a **per-repo sqlite code graph**: it resolved a
``.db`` path via ``Settings.get_database_manager()``, cloned the repository into
a coding-github-local workspace, ran a ``FileAnalyzer`` over the changed files,
and INSERTed rows into a sqlite ``nodes`` table. Every part of that was
monolith-era state that now belongs to coding-analysis (Postgres):

* ``Settings.get_database_manager()`` exists on **neither** ``Settings`` nor
  ``BaseMicroserviceSettings`` — the very first call raised ``AttributeError``,
  so the push path could never complete. PR #64 fixed the *import* that reached
  this code and explicitly did not claim the path worked. It didn't.
* ``processing.analyzers.file_analyzer`` **does not exist in this service** at
  all (verified: there is no ``src/processing`` package here). Its
  ``ImportError`` was caught and downgraded to ``status="updated_basic"`` — a
  *success-shaped* answer for an analysis that never ran.
* Even had both worked, a second code graph in sqlite would be a divergent copy
  of the graph coding-analysis serves — exactly what the microservice split
  exists to prevent.

The graph refresh is therefore **delegated** to coding-analysis via the same
``/api/analysis/scan/scan`` kickoff provisioning already uses
(``services.github_provisioning._trigger_full_scan``), against the container
that actually holds the checkout. Nothing is written here.

⚠️ KNOWN GAP, deliberately surfaced rather than papered over: an inbound GitHub
push carries **no tenant identity**, and container lookup is tenant-scoped. A
push can therefore only be resolved within the tenant this service can reach.
When the repo cannot be resolved to a container the result says so
(``status="no_container"``) instead of reporting a successful update.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from services.github.github_service import get_github_service

from .repo_operations import RepositoryOperations

logger = logging.getLogger(__name__)


# TypedDict definitions for type safety
class UpdateStatsDict(TypedDict):
    """Statistics for repository update operations."""

    added_nodes: int
    modified_nodes: int
    deleted_nodes: int
    files_processed: int
    status: str
    error: str | None
    commits_behind: int | None


class RepositoryUpdateResponseDict(TypedDict):
    """Response data for repository update operations."""

    status: str
    repo_id: str
    branch: str
    commits_processed: int
    files_changed: int
    update_stats: UpdateStatsDict
    timestamp: str


class CommitFileDict(TypedDict):
    """File information from git commit."""

    filename: str
    status: str
    additions: int | None
    deletions: int | None
    previous_filename: str | None


def _empty_stats(status: str, error: str | None = None) -> UpdateStatsDict:
    """Stats for an update this service did not itself perform.

    The node/edge counts are **0 because nothing was counted here**, not because
    a graph was found to be empty. coding-analysis owns the graph and runs the
    scan asynchronously, so this service genuinely cannot know the counts — and
    inventing them is how the old sqlite path reported "success" for work that
    never happened.
    """
    return {
        "added_nodes": 0,
        "modified_nodes": 0,
        "deleted_nodes": 0,
        "files_processed": 0,
        "status": status,
        "error": error,
        "commits_behind": None,
    }


class GitHubWebhookHandler:
    """Handles GitHub webhook events by delegating graph work to its owner."""

    def __init__(self) -> None:
        """Initialize webhook handler."""
        self.repo_operations = RepositoryOperations()

    async def handle_repository_update(
        self,
        repo_name: str,
        branch: str,
        changed_files: list[str],
        commit_shas: list[str],
        tenant_id: str | None = None,
    ) -> RepositoryUpdateResponseDict:
        """Refresh the code graph for ``repo_name`` after a push.

        Resolves the repository to the container holding its checkout, then asks
        coding-analysis to re-scan it. This service writes no graph state.

        Args:
            repo_name: Repository name in format "owner/repo"
            branch: Branch that was updated
            changed_files: File paths the webhook reported as changed
            commit_shas: Commit SHAs involved in the update
            tenant_id: Tenant the push belongs to. Container lookup is
                tenant-scoped; see the module docstring on why a webhook may not
                carry one.

        Returns:
            A dict whose ``status`` distinguishes the outcomes the old code
            collapsed into ``"success"``:

            * ``scan_requested`` — coding-analysis accepted the re-scan kickoff.
              The scan itself then runs in the background there.
            * ``no_container``   — no container for this repo in the tenant this
              service can see. Nothing was refreshed. Not an error, and not a
              success: there is simply no graph to update.
            * ``error``          — the lookup or the kickoff failed.
        """
        repo_id = repo_name.replace("/", "-").lower()
        logger.info(
            "Push received for %s on %s: %d changed files across %d commits",
            repo_name,
            branch,
            len(changed_files),
            len(commit_shas),
        )

        base: RepositoryUpdateResponseDict = {
            "status": "error",
            "repo_id": repo_id,
            "branch": branch,
            "commits_processed": 0,
            "files_changed": 0,
            "update_stats": _empty_stats("error"),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            container_name = await self._resolve_container_name(repo_name, tenant_id)
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            logger.error("Could not resolve a container for %s: %s", repo_name, e, exc_info=True)
            base["update_stats"] = _empty_stats("error", str(e))
            return base

        if not container_name:
            # No container => no graph => nothing to refresh. Said plainly so it
            # can never read as "the graph was updated".
            logger.warning(
                "No container found for %s (tenant=%s) — no code graph to refresh. "
                "A push for a repo nobody has provisioned is a no-op, not a failure.",
                repo_name,
                tenant_id or "<none>",
            )
            base["status"] = "no_container"
            base["update_stats"] = _empty_stats("no_container")
            return base

        try:
            # The same kickoff provisioning uses. Fire-and-forget by design:
            # coding-analysis runs the scan in the background and owns the result.
            from services.github_provisioning import _trigger_full_scan

            await _trigger_full_scan(container_name, tenant_id)
        except (OSError, RuntimeError) as e:
            logger.error(
                "Code-graph re-scan kickoff failed for %s (container=%s): %s",
                repo_name,
                container_name,
                e,
                exc_info=True,
            )
            base["update_stats"] = _empty_stats("error", str(e))
            return base

        logger.info(
            "Requested code-graph re-scan for %s (container=%s) after push to %s",
            repo_name,
            container_name,
            branch,
        )
        return {
            "status": "scan_requested",
            "repo_id": repo_id,
            "branch": branch,
            "commits_processed": len(commit_shas),
            "files_changed": len(changed_files),
            "update_stats": _empty_stats("scan_requested"),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _resolve_container_name(self, repo_name: str, tenant_id: str | None) -> str | None:
        """Return the container holding ``repo_name``'s checkout, or ``None``.

        Reuses the containers adapter's ``find_existing_container`` — the only
        repo-url -> container matcher in this service. It already normalises the
        url and re-checks the container's tenant, which is why this does not
        re-implement the match: a second matcher would be a second place for the
        tenant check to drift.

        Note the lookup lives on ``GitHubProvisioningService.docker_service``
        (the ``_ContainersHTTPAdapter``), not on the service itself.
        """
        from services.github_provisioning import GitHubProvisioningService

        service = GitHubProvisioningService()
        container = await service.docker_service.find_existing_container(
            f"https://github.com/{repo_name}", tenant_id=tenant_id
        )
        if container is None:
            return None
        if isinstance(container, dict):
            name = container.get("container_name")
        else:
            name = getattr(container, "container_name", None)
        return str(name) if name else None

    async def _process_commit_changes(
        self, repo_name: str, commit_shas: list[str]
    ) -> dict[str, list[Path]]:
        """Categorise the files touched by up to 5 commits, via the GitHub API."""
        changes_dict: dict[str, list[Path]] = {
            "added": [],
            "modified": [],
            "deleted": [],
        }

        from services.github_integration import is_platform_owned

        if not is_platform_owned((repo_name or "").split("/", 1)[0]):
            # ⛔ A webhook carries no fr tenant, so a customer repo has no
            # credential here at all (and no PAT fallback exists).
            logger.info(
                "skipped: customer repo, no tenant context on webhook (%s); "
                "commit changes not fetched, no PAT fallback",
                repo_name,
            )
            return changes_dict

        async with await get_github_service(repo_name) as github_service:
            for commit_sha in commit_shas[:5]:  # Process up to 5 commits
                try:
                    commit_files = await github_service.get_commit_files(repo_name, commit_sha)

                    for file_info in commit_files:
                        # Type-safe file info processing
                        file_info_dict: CommitFileDict = {
                            "filename": file_info.get("filename", ""),
                            "status": file_info.get("status", ""),
                            "additions": file_info.get("additions"),
                            "deletions": file_info.get("deletions"),
                            "previous_filename": file_info.get("previous_filename"),
                        }

                        file_path = Path(file_info_dict["filename"])
                        status = file_info_dict["status"]

                        if status == "added":
                            changes_dict["added"].append(file_path)
                        elif status in ["modified", "changed"]:
                            changes_dict["modified"].append(file_path)
                        elif status == "removed":
                            changes_dict["deleted"].append(file_path)
                        elif status == "renamed":
                            # Handle renames as delete + add
                            old_path = Path(file_info_dict["previous_filename"] or "")
                            changes_dict["deleted"].append(old_path)
                            changes_dict["added"].append(file_path)

                except (KeyError, ValueError, OSError) as e:
                    logger.warning("Could not process commit %s: %s", commit_sha, e)
                    continue

        # Remove duplicates
        for change_type in changes_dict:
            changes_dict[change_type] = list(set(changes_dict[change_type]))

        logger.info(
            "Processed changes: %d added, %d modified, %d deleted",
            len(changes_dict["added"]),
            len(changes_dict["modified"]),
            len(changes_dict["deleted"]),
        )

        return changes_dict
