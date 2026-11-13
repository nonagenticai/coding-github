"""
GitHub Actions Workflow Trigger Service

This service handles triggering GitHub Actions workflows via workflow_dispatch events.
Used for triggering comprehensive validation after fast validation succeeds.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from github import Github, GithubException

logger = logging.getLogger(__name__)


class GitHubActionsWorkflowTrigger:
    """Service for triggering GitHub Actions workflows."""

    def __init__(self, github_token: str | None = None):
        """
        Initialize the workflow trigger service.

        Args:
            github_token: A caller-supplied token with workflow permissions.

        ⛔ There is NO platform-PAT fallback (owner ruling 2026-09-30). Without a
        caller-supplied token the client is ``None`` and every trigger refuses
        with ``ValueError`` — which is what production did while the old PAT
        setting was empty.
        """
        self.github_token = github_token or None
        self.github_client = Github(self.github_token) if self.github_token else None

    async def trigger_pr_comprehensive_validation(
        self,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        pr_sha: str,
        validation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Trigger comprehensive PR validation workflow.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            pr_number: Pull request number
            pr_sha: Pull request HEAD SHA
            validation_id: Optional validation tracking ID

        Returns:
            Dict containing trigger status and validation ID

        Raises:
            Exception: If workflow trigger fails
        """
        if not self.github_client:
            raise ValueError("GitHub client not initialized - check authentication")

        # Generate validation ID if not provided
        if not validation_id:
            validation_id = f"pr-val-{uuid.uuid4().hex[:12]}"

        try:
            # Get the repository
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")

            # Trigger workflow_dispatch event
            workflow_id = "pr-comprehensive-validation.yml"

            # Prepare workflow inputs
            workflow_inputs = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "pr_number": str(pr_number),
                "pr_sha": pr_sha,
                "validation_id": validation_id,
            }

            logger.info(
                f"Triggering workflow {workflow_id} for PR #{pr_number} in {repo_owner}/{repo_name}"
            )

            # Get the workflow
            workflow = repo.get_workflow(workflow_id)

            # Create workflow dispatch event
            success = workflow.create_dispatch(
                ref="main",  # Or the default branch
                inputs=workflow_inputs,
            )

            if success:
                logger.info(
                    f"Successfully triggered workflow for PR #{pr_number}, validation_id: {validation_id}"
                )
                return {
                    "status": "triggered",
                    "validation_id": validation_id,
                    "workflow_id": workflow_id,
                    "repo": f"{repo_owner}/{repo_name}",
                    "pr_number": pr_number,
                }
            else:
                raise Exception("Failed to trigger workflow dispatch")

        except GithubException as e:
            logger.error(f"GitHub API error triggering workflow: {e.status} - {e.data}")
            raise Exception(
                f"Failed to trigger workflow: {e.status} - {e.data.get('message', str(e))}"
            ) from e
        except Exception as e:
            logger.error(f"Error triggering workflow: {e}")
            raise

    async def trigger_workflow_via_api(
        self,
        repo_owner: str,
        repo_name: str,
        workflow_id: str,
        ref: str,
        inputs: dict[str, str],
    ) -> dict[str, Any]:
        """
        Trigger workflow using direct GitHub REST API call.

        This is an alternative method that uses the REST API directly
        instead of PyGithub, which can be more reliable for workflow_dispatch.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            workflow_id: Workflow filename or ID
            ref: Git reference (branch, tag, or SHA)
            inputs: Workflow input parameters

        Returns:
            Dict containing trigger status

        Raises:
            Exception: If workflow trigger fails
        """
        if not self.github_token:
            raise ValueError("GitHub token not available")

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        payload = {"ref": ref, "inputs": inputs}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 204:
                    logger.info(
                        f"Successfully triggered workflow {workflow_id} for {repo_owner}/{repo_name}"
                    )
                    return {
                        "status": "triggered",
                        "workflow_id": workflow_id,
                        "repo": f"{repo_owner}/{repo_name}",
                        "ref": ref,
                    }
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("message", "Unknown error")
                    logger.error(
                        f"Failed to trigger workflow: {response.status_code} - {error_msg}"
                    )
                    raise Exception(
                        f"Workflow trigger failed: {response.status_code} - {error_msg}"
                    )

        except httpx.TimeoutException as e:
            logger.error(f"Timeout triggering workflow: {e}")
            raise Exception("Workflow trigger timed out") from e
        except Exception as e:
            logger.error(f"Error triggering workflow via API: {e}")
            raise

    async def check_workflow_status(
        self, repo_owner: str, repo_name: str, run_id: int
    ) -> dict[str, Any]:
        """
        Check the status of a workflow run.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            run_id: Workflow run ID

        Returns:
            Dict containing workflow run status

        Raises:
            Exception: If status check fails
        """
        if not self.github_client:
            raise ValueError("GitHub client not initialized")

        try:
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            run = repo.get_workflow_run(run_id)

            return {
                "id": run.id,
                "status": run.status,
                "conclusion": run.conclusion,
                "created_at": run.created_at.isoformat(),
                "updated_at": run.updated_at.isoformat(),
                "html_url": run.html_url,
            }

        except GithubException as e:
            logger.error(f"GitHub API error checking workflow status: {e}")
            raise Exception(f"Failed to check workflow status: {e}") from e
        except Exception as e:
            logger.error(f"Error checking workflow status: {e}")
            raise


# Singleton instance
_workflow_trigger: GitHubActionsWorkflowTrigger | None = None


def get_workflow_trigger(
    github_token: str | None = None,
) -> GitHubActionsWorkflowTrigger:
    """
    Get or create the workflow trigger service instance.

    Args:
        github_token: Optional GitHub token to use

    Returns:
        GitHubActionsWorkflowTrigger instance
    """
    global _workflow_trigger
    if _workflow_trigger is None or github_token:
        _workflow_trigger = GitHubActionsWorkflowTrigger(github_token)
    return _workflow_trigger
