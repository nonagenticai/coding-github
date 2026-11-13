"""
PR Validation Orchestrator

Coordinates fast validation and triggers GitHub Actions for comprehensive validation.
This implements the hybrid validation approach: fast webhook validation followed by
comprehensive GitHub Actions validation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from schemas.github_api_models import ValidatePRRequest
from services.github_actions_trigger import get_workflow_trigger
from services.pr_validation.integration.github_validation_service import (
    PRGitHubValidationService,
)

logger = logging.getLogger(__name__)


class PRValidationOrchestrator:
    """
    Orchestrates the two-phase PR validation process:
    1. Fast validation (webhook-driven, < 30 seconds)
    2. Comprehensive validation (GitHub Actions, minutes)
    """

    def __init__(self):
        self.fast_validator = PRGitHubValidationService()
        self.workflow_trigger = get_workflow_trigger()
        # In-memory storage for validation tracking (could be Redis in production)
        self.validations: dict[str, dict[str, Any]] = {}

    async def validate_pr_with_actions(
        self,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        pr_files: list,
        pr_metadata: dict[str, Any] | None = None,
        trigger_comprehensive: bool = True,
    ) -> dict[str, Any]:
        """
        Perform two-phase validation of a PR.

        Phase 1: Fast validation (immediate)
        Phase 2: Comprehensive validation via GitHub Actions (if fast validation passes)

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            pr_number: Pull request number
            pr_files: List of PR files from GitHub API
            pr_metadata: Optional PR metadata
            trigger_comprehensive: Whether to trigger comprehensive validation

        Returns:
            Dict containing validation results and status
        """
        validation_id = f"pr-val-{uuid.uuid4().hex[:12]}"

        try:
            # Phase 1: Fast Validation
            logger.info(
                f"[{validation_id}] Starting fast validation for PR #{pr_number} in {repo_owner}/{repo_name}"
            )

            fast_request = ValidatePRRequest(
                repo_owner=repo_owner,
                repo_name=repo_name,
                pr_number=pr_number,
                validation_types=["security", "quality"],  # Quick checks only
            )

            fast_results = await self.fast_validator.validate_github_pr(
                fast_request, pr_files, pr_metadata
            )

            # Calculate fast validation metrics
            fast_passed = all(v.passed for v in fast_results)
            fast_score = (
                sum(v.score for v in fast_results) / len(fast_results) if fast_results else 0
            )

            # Store validation state
            self.validations[validation_id] = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "pr_number": pr_number,
                "fast_validation": {
                    "passed": fast_passed,
                    "score": fast_score,
                    "results": [
                        {
                            "validation_type": v.validation_type,
                            "passed": v.passed,
                            "score": v.score,
                            "issue_count": len(v.issues),
                        }
                        for v in fast_results
                    ],
                },
                "comprehensive_validation": {"status": "not_triggered"},
            }

            logger.info(
                f"[{validation_id}] Fast validation completed: passed={fast_passed}, score={fast_score:.1f}"
            )

            # Phase 2: Trigger Comprehensive Validation (if fast validation passed)
            if fast_passed and trigger_comprehensive:
                try:
                    logger.info(
                        f"[{validation_id}] Fast validation passed, triggering comprehensive validation"
                    )

                    # Get PR SHA from metadata
                    pr_sha = None
                    if pr_metadata:
                        if "head" in pr_metadata and isinstance(pr_metadata["head"], dict):
                            pr_sha = pr_metadata["head"].get("sha")

                    if not pr_sha and pr_files:
                        # Try to extract SHA from file metadata
                        for file in pr_files:
                            if hasattr(file, "sha"):
                                pr_sha = str(file.sha)
                                break

                    if not pr_sha:
                        pr_sha = f"pr-{pr_number}-head"
                        logger.warning(
                            f"[{validation_id}] Could not determine PR SHA, using fallback: {pr_sha}"
                        )

                    # Trigger GitHub Actions workflow
                    trigger_result = (
                        await self.workflow_trigger.trigger_pr_comprehensive_validation(
                            repo_owner=repo_owner,
                            repo_name=repo_name,
                            pr_number=pr_number,
                            pr_sha=pr_sha,
                            validation_id=validation_id,
                        )
                    )

                    self.validations[validation_id]["comprehensive_validation"] = {
                        "status": "triggered",
                        "workflow_id": trigger_result.get("workflow_id"),
                    }

                    logger.info(
                        f"[{validation_id}] Successfully triggered comprehensive validation"
                    )

                    return {
                        "validation_id": validation_id,
                        "status": "fast_validation_passed",
                        "fast_validation": {
                            "passed": fast_passed,
                            "score": fast_score,
                            "results": fast_results,
                        },
                        "comprehensive_validation": {
                            "status": "triggered",
                            "message": "Comprehensive validation workflow has been triggered",
                        },
                    }

                except Exception as trigger_error:
                    logger.error(
                        f"[{validation_id}] Failed to trigger comprehensive validation: {trigger_error}"
                    )

                    self.validations[validation_id]["comprehensive_validation"] = {
                        "status": "trigger_failed",
                        "error": str(trigger_error),
                    }

                    # Still return success for fast validation
                    return {
                        "validation_id": validation_id,
                        "status": "fast_validation_passed",
                        "fast_validation": {
                            "passed": fast_passed,
                            "score": fast_score,
                            "results": fast_results,
                        },
                        "comprehensive_validation": {
                            "status": "trigger_failed",
                            "error": str(trigger_error),
                            "message": "Fast validation passed, but comprehensive validation could not be triggered",
                        },
                    }

            else:
                # Fast validation failed or comprehensive not requested
                reason = "fast_validation_failed" if not fast_passed else "not_requested"
                logger.info(f"[{validation_id}] Not triggering comprehensive validation: {reason}")

                return {
                    "validation_id": validation_id,
                    "status": reason,
                    "fast_validation": {
                        "passed": fast_passed,
                        "score": fast_score,
                        "results": fast_results,
                    },
                    "comprehensive_validation": {
                        "status": "not_triggered",
                        "reason": reason,
                    },
                }

        except Exception as e:
            logger.error(
                f"[{validation_id}] PR validation orchestration failed: {e}",
                exc_info=True,
            )
            return {
                "validation_id": validation_id,
                "status": "error",
                "error": str(e),
                "fast_validation": {"passed": False, "score": 0},
                "comprehensive_validation": {"status": "not_triggered"},
            }

    def get_validation_status(self, validation_id: str) -> dict[str, Any] | None:
        """
        Get the current status of a validation.

        Args:
            validation_id: Validation tracking ID

        Returns:
            Dict containing validation status or None if not found
        """
        return self.validations.get(validation_id)

    async def update_comprehensive_results(
        self, validation_id: str, results: dict[str, Any]
    ) -> bool:
        """
        Update validation with comprehensive results from GitHub Actions.

        This is called by the API endpoint that receives results from the Actions workflow.

        Args:
            validation_id: Validation tracking ID
            results: Comprehensive validation results

        Returns:
            True if update successful, False otherwise
        """
        if validation_id not in self.validations:
            logger.warning(f"Validation ID {validation_id} not found for results update")
            return False

        try:
            self.validations[validation_id]["comprehensive_validation"] = {
                "status": "completed",
                "results": results,
            }

            logger.info(f"[{validation_id}] Comprehensive validation results updated")
            return True

        except Exception as e:
            logger.error(f"[{validation_id}] Failed to update comprehensive results: {e}")
            return False


# Singleton instance
_orchestrator: PRValidationOrchestrator | None = None


def get_pr_validation_orchestrator() -> PRValidationOrchestrator:
    """
    Get or create the PR validation orchestrator instance.

    Returns:
        PRValidationOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PRValidationOrchestrator()
    return _orchestrator
