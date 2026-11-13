"""
PR Actions Endpoint for automated pull request management.

This endpoint provides automation capabilities for pull requests
based on validation results.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth.tenant import require_tenant
from coding_client.exceptions import (
    AuthenticationException,
)
from coding_client.models.responses import (  # TODO: verify coding-client import
    bad_request,
    error_response,
    internal_error,
)
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from routers.github_pull_request_operations import validate_pull_request
from schemas.github_api_models import (
    PRActionsProcessedResponse,
    ResponseStatus,
    ValidatePRRequest,
    ValidatePRResponse,
    ValidationWebhookResponse,
)
from services.andon import require_andon_released
from services.github_integration import GitHubAuthError, is_platform_owned, resolve_github_token
from services.pr_validation.actions.pr_auto_closer import PRActionOutcome, PRAutoCloser
from services.tenant_binding import (
    PERMS_PR_ACTIONS,
    PlatformRepoScopedMintFailedError,
    TenantRepoTokenError,
    mint_allowlisted_platform_token,
    platform_repo_grant,
    resolve_customer_repo_token,
    tenant_error_http_exception,
)
from utils.github_validation_helpers import (
    container_codebase_path,
    validate_and_get_container_for_tenant,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _raise_if_actions_failed(outcome: PRActionOutcome, actions_taken: dict[str, Any]) -> None:
    """Turn any failed PR action into a visible 502 instead of a success.

    The caller asked us to change something on GitHub. If any part of that did
    not happen we must not answer 200/"success" — the response would claim work
    that was never done. The detail carries the truthful per-action flags so the
    caller can still see what *did* land.
    """
    if outcome.succeeded:
        return
    logger.error(
        "PR actions did not fully complete: %s (actions_taken=%s)",
        outcome.errors,
        actions_taken,
    )
    raise error_response(
        status_code=status.HTTP_502_BAD_GATEWAY,
        message="One or more PR actions failed against GitHub",
        error_code="PR_ACTIONS_FAILED",
        details={"actions_taken": actions_taken, "errors": outcome.errors},
    )


class ProcessValidationRequest(BaseModel):
    """Request model for processing PR validation results."""

    repo_owner: str = Field(..., min_length=1, description="Repository owner")
    repo_name: str = Field(..., min_length=1, description="Repository name")
    pr_number: int = Field(..., ge=1, description="Pull request number")
    validation_response: dict[str, Any] = Field(..., description="Validation response data")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration options")
    container_name: str | None = Field(
        None, description="Optional container name to link PR to provisioned container"
    )

    @field_validator("repo_owner", "repo_name")
    @classmethod
    def validate_string_fields(cls, v: str) -> str:
        """Validate string fields contain only valid characters."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")

        # Check for invalid characters that might cause parsing issues
        invalid_chars = [
            "\x00",
            "\x01",
            "\x02",
            "\x03",
            "\x04",
            "\x05",
            "\x06",
            "\x07",
            "\x08",
            "\x09",
            "\x0a",
            "\x0b",
            "\x0c",
            "\x0d",
            "\x0e",
            "\x0f",
            "\x10",
        ]
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Invalid character found in string: {repr(char)}")

        return v.strip()


class ValidationWebhookRequest(BaseModel):
    """Request model for validation webhook."""

    repo_owner: str = Field(..., min_length=1, description="Repository owner")
    repo_name: str = Field(..., min_length=1, description="Repository name")
    pr_number: int = Field(..., ge=1, description="Pull request number")
    auto_close_enabled: bool = Field(
        default=False, description="Enable auto-close on critical issues"
    )
    github_token: str | None = Field(None, description="GitHub access token")
    container_name: str | None = Field(
        None, description="Optional container name to link PR to provisioned container"
    )

    @field_validator("repo_owner", "repo_name")
    @classmethod
    def validate_string_fields(cls, v: str) -> str:
        """Validate string fields contain only valid characters."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")

        # Check for invalid characters that might cause parsing issues
        invalid_chars = [
            "\x00",
            "\x01",
            "\x02",
            "\x03",
            "\x04",
            "\x05",
            "\x06",
            "\x07",
            "\x08",
            "\x09",
            "\x0a",
            "\x0b",
            "\x0c",
            "\x0d",
            "\x0e",
            "\x0f",
            "\x10",
        ]
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Invalid character found in string: {repr(char)}")

        return v.strip()


async def _platform_pr_actions_token(
    owner: str, repo: str, tenant_id: str | None, site: str
) -> str | None:
    """Credential for a PLATFORM repo's PR actions — the GitHub App, never a PAT.

    * Allowlisted non-platform tenant → an internal-App token for THIS ONE repo
      with ``PERMS_PR_ACTIONS``; a failed mint is an HTTP error.
    * Otherwise :func:`services.github_integration.resolve_github_token` — the
      platform-tenant gate (403 ``platform_repo_forbidden`` / tenantless refusal),
      then the internal App's installation token for this repo.
    * ⛔ No personal-access-token fallback exists (owner ruling 2026-09-30). If
      the App cannot mint, this returns ``None`` and the caller fails closed
      with ``GITHUB_TOKEN_REQUIRED``.
    """
    import asyncio

    if platform_repo_grant(tenant_id, owner, repo) == "scoped":
        try:
            scoped = await asyncio.to_thread(
                mint_allowlisted_platform_token,
                owner,
                repo,
                PERMS_PR_ACTIONS,
                tenant_id=tenant_id,
            )
        except TenantRepoTokenError as e:
            raise tenant_error_http_exception(e) from e
        if not scoped.token:
            raise tenant_error_http_exception(
                PlatformRepoScopedMintFailedError(
                    f"Empty repo-scoped token for allowlisted '{owner}/{repo}'.",
                    owner=owner,
                    repo=repo,
                )
            )
        return scoped.token

    def _resolve():
        return resolve_github_token(
            owner=owner,
            repo=repo,
            tenant_id=tenant_id,
            permissions=PERMS_PR_ACTIONS,
            site=site,
        )

    try:
        resolved = await asyncio.to_thread(_resolve)
    except TenantRepoTokenError as e:
        raise tenant_error_http_exception(e) from e
    except GitHubAuthError as e:
        logger.warning(
            "pr_actions_app_token_unavailable site=%s repo=%s/%s -- %s; refusing (no PAT fallback)",
            site,
            owner,
            repo,
            e,
        )
        return None
    return resolved.token


async def _customer_pr_actions_token(owner: str, repo: str, tenant_id: str | None) -> str:
    """Tenant-bound public-App token for a CUSTOMER repo's PR actions.

    Refusals surface as HTTPException (e.g. 409 ``github_app_not_installed``);
    there is no PAT fallback.
    """
    import asyncio

    try:
        scoped = await asyncio.to_thread(
            resolve_customer_repo_token, owner, repo, tenant_id, PERMS_PR_ACTIONS
        )
    except TenantRepoTokenError as e:
        raise tenant_error_http_exception(e) from e
    return scoped.token


@router.post(
    "/process-validation",
    operation_id="process_pr_validation",
    dependencies=[Depends(require_andon_released)],
    response_model=PRActionsProcessedResponse,
    summary="Process PR Validation Results",
    description="""
    Automatically process pull request validation results with configurable actions.

    **Available actions:**
    - Add validation status labels
    - Post detailed validation comments
    - Request changes for failed validations
    - Auto-close PRs with critical issues (if enabled)

    **Usage:**
    ```json
    {
        "repo_owner": "myorg",
        "repo_name": "myrepo",
        "pr_number": 123,
        "validation_response": {...},
        "config": {
            "auto_close_enabled": false,
            "add_labels": true,
            "post_comment": true
        }
    }
    ```

    **Credential:** the GitHub App (internal App for platform repos, tenant-bound public App otherwise)
    """,
    responses={
        200: {"description": "Every requested action was performed successfully"},
        400: {"description": "Invalid request"},
        401: {"description": "GitHub authentication failed"},
        500: {"description": "Processing error"},
        502: {"description": "One or more PR actions failed against GitHub"},
    },
)
async def process_pr_validation(
    request: ProcessValidationRequest,
    tenant_id: str = Depends(require_tenant),
) -> PRActionsProcessedResponse:
    """
    Process PR validation results and take automated actions.

    This endpoint processes validation results and can perform various automated
    actions based on the configuration, including adding labels, posting comments,
    requesting changes, and auto-closing PRs.

    Args:
        request: ProcessValidationRequest containing validation data and config

    Returns:
        Dict[str, Any]: Response containing status and actions taken

    Raises:
        HTTPException: If validation fails or processing errors occur
        AuthenticationException: If GitHub token is missing
    """
    try:
        # Validate container if provided (optional)
        if request.container_name:
            container_info = await validate_and_get_container_for_tenant(
                request.container_name, tenant_id
            )
            logger.info(
                "PR validation linked to container '%s' at path: %s",
                request.container_name,
                container_codebase_path(container_info),
            )

        # Extract parameters from Pydantic model
        repo_owner = request.repo_owner
        repo_name = request.repo_name
        pr_number = request.pr_number
        validation_response = request.validation_response
        config = request.config

        # Get GitHub token. Platform repo: the internal GitHub App (platform
        # tenants only). Customer repo: ONLY the tenant-bound public App.
        if is_platform_owned(str(repo_owner)):
            github_token = await _platform_pr_actions_token(
                str(repo_owner),
                str(repo_name),
                tenant_id,
                "pr_actions_process_validation",
            )
        else:
            github_token = await _customer_pr_actions_token(
                str(repo_owner), str(repo_name), tenant_id
            )
        if not github_token:
            raise AuthenticationException(
                message="GitHub token not configured",
                error_code="GITHUB_TOKEN_REQUIRED",
            )

        # Create auto-closer instance
        auto_closer = PRAutoCloser(github_token, config)

        # Convert validation response to proper type if needed
        validated_response: ValidatePRResponse
        if isinstance(validation_response, dict):
            # Validate that the dict has required fields
            required_fields = [
                "overall_score",
                "passed",
                "validations",
                "recommendation",
            ]
            missing_fields = [
                field for field in required_fields if field not in validation_response
            ]

            if missing_fields:
                raise bad_request(
                    f"Invalid validation_response: missing required fields: {missing_fields}"
                )

            try:
                validated_response = ValidatePRResponse(**validation_response)
            except Exception as e:
                raise bad_request(f"Failed to parse validation_response: {str(e)}") from e
        elif isinstance(validation_response, ValidatePRResponse):
            validated_response = validation_response
        else:
            raise bad_request("validation_response must be a dict or ValidatePRResponse")

        # Process the validation results. The outcome records what ACTUALLY
        # happened — never re-derive actions_taken from the config flags, which
        # only say what we were asked to attempt.
        outcome = await auto_closer.process_validation_result(
            str(repo_owner), str(repo_name), int(pr_number), validated_response
        )
        actions_taken = outcome.action_flags()
        _raise_if_actions_failed(outcome, actions_taken)

        return PRActionsProcessedResponse(
            status=ResponseStatus.SUCCESS,
            message="PR validation results processed successfully",
            actions_taken=actions_taken,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing PR validation: %s", e)
        raise internal_error(f"Failed to process PR validation: {str(e)}") from e


@router.post(
    "/validation-webhook",
    operation_id="trigger_validation_webhook",
    dependencies=[Depends(require_andon_released)],
    response_model=ValidationWebhookResponse,
    summary="Validation Webhook",
    description="""
    Webhook endpoint for automated PR validation and actions.

    This endpoint can be called by:
    - GitHub Actions
    - CI/CD systems
    - Custom automation tools

    It will:
    1. Validate the pull request
    2. Process results according to configuration
    3. Take automated actions (labels, comments, close)

    **Usage with curl:**
    ```bash
    curl -X POST http://your-api/api/webhooks/pr-actions/validation-webhook \\
      -H "Content-Type: application/json" \\
      -H "X-GitHub-Token: $TOKEN" \\
      -d '{
        "repo_owner": "myorg",
        "repo_name": "myrepo",
        "pr_number": 123,
        "auto_close_enabled": false
      }'
    ```
    """,
    responses={
        200: {"description": "Validation ran and every requested action succeeded"},
        400: {"description": "Invalid webhook payload"},
        401: {"description": "Authentication failed"},
        404: {"description": "Repository or pull request not found"},
        500: {"description": "Webhook processing error"},
        502: {"description": "One or more PR actions failed against GitHub"},
    },
)
async def validation_webhook(
    request: ValidationWebhookRequest,
    tenant_id: str = Depends(require_tenant),
) -> ValidationWebhookResponse:
    """
    Webhook endpoint that validates PR and processes results automatically.

    This endpoint performs validation on a pull request and then automatically
    processes the results according to the configuration. It's designed to be
    called by CI/CD systems or GitHub Actions.

    Args:
        request: ValidationWebhookRequest containing PR details and config

    Returns:
        Dict[str, Any]: Response containing validation results and actions taken

    Raises:
        HTTPException: If validation or processing fails
        AuthenticationException: If GitHub token is missing
    """
    try:
        # Validate container if provided (optional)
        if request.container_name:
            container_info = await validate_and_get_container_for_tenant(
                request.container_name, tenant_id
            )
            logger.info(
                "Webhook PR validation linked to container '%s' at path: %s",
                request.container_name,
                container_codebase_path(container_info),
            )

        # Extract GitHub token: the caller's explicit token wins (its own chosen
        # identity). Otherwise a platform repo uses the internal GitHub App and a
        # customer repo ONLY the tenant-bound public App.
        github_token = request.github_token
        if not (github_token and github_token.strip()):
            if is_platform_owned(request.repo_owner):
                github_token = await _platform_pr_actions_token(
                    request.repo_owner,
                    request.repo_name,
                    tenant_id,
                    "pr_actions_validation_webhook",
                )
            else:
                github_token = await _customer_pr_actions_token(
                    request.repo_owner, request.repo_name, tenant_id
                )

        if not github_token or not github_token.strip():
            raise AuthenticationException(
                message="GitHub token required",
                error_code="GITHUB_TOKEN_REQUIRED",
            )

        # Additional validation for meaningful repo names (not just "0" or other single chars)
        if len(request.repo_owner) < 2 or len(request.repo_name) < 2:
            raise bad_request("Repository owner and name must be at least 2 characters long")

        # First, create a validation request
        validation_request = ValidatePRRequest(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name,
            pr_number=request.pr_number,
            validation_types=["security", "performance", "quality"],
        )

        # Run REAL validation via the shared PR-validation logic so the webhook
        # reflects the actual PR state — never a fabricated pass. validate_pull_request
        # raises proper 404/403/500 on GitHub errors, propagating honest failures.
        logger.info(
            "Processing validation webhook for PR %s/%s#%d",
            validation_request.repo_owner,
            validation_request.repo_name,
            validation_request.pr_number,
        )
        validation_response = await validate_pull_request(validation_request, tenant_id)

        # Process the validation results
        config = {
            "auto_close_enabled": request.auto_close_enabled,
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": True,
            "request_changes": True,
        }

        auto_closer = PRAutoCloser(github_token, config)
        outcome = await auto_closer.process_validation_result(
            request.repo_owner,
            request.repo_name,
            request.pr_number,
            validation_response,
        )
        # validation_performed is the one flag we can honestly assert here:
        # validate_pull_request above either returned or raised.
        actions_taken: dict[str, Any] = {
            "validation_performed": True,
            "labels_updated": outcome.labels_added,
            "comment_posted": outcome.comment_posted,
            "changes_requested": outcome.changes_requested,
            "changes_request_degraded": outcome.changes_request_degraded,
            "pr_closed": outcome.pr_closed,
        }
        _raise_if_actions_failed(outcome, actions_taken)

        return ValidationWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="Webhook processed successfully",
            validation_score=validation_response.overall_score,
            recommendation=validation_response.recommendation,
            actions_taken=actions_taken,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Webhook processing error: %s", e)
        raise internal_error(f"Webhook processing failed: {str(e)}") from e
