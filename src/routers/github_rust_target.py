"""POST /api/repositories/create-rust-target-repo — the Python→Rust TARGET repo.

George 2026-11-03: creating the writable Rust target of a rewrite is a coding API op
(``create_rust_target_repo``), replacing the manual gitops
``scripts/new-rust-target.sh``. All behaviour lives in :mod:`services.rust_target`;
this module only maps it onto HTTP. ``operation_id`` is load-bearing: the generated
``@nonagentic/n8n-nodes-coding`` node derives its operation name from it.
"""

from __future__ import annotations

import asyncio

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import error_response
from fastapi import APIRouter, Depends

from schemas.github_api_models import CreateRustTargetRepoRequest, CreateRustTargetRepoResponse
from services import rust_target
from services.andon import require_andon_released

router = APIRouter()


@router.post(
    "/create-rust-target-repo",
    operation_id="create_rust_target_repo",
    dependencies=[Depends(require_andon_released)],
    response_model=CreateRustTargetRepoResponse,
    status_code=201,
    summary="Create Rust Target Repo",
    description=(
        "Create the PRIVATE Rust target repo of a Python→Rust rewrite in conveyor-coding "
        "from nonagenticai/template-rust-service (renamed, self-contained CI, one initial "
        "commit on main). Every refusal carries a machine `error_code` in "
        "`error.details.error_code`; the per-status list is under Responses."
    ),
    responses={
        403: {
            "description": (
                "`platform_repo_forbidden`: the tenant holds no write grant for the target. "
                "`platform_repo_read_only`: the target is granted read-only. "
                "`source_not_granted`: a source is not in the tenant's allowlist (checked "
                "before any GitHub call). `github_app_permission_missing`: GitHub refused "
                'for lack of "Repository permissions → Administration: write" or '
                '"→ Workflows: write" on the public App (`details.missing_permission`).'
            )
        },
        409: {
            "description": (
                "`target_exists`: the target repository already exists (also after a "
                "create that succeeded but whose push failed — the repo is then EMPTY). "
                "`github_app_not_installed`: the public App is not installed on "
                "conveyor-coding as an Organization."
            )
        },
        422: {
            "description": (
                "`invalid_target`, `target_owner_not_allowed` (not conveyor-coding), "
                "`target_name_not_rs`, `target_name_invalid` (not lowercase kebab), "
                "`invalid_source` (not owner/repo, or `.`/`..`), `source_owner_not_allowed`, "
                "`source_is_target`, or `source_not_usable` (missing, not visible, not "
                "PRIVATE, or owned elsewhere — one uniform answer). Request-shape "
                "errors use the standard validation body."
            ),
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
            },
        },
        502: {
            "description": (
                "`github_error`, `github_app_mint_failed`, `template_unreadable` or "
                "`git_failed`: GitHub or git failed; nothing was created unless the message "
                "says the repo was CREATED but is EMPTY."
            )
        },
        503: {"description": "`github_public_app_not_configured`."},
        504: {
            "description": (
                "`deadline_exceeded`: the op's total budget (85 s) was spent; the message "
                "says whether an EMPTY repo was left behind."
            )
        },
    },
)
async def create_rust_target_repo(
    request: CreateRustTargetRepoRequest, tenant_id: str = Depends(require_tenant)
) -> CreateRustTargetRepoResponse:
    """Create, populate and return the target repository."""
    try:
        result = await asyncio.to_thread(
            rust_target.create_rust_target, request.target, request.sources, tenant_id
        )
    except rust_target.RustTargetError as e:
        raise error_response(
            status_code=e.status,
            message=str(e),
            error_code=e.code,
            details={"target": request.target, "error_code": e.code, **e.details},
        ) from e
    return CreateRustTargetRepoResponse(
        target=result.target,
        url=result.url,
        default_branch=result.default_branch,
        head_sha=result.head_sha,
        template_sha=result.template_sha,
    )
