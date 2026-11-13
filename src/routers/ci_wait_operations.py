"""CI-checks wait: ``POST /ci-wait`` (202 + wait id) and ``GET /ci-wait/{wait_id}``.

Why it exists: the conveyor (n8n) opens a draft PR and must continue only once
CI on the PR head has finished. It parks the cell on a Wait node and registers
the node's resume URL here; this service POSTs the result to it when every
check on the head SHA has completed — no polling from n8n, no time budget
(George 2026-10-11). Evaluation rules: :mod:`services.ci_waits.evaluator`;
triggers and delivery: :mod:`services.ci_waits.manager`.

Same tenant/repository authorisation as the PR routes, reused: ``require_tenant``
and ``_github`` (App-first credential ladder + platform-repo tenant gate) resolve
the head SHA at registration, and the background evaluations run under the
registering tenant through the same ladder.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from coding_client.auth.tenant import require_tenant
from coding_client.models.responses import (
    conflict,
    internal_error,
    not_found,
    service_unavailable,
    validation_error,
)
from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from pydantic import BaseModel, ConfigDict, Field

from routers.github_issue_operations import _github
from schemas.github_api_models import IssueRef
from services.ci_waits.callback import callback_url_allowed
from services.ci_waits.evaluator import PERMS_CI_WAIT
from services.ci_waits.manager import (
    CIWaitManager,
    _json_list,
    get_ci_wait_manager,
    status_path,
)
from services.github.github_service import GithubException
from services.github_integration import GitHubAuthError
from services.tenant_binding import github_auth_http_exception

logger = logging.getLogger(__name__)
router = APIRouter()

#: Retry-After (seconds) for a replay that lands while the holder's callback is in flight.
REPLAY_RETRY_AFTER_S = 2


class CIWaitRequest(IssueRef):
    """Body of ``wait_for_pr_checks``."""

    pr_number: int = Field(..., ge=1, description="Pull request number")
    callback_url: str = Field(
        ...,
        max_length=2048,
        description=(
            "URL POSTed once with the wait's result when the PR head's checks have all "
            "completed (or the head moved, the PR closed unmerged, or no CI exists). Must "
            "sit under a configured CI_CALLBACK_URL_PREFIXES prefix (an n8n Wait-node "
            "resume URL), else 422. Also the idempotency key: while a wait of this tenant "
            "with the same callback_url is open or its callback is still pending, a "
            "repeated POST returns that wait (202, same wait_id)."
        ),
        json_schema_extra={"x-n8n-param-name": "ciWaitCallbackUrl"},
    )
    required_checks: list[str] = Field(
        default_factory=list,
        max_length=50,
        description=(
            "Check names (check-run name or status context) that must EXIST on the head "
            "before the wait can settle — guards a second workflow whose runs are not "
            "created yet. Every check present gates the result either way."
        ),
        json_schema_extra={"x-n8n-param-name": "ciWaitRequiredChecks"},
    )
    expected_checks: int = Field(
        default=1,
        ge=1,
        le=500,
        description="Minimum number of checks that must exist before the wait can settle.",
        json_schema_extra={"x-n8n-param-name": "ciWaitExpectedChecks"},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "owner": "nonagenticai",
                "repo": "ems",
                "pr_number": 12,
                "callback_url": "https://callback.example.com/webhook-waiting/4711",
            }
        }
    )


class CIWaitLinks(BaseModel):
    status: str = Field(..., description="Path of the wait's status route")


class CIWaitAccepted(BaseModel):
    """202 body: the wait exists; its result arrives at ``callback_url``."""

    wait_id: str
    repo: str
    pr_number: int
    head_sha: str = Field(..., description="The head SHA the wait is pinned to")
    status: str = Field(..., description="'waiting', or the existing wait's status on a replay")
    links: CIWaitLinks


class CIWaitCheck(BaseModel):
    name: str | None = None
    kind: str | None = Field(None, description="check_run | status | check_suite | missing")
    status: str | None = None
    conclusion: str | None = None
    url: str | None = None


class CIWaitStatus(BaseModel):
    """Wait document; the completion callback POSTs the same result fields."""

    wait_id: str
    repo: str
    pr_number: int
    head_sha: str
    status: str = Field(..., description="waiting | resolved")
    overall: str | None = Field(
        None,
        description=(
            "success | failure | skipped (all checks skipped/neutral) | superseded (head "
            "moved: see new_head_sha) | pr_closed (closed unmerged before CI settled) | "
            "no_checks (nothing reported and nothing will: no active workflow, only "
            "empty completed suites, or no suite created within the grace window) | "
            "error (can never be evaluated). failure also covers a workflow that could "
            "not start (check_suite item) and required/expected checks that never "
            "appeared once CI finished (missing items)"
        ),
    )
    new_head_sha: str | None = None
    checks: list[CIWaitCheck] = Field(
        default_factory=list, description="Latest evaluation's checks on head_sha"
    )
    required_checks: list[str] = Field(default_factory=list)
    expected_checks: int = 1
    error: str | None = None
    eval_count: int = 0
    last_evaluated_at: datetime | None = None
    last_eval_error: str | None = Field(
        None, description="Why the last evaluation failed (it is retried with backoff)"
    )
    callback_state: str = Field(
        ...,
        description=(
            "none | sending | retrying (not accepted yet, resent with capped backoff until "
            "it is) | delivered | refused | disallowed"
        ),
    )
    created_at: datetime
    completed_at: datetime | None = None
    links: CIWaitLinks


def _manager_or_503() -> CIWaitManager:
    manager = get_ci_wait_manager()
    if manager is None:
        raise service_unavailable("CI-checks waits are not available: no wait store is configured")
    return manager


def _status_from_row(row: dict[str, Any]) -> CIWaitStatus:
    return CIWaitStatus(
        wait_id=row["wait_id"],
        repo=f"{row['owner']}/{row['repo']}",
        pr_number=row["pr_number"],
        head_sha=row["head_sha"],
        status=row["status"],
        overall=row["overall"],
        new_head_sha=row["new_head_sha"],
        checks=[CIWaitCheck(**c) for c in _json_list(row["checks"])],
        required_checks=_json_list(row["required_checks"]),
        expected_checks=row["expected_checks"],
        error=row["error"],
        eval_count=row["eval_count"],
        last_evaluated_at=row["last_evaluated_at"],
        last_eval_error=row["last_eval_error"],
        callback_state=row["callback_state"],
        created_at=row["created_at"],
        completed_at=row["resolved_at"],
        links=CIWaitLinks(status=status_path(row["wait_id"])),
    )


async def _resolve_head_sha(request: CIWaitRequest, tenant_id: str) -> str:
    """The PR's current head SHA, under the PR routes' credential ladder + tenant gate."""
    try:
        async with _github(request.owner, request.repo, tenant_id, PERMS_CI_WAIT) as service:
            repo = await service.get_repository(request.full_name())
            pr = await service._execute_with_retry(repo.get_pull, request.pr_number)
            return str(pr.head.sha)
    except HTTPException:
        raise
    except GitHubAuthError as e:
        raise github_auth_http_exception(e) from e
    except GithubException as e:
        if getattr(e, "status", None) == 404:
            raise not_found("Pull request", f"{request.full_name()}#{request.pr_number}") from e
        if getattr(e, "status", None) in (401, 403):
            raise HTTPException(
                status_code=e.status, detail=f"GitHub refused reading {request.full_name()}"
            ) from e
        raise internal_error(f"Failed to read pull request: {e}") from e
    except ValueError as e:
        raise not_found("Pull request", str(request.pr_number)) from e


@router.post(
    "/ci-wait",
    response_model=CIWaitAccepted,
    operation_id="wait_for_pr_checks",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Wait for a PR's checks to finish (202 + completion callback)",
    description="""
    Pin the PR's current head SHA and return immediately with a wait id. When every
    check run and commit status on that SHA has completed, the result is POSTed once to
    `callback_url` (an n8n Wait-node resume URL): `overall` = `success` | `failure` |
    `skipped`, with each check's name, conclusion and url.

    It also resolves — never parks forever — when the head moves (`superseded` +
    `new_head_sha`), the PR is closed unmerged before CI settles (`pr_closed`), the SHA
    has no checks and none will come (`no_checks`: no active workflow, or no check
    suite created for the SHA within the stale window), a workflow fails to start
    (`failure`), required/expected checks never appear once CI finished (`failure`
    with `missing` items), or GitHub says
    the PR can no longer be read (`error`). No polling: evaluated at registration, on
    GitHub's check/status/workflow/PR deliveries for the repository, and at startup.
    `GET /ci-wait/{wait_id}` reads the same document.
    """,
    responses={
        202: {"description": "Wait accepted"},
        401: {"description": "Authentication required"},
        403: {"description": "The tenant may not read this repository"},
        404: {"description": "Pull request not found"},
        409: {"description": "callback_url is held by an open wait for another PR"},
        422: {"description": "callback_url not under an allowed prefix"},
        503: {
            "description": "Wait store unavailable, or (with Retry-After) the previous wait "
            "for this callback_url is delivering its callback right now"
        },
    },
)
async def wait_for_pr_checks(
    response: Response,
    request: CIWaitRequest,
    tenant_id: str = Depends(require_tenant),
) -> CIWaitAccepted:
    if not callback_url_allowed(request.callback_url):
        # Refused before anything is read or persisted: this pod POSTs the URL from
        # inside the cluster, so only configured prefixes are accepted.
        raise validation_error(
            "callback_url is not under an allowed callback prefix", field="callback_url"
        )
    manager = _manager_or_503()
    head_sha = await _resolve_head_sha(request, tenant_id)

    wait_id = str(uuid.uuid7())
    try:
        row, created = await manager.store.create_wait(
            wait_id=wait_id,
            tenant_id=tenant_id,
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            head_sha=head_sha,
            required_checks=list(dict.fromkeys(request.required_checks)),
            expected_checks=request.expected_checks,
            callback_url=request.callback_url,
        )
    except Exception as exc:
        logger.exception("CI wait: could not create wait for %s", request.full_name())
        raise service_unavailable("CI wait store unavailable") from exc

    if not created:
        same_pr = (
            row["repo_key"] == request.full_name().lower() and row["pr_number"] == request.pr_number
        )
        if not same_pr:
            raise conflict("CI wait", "callback_url is already used by an open wait for another PR")
        if row["status"] != "waiting" and row["callback_state"] == "sending":
            exc = service_unavailable(
                "callback_url's previous wait is delivering its callback; retry shortly"
            )
            exc.headers = {"Retry-After": str(REPLAY_RETRY_AFTER_S)}
            raise exc
        logger.info("CI wait replay: tenant=%s, existing wait=%s", tenant_id, row["wait_id"])
    else:
        logger.info(
            "CI wait accepted: tenant=%s, %s#%s head=%s, wait=%s",
            tenant_id,
            request.full_name(),
            request.pr_number,
            head_sha[:12],
            wait_id,
        )
        # CI may already be done: evaluate now, in the background. A callback that
        # lands before the cell parks gets n8n's 409 "running already" and is retried.
        manager.schedule_evaluation(row)

    link = status_path(row["wait_id"])
    response.headers["Location"] = link
    return CIWaitAccepted(
        wait_id=row["wait_id"],
        repo=f"{row['owner']}/{row['repo']}",
        pr_number=row["pr_number"],
        head_sha=row["head_sha"],
        status=row["status"],
        links=CIWaitLinks(status=link),
    )


@router.get(
    "/ci-wait/{wait_id}",
    response_model=CIWaitStatus,
    operation_id="get_ci_wait_status",
    summary="Get CI-checks wait status",
    description=(
        "Read a CI-checks wait. Tenant-scoped: a wait of another tenant is a 404, "
        "indistinguishable from none."
    ),
    responses={
        404: {"description": "No such wait for this tenant"},
        503: {"description": "Wait store unavailable"},
    },
)
async def get_ci_wait_status(
    wait_id: str = Path(..., description="Wait id returned by wait_for_pr_checks"),
    tenant_id: str = Depends(require_tenant),
) -> CIWaitStatus:
    manager = _manager_or_503()
    try:
        canonical = str(uuid.UUID(wait_id))
    except ValueError:
        raise not_found("CI wait", wait_id) from None
    try:
        row = await manager.store.get_wait(wait_id=canonical, tenant_id=tenant_id)
    except Exception as exc:
        logger.exception("CI wait: status read failed for %s", wait_id)
        raise service_unavailable("CI wait store unavailable") from exc
    if row is None:
        raise not_found("CI wait", wait_id)
    return _status_from_row(row)
