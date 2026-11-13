"""
Read ONE file from a repository at a ref, without a container.

WHY THIS MODULE EXISTS
----------------------
The conveyor has to learn a repository's real test command (``pyproject.toml``,
``package.json``, ...) before it provisions anything. The only file read that
existed (``GET /read-file``, ``github_repository_ops``) reads from a provisioned
WORKSPACE container, so "which command runs this repo's tests" could only be
answered after paying for the thing the answer was needed to configure.

``POST /get-repository-file`` reads the blob straight from GitHub at a ref.

⛔ A MISSING FILE IS DATA, NOT AN ERROR. "Is there a ``package.json`` at this
ref?" is the question the caller is asking, so an absent path answers **200
``{exists: false}``**. Only a missing REPOSITORY or REF is a 404 — those mean the
question itself was wrong. Collapsing the two would make "this repo has no
pyproject" indistinguishable from "this repo does not exist".

⛔ EVERY ERROR THIS ROUTE RAISES CARRIES A TOP-LEVEL ``code``. (Request-schema
validation failures — bad owner/repo format, over-long ref, missing fields — are
rejected by FastAPI before the handler runs and carry only the shared envelope
``{"error": {"code": "HTTP_422", ...}}``; callers must treat a 422 without a
top-level ``code`` as ``invalid_request``.) The shared coding-client handler
renders an ``HTTPException`` as ``{"error": {"code": "HTTP_<status>", ...}}`` and
keeps the machine code only in ``error.details.error_code``. Consumers of this
route read ``code`` directly, so every failure here is returned as a
``JSONResponse`` that carries BOTH: the top-level ``code`` and the established
``error`` envelope (``error.details.error_code`` equal to it), so a caller written
against either shape reads the same answer.

TENANCY AND CREDENTIAL — ⛔ no new seam, no PAT. The handler opens
``routers.github_issue_operations._github`` exactly as ``get-repository`` does
(``routers/github_repo_read_operations.py``). That resolves through
``services.github_integration.resolve_github_token``: a platform-owned owner runs
``check_platform_repo_tenant`` (403 ``platform_repo_forbidden`` under
``CODING_GITHUB_PLATFORM_TENANT_ENFORCE``) and then mints from
``get_installation_token_for(owner, repo)``; a customer owner gets the
tenant-bound public-App token scoped to this one repo with ``contents: read``.
There is no PAT rung on either path, and a refusal reaches the caller as its own
403/409, never as a 500.

READ-ONLY: no writes, no andon gate (a read must not be coupled to fr's health —
the andon census in ``tests/unit/test_andon_chokepoint.py`` pins that read routes
carry no guard), and nothing is cached: every call reads GitHub with the caller's
own credential.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

from coding_client.auth.tenant import require_tenant
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from github import GithubException
from pydantic import BaseModel, Field, field_validator

from routers.github_issue_operations import _github
from routers.github_pr_status_operations import _access_denied, _github_status, _is_github_404
from services.github_integration import GitHubAuthError
from services.tenant_binding import PERMS_REPO_READ, github_auth_http_exception

logger = logging.getLogger(__name__)
router = APIRouter()

#: Files above this many bytes are refused (``413 file_too_large``). GitHub's
#: contents API stops inlining content at 1 MB anyway; a build manifest is far
#: below it, and the response travels through n8n's execution store.
MAX_FILE_BYTES = 1024 * 1024

_OWNER_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"
_REPO_PATTERN = r"^[A-Za-z0-9._-]{1,100}$"


class GetRepositoryFileRequest(BaseModel):
    """Which file, in which repository, at which ref."""

    owner: str = Field(..., pattern=_OWNER_PATTERN, description="Repository owner (user or org)")
    repo: str = Field(..., pattern=_REPO_PATTERN, description="Repository name")
    ref: str = Field(
        ...,
        max_length=255,
        description="Branch, tag or commit SHA. Resolved to a commit SHA (`ref_sha`).",
    )
    path: str = Field(
        ...,
        max_length=4096,
        description=(
            "Repository-relative file path, e.g. `pyproject.toml` or `api/package.json`. "
            "Absolute paths, `..` segments and NUL are refused (`422 invalid_path`)."
        ),
    )

    @field_validator("repo")
    @classmethod
    def _not_a_dot_segment(cls, v: str) -> str:
        # "." / ".." would turn ``owner/repo`` into a different URL path.
        if v in (".", ".."):
            raise ValueError("repo may not be '.' or '..'")
        return v

    def full_name(self) -> str:
        """PyGithub's get_repo takes 'owner/repo'."""
        return f"{self.owner}/{self.repo}"


class RepositoryFileResponse(BaseModel):
    """The file, or the fact that it is not there."""

    exists: bool = Field(..., description="False when the path is absent at `ref_sha`")
    ref_sha: str = Field(..., description="Commit SHA `ref` resolved to; the read is pinned to it")
    path: str
    content: str | None = Field(default=None, description="UTF-8 text; null when absent")
    sha: str | None = Field(default=None, description="Git blob SHA; null when absent")
    size: int | None = Field(default=None, description="Bytes; null when absent")


class RepositoryFileError(BaseModel):
    """Every failure: a top-level machine ``code`` plus the shared ``error`` envelope."""

    code: str
    message: str
    error: dict[str, Any]


class _Refusal(Exception):  # noqa: N818 - internal control flow, never escapes
    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


def _coded(status: int, code: str, message: str, details: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "error": {
                "code": f"HTTP_{status}",
                "message": message,
                "details": {**details, "error_code": code},
            },
        },
    )


def _from_http_exception(e: HTTPException) -> JSONResponse:
    """Re-shape an HTTPException (tenant refusal, auth, access-denied) with its code."""
    detail = e.detail if isinstance(e.detail, dict) else {}
    raw_details = detail.get("details")
    details: dict[str, Any] = dict(raw_details) if isinstance(raw_details, dict) else {}
    code = str(
        details.get("error_code") or detail.get("error_code") or f"http_{e.status_code}"
    ).lower()
    message = str(detail.get("message") or e.detail)
    return _coded(e.status_code, code, message, details)


def _validate_path(path: str) -> str:
    if not path or "\x00" in path:
        raise _Refusal(422, "invalid_path", "path must be a non-empty string without NUL")
    if path.startswith(("/", "\\")):
        raise _Refusal(422, "invalid_path", "path must be repository-relative, not absolute")
    segments = path.replace("\\", "/").split("/")
    if ".." in segments or "." in segments:
        raise _Refusal(422, "invalid_path", "path may not contain '.' or '..' segments")
    if any(s == "" for s in segments):
        raise _Refusal(422, "invalid_path", "path may not contain empty segments")
    return path


def _validate_ref(ref: str) -> str:
    if not ref or not ref.strip() or "\x00" in ref:
        raise _Refusal(422, "invalid_ref", "ref must be a non-empty string without NUL")
    return ref


def _resolve_ref_sha(repo: Any, ref: str) -> str:
    """Resolve ``ref`` to its commit SHA.

    ⛔ Reads ``raw_data`` — which forces the GET — not ``.sha``: a lazy PyGithub
    ``Commit`` built from a URL could otherwise answer with the ref string itself
    (``"main"``) and a caller would pin a "SHA" that is a branch name.
    """
    return str(repo.get_commit(ref).raw_data["sha"])


def _read_blob(repo: Any, path: str, ref_sha: str) -> dict[str, Any] | None:
    """The file at ``ref_sha``, or ``None`` when absent. Raises ``_Refusal`` otherwise."""
    try:
        found = repo.get_contents(path, ref=ref_sha)
    except GithubException as e:
        if _is_github_404(e):
            return None
        raise
    if isinstance(found, list):
        raise _Refusal(422, "not_a_file", f"'{path}' is a directory at {ref_sha}", path=path)
    if found.type != "file":
        raise _Refusal(
            422, "not_a_file", f"'{path}' is a {found.type}, not a file", path=path, type=found.type
        )
    size = int(found.size or 0)
    if size > MAX_FILE_BYTES:
        raise _Refusal(
            413,
            "file_too_large",
            f"'{path}' is {size} bytes; the limit is {MAX_FILE_BYTES}",
            path=path,
            size=size,
            limit=MAX_FILE_BYTES,
        )
    if found.encoding != "base64" or found.content is None:
        raise _Refusal(
            502,
            "github_unexpected_encoding",
            f"GitHub returned '{path}' with encoding {found.encoding!r}",
            path=path,
        )
    try:
        raw = base64.b64decode(found.content)
    except (binascii.Error, ValueError) as e:
        raise _Refusal(502, "github_unexpected_encoding", f"undecodable content: {e}") from e
    if b"\x00" in raw:
        raise _Refusal(422, "binary_file", f"'{path}' is a binary file", path=path, size=size)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _Refusal(422, "binary_file", f"'{path}' is not UTF-8 text", path=path) from e
    return {"content": text, "sha": found.sha, "size": size}


_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": RepositoryFileError, "description": desc}
    for status, desc in {
        401: "`github_unauthorized` — the credential was rejected",
        403: "`platform_repo_forbidden` / `github_forbidden` / `github_rate_limited`",
        404: "`repository_not_found` / `ref_not_found`",
        409: "`github_app_not_installed` — no tenant binding for this owner",
        412: "`github_auth_failed` — the GitHub App could not mint; no PAT fallback",
        413: "`file_too_large` — above 1 MiB",
        422: "`invalid_path` / `invalid_ref` / `not_a_file` / `binary_file`",
    }.items()
}


@router.post(
    "/get-repository-file",
    operation_id="get_repository_file",
    response_model=RepositoryFileResponse,
    responses=_ERROR_RESPONSES,
    summary="Get Repository File",
    description=(
        "Read ONE file from a repository at a ref (branch, tag or SHA), without a "
        "container. `ref` is resolved to `ref_sha` and the read is pinned to it. "
        "**A missing file is data, not an error**: `200 {exists: false, ref_sha}`. "
        "`404` means the repository (`repository_not_found`) or the ref "
        "(`ref_not_found`) does not exist for this credential. `422` for `invalid_path` "
        "(absolute, `..`, NUL), `not_a_file` (a directory or submodule) and "
        "`binary_file`; `413 file_too_large` above 1 MiB. Every error raised by the "
        "route carries a top-level `code`; a request-schema 422 carries only the shared "
        "`error` envelope (treat it as `invalid_request`)."
    ),
)
async def get_repository_file(
    request: GetRepositoryFileRequest, tenant_id: str = Depends(require_tenant)
) -> RepositoryFileResponse | JSONResponse:
    """Read one file at a ref."""
    full_name = request.full_name()
    stage = "repository"
    try:
        path = _validate_path(request.path)
        ref = _validate_ref(request.ref)
        async with _github(request.owner, request.repo, tenant_id, PERMS_REPO_READ) as service:
            repo = await service.get_repository(full_name)
            stage = "ref"
            ref_sha = await service._execute_with_retry(_resolve_ref_sha, repo, ref)
            stage = "path"
            blob = await service._execute_with_retry(_read_blob, repo, path, ref_sha)
        if blob is None:
            return RepositoryFileResponse(exists=False, ref_sha=ref_sha, path=path)
        return RepositoryFileResponse(exists=True, ref_sha=ref_sha, path=path, **blob)
    except _Refusal as r:
        return _coded(r.status, r.code, r.message, {"repository": full_name, **r.details})
    except HTTPException as e:
        # Includes the platform-repo tenant refusal (403 platform_repo_forbidden)
        # raised inside _github. It keeps its status and its code.
        return _from_http_exception(e)
    except GitHubAuthError as e:
        mapped = github_auth_http_exception(e)
        if isinstance(mapped.detail, dict):
            return _from_http_exception(mapped)
        return _coded(412, "github_auth_failed", str(e), {"repository": full_name})
    except Exception as e:
        denied = _access_denied(e, full_name)
        if denied is not None:
            return _from_http_exception(denied)
        status = _github_status(e)
        if stage == "repository" and (_is_github_404(e) or isinstance(e, ValueError)):
            return _coded(
                404,
                "repository_not_found",
                f"Repository '{full_name}' not found for this credential (a private repo "
                "it may not see also reads as absent)",
                {"repository": full_name},
            )
        if stage == "ref" and status in (404, 422):
            # GitHub answers an unknown SHA with 422 "No commit found", and an unknown
            # branch name with 404. Both mean the same thing to the caller.
            return _coded(
                404,
                "ref_not_found",
                f"Ref '{request.ref}' not found in '{full_name}'",
                {"repository": full_name, "ref": request.ref},
            )
        logger.exception("get-repository-file failed for %s at stage %s", full_name, stage)
        return _coded(
            500,
            "internal_error",
            f"Failed to read repository file: {type(e).__name__}",
            {"repository": full_name},
        )
