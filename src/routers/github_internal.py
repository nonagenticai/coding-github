"""Internal service-to-service GitHub endpoints (``/api/internal``).

Vends a **scoped-down, short-lived** GitHub credential to sibling coding-* pods.
coding-containers has no GitHub credential of its own and no egress to
api.github.com, so without this it can only clone a repo when the project
carries a user PAT — a project created from the fr UI without one can never get
a workspace container.

⚠️ THREE THINGS HERE ARE LOAD-BEARING — do not "tidy" any of them away.

1. **The ``/api/internal`` prefix.** coding-gateway's ``PROXY_ROUTES``
   (``coding-gateway/src/routers/proxy.py``) is an explicit prefix ALLOWLIST and
   404s anything unmatched. ``/api/repositories`` IS in that allowlist, so
   nesting a token-vending endpoint under it would publish it **to the public
   internet**. ``/api/internal`` is deliberately absent from the allowlist and
   must stay absent.

2. **The ``X-Internal-Token`` guard, not ``require_tenant()``.** ``require_tenant``
   is not authentication: it accepts a bare ``X-Tenant-ID`` header with no token
   at all, and it falls open for any caller inside ``TRUSTED_CIDRS`` — which
   includes ``10.0.0.0/8``, i.e. the pod (10.42/16) and service (10.43/16) CIDRs.
   Every in-cluster caller is therefore "trusted", so it cannot guard an endpoint
   that hands out credentials. A shared secret can.

3. **This route reads no tenant — but the tenant middleware still runs on it.**
   Not using ``require_tenant`` (point 2) does NOT put ``/api/internal`` outside
   the tenant machinery. ``TenantMiddleware``/``AuthMiddleware`` are installed
   **globally** in ``main.py`` with **no path exemption**, so every request here
   traverses them first. ``coding_client.auth._onbehalf.resolve_tenant_assertion``
   refuses on **disagreement**: a caller presenting its own licence AND an
   ``X-Tenant-Id`` naming a different tenant than that licence is rejected with
   **403 ``TENANT_ASSERTION_NOT_PERMITTED``** unless its subject is in this
   service's ``CODING_ONBEHALF_SUBJECTS`` — which lives in the **Deployment env**
   here, not a configmap as it does on the sibling services. Grep both forms; a
   search that only covers configmaps concludes this service has no allowlist.

   ⚠️ **Never read that list positionally.** Its entries are sorted **ascending by
   subject id**, not grouped or ordered by service, and the lists differ per
   service (this one names 3 subjects; coding-containers' names 7). Mapping
   "nth entry" to "nth service" gives a confidently wrong answer. Resolve a
   subject by introspecting the pod's own ``CODING_SERVICE_TOKEN``, or read it
   straight out of the refusal line — which names the subject verbatim.

   That 403 happens **before this module runs**, so no amount of reading
   ``mint_clone_token`` explains it, and the response body is the only place the
   real reason appears.

   ⚠️ **The correct call therefore OMITS ``X-Tenant-Id`` entirely.** The handler
   below takes only ``CloneTokenRequest`` and never reads a tenant, so sending one
   buys nothing and can only ever cost a 403. This is not hypothetical: it blocked
   every PAT-less container provision (coding-containers#101, 2026-08-16) — a
   caller added its own licence for good reasons, which silently promoted a header
   that had always been ignored into an *evaluated assertion*. Measured in-pod,
   same request, only the header differing: with ``X-Tenant-Id`` → 403; without →
   200 and a real App token.

4. **Customer repos are tenant-bound, and the tenant travels in the JSON BODY.**
   For an owner other than ``system_github_owner`` the token comes ONLY from the
   PUBLIC App, only for an installation fr has bound to ``CloneTokenRequest.tenant_id``
   (see ``services.tenant_binding``). The tenant is a BODY field, never the
   ``X-Tenant-Id`` header, for the reason in point 3. No tenant ⇒
   ``reason: tenant_required``; no binding ⇒ ``reason: not_installed``; fr
   unreachable ⇒ 503 (fail closed, never "unbound"). The body tenant is an
   assertion by an AUTHENTICATED internal caller (per-caller secret) — this
   route trusts that caller to forward the tenant it was itself given.

Nothing here ever logs a caller secret or the minted token. Each caller has
its own secret (see ``require_internal_token``); the log names the caller.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from config.settings import get_settings
from services.andon import enforce_andon
from services.github_app_auth import (
    _SCOPE_PERMISSIONS,
    GitHubAppAuthError,
    GitHubAppNotConfiguredError,
    GitHubAppNotInstalledError,
    RefHead,
    RepoPushMetadata,
    TokenScope,
    get_branch_head,
    get_repo_push_metadata,
    get_repo_visibility,
    get_scoped_clone_token,
)
from services.github_integration import is_platform_owned, is_sandbox_owned
from services.repo_visibility_guard import (
    RepoVisibilityRefusedError,
    assert_repo_not_public_lookup,
    visibility_refused_http,
)
from services.tenant_binding import (
    NotInstalledForTenantError,
    PlatformRepoForbiddenError,
    PlatformRepoReadOnlyError,
    PublicAppNotConfiguredError,
    TenantBindingUnavailableError,
    TenantRepoTokenError,
    TenantRequiredError,
    check_platform_repo_tenant,
    is_read_only_grant,
    mint_allowlisted_platform_token,
    platform_repo_grant,
    resolve_customer_repo_token,
    resolve_sandbox_repo_token,
    tenant_error_http_exception,
)

CloneTokenReason = Literal[
    "not_installed", "not_configured", "mint_failed", "tenant_required", "platform_repo_forbidden"
]

logger = logging.getLogger(__name__)
router = APIRouter()


class CloneTokenRequest(BaseModel):
    """Request for a clone credential for a single repository."""

    repo_url: str = Field(
        ...,
        min_length=1,
        description="Repository URL, e.g. https://github.com/owner/repo",
        examples=["https://github.com/nonagenticai/coding-github"],
    )
    scope: TokenScope = Field(
        "clone",
        description=(
            "Permission set to mint. 'clone' = contents:read (default); 'pr-read' = "
            "contents:read + pull_requests:read (needed for GET /pulls/{n}/files); "
            "'push' = contents:write on this one repo (branch push only). No other "
            "scope is grantable; pull_requests/workflows write are never issued here."
        ),
    )
    tenant_id: str | None = Field(
        None,
        description=(
            "Tenant on whose behalf the clone happens. REQUIRED for any owner outside "
            "the platform org (the token is then minted by the public GitHub App, only "
            "for an installation bound to this tenant). Sent in the BODY, never as "
            "X-Tenant-Id (see module docstring)."
        ),
    )
    branch: str | None = Field(
        None,
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
        description=(
            "scope='push' only (ignored otherwise): the branch the caller is about to "
            "push. When set and a push token is minted, the response carries "
            "``branch_head`` (and ``default_branch_head`` when the branch does not exist "
            "yet) read from GitHub with the same token, so the caller needs no GitHub "
            "egress of its own to find the push base."
        ),
    )


class RefHeadModel(BaseModel):
    """``refs/heads/<branch>`` on GitHub. ``state`` other than ``present`` carries no sha.

    ``unknown`` means the lookup failed and must be read as a refusal, never as
    ``absent``.
    """

    state: Literal["present", "absent", "unknown", "empty_repository"]
    sha: str | None = None

    @classmethod
    def of(cls, head: RefHead) -> RefHeadModel:
        return cls(state=head.state, sha=head.sha)


class CloneTokenResponse(BaseModel):
    """A clone credential, or an explicit "no credential" answer.

    ``token`` is ``None`` (with ``source="none"``) when the App is unconfigured or
    not installed on that owner. That is NOT an error: a ``public`` repo clones
    fine with no credential, so the caller decides what to do.

    ``reason`` says WHY there is no token, and it is the field that makes a
    tokenless answer auditable. ``not_installed`` is the routine, expected miss
    (the App simply does not cover that owner); ``mint_failed`` means the App
    should have been able to answer and could not — a defect, and one that
    ``source="none"`` alone reported identically to the routine miss. Callers
    that prefer the App token and fall back to a BYO PAT cannot otherwise tell a
    GitHub outage from an uninstalled owner in their own logs.

    ``reason`` is ``None`` exactly when ``token`` is present. It is an OPTIONAL
    field on the wire: a caller pinned to an older coding-github must keep
    working, so consumers treat a missing ``reason`` as "unknown", never as
    ``not_installed``.
    """

    token: str | None = Field(
        None, description="Short-lived single-repo token for the requested scope (ghs_…)"
    )
    expires_at: str | None = Field(None, description="ISO8601 expiry of the token")
    visibility: Literal["public", "private"] = Field(..., description="Repository visibility")
    source: Literal["app", "none"] = Field(..., description="Where the token came from")
    reason: CloneTokenReason | None = Field(
        None,
        description=(
            "Why no token was minted. Absent/None when a token WAS minted. "
            "'not_installed' is routine; 'mint_failed' is a defect; 'tenant_required' "
            "means a non-platform owner was requested without a tenant_id; "
            "'platform_repo_forbidden' means a platform-owned repo was requested for a "
            "tenant_id that is not a platform tenant."
        ),
    )
    # ── Push-scope repository facts (additive; coding-containers >= the PR that
    # consumes them refuses a push when they are absent, i.e. fails closed against
    # an older coding-github). Read from GitHub BY THIS SERVICE with the minted
    # token, never by the caller's workspace.
    private: bool | None = Field(
        None,
        description=(
            "scope='push' with a token: GitHub's ``private`` field, which the push guard "
            "has already required to be exactly ``true``. ``None`` for every other "
            "answer (not looked up). Never inferred from ``visibility``."
        ),
    )
    default_branch: str | None = Field(
        None,
        description=(
            "scope='push' with a token: the repository's default branch as GitHub "
            "reports it. ``None`` = not looked up or GitHub named none (unknown)."
        ),
    )
    branch_head: RefHeadModel | None = Field(
        None,
        description="scope='push' with a token and ``branch`` set: head of that branch.",
    )
    default_branch_head: RefHeadModel | None = Field(
        None,
        description=(
            "scope='push' with a token, ``branch`` set and that branch ``absent``: head "
            "of ``default_branch`` (the base a new branch is pushed against)."
        ),
    )


def require_internal_token(
    request: Request,
    x_internal_token: str | None = Header(None, alias="X-Internal-Token"),
) -> str:
    """Guard ``/api/internal`` with per-caller secrets compared in constant time.

    Accepted callers come from ``Settings.internal_callers``: the legacy
    ``CODING_GITHUB_INTERNAL_SERVICE_TOKEN`` (caller ``coding-containers``) plus
    every entry of ``CODING_GITHUB_INTERNAL_CALLER_TOKENS``. Each caller holds its
    own secret, so one service's token never authenticates another.

    The presented token is compared against EVERY entry with no early exit, so
    the response time does not reveal which entry (if any) matched.

    Fail-closed: no configured caller rejects everything rather than degrading
    to an open token-vending endpoint. Returns the caller name and stashes it on
    ``request.state.internal_caller``.
    """
    callers = get_settings().internal_callers
    if not callers:
        logger.error(
            "No internal caller is configured (CODING_GITHUB_INTERNAL_SERVICE_TOKEN / "
            "CODING_GITHUB_INTERNAL_CALLER_TOKENS) — rejecting internal request "
            "(endpoint disabled)."
        )
        raise HTTPException(status_code=403, detail="Forbidden")
    # compare_digest on bytes: a non-ASCII header would raise on str comparison.
    supplied = (x_internal_token or "").encode("utf-8")
    matched: str | None = None
    for caller, expected in callers.items():
        if hmac.compare_digest(supplied, expected.encode("utf-8")) and matched is None:
            matched = caller
    if matched is None:
        logger.warning("Rejected internal request with missing/invalid X-Internal-Token")
        raise HTTPException(status_code=403, detail="Forbidden")
    request.state.internal_caller = matched
    return matched


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Extract ``(owner, repo)`` from an HTTPS or SSH GitHub URL."""
    candidate = repo_url.strip()
    if candidate.startswith("git@"):
        path = candidate.split(":", 1)[-1]
    else:
        path = urlparse(candidate).path
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Invalid repository URL: {repo_url}")
    return parts[0], parts[1]


_MintResult = tuple[str | None, str | None, Literal["app", "none"], "CloneTokenReason | None"]


async def _platform_clone_token(owner: str, repo: str, scope: TokenScope) -> _MintResult:
    """The INTERNAL App's scoped clone token for a platform-owned repo (unchanged path)."""
    reason: CloneTokenReason | None = None
    try:
        scoped = await asyncio.to_thread(get_scoped_clone_token, owner, repo, scope=scope)
        return scoped.token, scoped.expires_at, "app", None
    except GitHubAppNotInstalledError as e:
        # ROUTINE. The App does not cover this owner. A public repo still clones
        # without a credential and a caller with a BYO PAT is unaffected, so this
        # is a 200 with source="none" at INFO — not a failure.
        reason = "not_installed"
        logger.info("No App installation covers %s/%s: %s", owner, repo, e)
    except GitHubAppNotConfiguredError as e:
        # This pod has no App identity at all. Every repo will miss, so say it
        # once, loudly — it is a deployment defect, not a per-repo answer.
        reason = "not_configured"
        logger.error(
            "GitHub App is not configured on this coding-github — NO repo can get an "
            "App clone token until it is: %s",
            e,
        )
    except GitHubAppAuthError as e:
        # The App should have been able to answer and did not (suspended install,
        # rate limit, 5xx, network, refused mint). Distinct from "not installed":
        # this one is a defect and must never be logged as a routine miss. A bare
        # GitHubAppAuthError lands here too — fail loud rather than silent.
        reason = "mint_failed"
        logger.error(
            "App clone-token mint FAILED for %s/%s (the App may well be installed; "
            "the attempt itself failed): %s",
            owner,
            repo,
            e,
        )
    return None, None, "none", reason


async def _customer_clone_token(
    owner: str, repo: str, scope: TokenScope, tenant_id: str | None
) -> _MintResult:
    """Tenant-bound PUBLIC-App token for a customer repo, or a tokenless answer.

    Never touches the internal App or any PAT. fr being unable to answer is a
    503 (fail closed) — it must never read as "not installed".
    """
    if not tenant_id:
        logger.info("Customer repo %s/%s requested without tenant_id: no token", owner, repo)
        return None, None, "none", "tenant_required"
    try:
        scoped = await asyncio.to_thread(
            resolve_customer_repo_token, owner, repo, tenant_id, dict(_SCOPE_PERMISSIONS[scope])
        )
        return scoped.token, scoped.expires_at, "app", None
    except TenantRequiredError:
        return None, None, "none", "tenant_required"
    except NotInstalledForTenantError as e:
        logger.info("No tenant-bound App installation for %s/%s: %s", owner, repo, e)
        return None, None, "none", "not_installed"
    except TenantBindingUnavailableError as e:
        logger.error("fr binding lookup failed for %s/%s — refusing: %s", owner, repo, e)
        raise HTTPException(
            status_code=503,
            detail={"error_code": e.code, "message": str(e), "owner": owner},
        ) from e
    except PublicAppNotConfiguredError as e:
        logger.error("Public GitHub App not configured — no customer clone token: %s", e)
        return None, None, "none", "not_configured"
    except TenantRepoTokenError as e:
        logger.error("Public-App clone-token mint FAILED for %s/%s: %s", owner, repo, e)
        return None, None, "none", "mint_failed"


async def _sandbox_clone_token(
    owner: str, repo: str, scope: TokenScope, tenant_id: str | None, caller: str | None
) -> _MintResult:
    """One-repo PUBLIC-App token for a SANDBOX-owned repo, gated like a platform repo.

    Platform tenant or an allowlist entry for exactly this repo → token. Anyone
    else (and no tenant) → ``platform_repo_forbidden`` under enforce. fr is never
    consulted. Under OBSERVE a would-refuse keeps today's answer (customer path).
    """
    try:
        scoped = await asyncio.to_thread(
            lambda: resolve_sandbox_repo_token(
                owner,
                repo,
                tenant_id,
                dict(_SCOPE_PERMISSIONS[scope]),
                site="clone_token",
                caller=caller,
            )
        )
    except PlatformRepoForbiddenError:
        return None, None, "none", "platform_repo_forbidden"
    except PlatformRepoReadOnlyError as e:
        # A write scope on a ``:read`` grant: a loud 403, never a tokenless 200.
        raise tenant_error_http_exception(e) from e
    except NotInstalledForTenantError as e:
        logger.info("Public App not installed on sandbox repo %s/%s: %s", owner, repo, e)
        return None, None, "none", "not_installed"
    except PublicAppNotConfiguredError as e:
        logger.error("Public GitHub App not configured — no sandbox clone token: %s", e)
        return None, None, "none", "not_configured"
    except TenantRepoTokenError as e:
        logger.error("Sandbox clone-token mint FAILED for %s/%s: %s", owner, repo, e)
        return None, None, "none", "mint_failed"
    if scoped is None:
        # OBSERVE mode, would-refuse: unchanged behaviour (the customer path).
        return await _customer_clone_token(owner, repo, scope, tenant_id)
    return scoped.token, scoped.expires_at, "app", None


async def _allowlisted_platform_clone_token(
    owner: str, repo: str, scope: TokenScope, tenant_id: str | None
) -> _MintResult:
    """One-repo INTERNAL-App token for an allowlisted (non-platform tenant, repo) grant.

    ⛔ A failure is an HTTP 502 ``platform_repo_scoped_mint_failed`` — NEVER a
    200 with ``token=None``: coding-containers reads a tokenless answer as
    "clone anonymously or use my own credential", and its own credential is the
    org PAT. Refusing loudly is the only answer that cannot become a PAT.
    """
    try:
        scoped = await asyncio.to_thread(
            mint_allowlisted_platform_token,
            owner,
            repo,
            dict(_SCOPE_PERMISSIONS[scope]),
            tenant_id=tenant_id,
        )
    except TenantRepoTokenError as e:
        logger.error("Allowlisted platform clone-token mint FAILED for %s/%s: %s", owner, repo, e)
        raise tenant_error_http_exception(e) from e
    return scoped.token, scoped.expires_at, "app", None


class _PrivateFlag:
    """Adapter so the raw ``private`` field reaches the ONE shared guard
    (``assert_repo_not_public`` reads ``.private``); no second copy of the verdict."""

    __slots__ = ("private",)

    def __init__(self, private: object) -> None:
        self.private = private


def _platform_clone_allowed(
    owner: str, repo: str, tenant_id: str | None, caller: str | None
) -> bool:
    """False ONLY when enforcing and ``tenant_id`` is not a platform tenant.

    STRICT: an absent ``tenant_id`` is NOT a platform tenant. In observe mode
    (enforce off) it is logged as ``would_refuse`` (``tenant=none``, with the
    authenticated internal ``caller``) and the platform mint proceeds as before.
    """
    try:
        check_platform_repo_tenant(owner, repo, tenant_id, site="clone_token", caller=caller)
    except PlatformRepoForbiddenError:
        return False
    return True


@router.post(
    "/github/clone-token",
    response_model=CloneTokenResponse,
    operation_id="mint_internal_clone_token",
    dependencies=[Depends(require_internal_token)],
    summary="Mint a scoped, short-lived clone token for a repository",
)
async def mint_clone_token(request: CloneTokenRequest, http_request: Request) -> CloneTokenResponse:
    """Return a single-repo token for ``scope`` (or ``None`` with a ``reason``).

    Both helpers are blocking (PyGithub + httpx), so they run off the event loop.

    Note: ``scope='push'`` is also granted for PLATFORM-owned repos (internal App,
    contents:write on that one repo) — accepted, same single-repo bound as customers.

    ⛔ PLATFORM-owned repo + a ``tenant_id`` that is absent or NOT in
    ``CODING_GITHUB_PLATFORM_TENANTS`` → with ``CODING_GITHUB_PLATFORM_TENANT_ENFORCE``
    true: ``token=None``, reason ``platform_repo_forbidden``, the internal App is
    not consulted. With it false (observe, the default): logged as
    ``platform_repo_tenant_gate decision=would_refuse`` and minted as before.

    ⛔ An absent ``tenant_id`` is treated as a non-platform tenant: refused under
    enforce, logged (``tenant=none``, plus the authenticated caller) in observe.
    The caller behind this route is tenant-facing (coding-containers'
    ``POST /api/containers/``), so a tenantless request would otherwise hand any
    licensed tenant a platform-repo token. fr already sends ``tenant_id`` and
    coding-containers #138 adds it; the observe window is what proves no caller
    still sends ``None`` before enforce is switched on.
    """
    owner, repo = parse_repo_url(request.repo_url)

    # ⛔ ANDON BRAKE — SCOPE-CONDITIONAL, and the condition is the whole point.
    #
    # ``scope='push'`` mints ``contents:write`` on the repo (``_SCOPE_PERMISSIONS``
    # in ``github_app_auth``). That credential leaves this service and is used by
    # coding-containers to push commits — a GitHub write that does NOT traverse
    # any other chokepoint in this repo, so if the brake did not cover it a worker
    # could push while the cord is pulled even though ``apply_code_changes`` is
    # gated.
    #
    # ⛔ The read scopes (``clone`` = contents:read, ``pr-read`` = contents:read +
    # pull_requests:read) are DELIBERATELY NOT gated. They mint a read-only
    # credential, and gating them would stop coding-containers cloning a repo at
    # all — i.e. the cord would block the one thing an operator needs while it is
    # pulled: provisioning a workspace to look at the code. A brake that blocks
    # observation makes the incident it fires on undiagnosable.
    #
    # This is the one gate expressed inline rather than as a route dependency,
    # because it turns on a BODY FIELD and a route-level dependency cannot see one.
    # It calls the same ``services.andon`` predicate — there is no second copy of
    # the verdict logic.
    if request.scope == "push":
        await enforce_andon("mint_clone_token")
        # ⛔ READ-ONLY GRANT (``owner/repo:read`` in the allowlist): scope=push is
        # refused HERE, explicitly, before any mint — and again at the mint
        # chokepoint (``tenant_binding.refuse_write_on_read_only_grant``), so the
        # refusal does not depend on ``_SCOPE_PERMISSIONS['push']`` staying a
        # write set. 403, never a tokenless 200 (coding-containers would read that
        # as "use my own credential").
        if is_read_only_grant(request.tenant_id, owner, repo):
            raise tenant_error_http_exception(
                PlatformRepoReadOnlyError(
                    f"'{owner}/{repo}' is granted READ-ONLY to this tenant; "
                    "scope='push' is refused.",
                    owner=owner,
                    repo=repo,
                )
            )

    token: str | None = None
    expires_at: str | None = None
    source: Literal["app", "none"] = "none"
    reason: CloneTokenReason | None = None
    if is_sandbox_owned(owner):
        # Platform-owned sandbox org: platform gate, PUBLIC App, no fr lookup.
        token, expires_at, source, reason = await _sandbox_clone_token(
            owner,
            repo,
            request.scope,
            request.tenant_id,
            getattr(http_request.state, "internal_caller", None),
        )
    elif not is_platform_owned(owner):
        token, expires_at, source, reason = await _customer_clone_token(
            owner, repo, request.scope, request.tenant_id
        )
    elif platform_repo_grant(request.tenant_id, owner, repo) == "scoped":
        # Allowlisted non-platform tenant: this ONE repo, internal App, no fallback.
        token, expires_at, source, reason = await _allowlisted_platform_clone_token(
            owner, repo, request.scope, request.tenant_id
        )
    elif not _platform_clone_allowed(
        owner, repo, request.tenant_id, getattr(http_request.state, "internal_caller", None)
    ):
        reason = "platform_repo_forbidden"
    else:
        token, expires_at, source, reason = await _platform_clone_token(owner, repo, request.scope)

    # ⛔ PUBLIC-REPO GUARD on the write scope. A ``push`` token is the ONE conveyor
    # write that bypasses every other visibility guard in this service (the container
    # pushes with it directly), so this is the chokepoint: a write-capable token is
    # handed out ONLY for a repo GitHub confirms ``private: true``. Public → 403
    # ``public_repo_refused``; absent/None/404/lookup error → 503
    # ``repo_visibility_unknown`` (fail closed). The token was minted to perform the
    # lookup (a private repo is invisible without it) and is WITHHELD on refusal.
    # Read scopes are not touched: cloning a public repo is not a write.
    #
    # The SAME GET also yields ``default_branch``, and with ``request.branch`` the
    # branch heads: coding-containers has no GitHub egress, so these are the facts
    # its push route refuses on (public repo, default-branch push, unknown base).
    private: bool | None = None
    default_branch: str | None = None
    branch_head: RefHeadModel | None = None
    default_branch_head: RefHeadModel | None = None
    if request.scope == "push" and token is not None:
        push_token = token
        fetched: list[RepoPushMetadata] = []

        def _lookup() -> _PrivateFlag:
            metadata = get_repo_push_metadata(owner, repo, push_token)
            fetched.append(metadata)
            return _PrivateFlag(metadata.private)

        try:
            await asyncio.to_thread(assert_repo_not_public_lookup, _lookup, f"{owner}/{repo}")
        except RepoVisibilityRefusedError as e:
            logger.warning(
                "Push token REFUSED for %s/%s (%s); token withheld", owner, repo, e.error_code
            )
            raise visibility_refused_http(e) from e
        visibility = "private"
        private = True
        default_branch = fetched[0].default_branch if fetched else None
        if request.branch:
            head = await asyncio.to_thread(get_branch_head, owner, repo, request.branch, push_token)
            branch_head = RefHeadModel.of(head)
            if head.state == "absent" and default_branch:
                base = await asyncio.to_thread(
                    get_branch_head, owner, repo, default_branch, push_token
                )
                default_branch_head = RefHeadModel.of(base)
    else:
        try:
            visibility = await asyncio.to_thread(get_repo_visibility, owner, repo, token)
        except GitHubAppAuthError as e:
            logger.warning("Could not resolve visibility for %s/%s: %s", owner, repo, e)
            raise HTTPException(
                status_code=502,
                detail=f"Could not resolve repository visibility for {owner}/{repo}",
            ) from e

    logger.info(
        "Clone token request served (caller=%s, repo=%s/%s, scope=%s, visibility=%s, "
        "source=%s, reason=%s)",
        getattr(http_request.state, "internal_caller", "-"),
        owner,
        repo,
        request.scope,
        visibility,
        source,
        reason or "-",
    )
    return CloneTokenResponse(
        token=token,
        expires_at=expires_at,
        visibility=visibility,  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
        source=source,
        reason=reason,
        private=private,
        default_branch=default_branch,
        branch_head=branch_head,
        default_branch_head=default_branch_head,
    )
