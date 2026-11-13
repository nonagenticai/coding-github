"""Tenant-bound GitHub credentials for CUSTOMER repositories.

⛔ THE INVARIANT this module enforces (plan: public-github-app-usable.md):

    A GitHub token for a repo outside ``system_github_owner`` is issued ONLY via
    the PUBLIC App, ONLY for an installation that fr has bound to the REQUESTING
    tenant, and ONLY if GitHub confirms that installation covers the repo.
    Anything else is refused. There is never a PAT fallback for a customer repo.

Three independent facts must agree before a token is minted:

1. fr's binding: ``GET {fr_url}/api/internal/github-app/binding?tenant_id=&owner=``
   names the installation this tenant installed on ``owner`` (or ``null``).
2. GitHub's answer: the public App's JWT ``GET /repos/{owner}/{repo}/installation``
   names the installation that actually covers the repo.
3. They are EQUAL. A mismatch means the repo is covered by an installation that
   belongs to someone else (or the tenant's installation does not include the
   repo) — refused, never "close enough".

Every failure FAILS CLOSED. In particular an fr error (timeout, 5xx, bad JSON,
non-integer id) raises :class:`TenantBindingUnavailableError`; it is never read as
"unbound" and never as "bound".

Platform-owned repos (``system_github_owner``) do not come through the public
App — they keep the internal App path in :mod:`services.github_integration`. But
they ARE tenant-gated: :func:`check_platform_repo_tenant` refuses (403
``platform_repo_forbidden``) any tenant not in ``CODING_GITHUB_PLATFORM_TENANTS``.
Without it every licensed tenant could drive the internal App against the
platform's own repositories.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Literal, NoReturn

import httpx
from coding_client.models.responses import error_response
from fastapi import HTTPException

from config import settings as _settings_module
from config.settings import get_settings
from services import github_app_auth as _github_app_auth
from services.github_app_auth import (
    GitHubAppAuthError,
    GitHubAppMintError,
    GitHubAppNotConfiguredError,
    GitHubAppNotInstalledError,
    GitHubAppOwnerMismatchError,
    ScopedToken,
    mint_public_repo_token,
    public_app_configured,
    resolve_public_installation,
    resolve_public_installation_id,
)
from services.github_integration import GitHubAuthError, is_platform_owned, is_sandbox_owned

logger = logging.getLogger(__name__)

BINDING_PATH = "/api/internal/github-app/binding"
BINDING_TTL_SECONDS = 60.0
_HTTP_TIMEOUT_SECONDS = 5.0

# Fixed permission sets callers choose from. The public App must have been
# GRANTED at least these at registration, or GitHub refuses the mint (422).
PERMS_CLONE: dict[str, str] = {"contents": "read"}
PERMS_PR_READ: dict[str, str] = {"contents": "read", "pull_requests": "read"}
PERMS_REPO_READ: dict[str, str] = {"contents": "read"}
PERMS_AUTHORING: dict[str, str] = {"contents": "write", "pull_requests": "write"}
PERMS_ISSUES: dict[str, str] = {"issues": "write"}
PERMS_PR_STATE: dict[str, str] = {"pull_requests": "read"}
PERMS_PR_CHECKS: dict[str, str] = {"pull_requests": "read", "checks": "read", "statuses": "read"}
# ⚠️ `actions: read` is a DISTINCT grant from `checks: read`. `GET /actions/runs`
# is refused by a checks-only token, so list-workflow-runs cannot reuse
# PERMS_PR_CHECKS. If the public App was never GRANTED Actions:read at
# registration the mint answers 422 and this surfaces as 502
# `github_app_mint_failed` — a loud refusal naming the rung, not a downgrade.
PERMS_ACTIONS_READ: dict[str, str] = {"actions": "read"}
PERMS_PR_ACTIONS: dict[str, str] = {"pull_requests": "write", "issues": "write"}
PERMS_MERGE: dict[str, str] = {
    "contents": "write",
    "pull_requests": "write",
    "checks": "read",
    "statuses": "read",
}


# ---------------------------------------------------------------------------
# Errors. All subclass GitHubAuthError, so every pre-existing
# ``except GitHubAuthError`` in the routers still refuses (412) even where the
# richer mapping below was not wired — the fallback is a refusal, not a token.
# ---------------------------------------------------------------------------
class TenantRepoTokenError(GitHubAuthError):
    """Base: no tenant-bound credential may be issued for this repo."""

    code = "github_app_error"
    http_status = 502

    def __init__(self, message: str, owner: str | None = None, repo: str | None = None):
        super().__init__(message)
        self.owner = owner
        self.repo = repo


class OwnerRequiredError(TenantRepoTokenError):
    """A credential was requested with no repository owner (no repo context)."""

    code = "github_owner_required"
    http_status = 400


class TenantRequiredError(TenantRepoTokenError):
    """A customer repo was requested without a tenant to bind it to."""

    code = "tenant_required"
    http_status = 403


class NotInstalledForTenantError(TenantRepoTokenError):
    """No installation bound to THIS tenant covers the repo (or GitHub disagrees)."""

    code = "github_app_not_installed"
    http_status = 409


class TenantBindingUnavailableError(TenantRepoTokenError):
    """fr could not answer the binding question. Fail closed."""

    code = "github_binding_unavailable"
    http_status = 503


class PublicAppNotConfiguredError(TenantRepoTokenError):
    """This coding-github has no public App identity."""

    code = "github_public_app_not_configured"
    http_status = 503


class PublicAppMintFailedError(TenantRepoTokenError):
    """The binding checked out but GitHub failed to answer / refused the mint."""

    code = "github_app_mint_failed"
    http_status = 502


class PlatformRepoForbiddenError(TenantRepoTokenError):
    """A platform-owned repo was requested by a tenant that is not a platform tenant."""

    code = "platform_repo_forbidden"
    http_status = 403


def is_platform_tenant(tenant_id: str | None) -> bool:
    """True only for a tenant listed in ``CODING_GITHUB_PLATFORM_TENANTS``.

    Absent, blank and non-UUID values are never platform tenants.
    """
    if not tenant_id or not str(tenant_id).strip():
        return False
    try:
        normalised = str(uuid.UUID(str(tenant_id).strip()))
    except ValueError:
        return False
    # Via the module, not this module's ``get_settings`` name: suites swap that
    # name for a binding-only double, and the allowlist must stay the real one.
    return normalised in _settings_module.get_settings().platform_tenant_ids


GATE_LOG_TOKEN = "platform_repo_tenant_gate"


def platform_tenant_enforced() -> bool:
    """``CODING_GITHUB_PLATFORM_TENANT_ENFORCE`` (False = observe only)."""
    return bool(getattr(_settings_module.get_settings(), "platform_tenant_enforce", False))


def check_platform_repo_tenant(
    owner: str, repo: str | None, tenant_id: str | None, *, site: str, caller: str | None = None
) -> bool:
    """⛔ Gate a platform-owned ``owner/repo`` to platform tenants only.

    STRICT: an absent tenant counts as non-platform. Callers that legitimately
    have no tenant (service-to-service) must not call this.

    Returns True when allowed. For a non-platform tenant it ALWAYS emits one
    WARNING carrying ``platform_repo_tenant_gate decision=<refused|would_refuse>``
    , ``caller_path=<site>`` and ``caller=<internal caller|->``; it then raises :class:`PlatformRepoForbiddenError`
    only when ``CODING_GITHUB_PLATFORM_TENANT_ENFORCE`` is true, and otherwise
    returns False so the caller proceeds with its previous behaviour (OBSERVE).

    Rollout: observe in prod, grep ``platform_repo_tenant_gate
    decision=would_refuse`` over ≥24h of conveyor traffic, then flip the env in
    gitops.
    """
    if is_platform_tenant(tenant_id):
        return True
    enforce = platform_tenant_enforced()
    logger.warning(
        "%s decision=%s tenant=%s owner=%s repo=%s caller_path=%s caller=%s",
        GATE_LOG_TOKEN,
        "refused" if enforce else "would_refuse",
        tenant_id if tenant_id else "none",
        owner,
        repo,
        site,
        caller or "-",
    )
    if enforce:
        raise PlatformRepoForbiddenError(
            f"'{owner}/{repo}' is a platform-owned repository; this tenant may not act on it.",
            owner=owner,
            repo=repo,
        )
    return False


def is_tenantless(tenant_id: str | None) -> bool:
    """True for an absent or blank tenant (``None``, ``""``, whitespace)."""
    return tenant_id is None or not str(tenant_id).strip()


def refuse_tenantless_platform_repo(owner: str | None, repo: str | None, *, site: str) -> NoReturn:
    """⛔ A platform-owned repo requested with NO tenant is refused, ALWAYS.

    Not gated on ``CODING_GITHUB_PLATFORM_TENANT_ENFORCE``: a request that names
    no tenant can never be proven to be a platform tenant, and every rung past
    this point hands out a platform (org-wide) credential. Measured before this
    was closed (2026-09-28, 7d Loki + code): zero live callers reach it — every
    tenant-facing route is ``require_tenant`` — so failing closed breaks nothing
    that is used and removes the silent platform identity.
    """
    logger.warning(
        "%s decision=refused tenant=none owner=%s repo=%s caller_path=%s caller=-",
        GATE_LOG_TOKEN,
        owner,
        repo,
        site,
    )
    raise PlatformRepoForbiddenError(
        f"'{owner}/{repo}' is a platform-owned repository; a request with no tenant "
        "may not act on it.",
        owner=owner,
        repo=repo,
    )


class PlatformRepoScopedMintFailedError(TenantRepoTokenError):
    """An allowlisted (tenant, platform repo) grant could not be minted. No fallback."""

    code = "platform_repo_scoped_mint_failed"
    http_status = 502


class PlatformRepoReadOnlyError(TenantRepoTokenError):
    """A WRITE was requested on a platform repo the tenant holds only a ``:read`` grant for."""

    code = "platform_repo_read_only"
    http_status = 403


PlatformRepoGrant = Literal["platform", "scoped", "refuse"]


def _normalise_tenant(tenant_id: str | None) -> str | None:
    if not tenant_id or not str(tenant_id).strip():
        return None
    try:
        return str(uuid.UUID(str(tenant_id).strip()))
    except ValueError:
        return None


def platform_repo_grant(
    tenant_id: str | None, owner: str | None, repo: str | None
) -> PlatformRepoGrant:
    """⛔ What ``tenant_id`` may hold for the platform-owned ``owner/repo``.

    * ``"platform"`` — a platform tenant (``CODING_GITHUB_PLATFORM_TENANTS``):
      the pre-existing internal-App path, unchanged.
    * ``"scoped"`` — a NON-platform tenant whose
      ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST`` entry names EXACTLY this
      ``owner/repo`` (case-insensitive). The caller MUST serve it only via
      :func:`mint_allowlisted_platform_token` (one repo, internal App) and must
      never reach the org-wide installation token or any PAT.
    * ``"refuse"`` — everything else, including a non-platform owner, an
      absent/blank/non-UUID tenant and an absent repo. Callers route this to the
      existing :func:`check_platform_repo_tenant` (403 under enforce).
    """
    if is_platform_tenant(tenant_id):
        return "platform"
    tenant = _normalise_tenant(tenant_id)
    if tenant is None or not owner or not repo or not is_platform_owned(owner):
        return "refuse"
    grants = getattr(_settings_module.get_settings(), "platform_repo_grants", None) or {}
    allowed = grants.get(tenant)
    if not allowed:
        return "refuse"
    if f"{owner.strip().lower()}/{repo.strip().lower()}" in allowed:
        return "scoped"
    return "refuse"


def is_read_only_grant(tenant_id: str | None, owner: str | None, repo: str | None) -> bool:
    """True when ``tenant_id``'s allowlist names ``owner/repo`` with the ``:read`` suffix.

    Only meaningful for a ``"scoped"`` grant (platform or sandbox): a platform
    tenant is never read-only, and a tenant with no grant is refused elsewhere.
    """
    if is_platform_tenant(tenant_id):
        return False
    tenant = _normalise_tenant(tenant_id)
    if tenant is None or not owner or not repo:
        return False
    read_only = (
        getattr(_settings_module.get_settings(), "platform_repo_read_only_grants", None) or {}
    )
    return f"{owner.strip().lower()}/{repo.strip().lower()}" in (read_only.get(tenant) or ())


def refuse_write_on_read_only_grant(
    owner: str, repo: str, tenant_id: str | None, permissions: dict[str, str], *, site: str
) -> None:
    """⛔ THE chokepoint: a ``:read`` grant never mints a ``write`` permission.

    Called by both scoped mints (:func:`mint_allowlisted_platform_token`,
    :func:`resolve_sandbox_repo_token`), so every route — PR create, push,
    merge, labels, comments, issue writes, PR-body edits, and any route added
    later — that asks for a write permission set on a read-only grant is refused
    403 ``platform_repo_read_only`` before GitHub is contacted.
    """
    if not is_read_only_grant(tenant_id, owner, repo):
        return
    writes = sorted(k for k, v in permissions.items() if v != "read")
    if not writes:
        return
    logger.warning(
        "%s decision=read_only_refused tenant=%s owner=%s repo=%s perms=%s caller_path=%s",
        GATE_LOG_TOKEN,
        tenant_id,
        owner,
        repo,
        ",".join(f"{k}:{v}" for k, v in sorted(permissions.items())),
        site,
    )
    raise PlatformRepoReadOnlyError(
        f"'{owner}/{repo}' is granted READ-ONLY to this tenant; refusing a token with "
        f"write permission(s): {', '.join(writes)}.",
        owner=owner,
        repo=repo,
    )


def mint_allowlisted_platform_token(
    owner: str, repo: str, permissions: dict[str, str], *, tenant_id: str | None
) -> ScopedToken:
    """INTERNAL-App token restricted to ``repositories=[repo]`` for an allowlisted grant.

    Re-checks :func:`platform_repo_grant` itself (``"scoped"`` or it raises
    :class:`PlatformRepoForbiddenError`), so it can never be called into minting
    for a repo the allowlist does not name. Goes through ``_mint_repo_scoped_token``
    ONLY — never ``_mint_installation_token`` / ``_token_by_id`` (the org-wide,
    PR-opening installation token). Every failure raises
    :class:`PlatformRepoScopedMintFailedError`; there is no fallback rung.

    ``owner``/``repo`` are stripped of surrounding whitespace BEFORE the grant
    check and the mint, so the name that was authorised is exactly the name
    minted for (``repositories=[repo]``) — never a padded variant GitHub would
    reject or resolve differently.
    """
    owner = (owner or "").strip()
    repo = (repo or "").strip()
    if platform_repo_grant(tenant_id, owner, repo) != "scoped":
        raise PlatformRepoForbiddenError(
            f"'{owner}/{repo}' is not on this tenant's platform repo allowlist.",
            owner=owner,
            repo=repo,
        )
    if not permissions:
        raise ValueError("An allowlisted platform repo token needs an explicit permission set")
    for name, level in permissions.items():
        if level not in ("read", "write") or not name.replace("_", "").isalpha():
            raise ValueError(f"Invalid permission {name}={level!r}")
    refuse_write_on_read_only_grant(
        owner, repo, tenant_id, permissions, site="mint_allowlisted_platform_token"
    )
    try:
        installation_id = _github_app_auth._resolve_installation_id(
            owner, repo, _github_app_auth.INTERNAL_APP
        )
        token = _github_app_auth._mint_repo_scoped_token(
            _github_app_auth.INTERNAL_APP,
            installation_id,
            owner,
            repo,
            dict(permissions),
            noun="allowlisted platform repo token",
        )
    except GitHubAppAuthError as e:
        raise PlatformRepoScopedMintFailedError(
            f"Could not mint a repo-scoped token for allowlisted '{owner}/{repo}': {e}",
            owner=owner,
            repo=repo,
        ) from e
    logger.info(
        "%s decision=scoped tenant=%s owner=%s repo=%s perms=%s",
        GATE_LOG_TOKEN,
        tenant_id,
        owner,
        repo,
        ",".join(f"{k}:{v}" for k, v in sorted(permissions.items())),
    )
    return token


# ---------------------------------------------------------------------------
# Sandbox orgs (CODING_GITHUB_SANDBOX_GITHUB_OWNERS): platform-owned, served by
# the PUBLIC App, gated like platform repos. No fr binding is ever consulted.
# ---------------------------------------------------------------------------
def sandbox_repo_grant(
    tenant_id: str | None, owner: str | None, repo: str | None
) -> PlatformRepoGrant:
    """``"platform"`` (platform tenant), ``"scoped"`` (allowlisted for exactly
    this ``owner/repo``, or by an ``owner/*suffix`` name pattern) or ``"refuse"``
    — for a SANDBOX-owned repo only.

    Kept separate from :func:`platform_repo_grant` on purpose: a ``"scoped"``
    verdict there is served by the INTERNAL App, which is not installed on a
    sandbox org.
    """
    if not owner or not repo or not is_sandbox_owned(owner):
        return "refuse"
    if is_platform_tenant(tenant_id):
        return "platform"
    tenant = _normalise_tenant(tenant_id)
    if tenant is None:
        return "refuse"
    grants = getattr(_settings_module.get_settings(), "platform_repo_grants", None) or {}
    # Exact entries AND sandbox name patterns (``conveyor-coding/*-rs``). Patterns
    # are accepted at boot only for a sandbox owner, so platform_repo_grant above
    # (internal App, exact match only) can never be widened by one.
    if _settings_module.grant_covers(grants.get(tenant) or frozenset(), owner, repo):
        return "scoped"
    return "refuse"


def _assert_sandbox_installation(
    installation: object, owner: str, repo: str, tenant_id: str | None, *, site: str
) -> None:
    """⛔ The public-App installation covering a sandbox repo must BE the sandbox org's.

    ``account.login`` must equal the configured sandbox ``owner`` (case-insensitive)
    and ``account.type`` must be ``Organization``. Anything else — a repo that was
    transferred / redirected to another account, a user-account installation, or
    GitHub omitting the account — is refused BEFORE a token is minted, with the same
    ``NotInstalledForTenantError`` (reason ``not_installed``) the customer path uses
    when GitHub's installation disagrees with fr's binding: "no installation we
    trust covers this repo". It is not ``platform_repo_forbidden``, which names the
    TENANT as the problem; here the tenant was allowed and the installation is wrong.
    """
    login = str(getattr(installation, "account_login", "") or "")
    acct_type = str(getattr(installation, "account_type", "") or "")
    if login.strip().lower() == owner.strip().lower() and acct_type == "Organization":
        return
    logger.error(
        "%s decision=sandbox_installation_mismatch tenant=%s owner=%s repo=%s installation=%s "
        "account_login=%s account_type=%s caller_path=%s — REFUSED: the installation "
        "covering this repo is not the sandbox org's",
        GATE_LOG_TOKEN,
        tenant_id if tenant_id else "none",
        owner,
        repo,
        getattr(installation, "id", "?"),
        login or "-",
        acct_type or "-",
        site,
    )
    raise NotInstalledForTenantError(
        f"The public GitHub App installation covering '{owner}/{repo}' belongs to "
        f"'{login or '?'}' ({acct_type or '?'}), not sandbox org '{owner}'; refused.",
        owner=owner,
        repo=repo,
    )


def resolve_sandbox_repo_token(
    owner: str,
    repo: str,
    tenant_id: str | None,
    permissions: dict[str, str],
    *,
    site: str,
    caller: str | None = None,
) -> ScopedToken | None:
    """⛔ PUBLIC-App token for ONE sandbox-owned repo, gated like a platform repo.

    * no tenant → :class:`PlatformRepoForbiddenError`, ALWAYS (not ENFORCE-gated);
    * a platform tenant, or a tenant whose ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST``
      entry names exactly this repo → minted;
    * any other tenant → :func:`check_platform_repo_tenant` (same log line and
      ENFORCE flag): raises ``platform_repo_forbidden`` under enforce; under
      OBSERVE returns ``None`` so the caller keeps TODAY's behaviour (the
      customer path) — observe never changes an answer.

    The mint: the public App's JWT ``GET /repos/{o}/{r}/installation`` — whose
    ``account`` must be the sandbox org (login match, type ``Organization``;
    :func:`_assert_sandbox_installation`) — then a token restricted to
    ``repositories=[repo]`` with exactly ``permissions``, whose returned repository
    must be owned by ``owner`` (else discarded). Either mismatch →
    :class:`NotInstalledForTenantError`.
    fr is NEVER called. Unconfigured public App →
    :class:`PublicAppNotConfiguredError`; not installed / not covering the repo →
    :class:`NotInstalledForTenantError`; GitHub failure →
    :class:`PublicAppMintFailedError`. No fallback rung exists.
    """
    owner = (owner or "").strip()
    repo = (repo or "").strip()
    if is_tenantless(tenant_id):
        refuse_tenantless_platform_repo(owner, repo, site=site)
    grant = sandbox_repo_grant(tenant_id, owner, repo)
    if grant == "refuse" and not check_platform_repo_tenant(
        owner, repo, tenant_id, site=site, caller=caller
    ):
        return None
    if grant == "scoped":
        refuse_write_on_read_only_grant(owner, repo, tenant_id, permissions, site=site)
    if not public_app_configured():
        raise PublicAppNotConfiguredError(
            "The public GitHub App is not configured on coding-github "
            "(CODING_GITHUB_PUBLIC_APP_ID / _PRIVATE_KEY); no sandbox-repo token.",
            owner=owner,
            repo=repo,
        )
    try:
        installation = resolve_public_installation(owner, repo)
        installation_id = installation.id
        _assert_sandbox_installation(installation, owner, repo, tenant_id, site=site)
        token = mint_public_repo_token(
            installation_id, owner, repo, dict(permissions), expected_owner=owner
        )
    except GitHubAppOwnerMismatchError as e:
        # The minted token's repository is owned by another account (transfer /
        # redirect). Same refusal as an account mismatch; the token is discarded.
        raise NotInstalledForTenantError(
            f"The public GitHub App's token for '{owner}/{repo}' is not for a repository "
            f"owned by sandbox org '{owner}'; refused.",
            owner=owner,
            repo=repo,
        ) from e
    except GitHubAppNotInstalledError as e:
        raise NotInstalledForTenantError(
            f"The public GitHub App is not installed on sandbox org '{owner}' for "
            f"'{owner}/{repo}'.",
            owner=owner,
            repo=repo,
        ) from e
    except GitHubAppNotConfiguredError as e:
        raise PublicAppNotConfiguredError(str(e), owner=owner, repo=repo) from e
    except GitHubAppAuthError as e:
        raise PublicAppMintFailedError(
            f"Could not mint a sandbox-repo token for '{owner}/{repo}': {e}",
            owner=owner,
            repo=repo,
        ) from e
    logger.info(
        "%s decision=sandbox_%s tenant=%s owner=%s repo=%s installation=%s perms=%s caller_path=%s",
        GATE_LOG_TOKEN,
        grant,
        tenant_id,
        owner,
        repo,
        installation_id,
        ",".join(f"{k}:{v}" for k, v in sorted(permissions.items())),
        site,
    )
    return token


def tenant_error_http_exception(e: TenantRepoTokenError) -> HTTPException:
    """Map a refusal to a clear HTTP error naming the owner and a machine code."""
    return error_response(
        status_code=e.http_status,
        message=str(e),
        error_code=e.code,
        # The code is repeated in ``details`` because coding-client's shared HTTP
        # handler renders the wire ``error.code`` as ``HTTP_<status>`` and keeps
        # only ``message`` + ``details`` — without this the machine code never
        # reaches the caller.
        details={"owner": e.owner, "repo": e.repo, "error_code": e.code},
    )


def github_auth_http_exception(e: Exception) -> HTTPException:
    """The single mapping for ``except GitHubAuthError`` in routers.

    A tenant refusal keeps its own status + code; any other GitHubAuthError is
    the historical 412 with the message as detail (unchanged).
    """
    if isinstance(e, TenantRepoTokenError):
        return tenant_error_http_exception(e)
    return HTTPException(status_code=412, detail=str(e))


# ---------------------------------------------------------------------------
# fr binding client (60s TTL cache keyed (tenant, owner.lower()))
# ---------------------------------------------------------------------------
_binding_cache: dict[tuple[str, str], tuple[int | None, float]] = {}
_binding_lock = threading.Lock()


def reset_binding_cache() -> None:
    """Test helper."""
    with _binding_lock:
        _binding_cache.clear()


def _binding_request_parts(
    tenant_id: str, owner: str
) -> tuple[str, dict[str, str], dict[str, str]]:
    settings = get_settings()
    fr_url = (getattr(settings, "fr_url", "") or "").rstrip("/")
    token = getattr(settings, "fr_binding_token", "") or ""
    if not fr_url or not token:
        raise TenantBindingUnavailableError(
            "fr binding lookup is not configured (CODING_GITHUB_FR_URL / "
            "CODING_GITHUB_FR_BINDING_TOKEN); refusing to issue a customer-repo token.",
            owner=owner,
        )
    return (
        f"{fr_url}{BINDING_PATH}",
        {"tenant_id": tenant_id, "owner": owner},
        {"X-Internal-Token": token, "Accept": "application/json"},
    )


def _parse_binding(response: httpx.Response, tenant_id: str, owner: str) -> int | None:
    if response.status_code != 200:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' failed (HTTP {response.status_code}); "
            "refusing (fail closed).",
            owner=owner,
        )
    try:
        payload: Any = response.json()
    except ValueError as e:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' returned non-JSON; refusing.", owner=owner
        ) from e
    if not isinstance(payload, dict) or "installation_id" not in payload:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' returned an unexpected shape; refusing.",
            owner=owner,
        )
    raw = payload["installation_id"]
    if raw is None:
        return None
    # bool is an int subclass; a JSON true must not become installation 1.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' returned an invalid installation_id; refusing.",
            owner=owner,
        )
    return raw


def _cache_get(key: tuple[str, str]) -> tuple[bool, int | None]:
    with _binding_lock:
        hit = _binding_cache.get(key)
        if hit is not None and hit[1] > time.monotonic():
            return True, hit[0]
    return False, None


def _cache_put(key: tuple[str, str], value: int | None) -> None:
    with _binding_lock:
        _binding_cache[key] = (value, time.monotonic() + BINDING_TTL_SECONDS)


def lookup_binding(tenant_id: str, owner: str) -> int | None:
    """Installation id fr has bound to ``tenant_id`` for ``owner``, or ``None``.

    Raises :class:`TenantBindingUnavailableError` on ANY fr failure (never None).
    """
    if not tenant_id:
        raise TenantRequiredError(f"A tenant is required for owner '{owner}'.", owner=owner)
    key = (str(tenant_id), owner.lower())
    hit, value = _cache_get(key)
    if hit:
        return value
    url, params, headers = _binding_request_parts(str(tenant_id), owner)
    try:
        response = httpx.get(url, params=params, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as e:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' could not reach fr: {type(e).__name__}; "
            "refusing (fail closed).",
            owner=owner,
        ) from e
    value = _parse_binding(response, str(tenant_id), owner)
    _cache_put(key, value)
    return value


async def alookup_binding(tenant_id: str, owner: str) -> int | None:
    """Async twin of :func:`lookup_binding` (same cache, same fail-closed rules)."""
    if not tenant_id:
        raise TenantRequiredError(f"A tenant is required for owner '{owner}'.", owner=owner)
    key = (str(tenant_id), owner.lower())
    hit, value = _cache_get(key)
    if hit:
        return value
    url, params, headers = _binding_request_parts(str(tenant_id), owner)
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url, params=params, headers=headers)
    except httpx.HTTPError as e:
        raise TenantBindingUnavailableError(
            f"fr binding lookup for owner '{owner}' could not reach fr: {type(e).__name__}; "
            "refusing (fail closed).",
            owner=owner,
        ) from e
    value = _parse_binding(response, str(tenant_id), owner)
    _cache_put(key, value)
    return value


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------
def resolve_customer_repo_token(
    owner: str,
    repo: str,
    tenant_id: str | None,
    permissions: dict[str, str],
    *,
    site: str = "customer_repo_token",
    caller: str | None = None,
) -> ScopedToken:
    """Tenant-bound public-App token for a NON-platform ``owner/repo``.

    Raises a :class:`TenantRepoTokenError` subclass on every refusal. Never
    returns without a token.

    ⛔ A SANDBOX owner (``CODING_GITHUB_SANDBOX_GITHUB_OWNERS``) is routed to
    :func:`resolve_sandbox_repo_token` FIRST — platform-gated, no fr lookup. Only
    an OBSERVE-mode "would refuse" falls through to the customer path below.
    """
    if is_sandbox_owned(owner):
        sandbox = resolve_sandbox_repo_token(
            owner, repo, tenant_id, permissions, site=site, caller=caller
        )
        if sandbox is not None:
            return sandbox
    if not tenant_id:
        raise TenantRequiredError(
            f"A tenant is required to obtain a GitHub credential for '{owner}/{repo}' "
            "(repositories outside the platform org are tenant-bound).",
            owner=owner,
            repo=repo,
        )
    bound_id = lookup_binding(tenant_id, owner)
    if bound_id is None:
        raise NotInstalledForTenantError(
            f"The GitHub App is not installed on '{owner}' for this tenant. Install it from "
            "Settings → GitHub and grant access to the repository.",
            owner=owner,
            repo=repo,
        )
    if not public_app_configured():
        raise PublicAppNotConfiguredError(
            "The public GitHub App is not configured on coding-github "
            "(CODING_GITHUB_PUBLIC_APP_ID / _PRIVATE_KEY).",
            owner=owner,
            repo=repo,
        )
    try:
        actual_id = resolve_public_installation_id(owner, repo)
    except GitHubAppNotInstalledError as e:
        raise NotInstalledForTenantError(
            f"The GitHub App installation on '{owner}' does not cover '{owner}/{repo}'. "
            "Grant the App access to this repository on GitHub.",
            owner=owner,
            repo=repo,
        ) from e
    except GitHubAppNotConfiguredError as e:
        raise PublicAppNotConfiguredError(str(e), owner=owner, repo=repo) from e
    except GitHubAppAuthError as e:
        raise PublicAppMintFailedError(
            f"Could not confirm the GitHub App installation for '{owner}/{repo}': {e}",
            owner=owner,
            repo=repo,
        ) from e
    if actual_id != bound_id:
        # ⛔ The tenant check. GitHub says a DIFFERENT installation covers this
        # repo than the one fr bound to this tenant.
        logger.warning(
            "Refusing customer-repo token: %s/%s is covered by installation %s but tenant %s "
            "is bound to %s",
            owner,
            repo,
            actual_id,
            tenant_id,
            bound_id,
        )
        raise NotInstalledForTenantError(
            f"The GitHub App installation covering '{owner}/{repo}' is not the one bound to "
            "this tenant.",
            owner=owner,
            repo=repo,
        )
    try:
        token = mint_public_repo_token(actual_id, owner, repo, permissions)
    except GitHubAppNotConfiguredError as e:
        raise PublicAppNotConfiguredError(str(e), owner=owner, repo=repo) from e
    except GitHubAppMintError as e:
        raise PublicAppMintFailedError(str(e), owner=owner, repo=repo) from e
    logger.info(
        "Tenant-bound public-App token issued (tenant=%s, repo=%s/%s, installation=%s, perms=%s)",
        tenant_id,
        owner,
        repo,
        actual_id,
        ",".join(f"{k}:{v}" for k, v in sorted(permissions.items())),
    )
    return token


def resolve_tenant_repo_token(
    owner: str, repo: str, tenant_id: str | None, permissions: dict[str, str]
) -> str:
    """Token for ``owner/repo`` on behalf of ``tenant_id``.

    * Platform owner → ONLY for a platform tenant (:func:`check_platform_repo_tenant`,
      strict: an absent tenant is refused too), then the internal-App path
      (:func:`services.github_integration.resolve_github_token`): App mandatory,
      :class:`GitHubAuthError` if it cannot mint. ``permissions`` does not narrow
      it (the internal path never did).
    * Anyone else → :func:`resolve_customer_repo_token` (tenant-bound public App).
    """
    if is_platform_owned(owner):
        from services.github_integration import resolve_github_token

        if platform_repo_grant(tenant_id, owner, repo) == "scoped":
            # Allowlisted non-platform tenant: ONE repo, internal App, no fallback.
            return mint_allowlisted_platform_token(
                owner, repo, permissions, tenant_id=tenant_id
            ).token
        if is_tenantless(tenant_id):
            refuse_tenantless_platform_repo(owner, repo, site="resolve_tenant_repo_token")
        check_platform_repo_tenant(owner, repo, tenant_id, site="resolve_tenant_repo_token")
        # tenant_id IS forwarded: resolve_github_token refuses a platform repo
        # with no tenant, so a tenantless call here can never reach the org
        # token. (Under OBSERVE a non-platform tenant is logged twice; that is
        # the price of never having a tenantless path into the ladder.)
        resolved = resolve_github_token(owner=owner, repo=repo, tenant_id=tenant_id)
        if not resolved.token:
            raise GitHubAuthError(f"No GitHub credential for platform repo {owner}/{repo}.")
        return resolved.token
    return resolve_customer_repo_token(owner, repo, tenant_id, permissions).token
