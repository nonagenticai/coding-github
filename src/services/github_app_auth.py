"""GitHub App installation-token authentication for coding-github.

Turns a GitHub App identity (App ID + installation ID + private-key PEM) into a
short-lived installation access token (``ghs_…``) so the platform mints its own
credentials for PR-producing flows without a per-run personal access token.

Reuses PyGithub's :class:`github.Auth.AppAuth` / ``AppInstallationAuth`` — which
already do RS256 JWT signing, the ``POST /app/installations/{id}/access_tokens``
exchange, and token caching/refresh — so **no new dependency is added**.

The PEM is loaded from a mounted file
(``CODING_GITHUB_GITHUB_APP_PRIVATE_KEY_PATH``, mirroring the arc-runners
``github-app-credentials`` secret) with an inline-env fallback
(``CODING_GITHUB_GITHUB_APP_PRIVATE_KEY``) for dev/test.

The App is a *fallback* token source, not a mandatory one: every helper raises
:class:`GitHubAppAuthError` (not a bare ``Exception``) when unconfigured so the
token resolver can fall through to the next source.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast
from urllib.parse import quote

import httpx

from config.settings import get_settings

try:
    from github import Auth, GithubIntegration

    HAS_GITHUB = True
except ImportError:  # pragma: no cover - exercised only when PyGithub absent
    Auth = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    GithubIntegration = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    HAS_GITHUB = False

logger = logging.getLogger(__name__)


class GitHubAppAuthError(Exception):
    """GitHub App auth is unconfigured or failed.

    Callers treat this as "fall through to the next token source" rather than a
    hard failure — the App is one rung in :func:`resolve_github_token`.

    The three subclasses below all inherit from this, so every existing
    ``except GitHubAppAuthError`` keeps working unchanged. They exist because
    "the App is not installed on this owner" and "the App IS installed and the
    mint FAILED" are operationally opposite events that this class alone cannot
    tell apart: the first is the normal, expected answer for any owner outside
    the App's installations, the second is a defect. Collapsing them made a
    GitHub outage indistinguishable from a routine miss in every downstream log.
    """


class GitHubAppNotConfiguredError(GitHubAppAuthError):
    """No usable App identity here (no app id / no private key / no PyGithub)."""


class GitHubAppNotInstalledError(GitHubAppAuthError):
    """The App exists but is not installed on this owner/repo (GitHub said 404).

    Expected and normal — most owners are not customers of this App.
    """


class GitHubAppMintError(GitHubAppAuthError):
    """The App should have been able to answer, and the attempt FAILED.

    Covers a network failure, a non-404 GitHub error while resolving the
    installation (e.g. a suspended installation, rate limiting, a 5xx), a
    refused token mint, and an empty token in a 2xx body.

    ``status_code`` is GitHub's HTTP status when GitHub answered and refused the
    mint (e.g. 422 when the installation was never granted a requested
    permission); ``None`` for a network failure or an empty token.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GitHubAppOwnerMismatchError(GitHubAppAuthError):
    """GitHub's answer names a DIFFERENT account/owner than the one requested.

    Raised only by the sandbox (platform-owned, public-App) path: the installation
    that covers ``owner/repo`` belongs to another account, or the minted token's
    repository is owned by someone else (a transferred / redirected repo, a
    user-account installation). The token, if one was minted, is discarded.
    """


# Module-level caches. A single GitHub App can be installed on many orgs (one
# installation id each) — the bulk dep-upgrade campaign resolves the right
# installation PER OWNER (Phase D3). We mint installation tokens via the
# app-level ``GithubIntegration.get_access_token(id)`` (the working PyGithub
# path; a bare ``AppAuth.get_installation_auth(id).token`` raises
# "withRequester must be called first" when used standalone) and cache the
# minted token per installation id with a TTL well inside GitHub's ~60-minute
# expiry. The app-level integration (JWT signer) is cached once.
_TOKEN_TTL_SECONDS = 50 * 60  # refresh ~10min before GitHub's ~60min expiry
_token_by_id: dict[int, tuple[str, float]] = {}  # id -> (token, monotonic_expiry)

# ---------------------------------------------------------------------------
# App identities. There are TWO GitHub Apps and they must never share state.
# ---------------------------------------------------------------------------
# ``internal`` — the platform App (code-graph-helper-app), installed on
#   ``system_github_owner``. Every pre-existing helper in this module defaults
#   to it, so its behaviour is unchanged.
# ``public`` — the customer-installable App. It only ever mints REPO-SCOPED
#   tokens, and only via ``services.tenant_binding.resolve_tenant_repo_token``,
#   which checks fr's tenant binding first. Nothing else may call its mint.
#
# Every cache below is keyed by identity: the JWT signer, the PyGithub
# integration and the scoped-token cache. Installation ids are GitHub-global,
# so an id-only key could not collide across Apps in practice — but a cache
# that CAN serve one App's credential for the other is a cache that will, the
# day someone reuses a key shape. So they are simply separate dicts.
AppIdentity = Literal["internal", "public"]
INTERNAL_APP: AppIdentity = "internal"
PUBLIC_APP: AppIdentity = "public"


class _AppSettingNames(NamedTuple):
    label: str  # human label used in error messages
    app_id: str
    inline_key: str
    key_path: str
    key_path_env: str
    inline_key_env: str


_APP_SETTINGS: dict[str, _AppSettingNames] = {
    INTERNAL_APP: _AppSettingNames(
        "GitHub App",
        "github_app_id",
        "github_app_private_key",
        "github_app_private_key_path",
        "CODING_GITHUB_GITHUB_APP_PRIVATE_KEY_PATH",
        "CODING_GITHUB_GITHUB_APP_PRIVATE_KEY",
    ),
    PUBLIC_APP: _AppSettingNames(
        "Public GitHub App",
        "public_app_id",
        "public_app_private_key",
        "public_app_private_key_path",
        "CODING_GITHUB_PUBLIC_APP_PRIVATE_KEY_PATH",
        "CODING_GITHUB_PUBLIC_APP_PRIVATE_KEY",
    ),
}

_app_auth_by_app: dict[str, object] = {}
_app_integration_by_app: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Scoped clone tokens (read-only, single-repo) — SEPARATE cache. Do not merge.
# ---------------------------------------------------------------------------
# ⚠️ INVARIANT: a scoped token MUST NEVER enter ``_token_by_id``.
#
# ``_token_by_id`` is keyed on the installation id ALONE, and it backs
# ``get_installation_token()`` / ``get_installation_token_for()`` — the tokens
# that OPEN PULL REQUESTS. A clone token is deliberately crippled
# (``contents:read`` on ONE repo). If a scoped mint ever wrote into
# ``_token_by_id``, the next PR-opening call for that installation would get a
# read-only single-repo token back from the cache and every AA pull request
# would fail — silently, and only until the TTL expired. So scoped tokens live
# in their own dict under a key that carries the full scope
# ``(installation_id, owner, repo, permission-scope)``; the two caches share
# nothing but the App JWT signer. ``test_scoped_mint_does_not_poison_pr_token``
# pins this.
_CLONE_SCOPE = "contents:read"

# The ONLY permission sets a scoped token may carry. Keyed by the public scope
# name the internal endpoint accepts; there is no way to pass free-form
# permissions through. ``pr-read`` exists because
# ``GET /repos/{o}/{r}/pulls/{n}/files`` 403s ("Resource not accessible by
# integration") on a contents-only token — it needs ``pull_requests:read``.
#
# ``push`` (contents:write, ONE repo) is the ONLY write scope: coding-containers
# pushes the workspace branch with it. ``pull_requests``/``workflows`` write are
# deliberately NOT expressible here — opening a PR stays in coding-github, and a
# workflow-write token in a container that runs generated code could rewrite CI.
# For a customer owner it is minted only through the tenant-bound public App,
# exactly like ``clone``/``pr-read``. Its cache key carries ``contents:write``,
# so it can never be served for a read scope or vice versa.
TokenScope = Literal["clone", "pr-read", "push"]
_SCOPE_PERMISSIONS: dict[str, dict[str, str]] = {
    "clone": {"contents": "read"},
    "pr-read": {"contents": "read", "pull_requests": "read"},
    "push": {"contents": "write"},
}


def _scope_label(permissions: dict[str, str]) -> str:
    """Cache-key/log label for a permission set, e.g. ``contents:read,pull_requests:read``."""
    return ",".join(f"{k}:{v}" for k, v in sorted(permissions.items()))


_EXPIRY_SAFETY_MARGIN_SECONDS = 5 * 60  # stop serving a cached token 5min early
_GITHUB_API_BASE = "https://api.github.com"
_HTTP_TIMEOUT_SECONDS = 10.0

# (installation_id, owner, repo, scope) -> (token, expires_at_iso, monotonic_expiry)
_scoped_token_by_repo: dict[tuple[int, str, str, str], tuple[str, str, float]] = {}
# The PUBLIC App's repo-scoped tokens. Same key shape (owner/repo lower-cased),
# separate dict: a public-App token is never served for an internal request
# and vice versa.
_public_scoped_token_by_repo: dict[tuple[int, str, str, str], tuple[str, str, float]] = {}
_SCOPED_CACHES: dict[str, dict[tuple[int, str, str, str], tuple[str, str, float]]] = {
    INTERNAL_APP: _scoped_token_by_repo,
    PUBLIC_APP: _public_scoped_token_by_repo,
}


class ScopedToken(NamedTuple):
    """A single-repo, read-only installation token and its GitHub expiry."""

    token: str
    expires_at: str


class AppInstallation(NamedTuple):
    """An installation id plus the account GitHub says it is installed on."""

    id: int
    account_login: str
    account_type: str


def _github_headers(credential: str | None = None) -> dict[str, str]:
    """Standard GitHub REST headers; ``credential`` may be an App JWT or a token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    return headers


def _load_private_key(app: AppIdentity = INTERNAL_APP) -> str:
    """Return the App private-key PEM, preferring a mounted file over inline env."""
    names = _APP_SETTINGS[app]
    settings = get_settings()
    path = getattr(settings, names.key_path, "") or ""
    if path:
        try:
            pem = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            raise GitHubAppNotConfiguredError(
                f"Cannot read {names.label} private key file {path}: {e}"
            ) from e
        if pem.strip():
            return pem
        logger.warning("%s private key file %s is empty; trying inline key", names.label, path)
    inline = getattr(settings, names.inline_key, "") or ""
    if inline.strip():
        return inline
    raise GitHubAppNotConfiguredError(
        f"No {names.label} private key configured "
        f"(neither {names.key_path_env} file nor inline key)."
    )


def public_app_configured() -> bool:
    """True when the PUBLIC App has an id and a private key (file or inline)."""
    if not HAS_GITHUB:
        return False
    names = _APP_SETTINGS[PUBLIC_APP]
    settings = get_settings()
    if not getattr(settings, names.app_id, ""):
        return False
    path = getattr(settings, names.key_path, "") or ""
    if path and Path(path).is_file():
        return True
    return bool(getattr(settings, names.inline_key, "") or "")


def app_auth_configured() -> bool:
    """True when App ID + installation ID + a private key are all present."""
    if not HAS_GITHUB:
        return False
    settings = get_settings()
    if not (settings.github_app_id and getattr(settings, "github_app_installation_id", "")):
        return False
    path = getattr(settings, "github_app_private_key_path", "") or ""
    if path and Path(path).is_file():
        return True
    return bool(getattr(settings, "github_app_private_key", "") or "")


def _get_app_auth(app: AppIdentity = INTERNAL_APP):
    """Return a cached ``Auth.AppAuth`` — the App identity / RS256 JWT signer.

    Shared by the PyGithub integration (below) and by the raw scoped-token mint,
    which needs the App JWT directly because PyGithub cannot express
    repository-scoping (see :func:`get_scoped_clone_token`). Cached PER App.
    """
    cached = _app_auth_by_app.get(app)
    if cached is not None:
        return cached
    names = _APP_SETTINGS[app]
    if not HAS_GITHUB:
        raise GitHubAppNotConfiguredError(f"PyGithub is not available for {names.label} auth.")
    settings = get_settings()
    app_id = getattr(settings, names.app_id, "") or ""
    if not app_id:
        raise GitHubAppNotConfiguredError(f"{names.label} ID not configured.")
    private_key = _load_private_key(app)
    try:
        auth = Auth.AppAuth(app_id, private_key)
    except Exception as e:
        raise GitHubAppNotConfiguredError(f"Failed to build {names.label} auth: {e}") from e
    _app_auth_by_app[app] = auth
    return auth


def _get_app_integration(app: AppIdentity = INTERNAL_APP):
    """Return a cached app-level ``GithubIntegration`` (the JWT signer), per App.

    Resolves owner→installation and mints installation tokens — the entry point
    for the multi-installation, per-owner token resolution (Phase D3).
    """
    cached = _app_integration_by_app.get(app)
    if cached is not None:
        return cached
    app_auth = _get_app_auth(app)
    try:
        integration = GithubIntegration(auth=app_auth)
    except Exception as e:
        raise GitHubAppNotConfiguredError(
            f"Failed to build {_APP_SETTINGS[app].label} integration: {e}"
        ) from e
    _app_integration_by_app[app] = integration
    return integration


def _mint_installation_token(installation_id: int) -> str:
    """Mint (or return a cached) installation token for ``installation_id``.

    Uses ``GithubIntegration.get_access_token(id)`` — the working PyGithub path.
    Cached with a TTL inside GitHub's ~60-minute token lifetime.
    """
    cached = _token_by_id.get(installation_id)
    if cached is not None and cached[1] > time.monotonic():
        return cached[0]
    integration = _get_app_integration()
    try:
        authorization = integration.get_access_token(installation_id)
    except Exception as e:
        raise GitHubAppMintError(
            f"Failed to mint GitHub App installation token (id={installation_id}): {e}"
        ) from e
    token = getattr(authorization, "token", None)
    if not token:
        raise GitHubAppMintError("GitHub App installation token was empty.")
    _token_by_id[installation_id] = (token, time.monotonic() + _TOKEN_TTL_SECONDS)
    logger.info("GitHub App installation token minted (installation_id=%s)", installation_id)
    return token


def get_installation_token() -> str:
    """Return a raw installation access token (``ghs_…``) for the home installation."""
    settings = get_settings()
    installation_id = getattr(settings, "github_app_installation_id", "")
    if not (settings.github_app_id and installation_id):
        raise GitHubAppNotConfiguredError("GitHub App ID / installation ID not configured.")
    return _mint_installation_token(int(installation_id))


def _lookup_installation(owner: str, repo: str, app: AppIdentity = INTERNAL_APP) -> AppInstallation:
    """Return the App installation that covers ``owner/repo`` (id + account).

    Raises :class:`GitHubAppNotInstalledError` when the App isn't installed there
    (a 404 from GitHub) so callers can fall through to another credential source.

    ⚠️ Only a 404 means "not installed". Every other failure (a suspended
    installation, a rate-limit, a 5xx, a network error) means the question was
    never answered — that is :class:`GitHubAppMintError`, not an absence. The
    old code raised the same error for both, so a GitHub outage was reported
    downstream as "the App is not installed on that owner" and the operator had
    no way to see the difference.
    """
    integration = _get_app_integration(app)
    label = _APP_SETTINGS[app].label
    try:
        installation = integration.get_repo_installation(owner, repo)
    except Exception as e:
        # PyGithub's GithubException carries .status; a non-HTTP failure has none.
        if getattr(e, "status", None) == 404:
            raise GitHubAppNotInstalledError(
                f"{label} is not installed on {owner}/{repo}: {e}"
            ) from e
        raise GitHubAppMintError(
            f"Could not resolve the {label} installation for {owner}/{repo} "
            f"(the App may be installed; the lookup itself failed): {e}"
        ) from e
    # raw_data only: never a lazy PyGithub fetch. Missing ⇒ "" (callers that
    # check the account then fail CLOSED).
    raw = getattr(installation, "raw_data", None)
    account = raw.get("account") if isinstance(raw, dict) else None
    if not isinstance(account, dict):
        account = {}
    return AppInstallation(
        int(installation.id), str(account.get("login") or ""), str(account.get("type") or "")
    )


def _resolve_installation_id(owner: str, repo: str, app: AppIdentity = INTERNAL_APP) -> int:
    """Return the id of the App installation that covers ``owner/repo``.

    See :func:`_lookup_installation` for the error contract.
    """
    return _lookup_installation(owner, repo, app).id


def get_installation_token_for(owner: str, repo: str) -> str:
    """Return an installation access token for the App's install on ``owner/repo``.

    Resolves the installation id that covers the repo (the App may be installed
    on many orgs — one id each) and mints/returns-cached that installation's
    token. Raises :class:`GitHubAppAuthError` (so the resolver falls through to
    the next token source) when the App isn't configured or isn't installed on
    that owner.

    This is the FULL-PERMISSION installation token (it opens pull requests). It
    is deliberately NOT the token handed to a workspace container — see
    :func:`get_scoped_clone_token`.
    """
    return _mint_installation_token(_resolve_installation_id(owner, repo))


def _cache_ttl_for(expires_at: str) -> float:
    """Seconds a token may be served from cache, derived from GitHub's expiry.

    Stops serving ``_EXPIRY_SAFETY_MARGIN_SECONDS`` early so a token handed out
    on the last cache hit is still valid for the clone that follows. Falls back
    to the standard TTL when GitHub's timestamp is missing/unparseable, and
    returns 0 (never cache) for an already-expired token.
    """
    if not expires_at:
        return _TOKEN_TTL_SECONDS
    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        logger.warning("Unparseable expires_at from GitHub; using default token TTL")
        return _TOKEN_TTL_SECONDS
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    remaining = (expiry - datetime.now(UTC)).total_seconds() - _EXPIRY_SAFETY_MARGIN_SECONDS
    return max(0.0, min(remaining, _TOKEN_TTL_SECONDS))


def get_scoped_clone_token(
    owner: str, repo: str, scope: TokenScope = "clone", *, fresh: bool = False
) -> ScopedToken:
    """Mint a read-only, single-repo installation token for cloning ``owner/repo``.

    The token GitHub returns is restricted to ``contents:read`` on ``repo`` ONLY
    and expires in ~1h — safe to hand to a workspace container, unlike the
    full-permission installation token from :func:`get_installation_token_for`.

    Why a raw HTTP POST instead of PyGithub: repository-scoping is not
    expressible through PyGithub (2.9.1). ``GithubIntegration.get_access_token``
    hardcodes its request body to ``{"permissions": permissions}`` and the
    package has no ``repositories`` kwarg anywhere — so a PyGithub-minted token
    would be ``contents:read`` across **every repo in the installation**. We
    therefore POST the access-tokens endpoint ourselves, reusing the App JWT
    that ``Auth.AppAuth`` already signs (no hand-rolled RSA).

    ``scope`` selects one of the fixed permission sets in ``_SCOPE_PERMISSIONS``
    (``clone`` = contents:read, ``pr-read`` = contents:read + pull_requests:read).
    The scope is part of the cache key, so a token cached for one scope is never
    served for the other.

    Raises :class:`GitHubAppAuthError` when the App is unconfigured, not
    installed on ``owner``, or GitHub refuses the mint; :class:`ValueError` for
    an unknown ``scope``.
    """
    permissions = _SCOPE_PERMISSIONS.get(scope)
    if permissions is None:
        raise ValueError(f"Unknown scoped-token scope {scope!r}")
    installation_id = _resolve_installation_id(owner, repo)
    return _mint_repo_scoped_token(
        INTERNAL_APP,
        installation_id,
        owner,
        repo,
        permissions,
        noun="scoped clone token",
        fresh=fresh,
    )


def _mint_repo_scoped_token(
    app: AppIdentity,
    installation_id: int,
    owner: str,
    repo: str,
    permissions: dict[str, str],
    *,
    noun: str,
    expected_owner: str | None = None,
    fresh: bool = False,
) -> ScopedToken:
    """POST ``/app/installations/{id}/access_tokens`` restricted to ONE repo.

    ``fresh=True`` neither reads nor writes the cache: the caller owns the token and
    REVOKES it when done (``create_rust_target_repo``). A revoked token must never
    sit in a cache that other requests are served from.

    Shared by the internal clone-token mint and the public App's tenant-bound
    mint. The cache is chosen by ``app`` (never shared across Apps) and keyed by
    the full scope, so a token for one permission set is never served for
    another.

    ``expected_owner`` (sandbox path): GitHub's response must list exactly ONE
    repository, named ``repo`` and owned by ``expected_owner`` (case-insensitive);
    otherwise the fresh token is discarded (never cached, never returned) and
    :class:`GitHubAppOwnerMismatchError` is raised.
    """
    scope_label = _scope_label(permissions)
    cache = _SCOPED_CACHES[app]
    if app == INTERNAL_APP:
        key = (installation_id, owner, repo, scope_label)
    else:
        key = (installation_id, owner.lower(), repo.lower(), scope_label)
    cached = None if fresh else cache.get(key)
    if cached is not None and cached[2] > time.monotonic():
        return ScopedToken(cached[0], cached[1])

    app_jwt = _get_app_auth(app).create_jwt()
    try:
        response = httpx.post(
            f"{_GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens",
            json={"permissions": dict(permissions), "repositories": [repo]},
            headers=_github_headers(app_jwt),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        raise GitHubAppMintError(
            f"Failed to reach GitHub to mint a {noun} for {owner}/{repo}: {e}"
        ) from e
    if response.status_code not in (200, 201):
        # Never log the body — it carries the credential on success.
        raise GitHubAppMintError(
            f"GitHub refused a {noun} for {owner}/{repo} (HTTP {response.status_code}).",
            status_code=response.status_code,
        )

    payload = response.json()
    token = payload.get("token") or ""
    expires_at = payload.get("expires_at") or ""
    if not token:
        raise GitHubAppMintError(f"GitHub returned an empty {noun}.")
    if expected_owner is not None:
        _assert_minted_repo_owner(payload, expected_owner, owner, repo)

    # NOTE: writes to the per-App scoped cache ONLY. Never _token_by_id — see
    # the invariant at the top of this module.
    if not fresh:
        cache[key] = (token, expires_at, time.monotonic() + _cache_ttl_for(expires_at))
    logger.info(
        "%s minted (app=%s, installation_id=%s, repo=%s/%s, scope=%s, expires_at=%s)",
        noun[:1].upper() + noun[1:],
        app,
        installation_id,
        owner,
        repo,
        scope_label,
        expires_at,
    )
    return ScopedToken(token, expires_at)


def _assert_minted_repo_owner(payload: object, expected_owner: str, owner: str, repo: str) -> None:
    """⛔ The minted token's single repository must be ``expected_owner/repo``."""
    body: dict[str, Any] = cast("dict[str, Any]", payload) if isinstance(payload, dict) else {}
    repos = body.get("repositories")
    seen: list[str] = []
    if isinstance(repos, list):
        for r in repos:
            r = r if isinstance(r, dict) else {}
            o = r.get("owner") if isinstance(r.get("owner"), dict) else {}
            seen.append(f"{o.get('login') or '?'}/{r.get('name') or '?'}")
    want = f"{expected_owner.strip().lower()}/{repo.strip().lower()}"
    if [x.lower() for x in seen] != [want]:
        logger.error(
            "OWNER MISMATCH: refusing public-App token for %s/%s; GitHub's token covers %s, "
            "expected exactly %s. Token discarded.",
            owner,
            repo,
            seen or "no repositories",
            want,
        )
        raise GitHubAppOwnerMismatchError(
            f"GitHub minted a token for {seen or 'no repositories'}, not {want}; refused."
        )


_ALLOWED_PERMISSION_LEVELS = frozenset({"read", "write"})


def resolve_public_installation_id(owner: str, repo: str) -> int:
    """The PUBLIC App's installation id covering ``owner/repo`` (App-JWT lookup).

    404 → :class:`GitHubAppNotInstalledError`; anything else →
    :class:`GitHubAppMintError`; unconfigured → :class:`GitHubAppNotConfiguredError`.
    Callers MUST compare the result with the tenant's fr binding before minting.
    """
    return _resolve_installation_id(owner, repo, PUBLIC_APP)


def resolve_public_installation(owner: str, repo: str) -> AppInstallation:
    """Like :func:`resolve_public_installation_id`, plus the installation's account."""
    return _lookup_installation(owner, repo, PUBLIC_APP)


def mint_public_repo_token(
    installation_id: int,
    owner: str,
    repo: str,
    permissions: dict[str, str],
    *,
    expected_owner: str | None = None,
    fresh: bool = False,
) -> ScopedToken:
    """Mint a PUBLIC-App token scoped to ``repo`` with exactly ``permissions``.

    ``fresh=True``: uncached, for a caller that revokes it (see ``_mint_repo_scoped_token``).

    ⛔ Performs NO tenant check. The only legitimate caller is
    :func:`services.tenant_binding.resolve_tenant_repo_token`, which verifies
    the fr binding and GitHub's installation id first.
    """
    if not permissions:
        raise ValueError("A public-App token needs an explicit permission set")
    for name, level in permissions.items():
        if level not in _ALLOWED_PERMISSION_LEVELS or not name.replace("_", "").isalpha():
            raise ValueError(f"Invalid permission {name}={level!r}")
    return _mint_repo_scoped_token(
        PUBLIC_APP,
        installation_id,
        owner,
        repo,
        permissions,
        noun="tenant repo token",
        expected_owner=expected_owner,
        fresh=fresh,
    )


def get_repo_visibility(owner: str, repo: str, token: str | None = None) -> str:
    """Return ``"public"`` or ``"private"`` for ``owner/repo``.

    coding-github is the only coding-* service with egress to api.github.com, so
    it answers this for siblings: a public repo can be cloned with no credential
    at all, a private one cannot.

    ``token`` (when supplied) authenticates the lookup — an installation token
    carries implicit ``metadata:read``, so a scoped clone token is enough. Without
    one only public repos are visible, so a 404 is reported as ``"private"``: the
    fail-safe answer, meaning "a credential is required to see this".
    """
    try:
        response = httpx.get(
            f"{_GITHUB_API_BASE}/repos/{owner}/{repo}",
            headers=_github_headers(token),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        raise GitHubAppAuthError(
            f"Failed to reach GitHub to resolve visibility of {owner}/{repo}: {e}"
        ) from e
    if response.status_code == 200:
        # Absent `private` ⇒ assume private (fail safe, don't guess "cloneable").
        return "private" if response.json().get("private", True) else "public"
    if response.status_code == 404:
        return "private"
    raise GitHubAppAuthError(
        f"Failed to resolve visibility of {owner}/{repo} (HTTP {response.status_code})."
    )


class RepoPushMetadata(NamedTuple):
    """What the push-scope mint reads from ONE ``GET /repos/{o}/{r}``.

    ``private`` is GitHub's RAW field (``None`` when absent or on a 404) so the
    fail-closed guard sees an unknown as unknown. ``default_branch`` is ``None``
    unless GitHub named a non-empty string.
    """

    private: object
    default_branch: str | None


def get_repo_push_metadata(owner: str, repo: str, token: str) -> RepoPushMetadata:
    """``private`` + ``default_branch`` for ``owner/repo``, looked up WITH ``token``.

    ⛔ Unlike :func:`get_repo_visibility` this does NOT fold "could not tell" into
    ``"private"``. That fold is the right answer for a CLONE (it means "a credential is
    needed") and the wrong one for a WRITE: the push-scope guard must see an absent
    field or a 404 as UNKNOWN, so both come back as ``None`` and
    ``assert_repo_not_public`` refuses. A transport/HTTP failure raises
    :class:`GitHubAppAuthError`.

    coding-containers has no egress to api.github.com, so this service is the only
    place its push route can learn either fact from (it refuses a public repo and a
    push to the default branch, and both answers must come from GitHub, never from
    the workspace).
    """
    try:
        response = httpx.get(
            f"{_GITHUB_API_BASE}/repos/{owner}/{repo}",
            headers=_github_headers(token),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        raise GitHubAppAuthError(
            f"Failed to reach GitHub to resolve visibility of {owner}/{repo}: {e}"
        ) from e
    if response.status_code == 200:
        body = response.json()
        if not isinstance(body, dict):
            return RepoPushMetadata(private=None, default_branch=None)
        branch = body.get("default_branch")
        return RepoPushMetadata(
            private=body.get("private"),
            default_branch=branch if isinstance(branch, str) and branch else None,
        )
    if response.status_code == 404:
        return RepoPushMetadata(private=None, default_branch=None)
    raise GitHubAppAuthError(
        f"Failed to resolve visibility of {owner}/{repo} (HTTP {response.status_code})."
    )


def get_repo_private_flag(owner: str, repo: str, token: str) -> object:
    """GitHub's raw ``private`` field (see :func:`get_repo_push_metadata`)."""
    return get_repo_push_metadata(owner, repo, token).private


RefHeadState = Literal["present", "absent", "unknown", "empty_repository"]

#: A full commit id (SHA-1 or SHA-256) as the GitHub API prints it.
_COMMIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class RefHead(NamedTuple):
    """The head of ``refs/heads/<branch>`` as GitHub reports it.

    ``present`` ⇒ ``sha`` is a full commit id; ``absent`` ⇒ GitHub 404'd the ref;
    ``empty_repository`` ⇒ GitHub says the repo has no commits (409 "Git Repository
    is empty."); ``unknown`` ⇒ anything else (transport error, other status,
    malformed body). Callers must treat ``unknown`` as a refusal, never as absent.
    """

    state: RefHeadState
    sha: str | None = None


def get_branch_head(owner: str, repo: str, branch: str, token: str) -> RefHead:
    """``GET /repos/{o}/{r}/git/ref/heads/{branch}`` (exact match). Never raises."""
    url = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/git/ref/heads/{quote(branch, safe='/')}"
    try:
        response = httpx.get(url, headers=_github_headers(token), timeout=_HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as e:
        logger.error("branch-head lookup for %s/%s failed: %s", owner, repo, type(e).__name__)
        return RefHead("unknown")
    if response.status_code == 404:
        return RefHead("absent")
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code == 409:
        message = body.get("message") if isinstance(body, dict) else None
        if isinstance(message, str) and "empty" in message.lower():
            return RefHead("empty_repository")
    if response.status_code != 200 or not isinstance(body, dict):
        logger.error(
            "branch-head lookup for %s/%s failed with HTTP %s", owner, repo, response.status_code
        )
        return RefHead("unknown")
    obj = body.get("object")
    sha = obj.get("sha") if isinstance(obj, dict) else None
    kind = obj.get("type") if isinstance(obj, dict) else None
    if kind != "commit" or not isinstance(sha, str) or not _COMMIT_SHA_RE.match(sha):
        return RefHead("unknown")
    return RefHead("present", sha)


def reset_cache() -> None:
    """Clear the cached installation/scoped tokens + app integration (test helper)."""
    _token_by_id.clear()
    _scoped_token_by_repo.clear()
    _public_scoped_token_by_repo.clear()
    _app_integration_by_app.clear()
    _app_auth_by_app.clear()


def resolve_public_org_installation(org: str) -> AppInstallation:
    """The PUBLIC App's installation on the ORGANISATION ``org`` (App-JWT lookup).

    Used ONLY by ``create_rust_target_repo``: a repository that does not exist yet
    has no ``/repos/{o}/{r}/installation``, so creating one needs the org's.
    404 → :class:`GitHubAppNotInstalledError`; anything else →
    :class:`GitHubAppMintError`.
    """
    integration = _get_app_integration(PUBLIC_APP)
    label = _APP_SETTINGS[PUBLIC_APP].label
    try:
        installation = integration.get_org_installation(org)
    except Exception as e:
        if getattr(e, "status", None) == 404:
            raise GitHubAppNotInstalledError(f"{label} is not installed on org {org}: {e}") from e
        raise GitHubAppMintError(
            f"Could not resolve the {label} installation for org {org}: {e}"
        ) from e
    raw = getattr(installation, "raw_data", None)
    account = raw.get("account") if isinstance(raw, dict) else None
    if not isinstance(account, dict):
        account = {}
    return AppInstallation(
        int(installation.id), str(account.get("login") or ""), str(account.get("type") or "")
    )


def mint_public_org_token(installation_id: int, permissions: dict[str, str]) -> str:
    """Mint an UNCACHED, installation-wide PUBLIC-App token with exactly ``permissions``.

    ⛔ Installation-wide (no ``repositories`` restriction), because the repository it
    exists to create does not exist yet. The only legitimate caller is
    ``services.rust_target`` (``create_rust_target_repo``), which gates the tenant
    and the sandbox installation first and uses the token for ONE create call.
    Never cached, never returned to a caller. A refusal keeps GitHub's status on
    :class:`GitHubAppMintError` (422 = the installation was never GRANTED a
    requested permission).
    """
    if not permissions:
        raise ValueError("An org token needs an explicit permission set")
    for name, level in permissions.items():
        if level not in _ALLOWED_PERMISSION_LEVELS or not name.replace("_", "").isalpha():
            raise ValueError(f"Invalid permission {name}={level!r}")
    app_jwt = _get_app_auth(PUBLIC_APP).create_jwt()
    try:
        response = httpx.post(
            f"{_GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens",
            json={"permissions": dict(permissions)},
            headers=_github_headers(app_jwt),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        raise GitHubAppMintError(f"Failed to reach GitHub to mint an org token: {e}") from e
    if response.status_code not in (200, 201):
        raise GitHubAppMintError(
            f"GitHub refused an org token with {_scope_label(permissions)} "
            f"(HTTP {response.status_code}).",
            status_code=response.status_code,
        )
    token = (response.json() or {}).get("token") or ""
    if not token:
        raise GitHubAppMintError("GitHub returned an empty org token.")
    logger.info(
        "Public App org token minted (installation_id=%s, scope=%s)",
        installation_id,
        _scope_label(permissions),
    )
    return token
