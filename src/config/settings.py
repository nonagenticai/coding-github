"""Service configuration for coding-github.

Per ``gitops-production/docs/config-externalization-policy.md`` and the
2026-05-09 user directive (memory: ``feedback-no-hardcoded-urls-ports-in-code``):
service addresses MUST come from gitops env. Missing env in production must
crash boot loudly instead of silently routing traffic to a dead in-cluster
service name. Defaults are empty strings; ``_REQUIRED_IN_PRODUCTION`` enforces
presence at startup when ``CODING_GITHUB_ENVIRONMENT=production``.

Mirrors the canonical ``inference-router/src/settings.py`` shape and sibling
PRs coding-gateway#5 / coding-testing#3 / coding-analysis#3.
"""

import json
import re
import uuid
from functools import lru_cache
from typing import ClassVar, Literal

from coding_client.config import BaseMicroserviceSettings
from pydantic import AliasChoices, Field, PrivateAttr

# The caller name ``internal_service_token`` authenticates. That setting predates
# per-caller secrets and has only ever held coding-containers' copy.
LEGACY_INTERNAL_CALLER = "coding-containers"
_CALLER_TOKENS_ENV = "CODING_GITHUB_INTERNAL_CALLER_TOKENS"
_LEGACY_TOKEN_ENV = "CODING_GITHUB_INTERNAL_SERVICE_TOKEN"
_PLATFORM_TENANTS_ENV = "CODING_GITHUB_PLATFORM_TENANTS"
_PLATFORM_REPO_ALLOWLIST_ENV = "CODING_GITHUB_PLATFORM_REPO_ALLOWLIST"
_SANDBOX_OWNERS_ENV = "CODING_GITHUB_SANDBOX_GITHUB_OWNERS"
# A GitHub owner / repository name: letters, digits, ``-``, ``_``, ``.``.
# \Z, not $: `$` also matches before a trailing newline, so "repo\n" would pass.
_GITHUB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\Z")

# fr's five INTERNAL tenants (…0003 is EXECUTION, the conveyor). These — and only
# these — may obtain a credential for a platform-owned (``system_github_owner``)
# repository through the tenant-facing routes.
#
# ⚠️ THIS DEFAULT IS NOT THE SOURCE OF TRUTH, and must not become one again.
# Until 2026-09-22 neither ``CODING_GITHUB_PLATFORM_TENANTS`` nor
# ``CODING_GITHUB_PLATFORM_TENANT_ENFORCE`` was set anywhere in the cluster —
# measured in the running pod, not inferred — so this literal WAS the live
# policy, an invisible cross-repo dependency on a line in another repo's source.
# The list is now DECLARED in gitops-production at
#   clusters/strato/services/coding-github/deployment.yaml
# and the env wins. This default survives only so a local run and the test
# suite have a value; if it ever disagrees with that file, the file is right.
# fr keeps its own copy of the same five (`INTERNAL_TENANT_IDS`), pinned there —
# no test spans the two repos, so the agreement is maintained by these comments
# and by the declaration, not by CI.
DEFAULT_PLATFORM_TENANTS = ",".join(f"0195f48b-d000-7000-8000-00000000000{i}" for i in range(1, 6))


def parse_platform_tenants(raw: str) -> frozenset[str]:
    """Parse ``CODING_GITHUB_PLATFORM_TENANTS`` (comma-separated tenant UUIDs).

    Each entry is normalised to the canonical lowercase UUID form. Raises
    ``RuntimeError`` naming the KEY when the list is empty or any entry is not a
    UUID — an empty allowlist would silently lock the platform's own tenants out
    of its own repos, and a typo would do the same for one of them.
    """
    entries = [part.strip() for part in (raw or "").split(",")]
    entries = [e for e in entries if e]
    if not entries:
        raise RuntimeError(
            f"{_PLATFORM_TENANTS_ENV} is empty; it must list at least one tenant UUID "
            "(the tenants allowed to act on platform-owned repositories)"
        )
    tenants: set[str] = set()
    for entry in entries:
        try:
            tenants.add(str(uuid.UUID(entry)))
        except ValueError:
            raise RuntimeError(
                f"{_PLATFORM_TENANTS_ENV} contains an entry that is not a UUID: {entry!r}"
            ) from None
    return frozenset(tenants)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise RuntimeError(f"{_PLATFORM_REPO_ALLOWLIST_ENV} lists tenant {key!r} twice")
        seen[key] = value
    return seen


def parse_sandbox_github_owners(raw: str, *, platform_owner: str) -> frozenset[str]:
    """Parse ``CODING_GITHUB_SANDBOX_GITHUB_OWNERS`` (comma list of GitHub org names).

    A sandbox owner is an org the PLATFORM owns but that the internal App cannot
    be installed on (it is private to ``system_github_owner``); its repos are
    served with a PUBLIC-App token, gated exactly like platform repos (platform
    tenants + ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST``), never via an fr binding.
    Empty/unset ⇒ no sandbox owners (every such org takes the customer path).

    Returns lowercase names. Raises ``RuntimeError`` (boot fails) on an invalid
    name or when the platform org itself is listed.
    """
    owners: set[str] = set()
    platform_lc = (platform_owner or "").strip().lower()
    for part in (raw or "").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if not _GITHUB_NAME_RE.match(name) or name in (".", ".."):
            raise RuntimeError(f"{_SANDBOX_OWNERS_ENV} has an invalid GitHub owner: {name!r}")
        if name == platform_lc:
            raise RuntimeError(
                f"{_SANDBOX_OWNERS_ENV} lists the platform org {name!r} "
                "(CODING_GITHUB_SYSTEM_GITHUB_OWNER); a sandbox owner must be a different org"
            )
        owners.add(name)
    return frozenset(owners)


READ_ONLY_GRANT_SUFFIX = ":read"

# A NAME-PATTERN grant: ``"<sandbox owner>/*<suffix>"`` (e.g. ``conveyor-coding/*-rs``)
# grants every repository of that sandbox owner whose name ends in ``<suffix>``, so a
# new Python→Rust target (``create_rust_target_repo``) is usable without one gitops
# edit per repo. Deliberately narrow: ONE leading ``*`` on the NAME segment followed
# by a non-empty literal suffix; never on the owner; never a platform-org owner (the
# internal App mints those, and a wildcard there would reach production repos);
# always a FULL (write) grant — ``:read`` patterns are refused.
_NAME_PATTERN_RE = re.compile(r"^\*[A-Za-z0-9_.-]+\Z")


def grant_covers(
    entries: frozenset[str] | set[str] | tuple[str, ...], owner: str, repo: str
) -> bool:
    """True when ``entries`` (canonical lowercase grants) name ``owner/repo``.

    An exact ``owner/repo`` entry, or a name-pattern entry ``owner/*<suffix>`` whose
    owner matches and whose suffix ends a LONGER, valid repository name (so the
    pattern never matches a bare ``-rs`` or ``.``/``..``).
    """
    o = (owner or "").strip().lower()
    r = (repo or "").strip().lower()
    if not o or not r or not _GITHUB_NAME_RE.match(r) or r in (".", ".."):
        return False
    if f"{o}/{r}" in entries:
        return True
    for entry in entries:
        e_owner, _, e_name = entry.partition("/")
        if not e_name.startswith("*") or e_owner != o:
            continue
        suffix = e_name[1:]
        if r.endswith(suffix) and len(r) > len(suffix):
            return True
    return False


def parse_platform_repo_allowlist(
    raw: str,
    *,
    platform_owner: str,
    platform_tenants: frozenset[str],
    sandbox_owners: frozenset[str] = frozenset(),
) -> dict[str, frozenset[str]]:
    """Every granted ``owner/repo`` per tenant, full AND read-only (suffix stripped).

    See :func:`parse_platform_repo_allowlist_modes`, which this wraps; a caller
    that needs to know which grants are read-only must use that instead.
    """
    return parse_platform_repo_allowlist_modes(
        raw,
        platform_owner=platform_owner,
        platform_tenants=platform_tenants,
        sandbox_owners=sandbox_owners,
    )[0]


def _check_name_pattern(
    entry: str,
    name: str,
    tenant: str,
    is_read_only: bool,
    owner_lc: str,
    sandbox_owners: frozenset[str],
) -> None:
    """Boot-fail unless ``name`` is a valid ``<sandbox owner>/*<suffix>`` pattern."""
    where = f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] entry {entry!r}"
    if is_read_only:
        raise RuntimeError(f"{where}: a name pattern is always a write grant; ':read' is refused")
    parts = name.split("/")
    if len(parts) != 2 or not _GITHUB_NAME_RE.match(parts[0]):
        raise RuntimeError(f"{where} is not 'owner/*suffix' (a wildcard owner is refused)")
    if not _NAME_PATTERN_RE.match(parts[1]):
        raise RuntimeError(
            f"{where}: the only accepted pattern is ONE leading '*' on the repository name "
            "followed by a non-empty literal suffix, e.g. 'conveyor-coding/*-rs'"
        )
    owner = parts[0].lower()
    if owner == owner_lc or owner not in sandbox_owners:
        raise RuntimeError(
            f"{where}: a name pattern is accepted only for a sandbox owner "
            f"({_SANDBOX_OWNERS_ENV}), never for the platform org {owner_lc!r}"
        )


def parse_platform_repo_allowlist_modes(
    raw: str,
    *,
    platform_owner: str,
    platform_tenants: frozenset[str],
    sandbox_owners: frozenset[str] = frozenset(),
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Parse ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST`` (JSON ``{tenant: ["owner/repo"]}``).

    An entry is either ``"owner/repo"`` (a FULL grant — clone, push, PR, merge …)
    or ``"owner/repo:read"`` (a READ-ONLY grant: every token minted for it carries
    only ``read`` permissions; scope=push, PR creation, issue/label writes, merge
    and every other write is refused 403 ``platform_repo_read_only``). ``:read`` is
    the only accepted suffix, and one repo may appear at most once per tenant (in
    either form) — ``["o/r", "o/r:read"]`` is ambiguous and fails boot.

    Returns ``(all_grants, read_only_grants)``, both
    ``{canonical tenant uuid: frozenset({"owner/repo" lowercase})}``;
    ``read_only_grants[t]`` is a subset of ``all_grants[t]`` and tenants with no
    read-only entry are absent from it.

    ⛔ The ONLY way a NON-platform tenant may act on a platform-owned repo: each
    listed ``owner/repo`` is served with an INTERNAL-App token minted for that ONE
    repository (``services.tenant_binding.mint_allowlisted_platform_token``) —
    never the org-wide installation token, never a PAT. Empty/unset ⇒ no grants.

    Returns ``{canonical tenant uuid: frozenset({"owner/repo" lowercase})}``.
    Raises ``RuntimeError`` naming the KEY (boot fails) when:

    * the value is not valid JSON, or not an object of string lists;
    * a tenant key is not a UUID, or is listed twice (after normalisation);
    * a tenant is ALSO a platform tenant (it already has everything — a grant
      here would be a misconfiguration nobody could read correctly);
    * an entry is not exactly ``owner/repo``, or its owner is neither the
      platform org (``system_github_owner``) nor a configured sandbox owner
      (``CODING_GITHUB_SANDBOX_GITHUB_OWNERS``); compared case-insensitively. A
      customer repo is tenant-bound via the public App and must never appear here.
      A sandbox-owned entry is served with a one-repo PUBLIC-App token;
    * a NAME PATTERN (``owner/*suffix``) is malformed, carries ``:read``, or is
      not owned by a sandbox owner (see :func:`grant_covers`).
    """
    if not (raw or "").strip():
        return {}, {}
    owner_lc = (platform_owner or "").strip().lower()
    if not owner_lc:
        raise RuntimeError(
            f"{_PLATFORM_REPO_ALLOWLIST_ENV} is set but CODING_GITHUB_SYSTEM_GITHUB_OWNER is "
            "empty; a platform repo grant needs a platform org"
        )
    try:
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ValueError:
        raise RuntimeError(
            f"{_PLATFORM_REPO_ALLOWLIST_ENV} is not valid JSON; expected an object "
            '{"<tenant uuid>": ["owner/repo", ...], ...}'
        ) from None
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{_PLATFORM_REPO_ALLOWLIST_ENV} must be a JSON object {{tenant: [owner/repo]}}, "
            f"got {type(parsed).__name__}"
        )
    grants: dict[str, frozenset[str]] = {}
    read_only: dict[str, frozenset[str]] = {}
    for key, repos in parsed.items():
        try:
            tenant = str(uuid.UUID(str(key).strip()))
        except ValueError:
            raise RuntimeError(
                f"{_PLATFORM_REPO_ALLOWLIST_ENV} has a key that is not a tenant UUID: {key!r}"
            ) from None
        if tenant in grants:
            raise RuntimeError(f"{_PLATFORM_REPO_ALLOWLIST_ENV} lists tenant {tenant!r} twice")
        if tenant in platform_tenants:
            raise RuntimeError(
                f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] is a platform tenant "
                f"({_PLATFORM_TENANTS_ENV}); it must not also carry a repo allowlist"
            )
        if not isinstance(repos, list):
            raise RuntimeError(
                f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] must be a list of 'owner/repo', "
                f"got {type(repos).__name__}"
            )
        entries: set[str] = set()
        read_entries: set[str] = set()
        for entry in repos:
            if not isinstance(entry, str):
                raise RuntimeError(
                    f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] has a non-string entry "
                    f"({type(entry).__name__})"
                )
            is_read_only = entry.endswith(READ_ONLY_GRANT_SUFFIX)
            name = entry[: -len(READ_ONLY_GRANT_SUFFIX)] if is_read_only else entry
            if ":" in name:
                raise RuntimeError(
                    f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] entry {entry!r} has an unknown "
                    f"access suffix; the only one accepted is {READ_ONLY_GRANT_SUFFIX!r}"
                )
            if "*" in name:
                _check_name_pattern(entry, name, tenant, is_read_only, owner_lc, sandbox_owners)
                canonical = name.lower()
                if canonical in entries:
                    raise RuntimeError(
                        f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] lists {canonical!r} more "
                        "than once"
                    )
                entries.add(canonical)
                continue
            parts = name.split("/")
            if (
                len(parts) != 2
                or not all(_GITHUB_NAME_RE.match(p) for p in parts)
                or parts[1] in (".", "..")
            ):
                raise RuntimeError(
                    f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] entry {entry!r} is not "
                    "'owner/repo'"
                )
            if parts[0].lower() != owner_lc and parts[0].lower() not in sandbox_owners:
                raise RuntimeError(
                    f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] entry {entry!r} is not owned "
                    f"by the platform org {owner_lc!r} or a sandbox owner "
                    f"({_SANDBOX_OWNERS_ENV})"
                )
            canonical = f"{parts[0].lower()}/{parts[1].lower()}"
            if canonical in entries:
                raise RuntimeError(
                    f"{_PLATFORM_REPO_ALLOWLIST_ENV}[{tenant!r}] lists {canonical!r} more than "
                    "once (a repo is either a full grant or ':read', never both)"
                )
            entries.add(canonical)
            if is_read_only:
                read_entries.add(canonical)
        grants[tenant] = frozenset(entries)
        if read_entries:
            read_only[tenant] = frozenset(read_entries)
    return grants, read_only


def parse_internal_caller_tokens(raw: str) -> dict[str, str]:
    """Parse ``CODING_GITHUB_INTERNAL_CALLER_TOKENS`` (a JSON ``{caller: token}``).

    Raises ``RuntimeError`` naming the offending KEY — never a token value, and
    never the raw input (which is made of tokens). Empty/blank ⇒ ``{}``.
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        # Deliberately no ``from e``: the decode error quotes the input document.
        raise RuntimeError(
            f"{_CALLER_TOKENS_ENV} is not valid JSON; expected an object "
            '{"<caller>": "<token>", ...}'
        ) from None
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{_CALLER_TOKENS_ENV} must be a JSON object {{caller: token}}, "
            f"got {type(parsed).__name__}"
        )
    callers: dict[str, str] = {}
    seen: dict[str, str] = {}
    for caller, token in parsed.items():
        if not caller.strip():
            raise RuntimeError(f"{_CALLER_TOKENS_ENV} has an empty caller name")
        if not isinstance(token, str):
            raise RuntimeError(
                f"{_CALLER_TOKENS_ENV}[{caller!r}] must be a string, got {type(token).__name__}"
            )
        if not token.strip():
            raise RuntimeError(f"{_CALLER_TOKENS_ENV}[{caller!r}] is an empty token")
        if token in seen:
            raise RuntimeError(
                f"{_CALLER_TOKENS_ENV}: callers {seen[token]!r} and {caller!r} share the "
                "same token — every caller must hold its own secret"
            )
        seen[token] = caller
        callers[caller] = token
    return callers


WebhookMode = Literal["outcome_only", "all"]


def resolve_webhook_mode(raw: str | None) -> WebhookMode:
    """Map a raw CODING_GITHUB_WEBHOOK_MODE value to a mode.

    Only the exact string ``"all"`` selects "all"; unset, empty or ANY other
    value (typos, wrong case, stray whitespace) fails safe to "outcome_only".
    """
    return "all" if raw == "all" else "outcome_only"


class Settings(BaseMicroserviceSettings):
    service_name: str = "coding-github"
    environment: str = "production"
    debug: bool = False

    # Database — sourced from gitops env (coding-github Deployment env block).
    database_url: str = ""

    # GitHub — credentials, not internal-service URLs. ⛔ There is NO personal
    # access token here, by owner ruling (2026-09-30): every GitHub call is made
    # as the internal App (platform repos, platform tenants only) or the public
    # App via the fr tenant binding. A tenant's own BYO token is not a platform
    # PAT and is resolved per request, never from settings.
    github_app_id: str = ""
    github_app_installation_id: str = ""
    github_app_private_key: str = ""
    # PEM source: prefer a mounted file (mirrors arc-runners github-app-credentials);
    # ``github_app_private_key`` (inline) is the dev/test fallback.
    github_app_private_key_path: str = ""
    github_webhook_secret: str = ""

    # PUBLIC GitHub App — the ONLY identity that may mint a token for a repo
    # outside ``system_github_owner`` (a customer repo). Customers install it
    # through fr; fr records which tenant owns each installation. A token is
    # issued only when fr says the installation is bound to the REQUESTING
    # tenant and GitHub confirms it covers the repo (see
    # ``services.tenant_binding.resolve_tenant_repo_token``). Unset ⇒ every
    # customer-repo request fails closed.
    public_app_id: str = ""
    public_app_private_key: str = ""
    public_app_private_key_path: str = ""
    # fr's internal binding lookup (``GET {fr_url}/api/internal/github-app/binding``)
    # and coding-github's own caller secret for it (``X-Internal-Token``).
    fr_url: str = ""
    fr_binding_token: str = ""

    # The GitHub owner whose repositories are PLATFORM repos (internal App).
    system_github_owner: str = "nonagenticai"
    # ⛔ The tenants allowed to act on a ``system_github_owner`` repo via the
    # tenant-facing (``require_tenant``) routes. Any other tenant is refused with
    # 403 ``platform_repo_forbidden`` — without this, every licensed tenant could
    # drive the INTERNAL App against the platform's own repositories. Parsed at
    # boot (``model_post_init``); empty/invalid crashes startup.
    platform_tenants: str = DEFAULT_PLATFORM_TENANTS
    _platform_tenant_ids: frozenset[str] = PrivateAttr(default_factory=frozenset)
    # OBSERVE by default. False ⇒ a non-platform tenant on a platform repo is only
    # LOGGED (``platform_repo_tenant_gate decision=would_refuse``) and the old
    # behaviour proceeds; True ⇒ it is refused (403 / clone-token reason
    # ``platform_repo_forbidden``). Rollout: run observe in prod, grep
    # ``platform_repo_tenant_gate decision=would_refuse`` over ≥24h of conveyor
    # traffic, and only when it is empty (or every hit is a real outsider) flip
    # CODING_GITHUB_PLATFORM_TENANT_ENFORCE=true in gitops. The live tenant comes
    # from licence introspection / on-behalf-of assertion for coding-gateway,
    # coding-containers and fr, so the allowlist is unproven until observed.
    platform_tenant_enforce: bool = False
    # ⛔ Per-tenant exceptions to the gate above, for NON-platform tenants only:
    # JSON ``{"<tenant uuid>": ["<system_github_owner>/<repo>", ...]}``; an entry
    # suffixed ``:read`` is a READ-ONLY grant (read permissions only, every write
    # refused 403 ``platform_repo_read_only``). A listed
    # repo is served ONLY with an internal-App token minted for that one repo
    # (``tenant_binding.mint_allowlisted_platform_token``); every other platform
    # repo stays ``platform_repo_forbidden``. Empty (default) ⇒ no grants.
    # Parsed at boot; malformed / non-UUID / foreign owner / platform tenant
    # crashes startup (``parse_platform_repo_allowlist``).
    platform_repo_allowlist: str = ""
    # ⛔ Platform-owned SANDBOX orgs (comma list, e.g. ``conveyor-coding``). The
    # internal App is private to ``system_github_owner`` and cannot be installed
    # there, so a sandbox repo is served with a one-repo PUBLIC-App token — but
    # gated like a platform repo (platform tenants + the allowlist above), with
    # NO fr binding lookup. Empty (default) ⇒ inert: such an org takes the
    # customer path. Must not name ``system_github_owner`` (boot fails).
    sandbox_github_owners: str = ""
    _sandbox_owners: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _platform_repo_grants: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)
    # The subset of the grants above declared ``owner/repo:read`` (read-only).
    _platform_repo_read_only_grants: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)

    # Service URLs (inter-service communication).
    # Real values come from the coding-github Deployment env block in
    # gitops-production. Empty defaults force production boot to fail loudly
    # if any are missing.
    coding_gateway_url: str = ""
    coding_containers_url: str = ""
    coding_security_url: str = ""
    coding_analysis_url: str = ""

    # Internal service-to-service shared secret.
    # Guards /api/internal/* — endpoints that vend credentials to sibling
    # coding-* pods (e.g. the scoped clone token coding-containers needs to
    # `git clone` a private repo). These CANNOT be guarded by require_tenant():
    # coding-client's tenant/auth resolution accepts a bare X-Tenant-ID header
    # and additionally falls open for any caller inside TRUSTED_CIDRS (which
    # includes 10.0.0.0/8 ⊃ the pod + svc CIDRs), so every in-cluster caller
    # would be "authenticated". A shared secret is the actual gate.
    # Empty (the default) ⇒ fail-closed: every /api/internal request is 403.
    #
    # ``internal_service_token`` authenticates caller ``coding-containers`` (it
    # predates per-caller secrets). Every OTHER caller gets its own secret via
    # ``internal_caller_tokens`` — a JSON object ``{"<caller>": "<token>"}`` —
    # so no two services ever share a credential. Parsed and validated at boot
    # (see ``model_post_init``); a malformed value crashes startup.
    internal_service_token: str = ""
    internal_caller_tokens: str = ""
    _internal_callers: dict[str, str] = PrivateAttr(default_factory=dict)

    # Secrets API
    secrets_api_url: str = ""
    ca_cert_path: str = "/certs/ca-cert.pem"

    # Outcome-event forwarding (George's ruling: no timer may write on a clock).
    # Replaces the n8n workflow "AA Observability - GitHub Outcome Poll", a
    # 30-minute timer that scraped GitHub for PR-merged / PR-closed-without-merge
    # / post-merge CI results. The webhook receiver (routers/github_webhooks.py)
    # now PUSHES those three outcomes to n8n's `aa-github-outcome` webhook the
    # moment a verified GitHub delivery reports them (services/outcome_forward.py).
    # EITHER field empty (both default empty) disables forwarding entirely --
    # logged once, never a hard failure, because an unconfigured forward target
    # must not turn a verified GitHub delivery into anything but a 2xx for GitHub.
    # n8n's `aa-github-outcome` trigger validates `Authorization: Bearer <token>`
    # against its own AA_GITHUB_OUTCOME_WEBHOOK_TOKEN, so a URL configured with no
    # token would be forwarded and refused by n8n on every delivery -- treated
    # identically to "not configured" here rather than attempted and dropped.
    outcome_forward_url: str = ""
    outcome_forward_token: str = ""

    # Which webhook events the receiver acts on (CODING_GITHUB_WEBHOOK_MODE).
    # Code default "outcome_only": after HMAC verification only PR-closed and
    # workflow_run-completed are forwarded to n8n; every other event is ignored
    # so dormant handlers (PR validation, push analysis, comments) are never
    # woken. "all" restores the legacy full dispatch and must be set explicitly.
    # Always read through ``resolve_webhook_mode`` -- unknown values mean
    # outcome_only, never "all".
    webhook_mode: str = "outcome_only"

    # ------------------------------------------------------------------
    # CI-checks wait (``wait_for_pr_checks``) — durable store + completion callback.
    #
    # The store is this service's OWN Postgres database (``coding_github``, one
    # service / one DB / one user) reached through PgBouncer in transaction mode,
    # exactly like coding-containers' exec-job store. ⛔ It deliberately does NOT
    # fall back to ``database_url``: that DSN names the shared ``coding`` database
    # (another service's), and the wait table must never be created there. Unset =
    # the wait routes answer 503 and no sweeper runs; nothing else changes.
    # ------------------------------------------------------------------
    pgbouncer_url: str = Field(
        default="",
        validation_alias=AliasChoices("PGBOUNCER_URL", "CODING_GITHUB_PGBOUNCER_URL"),
        description="Pooled DSN of the coding_github DB (CI-checks wait store)",
    )
    # Comma-separated URL prefixes a wait's ``callback_url`` must sit under (e.g. an
    # n8n Wait-node resume base ``http://host:5678/webhook-waiting/``). EMPTY = every
    # callback_url is refused (fail closed): this pod POSTs it from inside the
    # cluster, so an open value is an SSRF primitive. Same semantics as
    # coding-containers EXEC_CALLBACK_URL_PREFIXES.
    ci_callback_url_prefixes: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CI_CALLBACK_URL_PREFIXES", "CODING_GITHUB_CI_CALLBACK_URL_PREFIXES"
        ),
        description="Allowed callback_url prefixes for CI-checks waits (empty = off)",
    )
    # Database sweep only (due callback retries, due evaluation retries) — it never
    # polls GitHub for CI state; a wait is evaluated on registration, on a verified
    # webhook delivery and at startup.
    ci_wait_sweep_interval_seconds: int = Field(default=20, ge=1, le=600)
    # Lost-webhook backstop: ANY verified delivery (any repo) also re-evaluates open
    # waits not evaluated for this long. Event-driven, not a timer.
    ci_wait_stale_after_seconds: int = Field(default=600, ge=30, le=86400)

    # ``extra="ignore"``: the retired PAT env vars (the removed ``github_token`` /
    # system-token / token-pool fields) may still be set in the pod or a local
    # .env until gitops drops them; they must not crash boot.
    model_config = {"env_prefix": "CODING_GITHUB_", "env_file": ".env", "extra": "ignore"}

    # Required-in-production env vars; enforced by the shared base
    # ``model_post_init``. Missing any of these when ``environment=production``
    # raises a single combined RuntimeError naming each missing field.
    _SERVICE_NAME: ClassVar[str] = "coding-github"
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "database_url",
        "coding_gateway_url",
        "coding_containers_url",
        "coding_security_url",
        "coding_analysis_url",
        "secrets_api_url",
    )

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        callers = parse_internal_caller_tokens(self.internal_caller_tokens)
        legacy = self.internal_service_token
        if legacy:
            if LEGACY_INTERNAL_CALLER in callers:
                raise RuntimeError(
                    f"{_CALLER_TOKENS_ENV}[{LEGACY_INTERNAL_CALLER!r}] conflicts with "
                    f"{_LEGACY_TOKEN_ENV}, which already authenticates that caller — set one"
                )
            for caller, token in callers.items():
                if token == legacy:
                    raise RuntimeError(
                        f"{_CALLER_TOKENS_ENV}[{caller!r}] reuses {_LEGACY_TOKEN_ENV} "
                        f"({LEGACY_INTERNAL_CALLER}'s secret) — every caller must hold its own"
                    )
            callers = {LEGACY_INTERNAL_CALLER: legacy, **callers}
        self._internal_callers = callers
        self._platform_tenant_ids = parse_platform_tenants(self.platform_tenants)
        self._sandbox_owners = parse_sandbox_github_owners(
            self.sandbox_github_owners, platform_owner=self.system_github_owner
        )
        self._platform_repo_grants, self._platform_repo_read_only_grants = (
            parse_platform_repo_allowlist_modes(
                self.platform_repo_allowlist,
                platform_owner=self.system_github_owner,
                platform_tenants=self._platform_tenant_ids,
                sandbox_owners=self._sandbox_owners,
            )
        )

    @property
    def ci_callback_prefixes(self) -> tuple[str, ...]:
        """The allowed CI-wait callback URL prefixes; empty means callbacks are off."""
        return tuple(p.strip() for p in self.ci_callback_url_prefixes.split(",") if p.strip())

    @property
    def ci_wait_dsn(self) -> str:
        """The CI-wait store DSN, or "" when no store is configured."""
        return self.pgbouncer_url

    @property
    def platform_tenant_ids(self) -> frozenset[str]:
        """Normalised (lowercase UUID) tenants allowed on platform-owned repos."""
        return self._platform_tenant_ids

    @property
    def sandbox_owners(self) -> frozenset[str]:
        """Lowercase platform-owned sandbox orgs (public-App, platform-gated)."""
        return self._sandbox_owners

    @property
    def platform_repo_grants(self) -> dict[str, frozenset[str]]:
        """``{tenant uuid: {"owner/repo" lowercase}}`` from the repo allowlist."""
        return dict(self._platform_repo_grants)

    @property
    def platform_repo_read_only_grants(self) -> dict[str, frozenset[str]]:
        """``{tenant uuid: {"owner/repo" lowercase}}`` of the ``:read`` (read-only) grants."""
        return dict(self._platform_repo_read_only_grants)

    @property
    def internal_callers(self) -> dict[str, str]:
        """Every caller accepted on ``/api/internal``, as ``{caller: token}``."""
        return dict(self._internal_callers)

    # ------------------------------------------------------------------
    # Environment predicates — the single resolution point.
    #
    # ``environment`` is configured as CODING_GITHUB_ENVIRONMENT (env_prefix
    # above). Reading the bare ``ENVIRONMENT`` name off os.environ instead —
    # which four call sites used to do independently — returns "" in
    # production, because the Deployment sets no such variable. Route every
    # environment decision through these two properties so a fifth site
    # cannot drift; ``tests/unit/test_environment_setting_resolution.py``
    # fails the build if a bare read reappears.
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """True when this service is running as production."""
        return self.environment.strip().lower() == "production"

    @property
    def is_dev_or_test(self) -> bool:
        """True only for environments where security bypasses are legitimate.

        Deliberately excludes ``staging``: staging is neither production nor a
        place where validation bypasses may be unlocked.
        """
        return self.environment.strip().lower() in ("development", "dev", "testing", "test")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_feature_flags() -> dict[str, bool]:
    """Load feature flags from environment.

    This is a simplified version for the coding-github microservice.
    In the monolith, this loaded from a YAML file. Here we use env vars.
    """
    import os

    flags: dict[str, bool] = {}
    prefix = "CODING_GITHUB_FEATURE_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            flag_name = key[len(prefix) :].lower()
            flags[flag_name] = value.lower() in ("true", "1", "yes", "on")
    return flags
