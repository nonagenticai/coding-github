"""Create the WRITABLE Rust target repo of a Python→Rust rewrite (``create_rust_target_repo``).

Port of gitops-production ``scripts/new-rust-target.sh`` (conveyor-coding branch only),
which needed George's personal ``gh`` login. George 2026-11-03: this is a coding API
op, authenticated by the PUBLIC App (``nonagenticai``, id 5029921) installed on the
sandbox org ``conveyor-coding``. That App must hold **Administration: write** (create
the repo) and **Workflows: write** (the initial commit carries ``.github/workflows/``);
a refusal names exactly which one is missing.

Steps, in an order that keeps the window between "repo exists" and "repo usable" to
ONE push:

1. validate (422): owner ``conveyor-coding`` only, lowercase-kebab name ending ``-rs``,
   every source ``owner/repo`` under ``nonagenticai``/``conveyor-coding``, not the target;
2. gate the TENANT (403): a platform tenant, or a write grant covering the target in
   ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST`` (exact, or the ``conveyor-coding/*-rs``
   name pattern). Fails closed regardless of ``PLATFORM_TENANT_ENFORCE`` — a new op
   has no previous behaviour to observe;
3. gate every SOURCE (403 ``source_not_granted``): the tenant must hold a read or write
   grant for it. Steps 1-3 make no network call, so an ungranted name reveals nothing;
4. the public App's installation ON THE ORG must be the sandbox org's (409 otherwise);
5. the target must not exist (409) — checked BEFORE any source is read;
6. every source must EXIST and be PRIVATE, owned by the owner named (422, ONE uniform
   ``source_not_usable`` answer);
7. clone ``nonagenticai/template-rust-service`` (internal App, contents:read, one repo),
   run ``scripts/rename.sh <name> --self-contained-ci``, collapse history to ONE commit
   ``Initial commit from template-rust-service@<sha>``;
8. ``POST /orgs/{org}/repos`` (PRIVATE, empty), then push that commit to ``main`` with a
   token restricted to the new repo (contents+workflows write).

Every installation token the op mints (org admin, source reads, template read, push) is
minted uncached and revoked (``DELETE /installation/token``) on every exit, and the
whole op runs inside a ``_TOTAL_BUDGET_SECONDS`` deadline (504 past it) so
the caller's 150 s client timeout never fires first.

A failure after step 8's create leaves an EMPTY private repo; the error says so and a
retry answers 409 (by design — the op never deletes a repository).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings as _settings_module
from services import github_app_auth as _ga
from services import tenant_binding as _tb
from services.github_app_auth import (
    GitHubAppAuthError,
    GitHubAppMintError,
    GitHubAppNotInstalledError,
    GitHubAppOwnerMismatchError,
)
from utils.credential_redaction import redact_credentials

logger = logging.getLogger(__name__)

TARGET_OWNER = "conveyor-coding"
SOURCE_OWNERS = frozenset({"nonagenticai", "conveyor-coding"})
TEMPLATE_OWNER = "nonagenticai"
TEMPLATE_REPO = "template-rust-service"
DEFAULT_BRANCH = "main"
TARGET_SUFFIX = "-rs"

#: The org-wide token only creates the repository (and reads metadata).
PERMS_CREATE: dict[str, str] = {"administration": "write"}
#: The push token, restricted to the new repository.
PERMS_PUSH: dict[str, str] = {"contents": "write", "workflows": "write"}

# Mirrors the template's rename.sh: lowercase kebab case (crate + image + k8s name).
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]\Z")
_SLUG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+)\Z")

GIT_BASE_URL = "https://github.com"
GITHUB_API_BASE = "https://api.github.com"
# ⏱ TOTAL budget. The n8n node (and coding-gateway's x-request-timeout-ms) gives a
# POST 150 s; this op must answer within 120 s. Every step this module controls is
# bounded by what is LEFT of _TOTAL_BUDGET_SECONDS, capped per step, and the deadline
# is checked BEFORE each App call in services.github_app_auth, which is not
# deadline-aware. So at most ONE App call can run past the deadline. The longest is
# get_scoped_clone_token = installation lookup (PyGithub, 15 s) + mint (httpx, 10 s)
# = 25 s. Then the revoke phase: up to 23 tokens (org + 20 sources + template + push),
# revoked CONCURRENTLY and joined within _REVOKE_JOIN_SECONDS (= the 5 s per-call
# timeout + 1 s join slack; a sequential loop would be 23 × 5 s = 115 s on its own).
# 85 + 25 + 6 = 116 ≤ 120 s.
# tests/unit/test_create_rust_target_repo.py drives the op on a fake clock with 20
# sources and every call taking its full timeout, and asserts that bound. The
# template is ~35 KB, so these caps are generous for every real run.
_TOTAL_BUDGET_SECONDS = 85.0
_HTTP_STEP_CAP_SECONDS = 10.0
_GIT_NETWORK_CAP_SECONDS = 40.0  # clone, push
_GIT_LOCAL_CAP_SECONDS = 15.0  # rename.sh, add, commit, rev-parse
_REVOKE_TIMEOUT_SECONDS = 5.0
_REVOKE_JOIN_SECONDS = _REVOKE_TIMEOUT_SECONDS + 1.0  # whole revoke phase
_BOT_NAME = "nonagenticai[bot]"
_BOT_EMAIL = "nonagenticai[bot]@users.noreply.github.com"
# Env passed through to git/bash; everything else (GIT_DIR, a stray credential
# helper config, …) is dropped.
_PASSTHROUGH_ENV = ("PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_SSL_CAINFO", "LANG")
# GitHub's refusal when a token without workflows:write pushes .github/workflows/.
_WORKFLOW_REFUSAL_RE = re.compile(r"without `?workflows?`? (scope|permission)", re.IGNORECASE)


class RustTargetError(Exception):
    """A refusal with an HTTP status and a machine ``error_code``."""

    def __init__(self, status: int, code: str, message: str, **details: Any):
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = details


class _Deadline:
    """What is left of the op's total budget; refuses 504 once it is spent."""

    def __init__(self, seconds: float):
        self._end = time.monotonic() + seconds

    def left(self, cap: float) -> float:
        remaining = self._end - time.monotonic()
        if remaining <= 0:
            raise RustTargetError(
                504,
                "deadline_exceeded",
                f"create_rust_target_repo exceeded its {_TOTAL_BUDGET_SECONDS:.0f}s budget",
            )
        return min(cap, remaining)

    def check(self) -> None:
        """Refuse 504 if the budget is spent (before a step that is not deadline-aware)."""
        self.left(float("inf"))


@dataclass(frozen=True)
class RustTargetResult:
    target: str
    url: str
    default_branch: str
    head_sha: str
    template_sha: str


#: GitHub's settings-page label for each permission this op needs. ⚠️ REPOSITORY
#: Administration, not "Organization permissions → Administration": the org-level one
#: does NOT authorise ``POST /orgs/{org}/repos``.
_PERMISSION_LABELS = {
    "administration": "Repository permissions → Administration: write",
    "workflows": "Repository permissions → Workflows: write",
}


def _missing_permission(permission: str, what: str) -> RustTargetError:
    label = _PERMISSION_LABELS[permission]
    return RustTargetError(
        403,
        "github_app_permission_missing",
        f"GitHub refused {what}: the public GitHub App (id 5029921) installation on "
        f"'{TARGET_OWNER}' lacks \"{label}\". Grant it on the App and accept it on the "
        f"{TARGET_OWNER} installation.",
        missing_permission=permission,
    )


def validate(target: str, sources: list[str]) -> tuple[str, str, list[tuple[str, str]]]:
    """Return ``(owner, name, [(source_owner, source_repo)])`` or raise 422."""
    m = _SLUG_RE.match((target or "").strip())
    if not m:
        raise RustTargetError(422, "invalid_target", f"target {target!r} is not 'owner/name'")
    owner, name = m.group(1), m.group(2)
    if owner.lower() != TARGET_OWNER:
        raise RustTargetError(
            422,
            "target_owner_not_allowed",
            f"target owner {owner!r} is refused: Rust targets are created in "
            f"{TARGET_OWNER!r} only (nonagenticai targets need registry secrets this op "
            "does not set)",
        )
    if not name.endswith(TARGET_SUFFIX):
        raise RustTargetError(
            422, "target_name_not_rs", f"target name {name!r} does not end in {TARGET_SUFFIX!r}"
        )
    if not _NAME_RE.match(name):
        raise RustTargetError(
            422,
            "target_name_invalid",
            f"target name {name!r} is not lowercase kebab case ([a-z][a-z0-9-]*[a-z0-9])",
        )
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        sm = _SLUG_RE.match((source or "").strip())
        if not sm:
            raise RustTargetError(422, "invalid_source", f"source {source!r} is not 'owner/repo'")
        s_owner, s_repo = sm.group(1), sm.group(2)
        if s_repo in (".", ".."):
            raise RustTargetError(422, "invalid_source", f"source {source!r} is not 'owner/repo'")
        if s_owner.lower() not in SOURCE_OWNERS:
            raise RustTargetError(
                422,
                "source_owner_not_allowed",
                f"source owner {s_owner!r} is not one of {sorted(SOURCE_OWNERS)}",
            )
        key = f"{s_owner.lower()}/{s_repo.lower()}"
        if key == f"{TARGET_OWNER}/{name}":
            raise RustTargetError(422, "source_is_target", f"source {source!r} is the target")
        if key in seen:
            continue
        seen.add(key)
        parsed.append((s_owner, s_repo))
    return TARGET_OWNER, name, parsed


def gate_tenant(tenant_id: str | None, owner: str, name: str) -> None:
    """403 unless the tenant may WRITE ``owner/name`` (platform tenant or write grant)."""
    grant = _tb.sandbox_repo_grant(tenant_id, owner, name)
    if grant == "refuse":
        logger.warning(
            "%s decision=refused tenant=%s owner=%s repo=%s caller_path=create_rust_target_repo",
            _tb.GATE_LOG_TOKEN,
            tenant_id or "none",
            owner,
            name,
        )
        raise RustTargetError(
            403,
            "platform_repo_forbidden",
            f"tenant may not create '{owner}/{name}': it is neither a platform tenant nor "
            "granted the target in CODING_GITHUB_PLATFORM_REPO_ALLOWLIST "
            f"(e.g. '{owner}/*{TARGET_SUFFIX}')",
        )
    if grant == "scoped" and _tb.is_read_only_grant(tenant_id, owner, name):
        raise RustTargetError(
            403,
            "platform_repo_read_only",
            f"'{owner}/{name}' is granted READ-ONLY to this tenant; creating it is a write",
        )


def gate_sources(tenant_id: str | None, sources: list[tuple[str, str]]) -> None:
    """403 unless the tenant holds a grant (read or write) for EVERY source.

    Checked before any GitHub call, so an ungranted name costs no network round trip
    and its answer cannot depend on whether the repo exists. A platform tenant holds
    every platform/sandbox repo; anyone else needs a ``CODING_GITHUB_PLATFORM_REPO_ALLOWLIST``
    entry — which a source needs anyway (``owner/repo:read``) for the workspace to clone it.
    """
    if _tb.is_platform_tenant(tenant_id):
        return
    tenant = _tb._normalise_tenant(tenant_id)
    grants = getattr(_settings_module.get_settings(), "platform_repo_grants", None) or {}
    entries = (grants.get(tenant) or frozenset()) if tenant else frozenset()
    for s_owner, s_repo in sources:
        if not _settings_module.grant_covers(entries, s_owner, s_repo):
            raise RustTargetError(
                403,
                "source_not_granted",
                f"source {s_owner}/{s_repo} is not granted to this tenant in "
                f"CODING_GITHUB_PLATFORM_REPO_ALLOWLIST (add '{s_owner}/{s_repo}:read')",
            )


def _http_client() -> httpx.Client:
    """The GitHub REST client (tests swap this for an ``httpx.MockTransport`` one).

    Each request passes its own deadline-bounded ``timeout``; this is only the default.
    """
    return httpx.Client(base_url=GITHUB_API_BASE, timeout=_HTTP_STEP_CAP_SECONDS)


def _auth(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }


def _source_unusable(owner: str, repo: str) -> RustTargetError:
    """ONE answer for missing / not visible / not private / owned elsewhere.

    ⛔ Uniform on purpose: a tenant may name any ``nonagenticai/*`` source, so
    distinct answers would let it probe whether a PRIVATE platform repo exists.
    """
    return RustTargetError(
        422,
        "source_not_usable",
        f"source {owner}/{repo} is not an existing PRIVATE repository the platform can read",
    )


def _require_private_source(
    client: httpx.Client, owner: str, repo: str, token: str, deadline: _Deadline
) -> None:
    resp = client.get(
        f"/repos/{owner}/{repo}",
        headers=_auth(token),
        timeout=deadline.left(_HTTP_STEP_CAP_SECONDS),
    )
    if resp.status_code == 404:
        raise _source_unusable(owner, repo)
    if resp.status_code != 200:
        raise RustTargetError(
            502,
            "github_error",
            f"GitHub answered HTTP {resp.status_code} reading source {owner}/{repo}",
        )
    body = resp.json() or {}
    actual_owner = str((body.get("owner") or {}).get("login") or "")
    if actual_owner.lower() != owner.lower() or body.get("private") is not True:
        raise _source_unusable(owner, repo)


def _source_token(owner: str, repo: str, org_token: str) -> str:
    """A credential that can read ``owner/repo``'s metadata."""
    if owner.lower() == TARGET_OWNER:
        return org_token
    try:
        return _ga.get_scoped_clone_token(owner, repo, "clone", fresh=True).token
    except GitHubAppNotInstalledError as e:
        raise _source_unusable(owner, repo) from e
    except GitHubAppAuthError as e:
        raise RustTargetError(
            502, "github_app_mint_failed", f"cannot read source {owner}/{repo}: {e}"
        ) from e


def _git_env(workdir: str, token: str | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update(
        HOME=workdir,
        TMPDIR=workdir,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME=_BOT_NAME,
        GIT_AUTHOR_EMAIL=_BOT_EMAIL,
        GIT_COMMITTER_NAME=_BOT_NAME,
        GIT_COMMITTER_EMAIL=_BOT_EMAIL,
    )
    if token:
        # Via env (GIT_CONFIG_*), never argv (visible in ps) and never the URL
        # (git echoes a URL's userinfo into its own error text).
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0=f"http.{GIT_BASE_URL}/.extraheader",
            GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}",
        )
    return env


def _run(
    args: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    what: str,
    timeout: float,
    secret: str = "",
) -> str:
    """Run ``args`` in its OWN process group; on timeout SIGKILL the whole group.

    ⛔ ``subprocess.run(timeout=)`` kills only the direct child: ``git push`` forks
    ``git-remote-https``, which holds the token in its environment and would outlive
    the request. Killing the group (``start_new_session=True``) takes it down too.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        raise RustTargetError(
            502, "git_failed", f"{what} failed: {redact_credentials(str(e), secret)}"
        ) from None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()  # reap the child and close its pipes
        raise RustTargetError(
            504, "deadline_exceeded", f"{what} did not finish within {timeout:.0f}s"
        ) from None
    proc = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    if proc.returncode != 0:
        output = redact_credentials((proc.stderr or "") + (proc.stdout or ""), secret).strip()
        if _WORKFLOW_REFUSAL_RE.search(output):
            raise _missing_permission("workflows", f"the push of {what}")
        raise RustTargetError(
            502, "git_failed", f"{what} failed (exit {proc.returncode}): {output[-2000:]}"
        )
    return proc.stdout.strip()


def build_initial_commit(
    workdir: str, name: str, template_token: str, deadline: _Deadline | None = None
) -> tuple[str, str, str]:
    """Clone the template, rename it, collapse to one commit. Returns ``(clone, tpl, head)``."""
    deadline = deadline or _Deadline(_TOTAL_BUDGET_SECONDS)
    clone = os.path.join(workdir, name)
    plain = _git_env(workdir)

    def local(args: list[str], what: str) -> str:
        return _run(
            args, cwd=clone, env=plain, what=what, timeout=deadline.left(_GIT_LOCAL_CAP_SECONDS)
        )

    _run(
        [
            "git",
            "clone",
            "--quiet",
            f"{GIT_BASE_URL}/{TEMPLATE_OWNER}/{TEMPLATE_REPO}.git",
            clone,
        ],
        cwd=workdir,
        env=_git_env(workdir, template_token),
        what=f"clone of {TEMPLATE_OWNER}/{TEMPLATE_REPO}",
        timeout=deadline.left(_GIT_NETWORK_CAP_SECONDS),
        secret=template_token,
    )
    template_sha = local(["git", "rev-parse", "HEAD"], "rev-parse")
    local(["bash", "scripts/rename.sh", name, "--self-contained-ci"], "scripts/rename.sh")
    # An orphan branch drops every template commit from what is pushed.
    local(["git", "checkout", "-q", "--orphan", "main-init"], "orphan")
    local(["git", "add", "-A"], "git add")
    local(
        ["git", "commit", "-q", "-m", f"Initial commit from {TEMPLATE_REPO}@{template_sha}"],
        "git commit",
    )
    head_sha = local(["git", "rev-parse", "HEAD"], "rev-parse")
    return clone, template_sha, head_sha


def _org_installation() -> int:
    if not _ga.public_app_configured():
        raise RustTargetError(
            503,
            "github_public_app_not_configured",
            "The public GitHub App is not configured on coding-github; cannot create repos",
        )
    try:
        installation = _ga.resolve_public_org_installation(TARGET_OWNER)
    except GitHubAppNotInstalledError as e:
        raise RustTargetError(
            409,
            "github_app_not_installed",
            f"The public GitHub App is not installed on org '{TARGET_OWNER}'",
        ) from e
    except GitHubAppAuthError as e:
        raise RustTargetError(502, "github_app_mint_failed", str(e)) from e
    if (
        installation.account_login.lower() != TARGET_OWNER
        or installation.account_type != "Organization"
    ):
        raise RustTargetError(
            409,
            "github_app_not_installed",
            f"The public App installation for '{TARGET_OWNER}' belongs to "
            f"'{installation.account_login or '?'}' ({installation.account_type or '?'}); "
            "refused",
        )
    return installation.id


def _mint_org_token(installation_id: int) -> str:
    try:
        return _ga.mint_public_org_token(installation_id, PERMS_CREATE)
    except GitHubAppMintError as e:
        if e.status_code == 422:
            raise _missing_permission("administration", "the org token mint") from e
        raise RustTargetError(502, "github_app_mint_failed", str(e)) from e
    except GitHubAppAuthError as e:
        raise RustTargetError(502, "github_app_mint_failed", str(e)) from e


def _revoke(client: httpx.Client, token: str) -> None:
    """``DELETE /installation/token`` — every token the op minted dies with the op.

    Never raises and never logs the token: a revoke failure only shortens nothing
    (GitHub expires the token within the hour anyway); it must not turn a created
    repository into a failed request.
    """
    try:
        resp = client.delete(
            "/installation/token", headers=_auth(token), timeout=_REVOKE_TIMEOUT_SECONDS
        )
        if resp.status_code != 204:
            logger.warning("token revoke answered HTTP %s (not revoked)", resp.status_code)
    except Exception as e:  # noqa: BLE001 — best effort by contract
        logger.warning("token revoke failed: %s", redact_credentials(str(e), token))


def _revoke_all(client: httpx.Client, tokens: list[str]) -> None:
    """Revoke every token CONCURRENTLY; the phase as a whole waits at most
    ``_REVOKE_TIMEOUT_SECONDS`` (+1 s slack), however many tokens there are.

    One daemon thread per token over the shared (thread-safe) ``httpx.Client``. Each
    ``_revoke`` is itself bounded by ``_REVOKE_TIMEOUT_SECONDS`` and never raises; a
    straggler past the join budget is abandoned (it ends at its own timeout) rather
    than holding up the answer.
    """
    threads = [
        threading.Thread(target=_revoke, args=(client, token), daemon=True)
        for token in reversed(tokens)
    ]
    for thread in threads:
        thread.start()
    budget_end = time.monotonic() + _REVOKE_JOIN_SECONDS
    for thread in threads:
        thread.join(max(0.0, budget_end - time.monotonic()))


def _create_repo(
    client: httpx.Client, owner: str, name: str, token: str, sources: str, deadline: _Deadline
) -> str:
    resp = client.post(
        f"/orgs/{owner}/repos",
        headers=_auth(token),
        json={
            "name": name,
            "private": True,
            "auto_init": False,
            "description": f"Rust rewrite of {sources}"[:350],
        },
        timeout=deadline.left(_HTTP_STEP_CAP_SECONDS),
    )
    if resp.status_code == 201:
        return str((resp.json() or {}).get("html_url") or f"{GIT_BASE_URL}/{owner}/{name}")
    if resp.status_code == 422 and "already exists" in resp.text:
        raise RustTargetError(409, "target_exists", f"target {owner}/{name} already exists")
    if resp.status_code == 403:
        raise _missing_permission("administration", f"POST /orgs/{owner}/repos")
    raise RustTargetError(
        502, "github_error", f"GitHub answered HTTP {resp.status_code} creating {owner}/{name}"
    )


def _mint_push_token(installation_id: int, owner: str, name: str) -> str:
    try:
        return _ga.mint_public_repo_token(
            installation_id, owner, name, dict(PERMS_PUSH), expected_owner=owner, fresh=True
        ).token
    except GitHubAppMintError as e:
        if e.status_code == 422:
            raise _missing_permission("workflows", "the push token mint") from e
        raise RustTargetError(502, "github_app_mint_failed", str(e)) from e
    except (GitHubAppOwnerMismatchError, GitHubAppAuthError) as e:
        raise RustTargetError(502, "github_app_mint_failed", str(e)) from e


def create_rust_target(
    target: str,
    sources: list[str],
    tenant_id: str | None,
    *,
    http_client: Callable[[], httpx.Client] | None = None,
) -> RustTargetResult:
    """Run the whole op synchronously (the route calls it in a worker thread)."""
    deadline = _Deadline(_TOTAL_BUDGET_SECONDS)
    owner, name, parsed_sources = validate(target, sources)
    gate_tenant(tenant_id, owner, name)
    gate_sources(tenant_id, parsed_sources)
    deadline.check()
    installation_id = _org_installation()
    with (http_client or _http_client)() as client:
        deadline.check()
        org_token = _mint_org_token(installation_id)
        # EVERY token this op mints lands here and is revoked on every exit. All are
        # minted uncached (fresh=True), so revoking one cannot break another request.
        minted: list[str] = [org_token]
        try:
            return _create_with(
                client, deadline, installation_id, org_token, owner, name, parsed_sources, minted
            )
        finally:
            _revoke_all(client, minted)


def _create_with(
    client: httpx.Client,
    deadline: _Deadline,
    installation_id: int,
    org_token: str,
    owner: str,
    name: str,
    parsed_sources: list[tuple[str, str]],
    minted: list[str],
) -> RustTargetResult:
    # Target BEFORE sources: an existing target answers 409 without a single source read.
    existing = client.get(
        f"/repos/{owner}/{name}",
        headers=_auth(org_token),
        timeout=deadline.left(_HTTP_STEP_CAP_SECONDS),
    )
    if existing.status_code == 200:
        raise RustTargetError(409, "target_exists", f"target {owner}/{name} already exists")
    if existing.status_code != 404:
        raise RustTargetError(
            502,
            "github_error",
            f"GitHub answered HTTP {existing.status_code} checking {owner}/{name}",
        )
    for s_owner, s_repo in parsed_sources:
        deadline.check()
        token = _source_token(s_owner, s_repo, org_token)
        if token != org_token:
            minted.append(token)
        _require_private_source(client, s_owner, s_repo, token, deadline)
    deadline.check()
    try:
        template_token = _ga.get_scoped_clone_token(TEMPLATE_OWNER, TEMPLATE_REPO, fresh=True).token
        minted.append(template_token)
    except GitHubAppAuthError as e:
        raise RustTargetError(
            502,
            "template_unreadable",
            f"cannot mint a read token for {TEMPLATE_OWNER}/{TEMPLATE_REPO}: {e}",
        ) from e
    with tempfile.TemporaryDirectory(prefix="rust-target-") as workdir:
        clone, template_sha, head_sha = build_initial_commit(
            workdir, name, template_token, deadline
        )
        source_list = ", ".join(f"{o}/{r}" for o, r in parsed_sources)
        url = _create_repo(client, owner, name, org_token, source_list, deadline)
        logger.info("rust target created %s/%s", owner, name)
        try:
            deadline.check()
            push_token = _mint_push_token(installation_id, owner, name)
            minted.append(push_token)
            _run(
                [
                    "git",
                    "push",
                    "--quiet",
                    f"{GIT_BASE_URL}/{owner}/{name}.git",
                    f"HEAD:refs/heads/{DEFAULT_BRANCH}",
                ],
                cwd=clone,
                env=_git_env(workdir, push_token),
                what=f"{owner}/{name} main",
                timeout=deadline.left(_GIT_NETWORK_CAP_SECONDS),
                secret=push_token,
            )
        except RustTargetError as e:
            e.details["created_empty_repo"] = f"{owner}/{name}"
            raise RustTargetError(
                e.status,
                e.code,
                f"{e} — NOTE: {owner}/{name} was CREATED but is EMPTY; a retry answers "
                "409 until it is deleted",
                **e.details,
            ) from e
    return RustTargetResult(
        target=f"{owner}/{name}",
        url=url,
        default_branch=DEFAULT_BRANCH,
        head_sha=head_sha,
        template_sha=template_sha,
    )
