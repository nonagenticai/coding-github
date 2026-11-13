"""GitHub integration service for creating pull requests with bug fixes."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, NamedTuple

from config.settings import get_settings
from services.repo_visibility_guard import RepoVisibilityRefusedError, assert_repo_not_public
from utils.github_error_handler import GitHubErrorHandler, retry_github_call

if TYPE_CHECKING:
    from github import Auth, Github
    from github.GithubException import GithubException
    from github.Repository import Repository

try:
    from github import Auth, Github
    from github.GithubException import GithubException
    from github.Repository import Repository

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    Auth = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    Github = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    GithubException = Exception  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    Repository = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]

FixValidator = None  # Stub for patching in tests

if TYPE_CHECKING:
    from github.AuthenticatedUser import AuthenticatedUser as User
else:
    User = Any  # Runtime fallback

logger = logging.getLogger(__name__)
settings = get_settings()


class GitHubAuthError(Exception):
    """No GitHub token available — fork/apply-changes/create-PR cannot proceed.

    Distinct from RuntimeError so router handlers can map it to HTTP 412
    (Precondition Failed) without colliding with the existing
    `except RuntimeError → not_found(Container)` branch.
    """


class BaseRefNotFoundError(ValueError):
    """The requested base for a new branch resolves to nothing on the repo.

    Raised instead of silently cutting the branch from the default branch: a
    fix branch cut from the wrong commit applies its diff to code the diff was
    never written against, and nothing downstream can tell. Router handlers map
    it to HTTP 400.
    """


class DefaultBranchWriteRefusedError(PermissionError):
    """A commit or ref update was aimed at the repository's default branch.

    Every producer in this service writes a FEATURE branch and opens a PR; none
    has a legitimate reason to commit straight onto ``main``. Before this guard,
    ``apply-changes`` with ``branch_name="main"`` (and coding-security's
    ``commit=true, branch_name="main"``, which calls it) found the ref already
    existing at the base SHA, "reused" it, and committed directly to the default
    branch — no PR, no review, no CI gate. The denylist in n8n guarded only
    prose-origin workers; typed callers were unguarded.

    Always refused — there is deliberately no override flag. A body field would
    be caller-settable (the gateway forwards the body unchanged and n8n's
    generic coding dispatch passes unknown keys through), so any "allow" switch
    is a switch every caller holds. Router handlers map it to HTTP 403
    ``default_branch_write_refused``.
    """

    error_code = "default_branch_write_refused"


# Long-lived names refused even when the repo's reported default is different:
# a ``develop``-default repo's ``main`` is still somebody's release line.
_ALWAYS_PROTECTED_BRANCHES = frozenset({"main", "master"})


def bare_branch_name(branch_name: str) -> str:
    """``owner:branch`` → ``branch``; a bare name is returned unchanged."""
    return branch_name.split(":", 1)[1] if ":" in branch_name else branch_name


def protected_branches(repo: Any) -> set[str]:
    """The branch names a write must not target on ``repo``: its default
    branch (read from the already-fetched repo object, so no API call) plus
    ``main``/``master``.

    The workspace's base is deliberately NOT added: a workspace provisioned on
    a PR's head branch legitimately commits more work onto that same branch.
    """
    names = set(_ALWAYS_PROTECTED_BRANCHES)
    default = getattr(repo, "default_branch", None)
    if isinstance(default, str) and default:
        names.add(default)
    return names


_REF_PREFIXES = ("refs/heads/", "heads/", "refs/", "/")


def normalize_branch_ref(branch_name: str) -> str:
    """Reduce any spelling of a branch ref to its comparable bare name.

    ``" refs/heads/Main "``, ``"heads/main"``, ``"owner:refs/heads/main"`` and
    ``"main"`` all normalise to ``"main"``: whitespace stripped, an
    ``owner:`` qualifier dropped, ``refs/heads/``/``heads/`` prefixes removed
    (repeatedly), and case folded. Used ONLY for the protection comparison —
    over-matching here can only refuse more, never permit more.
    """
    name = bare_branch_name(branch_name.strip()).strip()
    changed = True
    while changed:
        changed = False
        for prefix in _REF_PREFIXES:
            if name.casefold().startswith(prefix):
                name = name[len(prefix) :].strip()
                changed = True
    return name.casefold()


def assert_branch_writable(repo: Any, branch_name: str) -> None:
    """Raise :class:`DefaultBranchWriteRefusedError` for a protected target.

    Unconditional: there is no parameter that bypasses it.
    """
    target = normalize_branch_ref(branch_name)
    protected = protected_branches(repo)
    if target in {normalize_branch_ref(b) for b in protected}:
        full_name = getattr(repo, "full_name", None)
        where = f" on {full_name}" if isinstance(full_name, str) else ""
        raise DefaultBranchWriteRefusedError(
            f"Refusing to write to protected branch '{target}'{where} "
            f"(protected: {', '.join(sorted(protected))}). Write a feature branch "
            "and open a pull request."
        )


# A base that is not a branch head may be a commit (full or abbreviated SHA).
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

# The only bases allowed to fall back to the repo's default branch. They are
# the values callers send when nobody chose a base (fr's `branch` defaults to
# "main"), so on a `master`-default repo the fallback is the caller's intent.
# Any other name, and any SHA, was chosen deliberately and must resolve exactly.
_IMPLICIT_DEFAULT_BASES = frozenset({"main", "master"})


class GitHubRepoInfo(NamedTuple):
    """Information about the repository being used for operations."""

    repository: Repository
    is_fork: bool
    is_clone: bool
    original_url: str | None


class ResolvedToken(NamedTuple):
    """A resolved GitHub token, its provenance, and WHICH rung produced it.

    ``is_app`` is load-bearing: an App identity has no bot user account to fork
    into, so App-sourced tokens must drive the *same-repo* (no-fork) PR path.

    ``rung`` names the ladder step that produced the token (``user`` / ``app`` /
    ``env`` / ``pool`` / ``system`` / ``none``). It exists because this ladder
    used to downgrade SILENTLY: on 2026-08-15 a per-project PAT without
    ``contents: write`` reached ``POST /git/refs`` and GitHub answered *"Resource
    not accessible by personal access token"*. That message names the credential
    TYPE that was used and says nothing about WHY a weaker one was chosen, so
    three separate sessions diagnosed it as an n8n defect. Callers surface this
    field so the next occurrence names itself.
    """

    token: str | None
    is_app: bool
    rung: str = "unknown"


def is_platform_owned(owner: str | None) -> bool:
    """True when ``owner`` is the platform's own GitHub org.

    Own repos MUST be authored by the GitHub App (see
    :func:`resolve_github_token`); a customer's BYO PAT is for customer repos.
    Reuses the existing ``system_github_owner`` setting rather than introducing
    a second notion of "our org".
    """
    # getattr, not attribute access: several suites swap ``settings`` for a
    # SimpleNamespace double. An absent setting means "no platform org known",
    # which must degrade to the ordinary third-party ladder rather than raise.
    configured = (getattr(settings, "system_github_owner", "") or "").strip().lower()
    return bool(owner) and bool(configured) and owner.strip().lower() == configured


def is_sandbox_owned(owner: str | None) -> bool:
    """True when ``owner`` is a platform-owned SANDBOX org (``sandbox_github_owners``).

    A sandbox org is the platform's own, but the internal App (private to
    ``system_github_owner``) cannot be installed there, so its repos are served
    by the PUBLIC App — gated like platform repos (platform tenants + the repo
    allowlist), never via an fr binding. See
    :func:`services.tenant_binding.resolve_sandbox_repo_token`. Empty setting ⇒
    always False (such an org takes the customer path). Never True for the
    platform org itself.
    """
    if not owner or not owner.strip() or is_platform_owned(owner):
        return False
    # getattr: suites swap ``settings`` for a SimpleNamespace double (no sandbox).
    owners = getattr(settings, "sandbox_owners", None) or frozenset()
    return owner.strip().lower() in owners


def resolve_github_token(
    user_token: str | None = None,
    owner: str | None = None,
    repo: str | None = None,
    tenant_id: str | None = None,
    permissions: dict[str, str] | None = None,
    site: str = "resolve_github_token",
) -> ResolvedToken:
    """Resolve a GitHub token by precedence, and NEVER downgrade silently.

    ``site`` names the caller in the platform-tenant gate's log line
    (``caller_path=``) so an observe-mode ``would_refuse`` is attributable.

    explicit user/container token (the tenant's OWN BYO credential) > GitHub App
    installation token: the internal App for a platform-owned repo (platform
    tenants only), the tenant-bound public App for any other owner. ⛔ There is
    no personal-access-token rung of any kind (owner ruling 2026-09-30).

    The App is the platform's PRIMARY identity once configured (App-first):
    when ``owner``/``repo`` are known the token is minted from the App's
    installation on THAT owner (``get_installation_token_for``), so the bulk
    campaign can act across many orgs that each installed the App; without that
    context the home installation token is used. App installations carry much
    higher rate limits than a PAT. An explicit per-request ``user_token`` still wins (handled by the
    caller before this is reached). A token minted from the App carries
    ``is_app=True`` so callers use the same-repo path
    (see :meth:`GitHubIntegration._ensure_fork`).

    ⛔ **There is no PAT rung below the App for a platform-owned repo.** There
    used to be (env / pool / system tokens, all removed 2026-09-30), and that is the whole defect: when the App rung
    raised, resolution quietly continued to a PAT that could not write to the
    repo, and the failure surfaced 7 minutes later as a GitHub 403 attributed to
    the wrong subsystem. For ``is_platform_owned(owner)`` the App is mandatory —
    if it cannot mint, this raises :class:`GitHubAuthError` NAMING the rung that
    failed.

    ⛔ **A NON-platform owner (a customer repo) never reaches ANY rung below.**
    It is resolved by :func:`services.tenant_binding.resolve_customer_repo_token`
    — the PUBLIC App, only for an installation fr has bound to ``tenant_id``,
    only if GitHub confirms it covers the repo, scoped to that one repo with
    ``permissions``. Every refusal raises a ``TenantRepoTokenError`` (a
    :class:`GitHubAuthError`). There is no env / PAT / pool / system fallback and
    the internal App is not consulted for customer owners.

    ⛔ **``owner`` is REQUIRED** (unless ``user_token`` is given). An ownerless
    call used to return the HOME installation token — org-wide, ungated by any
    tenant check — or a PAT. It raises ``OwnerRequiredError`` (400, a
    :class:`GitHubAuthError`) now. With an
    owner every request is either a customer repo (above) or a platform repo
    (App mandatory).

    ⛔ **A platform-owned repo with ``tenant_id=None`` (or blank) is refused**
    (``platform_repo_forbidden``, 403) regardless of the ENFORCE flag: no tenant
    can never be a platform tenant, and every rung below is the org identity.
    """
    if user_token:
        return ResolvedToken(user_token, False, "user")
    if not owner or not str(owner).strip():
        from services.tenant_binding import OwnerRequiredError

        raise OwnerRequiredError(
            "A repository owner is required to resolve a GitHub credential; refusing to "
            "hand out the home installation token for a request with no repo context.",
            owner=owner,
            repo=repo,
        )
    if not is_platform_owned(owner):
        from services.tenant_binding import (
            PERMS_AUTHORING,
            TenantRepoTokenError,
            resolve_customer_repo_token,
        )

        if not repo:
            raise TenantRepoTokenError(
                f"A repository is required to obtain a credential for owner '{owner}'.",
                owner=owner,
            )
        scoped = resolve_customer_repo_token(owner, repo, tenant_id, permissions or PERMS_AUTHORING)
        return ResolvedToken(scoped.token, True, "public_app")
    # Past the customer branch, ``owner`` is ALWAYS the platform org: only a
    # platform tenant (CODING_GITHUB_PLATFORM_TENANTS) may get the internal App's
    # token, an allowlisted tenant gets a one-repo token, and NO tenant is refused.
    from services.tenant_binding import (
        PERMS_AUTHORING,
        check_platform_repo_tenant,
        is_tenantless,
        mint_allowlisted_platform_token,
        platform_repo_grant,
        refuse_tenantless_platform_repo,
    )

    if is_tenantless(tenant_id):
        refuse_tenantless_platform_repo(owner, repo, site=site)

    if platform_repo_grant(tenant_id, owner, repo) == "scoped":
        # ⛔ Allowlisted NON-platform tenant (CODING_GITHUB_PLATFORM_REPO_ALLOWLIST):
        # an internal-App token for THIS ONE repo, decided before the App rung.
        # A failed mint RAISES — never the org-wide
        # installation token, never a PAT.
        scoped = mint_allowlisted_platform_token(
            owner or "", repo or "", permissions or PERMS_AUTHORING, tenant_id=tenant_id
        )
        return ResolvedToken(scoped.token, True, "app_repo_scoped")
    check_platform_repo_tenant(owner or "", repo, tenant_id, site=site)
    app_failure: str | None = None
    # App-first: the platform identity, per-owner when the repo is known.
    try:
        from services.github_app_auth import (
            GitHubAppAuthError,
            app_auth_configured,
            get_installation_token,
            get_installation_token_for,
        )

        if app_auth_configured():
            try:
                if owner and repo:
                    return ResolvedToken(get_installation_token_for(owner, repo), True, "app")
                return ResolvedToken(get_installation_token(), True, "app")
            except GitHubAppAuthError as e:
                # App configured but not installed on this owner (or home id
                # unset). For our OWN org that is fatal — see the raise below.
                app_failure = f"App not usable for {owner}/{repo}: {e}"
                logger.info("GitHub App not usable for %s/%s: %s", owner, repo, e)
        else:
            app_failure = "GitHub App auth is not configured"
    except Exception as e:  # noqa: BLE001 - never let token resolution crash callers
        app_failure = f"GitHub App auth error: {e}"
        logger.warning("GitHub App auth error: %s", e)

    # Fail CLOSED, naming the rung. Substituting a platform PAT here is what
    # produced the 2026-08-15 403 and three wrong root causes. (``owner`` is the
    # platform org here: customer owners returned above and an owner is required.)
    raise GitHubAuthError(
        f"GitHub App could not authenticate for platform-owned repo "
        f"{owner}/{repo} ({app_failure or 'unknown App failure'}). Refusing to "
        f"fall back to a platform PAT: rung 'app' is mandatory for "
        f"'{settings.system_github_owner}' repos."
    )


class GitHubIntegration:
    """Real GitHub integration service using PyGithub."""

    def __init__(
        self,
        github_token: str | None = None,
        codebase_path: str | None = None,
        is_app_auth: bool = False,
        target_owner: str | None = None,
        target_repo: str | None = None,
        tenant_id: str | None = None,
        permissions: dict[str, str] | None = None,
    ):
        """Initialize GitHub integration.

        ``tenant_id`` / ``permissions`` are REQUIRED in effect for a customer
        (non-platform) ``target_owner``: the credential is then the tenant-bound
        public-App token (see :func:`resolve_github_token`), and construction
        raises a ``TenantRepoTokenError`` when the tenant has no binding.

        Args:
            github_token: GitHub token. When ``None``, the token is resolved via
                :func:`resolve_github_token` (user/container > env > App > system),
                which also determines ``is_app_auth``. When a token is supplied
                explicitly, ``is_app_auth`` is taken from the argument.
            codebase_path: Path to codebase for resolve validation (defaults to cwd)
            is_app_auth: Whether the supplied ``github_token`` is a GitHub App
                installation token (only consulted when ``github_token`` is given).
        """
        if not HAS_GITHUB:
            logger.warning(
                "GitHub library not available - integration service will have limited functionality"
            )

        if github_token is not None:
            self.github_token: str | None = github_token
            self.is_app_auth = is_app_auth
            # An explicitly supplied token skips the ladder entirely; the only
            # caller that does this is the customer-BYO path.
            self.token_rung: str = "byo"
        else:
            # target_owner/repo (when the caller knows them) lets the App mint a
            # per-owner installation token (Phase D3). There is no PAT rung:
            # raises GitHubAuthError (→ HTTP 412) for a platform-owned repo
            # whose App cannot mint.
            resolved = resolve_github_token(
                owner=target_owner,
                repo=target_repo,
                tenant_id=tenant_id,
                permissions=permissions,
            )
            self.github_token = resolved.token
            self.is_app_auth = resolved.is_app
            self.token_rung = resolved.rung
        logger.info(
            "GitHub authoring identity for %s/%s: rung=%s is_app=%s",
            target_owner,
            target_repo,
            self.token_rung,
            self.is_app_auth,
        )
        self.github_client = None
        self.codebase_path = codebase_path or os.getcwd()
        self.fix_validator = None
        if self.github_token:
            try:
                auth = Auth.Token(self.github_token)
                self.github_client = Github(auth=auth)
                logger.info("GitHub client initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize GitHub client: %s", e)

    def create_pull_request(
        self,
        repo_owner: str,
        repo_name: str,
        title: str,
        message: str,
        head_branch: str,
        base_branch: str = "main",
        resolves: list[dict[str, Any]] | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Create a pull request with bug fixes.

        Args:
            repo_owner: Repository owner username
            repo_name: Repository name
            title: PR title
            message: PR description/message
            head_branch: Source branch name
            base_branch: Target branch name (default: main)
            resolves: List of resolve dictionaries with file_path and content
            draft: Open the PR as a draft (not ready for review). The AA
                delivery pipeline opens during DOING as draft and only marks
                ready once QA is green.

        Returns:
            Dictionary with PR creation result
        """
        try:
            repo_info = self._ensure_fork(repo_owner, repo_name)
            fork_repo = repo_info.repository
            # ⛔ The resolved WRITE target (a fork/clone `_ensure_fork` redirected to)
            # is never public either — checked before `_create_branch` writes to it.
            assert_repo_not_public(
                fork_repo, getattr(fork_repo, "full_name", None) or f"{repo_owner}/{repo_name}"
            )
            # Only reset a stale existing branch to base when THIS call will also
            # re-apply files (``resolves`` provided). For the pure PR-open step —
            # the AA refactoring procedure applies in a SEPARATE prior step and
            # calls this with resolves=None — resetting would wipe that apply's
            # commit and yield "No commits between" → the card strands in BLOCKED.
            self._create_branch(
                fork_repo,
                head_branch,
                base_branch,
                reset_existing_to_base=bool(resolves),
            )
            if resolves:
                actual_branch_name = bare_branch_name(head_branch)
                self._update_files(fork_repo, actual_branch_name, resolves)
            if not self.github_client:
                raise GitHubAuthError(
                    "No GitHub token configured. Supply 'user_token' in the request "
                    "or install the GitHub App for this tenant (no PAT fallback exists)."
                )
            original_repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            if self.is_app_auth:
                # Same-repo PR: head is a plain branch name on the target repo,
                # never an ``owner:branch`` cross-fork reference (the App has no
                # bot account to own a fork).
                head_ref = head_branch.split(":", 1)[1] if ":" in head_branch else head_branch
                logger.info("App auth: same-repo head reference: %s", head_ref)
            elif ":" in head_branch:
                head_ref = head_branch
                logger.info("Using provided head reference: %s", head_ref)
            elif repo_info.is_fork or repo_info.is_clone:
                # Cross-repo PR: the branch lives on the fork/clone under the
                # token user's own account, so the head must be ``login:branch``.
                user_login = self.github_client.get_user().login
                head_ref = f"{user_login}:{head_branch}"
                logger.info("Created cross-repo head reference: %s", head_ref)
            else:
                # Same-repo PR (owned or direct push): the branch lives on the
                # target repo itself, so a plain branch ref is correct — an
                # ``owner:branch`` prefix would be wrong whenever login != owner.
                head_ref = head_branch
                logger.info("Same-repo head reference: %s", head_ref)
            pr = retry_github_call(
                lambda: original_repo.create_pull(
                    title=title, body=message, head=head_ref, base=base_branch, draft=draft
                ),
                op_name=f"create_pull {repo_owner}/{repo_name}",
                retryable=GitHubErrorHandler.is_retryable,
            )
            logger.info(
                "Successfully created %sPR #%s: %s",
                "draft " if draft else "",
                pr.number,
                pr.html_url,
            )
            return {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "draft": bool(getattr(pr, "draft", draft)),
                "message": f"Pull request created successfully: #{pr.number}",
                "branch_name": head_branch,
                "fork_url": fork_repo.html_url,
                "is_fork": repo_info.is_fork,
                "is_clone": repo_info.is_clone,
                "original_url": repo_info.original_url,
                "working_repo_url": fork_repo.html_url,
            }
        except GithubException as e:
            error_message = str(e)
            if hasattr(e, "data") and isinstance(e.data, dict):
                # GitHub's 422 puts a generic "Validation Failed" in `message`
                # and the ACTUAL reason ("A pull request already exists…",
                # "No commits between…") in `errors[].message`. Fold both in so
                # callers can classify the failure.
                base_msg = str(e.data.get("message") or e)
                nested = " ".join(
                    str(err.get("message", ""))
                    for err in (e.data.get("errors") or [])
                    if isinstance(err, dict)
                ).strip()
                error_message = f"{base_msg} {nested}".strip() if nested else base_msg
            low = error_message.lower()
            # Idempotency: re-running the same upgrade (same head→base) hits
            # GitHub's "A pull request already exists" (422). Return the existing
            # open PR instead of failing so callers can treat "ensure a PR
            # exists" as idempotent.
            if "pull request already exists" in low:
                try:
                    existing = self._find_open_pr(repo_owner, repo_name, head_ref, base_branch)
                except Exception as lookup_exc:  # noqa: BLE001 — lookup is best-effort
                    logger.warning("PR exists for %s but lookup failed: %s", head_ref, lookup_exc)
                    existing = None
                if existing is not None:
                    logger.info(
                        "Existing PR #%s for %s -> %s: %s",
                        existing.number,
                        head_ref,
                        base_branch,
                        existing.html_url,
                    )
                    return {
                        "success": True,
                        "pr_number": existing.number,
                        "pr_url": existing.html_url,
                        "draft": bool(getattr(existing, "draft", False)),
                        "message": f"Pull request already exists: #{existing.number}",
                        "branch_name": head_branch,
                        "already_existed": True,
                    }
            # Idempotency: an empty-diff apply (every change a no-op, or the
            # branch already at base) leaves nothing to open a PR for. GitHub
            # rejects create_pull with a 422 whose errors[].message contains
            # "No commits between …". That is benign for the AA refactoring
            # loop — there is simply nothing to deliver — so return success
            # with a clear no_commits flag instead of hard-failing the loop.
            if "no commits between" in low:
                logger.info(
                    "No commits between %s and %s; no pull request opened",
                    base_branch,
                    head_ref,
                )
                return {
                    "success": True,
                    "no_commits": True,
                    "pr_number": None,
                    "pr_url": None,
                    "branch_name": head_branch,
                    "message": (
                        f"No commits between {base_branch} and {head_ref}; no pull request opened"
                    ),
                }
            logger.error("GitHub API error creating PR: %s", e)
            return {
                "success": False,
                "error": f"GitHub API error: {error_message}",
                "status_code": getattr(e, "status", 500),
            }
        except (DefaultBranchWriteRefusedError, RepoVisibilityRefusedError):
            # A policy refusal, not a GitHub failure: let the router map it to
            # its own 403 instead of folding it into a generic 400.
            raise
        except Exception as e:
            logger.error("Unexpected error creating PR: %s", e)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _find_open_pr(
        self, repo_owner: str, repo_name: str, head_ref: str, base_branch: str
    ) -> Any | None:
        """Return the open PR for ``head_ref`` -> ``base_branch``, or None.

        ``head_ref`` may be a plain branch (same-repo) or ``owner:branch``
        (cross-fork). ``Repository.get_pulls`` expects ``owner:branch`` for the
        ``head`` filter, so qualify a plain branch with ``repo_owner``.
        """
        if not self.github_client:
            return None
        repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
        branch = head_ref.split(":", 1)[1] if ":" in head_ref else head_ref
        # Cheap path: GitHub's qualified head filter (owner:branch).
        head_filter = head_ref if ":" in head_ref else f"{repo_owner}:{branch}"
        try:
            for pr in repo.get_pulls(state="open", head=head_filter, base=base_branch):
                return pr
        except Exception as exc:  # noqa: BLE001 — fall back to a branch-ref scan
            logger.debug("get_pulls head filter %r failed: %s", head_filter, exc)
        # Robust fallback: the head owner prefix depends on auth mode (App vs
        # PAT vs fork) and may not match the filter, so scan open PRs to base
        # and match on the branch ref directly.
        for pr in repo.get_pulls(state="open", base=base_branch):
            if getattr(pr.head, "ref", None) == branch:
                return pr
        return None

    def mark_pull_request_ready(
        self, repo_owner: str, repo_name: str, pr_number: int
    ) -> dict[str, Any]:
        """Flip a draft PR to ready-for-review.

        The REST ``pulls.edit`` route cannot change the draft flag, so this uses
        GitHub's GraphQL ``markPullRequestReadyForReview`` mutation. Idempotent:
        if the PR is already ready for review it is returned unchanged with
        ``already_ready=True``.
        """
        try:
            if not self.github_client:
                raise GitHubAuthError(
                    "No GitHub token configured. Supply 'user_token' in the request "
                    "or install the GitHub App for this tenant (no PAT fallback exists)."
                )
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            pr = repo.get_pull(pr_number)
            if not getattr(pr, "draft", False):
                logger.info("PR #%s already ready for review (no-op)", pr_number)
                return {
                    "success": True,
                    "pr_number": pr.number,
                    "pr_url": pr.html_url,
                    "draft": False,
                    "pr_status": pr.state,
                    "already_ready": True,
                }
            node_id = getattr(pr, "node_id", None)
            if not node_id:
                raise RuntimeError(f"PR #{pr_number} has no GraphQL node_id")
            mutation = (
                "mutation($id: ID!) {"
                " markPullRequestReadyForReview(input: {pullRequestId: $id}) {"
                " pullRequest { number isDraft url state } } }"
            )
            _headers, data = self.github_client.requester.graphql_query(mutation, {"id": node_id})
            payload = (data or {}).get("data", {}).get("markPullRequestReadyForReview", {}) or {}
            pr_data = payload.get("pullRequest") or {}
            logger.info("Marked PR #%s ready for review", pr_number)
            return {
                "success": True,
                "pr_number": pr_data.get("number", pr.number),
                "pr_url": pr_data.get("url", pr.html_url),
                "draft": bool(pr_data.get("isDraft", False)),
                "pr_status": pr_data.get("state", pr.state),
                "already_ready": False,
            }
        except GitHubAuthError:
            raise
        except GithubException as e:
            logger.error("GitHub API error marking PR ready: %s", e)
            return {
                "success": False,
                "error": f"GitHub API error: {e}",
                "status_code": getattr(e, "status", 500),
            }
        except Exception as e:
            logger.error("Unexpected error marking PR ready: %s", e)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def update_pull_request_body(
        self, repo_owner: str, repo_name: str, pr_number: int, body: str
    ) -> dict[str, Any]:
        """Replace an open PR's description.

        Refuses a public repo (``RepoVisibilityRefusedError`` propagates) and a
        non-open PR (``status_code`` 409). Idempotent: if the body is already
        identical, ``edit`` is not called and ``changed=False`` is returned.
        """
        try:
            if not self.github_client:
                raise GitHubAuthError(
                    "No GitHub token configured. Supply 'user_token' in the request "
                    "or install the GitHub App for this tenant (no PAT fallback exists)."
                )
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            # ⛔ A PUBLIC repo is never a conveyor target — refuse before the write.
            assert_repo_not_public(
                repo, getattr(repo, "full_name", None) or f"{repo_owner}/{repo_name}"
            )
            pr = repo.get_pull(pr_number)
            if pr.state != "open":
                return {
                    "success": False,
                    "error": f"pull request #{pr_number} is {pr.state}, not open",
                    "status_code": 409,
                }
            result = {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "pr_status": pr.state,
            }
            if (pr.body or "") == body:
                logger.info("PR #%s body already up to date (no-op)", pr_number)
                return {**result, "changed": False}
            pr.edit(body=body)
            logger.info("Updated PR #%s body", pr_number)
            return {**result, "changed": True}
        except (GitHubAuthError, RepoVisibilityRefusedError):
            raise
        except GithubException as e:
            logger.error("GitHub API error updating PR body: %s", e)
            return {
                "success": False,
                "error": f"GitHub API error: {e}",
                "status_code": getattr(e, "status", 500),
            }
        except Exception as e:
            logger.error("Unexpected error updating PR body: %s", e)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    @staticmethod
    def _token_can_push(repo: Repository) -> bool:
        """True only when the authenticated token has push (write) access to ``repo``.

        PyGithub populates ``repo.permissions`` on an authenticated ``get_repo``;
        ``permissions.push`` is a real ``bool``. The strict ``is True`` check
        means anything that is not a genuine push grant (no permissions block, a
        truthy stand-in) is treated as no-push — so we only skip forking when we
        can actually push the branch.
        """
        permissions = getattr(repo, "permissions", None)
        return getattr(permissions, "push", False) is True

    def _ensure_fork(self, repo_owner: str, repo_name: str) -> GitHubRepoInfo:
        """Ensure a fork exists with comprehensive error handling."""
        if not self.github_client:
            raise GitHubAuthError(
                "No GitHub token configured. Supply 'user_token' in the request "
                "or install the GitHub App for this tenant (no PAT fallback exists)."
            )
        if self.is_app_auth:
            # A GitHub App has no bot user account to fork into. With Contents +
            # Pull-requests write on the target repo, operate on it directly and
            # open a same-repo PR (branch in place).
            try:
                target_repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            except GithubException as e:
                if e.status == 404:
                    raise ValueError(f"Repository {repo_owner}/{repo_name} not found") from e
                if e.status == 403:
                    raise ValueError(f"Access denied to repository {repo_owner}/{repo_name}") from e
                raise
            logger.info(
                "App auth: using target repository directly for same-repo PR: %s",
                target_repo.full_name,
            )
            return GitHubRepoInfo(
                repository=target_repo,
                is_fork=False,
                is_clone=False,
                original_url=None,
            )
        try:
            user = self.github_client.get_user()
            if not hasattr(user, "login") or not user.login:
                raise ValueError("Could not determine GitHub user login")
            if user.login.lower() == repo_owner.lower():
                try:
                    original_repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
                    logger.info(
                        "User owns repository, using original: %s",
                        original_repo.full_name,
                    )
                    return GitHubRepoInfo(
                        repository=original_repo,
                        is_fork=False,
                        is_clone=False,
                        original_url=None,
                    )
                except GithubException as e:
                    if e.status == 404:
                        raise ValueError(f"Repository {repo_owner}/{repo_name} not found") from e
                    elif e.status == 403:
                        raise ValueError(
                            f"Access denied to repository {repo_owner}/{repo_name}"
                        ) from e
                    else:
                        raise
            try:
                fork_repo = self.github_client.get_repo(f"{user.login}/{repo_name}")
                # A moved/renamed target redirects the user-namespace lookup
                # straight back to the canonical repo (e.g. a repo transferred
                # from the PAT user's account to an org). That is NOT a clone —
                # if we can push to it, it is a direct same-repo PR. Without
                # this, the redirect falls through to the is_clone branch below
                # and create_pull_request builds a wrong ``login:branch`` head.
                if getattr(
                    fork_repo, "full_name", ""
                ).lower() == f"{repo_owner}/{repo_name}".lower() and self._token_can_push(
                    fork_repo
                ):
                    logger.info(
                        "User-namespace lookup resolved to canonical %s with push access "
                        "— using direct same-repo PR (no fork)",
                        fork_repo.full_name,
                    )
                    return GitHubRepoInfo(
                        repository=fork_repo,
                        is_fork=False,
                        is_clone=False,
                        original_url=None,
                    )
                if fork_repo.fork and fork_repo.parent:
                    parent_full_name = fork_repo.parent.full_name.lower()
                    target_full_name = f"{repo_owner}/{repo_name}".lower()
                    if parent_full_name == target_full_name:
                        logger.info("Using existing fork: %s", fork_repo.full_name)
                        return GitHubRepoInfo(
                            repository=fork_repo,
                            is_fork=True,
                            is_clone=False,
                            original_url=f"https://github.com/{repo_owner}/{repo_name}",
                        )
                    logger.warning(
                        "Found fork %s but it's a fork of %s, not %s",
                        fork_repo.full_name,
                        parent_full_name,
                        target_full_name,
                    )
                logger.info(
                    "Using existing user repository (not a fork): %s",
                    fork_repo.full_name,
                )
                return GitHubRepoInfo(
                    repository=fork_repo,
                    is_fork=False,
                    is_clone=True,
                    original_url=f"https://github.com/{repo_owner}/{repo_name}",
                )
            except GithubException as e:
                if e.status != 404:
                    logger.warning("Error checking for existing fork: %s", e)
            try:
                original_repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            except GithubException as e:
                if e.status == 404:
                    raise ValueError(
                        f"Original repository {repo_owner}/{repo_name} not found"
                    ) from e
                elif e.status == 403:
                    raise ValueError(
                        f"Access denied to original repository {repo_owner}/{repo_name}"
                    ) from e
                else:
                    raise
            # Direct same-repo PR: the ``login == repo_owner`` check above only
            # catches repos owned by the token's *personal* account. When the
            # token can push to the target (e.g. an org repo like
            # ``nonagenticai/*`` driven by a member PAT), forking is unnecessary
            # and wrong — branch in place and open a same-repo PR, exactly as the
            # owned-repo path does.
            if self._token_can_push(original_repo):
                logger.info(
                    "Token has push access to %s/%s — opening a direct same-repo PR (no fork)",
                    repo_owner,
                    repo_name,
                )
                return GitHubRepoInfo(
                    repository=original_repo,
                    is_fork=False,
                    is_clone=False,
                    original_url=None,
                )
            try:
                create_fork = getattr(user, "create_fork", None)
                if create_fork:
                    # Throttle-only retry: fork creation is a prime trigger for
                    # GitHub's secondary (abuse) rate limit during a mass
                    # campaign. Generic 5xx is NOT retried here — it falls
                    # through to the copy-repo fallback below.
                    fork_repo = retry_github_call(
                        lambda: create_fork(original_repo),
                        op_name=f"create_fork {repo_owner}/{repo_name}",
                        retryable=GitHubErrorHandler.is_throttled,
                    )
                else:
                    raise GithubException(403, "User does not support forking", None)
                logger.info("Created fork: %s", fork_repo.full_name)
                # Forks are created asynchronously — the repo object exists
                # immediately but its git refs can 404 for a few seconds, which
                # made the old fixed `time.sleep(2)` both flaky (too short under
                # load) and wasteful (too long when ready). Poll until the
                # default branch resolves; if it never becomes push-ready, fall
                # back to the copy-repo path rather than pushing into a half-born
                # fork.
                if self._wait_for_fork_ready(fork_repo):
                    return GitHubRepoInfo(
                        repository=fork_repo,
                        is_fork=True,
                        is_clone=False,
                        original_url=original_repo.html_url,
                    )
                logger.warning(
                    "Fork %s not push-ready within timeout; falling back to copy-repo",
                    getattr(fork_repo, "full_name", f"{user.login}/{repo_name}"),
                )
                cloned_repo = self._create_user_repository_from_original(original_repo, user)
                return GitHubRepoInfo(
                    repository=cloned_repo,
                    is_fork=False,
                    is_clone=True,
                    original_url=original_repo.html_url,
                )
            except GithubException as fork_error:
                logger.warning(
                    "Forking failed (%s), attempting to clone repository content",
                    fork_error,
                )
                if fork_error.status == 403:
                    error_msg = "Forking disabled by repository owner or insufficient permissions"
                elif fork_error.status == 422:
                    error_msg = "Fork already exists or cannot be created"
                else:
                    error_msg = f"GitHub API error: {fork_error}"
                logger.info("Fork creation failed: %s. Creating clone repository.", error_msg)
                cloned_repo = self._create_user_repository_from_original(original_repo, user)
                return GitHubRepoInfo(
                    repository=cloned_repo,
                    is_fork=False,
                    is_clone=True,
                    original_url=original_repo.html_url,
                )
        except Exception as e:
            logger.error("Error ensuring fork for %s/%s: %s", repo_owner, repo_name, e)
            raise

    def _wait_for_fork_ready(
        self,
        fork_repo: Repository,
        *,
        timeout: float = 30.0,
        interval: float = 2.0,
    ) -> bool:
        """Poll a freshly created fork until its default branch is push-ready.

        Returns ``True`` once the default branch ref resolves (the fork has been
        populated and can accept a branch + commits), or ``False`` if it never
        becomes ready within ``timeout`` seconds. Best-effort: any error during
        the readiness probe is treated as "not ready yet" and retried until the
        deadline.
        """
        deadline = time.monotonic() + timeout
        default_branch = getattr(fork_repo, "default_branch", None) or "main"
        while True:
            try:
                fork_repo.get_branch(default_branch)
                return True
            except GithubException as e:
                logger.debug(
                    "Fork %s not push-ready yet: %s",
                    getattr(fork_repo, "full_name", "?"),
                    e,
                )
            except Exception as e:  # noqa: BLE001 — readiness probe is best-effort
                logger.debug("Fork readiness probe error: %s", e)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(interval, remaining))

    def _create_user_repository_from_original(
        self, original_repo: Repository, user: Any
    ) -> Repository:
        """Create a new repository under user's account with content from original repository.

        This is used when forking is not allowed or fails.
        """
        try:
            base_name = original_repo.name
            new_repo_name = base_name
            counter = 1
            while True:
                try:
                    if self.github_client is None:
                        raise GitHubAuthError(
                            "No GitHub token configured. Supply 'user_token' in the request "
                            "or install the GitHub App for this tenant (no PAT fallback exists)."
                        )
                    self.github_client.get_repo(f"{user.login}/{new_repo_name}")
                    new_repo_name = f"{base_name}-clone-{counter}"
                    counter += 1
                except GithubException as e:
                    if e.status == 404:
                        break
                    else:
                        raise
            new_repo = user.create_repo(
                name=new_repo_name,
                description=f"Clone of {original_repo.full_name} (forking not available)",
                # ⛔ ALWAYS private: copying the upstream's flag made a public upstream
                # produce a PUBLIC repo that the conveyor then commits into.
                private=True,
                auto_init=True,
            )
            logger.info("Created new repository: %s", new_repo.full_name)
            # ⛔ Confirm GitHub actually made it private before committing into it
            # (fail closed on an absent/unknown flag).
            assert_repo_not_public(new_repo, getattr(new_repo, "full_name", new_repo_name))
            self._copy_repository_content(original_repo, new_repo)
            return new_repo
        except GithubException as e:
            logger.error("Error creating user repository from original: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error creating user repository: %s", e)
            raise

    def _copy_repository_content(self, source_repo: Repository, target_repo: Repository) -> None:
        """Copy essential files from source repository to target repository."""
        try:
            default_branch = source_repo.default_branch
            essential_files = [
                "README.md",
                "README.rst",
                "README.txt",
                "README",
                "LICENSE",
                "LICENSE.md",
                "LICENSE.txt",
                "requirements.txt",
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                ".gitignore",
                "Dockerfile",
                "docker-compose.yml",
                "package.json",
                "Cargo.toml",
                "go.mod",
            ]
            for file_name in essential_files:
                try:
                    file_content = source_repo.get_contents(file_name, ref=default_branch)
                    if isinstance(file_content, list):
                        continue
                    target_repo.create_file(
                        path=file_name,
                        message=f"Copy {file_name} from {source_repo.full_name}",
                        content=file_content.decoded_content.decode("utf-8"),
                        branch=target_repo.default_branch,
                    )
                    logger.info("Copied %s to new repository", file_name)
                except GithubException as e:
                    if e.status == 404:
                        continue
                    else:
                        logger.warning("Failed to copy %s: %s", file_name, e)
                        continue
                except UnicodeDecodeError:
                    logger.warning("Skipped %s due to encoding issues", file_name)
                    continue
            clone_readme = "# {target_repo.name}\n\nThis repository is a clone of [{source_repo.full_name}]({source_repo.html_url}).\n\nOriginal repository: {source_repo.html_url}\nClone created: Due to forking restrictions on the original repository\nPurpose: Enable collaborative development and contributions\n\n## Getting Started\n\nThis clone contains the essential configuration files from the original repository.\nYou may need to copy additional source files manually if needed for your development work.\n\n## Contributing Back\n\nTo contribute changes back to the original repository:\n1. Make your changes in this repository\n2. Create a patch or diff of your changes\n3. Submit the changes to the original repository through their preferred contribution method\n"
            try:
                try:
                    readme_content = target_repo.get_contents("README.md")
                    if isinstance(readme_content, list):
                        readme_file = readme_content[0]
                    else:
                        readme_file = readme_content
                    target_repo.update_file(
                        path="README.md",
                        message="Update README with clone information",
                        content=clone_readme,
                        sha=readme_file.sha,
                        branch=target_repo.default_branch,
                    )
                except GithubException:
                    target_repo.create_file(
                        path="README.md",
                        message="Add README with clone information",
                        content=clone_readme,
                        branch=target_repo.default_branch,
                    )
                logger.info("Added clone information to README")
            except Exception as e:
                logger.warning("Failed to update README with clone info: %s", e)
        except Exception as e:
            logger.warning("Error copying repository content: %s", e)

    def _resolve_base_sha(self, repo: Repository, base: str) -> tuple[str, str]:
        """Resolve ``base`` to the commit a new branch must start from.

        Returns ``(sha, label)`` where ``label`` names what was actually used.
        Order: a branch head of that name, then a commit SHA, then — only for an
        implicit default base (``main``/``master``) — the repo's default branch.
        Anything else raises :class:`BaseRefNotFoundError`.
        """
        try:
            return repo.get_git_ref(f"heads/{base}").object.sha, base
        except GithubException as e:
            if getattr(e, "status", None) != 404:
                raise
            not_found = e

        if _COMMIT_SHA_RE.match(base):
            try:
                commit_sha = repo.get_commit(base).sha
            except GithubException as e:
                if getattr(e, "status", None) not in (404, 422):
                    raise
                raise BaseRefNotFoundError(
                    f"Base '{base}' is neither a branch nor a commit on {repo.full_name}"
                ) from e
            logger.info("Base '%s' is a commit, not a branch; using %s", base, commit_sha)
            return commit_sha, commit_sha

        if base in _IMPLICIT_DEFAULT_BASES:
            default_branch = repo.default_branch
            logger.warning(
                "Base branch '%s' not found, using default branch '%s'",
                base,
                default_branch,
            )
            try:
                return repo.get_git_ref(f"heads/{default_branch}").object.sha, default_branch
            except GithubException:
                raise BaseRefNotFoundError(
                    f"Neither base branch '{base}' nor default branch "
                    f"'{default_branch}' found on {repo.full_name}"
                ) from not_found

        raise BaseRefNotFoundError(
            f"Base branch '{base}' not found on {repo.full_name}; refusing to "
            "fall back to the default branch for an explicitly requested base"
        ) from not_found

    def _create_branch(
        self,
        repo: Repository,
        branch_name: str,
        base_branch: str = "main",
        reset_existing_to_base: bool = True,
    ) -> Any:
        """Create a new branch in the repository with updated error handling.

        ``reset_existing_to_base`` controls what happens when the branch already
        exists at a SHA other than base. When True (the apply path: this call —
        or its caller — will re-apply files immediately after), a stale leftover
        ref is force-reset to base so the re-apply starts clean. When False (the
        pure PR-open path, where a SEPARATE prior step already populated the
        branch), the existing commits are preserved — resetting them would wipe
        the apply step's real diff and yield GitHub's "No commits between".
        """
        try:
            if ":" in branch_name:
                actual_branch_name = branch_name.split(":", 1)[1]
                logger.info(
                    "Extracted branch name '%s' from '%s'",
                    actual_branch_name,
                    branch_name,
                )
            else:
                actual_branch_name = branch_name
            if not re.match("^[a-zA-Z0-9._/-]+$", actual_branch_name):
                raise ValueError(f"Invalid branch name format: {actual_branch_name}")
            base_sha, base_branch = self._resolve_base_sha(repo, base_branch)
            try:
                branch_ref = repo.create_git_ref(
                    ref=f"refs/heads/{actual_branch_name}", sha=base_sha
                )
                logger.info("Created branch '%s' from '%s'", actual_branch_name, base_branch)
                return branch_ref
            except GithubException as e:
                if e.status == 422:
                    logger.info("Branch '%s' already exists", actual_branch_name)
                    existing_ref = repo.get_git_ref(f"heads/{actual_branch_name}")
                    if existing_ref.object.sha != base_sha and reset_existing_to_base:
                        # A force-reset IS a force-push: never onto the
                        # default branch.
                        assert_branch_writable(repo, actual_branch_name)
                        # A reused per-drive refactor branch left over from a
                        # prior apply is a throwaway: force it back to base so
                        # the re-apply starts from a clean base and can produce
                        # a real commit. Serving the stale ref silently makes
                        # the next apply a no-op (identical content already on
                        # the old tip) → zero commits → "No commits between"
                        # → the loop strands. Force-reset, don't just warn.
                        logger.warning(
                            "Existing branch '%s' is at %s, not base '%s' (%s); "
                            "force-resetting to base for a clean re-apply.",
                            actual_branch_name,
                            existing_ref.object.sha,
                            base_branch,
                            base_sha,
                        )
                        existing_ref.edit(sha=base_sha, force=True)
                        logger.info(
                            "Reset branch '%s' to base '%s' (%s)",
                            actual_branch_name,
                            base_branch,
                            base_sha,
                        )
                    elif existing_ref.object.sha != base_sha:
                        # Branch is ahead of base and we are NOT re-applying in
                        # this call: those commits are a SEPARATE prior apply
                        # step's real work (the architecture-refactoring procedure
                        # applies in step 8 and opens the PR in step 9). Preserve
                        # them so the PR opens against a non-empty diff instead of
                        # being force-reset to base → "No commits between" → the
                        # card stranding in BLOCKED before QA.
                        logger.info(
                            "Existing branch '%s' is ahead of base '%s' at %s; "
                            "preserving its commits (no re-apply in this call).",
                            actual_branch_name,
                            base_branch,
                            existing_ref.object.sha,
                        )
                    return existing_ref
                raise
        except Exception as e:
            logger.error("Error creating branch %s: %s", branch_name, e)
            raise

    def _update_files(
        self, repo: Repository, branch_name: str, changes: list[dict[str, Any]]
    ) -> str:
        """File update with proper change_type support.

        Backward-compatible wrapper returning only the last commit SHA (an empty
        string when every change was a no-op). Callers that must distinguish a
        real apply from a zero-commit no-op should use :meth:`_apply_changes`.
        """
        return self._apply_changes(repo, branch_name, changes)[0]

    def _apply_changes(
        self, repo: Repository, branch_name: str, changes: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Apply file changes, returning ``(last_commit_sha, committed_count)``.

        The per-file helpers return ``None`` when a write is a no-op (identical
        content, or a delete of a missing file), so a change that produces no
        commit is NOT counted. ``committed_count == 0`` therefore means every
        requested change was a no-op and zero commits landed — the caller must
        treat that as benign "nothing to commit" rather than a false success
        (and never feed it to create-pull-request, which would 422 with
        "No commits between").
        """
        # Every commit this service makes goes through here, so this is the
        # chokepoint: no file write lands on the default branch.
        assert_branch_writable(repo, branch_name)
        last_commit_sha = ""
        committed = 0
        for change in changes:
            file_path = change.get("file_path")
            content = change.get("content")
            change_type = change.get("change_type", "modified")
            try:
                if change_type == "deleted":
                    if file_path is None:
                        raise ValueError("file_path is required for delete operation")
                    commit_sha = self._delete_file(repo, branch_name, file_path)
                    if commit_sha:
                        last_commit_sha = commit_sha
                        committed += 1
                elif change_type == "renamed":
                    old_path = change.get("old_file_path")
                    if not old_path:
                        raise ValueError("old_file_path required for rename operation")
                    if file_path is None:
                        raise ValueError("file_path is required for rename operation")
                    if content is None:
                        raise ValueError("content is required for rename operation")
                    commit_sha = self._rename_file(repo, branch_name, old_path, file_path, content)
                    if commit_sha:
                        last_commit_sha = commit_sha
                        committed += 1
                elif change_type in ["added", "modified"]:
                    if file_path is None:
                        raise ValueError("file_path is required for add/modify operation")
                    if content is None:
                        raise ValueError("content is required for add/modify operation")
                    commit_sha = self._create_or_update_file(repo, branch_name, file_path, content)
                    if commit_sha:
                        last_commit_sha = commit_sha
                        committed += 1
                else:
                    logger.warning("Unknown change_type: %s", change_type)
                    if file_path is None or content is None:
                        raise ValueError("file_path and content are required")
                    commit_sha = self._create_or_update_file(repo, branch_name, file_path, content)
                    if commit_sha:
                        last_commit_sha = commit_sha
                        committed += 1
            except Exception as e:
                logger.error("Failed to apply change %s to %s: %s", change_type, file_path, e)
                raise
        return last_commit_sha, committed

    def _create_or_update_file(
        self, repo: Repository, branch_name: str, file_path: str, content: str
    ) -> str | None:
        """Create new file or update existing file with updated error handling."""
        try:
            if isinstance(content, bytes):
                try:
                    content = content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                        try:
                            content = content.decode(encoding)
                            logger.warning("File %s decoded using %s", file_path, encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError(f"Cannot decode file content for {file_path}") from exc
            if not content.strip():
                logger.warning("Empty content for file %s", file_path)
            try:
                content.encode("utf-8")
            except UnicodeEncodeError as e:
                logger.error("Content contains invalid UTF-8 characters for %s: %s", file_path, e)
                content = content.encode("utf-8", errors="replace").decode("utf-8")
            try:
                file_contents = repo.get_contents(file_path, ref=branch_name)
                if isinstance(file_contents, list):
                    if not file_contents:
                        raise GithubException(404, "File not found")
                    if len(file_contents) > 1:
                        raise ValueError(
                            f"Multiple files found for path: {file_path} - this might be a directory"
                        )
                    file_content = file_contents[0]
                else:
                    file_content = file_contents
                try:
                    existing_content = file_content.decoded_content.decode("utf-8")
                    if existing_content.strip() == content.strip():
                        logger.info("No changes needed for file: %s", file_path)
                        return None
                except Exception as e:
                    logger.warning("Could not compare existing content for %s: %s", file_path, e)
                update_result = repo.update_file(
                    path=file_path,
                    message=f"Update {file_path}",
                    content=content,
                    sha=file_content.sha,
                    branch=branch_name,
                )
                logger.info("Updated existing file: %s", file_path)
                return update_result["commit"].sha
            except GithubException as e:
                if e.status == 404:
                    try:
                        create_result = repo.create_file(
                            path=file_path,
                            message=f"Create {file_path}",
                            content=content,
                            branch=branch_name,
                        )
                        logger.info("Created new file: %s", file_path)
                        return create_result["commit"].sha
                    except GithubException as create_error:
                        if "path" in str(create_error).lower():
                            logger.error(
                                "Failed to create file due to path issue: %s",
                                create_error,
                            )
                            raise ValueError(
                                f"Invalid file path or directory conflict: {file_path}"
                            ) from create_error
                        raise
                if e.status in (409, 422):
                    # Optimistic-concurrency conflict: a concurrent writer changed
                    # this file between our get_contents (sha read) and update_file
                    # — a TOCTOU on the shared fork branch. The AA refactoring loop
                    # hits this when a sibling apply / tournament candidate writes the
                    # same branch, or when a slow PUT's first attempt already landed
                    # and a retry sees the new sha. Re-fetch: if the file now already
                    # holds OUR desired content the concurrent write applied our change
                    # → return None (idempotent no-op) so committed_count stays accurate
                    # and the apply does NOT 500 over an already-applied change (which
                    # was stranding architecture_refactoring cards in BLOCKED despite a
                    # valid PR). Otherwise retry the update once against the fresh sha.
                    try:
                        fresh = repo.get_contents(file_path, ref=branch_name)
                        if isinstance(fresh, list):
                            fresh = fresh[0]
                        fresh_content = fresh.decoded_content.decode("utf-8")
                        if fresh_content.strip() == content.strip():
                            logger.info(
                                "File %s already at desired content after a concurrent "
                                "write; idempotent no-op (conflict %s)",
                                file_path,
                                e.status,
                            )
                            return None
                        retry_result = repo.update_file(
                            path=file_path,
                            message=f"Update {file_path}",
                            content=content,
                            sha=fresh.sha,
                            branch=branch_name,
                        )
                        logger.info(
                            "Updated %s after refetching sha post-conflict (%s)",
                            file_path,
                            e.status,
                        )
                        return retry_result["commit"].sha
                    except GithubException as conflict_retry_error:
                        logger.error(
                            "Conflict retry for %s failed: %s",
                            file_path,
                            conflict_retry_error,
                        )
                        raise
                raise
        except Exception as e:
            logger.error("Error creating/updating file %s: %s", file_path, e)
            raise

    def _delete_file(self, repo: Repository, branch_name: str, file_path: str) -> str | None:
        """Delete a file from the repository."""
        try:
            file_content = repo.get_contents(file_path, ref=branch_name)
            if isinstance(file_content, list):
                raise ValueError(f"Cannot delete directory: {file_path}")
            delete_result = repo.delete_file(
                path=file_path,
                message=f"Delete {file_path}",
                sha=file_content.sha,
                branch=branch_name,
            )
            logger.info("Deleted file: %s", file_path)
            if "commit" in delete_result and hasattr(delete_result["commit"], "sha"):
                return str(delete_result["commit"].sha)
            return None
        except ValueError:
            raise
        except GithubException as e:
            if e.status == 404:
                logger.warning("File not found for deletion: %s", file_path)
                return None
            raise

    def _rename_file(
        self,
        repo: Repository,
        branch_name: str,
        old_path: str,
        new_path: str,
        content: str | None = None,
    ) -> str | None:
        """Rename/move a file (create new + delete old)."""
        try:
            if content is None:
                old_file = repo.get_contents(old_path, ref=branch_name)
                if isinstance(old_file, list):
                    raise ValueError(f"Cannot rename directory: {old_path}")
                content = old_file.decoded_content.decode("utf-8")
            create_commit_sha = self._create_or_update_file(repo, branch_name, new_path, content)
            delete_commit_sha = self._delete_file(repo, branch_name, old_path)
            logger.info("Renamed file: %s -> %s", old_path, new_path)
            return delete_commit_sha or create_commit_sha
        except Exception as e:
            logger.error("Failed to rename %s to %s: %s", old_path, new_path, e)
            raise

    def validate_fix(self, fix_data: dict[str, Any]) -> dict[str, Any]:
        """Validate a fix before applying it."""
        logger.debug("Fix validation stub")
        return {
            "valid": True,
            "warnings": ["Fix validation unavailable - coding-analysis service required"],
        }

    @staticmethod
    async def create_upstream_pull_request(
        container_name: str,
        title: str,
        message: str,
        head_branch: str,
        base_branch: str = "main",
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a pull request from system fork to user's private repository.

        This method is used in the dual-token workflow where:
        1. User provisions their private repo (with their token)
        2. System creates fork under @nonagenticai account
        3. Work is done on system fork
        4. PR is created back to user's private repo (using user's token)

        Args:
            container_name: Container name containing the repository metadata
            title: PR title
            message: PR description/body
            head_branch: Source branch on system fork (e.g., "nonagenticai:feature-branch")
            base_branch: Target branch on user's repo (default: "main")

        Returns:
            Dictionary with PR creation result

        A ``@staticmethod``: it resolves its own credential per branch below (with
        ``tenant_id``), so it needs no instance — and constructing one with no
        owner would resolve an ownerless credential, which is refused.
        """
        try:
            from main import get_workspaces_client
            from services.github_token_manager import get_github_token_manager
            from services.outbound import optional_sibling_headers

            client = get_workspaces_client()
            # Forward the project tenant so coding-containers' tenant-scoped
            # GET resolves the container (else a cross-tenant lookup 404s).
            # Present coding-github's OWN licence alongside the tenant assertion.
            # coding-containers runs TRUST_TENANT_HEADER=false, so a header-only
            # call 401s and this method fails with "Unexpected error creating
            # upstream PR: ... returned 401" -- measured live 2026-08-14 06:06:57.
            container_headers = optional_sibling_headers(tenant_id) or None
            raw_container = await client.get_workspace(container_name, headers=container_headers)
            # coding-containers wraps the model in an envelope:
            # {"status": ..., "message": ..., "container": {<ContainerInfo>}}.
            # Unwrap to the ContainerInfo (tolerate an already-flat payload).
            container_info = (
                raw_container.get("container", raw_container)
                if isinstance(raw_container, dict)
                else raw_container
            )

            # Extract metadata
            user_id = container_info.get("user_id")
            original_repo_owner = container_info.get("original_repo_owner")
            original_repo_name = container_info.get("original_repo_name")
            # No system fork is created when we hold the user's BYO token (the
            # branch is pushed straight to the user's repo). Fall back to the
            # repo owner so the PR is a same-repo PR (head=branch, base=main).
            system_fork_owner = container_info.get("system_fork_owner") or original_repo_owner
            secret_config_id = container_info.get("secret_config_id")

            if not all([original_repo_owner, original_repo_name]):
                raise ValueError(
                    f"Container {container_name} missing repository metadata for PR creation."
                )

            # Resolve the user's GitHub token from secrets-service (the
            # cross-service source of truth — the in-memory token store does not
            # survive cross-process). Falls back to the legacy in-memory store.
            from services.secret_resolver import resolve_git_token

            # ⛔ PLATFORM-OWNED REPOS ARE AUTHORED BY THE APP HERE TOO.
            # This is a SECOND, independent token resolution — it does not go
            # through resolve_github_token, so App-first in the apply-changes
            # router would not reach it. Left alone, a drive on our own repo
            # would have the App create the branch and then die HERE with
            # "User token not found" the moment no project PAT was stored
            # (which is exactly what dropping GH_DEMO_PAT produces).
            token_rung = "byo"
            # ⛔ A SANDBOX org is platform-owned as well: App only (public App,
            # platform-gated inside resolve_github_token). Stored project/user
            # tokens are skipped exactly as for the platform org.
            if is_platform_owned(original_repo_owner) or is_sandbox_owned(original_repo_owner):
                # tenant_id is passed so a non-platform tenant is refused
                # (platform_repo_forbidden) rather than handed the internal App.
                resolved = resolve_github_token(
                    owner=original_repo_owner, repo=original_repo_name, tenant_id=tenant_id
                )
                user_token = resolved.token
                token_rung = resolved.rung
            else:
                # CUSTOMER repo. The project's BYO token (tenant-scoped via
                # secret_config_id + tenant_id) and a token the user registered
                # for this container are the CALLER's own chosen identities and
                # win when present. Otherwise the ONLY platform-issued credential
                # is the tenant-bound public App (raises TenantRepoTokenError on
                # refusal) — never a platform PAT.
                user_token = await resolve_git_token(secret_config_id, tenant_id)
                if not user_token:
                    token_manager = get_github_token_manager()
                    user_token = await token_manager.get_user_token(
                        user_id=user_id or "", container_id=container_name
                    )
                if not user_token:
                    import asyncio

                    from services.tenant_binding import (
                        PERMS_AUTHORING,
                        resolve_customer_repo_token,
                    )

                    scoped = await asyncio.to_thread(
                        resolve_customer_repo_token,
                        original_repo_owner,
                        original_repo_name,
                        tenant_id,
                        PERMS_AUTHORING,
                    )
                    user_token = scoped.token
                    token_rung = "public_app"

            if not user_token:
                raise ValueError(
                    f"No GitHub token for container={container_name} "
                    f"(rung={token_rung}, secret_config_id={secret_config_id}) — "
                    f"the PR to {original_repo_owner}/{original_repo_name} cannot be opened."
                )
            logger.info(
                "Upstream PR authoring identity for %s/%s: rung=%s",
                original_repo_owner,
                original_repo_name,
                token_rung,
            )

            logger.info(
                f"Creating upstream PR from {system_fork_owner}:{head_branch} "
                f"to {original_repo_owner}/{original_repo_name}:{base_branch}"
            )

            # Initialize GitHub client with user's token
            auth = Auth.Token(user_token)
            github_client = Github(auth=auth)

            try:
                # Get the original (user's) repository
                original_repo = github_client.get_repo(
                    f"{original_repo_owner}/{original_repo_name}"
                )

                # ⛔ A PUBLIC repo is never a conveyor target (403 public_repo_refused;
                # unknown visibility is 503 repo_visibility_unknown, never a pass).
                try:
                    assert_repo_not_public(
                        original_repo, f"{original_repo_owner}/{original_repo_name}"
                    )
                except RepoVisibilityRefusedError as e:
                    logger.warning("Upstream PR refused: %s (%s)", e.error_code, e)
                    return {
                        "success": False,
                        "error": str(e),
                        "status_code": e.status_code,
                        "error_code": e.error_code,
                    }

                # Create head reference (system_fork:branch)
                if ":" in head_branch:
                    head_ref = head_branch
                else:
                    head_ref = f"{system_fork_owner}:{head_branch}"

                logger.info(f"Creating PR with head_ref={head_ref}, base={base_branch}")

                # Defensive pre-check: create_pull returns an opaque GitHub 422
                # "Validation Failed: head invalid" when the head branch does not
                # exist on the remote. That happens when an upstream push step
                # silently failed (e.g. a large-repo push timed out), leaving the
                # branch absent. Verify the head ref exists first and return a
                # clear, actionable error instead of the opaque 422.
                bare_head = head_ref.split(":", 1)[1] if ":" in head_ref else head_ref
                try:
                    fork_repo = (
                        original_repo
                        if system_fork_owner == original_repo_owner
                        else github_client.get_repo(f"{system_fork_owner}/{original_repo_name}")
                    )
                    fork_repo.get_git_ref(f"heads/{bare_head}")
                except GithubException:
                    logger.error(
                        "Upstream PR aborted: head branch '%s' not found on %s/%s "
                        "— the push step did not complete (branch missing on remote).",
                        bare_head,
                        system_fork_owner,
                        original_repo_name,
                    )
                    return {
                        "success": False,
                        "error": (
                            f"Head branch '{bare_head}' not found on "
                            f"{system_fork_owner}/{original_repo_name} — the push "
                            f"step did not complete (the branch was never pushed "
                            f"to the remote)."
                        ),
                        "status_code": 422,
                    }

                # Create the pull request
                pr = original_repo.create_pull(
                    title=title, body=message, head=head_ref, base=base_branch
                )

                logger.info(
                    f"Successfully created upstream PR #{pr.number} to {original_repo_owner}/{original_repo_name}"
                )

                return {
                    "success": True,
                    "pr_number": pr.number,
                    "pr_url": pr.html_url,
                    "message": f"Pull request created successfully: #{pr.number}",
                    "original_repo": f"{original_repo_owner}/{original_repo_name}",
                    "system_fork": system_fork_owner,
                    "branch_name": head_branch,
                }

            finally:
                github_client.close()

        except GithubException as e:
            logger.error(f"GitHub API error creating upstream PR: {e}")
            error_message = str(e)
            if hasattr(e, "data") and isinstance(e.data, dict):
                error_message = str(e.data.get("message") or e)
            return {
                "success": False,
                "error": f"GitHub API error: {error_message}",
                "status_code": getattr(e, "status", 500),
            }
        except Exception as e:
            from services.tenant_binding import TenantRepoTokenError

            if isinstance(e, TenantRepoTokenError):
                logger.warning("Upstream PR refused: %s (%s)", e.code, e)
                return {
                    "success": False,
                    "error": str(e),
                    "status_code": e.http_status,
                    "error_code": e.code,
                }
            logger.error(f"Unexpected error creating upstream PR: {e}", exc_info=True)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def generate_pr_description(
        self, resolves: list[dict[str, Any]], issue_description: str = ""
    ) -> str:
        """Generate a comprehensive PR description."""
        description_parts: list[Any] = []
        description_parts.append("## Code Updates")
        description_parts.append("This pull request contains automated code updates and bug fixes.")
        if issue_description:
            description_parts.append(f"\n## Context\n{issue_description}")
        description_parts.append("\n## Changes Made")
        for i, resolve in enumerate(resolves, 1):
            file_path = resolve.get("file_path", "Unknown file")
            fix_type = resolve.get("fix_type", "Unknown fix")
            description_parts.append(f"{i}. **{file_path}**: {fix_type}")
        description_parts.extend(
            [
                "\n## Testing",
                "- [ ] Code compiles without errors",
                "- [ ] All existing tests pass",
                "- [ ] New tests added for bug fixes",
                "- [ ] Manual testing completed",
                "\n## Checklist",
                "- [ ] Code follows project style guidelines",
                "- [ ] Self-review completed",
                "- [ ] Documentation updated if needed",
                "- [ ] No breaking changes introduced",
            ]
        )
        return "\n".join(description_parts)
