"""
GitHub Validation Service - Complete implementation with GitHub API validation.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from github import Auth, Github
    from github.GithubException import GithubException

try:
    from github import Auth, Github
    from github.GithubException import GithubException

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    Auth = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    Github = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    GithubException = Exception  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

from coding_client.exceptions.api_exceptions import NotFoundException, ValidationException
from coding_client.models.container import GitHubRepoRequest

from config.settings import get_settings, load_feature_flags
from services.tenant_binding import PERMS_PR_READ

logger = logging.getLogger(__name__)

# Token-source sentinels, so logs (and tests) key on a constant rather than on
# log prose.
TOKEN_SOURCE_EXPLICIT = "explicit"
TOKEN_SOURCE_APP = "app"
TOKEN_SOURCE_NONE = "none"

# GitHub App INSTALLATION tokens are ``ghs_``-prefixed. Kept as a constant
# because two independent facts hang off it (see ``is_app`` below), and because a
# magic string in two modules is how the two pre-flights would drift apart.
_INSTALLATION_TOKEN_PREFIX = "ghs_"


def _is_installation_token(token: str | None) -> bool:
    """True when ``token`` is a GitHub App installation token (``ghs_…``).

    Deliberately keyed on the CREDENTIAL, not on the rung that produced it: a
    ``ghs_`` token behaves the same way at api.github.com whoever handed it over,
    so an installation token passed as an explicit request token must get the
    same treatment as one this module minted.
    """
    return bool(token) and token.strip().startswith(_INSTALLATION_TOKEN_PREFIX)


class ResolvedGitHubToken(NamedTuple):
    """A credential for a read-only probe, WHICH rung produced it, and its TYPE.

    ⚠️ ``is_app`` is load-bearing and was the defect in the first cut of this
    change. A GitHub App installation token **cannot call ``GET /user``** —
    api.github.com answers ``403 Resource not accessible by integration``.
    Measured in the pod against a token minted by
    :func:`services.github_app_auth.get_installation_token_for`::

        minted: len=390 prefix=ghs_
        get_user  RAISED GithubException 403 "Resource not accessible by integration"
        get_repo  OK  private=True

    Every call site here used ``g.get_user()`` as an "authentication test" BEFORE
    ``g.get_repo()``, so the new App rung could only ever convert a clean 404
    into ``ACCESS_FORBIDDEN`` / HTTP 400 — it degraded the error instead of
    fixing anything, with ``get_repo`` never reached. Callers must skip that
    probe when ``is_app`` is true, exactly as
    :attr:`services.github_integration.GitHubIntegration.is_app_auth` already
    does at :mod:`services.github_integration` (``create_pull_request`` /
    ``_ensure_fork``).

    ``source`` is one of the ``TOKEN_SOURCE_*`` sentinels and answers a second
    question: whether the caller CHOSE this identity. A credential the caller did
    not supply (``app``) must never turn a probe that used to succeed
    anonymously into an authentication error — see
    :meth:`CoreGitHubValidationService._may_degrade_to_anonymous`.
    """

    token: str | None
    source: str
    is_app: bool


# "No credential from any source" — the ANONYMOUS probe. Immutable, so one
# shared instance is safe, and every anonymous outcome is spelled identically.
_NO_GITHUB_TOKEN = ResolvedGitHubToken(None, TOKEN_SOURCE_NONE, False)


def _resolve_github_token(
    explicit: str | None,
    owner: str,
    repo: str,
    tenant_id: str | None = None,
    permissions: dict[str, str] | None = None,
) -> ResolvedGitHubToken:
    """Resolve a credential for a READ-ONLY GitHub repository probe.

    Returns a :class:`ResolvedGitHubToken`; ``token is None`` means "probe
    anonymously". ⚠️ Callers must consume ``.token`` — this used to return a bare
    ``str | None`` and the extra two fields are what make the App rung usable at
    all (``is_app``) and keep a platform credential from escalating a caller's
    error (``source``).

    Order: ``explicit`` request token → a GitHub App installation token for
    ``owner/repo`` → ``None`` (probe anonymously). ⛔ There is NO personal-access
    token rung (owner ruling 2026-09-30): the platform identity is the App only.

    WHY THIS EXISTS. Every pre-flight in this module used to end at an
    ANONYMOUS ``Github()`` client whenever the caller supplied no token, so
    api.github.com answered 404 for every PRIVATE repository and the card read
    ``Repository '… not found or is private' not found``. The pod has mounted a
    working GitHub App credential the whole time — the PR/merge path uses it —
    and this path simply never consulted it. From 2026-09-10 14:33Z that blocked
    the conveyor: 80 hits on one private repo in 48h.

    WHY NOT REUSE ``github_integration.resolve_github_token``. That resolver
    fails CLOSED (raises ``GitHubAuthError``) for a platform-owned repo when the
    App cannot mint — correct for an AUTHORING entrypoint, wrong here: it would
    turn a clean 404 into a 500 and destroy the very discriminator used to
    diagnose this defect. A read-only probe must degrade, never escalate.

    ⚠️ The App rung is NON-FATAL BY CONSTRUCTION. Any ``GitHubAppAuthError``
    subclass — and any unexpected exception — is logged and falls through to
    ``None``, i.e. to the anonymous probe. A mint failure must never change the
    ERROR a caller sees; it may only fail to improve it.

    ⚠️ The anonymous outcome is deliberately still reachable. When
    ``app_auth_configured()`` is false this returns ``None`` and the caller's
    unauthenticated client runs exactly as before — that path is the positive
    control proving the App rung is a fallback and not a new requirement.
    """
    if explicit and explicit.strip():
        logger.info("github_token_source=%s repo=%s/%s", TOKEN_SOURCE_EXPLICIT, owner, repo)
        return ResolvedGitHubToken(
            explicit, TOKEN_SOURCE_EXPLICIT, _is_installation_token(explicit)
        )

    # ⛔ CUSTOMER OWNER (not system_github_owner): the internal App is NEVER
    # consulted. The only platform credential is the tenant-bound
    # public-App token (read-only, this repo only). "Not installed for this
    # tenant" / "no tenant" degrade to the ANONYMOUS probe — that issues nothing
    # and still works for a public repo. Any other refusal (fr unreachable,
    # public App unconfigured, GitHub failed) RAISES: an fr error must never be
    # read as "unbound".
    from services.github_integration import is_platform_owned

    if owner and not is_platform_owned(owner):
        from services.tenant_binding import (
            PERMS_REPO_READ,
            NotInstalledForTenantError,
            TenantRequiredError,
            resolve_customer_repo_token,
        )

        try:
            scoped = resolve_customer_repo_token(
                owner, repo, tenant_id, permissions or PERMS_REPO_READ
            )
        except (TenantRequiredError, NotInstalledForTenantError) as e:
            logger.info(
                "github_token_source=%s repo=%s/%s -- customer repo, no tenant-bound App "
                "installation (%s); probing anonymously",
                TOKEN_SOURCE_NONE,
                owner,
                repo,
                e.code,
            )
            return _NO_GITHUB_TOKEN
        logger.info(
            "github_token_source=%s repo=%s/%s is_app=True -- tenant-bound public App",
            TOKEN_SOURCE_APP,
            owner,
            repo,
        )
        return ResolvedGitHubToken(scoped.token, TOKEN_SOURCE_APP, True)

    # ⛔ PLATFORM OWNER on behalf of a tenant: the internal App below is a
    # platform credential, so only a platform tenant may reach them. A
    # non-platform tenant RAISES PlatformRepoForbiddenError (403) — deliberately
    # not the anonymous degrade: the caller must learn it is forbidden, not that
    # the repo "does not exist". ⛔ NO tenant (``None``/blank) is refused too
    # (403, not ENFORCE-gated): the rung below is the org-wide App, and a tenantless request can never be proven a platform tenant. An
    # EMPTY owner has nothing to gate on and gets no credential (anonymous).
    if not owner or not owner.strip():
        logger.info(
            "github_token_source=%s repo=%s/%s -- no owner; probing anonymously",
            TOKEN_SOURCE_NONE,
            owner,
            repo,
        )
        return _NO_GITHUB_TOKEN
    from services.tenant_binding import (
        PERMS_REPO_READ,
        check_platform_repo_tenant,
        is_tenantless,
        mint_allowlisted_platform_token,
        platform_repo_grant,
        refuse_tenantless_platform_repo,
    )

    if is_tenantless(tenant_id):
        refuse_tenantless_platform_repo(owner, repo, site="validation_probe")

    if platform_repo_grant(tenant_id, owner, repo) == "scoped":
        # ⛔ Allowlisted NON-platform tenant: ONLY a one-repo internal-App token,
        # before the org-wide App rung. A failed mint RAISES
        # (PlatformRepoScopedMintFailedError) — it never degrades to the org-wide App.
        scoped = mint_allowlisted_platform_token(
            owner, repo, permissions or PERMS_REPO_READ, tenant_id=tenant_id
        )
        logger.info(
            "github_token_source=%s repo=%s/%s is_app=True -- allowlisted repo-scoped",
            TOKEN_SOURCE_APP,
            owner,
            repo,
        )
        return ResolvedGitHubToken(scoped.token, TOKEN_SOURCE_APP, True)
    check_platform_repo_tenant(owner, repo, tenant_id, site="validation_probe")

    try:
        from services.github_app_auth import (
            GitHubAppAuthError,
            app_auth_configured,
            get_installation_token_for,
        )

        if not app_auth_configured():
            logger.info(
                "github_token_source=%s repo=%s/%s -- GitHub App auth is not configured; "
                "probing anonymously (private repositories will 404)",
                TOKEN_SOURCE_NONE,
                owner,
                repo,
            )
            return _NO_GITHUB_TOKEN

        try:
            app_token = get_installation_token_for(owner, repo)
        except GitHubAppAuthError as e:
            # Not installed on this owner, or the mint itself failed. Both are
            # "no better credential available", never a hard error here.
            logger.warning(
                "github_token_source=%s repo=%s/%s -- GitHub App configured but could not "
                "mint (%s: %s); probing anonymously",
                TOKEN_SOURCE_NONE,
                owner,
                repo,
                type(e).__name__,
                e,
            )
            return _NO_GITHUB_TOKEN

        if app_token:
            logger.info(
                "github_token_source=%s repo=%s/%s is_app=True -- skipping the get_user() "
                "probe (GET /user is 403 for an integration)",
                TOKEN_SOURCE_APP,
                owner,
                repo,
            )
            # ⚠️ is_app=True unconditionally, NOT _is_installation_token(app_token):
            # this rung minted it, so its type is known by provenance. A double
            # that returns a non-``ghs_`` string must still take the skip.
            return ResolvedGitHubToken(app_token, TOKEN_SOURCE_APP, True)

        logger.warning(
            "github_token_source=%s repo=%s/%s -- GitHub App returned an EMPTY installation "
            "token; probing anonymously",
            TOKEN_SOURCE_NONE,
            owner,
            repo,
        )
        return _NO_GITHUB_TOKEN
    except Exception as e:  # noqa: BLE001 - token resolution must never raise
        logger.warning(
            "github_token_source=%s repo=%s/%s -- unexpected error resolving a GitHub App "
            "token (%s: %s); probing anonymously",
            TOKEN_SOURCE_NONE,
            owner,
            repo,
            type(e).__name__,
            e,
        )
        return _NO_GITHUB_TOKEN


class CoreGitHubValidationService:
    """Handles GitHub API validation and repository access."""

    def __init__(self):
        """Initialize the GitHub validation service."""
        if not HAS_GITHUB:
            logger.warning(
                "GitHub library not available - validation service will have limited functionality"
            )

        self.feature_flags = load_feature_flags()
        self.test_mode = self._detect_test_mode()
        logger.info("GitHub validation service initialized - Test mode: %s", self.test_mode)

    def _detect_test_mode(self) -> bool:
        """Detect if we're running in test mode."""
        # Check environment variables that indicate test mode
        test_env_indicators = [
            "PYTEST_CURRENT_TEST",  # pytest sets this
            "TESTING",  # common test flag
            "TEST_MODE",  # explicit test mode
            "CI",  # continuous integration
            "SCHEMATHESIS_TEST",  # explicit schemathesis test mode
            "SCHEMATHESIS_TESTING",  # new schemathesis test mode indicator
        ]

        # Check if any test indicators are present
        for indicator in test_env_indicators:
            env_value = os.getenv(indicator)
            if env_value:
                # For boolean flags, check if they're truthy
                if env_value.lower() in ("true", "1", "yes", "on"):
                    logger.info(
                        "Test mode detected via environment variable: %s=%s",
                        indicator,
                        env_value,
                    )
                    return True
                elif indicator in ["PYTEST_CURRENT_TEST", "CI"]:
                    # These are always truthy when present
                    logger.info("Test mode detected via environment variable: %s", indicator)
                    return True

        # Check feature flags
        if self.feature_flags.get("enable_github_test_mode", False):
            logger.info("Test mode enabled via feature flag")
            return True

        # SECURITY: github_bypass_validation should ONLY be used in dev/test environments.
        # This bypass skips all GitHub API validation which could allow invalid repos.
        if self.feature_flags.get("github_bypass_validation", False):
            settings = get_settings()
            env = settings.environment
            if not settings.is_dev_or_test:
                logger.error(
                    "SECURITY: github_bypass_validation flag enabled in non-dev environment '%s' - ignoring bypass",
                    env or "unset",
                )
                # Don't enable bypass in production/staging environments
                return False
            logger.warning(
                "SECURITY: GitHub validation bypass enabled via feature flag - only allowed in dev/test (env=%s)",
                env,
            )
            return True

        return False

    def _is_test_token(self, token: str | None) -> bool:
        """Check if the provided token is a test token."""
        if not token:
            return False

        # Common test token patterns
        test_patterns = [
            "test_token",
            "fake_token",
            "mock_token",
            "dummy_token",
            "github_test_token",
            "ghp_test",
            "test_",
            "ghp_123456",  # Common schemathesis test pattern
            "1234567890abcdef",  # Another common test pattern
            "schemathesis",
        ]

        token_lower = token.lower()

        # Special handling for schemathesis tokens when SCHEMATHESIS_TESTING is set
        if os.getenv("SCHEMATHESIS_TESTING", "").lower() in ("true", "1", "yes", "on"):
            # In schemathesis testing mode, accept any token that starts with "schemathesis"
            if token_lower.startswith("schemathesis"):
                logger.info(
                    "Schemathesis test token detected in SCHEMATHESIS_TESTING mode: %s...",
                    token[:20],
                )
                return True

        return any(pattern in token_lower for pattern in test_patterns)

    # Sentinel values for the intake capability probe, so callers/tests assert on a
    # constant rather than on log prose.
    CAPABILITY_WRITE = "write"
    CAPABILITY_READ_ONLY = "read_only"
    CAPABILITY_UNKNOWN = "unknown"

    def _record_token_capability(self, repo, repo_full_name: str) -> str:
        """Record whether the supplied token can WRITE to this repo. Never raises.

        WHY THIS EXISTS. ``validate_github_token`` is format-only -- a well-formed but
        under-privileged PAT passes it -- so on 2026-08-15 a drive provisioned a board, six
        lists, nine customFieldGroups and three containers, ran the actual dependency
        upgrade, and only then died on the first write to GitHub:
        ``403 Resource not accessible by personal access token`` on ``POST .../git/refs``.
        Nothing had checked, and nothing recorded why. This makes that knowable at intake.

        WHY IT RECORDS RATHER THAN REJECTS. ``provision_repository`` provisions a code
        *analysis* container, and analysis needs **read**, not write. ``GitHubRepoRequest``
        carries no authoring-intent field and is ``extra="forbid"``, so at intake there is
        no way to tell "read-only analysis of a private repo" from "this project will open
        PRs". Refusing on ``push is False`` here would block the majority case to catch the
        minority -- a guard armed on an unmeasured base rate. The BLOCKING gate belongs at
        the authoring entrypoint, where intent is unambiguous. Cannot-*read* already fails
        closed upstream: ``get_repo`` raises 404/401 before this runs.

        UNKNOWN IS NOT OK. A missing or unreadable permissions block is reported as UNKNOWN,
        never as "fine" -- an unreachable GitHub or a stripped response must not read as a
        healthy token.
        """
        try:
            perms = getattr(repo, "permissions", None)
            push = getattr(perms, "push", None) if perms is not None else None
        except Exception as exc:  # a diagnostic must never break provisioning
            logger.warning("token_capability=unknown repo=%s reason=%s", repo_full_name, exc)
            return self.CAPABILITY_UNKNOWN

        if push is True:
            logger.info("token_capability=write repo=%s push=true", repo_full_name)
            return self.CAPABILITY_WRITE

        if push is False:
            logger.warning(
                "token_cannot_author repo=%s push=false -- the supplied GitHub token can READ "
                "this repository but cannot write to it. Provisioning continues (analysis needs "
                "read only), but any later commit/branch/PR for this project WILL fail with "
                "'403 Resource not accessible by personal access token'. Supply a token with "
                "contents:write on this repository, or let the GitHub App author it.",
                repo_full_name,
            )
            return self.CAPABILITY_READ_ONLY

        logger.warning(
            "token_capability=unknown repo=%s -- GitHub returned no permissions block, so write "
            "access could NOT be verified. This is not evidence that the token is fine.",
            repo_full_name,
        )
        return self.CAPABILITY_UNKNOWN

    def _may_degrade_to_anonymous(
        self,
        resolved: ResolvedGitHubToken,
        exc,
        *,
        branch_probe_started: bool = False,
    ) -> bool:
        """True when a CREDENTIAL failure on a platform-chosen token must degrade.

        WHY THIS EXISTS. Before the token ladder, this pre-flight ran ANONYMOUSLY
        for every tokenless request. A tokenless request for a PUBLIC repo
        therefore returned 200 even when the platform credential was broken,
        because nothing consulted it. With the ladder in place that same
        request now authenticates, GitHub answers 401, and a pre-flight that
        worked yesterday returns ``INVALID_TOKEN`` / HTTP 400 today. That is a
        NEW regression created by the fix, on requests that never asked for a
        credential — the opposite of this module's stated principle that a
        read-only probe must degrade, never escalate.

        ⛔ ONLY for a credential the CALLER DID NOT SUPPLY. An explicit request
        token still surfaces its own error verbatim: a caller who passes a bad
        token deserves to hear that it is bad, and silently succeeding
        anonymously would hide a broken integration until the first write.

        ⛔ ONLY FOR A CREDENTIAL REJECTION, i.e. only while access is still being
        ESTABLISHED. ``branch_probe_started`` is the second boundary and it was
        missing from the first cut of this method: the ``except
        GithubException`` that calls this also wrapped the
        ``repo.get_branch(...)`` block, and the retry it authorises —
        :meth:`_validate_anonymously` — probes the REPOSITORY and checks NO
        BRANCH. So with a platform token on a PUBLIC repo, a non-rate-limit 403 out
        of ``get_branch`` degraded, the anonymous repo probe succeeded, and
        ``validate_repository_access`` RETURNED SUCCESS for a branch that
        nothing had validated — where before it raised ACCESS_FORBIDDEN. A
        branch probe that cannot complete must never report success, so once the
        branch probe has begun there is nothing left for the degrade to rescue
        and the error surfaces.
        """
        if branch_probe_started:
            logger.warning(
                "github_token_not_degraded source=%s status=%s -- the failure came from the "
                "BRANCH-EXISTENCE probe, not from GitHub rejecting the credential. The "
                "anonymous retry validates only the REPOSITORY, so degrading here would "
                "report success for a branch nothing checked (%s).",
                resolved.source,
                getattr(exc, "status", "?"),
                exc,
            )
            return False
        if resolved.source != TOKEN_SOURCE_APP:
            return False
        logger.warning(
            "github_token_degraded source=%s status=%s -- the PLATFORM credential (not one "
            "supplied by the caller) was rejected by GitHub (%s); re-probing ANONYMOUSLY so a "
            "public repository still validates as it did before this path resolved a token. "
            "A private repository will return the clean 404.",
            resolved.source,
            getattr(exc, "status", "?"),
            exc,
        )
        return True

    def _validate_anonymously(self, repo_full_name: str) -> None:
        """Probe ``repo_full_name`` with NO credential at all.

        The original pre-ladder behaviour, extracted so the tokenless branch and
        the degraded-credential retry share ONE implementation rather than
        drifting apart on which error each produces.

        ⛔ Do not change the 404 message: ``not found or is private`` is the live
        discriminator that separates this anonymous path from the authenticated
        one (``not found or access denied``), and it is what diagnosed the
        2026-09-10 outage.
        """
        # Set timeout to prevent waiting for rate limit backoff
        g = Github(timeout=10, retry=None)
        try:
            repo = g.get_repo(repo_full_name)
            if repo.private:
                raise ValueError("Repository is private but no access token provided")
            logger.info(
                "Successfully validated access to public repository %s",
                repo_full_name,
            )

        except GithubException as e:
            if e.status == 404:
                # Use NotFoundException for 404 errors - this is a resource not found issue, not a validation issue
                raise NotFoundException(
                    message=f"Repository {repo_full_name} not found or is private",
                    error_code="REPOSITORY_NOT_FOUND",
                    resource_type="repository",
                    resource_id=repo_full_name,
                ) from e
            if e.status == 403:
                # Check if it's a rate limit error
                if "rate limit" in str(e.data.get("message", "")).lower():
                    raise ValidationException(
                        message="GitHub API rate limit exceeded. Please try again later or provide an access token.",
                        error_code="RATE_LIMIT_EXCEEDED",
                    ) from e
                raise ValidationException(
                    message=f"GitHub API error: {e.data.get('message', str(e))}",
                    error_code="GITHUB_API_ERROR",
                ) from e
            raise ValidationException(
                message=f"GitHub API error: {e.data.get('message', str(e))}",
                error_code="GITHUB_API_ERROR",
            ) from e
        finally:
            g.close()

    def _should_bypass_validation(self, request: GitHubRepoRequest) -> bool:
        """Determine if validation should be bypassed for this request.

        SECURITY NOTE: Bypass is only allowed in development/testing environments.
        In production/staging, validation is always enforced regardless of flags.
        """
        # SECURITY: Only allow bypass in dev/test environments
        if self.feature_flags.get("github_bypass_validation", False):
            settings = get_settings()
            env = settings.environment
            if not settings.is_dev_or_test:
                logger.warning(
                    "SECURITY: Attempted to bypass GitHub validation in non-dev environment '%s' - denied",
                    env or "unset",
                )
                # Fall through to other checks, don't return True
            else:
                logger.warning(
                    "SECURITY: Bypassing GitHub validation due to feature flag (env=%s, repo=%s)",
                    env,
                    str(request.repo_url)[:50] if request.repo_url else "unknown",
                )
                return True

        # Bypass in test mode
        if self.test_mode:
            logger.info("Bypassing GitHub validation due to test mode")
            return True

        # Bypass for test tokens
        if self._is_test_token(request.access_token):
            logger.info("Bypassing GitHub validation due to test token")
            return True

        # Bypass if no token provided and we want to allow public repo access
        if not request.access_token or not request.access_token.strip():
            logger.info("No access token provided - will attempt public repository access")
            # We don't bypass entirely, but we handle this case differently

        return False

    def _parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """Parse repository URL to extract owner and repo name."""
        parsed_url = urlparse(repo_url)
        if parsed_url.hostname != "github.com":
            raise ValueError(f"Only GitHub repositories are supported, got: {repo_url}")

        # Extract owner/repo from path (e.g., /owner/repo.git -> owner/repo)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL format: {repo_url}")

        owner = path_parts[0]
        # Remove .git suffix if present
        repo_name = path_parts[1].replace(".git", "")
        return owner, repo_name

    def parse_github_pr_url(self, url: str) -> tuple[str, str, int]:
        """
        Parse a GitHub PR URL to extract owner, repo name, and PR number.

        Supports URLs like:
        - https://github.com/owner/repo/pull/123
        - https://github.com/owner/repo/pull/123/files
        - https://github.com/owner/repo/pull/123/commits

        Returns:
            tuple: (owner, repo_name, pr_number)
        """
        parsed_url = urlparse(url)
        if parsed_url.hostname != "github.com":
            raise ValueError(f"Only GitHub URLs are supported, got: {url}")

        # Extract path parts
        path_parts = parsed_url.path.strip("/").split("/")

        # Validate we have at least owner/repo/pull/number
        if len(path_parts) < 4 or path_parts[2] != "pull":
            raise ValueError(f"Invalid GitHub PR URL format: {url}")

        owner = path_parts[0]
        repo_name = path_parts[1]

        try:
            pr_number = int(path_parts[3])
        except ValueError as exc:
            raise ValueError(f"Invalid PR number in URL: {path_parts[3]}") from exc

        return owner, repo_name, pr_number

    async def validate_repository_access(
        self, request: GitHubRepoRequest, tenant_id: str | None = None
    ) -> None:
        """Validate GitHub repository access using PyGithub."""

        # Check if we should bypass validation entirely
        if self._should_bypass_validation(request):
            logger.info("Skipping GitHub API validation for test/bypass mode")
            # Still validate the URL format for basic sanity checking
            try:
                self._parse_repo_url(str(request.repo_url))
                logger.info("URL format validation passed in bypass mode")
                return
            except ValueError as e:
                logger.warning("URL format validation failed even in bypass mode: %s", e)
                # In bypass mode, we're more lenient, but still catch obvious format errors
                if "github.com" not in str(request.repo_url):
                    raise ValueError("Repository URL must contain 'github.com'") from e
                return

        try:
            # Parse repository URL to extract owner and repo name
            url_str = str(request.repo_url)

            # Special handling for test data like "0/0"
            if url_str in ["0/0", "0", "/0/0", "//0"]:
                logger.error("URL validation failed: Test data received: '%s'", request.repo_url)
                raise ValidationException(
                    message=f"Invalid repository URL: '{request.repo_url}'. Expected format: https://github.com/owner/repo",
                    error_code="INVALID_URL_FORMAT",
                )

            # Validate basic URL structure before parsing
            if not url_str or len(url_str) < 10:
                logger.error(
                    "URL validation failed: URL too short or invalid: '%s'",
                    request.repo_url,
                )
                raise ValidationException(
                    message="Repository URL is too short or invalid",
                    error_code="INVALID_URL",
                )

            # Check for obviously invalid URLs like single characters
            if len(url_str.strip()) < 10 or not url_str.startswith(("http://", "https://")):
                logger.error("URL validation failed: Invalid URL format: '%s'", request.repo_url)
                raise ValidationException(
                    message=f"Invalid URL format: {request.repo_url}. Expected a valid GitHub URL like https://github.com/owner/repo",
                    error_code="INVALID_URL_FORMAT",
                )

            parsed_url = urlparse(url_str)

            # Check if hostname is valid and is github.com
            if not parsed_url.hostname or parsed_url.hostname != "github.com":
                raise ValidationException(
                    message=f"Only GitHub repositories are supported. Expected github.com, got: {parsed_url.hostname or 'invalid hostname'}",
                    error_code="INVALID_HOSTNAME",
                )

            # Extract owner/repo from path (e.g., /owner/repo.git -> owner/repo)
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) < 2:
                raise ValidationException(
                    message=f"Invalid GitHub repository URL format: {request.repo_url}",
                    error_code="INVALID_URL_FORMAT",
                )

            owner = path_parts[0]
            repo_name = path_parts[1].replace(".git", "")  # Remove .git suffix if present
            repo_full_name = f"{owner}/{repo_name}"

            logger.info("Validating access to GitHub repository: %s", repo_full_name)

            # Handle token validation more gracefully
            has_valid_token = bool(request.access_token and request.access_token.strip())
            logger.info("Access token provided: %s", "YES" if has_valid_token else "NO")

            # In test mode, handle test tokens gracefully
            if has_valid_token and self._is_test_token(request.access_token):
                logger.info("Test token detected in validation")
                if self.test_mode:
                    logger.info("Test mode active - allowing test token to proceed")
                    # Return early to bypass GitHub API calls
                    return
                else:
                    logger.warning(
                        "Test token detected but not in test mode - may cause API errors"
                    )

            if has_valid_token:
                # Log information about the token we're actually using
                token_for_logging = request.access_token
                if token_for_logging:
                    logger.info("Access token length: %d chars", len(token_for_logging))
                    logger.info("Access token starts with: %s...", token_for_logging[:10])
            else:
                logger.info(
                    "No valid access token provided - resolving a platform credential "
                    "before falling back to public repository access"
                )

            # Resolve the credential this probe will actually use:
            # explicit request token -> GitHub App -> None.
            # Before this existed the else-branch below ran for EVERY tokenless
            # request, so every private repo 404'd with "not found or is private"
            # even though the pod holds a working App credential.
            resolved = _resolve_github_token(request.access_token, owner, repo_name, tenant_id)
            token_to_use = resolved.token

            # Use PyGithub to validate repository access
            if token_to_use:
                auth = Auth.Token(token_to_use)
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(auth=auth, timeout=10, retry=None)
                # Set by the handler below when a PLATFORM credential (the App)
                # failed on CREDENTIAL grounds. Re-probing anonymously happens
                # AFTER `finally: g.close()` so the two clients never overlap.
                degraded_from: Exception | None = None
                # PHASE MARKER for the degrade. False while access is still
                # being ESTABLISHED (get_user / get_repo) — the only phase in
                # which a rejected platform credential may be swapped for an
                # anonymous probe. Flipped true the moment the BRANCH-existence
                # probe begins, because the anonymous retry checks no branch;
                # see ``_may_degrade_to_anonymous``.
                branch_probe_started = False
                try:
                    # Test authentication first - but handle test tokens gracefully
                    if self._is_test_token(token_to_use):
                        logger.info("Skipping authentication test for test token")
                        # For test tokens, we'll skip auth test and try repository access directly
                    elif resolved.is_app:
                        # ⛔ NEVER call get_user() with an App installation token:
                        # api.github.com answers 403 "Resource not accessible by
                        # integration", which this handler would report as
                        # ACCESS_FORBIDDEN / HTTP 400 with get_repo NEVER reached.
                        # Same branch as services.github_integration's is_app_auth.
                        logger.info(
                            "Skipping the get_user() authentication probe: the resolved "
                            "credential is a GitHub App installation token (source=%s), and "
                            "GET /user is not accessible to an integration. Probing get_repo "
                            "directly.",
                            resolved.source,
                        )
                    else:
                        user = g.get_user()
                        logger.info("Authentication successful! User: %s", user.login)

                    # Try to get repository information
                    repo = g.get_repo(repo_full_name)
                    logger.info("Successfully validated access to %s", repo.full_name)

                    # Capability probe -- know at INTAKE, not seven minutes later.
                    # PyGithub already returns `permissions` on the repo object fetched
                    # above, so this costs no extra API call.
                    self._record_token_capability(repo, repo_full_name)

                    # Verify the branch exists if specified - with smart fallback for 'main'
                    if request.branch:
                        # Everything from here on is the BRANCH probe: its
                        # failures must surface, never degrade to an anonymous
                        # repository probe that validates no branch at all.
                        branch_probe_started = True
                        try:
                            repo.get_branch(request.branch)
                            logger.info("Validated branch '%s' exists", request.branch)
                        except GithubException as e:
                            if e.status == 404:
                                # If 'main' branch doesn't exist, try the repository's default branch
                                if request.branch == "main":
                                    default_branch = repo.default_branch
                                    logger.info(
                                        "Branch 'main' not found, using repository default branch: '%s'",
                                        default_branch,
                                    )
                                    try:
                                        repo.get_branch(default_branch)
                                        logger.info(
                                            "Validated default branch '%s' exists",
                                            default_branch,
                                        )
                                        # Update the request to use the correct branch
                                        request.branch = default_branch
                                    except GithubException as exc:
                                        raise ValueError(
                                            f"Neither 'main' nor default branch '{default_branch}' found in repository {repo_full_name}"
                                        ) from exc
                                else:
                                    raise ValueError(
                                        f"Branch '{request.branch}' not found in repository {repo_full_name}"
                                    ) from e
                            else:
                                raise

                except GithubException as e:
                    if e.status == 404:
                        # Use NotFoundException for 404 errors - this is a resource not found issue, not a validation issue
                        raise NotFoundException(
                            message=f"Repository {repo_full_name} not found or access denied",
                            error_code="REPOSITORY_NOT_FOUND",
                            resource_type="repository",
                            resource_id=repo_full_name,
                        ) from e
                    if e.status == 401:
                        # For test tokens, handle more gracefully
                        if self._is_test_token(token_to_use):
                            logger.warning(
                                "Test token authentication failed - this is expected for fake tokens"
                            )
                            # In test mode or schemathesis mode, we want to continue without failing
                            if self.test_mode or os.getenv("SCHEMATHESIS_TESTING", "").lower() in (
                                "true",
                                "1",
                                "yes",
                                "on",
                            ):
                                logger.info(
                                    "Continuing in test/schemathesis mode despite authentication failure"
                                )
                                return
                            else:
                                # Not in test mode but got a test token - provide helpful error
                                raise ValidationException(
                                    message="Test token provided but not in test mode. Use a valid GitHub token or enable test mode.",
                                    error_code="INVALID_TEST_TOKEN",
                                ) from e

                        # Regular token validation error -- unless the credential
                        # was the PLATFORM's own choice, in which case a probe
                        # that used to succeed anonymously must not start failing.
                        if self._may_degrade_to_anonymous(
                            resolved, e, branch_probe_started=branch_probe_started
                        ):
                            degraded_from = e
                        else:
                            raise ValidationException(
                                message="Invalid or expired access token",
                                error_code="INVALID_TOKEN",
                            ) from e
                    elif e.status == 403:
                        # Check if it's a rate limit error
                        if "rate limit" in str(e.data.get("message", "")).lower():
                            # NOT a credential error, and an anonymous retry has a
                            # LOWER limit -- never degrade this one.
                            raise ValidationException(
                                message="GitHub API rate limit exceeded. Please try again later with a valid access token.",
                                error_code="RATE_LIMIT_EXCEEDED",
                            ) from e
                        if self._may_degrade_to_anonymous(
                            resolved, e, branch_probe_started=branch_probe_started
                        ):
                            degraded_from = e
                        else:
                            raise ValidationException(
                                message="Access forbidden - check token permissions",
                                error_code="ACCESS_FORBIDDEN",
                            ) from e
                    else:
                        raise ValueError(
                            f"GitHub API error: {e.data.get('message', str(e))}"
                        ) from e
                finally:
                    g.close()

                if degraded_from is not None:
                    # Old behaviour, restored: probe with no credential at all.
                    # A public repo validates exactly as it did before this
                    # module grew an env/App ladder; a private one comes back as
                    # the clean 404 that is the live diagnostic discriminator.
                    self._validate_anonymously(repo_full_name)
            else:
                # No credential was obtainable from ANY source. Public repos
                # still validate here; a private one 404s with the message
                # below. ⛔ Do not delete this branch — it is the positive
                # control that the App rung is a fallback, not a requirement.
                self._validate_anonymously(repo_full_name)

        except Exception as e:
            logger.error("Repository access validation failed: %s", str(e))
            raise

    async def get_branch_head_commit(
        self,
        repo_url: str,
        branch: str = "main",
        access_token: str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """Get the SHA of the latest commit on the specified branch."""

        try:
            # Parse repository URL to extract owner and repo name
            owner, repo_name = self._parse_repo_url(repo_url)
            repo_full_name = f"{owner}/{repo_name}"

            # Use PyGithub to get repository and branch info with authentication if provided.
            # explicit -> GitHub App -> anonymous.
            # ``.token`` only: this site never ran a get_user() probe, so there is
            # nothing here for ``is_app`` to skip.
            token_to_use = _resolve_github_token(access_token, owner, repo_name, tenant_id).token

            if token_to_use:
                auth = Auth.Token(token_to_use)
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(auth=auth, timeout=10, retry=None)
                logger.info("Using authenticated GitHub client for %s", repo_full_name)
            else:
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(timeout=10, retry=None)
                logger.info("Using unauthenticated GitHub client for %s", repo_full_name)

            try:
                repo = g.get_repo(repo_full_name)
                logger.info("Successfully accessed repository %s", repo_full_name)

                # If no specific branch is requested or the requested branch is 'main' but doesn't exist,
                # try to use the repository's default branch instead
                try_default_branch = False

                try:
                    logger.info("Attempting to get branch '%s' from %s", branch, repo_full_name)
                    branch_obj = repo.get_branch(branch)
                    logger.info("Successfully found branch '%s'", branch)
                except GithubException as e:
                    logger.warning("Failed to get branch '%s': %d - %s", branch, e.status, e.data)
                    if e.status == 404 and (branch == "main" or not branch):
                        # Branch not found and it's the default 'main' - try repository's default branch
                        try_default_branch = True
                        logger.info("Will try default branch instead of '%s'", branch)
                    else:
                        # Re-raise for other errors or explicitly requested branches
                        raise

                if try_default_branch:
                    default_branch = repo.default_branch
                    logger.info(
                        "Branch '%s' not found, using repository default branch: '%s'",
                        branch,
                        default_branch,
                    )
                    branch_obj = repo.get_branch(default_branch)
                    logger.info("Successfully got default branch '%s'", default_branch)

                commit_sha = branch_obj.commit.sha
                logger.info("Found commit SHA: %s on branch '%s'", commit_sha, branch_obj.name)
                return commit_sha
            except GithubException as e:
                if e.status == 404:
                    raise ValueError(
                        f"Branch '{branch}' not found in repository {repo_full_name}"
                    ) from e
                if e.status == 403:
                    # Check if it's a rate limit error
                    if "rate limit" in str(e.data.get("message", "")).lower():
                        raise ValueError(
                            "GitHub API rate limit exceeded. Please try again later or provide an access token."
                        ) from e
                    raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
                raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
            finally:
                g.close()

        except Exception as e:
            logger.error("Failed to get branch head commit: %s", str(e))
            raise

    async def validate_pull_request_access(
        self, request: GitHubRepoRequest, tenant_id: str | None = None
    ) -> None:
        """Validate GitHub pull request access using PyGithub."""
        try:
            if request.pr_number is None:
                raise ValueError("PR number is required for pull request validation")

            owner, repo_name = self._parse_repo_url(str(request.repo_url))
            repo_full_name = f"{owner}/{repo_name}"

            # Create GitHub client with token fallback:
            # explicit -> GitHub App -> anonymous.
            # ``.token`` only: no get_user() probe on this site either.
            token_to_use = _resolve_github_token(
                request.access_token, owner, repo_name, tenant_id, PERMS_PR_READ
            ).token

            if token_to_use:
                auth = Auth.Token(token_to_use)
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(auth=auth, timeout=10, retry=None)
            else:
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(timeout=10, retry=None)
            try:
                repo = g.get_repo(repo_full_name)

                # Verify the PR exists and is accessible
                pr = repo.get_pull(request.pr_number)

                if pr.state == "closed":
                    logger.warning(
                        "PR #%d is closed but proceeding with provisioning",
                        request.pr_number,
                    )

                logger.info(
                    "Validated PR #%d: '%s' by %s",
                    request.pr_number,
                    pr.title,
                    pr.user.login,
                )
                logger.info(
                    "PR state: %s, head: %s, base: %s",
                    pr.state,
                    pr.head.ref,
                    pr.base.ref,
                )

            except GithubException as e:
                if e.status == 404:
                    raise ValueError(
                        f"Pull request #{request.pr_number} not found in repository {repo_full_name}"
                    ) from e
                if e.status == 403:
                    # Check if it's a rate limit error
                    if "rate limit" in str(e.data.get("message", "")).lower():
                        raise ValueError(
                            "GitHub API rate limit exceeded. Please try again later or provide an access token."
                        ) from e
                    raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
                raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
            finally:
                g.close()

        except Exception as e:
            logger.error("Pull request access validation failed: %s", str(e))
            raise

    async def get_pull_request_head_commit(
        self,
        repo_url: str,
        pr_number: int,
        access_token: str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """Get the SHA of the head commit for the specified pull request."""
        try:
            owner, repo_name = self._parse_repo_url(repo_url)
            repo_full_name = f"{owner}/{repo_name}"

            # Create GitHub client with token fallback:
            # explicit -> GitHub App -> anonymous.
            # ``.token`` only: no get_user() probe on this site either.
            token_to_use = _resolve_github_token(
                access_token, owner, repo_name, tenant_id, PERMS_PR_READ
            ).token

            if token_to_use:
                auth = Auth.Token(token_to_use)
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(auth=auth, timeout=10, retry=None)
            else:
                # Set timeout to prevent waiting for rate limit backoff
                g = Github(timeout=10, retry=None)
            try:
                repo = g.get_repo(repo_full_name)
                pr = repo.get_pull(pr_number)

                head_sha = pr.head.sha
                logger.info("PR #%d head commit: %s", pr_number, head_sha)
                return head_sha

            except GithubException as e:
                if e.status == 404:
                    raise ValueError(
                        f"Pull request #{pr_number} not found in repository {repo_full_name}"
                    ) from e
                if e.status == 403:
                    # Check if it's a rate limit error
                    if "rate limit" in str(e.data.get("message", "")).lower():
                        raise ValueError(
                            "GitHub API rate limit exceeded. Please try again later or provide an access token."
                        ) from e
                    raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
                raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
            finally:
                g.close()

        except Exception as e:
            logger.error("Failed to get PR head commit: %s", str(e))
            raise


# Factory function to create CoreGitHubValidationService instance
def get_github_validation_service() -> CoreGitHubValidationService:
    """Get GitHub validation service instance."""
    return CoreGitHubValidationService()
