"""
Production GitHub API integration service.
Replaces mock implementations with full GitHub API functionality.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp
    from github import Github, GithubException, RateLimitExceededException
    from github.Repository import Repository

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]

try:
    from github import Github, GithubException, RateLimitExceededException
    from github.Repository import Repository

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    Github = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    GithubException = Exception  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    RateLimitExceededException = Exception  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
    Repository = None  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
from coding_client.models.github import PRFileChange, PullRequest
from pydantic import BaseModel

from config.settings import get_settings
from services.github.secure_token_manager import (
    get_secure_token_manager,
)
from services.repo_visibility_guard import assert_repo_not_public
from utils.github_utils import parse_rate_limit_reset

logger = logging.getLogger(__name__)

# Check-run conclusions that REFUSE a merge. `skipped`/`neutral` are absent on purpose:
# they neither refuse nor count as a pass (see get_pull_request_checks).
_FAILING_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "stale", "startup_failure"}
)

# Only conveyor branches are ever deleted after a merge. A human's `feature/x`, and the
# default branch, are never touched — whatever GitHub's own auto-delete setting says.
AA_BRANCH_PREFIX = "aa/"

# ⛔ GitHub's compare endpoint (``GET /repos/{o}/{r}/compare/{base}...{head}``) returns
# AT MOST 300 changed files, all on the FIRST page — its pagination walks COMMITS, not
# files — and it sends NO truncation flag. PyGithub 2.9.1 ``Comparison.files``
# docstring: "Only the first 300 changed files"; GitHub REST docs for "Compare two
# commits" state the same 300-file response limit. So a list of exactly 300 is
# indistinguishable from a list that was cut at 300: ``compare_files`` REFUSES at the
# cap instead of returning a list that might be partial. A changed-files fence fed a
# truncated list would let a foreign change slip past it.
GITHUB_COMPARE_FILES_CAP = 300


def commit_handle(repo: Any, sha: str) -> Any:
    """A PyGithub ``Commit`` addressing ``sha`` WITHOUT ``GET /commits/{sha}``.

    ``Repository.get_commit`` completes the object eagerly, and that GET needs
    ``contents: read`` — which the check-reading permission sets (``PERMS_PR_CHECKS``,
    ``PERMS_CI_WAIT``) deliberately do not hold, so a repo-scoped token answers 403
    "Resource not accessible by integration" (live 2026-10-11). The check-run,
    check-suite and combined-status methods build their URL from ``self.url`` and
    never complete the commit, so they need only ``checks``/``statuses: read``.
    Do NOT read commit attributes (``sha``, ``files``, ...) off the handle: that
    completes it with exactly the GET this exists to avoid.
    """
    import urllib.parse

    from github.Commit import Commit

    quoted = urllib.parse.quote(sha, safe="")
    return Commit(repo.requester, url=f"{repo.url}/commits/{quoted}", completed=False)


class CompareFilesTruncatedError(Exception):
    """GitHub returned ``count`` >= :data:`GITHUB_COMPARE_FILES_CAP` changed files, so
    the list may be truncated and is withheld.

    ⚠️ Deliberately NOT a ``ValueError``: the routers map ``ValueError`` to 404, and
    "too many files to list exactly" must never read as "commit not found".
    """

    def __init__(self, count: int) -> None:
        self.count = count
        self.cap = GITHUB_COMPARE_FILES_CAP
        super().__init__(
            f"GitHub returned {count} changed files, at or above its {self.cap}-file "
            "compare cap; the list may be truncated and is not returned"
        )


def aa_head_branch_deletable(ref: object, default_branch: object, base_ref: object) -> str | None:
    """``None`` when ``ref`` may be deleted, else the reason it may not.

    Fails CLOSED: an unknown ref or an unknown default branch is never deleted.
    """
    if not isinstance(ref, str) or not ref:
        return "head_branch_unknown"
    if not ref.startswith(AA_BRANCH_PREFIX) or ref == AA_BRANCH_PREFIX:
        return "not_aa_branch"
    if not isinstance(default_branch, str) or not default_branch:
        return "default_branch_unknown"
    if ref == default_branch or ref == base_ref:
        return "is_default_or_base_branch"
    return None


class RateLimitInfo(BaseModel):
    """GitHub API rate limit information."""

    remaining: int
    reset_at: datetime
    limit: int
    used: int
    #: True when the probe itself failed and these numbers are a placeholder, not
    #: GitHub's answer. A caller must never SLEEP on a placeholder: the old code
    #: did, and every GitHub call paid ~1 s whenever the probe broke (D14).
    probe_failed: bool = False


@dataclass
class GitHubConfig:
    """GitHub service configuration."""

    token_id: str
    #: An already-resolved token, used verbatim when present.
    #:
    #: ``token_id`` is a KEY into :class:`SecureTokenManager`'s on-disk store, not a
    #: credential. Callers that resolve a token themselves (the App ladder in
    #: ``services.github_integration.resolve_github_token``) have nothing to look up:
    #: passing the token as ``token_id`` made ``retrieve_token`` search for a file named
    #: after the secret, miss, and raise -- surfacing as a 404 on a route whose real
    #: problem was authentication. Setting this field skips the store entirely; keep
    #: ``token_id`` as a non-secret provenance label (e.g. ``"rung:app"``).
    token: str | None = None
    base_url: str = "https://api.github.com"
    per_page: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0


class GitHubService:
    """Production GitHub API integration service."""

    def __init__(self, config: GitHubConfig):
        if not HAS_GITHUB:
            logger.warning(
                "GitHub library not available - GitHubService will have limited functionality"
            )

        self.config = config
        self._token_manager = get_secure_token_manager()
        self._github: Github | None = None
        self._async_client: Any = None
        self._rate_limit_lock = asyncio.Lock()
        self._last_rate_check = datetime.min.replace(tzinfo=UTC)

    async def __aenter__(self) -> GitHubService:
        if self._async_client is None:
            token = self.config.token or await self._token_manager.retrieve_token(
                self.config.token_id
            )
            if not token:
                # SCHEMATHESIS_TESTING bypass only allowed in non-production environments
                is_testing = os.getenv("SCHEMATHESIS_TESTING", "").lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
                is_production = get_settings().is_production
                if is_testing and not is_production:
                    logger.info("Using mock token for schemathesis testing")
                    token = f"schemathesis_mock_token_{self.config.token_id}"
                else:
                    raise ValueError(f"GitHub token '{self.config.token_id}' not found or expired")
            if token.startswith("schemathesis"):
                logger.info("Setting up mock GitHub client for schemathesis testing")
                assert aiohttp is not None, "aiohttp is required"
                self._async_client = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "CodeGraphHelper-Schemathesis/1.0",
                        "X-Test-Mode": "schemathesis",
                    }
                )
                self._github = None
            else:
                assert aiohttp is not None, "aiohttp is required"
                self._async_client = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "CodeGraphHelper/1.0",
                    }
                )
                if self._github is None:
                    self._github = Github(token, base_url=self.config.base_url)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

    async def check_rate_limit(self) -> RateLimitInfo:
        """Check current GitHub API rate limit status."""
        if not self._github:
            raise RuntimeError("GitHub client not initialized. Use async context manager.")
        try:
            overview = await asyncio.to_thread(self._github.get_rate_limit)
            # PyGithub >= 2.4 returns a ``RateLimitOverview``; the per-resource
            # buckets live under ``.resources`` (``.core`` on the overview itself
            # does not exist and raised AttributeError on every probe in prod).
            core = overview.resources.core
            return RateLimitInfo(
                remaining=core.remaining,
                reset_at=core.reset.replace(tzinfo=UTC),
                limit=core.limit,
                used=core.limit - core.remaining,
            )
        except Exception as e:
            # The probe failed, so we do NOT know the budget. Report a conservative
            # placeholder (remaining=0, never the old optimistic 1000) but flag it,
            # so ``_wait_for_rate_limit`` does not sleep on it: a sleep keyed on an
            # unknown is pure latency, and real exhaustion still surfaces as
            # ``RateLimitExceededException`` on the call itself, which
            # ``_execute_with_retry`` backs off on with GitHub's actual reset time.
            logger.warning(
                "Rate-limit probe failed (%s: %s); not throttling on an unknown budget",
                type(e).__name__,
                e,
            )
            return RateLimitInfo(
                remaining=0,
                reset_at=datetime.now(UTC),
                limit=5000,
                used=5000,
                probe_failed=True,
            )

    async def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exceeded."""
        async with self._rate_limit_lock:
            now = datetime.now(UTC)
            if (now - self._last_rate_check).total_seconds() < 60:
                return
            self._last_rate_check = now
            rate_info = await self.check_rate_limit()
            if rate_info.probe_failed:
                return
            if rate_info.remaining < 10:
                wait_time = (rate_info.reset_at - now).total_seconds()
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit low ({rate_info.remaining} remaining). Waiting {wait_time:.0f} seconds..."
                    )
                    await asyncio.sleep(wait_time + 1)

    async def _execute_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute function with retry logic for transient failures."""
        for attempt in range(self.config.max_retries):
            try:
                await self._wait_for_rate_limit()
                # PyGithub is synchronous; run it off the event loop so a slow
                # GitHub call doesn't block every other coroutine on this worker.
                return await asyncio.to_thread(func, *args, **kwargs)
            except RateLimitExceededException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait_time = (
                    getattr(e, "reset_time", datetime.now(UTC).timestamp() + 60)
                    - datetime.now(UTC).timestamp()
                )
                await asyncio.sleep(max(wait_time, self.config.retry_delay))
            except GithubException as e:
                if e.status >= 500 and attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
        # Loop exhausted without returning or raising (e.g. max_retries <= 0):
        # fail loudly instead of returning None and surfacing as an
        # AttributeError far from the cause.
        raise RuntimeError(f"GitHub call did not execute (max_retries={self.config.max_retries})")

    async def get_repository(self, repo_name: str) -> Repository:
        """Get repository object."""
        if self._github is None:
            raise RuntimeError("GitHub client not initialized. Use async context manager.")
        result = await self._execute_with_retry(self._github.get_repo, repo_name)
        from github.Repository import Repository

        if not isinstance(result, Repository):
            raise ValueError(f"Expected Repository object, got {type(result)}")
        return result

    async def get_repository_info(self, repo_name: str) -> dict[str, Any]:
        """Get repository metadata."""
        repo = await self.get_repository(repo_name)
        return {
            "name": repo.name,
            "full_name": repo.full_name,
            "owner": repo.owner.login,
            "description": repo.description,
            "language": repo.language,
            "default_branch": repo.default_branch,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "private": repo.private,
            "url": repo.html_url,
            "clone_url": repo.clone_url,
            "ssh_url": repo.ssh_url,
            "created_at": repo.created_at.isoformat() if repo.created_at else None,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
        }

    async def clone_repository(self, repo_url: str, target_dir: str) -> str:
        """Clone repository to local directory."""
        import git

        parts = repo_url.rstrip("/").split("/")
        repo_name = f"{parts[-2]}/{parts[-1]}"
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        token = self.config.token or await self._token_manager.retrieve_token(self.config.token_id)
        auth_url = repo_url.replace("https://", f"https://{token}@")
        try:
            # GitPython clone is synchronous and can take seconds — run it off
            # the event loop so it doesn't stall every other coroutine.
            await asyncio.to_thread(git.Repo.clone_from, auth_url, target_dir)
            logger.info("Successfully cloned %s to %s", repo_name, target_dir)
            return target_dir
        except Exception as e:
            # Mask token in error messages to prevent credential leakage
            sanitized_msg = str(e).replace(token, "***") if token else str(e)
            logger.error("Failed to clone repository: %s", sanitized_msg)
            raise type(e)(sanitized_msg) from None

    async def get_pull_request(self, repo_name: str, pr_number: int) -> PullRequest:
        """Get pull request details."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        return PullRequest(
            id=f"pr-{pr.id}",
            number=pr.number,
            title=pr.title,
            description=pr.body or "",
            url=pr.html_url,
            status=pr.state,
            created_at=pr.created_at,
            updated_at=pr.updated_at,
            merged_at=pr.merged_at,
            author=pr.user.login,
            base_branch=pr.base.ref,
            head_branch=pr.head.ref,
            labels=[label.name for label in pr.labels],
            reviewers=[reviewer.login for reviewer in pr.requested_reviewers],
        )

    async def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> PullRequest:
        """Create a new pull request."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(
            repo.create_pull,
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft,
        )
        logger.info(f"Created PR #{pr.number} in {repo_name}: {title}")
        return await self.get_pull_request(repo_name, pr.number)

    async def list_pull_request_files(self, repo_name: str, pr_number: int) -> list[PRFileChange]:
        """List all files changed in a pull request."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        files = []
        page = 0
        while True:
            pr_files = await self._execute_with_retry(pr.get_files().get_page, page)
            if not pr_files:
                break
            for file in pr_files:
                files.append(
                    PRFileChange(
                        file_path=file.filename,
                        change_type=file.status,
                        additions=file.additions,
                        deletions=file.deletions,
                        content=None,
                        old_content=None,
                        patch=file.patch or "",
                    )
                )
            page += 1
        return files

    async def read_pull_request_file(
        self, repo_name: str, pr_number: int, filepath: str, ref: str = "head"
    ) -> str:
        """Read file content from a pull request."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        if ref == "head":
            sha = pr.head.sha
        elif ref == "base":
            sha = pr.base.sha
        else:
            sha = ref
        try:
            content = await self._execute_with_retry(repo.get_contents, filepath, ref=sha)
        except GithubException as e:
            if e.status == 404:
                raise FileNotFoundError(f"File {filepath} not found in PR")
            raise
        if isinstance(content, list):
            raise ValueError(f"Path {filepath} is a directory")
        if content.encoding == "base64":
            decoded = base64.b64decode(content.content).decode("utf-8")
            return str(decoded)
        else:
            decoded = content.decoded_content.decode("utf-8")
            return str(decoded)

    async def add_pull_request_comment(
        self, repo_name: str, pr_number: int, comment: str
    ) -> dict[str, Any]:
        """Add a comment to a pull request."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        issue_comment = await self._execute_with_retry(pr.create_issue_comment, comment)
        return {
            "id": issue_comment.id,
            "body": issue_comment.body,
            "author": issue_comment.user.login,
            "created_at": issue_comment.created_at.isoformat(),
            "updated_at": issue_comment.updated_at.isoformat(),
        }

    async def add_pull_request_review_comment(
        self,
        repo_name: str,
        pr_number: int,
        comment: str,
        filepath: str,
        line: int,
        side: str = "RIGHT",
    ) -> dict[str, Any]:
        """Add a review comment on a specific line."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        review_comment = await self._execute_with_retry(
            pr.create_review_comment, body=comment, path=filepath, line=line, side=side
        )
        return {
            "id": review_comment.id,
            "body": review_comment.body,
            "author": review_comment.user.login,
            "created_at": review_comment.created_at.isoformat(),
            "updated_at": review_comment.updated_at.isoformat(),
            "filepath": filepath,
            "line": line,
        }

    async def read_file(self, repo_name: str, filepath: str, ref: str = "main") -> str:
        """Read file content from repository."""
        repo = await self.get_repository(repo_name)
        try:
            content = await self._execute_with_retry(repo.get_contents, filepath, ref=ref)
        except GithubException as e:
            if e.status == 404:
                raise FileNotFoundError(f"File {filepath} not found")
            raise
        if isinstance(content, list):
            raise ValueError(f"Path {filepath} is a directory")
        if content.encoding == "base64":
            decoded = base64.b64decode(content.content).decode("utf-8")
            return str(decoded)
        else:
            decoded = content.decoded_content.decode("utf-8")
            return str(decoded)

    async def list_files(
        self, repo_name: str, path: str = "", ref: str = "main", recursive: bool = False
    ) -> list[dict[str, Any]]:
        """List files in repository."""
        repo = await self.get_repository(repo_name)

        async def _list_contents(path: str) -> list[dict[str, Any]]:
            contents = await self._execute_with_retry(repo.get_contents, path, ref=ref)
            if not isinstance(contents, list):
                contents = [contents]
            files = []
            for content in contents:
                file_info = {
                    "path": content.path,
                    "name": content.name,
                    "type": content.type,
                    "size": content.size,
                    "sha": content.sha,
                }
                files.append(file_info)
                if recursive and content.type == "dir":
                    subfiles = await _list_contents(content.path)
                    files.extend(subfiles)
            return files

        return await _list_contents(path)

    async def create_file(
        self,
        repo_name: str,
        filepath: str,
        content: str,
        commit_message: str,
        branch: str = "main",
    ) -> dict[str, Any]:
        """Create a new file in repository."""
        repo = await self.get_repository(repo_name)
        result = await self._execute_with_retry(
            repo.create_file,
            path=filepath,
            message=commit_message,
            content=content,
            branch=branch,
        )
        return {
            "sha": result["commit"].sha,
            "url": result["content"].html_url,
            "commit": {
                "sha": result["commit"].sha,
                "message": result["commit"].message,
                "author": result["commit"].author.name,
            },
        }

    async def update_file(
        self,
        repo_name: str,
        filepath: str,
        content: str,
        commit_message: str,
        sha: str,
        branch: str = "main",
    ) -> dict[str, Any]:
        """Update an existing file in repository."""
        repo = await self.get_repository(repo_name)
        result = await self._execute_with_retry(
            repo.update_file,
            path=filepath,
            message=commit_message,
            content=content,
            sha=sha,
            branch=branch,
        )
        return {
            "sha": result["commit"].sha,
            "url": result["content"].html_url,
            "commit": {
                "sha": result["commit"].sha,
                "message": result["commit"].message,
                "author": result["commit"].author.name,
            },
        }

    async def create_branch(
        self, repo_name: str, branch_name: str, from_branch: str = "main"
    ) -> dict[str, Any]:
        """Create a new branch."""
        repo = await self.get_repository(repo_name)
        source_ref = await self._execute_with_retry(repo.get_git_ref, f"heads/{from_branch}")
        new_ref = await self._execute_with_retry(
            repo.create_git_ref,
            ref=f"refs/heads/{branch_name}",
            sha=source_ref.object.sha,
        )
        return {"re": new_ref.ref, "sha": new_ref.object.sha, "url": new_ref.url}

    async def delete_branch(self, repo_name: str, branch_name: str) -> bool:
        """Delete a branch."""
        repo = await self.get_repository(repo_name)
        try:
            ref = await self._execute_with_retry(repo.get_git_ref, f"heads/{branch_name}")
            ref.delete()
            return True
        except GithubException as e:
            if e.status == 404:
                return False
            raise

    async def create_webhook(
        self, repo_name: str, url: str, events: list[str], secret: str | None = None
    ) -> dict[str, Any]:
        """Create webhook for repository."""
        repo = await self.get_repository(repo_name)
        config = {"url": url, "content_type": "json"}
        if secret:
            config["secret"] = secret
        hook = await self._execute_with_retry(
            repo.create_hook,
            name="web",
            config=config,
            events=events,
            active=True,
        )
        return {
            "id": hook.id,
            "url": hook.config.get("url"),
            "events": hook.events,
            "active": hook.active,
        }

    async def verify_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify webhook signature."""
        import hashlib
        import hmac

        expected_signature = (
            "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        )
        return hmac.compare_digest(expected_signature, signature)

    async def create_issue(
        self, repo_name: str, title: str, body: str, labels: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a new issue in the repository."""
        repo = await self.get_repository(repo_name)
        issue_data: dict[str, Any] = {"title": title, "body": body}
        if labels:
            issue_data["labels"] = labels
        issue = await self._execute_with_retry(repo.create_issue, **issue_data)
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
        }

    def _issue_payload(self, issue: Any) -> dict[str, Any]:
        """Normalise a PyGithub Issue to the shape create_issue already returns."""
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
            "body": issue.body,
            "labels": [label.name for label in issue.labels],
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
        }

    async def list_issues(
        self,
        repo_name: str,
        state: str = "open",
        labels: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List issues, EXCLUDING pull requests.

        ⚠️ GitHub models a pull request as an issue, so `repo.get_issues()` returns PRs
        too — on an active repo they are usually the majority. Every one of them has a
        non-None `.pull_request`, which is the only reliable discriminator (the REST
        payload carries a `pull_request` key; a plain issue does not). Without this
        filter "list the open issues" silently answers with the open PRs, which reads
        as a working endpoint returning plausible data.
        """
        repo = await self.get_repository(repo_name)
        kwargs: dict[str, Any] = {"state": state}
        if labels:
            kwargs["labels"] = labels
        paginated = await self._execute_with_retry(repo.get_issues, **kwargs)

        def _collect() -> list[Any]:
            out: list[Any] = []
            for issue in paginated:
                if issue.pull_request is not None:
                    continue
                out.append(issue)
                if len(out) >= limit:
                    break
            return out

        # Pagination fetches lazily and synchronously, so walking it is another blocking
        # call and belongs off the event loop for the same reason _execute_with_retry does.
        issues = await asyncio.to_thread(_collect)
        return [self._issue_payload(i) for i in issues]

    async def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        """Fetch a single issue. Raises if the number refers to a pull request."""
        repo = await self.get_repository(repo_name)
        issue = await self._execute_with_retry(repo.get_issue, issue_number)
        if issue.pull_request is not None:
            raise ValueError(f"#{issue_number} is a pull request, not an issue")
        return self._issue_payload(issue)

    async def comment_on_issue(
        self, repo_name: str, issue_number: int, body: str
    ) -> dict[str, Any]:
        """Post a comment on an issue."""
        repo = await self.get_repository(repo_name)
        issue = await self._execute_with_retry(repo.get_issue, issue_number)
        comment = await self._execute_with_retry(issue.create_comment, body)
        return {
            "issue_number": issue_number,
            "comment_id": comment.id,
            "url": comment.html_url,
            "created_at": comment.created_at.isoformat() if comment.created_at else None,
        }

    async def add_issue_labels(
        self, repo_name: str, issue_number: int, labels: list[str]
    ) -> dict[str, Any]:
        """Add labels to an EXISTING issue, returning the issue's resulting label set.

        `create_issue` has always accepted `labels`, but only at creation. Nothing in this
        service could label an issue that already existed, so the conveyor's GitHub-issue
        intake was reduced to a raw `POST /issues/{n}/labels` with a PAT that holds no
        Issues permission at all.

        MEASURED BEHAVIOUR (2026-09-17, live against
        `POST /repos/nonagenticai/enterprise_mcp_server/issues/162/labels`) — these are
        observations, not assumptions, because the intake RE-RUNS and both of them decide
        whether a re-run is safe:

        * **Adding a label already on the issue is a 200 no-op.** The response is the
          issue's label list, unchanged, with no duplicate entry and no error. So this
          call is IDEMPOTENT and the intake may replay it freely.
        * **Adding a label the repository does not define CREATES it**, on the repository,
          with GitHub's default grey `ededed` and a null description — then attaches it.
          This is additive and silent: a typo'd label name does not fail, it permanently
          adds a new label to the repo. Callers that want "attach only if defined" must
          check `GET /repos/{o}/{r}/labels` first; this method deliberately does not, so
          that a first-run intake can mint `conveyor:queued` without a provisioning step.
        * A label name over **50** characters is refused by GitHub with 422. The request
          model rejects it locally first.
        * An unknown issue number is a GitHub **404**; an empty `labels` array is a **422**.

        Unlike `get_issue`, this does NOT refuse a pull request. GitHub models a PR as an
        issue and labelling one is a legitimate operation — the sibling MUTATORS
        (`comment_on_issue`, `close_issue`) do not refuse PRs either. The PR guard exists
        only on the READ paths, where PR data masquerading as issue data is what misleads.
        """
        if not labels:
            # GitHub answers an empty array with 422. Failing here keeps the round trip
            # unspent and makes "the caller sent nothing" distinguishable from "GitHub
            # rejected the names".
            raise ValueError("labels must not be empty")
        repo = await self.get_repository(repo_name)
        issue = await self._execute_with_retry(repo.get_issue, issue_number)
        await self._execute_with_retry(issue.add_to_labels, *labels)
        # Re-read rather than trusting the local object: add_to_labels returns None and
        # leaves `issue.labels` stale, so reporting it would echo the PRE-add set and an
        # idempotent re-add would look like it had done nothing.
        refreshed = await self._execute_with_retry(repo.get_issue, issue_number)
        return self._issue_payload(refreshed)

    async def close_issue(
        self, repo_name: str, issue_number: int, comment: str | None = None
    ) -> dict[str, Any]:
        """Close an issue, optionally leaving a closing comment first.

        The comment is posted BEFORE the state change so that a failure to comment
        leaves the issue open and the caller retryable, rather than closing it silently
        with the explanation lost.
        """
        repo = await self.get_repository(repo_name)
        issue = await self._execute_with_retry(repo.get_issue, issue_number)
        if comment:
            await self._execute_with_retry(issue.create_comment, comment)
        await self._execute_with_retry(issue.edit, state="closed")
        refreshed = await self._execute_with_retry(repo.get_issue, issue_number)
        return self._issue_payload(refreshed)

    async def get_pull_request_checks(
        self,
        repo_name: str,
        pr_number: int,
        expected_checks: int | None = None,
        required_checks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read CI state for a PR's head commit, as a THREE-VALUED verdict.

        ⛔ "NO FAILURES YET" IS NOT "PASSED". A check-run list read before the jobs have
        finished contains zero failures and reads as green. That is not a theoretical
        race: measured 2026-08-26 the org CI pool is a SINGLE node — every runner pod
        lands on `worker2` — and a PR sat at `queued=11, in_progress=0` for 25+ minutes
        while runners churned healthily. So the window in which "no failures" means
        "nothing has run" is minutes-to-hours wide, every time.

        The verdict is therefore `passed` | `failed` | `pending`, and `passed` requires
        that every check has COMPLETED. `expected_checks`, when supplied, additionally
        requires that the expected NUMBER of checks exists — otherwise a commit whose
        workflows have not been created yet reports `total == completed == 0`, which
        satisfies "all completed" vacuously and is the same green-on-nothing shape.

        ⛔ ONLY `success` COUNTS TOWARD THE FLOOR. `skipped`/`neutral` are neither failures
        nor passes: they do not refuse, and they do not count. Counting them let N skipped
        checks + 0 successes satisfy `expected_checks: N` — a required auditor that was
        skipped read as passed. The floor is `expected_checks` successes, or at least ONE
        success when no count is given (an all-skipped head has proven nothing).

        `required_checks` names checks that must each exist on the head SHA with a
        `success` conclusion (latest run per name, by id). A named check that is missing
        or still running is `pending`; one that completed with any other conclusion —
        skipped and neutral included — is `failed`.

        Every page of check runs is read: iterating the PyGithub PaginatedList follows
        the `Link: rel="next"` header to the last page.
        """
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        head_sha = pr.head.sha
        # Not repo.get_commit: that GET needs contents:read (see commit_handle).
        commit = commit_handle(repo, head_sha)
        runs = await self._execute_with_retry(commit.get_check_runs)

        def _collect() -> list[dict[str, Any]]:
            # Iterating the PaginatedList fetches EVERY page (synchronously — off-loop).
            return [
                {
                    "name": r.name,
                    "status": r.status,
                    "conclusion": r.conclusion,
                    "_id": getattr(r, "id", None),
                    "_seq": i,
                }
                for i, r in enumerate(runs)
            ]

        raw = await asyncio.to_thread(_collect)
        total = len(raw)
        completed = [c for c in raw if c["status"] == "completed"]
        failed = [c for c in completed if c["conclusion"] in _FAILING_CONCLUSIONS]
        # Only `success` is a pass. `skipped`/`neutral` do not refuse and do not count.
        succeeded = [c for c in completed if c["conclusion"] == "success"]
        floor = expected_checks if expected_checks is not None else 1

        # Required names: the LATEST run per name (highest id; list order breaks ties).
        latest: dict[str, dict[str, Any]] = {}
        for c in raw:
            prev = latest.get(c["name"])
            if prev is None or (c["_id"] or 0, c["_seq"]) >= (prev["_id"] or 0, prev["_seq"]):
                latest[c["name"]] = c
        required = list(dict.fromkeys(required_checks or []))
        missing_required = [n for n in required if n not in latest]
        running_required = [
            n for n in required if n in latest and latest[n]["status"] != "completed"
        ]
        unsatisfied_required = [
            n
            for n in required
            if n in latest
            and latest[n]["status"] == "completed"
            and latest[n]["conclusion"] != "success"
        ]

        if failed or unsatisfied_required:
            verdict = "failed"
        elif total == 0:
            verdict = "pending"  # no checks created yet — never "passed"
        elif len(completed) < total:
            verdict = "pending"
        elif missing_required or running_required:
            verdict = "pending"  # a named check has not been created / not finished
        elif len(succeeded) < floor:
            verdict = "pending"  # fewer SUCCESSES than the caller requires
        else:
            verdict = "passed"

        checks = [{k: v for k, v in c.items() if not k.startswith("_")} for c in raw]
        return {
            "pr_number": pr_number,
            "head_sha": head_sha,
            "verdict": verdict,
            "total": total,
            "completed": len(completed),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "failed_names": [c["name"] for c in failed],
            "missing_required": missing_required + running_required,
            "unsatisfied_required": unsatisfied_required,
            "checks": checks,
        }

    async def merge_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        merge_method: str = "squash",
        commit_title: str | None = None,
        commit_message: str | None = None,
        require_checks: bool = True,
        expected_checks: int | None = None,
        sha: str | None = None,
        require_up_to_date: bool = False,
        required_checks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Merge a PR, refusing by default unless CI has actually PASSED.

        `sha` pins the merge to one head commit (refused as `head_moved` otherwise, both
        by a pre-compare here and by GitHub's own 409). `require_up_to_date` refuses as
        `base_moved` when that head is behind the base branch tip.
        On a successful merge an ``aa/*`` head branch in the same repository is deleted;
        any other branch is left alone.

        `require_checks` gates on the three-valued verdict above, so a `pending` PR is
        refused just as firmly as a `failed` one. Collapsing pending into "not failed"
        here would merge on an unrun pipeline, which on a single-node CI pool is the
        common case rather than the rare one.
        """
        if require_checks:
            state = await self.get_pull_request_checks(
                repo_name,
                pr_number,
                expected_checks=expected_checks,
                required_checks=required_checks,
            )
            if state["verdict"] != "passed":
                refusal: dict[str, Any] = {
                    "merged": False,
                    "reason": f"checks_{state['verdict']}",
                    "pr_number": pr_number,
                    "checks": state,
                }
                named = state.get("missing_required", []) + state.get("unsatisfied_required", [])
                if named:
                    refusal["message"] = (
                        "required checks not successful: "
                        f"missing/running={state.get('missing_required', [])} "
                        f"non-success={state.get('unsatisfied_required', [])}"
                    )
                elif (
                    state["verdict"] == "pending"
                    and 0 < state.get("completed", 0) == state.get("total", 0)
                    and state.get("succeeded", 0) < (expected_checks or 1)
                ):
                    refusal["message"] = (
                        f"{state.get('succeeded', 0)} successful checks, "
                        f"{expected_checks or 1} required (skipped/neutral do not count)"
                    )
                return refusal
            # The verdict grades its own head snapshot; a pinned merge must not ride on
            # checks that graded a different commit.
            if sha is not None and (state.get("head_sha") or "").lower() != sha.lower():
                return {
                    "merged": False,
                    "reason": "head_moved",
                    "pr_number": pr_number,
                    "head_sha": state.get("head_sha"),
                    "checks": state,
                    "message": f"checks graded {state.get('head_sha')}, merge was pinned to {sha}",
                }

        repo = await self.get_repository(repo_name)
        # ⛔ A PUBLIC repo is never a conveyor target: refused (403 public_repo_refused)
        # before the merge or the aa/* branch delete, off the metadata just fetched.
        # Unknown visibility is a 503, never a pass.
        assert_repo_not_public(repo, repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        head_sha = getattr(getattr(pr, "head", None), "sha", None)
        if pr.merged:
            # Replay no-op: report the existing merge commit and the head that was merged,
            # so a caller that pinned `sha` can tell "my merge already happened" from
            # "someone merged a different head".
            return {
                "merged": False,
                "reason": "already_merged",
                "pr_number": pr_number,
                "sha": getattr(pr, "merge_commit_sha", None),
                "head_sha": head_sha,
            }
        if sha is not None and (head_sha or "").lower() != sha.lower():
            return {
                "merged": False,
                "reason": "head_moved",
                "pr_number": pr_number,
                "head_sha": head_sha,
                "message": f"PR head is {head_sha}, merge was pinned to {sha}",
            }
        # `mergeable` is None while GitHub computes it — a tri-state, and None is NOT False.
        if pr.mergeable is False:
            return {
                "merged": False,
                "reason": "conflicts",
                "pr_number": pr_number,
                "head_sha": head_sha,
            }
        if require_up_to_date:
            target = sha or head_sha
            comparison = await self._execute_with_retry(repo.compare, pr.base.ref, target)
            behind_by = int(getattr(comparison, "behind_by", 0) or 0)
            if behind_by > 0:
                return {
                    "merged": False,
                    "reason": "base_moved",
                    "pr_number": pr_number,
                    "head_sha": head_sha,
                    "behind_by": behind_by,
                    "message": f"{target} is {behind_by} commit(s) behind {pr.base.ref}",
                }

        merge_kwargs: dict[str, Any] = {
            "commit_title": commit_title or pr.title,
            "commit_message": commit_message or "",
            "merge_method": merge_method,
        }
        if sha is not None:
            # PyGithub asserts `sha` is a str or NotSet — never pass None.
            merge_kwargs["sha"] = sha
        try:
            result = await self._execute_with_retry(pr.merge, **merge_kwargs)
        except GithubException as e:
            # 409: head changed between our read and GitHub's merge (only possible to
            # detect this way when `sha` was forwarded). 405: not mergeable — draft,
            # branch protection, required reviews. Both are refusals, not server errors.
            reason = {409: "head_moved", 405: "merge_blocked"}.get(e.status)
            if reason is None:
                raise
            data = e.data if isinstance(e.data, dict) else {}
            return {
                "merged": False,
                "reason": reason,
                "pr_number": pr_number,
                "head_sha": head_sha,
                "message": data.get("message") or str(e),
            }
        merged = bool(getattr(result, "merged", False))
        out: dict[str, Any] = {
            "merged": merged,
            "sha": getattr(result, "sha", None),
            "message": getattr(result, "message", None),
            "pr_number": pr_number,
            "head_sha": head_sha,
        }
        if merged:
            out.update(await self._delete_merged_aa_head_branch(repo, pr, repo_name))
        return out

    async def _delete_merged_aa_head_branch(
        self, repo: Any, pr: Any, repo_name: str
    ) -> dict[str, Any]:
        """Delete the merged PR's head branch iff it is an ``aa/*`` branch.

        Best-effort: the merge already happened, so a failure here is REPORTED
        (``head_branch_delete_skipped``) and never turns the merge into an error.
        A head on a different repository (a fork) is skipped: this token was
        scoped to ``repo_name``.
        """
        head = getattr(pr, "head", None)
        ref = getattr(head, "ref", None)
        base_ref = getattr(getattr(pr, "base", None), "ref", None)
        out: dict[str, Any] = {
            "head_branch": ref if isinstance(ref, str) else None,
            "head_branch_deleted": False,
        }
        refusal = aa_head_branch_deletable(ref, getattr(repo, "default_branch", None), base_ref)
        if refusal is None:
            head_repo = getattr(getattr(head, "repo", None), "full_name", None)
            if not isinstance(head_repo, str) or head_repo.lower() != repo_name.lower():
                refusal = "head_on_other_repository"
        if refusal is not None or not isinstance(ref, str):
            out["head_branch_delete_skipped"] = refusal or "head_branch_unknown"
            return out
        try:
            deleted = await self.delete_branch(repo_name, ref)
        except Exception as e:  # the merge stands: report, never raise
            logger.warning("Merged PR head %s not deleted: %s", ref, e)
            out["head_branch_delete_skipped"] = f"delete_failed: {e}"
            return out
        out["head_branch_deleted"] = deleted
        if not deleted:
            out["head_branch_delete_skipped"] = "already_absent"
        return out

    async def get_commit_files(self, repo_name: str, commit_sha: str) -> list[dict[str, Any]]:
        """Get files changed in a specific commit."""
        repo = await self.get_repository(repo_name)
        commit = await self._execute_with_retry(repo.get_commit, commit_sha)
        files = []
        for file_info in commit.files:
            files.append(
                {
                    "filename": file_info.filename,
                    "status": file_info.status,
                    "additions": file_info.additions,
                    "deletions": file_info.deletions,
                    "changes": file_info.changes,
                    "patch": getattr(file_info, "patch", None),
                }
            )
        return files

    async def get_pull_requests(
        self, repo_name: str, state: str = "open", limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get list of pull requests from repository."""
        repo = await self.get_repository(repo_name)
        prs = await self._execute_with_retry(
            repo.get_pulls,
            state=state,
            sort="updated",
            direction="desc",
        )
        results = []
        count = 0
        for pr in prs:
            if limit and count >= limit:
                break
            results.append(
                {
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "url": pr.html_url,
                    "author": pr.user.login,
                    "created_at": pr.created_at.isoformat() if pr.created_at else None,
                    "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                    "head_branch": pr.head.ref,
                    "base_branch": pr.base.ref,
                }
            )
            count += 1
        return results

    @staticmethod
    def _pr_state(pr: Any, *, from_list: bool = False) -> dict[str, Any]:
        """The typed PR read shape shared by get-pull-request and list-pull-requests.

        ``from_list``: GitHub's LIST endpoint omits ``merged`` and ``mergeable``
        (mergeability is computed per-PR on demand). Touching either attribute on a
        PyGithub list item silently triggers a lazy completion — one extra API call PER
        PR — so for list items ``merged`` is derived from ``merged_at`` (a PR is merged
        iff it has a merge timestamp) and ``mergeable`` is reported as ``None``
        (unknown), never guessed. Callers needing mergeability read the single PR.

        ``merge_commit_sha``, ``labels`` and ``base_default_branch`` are safe on a list
        item: all ARE carried by the list payload (GitHub's ``pull-request-simple``), so
        reading them finds the attribute already set and does NOT complete. That is
        measured, not assumed — see ``TestNoLazyCompletion`` in
        tests/unit/test_github_pr_read_routes.py, which counts requester calls against
        real PyGithub and would go red if any of them ever started completing.

        ``base_default_branch`` comes from ``pr.base.repo.default_branch``, the nested
        repository object GitHub embeds in the PR payload — so the trunk comparison
        ``base != base_default_branch`` costs no second ``getRepository`` call. ⚠️ Read
        through a truthiness guard because ``pr.base.repo`` is ``None`` when the base
        repository is gone; ``None`` propagates as UNKNOWN rather than raising and
        turning an ordinary PR read into a 500. ⛔ If GitHub ever stopped embedding
        ``default_branch`` in that nested repo, reading it would trigger ONE lazy
        completion per PR — the negative control in ``TestNoLazyCompletion`` proves the
        counter can see exactly that.
        """
        merged_at = pr.merged_at
        base_repo = pr.base.repo
        return {
            "number": pr.number,
            "state": pr.state,
            "merged": merged_at is not None if from_list else bool(pr.merged),
            "merged_at": merged_at.isoformat() if merged_at else None,
            "mergeable": None if from_list else pr.mergeable,
            "merge_commit_sha": pr.merge_commit_sha or None,
            "labels": [label.name for label in pr.labels],
            "head_sha": pr.head.sha,
            "base": pr.base.ref,
            "base_default_branch": base_repo.default_branch if base_repo else None,
            "head": pr.head.ref,
            "title": pr.title,
            "url": pr.html_url,
            "draft": bool(pr.draft),
        }

    async def get_pull_request_state(self, repo_name: str, pr_number: int) -> dict[str, Any]:
        """Read one PR in the typed read shape (includes live ``mergeable``)."""
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        # Attribute reads on a fully-fetched PR are local; the helper runs off-loop anyway
        # so a lazy completion (if PyGithub ever needs one) cannot block the event loop.
        state = await asyncio.to_thread(self._pr_state, pr)
        # Same snapshot the Phase B reads take, and for the same caller:
        # landing-reconciler reads X-RateLimit-* off the raw response today. It is read
        # off the REQUESTER, not off ``pr``, so it touches no PyGithub attribute and
        # cannot trigger a lazy completion — measured in TestNoLazyCompletion.
        state["rate_limit"] = self._rate_limit_snapshot()
        return state

    @staticmethod
    def _paginate_pulls(repo: Any, params: dict[str, Any]) -> Any:
        """``GET /repos/{o}/{r}/pulls`` as a lazy PaginatedList with explicit params.

        ``Repository.get_pulls`` (PyGithub 2.9) takes no ``per_page`` — passing one is a
        TypeError — and the client-wide ``Github.per_page`` does not reach a repo whose
        requester was copied (``get_repo(lazy=True)`` copies it). Building the same
        PaginatedList ``get_pulls`` builds, with ``per_page`` in its first-page params,
        is the only form that reliably puts ``per_page`` on the wire. No I/O here.
        """
        from github.PaginatedList import PaginatedList
        from github.PullRequest import PullRequest as GHPullRequest

        return PaginatedList(GHPullRequest, repo.requester, f"{repo.url}/pulls", params)

    async def list_pull_request_states(
        self,
        repo_name: str,
        state: str = "open",
        limit: int = 30,
        head: str | None = None,
    ) -> list[dict[str, Any]]:
        """List PRs (most recently updated first) in the typed read shape.

        ``head`` is a branch filter. GitHub's API only honours ``owner:branch``; a bare
        branch name is silently IGNORED by the API (returning every PR), so a bare name
        is qualified with the repository owner here.
        """
        repo = await self.get_repository(repo_name)
        params: dict[str, Any] = {
            "state": state,
            "sort": "updated",
            "direction": "desc",
            # One request for limit<=100 instead of ceil(limit/30) at GitHub's default.
            "per_page": min(limit, 100),
        }
        if head:
            params["head"] = head if ":" in head else f"{repo_name.split('/')[0]}:{head}"
        prs = await self._execute_with_retry(self._paginate_pulls, repo, params)

        def _collect() -> list[dict[str, Any]]:
            # Iterating a PaginatedList fetches pages synchronously — keep it off-loop.
            out: list[dict[str, Any]] = []
            for pr in prs:
                if len(out) >= limit:
                    break
                out.append(self._pr_state(pr, from_list=True))
            return out

        return await asyncio.to_thread(_collect)

    # -----------------------------------------------------------------------
    # Phase B reads: repository metadata, PR changed files, Actions runs by SHA.
    # Each returns its payload WITH a rate-limit snapshot, because the n8n callers
    # these replace read GitHub's X-RateLimit-* headers off the raw response today.
    # -----------------------------------------------------------------------

    def _rate_limit_snapshot(self) -> dict[str, Any]:
        """GitHub's ``X-RateLimit-*`` from the LAST response, with NO extra API call.

        ⛔ Read off ``Github.requester`` (a plain attribute the Requester rewrites from
        every response's headers), NOT ``Github.rate_limiting`` — that PROPERTY calls
        ``get_rate_limit()`` whenever it finds its seed value, so "surface the headers
        we already have" would silently become one more request per route call, on the
        very endpoints added to stop burning quota.

        PyGithub seeds ``(-1, -1)`` / ``0`` before any response. Those seeds are mapped
        to ``None`` (unknown), never passed through: ``remaining = 0`` means an
        exhausted quota and a caller that backs off on it must not be handed that by a
        missing header.

        ⛔ The reset goes through the SHARED total conversion
        :func:`utils.github_utils.parse_rate_limit_reset`, never an inline
        ``datetime.fromtimestamp``. This method shipped in #110 with the inline form and
        it is ALREADY LIVE: a fronting layer sending the reset in MILLISECONDS made
        ``fromtimestamp`` raise ValueError, which ``get_repository_state``'s caller
        catches as ``not_found`` — so ``get-repository`` answered **404 for a repository
        that exists**. Same defect as the error path in
        ``routers.github_pr_status_operations``; one helper so it cannot be fixed in one
        place and forgotten in the other.
        """
        requester = getattr(self._github, "requester", None) if self._github else None
        pair = getattr(requester, "rate_limiting", None)
        remaining: int | None = None
        limit: int | None = None
        if isinstance(pair, tuple) and len(pair) == 2:
            raw_remaining, raw_limit = pair
            remaining = (
                raw_remaining if isinstance(raw_remaining, int) and raw_remaining >= 0 else None
            )
            limit = raw_limit if isinstance(raw_limit, int) and raw_limit >= 0 else None
        reset_epoch, reset_at = parse_rate_limit_reset(
            getattr(requester, "rate_limiting_resettime", None)
        )
        return {
            "remaining": remaining,
            "limit": limit,
            "reset_epoch": reset_epoch,
            "reset_at": reset_at,
        }

    @staticmethod
    def _repo_permissions(repo: Any) -> dict[str, bool]:
        """The acting credential's permissions, or ``{}`` when GitHub did not say.

        ⛔ An ABSENT permissions block must not be rendered as five ``false`` flags.
        GitHub omits the block for a metadata-only read; five falses would make "you
        may not push" indistinguishable from "we were not told", and a delivery sweep
        branching on ``push`` would refuse work it is in fact allowed to do.
        """
        perms = getattr(repo, "permissions", None)
        if perms is None:
            return {}
        out: dict[str, bool] = {}
        for name in ("admin", "maintain", "push", "triage", "pull"):
            value = getattr(perms, name, None)
            if isinstance(value, bool):
                out[name] = value
        return out

    async def get_repository_state(self, repo_name: str) -> dict[str, Any]:
        """Repository metadata in the typed read shape, plus a rate-limit snapshot.

        Raises whatever PyGithub raises: a 404 stays a 404 and a 401/403 stays a
        401/403. The router MUST NOT collapse them — see the module docstring of
        ``routers.github_repo_read_operations``.
        """
        repo = await self.get_repository(repo_name)

        def _shape() -> dict[str, Any]:
            return {
                "full_name": repo.full_name,
                "name": repo.name,
                "owner": repo.owner.login,
                "default_branch": repo.default_branch,
                "archived": bool(repo.archived),
                "disabled": bool(getattr(repo, "disabled", False)),
                "private": bool(repo.private),
                "visibility": getattr(repo, "visibility", None),
                "fork": bool(repo.fork),
                "description": repo.description,
                "html_url": repo.html_url,
                "permissions": self._repo_permissions(repo),
            }

        state = await asyncio.to_thread(_shape)
        state["rate_limit"] = self._rate_limit_snapshot()
        return state

    async def get_branch_head(self, repo_name: str, branch: str) -> dict[str, Any]:
        """``GET /repos/{o}/{r}/branches/{branch}``: the branch's HEAD SHA + protection.

        PyGithub's ``Repository.get_branch`` URL-quotes the name (``/`` kept as a path
        separator, which GitHub's branches endpoint accepts for ``feature/x``). Raises
        whatever PyGithub raises: a missing repo OR branch is a 404 and stays one.
        """
        repo = await self.get_repository(repo_name)
        found = await self._execute_with_retry(repo.get_branch, branch)

        def _shape() -> dict[str, Any]:
            return {
                "branch": found.name,
                "sha": found.commit.sha,
                "protected": bool(found.protected),
            }

        state = await asyncio.to_thread(_shape)
        state["rate_limit"] = self._rate_limit_snapshot()
        return state

    # GitHub compare statuses under which every commit of ``base`` is in ``head``.
    _BASE_CONTAINED_STATUSES = frozenset({"ahead", "identical"})

    async def compare_commits(self, repo_name: str, base: str, head: str) -> dict[str, Any]:
        """``GET /repos/{o}/{r}/compare/{base}...{head}``: is ``base`` in ``head``?

        ⛔ ``base_is_ancestor_of_head`` is derived from GitHub's ``status`` ONLY.
        ``ahead_by > 0`` is also true of a ``diverged`` pair, so reading it as
        "contained" would wave through a child merge that never reached the run branch.
        Raises whatever PyGithub raises: an unknown repo, sha or ref is a 404 and stays
        one — never a 200 with ``false``.
        """
        repo = await self.get_repository(repo_name)

        def _fetch() -> Any:
            # ⚠️ PyGithub's ``Repository.compare`` is LAZY: it builds a ``Comparison``
            # with no I/O and the GET fires on first attribute access. Touch ``status``
            # HERE so the request (and its 404) happens inside the retry wrapper, not
            # later in ``_shape`` where nothing would retry it.
            comparison = repo.compare(base, head)
            _ = comparison.status
            return comparison

        comparison = await self._execute_with_retry(_fetch)

        def _shape() -> dict[str, Any]:
            status = comparison.status
            return {
                "status": status,
                "ahead_by": comparison.ahead_by,
                "behind_by": comparison.behind_by,
                "merge_base_sha": comparison.merge_base_commit.sha,
                "base_is_ancestor_of_head": status in self._BASE_CONTAINED_STATUSES,
            }

        state = await asyncio.to_thread(_shape)
        state["rate_limit"] = self._rate_limit_snapshot()
        return state

    async def compare_files(self, repo_name: str, base: str, head: str) -> dict[str, Any]:
        """``GET /repos/{o}/{r}/compare/{base}...{head}``: the EXACT changed-file list.

        Same comparison fields as :meth:`compare_commits` (``base_is_ancestor_of_head``
        derived from ``status`` only) plus ``files`` sorted by filename.
        ``previous_filename`` is set only for a ``renamed`` file.

        ⛔ Raises :class:`CompareFilesTruncatedError` when GitHub returns
        ``>= GITHUB_COMPARE_FILES_CAP`` files — the list is never returned then, because
        GitHub gives no signal that it cut it. Raises whatever PyGithub raises otherwise:
        an unknown repo, sha or ref is a 404 and stays one.
        """
        repo = await self.get_repository(repo_name)

        def _fetch() -> Any:
            # ⚠️ ``Repository.compare`` is LAZY (see ``compare_commits``): touch
            # ``status`` AND ``files`` here so the GET and its 404 happen inside the
            # retry wrapper, not later in ``_shape`` where nothing would retry them.
            comparison = repo.compare(base, head)
            _ = comparison.status
            _ = comparison.files
            return comparison

        comparison = await self._execute_with_retry(_fetch)

        def _shape() -> dict[str, Any]:
            raw_files = list(comparison.files or [])
            if len(raw_files) >= GITHUB_COMPARE_FILES_CAP:
                raise CompareFilesTruncatedError(len(raw_files))
            files = sorted(
                (
                    {
                        "filename": f.filename,
                        "status": f.status,
                        "previous_filename": (
                            f.previous_filename if f.status == "renamed" else None
                        ),
                    }
                    for f in raw_files
                ),
                key=lambda item: item["filename"],
            )
            status = comparison.status
            return {
                "status": status,
                "ahead_by": comparison.ahead_by,
                "behind_by": comparison.behind_by,
                "merge_base_sha": comparison.merge_base_commit.sha,
                "base_is_ancestor_of_head": status in self._BASE_CONTAINED_STATUSES,
                "files": files,
                "files_count": len(files),
            }

        state = await asyncio.to_thread(_shape)
        state["rate_limit"] = self._rate_limit_snapshot()
        return state

    @staticmethod
    def _paginate_pr_files(repo: Any, pr: Any, per_page: int) -> Any:
        """``GET /repos/{o}/{r}/pulls/{n}/files`` as a PaginatedList with ``per_page``.

        Same reason as :meth:`_paginate_pulls`: ``PullRequest.get_files()`` takes no
        ``per_page``, and the client-wide ``Github.per_page`` does not reach a requester
        that was copied. Building the PaginatedList with ``per_page`` in its first-page
        params is the only form that reliably puts it on the wire — and without it every
        page is GitHub's default 30, so a 100-file PR costs 4 requests instead of 1.
        No I/O here.
        """
        from github.File import File
        from github.PaginatedList import PaginatedList

        return PaginatedList(File, repo.requester, f"{pr.url}/files", {"per_page": per_page})

    async def list_pull_request_file_page(
        self, repo_name: str, pr_number: int, page: int = 1, per_page: int = 100
    ) -> dict[str, Any]:
        """ONE page of a PR's changed files (1-based ``page``).

        Deliberately NOT :meth:`list_pull_request_files`, which walks every page into
        memory: delivery-sweep asks for ``per_page=100`` and wants to stop early, and a
        route that cannot be stopped turns a 3000-file PR into 30 GitHub calls the
        caller never asked for.
        """
        repo = await self.get_repository(repo_name)
        pr = await self._execute_with_retry(repo.get_pull, pr_number)
        paginated = self._paginate_pr_files(repo, pr, per_page)
        # get_page is 0-based; the wire/route contract is 1-based.
        raw = await self._execute_with_retry(paginated.get_page, page - 1)

        def _collect() -> list[dict[str, Any]]:
            return [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changes": f.changes,
                    # `or ""` would erase the difference between "no patch supplied"
                    # (binary / too large) and "an empty patch". Keep None as None.
                    "patch": getattr(f, "patch", None),
                    "sha": getattr(f, "sha", None),
                    "previous_filename": getattr(f, "previous_filename", None),
                    "blob_url": getattr(f, "blob_url", None),
                }
                for f in raw
            ]

        files = await asyncio.to_thread(_collect)
        return {
            "pr_number": pr_number,
            "page": page,
            "per_page": per_page,
            "count": len(files),
            # GitHub's files endpoint carries no total_count, so a FULL page is the
            # only evidence another page may exist. Never a promise that one does.
            "has_more": len(files) == per_page,
            "files": files,
            "rate_limit": self._rate_limit_snapshot(),
        }

    @staticmethod
    def _paginate_workflow_runs(repo: Any, params: dict[str, Any]) -> Any:
        """``GET /repos/{o}/{r}/actions/runs`` as a PaginatedList with explicit params.

        ``Repository.get_workflow_runs`` takes no ``per_page`` (PyGithub 2.9), exactly
        like ``get_pulls``. Note ``list_item="workflow_runs"``: this endpoint wraps its
        array in an object, and omitting it yields zero runs — which would read as the
        vacuous "no runs for this SHA". No I/O here.
        """
        from github.PaginatedList import PaginatedList
        from github.WorkflowRun import WorkflowRun

        return PaginatedList(
            WorkflowRun,
            repo.requester,
            f"{repo.url}/actions/runs",
            params,
            list_item="workflow_runs",
        )

    async def list_workflow_runs_for_sha(
        self, repo_name: str, head_sha: str, per_page: int = 30
    ) -> dict[str, Any]:
        """Actions runs whose head commit is ``head_sha``, newest first.

        ⛔ NOT the same question as :meth:`get_pull_request_checks`, which is keyed by
        PR number and reads CHECK RUNS. A commit can be pushed with no PR at all, and
        github-outcome-poll polls a SHA. An empty result means "GitHub has created no
        run for this SHA yet" — never "passed".
        """
        repo = await self.get_repository(repo_name)
        params: dict[str, Any] = {"head_sha": head_sha, "per_page": per_page}
        paginated = self._paginate_workflow_runs(repo, params)
        raw = await self._execute_with_retry(paginated.get_page, 0)

        def _collect() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for run in raw:
                created = getattr(run, "created_at", None)
                updated = getattr(run, "updated_at", None)
                out.append(
                    {
                        "id": run.id,
                        "name": getattr(run, "name", None),
                        "status": getattr(run, "status", None),
                        "conclusion": getattr(run, "conclusion", None),
                        "html_url": getattr(run, "html_url", None),
                        "head_sha": run.head_sha,
                        "head_branch": getattr(run, "head_branch", None),
                        "event": getattr(run, "event", None),
                        "path": getattr(run, "path", None),
                        "workflow_id": getattr(run, "workflow_id", None),
                        "run_number": getattr(run, "run_number", None),
                        "run_attempt": getattr(run, "run_attempt", None),
                        "created_at": created.isoformat() if created else None,
                        "updated_at": updated.isoformat() if updated else None,
                    }
                )
            return out

        runs = await asyncio.to_thread(_collect)
        return {
            "head_sha": head_sha,
            "count": len(runs),
            "runs": runs,
            "rate_limit": self._rate_limit_snapshot(),
        }

    async def create_pr_comment(
        self, repo_name: str, pr_number: int, comment: str
    ) -> dict[str, Any]:
        """Create a comment on a pull request."""
        return await self.add_pull_request_comment(repo_name, pr_number, comment)


class PlatformPATRefusedError(ValueError):
    """A tenantless path was asked to act on a repo outside ``system_github_owner``."""


async def get_github_service(repo_name: str) -> GitHubService:
    """GitHubService for the tenantless webhook / analysis paths — DISABLED.

    ⛔ There is no platform personal access token (owner ruling 2026-09-30).
    Platform GitHub access is the internal GitHub App, gated to platform
    TENANTS (``services.github_integration.resolve_github_token``), and nothing
    on the paths that call this carries a tenant. So this has no credential to
    offer and raises ``ValueError`` — exactly what it did in production while the
    old PAT setting was empty. A non-platform (or unparseable) ``repo_name``
    still raises :class:`PlatformPATRefusedError` first.

    The only exception is the SCHEMATHESIS_TESTING bypass (never in production),
    which substitutes a fake token so contract fuzzing can reach the handlers.
    """
    from services.github_integration import is_platform_owned

    owner = (repo_name or "").split("/", 1)[0]
    if not is_platform_owned(owner):
        raise PlatformPATRefusedError(
            f"Refusing a tenantless GitHub call for non-platform repo '{repo_name}' "
            "(customer repos are tenant-bound; no PAT fallback)."
        )
    settings = get_settings()
    token: str | None = None
    # SCHEMATHESIS_TESTING bypass only allowed in non-production environments
    is_testing = os.getenv("SCHEMATHESIS_TESTING", "").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    is_production = settings.is_production
    if is_testing and not is_production:
        logger.info("Using schemathesis token for testing")
        token = "schemathesis_test_token"
    if not token:
        raise ValueError(
            f"No GitHub credential for '{repo_name}': this path carries no tenant, so the "
            "GitHub App cannot be used, and there is no PAT fallback."
        )
    config = GitHubConfig(token_id="schemathesis", token=token)
    return GitHubService(config)
