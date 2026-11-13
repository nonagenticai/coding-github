"""GitHub's REST API, faked UNDER real PyGithub, enforcing the minted permissions.

Only ``Requester.requestJsonAndCheck`` is replaced: everything above it — the
Repository / PullRequest / Commit objects, PaginatedList, PyGithub's own
exception classification (``Requester.createException``) — is the real library,
so a call the code makes that GitHub would refuse is refused here the same way.

``ENDPOINTS`` is GitHub's documented fine-grained grant per endpoint. A request
for an endpoint whose grant the code did not ask to mint answers exactly what
GitHub answered live on 2026-10-11 (``GET /commits/{sha}`` with a token minted
without ``contents``): 403 "Resource not accessible by integration".
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import patch
from urllib.parse import urlparse

from github import Auth, Github
from github.Requester import Requester

API = "https://api.github.com"
SHA = "a" * 40

_REPO = r"^/repos/[^/]+/[^/]+"
#: (path pattern, the fine-grained permission GitHub requires; None = metadata).
ENDPOINTS: list[tuple[str, str | None]] = [
    (_REPO + r"$", None),
    (_REPO + r"/pulls/\d+$", "pull_requests"),
    (_REPO + r"/commits/[^/]+$", "contents"),
    (_REPO + r"/commits/[^/]+/check-runs$", "checks"),
    (_REPO + r"/commits/[^/]+/check-suites$", "checks"),
    (_REPO + r"/commits/[^/]+/status$", "statuses"),
    (_REPO + r"/actions/workflows$", "actions"),
]

NOT_ACCESSIBLE = {
    "message": "Resource not accessible by integration",
    "documentation_url": "https://docs.github.com/rest/commits/commits#get-a-commit",
    "status": "403",
}


def required_permission(path: str) -> str | None:
    for pattern, perm in ENDPOINTS:
        if re.match(pattern, path):
            return perm
    raise AssertionError(f"unexpected GitHub call: {path}")


class FakeGitHubRest:
    """One repo, one PR; answers by path; records every path requested."""

    def __init__(
        self,
        *,
        owner: str = "nonagenticai",
        repo: str = "ems",
        number: int = 7,
        head: str = SHA,
        state: str = "open",
        merged: bool = False,
        runs: list[dict[str, Any]] | None = None,
        statuses: list[dict[str, Any]] | None = None,
        suites: list[dict[str, Any]] | None = None,
        workflows: list[dict[str, Any]] | None = None,
        fail: tuple[int, dict[str, Any], dict[str, Any]] | None = None,
    ):
        self.full = f"{owner}/{repo}"
        self.number, self.head, self.state, self.merged = number, head, state, merged
        self.runs = runs or []
        self.statuses = statuses or []
        self.suites = suites or []
        self.workflows = workflows or []
        #: (status, headers, body) answered by every non-metadata call, when set.
        self.fail = fail
        #: The permission set the code under test asked to MINT (set by ``patched``).
        self.permissions: dict[str, str] = {}
        self.paths: list[str] = []

    def answer(self, verb: str, url: str) -> tuple[dict[str, Any], Any]:
        assert verb == "GET", f"the CI-wait read must never write: {verb} {url}"
        path = urlparse(url).path
        self.paths.append(path)
        perm = required_permission(path)
        if perm is not None and perm not in self.permissions:
            raise Requester.createException(403, {}, dict(NOT_ACCESSIBLE))
        if perm is not None and self.fail is not None:
            status, headers, body = self.fail
            raise Requester.createException(status, headers, body)
        base = f"{API}/repos/{self.full}"
        if path.endswith("/check-runs"):
            return {}, {"total_count": len(self.runs), "check_runs": self.runs}
        if path.endswith("/check-suites"):
            return {}, {"total_count": len(self.suites), "check_suites": self.suites}
        if path.endswith("/status"):
            return {}, {"state": "pending", "sha": self.head, "statuses": self.statuses}
        if path.endswith("/actions/workflows"):
            return {}, {"total_count": len(self.workflows), "workflows": self.workflows}
        if "/pulls/" in path:
            return {}, {
                "url": f"{base}/pulls/{self.number}",
                "number": self.number,
                "state": self.state,
                "merged": self.merged,
                "head": {"sha": self.head, "ref": "feature"},
            }
        if "/commits/" in path:
            return {}, {"url": f"{base}/commits/{self.head}", "sha": self.head}
        return {}, {"url": base, "full_name": self.full, "name": self.full.split("/")[1]}


def check_run(rid: int, name: str, status="completed", conclusion="success", suite=1):
    return {
        "id": rid,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/x/runs/{rid}",
        "check_suite": {"id": suite},
    }


def check_suite(sid: int, status="completed", conclusion="success", runs=1, app="github-actions"):
    return {
        "id": sid,
        "app": {"slug": app},
        "status": status,
        "conclusion": conclusion,
        "latest_check_runs_count": runs,
    }


class RealPyGithubService:
    """The GitHubService surface fetch_snapshot uses, over a REAL ``Github`` client."""

    def __init__(self) -> None:
        self.gh = Github(auth=Auth.Token("ghs_fake"), retry=None)

    async def get_repository(self, name: str):
        return await asyncio.to_thread(self.gh.get_repo, name)

    async def _execute_with_retry(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)


def real_github_service() -> Any:
    """A production ``GitHubService`` whose PyGithub client is real (over ``rest``)."""
    from services.github.github_service import GitHubConfig, GitHubService

    svc = GitHubService(GitHubConfig(token_id="rung:test", token="ghs_fake"))
    real = RealPyGithubService()
    svc.get_repository = real.get_repository  # type: ignore[method-assign]
    svc._execute_with_retry = real._execute_with_retry  # type: ignore[method-assign]
    return svc


@contextmanager
def patched(
    rest: FakeGitHubRest,
    seen: list | None = None,
    *,
    seam: str = "routers.github_issue_operations._github",
    service=RealPyGithubService,
):
    """Route a ``_github`` seam to a real PyGithub client over ``rest``.

    The permission set the code passes to ``_github`` (the set it would MINT) is
    what ``rest`` enforces — so a call outside it is refused like GitHub does.
    ``seam`` is where the caller looks ``_github`` up (a router that did
    ``from routers.github_issue_operations import _github`` holds its own name).
    """

    @asynccontextmanager
    async def fake_github(owner, repo_name, tenant_id, permissions):
        if seen is not None:
            seen.append((owner, repo_name, tenant_id, dict(permissions or {})))
        rest.permissions = dict(permissions or {})
        yield service()

    def request(self, verb, url, parameters=None, headers=None, input=None, **kw):
        return rest.answer(verb, url)

    with (
        patch(seam, new=fake_github),
        patch.object(Requester, "requestJsonAndCheck", new=request),
    ):
        yield rest


def rate_limit_overview(remaining: int = 4999, limit: int = 5000, reset: int = 4070908800):
    """A REAL PyGithub ``RateLimitOverview`` (what ``Github.get_rate_limit()`` returns).

    Use this instead of ``MagicMock(core=...)``: that mock had the PRE-2.4 shape the
    code assumed, so the suite stayed green while every probe in prod raised
    ``'RateLimitOverview' object has no attribute 'core'`` (D14).
    """
    from unittest.mock import MagicMock

    from github.RateLimitOverview import RateLimitOverview

    bucket = {"limit": limit, "remaining": remaining, "used": limit - remaining, "reset": reset}
    return RateLimitOverview(MagicMock(), {}, {"resources": {"core": bucket}, "rate": bucket})
