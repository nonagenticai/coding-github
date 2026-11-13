"""A GitHub refusal reaches the caller as a TYPED status, never a 500.

Live 2026-10-12 (licence-auth sweep through api.nonagentic.ai/coding):
``POST /api/repositories/get-issue {owner: nonagenticai, repo: ems, issue_number: 1}``
answered ``500 "Failed to get issue: Resource not accessible by integration: 403"``.
ems#1 is a pull request; the route mints an issues-only App token, and GitHub
refuses a PR read through the issues API without pull_requests access. That is a
permission fact about the credential, not a server fault, so it must be a 403
with ``error_code=github_app_permission_denied``.

Every issue route and the PR-status routes that open ``_github`` share this
mapping; each is driven below with the exact live GitHub body.

MUTATION (measured): delete the ``_github_refusal`` call in ``get_issue`` (or the
``_access_denied`` call in a PR-status route) and its row here goes red with 500.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from github import GithubException, UnknownObjectException

LIVE_403 = GithubException(
    403,
    {
        "message": "Resource not accessible by integration",
        "documentation_url": "https://docs.github.com/rest/issues/issues#get-an-issue",
        "status": "403",
    },
    {"x-ratelimit-remaining": "4990", "x-ratelimit-limit": "5000"},
)
OTHER_403 = GithubException(403, {"message": "Forbidden by org SAML policy"}, None)
BAD_CREDS = GithubException(401, {"message": "Bad credentials"}, None)
MISSING = UnknownObjectException(404, {"message": "Not Found"}, None)

BASE = {"owner": "nonagenticai", "repo": "ems"}
ISSUE_ROUTES = [
    ("/api/repositories/get-issue", {**BASE, "issue_number": 1}, "get_issue"),
    ("/api/repositories/list-issues", BASE, "list_issues"),
    ("/api/repositories/create-issue", {**BASE, "title": "t"}, "create_issue"),
    (
        "/api/repositories/comment-on-issue",
        {**BASE, "issue_number": 1, "body": "b"},
        "comment_on_issue",
    ),
    (
        "/api/repositories/add-issue-labels",
        {**BASE, "issue_number": 1, "labels": ["x"]},
        "add_issue_labels",
    ),
    ("/api/repositories/close-issue", {**BASE, "issue_number": 1}, "close_issue"),
]
PR_STATUS_ROUTES = [
    ("/api/repositories/pr-checks", {**BASE, "pr_number": 1}, "get_pull_request_checks"),
    ("/api/repositories/list-pull-requests", BASE, "list_pull_request_states"),
    ("/api/repositories/merge-pull-request", {**BASE, "pr_number": 1}, "merge_pull_request"),
]


class _RaisingService:
    def __init__(self, method: str, exc: Exception) -> None:
        self._method = method
        self._exc = exc

    def __getattr__(self, name: str):
        async def _call(*_a, **_k):
            raise self._exc

        return _call


@pytest.fixture
def github_raises(monkeypatch):
    """Swap ``_github`` (both import sites) for a service whose call raises ``exc``."""
    holder: dict[str, Exception] = {}

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        yield _RaisingService("any", holder["exc"])

    monkeypatch.setattr("routers.github_issue_operations._github", _fake_github)
    monkeypatch.setattr("routers.github_pr_status_operations._github", _fake_github)

    # The andon brake gates the write routes; released for this suite.
    from main import app
    from services.andon import require_andon_released

    app.dependency_overrides[require_andon_released] = lambda: None
    yield holder
    app.dependency_overrides.pop(require_andon_released, None)


def _code(resp) -> str | None:
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    details = (err or {}).get("details") or {}
    return details.get("error_code") or (err or {}).get("code")


@pytest.mark.parametrize(("url", "body", "_name"), ISSUE_ROUTES + PR_STATUS_ROUTES)
class TestGithubRefusalIsTyped:
    async def test_app_permission_403_is_a_typed_403(
        self, async_client, github_raises, url, body, _name
    ) -> None:
        github_raises["exc"] = LIVE_403
        resp = await async_client.post(url, json=body)
        assert resp.status_code == 403, resp.text
        assert _code(resp) == "github_app_permission_denied", resp.text

    async def test_other_403_stays_github_forbidden(
        self, async_client, github_raises, url, body, _name
    ) -> None:
        github_raises["exc"] = OTHER_403
        resp = await async_client.post(url, json=body)
        assert resp.status_code == 403, resp.text
        assert _code(resp) == "github_forbidden", resp.text

    async def test_bad_credentials_is_401(
        self, async_client, github_raises, url, body, _name
    ) -> None:
        github_raises["exc"] = BAD_CREDS
        resp = await async_client.post(url, json=body)
        assert resp.status_code == 401, resp.text
        assert _code(resp) == "github_unauthorized", resp.text


@pytest.mark.parametrize(("url", "body", "_name"), ISSUE_ROUTES)
async def test_issue_route_github_404_is_404_not_500(
    async_client, github_raises, url, body, _name
) -> None:
    github_raises["exc"] = MISSING
    resp = await async_client.post(url, json=body)
    assert resp.status_code == 404, resp.text


async def test_an_unknown_failure_is_still_a_500(async_client, github_raises) -> None:
    """Positive control: the mapping is for GitHub's statuses only; a genuine fault
    still answers 500, so the 403 above is not 'everything is a 403'."""
    github_raises["exc"] = RuntimeError("boom")
    resp = await async_client.post(ISSUE_ROUTES[0][0], json=ISSUE_ROUTES[0][1])
    assert resp.status_code == 500, resp.text
