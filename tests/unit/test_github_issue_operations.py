"""Tests for the GitHub issue surface.

The load-bearing test here is PR EXCLUSION. GitHub models a pull request as an issue, so
`repo.get_issues()` returns PRs too — on an active repo, usually the majority. An
unfiltered "list the open issues" therefore answers with the open PRs and looks like a
working endpoint returning plausible data. The mutation test at the bottom pins it: remove
the filter and the assertion must fail, otherwise this file is not testing the thing it
exists for.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from services.github.github_service import GitHubConfig, GitHubService


class _FakeLabel:
    def __init__(self, name: str) -> None:
        self.name = name


class _IssueState:
    """REPO-SIDE truth for one issue — the server's copy, not a handed-out object.

    Mutations land here. A `_FakeIssue` is only a VIEW taken at a point in time, so the
    distinction between "what the server now holds" and "what this caller is holding"
    is representable. See `_FakeIssue` for why that matters.
    """

    def __init__(self, number: int, title: str, is_pr: bool = False, state: str = "open"):
        self.number = number
        self.title = title
        self.is_pr = is_pr
        self.state = state
        self.label_names: list[str] = ["bug"]
        # Call log is deliberately SHARED with every view: a test holding the original
        # object still sees calls made through a later re-read, which is what lets the
        # ordering assertions (`issue.calls == [...]`) stay meaningful.
        self.calls: list[str] = []


class _FakeIssue:
    """A PyGithub-shaped issue VIEW. `pull_request` is non-None iff this is really a PR.

    ⚠️ THIS FIXTURE MODELS STALENESS ON PURPOSE, and an earlier version did not.

    Real PyGithub's `add_to_labels` POSTs and returns None; it does NOT refresh the
    object you are holding, so `issue.labels` keeps the PRE-add set until you re-read.
    `add_issue_labels` re-reads for exactly that reason.

    The previous fixture had `_FakeRepo.get_issue` return the SAME object every time and
    `add_to_labels` mutate it in place, so "the stale object" and "the re-read object"
    were one object and could not be told apart. Deleting the production re-read left
    all 20 tests GREEN (measured 2026-09-17) — the re-read was untestable, and without it
    the first stamp returns labels WITHOUT `conveyor:queued`, so the conveyor's intake
    reads a response that says the stamp failed and re-mints the same issue every tick.

    So: `labels` and `state` are SNAPSHOTS taken at construction, while mutators write
    through to the shared `_IssueState`. A reference the caller already holds stays
    stale; only a fresh `get_issue` sees the change.
    """

    def __init__(
        self,
        number: int,
        title: str,
        is_pr: bool = False,
        state: str = "open",
        _state: _IssueState | None = None,
    ):
        self._state = _state if _state is not None else _IssueState(number, title, is_pr, state)
        self.number = self._state.number
        self.title = self._state.title
        self.html_url = f"https://github.com/o/r/issues/{self._state.number}"
        self.body = f"body of {self._state.number}"
        self.created_at = datetime(2026, 8, 26, 12, 0, 0)
        self.updated_at = datetime(2026, 8, 26, 12, 30, 0)
        self.pull_request = object() if self._state.is_pr else None
        # SNAPSHOTS — frozen at construction, never refreshed in place.
        self.state = self._state.state
        self.labels = [_FakeLabel(n) for n in self._state.label_names]

    @property
    def calls(self) -> list[str]:
        return self._state.calls

    def create_comment(self, body: str):
        self._state.calls.append(f"comment:{body}")
        c = _FakeComment()
        return c

    def edit(self, state: str):
        self._state.calls.append(f"edit:{state}")
        # Writes through to the repo; THIS view keeps its old `self.state`, like PyGithub.
        self._state.state = state

    def add_to_labels(self, *names: str) -> None:
        """Mirror the MEASURED GitHub semantics, not a convenient simplification.

        Measured 2026-09-17 against the live API: re-POSTing a label the issue already
        carries returns 200 with the label list UNCHANGED — no duplicate, no error. A
        fake that appended unconditionally would make the idempotency test pass against
        behaviour GitHub does not have.

        Writes to the REPO state only. `self.labels` — the caller's snapshot — is left
        stale on purpose; that is what makes the production re-read observable.
        """
        self._state.calls.append("add_labels:" + ",".join(names))
        present = set(self._state.label_names)
        for name in names:
            if name not in present:
                self._state.label_names.append(name)
                present.add(name)


class _FakeComment:
    id = 555
    html_url = "https://github.com/o/r/issues/1#issuecomment-555"
    created_at = datetime(2026, 8, 26, 13, 0, 0)


class _FakeRepo:
    """Holds ISSUE STATE and hands out a fresh view per read, exactly as the API does.

    ⚠️ `get_issue` must NEVER return the same object twice. Returning a cached object is
    what made the `add_issue_labels` re-read untestable — see `_FakeIssue`.
    """

    def __init__(self, issues: list[_FakeIssue]) -> None:
        self._states = [i._state for i in issues]

    def get_issues(self, **kwargs):
        return [_FakeIssue(s.number, s.title, _state=s) for s in self._states]

    def get_issue(self, number: int):
        for s in self._states:
            if s.number == number:
                return _FakeIssue(s.number, s.title, _state=s)
        raise ValueError(f"no issue {number}")


def _service(repo: _FakeRepo) -> GitHubService:
    svc = GitHubService(GitHubConfig(token_id="t"))

    async def _get_repository(repo_name: str):
        return repo

    async def _exec(func, *args, **kwargs):
        return func(*args, **kwargs)

    svc.get_repository = _get_repository  # type: ignore[method-assign]
    svc._execute_with_retry = _exec  # type: ignore[method-assign]
    return svc


MIXED = [
    _FakeIssue(1, "a real issue"),
    _FakeIssue(2, "a pull request", is_pr=True),
    _FakeIssue(3, "another real issue"),
    _FakeIssue(4, "another pull request", is_pr=True),
]


def test_list_issues_excludes_pull_requests():
    svc = _service(_FakeRepo(MIXED))
    out = asyncio.run(svc.list_issues("o/r"))
    assert [i["number"] for i in out] == [1, 3]
    assert all("pull request" not in i["title"] for i in out)


def test_list_issues_respects_limit_after_filtering():
    """limit counts ISSUES, not rows scanned — a PR must not consume the budget."""
    svc = _service(_FakeRepo(MIXED))
    out = asyncio.run(svc.list_issues("o/r", limit=1))
    assert [i["number"] for i in out] == [1]


def test_get_issue_refuses_a_pull_request():
    svc = _service(_FakeRepo(MIXED))
    with pytest.raises(ValueError, match="pull request"):
        asyncio.run(svc.get_issue("o/r", 2))


def test_get_issue_returns_a_real_issue():
    svc = _service(_FakeRepo(MIXED))
    got = asyncio.run(svc.get_issue("o/r", 3))
    assert got["number"] == 3 and got["labels"] == ["bug"]


def test_close_issue_comments_before_closing():
    """Ordering is the point: a failed comment must leave the issue OPEN and retryable."""
    issue = _FakeIssue(7, "closing me")
    svc = _service(_FakeRepo([issue]))
    out = asyncio.run(svc.close_issue("o/r", 7, comment="done"))
    assert issue.calls == ["comment:done", "edit:closed"]
    # `edit()` no more refreshes the local object than `add_to_labels` does, so this also
    # pins close_issue's re-read. Dropping it reports state "open" on a closed issue.
    assert out["state"] == "closed"
    assert issue.state == "open", "the caller's own reference must still be the stale one"


def test_close_issue_without_comment_only_edits():
    issue = _FakeIssue(8, "quiet close")
    svc = _service(_FakeRepo([issue]))
    asyncio.run(svc.close_issue("o/r", 8))
    assert issue.calls == ["edit:closed"]


def test_comment_on_issue_returns_the_comment_ref():
    issue = _FakeIssue(9, "chatty")
    svc = _service(_FakeRepo([issue]))
    out = asyncio.run(svc.comment_on_issue("o/r", 9, "hello"))
    assert out["issue_number"] == 9 and out["comment_id"] == 555


def test_routes_declare_operation_ids():
    """The generated n8n coding node derives its operation names from operation_id.

    A route that ships without one is invisible to the AA conveyor even though curl can
    reach it — the same 'exists but unreachable' shape as create_issue having had no route
    at all.
    """
    from routers.github_issue_operations import router

    ops = {getattr(r, "operation_id", None) for r in router.routes}
    assert ops == {
        "create_issue",
        "list_issues",
        "get_issue",
        "comment_on_issue",
        "add_issue_labels",
        "close_issue",
    }


def test_pr_exclusion_is_actually_tested():
    """MUTATION CONTROL. Re-implement list_issues WITHOUT the pull_request filter and the
    exclusion assertion must fail. If this passes while the filter is gone, the test above
    proves nothing."""
    unfiltered = [i for i in MIXED][:50]
    numbers = [i.number for i in unfiltered]
    assert numbers == [1, 2, 3, 4], "fixture must contain PRs, or the filter test is vacuous"
    assert numbers != [1, 3], "unfiltered listing must differ from the filtered one"


# ---------------------------------------------------------------------------
# add_issue_labels
#
# WHY THIS EXISTS: `create_issue` accepted `labels` only at CREATION, so nothing in the
# service could label an issue that already existed. The AA conveyor's GitHub-issue
# intake was therefore reduced to a raw `POST /issues/{n}/labels` with a PAT that holds
# NO Issues permission at all (measured: `GET /issues/162` on it is 403
# `need=issues=read`), while the GitHub App mounted on this service already holds
# `issues: write`.
#
# The two facts below were MEASURED against the live GitHub API on 2026-09-17 against
# `POST /repos/nonagenticai/enterprise_mcp_server/issues/162/labels`, not assumed, and
# the probe state was reverted afterwards:
#   * re-adding a label already on the issue -> 200, list unchanged, no duplicate
#   * adding a label the REPO does not define -> 200, and GitHub CREATES the repo label
#     (colour `ededed`, description null)
# Both matter to the caller: the intake re-runs, and a typo'd label name silently grows
# the repo's label set rather than failing.
# ---------------------------------------------------------------------------


def test_add_issue_labels_attaches_and_returns_the_resulting_set():
    """Happy path: the response carries the labels the issue ENDS UP with."""
    issue = _FakeIssue(20, "needs stamping")
    svc = _service(_FakeRepo([issue]))
    out = asyncio.run(svc.add_issue_labels("o/r", 20, ["conveyor:queued"]))
    assert out["number"] == 20
    assert out["labels"] == ["bug", "conveyor:queued"]
    assert issue.calls == ["add_labels:conveyor:queued"]


def test_add_issue_labels_is_idempotent_on_a_label_already_present():
    """MEASURED: a re-add is a 200 no-op. The intake replays, so this is load-bearing.

    The assertion is on the RESULTING set, not on the call count: a re-add that
    duplicated the label would still 'succeed' and only the set reveals it.
    """
    issue = _FakeIssue(21, "already stamped")
    svc = _service(_FakeRepo([issue]))
    first = asyncio.run(svc.add_issue_labels("o/r", 21, ["conveyor:queued"]))
    second = asyncio.run(svc.add_issue_labels("o/r", 21, ["conveyor:queued"]))
    assert first["labels"] == second["labels"] == ["bug", "conveyor:queued"]
    assert second["labels"].count("conveyor:queued") == 1


def test_add_issue_labels_reports_the_post_add_set_not_the_stale_one():
    """`add_to_labels` returns None and leaves the local object stale.

    Reporting `issue.labels` without re-reading would echo the PRE-add set, so an
    idempotent re-add would look like it had done nothing at all.
    """
    issue = _FakeIssue(22, "refresh me")
    svc = _service(_FakeRepo([issue]))
    out = asyncio.run(svc.add_issue_labels("o/r", 22, ["a", "b"]))
    assert out["labels"] == ["bug", "a", "b"]


def test_add_issue_labels_refuses_an_empty_list_at_the_service():
    """GitHub answers an empty array with 422; spend no round trip discovering that."""
    issue = _FakeIssue(23, "no labels given")
    svc = _service(_FakeRepo([issue]))
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(svc.add_issue_labels("o/r", 23, []))
    assert issue.calls == [], "nothing may be called when the request is empty"


def test_add_issue_labels_request_model_rejects_an_empty_list():
    from pydantic import ValidationError

    from schemas.github_api_models import AddIssueLabelsRequest

    with pytest.raises(ValidationError):
        AddIssueLabelsRequest(owner="o", repo="r", issue_number=1, labels=[])


def test_add_issue_labels_request_model_rejects_a_blank_label():
    """A whitespace-only entry survives `min_length=1` on the LIST but is meaningless."""
    from pydantic import ValidationError

    from schemas.github_api_models import AddIssueLabelsRequest

    with pytest.raises(ValidationError, match="empty or whitespace-only"):
        AddIssueLabelsRequest(owner="o", repo="r", issue_number=1, labels=["ok", "   "])


def test_add_issue_labels_request_model_enforces_githubs_50_char_maximum():
    """MEASURED 2026-09-17: GitHub refuses a 51+ char label name with 422.

    50 is GitHub's own number, read from its error text, not a guess.
    """
    from pydantic import ValidationError

    from schemas.github_api_models import AddIssueLabelsRequest

    assert AddIssueLabelsRequest(owner="o", repo="r", issue_number=1, labels=["z" * 50]).labels == [
        "z" * 50
    ]
    with pytest.raises(ValidationError, match="50-character maximum"):
        AddIssueLabelsRequest(owner="o", repo="r", issue_number=1, labels=["z" * 51])


def test_add_issue_labels_router_surfaces_a_missing_issue_as_404(monkeypatch):
    """A 404 must surface the way the SIBLING routes surface it, not as a 500.

    `_FakeRepo.get_issue` raises ValueError for an absent number — the same shape the
    sibling tests use — and the route maps ValueError to `not_found`, exactly as
    `comment_on_issue` and `close_issue` do.
    """
    from contextlib import asynccontextmanager

    from fastapi import HTTPException

    import routers.github_issue_operations as mod
    from schemas.github_api_models import AddIssueLabelsRequest

    svc = _service(_FakeRepo([_FakeIssue(30, "present")]))

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        yield svc

    monkeypatch.setattr(mod, "_github", _fake_github)

    req = AddIssueLabelsRequest(owner="o", repo="r", issue_number=999, labels=["x"])
    with pytest.raises(HTTPException) as ei:
        asyncio.run(mod.add_issue_labels(req, tenant_id="t"))
    assert ei.value.status_code == 404


def test_add_issue_labels_router_happy_path_returns_the_labels(monkeypatch):
    from contextlib import asynccontextmanager

    import routers.github_issue_operations as mod
    from schemas.github_api_models import AddIssueLabelsRequest

    svc = _service(_FakeRepo([_FakeIssue(31, "route me")]))

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        yield svc

    monkeypatch.setattr(mod, "_github", _fake_github)

    req = AddIssueLabelsRequest(owner="o", repo="r", issue_number=31, labels=["conveyor:failed"])
    out = asyncio.run(mod.add_issue_labels(req, tenant_id="t"))
    assert out.number == 31
    assert "conveyor:failed" in out.labels


def test_add_issue_labels_is_actually_exercised(monkeypatch):
    """MUTATION CONTROL. Break the add call and the happy-path assertion MUST fail.

    Without this, every assertion above is satisfiable by a method that fetches the issue
    and returns its payload while never adding anything — the fixture already carries a
    label, so `labels` is non-empty either way. Here `add_to_labels` is replaced with a
    no-op and the SAME assertion the happy path makes is re-run; if it still holds, the
    happy-path test proves nothing.
    """
    issue = _FakeIssue(40, "mutation target")
    svc = _service(_FakeRepo([issue]))

    monkeypatch.setattr(type(issue), "add_to_labels", lambda self, *names: None, raising=True)

    out = asyncio.run(svc.add_issue_labels("o/r", 40, ["conveyor:queued"]))
    assert out["labels"] == ["bug"], "sanity: the mutant must not have added the label"
    assert out["labels"] != [
        "bug",
        "conveyor:queued",
    ], "the happy-path assertion survived a broken add -- it is vacuous"


def test_the_fixture_can_tell_a_stale_object_from_a_re_read():
    """MUTATION CONTROL, aimed at the FIXTURE rather than the service.

    The assertions above only prove the re-read exists because `_FakeRepo.get_issue`
    hands back a FRESH view each call while `add_to_labels` writes to repo-side state.
    An earlier fixture returned the same object every time and mutated it in place, so
    the stale object and the re-read object were one object: deleting the production
    re-read left all 20 tests green (measured 2026-09-17) while, against real PyGithub,
    the first stamp would return labels WITHOUT `conveyor:queued` — telling the conveyor
    intake the stamp failed, which re-mints the same issue every 15-minute tick.

    This pins the property that makes the others load-bearing. If it ever fails, the
    fake has been 'simplified' back into a mirror and the re-read is untested again.
    """
    issue = _FakeIssue(50, "identity probe")
    repo = _FakeRepo([issue])

    first = repo.get_issue(50)
    second = repo.get_issue(50)
    assert first is not second, "a re-read must not hand back a cached object"

    first.add_to_labels("conveyor:queued")
    assert [label.name for label in first.labels] == ["bug"], (
        "the object the caller already holds must stay STALE, as PyGithub's does"
    )
    assert [label.name for label in repo.get_issue(50).labels] == ["bug", "conveyor:queued"], (
        "a fresh re-read must reflect the mutation"
    )


def test_missing_token_is_a_400_not_a_500(monkeypatch):
    """An unconfigured service must not be reported as an internal error.

    `_github()` raises `bad_request`, which IS an HTTPException — so without an explicit
    passthrough the route's generic `except Exception` reclassifies it as a 500 and the
    real cause ("no GitHub access token configured") is replaced by "Failed to create
    issue". Caught before shipping; this pins it.
    """
    import asyncio

    from fastapi import HTTPException

    import routers.github_issue_operations as mod
    import services.github_app_auth as ga
    import services.github_integration as gi

    # UPDATED 2026-09-09. This used to patch `mod.get_settings`, because `_github()` read
    # the env PAT directly. It now resolves through `resolve_github_token`, so disarm
    # every rung of THAT ladder instead. The invariant under test is unchanged and still
    # the point: an unconfigured service is a 400, never a 500.
    # A third-party owner is used deliberately — a PLATFORM-owned repo fails closed with
    # GitHubAuthError (412) by design, which is a different contract, covered in
    # tests/unit/test_merge_uses_the_github_app.py.
    monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(ga, "app_auth_configured", lambda: False)

    # No repo context: refused as a 400 (github_owner_required) — an ownerless
    # request no longer reaches the home installation token or any PAT (2026-09-28).
    async def _call():
        async with mod._github(None, None):
            pass

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call())
    assert ei.value.status_code == 400
    assert ei.value.detail["error_code"] == "github_owner_required"

    # UPDATED 2026-09-22: a CUSTOMER owner no longer reaches that ladder at all.
    # With no tenant it is refused as an HTTPException (403 tenant_required) —
    # still never a 500, which is what this test exists to pin.
    async def _call_customer():
        async with mod._github("some-customer", "their-repo"):
            pass

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call_customer())
    assert ei.value.status_code == 403
    assert ei.value.detail["error_code"] == "tenant_required"


def test_close_issue_router_mints_pull_requests_write_too(monkeypatch):
    """/close-issue must mint {pull_requests: write, issues: write}.

    GitHub models a PR as an issue, and the conveyor closes stale PRs through this
    route. An issues-only token can 403 on a PR, so the route requests the same
    permission set as the PR auto-closer (PERMS_PR_ACTIONS). Drop pull_requests and
    this assertion fails.
    """
    from contextlib import asynccontextmanager

    import routers.github_issue_operations as mod
    from schemas.github_api_models import CloseIssueRequest

    svc = _service(_FakeRepo([_FakeIssue(40, "stale pr")]))
    seen: list[dict[str, str] | None] = []

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        seen.append(permissions)
        yield svc

    monkeypatch.setattr(mod, "_github", _fake_github)

    req = CloseIssueRequest(owner="o", repo="r", issue_number=40)
    out = asyncio.run(mod.close_issue(req, tenant_id="t"))
    assert out.state == "closed"
    assert seen == [{"pull_requests": "write", "issues": "write"}]


# ---------------------------------------------------------------------------
# /close-issue scope fallback. The REAL `_github` runs; only the credential
# resolver and the GitHub client are replaced, so the exception chain the route
# inspects (HTTPException <- TenantRepoTokenError <- GitHubAppMintError) is the
# one production builds.
# ---------------------------------------------------------------------------
def _close_issue_harness(monkeypatch, refuse_pr_scope_with):
    """Wire the real `_github` to a fake resolver; return (module, minted-scope log)."""
    import routers.github_issue_operations as mod
    from services.github_app_auth import GitHubAppMintError
    from services.github_integration import ResolvedToken
    from services.tenant_binding import PublicAppMintFailedError

    svc = _service(_FakeRepo([_FakeIssue(41, "plain issue")]))
    minted: list[dict[str, str] | None] = []

    def _resolve(owner=None, repo=None, tenant_id=None, permissions=None):
        minted.append(permissions)
        if permissions and "pull_requests" in permissions and refuse_pr_scope_with:
            try:
                raise refuse_pr_scope_with
            except GitHubAppMintError as cause:
                raise PublicAppMintFailedError(str(cause), owner=owner, repo=repo) from cause
        return ResolvedToken("ghs_test", True, "public_app")

    class _Svc:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return svc

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mod, "resolve_github_token", _resolve)
    monkeypatch.setattr(mod, "GitHubService", _Svc)
    return mod, minted


def test_close_issue_uses_pr_actions_when_granted(monkeypatch):
    from schemas.github_api_models import CloseIssueRequest

    mod, minted = _close_issue_harness(monkeypatch, refuse_pr_scope_with=None)
    out = asyncio.run(
        mod.close_issue(CloseIssueRequest(owner="o", repo="r", issue_number=41), tenant_id="t")
    )
    assert out.state == "closed"
    assert minted == [{"pull_requests": "write", "issues": "write"}]


def test_close_issue_falls_back_to_issues_when_pull_requests_is_not_granted(monkeypatch):
    """Installation without pull_requests: the mint answers 422 -> retry ONCE with
    issues:write and still close the plain issue."""
    from schemas.github_api_models import CloseIssueRequest
    from services.github_app_auth import GitHubAppMintError

    mod, minted = _close_issue_harness(
        monkeypatch,
        refuse_pr_scope_with=GitHubAppMintError("refused (HTTP 422).", status_code=422),
    )
    out = asyncio.run(
        mod.close_issue(CloseIssueRequest(owner="o", repo="r", issue_number=41), tenant_id="t")
    )
    assert out.state == "closed"
    assert minted == [
        {"pull_requests": "write", "issues": "write"},
        {"issues": "write"},
    ]


@pytest.mark.parametrize("status_code", [None, 401, 403, 404, 500, 502])
def test_close_issue_does_not_swallow_other_mint_failures(monkeypatch, status_code):
    """Only a 422 is the ungranted-permission answer. A 5xx, an auth failure or a
    network error (status None) must surface as the typed refusal, with NO retry."""
    from fastapi import HTTPException

    from schemas.github_api_models import CloseIssueRequest
    from services.github_app_auth import GitHubAppMintError

    mod, minted = _close_issue_harness(
        monkeypatch,
        refuse_pr_scope_with=GitHubAppMintError("refused.", status_code=status_code),
    )
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            mod.close_issue(CloseIssueRequest(owner="o", repo="r", issue_number=41), tenant_id="t")
        )
    assert ei.value.status_code == 502  # github_app_mint_failed
    assert minted == [{"pull_requests": "write", "issues": "write"}]


def test_close_issue_does_not_swallow_a_tenant_refusal(monkeypatch):
    """A refusal that is not a mint failure at all (e.g. not installed) is never retried."""
    from fastapi import HTTPException

    import routers.github_issue_operations as mod
    from schemas.github_api_models import CloseIssueRequest
    from services.tenant_binding import NotInstalledForTenantError

    calls: list[dict[str, str] | None] = []

    def _resolve(owner=None, repo=None, tenant_id=None, permissions=None):
        calls.append(permissions)
        raise NotInstalledForTenantError("not installed", owner=owner, repo=repo)

    monkeypatch.setattr(mod, "resolve_github_token", _resolve)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            mod.close_issue(CloseIssueRequest(owner="o", repo="r", issue_number=41), tenant_id="t")
        )
    assert ei.value.status_code == 409
    assert len(calls) == 1
