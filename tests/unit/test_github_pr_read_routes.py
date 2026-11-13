"""Tests for the typed PR READ routes: get-pull-request and list-pull-requests.

These back the conveyor's typed ops ``github.getPullRequest`` / ``github.listPullRequests``.
The GitHub client is mocked at the same boundary as ``test_github_pr_status_operations``
(``get_repository`` + ``_execute_with_retry``), and ``_github`` is swapped so the real
router -> service -> shape path runs end to end over HTTP.

Properties pinned, each with the mutant that must turn it red:
- a GitHub 404 is a 404, not a 500 (mutant: drop ``_is_github_404`` -> 500);
- ``state`` / ``head`` / ``limit`` reach GitHub (mutant: hard-code ``state="open"``);
- list items never touch ``pr.mergeable``/``pr.merged`` (mutant: reuse the single-PR
  shape for list items -> the lazy-completion guard raises);
- the list shape costs ZERO extra API calls, counted against real PyGithub in
  ``TestNoLazyCompletion`` (mutant: read ``pr.mergeable`` in the list path -> N calls);
- 401, 403-with-quota-exhausted and 403-auth are THREE distinguishable answers on
  get-pull-request, and none of them is a 500 or a 404 (``TestAuthDiscrimination``).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from github import BadCredentialsException, GithubException, UnknownObjectException

from services.github.github_service import GitHubConfig, GitHubService

_SVC = "services.github.github_service.GitHubService"


_MISSING = object()


class _BaseRepo:
    """The repository object GitHub embeds under ``base.repo`` — carries default_branch."""

    def __init__(self, default_branch: str = "main") -> None:
        self.default_branch = default_branch


class _Ref:
    def __init__(self, ref: str, sha: str = "", repo: object = _MISSING) -> None:
        self.ref, self.sha = ref, sha
        # Default: a base repo reporting 'main'. Pass repo=None for the deleted-base-
        # repository case, which PyGithub surfaces as ``pr.base.repo is None``.
        self.repo = _BaseRepo() if repo is _MISSING else repo


class _Label:
    def __init__(self, name: str) -> None:
        self.name = name


class _PR:
    def __init__(
        self,
        number: int = 7,
        *,
        merged_at=None,
        mergeable=True,
        draft=False,
        merge_commit_sha="",
        labels=(),
        base_ref="main",
        base_repo=_MISSING,
    ):
        self.number = number
        self.state = "closed" if merged_at else "open"
        self.merged_at = merged_at
        self._merged = merged_at is not None
        self._mergeable = mergeable
        self.merge_commit_sha = merge_commit_sha
        self.labels = [_Label(n) for n in labels]
        self.head = _Ref(f"feat/x-{number}", sha=f"sha{number}")
        self.base = _Ref(base_ref, repo=base_repo)
        self.title = f"PR {number}"
        self.html_url = f"https://github.com/o/r/pull/{number}"
        self.draft = draft

    @property
    def merged(self):
        return self._merged

    @property
    def mergeable(self):
        return self._mergeable


class _ListPR(_PR):
    """A list-endpoint item: touching merged/mergeable would cost a lazy API call."""

    @property
    def merged(self):
        raise AssertionError("list item read pr.merged (lazy completion = extra API call)")

    @property
    def mergeable(self):
        raise AssertionError("list item read pr.mergeable (lazy completion = extra API call)")


class _Repo:
    def __init__(self, pr=None, pulls=None, missing=False) -> None:
        self._pr, self._pulls, self._missing = pr, pulls or [], missing
        self.get_pulls_calls: list[dict] = []

    def get_pull(self, n):
        if self._missing:
            raise UnknownObjectException(404, {"message": "Not Found"}, None)
        return self._pr

    def paginate(self, params):
        self.get_pulls_calls.append(dict(params))
        return iter(self._pulls)


class _Requester:
    """PyGithub's Requester exposes rate_limiting as a plain, header-updated attr.

    Same stand-in as ``test_github_repo_read_routes``: ``_rate_limit_snapshot`` reads
    these two attributes and nothing else, which is the whole reason it costs no call.
    """

    def __init__(self, remaining=4321, limit=5000, reset=1790000000) -> None:
        self.rate_limiting = (remaining, limit)
        self.rate_limiting_resettime = reset


@pytest.fixture
def repo(monkeypatch):
    holder: dict = {"repo": _Repo(pr=_PR())}

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        s = GitHubService(GitHubConfig(token_id="t"))
        s._github = type("G", (), {"requester": holder.get("requester", _Requester())})()

        async def _gr(_name):
            boom = holder.get("get_repository_raises")
            if boom is not None:
                raise boom
            return holder["repo"]

        async def _ex(func, *a, **k):
            return func(*a, **k)

        s.get_repository = _gr  # type: ignore[method-assign]
        s._execute_with_retry = _ex  # type: ignore[method-assign]
        # The real _paginate_pulls is covered against real PyGithub in TestPerPage.
        s._paginate_pulls = lambda r, params: r.paginate(params)  # type: ignore[method-assign]
        holder["service"] = s
        yield s

    monkeypatch.setattr("routers.github_pr_status_operations._github", _fake_github)
    return holder


GET = "/api/repositories/get-pull-request"
LIST = "/api/repositories/list-pull-requests"
KEYS = {
    "number",
    "state",
    "merged",
    "merged_at",
    "mergeable",
    "merge_commit_sha",
    "labels",
    "head_sha",
    "base",
    "base_default_branch",
    "head",
    "title",
    "url",
    "draft",
    "rate_limit",
}


class TestGetPullRequest:
    async def test_success_returns_the_typed_shape(self, async_client, repo) -> None:
        when = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
        repo["repo"] = _Repo(
            pr=_PR(
                12,
                merged_at=when,
                mergeable=None,
                draft=True,
                merge_commit_sha="mc12",
                labels=["bug", "ready"],
            )
        )
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 12})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body) == KEYS
        assert body == {
            "number": 12,
            "state": "closed",
            "merged": True,
            "merged_at": when.isoformat(),
            "mergeable": None,
            "merge_commit_sha": "mc12",
            "labels": ["bug", "ready"],
            "head_sha": "sha12",
            "base": "main",
            "base_default_branch": "main",
            "head": "feat/x-12",
            "title": "PR 12",
            "url": "https://github.com/o/r/pull/12",
            "draft": True,
            "rate_limit": {
                "remaining": 4321,
                "limit": 5000,
                "reset_epoch": 1790000000,
                "reset_at": datetime.fromtimestamp(1790000000, UTC).isoformat(),
            },
        }

    async def test_missing_pr_is_404_not_500(self, async_client, repo) -> None:
        repo["repo"] = _Repo(missing=True)
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 999})
        assert resp.status_code == 404, resp.text

    async def test_millisecond_reset_returns_the_pr_not_a_404(self, async_client, repo) -> None:
        """⛔ THE SUCCESS-PATH HALF of the same defect — shipped LIVE in #110 via
        ``GitHubService._rate_limit_snapshot``, and extended to this route by this PR.

        A fronting layer reporting the reset in MILLISECONDS made the inline
        ``datetime.fromtimestamp`` raise ValueError *inside the service*, where the
        route's ``except ValueError -> not_found`` swallowed it: **a PR that exists
        answered 404**. Nothing about the PR was missing; a header was.

        ⚠️ ``remaining``/``limit`` must still arrive — the rate-limit snapshot is the
        reason these routes exist (they replaced n8n workflows reading the headers off
        the raw response), so degrading the whole snapshot to null would be a quieter
        version of the same loss."""
        repo["requester"] = _Requester(remaining=17, limit=5000, reset=1790000000000)
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 7})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["number"] == 7
        assert body["rate_limit"] == {
            "remaining": 17,
            "limit": 5000,
            "reset_epoch": None,
            "reset_at": None,
        }

    async def test_missing_pr_number_is_422(self, async_client, repo) -> None:
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 422, resp.text

    async def test_pr_number_zero_is_422(self, async_client, repo) -> None:
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 0})
        assert resp.status_code == 422, resp.text


class TestListPullRequests:
    async def test_success_items_share_the_shape_and_skip_lazy_fields(
        self, async_client, repo
    ) -> None:
        when = datetime(2026, 9, 21, tzinfo=UTC)
        repo["repo"] = _Repo(
            pulls=[
                _ListPR(1, labels=["wip"]),
                _ListPR(2, merged_at=when, merge_commit_sha="mc2"),
            ]
        )
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        items = resp.json()["items"]
        assert [i["number"] for i in items] == [1, 2]
        assert all(set(i) == KEYS for i in items)
        assert [i["merged"] for i in items] == [False, True]
        assert all(i["mergeable"] is None for i in items)
        # merge_commit_sha/labels ARE carried by the list payload, so list items get
        # them too — an empty sha normalises to None rather than "".
        assert [i["merge_commit_sha"] for i in items] == [None, "mc2"]
        assert [i["labels"] for i in items] == [["wip"], []]
        # base.repo.default_branch is carried by the list payload too, so unlike
        # `mergeable` this is POPULATED on list items — see TestBaseDefaultBranch.
        assert [i["base_default_branch"] for i in items] == ["main", "main"]
        # Defaults reach GitHub: open, newest-updated first, no head filter.
        assert repo["repo"].get_pulls_calls == [
            {"state": "open", "sort": "updated", "direction": "desc", "per_page": 30}
        ]

    @pytest.mark.parametrize("state", ["open", "closed", "all"])
    async def test_state_filter_is_passed_through(self, async_client, repo, state) -> None:
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r", "state": state})
        assert resp.status_code == 200, resp.text
        assert repo["repo"].get_pulls_calls[0]["state"] == state

    async def test_bare_head_is_owner_qualified_and_qualified_head_passes_through(
        self, async_client, repo
    ) -> None:
        await async_client.post(LIST, json={"owner": "o", "repo": "r", "head": "feat/x"})
        await async_client.post(LIST, json={"owner": "o", "repo": "r", "head": "fork:feat/y"})
        heads = [c["head"] for c in repo["repo"].get_pulls_calls]
        assert heads == ["o:feat/x", "fork:feat/y"]

    async def test_limit_truncates(self, async_client, repo) -> None:
        repo["repo"] = _Repo(pulls=[_ListPR(n) for n in range(1, 11)])
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r", "limit": 3})
        assert resp.status_code == 200, resp.text
        assert [i["number"] for i in resp.json()["items"]] == [1, 2, 3]

    @pytest.mark.parametrize(
        "extra", [{"limit": 0}, {"limit": 101}, {"state": "merged"}, {"head": ""}]
    )
    async def test_invalid_input_is_422(self, async_client, repo, extra) -> None:
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r", **extra})
        assert resp.status_code == 422, resp.text

    async def test_missing_repo_is_404(self, async_client, repo, monkeypatch) -> None:
        def _boom(_params):
            raise UnknownObjectException(404, {"message": "Not Found"}, None)

        monkeypatch.setattr(repo["repo"], "paginate", _boom)
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "nope"})
        assert resp.status_code == 404, resp.text


class TestBaseDefaultBranch:
    """``base_default_branch`` — the field that makes the trunk comparison one call.

    Two n8n call sites evaluate ``base.ref !== base.repo.default_branch``
    (``workflows/patterns/approval-callback.json :: Check Merge Base`` and
    ``workflows/system/delivery-sweep.json :: Observe Delivery State``). Before this
    field the only ``default_branch`` on this service lived on
    ``RepositoryStateResponse``, so converting either workflow cost a second typed call
    — and both are at zero node headroom against their scorecard baselines.

    ⭐ Unlike ``mergeable`` and ``rate_limit``, this IS populated on list items, so the
    comparison works on both routes. Measured live 2026-09-24, unauthenticated:
    ``GET /repos/octocat/Hello-World/pulls`` items each carry
    ``base.repo.default_branch`` (and omit ``merged``/``mergeable``, matching what
    ``_pr_state``'s docstring already claims), as does ``GET .../pulls/11277``.

    MUTANT: delete the ``"base_default_branch"`` line from ``_pr_state`` -> the two
    populate tests below go red while ``test_a_consumer_that_ignores_the_field_is
    _unaffected`` stays green, which is the whole point of defaulting it.
    """

    async def test_single_read_populates_it_and_the_comparison_answers_same(
        self, async_client, repo
    ) -> None:
        """The trunk case: base IS the default branch -> the two agree."""
        repo["repo"] = _Repo(pr=_PR(31, base_ref="main", base_repo=_BaseRepo("main")))
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 31})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["base"] == "main"
        assert body["base_default_branch"] == "main"
        assert body["base"] == body["base_default_branch"], "trunk PR must compare equal"

    async def test_single_read_discriminates_a_non_default_base(self, async_client, repo) -> None:
        """⛔ The guard delivery-sweep needs: merged into a THROWAWAY branch.

        Without the field the comparison is unanswerable and ``merged_into_non_default``
        becomes unreachable, so this PR would be recorded as a plain ``merged`` and the
        card would ADVANCE. The field must not be the base ref echoed back."""
        repo["repo"] = _Repo(
            pr=_PR(
                32,
                merged_at=datetime(2026, 9, 24, tzinfo=UTC),
                base_ref="throwaway/integration",
                base_repo=_BaseRepo("main"),
            )
        )
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 32})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["merged"] is True
        assert body["base"] == "throwaway/integration"
        assert body["base_default_branch"] == "main"
        assert body["base"] != body["base_default_branch"], "must detect a non-default base"

    async def test_list_items_are_populated_not_null(self, async_client, repo) -> None:
        """⭐ THE ANSWER TO THE HAZARD QUESTION, asserted rather than assumed.

        ``mergeable`` is always ``null`` on list items and ``rate_limit`` is all-null
        there; this field is NOT. Reading it off a list item costs nothing — measured
        against real PyGithub in ``TestNoLazyCompletion``."""
        repo["repo"] = _Repo(
            pulls=[
                _ListPR(1, base_ref="main", base_repo=_BaseRepo("main")),
                _ListPR(2, base_ref="release/2.0", base_repo=_BaseRepo("main")),
            ]
        )
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        items = resp.json()["items"]
        assert [i["base_default_branch"] for i in items] == ["main", "main"]
        assert all(i["mergeable"] is None for i in items), "contrast: mergeable IS null here"
        assert [i["base"] != i["base_default_branch"] for i in items] == [False, True]

    async def test_a_deleted_base_repo_is_null_and_still_a_200(self, async_client, repo) -> None:
        """PyGithub reports ``pr.base.repo is None`` when the base repository is gone.

        That must degrade to ``null`` = UNKNOWN, never raise — an AttributeError in the
        shape helper would turn an ordinary PR read into a 500 for every consumer,
        including the ones that never asked for this field."""
        repo["repo"] = _Repo(pr=_PR(33, base_repo=None))
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 33})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["base_default_branch"] is None
        assert body["base"] == "main", "the rest of the shape is intact"

    def test_a_consumer_that_ignores_the_field_is_unaffected(self) -> None:
        """POSITIVE CONTROL for the whole additive claim.

        The exact pre-change (#111-era) payload — no ``base_default_branch`` key at all
        — must still validate, every existing field unchanged, and the new field default
        to ``None``. If this ever needs updating, the field stopped being additive."""
        from schemas.github_api_models import PullRequestState

        pre_change = {
            "number": 9,
            "state": "open",
            "merged": False,
            "merged_at": None,
            "mergeable": True,
            "merge_commit_sha": "abc123",
            "labels": ["bug"],
            "head_sha": "deadbeef",
            "base": "main",
            "head": "feat/x",
            "title": "PR 9",
            "url": "https://github.com/o/r/pull/9",
            "draft": False,
        }
        state = PullRequestState(**pre_change)

        assert state.base_default_branch is None, "must default, not be required"
        assert state.model_dump(exclude={"base_default_branch", "rate_limit"}) == pre_change
        assert PullRequestState.model_fields["base_default_branch"].is_required() is False


class TestResponseShapeDrift:
    """pydantic's ValidationError IS a ValueError. MUTANT: drop the ValidationError
    clause -> the ``except ValueError`` 404 branch swallows our own bug."""

    async def test_malformed_single_pr_dict_is_500_not_404(
        self, async_client, repo, monkeypatch
    ) -> None:
        async def _bad(self, repo_name, pr_number):
            return {"number": "not-an-int", "state": "open"}

        monkeypatch.setattr(f"{_SVC}.get_pull_request_state", _bad)
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 1})
        assert resp.status_code == 500, resp.text
        assert "not-an-int" not in resp.text  # values stay out of the message

    async def test_malformed_list_item_is_500_not_404(
        self, async_client, repo, monkeypatch
    ) -> None:
        async def _bad(self, repo_name, state="open", limit=30, head=None):
            return [{"number": 1}]

        monkeypatch.setattr(f"{_SVC}.list_pull_request_states", _bad)
        resp = await async_client.post(LIST, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 500, resp.text


class TestPerPage:
    """Against REAL PyGithub, no network: ``get_repo(lazy=True)`` and a PaginatedList
    are both constructed without I/O. MUTANT: pass ``per_page=`` to ``repo.get_pulls``
    (TypeError in PyGithub 2.9) or set ``Github.per_page`` (does not reach a lazy repo's
    copied requester) -> these fail. A fake-only test could not see either."""

    @pytest.mark.parametrize(("limit", "expected"), [(100, 100), (30, 30), (1, 1)])
    async def test_per_page_is_on_the_first_request(self, limit, expected) -> None:
        from github import Auth, Github

        real_repo = Github(auth=Auth.Token("t"), lazy=True).get_repo("o/r")
        s = GitHubService(GitHubConfig(token_id="t"))
        captured: list = []

        async def _gr(_name):
            return real_repo

        async def _ex(func, *a, **k):
            captured.append(func(*a, **k))
            return []  # stop before iteration would hit the network

        s.get_repository = _gr  # type: ignore[method-assign]
        s._execute_with_retry = _ex  # type: ignore[method-assign]

        await s.list_pull_request_states("o/r", state="all", limit=limit, head="feat/x")

        (plist,) = captured
        assert plist._PaginatedList__firstUrl.endswith("/repos/o/r/pulls")
        assert plist._PaginatedList__firstParams == {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": expected,
            "head": "o:feat/x",
        }


class TestAuthDiscrimination:
    """⛔ THE load-bearing group, and the named blocker for migrating
    ``workflows/system/landing-reconciler.json``'s ``Observe PR Merge State`` onto
    this route.

    That node fails the whole reconcile tick CLOSED on an auth failure, and it
    branches three ways on what GitHub said, because the three demand three different
    operator actions:

    * **401** -> the credential is bad. Rotate it.
    * **403 + ``X-RateLimit-Remaining: 0``** -> the credential is FINE and the hour's
      quota is spent. ⛔ DO NOT ROTATE — wait for the reset. This is the assertion the
      group exists for.
    * **403 without it** -> a real permission denial. Rotating may help.

    Before this change every one of them fell through ``except Exception ->
    internal_error`` and arrived as one opaque **500**, so the migration would have
    collapsed all three into "observe_failed" — and a card merged to a side branch
    would grade as landed and move to DONE (DEFECT 4).

    MUTANTS this group must refuse, each verified RED:
    1. drop the ``_access_denied`` call from the handler -> 401 is a 500 again;
    2. collapse the rate-limit branch into the generic auth branch -> the two 403s
       become indistinguishable;
    3. fold 401/403 into ``not_found`` -> the refusal reads as ``pr_not_found``.
    """

    HEADERS_EXHAUSTED = {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1790000000",
    }
    # A real 403 permission denial still carries the headers — with quota LEFT.
    HEADERS_HEALTHY = {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Remaining": "4987",
        "X-RateLimit-Reset": "1790000000",
    }
    # Same exhausted quota, but the reset arrives in MILLISECONDS — which edge proxies
    # and GHE fronting layers do send. See TestRateLimitResetIsUnparseable.
    HEADERS_EXHAUSTED_MS = {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1790000000000",
    }

    @staticmethod
    async def _get(async_client, repo, exc):
        repo["get_repository_raises"] = exc
        resp = await async_client.post(GET, json={"owner": "o", "repo": "r", "pr_number": 7})
        # The app's exception handler re-envelopes HTTPException.detail as
        # {"error": {..., "details": {...}}} — which is exactly why _access_denied
        # repeats error_code INSIDE details: the top-level "code" is only "HTTP_403".
        details = resp.json().get("error", {}).get("details", {})
        return resp.status_code, details

    async def test_bad_credentials_is_401_not_500_and_not_404(self, async_client, repo) -> None:
        status, details = await self._get(
            async_client, repo, BadCredentialsException(401, {"message": "Bad"}, None)
        )
        assert status == 401, details
        assert details["error_code"] == "github_unauthorized"

    async def test_exhausted_quota_says_do_not_rotate(self, async_client, repo) -> None:
        """⭐ THE assertion. A 403 whose headers say the quota is spent must be
        reported as a rate limit, carrying the reset, and must say rotation is not the
        remedy."""
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {"message": "API rate limit exceeded"}, self.HEADERS_EXHAUSTED),
        )
        assert status == 403, details
        assert details["error_code"] == "github_rate_limited"
        assert details["credential_rotation_would_not_help"] is True
        assert details["rate_limit"]["remaining"] == 0
        assert details["rate_limit"]["limit"] == 5000
        assert details["rate_limit"]["reset_epoch"] == 1790000000
        assert (
            details["rate_limit"]["reset_at"] == datetime.fromtimestamp(1790000000, UTC).isoformat()
        )

    async def test_permission_denial_is_not_reported_as_a_rate_limit(
        self, async_client, repo
    ) -> None:
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {"message": "Resource not accessible"}, self.HEADERS_HEALTHY),
        )
        assert status == 403, details
        assert details["error_code"] == "github_forbidden"
        assert details.get("credential_rotation_would_not_help") is None

    async def test_403_without_any_rate_limit_headers_is_a_plain_denial(
        self, async_client, repo
    ) -> None:
        """⛔ Absent headers are NOT ``remaining == 0``. A 403 that says nothing about
        the quota must not be reported as an exhausted quota — that would tell the
        caller to sit out the hour on what is actually a missing scope."""
        status, details = await self._get(
            async_client, repo, GithubException(403, {"message": "Forbidden"}, None)
        )
        assert status == 403, details
        assert details["error_code"] == "github_forbidden"
        assert details["rate_limit"] is None

    async def test_headers_present_but_silent_about_the_quota_is_a_plain_denial(
        self, async_client, repo
    ) -> None:
        """⛔ The case the `headers is None` test does NOT reach: a 403 that HAS
        headers and simply says nothing about the quota (a proxy, or GitHub omitting
        them on a permission denial). Absent must stay UNKNOWN. Read as `remaining ==
        0`, the conveyor sits out the hour on what is really a missing scope — and
        the earlier version of this group let exactly that mutant live."""
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {"message": "Forbidden"}, {"content-type": "application/json"}),
        )
        assert status == 403, details
        assert details["error_code"] == "github_forbidden"
        assert details["rate_limit"] is None

    async def test_lowercase_headers_are_read_the_same_way(self, async_client, repo) -> None:
        """The casing is the transport's choice, and an exact-match lookup would read
        every lowercased header as absent — failing in the PERMISSIVE direction."""
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {}, {k.lower(): v for k, v in self.HEADERS_EXHAUSTED.items()}),
        )
        assert status == 403, details
        assert details["error_code"] == "github_rate_limited"

    async def test_the_three_auth_answers_are_pairwise_distinct(self, async_client, repo) -> None:
        """The property in one assertion: three inputs, three DIFFERENT answers, and
        none of them 404 or 500. A mutant that collapses any pair turns this red."""
        answers = []
        for exc in (
            BadCredentialsException(401, {}, None),
            GithubException(403, {}, self.HEADERS_EXHAUSTED),
            GithubException(403, {}, self.HEADERS_HEALTHY),
        ):
            status, details = await self._get(async_client, repo, exc)
            answers.append((status, details["error_code"]))
        assert len(set(answers)) == 3, answers
        assert {s for s, _ in answers} <= {401, 403}, answers

    async def test_404_is_still_a_404(self, async_client, repo) -> None:
        """``pr_not_found`` must keep working: the new branch runs BEFORE the 404 one,
        so a regression there would show up as a 403 on a missing PR."""
        status, _ = await self._get(
            async_client, repo, UnknownObjectException(404, {"message": "Not Found"}, None)
        )
        assert status == 404

    async def test_a_500_is_still_a_500(self, async_client, repo) -> None:
        """Control: the generic clause is still reachable, so the 401/403 results above
        are the new branch firing and not the whole handler having stopped raising."""
        status, _ = await self._get(async_client, repo, RuntimeError("boom"))
        assert status == 500

    async def test_millisecond_reset_still_says_rate_limited_and_do_not_rotate(
        self, async_client, repo
    ) -> None:
        """⭐ THE REGRESSION, end to end. A 403 + ``X-RateLimit-Remaining: 0`` whose
        reset arrives in MILLISECONDS used to raise ``ValueError: year must be in
        1..9999, not 58692`` from ``datetime.fromtimestamp`` — OUTSIDE ``_int``'s
        try/except, and therefore outside the route's own ``except Exception`` clause
        too, so the response carried no ``error_code`` at all. Measured on ``9ebd823``:
        all four routes collapsed a 403 into an opaque 500, the exact direction this PR
        exists to prevent.

        ⛔ AND THE ASSERTION THAT MATTERS MOST IS ``remaining == 0``: the whole meaning
        of ``github_rate_limited`` is "the credential is fine, ⛔ do NOT rotate it".
        Dropping ``remaining``/``limit`` because a DATE would not parse would hand the
        conveyor back the token rotation this route was built to prevent.
        """
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {"message": "API rate limit exceeded"}, self.HEADERS_EXHAUSTED_MS),
        )
        assert status == 403, details
        assert details["error_code"] == "github_rate_limited"
        assert details["credential_rotation_would_not_help"] is True
        # The signal survives the unparseable date.
        assert details["rate_limit"]["remaining"] == 0
        assert details["rate_limit"]["limit"] == 5000
        # ⛔ Unknown, never a fabricated date and never the raw ms epoch.
        assert details["rate_limit"]["reset_at"] is None
        assert details["rate_limit"]["reset_epoch"] is None

    async def test_millisecond_reset_still_reads_as_a_403_not_a_500(
        self, async_client, repo
    ) -> None:
        """The property in the crudest possible form, because it is the one an auditor
        measured: whatever we do with the date, the STATUS must stay GitHub's own and
        the response must still carry a machine-readable code. A mutant that lets the
        ValueError escape again returns 500 with ``details == {}`` and turns this red
        on the first assertion."""
        status, details = await self._get(
            async_client,
            repo,
            GithubException(403, {}, self.HEADERS_EXHAUSTED_MS),
        )
        assert status == 403, details
        assert details.get("error_code") == "github_rate_limited"


class TestNoLazyCompletion:
    """The performance contract of the list shape, MEASURED against real PyGithub.

    ``_pr_state``'s docstring records that touching ``merged``/``mergeable`` on a list
    item triggers a lazy completion — one extra GitHub API call PER PR. Adding
    ``merge_commit_sha`` and ``labels`` is only safe because GitHub's list payload
    (``pull-request-simple``) already carries both. That is a claim about the wire
    format, so it is measured here rather than reasoned about: real ``PullRequest``
    objects are built from a real list payload and the requester's calls are COUNTED.

    Verified against the live API 2026-09-23 (``GET /repos/{o}/{r}/pulls``): the list
    item carries ``merge_commit_sha`` and ``labels``, and omits ``merged``/``mergeable``.

    ⭐ EXTENDED for ``rate_limit``. The obvious worry — "surfacing X-RateLimit-* costs
    a call" — is FALSE HERE, and the reason is structural, not lucky:
    ``_rate_limit_snapshot`` reads ``Github.requester.rate_limiting``, a plain
    attribute the Requester rewrites from every response's headers. It never touches
    the PR object, so it cannot trigger a completion, and it is deliberately NOT
    ``Github.rate_limiting`` (that PROPERTY calls ``get_rate_limit()`` when it sees
    its seed, which WOULD be a call). Measured below, with a control proving the
    counter can still see a call on that same path.

    MUTANT: read ``pr.mergeable`` (or ``pr.merged``) in the list path -> the count goes
    from 0 to N and this turns red. MUTANT: point ``_rate_limit_snapshot`` at
    ``self._github.rate_limiting`` -> the snapshot test goes from 0 calls to 1.
    """

    # A real ``pull-request-simple`` item, trimmed to the keys the shape reads.
    LIST_ITEM = {
        "url": "https://api.github.com/repos/o/r/pulls/7",
        "number": 7,
        "state": "open",
        "title": "PR 7",
        "html_url": "https://github.com/o/r/pull/7",
        "draft": False,
        "merged_at": None,
        "merge_commit_sha": "abc123",
        "labels": [{"id": 1, "name": "bug"}, {"id": 2, "name": "ready"}],
        "head": {"ref": "feat/x", "sha": "deadbeef", "label": "o:feat/x"},
        # ``base.repo`` is a FULL repository object in the list payload, and it carries
        # ``default_branch`` — measured live 2026-09-24, unauthenticated, on
        # ``GET /repos/octocat/Hello-World/pulls`` (every item) and on the single PR.
        "base": {
            "ref": "main",
            "sha": "cafe",
            "label": "o:main",
            "repo": {
                "id": 1,
                "name": "r",
                "full_name": "o/r",
                "default_branch": "main",
                "url": "https://api.github.com/repos/o/r",
                "owner": {"login": "o", "id": 1},
            },
        },
    }
    N = 5

    @classmethod
    def _items(cls, *, omit=(), strip_base_default_branch=False):
        """N real PyGithub list items sharing one call-counting requester."""
        from github import Auth, Github
        from github.PullRequest import PullRequest

        requester = Github(auth=Auth.Token("t"), lazy=True).requester
        calls: list[tuple[str, str]] = []

        def _counted(verb, url, **kwargs):
            calls.append((verb, url))
            # Serves both a PR completion and a base-repo completion.
            full = dict(cls.LIST_ITEM) | {
                "merged": False,
                "mergeable": True,
                "default_branch": "FETCHED-BY-AN-EXTRA-CALL",
            }
            return {}, full

        requester.requestJsonAndCheck = _counted  # type: ignore[method-assign]
        payload = {k: v for k, v in cls.LIST_ITEM.items() if k not in omit}
        if strip_base_default_branch:
            base = dict(payload["base"])
            base["repo"] = {k: v for k, v in base["repo"].items() if k != "default_branch"}
            payload["base"] = base
        prs = [
            PullRequest(requester, {}, dict(payload, number=n), completed=False)
            for n in range(1, cls.N + 1)
        ]
        return prs, calls

    def test_the_list_shape_costs_zero_extra_api_calls(self) -> None:
        prs, calls = self._items()
        states = [GitHubService._pr_state(pr, from_list=True) for pr in prs]

        assert len(states) == self.N
        assert calls == [], f"list shape completed {len(calls)} time(s): {calls}"
        # ...and it really did read the two new fields, so the 0 is not vacuous.
        assert all(s["merge_commit_sha"] == "abc123" for s in states)
        assert all(s["labels"] == ["bug", "ready"] for s in states)
        assert all(s["base_default_branch"] == "main" for s in states)

    def test_control_the_counter_can_see_a_completion(self) -> None:
        """Positive control: ``mergeable`` is the field the docstring warns about.
        Without this, a broken counter would make the test above pass vacuously."""
        prs, calls = self._items()
        for pr in prs:
            _ = pr.mergeable
        assert len(calls) == self.N, f"expected {self.N} completions, saw {len(calls)}"

    # -- rate_limit: added to the single-PR read; must cost nothing ----------

    @classmethod
    def _counted_service(cls, *, complete: bool):
        """A real GitHubService over a real, call-COUNTING PyGithub requester.

        ``complete=False`` builds the PR from the LIST payload (no ``merged`` /
        ``mergeable``), which is what makes the control below able to see a call.
        """
        from github import Auth, Github
        from github.PullRequest import PullRequest

        client = Github(auth=Auth.Token("t"), lazy=True)
        requester = client.requester
        calls: list[tuple[str, str]] = []

        def _counted(verb, url, **kwargs):
            calls.append((verb, url))
            return {}, dict(cls.LIST_ITEM) | {"merged": False, "mergeable": True}

        requester.requestJsonAndCheck = _counted  # type: ignore[method-assign]
        # What a real response's headers leave behind on the requester.
        requester.rate_limiting = (4321, 5000)  # type: ignore[assignment]
        requester.rate_limiting_resettime = 1790000000  # type: ignore[assignment]

        payload = dict(cls.LIST_ITEM)
        if complete:
            payload |= {"merged": False, "mergeable": True}
        pr = PullRequest(requester, {}, payload, completed=complete)

        svc = GitHubService(GitHubConfig(token_id="t"))
        svc._github = client  # type: ignore[assignment]
        return svc, pr, calls

    async def test_the_rate_limit_snapshot_costs_zero_extra_api_calls(self) -> None:
        svc, pr, calls = self._counted_service(complete=True)
        state = await asyncio.to_thread(GitHubService._pr_state, pr)
        state["rate_limit"] = svc._rate_limit_snapshot()

        assert calls == [], f"the single-PR read made {len(calls)} call(s): {calls}"
        # ...and the 0 is not vacuous: the snapshot really carries the headers.
        assert state["rate_limit"] == {
            "remaining": 4321,
            "limit": 5000,
            "reset_epoch": 1790000000,
            "reset_at": datetime.fromtimestamp(1790000000, UTC).isoformat(),
        }

    async def test_control_the_counter_can_see_a_call_on_this_very_path(self) -> None:
        """Negative control for the 0 above. Same service, same requester, same
        ``_pr_state`` — only the PR is INCOMPLETE, so ``pr.merged`` completes once.
        Without this, a counter that had stopped counting would read as 'free'."""
        svc, pr, calls = self._counted_service(complete=False)
        await asyncio.to_thread(GitHubService._pr_state, pr)
        assert len(calls) == 1, f"expected 1 completion, saw {len(calls)}: {calls}"

    def test_control_the_snapshot_is_not_reading_the_property_that_costs_a_call(
        self,
    ) -> None:
        """⛔ The distinction the snapshot's docstring turns on, pinned. Blank the
        requester's seeded attribute and the snapshot must report UNKNOWN (all null)
        and STILL make no call — not fall back to ``Github.rate_limiting``, which
        would issue ``GET /rate_limit`` on exactly the endpoints added to save quota."""
        svc, _pr, calls = self._counted_service(complete=True)
        svc._github.requester.rate_limiting = (-1, -1)  # PyGithub's pre-response seed
        svc._github.requester.rate_limiting_resettime = 0

        snapshot = svc._rate_limit_snapshot()

        assert calls == [], f"the snapshot made {len(calls)} call(s): {calls}"
        assert snapshot == {
            "remaining": None,
            "limit": None,
            "reset_epoch": None,
            "reset_at": None,
        }, "the seed must be reported as UNKNOWN, never as an exhausted quota"

    def test_control_a_field_absent_from_the_payload_does_complete(self) -> None:
        """Negative control: the 0 above is because the fields are PRESENT in the
        payload, not because reading them can never complete. Strip them and each PR
        completes exactly once (one GET serves both fields)."""
        prs, calls = self._items(omit=("merge_commit_sha", "labels"))
        for pr in prs:
            _ = pr.merge_commit_sha
            _ = pr.labels
        assert len(calls) == self.N, f"expected {self.N} completions, saw {len(calls)}"

    def test_control_base_default_branch_absent_from_the_nested_repo_does_complete(
        self,
    ) -> None:
        """⛔ The negative control for ``base_default_branch`` specifically, because its
        0 rests on a claim one level DEEPER than the others: not 'the key is in the PR
        payload' but 'the key is in the repository object NESTED under base'. Strip only
        ``base.repo.default_branch`` and each PR completes its BASE REPO exactly once —
        ``GET /repos/o/r``, a different URL from a PR completion, which is what makes
        this a distinct measurement rather than a re-run of the one above."""
        prs, calls = self._items(strip_base_default_branch=True)
        values = [pr.base.repo.default_branch for pr in prs]

        assert len(calls) == self.N, f"expected {self.N} completions, saw {len(calls)}"
        assert all(url.endswith("/repos/o/r") for _verb, url in calls), calls
        assert all(v == "FETCHED-BY-AN-EXTRA-CALL" for v in values), values
