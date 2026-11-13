"""Tests for the Phase B read routes: get-repository, list-pull-request-files,
list-workflow-runs.

These three replace direct ``api.github.com`` calls made from n8n with a raw org PAT.
The GitHub client is mocked at the SAME boundary as ``test_github_pr_read_routes``
(``get_repository`` + ``_execute_with_retry``) and ``_github`` is swapped so the real
router -> service -> pydantic shape path runs end to end over HTTP.

Properties pinned, each named with the mutant that must turn it red:

- a GitHub 404 is a 404, a 401 is a 401 and a 403 is a 403 (mutant: drop
  ``_access_denied`` -> 401/403 become 500, or fold them into ``not_found`` -> 404);
- an ABSENT ``permissions`` block is ``null`` flags, not five ``false`` ones
  (mutant: ``bool(getattr(perms, k, False))`` -> a metadata-only read claims
  "you may not push");
- ``page``/``per_page`` reach GitHub and ``has_more`` follows the page fill
  (mutant: hard-code ``per_page=30``, or ``has_more=True``);
- a platform-repo refusal keeps its 403 ``platform_repo_forbidden`` envelope
  (mutant: remove the ``except HTTPException: raise`` -> 500);
- the rate-limit snapshot never invents a number (mutant: pass PyGithub's ``(-1,-1)``
  seed through -> a caller that backs off on ``remaining == 0`` stalls forever);
- ``head_sha`` must be hex (mutant: drop the pattern -> a branch name is accepted and
  GitHub answers an empty list, which reads as "no runs" = vacuous green).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from github import BadCredentialsException, GithubException, UnknownObjectException

from services.github.github_service import GitHubConfig, GitHubService
from services.github_integration import GitHubAuthError

GET_REPO = "/api/repositories/get-repository"
PR_FILES = "/api/repositories/list-pull-request-files"
RUNS = "/api/repositories/list-workflow-runs"
GET_BRANCH = "/api/repositories/get-branch"
COMPARE = "/api/repositories/compare"
COMPARE_FILES = "/api/repositories/compare/files"

_SVC = "services.github.github_service.GitHubService"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------
class _Perms:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Owner:
    def __init__(self, login: str) -> None:
        self.login = login


class _File:
    def __init__(self, name: str, *, patch=None, status="modified", previous=None) -> None:
        self.filename = name
        self.status = status
        self.additions = 3
        self.deletions = 1
        self.changes = 4
        self.patch = patch
        self.sha = f"blob-{name}"
        self.previous_filename = previous
        self.blob_url = f"https://github.com/o/r/blob/{name}"


class _Run:
    def __init__(self, run_id: int, *, status="completed", conclusion="success") -> None:
        self.id = run_id
        self.name = f"run-{run_id}"
        self.status = status
        self.conclusion = conclusion
        self.html_url = f"https://github.com/o/r/actions/runs/{run_id}"
        self.head_sha = "abc1234"
        self.head_branch = "main"
        self.event = "push"
        self.path = ".github/workflows/tests.yml"
        self.workflow_id = 99
        self.run_number = run_id
        self.run_attempt = 1
        self.created_at = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)
        self.updated_at = datetime(2026, 9, 22, 8, 5, tzinfo=UTC)


class _PR:
    url = "https://api.github.com/repos/o/r/pulls/7"


HEAD_SHA = "ab" * 20  # 40 hex, low entropy: a fake sha the secret scanner must not flag


class _Branch:
    def __init__(self, name: str) -> None:
        self.name = name
        self.commit = type("_Commit", (), {"sha": HEAD_SHA})()
        self.protected = name == "main"


MERGE_BASE_SHA = "cd" * 20  # 40 hex, low entropy


class _CmpFile:
    """A compare ``File``: only the three attributes ``compare_files`` reads."""

    def __init__(self, filename: str, status: str = "modified", previous=None) -> None:
        self.filename = filename
        self.status = status
        self.previous_filename = previous


class _Comparison:
    def __init__(self, status: str, ahead_by: int = 0, behind_by: int = 0, files=None) -> None:
        self.status = status
        self.ahead_by = ahead_by
        self.behind_by = behind_by
        self.merge_base_commit = type("_Commit", (), {"sha": MERGE_BASE_SHA})()
        self.files = files if files is not None else []


class _Repo:
    """A repository double. ``raises`` is raised by ``get_repository`` itself."""

    url = "https://api.github.com/repos/o/r"

    def __init__(self, *, permissions=..., files=None, runs=None, archived=False) -> None:
        self.full_name = "o/r"
        self.name = "r"
        self.owner = _Owner("o")
        self.default_branch = "main"
        self.archived = archived
        self.disabled = False
        self.private = True
        self.visibility = "private"
        self.fork = False
        self.description = "a repo"
        self.html_url = "https://github.com/o/r"
        if permissions is not ...:
            self.permissions = permissions
        self._files = files if files is not None else []
        self._runs = runs if runs is not None else []
        self.file_page_calls: list[tuple[dict, int]] = []
        self.run_page_calls: list[tuple[dict, int]] = []
        self.branch_calls: list[str] = []
        self.branch_raises: Exception | None = None
        self.compare_calls: list[tuple[str, str]] = []
        self.compare_result = _Comparison("ahead", ahead_by=2)
        self.compare_raises: Exception | None = None

    def get_pull(self, _n):
        return _PR()

    def get_branch(self, name):
        self.branch_calls.append(name)
        if self.branch_raises is not None:
            raise self.branch_raises
        return _Branch(name)

    def compare(self, base, head):
        self.compare_calls.append((base, head))
        if self.compare_raises is not None:
            raise self.compare_raises
        return self.compare_result

    # Stand-ins for the PaginatedList objects the service builds.
    def paginate_files(self, params):
        repo = self

        class _P:
            def get_page(self, page):
                repo.file_page_calls.append((dict(params), page))
                return list(repo._files)

        return _P()

    def paginate_runs(self, params):
        repo = self

        class _P:
            def get_page(self, page):
                repo.run_page_calls.append((dict(params), page))
                return list(repo._runs)

        return _P()


class _Requester:
    """PyGithub's Requester exposes rate_limiting as a plain, header-updated attr."""

    def __init__(self, remaining=4321, limit=5000, reset=1790000000) -> None:
        self.rate_limiting = (remaining, limit)
        self.rate_limiting_resettime = reset


@pytest.fixture
def repo(monkeypatch):
    """Swap ``_github`` for a service wired to a fake repo. ``holder`` is mutable."""
    holder: dict = {"repo": _Repo(permissions=_Perms(admin=False, push=True, pull=True))}

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        holder.setdefault("perms_asked", []).append(permissions)
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
        # The REAL _paginate_* builders are covered against real PyGithub in
        # TestPaginationAgainstRealPyGithub; here they are stubbed so no network.
        s._paginate_pr_files = lambda r, pr, per_page: r.paginate_files(  # type: ignore[method-assign]
            {"per_page": per_page}
        )
        s._paginate_workflow_runs = lambda r, params: r.paginate_runs(params)  # type: ignore[method-assign]
        holder["service"] = s
        yield s

    monkeypatch.setattr("routers.github_repo_read_operations._github", _fake_github)
    return holder


# ---------------------------------------------------------------------------
# get-repository
# ---------------------------------------------------------------------------
class TestGetRepository:
    async def test_success_returns_the_typed_shape(self, async_client, repo) -> None:
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["full_name"] == "o/r"
        assert body["default_branch"] == "main"
        assert body["archived"] is False
        assert body["disabled"] is False
        assert body["private"] is True
        assert body["visibility"] == "private"
        assert body["fork"] is False
        assert body["html_url"] == "https://github.com/o/r"
        assert body["permissions"] == {
            "admin": False,
            "maintain": None,
            "push": True,
            "triage": None,
            "pull": True,
        }

    async def test_archived_is_reported(self, async_client, repo) -> None:
        repo["repo"] = _Repo(permissions=_Perms(push=True), archived=True)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.json()["archived"] is True

    async def test_absent_permissions_block_is_null_not_false(self, async_client, repo) -> None:
        """MUTANT: default the flags to False -> "you may not push" becomes
        indistinguishable from "GitHub did not say", and a sweep skips work it may do."""
        repo["repo"] = _Repo(permissions=None)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        perms = resp.json()["permissions"]
        assert perms == dict.fromkeys(("admin", "maintain", "push", "triage", "pull"))
        assert not any(v is False for v in perms.values())

    async def test_missing_permissions_attribute_is_also_null(self, async_client, repo) -> None:
        repo["repo"] = _Repo()  # attribute not set at all
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        assert all(v is None for v in resp.json()["permissions"].values())

    async def test_rate_limit_headers_are_surfaced(self, async_client, repo) -> None:
        repo["requester"] = _Requester(remaining=17, limit=5000, reset=1790000000)
        body = (await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})).json()
        assert body["rate_limit"]["remaining"] == 17
        assert body["rate_limit"]["limit"] == 5000
        assert body["rate_limit"]["reset_epoch"] == 1790000000
        assert body["rate_limit"]["reset_at"] == datetime.fromtimestamp(1790000000, UTC).isoformat()

    async def test_unknown_rate_limit_is_null_never_zero(self, async_client, repo) -> None:
        """MUTANT: pass PyGithub's ``(-1, -1)`` / ``0`` seed straight through. A caller
        that backs off on ``remaining == 0`` would then stall on a MISSING header."""
        repo["requester"] = _Requester(remaining=-1, limit=-1, reset=0)
        rl = (await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})).json()[
            "rate_limit"
        ]
        assert rl == {"remaining": None, "limit": None, "reset_epoch": None, "reset_at": None}

    async def test_millisecond_reset_returns_the_repository_not_a_404(
        self, async_client, repo
    ) -> None:
        """⛔ THIS BUG WAS ALREADY LIVE, shipped in #110's ``_rate_limit_snapshot``.

        Measured on both ``origin/main`` and the branch: a repository that EXISTS
        returned **404** when ``rate_limiting_resettime`` was a millisecond epoch. The
        inline ``datetime.fromtimestamp(reset_epoch, UTC)`` raised ``ValueError: year
        must be in 1..9999, not 58692`` inside the service, and
        ``get_repository_state``'s route catches ``ValueError`` as ``not_found`` — so a
        bad HEADER was reported as a missing REPOSITORY. A caller cannot distinguish
        that from a repo it should stop sweeping.

        ⚠️ The positive control is the ``remaining``/``limit`` assertion: this must be
        fixed by making the RESET unknown, not by degrading the whole snapshot."""
        repo["requester"] = _Requester(remaining=17, limit=5000, reset=1790000000000)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["full_name"] == "o/r"
        assert body["rate_limit"] == {
            "remaining": 17,
            "limit": 5000,
            "reset_epoch": None,
            "reset_at": None,
        }


class TestGetRepositoryFailureModesStayDistinct:
    """⛔ THE load-bearing group. 404 / 401 / 403 mean three different things to
    delivery-sweep and landing-reconciler; collapsing any pair is silent."""

    async def test_missing_repository_is_404(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {"message": "Not Found"}, None)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "nope"})
        assert resp.status_code == 404, resp.text

    async def test_bad_credentials_is_401_not_404_and_not_500(self, async_client, repo) -> None:
        repo["get_repository_raises"] = BadCredentialsException(401, {"message": "Bad"}, None)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "github_unauthorized"

    async def test_forbidden_is_403_not_404_and_not_500(self, async_client, repo) -> None:
        repo["get_repository_raises"] = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "github_forbidden"

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (UnknownObjectException(404, {}, None), 404),
            (BadCredentialsException(401, {}, None), 401),
            (GithubException(403, {}, None), 403),
            (GithubException(500, {}, None), 500),
        ],
    )
    async def test_every_status_maps_to_itself(self, async_client, repo, exc, expected) -> None:
        """MUTATION CONTROL: a router that answered one status for all four would agree
        with a naive ``always 404`` (or ``always 500``) implementation on 3 of 4 rows.
        Requiring the WHOLE mapping is what makes the group non-decorative."""
        repo["get_repository_raises"] = exc
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == expected, resp.text

    async def test_a_404_and_a_403_do_not_share_a_status(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {}, None)
        missing = (await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})).status_code
        repo["get_repository_raises"] = GithubException(403, {}, None)
        denied = (await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})).status_code
        assert missing != denied
        assert (missing, denied) == (404, 403)


# ---------------------------------------------------------------------------
# list-pull-request-files
# ---------------------------------------------------------------------------
class TestListPullRequestFiles:
    async def test_success_returns_the_typed_shape(self, async_client, repo) -> None:
        repo["repo"] = _Repo(
            permissions=_Perms(pull=True),
            files=[_File("a.py", patch="@@ -1 +1 @@"), _File("logo.png", patch=None)],
        )
        resp = await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pr_number"] == 7
        assert [f["filename"] for f in body["files"]] == ["a.py", "logo.png"]
        first = body["files"][0]
        assert first["status"] == "modified"
        assert (first["additions"], first["deletions"], first["changes"]) == (3, 1, 4)
        assert first["patch"] == "@@ -1 +1 @@"

    async def test_absent_patch_is_null_not_empty_string(self, async_client, repo) -> None:
        """MUTANT: ``file.patch or ""`` -> "GitHub sent no patch" (binary / too large)
        becomes indistinguishable from "the patch was empty"."""
        repo["repo"] = _Repo(permissions=_Perms(), files=[_File("logo.png", patch=None)])
        body = (
            await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        ).json()
        assert body["files"][0]["patch"] is None

    async def test_renamed_file_carries_previous_filename(self, async_client, repo) -> None:
        repo["repo"] = _Repo(
            permissions=_Perms(), files=[_File("new.py", status="renamed", previous="old.py")]
        )
        body = (
            await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        ).json()
        assert body["files"][0]["previous_filename"] == "old.py"
        assert body["files"][0]["status"] == "renamed"

    async def test_default_per_page_is_100_and_page_is_one(self, async_client, repo) -> None:
        """delivery-sweep asks for per_page=100. MUTANT: hard-code GitHub's default 30
        -> a 100-file PR silently truncates to 30 and the sweep grades the wrong diff."""
        await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        assert repo["repo"].file_page_calls == [({"per_page": 100}, 0)]

    async def test_page_is_one_based_on_the_wire_and_zero_based_to_github(
        self, async_client, repo
    ) -> None:
        """MUTANT: forward ``page`` unchanged -> page 1 fetches GitHub's page 2 and the
        FIRST page of every PR is never read."""
        await async_client.post(
            PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7, "page": 3, "per_page": 50}
        )
        assert repo["repo"].file_page_calls == [({"per_page": 50}, 2)]

    async def test_full_page_sets_has_more_short_page_clears_it(self, async_client, repo) -> None:
        repo["repo"] = _Repo(permissions=_Perms(), files=[_File(f"f{i}.py") for i in range(2)])
        full = (
            await async_client.post(
                PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7, "per_page": 2}
            )
        ).json()
        short = (
            await async_client.post(
                PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7, "per_page": 3}
            )
        ).json()
        assert (full["count"], full["has_more"]) == (2, True)
        assert (short["count"], short["has_more"]) == (2, False)

    async def test_empty_page_is_200_with_no_more(self, async_client, repo) -> None:
        repo["repo"] = _Repo(permissions=_Perms(), files=[])
        body = (
            await async_client.post(
                PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7, "page": 9}
            )
        ).json()
        assert body["count"] == 0 and body["has_more"] is False and body["files"] == []

    async def test_rate_limit_is_surfaced(self, async_client, repo) -> None:
        repo["requester"] = _Requester(remaining=9, limit=5000, reset=1790000000)
        body = (
            await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        ).json()
        assert body["rate_limit"]["remaining"] == 9

    @pytest.mark.parametrize(
        "extra",
        [{"per_page": 0}, {"per_page": 101}, {"page": 0}, {"pr_number": 0}, {}],
    )
    async def test_invalid_input_is_422(self, async_client, repo, extra) -> None:
        payload = {"owner": "o", "repo": "r", "pr_number": 7, **extra}
        if not extra:
            payload.pop("pr_number")  # pr_number is required
        resp = await async_client.post(PR_FILES, json=payload)
        assert resp.status_code == 422, resp.text

    async def test_missing_pr_is_404(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {"message": "NF"}, None)
        resp = await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 999})
        assert resp.status_code == 404, resp.text

    async def test_forbidden_is_403(self, async_client, repo) -> None:
        repo["get_repository_raises"] = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# list-workflow-runs
# ---------------------------------------------------------------------------
class TestListWorkflowRuns:
    async def test_success_returns_the_typed_shape(self, async_client, repo) -> None:
        repo["repo"] = _Repo(
            permissions=_Perms(), runs=[_Run(1), _Run(2, status="in_progress", conclusion=None)]
        )
        resp = await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["head_sha"] == "abc1234"
        assert body["count"] == 2
        first = body["runs"][0]
        assert first["id"] == 1
        assert first["name"] == "run-1"
        assert first["status"] == "completed"
        assert first["conclusion"] == "success"
        assert first["html_url"] == "https://github.com/o/r/actions/runs/1"
        assert first["head_sha"] == "abc1234"
        assert first["created_at"] == datetime(2026, 9, 22, 8, 0, tzinfo=UTC).isoformat()
        assert first["path"] == ".github/workflows/tests.yml"

    async def test_running_run_has_null_conclusion(self, async_client, repo) -> None:
        """A ``null`` conclusion is neither a pass nor a failure — same property as
        the three-valued ``/pr-checks`` verdict."""
        repo["repo"] = _Repo(
            permissions=_Perms(), runs=[_Run(5, status="in_progress", conclusion=None)]
        )
        body = (
            await async_client.post(RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"})
        ).json()
        assert body["runs"][0]["conclusion"] is None
        assert body["runs"][0]["status"] == "in_progress"

    async def test_head_sha_and_per_page_reach_github(self, async_client, repo) -> None:
        """MUTANT: drop ``head_sha`` from the params -> EVERY run in the repo comes
        back and github-outcome-poll grades another commit's CI."""
        await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "deadbeef", "per_page": 7}
        )
        assert repo["repo"].run_page_calls == [({"head_sha": "deadbeef", "per_page": 7}, 0)]

    async def test_no_runs_is_an_empty_list_not_an_error(self, async_client, repo) -> None:
        """⛔ ``count == 0`` is "not created yet", never "passed". It must be a 200 with
        an empty list so the caller can keep polling — a 404 would read as terminal."""
        repo["repo"] = _Repo(permissions=_Perms(), runs=[])
        resp = await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "head_sha": "abc1234",
            "count": 0,
            "runs": [],
            "rate_limit": {
                "remaining": 4321,
                "limit": 5000,
                "reset_epoch": 1790000000,
                "reset_at": datetime.fromtimestamp(1790000000, UTC).isoformat(),
            },
        }

    @pytest.mark.parametrize("bad", ["main", "feat/x", "abc", "zzzzzzz", "", "a" * 41])
    async def test_non_hex_head_sha_is_422(self, async_client, repo, bad) -> None:
        """MUTANT: drop the hex pattern -> a branch name is accepted, GitHub's head_sha
        filter matches nothing, and an EMPTY list reads as "this commit has no runs"."""
        resp = await async_client.post(RUNS, json={"owner": "o", "repo": "r", "head_sha": bad})
        assert resp.status_code == 422, resp.text

    @pytest.mark.parametrize("per_page", [0, 101])
    async def test_invalid_per_page_is_422(self, async_client, repo, per_page) -> None:
        resp = await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234", "per_page": per_page}
        )
        assert resp.status_code == 422, resp.text

    async def test_missing_repository_is_404(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {"message": "NF"}, None)
        resp = await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"}
        )
        assert resp.status_code == 404, resp.text

    async def test_forbidden_is_403(self, async_client, repo) -> None:
        repo["get_repository_raises"] = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(
            RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"}
        )
        assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# get-branch
# ---------------------------------------------------------------------------
class TestGetBranch:
    async def test_success_returns_head_sha(self, async_client, repo) -> None:
        resp = await async_client.post(
            GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "main"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["owner"] == "o"
        assert body["repo"] == "r"
        assert body["branch"] == "main"
        assert body["sha"] == HEAD_SHA
        assert body["protected"] is True
        assert body["rate_limit"]["remaining"] == 4321
        assert repo["repo"].branch_calls == ["main"]

    async def test_branch_with_slash_is_passed_through(self, async_client, repo) -> None:
        """MUTANT: split on ``/`` or pre-quote the name -> ``feature/x`` reaches
        PyGithub as ``feature`` or ``feature%2Fx`` (double-quoted on the wire)."""
        resp = await async_client.post(
            GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "feature/x"}
        )
        assert resp.status_code == 200, resp.text
        assert repo["repo"].branch_calls == ["feature/x"]
        assert resp.json()["branch"] == "feature/x"
        assert resp.json()["protected"] is False

    async def test_missing_branch_is_404(self, async_client, repo) -> None:
        repo["repo"].branch_raises = GithubException(404, {"message": "Branch not found"}, None)
        resp = await async_client.post(
            GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "gone"}
        )
        assert resp.status_code == 404, resp.text

    async def test_missing_repository_is_404(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {"message": "NF"}, None)
        resp = await async_client.post(
            GET_BRANCH, json={"owner": "o", "repo": "nope", "branch": "main"}
        )
        assert resp.status_code == 404, resp.text

    async def test_forbidden_is_403_not_404(self, async_client, repo) -> None:
        repo["repo"].branch_raises = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(
            GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "main"}
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "github_forbidden"

    @pytest.mark.parametrize("bad", ["a..b", "", "has space", "tab\tx", "a" * 256])
    async def test_invalid_branch_is_422(self, async_client, repo, bad) -> None:
        resp = await async_client.post(GET_BRANCH, json={"owner": "o", "repo": "r", "branch": bad})
        assert resp.status_code == 422, resp.text
        assert repo["repo"].branch_calls == []

    async def test_missing_branch_field_is_422(self, async_client, repo) -> None:
        resp = await async_client.post(GET_BRANCH, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 422, resp.text

    async def test_missing_tenant_is_401_and_github_is_never_asked(
        self, async_client, repo
    ) -> None:
        """Without the suite-wide tenant override, ``require_tenant`` sees no tenant
        and refuses before ``_github`` mints any token."""
        from coding_client.auth.tenant import require_tenant

        from main import app

        saved = app.dependency_overrides.pop(require_tenant, None)
        try:
            resp = await async_client.post(
                GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "main"}
            )
        finally:
            if saved is not None:
                app.dependency_overrides[require_tenant] = saved
        assert resp.status_code == 401, resp.text
        assert "perms_asked" not in repo


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
_CMP = {"owner": "o", "repo": "r", "base": "abc1234", "head": "run/branch"}


class TestCompareCommits:
    async def test_ahead_means_base_is_ancestor(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison("ahead", ahead_by=3, behind_by=0)
        resp = await async_client.post(COMPARE, json=_CMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["owner"] == "o"
        assert body["repo"] == "r"
        assert body["base"] == "abc1234"
        assert body["head"] == "run/branch"
        assert body["status"] == "ahead"
        assert (body["ahead_by"], body["behind_by"]) == (3, 0)
        assert body["merge_base_sha"] == MERGE_BASE_SHA
        assert body["base_is_ancestor_of_head"] is True
        assert body["rate_limit"]["remaining"] == 4321
        # base/head reach PyGithub in order, unmodified (slash kept).
        assert repo["repo"].compare_calls == [("abc1234", "run/branch")]

    async def test_identical_means_base_is_ancestor(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison("identical")
        body = (await async_client.post(COMPARE, json=_CMP)).json()
        assert body["status"] == "identical"
        assert body["base_is_ancestor_of_head"] is True

    async def test_behind_is_not_ancestor(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison("behind", behind_by=4)
        body = (await async_client.post(COMPARE, json=_CMP)).json()
        assert body["status"] == "behind"
        assert body["base_is_ancestor_of_head"] is False

    async def test_diverged_is_not_ancestor(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison("diverged", ahead_by=2, behind_by=5)
        body = (await async_client.post(COMPARE, json=_CMP)).json()
        assert body["status"] == "diverged"
        assert body["base_is_ancestor_of_head"] is False

    @pytest.mark.parametrize(
        ("status", "ahead_by", "behind_by", "expected"),
        [
            ("ahead", 1, 0, True),
            ("identical", 0, 0, True),
            ("behind", 0, 1, False),
            ("diverged", 7, 1, False),
            ("diverged", 7, 0, False),
            ("behind", 3, 3, False),
        ],
    )
    async def test_ancestor_flag_is_derived_from_status_only(
        self, async_client, repo, status, ahead_by, behind_by, expected
    ) -> None:
        """MUTANT: ``base_is_ancestor_of_head = ahead_by > 0`` (or ``behind_by == 0``,
        or a hard-coded ``True``) -> a diverged child merge that never reached the run
        branch reads as landed. Rows with ``ahead_by > 0`` AND ``False`` kill the
        first; ``behind_by == 0`` AND ``False`` kills the second; the ``False`` rows
        kill a constant. The flag must follow ``status`` and nothing else."""
        repo["repo"].compare_result = _Comparison(status, ahead_by=ahead_by, behind_by=behind_by)
        body = (await async_client.post(COMPARE, json=_CMP)).json()
        assert body["base_is_ancestor_of_head"] is expected

    async def test_unknown_sha_or_ref_is_404_not_200_false(self, async_client, repo) -> None:
        """⛔ "not found" must never read as "not contained": a caller gating on the
        flag would otherwise treat a typo'd SHA as a real, un-landed merge."""
        repo["repo"].compare_raises = UnknownObjectException(404, {"message": "Not Found"}, None)
        resp = await async_client.post(COMPARE, json=_CMP)
        assert resp.status_code == 404, resp.text
        assert "base_is_ancestor_of_head" not in resp.text

    async def test_unknown_ref_as_plain_github_404_is_404(self, async_client, repo) -> None:
        repo["repo"].compare_raises = GithubException(404, {"message": "No commit found"}, None)
        resp = await async_client.post(COMPARE, json=_CMP)
        assert resp.status_code == 404, resp.text

    async def test_missing_repository_is_404(self, async_client, repo) -> None:
        repo["get_repository_raises"] = UnknownObjectException(404, {"message": "NF"}, None)
        resp = await async_client.post(COMPARE, json={**_CMP, "repo": "nope"})
        assert resp.status_code == 404, resp.text

    async def test_forbidden_is_403_not_404(self, async_client, repo) -> None:
        repo["repo"].compare_raises = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(COMPARE, json=_CMP)
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "github_forbidden"

    @pytest.mark.parametrize("field", ["base", "head"])
    @pytest.mark.parametrize("bad", ["a..b", "main...x", "", "has space", "a" * 256])
    async def test_invalid_ref_is_422(self, async_client, repo, field, bad) -> None:
        resp = await async_client.post(COMPARE, json={**_CMP, field: bad})
        assert resp.status_code == 422, resp.text
        assert repo["repo"].compare_calls == []

    @pytest.mark.parametrize("field", ["base", "head"])
    async def test_missing_field_is_422(self, async_client, repo, field) -> None:
        payload = dict(_CMP)
        payload.pop(field)
        resp = await async_client.post(COMPARE, json=payload)
        assert resp.status_code == 422, resp.text

    async def test_missing_tenant_is_401_and_github_is_never_asked(
        self, async_client, repo
    ) -> None:
        from coding_client.auth.tenant import require_tenant

        from main import app

        saved = app.dependency_overrides.pop(require_tenant, None)
        try:
            resp = await async_client.post(COMPARE, json=_CMP)
        finally:
            if saved is not None:
                app.dependency_overrides[require_tenant] = saved
        assert resp.status_code == 401, resp.text
        assert "perms_asked" not in repo
        assert repo["repo"].compare_calls == []


def _n_files(n: int) -> list[_CmpFile]:
    return [_CmpFile(f"f{i:04d}.py") for i in range(n)]


class TestCompareFiles:
    async def test_exact_list_sorted_with_previous_filename_only_on_rename(
        self, async_client, repo
    ) -> None:
        repo["repo"].compare_result = _Comparison(
            "ahead",
            ahead_by=1,
            files=[
                _CmpFile("src/new_name.py", "renamed", previous="src/old_name.py"),
                _CmpFile("docs/added.md", "added"),
                # GitHub only sends previous_filename on a rename; a stray one on any
                # other status must not leak into the response.
                _CmpFile("a_removed.py", "removed", previous="ignored.py"),
            ],
        )
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["files"] == [
            {"filename": "a_removed.py", "status": "removed", "previous_filename": None},
            {"filename": "docs/added.md", "status": "added", "previous_filename": None},
            {
                "filename": "src/new_name.py",
                "status": "renamed",
                "previous_filename": "src/old_name.py",
            },
        ]
        assert body["files_count"] == 3
        assert body["status"] == "ahead"
        assert body["merge_base_sha"] == MERGE_BASE_SHA
        assert body["base_is_ancestor_of_head"] is True
        assert body["rate_limit"]["remaining"] == 4321
        assert repo["repo"].compare_calls == [("abc1234", "run/branch")]

    async def test_299_files_is_the_last_exact_answer(self, async_client, repo) -> None:
        """Boundary: one below the cap is returned in full."""
        repo["repo"].compare_result = _Comparison("ahead", ahead_by=1, files=_n_files(299))
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["files_count"] == 299
        assert len(body["files"]) == 299

    async def test_300_files_is_refused_as_truncated_not_returned(self, async_client, repo) -> None:
        """⛔ MUST FAIL CLOSED. GitHub caps compare at 300 files with no truncation flag,
        so 300 is indistinguishable from "cut at 300". MUTANT: return the list (or use
        ``> 300``) -> a 200 with a possibly partial list and the fence is blind."""
        repo["repo"].compare_result = _Comparison("ahead", ahead_by=1, files=_n_files(300))
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["error"]["details"]["error_code"] == "compare_files_truncated"
        assert body["error"]["details"]["count"] == 300
        assert "files" not in body
        assert "files" not in body["error"]["details"]
        assert "f0000.py" not in resp.text

    async def test_true_diff_of_301_capped_to_300_by_github_is_refused(
        self, async_client, repo
    ) -> None:
        """What GitHub actually sends for a 301-file diff: the first 300, no flag."""
        true_diff = _n_files(301)
        repo["repo"].compare_result = _Comparison("ahead", ahead_by=1, files=true_diff[:300])
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["error"]["details"]["error_code"] == "compare_files_truncated"
        assert body["error"]["details"]["count"] == 300
        assert "files" not in body

    async def test_diverged_returns_normally_and_is_not_ancestor(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison(
            "diverged", ahead_by=2, behind_by=5, files=[_CmpFile("x.py")]
        )
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "diverged"
        assert body["base_is_ancestor_of_head"] is False
        assert body["files_count"] == 1

    async def test_identical_is_an_empty_exact_list(self, async_client, repo) -> None:
        repo["repo"].compare_result = _Comparison("identical")
        body = (await async_client.post(COMPARE_FILES, json=_CMP)).json()
        assert body["files"] == []
        assert body["files_count"] == 0

    async def test_unknown_sha_or_ref_is_404(self, async_client, repo) -> None:
        repo["repo"].compare_raises = UnknownObjectException(404, {"message": "Not Found"}, None)
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 404, resp.text
        assert "files_count" not in resp.text

    async def test_unknown_ref_as_plain_github_404_is_404(self, async_client, repo) -> None:
        repo["repo"].compare_raises = GithubException(404, {"message": "No commit found"}, None)
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 404, resp.text

    async def test_forbidden_is_403_not_404(self, async_client, repo) -> None:
        repo["repo"].compare_raises = GithubException(403, {"message": "Forbidden"}, None)
        resp = await async_client.post(COMPARE_FILES, json=_CMP)
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "github_forbidden"

    @pytest.mark.parametrize("field", ["base", "head"])
    @pytest.mark.parametrize("bad", ["a..b", "main...x", "", "has space", "a" * 256])
    async def test_invalid_ref_is_422(self, async_client, repo, field, bad) -> None:
        resp = await async_client.post(COMPARE_FILES, json={**_CMP, field: bad})
        assert resp.status_code == 422, resp.text
        assert repo["repo"].compare_calls == []


class _LazyComparison:
    """PyGithub's ``Comparison`` shape: built with no I/O, GET fires on ``status``.

    ``raises`` is raised on every ``status`` read, standing in for the deferred 404 /
    5xx. The route-level ``_Repo.compare`` raises EAGERLY, which is why it cannot see
    whether the touch in ``_fetch`` exists.
    """

    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises
        self.ahead_by = 1
        self.behind_by = 0
        self.merge_base_commit = type("_Commit", (), {"sha": MERGE_BASE_SHA})()

    @property
    def status(self) -> str:
        if self._raises is not None:
            raise self._raises
        return "ahead"


class TestCompareLazyFetchIsInsideRetry:
    """⛔ ``_ = comparison.status`` in ``compare_commits._fetch`` is what puts the GET
    (and its 404) inside ``_execute_with_retry``.

    MUTANT: delete that line -> ``_fetch`` returns the unfetched object, the wrapper
    sees no error and retries nothing, and the request fires later in ``_shape`` where
    nothing retries it. Every route-level compare test stays green under that mutant.
    """

    @staticmethod
    def _service(comparisons: list[_LazyComparison], seen: list[Exception], retries: int):
        s = GitHubService(GitHubConfig(token_id="t"))
        s._github = type("G", (), {"requester": _Requester()})()

        class _LazyRepo:
            def compare(self, _base, _head):
                return comparisons.pop(0)

        async def _gr(_name):
            return _LazyRepo()

        async def _ex(func, *a, **k):
            for attempt in range(retries + 1):
                try:
                    return func(*a, **k)
                except Exception as e:
                    seen.append(e)
                    if attempt == retries:
                        raise
            raise AssertionError("unreachable")

        s.get_repository = _gr  # type: ignore[method-assign]
        s._execute_with_retry = _ex  # type: ignore[method-assign]
        return s

    async def test_deferred_404_surfaces_inside_the_retry_wrapper(self) -> None:
        missing = UnknownObjectException(404, {"message": "Not Found"}, None)
        seen: list[Exception] = []
        s = self._service([_LazyComparison(raises=missing)], seen, retries=0)

        with pytest.raises(UnknownObjectException):
            await s.compare_commits("o/r", "abc1234", "main")

        assert seen == [missing]

    async def test_deferred_transient_error_is_retried(self) -> None:
        flaky = GithubException(502, {"message": "Bad Gateway"}, None)
        seen: list[Exception] = []
        s = self._service([_LazyComparison(raises=flaky), _LazyComparison()], seen, retries=1)

        body = await s.compare_commits("o/r", "abc1234", "main")

        assert seen == [flaky]
        assert body["status"] == "ahead"
        assert body["base_is_ancestor_of_head"] is True


# ---------------------------------------------------------------------------
# The credential seam
# ---------------------------------------------------------------------------
class TestCredentialSeam:
    async def test_each_route_asks_for_its_own_permission_set(self, async_client, repo) -> None:
        """⛔ ``actions: read`` is a DISTINCT grant from ``checks: read``: a PR-checks
        token cannot read ``GET /actions/runs``. MUTANT: reuse PERMS_PR_CHECKS for
        list-workflow-runs -> a 403 from GitHub that names nothing."""
        from services.tenant_binding import PERMS_ACTIONS_READ, PERMS_PR_STATE, PERMS_REPO_READ

        await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        await async_client.post(RUNS, json={"owner": "o", "repo": "r", "head_sha": "abc1234"})
        await async_client.post(GET_BRANCH, json={"owner": "o", "repo": "r", "branch": "main"})
        await async_client.post(COMPARE, json=_CMP)
        await async_client.post(COMPARE_FILES, json=_CMP)
        assert repo["perms_asked"] == [
            PERMS_REPO_READ,
            PERMS_PR_STATE,
            PERMS_ACTIONS_READ,
            PERMS_REPO_READ,
            PERMS_REPO_READ,
            PERMS_REPO_READ,
        ]

    @pytest.mark.parametrize(
        ("url", "payload"),
        [
            (GET_REPO, {}),
            (PR_FILES, {"pr_number": 7}),
            (RUNS, {"head_sha": "abc1234"}),
            (GET_BRANCH, {"branch": "main"}),
            (COMPARE, {"base": "abc1234", "head": "main"}),
            (COMPARE_FILES, {"base": "abc1234", "head": "main"}),
        ],
    )
    async def test_platform_repo_refusal_keeps_its_shape_not_a_500(
        self, async_client, url, payload, enforce_platform_tenants
    ) -> None:
        """⛔ THE refusal test. A non-platform tenant asking for a platform-owned repo
        must get 403 ``platform_repo_forbidden`` through the REAL seam
        (``_github`` -> resolve_github_token -> check_platform_repo_tenant), not a 500.

        MUTANT: delete ``except HTTPException: raise`` from the handler -> the generic
        clause below it reclassifies the refusal as 500 "Failed to ..." and every caller
        loses the machine-readable code.
        """
        from coding_client.testing import override_tenant_auth

        from main import app

        # `test-owner` is CODING_GITHUB_SYSTEM_GITHUB_OWNER in tests/conftest.py.
        undo = override_tenant_auth(app, tenant_id="11111111-1111-1111-1111-111111111111")
        try:
            resp = await async_client.post(
                url, json={"owner": "test-owner", "repo": "coding-github", **payload}
            )
        finally:
            undo()
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"] == {
            "owner": "test-owner",
            "repo": "coding-github",
            "error_code": "platform_repo_forbidden",
        }


class TestResponseShapeDrift:
    """pydantic's ValidationError IS a ValueError. MUTANT: drop the ValidationError
    clause -> the ``except ValueError`` 404 branch swallows our own bug."""

    async def test_malformed_repository_dict_is_500_not_404(
        self, async_client, repo, monkeypatch
    ) -> None:
        async def _bad(self, repo_name):
            return {"full_name": "o/r"}  # missing every other required field

        monkeypatch.setattr(f"{_SVC}.get_repository_state", _bad)
        resp = await async_client.post(GET_REPO, json={"owner": "o", "repo": "r"})
        assert resp.status_code == 500, resp.text

    async def test_malformed_file_page_is_500_not_404(
        self, async_client, repo, monkeypatch
    ) -> None:
        async def _bad(self, repo_name, pr_number, page=1, per_page=100):
            return {"pr_number": pr_number, "files": [{"filename": "a.py"}]}

        monkeypatch.setattr(f"{_SVC}.list_pull_request_file_page", _bad)
        resp = await async_client.post(PR_FILES, json={"owner": "o", "repo": "r", "pr_number": 7})
        assert resp.status_code == 500, resp.text


class TestPaginationAgainstRealPyGithub:
    """Against REAL PyGithub, no network: ``get_repo(lazy=True)`` and a PaginatedList
    are both constructed without I/O.

    MUTANT: pass ``per_page=`` to ``pr.get_files()`` / ``repo.get_workflow_runs()``
    (TypeError in PyGithub 2.9), or drop ``list_item="workflow_runs"`` (the runs
    endpoint wraps its array in an object, so the list reads as EMPTY — the vacuous
    "no runs for this SHA"). A fake-only test could not see either.
    """

    @staticmethod
    def _real_repo():
        from github import Auth, Github

        return Github(auth=Auth.Token("t"), lazy=True).get_repo("o/r")

    @pytest.mark.parametrize("per_page", [100, 30, 1])
    async def test_pr_files_per_page_is_on_the_first_request(self, per_page) -> None:
        repo = self._real_repo()
        # A real get_pull() would hit the network; _paginate_pr_files only reads pr.url.
        pr = type("_PRUrl", (), {"url": f"{repo.url}/pulls/7"})()
        plist = GitHubService._paginate_pr_files(repo, pr, per_page)
        assert plist._PaginatedList__firstUrl.endswith("/repos/o/r/pulls/7/files")
        assert plist._PaginatedList__firstParams == {"per_page": per_page}

    async def test_workflow_runs_url_params_and_list_item(self) -> None:
        repo = self._real_repo()
        params = {"head_sha": "abc1234", "per_page": 30}
        plist = GitHubService._paginate_workflow_runs(repo, params)
        assert plist._PaginatedList__firstUrl.endswith("/repos/o/r/actions/runs")
        assert plist._PaginatedList__firstParams == params
        assert plist._PaginatedList__list_item == "workflow_runs"


class TestErrorMappingIsTheSameOnAllThreeRoutes:
    """One table, three routes. A per-route hand-written mapping drifts; this fails the
    moment one handler loses a clause the other two keep.

    MUTANT: delete ``except GitHubAuthError`` from any ONE handler -> that row answers
    500 while its two siblings answer 412, and the table catches exactly that.
    """

    ROUTES = [
        (GET_REPO, {"owner": "o", "repo": "r"}, "get_repository_state"),
        (PR_FILES, {"owner": "o", "repo": "r", "pr_number": 7}, "list_pull_request_file_page"),
        (RUNS, {"owner": "o", "repo": "r", "head_sha": "abc1234"}, "list_workflow_runs_for_sha"),
        (GET_BRANCH, {"owner": "o", "repo": "r", "branch": "main"}, "get_branch_head"),
        (COMPARE, _CMP, "compare_commits"),
        (COMPARE_FILES, _CMP, "compare_files"),
    ]

    @pytest.mark.parametrize(("url", "payload", "method"), ROUTES)
    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            # A non-tenant GitHubAuthError keeps the historical 412 refusal.
            (lambda: GitHubAuthError("App not usable"), 412),
            # A bare ValueError is "not there".
            (lambda: ValueError("no such thing"), 404),
            # Anything else is OUR bug and must say 500, not pretend it is a 404.
            (lambda: RuntimeError("GitHub client not initialized"), 500),
        ],
    )
    async def test_status_mapping(
        self, async_client, repo, monkeypatch, url, payload, method, factory, expected
    ) -> None:
        async def _raise(self, *a, **k):
            raise factory()

        monkeypatch.setattr(f"{_SVC}.{method}", _raise)
        resp = await async_client.post(url, json=payload)
        assert resp.status_code == expected, resp.text

    @pytest.mark.parametrize(("url", "payload", "method"), ROUTES)
    async def test_shape_drift_is_500_not_404_on_every_route(
        self, async_client, repo, monkeypatch, url, payload, method
    ) -> None:
        """pydantic's ValidationError IS a ValueError: without the earlier clause the
        404 branch swallows our own response-shape bug on whichever route lost it."""

        async def _bad(self, *a, **k):
            return {"count": "not-an-int"}

        monkeypatch.setattr(f"{_SVC}.{method}", _bad)
        resp = await async_client.post(url, json=payload)
        assert resp.status_code == 500, resp.text
        assert "not-an-int" not in resp.text  # values stay out of the message
