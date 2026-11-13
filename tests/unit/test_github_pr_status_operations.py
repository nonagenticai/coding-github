"""Tests for PR check-state and merge.

THE PROPERTY: "no failures yet" is not "passed". A check-run list read before jobs finish
contains zero failures and reads green. Measured 2026-08-26 the org CI pool is a SINGLE
node — every runner pod lands on worker2 — and a PR sat at queued=11/in_progress=0 for 25+
minutes. So the window where "no failures" means "nothing ran" is minutes-to-hours wide.

The mutation control at the bottom re-implements the verdict two-valued (failed == 0 =>
passed) and requires the pending cases to disagree. If it ever agrees, these tests are not
testing the thing this module exists for.
"""

from __future__ import annotations

import asyncio

import pytest

from services.github import github_service as gs
from services.github.github_service import GitHubConfig, GitHubService

_REAL_COMMIT_HANDLE = gs.commit_handle


@pytest.fixture(autouse=True)
def _commit_seam(monkeypatch):
    """The fakes hand out their commit through the same seam production uses."""
    monkeypatch.setattr(gs, "commit_handle", lambda repo, sha: repo.commit_for(sha))


class _Run:
    def __init__(self, name: str, status: str, conclusion: str | None):
        self.name, self.status, self.conclusion = name, status, conclusion


class _Head:
    sha = "deadbeef"


class _PR:
    def __init__(self, merged=False, mergeable=True, title="t"):
        self.merged, self.mergeable, self.title = merged, mergeable, title
        self.head = _Head()
        self.merge_calls: list[dict] = []

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        return type("R", (), {"merged": True, "sha": "abc123", "message": "ok"})()


class _Commit:
    def __init__(self, runs):
        self._runs = runs

    def get_check_runs(self):
        return list(self._runs)


class _Repo:
    private = True  # a private target repo (a public one is refused before merge)

    def __init__(self, runs, pr=None):
        self._commit = _Commit(runs)
        self._pr = pr or _PR()

    def get_pull(self, n):
        return self._pr

    # No ``get_commit``: GET /commits/{sha} needs contents:read (see commit_handle).
    def commit_for(self, sha):
        return self._commit


def _svc(repo) -> GitHubService:
    s = GitHubService(GitHubConfig(token_id="t"))

    async def _gr(repo_name):
        return repo

    async def _ex(func, *a, **k):
        return func(*a, **k)

    s.get_repository = _gr  # type: ignore[method-assign]
    s._execute_with_retry = _ex  # type: ignore[method-assign]
    return s


def _verdict(runs, expected=None, pr=None) -> dict:
    return asyncio.run(
        _svc(_Repo(runs, pr)).get_pull_request_checks("o/r", 1, expected_checks=expected)
    )


ALL_GREEN = [_Run("a", "completed", "success"), _Run("b", "completed", "success")]
STILL_QUEUED = [_Run("a", "completed", "success"), _Run("b", "queued", None)]
ONE_FAILED = [_Run("a", "completed", "success"), _Run("b", "completed", "failure")]
WITH_SKIPS = [_Run("a", "completed", "success"), _Run("b", "completed", "skipped")]


def test_all_completed_and_clean_is_passed():
    assert _verdict(ALL_GREEN)["verdict"] == "passed"


def test_queued_check_is_pending_not_passed():
    """THE load-bearing case: zero failures, and it must NOT be passed."""
    v = _verdict(STILL_QUEUED)
    assert v["failed"] == 0
    assert v["verdict"] == "pending"


def test_no_checks_at_all_is_pending_not_passed():
    """total == completed == 0 satisfies 'all completed' vacuously. Must not be passed."""
    v = _verdict([])
    assert v["total"] == 0 and v["verdict"] == "pending"


def test_fewer_checks_than_expected_is_pending():
    v = _verdict(ALL_GREEN, expected=5)
    assert v["verdict"] == "pending"


def test_expected_met_is_passed():
    assert _verdict(ALL_GREEN, expected=2)["verdict"] == "passed"


def test_failure_is_failed_and_names_it():
    v = _verdict(ONE_FAILED)
    assert v["verdict"] == "failed" and v["failed_names"] == ["b"]


def test_skipped_is_not_a_failure_and_not_outstanding():
    """A conditional job that did not apply is completed and non-failing."""
    assert _verdict(WITH_SKIPS)["verdict"] == "passed"


def test_cancelled_and_timed_out_count_as_failures():
    v = _verdict([_Run("a", "completed", "cancelled"), _Run("b", "completed", "timed_out")])
    assert v["verdict"] == "failed" and sorted(v["failed_names"]) == ["a", "b"]


def test_merge_refuses_when_checks_pending():
    """Refused as firmly as a failure — this is the whole point on a single-node pool."""
    repo = _Repo(STILL_QUEUED)
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1))
    assert out["merged"] is False and out["reason"] == "checks_pending"
    assert repo._pr.merge_calls == []


def test_merge_refuses_when_checks_failed():
    repo = _Repo(ONE_FAILED)
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1))
    assert out["merged"] is False and out["reason"] == "checks_failed"
    assert repo._pr.merge_calls == []


def test_merge_proceeds_when_passed():
    repo = _Repo(ALL_GREEN)
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1, commit_title="T"))
    assert out["merged"] is True and out["sha"] == "abc123"
    assert repo._pr.merge_calls[0]["merge_method"] == "squash"


def test_require_checks_false_bypasses_the_gate():
    repo = _Repo(STILL_QUEUED)
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1, require_checks=False))
    assert out["merged"] is True


def test_already_merged_is_reported_not_retried():
    repo = _Repo(ALL_GREEN, pr=_PR(merged=True))
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1))
    assert out["merged"] is False and out["reason"] == "already_merged"


def test_conflicts_are_reported():
    repo = _Repo(ALL_GREEN, pr=_PR(mergeable=False))
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1))
    assert out["merged"] is False and out["reason"] == "conflicts"


def test_mergeable_none_is_not_treated_as_conflict():
    """`mergeable` is None while GitHub computes it. None is NOT False."""
    repo = _Repo(ALL_GREEN, pr=_PR(mergeable=None))
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1))
    assert out["merged"] is True


@pytest.mark.parametrize("runs", [STILL_QUEUED, []])
def test_mutation_a_two_valued_verdict_would_call_these_passed(runs):
    """MUTATION CONTROL. Re-implement the verdict the naive way — no failures => passed —
    and it must DISAGREE with the real one on exactly the pending cases. If these ever
    agree, the three-valued logic is not being exercised and every test above is decorative.
    """
    real = _verdict(runs)["verdict"]
    naive = "failed" if any(r.conclusion == "failure" for r in runs) else "passed"
    assert real == "pending"
    assert naive == "passed"
    assert real != naive


# ---------------------------------------------------------------------------
# SHA-pinned merge (G12 landing in-chain). The conveyor lands ONLY the tree it tested:
# `sha` pins the head, `require_up_to_date` refuses a head behind base. Every refusal is
# `merged: false` + reason, never a 500, and never a call to `pr.merge`.
# ---------------------------------------------------------------------------

from services.github.github_service import GithubException  # noqa: E402

PINNED = "a" * 40
OTHER = "b" * 40


class _Base:
    ref = "main"


class _PinPR(_PR):
    def __init__(self, head_sha=PINNED, merged=False, merge_exc=None, merge_commit_sha=None):
        super().__init__(merged=merged)
        self.head = type("H", (), {"sha": head_sha})()
        self.base = _Base()
        self.merge_commit_sha = merge_commit_sha
        self._merge_exc = merge_exc

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        if self._merge_exc is not None:
            raise self._merge_exc
        return type("R", (), {"merged": True, "sha": "c" * 40, "message": "ok"})()


class _PinRepo(_Repo):
    def __init__(self, pr, behind_by=0):
        super().__init__(ALL_GREEN, pr=pr)
        self.behind_by = behind_by
        self.compare_calls: list[tuple] = []

    def compare(self, base, head):
        self.compare_calls.append((base, head))
        return type("C", (), {"behind_by": self.behind_by, "ahead_by": 1})()


def _merge(repo, **kw):
    kw.setdefault("require_checks", False)
    return asyncio.run(_svc(repo).merge_pull_request("o/r", 1, merge_method="merge", **kw))


def test_sha_reaches_pr_merge():
    repo = _PinRepo(_PinPR())
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is True
    assert repo._pr.merge_calls[0]["sha"] == PINNED


def test_moved_head_is_refused_without_calling_merge():
    repo = _PinRepo(_PinPR(head_sha=OTHER))
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is False and out["reason"] == "head_moved"
    assert out["head_sha"] == OTHER
    assert repo._pr.merge_calls == []


def test_github_409_is_head_moved():
    exc = GithubException(409, {"message": "Head branch was modified."}, None)
    repo = _PinRepo(_PinPR(merge_exc=exc))
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is False and out["reason"] == "head_moved"
    assert repo._pr.merge_calls[0]["sha"] == PINNED


def test_behind_base_is_base_moved_when_up_to_date_required():
    repo = _PinRepo(_PinPR(), behind_by=2)
    out = _merge(repo, sha=PINNED, require_up_to_date=True)
    assert out["merged"] is False and out["reason"] == "base_moved"
    assert out["behind_by"] == 2
    assert repo.compare_calls == [("main", PINNED)]
    assert repo._pr.merge_calls == []


def test_up_to_date_head_merges():
    repo = _PinRepo(_PinPR(), behind_by=0)
    out = _merge(repo, sha=PINNED, require_up_to_date=True)
    assert out["merged"] is True
    assert repo.compare_calls == [("main", PINNED)]


def test_behind_base_is_ignored_when_up_to_date_not_required():
    repo = _PinRepo(_PinPR(), behind_by=2)
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is True
    assert repo.compare_calls == []


def test_draft_405_is_merge_blocked_not_an_error():
    exc = GithubException(405, {"message": "Pull Request is still a draft"}, None)
    repo = _PinRepo(_PinPR(merge_exc=exc))
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is False and out["reason"] == "merge_blocked"
    assert "draft" in out["message"]


def test_other_github_errors_still_raise():
    exc = GithubException(422, {"message": "nope"}, None)
    repo = _PinRepo(_PinPR(merge_exc=exc))
    with pytest.raises(GithubException):
        _merge(repo, sha=PINNED)


def test_already_merged_returns_merge_commit_sha_and_head():
    repo = _PinRepo(_PinPR(merged=True, merge_commit_sha="d" * 40))
    out = _merge(repo, sha=PINNED)
    assert out["merged"] is False and out["reason"] == "already_merged"
    assert out["sha"] == "d" * 40 and out["head_sha"] == PINNED
    assert repo._pr.merge_calls == []


def test_no_sha_is_legacy_behaviour():
    """No `sha`: nothing is forwarded (PyGithub asserts str|NotSet), no compare, merges."""
    repo = _PinRepo(_PinPR(head_sha=OTHER), behind_by=5)
    out = _merge(repo)
    assert out["merged"] is True
    assert "sha" not in repo._pr.merge_calls[0]
    assert repo.compare_calls == []


def test_merge_request_validates_sha_shape():
    from pydantic import ValidationError

    from schemas.github_api_models import MergePRRequest

    ok = MergePRRequest(owner="o", repo="r", pr_number=1, sha=PINNED)
    assert ok.sha == PINNED and ok.require_up_to_date is False
    for bad in ("abc123", "z" * 40, "a" * 41):
        with pytest.raises(ValidationError):
            MergePRRequest(owner="o", repo="r", pr_number=1, sha=bad)
    legacy = MergePRRequest(owner="o", repo="r", pr_number=1)
    assert legacy.sha is None and legacy.require_up_to_date is False


def test_route_passes_sha_and_require_up_to_date_through(monkeypatch):
    import contextlib

    from routers import github_pr_status_operations as mod
    from schemas.github_api_models import MergePRRequest

    seen: dict = {}

    class _Svc:
        async def merge_pull_request(self, **kw):
            seen.update(kw)
            return {"merged": True, "pr_number": 1, "sha": "c" * 40}

    @contextlib.asynccontextmanager
    async def _fake(*a, **k):
        yield _Svc()

    monkeypatch.setattr(mod, "_github", _fake)
    req = MergePRRequest(owner="o", repo="r", pr_number=1, sha=PINNED, require_up_to_date=True)
    out = asyncio.run(mod.merge_pull_request(req, tenant_id="t"))
    assert out.merged is True
    assert seen["sha"] == PINNED and seen["require_up_to_date"] is True


def test_pinned_merge_refuses_checks_that_graded_another_head():
    """Checks read head B green, head is back at the pin A by get_pull: A was never graded."""
    repo = _PinRepo(_PinPR())
    svc = _svc(repo)

    async def _checks(*_a, **_k):
        return {"verdict": "passed", "head_sha": OTHER}

    svc.get_pull_request_checks = _checks
    out = asyncio.run(svc.merge_pull_request("o/r", 1, merge_method="merge", sha=PINNED))
    assert out["merged"] is False and out["reason"] == "head_moved"
    assert repo._pr.merge_calls == []


def test_pinned_merge_accepts_checks_that_graded_the_pin():
    repo = _PinRepo(_PinPR())
    svc = _svc(repo)

    async def _checks(*_a, **_k):
        return {"verdict": "passed", "head_sha": PINNED}

    svc.get_pull_request_checks = _checks
    out = asyncio.run(svc.merge_pull_request("o/r", 1, merge_method="merge", sha=PINNED))
    assert out["merged"] is True


# ── /pr-checks over REAL PyGithub, GitHub's permission model enforced ──
# Live 2026-10-11 (ci-wait, same call): GET /commits/{sha} under a token minted
# without contents:read answers 403 "Resource not accessible by integration".


async def test_pr_checks_route_reads_within_its_minted_permissions(monkeypatch):
    from routers import github_pr_status_operations as route
    from schemas.github_api_models import PRChecksRequest
    from services.tenant_binding import PERMS_PR_CHECKS
    from tests.unit import github_rest_fake as rest

    monkeypatch.setattr(gs, "commit_handle", _REAL_COMMIT_HANDLE)  # not the fakes' seam
    fake = rest.FakeGitHubRest(
        owner="o",
        repo="r",
        number=1,
        runs=[rest.check_run(1, "a"), rest.check_run(2, "b")],
    )
    seen: list = []
    with rest.patched(
        fake,
        seen,
        seam="routers.github_pr_status_operations._github",
        service=rest.real_github_service,
    ):
        out = await route.get_pull_request_checks(
            PRChecksRequest(owner="o", repo="r", pr_number=1), tenant_id="t1"
        )
    assert [s[3] for s in seen] == [PERMS_PR_CHECKS]
    assert "contents" not in PERMS_PR_CHECKS
    assert {rest.required_permission(p) for p in fake.paths} - {None} <= set(PERMS_PR_CHECKS)
    assert f"/repos/o/r/commits/{rest.SHA}" not in fake.paths
    assert out.verdict == "passed" and out.head_sha == rest.SHA and out.succeeded == 2
