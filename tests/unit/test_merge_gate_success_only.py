"""The merge gate counts ONLY `success` toward the floor.

THE BUG (george-rulings, 2026-09-30): `skipped`/`neutral` were "completed and
non-failing" AND counted in the total compared with `expected_checks`. So N skipped
checks + 0 successes satisfied `expected_checks: N`, and a required auditor that got
skipped read as passed. n8n approval-callback sends `requireChecks: true` +
`expectedChecks: N`, so this was the conveyor's own merge floor.

Also covered: named `required_checks` (latest run per name must be `success`), and
that EVERY page of check runs is read — with a real PyGithub PaginatedList walking a
`Link: rel="next"` header, not a Python list pretending to be one.
"""

from __future__ import annotations

import asyncio

import pytest
from github import Consts
from github.CheckRun import CheckRun
from github.PaginatedList import PaginatedList

from services.github.github_service import GitHubConfig, GitHubService


@pytest.fixture(autouse=True)
def _commit_seam(monkeypatch):
    """The fake repo hands out its commit through the seam production uses.

    Production never calls ``repo.get_commit`` here: that GET needs contents:read
    (see ``services.github.github_service.commit_handle``).
    """
    monkeypatch.setattr(
        "services.github.github_service.commit_handle", lambda repo, sha: repo.commit_for(sha)
    )


class _Run:
    def __init__(self, name: str, conclusion: str | None, status: str = "completed", id=None):
        self.name, self.status, self.conclusion, self.id = name, status, conclusion, id


class _PR:
    def __init__(self):
        self.merged, self.mergeable, self.title = False, True, "t"
        self.head = type("H", (), {"sha": "deadbeef"})()
        self.merge_calls: list[dict] = []

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        return type("R", (), {"merged": True, "sha": "abc123", "message": "ok"})()


class _Repo:
    private = True

    def __init__(self, runs):
        self._runs = runs
        self._pr = _PR()

    def get_pull(self, n):
        return self._pr

    def commit_for(self, sha):
        runs = self._runs
        return type("C", (), {"get_check_runs": lambda self: runs})()


def _svc(repo) -> GitHubService:
    s = GitHubService(GitHubConfig(token_id="t"))

    async def _gr(repo_name):
        return repo

    async def _ex(func, *a, **k):
        return func(*a, **k)

    s.get_repository = _gr  # type: ignore[method-assign]
    s._execute_with_retry = _ex  # type: ignore[method-assign]
    return s


def _merge(runs, **kw):
    repo = _Repo(runs)
    out = asyncio.run(_svc(repo).merge_pull_request("o/r", 1, **kw))
    return out, repo


N = 3


# (a) THE BUG ---------------------------------------------------------------
def test_a_n_skipped_zero_success_with_expected_n_is_refused():
    runs = [_Run(f"c{i}", "skipped") for i in range(N)]
    out, repo = _merge(runs, expected_checks=N)
    assert out["merged"] is False
    assert out["reason"] == "checks_pending"
    assert out["checks"]["succeeded"] == 0
    assert "skipped/neutral do not count" in out["message"]
    assert repo._pr.merge_calls == []


def test_a_neutral_counts_no_more_than_skipped():
    runs = [_Run("a", "success"), _Run("b", "neutral"), _Run("c", "skipped")]
    out, repo = _merge(runs, expected_checks=N)
    assert out["merged"] is False and out["checks"]["succeeded"] == 1
    assert repo._pr.merge_calls == []


def test_all_skipped_without_a_count_is_not_passed():
    """No count given: at least ONE success is required, or nothing was proven."""
    out, repo = _merge([_Run("a", "skipped"), _Run("b", "neutral")])
    assert out["merged"] is False and out["reason"] == "checks_pending"
    assert repo._pr.merge_calls == []


# (b) ------------------------------------------------------------------------
def test_b_n_success_is_allowed():
    out, repo = _merge([_Run(f"c{i}", "success") for i in range(N)], expected_checks=N)
    assert out["merged"] is True
    assert len(repo._pr.merge_calls) == 1


def test_action_required_and_stale_refuse():
    for bad in ("action_required", "stale"):
        runs = [_Run("a", "success"), _Run("b", "success"), _Run("x", bad)]
        out, repo = _merge(runs, expected_checks=2)
        assert out["merged"] is False and out["reason"] == "checks_failed", bad
        assert out["checks"]["failed_names"] == ["x"]
        assert repo._pr.merge_calls == []


# (c) ------------------------------------------------------------------------
def test_c_required_name_skipped_is_refused():
    runs = [_Run("unit", "success"), _Run("lint", "success"), _Run("auditor", "skipped")]
    out, repo = _merge(runs, expected_checks=2, required_checks=["auditor"])
    assert out["merged"] is False and out["reason"] == "checks_failed"
    assert out["checks"]["unsatisfied_required"] == ["auditor"]
    assert "auditor" in out["message"]
    assert repo._pr.merge_calls == []


def test_required_name_missing_is_refused_and_named():
    out, repo = _merge(
        [_Run("unit", "success")], expected_checks=1, required_checks=["auditor", "unit"]
    )
    assert out["merged"] is False and out["reason"] == "checks_pending"
    assert out["checks"]["missing_required"] == ["auditor"]
    assert "auditor" in out["message"]
    assert repo._pr.merge_calls == []


def test_required_name_latest_run_wins_by_id():
    """A re-run: the older run succeeded, the LATEST was skipped -> refused. And back."""
    stale_ok = [_Run("auditor", "success", id=10), _Run("auditor", "skipped", id=20)]
    out, _ = _merge(stale_ok, required_checks=["auditor"])
    assert out["merged"] is False and out["checks"]["unsatisfied_required"] == ["auditor"]
    fresh_ok = [_Run("auditor", "skipped", id=10), _Run("auditor", "success", id=20)]
    out, _ = _merge(fresh_ok, required_checks=["auditor"])
    assert out["merged"] is True


# (d) ------------------------------------------------------------------------
def test_d_required_success_others_skipped_count_met_by_successes_is_allowed():
    runs = [
        _Run("auditor", "success"),
        _Run("unit", "success"),
        _Run("e2e", "skipped"),
        _Run("deploy-preview", "neutral"),
    ]
    out, repo = _merge(runs, expected_checks=2, required_checks=["auditor"])
    assert out["merged"] is True, out
    assert len(repo._pr.merge_calls) == 1


def test_d_control_same_runs_count_above_successes_is_refused():
    """Control for (d): the skipped/neutral pair must not top the count up to 4."""
    runs = [
        _Run("auditor", "success"),
        _Run("unit", "success"),
        _Run("e2e", "skipped"),
        _Run("deploy-preview", "neutral"),
    ]
    out, repo = _merge(runs, expected_checks=4, required_checks=["auditor"])
    assert out["merged"] is False and out["checks"]["succeeded"] == 2
    assert repo._pr.merge_calls == []


# (e) pagination through a REAL PaginatedList --------------------------------
class _PagedRequester:
    """Serves 150 check runs as a 100-run page + a 50-run page linked by rel=next."""

    per_page = Consts.DEFAULT_PER_PAGE
    is_not_lazy = False
    is_lazy = True

    def __init__(self, last_page_conclusion: str):
        self.last = last_page_conclusion
        self.calls: list[str] = []

    def requestJsonAndCheck(self, verb, url, parameters=None, headers=None, **_):  # noqa: N802 (PyGithub API)
        self.calls.append(url)
        page = 2 if "page=2" in url else 1
        n = 100 if page == 1 else 50
        conclusion = "success" if page == 1 else self.last
        runs = [
            {
                "id": page * 1000 + i,
                "name": f"p{page}-{i}",
                "status": "completed",
                "conclusion": conclusion,
            }
            for i in range(n)
        ]
        hdr = (
            {"link": '<https://api.github.com/r/commits/x/check-runs?page=2>; rel="next"'}
            if page == 1
            else {}
        )
        return hdr, {"total_count": 150, "check_runs": runs}


def _paged(last_page_conclusion: str):
    req = _PagedRequester(last_page_conclusion)
    pl = PaginatedList(
        CheckRun,
        req,
        "https://api.github.com/r/commits/x/check-runs",
        {},
        list_item="check_runs",
    )
    return pl, req


def test_e_more_than_100_runs_across_pages_are_all_counted():
    pl, req = _paged("success")
    out, _ = _merge(pl, expected_checks=150)
    assert len(req.calls) == 2, req.calls  # the second page WAS fetched
    assert out["merged"] is True
    assert out.get("checks") is None or out["checks"]["succeeded"] == 150


def test_e_a_failure_on_page_two_refuses():
    """Positive control: a page-1-only reader would call this passed."""
    pl, req = _paged("failure")
    out, repo = _merge(pl, expected_checks=100)
    assert len(req.calls) == 2
    assert out["merged"] is False and out["reason"] == "checks_failed"
    assert len(out["checks"]["failed_names"]) == 50
    assert repo._pr.merge_calls == []


def test_e_page_two_successes_are_needed_to_meet_the_floor():
    pl, _ = _paged("skipped")
    out, _ = _merge(pl, expected_checks=101)
    assert out["merged"] is False and out["checks"]["succeeded"] == 100


# (f) mutation control --------------------------------------------------------
def test_f_mutation_counting_completed_non_failures_would_pass_case_a():
    """Re-implement the OLD floor (completed & not failing counts) and it must DISAGREE
    with the real gate on case (a). If they ever agree, (a) is not testing the fix.
    The source-level mutation (revert the filter in github_service.py, watch (a) go red)
    is recorded in the PR/commit notes."""
    runs = [_Run(f"c{i}", "skipped") for i in range(N)]
    real, _ = _merge(runs, expected_checks=N)
    old_floor_met = (
        sum(1 for r in runs if r.status == "completed" and r.conclusion not in ("failure",)) >= N
    )
    assert old_floor_met is True
    assert real["merged"] is False
