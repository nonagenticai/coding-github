"""CI-wait decision rules (pure) and the GitHub read that feeds them (faked PyGithub)."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from services.ci_waits import evaluator as ev
from services.ci_waits.evaluator import (
    FinalEvaluationError,
    Snapshot,
    decide,
    fetch_snapshot,
)

SHA = "a" * 40
NEW = "b" * 40


def wait(**kw):
    base = {
        "wait_id": "w1",
        "tenant_id": "t1",
        "owner": "nonagenticai",
        "repo": "ems",
        "pr_number": 7,
        "head_sha": SHA,
        "required_checks": [],
        "expected_checks": 1,
    }
    base.update(kw)
    return base


def run(name, status="completed", conclusion="success"):
    return {"kind": "check_run", "name": name, "status": status, "conclusion": conclusion}


def st(name, state):
    return {"kind": "status", "name": name, "status": state, "conclusion": None}


def snap(checks, *, head=SHA, state="open", merged=False, workflows=None, suites=()):
    s = Snapshot(
        head_sha=head, state=state, merged=merged, checks=checks, active_workflows=workflows
    )
    s.suites = list(suites)
    return s


def suite(sid, status="completed", conclusion="success", runs=1, app="github-actions"):
    return {"id": sid, "app": app, "status": status, "conclusion": conclusion, "runs": runs}


#: The measured gitops-production shape: an app with checks:write that never reports.
PHANTOM = suite(9, status="queued", conclusion=None, runs=0, app="code-graph-helper-app")
FRESH = {"age_seconds": 5.0}
OLD = {"age_seconds": 601.0}


# ── decide ────────────────────────────────────────────────────────────


def test_all_complete_and_green_is_success():
    d = decide(snap([run("lint"), run("test"), st("ci/ext", "success")]), wait())
    assert d is not None and d.overall == "success" and len(d.checks) == 3


def test_one_still_running_keeps_waiting():
    assert decide(snap([run("lint"), run("test", "in_progress", None)]), wait()) is None


def test_a_failure_does_not_resolve_until_everything_completed():
    """'All complete' is the contract: an early failure still waits for the rest."""
    checks = [run("lint", conclusion="failure"), run("test", "queued", None)]
    assert decide(snap(checks), wait()) is None


@pytest.mark.parametrize("conclusion", sorted(ev.FAILING_CONCLUSIONS))
def test_failing_conclusions_fail(conclusion):
    d = decide(snap([run("lint"), run("test", conclusion=conclusion)]), wait())
    assert d is not None and d.overall == "failure"


@pytest.mark.parametrize("state", ["failure", "error"])
def test_failing_status_fails(state):
    d = decide(snap([run("lint"), st("ext", state)]), wait())
    assert d is not None and d.overall == "failure"


def test_pending_status_keeps_waiting():
    assert decide(snap([run("lint"), st("ext", "pending")]), wait()) is None


def test_all_skipped_is_skipped_never_success():
    d = decide(snap([run("a", conclusion="skipped"), run("b", conclusion="neutral")]), wait())
    assert d is not None and d.overall == "skipped"


def test_new_head_is_superseded_with_the_new_sha():
    d = decide(snap([], head=NEW), wait())
    assert d is not None and d.overall == "superseded" and d.new_head_sha == NEW


def test_no_checks_and_no_active_workflow_resolves_no_checks():
    d = decide(snap([], workflows=0), wait())
    assert d is not None and d.overall == "no_checks"


def test_no_checks_yet_but_workflows_exist_keeps_waiting():
    assert decide(snap([], workflows=2), wait()) is None


def test_no_checks_and_workflow_count_unknown_keeps_waiting():
    assert decide(snap([], workflows=None), wait()) is None


def test_required_check_missing_keeps_waiting_even_if_the_rest_is_green():
    assert decide(snap([run("lint")]), wait(required_checks=["test"])) is None
    d = decide(snap([run("lint"), run("test")]), wait(required_checks=["test"]))
    assert d is not None and d.overall == "success"


def test_required_check_may_be_a_status_context():
    d = decide(snap([run("lint"), st("ci/ext", "success")]), wait(required_checks=["ci/ext"]))
    assert d is not None and d.overall == "success"


def test_expected_checks_floor():
    assert decide(snap([run("lint")]), wait(expected_checks=2)) is None
    d = decide(snap([run("lint"), run("test")]), wait(expected_checks=2))
    assert d is not None and d.overall == "success"


def test_closed_unmerged_before_ci_settles_is_pr_closed():
    d = decide(snap([run("t", "queued", None)], state="closed"), wait())
    assert d is not None and d.overall == "pr_closed"
    d = decide(snap([], state="closed", workflows=3), wait())
    assert d is not None and d.overall == "pr_closed"


def test_closed_unmerged_with_ci_settled_reports_the_ci_result():
    d = decide(snap([run("t", conclusion="failure")], state="closed"), wait())
    assert d is not None and d.overall == "failure"


def test_merged_pr_keeps_waiting_for_its_checks():
    assert (
        decide(snap([run("t", "in_progress", None)], state="closed", merged=True), wait()) is None
    )


# ── check suites (startup_failure, multi-workflow, phantoms, missing) ──


def test_startup_failure_suite_with_zero_runs_is_a_failure_not_a_forever_wait():
    """Measured: gitops check-suite 89355492045 completed/startup_failure, 0 runs."""
    s = snap([], workflows=3, suites=[suite(89355492045, conclusion="startup_failure", runs=0)])
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "failure"
    assert d.checks[0]["kind"] == "check_suite"
    assert d.checks[0]["conclusion"] == "startup_failure"


@pytest.mark.parametrize("conclusion", sorted(ev.FAILING_CONCLUSIONS))
def test_failing_empty_suite_fails_even_beside_green_runs(conclusion):
    s = snap(
        [run("lint"), run("test")],
        suites=[suite(1, runs=2), suite(2, conclusion=conclusion, runs=0)],
    )
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "failure"


def test_incomplete_actions_suite_without_runs_blocks_settling():
    """Workflow A green, workflow B's suite exists but has no runs yet: not settled."""
    s = snap([run("lint")], suites=[suite(1), suite(2, status="queued", conclusion=None, runs=0)])
    assert decide(s, wait(**FRESH)) is None


def test_incomplete_suite_with_runs_blocks_settling():
    s = snap([run("lint")], suites=[suite(1), suite(2, status="in_progress", conclusion=None)])
    assert decide(s, wait(**FRESH)) is None


def test_stuck_empty_actions_suite_past_grace_fails_never_succeeds():
    s = snap([run("lint")], suites=[suite(1), suite(2, status="queued", conclusion=None, runs=0)])
    d = decide(s, wait(**OLD))
    assert d is not None and d.overall == "failure"
    assert any(c["conclusion"] == "stale" for c in d.checks)


def test_phantom_app_suite_never_blocks():
    s = snap([run("lint")], suites=[suite(1), PHANTOM])
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "success"


def test_suites_completed_with_no_runs_at_all_is_no_checks():
    s = snap([], workflows=2, suites=[suite(1, conclusion="skipped", runs=0), PHANTOM])
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "no_checks"


def test_workflows_but_no_suite_for_the_sha_resolves_no_checks_after_grace():
    """paths-ignore / [skip ci]: GitHub never creates a suite; must not park forever."""
    assert decide(snap([], workflows=2, suites=[PHANTOM]), wait(**FRESH)) is None
    assert decide(snap([], workflows=2), wait()) is None  # age unknown: keep waiting
    d = decide(snap([], workflows=2, suites=[PHANTOM]), wait(**OLD))
    assert d is not None and d.overall == "no_checks"


def test_grace_is_the_callers_window():
    s = snap([], workflows=2)
    assert decide(s, wait(age_seconds=100.0), grace_seconds=120) is None
    d = decide(s, wait(age_seconds=130.0), grace_seconds=120)
    assert d is not None and d.overall == "no_checks"


def test_required_check_never_appeared_once_suites_completed_is_failure():
    s = snap([run("lint")], suites=[suite(1)])
    d = decide(s, wait(required_checks=["test"], **FRESH))
    assert d is not None and d.overall == "failure"
    assert {"kind": "missing", "name": "test"}.items() <= d.checks[-1].items()


def test_expected_floor_unmet_once_suites_completed_is_failure():
    s = snap([run("lint")], suites=[suite(1)])
    d = decide(s, wait(expected_checks=3, **FRESH))
    assert d is not None and d.overall == "failure"
    assert d.checks[-1]["kind"] == "missing"


def test_missing_check_on_a_merged_pr_resolves():
    s = snap([run("lint")], state="closed", merged=True, suites=[suite(1)])
    d = decide(s, wait(required_checks=["test"], **FRESH))
    assert d is not None and d.overall == "failure"


def test_missing_with_statuses_only_waits_until_grace():
    s = snap([st("ci/ext", "success")])
    assert decide(s, wait(required_checks=["ci/other"], **FRESH)) is None
    d = decide(s, wait(required_checks=["ci/other"], **OLD))
    assert d is not None and d.overall == "failure"


# ── fetch_snapshot (faked PyGithub behind the real _github seam) ─────


class FakeRun(SimpleNamespace):
    pass


def _fsuite(sid, status="completed", conclusion="success", runs=1, app="github-actions"):
    return SimpleNamespace(
        id=sid,
        app=SimpleNamespace(slug=app),
        status=status,
        conclusion=conclusion,
        latest_check_runs_count=runs,
    )


def _gh_repo(
    *, head=SHA, state="open", merged=False, runs=(), statuses=(), workflows=(), suites=()
):
    commit = SimpleNamespace(
        get_check_suites=lambda: list(suites),
        get_check_runs=lambda: list(runs),
        get_combined_status=lambda: SimpleNamespace(statuses=list(statuses)),
    )
    pr = SimpleNamespace(head=SimpleNamespace(sha=head), state=state, merged=merged)
    calls = {"commit": []}

    def commit_for(sha):
        calls["commit"].append(sha)
        return commit

    # No ``get_commit``: GET /commits/{sha} needs contents:read (see PERMS_CI_WAIT).
    repo = SimpleNamespace(
        get_pull=lambda n: pr,
        commit_for=commit_for,
        get_workflows=lambda: list(workflows),
    )
    return repo, calls


class FakeService:
    def __init__(self, repo):
        self.repo = repo

    async def get_repository(self, name):
        return self.repo

    async def _execute_with_retry(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@contextmanager
def _patch_github(repo=None, exc: Exception | None = None, seen: list | None = None):
    @asynccontextmanager
    async def fake(owner, repo_name, tenant_id, permissions):
        if seen is not None:
            seen.append((owner, repo_name, tenant_id, permissions))
        if exc is not None:
            raise exc
        yield FakeService(repo)

    with (
        patch("routers.github_issue_operations._github", new=fake),
        patch.object(ev, "_commit_handle", new=lambda r, sha: r.commit_for(sha)),
    ):
        yield


async def test_fetch_reads_latest_run_per_name_and_statuses_under_the_waits_tenant():
    runs = [
        FakeRun(id=1, name="test", status="completed", conclusion="failure", html_url="u1"),
        FakeRun(id=5, name="test", status="completed", conclusion="success", html_url="u5"),
        FakeRun(id=3, name="lint", status="in_progress", conclusion=None, html_url="u3"),
    ]
    statuses = [SimpleNamespace(context="ext", state="success", target_url="t")]
    repo, calls = _gh_repo(runs=runs, statuses=statuses)
    seen: list = []
    with _patch_github(repo, seen=seen):
        s = await fetch_snapshot(wait())
    assert seen == [("nonagenticai", "ems", "t1", ev.PERMS_CI_WAIT)]
    assert calls["commit"] == [SHA]  # the STORED sha, never anything else
    by_name = {c["name"]: c for c in s.checks}
    assert by_name["test"]["conclusion"] == "success" and by_name["test"]["url"] == "u5"
    assert by_name["lint"]["status"] == "in_progress"
    assert by_name["ext"] == {
        "kind": "status",
        "name": "ext",
        "status": "success",
        "conclusion": "success",
        "url": "t",
    }
    assert s.active_workflows is None  # only read when there are no checks


async def test_same_job_name_in_two_workflows_keeps_both_and_fails():
    """Repro: 'test' failure id 100 (suite A) + 'test' success id 101 (suite B)."""
    runs = [
        FakeRun(id=100, name="test", status="completed", conclusion="failure", check_suite_id=1),
        FakeRun(id=101, name="test", status="completed", conclusion="success", check_suite_id=2),
    ]
    repo, _ = _gh_repo(runs=runs, suites=[_fsuite(1), _fsuite(2)])
    with _patch_github(repo):
        s = await fetch_snapshot(wait())
    assert sorted(c["conclusion"] for c in s.checks) == ["failure", "success"]
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "failure"


async def test_rerun_in_the_same_suite_still_supersedes():
    runs = [
        FakeRun(id=100, name="test", status="completed", conclusion="failure", check_suite_id=1),
        FakeRun(id=101, name="test", status="completed", conclusion="success", check_suite_id=1),
    ]
    repo, _ = _gh_repo(runs=runs, suites=[_fsuite(1)])
    with _patch_github(repo):
        s = await fetch_snapshot(wait())
    assert [c["conclusion"] for c in s.checks] == ["success"]


async def test_fetch_reads_suites_with_app_and_run_count():
    repo, _ = _gh_repo(
        suites=[
            _fsuite(7, conclusion="startup_failure", runs=0),
            _fsuite(8, status="queued", conclusion=None, runs=0, app="code-graph-helper-app"),
        ],
        workflows=[SimpleNamespace(state="active")],
    )
    with _patch_github(repo):
        s = await fetch_snapshot(wait())
    assert s.suites == [
        {
            "id": 7,
            "app": "github-actions",
            "status": "completed",
            "conclusion": "startup_failure",
            "runs": 0,
        },
        {
            "id": 8,
            "app": "code-graph-helper-app",
            "status": "queued",
            "conclusion": None,
            "runs": 0,
        },
    ]
    d = decide(s, wait(**FRESH))
    assert d is not None and d.overall == "failure"


async def test_fetch_moved_head_does_not_read_checks():
    repo, calls = _gh_repo(head=NEW)
    with _patch_github(repo):
        s = await fetch_snapshot(wait())
    assert s.head_sha == NEW and calls["commit"] == []


async def test_fetch_counts_active_workflows_only_when_no_checks():
    wfs = [SimpleNamespace(state="active"), SimpleNamespace(state="disabled_manually")]
    repo, _ = _gh_repo(workflows=wfs)
    with _patch_github(repo):
        s = await fetch_snapshot(wait())
    assert s.checks == [] and s.active_workflows == 1


@pytest.mark.parametrize("status", [401, 403, 404, 409, 412, 422])
async def test_tenant_gate_refusal_is_final(status):
    with (
        _patch_github(exc=HTTPException(status_code=status, detail="refused")),
        pytest.raises(FinalEvaluationError),
    ):
        await fetch_snapshot(wait())


@pytest.mark.parametrize("status", [400, 502, 503])
async def test_ladder_outage_is_transient(status):
    with (
        _patch_github(exc=HTTPException(status_code=status, detail="later")),
        pytest.raises(HTTPException),
    ):
        await fetch_snapshot(wait())


async def test_github_404_is_final_and_5xx_transient():
    class GhError(Exception):
        def __init__(self, status):
            self.status = status

    with _patch_github(exc=GhError(404)), pytest.raises(FinalEvaluationError):
        await fetch_snapshot(wait())
    with _patch_github(exc=GhError(502)), pytest.raises(GhError):
        await fetch_snapshot(wait())


# ── fetch_snapshot over REAL PyGithub, with GitHub's permission model enforced ──
# (tests/unit/github_rest_fake.py: only Requester.requestJsonAndCheck is faked.)

from github.GithubException import GithubException  # noqa: E402

from tests.unit import github_rest_fake as rest  # noqa: E402


def test_perms_ci_wait_is_the_read_set_without_contents():
    assert ev.PERMS_CI_WAIT == {
        "pull_requests": "read",
        "checks": "read",
        "statuses": "read",
        "actions": "read",
    }


async def test_every_call_is_covered_by_the_minted_permissions():
    """Live 2026-10-11: GET /commits/{sha} under PERMS_CI_WAIT → 403, wait parked."""
    fake = rest.FakeGitHubRest(
        runs=[rest.check_run(1, "test")],
        statuses=[{"context": "ext", "state": "success", "target_url": "t"}],
        suites=[rest.check_suite(1)],
    )
    seen: list = []
    with rest.patched(fake, seen):
        s = await fetch_snapshot(wait())
    assert seen == [("nonagenticai", "ems", "t1", ev.PERMS_CI_WAIT)]
    needed = {rest.required_permission(p) for p in fake.paths} - {None}
    assert needed <= set(ev.PERMS_CI_WAIT)
    assert f"/repos/nonagenticai/ems/commits/{SHA}" not in fake.paths
    assert f"/repos/nonagenticai/ems/commits/{SHA}/check-runs" in fake.paths
    assert {c["name"] for c in s.checks} == {"test", "ext"}
    assert s.suites[0]["app"] == "github-actions"


async def test_no_checks_reads_workflows_within_the_minted_permissions():
    fake = rest.FakeGitHubRest(workflows=[{"id": 1, "state": "active"}])
    with rest.patched(fake):
        s = await fetch_snapshot(wait())
    assert s.checks == [] and s.active_workflows == 1
    assert fake.paths[-1] == "/repos/nonagenticai/ems/actions/workflows"


@pytest.mark.parametrize(
    ("runs", "workflows"),
    [
        ([rest.check_run(1, "test", status="in_progress", conclusion=None)], []),
        ([], [{"id": 1, "state": "active"}]),
    ],
    ids=["ci-running", "no-suite-yet"],
)
async def test_closed_pr_resolves_pr_closed(runs, workflows):
    """The live probe's shape: a CLOSED, unmerged PR must resolve, never park."""
    fake = rest.FakeGitHubRest(state="closed", runs=runs, workflows=workflows)
    with rest.patched(fake):
        s = await fetch_snapshot(wait(pr_number=1))
    d = decide(s, wait(pr_number=1, **FRESH))
    assert d is not None and d.overall == "pr_closed"


async def test_permission_403_is_final_with_githubs_message():
    fake = rest.FakeGitHubRest(fail=(403, {}, dict(rest.NOT_ACCESSIBLE)))
    with rest.patched(fake), pytest.raises(FinalEvaluationError) as err:
        await fetch_snapshot(wait())
    assert "403" in str(err.value) and "Resource not accessible by integration" in str(err.value)


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"x-ratelimit-remaining": "0"}, "API rate limit exceeded for installation ID 1."),
        ({}, "You have exceeded a secondary rate limit. Please wait a few minutes."),
        ({"Retry-After": "60"}, "Forbidden"),
        ({"X-RateLimit-Remaining": "0"}, "Forbidden"),
    ],
    ids=["primary", "secondary", "retry-after", "remaining-0"],
)
async def test_rate_limit_403_stays_transient(headers, message):
    fake = rest.FakeGitHubRest(fail=(403, headers, {"message": message}))
    with rest.patched(fake), pytest.raises(GithubException) as err:
        await fetch_snapshot(wait())
    assert not isinstance(err.value, FinalEvaluationError) and err.value.status == 403


async def test_github_422_is_final():
    body = {"message": f"No commit found for SHA: {SHA}"}
    with rest.patched(rest.FakeGitHubRest(fail=(422, {}, body))):
        with pytest.raises(FinalEvaluationError, match="422.*No commit found"):
            await fetch_snapshot(wait())


async def test_github_401_is_transient():
    """A revoked-but-cached installation token heals at the next mint."""
    with rest.patched(rest.FakeGitHubRest(fail=(401, {}, {"message": "Bad credentials"}))):
        with pytest.raises(GithubException) as err:
            await fetch_snapshot(wait())
    assert err.value.status == 401
