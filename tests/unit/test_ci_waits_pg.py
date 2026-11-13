"""CI-checks waits against REAL Postgres: triggers, resolution, callbacks, races.

Needs ``TEST_DATABASE_URL`` (or ``DATABASE_URL``): in CI the archetype's per-job
ephemeral postgres:18 sidecar (``has_postgres: true``); locally an ephemeral
``postgres:18`` container. Skipped when neither is set. The schema comes from
the real alembic migration, so these tests also prove the migration.

GitHub (the snapshot fetcher) and the HTTP client are fakes; everything between
them — the store's conditional transitions, the evaluation triggers, the sweep
and the single-sender callback hand-off — is the production code against
Postgres. Shape copied from coding-containers tests/unit/test_exec_jobs_pg.py.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import psycopg
import pytest

from db import session as db_session
from services.ci_waits import callback as cb
from services.ci_waits.evaluator import FinalEvaluationError, Snapshot
from services.ci_waits.manager import CIWaitManager, callback_body
from services.ci_waits.store import CIWaitStore
from tests.unit import github_rest_fake as rest
from tests.unit.test_ci_wait_manager_shutdown import finishes_within

DSN = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
if DSN.startswith("sqlite"):
    DSN = ""
pytestmark = pytest.mark.skipif(not DSN, reason="needs TEST_DATABASE_URL / DATABASE_URL")

PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
TENANT = "5f868d42-03f8-51f0-a93c-3e83070dc196"
SHA = "a" * 40
NEW = "b" * 40


@pytest.fixture(scope="module")
def migrated():
    from alembic.config import Config

    from alembic import command

    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = DSN
    try:
        command.upgrade(cfg, "head")
        command.upgrade(cfg, "head")  # idempotent: a re-run of the initContainer is a no-op
    finally:
        if old is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old


@pytest.fixture
async def store(migrated, monkeypatch):
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "pgbouncer_url", DSN)
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        conn.execute("TRUNCATE ci_waits")
    yield CIWaitStore()
    await db_session.close_pool()


class Recorder:
    """Stands in for httpx.AsyncClient across every sender; records POST bodies."""

    def __init__(self, status: int = 200):
        self.status = status
        self.posts: list[tuple[str, dict]] = []
        self.script: list = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        r = self.script.pop(0) if self.script else self.status
        if isinstance(r, Exception):
            raise r
        status, text = r if isinstance(r, tuple) else (r, "")
        return SimpleNamespace(status_code=status, text=text)


@pytest.fixture
def recorder():
    rec = Recorder()
    with (
        patch.object(cb.httpx, "AsyncClient", new=rec),
        patch.object(
            cb, "get_settings", return_value=SimpleNamespace(ci_callback_prefixes=(PREFIX,))
        ),
    ):
        yield rec


@pytest.fixture
def one_attempt_rounds(monkeypatch):
    """One POST per round, so a test drives rounds without the 2 s / 4 s sleeps."""
    monkeypatch.setattr(cb, "CALLBACK_ATTEMPTS", 1)


class FakeGitHub:
    """The snapshot fetcher: GitHub's answer per (repo, pr), mutable; counts reads."""

    def __init__(self):
        self.answers: dict[tuple[str, int], Snapshot | Exception] = {}
        self.calls: list[str] = []

    def set(self, snapshot: Snapshot | Exception, repo: str = "ems", pr: int = 7) -> None:
        self.answers[(repo, pr)] = snapshot

    async def __call__(self, wait):
        self.calls.append(wait["wait_id"])
        answer = self.answers[(wait["repo"], wait["pr_number"])]
        if isinstance(answer, Exception):
            raise answer
        return answer


def run(name, status="completed", conclusion="success"):
    return {
        "kind": "check_run",
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "url": f"https://github.com/x/{name}",
    }


def pending(head=SHA):
    return Snapshot(head, "open", False, [run("ci", "in_progress", None)])


def green(head=SHA):
    return Snapshot(head, "open", False, [run("ci"), run("lint")])


def _manager(store, gh, owner="pod-a", stale=600) -> CIWaitManager:
    return CIWaitManager(store, owner=owner, stale_after_seconds=stale, fetcher=gh)


async def _wait(store, *, repo="ems", pr=7, sha=SHA, url=None, tenant=TENANT, **kw):
    row, created = await store.create_wait(
        wait_id=str(uuid.uuid7()),
        tenant_id=tenant,
        owner="nonagenticai",
        repo=repo,
        pr_number=pr,
        head_sha=sha,
        required_checks=kw.get("required", []),
        expected_checks=kw.get("expected", 1),
        callback_url=url or PREFIX + uuid.uuid4().hex,
    )
    assert created
    return row


def _sql(query: str, *params):
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else []


async def _row(store, wait):
    return await store.get_wait(wait_id=wait["wait_id"], tenant_id=wait["tenant_id"])


def _delivery(repo="nonagenticai/ems", sha=SHA):
    return {
        "action": "completed",
        "check_suite": {"head_sha": sha},
        "repository": {"full_name": repo},
    }


# ── registration-time evaluation ──────────────────────────────────────


async def test_checks_already_complete_at_registration_call_back_immediately(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    m = _manager(store, gh)
    w = await _wait(store)
    assert await m.evaluate(w) == "success"
    row = await _row(store, w)
    assert row["status"] == "resolved" and row["overall"] == "success"
    assert row["callback_state"] == "delivered"
    ((url, body),) = recorder.posts
    assert url == w["callback_url"]
    assert body["wait_id"] == w["wait_id"] and body["overall"] == "success"
    assert body["repo"] == "nonagenticai/ems" and body["pr_number"] == 7
    assert body["head_sha"] == SHA and body["new_head_sha"] is None
    assert [(c["name"], c["conclusion"], c["url"]) for c in body["checks"]] == [
        ("ci", "success", "https://github.com/x/ci"),
        ("lint", "success", "https://github.com/x/lint"),
    ]
    assert body["completed_at"] is not None
    assert body["links"]["status"] == f"/api/repositories/ci-wait/{w['wait_id']}"


async def test_schedule_evaluation_runs_in_the_background(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    m = _manager(store, gh)
    w = await _wait(store)
    m.schedule_evaluation(w)
    await asyncio.gather(*list(m._tasks))
    assert (await _row(store, w))["callback_state"] == "delivered"


async def test_pending_at_registration_records_and_sends_nothing(store, recorder):
    gh = FakeGitHub()
    gh.set(pending())
    m = _manager(store, gh)
    w = await _wait(store)
    assert await m.evaluate(w) is None
    row = await _row(store, w)
    assert row["status"] == "waiting" and row["eval_count"] == 1
    assert row["checks"][0]["status"] == "in_progress"
    assert row["next_eval_at"] is None  # a clean pending evaluation schedules NOTHING
    assert recorder.posts == []
    assert await m.sweep_once() == 0  # and the sweep never asks GitHub about it
    assert gh.calls == [w["wait_id"]]


# ── webhook triggers ──────────────────────────────────────────────────


async def test_delivery_for_the_repo_resolves_the_wait(store, recorder):
    gh = FakeGitHub()
    gh.set(pending())
    m = _manager(store, gh)
    w = await _wait(store)
    await m.evaluate(w)
    gh.set(green())
    assert await m.on_delivery("check_suite", _delivery()) == 1
    assert (await _row(store, w))["overall"] == "success"
    assert len(recorder.posts) == 1
    # a later delivery finds nothing open: no second callback
    assert await m.on_delivery("check_suite", _delivery()) == 0
    assert len(recorder.posts) == 1


@pytest.mark.parametrize("event", ["check_run", "workflow_run", "status", "pull_request"])
async def test_every_ci_event_is_a_trigger(store, recorder, event):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    assert await _manager(store, gh).on_delivery(event, _delivery()) == 1
    assert (await _row(store, w))["status"] == "resolved"


async def test_delivery_bytes_payload_is_parsed(store, recorder):
    import json

    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    await _manager(store, gh).on_delivery("check_suite", json.dumps(_delivery()).encode())
    assert (await _row(store, w))["status"] == "resolved"


async def test_webhook_for_another_sha_never_resolves_the_wait(store, recorder):
    """A completed suite for ANOTHER sha is a trigger, not an answer: the wait is
    evaluated against its STORED sha, whose checks are still running."""
    gh = FakeGitHub()
    gh.set(pending())
    w = await _wait(store)
    await _manager(store, gh).on_delivery("check_suite", _delivery(sha="c" * 40))
    row = await _row(store, w)
    assert row["status"] == "waiting" and recorder.posts == []


async def test_delivery_for_another_repo_does_not_touch_a_fresh_wait(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    _sql("UPDATE ci_waits SET last_evaluated_at = now() WHERE id = %s", w["wait_id"])
    assert await _manager(store, gh).on_delivery("check_suite", _delivery("nonagenticai/kag")) == 0
    assert gh.calls == [] and (await _row(store, w))["status"] == "waiting"


async def test_repo_match_is_case_insensitive(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    assert await _manager(store, gh).on_delivery("check_suite", _delivery("NonAgenticAI/EMS")) == 1
    assert (await _row(store, w))["status"] == "resolved"


async def test_superseded_head_resolves_with_the_new_sha(store, recorder):
    gh = FakeGitHub()
    gh.set(pending())
    m = _manager(store, gh)
    w = await _wait(store)
    await m.evaluate(w)
    gh.set(pending(head=NEW))  # a new push: the new head's CI is only starting
    await m.on_delivery(
        "pull_request", {"action": "synchronize", "repository": {"full_name": "nonagenticai/ems"}}
    )
    row = await _row(store, w)
    assert row["overall"] == "superseded" and row["new_head_sha"] == NEW
    ((_, body),) = recorder.posts
    assert body["overall"] == "superseded" and body["new_head_sha"] == NEW
    assert body["head_sha"] == SHA and body["checks"] == []


# ── lost-webhook coverage ─────────────────────────────────────────────


async def test_lost_webhook_is_covered_by_startup_evaluation(store, recorder):
    """The completing delivery never arrived (pod down, andon, GitHub hiccup).
    A fresh process evaluates every open wait at startup."""
    gh = FakeGitHub()
    gh.set(pending())
    w = await _wait(store)
    await _manager(store, gh, "pod-a").evaluate(w)
    gh.set(green())  # CI finished; the delivery saying so was lost
    restarted = _manager(CIWaitStore(), gh, "pod-b")
    assert await restarted.startup_evaluate() == 1
    row = await _row(store, w)
    assert row["overall"] == "success" and row["callback_owner"] == "pod-b"
    assert len(recorder.posts) == 1


async def test_start_runs_the_startup_evaluation(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    m = _manager(store, gh)
    m.start(interval=3600)
    for _ in range(100):
        if (await _row(store, w))["callback_state"] == "delivered":
            break
        await asyncio.sleep(0.02)
    await m.shutdown()
    assert (await _row(store, w))["callback_state"] == "delivered"


async def test_shutdown_survives_a_cancel_swallowed_by_the_real_pool(store, recorder):
    """psycopg_pool's getconn swallows a CancelledError raised during its
    connection check. CI hit that window by chance (300 s hang); here the check
    is slowed so the shutdown cancel lands inside it every time."""
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)  # opens the real pool on this loop
    pool = await db_session.get_pool()
    real_check = pool._check
    sweeper_checking = asyncio.Event()
    m = _manager(store, gh)

    async def slow_check(conn):
        await real_check(conn)
        if asyncio.current_task() is not m._sweeper:
            return
        sweeper_checking.set()
        try:
            await asyncio.sleep(0.3)
        finally:
            sweeper_checking.clear()

    pool._check = slow_check
    try:
        m.start(interval=3600)
        for _ in range(200):
            if (await _row(store, w))["callback_state"] == "delivered":
                break
            await asyncio.sleep(0.02)
        await asyncio.wait_for(sweeper_checking.wait(), 5)  # sweeper inside getconn's check
        sweeper = m._sweeper
        # Before the fix: the 3600 s sleep. Not wait_for (see finishes_within).
        assert await finishes_within(m.shutdown(), 5)
        assert sweeper is not None and sweeper.done()
    finally:
        pool._check = real_check
    assert (await _row(store, w))["callback_state"] == "delivered"


async def test_any_delivery_reevaluates_stale_waits_of_other_repos(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    _sql(
        "UPDATE ci_waits SET last_evaluated_at = now() - interval '700 seconds' WHERE id = %s",
        w["wait_id"],
    )
    # a delivery for an unrelated repo, even an unrelated event, reaches the stale wait
    assert await _manager(store, gh, stale=600).on_delivery("installation_repositories", {}) == 1
    assert (await _row(store, w))["status"] == "resolved"


# ── evaluation failures ───────────────────────────────────────────────


async def test_transient_evaluation_failure_is_retried_by_the_sweep(store, recorder):
    gh = FakeGitHub()
    gh.set(httpx.ConnectError("github down"))
    m = _manager(store, gh)
    w = await _wait(store)
    assert await m.evaluate(w) is None
    row = await _row(store, w)
    assert row["status"] == "waiting" and row["eval_failures"] == 1
    assert "github down" in row["last_eval_error"] and row["next_eval_at"] is not None
    assert await m.sweep_once() == 0  # not due yet
    _sql(
        "UPDATE ci_waits SET next_eval_at = now() - interval '1 second' WHERE id = %s", w["wait_id"]
    )
    gh.set(green())
    assert await m.sweep_once() == 1
    row = await _row(store, w)
    assert row["overall"] == "success" and row["callback_state"] == "delivered"


async def test_due_evaluation_retry_is_claimed_by_one_replica(store, recorder):
    gh = FakeGitHub()
    gh.set(httpx.ConnectError("down"))
    w = await _wait(store)
    await _manager(store, gh).evaluate(w)
    _sql(
        "UPDATE ci_waits SET next_eval_at = now() - interval '1 second' WHERE id = %s", w["wait_id"]
    )
    gh.set(green())
    a, b = _manager(CIWaitStore(), gh, "pod-b"), _manager(CIWaitStore(), gh, "pod-c")
    counts = await asyncio.gather(a.sweep_once(), b.sweep_once())
    assert sum(counts) == 1 and len(recorder.posts) == 1


async def test_final_evaluation_error_resolves_error_and_calls_back(store, recorder):
    gh = FakeGitHub()
    gh.set(FinalEvaluationError("GitHub 404: gone"))
    w = await _wait(store)
    assert await _manager(store, gh).evaluate(w) == "error"
    row = await _row(store, w)
    assert row["overall"] == "error" and "404" in row["error"]
    ((_, body),) = recorder.posts
    assert body["overall"] == "error" and "404" in body["error"]


async def test_no_checks_and_no_workflows_resolves_at_registration(store, recorder):
    gh = FakeGitHub()
    gh.set(Snapshot(SHA, "open", False, [], active_workflows=0))
    w = await _wait(store)
    assert await _manager(store, gh).evaluate(w) == "no_checks"
    assert recorder.posts[0][1]["overall"] == "no_checks"


async def test_suiteless_sha_resolves_no_checks_via_the_stale_backstop(store, recorder):
    """Workflows exist, GitHub never made a suite (paths-ignore): no forever park.

    The grace is measured on the DATABASE clock (``age_seconds``) and reached by
    the stale backstop (a delivery for ANOTHER repository).
    """
    gh = FakeGitHub()
    gh.set(Snapshot(SHA, "open", False, [], active_workflows=2))
    m = _manager(store, gh, stale=600)
    w = await _wait(store)
    assert 0 <= w["age_seconds"] < 60
    assert await m.evaluate(w) is None
    _sql(
        "UPDATE ci_waits SET created_at = now() - interval '700 seconds',"
        " last_evaluated_at = now() - interval '700 seconds' WHERE id = %s",
        w["wait_id"],
    )
    assert await m.on_delivery("push", _delivery(repo="nonagenticai/other")) == 1
    row = await _row(store, w)
    assert row["overall"] == "no_checks"
    assert recorder.posts[0][1]["overall"] == "no_checks"


# ── the REAL GitHub read (real PyGithub, GitHub's permission model enforced) ──


def _real_fetch_manager(store) -> CIWaitManager:
    """No fetcher override: the production ``fetch_snapshot`` reads the fake REST API."""
    return CIWaitManager(store, owner="pod-a", stale_after_seconds=600)


async def test_live_closed_pr_shape_resolves_pr_closed_and_calls_back(store, recorder):
    """Live 2026-10-11: ems PR #1 (CLOSED) parked with a contents-403, eval_count climbing."""
    fake = rest.FakeGitHubRest(
        number=1,
        state="closed",
        runs=[rest.check_run(1, "test", status="in_progress", conclusion=None)],
        suites=[rest.check_suite(1, status="in_progress", conclusion=None)],
    )
    w = await _wait(store, pr=1)
    with rest.patched(fake):
        assert await _real_fetch_manager(store).evaluate(w) == "pr_closed"
    row = await _row(store, w)
    assert row["overall"] == "pr_closed" and row["last_eval_error"] is None
    ((_, body),) = recorder.posts
    assert body["overall"] == "pr_closed"


async def test_permission_403_resolves_error_and_calls_back(store, recorder):
    fake = rest.FakeGitHubRest(fail=(403, {}, dict(rest.NOT_ACCESSIBLE)))
    w = await _wait(store)
    with rest.patched(fake):
        assert await _real_fetch_manager(store).evaluate(w) == "error"
    row = await _row(store, w)
    assert row["overall"] == "error" and "Resource not accessible by integration" in row["error"]
    ((_, body),) = recorder.posts
    assert body["overall"] == "error" and "403" in body["error"]


async def test_rate_limit_403_is_retried_not_resolved(store, recorder):
    body = {"message": "API rate limit exceeded for installation ID 1."}
    fake = rest.FakeGitHubRest(fail=(403, {"x-ratelimit-remaining": "0"}, body))
    w = await _wait(store)
    with rest.patched(fake):
        assert await _real_fetch_manager(store).evaluate(w) is None
    row = await _row(store, w)
    assert row["status"] == "waiting" and row["overall"] is None
    assert "rate limit" in row["last_eval_error"].lower() and row["next_eval_at"] is not None
    assert recorder.posts == []


# ── single resolution / single sender ─────────────────────────────────


async def test_concurrent_evaluations_resolve_and_call_back_once(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    managers = [_manager(CIWaitStore(), gh, f"pod-{n}") for n in "abcd"]
    results = await asyncio.gather(*(m.evaluate(w) for m in managers))
    assert sorted(r for r in results if r) == ["success"]
    assert len(recorder.posts) == 1


async def test_stuck_callback_is_resent_by_exactly_one_replica(store, recorder):
    w = await _wait(store)
    row = await store.resolve(
        wait_id=w["wait_id"],
        owner="pod-a",
        overall="success",
        checks=[],
        callback_lease_seconds=0,  # the sender died; its claim has lapsed
    )
    assert row is not None and row["callback_state"] == "sending"
    gh = FakeGitHub()
    a, b = _manager(CIWaitStore(), gh, "pod-b"), _manager(CIWaitStore(), gh, "pod-c")
    counts = await asyncio.gather(a.sweep_once(), b.sweep_once())
    assert sum(counts) == 1
    assert [body["wait_id"] for _, body in recorder.posts] == [w["wait_id"]]
    assert (await _row(store, w))["callback_state"] == "delivered"


async def test_a_live_send_claim_is_not_stolen(store, recorder):
    w = await _wait(store)
    await store.resolve(
        wait_id=w["wait_id"],
        owner="pod-a",
        overall="success",
        checks=[],
        callback_lease_seconds=120,
    )
    assert await _manager(store, FakeGitHub(), "pod-b").sweep_once() == 0
    assert recorder.posts == []
    assert not await store.record_callback(wait_id=w["wait_id"], owner="pod-b", outcome="delivered")
    assert await store.record_callback(wait_id=w["wait_id"], owner="pod-a", outcome="delivered")


# ── durable callback retries (precedent D1/D2) ────────────────────────

N8N_RUNNING = (409, '{"code":409,"message":"The execution \\"4711\\" is running already."}')
N8N_FINISHED = (409, '{"code":409,"message":"The execution \\"4711 has finished already."}')


def _make_due(wait_id: str) -> None:
    _sql(
        "UPDATE ci_waits SET callback_next_attempt_at = now() - interval '1 second' "
        "WHERE id = %s AND callback_state = 'retrying'",
        wait_id,
    )


async def test_resolved_before_the_cell_parks_is_retried_until_it_does(
    store, recorder, one_attempt_rounds
):
    """CI already green at registration: the callback beats the cell to its Wait
    node; n8n answers 409 "running already". It must land once the cell parks."""
    recorder.script = [N8N_RUNNING]
    gh = FakeGitHub()
    gh.set(green())
    m = _manager(store, gh)
    w = await _wait(store)
    await m.evaluate(w)
    row = await _row(store, w)
    assert row["callback_state"] == "retrying" and row["callback_owner"] is None
    assert await m.sweep_once() == 0  # not due: no hot loop
    _make_due(w["wait_id"])
    assert await m.sweep_once() == 1
    row = await _row(store, w)
    assert row["callback_state"] == "delivered" and row["callback_attempts"] == 2
    assert len(gh.calls) == 1  # the resend never re-reads GitHub


async def test_outage_is_retried_across_replicas_and_never_given_up(
    store, recorder, one_attempt_rounds
):
    down = httpx.ConnectError("connection refused")
    recorder.script = [down, (502, ""), down, (503, ""), down]
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    await _manager(store, gh, "pod-a").evaluate(w)
    for n in "bcde":
        assert (await _row(store, w))["callback_state"] == "retrying"
        _make_due(w["wait_id"])
        assert await _manager(CIWaitStore(), gh, f"pod-{n}").sweep_once() == 1
    _make_due(w["wait_id"])
    assert await _manager(store, gh, "pod-z").sweep_once() == 1
    row = await _row(store, w)
    assert row["callback_state"] == "delivered" and row["callback_attempts"] == 6


async def test_finished_execution_is_final_and_never_resent(store, recorder, one_attempt_rounds):
    recorder.script = [N8N_FINISHED]
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    await _manager(store, gh).evaluate(w)
    row = await _row(store, w)
    assert row["callback_state"] == "refused" and row["callback_next_attempt_at"] is None
    assert await _manager(store, gh).sweep_once() == 0


async def test_retry_backoff_is_scheduled_by_the_database_and_capped(
    store, recorder, one_attempt_rounds
):
    recorder.status = 503
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    _sql("UPDATE ci_waits SET callback_attempts = 3000 WHERE id = %s", w["wait_id"])
    await _manager(store, gh).evaluate(w)
    ((delay,),) = _sql(
        "SELECT extract(epoch FROM callback_next_attempt_at - now()) FROM ci_waits WHERE id = %s",
        w["wait_id"],
    )
    assert 250 < float(delay) <= cb.RETRY_CAP_S


# ── idempotency on (tenant, callback_url) ─────────────────────────────


async def _create(store, url, **kw):
    return await store.create_wait(
        wait_id=str(uuid.uuid7()),
        tenant_id=kw.get("tenant", TENANT),
        owner="nonagenticai",
        repo=kw.get("repo", "ems"),
        pr_number=kw.get("pr", 7),
        head_sha=SHA,
        required_checks=[],
        expected_checks=1,
        callback_url=url,
    )


async def test_replayed_registration_returns_the_open_wait(store):
    url = PREFIX + "4711"
    first, created = await _create(store, url)
    again, created2 = await _create(store, url)
    assert created and not created2 and again["wait_id"] == first["wait_id"]


async def test_replay_while_callback_retrying_returns_the_same_wait(store):
    url = PREFIX + "4711"
    first, _ = await _create(store, url)
    await store.resolve(
        wait_id=first["wait_id"], owner="p", overall="success", checks=[], callback_lease_seconds=60
    )
    await store.record_callback(
        wait_id=first["wait_id"], owner="p", outcome="retrying", retry_in_seconds=60
    )
    again, created = await _create(store, url)
    assert not created and again["wait_id"] == first["wait_id"]
    assert again["callback_state"] == "retrying"


async def test_replay_while_sending_returns_the_holder(store):
    url = PREFIX + "4711"
    first, _ = await _create(store, url)
    await store.resolve(
        wait_id=first["wait_id"], owner="p", overall="success", checks=[], callback_lease_seconds=60
    )
    again, created = await _create(store, url)
    assert not created and again["callback_state"] == "sending"


@pytest.mark.parametrize("final", ["delivered", "refused", "disallowed"])
async def test_key_frees_up_once_the_callback_is_final(store, final):
    url = PREFIX + "4711"
    first, _ = await _create(store, url)
    await store.resolve(
        wait_id=first["wait_id"], owner="p", overall="success", checks=[], callback_lease_seconds=60
    )
    await store.record_callback(wait_id=first["wait_id"], owner="p", outcome=final)
    second, created = await _create(store, url)
    assert created and second["wait_id"] != first["wait_id"]


async def test_same_url_other_tenant_is_independent(store):
    url = PREFIX + "4711"
    a, _ = await _create(store, url)
    b, created = await _create(store, url, tenant="other-tenant")
    assert created and a["wait_id"] != b["wait_id"]


async def test_concurrent_duplicate_registrations_create_one_wait(store):
    url = PREFIX + "4711"
    results = await asyncio.gather(*(_create(CIWaitStore(), url) for _ in range(8)))
    assert sum(created for _, created in results) == 1
    assert len({row["wait_id"] for row, _ in results}) == 1


async def test_database_itself_rejects_a_second_open_wait_per_callback_url(store):
    url = PREFIX + "4711"
    await _create(store, url)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _sql(
            "INSERT INTO ci_waits (id, tenant_id, owner, repo, repo_key, pr_number, head_sha, "
            "callback_url, status) VALUES (%s, %s, 'o', 'r', 'o/r', 1, 's', %s, 'waiting')",
            str(uuid.uuid4()),
            TENANT,
            url,
        )


async def test_status_read_is_tenant_scoped(store):
    w = await _wait(store)
    assert await store.get_wait(wait_id=w["wait_id"], tenant_id=TENANT) is not None
    assert await store.get_wait(wait_id=w["wait_id"], tenant_id="someone-else") is None


async def test_callback_body_matches_the_stored_row(store, recorder):
    gh = FakeGitHub()
    gh.set(green())
    w = await _wait(store)
    await _manager(store, gh).evaluate(w)
    row = await _row(store, w)
    assert recorder.posts[0][1] == callback_body(row)
