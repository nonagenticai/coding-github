"""``wait_for_pr_checks`` / ``get_ci_wait_status`` routes.

The head-SHA read (``_resolve_head_sha``) is faked; the store is REAL Postgres
where the test needs persistence (skipped without ``TEST_DATABASE_URL`` /
``DATABASE_URL``), and a stub where it only needs the refusal paths.
"""

from __future__ import annotations

import importlib
import os
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import psycopg
import pytest

from db import session as db_session
from services.ci_waits import callback as cb
from services.ci_waits import manager as mgr
from services.ci_waits.manager import CIWaitManager
from services.ci_waits.store import CIWaitStore
from services.github.github_service import GithubException

DSN = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
if DSN.startswith("sqlite"):
    DSN = ""
needs_pg = pytest.mark.skipif(not DSN, reason="needs TEST_DATABASE_URL / DATABASE_URL")

# Patched by dotted path, never via a module object captured at import: test_main's
# _fresh_main() drops every routers.* module, so main.app may serve a newer copy.
ROUTE = "routers.ci_wait_operations"
PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
SHA = "a" * 40
TENANT = "test-tenant"  # what coding_client.testing.override_tenant_auth resolves to
BODY = {"owner": "nonagenticai", "repo": "ems", "pr_number": 7, "callback_url": PREFIX + "4711"}


@pytest.fixture
def allow_prefix():
    with patch.object(
        cb, "get_settings", return_value=SimpleNamespace(ci_callback_prefixes=(PREFIX,))
    ):
        yield


@pytest.fixture
def head_sha():
    fake = AsyncMock(return_value=SHA)
    with patch(f"{ROUTE}._resolve_head_sha", new=fake):
        yield fake


class _NoEval(CIWaitManager):
    """Records scheduled evaluations instead of reading GitHub."""

    def __init__(self, store):
        super().__init__(store, owner="pod-test", fetcher=AsyncMock())
        self.scheduled: list[str] = []

    def schedule_evaluation(self, wait):
        self.scheduled.append(wait["wait_id"])


@pytest.fixture
async def pg_manager(monkeypatch):
    if not DSN:
        pytest.skip("needs TEST_DATABASE_URL / DATABASE_URL")
    from alembic.config import Config

    from alembic import command
    from config.settings import get_settings

    cfg = Config()
    cfg.set_main_option("script_location", "alembic")
    monkeypatch.setenv("DATABASE_URL", DSN)
    command.upgrade(cfg, "head")
    monkeypatch.setattr(get_settings(), "pgbouncer_url", DSN)
    with psycopg.connect(db_session.normalize_dsn(DSN), autocommit=True) as conn:
        conn.execute("TRUNCATE ci_waits")
    m = _NoEval(CIWaitStore())
    mgr.set_ci_wait_manager(m)
    yield m
    mgr.set_ci_wait_manager(None)
    await db_session.close_pool()


@pytest.fixture
def no_manager():
    mgr.set_ci_wait_manager(None)
    yield


# ── refusals before anything is read or stored ────────────────────────


async def test_callback_outside_prefix_is_422_before_reading_github(
    async_client, allow_prefix, head_sha, no_manager
):
    r = await async_client.post(
        "/api/repositories/ci-wait", json={**BODY, "callback_url": "http://evil:1/x"}
    )
    assert r.status_code == 422
    head_sha.assert_not_awaited()


async def test_no_prefixes_configured_refuses_every_callback(async_client, head_sha, no_manager):
    with patch.object(cb, "get_settings", return_value=SimpleNamespace(ci_callback_prefixes=())):
        r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    assert r.status_code == 422


async def test_no_wait_store_is_503(async_client, allow_prefix, head_sha, no_manager):
    r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    assert r.status_code == 503
    head_sha.assert_not_awaited()


async def test_status_without_store_is_503(async_client, no_manager):
    r = await async_client.get(f"/api/repositories/ci-wait/{uuid.uuid4()}")
    assert r.status_code == 503


async def test_tenant_refusal_from_the_ladder_reaches_the_caller(
    async_client, allow_prefix, monkeypatch
):
    """The SAME _github ladder + tenant gate as the PR routes decides access."""
    from services.tenant_binding import PlatformRepoForbiddenError, tenant_error_http_exception

    @asynccontextmanager
    async def refuse(owner, repo, tenant_id, permissions):
        raise tenant_error_http_exception(PlatformRepoForbiddenError("nonagenticai", "ems"))
        yield  # pragma: no cover

    stub = SimpleNamespace(store=SimpleNamespace(create_wait=AsyncMock()))
    mgr.set_ci_wait_manager(stub)  # type: ignore[arg-type]
    try:
        with patch(f"{ROUTE}._github", new=refuse):
            r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    finally:
        mgr.set_ci_wait_manager(None)
    assert r.status_code == 403
    stub.store.create_wait.assert_not_awaited()


async def test_missing_pr_is_404(async_client, allow_prefix):
    @asynccontextmanager
    async def ok(owner, repo, tenant_id, permissions):
        async def get_repository(name):
            def get_pull(n):
                raise GithubException(404, {"message": "Not Found"}, None)

            return SimpleNamespace(get_pull=get_pull)

        async def run(func, *a):
            return func(*a)

        yield SimpleNamespace(get_repository=get_repository, _execute_with_retry=run)

    stub = SimpleNamespace(store=SimpleNamespace(create_wait=AsyncMock()))
    mgr.set_ci_wait_manager(stub)  # type: ignore[arg-type]
    try:
        with patch(f"{ROUTE}._github", new=ok):
            r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    finally:
        mgr.set_ci_wait_manager(None)
    assert r.status_code == 404


async def test_store_failure_on_create_is_503(async_client, allow_prefix, head_sha):
    stub = SimpleNamespace(store=SimpleNamespace(create_wait=AsyncMock(side_effect=OSError("db"))))
    mgr.set_ci_wait_manager(stub)  # type: ignore[arg-type]
    try:
        r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    finally:
        mgr.set_ci_wait_manager(None)
    assert r.status_code == 503


# ── registration + status against Postgres ────────────────────────────


@needs_pg
async def test_accepts_202_pins_the_head_and_evaluates_at_once(
    async_client, allow_prefix, head_sha, pg_manager
):
    r = await async_client.post(
        "/api/repositories/ci-wait", json={**BODY, "required_checks": ["ci", "ci"]}
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["head_sha"] == SHA and body["status"] == "waiting"
    assert body["repo"] == "nonagenticai/ems" and body["pr_number"] == 7
    assert r.headers["Location"] == f"/api/repositories/ci-wait/{body['wait_id']}"
    assert pg_manager.scheduled == [body["wait_id"]]  # immediate evaluation
    req, tenant = head_sha.await_args.args
    assert (req.owner, req.repo, req.pr_number, tenant) == ("nonagenticai", "ems", 7, TENANT)

    s = await async_client.get(body["links"]["status"])
    assert s.status_code == 200
    st = s.json()
    assert st["status"] == "waiting" and st["overall"] is None
    assert st["required_checks"] == ["ci"] and st["callback_state"] == "none"


@needs_pg
async def test_replayed_post_returns_the_same_wait_and_does_not_reevaluate(
    async_client, allow_prefix, head_sha, pg_manager
):
    a = await async_client.post("/api/repositories/ci-wait", json=BODY)
    b = await async_client.post("/api/repositories/ci-wait", json=BODY)
    assert a.status_code == b.status_code == 202
    assert a.json()["wait_id"] == b.json()["wait_id"]
    assert pg_manager.scheduled == [a.json()["wait_id"]]


@needs_pg
async def test_same_callback_for_another_pr_is_409(
    async_client, allow_prefix, head_sha, pg_manager
):
    await async_client.post("/api/repositories/ci-wait", json=BODY)
    r = await async_client.post("/api/repositories/ci-wait", json={**BODY, "pr_number": 8})
    assert r.status_code == 409


@needs_pg
async def test_replay_while_callback_sending_is_503_with_retry_after(
    async_client, allow_prefix, head_sha, pg_manager
):
    a = (await async_client.post("/api/repositories/ci-wait", json=BODY)).json()
    await pg_manager.store.resolve(
        wait_id=a["wait_id"], owner="p", overall="success", checks=[], callback_lease_seconds=60
    )
    r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    assert r.status_code == 503 and r.headers["Retry-After"] == "2"


@needs_pg
async def test_replay_while_callback_retrying_returns_the_resolved_wait(
    async_client, allow_prefix, head_sha, pg_manager
):
    a = (await async_client.post("/api/repositories/ci-wait", json=BODY)).json()
    store = pg_manager.store
    await store.resolve(
        wait_id=a["wait_id"], owner="p", overall="failure", checks=[], callback_lease_seconds=60
    )
    await store.record_callback(
        wait_id=a["wait_id"], owner="p", outcome="retrying", retry_in_seconds=60
    )
    r = await async_client.post("/api/repositories/ci-wait", json=BODY)
    assert r.status_code == 202
    assert r.json()["wait_id"] == a["wait_id"] and r.json()["status"] == "resolved"


@needs_pg
async def test_status_of_another_tenants_wait_is_404(async_client, allow_prefix, pg_manager):
    row, _ = await pg_manager.store.create_wait(
        wait_id=str(uuid.uuid7()),
        tenant_id="someone-else",
        owner="nonagenticai",
        repo="ems",
        pr_number=7,
        head_sha=SHA,
        required_checks=[],
        expected_checks=1,
        callback_url=PREFIX + "x",
    )
    r = await async_client.get(f"/api/repositories/ci-wait/{row['wait_id']}")
    assert r.status_code == 404


@needs_pg
async def test_status_non_uuid_is_404(async_client, pg_manager):
    r = await async_client.get("/api/repositories/ci-wait/not-a-uuid")
    assert r.status_code == 404


def test_operation_ids_are_stable():
    from fastapi.routing import iter_route_contexts

    from main import app

    # fastapi>=0.137 no longer flattens include_router() routes into app.routes;
    # iter_route_contexts is the public accessor for the effective, prefixed routes.
    ops = {
        (m, r.path): r.operation_id
        for r in iter_route_contexts(app.routes)
        if getattr(r, "operation_id", None)
        for m in getattr(r, "methods", ())
    }
    assert ops[("POST", "/api/repositories/ci-wait")] == "wait_for_pr_checks"
    assert ops[("GET", "/api/repositories/ci-wait/{wait_id}")] == "get_ci_wait_status"


def test_route_reuses_the_pr_routes_credential_ladder():
    route = importlib.import_module(ROUTE)
    ladder = importlib.import_module("routers.github_issue_operations")
    assert route._github is ladder._github


@pytest.mark.parametrize("dsn", ["", "postgresql://u@h:6432/coding_github"])
async def test_lifespan_runs_the_manager_only_with_a_wait_store(monkeypatch, dsn):
    main = importlib.import_module("main")
    started: list[float] = []
    monkeypatch.setattr(CIWaitManager, "start", lambda self, interval: started.append(interval))
    fake = SimpleNamespace(
        database_url="",
        coding_containers_url="http://coding-containers:8691",
        coding_security_url="http://coding-security:8693",
        ci_wait_dsn=dsn,
        ci_wait_stale_after_seconds=600,
        ci_wait_sweep_interval_seconds=20,
    )
    monkeypatch.setattr(main, "settings", fake)
    mgr.set_ci_wait_manager(None)
    async with main.lifespan(main.app):
        assert (mgr.get_ci_wait_manager() is not None) is bool(dsn)
    assert started == ([20] if dsn else [])
    assert mgr.get_ci_wait_manager() is None
