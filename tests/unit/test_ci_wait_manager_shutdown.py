"""CIWaitManager.shutdown is bounded even when a cancel is swallowed below it.

psycopg_pool 3.3.1 ``AsyncConnectionPool.getconn`` catches ``CancelledError``
raised during its connection check (``CLIENT_EXCEPTIONS`` includes it), puts the
connection back and retries. A cancel that lands there is therefore swallowed,
the sweeper finishes its round and goes into ``asyncio.sleep(interval)`` — and a
single ``cancel()`` + ``await`` waits the whole interval (an hour in production).
That hung CI (test_start_runs_the_startup_evaluation, 300 s timeout) and would
hang pod shutdown until SIGKILL. The real-pool reproduction is in
test_ci_waits_pg.py; these tests pin the manager's contract without a database.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from services.ci_waits.manager import CIWaitManager


class SwallowingStore:
    """A store whose sweep claim swallows the first N cancels, like getconn's check."""

    def __init__(self, swallow: int) -> None:
        self.swallow = swallow
        self.in_claim = asyncio.Event()
        self.claims = 0

    async def list_waiting(self, **_: Any) -> list[dict[str, Any]]:
        return []

    async def claim_due_callbacks(self, **_: Any) -> list[dict[str, Any]]:
        self.claims += 1
        self.in_claim.set()
        while True:
            try:
                await asyncio.sleep(3600)  # an in-flight connection check
            except asyncio.CancelledError:
                if self.swallow <= 0:
                    raise
                self.swallow -= 1
                return []  # swallowed: carry on as if nothing happened

    async def claim_due_evaluations(self, **_: Any) -> list[dict[str, Any]]:
        return []


async def test_a_swallowed_cancel_does_not_leave_shutdown_waiting_out_the_interval():
    store = SwallowingStore(swallow=1)
    m = CIWaitManager(store, owner="pod-a")  # type: ignore[arg-type]
    m.start(interval=3600)
    await asyncio.wait_for(store.in_claim.wait(), 2)
    sweeper = m._sweeper
    assert sweeper is not None

    # Before the fix: blocked for the 3600 s sleep. Not wait_for: cancelling the
    # old shutdown() made it swallow its OWN cancel and return "normally".
    assert await finishes_within(m.shutdown(), 2)

    assert sweeper.done()
    assert store.swallow == 0  # the cancel really was swallowed once
    assert store.claims == 1  # and no second sweep round started after it


async def test_shutdown_gives_up_on_a_task_that_never_stops(caplog):
    store = SwallowingStore(swallow=10**9)  # swallows every cancel, forever
    m = CIWaitManager(store, owner="pod-a")  # type: ignore[arg-type]
    m._sweep_loop = _stubborn_loop(store)  # type: ignore[method-assign]
    m.start(interval=3600)
    await asyncio.wait_for(store.in_claim.wait(), 2)
    sweeper = m._sweeper
    assert sweeper is not None

    loop = asyncio.get_running_loop()
    try:
        t0 = loop.time()
        with caplog.at_level(logging.WARNING, logger="services.ci_waits.manager"):
            assert await finishes_within(m.shutdown(timeout=0.5), 3)
        elapsed = loop.time() - t0

        assert 0.4 <= elapsed < 2.0  # bounded by ``timeout``, not by the task
        assert not sweeper.done()
        assert "did not stop within" in caplog.text
    finally:
        store.swallow = 0  # let the test's loop tear down cleanly whatever happened
        sweeper.cancel()
        await asyncio.wait({sweeper}, timeout=2)


async def finishes_within(coro: Any, seconds: float) -> bool:
    """True if ``coro`` completes within ``seconds``; never cancels it to find out
    (a cancel can make a hung coroutine return and read as a pass)."""
    task = asyncio.ensure_future(coro)
    done, _ = await asyncio.wait({task}, timeout=seconds)
    if not done:
        task.cancel()
        await asyncio.wait({task}, timeout=2)
        return False
    task.result()
    return True


def _stubborn_loop(store: SwallowingStore):
    async def loop(_interval: float) -> None:
        while True:
            await store.claim_due_callbacks()

    return loop
