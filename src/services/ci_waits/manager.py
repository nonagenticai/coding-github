"""Drives CI-checks waits: evaluation triggers, resolution, callback delivery, sweeper.

One ``CIWaitManager`` per process (``owner`` = pod hostname + pid + nonce).

Invariant this module exists for: a registered wait ends in exactly one
completion callback, and NO path in this service can strand it silently — no
GitHub polling, no time budget (George 2026-10-11). A wait is evaluated:

1. at registration (CI may already be done);
2. on every verified GitHub delivery naming its repository (``check_suite``,
   ``check_run``, ``workflow_run``, ``status``, ``pull_request``) — whatever SHA
   the delivery names, because a delivery for a NEW head is how a superseded
   wait is found, and because the evaluation always reads the STORED SHA;
3. on ANY verified delivery (any repository) when it has not been evaluated for
   ``ci_wait_stale_after_seconds`` — the lost-webhook backstop;
4. at startup, every open wait (a delivery lost while no pod was up, or a
   registration whose first evaluation died with its pod);
5. after a FAILED evaluation (GitHub 5xx/network/rate limit), as a due retry
   from the database sweep (5 s doubling, capped at 300 s) until one succeeds.

The sweep is over the DATABASE only (due evaluation retries, due callback
retries, lapsed send claims); it never asks GitHub about a wait that last
evaluated cleanly.

Resolution is one conditional UPDATE, so concurrent evaluations (two deliveries,
two replicas) produce one winner, and only the winner sends. Delivery semantics
are coding-containers' exec-job callbacks, unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from typing import Any

from .callback import CallbackOutcome, retry_delay_seconds, send_callback
from .evaluator import FinalEvaluationError, Snapshot, decide, fetch_snapshot
from .store import CIWaitStore

logger = logging.getLogger(__name__)

#: Deliveries that may change a wait's answer. Everything verified is a backstop
#: trigger for STALE waits; these also re-evaluate the named repository's waits.
CI_WAIT_EVENTS = frozenset({"check_suite", "check_run", "workflow_run", "status", "pull_request"})

# Covers one send_callback round's worst case (3 x 10 s timeouts + 2 s + 4 s
# backoff = 36 s) with margin, so a live sender's claim never lapses under it.
CALLBACK_LEASE_SECONDS = 120
#: How long a claimed evaluation retry is held before another sweeper may take it.
EVAL_CLAIM_SECONDS = 120
EVAL_RETRY_BASE_S = 5
EVAL_RETRY_CAP_S = 300
#: Concurrent GitHub evaluations per process.
EVAL_CONCURRENCY = 4
#: Upper bound on :meth:`CIWaitManager.shutdown` (well under a pod's grace period).
SHUTDOWN_TIMEOUT_S = 10.0
#: How often shutdown re-delivers cancel to a task that swallowed it.
SHUTDOWN_RECANCEL_S = 0.1

Fetcher = Callable[[dict[str, Any]], Awaitable[Snapshot]]


def make_owner_id() -> str:
    """A process-unique owner: hostname (pod name) + pid + nonce."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def status_path(wait_id: str) -> str:
    return f"/api/repositories/ci-wait/{wait_id}"


def eval_retry_delay_seconds(failures: int) -> int:
    """Delay before re-evaluating after ``failures`` consecutive failed evaluations."""
    return min(EVAL_RETRY_CAP_S, EVAL_RETRY_BASE_S * 2 ** min(max(failures - 1, 0), 10))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return list(value or [])


def callback_body(row: dict[str, Any]) -> dict[str, Any]:
    """The POSTed completion document (same fields as the status route)."""
    return {
        "wait_id": row["wait_id"],
        "repo": f"{row['owner']}/{row['repo']}",
        "pr_number": row["pr_number"],
        "head_sha": row["head_sha"],
        "overall": row["overall"],
        "new_head_sha": row["new_head_sha"],
        "checks": [
            {
                "name": c.get("name"),
                "kind": c.get("kind"),
                "status": c.get("status"),
                "conclusion": c.get("conclusion"),
                "url": c.get("url"),
            }
            for c in _json_list(row["checks"])
        ],
        "error": row["error"],
        "completed_at": _iso(row["resolved_at"]),
        "links": {"status": status_path(row["wait_id"])},
    }


def _repo_key(payload: dict[str, Any]) -> str | None:
    repo = payload.get("repository")
    if isinstance(repo, dict) and isinstance(repo.get("full_name"), str):
        return repo["full_name"].lower()
    return None


class CIWaitManager:
    """Owns this process's evaluations, deliveries and the database sweep loop."""

    def __init__(
        self,
        store: CIWaitStore,
        *,
        owner: str,
        stale_after_seconds: int = 600,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.store = store
        self.owner = owner
        self.stale_after_seconds = stale_after_seconds
        self._fetch: Fetcher = fetcher or fetch_snapshot
        self._sem = asyncio.Semaphore(EVAL_CONCURRENCY)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._sweeper: asyncio.Task[None] | None = None
        self._closing = False

    def _spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any] | None:
        if self._closing:
            # Not lost: the next pod's startup evaluation covers every open wait.
            coro.close()
            return None
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ── evaluation ────────────────────────────────────────────────────

    async def evaluate(self, wait: dict[str, Any]) -> str | None:
        """Evaluate one wait; resolve + deliver if settled. Returns ``overall`` or None.

        Never raises: a failed evaluation is recorded and retried by the sweeper.
        """
        wait_id = wait["wait_id"]
        async with self._sem:
            try:
                snapshot = await self._fetch(wait)
            except FinalEvaluationError as exc:
                logger.warning("CI wait %s: cannot be evaluated: %s", wait_id, exc)
                return await self._resolve(wait, "error", [], error=str(exc)[:2000])
            except Exception as exc:
                failures = int(wait.get("eval_failures") or 0) + 1
                retry_in = eval_retry_delay_seconds(failures)
                logger.warning(
                    "CI wait %s: evaluation failed (%s: %s); retry in %ss",
                    wait_id,
                    type(exc).__name__,
                    exc,
                    retry_in,
                )
                with contextlib.suppress(Exception):
                    await self.store.record_evaluation(
                        wait_id=wait_id,
                        error=f"{type(exc).__name__}: {exc}"[:2000],
                        retry_in_seconds=retry_in,
                    )
                return None
        decision = decide(snapshot, wait, grace_seconds=self.stale_after_seconds)
        if decision is None:
            try:
                await self.store.record_evaluation(wait_id=wait_id, checks=snapshot.checks)
            except Exception:
                logger.exception("CI wait %s: could not record a pending evaluation", wait_id)
            return None
        return await self._resolve(
            wait, decision.overall, decision.checks, new_head_sha=decision.new_head_sha
        )

    async def _resolve(
        self,
        wait: dict[str, Any],
        overall: str,
        checks: list[dict[str, Any]],
        *,
        new_head_sha: str | None = None,
        error: str | None = None,
    ) -> str | None:
        wait_id = wait["wait_id"]
        try:
            row = await self.store.resolve(
                wait_id=wait_id,
                owner=self.owner,
                overall=overall,
                checks=checks,
                new_head_sha=new_head_sha,
                error=error,
                callback_lease_seconds=CALLBACK_LEASE_SECONDS,
            )
        except Exception:
            # Still 'waiting': the next trigger (or the startup sweep) evaluates again.
            logger.exception("CI wait %s: could not persist resolution %s", wait_id, overall)
            return None
        if row is None:
            logger.info("CI wait %s: already resolved elsewhere; no callback from here", wait_id)
            return None
        logger.info(
            "CI wait %s resolved: %s/%s#%s %s overall=%s",
            wait_id,
            row["owner"],
            row["repo"],
            row["pr_number"],
            row["head_sha"][:12],
            overall,
        )
        await self.deliver(row)
        return overall

    async def evaluate_many(self, waits: list[dict[str, Any]]) -> None:
        if waits:
            await asyncio.gather(*(self.evaluate(w) for w in waits))

    def schedule_evaluation(self, wait: dict[str, Any]) -> None:
        """Evaluate in the background (registration: CI may already be done)."""
        self._spawn(self.evaluate(wait), name=f"ci-wait-eval-{wait['wait_id']}")

    # ── triggers ──────────────────────────────────────────────────────

    async def on_delivery(self, event: str, payload: bytes | dict[str, Any]) -> int:
        """Re-evaluate what a VERIFIED delivery may have changed. Returns waits evaluated.

        A relevant event re-evaluates every open wait on its repository plus every
        stale one; any other verified event only the stale ones.
        """
        try:
            data = json.loads(payload) if isinstance(payload, (bytes, str)) else payload
        except ValueError:
            data = {}
        repo_key = _repo_key(data) if event in CI_WAIT_EVENTS and isinstance(data, dict) else None
        try:
            waits = await self.store.list_waiting(
                repo_key=repo_key, stale_after_seconds=self.stale_after_seconds
            )
        except Exception:
            logger.exception("CI waits: could not list open waits for a %s delivery", event)
            return 0
        if waits:
            logger.info(
                "CI waits: %s delivery for %s re-evaluates %d open wait(s)",
                event,
                repo_key or "-",
                len(waits),
            )
        await self.evaluate_many(waits)
        return len(waits)

    def schedule_delivery(self, event: str, payload: bytes) -> None:
        """Fire-and-forget :meth:`on_delivery` (GitHub's 2xx never waits on it)."""
        self._spawn(self.on_delivery(event, payload), name=f"ci-wait-delivery-{event}")

    async def startup_evaluate(self) -> int:
        """Evaluate EVERY open wait once (covers deliveries lost while no pod ran)."""
        total = 0
        try:
            waits = await self.store.list_waiting()
        except Exception:
            logger.exception("CI waits: startup evaluation could not list open waits")
            return 0
        total = len(waits)
        if total:
            logger.info("CI waits: startup evaluation of %d open wait(s)", total)
        await self.evaluate_many(waits)
        return total

    # ── callbacks ─────────────────────────────────────────────────────

    async def deliver(self, row: dict[str, Any]) -> CallbackOutcome | None:
        """Send one round for a row this process holds the send claim on."""
        if row.get("callback_state") != "sending" or not row.get("callback_url"):
            return None
        outcome, attempts = await send_callback(
            row["callback_url"], callback_body(row), wait_id=row["wait_id"]
        )
        retry_in = None
        if outcome is CallbackOutcome.RETRY:
            retry_in = retry_delay_seconds(int(row.get("callback_attempts") or 0) + attempts)
        try:
            recorded = await self.store.record_callback(
                wait_id=row["wait_id"],
                owner=row["callback_owner"],
                outcome=outcome.value,
                attempts=attempts,
                retry_in_seconds=retry_in,
            )
            if not recorded:
                logger.warning("CI wait %s: callback claim moved before recording", row["wait_id"])
            elif retry_in is not None:
                logger.warning(
                    "CI wait %s: callback not accepted yet; next round in %ss",
                    row["wait_id"],
                    retry_in,
                )
        except Exception:
            # Unrecorded: the row stays 'sending' and is re-claimed once its
            # callback lease lapses, so the callback is still not lost.
            logger.exception("CI wait %s: could not record callback outcome", row["wait_id"])
        return outcome

    # ── database sweep ────────────────────────────────────────────────

    async def sweep_once(self) -> int:
        """(Re)send due callbacks and run due evaluation retries. Returns rows handled."""
        due_callbacks = await self.store.claim_due_callbacks(
            owner=self.owner, callback_lease_seconds=CALLBACK_LEASE_SECONDS
        )
        due_evals = await self.store.claim_due_evaluations(claim_seconds=EVAL_CLAIM_SECONDS)
        await asyncio.gather(
            *(self.deliver(r) for r in due_callbacks), *(self.evaluate(w) for w in due_evals)
        )
        return len(due_callbacks) + len(due_evals)

    async def _sweep_loop(self, interval: float) -> None:
        with contextlib.suppress(Exception):
            await self.startup_evaluate()
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CI wait sweep failed")
            if self._closing:
                # A cancel that something below swallowed (see shutdown) must
                # not be followed by a full ``interval`` of sleep.
                return
            await asyncio.sleep(interval)

    def start(self, interval: float) -> None:
        """Start the loop: startup evaluation of every open wait, then the DB sweep."""
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop(interval), name="ci-wait-sweeper")

    async def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Stop, within ``timeout`` seconds. Nothing is owned: open waits are
        re-evaluated by the next pod's startup sweep, and a ``sending`` claim
        lapses and is re-claimed.

        One ``cancel()`` is NOT enough: psycopg_pool's ``getconn`` catches
        ``CancelledError`` raised during its connection check (it puts the
        connection back and retries), so a cancel that lands there is swallowed
        and the sweeper would go on into its ``interval`` sleep — an hour in
        production, which hung pod shutdown (and CI) until killed. So cancel is
        re-delivered until every task has ended, and the whole stop is bounded.
        """
        self._closing = True
        tasks = [t for t in (self._sweeper, *self._tasks) if t is not None]
        self._sweeper = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        pending = {t for t in tasks if not t.done()}
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "CI waits: %d task(s) did not stop within %.0fs; abandoning them",
                    len(pending),
                    timeout,
                )
                break
            for task in pending:
                task.cancel()
            _done, pending = await asyncio.wait(
                pending, timeout=min(SHUTDOWN_RECANCEL_S, remaining)
            )
        for task in tasks:
            if task.done() and not task.cancelled():
                # Retrieve the outcome so an error is logged once, not as
                # "exception was never retrieved" at garbage collection.
                exc = task.exception()
                if exc is not None:
                    logger.warning("CI waits: task %s ended with %r", task.get_name(), exc)


_manager: CIWaitManager | None = None


def get_ci_wait_manager() -> CIWaitManager | None:
    """The process manager, or None when no wait store is configured."""
    return _manager


def set_ci_wait_manager(manager: CIWaitManager | None) -> None:
    global _manager
    _manager = manager
