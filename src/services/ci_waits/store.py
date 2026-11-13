"""PostgreSQL persistence for CI-checks waits (table ``ci_waits``).

Shape copied from coding-containers ``services/exec_jobs/store.py``. Every state
change is ONE conditional statement, so concurrent evaluators / replicas cannot
both win a transition:

* ``resolve`` matches only ``status = 'waiting'`` — under READ COMMITTED the loser
  re-evaluates the WHERE after the winner commits and matches nothing; the same
  statement moves ``callback_state`` to ``sending`` under ``callback_owner`` = the
  winner, so exactly one evaluator POSTs the callback;
* ``FOR UPDATE SKIP LOCKED`` keeps two sweepers from blocking on each other;
* a retryable callback is parked as ``retrying`` with ``callback_next_attempt_at``
  and re-claimed (back to ``sending``, one winner) by any replica's sweeper once
  due — durable across pod restarts, retried for as long as it is retryable;
* at most one wait HOLDING each ``(tenant_id, callback_url)`` (partial unique
  index ``ci_waits_active_callback_uidx``): a wait holds its key while
  ``waiting`` AND while its callback is still ``sending``/``retrying``. ``resolve``
  moves ``callback_state`` to ``sending`` in the SAME statement, so the key is
  never free between "resolved" and "callback settled".

All times are the database's ``now()`` — never a replica's clock.
"""

from __future__ import annotations

from typing import Any, LiteralString

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.session import get_connection

#: Callback states in which a resolved wait still holds its ``(tenant_id, callback_url)``.
CALLBACK_PENDING = ("sending", "retrying")

# The idempotency-key predicate. MUST equal the WHERE of the partial unique index
# ci_waits_active_callback_uidx (migration 0001) — ON CONFLICT infers the index
# from it — and is also the read-back filter.
_HOLDS_KEY = """
    status = 'waiting' OR callback_state IN ('sending', 'retrying')
"""

_COLUMNS = """
    id::text AS wait_id, tenant_id, owner, repo, repo_key, pr_number, head_sha,
    required_checks, expected_checks, callback_url, status, overall, new_head_sha,
    checks, error, eval_count, eval_failures, last_evaluated_at, last_eval_error,
    next_eval_at, callback_state, callback_owner, callback_lease_expires_at,
    callback_attempts, callback_next_attempt_at, created_at, resolved_at, updated_at,
    extract(epoch FROM now() - created_at)::float8 AS age_seconds
"""


#: ``_COLUMNS`` for the ``UPDATE ci_waits w ... FROM ...`` statements.
_W_COLUMNS = _COLUMNS.replace("id::text", "w.id::text")


class CIWaitStore:
    """Raw-psycopg store over ``ci_waits`` (one short transaction per call)."""

    async def _one(self, sql: LiteralString, params: dict[str, Any]) -> dict[str, Any] | None:
        async with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    async def _all(self, sql: LiteralString, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    async def create_wait(
        self,
        *,
        wait_id: str,
        tenant_id: str,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        required_checks: list[str],
        expected_checks: int,
        callback_url: str,
    ) -> tuple[dict[str, Any], bool]:
        """Insert a ``waiting`` row; ``(row, created)``.

        Idempotent on ``(tenant_id, callback_url)`` while a wait holds it (see
        ``_HOLDS_KEY``): a client retrying the POST gets the holder back with
        ``created = False``. The INSERT and the read-back are separate statements,
        so the holder can release the key in between; the read-back filters on the
        same predicate, so it then finds nothing and the INSERT is retried.
        """
        params = {
            "id": wait_id,
            "tenant_id": tenant_id,
            "owner": owner,
            "repo": repo,
            "repo_key": f"{owner}/{repo}".lower(),
            "pr_number": pr_number,
            "head_sha": head_sha,
            "required": Jsonb(list(required_checks)),
            "expected": expected_checks,
            "callback_url": callback_url,
        }
        for _ in range(5):
            row = await self._one(
                f"""
                INSERT INTO ci_waits (id, tenant_id, owner, repo, repo_key, pr_number,
                                      head_sha, required_checks, expected_checks,
                                      callback_url, status)
                VALUES (%(id)s, %(tenant_id)s, %(owner)s, %(repo)s, %(repo_key)s,
                        %(pr_number)s, %(head_sha)s, %(required)s, %(expected)s,
                        %(callback_url)s, 'waiting')
                ON CONFLICT (tenant_id, callback_url) WHERE {_HOLDS_KEY} DO NOTHING
                RETURNING {_COLUMNS}
                """,
                params,
            )
            if row is not None:
                return row, True
            existing = await self._one(
                f"""
                SELECT {_COLUMNS} FROM ci_waits
                 WHERE tenant_id = %(tenant_id)s AND callback_url = %(callback_url)s
                   AND ({_HOLDS_KEY})
                """,
                params,
            )
            if existing is not None:
                return existing, False
        raise RuntimeError("ci wait insert kept conflicting with a releasing holder")

    async def get_wait(self, *, wait_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Tenant-scoped read: a foreign wait is indistinguishable from none."""
        return await self._one(
            f"SELECT {_COLUMNS} FROM ci_waits WHERE id = %(id)s AND tenant_id = %(tenant_id)s",
            {"id": wait_id, "tenant_id": tenant_id},
        )

    async def list_waiting(
        self,
        *,
        repo_key: str | None = None,
        stale_after_seconds: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Open waits: all, one repository's, or those not evaluated recently.

        ``repo_key`` and ``stale_after_seconds`` combine with OR: a delivery for a
        repository re-evaluates every open wait on it plus every stale one.
        """
        return await self._all(
            f"""
            SELECT {_COLUMNS} FROM ci_waits
             WHERE status = 'waiting'
               AND (
                    (%(repo_key)s::text IS NULL AND %(stale)s::integer IS NULL)
                 OR repo_key = %(repo_key)s::text
                 OR (%(stale)s::integer IS NOT NULL
                     AND coalesce(last_evaluated_at, created_at)
                         < now() - (%(stale)s::integer * interval '1 second'))
               )
             ORDER BY created_at
             LIMIT %(limit)s
            """,
            {"repo_key": repo_key, "stale": stale_after_seconds, "limit": limit},
        )

    async def record_evaluation(
        self,
        *,
        wait_id: str,
        checks: list[dict[str, Any]] | None = None,
        error: str | None = None,
        retry_in_seconds: int | None = None,
    ) -> bool:
        """Persist a non-resolving evaluation of a still-waiting wait.

        ``error`` (with ``retry_in_seconds``) records a FAILED evaluation and
        schedules a retry from the database's ``now()``; without it the evaluation
        succeeded and is pending, so any scheduled retry is cleared.
        """
        if error is not None and retry_in_seconds is None:
            raise ValueError("a failed evaluation needs retry_in_seconds")
        row = await self._one(
            """
            UPDATE ci_waits
               SET eval_count = eval_count + 1,
                   last_evaluated_at = now(),
                   last_eval_error = %(error)s,
                   eval_failures = CASE WHEN %(error)s::text IS NULL THEN 0
                                        ELSE eval_failures + 1 END,
                   next_eval_at = CASE WHEN %(error)s::text IS NULL THEN NULL
                       ELSE now() + (%(retry_in)s * interval '1 second') END,
                   checks = coalesce(%(checks)s, checks),
                   updated_at = now()
             WHERE id = %(id)s AND status = 'waiting'
            RETURNING id
            """,
            {
                "id": wait_id,
                "error": error,
                "retry_in": retry_in_seconds or 0,
                "checks": Jsonb(checks) if checks is not None else None,
            },
        )
        return row is not None

    async def resolve(
        self,
        *,
        wait_id: str,
        owner: str,
        overall: str,
        checks: list[dict[str, Any]],
        callback_lease_seconds: int,
        new_head_sha: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """``waiting`` -> ``resolved`` + callback hand-off. Returns the row only to the winner."""
        return await self._one(
            f"""
            UPDATE ci_waits
               SET status = 'resolved', overall = %(overall)s, checks = %(checks)s,
                   new_head_sha = %(new_head_sha)s, error = %(error)s,
                   resolved_at = now(), updated_at = now(), next_eval_at = NULL,
                   eval_count = eval_count + 1, last_evaluated_at = now(),
                   callback_state = 'sending', callback_owner = %(owner)s,
                   callback_lease_expires_at =
                       now() + (%(callback_lease)s * interval '1 second')
             WHERE id = %(id)s AND status = 'waiting'
            RETURNING {_COLUMNS}
            """,
            {
                "id": wait_id,
                "owner": owner,
                "overall": overall,
                "checks": Jsonb(checks),
                "new_head_sha": new_head_sha,
                "error": error,
                "callback_lease": callback_lease_seconds,
            },
        )

    async def claim_due_evaluations(
        self, *, claim_seconds: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Claim waits whose failed evaluation is due for a retry (one winner per row).

        The claim pushes ``next_eval_at`` forward by ``claim_seconds``: a claimant
        that dies mid-evaluation leaves the wait due again once that lapses.
        """
        return await self._all(
            f"""
            WITH due AS (
                SELECT id FROM ci_waits
                 WHERE status = 'waiting' AND next_eval_at IS NOT NULL
                   AND next_eval_at <= now()
                 ORDER BY next_eval_at
                 LIMIT %(limit)s
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE ci_waits w
               SET next_eval_at = now() + (%(claim)s * interval '1 second'),
                   updated_at = now()
              FROM due d
             WHERE w.id = d.id AND w.status = 'waiting' AND w.next_eval_at <= now()
            RETURNING {_W_COLUMNS}
            """,
            {"claim": claim_seconds, "limit": limit},
        )

    async def claim_due_callbacks(
        self, *, owner: str, callback_lease_seconds: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Claim callbacks to (re)send: one winner per row across replicas.

        * ``sending`` with a lapsed lease — the sender died mid-send;
        * ``retrying`` whose ``callback_next_attempt_at`` is due — the last round
          ended retryable (n8n not parked yet, n8n/tailnet down, 5xx).
        """
        return await self._all(
            f"""
            WITH due AS (
                SELECT id FROM ci_waits
                 WHERE (callback_state = 'sending' AND callback_lease_expires_at < now())
                    OR (callback_state = 'retrying' AND callback_next_attempt_at <= now())
                 ORDER BY coalesce(callback_lease_expires_at, callback_next_attempt_at)
                 LIMIT %(limit)s
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE ci_waits w
               SET callback_state = 'sending', callback_owner = %(owner)s,
                   callback_next_attempt_at = NULL, updated_at = now(),
                   callback_lease_expires_at =
                       now() + (%(callback_lease)s * interval '1 second')
              FROM due d
             WHERE w.id = d.id
               AND ((w.callback_state = 'sending' AND w.callback_lease_expires_at < now())
                    OR (w.callback_state = 'retrying' AND w.callback_next_attempt_at <= now()))
            RETURNING {_W_COLUMNS}
            """,
            {"owner": owner, "callback_lease": callback_lease_seconds, "limit": limit},
        )

    async def record_callback(
        self,
        *,
        wait_id: str,
        owner: str,
        outcome: str,
        attempts: int = 1,
        retry_in_seconds: int | None = None,
    ) -> bool:
        """Persist a round's outcome; only the current callback owner may.

        ``outcome == 'retrying'`` releases the claim and schedules the next round
        ``retry_in_seconds`` from the database's ``now()``.
        """
        if outcome == "retrying" and retry_in_seconds is None:
            raise ValueError("a retrying outcome needs retry_in_seconds")
        row = await self._one(
            """
            UPDATE ci_waits
               SET callback_state = %(outcome)s, callback_lease_expires_at = NULL,
                   callback_owner = CASE WHEN %(outcome)s = 'retrying' THEN NULL
                                         ELSE callback_owner END,
                   callback_next_attempt_at = CASE WHEN %(outcome)s = 'retrying'
                       THEN now() + (%(retry_in)s * interval '1 second') ELSE NULL END,
                   callback_attempts = callback_attempts + %(attempts)s, updated_at = now()
             WHERE id = %(id)s AND callback_state = 'sending' AND callback_owner = %(owner)s
            RETURNING id
            """,
            {
                "id": wait_id,
                "owner": owner,
                "outcome": outcome,
                "attempts": attempts,
                "retry_in": retry_in_seconds or 0,
            },
        )
        return row is not None
