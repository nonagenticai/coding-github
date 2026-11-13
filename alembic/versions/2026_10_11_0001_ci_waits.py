"""ci_waits: durable "PR head checks finished" waits with a completion callback.

Revision ID: 0001
Revises:
Create Date: 2026-10-11

One row per ``wait_for_pr_checks`` registration. A wait is ``waiting`` until an
evaluation (at registration, on a verified GitHub delivery for its repository,
at startup, or a due retry after a failed evaluation) finds its stored head SHA
settled; then ONE conditional UPDATE moves it to ``resolved`` and hands the
callback to the winner (``callback_state = 'sending'`` under ``callback_owner``).

Callback columns and semantics are copied from coding-containers ``exec_jobs``
(0001 + 0002): ``retrying`` with ``callback_next_attempt_at`` (database clock),
re-claimed by any replica's sweeper, retried for as long as it stays retryable.

Idempotency: at most one wait per ``(tenant_id, callback_url)`` HOLDING it — a
wait holds its key while ``waiting`` AND while its callback is still
``sending``/``retrying`` (partial unique index ``ci_waits_active_callback_uidx``).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must equal services.ci_waits.store._HOLDS_KEY (ON CONFLICT infers the index from it).
HOLDS_KEY = "status = 'waiting' OR callback_state IN ('sending', 'retrying')"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ci_waits (
            id                        uuid PRIMARY KEY,
            tenant_id                 text NOT NULL,
            owner                     text NOT NULL,
            repo                      text NOT NULL,
            repo_key                  text NOT NULL,
            pr_number                 integer NOT NULL CHECK (pr_number > 0),
            head_sha                  text NOT NULL,
            required_checks           jsonb NOT NULL DEFAULT '[]'::jsonb,
            expected_checks           integer NOT NULL DEFAULT 1 CHECK (expected_checks >= 1),
            callback_url              text NOT NULL,
            status                    text NOT NULL CHECK (status IN ('waiting', 'resolved')),
            overall                   text CHECK (overall IS NULL OR overall IN
                ('success', 'failure', 'skipped', 'superseded', 'pr_closed', 'no_checks',
                 'error')),
            new_head_sha              text,
            checks                    jsonb NOT NULL DEFAULT '[]'::jsonb,
            error                     text,
            eval_count                integer NOT NULL DEFAULT 0,
            eval_failures             integer NOT NULL DEFAULT 0,
            last_evaluated_at         timestamptz,
            last_eval_error           text,
            next_eval_at              timestamptz,
            callback_state            text NOT NULL DEFAULT 'none'
                CHECK (callback_state IN
                       ('none', 'sending', 'retrying', 'delivered', 'refused', 'disallowed')),
            callback_owner            text,
            callback_lease_expires_at timestamptz,
            callback_attempts         integer NOT NULL DEFAULT 0,
            callback_next_attempt_at  timestamptz,
            created_at                timestamptz NOT NULL DEFAULT now(),
            resolved_at               timestamptz,
            updated_at                timestamptz NOT NULL DEFAULT now(),
            CHECK ((status = 'waiting') = (overall IS NULL))
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ci_waits_waiting_repo_idx "
        "ON ci_waits (repo_key) WHERE status = 'waiting'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ci_waits_eval_retry_idx "
        "ON ci_waits (next_eval_at) WHERE status = 'waiting' AND next_eval_at IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ci_waits_callback_sending_idx "
        "ON ci_waits (callback_lease_expires_at) WHERE callback_state = 'sending'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ci_waits_callback_retry_idx "
        "ON ci_waits (callback_next_attempt_at) WHERE callback_state = 'retrying'"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ci_waits_active_callback_uidx "
        f"ON ci_waits (tenant_id, callback_url) WHERE {HOLDS_KEY}"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ci_waits")
