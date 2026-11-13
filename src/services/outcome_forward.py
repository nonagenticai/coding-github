"""Forward verified GitHub webhook outcomes to n8n (push, not poll).

Replaces the n8n workflow "AA Observability - GitHub Outcome Poll" -- a
30-minute timer that scraped GitHub for PR-merged / PR-closed-without-merge /
post-merge CI results. George's ruling: no timer may write on a clock. Outcomes
are events GitHub already tells us about the moment they happen; this module
turns a VERIFIED GitHub delivery (HMAC already checked -- by
``SecureWebhookHandler._validate_request_security`` / ``verify_security`` --
before either public builder below is ever reached) into one of three
normalised outcome events and POSTs it to n8n's ``aa-github-outcome`` webhook,
best-effort.

⛔ MUST NEVER influence the HTTP response coding-github sends back to GitHub.
A forward failure, a forward timeout, or n8n being completely down must never
turn a successful, HMAC-verified delivery into anything but a 2xx for GitHub --
GitHub's own redelivery would otherwise retry a payload this service already
understood, for no gain and at n8n's expense, not GitHub's. Callers therefore
always ``asyncio.create_task(forward_outcome_event(...))`` rather than
``await`` it inline on the request path.

Configuration: ``CODING_GITHUB_OUTCOME_FORWARD_URL`` and
``CODING_GITHUB_OUTCOME_FORWARD_TOKEN``. EITHER empty (both default empty)
disables forwarding -- logged once per process, not once per request, so a
misconfiguration is visible without flooding logs on every delivery. n8n's
``aa-github-outcome`` webhook trigger itself validates
the ``x-aa-github-outcome-token: <token>`` header (constant-time, entry
shim of n8n_custom_nodes ``e.github-outcome``) against its own
``AA_GITHUB_OUTCOME_WEBHOOK_TOKEN``, so a URL configured without a token would
be forwarded and refused by n8n on every single delivery -- treated identically
to "not configured" here rather than attempted and silently dropped.

⛔ The token is NEVER logged, at any level, in this module.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

# 5s: this is a fire-and-forget notification, not a request GitHub is waiting
# on -- the HTTP response to GitHub has already been sent (or is about to be,
# independent of this call) by the time this runs; see the module docstring.
# Generous enough for a healthy n8n, short enough that a hung n8n cannot pin
# down the background-task pool.
_FORWARD_TIMEOUT_SECONDS = 5.0

OutcomeEventName = Literal["pr_merged", "pr_closed_unmerged", "ci_completed"]


class OutcomeEvent(TypedDict):
    source: str
    delivery_id: str | None
    event: OutcomeEventName
    repo: str | None
    pr_number: int | None
    head_sha: str | None
    merge_sha: str | None
    conclusion: str | None
    merged_at: str | None
    closed_at: str | None
    received_at: str


def build_pull_request_outcome(
    payload: dict[str, Any], delivery_id: str | None
) -> OutcomeEvent | None:
    """``pull_request`` webhook, ``action == "closed"`` -> pr_merged / pr_closed_unmerged.

    Callers gate on ``action == "closed"`` before calling this; it stays
    defensive and returns ``None`` for anything else so a future caller cannot
    silently misuse it for an action that was never closed.
    """
    if payload.get("action") != "closed":
        return None
    pr = payload.get("pull_request") or {}
    repo = payload.get("repository") or {}
    merged = bool(pr.get("merged"))
    return OutcomeEvent(
        source="github",
        delivery_id=delivery_id,
        event="pr_merged" if merged else "pr_closed_unmerged",
        repo=repo.get("full_name"),
        pr_number=pr.get("number"),
        head_sha=(pr.get("head") or {}).get("sha"),
        merge_sha=pr.get("merge_commit_sha") if merged else None,
        conclusion=None,
        merged_at=pr.get("merged_at"),
        closed_at=pr.get("closed_at"),
        received_at=datetime.now(UTC).isoformat(),
    )


def build_workflow_run_outcome(
    payload: dict[str, Any], delivery_id: str | None
) -> OutcomeEvent | None:
    """``workflow_run`` webhook, ``action == "completed"`` on the DEFAULT branch
    only -> ci_completed.

    ⚠️ Scoped to the default branch deliberately. ``workflow_run`` fires for
    every branch a workflow runs against (feature branches, PR heads under
    common trigger configs); the timer this replaces only ever asked about
    post-merge CI on the repo's trunk, so forwarding every branch's result
    would change what n8n receives, not merely how it arrives. The comparison
    costs no extra GitHub call: both ``workflow_run.head_branch`` and
    ``repository.default_branch`` are already in the webhook payload.
    """
    if payload.get("action") != "completed":
        return None
    run = payload.get("workflow_run") or {}
    repo = payload.get("repository") or {}
    default_branch = repo.get("default_branch")
    head_branch = run.get("head_branch")
    if not default_branch or head_branch != default_branch:
        return None
    return OutcomeEvent(
        source="github",
        delivery_id=delivery_id,
        event="ci_completed",
        repo=repo.get("full_name"),
        pr_number=None,
        head_sha=run.get("head_sha"),
        merge_sha=None,
        conclusion=run.get("conclusion"),
        merged_at=None,
        closed_at=None,
        received_at=datetime.now(UTC).isoformat(),
    )


_warned_disabled = False


def _forward_target() -> tuple[str, str] | None:
    """``(url, token)`` when forwarding is fully configured; ``None`` otherwise.

    Logs the reason exactly ONCE per process (not per request) so a
    misconfiguration is discoverable without flooding logs on every delivery.
    ⛔ Never logs the token itself.
    """
    global _warned_disabled
    settings = get_settings()
    url = (getattr(settings, "outcome_forward_url", "") or "").strip()
    token = (getattr(settings, "outcome_forward_token", "") or "").strip()
    if url and token:
        return url, token
    if not _warned_disabled:
        missing = []
        if not url:
            missing.append("CODING_GITHUB_OUTCOME_FORWARD_URL")
        if not token:
            missing.append("CODING_GITHUB_OUTCOME_FORWARD_TOKEN")
        logger.info(
            "GitHub outcome forwarding to n8n is disabled (%s not set); "
            "pr_merged/pr_closed_unmerged/ci_completed events will not be forwarded.",
            " and ".join(missing),
        )
        _warned_disabled = True
    return None


async def forward_outcome_event(event: OutcomeEvent) -> None:
    """POST ``event`` to n8n's ``aa-github-outcome`` webhook, best-effort.

    NEVER raises: every failure (disabled config, timeout, connection error,
    non-2xx) is caught and logged with ``delivery_id`` so a missed forward is
    diagnosable without correlating anything else. Exactly one attempt -- no
    retry loop -- because GitHub already redelivers a webhook this service
    fails to acknowledge, and retrying here would double that.
    """
    target = _forward_target()
    if target is None:
        return
    url, token = target
    try:
        async with httpx.AsyncClient(timeout=_FORWARD_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                json=event,
                headers={"x-aa-github-outcome-token": token},
            )
        if response.status_code >= 400:
            logger.warning(
                "Outcome forward to n8n failed: HTTP %s (delivery_id=%s event=%s)",
                response.status_code,
                event.get("delivery_id"),
                event.get("event"),
            )
        else:
            logger.info(
                "Forwarded %s to n8n (delivery_id=%s)",
                event.get("event"),
                event.get("delivery_id"),
            )
    except httpx.HTTPError as exc:
        logger.warning(
            "Outcome forward to n8n failed: %s (delivery_id=%s event=%s)",
            type(exc).__name__,
            event.get("delivery_id"),
            event.get("event"),
        )
