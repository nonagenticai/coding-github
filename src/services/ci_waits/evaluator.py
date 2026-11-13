"""Evaluate one CI-checks wait: read GitHub once, decide whether it has settled.

Split in two so the decision is a pure function of what GitHub said:

* :func:`fetch_snapshot` — the only GitHub I/O. It reads the PR (head SHA, state),
  then — only while the head is still the stored SHA — every check SUITE, every
  check run (latest per (suite, name), every page) and every commit status (latest
  per context) on that SHA, and, only when there are no check runs or statuses at
  all, whether the repository has any ACTIVE workflow. Authorisation is the SAME
  ladder and tenant gate the PR routes use (``routers.github_issue_operations._github``
  with ``PERMS_PR_CHECKS`` + actions read), under the tenant that registered the wait.
  ⛔ It never GETs the commit itself (``GET /commits/{sha}`` needs ``contents: read``,
  which ``PERMS_CI_WAIT`` deliberately does not hold — a repo-scoped token for it
  answers 403 "Resource not accessible by integration"): the check-run, check-suite
  and combined-status endpoints are addressed by the SHA directly. See
  :data:`PERMS_CI_WAIT` for the endpoint → permission table.
* :func:`decide` — the rules below, documented here because callers branch on them.

Check SUITES are read because a workflow that cannot start (a broken ``ci.yml``)
completes its suite ``startup_failure`` with ZERO check runs — invisible to a
runs-and-statuses-only reader, which then waits forever.

A suite is RELEVANT unless it is a *phantom*: not completed, zero check runs, and
not from ``github-actions``. Any app with ``checks: write`` gets a suite auto-created
on every push that stays ``queued`` with no runs if the app never reports (measured
2026-10-11: ``code-graph-helper-app`` does this on every gitops-production and
coding-github commit). Waiting on it would park every wait forever; an app that
has created no run has nothing to wait for.

``grace`` below is ``grace_seconds`` (the manager passes its stale-backstop window,
default 600 s) measured against the wait's age from the DATABASE clock
(``age_seconds``); the stale backstop re-evaluates a wait at least that often
while any delivery arrives, so a rule gated on it fires.

RULES (in order):

1. **superseded** — the PR head is no longer the stored SHA (a new push). Resolves
   at once with ``new_head_sha``; the old SHA's checks are not waited for.
2. **not settled** — a relevant suite is not completed, or a check run / status is
   not complete ⇒ keep waiting (``pr_closed`` instead if the PR was closed without
   merging). A ``github-actions`` suite still incomplete with ZERO runs past
   ``grace`` stops blocking and counts as a failing item (conclusion ``stale``):
   a queued Actions job always has a run, so such a suite will not report, and
   the wait must neither park forever nor settle green over it.
3. **suite failures** — a completed suite whose conclusion is failing (failure,
   startup_failure, timed_out, cancelled, action_required, stale) with ZERO runs is
   added as a failing ``check_suite`` item: it fails the wait.
4. **no checks at all** (no runs, statuses or failing suites):
   ``no_checks`` if the repository has NO active workflow; or if relevant suites
   exist and all completed (nothing more will report); else ``pr_closed`` if closed
   unmerged; else — workflows exist but GitHub created no suite for the SHA
   (``paths-ignore``, ``[skip ci]``) — ``no_checks`` once past ``grace``, keep
   waiting before that.
5. **missing checks** — everything visible is complete but a ``required_checks``
   name never appeared or fewer than ``expected_checks`` (default 1) exist:
   resolves ``failure`` with a ``kind: missing`` item per gap once nothing more can
   arrive — at once when relevant suites exist (all completed), else (statuses
   only) past ``grace``; ``pr_closed`` / keep waiting before that. Never parks
   forever, merged PR included.
6. **all-gating** — EVERY check run, status and failing suite gates the wait (this
   service cannot see which checks a base branch requires). Any failing item ⇒
   ``failure``; else at least one ``success`` ⇒ ``success``; else (all
   skipped/neutral) ⇒ ``skipped`` — an all-skipped head has proven nothing.

A merged PR keeps waiting for its head's checks like an open one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from github.GithubException import RateLimitExceededException
from github.Requester import Requester

from services.github.github_service import commit_handle
from services.tenant_binding import PERMS_PR_CHECKS

#: Check-run conclusions / status states that fail the wait.
FAILING_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure", "stale"}
)
FAILING_STATES = frozenset({"failure", "error"})
#: Settled but neither a pass nor a fail.
NEUTRAL_CONCLUSIONS = frozenset({"skipped", "neutral"})

#: The GitHub App whose suites are workflow runs; its suites are never phantoms.
ACTIONS_APP = "github-actions"
#: Default for ``decide(grace_seconds=...)`` (the manager's stale-backstop window).
DEFAULT_GRACE_SECONDS = 600

#: The PR-checks read set plus ``actions: read`` for the active-workflow count.
#: Every GitHub call :func:`fetch_snapshot` makes, and the grant it needs:
#:
#: * ``GET /repos/{o}/{r}``                              — metadata (always granted)
#: * ``GET /repos/{o}/{r}/pulls/{n}``                    — pull_requests: read
#: * ``GET /repos/{o}/{r}/commits/{sha}/check-runs``     — checks: read
#: * ``GET /repos/{o}/{r}/commits/{sha}/check-suites``   — checks: read
#: * ``GET /repos/{o}/{r}/commits/{sha}/status``         — statuses: read
#: * ``GET /repos/{o}/{r}/actions/workflows``            — actions: read
#:
#: ⛔ NOT ``GET /repos/{o}/{r}/commits/{sha}`` (contents: read). Do not add
#: ``contents`` here to make a ``get_commit`` work: that grants the wait's token
#: the tenant's whole source tree to read four check endpoints (live 2026-10-11).
PERMS_CI_WAIT: dict[str, str] = {**PERMS_PR_CHECKS, "actions": "read"}


class FinalEvaluationError(Exception):
    """GitHub answered definitively that this wait can never be evaluated.

    (PR or repository gone, or the tenant may no longer read it.) The wait
    resolves ``error`` rather than parking the cell forever.
    """


@dataclass
class Snapshot:
    """What GitHub said about one wait's PR, at one moment."""

    head_sha: str
    state: str  # "open" | "closed"
    merged: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    #: Check suites on the SHA: ``{id, app, status, conclusion, runs}``.
    suites: list[dict[str, Any]] = field(default_factory=list)
    #: Active workflows in the repo; only read when ``checks`` is empty (else None).
    active_workflows: int | None = None


@dataclass
class Decision:
    overall: str
    checks: list[dict[str, Any]]
    new_head_sha: str | None = None


#: Items carrying a check-run style status/conclusion (statuses carry a state).
_RUN_KINDS = frozenset({"check_run", "check_suite"})


def _check_complete(item: dict[str, Any]) -> bool:
    if item["kind"] in _RUN_KINDS:
        return item["status"] == "completed"
    return item["status"] != "pending"


def _check_failing(item: dict[str, Any]) -> bool:
    if item["kind"] in _RUN_KINDS:
        return item["conclusion"] in FAILING_CONCLUSIONS
    return item["status"] in FAILING_STATES


def _check_succeeded(item: dict[str, Any]) -> bool:
    if item["kind"] in _RUN_KINDS:
        return item["conclusion"] == "success"
    return item["status"] == "success"


def _past_grace(wait: dict[str, Any], grace_seconds: float) -> bool:
    age = wait.get("age_seconds")
    return age is not None and float(age) >= grace_seconds


def _suite_item(suite: dict[str, Any], conclusion: str) -> dict[str, Any]:
    return {
        "kind": "check_suite",
        "name": f"{suite.get('app') or 'unknown-app'} suite {suite.get('id')}",
        "status": "completed",
        "conclusion": conclusion,
        "url": None,
    }


def _missing(checks: list[dict[str, Any]], wait: dict[str, Any]) -> list[dict[str, Any]]:
    names = {c["name"] for c in checks}
    gaps = [n for n in (wait.get("required_checks") or []) if n not in names]
    missing = [
        {"kind": "missing", "name": n, "status": "missing", "conclusion": "missing", "url": None}
        for n in gaps
    ]
    expected = int(wait.get("expected_checks") or 1)
    if len(checks) + len(gaps) < expected:
        missing.append(
            {
                "kind": "missing",
                "name": f"expected {expected} checks, {len(checks)} reported",
                "status": "missing",
                "conclusion": "missing",
                "url": None,
            }
        )
    return missing


def decide(
    snapshot: Snapshot, wait: dict[str, Any], *, grace_seconds: float = DEFAULT_GRACE_SECONDS
) -> Decision | None:
    """Apply the module RULES; ``None`` means "not settled, keep waiting"."""
    if snapshot.head_sha != wait["head_sha"]:
        return Decision("superseded", [], new_head_sha=snapshot.head_sha)
    closed_unmerged = snapshot.state == "closed" and not snapshot.merged
    past_grace = _past_grace(wait, grace_seconds)

    relevant: list[dict[str, Any]] = []
    blocking = False
    extra: list[dict[str, Any]] = []
    for suite in snapshot.suites:
        runs = int(suite.get("runs") or 0)
        completed = suite.get("status") == "completed"
        if not completed and runs == 0:
            if suite.get("app") != ACTIONS_APP:
                continue  # phantom: an app that never reports (module docstring)
            relevant.append(suite)
            if past_grace:
                extra.append(_suite_item(suite, "stale"))
            else:
                blocking = True
            continue
        relevant.append(suite)
        if not completed:
            blocking = True
        elif runs == 0 and suite.get("conclusion") in FAILING_CONCLUSIONS:
            extra.append(_suite_item(suite, suite["conclusion"]))

    checks = snapshot.checks + extra
    if blocking or not all(_check_complete(c) for c in checks):
        return Decision("pr_closed", checks) if closed_unmerged else None
    if not checks:
        if snapshot.active_workflows == 0 or relevant:
            return Decision("no_checks", [])
        if closed_unmerged:
            return Decision("pr_closed", [])
        return Decision("no_checks", []) if past_grace else None
    missing = _missing(checks, wait)
    if missing:
        if relevant or past_grace:
            return Decision("failure", checks + missing)
        return Decision("pr_closed", checks) if closed_unmerged else None
    if any(_check_failing(c) for c in checks):
        return Decision("failure", checks)
    if any(_check_succeeded(c) for c in checks):
        return Decision("success", checks)
    return Decision("skipped", checks)


def _latest_check_runs(runs: Any) -> list[dict[str, Any]]:
    """Every page; the LATEST run per (check suite, name) — a re-run supersedes.

    Keyed by suite as well as name: two workflows may each have a job called
    ``test``, and the higher id of one must never hide the other's failure.
    """
    latest: dict[tuple[Any, str], Any] = {}
    for r in runs:
        key = (getattr(r, "check_suite_id", None), r.name)
        prev = latest.get(key)
        if prev is None or (getattr(r, "id", 0) or 0) >= (getattr(prev, "id", 0) or 0):
            latest[key] = r
    return [
        {
            "kind": "check_run",
            "name": r.name,
            "status": r.status,
            "conclusion": r.conclusion,
            "url": getattr(r, "html_url", None),
        }
        for r in latest.values()
    ]


def _suites(suites: Any) -> list[dict[str, Any]]:
    """Every page of the SHA's check suites, reduced to what :func:`decide` reads."""
    out: list[dict[str, Any]] = []
    for s in suites:
        app = getattr(s, "app", None)
        out.append(
            {
                "id": getattr(s, "id", None),
                "app": getattr(app, "slug", None) if app is not None else None,
                "status": s.status,
                "conclusion": s.conclusion,
                "runs": int(getattr(s, "latest_check_runs_count", 0) or 0),
            }
        )
    return out


def _latest_statuses(combined: Any) -> list[dict[str, Any]]:
    """Combined status already carries the latest status per context."""
    return [
        {
            "kind": "status",
            "name": s.context,
            "status": s.state,
            "conclusion": s.state if s.state != "pending" else None,
            "url": getattr(s, "target_url", None),
        }
        for s in combined.statuses
    ]


def _github_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _github_message(exc: BaseException) -> str:
    data = getattr(exc, "data", None)
    if isinstance(data, dict) and isinstance(data.get("message"), str):
        return data["message"]
    message = getattr(exc, "message", None)
    return message if isinstance(message, str) else ""


def _rate_limited(exc: BaseException) -> bool:
    """A GitHub refusal that says "later", not "no" (primary or secondary limit)."""
    if isinstance(exc, RateLimitExceededException):
        return True
    raw = getattr(exc, "headers", None)
    headers = {str(k).lower(): v for k, v in raw.items()} if isinstance(raw, dict) else {}
    if headers.get("retry-after") is not None:
        return True
    if str(headers.get("x-ratelimit-remaining", "")).strip() == "0":
        return True
    message = _github_message(exc).lower()
    return Requester.isRateLimitError(message) or "rate limit" in message


#: GitHub answers that are definitive for this wait's credential + PR:
#: 403 a permission refusal ("Resource not accessible by integration", SSO,
#: blocked repo) — never a rate limit, see :func:`_rate_limited`; 404 gone; 422
#: unprocessable (e.g. "No commit found for SHA" after a force-push + GC).
#: ⚠️ 401 is deliberately NOT here: it means the token just used is bad, and the
#: one bad token we can hold is a cached installation token revoked before its
#: expiry — the next mint heals it, so it is retried (a broken App identity fails
#: the MINT instead, which the ladder reports, not GitHub per call).
_FINAL_GITHUB_STATUSES = frozenset({403, 404, 422})


def _github_final(exc: BaseException) -> bool:
    status = _github_status(exc)
    if status not in _FINAL_GITHUB_STATUSES:
        return False
    return not (status == 403 and _rate_limited(exc))


#: Shared with ``/pr-checks``: a Commit handle with no ``GET /commits/{sha}``.
_commit_handle = commit_handle


async def fetch_snapshot(wait: dict[str, Any]) -> Snapshot:
    """Read GitHub for ``wait``. Raises :class:`FinalEvaluationError` on a definitive no.

    Final: the tenant gate's refusals, and GitHub 403 (permission, never a rate
    limit), 404 and 422. Any other exception is transient (network, 5xx, 401,
    429, a rate-limit 403): the caller records it and schedules a retry.
    """
    # Lazy: the credential ladder lives with the PR routes; importing it at module
    # load would make this service module depend on the router package at import.
    from fastapi import HTTPException

    from routers.github_issue_operations import _github

    owner, repo, number = wait["owner"], wait["repo"], int(wait["pr_number"])
    try:
        async with _github(owner, repo, wait["tenant_id"], PERMS_CI_WAIT) as service:
            gh_repo = await service.get_repository(f"{owner}/{repo}")
            pr = await service._execute_with_retry(gh_repo.get_pull, number)
            snap = Snapshot(head_sha=pr.head.sha, state=pr.state, merged=bool(pr.merged))
            if snap.head_sha != wait["head_sha"]:
                return snap
            commit = _commit_handle(gh_repo, wait["head_sha"])
            runs = await service._execute_with_retry(commit.get_check_runs)
            combined = await service._execute_with_retry(commit.get_combined_status)
            suites = await service._execute_with_retry(commit.get_check_suites)
            snap.suites = await asyncio.to_thread(lambda: _suites(suites))
            snap.checks = await asyncio.to_thread(
                lambda: _latest_check_runs(runs) + _latest_statuses(combined)
            )
            if not snap.checks:
                workflows = await service._execute_with_retry(gh_repo.get_workflows)
                snap.active_workflows = await asyncio.to_thread(
                    lambda: sum(1 for w in workflows if getattr(w, "state", None) == "active")
                )
            return snap
    except HTTPException as exc:
        # The tenant gate / credential ladder refused (403 platform_repo_forbidden,
        # 409 github_app_not_installed, 412 ...): definitive for this tenant. A 400
        # "no credential configured" or a 5xx from the ladder is NOT — retried.
        if exc.status_code in (401, 403, 404, 409, 412, 422):
            raise FinalEvaluationError(f"HTTP {exc.status_code}: {exc.detail}") from exc
        raise
    except Exception as exc:
        if _github_status(exc) == 404:
            raise FinalEvaluationError(
                f"GitHub 404: {owner}/{repo}#{number} no longer exists for this credential"
            ) from exc
        if _github_final(exc):
            # A permission 403 / a 422 does not heal by retrying: parking the cell
            # forever behind it (eval_count climbing) is the failure, not the safety.
            raise FinalEvaluationError(
                f"GitHub {_github_status(exc)}: {_github_message(exc) or exc} "
                f"({owner}/{repo}#{number})"
            ) from exc
        raise
