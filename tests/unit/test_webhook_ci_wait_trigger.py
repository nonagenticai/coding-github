"""Verified GitHub deliveries trigger CI-wait evaluations — and change nothing else.

* With a wait manager configured, every HMAC-verified delivery is handed to it
  (``schedule_delivery``); check/status deliveries outcome_only used to ignore
  are acknowledged as ``ci_wait_trigger`` and wake NO legacy handler.
* An unsigned/badly signed delivery never reaches the manager (401 first).
* PR-closed and workflow_run-completed are forwarded exactly as before.
* Without a manager (no wait store), behaviour is byte-for-byte the old one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.ci_waits import manager as mgr

os.environ.setdefault("ENABLE_AUTH", "false")

SECRET = "ci-wait-trigger-test-secret"  # pragma: allowlist secret
REPO = {"full_name": "nonagenticai/ems", "default_branch": "main"}


def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


class FakeManager:
    def __init__(self):
        self.deliveries: list[tuple[str, bytes]] = []

    def schedule_delivery(self, event, payload):
        self.deliveries.append((event, payload))


@pytest.fixture
def settings(monkeypatch):
    from config.settings import get_settings

    s = get_settings()
    monkeypatch.delenv("CODING_GITHUB_WEBHOOK_MODE", raising=False)
    monkeypatch.setattr(s, "github_webhook_secret", SECRET)
    monkeypatch.setattr(s, "webhook_mode", "outcome_only")
    return s


@pytest.fixture
def handlers(settings):
    import services.github.secure_webhook_handler as swh

    original = swh._webhook_handler
    swh._webhook_handler = None
    pr = AsyncMock(return_value=True)
    push = AsyncMock(return_value=True)
    with (
        patch.object(swh, "handle_pull_request_event", pr),
        patch.object(swh, "handle_push_event", push),
        patch("services.outcome_forward.forward_outcome_event", AsyncMock()) as fwd,
    ):
        yield {"pr": pr, "push": push, "fwd": fwd}
    swh._webhook_handler = original


@pytest.fixture
def manager():
    m = FakeManager()
    mgr.set_ci_wait_manager(m)  # type: ignore[arg-type]
    yield m
    mgr.set_ci_wait_manager(None)


@pytest.fixture
def no_manager():
    mgr.set_ci_wait_manager(None)
    yield


@pytest.fixture
def client(handlers):
    from routers.github_webhooks import router

    app = FastAPI()
    app.include_router(router, prefix="/github")
    return TestClient(app)


def _post(client, event, body: dict, *, signed: bool = True, bad: bool = False):
    raw = json.dumps(body).encode()
    headers = {"X-GitHub-Event": event, "Content-Type": "application/json"}
    if signed:
        headers["X-Hub-Signature-256"] = "sha256=" + "0" * 64 if bad else _sig(raw)
    return client.post("/github/webhook", content=raw, headers=headers), raw


CHECK_SUITE = {"action": "completed", "check_suite": {"head_sha": "a" * 40}, "repository": REPO}
CHECK_RUN = {"action": "completed", "check_run": {"head_sha": "a" * 40}, "repository": REPO}
STATUS = {"sha": "a" * 40, "state": "success", "repository": REPO}
PR_CLOSED = {
    "action": "closed",
    "pull_request": {
        "number": 7,
        "merged": True,
        "merged_at": "2026-10-01T00:00:00Z",
        "html_url": "https://github.com/nonagenticai/ems/pull/7",
        "head": {"sha": "abc", "ref": "feat"},
        "base": {"ref": "main"},
    },
    "repository": REPO,
}
WF_RUN_COMPLETED = {
    "action": "completed",
    "workflow_run": {"head_branch": "main", "head_sha": "sha1", "conclusion": "success"},
    "repository": REPO,
}


@pytest.mark.parametrize(
    ("event", "body"), [("check_suite", CHECK_SUITE), ("check_run", CHECK_RUN), ("status", STATUS)]
)
def test_check_deliveries_trigger_the_manager_and_wake_nothing_else(
    client, handlers, manager, event, body
):
    r, raw = _post(client, event, body)
    assert r.status_code == 202
    assert r.json() == {"status": "accepted", "reason": "ci_wait_trigger"}
    assert manager.deliveries == [(event, raw)]
    handlers["pr"].assert_not_awaited()
    handlers["push"].assert_not_awaited()
    handlers["fwd"].assert_not_called()


@pytest.mark.parametrize("bad", [False, True])
def test_unverified_delivery_never_reaches_the_manager(client, manager, bad):
    r, _ = _post(client, "check_suite", CHECK_SUITE, signed=bad, bad=bad)
    assert r.status_code == 401
    assert manager.deliveries == []


def test_pr_closed_is_forwarded_exactly_as_before_and_also_triggers(client, handlers, manager):
    r, raw = _post(client, "pull_request", PR_CLOSED)
    assert r.status_code == 200 and r.json()["processed"] is True
    assert handlers["fwd"].call_count == 1
    assert manager.deliveries == [("pull_request", raw)]
    handlers["pr"].assert_not_awaited()


def test_pr_synchronize_triggers_but_is_not_forwarded(client, handlers, manager):
    r, raw = _post(client, "pull_request", {**PR_CLOSED, "action": "synchronize"})
    assert r.status_code == 202 and r.json()["status"] == "ignored"
    assert manager.deliveries == [("pull_request", raw)]
    handlers["fwd"].assert_not_called()


def test_workflow_run_completed_is_forwarded_exactly_as_before_and_triggers(
    client, handlers, manager
):
    r, raw = _post(client, "workflow_run", WF_RUN_COMPLETED)
    assert r.status_code == 200 and r.json()["processed"] is True
    assert handlers["fwd"].call_count == 1
    assert manager.deliveries == [("workflow_run", raw)]


def test_unrelated_verified_event_is_still_ignored_but_feeds_the_stale_backstop(
    client, handlers, manager
):
    r, raw = _post(client, "push", {"ref": "refs/heads/main", "repository": REPO, "commits": []})
    assert r.status_code == 202 and r.json()["status"] == "ignored"
    assert manager.deliveries == [("push", raw)]
    handlers["push"].assert_not_awaited()


@pytest.mark.parametrize(
    ("event", "body"), [("check_suite", CHECK_SUITE), ("check_run", CHECK_RUN), ("status", STATUS)]
)
def test_without_a_wait_store_check_deliveries_are_ignored_as_before(
    client, handlers, no_manager, event, body
):
    r, _ = _post(client, event, body)
    assert r.status_code == 202
    assert r.json() == {"status": "ignored", "reason": "webhook_mode=outcome_only"}


def test_without_a_wait_store_pr_closed_forwarding_is_unchanged(client, handlers, no_manager):
    r, _ = _post(client, "pull_request", PR_CLOSED)
    assert r.status_code == 200 and handlers["fwd"].call_count == 1


def test_all_mode_also_triggers_after_verification(client, handlers, manager, settings):
    settings.webhook_mode = "all"
    r, raw = _post(client, "workflow_run", WF_RUN_COMPLETED)
    assert r.status_code == 200
    assert manager.deliveries == [("workflow_run", raw)]
    r, _ = _post(client, "workflow_run", WF_RUN_COMPLETED, signed=False)
    assert r.status_code == 401 and len(manager.deliveries) == 1
