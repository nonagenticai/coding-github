"""CODING_GITHUB_WEBHOOK_MODE (coding-github#131).

Default ``outcome_only``: after HMAC verification only pull_request-closed and
workflow_run-completed are forwarded; everything else is a 202 ``ignored`` that
wakes no handler. ``all`` keeps the legacy dispatch. Includes in-test mutants
that re-run this file against a patched copy of the source.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("ENABLE_AUTH", "false")

SECRET = "outcome-only-test-secret"  # pragma: allowlist secret
REPO_ROOT = Path(__file__).resolve().parents[2]

REPO = {"full_name": "org/repo", "default_branch": "main"}


def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def mode_env(monkeypatch):
    """Cached settings object with the secret set; webhook_mode left at its CODE
    default unless a test overrides it."""
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.delenv("CODING_GITHUB_WEBHOOK_MODE", raising=False)
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    return settings


@pytest.fixture
def handlers(mode_env):
    """Fresh SecureWebhookHandler whose registered PR/push handlers are mocks."""
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
def client(handlers):
    from routers.github_webhooks import router

    app = FastAPI()
    app.include_router(router, prefix="/github")
    return TestClient(app)


def _post(client, event, body: dict, *, signed: bool):
    raw = json.dumps(body).encode()
    headers = {"X-GitHub-Event": event, "Content-Type": "application/json"}
    if signed:
        headers["X-Hub-Signature-256"] = _sig(raw)
    return client.post("/github/webhook", content=raw, headers=headers)


PR_CLOSED = {
    "action": "closed",
    "pull_request": {
        "number": 7,
        "merged": True,
        "merged_at": "2026-10-01T00:00:00Z",
        "html_url": "https://github.com/org/repo/pull/7",
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
PR_OPENED = {**PR_CLOSED, "action": "opened"}
IGNORED_SIGNED = [
    ("pull_request", {**PR_CLOSED, "action": "synchronize"}),
    ("pull_request", PR_OPENED),
    ("push", {"ref": "refs/heads/main", "repository": REPO, "commits": []}),
    ("workflow_job", {"action": "completed", "repository": REPO}),
    ("check_run", {"action": "completed", "repository": REPO}),
    ("check_suite", {"action": "completed", "repository": REPO}),
    ("issue_comment", {"action": "created", "repository": REPO}),
    ("ping", {"zen": "x"}),
]


class TestResolveWebhookMode:
    @pytest.mark.parametrize("raw", [None, "", "ALL ", "ALL", " all", "bogus", "outcome_only"])
    def test_everything_but_exact_all_is_outcome_only(self, raw):
        from config.settings import resolve_webhook_mode

        assert resolve_webhook_mode(raw) == "outcome_only"

    def test_all_selects_all(self):
        from config.settings import resolve_webhook_mode

        assert resolve_webhook_mode("all") == "all"


class TestHmacBeforeModeFilter:
    @pytest.mark.parametrize("event", ["push", "workflow_job", "check_run", "ping"])
    def test_unsigned_ignored_event_is_401(self, client, event):
        assert _post(client, event, {"repository": REPO}, signed=False).status_code == 401

    def test_unsigned_pr_closed_is_401(self, client, handlers):
        assert _post(client, "pull_request", PR_CLOSED, signed=False).status_code == 401
        handlers["fwd"].assert_not_called()

    def test_empty_secret_is_401_even_for_ignored_event(self, client, mode_env, monkeypatch):
        import services.github.secure_webhook_handler as swh

        monkeypatch.setattr(mode_env, "github_webhook_secret", "")
        swh._webhook_handler = None
        assert _post(client, "push", {"repository": REPO}, signed=True).status_code == 401


class TestOutcomeOnlyIgnores:
    @pytest.mark.parametrize(("event", "body"), IGNORED_SIGNED)
    def test_signed_non_outcome_event_is_202_ignored_and_wakes_nothing(
        self, client, handlers, event, body
    ):
        r = _post(client, event, body, signed=True)
        assert r.status_code == 202
        assert r.json() == {"status": "ignored", "reason": "webhook_mode=outcome_only"}
        handlers["pr"].assert_not_called()
        handlers["push"].assert_not_called()
        handlers["fwd"].assert_not_called()


class TestOutcomeOnlyForwards:
    def test_pr_closed_merged_forwards_once(self, client, handlers):
        r = _post(client, "pull_request", PR_CLOSED, signed=True)
        assert r.status_code == 200
        data = r.json()
        assert (data["event_type"], data["action"], data["processed"]) == (
            "pull_request",
            "closed",
            True,
        )
        handlers["fwd"].assert_called_once()
        handlers["pr"].assert_not_called()

    def test_workflow_run_completed_forwards_once(self, client, handlers):
        r = _post(client, "workflow_run", WF_RUN_COMPLETED, signed=True)
        assert r.status_code == 200
        data = r.json()
        assert (data["event_type"], data["action"], data["processed"]) == (
            "workflow_run",
            "completed",
            True,
        )
        handlers["fwd"].assert_called_once()


class TestAllMode:
    def test_all_mode_pr_opened_reaches_validation_handler(
        self, client, handlers, mode_env, monkeypatch
    ):
        monkeypatch.setattr(mode_env, "webhook_mode", "all")
        r = _post(client, "pull_request", PR_OPENED, signed=True)
        assert r.status_code == 200
        handlers["pr"].assert_called_once()


# --------------------------------------------------------------------------
# Mutants: re-run this file against a patched COPY of the source tree.
# --------------------------------------------------------------------------

MUTANTS = {
    "default_flipped_to_all": (
        "src/config/settings.py",
        '    webhook_mode: str = "outcome_only"',
        '    webhook_mode: str = "all"',
    ),
    "mode_filter_before_hmac": (
        "src/routers/github_webhooks.py",
        "        if outcome_only or ci_wait is not None:\n",
        '        if outcome_only and x_github_event not in ("pull_request", "workflow_run"):\n'
        "            return _ignored_outcome_only(x_github_event, x_github_delivery)\n"
        "        if outcome_only or ci_wait is not None:\n",
    ),
}


def _run_mutant(tmp_path: Path, rel: str, old: str, new: str) -> subprocess.CompletedProcess:
    work = tmp_path / "copy"
    shutil.copytree(
        REPO_ROOT,
        work,
        ignore=shutil.ignore_patterns(
            ".git", ".worktrees", ".venv", "__pycache__", ".pytest_cache", "logs", "*.pyc"
        ),
    )
    target = work / rel
    text = target.read_text()
    assert text.count(old) == 1, f"mutation anchor not found exactly once in {rel}"
    target.write_text(text.replace(old, new))
    env = {**os.environ, "MUTANT_RUN": "1"}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "tests/unit/test_webhook_outcome_only.py",
            "-k",
            "not Mutant",
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.skipif(os.environ.get("MUTANT_RUN") == "1", reason="no recursion")
class TestMutantsAreKilled:
    @pytest.mark.parametrize("name", list(MUTANTS))
    def test_mutant_turns_suite_red(self, tmp_path, name):
        rel, old, new = MUTANTS[name]
        proc = _run_mutant(tmp_path, rel, old, new)
        assert proc.returncode != 0, f"mutant {name} SURVIVED:\n{proc.stdout[-2000:]}"
        assert "failed" in proc.stdout, proc.stdout[-2000:]
