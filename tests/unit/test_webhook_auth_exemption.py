"""The GitHub webhook route under the REAL auth middleware (ENABLE_AUTH on).

Every other webhook test builds a bare FastAPI app with no middleware, which is
exactly how #129/#133 shipped a route nobody could reach: in production
coding-client's ``AuthMiddleware`` 401'd every delivery -- signed ones included --
with ``{"detail": "Authentication required"}`` before the route's HMAC check ran.

These tests install the production middleware via ``main.install_auth_middleware``
and pin both halves: a signed delivery reaches the handler, and the exemption is
exactly ``POST`` + the exact path -- no lookalike reaches a handler without HMAC.
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

SECRET = "auth-exemption-test-secret"  # pragma: allowlist secret
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO = {"full_name": "org/repo", "default_branch": "main"}
AUTH_REQUIRED = {"detail": "Authentication required"}
HMAC_REFUSALS = {"Missing webhook signature", "Invalid webhook signature"}

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
WORKFLOW_JOB = {"action": "completed", "repository": REPO}


def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def handlers(monkeypatch):
    import services.github.secure_webhook_handler as swh
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.delenv("CODING_GITHUB_WEBHOOK_MODE", raising=False)
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    # Production behaviour: no dev/test API-key path, no trusted-network bypass.
    monkeypatch.setenv("ENVIRONMENT", "production")

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
    from main import GITHUB_WEBHOOK_PATH, install_auth_middleware
    from routers.github import router  # the SAME aggregate router main.py mounts

    assert GITHUB_WEBHOOK_PATH == "/api/repositories/webhooks/webhook"
    app = FastAPI()
    app.include_router(router, prefix="/api/repositories")

    @app.post("/api/repositories/protected")
    @app.get("/api/repositories/protected")
    async def protected():
        return {"ok": True}

    install_auth_middleware(app)
    return TestClient(app, base_url="http://testserver")


def _post(client, url: str, event: str, body: dict, *, signed: bool):
    raw = json.dumps(body).encode()
    headers = {"X-GitHub-Event": event, "Content-Type": "application/json"}
    if signed:
        headers["X-Hub-Signature-256"] = _sig(raw)
    return client.post(url, content=raw, headers=headers)


WEBHOOK_URL = "http://testserver/api/repositories/webhooks/webhook"


class TestSignedDeliveryReachesTheHandler:
    def test_signed_ignored_event_is_202(self, client, handlers):
        resp = _post(client, WEBHOOK_URL, "workflow_job", WORKFLOW_JOB, signed=True)
        assert resp.status_code == 202, resp.text
        assert resp.json()["status"] == "ignored"
        handlers["pr"].assert_not_awaited()
        handlers["fwd"].assert_not_awaited()

    def test_signed_pr_closed_reaches_the_forwarder(self, client, handlers):
        resp = _post(client, WEBHOOK_URL, "pull_request", PR_CLOSED, signed=True)
        assert resp.status_code == 200, resp.text
        handlers["fwd"].assert_awaited_once()


class TestHmacIsStillTheGate:
    def test_unsigned_is_refused_by_hmac_not_by_auth(self, client, handlers):
        resp = _post(client, WEBHOOK_URL, "pull_request", PR_CLOSED, signed=False)
        assert resp.status_code == 401
        assert resp.json()["detail"] in HMAC_REFUSALS
        handlers["fwd"].assert_not_awaited()

    def test_forged_signature_is_refused(self, client, handlers):
        raw = json.dumps(PR_CLOSED).encode()
        resp = client.post(
            WEBHOOK_URL,
            content=raw,
            headers={
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=" + "0" * 64,
            },
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid webhook signature"
        handlers["fwd"].assert_not_awaited()


class TestExemptionIsExactlyOneRoute:
    def test_get_on_the_webhook_path_still_needs_auth(self, client):
        resp = client.get(WEBHOOK_URL)
        assert resp.status_code == 401
        assert resp.json() == AUTH_REQUIRED

    @pytest.mark.parametrize("method", ["get", "post"])
    def test_other_routes_still_need_auth(self, client, method):
        resp = getattr(client, method)("http://testserver/api/repositories/protected")
        assert resp.status_code == 401
        assert resp.json() == AUTH_REQUIRED

    @pytest.mark.parametrize(
        "url",
        [
            "http://testserver/api/repositories/webhooks/webhook/",  # trailing slash
            "http://testserver//api/repositories/webhooks/webhook",  # double slash
            "http://testserver/api/repositories//webhooks/webhook",  # inner double slash
            "http://testserver/api/repositories/webhooks/%77ebhook",  # encoded segment
            "http://testserver/api/repositories/webhooks%2Fwebhook",  # encoded separator
            "http://testserver/API/repositories/webhooks/webhook",  # uppercase
            "http://testserver/api/repositories/webhooks/WEBHOOK",  # uppercase leaf
            "http://testserver/api/repositories/webhooks/webhook?x=1",  # query string
            "http://testserver/api/repositories/webhooks/webhook/../protected",
        ],
    )
    def test_lookalike_unsigned_never_reaches_a_handler(self, client, handlers, url):
        """Each variant is refused EITHER by auth OR by HMAC -- never served."""
        resp = _post(client, url, "pull_request", PR_CLOSED, signed=False)
        assert resp.status_code == 401, (url, resp.status_code, resp.text)
        detail = resp.json().get("detail")
        assert detail == AUTH_REQUIRED["detail"] or detail in HMAC_REFUSALS, (url, detail)
        handlers["pr"].assert_not_awaited()
        handlers["fwd"].assert_not_awaited()

    @pytest.mark.parametrize(
        "url",
        [
            "http://testserver/api/repositories/webhooks/webhook/",
            "http://testserver//api/repositories/webhooks/webhook",
            "http://testserver/API/repositories/webhooks/webhook",
        ],
    )
    def test_lookalike_signed_is_still_refused_by_auth(self, client, handlers, url):
        """A VALID signature on a non-exempt path buys nothing: auth still applies."""
        resp = _post(client, url, "pull_request", PR_CLOSED, signed=True)
        assert resp.status_code == 401, (url, resp.status_code, resp.text)
        assert resp.json() == AUTH_REQUIRED
        handlers["fwd"].assert_not_awaited()


_EXEMPTION = 'if request.method == "POST" and request.url.path == GITHUB_WEBHOOK_PATH:'
MUTANTS = {
    # The defect this file exists for: no exemption => signed deliveries 401.
    "no_exemption": ("src/main.py", _EXEMPTION, "if False:"),
    # Method dropped => GET on the webhook path skips auth.
    "any_method": (
        "src/main.py",
        _EXEMPTION,
        "if request.url.path == GITHUB_WEBHOOK_PATH:",
    ),
    # Prefix match => trailing-slash / child paths skip auth.
    "prefix_match": (
        "src/main.py",
        _EXEMPTION,
        'if request.method == "POST" and request.url.path.startswith(GITHUB_WEBHOOK_PATH):',
    ),
    # Case-insensitive match => uppercase lookalikes skip auth.
    "casefold_match": (
        "src/main.py",
        _EXEMPTION,
        'if request.method == "POST" and request.url.path.lower() == GITHUB_WEBHOOK_PATH:',
    ),
}


@pytest.mark.skipif(os.environ.get("MUTANT_RUN") == "1", reason="no recursion")
class TestMutantsAreKilled:
    @pytest.mark.parametrize("name", list(MUTANTS))
    def test_mutant_turns_suite_red(self, tmp_path, name):
        rel, old, new = MUTANTS[name]
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
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                "-p",
                "no:cacheprovider",
                "tests/unit/test_webhook_auth_exemption.py",
                "-k",
                "not Mutant",
            ],
            cwd=work,
            env={**os.environ, "MUTANT_RUN": "1"},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode != 0, f"mutant {name} SURVIVED:\n{proc.stdout[-2000:]}"
        assert "failed" in proc.stdout, proc.stdout[-2000:]
