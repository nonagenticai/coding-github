"""Tests for services/outcome_forward.py.

Covers: normalising pull_request/workflow_run payloads into outcome events,
and forwarding them to n8n -- including the x-aa-github-outcome-token header, both
"forwarding disabled" cases (empty URL, empty token), and that n8n being down
(or answering an error) never raises out of ``forward_outcome_event``.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from services import outcome_forward as of


@pytest.fixture(autouse=True)
def _reset_warned_disabled():
    """The "forwarding disabled" log-once flag is process-global; reset it so
    tests do not depend on execution order."""
    of._warned_disabled = False
    yield
    of._warned_disabled = False


@pytest.fixture
def configured_forwarding(monkeypatch):
    """URL + token both set -> forwarding enabled."""
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        settings, "outcome_forward_url", "http://n8n.example/webhook/aa-github-outcome"
    )
    monkeypatch.setattr(settings, "outcome_forward_token", "the-bearer-token")
    return settings


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in that records every POST it makes."""

    calls: list[dict] = []
    next_status: int = 200
    raise_error: Exception | None = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).calls.append({"url": url, "json": json, "headers": headers})
        if type(self).raise_error is not None:
            raise type(self).raise_error
        return _FakeResponse(type(self).next_status)


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.next_status = 200
    _FakeAsyncClient.raise_error = None
    yield


# ---------------------------------------------------------------------------
# build_pull_request_outcome
# ---------------------------------------------------------------------------


class TestBuildPullRequestOutcome:
    def test_merged_pr_produces_pr_merged(self):
        payload = {
            "action": "closed",
            "pull_request": {
                "number": 42,
                "merged": True,
                "merged_at": "2026-09-30T00:00:00Z",
                "closed_at": "2026-09-30T00:00:00Z",
                "merge_commit_sha": "merge-sha-123",
                "head": {"sha": "head-sha-456"},
            },
            "repository": {"full_name": "nonagenticai/coding-github"},
        }

        event = of.build_pull_request_outcome(payload, "delivery-1")

        assert event is not None
        assert event["source"] == "github"
        assert event["delivery_id"] == "delivery-1"
        assert event["event"] == "pr_merged"
        assert event["repo"] == "nonagenticai/coding-github"
        assert event["pr_number"] == 42
        assert event["head_sha"] == "head-sha-456"
        assert event["merge_sha"] == "merge-sha-123"
        assert event["conclusion"] is None
        assert event["merged_at"] == "2026-09-30T00:00:00Z"
        assert event["closed_at"] == "2026-09-30T00:00:00Z"
        assert event["received_at"]

    def test_closed_unmerged_pr_produces_pr_closed_unmerged(self):
        payload = {
            "action": "closed",
            "pull_request": {
                "number": 7,
                "merged": False,
                "merged_at": None,
                "closed_at": "2026-09-30T01:00:00Z",
                "head": {"sha": "head-sha-abc"},
            },
            "repository": {"full_name": "nonagenticai/coding-github"},
        }

        event = of.build_pull_request_outcome(payload, "delivery-2")

        assert event is not None
        assert event["event"] == "pr_closed_unmerged"
        assert event["merge_sha"] is None
        assert event["merged_at"] is None
        assert event["closed_at"] == "2026-09-30T01:00:00Z"

    def test_non_closed_action_returns_none(self):
        payload = {"action": "opened", "pull_request": {"number": 1}}
        assert of.build_pull_request_outcome(payload, "d") is None

    def test_missing_delivery_id_is_none(self):
        payload = {
            "action": "closed",
            "pull_request": {"number": 1, "merged": True},
            "repository": {"full_name": "o/r"},
        }
        event = of.build_pull_request_outcome(payload, None)
        assert event is not None
        assert event["delivery_id"] is None


# ---------------------------------------------------------------------------
# build_workflow_run_outcome
# ---------------------------------------------------------------------------


class TestBuildWorkflowRunOutcome:
    def test_completed_on_default_branch_produces_ci_completed(self):
        payload = {
            "action": "completed",
            "workflow_run": {
                "head_branch": "main",
                "head_sha": "sha-789",
                "conclusion": "success",
            },
            "repository": {"full_name": "nonagenticai/coding-github", "default_branch": "main"},
        }

        event = of.build_workflow_run_outcome(payload, "delivery-3")

        assert event is not None
        assert event["event"] == "ci_completed"
        assert event["repo"] == "nonagenticai/coding-github"
        assert event["pr_number"] is None
        assert event["head_sha"] == "sha-789"
        assert event["conclusion"] == "success"
        assert event["merge_sha"] is None

    def test_completed_on_non_default_branch_returns_none(self):
        payload = {
            "action": "completed",
            "workflow_run": {"head_branch": "feature/x", "head_sha": "sha-1"},
            "repository": {"full_name": "o/r", "default_branch": "main"},
        }
        assert of.build_workflow_run_outcome(payload, "d") is None

    def test_non_completed_action_returns_none(self):
        payload = {
            "action": "in_progress",
            "workflow_run": {"head_branch": "main"},
            "repository": {"full_name": "o/r", "default_branch": "main"},
        }
        assert of.build_workflow_run_outcome(payload, "d") is None

    def test_missing_default_branch_returns_none(self):
        """Unknown default branch must refuse, never guess a match."""
        payload = {
            "action": "completed",
            "workflow_run": {"head_branch": "main"},
            "repository": {"full_name": "o/r"},
        }
        assert of.build_workflow_run_outcome(payload, "d") is None


# ---------------------------------------------------------------------------
# forward_outcome_event
# ---------------------------------------------------------------------------


_SAMPLE_EVENT: of.OutcomeEvent = {
    "source": "github",
    "delivery_id": "delivery-9",
    "event": "pr_merged",
    "repo": "nonagenticai/coding-github",
    "pr_number": 42,
    "head_sha": "sha",
    "merge_sha": "merge-sha",
    "conclusion": None,
    "merged_at": "2026-09-30T00:00:00Z",
    "closed_at": "2026-09-30T00:00:00Z",
    "received_at": "2026-09-30T00:00:01Z",
}


class TestForwardOutcomeEvent:
    async def test_disabled_when_url_empty(self, monkeypatch):
        from config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "outcome_forward_url", "")
        monkeypatch.setattr(settings, "outcome_forward_token", "some-token")

        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            await of.forward_outcome_event(_SAMPLE_EVENT)

        assert _FakeAsyncClient.calls == []

    async def test_disabled_when_token_empty(self, monkeypatch):
        """Addendum requirement: an empty token disables forwarding even with a URL set."""
        from config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(
            settings, "outcome_forward_url", "http://n8n.example/webhook/aa-github-outcome"
        )
        monkeypatch.setattr(settings, "outcome_forward_token", "")

        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            await of.forward_outcome_event(_SAMPLE_EVENT)

        assert _FakeAsyncClient.calls == []

    async def test_sends_bearer_token_header_and_json_body(self, configured_forwarding):
        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            await of.forward_outcome_event(_SAMPLE_EVENT)

        assert len(_FakeAsyncClient.calls) == 1
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == "http://n8n.example/webhook/aa-github-outcome"
        assert call["headers"] == {"x-aa-github-outcome-token": "the-bearer-token"}
        assert call["json"] == _SAMPLE_EVENT

    async def test_non_2xx_response_is_logged_not_raised(self, configured_forwarding, caplog):
        import logging

        _FakeAsyncClient.next_status = 500
        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            with caplog.at_level(logging.WARNING):
                await of.forward_outcome_event(_SAMPLE_EVENT)  # must not raise

        assert "Outcome forward to n8n failed" in caplog.text
        assert "500" in caplog.text

    async def test_n8n_unreachable_is_swallowed_not_raised(self, configured_forwarding, caplog):
        """n8n being completely down must never propagate out of this function --
        that is what lets the webhook response to GitHub stay 2xx regardless."""
        import logging

        _FakeAsyncClient.raise_error = httpx.ConnectError("connection refused")
        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            with caplog.at_level(logging.WARNING):
                await of.forward_outcome_event(_SAMPLE_EVENT)  # must not raise

        assert "Outcome forward to n8n failed" in caplog.text
        assert "ConnectError" in caplog.text

    async def test_token_never_appears_in_logs(self, configured_forwarding, caplog):
        import logging

        _FakeAsyncClient.raise_error = httpx.ConnectError("boom")
        with patch("services.outcome_forward.httpx.AsyncClient", _FakeAsyncClient):
            with caplog.at_level(logging.DEBUG):
                await of.forward_outcome_event(_SAMPLE_EVENT)

        assert "the-bearer-token" not in caplog.text

    async def test_disabled_reason_logged_once_per_process(self, monkeypatch, caplog):
        import logging

        from config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "outcome_forward_url", "")
        monkeypatch.setattr(settings, "outcome_forward_token", "")

        with caplog.at_level(logging.INFO):
            await of.forward_outcome_event(_SAMPLE_EVENT)
            await of.forward_outcome_event(_SAMPLE_EVENT)

        disabled_lines = [r for r in caplog.records if "forwarding to n8n is disabled" in r.message]
        assert len(disabled_lines) == 1
