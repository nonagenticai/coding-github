"""Extended tests for services/github/secure_webhook_handler.py

Covers: rate limiting, IP blocking/unblocking, client IP extraction,
metrics tracking, sync handler execution, handler exceptions, event handler
functions (push, PR, issues), and the singleton factory.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from services.github.secure_webhook_handler import (
    SecureWebhookHandler,
    WebhookProcessingResult,
    WebhookSecurityLevel,
    get_secure_webhook_handler,
    handle_issues_event,
    handle_pull_request_event,
    handle_push_event,
)


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch):
    """These tests pin behaviour that uses the PLATFORM PAT, which since
    2026-09-22 may act only on ``system_github_owner`` repos; a customer repo is
    skipped/refused (tests/unit/test_platform_pat_customer_guard.py). So the
    fixture owner "org" is declared the platform org here.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "org")


# ── helpers ──────────────────────────────────────────────────────────────


def _make_handler(secret: str = "test-secret", level=WebhookSecurityLevel.STANDARD):
    return SecureWebhookHandler(webhook_secret=secret, security_level=level)


def _sign(payload: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _req(ip: str = "127.0.0.1", headers: dict | None = None):
    r = MagicMock()
    r.headers = headers or {}
    r.client = MagicMock()
    r.client.host = ip
    return r


# ── Signature verification edge cases ────────────────────────────────────


class TestVerifySignatureException:
    def test_returns_false_on_internal_error(self):
        """Exception during HMAC computation returns False."""
        handler = _make_handler()
        with patch(
            "services.github.secure_webhook_handler.hmac.new", side_effect=TypeError("boom")
        ):
            assert handler.verify_signature(b"payload", "sha256=abc") is False


# ── Rate limiting ────────────────────────────────────────────────────────


class TestRateLimiting:
    def test_rate_limit_enforced_at_capacity(self):
        """_check_rate_limit returns False when max requests exceeded."""
        handler = _make_handler()
        handler._max_requests_per_minute = 2
        assert handler._check_rate_limit("a") is True
        assert handler._check_rate_limit("a") is True
        assert handler._check_rate_limit("a") is False

    async def test_rate_limit_exceeded_raises_429(self):
        """Exceeding rate limit raises HTTP 429."""
        handler = _make_handler()
        handler._max_requests_per_minute = 0
        with pytest.raises(HTTPException) as exc:
            await handler._validate_request_security("9.9.9.9", b"p", "sha256=abc")
        assert exc.value.status_code == 429


# ── IP blocking ──────────────────────────────────────────────────────────


class TestIPBlocking:
    async def test_blocked_ip_raises_403(self):
        """Request from blocked IP raises 403."""
        handler = _make_handler()
        handler._blocked_ips.add("1.2.3.4")
        with pytest.raises(HTTPException) as exc:
            await handler._validate_request_security("1.2.3.4", b"p", "sig")
        assert exc.value.status_code == 403

    async def test_block_ip_temporarily_adds_ip_and_creates_task(self):
        """_block_ip_temporarily adds IP and schedules an unblock task."""
        handler = _make_handler()
        handler._block_ip_temporarily("6.6.6.6")
        assert "6.6.6.6" in handler._blocked_ips
        assert len(handler._background_tasks) == 1
        for task in handler._background_tasks:
            task.cancel()

    async def test_unblock_later_removes_ip(self):
        """The unblock_later coroutine removes the IP after the delay."""
        handler = _make_handler()
        with patch("services.github.secure_webhook_handler.asyncio.sleep", new_callable=AsyncMock):
            handler._block_ip_temporarily("7.7.7.7")
            assert "7.7.7.7" in handler._blocked_ips
            for task in list(handler._background_tasks):
                await task
            assert "7.7.7.7" not in handler._blocked_ips


# ── Client IP extraction ────────────────────────────────────────────────


class TestGetClientIP:
    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}, "10.0.0.1"),
            ({"X-Real-IP": "172.16.0.1"}, "172.16.0.1"),
            ({}, "192.168.1.1"),
        ],
        ids=["x-forwarded-for", "x-real-ip", "client-host"],
    )
    def test_ip_extraction_priority(self, headers, expected):
        """IP extraction follows X-Forwarded-For > X-Real-IP > client.host."""
        handler = _make_handler()
        req = _req(ip="192.168.1.1", headers=headers)
        assert handler._get_client_ip(req) == expected

    def test_unknown_when_no_client(self):
        """Returns 'unknown' when request has no client."""
        handler = _make_handler()
        req = MagicMock()
        req.headers = {}
        req.client = None
        assert handler._get_client_ip(req) == "unknown"


# ── Event processing internals ───────────────────────────────────────────


class TestProcessEvent:
    async def test_no_handlers_returns_false(self):
        """No registered handlers returns False."""
        handler = _make_handler()
        result = await handler._process_event("push", {"ref": "main"})
        assert result is False

    async def test_sync_handler_called(self):
        """Synchronous (non-async) handler is executed correctly."""
        handler = _make_handler()
        sync_fn = MagicMock(return_value=True)
        handler.register_event_handler("push", sync_fn)
        result = await handler._process_event("push", {"ref": "main"})
        assert result is True
        sync_fn.assert_called_once_with({"ref": "main"})

    async def test_handler_exception_results_in_false(self):
        """Handler that raises is caught; result counts as False."""
        handler = _make_handler()
        bad = AsyncMock(side_effect=ValueError("oops"))
        handler.register_event_handler("push", bad)
        result = await handler._process_event("push", {})
        assert result is False


class TestParseEventType:
    def test_unknown_event_returns_none(self):
        """Unknown event string returns None."""
        handler = _make_handler()
        assert handler._parse_event_type("unknown_event_xyz") is None

    @pytest.mark.parametrize(
        "event_type",
        ["push", "pull_request", "issue_comment", "issues"],
    )
    def test_known_event_types_parsed(self, event_type):
        """Known GitHub event types are parsed successfully."""
        handler = _make_handler()
        result = handler._parse_event_type(event_type)
        assert result is not None


# ── Metrics tracking ─────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_trimmed_to_max_history(self):
        """Metrics list is trimmed when exceeding max."""
        handler = _make_handler()
        handler._max_metrics_history = 3
        for i in range(5):
            result = WebhookProcessingResult(event_type="push", processing_time_ms=float(i))
            handler._add_metrics(result)
        assert len(handler._processed_events) == 3
        assert handler._processed_events[0].processing_time_ms == 2.0

    def test_empty_metrics(self):
        """Empty metrics returns total_events 0."""
        handler = _make_handler()
        m = handler.get_processing_metrics()
        assert m == {"total_events": 0}

    def test_metrics_with_events_includes_breakdown(self):
        """Populated metrics include event type breakdown and success rate."""
        handler = _make_handler()
        handler._add_metrics(
            WebhookProcessingResult(event_type="push", processed=True, processing_time_ms=10.0)
        )
        handler._add_metrics(
            WebhookProcessingResult(event_type="push", processed=False, processing_time_ms=20.0)
        )
        handler._add_metrics(
            WebhookProcessingResult(
                event_type="pull_request", processed=True, processing_time_ms=30.0
            )
        )
        handler.register_event_handler("push", AsyncMock())
        m = handler.get_processing_metrics()
        assert m["total_events"] == 3
        assert m["successful_events"] == 2
        assert m["success_rate"] == pytest.approx(2 / 3)
        assert m["average_processing_time_ms"] == pytest.approx(20.0)
        assert m["event_type_breakdown"]["push"] == 2
        assert m["event_type_breakdown"]["pull_request"] == 1
        assert "push" in m["active_handlers"]


# ── Generic processing exception ────────────────────────────────────────


class TestGenericProcessingException:
    async def test_generic_exception_yields_500_with_metrics(self):
        """Unexpected error during event processing yields HTTP 500 and records metrics."""
        handler = _make_handler("s", WebhookSecurityLevel.DEVELOPMENT)
        payload = json.dumps({"ref": "refs/heads/main"}).encode()

        with patch.object(handler, "_process_event", side_effect=RuntimeError("kaboom")):
            with pytest.raises(HTTPException) as exc:
                await handler.handle_webhook(request=_req(), payload=payload, event_type="push")
            assert exc.value.status_code == 500
            assert "kaboom" in exc.value.detail
        assert any(r.error == "kaboom" for r in handler._processed_events)


# ── Event handler functions ──────────────────────────────────────────────


class TestWebhookHandlersResolveRealModules:
    """The registered handlers must reach modules that EXIST in this service.

    Both handlers used to import pre-split monolith modules (``api.endpoints``,
    ``core.github_integration``) that do not exist in coding-github. The
    ``except Exception`` in each handler laundered the ``ModuleNotFoundError``
    into ``return False``, so every ``pull_request`` and every ``push`` webhook
    was a no-op that logged one line and reported "not processed".

    These tests patch only REAL, in-repo symbols — no ``sys.modules`` fakes —
    so they cannot pass against an implementation that reaches for a module
    this service does not ship.
    """

    async def test_pr_event_calls_in_repo_analyzer(self):
        analyzer = AsyncMock(return_value=None)
        with patch("routers.github_webhooks.analyze_pull_request_changes", analyzer):
            result = await handle_pull_request_event(
                {
                    "action": "opened",
                    "pull_request": {"number": 42},
                    "repository": {"full_name": "org/repo"},
                }
            )
        assert result is True, "pull_request webhook must be processed, not swallowed"
        analyzer.assert_awaited_once_with("org/repo", 42, "opened")

    async def test_push_event_completes_without_dying_on_absent_monolith_state(self, caplog):
        """Runs the push handler with NOTHING patched, against the real hub.

        This test used to assert the push path still FAILED, and said so
        explicitly: ``handle_repository_update`` depended on a monolith-era
        per-repo sqlite code graph via ``Settings.get_database_manager()`` — a
        method that exists on neither ``Settings`` nor
        ``BaseMicroserviceSettings`` — and that was called out as "a separate
        architecture-level fix, deliberately NOT claimed working here".

        That fix has now landed: the graph refresh is delegated to
        coding-analysis, this service owns no graph, and the push path runs to
        completion. What it must never do again:

        * import a module this service does not ship (``ModuleNotFoundError``),
        * die on ``Settings.get_database_manager`` (``AttributeError``),
        * or swallow a failure without a traceback.

        Nothing here is mocked, so the container lookup genuinely fails (no
        lifespan, so no containers client). That failure must be *handled and
        reported*, not fatal — which is exactly what is asserted.
        """
        with caplog.at_level(logging.ERROR):
            result = await handle_push_event(
                {
                    "repository": {"full_name": "org/repo"},
                    "ref": "refs/heads/main",
                    "commits": [{"id": "sha1", "modified": ["a.py"]}],
                }
            )

        # The webhook is processed: the handler no longer dies on absent state.
        assert result is True, "push webhook must complete, not die on monolith-era state"

        # The old failure modes must be gone, wherever they were logged.
        for record in caplog.records:
            if record.exc_info:
                exc_type = record.exc_info[0]
                assert not issubclass(exc_type, ModuleNotFoundError), (
                    "push handler must not import a module this service does not ship"
                )
                assert "get_database_manager" not in str(record.exc_info[1]), (
                    "push handler must not reach for the monolith-era sqlite graph"
                )

        # The original property this test was written for: this handler must
        # never swallow a failure with a bare str(e). It no longer swallows this
        # path at all, so assert the property conditionally rather than forcing a
        # failure to occur. (That a failed refresh is REPORTED — not silently
        # treated as done — is asserted directly on the return value in
        # test_webhook_handler.py::test_resolution_failure_is_reported_as_error.)
        swallowed = [
            r
            for r in caplog.records
            if r.name == "services.github.secure_webhook_handler" and r.levelno >= logging.ERROR
        ]
        assert all(r.exc_info for r in swallowed), (
            "a swallowed webhook failure must carry a traceback, not just str(e)"
        )

    async def test_negative_control_real_failure_still_returns_false(self):
        """Control: a genuine failure inside the analyzer must still be False.

        Guards against "fix" by making the handlers unconditionally return True.
        """
        analyzer = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("routers.github_webhooks.analyze_pull_request_changes", analyzer):
            result = await handle_pull_request_event(
                {
                    "action": "opened",
                    "pull_request": {"number": 42},
                    "repository": {"full_name": "org/repo"},
                }
            )
        assert result is False


class TestHandlePullRequestEvent:
    async def test_pr_opened_with_handler_registry(self):
        """Opened PR imports registry and calls analyze."""
        mock_handler = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_handler

        interfaces_mod = MagicMock()
        interfaces_mod.WebhookEventHandlerRegistry = mock_registry

        with patch.dict("sys.modules", {"interfaces": interfaces_mod}):
            result = await handle_pull_request_event(
                {
                    "action": "opened",
                    "pull_request": {"number": 42},
                    "repository": {"full_name": "org/repo"},
                }
            )
        assert result is True
        mock_handler.analyze_pull_request_changes.assert_awaited_once_with("org/repo", 42, "opened")

    async def test_pr_opened_fallback_to_in_repo_analyzer(self):
        """ValueError from registry falls back to the in-repo router analyzer.

        This used to inject a fake ``api.endpoints.github_webhooks`` into
        ``sys.modules``. That module does not exist in coding-github, so the
        test was green while production raised ModuleNotFoundError on every
        PR webhook. Patch the REAL symbol instead.
        """
        mock_registry = MagicMock()
        mock_registry.get.side_effect = ValueError("no handler")

        interfaces_mod = MagicMock()
        interfaces_mod.WebhookEventHandlerRegistry = mock_registry

        analyzer = AsyncMock(return_value=None)

        with (
            patch.dict("sys.modules", {"interfaces": interfaces_mod}),
            patch("routers.github_webhooks.analyze_pull_request_changes", analyzer),
        ):
            result = await handle_pull_request_event(
                {
                    "action": "synchronize",
                    "pull_request": {"number": 7},
                    "repository": {"full_name": "org/repo"},
                }
            )
        assert result is True
        analyzer.assert_awaited_once_with("org/repo", 7, "synchronize")

    async def test_pr_opened_import_error_returns_false(self):
        """Import failure returns False."""
        with patch.dict("sys.modules", {"interfaces": None}):
            result = await handle_pull_request_event(
                {
                    "action": "reopened",
                    "pull_request": {"number": 1},
                    "repository": {"full_name": "org/r"},
                }
            )
        assert result is False

    @pytest.mark.parametrize(
        ("action", "merged", "expected"),
        [
            ("closed", True, True),
            ("closed", False, True),
            ("labeled", None, False),
        ],
        ids=["merged", "closed-not-merged", "unhandled-action"],
    )
    async def test_pr_closed_and_unhandled_actions(self, action, merged, expected):
        """Closed PRs (merged or not) return True; unhandled actions return False."""
        event = {
            "action": action,
            "pull_request": {"number": 5},
            "repository": {"full_name": "org/repo"},
        }
        if merged is not None:
            event["pull_request"]["merged"] = merged
        result = await handle_pull_request_event(event)
        assert result is expected

    async def test_pr_merged_forwards_pr_merged_outcome(self):
        """Closed+merged forwards a pr_merged outcome event to n8n instead of
        n8n having to poll for it (George's ruling: no timer may write on a
        clock)."""
        import asyncio

        forward_mock = AsyncMock()
        event = {
            "action": "closed",
            "pull_request": {
                "number": 5,
                "merged": True,
                "merge_commit_sha": "merge-sha",
                "head": {"sha": "head-sha"},
            },
            "repository": {"full_name": "org/repo"},
            "_coding_github_delivery_id": "delivery-xyz",
        }
        with patch("services.outcome_forward.forward_outcome_event", forward_mock):
            result = await handle_pull_request_event(event)
            # Give the fire-and-forget asyncio.create_task a chance to run.
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending)

        assert result is True
        forward_mock.assert_awaited_once()
        forwarded = forward_mock.await_args.args[0]
        assert forwarded["event"] == "pr_merged"
        assert forwarded["delivery_id"] == "delivery-xyz"
        assert forwarded["pr_number"] == 5
        assert forwarded["merge_sha"] == "merge-sha"

    async def test_pr_closed_unmerged_forwards_pr_closed_unmerged_outcome(self):
        import asyncio

        forward_mock = AsyncMock()
        event = {
            "action": "closed",
            "pull_request": {"number": 6, "merged": False, "head": {"sha": "head-sha-2"}},
            "repository": {"full_name": "org/repo"},
            "_coding_github_delivery_id": "delivery-abc",
        }
        with patch("services.outcome_forward.forward_outcome_event", forward_mock):
            result = await handle_pull_request_event(event)
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending)

        assert result is True
        forward_mock.assert_awaited_once()
        forwarded = forward_mock.await_args.args[0]
        assert forwarded["event"] == "pr_closed_unmerged"
        assert forwarded["merge_sha"] is None


class TestHandlePushEvent:
    async def test_push_to_main_processes_commits(self):
        """Push to main resolves the in-repo hub and processes changed files.

        This used to inject a fake ``core.github_integration`` module into
        ``sys.modules``; no such module exists in coding-github, so the test
        was green while production raised ModuleNotFoundError on every push.
        Patch the REAL symbol instead.
        """
        mock_hub_instance = AsyncMock()
        mock_hub_class = MagicMock(return_value=mock_hub_instance)

        with patch(
            "services.github_integration_core.GitHubIntegrationHub",
            mock_hub_class,
        ):
            result = await handle_push_event(
                {
                    "repository": {"full_name": "org/repo"},
                    "ref": "refs/heads/main",
                    "commits": [
                        {
                            "id": "sha1",
                            "modified": ["a.py"],
                            "added": ["b.py"],
                            "removed": ["c.py"],
                        },
                        {"id": "sha2", "modified": ["d.py"], "added": [], "removed": []},
                    ],
                }
            )
        assert result is True
        call_args = mock_hub_instance.handle_repository_update.call_args
        assert set(call_args[0][2]) == {"a.py", "b.py", "c.py", "d.py"}
        assert call_args[0][3] == ["sha1", "sha2"]

    async def test_push_to_feature_branch_returns_true(self):
        """Push to non-protected branch returns True immediately."""
        result = await handle_push_event(
            {
                "repository": {"full_name": "org/repo"},
                "ref": "refs/heads/feature-xyz",
                "commits": [],
            }
        )
        assert result is True

    async def test_push_import_error_returns_false(self):
        """Import failure on push handler returns False."""
        with patch.dict("sys.modules", {"services.github_integration_core": None}):
            result = await handle_push_event(
                {
                    "repository": {"full_name": "org/repo"},
                    "ref": "refs/heads/develop",
                    "commits": [],
                }
            )
        assert result is False


class TestHandleIssuesEvent:
    async def test_issues_event_returns_true(self):
        """Issues handler always returns True."""
        result = await handle_issues_event(
            {
                "action": "opened",
                "issue": {"number": 99},
                "repository": {"full_name": "org/repo"},
            }
        )
        assert result is True


# ── Singleton factory ────────────────────────────────────────────────────


class TestGetSecureWebhookHandler:
    def test_creates_singleton_with_registered_handlers(self):
        """Factory creates handler with push, pull_request, and issues handlers."""
        import services.github.secure_webhook_handler as mod

        original = mod._webhook_handler
        mod._webhook_handler = None
        try:
            h = get_secure_webhook_handler(webhook_secret="my-test-secret")
            assert isinstance(h, SecureWebhookHandler)
            assert h.security_level == WebhookSecurityLevel.STANDARD
            assert "pull_request" in h._event_handlers
            assert "push" in h._event_handlers
            assert "issues" in h._event_handlers
            # Second call returns same instance
            assert get_secure_webhook_handler() is h
        finally:
            mod._webhook_handler = original

    def test_reads_settings_secret_when_no_secret_passed(self, monkeypatch):
        """When no secret is passed, reads Settings.github_webhook_secret --
        i.e. the PREFIXED ``CODING_GITHUB_GITHUB_WEBHOOK_SECRET``, the field
        this service's Deployment actually sets.

        ⛔ FIXED 2026-09-30: this used to read the unprefixed
        ``GITHUB_WEBHOOK_SECRET`` env var -- never set anywhere in gitops --
        so the configured secret had no effect in production. See
        ``get_secure_webhook_handler``'s docstring.
        """
        import services.github.secure_webhook_handler as mod
        from config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "github_webhook_secret", "settings-secret")

        original = mod._webhook_handler
        mod._webhook_handler = None
        try:
            h = get_secure_webhook_handler()
            assert h.webhook_secret == b"settings-secret"
        finally:
            mod._webhook_handler = original

    def test_empty_configured_secret_is_not_silently_substituted(self, monkeypatch):
        """An empty ``Settings.github_webhook_secret`` must NOT fall back to any
        hardcoded default (the old ``"dev_secret_change_in_production"``
        literal is gone entirely) -- the handler is built with a genuinely
        empty secret, and ``_validate_request_security`` is what then refuses
        every delivery (see ``TestEmptySecretFailsClosed`` below)."""
        import services.github.secure_webhook_handler as mod
        from config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "github_webhook_secret", "")

        original = mod._webhook_handler
        mod._webhook_handler = None
        try:
            h = get_secure_webhook_handler()
            assert h.webhook_secret == b""
        finally:
            mod._webhook_handler = original


class TestEmptySecretFailsClosed:
    """The HMAC fail-closed fix: an empty configured secret refuses every
    delivery outright, rather than verifying against a guessable default."""

    async def test_empty_secret_raises_401_even_with_a_signature_present(self):
        handler = SecureWebhookHandler(
            webhook_secret="", security_level=WebhookSecurityLevel.STANDARD
        )
        with pytest.raises(HTTPException) as exc_info:
            await handler._validate_request_security(
                "1.2.3.4", b'{"action":"opened"}', "sha256=whatever"
            )
        assert exc_info.value.status_code == 401
        assert "secret" in exc_info.value.detail.lower()

    async def test_empty_secret_raises_401_with_no_signature_at_all(self):
        handler = SecureWebhookHandler(
            webhook_secret="", security_level=WebhookSecurityLevel.STANDARD
        )
        with pytest.raises(HTTPException) as exc_info:
            await handler._validate_request_security("1.2.3.4", b"payload", None)
        assert exc_info.value.status_code == 401
