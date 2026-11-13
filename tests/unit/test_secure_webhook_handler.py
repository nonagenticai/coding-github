"""Tests for services/github/secure_webhook_handler.py

Verifies HMAC-SHA256 signature validation, event routing, security controls
(rate limiting, IP blocking), and webhook processing pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from services.github.secure_webhook_handler import (
    SecureWebhookHandler,
    WebhookSecurityLevel,
)


def _make_handler(secret: str = "test-secret", level=WebhookSecurityLevel.STANDARD):
    return SecureWebhookHandler(webhook_secret=secret, security_level=level)


def _sign_payload(payload: bytes, secret: str) -> str:
    """Compute valid HMAC-SHA256 signature for a payload."""
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _make_request(ip: str = "127.0.0.1", headers: dict | None = None):
    """Build a minimal mock FastAPI Request."""
    req = MagicMock()
    req.headers = headers or {}
    req.client = MagicMock()
    req.client.host = ip
    return req


class TestVerifySignature:
    """HMAC-SHA256 signature verification."""

    def test_valid_signature_returns_true(self):
        """Valid HMAC signature is accepted."""
        handler = _make_handler("my-secret")
        payload = b'{"action":"opened"}'
        sig = _sign_payload(payload, "my-secret")
        assert handler.verify_signature(payload, sig) is True

    def test_wrong_signature_returns_false(self):
        """Wrong HMAC signature is rejected."""
        handler = _make_handler("my-secret")
        payload = b'{"action":"opened"}'
        assert handler.verify_signature(payload, "sha256=badhex") is False

    def test_empty_signature_returns_false(self):
        """Empty signature string is rejected."""
        handler = _make_handler("my-secret")
        assert handler.verify_signature(b"payload", "") is False

    def test_signature_without_prefix_still_works(self):
        """Signature without sha256= prefix is still verified."""
        handler = _make_handler("my-secret")
        payload = b'{"test": true}'
        raw_sig = hmac.new(b"my-secret", payload, hashlib.sha256).hexdigest()
        assert handler.verify_signature(payload, raw_sig) is True

    @pytest.mark.parametrize(
        ("secret", "payload"),
        [
            ("", b"empty-secret"),
            ("secret", b""),
            ("s", b"x"),
        ],
        ids=["empty-secret", "empty-payload", "minimal"],
    )
    def test_signature_edge_cases(self, secret, payload):
        """Signature verification handles edge-case inputs correctly."""
        handler = _make_handler(secret)
        sig = _sign_payload(payload, secret)
        assert handler.verify_signature(payload, sig) is True

    def test_tampered_payload_rejected(self):
        """Signature computed on original payload fails for tampered payload."""
        handler = _make_handler("my-secret")
        original = b'{"action":"opened"}'
        sig = _sign_payload(original, "my-secret")
        tampered = b'{"action":"closed"}'
        assert handler.verify_signature(tampered, sig) is False


class TestHandleWebhookRouting:
    """Webhook event routing to registered handlers."""

    async def test_push_event_routed_to_handler(self):
        """Push events are dispatched to the registered push handler."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        push_handler = AsyncMock(return_value=True)
        handler.register_event_handler("push", push_handler)

        payload = json.dumps({"ref": "refs/heads/main", "commits": []}).encode()
        request = _make_request()

        result = await handler.handle_webhook(
            request=request,
            payload=payload,
            event_type="push",
            signature=None,
            delivery_id="del-1",
        )

        assert result.event_type == "push"
        assert result.processed is True
        assert result.processing_time_ms > 0
        push_handler.assert_awaited_once()

    async def test_pull_request_event_captures_action(self):
        """Pull request events extract the action field."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        pr_handler = AsyncMock(return_value=True)
        handler.register_event_handler("pull_request", pr_handler)

        payload = json.dumps({"action": "opened", "pull_request": {"number": 1}}).encode()
        request = _make_request()

        result = await handler.handle_webhook(
            request=request,
            payload=payload,
            event_type="pull_request",
            signature=None,
        )

        assert result.event_type == "pull_request"
        assert result.action == "opened"
        pr_handler.assert_awaited_once()

    async def test_no_handler_registered_returns_not_processed(self):
        """Events with no registered handlers return processed=False."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        # Use a supported event type that has no handler registered
        payload = json.dumps({"action": "created"}).encode()
        request = _make_request()

        result = await handler.handle_webhook(
            request=request,
            payload=payload,
            event_type="issues",
            signature=None,
        )
        assert result.processed is False

    async def test_multiple_handlers_any_true_means_processed(self):
        """When multiple handlers are registered, any returning True means processed."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        handler.register_event_handler("push", AsyncMock(return_value=False))
        handler.register_event_handler("push", AsyncMock(return_value=True))

        payload = json.dumps({"ref": "refs/heads/main"}).encode()
        result = await handler.handle_webhook(
            request=_make_request(),
            payload=payload,
            event_type="push",
            signature=None,
        )
        assert result.processed is True


class TestWebhookSecurityValidation:
    """Security validation: missing signatures, invalid signatures, blocked IPs."""

    async def test_missing_signature_in_standard_mode_raises_401(self):
        """Standard mode rejects requests without a signature."""
        handler = _make_handler("secret", WebhookSecurityLevel.STANDARD)
        payload = json.dumps({"action": "opened"}).encode()

        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_webhook(
                request=_make_request(),
                payload=payload,
                event_type="push",
                signature=None,
            )
        assert exc_info.value.status_code == 401

    async def test_invalid_signature_raises_401_and_blocks_ip(self):
        """Invalid signature raises 401 and temporarily blocks the source IP."""
        handler = _make_handler("real-secret")
        payload = json.dumps({"action": "opened"}).encode()

        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_webhook(
                request=_make_request(ip="5.5.5.5"),
                payload=payload,
                event_type="push",
                signature="sha256=wrong",
            )
        assert exc_info.value.status_code == 401
        assert "5.5.5.5" in handler._blocked_ips
        # Cancel background unblock task
        for task in handler._background_tasks:
            task.cancel()

    async def test_valid_signature_in_standard_mode_succeeds(self):
        """Standard mode accepts requests with a valid signature."""
        handler = _make_handler("real-secret", WebhookSecurityLevel.STANDARD)
        handler.register_event_handler("push", AsyncMock(return_value=True))
        payload = json.dumps({"ref": "refs/heads/main"}).encode()
        sig = _sign_payload(payload, "real-secret")

        result = await handler.handle_webhook(
            request=_make_request(),
            payload=payload,
            event_type="push",
            signature=sig,
        )
        assert result.processed is True

    async def test_malformed_json_payload_raises_400(self):
        """Malformed JSON payload raises 400."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_webhook(
                request=_make_request(),
                payload=b"not-json{{{",
                event_type="push",
                signature=None,
            )
        assert exc_info.value.status_code == 400

    async def test_unsupported_event_type_raises_400(self):
        """Unknown event types raise 400."""
        handler = _make_handler("s", WebhookSecurityLevel.DEVELOPMENT)
        payload = json.dumps({"x": 1}).encode()
        with pytest.raises(HTTPException) as exc_info:
            await handler.handle_webhook(
                request=_make_request(),
                payload=payload,
                event_type="totally_fake_event",
                signature=None,
            )
        assert exc_info.value.status_code == 400
        assert "Unsupported event type" in exc_info.value.detail

    async def test_development_mode_skips_signature_check(self):
        """Development mode processes without requiring a signature."""
        handler = _make_handler("secret", WebhookSecurityLevel.DEVELOPMENT)
        handler.register_event_handler("push", AsyncMock(return_value=True))

        payload = json.dumps({"ref": "refs/heads/main"}).encode()
        result = await handler.handle_webhook(
            request=_make_request(),
            payload=payload,
            event_type="push",
            signature=None,
        )
        assert result.processed is True
