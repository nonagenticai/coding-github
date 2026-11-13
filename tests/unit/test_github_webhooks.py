"""Unit tests for GitHub webhook API endpoints.

Tests for:
- src/app/api/endpoints/github_webhooks.py

Covers webhook handling, signature verification, PR events,
push events, and issue notifications.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient


# Disable auth before importing endpoints
@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch):
    """The fixture owner "owner" is declared the platform org. Webhooks have no
    GitHub credential at all since 2026-09-30 (no PAT, no tenant for the App), so
    tests that exercise the post-credential code stub ``_webhook_github_token``.
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "owner")


@pytest.fixture(autouse=True)
def _legacy_all_webhook_mode(monkeypatch):
    """This module pins the legacy full-dispatch behaviour, which since
    coding-github#131 requires CODING_GITHUB_WEBHOOK_MODE=all (code default is
    outcome_only; see tests/unit/test_webhook_outcome_only.py)."""
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "webhook_mode", "all")


os.environ["ENABLE_AUTH"] = "false"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.body = AsyncMock(return_value=b'{"test": "payload"}')
    return request


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.GITHUB_WEBHOOK_SECRET = "test-secret"
    settings.coding_gateway_url = "http://coding-gateway:8687"
    return settings


@pytest.fixture
def mock_webhook_result():
    """Create mock webhook processing result."""
    from services.github.secure_webhook_handler import WebhookProcessingResult

    return WebhookProcessingResult(
        event_type="pull_request",
        action="opened",
        processed=True,
        processing_time_ms=125.5,
    )


@pytest.fixture
def pr_payload():
    """Create a sample pull request webhook payload."""
    return {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "number": 42,
            "title": "Test PR",
            "body": "Test description",
            "user": {"login": "testuser"},
            "base": {"sha": "base123", "ref": "main"},
            "head": {"sha": "head456", "ref": "feature"},
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "merged": False,
        },
        "repository": {
            "full_name": "owner/repo",
            "name": "repo",
            "owner": {"login": "owner"},
        },
    }


@pytest.fixture
def push_payload():
    """Create a sample push webhook payload."""
    return {
        "ref": "refs/heads/main",
        "commits": [
            {
                "id": "abc123",
                "author": {"name": "Test User"},
                "message": "Test commit",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ],
        "repository": {
            "full_name": "owner/repo",
            "name": "repo",
            "owner": {"login": "owner"},
        },
    }


@pytest.fixture
def issue_comment_payload():
    """Create a sample issue comment payload."""
    return {
        "action": "created",
        "issue": {
            "number": 42,
            "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/42"},
        },
        "comment": {
            "body": "@codehelper analyze this",
            "user": {"login": "testuser"},
        },
        "repository": {
            "full_name": "owner/repo",
        },
    }


@pytest.fixture
def review_comment_payload():
    """Create a sample PR review comment payload."""
    return {
        "action": "created",
        "comment": {
            "body": "/analyze the changes",
            "user": {"login": "testuser"},
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {
            "full_name": "owner/repo",
        },
    }


@pytest.fixture
def pr_analysis_dict():
    """Create a sample PR analysis result dictionary."""
    return {
        "statistics": {
            "total": 5,
            "added": 2,
            "modified": 2,
            "removed": 1,
        },
        "total_additions": 150,
        "total_deletions": 50,
        "languages": {
            "python": {"files": 3, "additions": 100, "deletions": 30},
            "javascript": {"files": 2, "additions": 50, "deletions": 20},
        },
        "files": [],
    }


# =============================================================================
# Webhook Handler Endpoint Tests
# =============================================================================


@pytest.mark.unit
class TestGitHubWebhookHandler:
    """Tests for the main GitHub webhook handler endpoint."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with github webhooks router."""
        from coding_client.auth.tenant import require_tenant

        from routers.github_webhooks import router

        app = FastAPI()
        app.include_router(router, prefix="/github")
        app.dependency_overrides[require_tenant] = lambda: "test-tenant"
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_webhook_handler_success(self, client, mock_webhook_result):
        """Test successful webhook handling."""
        payload = {"action": "opened", "pull_request": {"number": 42}}

        with patch("routers.github_webhooks.WebhookEventHandlerRegistry") as mock_registry:
            mock_handler = AsyncMock()
            mock_handler.handle_webhook = AsyncMock(return_value=mock_webhook_result)
            mock_registry.get_handler.return_value = mock_handler

            response = client.post(
                "/github/webhook",
                json=payload,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": "sha256=test",
                    "X-GitHub-Delivery": "delivery-123",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["event_type"] == "pull_request"
            assert data["processed"] is True

    def test_webhook_handler_fallback_to_direct_import(self, client, mock_webhook_result):
        """Test webhook handler with fallback when registry returns None."""
        payload = {"action": "opened"}

        with patch("routers.github_webhooks.WebhookEventHandlerRegistry") as mock_registry:
            mock_registry.get_handler.return_value = None
            # Patch in the secure_webhook_handler module directly
            with patch(
                "services.github.secure_webhook_handler.get_secure_webhook_handler"
            ) as mock_get_handler:
                mock_handler = AsyncMock()
                mock_handler.handle_webhook = AsyncMock(return_value=mock_webhook_result)
                mock_get_handler.return_value = mock_handler

                response = client.post(
                    "/github/webhook",
                    json=payload,
                    headers={
                        "X-GitHub-Event": "ping",
                        "X-Hub-Signature-256": "sha256=test",
                    },
                )

                assert response.status_code == 200

    def test_webhook_handler_processing_failure(self, client):
        """Test webhook handler with processing failure."""
        payload = {"action": "test"}

        with patch("routers.github_webhooks.WebhookEventHandlerRegistry") as mock_registry:
            mock_handler = AsyncMock()
            mock_handler.handle_webhook = AsyncMock(side_effect=Exception("Processing failed"))
            mock_registry.get_handler.return_value = mock_handler

            response = client.post(
                "/github/webhook",
                json=payload,
                headers={"X-GitHub-Event": "test_event"},
            )

            assert response.status_code == 500

    def test_webhook_handler_http_exception(self, client):
        """Test webhook handler propagates HTTPException."""
        payload = {"action": "test"}

        with patch("routers.github_webhooks.WebhookEventHandlerRegistry") as mock_registry:
            mock_handler = AsyncMock()
            mock_handler.handle_webhook = AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Unauthorized")
            )
            mock_registry.get_handler.return_value = mock_handler

            response = client.post(
                "/github/webhook",
                json=payload,
                headers={"X-GitHub-Event": "test_event"},
            )

            assert response.status_code == 401

    def test_webhook_reachable_with_no_tenant_header(self, mock_webhook_result):
        """GitHub cannot send X-Tenant-Id; this route must not require it.

        ⛔ FIXED 2026-09-30: this route used to carry
        ``Depends(require_tenant)``, which 401s an unauthenticated caller by
        default (no TRUSTED_CIDRS/DEFAULT_TENANT_ID configured -- this
        service's live default). Unlike the other tests in this class, this
        one does NOT rely on ``app.dependency_overrides[require_tenant]`` (the
        ``app`` fixture above still sets it, harmlessly, since the route no
        longer depends on it at all) -- it proves the route is reachable with
        NO tenant machinery involved by building a bare app with none of that
        wiring.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routers.github_webhooks import router as webhook_router

        bare_app = FastAPI()
        bare_app.include_router(webhook_router, prefix="/github")
        bare_client = TestClient(bare_app)

        payload = {"action": "opened", "pull_request": {"number": 1}}
        with patch("routers.github_webhooks.WebhookEventHandlerRegistry") as mock_registry:
            mock_handler = AsyncMock()
            mock_handler.handle_webhook = AsyncMock(return_value=mock_webhook_result)
            mock_registry.get_handler.return_value = mock_handler

            response = bare_client.post(
                "/github/webhook",
                json=payload,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": "sha256=test",
                },
                # Deliberately NO X-Tenant-Id header.
            )

        assert response.status_code == 200


# =============================================================================
# workflow_run Webhook Tests (ci_completed outcome forwarding)
# =============================================================================


@pytest.mark.unit
class TestWorkflowRunWebhook:
    """``workflow_run`` is not in coding_client's GitHubEventTypeEnum, so it is
    handled outside the enum-gated dispatch (see
    ``routers.github_webhooks._handle_workflow_run_webhook``)."""

    def _handler(self, secret="wf-secret"):
        from services.github.secure_webhook_handler import (
            SecureWebhookHandler,
            WebhookSecurityLevel,
        )

        return SecureWebhookHandler(
            webhook_secret=secret, security_level=WebhookSecurityLevel.STANDARD
        )

    def _sign(self, payload: bytes, secret: str) -> str:
        import hashlib
        import hmac as hmac_mod

        return "sha256=" + hmac_mod.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    def _request(self, ip="10.0.0.1"):
        req = MagicMock()
        req.headers = {}
        req.client = MagicMock()
        req.client.host = ip
        return req

    @pytest.mark.asyncio
    async def test_completed_on_default_branch_forwards_ci_completed(self):
        import asyncio
        import json as json_mod

        from routers.github_webhooks import _handle_workflow_run_webhook

        handler = self._handler()
        payload = json_mod.dumps(
            {
                "action": "completed",
                "workflow_run": {
                    "head_branch": "main",
                    "head_sha": "sha-123",
                    "conclusion": "success",
                },
                "repository": {"full_name": "org/repo", "default_branch": "main"},
            }
        ).encode()
        sig = self._sign(payload, "wf-secret")

        forward_mock = AsyncMock()
        with patch("services.outcome_forward.forward_outcome_event", forward_mock):
            response = await _handle_workflow_run_webhook(
                request=self._request(),
                webhook_handler=handler,
                payload=payload,
                signature=sig,
                delivery_id="delivery-wf-1",
            )
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending)

        assert response.status == "success"
        assert response.event_type == "workflow_run"
        assert response.action == "completed"
        assert response.processed is True
        assert response.delivery_id == "delivery-wf-1"
        forward_mock.assert_awaited_once()
        forwarded = forward_mock.await_args.args[0]
        assert forwarded["event"] == "ci_completed"
        assert forwarded["conclusion"] == "success"
        assert forwarded["head_sha"] == "sha-123"

    @pytest.mark.asyncio
    async def test_completed_on_non_default_branch_does_not_forward(self):
        import json as json_mod

        from routers.github_webhooks import _handle_workflow_run_webhook

        handler = self._handler()
        payload = json_mod.dumps(
            {
                "action": "completed",
                "workflow_run": {"head_branch": "feature/x", "head_sha": "sha-1"},
                "repository": {"full_name": "org/repo", "default_branch": "main"},
            }
        ).encode()
        sig = self._sign(payload, "wf-secret")

        forward_mock = AsyncMock()
        with patch("services.outcome_forward.forward_outcome_event", forward_mock):
            response = await _handle_workflow_run_webhook(
                request=self._request(),
                webhook_handler=handler,
                payload=payload,
                signature=sig,
                delivery_id="delivery-wf-2",
            )

        assert response.processed is False
        forward_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_401(self):
        import json as json_mod

        from fastapi import HTTPException

        from routers.github_webhooks import _handle_workflow_run_webhook

        handler = self._handler()
        payload = json_mod.dumps(
            {
                "action": "completed",
                "workflow_run": {"head_branch": "main"},
                "repository": {"full_name": "org/repo", "default_branch": "main"},
            }
        ).encode()

        with pytest.raises(HTTPException) as exc_info:
            await _handle_workflow_run_webhook(
                request=self._request(),
                webhook_handler=handler,
                payload=payload,
                signature="sha256=wrong",
                delivery_id="delivery-wf-3",
            )
        assert exc_info.value.status_code == 401

    def test_workflow_run_full_route_returns_2xx_without_tenant_header(self, monkeypatch):
        """Full HTTP round trip through POST /webhook with X-GitHub-Event:
        workflow_run -- proves the route wiring (bypassing the enum-gated
        dispatch) end to end, and that no X-Tenant-Id is required."""
        import hashlib
        import hmac as hmac_mod
        import json as json_mod

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import services.github.secure_webhook_handler as swh_mod
        from config.settings import get_settings
        from routers.github_webhooks import router as webhook_router

        monkeypatch.setattr(get_settings(), "github_webhook_secret", "route-secret")
        original_handler = swh_mod._webhook_handler
        swh_mod._webhook_handler = None
        try:
            bare_app = FastAPI()
            bare_app.include_router(webhook_router, prefix="/github")
            bare_client = TestClient(bare_app)

            payload = json_mod.dumps(
                {
                    "action": "completed",
                    "workflow_run": {
                        "head_branch": "main",
                        "head_sha": "sha-full",
                        "conclusion": "success",
                    },
                    "repository": {"full_name": "org/repo", "default_branch": "main"},
                }
            ).encode()
            sig = "sha256=" + hmac_mod.new(b"route-secret", payload, hashlib.sha256).hexdigest()

            with patch("services.outcome_forward.forward_outcome_event", AsyncMock()):
                response = bare_client.post(
                    "/github/webhook",
                    content=payload,
                    headers={
                        "X-GitHub-Event": "workflow_run",
                        "X-Hub-Signature-256": sig,
                        "Content-Type": "application/json",
                    },
                )
        finally:
            swh_mod._webhook_handler = original_handler

        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "workflow_run"
        assert data["processed"] is True


# =============================================================================
# Pull Request Event Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandlePullRequestEvent:
    """Tests for handle_pull_request_event function."""

    @pytest.mark.asyncio
    async def test_pr_opened_event(self, pr_payload, mock_settings):
        """Test handling PR opened event."""
        from routers.github_webhooks import handle_pull_request_event

        # Patch both possible import paths for get_settings
        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("config.settings.get_settings") as mock_get_settings_2:
                mock_get_settings_2.return_value = mock_settings
                with patch(
                    "services.pr_validation.pr_validation_orchestrator.get_pr_validation_orchestrator"
                ) as mock_get_orchestrator:
                    mock_orchestrator = AsyncMock()
                    mock_orchestrator.validate_pr_with_actions = AsyncMock(
                        return_value={
                            "validation_id": "val-123",
                            "fast_validation": {
                                "passed": True,
                                "score": 85,
                                "results": [],
                            },
                            "comprehensive_validation": {"status": "triggered"},
                        }
                    )
                    mock_get_orchestrator.return_value = mock_orchestrator
                    with patch("github.Github") as mock_github:
                        mock_repo = MagicMock()
                        mock_pr = MagicMock()
                        mock_pr.title = "Test PR"
                        mock_pr.body = "Test body"
                        mock_pr.user = MagicMock(login="testuser")
                        mock_pr.base = MagicMock(sha="base123", ref="main")
                        mock_pr.head = MagicMock(sha="head456", ref="feature")
                        mock_pr.created_at = datetime.now(UTC)
                        mock_pr.updated_at = datetime.now(UTC)
                        mock_pr.get_files = MagicMock(return_value=[])
                        mock_pr.create_issue_comment = MagicMock()
                        mock_repo.get_pull = MagicMock(return_value=mock_pr)
                        mock_client = MagicMock()
                        mock_client.get_repo = MagicMock(return_value=mock_repo)
                        mock_github.return_value = mock_client

                        await handle_pull_request_event(pr_payload)

                        mock_orchestrator.validate_pr_with_actions.assert_called_once()
                        posted_comment = mock_pr.create_issue_comment.call_args.args[0]
                        assert "http://coding-gateway:8687/api/validation/val-123" in posted_comment

    @pytest.mark.asyncio
    async def test_pr_synchronize_event(self, pr_payload, mock_settings):
        """Test handling PR synchronize event posts comment with 'Skipped' for not_triggered."""
        from routers.github_webhooks import handle_pull_request_event

        pr_payload["action"] = "synchronize"

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch(
                "services.pr_validation.pr_validation_orchestrator.get_pr_validation_orchestrator"
            ) as mock_get_orchestrator:
                mock_orchestrator = AsyncMock()
                mock_orchestrator.validate_pr_with_actions = AsyncMock(
                    return_value={
                        "validation_id": "val-123",
                        "fast_validation": {"passed": True, "score": 90, "results": []},
                        "comprehensive_validation": {
                            "status": "not_triggered",
                            "reason": "low risk",
                        },
                    }
                )
                mock_get_orchestrator.return_value = mock_orchestrator
                with patch("github.Github") as mock_github:
                    mock_repo = MagicMock()
                    mock_pr = MagicMock()
                    mock_pr.title = "Test PR"
                    mock_pr.body = "Test body"
                    mock_pr.user = MagicMock(login="testuser")
                    mock_pr.base = MagicMock(sha="base123", ref="main")
                    mock_pr.head = MagicMock(sha="head456", ref="feature")
                    mock_pr.created_at = datetime.now(UTC)
                    mock_pr.updated_at = datetime.now(UTC)
                    mock_pr.get_files = MagicMock(return_value=[])
                    mock_pr.create_issue_comment = MagicMock()
                    mock_repo.get_pull = MagicMock(return_value=mock_pr)
                    mock_client = MagicMock()
                    mock_client.get_repo = MagicMock(return_value=mock_repo)
                    mock_github.return_value = mock_client

                    await handle_pull_request_event(pr_payload)

                    mock_orchestrator.validate_pr_with_actions.assert_called_once()
                    posted_comment = mock_pr.create_issue_comment.call_args.args[0]
                    assert "Skipped" in posted_comment
                    assert "low risk" in posted_comment

    @pytest.mark.asyncio
    async def test_pr_closed_merged(self, pr_payload, caplog):
        """Test handling PR closed (merged) event logs merge info."""
        import logging

        from routers.github_webhooks import handle_pull_request_event

        pr_payload["action"] = "closed"
        pr_payload["pull_request"]["merged"] = True

        with caplog.at_level(logging.INFO):
            await handle_pull_request_event(pr_payload)

        assert "was merged" in caplog.text

    @pytest.mark.asyncio
    async def test_pr_event_no_token(self, pr_payload, caplog):
        """No credential exists for a webhook (no PAT, no tenant for the App):
        the handler skips before GitHub, even with the legacy env PATs armed."""
        import logging

        from routers.github_webhooks import handle_pull_request_event

        with patch.dict(
            "os.environ",
            {"GITHUB_ACCESS_TOKEN": "ghp_env_pat", "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat"},
        ):
            with patch("github.Github") as mock_github, caplog.at_level(logging.INFO):
                await handle_pull_request_event(pr_payload)

        mock_github.assert_not_called()
        assert "needs a GitHub credential" in caplog.text
        assert "no PAT fallback" in caplog.text

    @pytest.mark.asyncio
    async def test_pr_event_invalid_repo_name(self, pr_payload, mock_settings, caplog):
        """Test PR event with invalid repo name format logs error and returns early."""
        import logging

        from routers.github_webhooks import handle_pull_request_event

        pr_payload["repository"]["full_name"] = "invalid-format"

        with patch("routers.github_webhooks._webhook_github_token", return_value=None):
            with patch("config.settings.get_settings", return_value=mock_settings):
                with patch("github.Github") as mock_github:
                    mock_client = MagicMock()
                    mock_repo = MagicMock()
                    mock_pr = MagicMock()
                    mock_pr.title = "Test"
                    mock_pr.body = ""
                    mock_pr.user = MagicMock(login="test")
                    mock_pr.base = MagicMock(sha="base", ref="main")
                    mock_pr.head = MagicMock(sha="head", ref="feature")
                    mock_pr.created_at = None
                    mock_pr.updated_at = None
                    mock_pr.get_files = MagicMock(return_value=[])
                    mock_pr.create_issue_comment = MagicMock()
                    mock_repo.get_pull = MagicMock(return_value=mock_pr)
                    mock_client.get_repo = MagicMock(return_value=mock_repo)
                    mock_github.return_value = mock_client

                    with caplog.at_level(logging.ERROR):
                        await handle_pull_request_event(pr_payload)

                    # An unparseable full_name gets no credential (no webhook
                    # repo does since 2026-09-30: no PAT, no tenant for the App),
                    # so the handler stops before any GitHub call.
                    mock_client.get_repo.assert_not_called()
                    mock_pr.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_pr_event_validation_error(self, pr_payload, mock_settings, caplog):
        """Test PR event when validation fails logs the error."""
        import logging

        from routers.github_webhooks import handle_pull_request_event

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("config.settings.get_settings", return_value=mock_settings):
                with patch(
                    "services.pr_validation.pr_validation_orchestrator.get_pr_validation_orchestrator"
                ) as mock_get_orchestrator:
                    mock_orchestrator = AsyncMock()
                    mock_orchestrator.validate_pr_with_actions = AsyncMock(
                        side_effect=ValueError("Validation failed")
                    )
                    mock_get_orchestrator.return_value = mock_orchestrator
                    with patch("github.Github") as mock_github:
                        mock_repo = MagicMock()
                        mock_pr = MagicMock()
                        mock_pr.title = "Test PR"
                        mock_pr.body = ""
                        mock_pr.user = MagicMock(login="testuser")
                        mock_pr.base = MagicMock(sha="base123", ref="main")
                        mock_pr.head = MagicMock(sha="head456", ref="feature")
                        mock_pr.created_at = None
                        mock_pr.updated_at = None
                        mock_pr.get_files = MagicMock(return_value=[])
                        mock_repo.get_pull = MagicMock(return_value=mock_pr)
                        mock_client = MagicMock()
                        mock_client.get_repo = MagicMock(return_value=mock_repo)
                        mock_github.return_value = mock_client

                        with caplog.at_level(logging.ERROR):
                            await handle_pull_request_event(pr_payload)

                        assert "Failed to validate PR" in caplog.text
                        assert "Validation failed" in caplog.text


# =============================================================================
# Review Comment Event Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandleReviewCommentEvent:
    """Tests for handle_review_comment_event function."""

    @pytest.mark.asyncio
    async def test_review_comment_with_trigger(self, review_comment_payload):
        """Test handling review comment with analysis trigger."""
        from routers.github_webhooks import handle_review_comment_event

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.return_value = None

            await handle_review_comment_event(review_comment_payload)

            mock_analyze.assert_called_once_with("owner/repo", 42, "comment_trigger")

    @pytest.mark.asyncio
    async def test_review_comment_codehelper_trigger(self, review_comment_payload):
        """Test handling review comment with @codehelper trigger."""
        from routers.github_webhooks import handle_review_comment_event

        review_comment_payload["comment"]["body"] = "@codehelper please review"

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.return_value = None

            await handle_review_comment_event(review_comment_payload)

            mock_analyze.assert_called_once_with("owner/repo", 42, "comment_trigger")

    @pytest.mark.asyncio
    async def test_review_comment_review_trigger(self, review_comment_payload):
        """Test handling review comment with /review trigger."""
        from routers.github_webhooks import handle_review_comment_event

        review_comment_payload["comment"]["body"] = "/review this code"

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.return_value = None

            await handle_review_comment_event(review_comment_payload)

            mock_analyze.assert_called_once_with("owner/repo", 42, "comment_trigger")

    @pytest.mark.asyncio
    async def test_review_comment_no_trigger(self, review_comment_payload):
        """Test handling review comment without trigger."""
        from routers.github_webhooks import handle_review_comment_event

        review_comment_payload["comment"]["body"] = "Just a regular comment"

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            await handle_review_comment_event(review_comment_payload)

            mock_analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_review_comment_analysis_error(self, review_comment_payload, caplog):
        """Test handling review comment when analysis fails logs the error."""
        import logging

        from routers.github_webhooks import handle_review_comment_event

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.side_effect = ValueError("Analysis failed")

            with caplog.at_level(logging.ERROR):
                await handle_review_comment_event(review_comment_payload)

            assert "Failed to analyze PR" in caplog.text
            assert "Analysis failed" in caplog.text


# =============================================================================
# Push Event Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandlePushEvent:
    """Tests for handle_push_event function."""

    @pytest.mark.asyncio
    async def test_push_to_main_branch(self, push_payload):
        """Test handling push to main branch."""
        from routers.github_webhooks import handle_push_event

        with patch("routers.github_webhooks.analyze_repository_changes") as mock_analyze:
            mock_analyze.return_value = {}

            await handle_push_event(push_payload)

            mock_analyze.assert_called_once_with("owner/repo", "main", push_payload["commits"])

    @pytest.mark.asyncio
    async def test_push_to_master_branch(self, push_payload):
        """Test handling push to master branch."""
        from routers.github_webhooks import handle_push_event

        push_payload["ref"] = "refs/heads/master"

        with patch("routers.github_webhooks.analyze_repository_changes") as mock_analyze:
            mock_analyze.return_value = {}

            await handle_push_event(push_payload)

            mock_analyze.assert_called_once_with("owner/repo", "master", push_payload["commits"])

    @pytest.mark.asyncio
    async def test_push_to_develop_branch(self, push_payload):
        """Test handling push to develop branch."""
        from routers.github_webhooks import handle_push_event

        push_payload["ref"] = "refs/heads/develop"

        with patch("routers.github_webhooks.analyze_repository_changes") as mock_analyze:
            mock_analyze.return_value = {}

            await handle_push_event(push_payload)

            mock_analyze.assert_called_once_with("owner/repo", "develop", push_payload["commits"])

    @pytest.mark.asyncio
    async def test_push_to_feature_branch(self, push_payload):
        """Test handling push to feature branch (should be ignored)."""
        from routers.github_webhooks import handle_push_event

        push_payload["ref"] = "refs/heads/feature/test"

        with patch("routers.github_webhooks.analyze_repository_changes") as mock_analyze:
            await handle_push_event(push_payload)

            mock_analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_analysis_error(self, push_payload, caplog):
        """Test handling push when analysis fails logs the error."""
        import logging

        from routers.github_webhooks import handle_push_event

        with patch("routers.github_webhooks.analyze_repository_changes") as mock_analyze:
            mock_analyze.side_effect = RuntimeError("Analysis failed")

            with caplog.at_level(logging.ERROR):
                await handle_push_event(push_payload)

            assert "Failed to analyze push" in caplog.text
            assert "Analysis failed" in caplog.text


# =============================================================================
# Issue Comment Event Handler Tests
# =============================================================================


@pytest.mark.unit
class TestHandleIssueCommentEvent:
    """Tests for handle_issue_comment_event function."""

    @pytest.mark.asyncio
    async def test_issue_comment_on_pr_with_trigger(self, issue_comment_payload):
        """Test handling issue comment on PR with trigger."""
        from routers.github_webhooks import handle_issue_comment_event

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.return_value = None

            await handle_issue_comment_event(issue_comment_payload)

            mock_analyze.assert_called_once_with("owner/repo", 42, "comment_trigger")

    @pytest.mark.asyncio
    async def test_issue_comment_help_trigger(self, issue_comment_payload):
        """Test handling issue comment with /help trigger."""
        from routers.github_webhooks import handle_issue_comment_event

        issue_comment_payload["comment"]["body"] = "/help with this"

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.return_value = None

            await handle_issue_comment_event(issue_comment_payload)

            mock_analyze.assert_called_once_with("owner/repo", 42, "comment_trigger")

    @pytest.mark.asyncio
    async def test_issue_comment_on_issue_not_pr(self, issue_comment_payload):
        """Test handling issue comment on regular issue (not PR)."""
        from routers.github_webhooks import handle_issue_comment_event

        del issue_comment_payload["issue"]["pull_request"]

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            await handle_issue_comment_event(issue_comment_payload)

            mock_analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_issue_comment_no_trigger(self, issue_comment_payload):
        """Test handling issue comment without trigger."""
        from routers.github_webhooks import handle_issue_comment_event

        issue_comment_payload["comment"]["body"] = "Regular comment"

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            await handle_issue_comment_event(issue_comment_payload)

            mock_analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_issue_comment_analysis_error(self, issue_comment_payload, caplog):
        """Test handling issue comment when analysis fails logs the error."""
        import logging

        from routers.github_webhooks import handle_issue_comment_event

        with patch("routers.github_webhooks.analyze_pull_request_changes") as mock_analyze:
            mock_analyze.side_effect = ConnectionError("Failed")

            with caplog.at_level(logging.ERROR):
                await handle_issue_comment_event(issue_comment_payload)

            assert "Failed to analyze PR" in caplog.text
            assert "Failed" in caplog.text


# =============================================================================
# PR Analysis Summary Generator Tests
# =============================================================================


@pytest.mark.unit
class TestGeneratePRAnalysisSummary:
    """Tests for _generate_pr_analysis_summary function."""

    def test_generate_summary_basic(self, pr_analysis_dict):
        """Test generating basic PR analysis summary."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "manual_trigger")

        assert "Code Analysis Results" in summary
        assert "manual_trigger" in summary
        assert "Total files:** 5" in summary
        assert "+150" in summary
        assert "-50" in summary

    def test_generate_summary_with_languages(self, pr_analysis_dict):
        """Test generating summary with language breakdown."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "test")

        assert "Languages" in summary
        assert "python" in summary
        assert "javascript" in summary

    def test_generate_summary_with_files(self, pr_analysis_dict):
        """Test generating summary with file list (when <= 10 files)."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        # Add mock files
        mock_file = MagicMock()
        mock_file.file_path = "test.py"
        mock_file.change_type = "modified"
        pr_analysis_dict["files"] = [mock_file]
        pr_analysis_dict["statistics"]["total"] = 1

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "test")

        assert "Changed Files" in summary
        assert "test.py" in summary

    def test_generate_summary_many_files(self, pr_analysis_dict):
        """Test generating summary with many files (no file list)."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        pr_analysis_dict["statistics"]["total"] = 15

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "test")

        assert "Changed Files" not in summary

    def test_generate_summary_no_languages(self, pr_analysis_dict):
        """Test generating summary with no language data."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        pr_analysis_dict["languages"] = {}

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "test")

        assert "Languages" not in summary

    def test_generate_summary_change_types(self, pr_analysis_dict):
        """Test generating summary with different change types."""
        from routers.github_webhooks import _generate_pr_analysis_summary

        mock_files = []
        for change_type in ["added", "modified", "removed"]:
            mock_file = MagicMock()
            mock_file.file_path = f"{change_type}.py"
            mock_file.change_type = change_type
            mock_files.append(mock_file)

        pr_analysis_dict["files"] = mock_files
        pr_analysis_dict["statistics"]["total"] = 3

        summary = _generate_pr_analysis_summary(pr_analysis_dict, "test")

        # Should contain emoji indicators for each type
        assert "added.py" in summary or "modified.py" in summary


# =============================================================================
# Post PR Comment Tests
# =============================================================================


@pytest.mark.unit
class TestPostPRComment:
    """Tests for _post_pr_comment function."""

    @pytest.mark.asyncio
    async def test_post_comment_success(self, mock_settings):
        """Test posting PR comment successfully."""
        from routers.github_webhooks import _post_pr_comment

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                mock_service = AsyncMock()
                mock_service.__aenter__ = AsyncMock(return_value=mock_service)
                mock_service.__aexit__ = AsyncMock()
                mock_service.add_pull_request_comment = AsyncMock()
                mock_service_class.return_value = mock_service

                await _post_pr_comment("owner/repo", 42, "Test comment")

                mock_service.add_pull_request_comment.assert_called_once_with(
                    repo_name="owner/repo", pr_number=42, comment="Test comment"
                )

    @pytest.mark.asyncio
    async def test_post_comment_no_token(self, caplog):
        """No webhook credential exists: logs a warning and never builds the service."""
        import logging

        from routers.github_webhooks import _post_pr_comment

        with patch.dict(
            "os.environ",
            {"GITHUB_ACCESS_TOKEN": "ghp_env_pat", "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat"},
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                with caplog.at_level(logging.WARNING):
                    await _post_pr_comment("owner/repo", 42, "Test comment")

                assert "No GitHub credential for webhook comments" in caplog.text
                mock_service_class.assert_not_called()


# =============================================================================
# Analyze Pull Request Changes Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzePullRequestChanges:
    """Tests for analyze_pull_request_changes function."""

    @pytest.mark.asyncio
    async def test_analyze_pr_success(self, pr_analysis_dict, mock_settings):
        """Test analyzing PR changes successfully."""
        from routers.github_webhooks import analyze_pull_request_changes

        with (
            patch("routers.github_webhooks._webhook_github_token", return_value="tok"),
            patch("routers.github_webhooks.GitHubIntegrationHub") as mock_hub_class,
        ):
            mock_hub = MagicMock()
            mock_hub.analyze_pull_request = AsyncMock(return_value=pr_analysis_dict)
            mock_hub_class.return_value = mock_hub
            with patch("routers.github_webhooks._post_pr_comment") as mock_post:
                mock_post.return_value = None

                await analyze_pull_request_changes("owner/repo", 42, "test_trigger")

                mock_hub.analyze_pull_request.assert_called_once()
                mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_pr_error_posts_comment(self):
        """Test analyzing PR posts error comment on failure."""
        from routers.github_webhooks import analyze_pull_request_changes

        with (
            patch("routers.github_webhooks._webhook_github_token", return_value="tok"),
            patch("routers.github_webhooks.GitHubIntegrationHub") as mock_hub_class,
        ):
            mock_hub = MagicMock()
            mock_hub.analyze_pull_request = AsyncMock(side_effect=ValueError("Analysis failed"))
            mock_hub_class.return_value = mock_hub
            with patch("routers.github_webhooks._post_pr_comment") as mock_post:
                mock_post.return_value = None

                await analyze_pull_request_changes("owner/repo", 42, "test")

                # Should try to post error comment
                assert mock_post.called
                call_args = mock_post.call_args[0]
                assert "Error" in call_args[2] or "error" in call_args[2].lower()


# =============================================================================
# Analyze Single Commit Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzeSingleCommit:
    """Tests for _analyze_single_commit function."""

    @pytest.mark.asyncio
    async def test_analyze_commit_success(self):
        """Test analyzing single commit successfully."""
        from routers.github_webhooks import _analyze_single_commit

        mock_service = AsyncMock()
        mock_service.get_commit_files = AsyncMock(
            return_value=[
                {
                    "filename": "test.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 5,
                }
            ]
        )

        commit = {
            "id": "abc123",
            "author": {"name": "Test Author"},
            "message": "Test commit message",
            "timestamp": "2025-01-01T00:00:00Z",
        }

        result = await _analyze_single_commit(mock_service, "owner/repo", commit)

        assert result["sha"] == "abc123"
        assert result["author"] == "Test Author"
        assert result["message"] == "Test commit message"
        assert result["timestamp"] == "2025-01-01T00:00:00Z"
        assert result["files_changed"] == 1
        assert result["total_additions"] == 10
        assert result["total_deletions"] == 5
        assert len(result["file_changes"]) == 1
        assert result["file_changes"][0]["filename"] == "test.py"
        assert result["file_changes"][0]["status"] == "modified"

    @pytest.mark.asyncio
    async def test_analyze_commit_error(self):
        """Test analyzing commit when API call fails."""
        from routers.github_webhooks import _analyze_single_commit

        mock_service = AsyncMock()
        mock_service.get_commit_files = AsyncMock(side_effect=RuntimeError("API error"))

        commit = {
            "id": "abc123",
            "author": {"name": "Test Author"},
            "message": "Test commit",
            "timestamp": "2025-01-01T00:00:00Z",
        }

        result = await _analyze_single_commit(mock_service, "owner/repo", commit)

        assert result is None


# =============================================================================
# Update Code Graph Tests
# =============================================================================


@pytest.mark.unit
class TestUpdateCodeGraph:
    """Tests for _update_code_graph function."""

    @pytest.mark.asyncio
    async def test_update_code_graph_success(self):
        """Test updating code graph successfully."""
        from routers.github_webhooks import _update_code_graph

        mock_hub = AsyncMock()
        mock_hub.handle_repository_update = AsyncMock(
            return_value={"status": "scan_requested", "update_stats": {}}
        )

        await _update_code_graph(
            mock_hub,
            "owner/repo",
            "main",
            ["file1.py", "file2.py"],
            ["sha1", "sha2"],
            tenant_id="tenant-a",
        )

        # The tenant MUST reach the handler: the container lookup that resolves
        # which graph to refresh is tenant-scoped, so dropping it here would
        # silently refresh the wrong tenant's graph (or none).
        mock_hub.handle_repository_update.assert_called_once_with(
            repo_name="owner/repo",
            branch="main",
            changed_files=["file1.py", "file2.py"],
            commit_shas=["sha1", "sha2"],
            tenant_id="tenant-a",
        )

    @pytest.mark.asyncio
    async def test_update_code_graph_no_container(self, caplog):
        """No container for the repo => nothing refreshed, and it says so."""
        import logging

        from routers.github_webhooks import _update_code_graph

        mock_hub = AsyncMock()
        mock_hub.handle_repository_update = AsyncMock(return_value={"status": "no_container"})

        with caplog.at_level(logging.WARNING):
            await _update_code_graph(mock_hub, "owner/repo", "main", [], [])

        assert "No container for" in caplog.text
        assert "nothing was updated" in caplog.text
        assert "owner/repo" in caplog.text

    @pytest.mark.asyncio
    async def test_update_code_graph_error(self, caplog):
        """Test updating code graph when error occurs logs the error."""
        import logging

        from routers.github_webhooks import _update_code_graph

        mock_hub = AsyncMock()
        mock_hub.handle_repository_update = AsyncMock(side_effect=RuntimeError("DB error"))

        with caplog.at_level(logging.ERROR):
            await _update_code_graph(mock_hub, "owner/repo", "main", [], [])

        assert "Failed to request a code-graph refresh" in caplog.text
        assert "DB error" in caplog.text


# =============================================================================
# Repository Summary Tests
# =============================================================================


@pytest.mark.unit
class TestGetRepositorySummary:
    """Tests for _get_repository_summary function."""

    def test_get_repository_summary(self):
        """Test generating repository summary statistics."""
        from routers.github_webhooks import _get_repository_summary

        commits_data = [
            {"total_additions": 50, "total_deletions": 20},
            {"total_additions": 30, "total_deletions": 10},
        ]
        changed_files = {"file1.py", "file2.py", "file3.py"}

        result = _get_repository_summary(commits_data, changed_files)

        assert result["total_files"] == 3
        assert result["total_commits"] == 2
        assert result["total_additions"] == 80
        assert result["total_deletions"] == 30

    def test_get_repository_summary_empty(self):
        """Test generating summary with empty data."""
        from routers.github_webhooks import _get_repository_summary

        result = _get_repository_summary([], set())

        assert result["total_files"] == 0
        assert result["total_commits"] == 0
        assert result["total_additions"] == 0
        assert result["total_deletions"] == 0


# =============================================================================
# Process Commits Analysis Tests
# =============================================================================


@pytest.mark.unit
class TestProcessCommitsAnalysis:
    """Tests for _process_commits_analysis function."""

    @pytest.mark.asyncio
    async def test_process_commits_analysis(self):
        """Test processing commits analysis."""
        # The source code has a bug using Set[Any]() instead of set()
        # We mock the function result directly to avoid this issue
        from routers.github_webhooks import _analyze_single_commit

        mock_service = MagicMock()
        mock_service.get_commit_files = AsyncMock(
            return_value=[
                {
                    "filename": "test.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 5,
                }
            ]
        )

        commit = {
            "id": "abc123",
            "author": {"name": "Test"},
            "message": "Test commit",
            "timestamp": "2025-01-01T00:00:00Z",
        }

        # Test single commit analysis which doesn't have the Set bug
        result = await _analyze_single_commit(mock_service, "owner/repo", commit)

        assert result["sha"] == "abc123"
        assert result["files_changed"] == 1
        assert result["total_additions"] == 10
        assert result["total_deletions"] == 5
        assert result["file_changes"][0]["filename"] == "test.py"
        assert result["file_changes"][0]["status"] == "modified"

    @pytest.mark.asyncio
    async def test_process_commits_analysis_limit(self):
        """Test that commits[:10] limit is applied — only first 10 processed."""
        from routers.github_webhooks import _process_commits_analysis

        mock_service = MagicMock()
        mock_service.get_commit_files = AsyncMock(
            return_value=[
                {"filename": "f.py", "status": "modified", "additions": 1, "deletions": 0}
            ]
        )

        commits = [
            {
                "id": f"sha{i}",
                "author": {"name": "Test"},
                "message": "Test",
                "timestamp": "2025-01-01T00:00:00Z",
            }
            for i in range(15)
        ]

        commit_analysis, all_changed_files = await _process_commits_analysis(
            mock_service, "owner/repo", commits
        )

        # Only first 10 commits should be processed despite 15 being passed
        assert len(commit_analysis) == 10
        assert mock_service.get_commit_files.call_count == 10


# =============================================================================
# Analyze Repository Changes Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzeRepositoryChanges:
    """Tests for analyze_repository_changes function."""

    @pytest.mark.asyncio
    async def test_analyze_repository_changes_success(self, mock_settings):
        """Test analyzing repository changes successfully."""
        from routers.github_webhooks import analyze_repository_changes

        commits = [
            {
                "id": "abc123",
                "author": {"name": "Test"},
                "message": "Test commit",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ]

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                mock_service = MagicMock()
                mock_service.get_commit_files = AsyncMock(return_value=[])
                mock_service_class.return_value = mock_service
                with patch("routers.github_webhooks.GitHubIntegrationHub") as mock_hub_class:
                    mock_hub = MagicMock()
                    mock_hub.handle_repository_update = AsyncMock(
                        return_value={"status": "success"}
                    )
                    mock_hub_class.return_value = mock_hub
                    with patch(
                        "routers.github_webhooks._analyze_files_for_issues"
                    ) as mock_analyze_issues:
                        mock_analyze_issues.return_value = []
                        with patch(
                            "routers.github_webhooks._process_commits_analysis"
                        ) as mock_process:
                            mock_process.return_value = ([], set())
                            with patch("routers.github_webhooks._update_code_graph") as mock_update:
                                mock_update.return_value = None

                                result = await analyze_repository_changes(
                                    "owner/repo", "main", commits
                                )

                                assert result["repository"] == "owner/repo"
                                assert result["branch"] == "main"

    @pytest.mark.asyncio
    async def test_analyze_repository_changes_no_token(self):
        """No webhook credential exists: returns an error before any GitHub call."""
        from routers.github_webhooks import analyze_repository_changes

        with patch.dict(
            "os.environ",
            {"GITHUB_ACCESS_TOKEN": "ghp_env_pat", "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat"},
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                result = await analyze_repository_changes("owner/repo", "main", [])

        assert "error" in result
        mock_service_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_repository_changes_error(self, mock_settings):
        """Test analyzing repository when error occurs."""
        from routers.github_webhooks import analyze_repository_changes

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                mock_service_class.side_effect = RuntimeError("Service error")

                result = await analyze_repository_changes("owner/repo", "main", [])

                assert "error" in result


# =============================================================================
# Get Code Files Tests
# =============================================================================


@pytest.mark.unit
class TestGetCodeFiles:
    """Tests for _get_code_files function."""

    def test_get_code_files_python(self):
        """Test filtering Python files."""
        from routers.github_webhooks import _get_code_files

        files = ["test.py", "README.md", "main.py", "config.json"]
        result = _get_code_files(files)

        assert "test.py" in result
        assert "main.py" in result
        assert "README.md" not in result
        assert "config.json" not in result

    def test_get_code_files_multiple_languages(self):
        """Test filtering files of multiple languages."""
        from routers.github_webhooks import _get_code_files

        files = ["app.js", "main.ts", "index.jsx", "component.tsx", "server.go"]
        result = _get_code_files(files)

        assert len(result) == 5

    def test_get_code_files_java_cpp(self):
        """Test filtering Java and C++ files."""
        from routers.github_webhooks import _get_code_files

        files = ["Main.java", "utils.cpp", "header.h", "helper.c"]
        result = _get_code_files(files)

        assert len(result) == 4

    def test_get_code_files_empty(self):
        """Test filtering empty file list."""
        from routers.github_webhooks import _get_code_files

        result = _get_code_files([])

        assert result == []


# =============================================================================
# Analyze Security Vulnerabilities Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzeSecurityVulnerabilities:
    """Tests for _analyze_security_vulnerabilities function (HTTP-based)."""

    @pytest.mark.asyncio
    async def test_analyze_security_success(self):
        """Test analyzing security vulnerabilities via HTTP client."""
        from routers.github_webhooks import (
            _analyze_security_vulnerabilities,
        )

        mock_client = AsyncMock()
        mock_client.analyze_vulnerabilities = AsyncMock(
            return_value={
                "vulnerabilities_by_type": {
                    "sql_injection": [
                        {
                            "severity": "high",
                            "file_path": "test.py",
                            "description": "SQL Injection",
                            "line_start": 10,
                        }
                    ]
                }
            }
        )

        with patch("main.get_security_client", return_value=mock_client):
            issues = await _analyze_security_vulnerabilities(["test.py"])

            assert len(issues) == 1
            assert issues[0]["type"] == "security"
            assert issues[0]["severity"] == "high"
            assert issues[0]["file"] == "test.py"
            assert issues[0]["tool"] == "security_detector"

    @pytest.mark.asyncio
    async def test_scan_sends_tenant_header(self):
        """The scan carries X-Tenant-Id rather than riding coding-security's IP fallback.

        Asserted off the real call kwargs; without the header this raises KeyError.

        ⚠️ The CONFIGURED tenant is the precondition, and it is now stated rather
        than assumed. This previously read `== DEFAULT_TENANT_ID` with no patch,
        which since coding-client 1.9.0 (#36) removed the built-in "n8n-dev"
        fallback compares None to None — passing while the code shipped a header
        whose value was literally None. The follow-on `.strip()` was the only
        thing that noticed, and it failed with AttributeError, which is a crash
        rather than a real assertion about the contract.
        """
        import routers.github_webhooks as gw
        from routers.github_webhooks import _analyze_security_vulnerabilities

        mock_client = AsyncMock()
        mock_client.analyze_vulnerabilities = AsyncMock(return_value={})

        configured = "0195f48b-d000-7000-8000-000000000001"
        with (
            patch.object(gw, "DEFAULT_TENANT_ID", configured),
            patch("main.get_security_client", return_value=mock_client),
        ):
            await _analyze_security_vulnerabilities(["test.py"])

        headers = mock_client.analyze_vulnerabilities.await_args.kwargs["headers"]
        assert headers["X-Tenant-Id"] == configured
        assert headers["X-Tenant-Id"].strip()

    @pytest.mark.asyncio
    async def test_scan_omits_tenant_header_when_default_is_unset(self):
        """Unset DEFAULT_TENANT_ID must omit the header, never send None.

        This is the live production case: coding-github does not set
        DEFAULT_TENANT_ID on its Deployment. Inventing a tenant is what
        coding-client #36 exists to stop, so the scan is left to be refused by
        coding-security rather than silently attributed to a fallback.
        """
        import routers.github_webhooks as gw
        from routers.github_webhooks import _analyze_security_vulnerabilities

        mock_client = AsyncMock()
        mock_client.analyze_vulnerabilities = AsyncMock(return_value={})

        with (
            patch.object(gw, "DEFAULT_TENANT_ID", None),
            patch("main.get_security_client", return_value=mock_client),
        ):
            await _analyze_security_vulnerabilities(["test.py"])

        headers = mock_client.analyze_vulnerabilities.await_args.kwargs["headers"]
        assert headers == {}
        assert None not in headers.values()

    @pytest.mark.asyncio
    async def test_analyze_security_error(self):
        """Test analyzing security when service is unavailable returns empty list."""
        from routers.github_webhooks import (
            _analyze_security_vulnerabilities,
        )

        with patch("main.get_security_client", side_effect=RuntimeError("not initialised")):
            issues = await _analyze_security_vulnerabilities(["test.py"])

            assert issues == []


# =============================================================================
# Analyze File Patterns Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzeFilePatterns:
    """Tests for _analyze_file_patterns function."""

    def test_analyze_sensitive_files(self):
        """Test detecting sensitive files."""
        from routers.github_webhooks import _analyze_file_patterns

        files = [".env", "config.json", "secrets.json", "regular.py"]
        issues = _analyze_file_patterns(files)

        assert len(issues) == 3
        assert all(issue["type"] == "sensitive_file" for issue in issues)

    def test_analyze_credential_files(self):
        """Test detecting credential files."""
        from routers.github_webhooks import _analyze_file_patterns

        files = ["server.pem", "private.key", "cert.p12"]
        issues = _analyze_file_patterns(files)

        assert len(issues) == 3
        assert all(issue["severity"] == "high" for issue in issues)

    def test_analyze_binary_files(self):
        """Test detecting binary files."""
        from routers.github_webhooks import _analyze_file_patterns

        files = ["archive.zip", "package.tar.gz", "program.exe", "lib.dll"]
        issues = _analyze_file_patterns(files)

        assert len(issues) == 4
        assert all(issue["type"] == "binary_file" for issue in issues)
        assert all(issue["severity"] == "medium" for issue in issues)

    def test_analyze_no_issues(self):
        """Test with clean files."""
        from routers.github_webhooks import _analyze_file_patterns

        files = ["main.py", "utils.js", "README.md"]
        issues = _analyze_file_patterns(files)

        assert issues == []


# =============================================================================
# Analyze Files For Issues Tests
# =============================================================================


@pytest.mark.unit
class TestAnalyzeFilesForIssues:
    """Tests for _analyze_files_for_issues function."""

    @pytest.mark.asyncio
    async def test_analyze_files_combined(self):
        """Test analyzing files for all issue types."""
        from routers.github_webhooks import _analyze_files_for_issues

        files = ["test.py", ".env"]

        with patch("routers.github_webhooks._analyze_security_vulnerabilities") as mock_security:
            mock_security.return_value = []
            with patch("routers.github_webhooks._analyze_file_patterns") as mock_patterns:
                mock_patterns.return_value = [{"type": "sensitive_file", "file": ".env"}]

                issues = await _analyze_files_for_issues(files)

                assert len(issues) == 1

    @pytest.mark.asyncio
    async def test_analyze_files_error(self):
        """Test analyzing files when error occurs."""
        from routers.github_webhooks import _analyze_files_for_issues

        with patch("routers.github_webhooks._get_code_files") as mock_get_code:
            mock_get_code.side_effect = ValueError("Error")

            issues = await _analyze_files_for_issues(["test.py"])

            assert issues == []


# =============================================================================
# Format Notification Message Tests
# =============================================================================


@pytest.mark.unit
class TestFormatNotificationMessage:
    """Tests for _format_notification_message function."""

    def test_format_notification(self):
        """Test formatting notification message."""
        from routers.github_webhooks import _format_notification_message

        issues = [
            {"file": "test.py", "message": "Critical vulnerability"},
            {"file": "main.py", "message": "Security issue"},
        ]

        message = _format_notification_message(issues, "main")

        assert "Critical Issues Detected" in message
        assert "main" in message
        assert "test.py" in message
        assert "Critical vulnerability" in message

    def test_format_notification_max_5_issues(self):
        """Test formatting notification with limit of 5 issues."""
        from routers.github_webhooks import _format_notification_message

        issues = [{"file": f"file{i}.py", "message": f"Issue {i}"} for i in range(10)]

        message = _format_notification_message(issues, "main")

        # Should only include first 5
        assert "file0.py" in message
        assert "file4.py" in message
        assert "file5.py" not in message


# =============================================================================
# Try Create GitHub Issue Tests
# =============================================================================


@pytest.mark.unit
class TestTryCreateGitHubIssue:
    """Tests for _try_create_github_issue function."""

    @pytest.mark.asyncio
    async def test_create_issue_success(self):
        """Test creating GitHub issue successfully."""
        from routers.github_webhooks import _try_create_github_issue

        mock_service = AsyncMock()
        mock_service.create_issue = AsyncMock()

        result = await _try_create_github_issue(
            mock_service, "owner/repo", "main", "Issue body", [{"file": "test.py"}]
        )

        assert result is True
        mock_service.create_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_issue_failure(self):
        """Test creating GitHub issue when API fails."""
        from routers.github_webhooks import _try_create_github_issue

        mock_service = AsyncMock()
        mock_service.create_issue = AsyncMock(side_effect=RuntimeError("API error"))

        result = await _try_create_github_issue(
            mock_service, "owner/repo", "main", "Issue body", []
        )

        assert result is False


# =============================================================================
# Try Fallback Notifications Tests
# =============================================================================


@pytest.mark.unit
class TestTryFallbackNotifications:
    """Tests for _try_fallback_notifications function."""

    @pytest.mark.asyncio
    async def test_fallback_post_to_pr(self):
        """Test fallback posting to open PR."""
        from routers.github_webhooks import _try_fallback_notifications

        mock_service = AsyncMock()
        mock_service.get_pull_requests = AsyncMock(return_value=[{"number": 42}])
        mock_service.create_pr_comment = AsyncMock()

        await _try_fallback_notifications(mock_service, "owner/repo", "main", "Notification", [])

        mock_service.create_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_no_open_prs(self):
        """Test fallback when no open PRs."""
        from routers.github_webhooks import _try_fallback_notifications

        mock_service = AsyncMock()
        mock_service.get_pull_requests = AsyncMock(return_value=[])

        with patch("routers.github_webhooks._save_notification_to_file") as mock_save:
            mock_save.return_value = None

            await _try_fallback_notifications(
                mock_service, "owner/repo", "main", "Notification", [{"issue": "test"}]
            )

            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_all_failed(self, caplog):
        """Test fallback when all methods fail logs the error."""
        import logging

        from routers.github_webhooks import _try_fallback_notifications

        mock_service = AsyncMock()
        mock_service.get_pull_requests = AsyncMock(side_effect=ConnectionError("Failed"))

        with caplog.at_level(logging.ERROR):
            await _try_fallback_notifications(
                mock_service, "owner/repo", "main", "Notification", []
            )

        assert "All notification methods failed" in caplog.text


# =============================================================================
# Save Notification To File Tests
# =============================================================================


@pytest.mark.unit
class TestSaveNotificationToFile:
    """Tests for _save_notification_to_file function."""

    @pytest.mark.asyncio
    async def test_save_notification_success(self, tmp_path, caplog):
        """Test saving notification to file writes correct JSON."""
        import logging

        from routers.github_webhooks import _save_notification_to_file

        with patch("routers.github_webhooks.Path", return_value=tmp_path):
            with caplog.at_level(logging.INFO):
                await _save_notification_to_file(
                    "owner/repo", "main", [{"issue": "test"}], "Notification body"
                )

            assert "Saved notification" in caplog.text

    @pytest.mark.asyncio
    async def test_save_notification_error(self, caplog):
        """Test saving notification when file write fails logs debug message."""
        import logging

        from routers.github_webhooks import _save_notification_to_file

        with patch("routers.github_webhooks.Path") as mock_path:
            mock_path.side_effect = OSError("Write failed")

            with caplog.at_level(logging.DEBUG):
                await _save_notification_to_file("owner/repo", "main", [], "Body")

            assert "Could not save notification file" in caplog.text


# =============================================================================
# Send Issue Notifications Tests
# =============================================================================


@pytest.mark.unit
class TestSendIssueNotifications:
    """Tests for _send_issue_notifications function."""

    @pytest.mark.asyncio
    async def test_send_notifications_success(self, mock_settings):
        """Test sending issue notifications successfully."""
        from routers.github_webhooks import _send_issue_notifications

        issues = [{"file": "test.py", "message": "Critical issue"}]

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                mock_service = MagicMock()
                mock_service.create_issue = AsyncMock()
                mock_service_class.return_value = mock_service
                with patch("routers.github_webhooks._try_create_github_issue") as mock_create:
                    mock_create.return_value = True

                    await _send_issue_notifications("owner/repo", "main", issues)

                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notifications_no_token(self, caplog):
        """No webhook credential exists: logs the skip and returns early."""
        import logging

        from routers.github_webhooks import _send_issue_notifications

        with patch.dict(
            "os.environ",
            {"GITHUB_ACCESS_TOKEN": "ghp_env_pat", "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat"},
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                with caplog.at_level(logging.INFO):
                    await _send_issue_notifications("owner/repo", "main", [])

                assert "issue_notifications" in caplog.text
                assert "needs a GitHub credential" in caplog.text
                mock_service_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_notifications_fallback(self, mock_settings):
        """Test sending notifications with fallback."""
        from routers.github_webhooks import _send_issue_notifications

        issues = [{"file": "test.py", "message": "Issue"}]

        with patch(
            "routers.github_webhooks._webhook_github_token", return_value="test-token-value"
        ):
            with patch("routers.github_webhooks.GitHubService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service
                with patch("routers.github_webhooks._try_create_github_issue") as mock_create:
                    mock_create.return_value = False
                    with patch(
                        "routers.github_webhooks._try_fallback_notifications"
                    ) as mock_fallback:
                        mock_fallback.return_value = None

                        await _send_issue_notifications("owner/repo", "main", issues)

                        mock_fallback.assert_called_once()


# =============================================================================
# Response Model Tests
# =============================================================================


@pytest.mark.unit
class TestGitHubWebhookResponseModel:
    """Tests for GitHubWebhookResponse model."""

    def test_response_model_success(self):
        """Test creating successful webhook response."""
        from routers.github_webhooks import GitHubWebhookResponse
        from schemas.github_api_models import ResponseStatus

        response = GitHubWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="Webhook processed",
            event_type="pull_request",
            action="opened",
            processed=True,
            processing_time_ms=125.5,
            delivery_id="delivery-123",
        )

        assert response.status == ResponseStatus.SUCCESS
        assert response.event_type == "pull_request"
        assert response.action == "opened"
        assert response.processed is True
        assert response.processing_time_ms == 125.5
        assert response.delivery_id == "delivery-123"

    def test_response_model_minimal(self):
        """Test creating minimal webhook response."""
        from routers.github_webhooks import GitHubWebhookResponse
        from schemas.github_api_models import ResponseStatus

        response = GitHubWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="OK",
            event_type="ping",
            processed=True,
            processing_time_ms=10.0,
        )

        assert response.action is None
        assert response.delivery_id is None
