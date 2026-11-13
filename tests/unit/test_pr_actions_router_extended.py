"""Extended tests for PR actions endpoints via real HTTP requests.

Tests the /api/webhooks/pr-actions/ endpoints through the ASGI transport,
mocking only the service layer (PRAutoCloser, validate_and_get_container_for_tenant).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from schemas.github_api_models import PRValidationResult, ValidatePRResponse
from tests.conftest import PLATFORM_TEST_TENANT


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch, platform_tenant, internal_app):
    """The fixture owner "myorg" is declared the platform org, and the internal
    GitHub App is configured: since 2026-09-30 it is the ONLY credential for a
    platform repo (no PAT fallback). Customer owners are tenant-bound public-App
    only (tests/unit/test_tenant_repo_token.py).
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "myorg")


BASE_URL = "/api/webhooks/pr-actions"


@pytest.fixture(autouse=True)
def _patch_validate_pull_request():
    """The validation-webhook now runs REAL PR validation; stub it so these
    router tests stay hermetic (no GitHub calls). Returns an approve verdict.

    This must be a real ValidatePRResponse, not a SimpleNamespace: the response
    is handed to PRAutoCloser, which also reads `.validations` (and each entry's
    `.issues`/`.suggestions`). A stub narrower than the real model silently
    turned every downstream action into an AttributeError.
    """
    with patch("routers.pr_actions.validate_pull_request", new_callable=AsyncMock) as mock:
        mock.return_value = ValidatePRResponse(
            status="success",
            message="done",
            overall_score=85.0,
            passed=True,
            validations=[
                PRValidationResult(
                    validation_type="security",
                    passed=True,
                    score=85.0,
                    issues=[],
                    suggestions=["Keep it up"],
                )
            ],
            recommendation="approve",
        )
        yield mock


# ===========================================================================
# ProcessValidationRequest validator edge cases
# ===========================================================================


class TestProcessValidationRequestValidators:
    """Pydantic validator edge cases tested via HTTP POST."""

    async def test_whitespace_only_repo_owner_rejected(self, async_client):
        """Whitespace-only repo_owner returns 422."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "   ",
                "repo_name": "myrepo",
                "pr_number": 1,
                "validation_response": {},
            },
        )
        assert resp.status_code == 422

    async def test_control_char_in_repo_owner_rejected(self, async_client):
        """Control character in repo_owner returns 422."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "my\x00org",
                "repo_name": "myrepo",
                "pr_number": 1,
                "validation_response": {},
            },
        )
        assert resp.status_code == 422

    async def test_control_char_in_repo_name_rejected(self, async_client):
        """Control character in repo_name returns 422."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "my\x05repo",
                "pr_number": 1,
                "validation_response": {},
            },
        )
        assert resp.status_code == 422

    async def test_pr_number_zero_rejected(self, async_client):
        """pr_number=0 returns 422 (must be >= 1)."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 0,
                "validation_response": {},
            },
        )
        assert resp.status_code == 422

    async def test_negative_pr_number_rejected(self, async_client):
        """Negative pr_number returns 422."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": -5,
                "validation_response": {},
            },
        )
        assert resp.status_code == 422


# ===========================================================================
# ValidationWebhookRequest validator edge cases
# ===========================================================================


class TestValidationWebhookRequestValidators:
    """Pydantic validator edge cases for /validation-webhook."""

    async def test_whitespace_only_repo_owner(self, async_client):
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={"repo_owner": "   ", "repo_name": "myrepo", "pr_number": 1},
        )
        assert resp.status_code == 422

    async def test_whitespace_only_repo_name(self, async_client):
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={"repo_owner": "myorg", "repo_name": "  ", "pr_number": 1},
        )
        assert resp.status_code == 422

    async def test_control_char_in_repo_owner(self, async_client):
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={"repo_owner": "org\x01name", "repo_name": "myrepo", "pr_number": 1},
        )
        assert resp.status_code == 422

    async def test_control_char_in_repo_name(self, async_client):
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={"repo_owner": "myorg", "repo_name": "repo\nname", "pr_number": 1},
        )
        assert resp.status_code == 422


# ===========================================================================
# process-validation — with container_name
# ===========================================================================


class TestProcessValidationWithContainer:
    @patch("routers.pr_actions.validate_and_get_container_for_tenant")
    async def test_container_name_validated(self, mock_validate_container, github_pr, async_client):
        """container_name triggers validate_and_get_container_for_tenant."""
        mock_validate_container.return_value = {"name": "c", "codebase_path": "/tmp/code"}

        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {
                    "status": "success",
                    "message": "done",
                    "overall_score": 85.0,
                    "passed": True,
                    "validations": [],
                    "recommendation": "approve",
                },
                "container_name": "my-container",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        mock_validate_container.assert_awaited_once_with("my-container", PLATFORM_TEST_TENANT)


class TestContainerRecordIsADictNotAModel:
    """D6 (live 2026-10-12): ``/process-validation`` with ``container_name`` answered
    500 ``'dict' object has no attribute 'codebase_path'``. coding-containers returns a
    plain dict envelope ``{"status", "message", "container": {...}}``; the router read
    ``container_info.codebase_path`` as if it were a model. These tests do NOT mock
    ``validate_and_get_container_for_tenant`` (the old ones did, with a SimpleNamespace,
    which is why the bug was invisible) — only the containers client, with the real
    envelope shape.

    MUTATION (measured): restore ``container_info.codebase_path`` in pr_actions and both
    route tests go red with 500.
    """

    ENVELOPE = {
        "status": "success",
        "message": "Container found",
        "container": {
            "name": "sweep-container",
            "container_id": "abc123",
            "status": "running",
            "codebase_path": "/app/data/repo",
            "repo_url": "https://github.com/myorg/myrepo",
        },
    }

    @pytest.fixture
    def containers_client(self):
        client = AsyncMock()
        client.get_workspace = AsyncMock(return_value=self.ENVELOPE)
        with patch("main.get_workspaces_client", return_value=client):
            yield client

    async def test_process_validation_with_container_name_is_not_a_500(
        self, containers_client, github_pr, async_client
    ):
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {
                    "status": "success",
                    "message": "done",
                    "overall_score": 85.0,
                    "passed": True,
                    "validations": [],
                    "recommendation": "approve",
                },
                "config": {"add_labels": True, "post_comment": True},
                "container_name": "sweep-container",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"
        containers_client.get_workspace.assert_awaited_once()

    async def test_validation_webhook_with_container_name_is_not_a_500(
        self, containers_client, github_pr, async_client
    ):
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "container_name": "sweep-container",
            },
        )
        assert resp.status_code == 200, resp.text

    async def test_helper_unwraps_the_envelope(self, containers_client):
        from utils.github_validation_helpers import (
            container_codebase_path,
            validate_and_get_container_for_tenant,
        )

        info = await validate_and_get_container_for_tenant("sweep-container", "t")
        assert info == self.ENVELOPE["container"]
        assert container_codebase_path(info) == "/app/data/repo"
        assert container_codebase_path({"name": "flat"}) is None
        assert container_codebase_path(SimpleNamespace(codebase_path="/m")) == "/m"


# ===========================================================================
# process-validation — parse failure & wrong type
# ===========================================================================


class TestProcessValidationParseFailure:
    async def test_validation_response_missing_required_fields(self, async_client):
        """validation_response missing required fields returns 400."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {"some_field": "value"},
            },
        )
        assert resp.status_code == 400

    async def test_validation_response_bad_data_types(self, async_client):
        """validation_response with wrong types returns 400."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {
                    "overall_score": "not-a-number",
                    "passed": True,
                    "validations": "not-a-list",
                    "recommendation": "approve",
                },
            },
        )
        assert resp.status_code == 400


# ===========================================================================
# validation-webhook — success paths
# ===========================================================================


class TestValidationWebhookSuccess:
    async def test_webhook_success_with_token(self, github_pr, async_client):
        """Successful webhook flow with github_token in payload."""
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "auto_close_enabled": False,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["validation_score"] == 85.0
        assert body["recommendation"] == "approve"
        assert body["actions_taken"]["validation_performed"] is True
        assert body["actions_taken"]["pr_closed"] is False
        # Ground truth for the two flags that used to come from config.
        github_pr.add_to_labels.assert_called_once()
        github_pr.create_issue_comment.assert_called_once()
        assert body["actions_taken"]["labels_updated"] is True
        assert body["actions_taken"]["comment_posted"] is True

    @patch("routers.pr_actions.validate_and_get_container_for_tenant")
    async def test_webhook_with_container_name(
        self, mock_validate_container, github_pr, async_client
    ):
        """Webhook with container_name validates the container."""
        mock_validate_container.return_value = {"name": "c", "codebase_path": "/tmp/code"}

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": "ghp_test_token",
                "container_name": "webhook-container",
            },
        )
        assert resp.status_code == 200
        mock_validate_container.assert_awaited_once_with("webhook-container", PLATFORM_TEST_TENANT)


# ===========================================================================
# validation-webhook — missing token
# ===========================================================================


def _no_app_and_armed_pats(monkeypatch):
    """Internal App NOT configured, every legacy PAT variable armed."""
    import services.github_app_auth as ga

    monkeypatch.setattr(ga, "app_auth_configured", lambda: False)
    for var in ("GITHUB_ACCESS_TOKEN", "GITHUB_TOKEN", "CODING_GITHUB_GITHUB_TOKEN"):
        monkeypatch.setenv(var, "ghp_legacy_pat_must_not_be_used")


class TestValidationWebhookMissingToken:
    async def test_webhook_no_token_returns_error(self, async_client, monkeypatch):
        """No github_token and no App: 401 — the armed env PATs are not a fallback."""
        _no_app_and_armed_pats(monkeypatch)
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": None,
            },
        )
        assert resp.status_code == 401, resp.text

    async def test_webhook_whitespace_token_returns_error(self, async_client, monkeypatch):
        """Whitespace-only token and no App: 401."""
        _no_app_and_armed_pats(monkeypatch)
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": "   ",
            },
        )
        assert resp.status_code == 401, resp.text


# ===========================================================================
# validation-webhook — short repo name
# ===========================================================================


class TestValidationWebhookShortRepoName:
    async def test_webhook_short_repo_name_returns_400(self, async_client):
        """repo_name < 2 chars returns 400."""
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "ab",
                "repo_name": "a",
                "pr_number": 5,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 400

    async def test_webhook_both_short_returns_400(self, async_client):
        """Both owner and name single char returns 400."""
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "x",
                "repo_name": "y",
                "pr_number": 5,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 400


# ===========================================================================
# validation-webhook — auto_close_enabled
# ===========================================================================


class TestValidationWebhookAutoClose:
    @patch("routers.pr_actions.PRAutoCloser", wraps=None)
    async def test_webhook_auto_close_enabled_approve_no_close(
        self, mock_autocloser_cls, github_pr, async_client
    ):
        """auto_close_enabled=True with approve recommendation => pr_closed=False.

        Still asserts the config handed to PRAutoCloser, so the class is spied
        on — but it delegates to the real implementation so `pr_closed` comes
        from the actual branch taken, not from the config.
        """
        from services.pr_validation.actions.pr_auto_closer import PRAutoCloser as RealAutoCloser

        mock_autocloser_cls.side_effect = RealAutoCloser

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "auto_close_enabled": True,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["actions_taken"]["pr_closed"] is False
        config_passed = mock_autocloser_cls.call_args[0][1]
        assert config_passed["auto_close_enabled"] is True


# ===========================================================================
# validation-webhook — downstream errors
# ===========================================================================


class TestValidationWebhookErrors:
    @patch("routers.pr_actions.PRAutoCloser")
    async def test_webhook_autocloser_error_returns_500(self, mock_autocloser_cls, async_client):
        """Unhandled exception from PRAutoCloser returns 500."""
        mock_instance = MagicMock()
        mock_instance.process_validation_result = AsyncMock(
            side_effect=RuntimeError("GitHub API error")
        )
        mock_autocloser_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 500

    @patch("routers.pr_actions.PRAutoCloser")
    async def test_webhook_http_exception_passthrough(self, mock_autocloser_cls, async_client):
        """HTTPException re-raised preserving status code."""
        mock_instance = MagicMock()
        mock_instance.process_validation_result = AsyncMock(
            side_effect=HTTPException(status_code=404, detail="PR not found")
        )
        mock_autocloser_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": "ghp_test_token",
            },
        )
        assert resp.status_code == 404


# ===========================================================================
# validation-webhook — env token fallback
# ===========================================================================


class TestValidationWebhookAppToken:
    @patch("routers.pr_actions.PRAutoCloser")
    @patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "ghp_env_token"}, clear=False)
    async def test_webhook_uses_the_internal_app_not_the_env_pat(
        self, mock_autocloser_cls, async_client, internal_app
    ):
        """No request token: the internal App token is used — never the env PAT
        (the ``GITHUB_ACCESS_TOKEN`` fallback was removed 2026-09-30)."""
        mock_instance = MagicMock()
        mock_instance.process_validation_result = AsyncMock()
        mock_autocloser_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": None,
            },
        )
        assert resp.status_code == 200
        mock_autocloser_cls.assert_called_once_with(
            internal_app,
            {
                "auto_close_enabled": False,
                "close_on_score_below": 20,
                "add_labels": True,
                "post_comment": True,
                "request_changes": True,
            },
        )
