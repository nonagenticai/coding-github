"""Unit tests for GitHubActionsWorkflowTrigger.

Verifies workflow dispatch via PyGithub, REST API dispatch,
workflow status checks, authentication, error handling, and the singleton factory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from github.GithubException import GithubException

from services.github_actions_trigger import GitHubActionsWorkflowTrigger


def _make_trigger(token="ghp_test"):
    """Build a trigger with a caller-supplied token and a mocked PyGithub client."""
    with patch("services.github_actions_trigger.Github") as mock_github_cls:
        mock_client = MagicMock()
        mock_github_cls.return_value = mock_client
        trigger = GitHubActionsWorkflowTrigger(github_token=token)
        trigger._mock_client = mock_client
    return trigger


def _make_trigger_no_token():
    """Build a trigger with no caller token — while every legacy PAT var is armed.

    ⛔ There is no platform-PAT fallback (owner ruling 2026-09-30): the trigger
    must stay credential-less even though the environment carries PATs.
    """
    with (
        patch.dict(
            "os.environ",
            {
                "GITHUB_ACCESS_TOKEN": "ghp_env_pat_must_not_be_used",
                "GITHUB_TOKEN": "ghp_env_pat_must_not_be_used",
                "CODING_GITHUB_GITHUB_TOKEN": "ghp_env_pat_must_not_be_used",
            },
        ),
        patch("services.github_actions_trigger.Github") as mock_github_cls,
    ):
        trigger = GitHubActionsWorkflowTrigger(github_token=None)
        mock_github_cls.assert_not_called()
    assert trigger.github_token is None
    return trigger


class TestWorkflowDispatchViaPyGithub:
    """Tests for trigger_pr_comprehensive_validation using PyGithub."""

    async def test_successful_dispatch(self):
        """Successful dispatch returns status=triggered with correct metadata."""
        trigger = _make_trigger()
        repo = MagicMock()
        workflow = MagicMock()
        workflow.create_dispatch.return_value = True
        repo.get_workflow.return_value = workflow
        trigger._mock_client.get_repo.return_value = repo

        result = await trigger.trigger_pr_comprehensive_validation(
            repo_owner="org",
            repo_name="repo",
            pr_number=42,
            pr_sha="abc123",
        )

        assert result["status"] == "triggered"
        assert result["pr_number"] == 42
        assert result["repo"] == "org/repo"
        workflow.create_dispatch.assert_called_once()
        dispatch_kwargs = workflow.create_dispatch.call_args
        inputs = dispatch_kwargs.kwargs.get("inputs") or dispatch_kwargs[1].get("inputs")
        assert inputs["pr_number"] == "42"
        assert inputs["pr_sha"] == "abc123"

    async def test_auto_generates_validation_id(self):
        """When validation_id is None, an auto-generated ID is used."""
        trigger = _make_trigger()
        repo = MagicMock()
        workflow = MagicMock()
        workflow.create_dispatch.return_value = True
        repo.get_workflow.return_value = workflow
        trigger._mock_client.get_repo.return_value = repo

        result = await trigger.trigger_pr_comprehensive_validation(
            repo_owner="org",
            repo_name="repo",
            pr_number=42,
            pr_sha="abc123",
            validation_id=None,
        )

        assert result["validation_id"].startswith("pr-val-")

    async def test_uses_provided_validation_id(self):
        """When validation_id is provided, it is used as-is."""
        trigger = _make_trigger()
        repo = MagicMock()
        workflow = MagicMock()
        workflow.create_dispatch.return_value = True
        repo.get_workflow.return_value = workflow
        trigger._mock_client.get_repo.return_value = repo

        result = await trigger.trigger_pr_comprehensive_validation(
            repo_owner="org",
            repo_name="repo",
            pr_number=1,
            pr_sha="def456",
            validation_id="custom-val-id",
        )

        assert result["validation_id"] == "custom-val-id"

    async def test_dispatch_returns_false_raises(self):
        """When create_dispatch returns False, an exception is raised."""
        trigger = _make_trigger()
        repo = MagicMock()
        workflow = MagicMock()
        workflow.create_dispatch.return_value = False
        repo.get_workflow.return_value = workflow
        trigger._mock_client.get_repo.return_value = repo

        with pytest.raises(Exception, match="Failed to trigger workflow dispatch"):
            await trigger.trigger_pr_comprehensive_validation(
                repo_owner="org", repo_name="repo", pr_number=42, pr_sha="abc123"
            )

    async def test_no_client_raises_value_error(self):
        """No github_client raises ValueError."""
        trigger = _make_trigger_no_token()
        trigger.github_client = None
        with pytest.raises(ValueError, match="GitHub client not initialized"):
            await trigger.trigger_pr_comprehensive_validation(
                repo_owner="org", repo_name="repo", pr_number=1, pr_sha="abc123"
            )

    async def test_github_exception_wrapped(self):
        """GithubException is caught and re-raised with a readable message."""
        trigger = _make_trigger()
        repo = MagicMock()
        repo.get_workflow.side_effect = GithubException(
            status=403, data={"message": "Resource not accessible"}
        )
        trigger._mock_client.get_repo.return_value = repo

        with pytest.raises(Exception, match="Failed to trigger workflow"):
            await trigger.trigger_pr_comprehensive_validation(
                repo_owner="org", repo_name="repo", pr_number=42, pr_sha="abc123"
            )

    async def test_generic_exception_propagates(self):
        """Non-GitHub exceptions propagate unchanged."""
        trigger = _make_trigger()
        repo = MagicMock()
        repo.get_workflow.side_effect = RuntimeError("unexpected")
        trigger._mock_client.get_repo.return_value = repo

        with pytest.raises(RuntimeError, match="unexpected"):
            await trigger.trigger_pr_comprehensive_validation(
                repo_owner="org", repo_name="repo", pr_number=42, pr_sha="abc123"
            )


class TestWorkflowDispatchViaRestAPI:
    """Tests for trigger_workflow_via_api using httpx."""

    async def test_successful_204_response(self):
        """HTTP 204 indicates successful dispatch."""
        trigger = _make_trigger(token="ghp_my_token")

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github_actions_trigger.httpx.AsyncClient", return_value=mock_http_client
        ):
            result = await trigger.trigger_workflow_via_api(
                repo_owner="org",
                repo_name="repo",
                workflow_id="ci.yml",
                ref="main",
                inputs={"key": "val"},
            )

        assert result["status"] == "triggered"
        call_kwargs = mock_http_client.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer ghp_my_token"

    async def test_non_204_response_raises(self):
        """Non-204 response raises with status code and error message."""
        trigger = _make_trigger(token="ghp_tok")

        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = '{"message":"Validation Failed"}'
        mock_response.json.return_value = {"message": "Validation Failed"}

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github_actions_trigger.httpx.AsyncClient", return_value=mock_http_client
        ):
            with pytest.raises(Exception, match="Workflow trigger failed.*422"):
                await trigger.trigger_workflow_via_api(
                    repo_owner="org",
                    repo_name="repo",
                    workflow_id="ci.yml",
                    ref="main",
                    inputs={"a": "b"},
                )

    async def test_empty_error_body_handled(self):
        """Non-204 with empty response body is handled gracefully."""
        trigger = _make_trigger(token="ghp_tok")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = ""
        mock_response.json.return_value = {}

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github_actions_trigger.httpx.AsyncClient", return_value=mock_http_client
        ):
            with pytest.raises(Exception, match="Workflow trigger failed.*500"):
                await trigger.trigger_workflow_via_api(
                    repo_owner="org",
                    repo_name="repo",
                    workflow_id="ci.yml",
                    ref="main",
                    inputs={},
                )

    async def test_timeout_raises(self):
        """httpx.TimeoutException is wrapped into a readable exception."""
        trigger = _make_trigger(token="ghp_tok")

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "services.github_actions_trigger.httpx.AsyncClient", return_value=mock_http_client
        ):
            with pytest.raises(Exception, match="timed out"):
                await trigger.trigger_workflow_via_api(
                    repo_owner="org",
                    repo_name="repo",
                    workflow_id="ci.yml",
                    ref="main",
                    inputs={},
                )

    async def test_no_token_raises_value_error(self):
        """No github_token raises ValueError."""
        trigger = _make_trigger_no_token()
        trigger.github_token = None
        with pytest.raises(ValueError, match="GitHub token not available"):
            await trigger.trigger_workflow_via_api(
                repo_owner="org",
                repo_name="repo",
                workflow_id="ci.yml",
                ref="main",
                inputs={},
            )


class TestCheckWorkflowStatus:
    """Tests for check_workflow_status."""

    async def test_completed_success(self):
        """Returns workflow run details for a completed run."""
        trigger = _make_trigger()
        repo = MagicMock()
        run = MagicMock()
        run.id = 99
        run.status = "completed"
        run.conclusion = "success"
        run.created_at = MagicMock(isoformat=MagicMock(return_value="2025-01-01T00:00:00Z"))
        run.updated_at = MagicMock(isoformat=MagicMock(return_value="2025-01-01T00:05:00Z"))
        run.html_url = "https://github.com/org/repo/actions/runs/99"
        repo.get_workflow_run.return_value = run
        trigger._mock_client.get_repo.return_value = repo

        result = await trigger.check_workflow_status("org", "repo", 99)

        assert result["status"] == "completed"
        assert result["conclusion"] == "success"
        assert result["id"] == 99
        assert result["html_url"] == "https://github.com/org/repo/actions/runs/99"

    async def test_completed_failure(self):
        """Returns failure conclusion when workflow fails."""
        trigger = _make_trigger()
        repo = MagicMock()
        run = MagicMock()
        run.id = 100
        run.status = "completed"
        run.conclusion = "failure"
        run.created_at = MagicMock(isoformat=MagicMock(return_value="2025-01-01T00:00:00Z"))
        run.updated_at = MagicMock(isoformat=MagicMock(return_value="2025-01-01T00:05:00Z"))
        run.html_url = "https://github.com/org/repo/actions/runs/100"
        repo.get_workflow_run.return_value = run
        trigger._mock_client.get_repo.return_value = repo

        result = await trigger.check_workflow_status("org", "repo", 100)
        assert result["conclusion"] == "failure"

    async def test_no_client_raises(self):
        """No github_client raises ValueError."""
        trigger = _make_trigger_no_token()
        trigger.github_client = None
        with pytest.raises(ValueError, match="GitHub client not initialized"):
            await trigger.check_workflow_status("org", "repo", 1)

    async def test_github_exception_on_status_check(self):
        """GithubException from get_workflow_run is wrapped."""
        trigger = _make_trigger()
        repo = MagicMock()
        repo.get_workflow_run.side_effect = GithubException(
            status=404, data={"message": "Not Found"}
        )
        trigger._mock_client.get_repo.return_value = repo

        with pytest.raises(Exception, match="Failed to check workflow status"):
            await trigger.check_workflow_status("org", "repo", 999)

    async def test_generic_exception_on_status_check(self):
        """Generic exception propagates unchanged."""
        trigger = _make_trigger()
        repo = MagicMock()
        repo.get_workflow_run.side_effect = RuntimeError("connection lost")
        trigger._mock_client.get_repo.return_value = repo

        with pytest.raises(RuntimeError, match="connection lost"):
            await trigger.check_workflow_status("org", "repo", 42)
