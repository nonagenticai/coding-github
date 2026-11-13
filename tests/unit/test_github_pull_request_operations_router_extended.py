"""Extended tests for PR operations endpoints via real HTTP requests.

Tests /api/repositories/ fork, apply-changes, create-pull-request, validate-pull-request
through the ASGI transport, mocking only the service layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

BASE = "/api/repositories"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_container_payload(**overrides):
    """A coding-containers GET envelope — what the shared client returns."""
    container = {
        "container_name": "test-container",
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "status": "running",
        "github_access_token": None,
    }
    container.update(overrides)
    return {"status": "success", "message": "ok", "container": container}


def _mock_workspaces_client(get_result=None, error=None):
    """Mock the shared coding-containers client used for unified resolution."""
    client = MagicMock()
    if error is not None:
        client.get_workspace = AsyncMock(side_effect=error)
    else:
        client.get_workspace = AsyncMock(
            return_value=get_result if get_result is not None else _make_container_payload()
        )
    return client


def _mock_fork_repo():
    fork = MagicMock()
    fork.html_url = "https://github.com/fork-owner/repo"
    fork.owner = SimpleNamespace(login="fork-owner")
    fork.name = "repo"
    fork.private = True  # the resolved WRITE target is private too (#118)
    return fork


# ===========================================================================
# Fork — fallback container search
# ===========================================================================


class TestForkUnifiedResolution:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_resolved_via_unified_client(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        """Fork resolves the container through the shared coding-containers client."""
        mock_get_client.return_value = _mock_workspaces_client(
            _make_container_payload(container_name="my-container")
        )

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "my-container"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["fork_owner"] == "fork-owner"
        assert body["original_url"] == "https://github.com/owner/repo"


class TestForkInvalidGithubUrl:
    @patch("routers.github_pull_request_operations.parse_github_url", return_value=(None, None))
    @patch("main.get_workspaces_client")
    async def test_fork_bad_url_returns_400(self, mock_get_client, mock_parse, async_client):
        mock_get_client.return_value = _mock_workspaces_client(
            _make_container_payload(repo_url="https://not-github.com/bad")
        )

        resp = await async_client.post(f"{BASE}/fork", json={"container_name": "test-container"})
        assert resp.status_code == 400


# ===========================================================================
# Apply changes — unified resolution & errors
# ===========================================================================


class TestApplyChangesUnifiedResolution:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_changes_resolved_via_unified_client(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client(
            _make_container_payload(container_name="card-ctr")
        )

        fork_repo = _mock_fork_repo()
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.return_value = SimpleNamespace(repository=fork_repo)
        mock_instance._create_branch.return_value = None
        mock_instance._apply_changes.return_value = ("sha456", 1)
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "card-ctr",
                "changes": [{"file_path": "a.py", "content": "x=1"}],
                "branch_name": "my-branch",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["changes_applied"] == 1
        assert body["commit_sha"] == "sha456"


class TestApplyChangesInvalidUrl:
    @patch(
        "routers.github_pull_request_operations.parse_github_url",
        return_value=("", ""),
    )
    @patch("main.get_workspaces_client")
    async def test_apply_changes_bad_url_returns_400(
        self, mock_get_client, mock_parse, async_client
    ):
        mock_get_client.return_value = _mock_workspaces_client(
            _make_container_payload(repo_url="https://bad-url.com")
        )

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 400


class TestApplyChangesRuntimeError:
    @patch("main.get_workspaces_client")
    async def test_runtime_error_returns_404(self, mock_get_client, async_client):
        mock_get_client.side_effect = RuntimeError("client not init")

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 404

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_generic_exception_returns_500(self, mock_gh_cls, mock_get_client, async_client):
        mock_get_client.return_value = _mock_workspaces_client()

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance._ensure_fork.side_effect = Exception("unexpected error")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/apply-changes",
            json={
                "container_name": "test-container",
                "changes": [{"file_path": "a.py", "content": "x"}],
            },
        )
        assert resp.status_code == 500


# ===========================================================================
# Create PR — missing fields
# ===========================================================================


class TestCreatePRMissingFields:
    async def test_create_pr_empty_required_field_returns_400(self, async_client):
        """Empty required fields returns 400."""
        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": " ",
                "original_repo": " ",
                "fork_owner": "fork",
                "fork_repo": "repo",
                "head_branch": "branch",
                "base_branch": "main",
                "title": "Title",
                "message": "Message",
            },
        )
        assert resp.status_code == 400


# ===========================================================================
# Create PR — real API path success & failure
# ===========================================================================


class TestCreatePRRealApi:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_api_success(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": True,
            "pr_number": 42,
            "pr_url": "https://github.com/owner/repo/pull/42",
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "feature/x",
                "base_branch": "main",
                "title": "Real PR",
                "message": "Description",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pr_number"] == 42

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_api_failure_returns_400(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.create_pull_request.return_value = {
            "success": False,
            "error": "Branch not found",
        }
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/create-pull-request",
            json={
                "original_owner": "real-owner",
                "original_repo": "real-repo",
                "fork_owner": "fork-user",
                "fork_repo": "real-repo",
                "head_branch": "feature/x",
                "base_branch": "main",
                "title": "Real PR",
                "message": "Description",
            },
        )
        assert resp.status_code == 400


# ===========================================================================
# Validate PR — input validation
# ===========================================================================


class TestValidatePRInputValidation:
    async def test_missing_repo_owner_returns_422(self, async_client):
        """Missing repo_owner returns 422."""
        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code == 422

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_no_github_client_returns_500(self, mock_gh_cls, async_client):
        """github_client is None returns 500."""
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client = None
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code == 500


# ===========================================================================
# Validate PR — repo/PR not found
# ===========================================================================


class TestValidatePRNotFound:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_pr_not_found_returns_404(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_repo.get_pull.side_effect = Exception("404 Not Found")
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 999},
        )
        assert resp.status_code == 404

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_pr_generic_error_returns_500(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_repo.get_pull.side_effect = Exception("Connection timeout")
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code == 500

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_pr_files_error_returns_500(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.side_effect = Exception("API rate limit")
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code == 500


# ===========================================================================
# Validate PR — legacy quality validation
# ===========================================================================


class TestValidatePRLegacyQuality:
    @patch("routers.github_pull_request_operations._validate_pr_quality", new_callable=AsyncMock)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_legacy_quality_validation(self, mock_gh_cls, mock_quality, async_client):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_quality.return_value = PRValidationResult(
            validation_type="quality", passed=True, score=90.0, issues=[], suggestions=[]
        )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["quality"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert len(body["validations"]) == 1


# ===========================================================================
# Validate PR — ImportError fallback
# ===========================================================================


class TestValidatePRImportErrorFallback:
    @patch(
        "routers.github_pull_request_operations.validate_pr_comprehensive",
        new_callable=AsyncMock,
    )
    @patch("routers.github_pull_request_operations._validate_pr_quality", new_callable=AsyncMock)
    @patch(
        "routers.github_pull_request_operations._validate_pr_performance", new_callable=AsyncMock
    )
    @patch("routers.github_pull_request_operations._validate_pr_security", new_callable=AsyncMock)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_import_error_falls_back_to_legacy(
        self, mock_gh_cls, mock_sec, mock_perf, mock_quality, mock_comprehensive, async_client
    ):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_comprehensive.side_effect = ImportError("module not found")

        for mock_fn, name, score in [
            (mock_sec, "security", 90.0),
            (mock_perf, "performance", 80.0),
            (mock_quality, "quality", 85.0),
        ]:
            mock_fn.return_value = PRValidationResult(
                validation_type=name, passed=True, score=score, issues=[], suggestions=[]
            )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security", "performance", "quality"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert len(body["validations"]) == 3


# ===========================================================================
# Validate PR — ValueError/TypeError fallback
# ===========================================================================


class TestValidatePRValueErrorFallback:
    @patch(
        "routers.github_pull_request_operations.validate_pr_comprehensive",
        new_callable=AsyncMock,
    )
    @patch("routers.github_pull_request_operations._validate_pr_quality", new_callable=AsyncMock)
    @patch(
        "routers.github_pull_request_operations._validate_pr_performance", new_callable=AsyncMock
    )
    @patch("routers.github_pull_request_operations._validate_pr_security", new_callable=AsyncMock)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_value_error_falls_back(
        self, mock_gh_cls, mock_sec, mock_perf, mock_quality, mock_comprehensive, async_client
    ):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_comprehensive.side_effect = ValueError("bad data")

        for mock_fn, name, score in [
            (mock_sec, "security", 90.0),
            (mock_perf, "performance", 80.0),
            (mock_quality, "quality", 85.0),
        ]:
            mock_fn.return_value = PRValidationResult(
                validation_type=name, passed=True, score=score, issues=[], suggestions=[]
            )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security", "performance", "quality"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["validations"]) == 3


# ===========================================================================
# Validate PR — recommendation logic
# ===========================================================================


class TestValidatePRRecommendation:
    @patch("routers.github_pull_request_operations._validate_pr_security", new_callable=AsyncMock)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_needs_work_recommendation(self, mock_gh_cls, mock_sec, async_client):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_sec.return_value = PRValidationResult(
            validation_type="security", passed=False, score=65.0, issues=[], suggestions=[]
        )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["recommendation"] == "needs_work"

    @patch("routers.github_pull_request_operations._validate_pr_security", new_callable=AsyncMock)
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_reject_recommendation(self, mock_gh_cls, mock_sec, async_client):
        from schemas.github_api_models import PRValidationResult

        mock_instance = MagicMock()

        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_repo.get_pull.return_value = mock_pr
        mock_instance.github_client.get_repo.return_value = mock_repo
        mock_gh_cls.return_value = mock_instance

        mock_sec.return_value = PRValidationResult(
            validation_type="security", passed=False, score=30.0, issues=[], suggestions=[]
        )

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={
                "repo_owner": "owner",
                "repo_name": "repo",
                "pr_number": 1,
                "validation_types": ["security"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["recommendation"] == "reject"


# ===========================================================================
# Validate PR — NotFoundException / PermissionException
# ===========================================================================


class TestValidatePRExceptionPropagation:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_repo_not_found(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client.get_repo.side_effect = Exception("404 not found")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        # Should map to 404 via NotFoundException
        assert resp.status_code in (404, 500)

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_permission_denied(self, mock_gh_cls, async_client):
        mock_instance = MagicMock()
        mock_instance.github_client.get_repo.return_value.private = True  # a private target repo
        mock_instance.github_client.get_repo.side_effect = Exception("403 forbidden")
        mock_gh_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE}/validate-pull-request",
            json={"repo_owner": "owner", "repo_name": "repo", "pr_number": 1},
        )
        assert resp.status_code in (403, 500)
