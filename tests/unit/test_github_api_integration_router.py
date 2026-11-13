"""Tests for routers/github_api_integration.py — comprehensive unit tests."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure stubs (same pattern as other test modules)
# ---------------------------------------------------------------------------


def _ensure_stubs():
    exc_mod = sys.modules.get("coding_client.exceptions")
    if exc_mod is None:
        return
    for name in (
        "AuthenticationException",
        "RateLimitException",
        "ServiceUnavailableException",
        "ValidationException",
    ):
        if not hasattr(exc_mod, name):
            setattr(exc_mod, name, type(name, (Exception,), {}))
    api_exc = sys.modules.get("coding_client.exceptions.api_exceptions")
    if api_exc:
        for name in (
            "AuthenticationException",
            "RateLimitException",
            "ServiceUnavailableException",
        ):
            if not hasattr(api_exc, name):
                setattr(api_exc, name, getattr(exc_mod, name))

    github_mod = sys.modules.get("github")
    if github_mod and not hasattr(github_mod, "__path__"):
        github_mod.__path__ = []
    for sub in (
        "github.PullRequest",
        "github.Repository",
        "github.GithubException",
        "github.AuthenticatedUser",
    ):
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            setattr(mod, sub.split(".")[-1], MagicMock)
            sys.modules[sub] = mod


_ensure_stubs()

from fastapi import HTTPException  # noqa: E402

from routers.github_api_integration import (  # noqa: E402
    GitHubOperationContext,
    calculate_validation_score,
    determine_pr_recommendation,
    extract_repo_info_from_url,
    format_container_status_response,
    format_github_error_message,
    get_container_by_name_or_id,
    sanitize_log_data,
    validate_file_changes,
    validate_github_repository_url,
    validate_pr_creation_request,
)

# ===========================================================================
# get_container_by_name_or_id
# ===========================================================================


class TestGetContainerByNameOrId:
    async def test_found_by_name_directly(self):
        """Container found by name via docker_service.find_container_by_name."""
        container = SimpleNamespace(container_name="my-ctr", id="ctr-1")
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(return_value=container)
        service.list_containers = AsyncMock(return_value=[])

        result = await get_container_by_name_or_id("my-ctr", service, "t1")
        assert result is container

    async def test_found_by_name_in_fallback_list(self):
        """Container found via list_containers fallback when direct lookup returns None."""
        target = SimpleNamespace(container_name="my-ctr", id="ctr-2")
        other = SimpleNamespace(container_name="other", id="ctr-3")
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        service.list_containers = AsyncMock(return_value=[other, target])

        result = await get_container_by_name_or_id("my-ctr", service, "t1")
        assert result is target

    async def test_found_by_id_in_fallback_list(self):
        """Container matched by its id attribute in the fallback list."""
        target = SimpleNamespace(container_name="some-name", id="ctr-99")
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        service.list_containers = AsyncMock(return_value=[target])

        result = await get_container_by_name_or_id("ctr-99", service, "t1")
        assert result is target

    async def test_not_found_raises_404(self):
        """HTTPException 404 raised when container is not found anywhere."""
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        service.list_containers = AsyncMock(return_value=[])

        with pytest.raises(HTTPException) as exc_info:
            await get_container_by_name_or_id("ghost", service, "t1")
        assert exc_info.value.status_code == 404

    async def test_service_error_raises_500(self):
        """HTTPException 500 raised when service raises an exception."""
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_container_by_name_or_id("anything", service, "t1")
        assert exc_info.value.status_code == 500

    async def test_fallback_list_with_no_id_attr(self):
        """Container without an id attribute should not match on id."""
        no_id = SimpleNamespace(container_name="no-id-ctr")  # no id attribute
        service = MagicMock()
        service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        service.list_containers = AsyncMock(return_value=[no_id])

        with pytest.raises(HTTPException) as exc_info:
            await get_container_by_name_or_id("some-id", service, "t1")
        assert exc_info.value.status_code == 404


# ===========================================================================
# validate_github_repository_url
# ===========================================================================


class TestValidateGithubRepositoryUrl:
    def test_valid_https_url(self):
        valid, err = validate_github_repository_url("https://github.com/owner/repo")
        assert valid is True
        assert err == ""

    def test_valid_http_url(self):
        valid, err = validate_github_repository_url("http://github.com/owner/repo")
        assert valid is True
        assert err == ""

    def test_valid_url_with_git_suffix(self):
        valid, err = validate_github_repository_url("https://github.com/owner/repo.git")
        assert valid is True
        assert err == ""

    def test_empty_string(self):
        valid, err = validate_github_repository_url("")
        assert valid is False
        assert "non-empty" in err

    def test_none_input(self):
        valid, err = validate_github_repository_url(None)
        assert valid is False
        assert "non-empty" in err

    def test_non_github_url(self):
        valid, err = validate_github_repository_url("https://gitlab.com/owner/repo")
        assert valid is False
        assert "GitHub URL" in err

    def test_missing_repo_name(self):
        valid, err = validate_github_repository_url("https://github.com/owner")
        assert valid is False
        assert "owner and repository" in err

    def test_empty_owner(self):
        valid, err = validate_github_repository_url("https://github.com//repo")
        assert valid is False
        assert "valid owner" in err

    def test_empty_repo(self):
        valid, err = validate_github_repository_url("https://github.com/owner/")
        assert valid is False
        assert "valid owner" in err or "empty" in err.lower()

    def test_only_git_suffix_as_repo(self):
        valid, err = validate_github_repository_url("https://github.com/owner/.git")
        assert valid is False
        assert "empty" in err.lower()

    def test_url_with_extra_path(self):
        """URL with additional path segments should still validate."""
        valid, err = validate_github_repository_url("https://github.com/owner/repo/tree/main")
        assert valid is True

    def test_whitespace_stripped(self):
        valid, err = validate_github_repository_url("  https://github.com/owner/repo  ")
        assert valid is True


# ===========================================================================
# extract_repo_info_from_url
# ===========================================================================


class TestExtractRepoInfoFromUrl:
    def test_basic_extraction(self):
        info = extract_repo_info_from_url("https://github.com/myorg/myrepo")
        assert info == {"owner": "myorg", "repo": "myrepo"}

    def test_strips_git_suffix(self):
        info = extract_repo_info_from_url("https://github.com/myorg/myrepo.git")
        assert info["repo"] == "myrepo"

    def test_http_url(self):
        info = extract_repo_info_from_url("http://github.com/owner/repo")
        assert info == {"owner": "owner", "repo": "repo"}

    def test_raises_on_invalid_url(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_repo_info_from_url("")

    def test_raises_on_non_github(self):
        with pytest.raises(ValueError, match="GitHub URL"):
            extract_repo_info_from_url("https://gitlab.com/a/b")


# ===========================================================================
# format_container_status_response
# ===========================================================================


class TestFormatContainerStatusResponse:
    def test_full_attributes(self):
        container = SimpleNamespace(
            container_name="ctr-1",
            status=SimpleNamespace(value="running"),
            repo_url="https://github.com/o/r",
            branch="main",
            api_endpoint="http://localhost:8080",
            internal_endpoint="http://internal:8080",
            public_endpoint="http://public:8080",
            code_graph_id="cg-1",
            description="Test container",
        )
        result = format_container_status_response(container)
        assert result["container_name"] == "ctr-1"
        assert result["status"] == "running"
        assert result["repo_url"] == "https://github.com/o/r"
        assert result["branch"] == "main"
        assert result["api_endpoint"] == "http://localhost:8080"
        assert result["internal_endpoint"] == "http://internal:8080"
        assert result["public_endpoint"] == "http://public:8080"
        assert result["code_graph_id"] == "cg-1"
        assert result["description"] == "Test container"
        assert "running" in result["message"]

    def test_missing_optional_attributes(self):
        """Container with minimal attributes still formats correctly."""
        container = SimpleNamespace(
            container_name="bare-ctr",
            status="stopped",
        )
        result = format_container_status_response(container)
        assert result["container_name"] == "bare-ctr"
        assert result["status"] == "stopped"
        assert result["repo_url"] is None
        assert result["branch"] is None
        assert result["code_graph_id"] is None
        assert "stopped" in result["message"]

    def test_status_with_value_attribute(self):
        """Status with .value property uses that value."""
        container = SimpleNamespace(
            container_name="c",
            status=SimpleNamespace(value="provisioning"),
        )
        result = format_container_status_response(container)
        assert result["status"] == "provisioning"

    def test_internal_endpoint_falls_back_to_api_endpoint(self):
        """When internal_endpoint is missing, falls back to api_endpoint."""
        container = SimpleNamespace(
            container_name="c",
            status="active",
            api_endpoint="http://api:80",
        )
        result = format_container_status_response(container)
        assert result["internal_endpoint"] == "http://api:80"
        assert result["public_endpoint"] == "http://api:80"


# ===========================================================================
# validate_pr_creation_request
# ===========================================================================


class TestValidatePrCreationRequest:
    def _valid_request(self, **overrides):
        data = {
            "original_owner": "owner",
            "original_repo": "repo",
            "fork_owner": "fork-owner",
            "fork_repo": "fork-repo",
            "head_branch": "feature",
            "base_branch": "main",
            "title": "Add feature",
            "message": "This adds a feature.",
        }
        data.update(overrides)
        return data

    def test_valid_request(self):
        valid, errors = validate_pr_creation_request(self._valid_request())
        assert valid is True
        assert errors == []

    def test_missing_required_field(self):
        data = self._valid_request()
        del data["title"]
        valid, errors = validate_pr_creation_request(data)
        assert valid is False
        assert any("title" in e for e in errors)

    def test_empty_required_field(self):
        valid, errors = validate_pr_creation_request(self._valid_request(title=""))
        assert valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_whitespace_only_field(self):
        valid, errors = validate_pr_creation_request(self._valid_request(title="   "))
        assert valid is False

    def test_title_too_long(self):
        valid, errors = validate_pr_creation_request(self._valid_request(title="x" * 201))
        assert valid is False
        assert any("200" in e for e in errors)

    def test_message_too_long(self):
        valid, errors = validate_pr_creation_request(self._valid_request(message="x" * 10001))
        assert valid is False
        assert any("10,000" in e or "10000" in e for e in errors)

    def test_dangerous_chars_in_head_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(head_branch="feat;rm -rf"))
        assert valid is False
        assert any("invalid character" in e.lower() for e in errors)

    def test_dangerous_chars_in_base_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(base_branch="main|echo"))
        assert valid is False
        assert any("invalid character" in e.lower() for e in errors)

    def test_pipe_in_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(head_branch="a||b"))
        assert valid is False

    def test_ampersand_in_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(base_branch="a&&b"))
        assert valid is False

    def test_backtick_in_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(head_branch="feat`id`"))
        assert valid is False

    def test_dollar_in_branch(self):
        valid, errors = validate_pr_creation_request(self._valid_request(head_branch="feat$HOME"))
        assert valid is False

    def test_multiple_missing_fields(self):
        valid, errors = validate_pr_creation_request({})
        assert valid is False
        assert len(errors) == 8  # all 8 required fields missing


# ===========================================================================
# sanitize_log_data
# ===========================================================================


class TestSanitizeLogData:
    def test_masks_access_token(self):
        data = {"access_token": "ghp_1234567890abcdef"}
        result = sanitize_log_data(data)
        assert "ghp_" in result["access_token"]
        assert "cdef" in result["access_token"]
        assert "***" in result["access_token"]

    def test_masks_password(self):
        data = {"password": "supersecret123"}
        result = sanitize_log_data(data)
        assert "***" in result["password"]
        assert result["password"] != "supersecret123"

    def test_masks_short_token(self):
        data = {"token": "short"}
        result = sanitize_log_data(data)
        assert result["token"] == "***masked***"

    def test_leaves_non_sensitive_fields(self):
        data = {"username": "testuser", "repo": "my-repo"}
        result = sanitize_log_data(data)
        assert result["username"] == "testuser"
        assert result["repo"] == "my-repo"

    def test_non_dict_returns_as_is(self):
        assert sanitize_log_data("string") == "string"
        assert sanitize_log_data(42) == 42

    def test_does_not_mutate_original(self):
        data = {"token": "my-secret-token-value"}
        _ = sanitize_log_data(data)
        assert data["token"] == "my-secret-token-value"

    def test_empty_sensitive_field_not_masked(self):
        data = {"token": ""}
        result = sanitize_log_data(data)
        assert result["token"] == ""

    def test_none_sensitive_field_not_masked(self):
        data = {"token": None}
        result = sanitize_log_data(data)
        assert result["token"] is None

    def test_masks_authorization(self):
        data = {"authorization": "Bearer abc123xyz789"}
        result = sanitize_log_data(data)
        assert "***" in result["authorization"]

    def test_masks_secret(self):
        data = {"secret": "webhook-secret-12345"}
        result = sanitize_log_data(data)
        assert "***" in result["secret"]


# ===========================================================================
# calculate_validation_score
# ===========================================================================


class TestCalculateValidationScore:
    def test_no_issues_returns_100(self):
        assert calculate_validation_score([]) == 100.0

    def test_critical_issue(self):
        issue = SimpleNamespace(severity="critical")
        score = calculate_validation_score([issue])
        assert score == 70.0  # 100 - 30

    def test_high_issue(self):
        issue = SimpleNamespace(severity="high")
        score = calculate_validation_score([issue])
        assert score == 85.0  # 100 - 15

    def test_medium_issue(self):
        issue = SimpleNamespace(severity="medium")
        score = calculate_validation_score([issue])
        assert score == 90.0  # 100 - 10

    def test_low_issue(self):
        issue = SimpleNamespace(severity="low")
        score = calculate_validation_score([issue])
        assert score == 95.0  # 100 - 5

    def test_unknown_severity_defaults_to_10(self):
        issue = SimpleNamespace(severity="unknown")
        score = calculate_validation_score([issue])
        assert score == 90.0  # 100 - 10 (default)

    def test_score_never_below_zero(self):
        issues = [SimpleNamespace(severity="critical") for _ in range(10)]
        score = calculate_validation_score(issues)
        assert score == 0.0

    def test_multiple_mixed_issues(self):
        issues = [
            SimpleNamespace(severity="critical"),  # -30
            SimpleNamespace(severity="high"),  # -15
            SimpleNamespace(severity="low"),  # -5
        ]
        score = calculate_validation_score(issues)
        assert score == 50.0  # 100 - 50

    def test_issue_without_severity_attr(self):
        """Issues without severity attribute default to 'medium' penalty (10)."""
        issue = SimpleNamespace()  # no severity attr
        score = calculate_validation_score([issue])
        assert score == 90.0


# ===========================================================================
# determine_pr_recommendation
# ===========================================================================


class TestDeterminePrRecommendation:
    def test_approve_high_score(self):
        assert determine_pr_recommendation(90.0, False) == "approve"

    def test_approve_at_80(self):
        assert determine_pr_recommendation(80.0, False) == "approve"

    def test_needs_work_at_70(self):
        assert determine_pr_recommendation(70.0, False) == "needs_work"

    def test_needs_work_at_60(self):
        assert determine_pr_recommendation(60.0, False) == "needs_work"

    def test_reject_below_60(self):
        assert determine_pr_recommendation(59.0, False) == "reject"

    def test_reject_at_0(self):
        assert determine_pr_recommendation(0.0, False) == "reject"

    def test_critical_issues_always_reject(self):
        assert determine_pr_recommendation(100.0, True) == "reject"

    def test_critical_overrides_high_score(self):
        assert determine_pr_recommendation(95.0, True) == "reject"


# ===========================================================================
# format_github_error_message
# ===========================================================================


class TestFormatGithubErrorMessage:
    def test_404_error(self):
        msg = format_github_error_message(Exception("404 Not Found"))
        assert "not found" in msg.lower()

    def test_not_found_text(self):
        msg = format_github_error_message(Exception("resource not found"))
        assert "not found" in msg.lower()

    def test_403_error(self):
        msg = format_github_error_message(Exception("403 Forbidden"))
        assert "denied" in msg.lower() or "permission" in msg.lower()

    def test_401_error(self):
        msg = format_github_error_message(Exception("401 Unauthorized"))
        assert "authentication" in msg.lower()

    def test_rate_limit(self):
        msg = format_github_error_message(Exception("rate limit exceeded"))
        assert "rate limit" in msg.lower()

    def test_timeout(self):
        msg = format_github_error_message(Exception("connection timeout"))
        assert "timed out" in msg.lower()

    def test_generic_error(self):
        msg = format_github_error_message(Exception("something weird happened"))
        assert "GitHub API error" in msg
        assert "something weird happened" in msg


# ===========================================================================
# validate_file_changes
# ===========================================================================


class TestValidateFileChanges:
    def test_valid_changes(self):
        changes = [{"file_path": "src/main.py", "content": "print('hi')"}]
        valid, errors = validate_file_changes(changes)
        assert valid is True
        assert errors == []

    def test_empty_list(self):
        valid, errors = validate_file_changes([])
        assert valid is False
        assert any("at least one" in e.lower() for e in errors)

    def test_too_many_changes(self):
        changes = [{"file_path": f"file{i}.py", "content": "x"} for i in range(51)]
        valid, errors = validate_file_changes(changes)
        assert valid is False
        assert any("maximum 50" in e for e in errors)

    def test_non_dict_change(self):
        valid, errors = validate_file_changes(["not-a-dict"])
        assert valid is False
        assert any("object" in e.lower() for e in errors)

    def test_missing_file_path(self):
        valid, errors = validate_file_changes([{"content": "x"}])
        assert valid is False
        assert any("file_path" in e for e in errors)

    def test_empty_file_path(self):
        valid, errors = validate_file_changes([{"file_path": "", "content": "x"}])
        assert valid is False
        assert any("file_path" in e for e in errors)

    def test_missing_content(self):
        valid, errors = validate_file_changes([{"file_path": "a.py"}])
        assert valid is False
        assert any("content" in e for e in errors)

    def test_absolute_path_rejected(self):
        valid, errors = validate_file_changes([{"file_path": "/etc/passwd", "content": "x"}])
        assert valid is False
        assert any("invalid" in e.lower() or "parent" in e.lower() for e in errors)

    def test_parent_traversal_rejected(self):
        valid, errors = validate_file_changes([{"file_path": "../secret.txt", "content": "x"}])
        assert valid is False
        assert any("parent" in e.lower() or "invalid" in e.lower() for e in errors)

    def test_null_byte_rejected(self):
        valid, errors = validate_file_changes([{"file_path": "file\x00.py", "content": "x"}])
        assert valid is False

    def test_content_too_large(self):
        huge = "x" * (1024 * 1024 + 1)
        valid, errors = validate_file_changes([{"file_path": "big.py", "content": huge}])
        assert valid is False
        assert any("1MB" in e for e in errors)

    def test_multiple_valid_changes(self):
        changes = [
            {"file_path": "a.py", "content": "a"},
            {"file_path": "b.py", "content": "b"},
            {"file_path": "c.py", "content": "c"},
        ]
        valid, errors = validate_file_changes(changes)
        assert valid is True

    def test_multiple_errors_accumulated(self):
        changes = [
            {"content": "no path"},  # missing file_path
            {"file_path": "ok.py"},  # missing content
        ]
        valid, errors = validate_file_changes(changes)
        assert valid is False
        assert len(errors) >= 2

    def test_non_string_file_path_triggers_path_error(self):
        """Non-string file_path value triggers the ValueError/TypeError handler."""
        changes = [{"file_path": 12345, "content": "x"}]
        valid, errors = validate_file_changes(changes)
        # Should still produce an error (either path validation or empty-path error)
        assert valid is False


# ===========================================================================
# GitHubOperationContext
# ===========================================================================


class TestGitHubOperationContext:
    def test_enter_returns_self(self):
        ctx = GitHubOperationContext("test-op")
        result = ctx.__enter__()
        assert result is ctx
        ctx.__exit__(None, None, None)

    def test_exit_returns_false(self):
        """Context manager should not suppress exceptions."""
        ctx = GitHubOperationContext("op")
        ctx.__enter__()
        assert ctx.__exit__(None, None, None) is False

    def test_exit_with_exception_returns_false(self):
        """Even with exceptions, context should not suppress them."""
        ctx = GitHubOperationContext("op")
        ctx.__enter__()
        assert ctx.__exit__(ValueError, ValueError("boom"), None) is False

    def test_with_container_name(self):
        ctx = GitHubOperationContext("clone", container_name="my-ctr")
        assert ctx.operation_name == "clone"
        assert ctx.container_name == "my-ctr"

    def test_context_manager_protocol(self):
        """Can be used as a standard context manager."""
        with GitHubOperationContext("test-op", container_name="c1") as ctx:
            assert ctx.operation_name == "test-op"
            assert ctx.container_name == "c1"

    def test_exception_propagates(self):
        """Exceptions raised inside the context should propagate."""
        with pytest.raises(RuntimeError, match="boom"):
            with GitHubOperationContext("op"):
                raise RuntimeError("boom")
