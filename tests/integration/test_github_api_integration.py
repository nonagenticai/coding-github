"""Tests for GitHub API Integration utilities.

This test suite validates the helper functions and utilities in github_api_integration.py,
including URL validation, repository info extraction, PR request validation, and more.

These are pure ``unittest.mock`` unit tests: they exercise standalone helper
functions and need no running service, database, or network. They are NOT
integration tests, so they carry no ``pytest.mark.integration`` marker and run
as part of the default CI suite.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from routers.github_api_integration import (
    GitHubOperationContext,
    calculate_validation_score,
    extract_repo_info_from_url,
    get_container_by_name_or_id,
    sanitize_log_data,
    validate_file_changes,
    validate_github_repository_url,
    validate_pr_creation_request,
)


class TestValidateGithubRepositoryUrl:
    """Test suite for validate_github_repository_url function."""

    def test_valid_https_url(self):
        """Test validation of valid HTTPS GitHub URL."""
        is_valid, error = validate_github_repository_url("https://github.com/owner/repo")
        assert is_valid is True
        assert error == ""

    def test_valid_https_url_with_git_suffix(self):
        """Test validation of GitHub URL with .git suffix."""
        is_valid, error = validate_github_repository_url("https://github.com/owner/repo.git")
        assert is_valid is True
        assert error == ""

    def test_valid_http_url(self):
        """Test validation of HTTP GitHub URL."""
        is_valid, error = validate_github_repository_url("http://github.com/owner/repo")
        assert is_valid is True
        assert error == ""

    def test_invalid_empty_string(self):
        """Test validation fails for empty string."""
        is_valid, error = validate_github_repository_url("")
        assert is_valid is False
        assert "non-empty string" in error

    def test_invalid_none_value(self):
        """Test validation fails for None value."""
        is_valid, error = validate_github_repository_url(None)  # type: ignore[invalid-argument-type]
        assert is_valid is False
        assert "non-empty string" in error

    def test_invalid_non_github_url(self):
        """Test validation fails for non-GitHub URL."""
        is_valid, error = validate_github_repository_url("https://gitlab.com/owner/repo")
        assert is_valid is False
        assert "GitHub URL" in error

    def test_invalid_missing_repo_name(self):
        """Test validation fails when repository name is missing."""
        is_valid, error = validate_github_repository_url("https://github.com/owner")
        assert is_valid is False
        assert "owner and repository name" in error

    def test_invalid_empty_owner(self):
        """Test validation fails when owner is empty."""
        is_valid, error = validate_github_repository_url("https://github.com//repo")
        assert is_valid is False
        assert "valid owner and repository name" in error

    def test_invalid_empty_repo_after_git_suffix(self):
        """Test validation fails when repo name is only .git."""
        is_valid, error = validate_github_repository_url("https://github.com/owner/.git")
        assert is_valid is False
        assert "cannot be empty" in error


class TestExtractRepoInfoFromUrl:
    """Test suite for extract_repo_info_from_url function."""

    def test_extract_from_valid_url(self):
        """Test extracting repo info from valid URL."""
        result = extract_repo_info_from_url("https://github.com/test-owner/test-repo")
        assert result["owner"] == "test-owner"
        assert result["repo"] == "test-repo"

    def test_extract_with_git_suffix(self):
        """Test extracting repo info from URL with .git suffix."""
        result = extract_repo_info_from_url("https://github.com/my-org/my-project.git")
        assert result["owner"] == "my-org"
        assert result["repo"] == "my-project"

    def test_extract_from_http_url(self):
        """Test extracting repo info from HTTP URL."""
        result = extract_repo_info_from_url("http://github.com/user/repository")
        assert result["owner"] == "user"
        assert result["repo"] == "repository"

    def test_extract_raises_on_invalid_url(self):
        """Test that ValueError is raised for invalid URL."""
        with pytest.raises(ValueError, match="non-empty string"):
            extract_repo_info_from_url("")

    def test_extract_raises_on_non_github_url(self):
        """Test that ValueError is raised for non-GitHub URL."""
        with pytest.raises(ValueError, match="GitHub URL"):
            extract_repo_info_from_url("https://bitbucket.org/owner/repo")


class TestValidatePrCreationRequest:
    """Test suite for validate_pr_creation_request function."""

    def test_valid_pr_request(self):
        """Test validation of complete and valid PR request."""
        request_data = {
            "original_owner": "upstream-org",
            "original_repo": "main-repo",
            "fork_owner": "my-org",
            "fork_repo": "forked-repo",
            "head_branch": "feature-branch",
            "base_branch": "main",
            "title": "Add new feature",
            "message": "This PR adds a new feature to the project.",
        }
        is_valid, errors = validate_pr_creation_request(request_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        request_data = {
            "original_owner": "upstream-org",
            "original_repo": "main-repo",
            # Missing fork_owner
            "fork_repo": "forked-repo",
            "head_branch": "feature-branch",
            "base_branch": "main",
            "title": "Add new feature",
            "message": "This PR adds a new feature.",
        }
        is_valid, errors = validate_pr_creation_request(request_data)
        assert is_valid is False
        assert any("fork_owner" in error for error in errors)

    def test_empty_field_value(self):
        """Test validation fails when field value is empty string."""
        request_data = {
            "original_owner": "upstream-org",
            "original_repo": "main-repo",
            "fork_owner": "my-org",
            "fork_repo": "forked-repo",
            "head_branch": "",  # Empty string
            "base_branch": "main",
            "title": "Add new feature",
            "message": "This PR adds a new feature.",
        }
        is_valid, errors = validate_pr_creation_request(request_data)
        assert is_valid is False
        assert any("head_branch" in error and "empty" in error for error in errors)

    def test_title_exceeds_max_length(self):
        """Test validation fails when title exceeds 200 characters."""
        request_data = {
            "original_owner": "upstream-org",
            "original_repo": "main-repo",
            "fork_owner": "my-org",
            "fork_repo": "forked-repo",
            "head_branch": "feature-branch",
            "base_branch": "main",
            "title": "A" * 201,  # 201 characters
            "message": "This PR adds a new feature.",
        }
        is_valid, errors = validate_pr_creation_request(request_data)
        assert is_valid is False
        assert any("title" in error and "200" in error for error in errors)

    def test_message_exceeds_max_length(self):
        """Test validation fails when message exceeds 10,000 characters."""
        request_data = {
            "original_owner": "upstream-org",
            "original_repo": "main-repo",
            "fork_owner": "my-org",
            "fork_repo": "forked-repo",
            "head_branch": "feature-branch",
            "base_branch": "main",
            "title": "Add new feature",
            "message": "X" * 10001,  # 10,001 characters
        }
        is_valid, errors = validate_pr_creation_request(request_data)
        assert is_valid is False
        assert any("message" in error and "10,000" in error for error in errors)

    def test_branch_with_dangerous_characters(self):
        """Test validation fails when branch contains dangerous characters."""
        dangerous_chars = [";", "&", "|", "`", "$", "&&", "||"]

        for char in dangerous_chars:
            request_data = {
                "original_owner": "upstream-org",
                "original_repo": "main-repo",
                "fork_owner": "my-org",
                "fork_repo": "forked-repo",
                "head_branch": f"feature{char}branch",
                "base_branch": "main",
                "title": "Add new feature",
                "message": "This PR adds a new feature.",
            }
            is_valid, errors = validate_pr_creation_request(request_data)
            assert is_valid is False
            assert any("invalid character" in error for error in errors)


class TestValidateFileChanges:
    """Test suite for validate_file_changes function."""

    def test_valid_file_changes(self):
        """Test validation of valid file changes."""
        changes = [
            {"file_path": "src/main.py", "content": "print('Hello, World!')"},
            {"file_path": "README.md", "content": "# Project Title"},
        ]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is True
        assert len(errors) == 0

    def test_empty_changes_list(self):
        """Test validation fails for empty changes list."""
        is_valid, errors = validate_file_changes([])
        assert is_valid is False
        assert any("at least one" in error.lower() for error in errors)

    def test_too_many_changes(self):
        """Test validation fails when changes exceed 50 files."""
        changes = [{"file_path": f"file_{i}.txt", "content": "data"} for i in range(51)]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("maximum 50" in error for error in errors)

    def test_missing_file_path(self):
        """Test validation fails when file_path is missing."""
        changes = [{"content": "some content"}]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("file_path" in error for error in errors)

    def test_missing_content_field(self):
        """Test validation fails when content field is missing."""
        changes = [{"file_path": "test.txt"}]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("content" in error for error in errors)

    def test_invalid_absolute_path(self):
        """Test validation fails for absolute paths."""
        changes = [{"file_path": "/etc/passwd", "content": "malicious"}]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("Invalid file path" in error or "parent directory" in error for error in errors)

    def test_invalid_parent_directory_traversal(self):
        """Test validation fails for parent directory traversal."""
        changes = [{"file_path": "../../../etc/passwd", "content": "malicious"}]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("parent directory" in error for error in errors)

    def test_content_size_exceeds_limit(self):
        """Test validation fails when content exceeds 1MB."""
        large_content = "X" * (1024 * 1024 + 1)  # 1MB + 1 byte
        changes = [{"file_path": "large.txt", "content": large_content}]
        is_valid, errors = validate_file_changes(changes)
        assert is_valid is False
        assert any("too large" in error for error in errors)


class TestSanitizeLogData:
    """Test suite for sanitize_log_data function."""

    def test_sanitize_access_token(self):
        """Test sanitization of access_token field."""
        data = {"access_token": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"}
        sanitized = sanitize_log_data(data)
        assert sanitized["access_token"] == "ghp_***wxyz"

    def test_sanitize_password(self):
        """Test sanitization of password field."""
        data = {"password": "SuperSecretPassword123!"}
        sanitized = sanitize_log_data(data)
        # Password is 23 chars, so first 4 and last 4 are kept
        assert sanitized["password"] == "Supe***123!"

    def test_sanitize_short_secret(self):
        """Test sanitization of short secret (8 chars or less)."""
        data = {"secret": "short"}
        sanitized = sanitize_log_data(data)
        assert sanitized["secret"] == "***masked***"

    def test_no_sensitive_data(self):
        """Test that non-sensitive data is not modified."""
        data = {"username": "testuser", "repo": "my-repo"}
        sanitized = sanitize_log_data(data)
        assert sanitized["username"] == "testuser"
        assert sanitized["repo"] == "my-repo"

    def test_non_dict_input(self):
        """Test that non-dict input is returned unchanged."""
        data = "not a dictionary"
        sanitized = sanitize_log_data(data)  # type: ignore[invalid-argument-type]
        assert sanitized == "not a dictionary"

    def test_mixed_sensitive_and_normal_data(self):
        """Test sanitization of mixed data."""
        data = {
            "username": "user123",
            "token": "very_long_secret_token_value",
            "repo_url": "https://github.com/user/repo",
        }
        sanitized = sanitize_log_data(data)
        assert sanitized["username"] == "user123"
        assert sanitized["token"] == "very***alue"
        assert sanitized["repo_url"] == "https://github.com/user/repo"


class TestCalculateValidationScore:
    """Test suite for calculate_validation_score function."""

    def test_no_issues_returns_perfect_score(self):
        """Test that no issues results in 100.0 score."""
        score = calculate_validation_score([])
        assert score == pytest.approx(100.0)

    def test_single_critical_issue(self):
        """Test score calculation with one critical issue."""

        class MockIssue:
            severity = "critical"

        score = calculate_validation_score([MockIssue()])
        assert score == pytest.approx(70.0)  # 100 - 30

    def test_multiple_issues_different_severities(self):
        """Test score calculation with mixed severity issues."""

        class MockIssue:
            def __init__(self, sev):
                self.severity = sev

        issues = [
            MockIssue("critical"),  # -30
            MockIssue("high"),  # -15
            MockIssue("medium"),  # -10
        ]
        score = calculate_validation_score(issues)
        assert score == pytest.approx(45.0)  # 100 - 55

    def test_score_doesnt_go_negative(self):
        """Test that score is capped at 0."""

        class MockIssue:
            severity = "critical"

        # 5 critical issues = 150 penalty, should cap at 0
        issues = [MockIssue() for _ in range(5)]
        score = calculate_validation_score(issues)
        assert score == pytest.approx(0.0)

    def test_unknown_severity_uses_default(self):
        """Test that unknown severity defaults to medium penalty (10)."""

        class MockIssue:
            severity = "unknown"

        score = calculate_validation_score([MockIssue()])
        assert score == pytest.approx(90.0)  # 100 - 10


class TestGitHubOperationContext:
    """Test suite for GitHubOperationContext context manager."""

    def test_context_manager_enter(self, caplog):
        """Test context manager __enter__ logs operation start."""
        import logging

        caplog.set_level(logging.INFO)

        with GitHubOperationContext("test_operation") as ctx:
            assert ctx is not None

        # Check that operation was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Starting GitHub operation: test_operation" in msg for msg in log_messages)

    def test_context_manager_with_container_name(self, caplog):
        """Test context manager logs container name when provided."""
        import logging

        caplog.set_level(logging.INFO)

        with GitHubOperationContext("test_operation", container_name="test-container"):
            pass

        log_messages = [record.message for record in caplog.records]
        assert any("Container: test-container" in msg for msg in log_messages)

    def test_context_manager_successful_exit(self, caplog):
        """Test context manager logs successful completion."""
        import logging

        caplog.set_level(logging.INFO)

        with GitHubOperationContext("test_operation"):
            pass

        log_messages = [record.message for record in caplog.records]
        assert any("GitHub operation completed: test_operation" in msg for msg in log_messages)

    def test_context_manager_exception_handling(self, caplog):
        """Test context manager logs errors but doesn't suppress exceptions."""
        import logging

        caplog.set_level(logging.ERROR)

        with pytest.raises(ValueError, match="Test error"):
            with GitHubOperationContext("test_operation"):
                raise ValueError("Test error")

        log_messages = [record.message for record in caplog.records]
        assert any("GitHub operation failed: test_operation" in msg for msg in log_messages)


@pytest.mark.asyncio
class TestGetContainerByNameOrId:
    """Test suite for get_container_by_name_or_id async function."""

    async def test_find_container_by_name(self):
        """Test finding container by name successfully."""
        # Create mock container
        mock_container = MagicMock()
        mock_container.container_name = "test-container"
        mock_container.status = "running"

        # Create mock service
        mock_service = MagicMock()
        mock_service.docker_service.find_container_by_name = AsyncMock(return_value=mock_container)

        result = await get_container_by_name_or_id("test-container", mock_service, "test-tenant")

        assert result == mock_container
        mock_service.docker_service.find_container_by_name.assert_called_once_with(
            "test-container", tenant_id="test-tenant"
        )

    async def test_find_container_by_id_fallback(self):
        """Test finding container by ID when name lookup fails."""
        # Create mock container
        mock_container = MagicMock()
        mock_container.container_name = "test-container"
        mock_container.id = "abc123"

        # Create mock service - name lookup returns None, list returns container
        mock_service = MagicMock()
        mock_service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        mock_service.list_containers = AsyncMock(return_value=[mock_container])

        result = await get_container_by_name_or_id("abc123", mock_service, "test-tenant")

        assert result == mock_container

    async def test_container_not_found_raises_error(self):
        """Test that an error is raised when container not found."""
        mock_service = MagicMock()
        mock_service.docker_service.find_container_by_name = AsyncMock(return_value=None)
        mock_service.list_containers = AsyncMock(return_value=[])

        with pytest.raises((HTTPException, Exception)):
            await get_container_by_name_or_id("nonexistent", mock_service, "test-tenant")

    async def test_service_error_raises_error(self):
        """Test that service errors result in an error being raised."""
        mock_service = MagicMock()
        mock_service.docker_service.find_container_by_name = AsyncMock(
            side_effect=Exception("Docker daemon error")
        )

        with pytest.raises((HTTPException, Exception)):
            await get_container_by_name_or_id("test-container", mock_service, "test-tenant")
