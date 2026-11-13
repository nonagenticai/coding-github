"""
Tests for GitHub Authentication and Validation Utilities

Comprehensive test suite for github_authentication.py covering:
- Token validation
- Repository URL validation
- Container name sanitization
- GitHub credentials validation
- Repository access permissions
- Secure container name generation
- Branch name safety validation
- PR request field validation
- Sensitive data masking

These are pure ``unittest.mock`` unit tests: they exercise standalone helper
functions and need no running service, database, or network. They are NOT
integration tests, so they carry no ``pytest.mark.integration`` marker and run
as part of the default CI suite.
"""

from unittest.mock import patch

from routers.github_authentication import (
    check_repository_access_permissions,
    create_auth_error,
    create_validation_error,
    generate_secure_container_name,
    is_safe_branch_name,
    mask_sensitive_data,
    sanitize_container_name,
    validate_access_token,
    validate_github_credentials,
    validate_pr_request_fields,
    validate_repository_url,
)


class TestValidateAccessToken:
    """Test suite for validate_access_token function."""

    @patch("routers.github_authentication.validate_github_token")
    def test_validate_access_token_valid(self, mock_validate):
        """Test validation with valid token."""
        mock_validate.return_value = True
        assert validate_access_token("ghp_validtoken123") is True
        mock_validate.assert_called_once_with("ghp_validtoken123")

    @patch("routers.github_authentication.validate_github_token")
    def test_validate_access_token_invalid(self, mock_validate):
        """Test validation with invalid token."""
        mock_validate.return_value = False
        assert validate_access_token("invalid_token") is False

    def test_validate_access_token_none(self):
        """Test validation with None token."""
        assert validate_access_token(None) is False

    def test_validate_access_token_empty_string(self):
        """Test validation with empty string token."""
        assert validate_access_token("") is False


class TestValidateRepositoryUrl:
    """Test suite for validate_repository_url function."""

    def test_validate_repository_url_https_valid(self):
        """Test validation with valid HTTPS GitHub URL."""
        assert validate_repository_url("https://github.com/owner/repo") is True

    def test_validate_repository_url_http_valid(self):
        """Test validation with valid HTTP GitHub URL."""
        assert validate_repository_url("http://github.com/owner/repo") is True

    def test_validate_repository_url_invalid_domain(self):
        """Test validation with non-GitHub URL."""
        assert validate_repository_url("https://gitlab.com/owner/repo") is False

    def test_validate_repository_url_empty_string(self):
        """Test validation with empty string."""
        assert validate_repository_url("") is False

    def test_validate_repository_url_none(self):
        """Test validation with None."""
        assert validate_repository_url(None) is False  # type: ignore[invalid-argument-type]

    def test_validate_repository_url_not_string(self):
        """Test validation with non-string type."""
        assert validate_repository_url(123) is False  # type: ignore[invalid-argument-type]


class TestSanitizeContainerName:
    """Test suite for sanitize_container_name function."""

    def test_sanitize_container_name_valid(self):
        """Test sanitization with valid name."""
        assert sanitize_container_name("my-container") == "my-container"

    def test_sanitize_container_name_uppercase(self):
        """Test sanitization converts uppercase to lowercase."""
        assert sanitize_container_name("MyContainer") == "mycontainer"

    def test_sanitize_container_name_special_chars(self):
        """Test sanitization removes special characters."""
        assert sanitize_container_name("my@container!") == "mycontainer"

    def test_sanitize_container_name_leading_dash(self):
        """Test sanitization removes leading dashes."""
        assert sanitize_container_name("--container") == "container"

    def test_sanitize_container_name_leading_underscore(self):
        """Test sanitization removes leading underscores."""
        assert sanitize_container_name("__container") == "container"

    def test_sanitize_container_name_mixed_invalid(self):
        """Test sanitization with mixed invalid characters."""
        assert sanitize_container_name("_-My@Container#123!") == "mycontainer123"

    def test_sanitize_container_name_length_limit(self):
        """Test sanitization enforces 63 character limit."""
        long_name = "a" * 100
        result = sanitize_container_name(long_name)
        assert len(result) == 63

    def test_sanitize_container_name_empty_string(self):
        """Test sanitization with empty string."""
        assert sanitize_container_name("") == ""

    def test_sanitize_container_name_all_invalid_chars(self):
        """Test sanitization with only invalid characters."""
        assert sanitize_container_name("@#$%^&*()") == ""


class TestValidateGithubCredentials:
    """Test suite for validate_github_credentials function."""

    @patch("routers.github_authentication.validate_github_token")
    def test_validate_github_credentials_valid_token(self, mock_validate):
        """Test credentials validation with valid token."""
        mock_validate.return_value = True
        result = validate_github_credentials("ghp_validtoken")

        assert result["valid"] is True
        assert result["token_provided"] is True
        assert result["token_valid"] is True
        assert result["message"] == "GitHub credentials are valid"

    @patch("routers.github_authentication.validate_github_token")
    def test_validate_github_credentials_invalid_token(self, mock_validate):
        """Test credentials validation with invalid token."""
        mock_validate.return_value = False
        result = validate_github_credentials("invalid_token")

        assert result["valid"] is False
        assert result["token_provided"] is True
        assert result["token_valid"] is False
        assert result["message"] == "Invalid or expired GitHub access token"

    def test_validate_github_credentials_no_token(self):
        """Test credentials validation with no token."""
        result = validate_github_credentials(None)

        assert result["valid"] is False
        assert result["token_provided"] is False
        assert result["token_valid"] is False
        assert result["message"] == "No GitHub access token provided"


class TestCheckRepositoryAccessPermissions:
    """Test suite for check_repository_access_permissions function."""

    def test_check_repository_access_permissions_invalid_url(self):
        """Test repository access check with invalid URL."""
        result = check_repository_access_permissions("https://gitlab.com/owner/repo")

        assert result["accessible"] is False
        assert result["message"] == "Invalid GitHub repository URL"

    @patch("routers.github_authentication.validate_github_token")
    def test_check_repository_access_permissions_with_valid_token(self, mock_validate):
        """Test repository access check with valid token."""
        mock_validate.return_value = True
        result = check_repository_access_permissions(
            "https://github.com/owner/repo", "ghp_validtoken"
        )

        assert result["accessible"] is True
        assert result["message"] == "Repository accessible with provided token"

    @patch("routers.github_authentication.validate_github_token")
    def test_check_repository_access_permissions_public_repo(self, mock_validate):
        """Test repository access check for public repo without token."""
        mock_validate.return_value = False
        result = check_repository_access_permissions("https://github.com/owner/repo", None)

        assert result["accessible"] is True
        assert result["public_repo"] is True
        assert result["token_required"] is True
        assert "public repository access" in result["message"]


class TestGenerateSecureContainerName:
    """Test suite for generate_secure_container_name function."""

    def test_generate_secure_container_name_basic(self):
        """Test container name generation with basic URL."""
        name = generate_secure_container_name("https://github.com/owner/repo", "main")
        assert name == "repo-main"

    def test_generate_secure_container_name_with_git_extension(self):
        """Test container name generation with .git extension."""
        name = generate_secure_container_name("https://github.com/owner/repo.git", "develop")
        assert name == "repo-develop"

    def test_generate_secure_container_name_special_branch(self):
        """Test container name generation with special characters in branch."""
        name = generate_secure_container_name("https://github.com/owner/repo", "feature/my-feature")
        assert name == "repo-featuremy-feature"

    def test_generate_secure_container_name_empty_url(self):
        """Test container name generation with empty URL."""
        name = generate_secure_container_name("", "main")
        assert name == ""


class TestIsSafeBranchName:
    """Test suite for is_safe_branch_name function."""

    def test_is_safe_branch_name_valid(self):
        """Test branch name validation with valid name."""
        assert is_safe_branch_name("main") is True
        assert is_safe_branch_name("feature-123") is True
        assert is_safe_branch_name("release/v1.0.0") is True

    def test_is_safe_branch_name_dangerous_chars(self):
        """Test branch name validation with dangerous characters."""
        assert is_safe_branch_name("branch;rm -rf") is False
        assert is_safe_branch_name("branch&whoami") is False
        assert is_safe_branch_name("branch|ls") is False
        assert is_safe_branch_name("branch`echo`") is False

    def test_is_safe_branch_name_command_injection(self):
        """Test branch name validation with command injection patterns."""
        assert is_safe_branch_name("branch&&ls") is False
        assert is_safe_branch_name("branch||pwd") is False
        assert is_safe_branch_name("branch$(whoami)") is False

    def test_is_safe_branch_name_empty_string(self):
        """Test branch name validation with empty string."""
        assert is_safe_branch_name("") is False

    def test_is_safe_branch_name_none(self):
        """Test branch name validation with None."""
        assert is_safe_branch_name(None) is False  # type: ignore[invalid-argument-type]

    def test_is_safe_branch_name_not_string(self):
        """Test branch name validation with non-string type."""
        assert is_safe_branch_name(123) is False  # type: ignore[invalid-argument-type]


class TestValidatePrRequestFields:
    """Test suite for validate_pr_request_fields function."""

    def test_validate_pr_request_fields_valid(self):
        """Test PR field validation with all valid fields."""
        fields = {
            "original_owner": "owner1",
            "original_repo": "repo1",
            "fork_owner": "owner2",
            "fork_repo": "repo2",
            "head_branch": "feature",
            "base_branch": "main",
            "title": "Test PR",
            "message": "PR description",
        }
        result = validate_pr_request_fields(fields)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_pr_request_fields_missing_required(self):
        """Test PR field validation with missing required fields."""
        fields = {
            "original_owner": "owner1",
            "title": "Test PR",
        }
        result = validate_pr_request_fields(fields)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_pr_request_fields_unsafe_branch(self):
        """Test PR field validation with unsafe branch name."""
        fields = {
            "original_owner": "owner1",
            "original_repo": "repo1",
            "fork_owner": "owner2",
            "fork_repo": "repo2",
            "head_branch": "feature;rm -rf /",
            "base_branch": "main",
            "title": "Test PR",
            "message": "PR description",
        }
        result = validate_pr_request_fields(fields)

        assert result["valid"] is False
        assert any("Unsafe branch name" in error for error in result["errors"])

    def test_validate_pr_request_fields_excessive_length(self):
        """Test PR field validation with excessively long fields."""
        fields = {
            "original_owner": "owner1",
            "original_repo": "repo1",
            "fork_owner": "owner2",
            "fork_repo": "repo2",
            "head_branch": "feature",
            "base_branch": "main",
            "title": "A" * 250,  # Exceeds max 200
            "message": "PR description",
        }
        result = validate_pr_request_fields(fields)

        # Should be valid but with warnings
        assert len(result["warnings"]) > 0
        assert any("longer than recommended" in warning for warning in result["warnings"])


class TestMaskSensitiveData:
    """Test suite for mask_sensitive_data function."""

    def test_mask_sensitive_data_token(self):
        """Test masking of access tokens."""
        data = {"access_token": "ghp_1234567890abcdefghij"}
        result = mask_sensitive_data(data)

        assert result["access_token"] == "ghp_...ghij"

    def test_mask_sensitive_data_short_token(self):
        """Test masking of short tokens."""
        data = {"token": "short"}
        result = mask_sensitive_data(data)

        assert result["token"] == "***"

    def test_mask_sensitive_data_multiple_fields(self):
        """Test masking of multiple sensitive fields."""
        data = {
            "access_token": "ghp_1234567890abcdefghij",
            "password": "secretpassword123",
            "username": "john_doe",  # Should not be masked
        }
        result = mask_sensitive_data(data)

        assert result["access_token"] == "ghp_...ghij"
        assert result["password"] == "secr...d123"
        assert result["username"] == "john_doe"

    def test_mask_sensitive_data_not_dict(self):
        """Test masking with non-dict input."""
        data = "not a dict"
        result = mask_sensitive_data(data)  # type: ignore[invalid-argument-type]

        assert result == "not a dict"


class TestCreateValidationError:
    """Test suite for create_validation_error function."""

    def test_create_validation_error_without_field(self):
        """Test validation error creation without field."""
        error = create_validation_error("Invalid input")

        assert error["error"] == "validation_error"
        assert error["message"] == "Invalid input"
        assert "field" not in error

    def test_create_validation_error_with_field(self):
        """Test validation error creation with field."""
        error = create_validation_error("Invalid email", "email")

        assert error["error"] == "validation_error"
        assert error["message"] == "Invalid email"
        assert error["field"] == "email"


class TestCreateAuthError:
    """Test suite for create_auth_error function."""

    def test_create_auth_error(self):
        """Test auth error creation."""
        error = create_auth_error("Invalid credentials")

        assert error["error"] == "authentication_error"
        assert error["message"] == "Invalid credentials"
        assert "timestamp" in error
