"""Tests for routers/github_authentication.py — comprehensive coverage."""

from __future__ import annotations

from unittest.mock import patch

# ── validate_access_token ──────────────────────────────────────────────


class TestValidateAccessToken:
    def test_none_token_returns_false(self):
        from routers.github_authentication import validate_access_token

        assert validate_access_token(None) is False

    def test_empty_string_returns_false(self):
        from routers.github_authentication import validate_access_token

        assert validate_access_token("") is False

    def test_valid_token_delegates_to_validate_github_token(self):
        from routers.github_authentication import validate_access_token

        with patch("routers.github_authentication.validate_github_token", return_value=True):
            assert validate_access_token("ghp_abc123") is True

    def test_invalid_token_returns_false(self):
        from routers.github_authentication import validate_access_token

        with patch("routers.github_authentication.validate_github_token", return_value=False):
            assert validate_access_token("bad-token") is False


# ── validate_repository_url ────────────────────────────────────────────


class TestValidateRepositoryUrl:
    def test_valid_https_url(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("https://github.com/owner/repo") is True

    def test_valid_http_url(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("http://github.com/owner/repo") is True

    def test_invalid_url_gitlab(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("https://gitlab.com/owner/repo") is False

    def test_empty_string(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("") is False

    def test_none_value(self):
        from routers.github_authentication import validate_repository_url

        # None is not a str but the function checks isinstance
        assert validate_repository_url(None) is False

    def test_non_string_value(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url(12345) is False

    def test_random_string(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("not-a-url") is False

    def test_github_url_without_full_prefix(self):
        from routers.github_authentication import validate_repository_url

        assert validate_repository_url("github.com/owner/repo") is False


# ── sanitize_container_name ────────────────────────────────────────────


class TestSanitizeContainerName:
    def test_empty_string(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("") == ""

    def test_none_value(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name(None) == ""

    def test_simple_name(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("mycontainer") == "mycontainer"

    def test_uppercase_converted_to_lowercase(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("MyContainer") == "mycontainer"

    def test_special_characters_removed(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("my@container!") == "mycontainer"

    def test_hyphens_and_underscores_preserved(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("my-container_1") == "my-container_1"

    def test_leading_dashes_stripped(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("--mycontainer") == "mycontainer"

    def test_leading_underscores_stripped(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("__mycontainer") == "mycontainer"

    def test_leading_mixed_stripped(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("-_-name") == "name"

    def test_truncated_to_63_chars(self):
        from routers.github_authentication import sanitize_container_name

        long_name = "a" * 100
        result = sanitize_container_name(long_name)
        assert len(result) == 63

    def test_all_invalid_chars_returns_empty(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("!!!") == ""

    def test_only_dashes_and_underscores_returns_empty(self):
        from routers.github_authentication import sanitize_container_name

        assert sanitize_container_name("---___") == ""


# ── validate_github_credentials ────────────────────────────────────────


class TestValidateGithubCredentials:
    def test_no_token_provided(self):
        from routers.github_authentication import validate_github_credentials

        result = validate_github_credentials(None)
        assert result["valid"] is False
        assert result["token_provided"] is False
        assert result["token_valid"] is False
        assert result["message"] == "No GitHub access token provided"

    def test_empty_token(self):
        from routers.github_authentication import validate_github_credentials

        result = validate_github_credentials("")
        assert result["valid"] is False
        assert result["token_provided"] is False

    def test_valid_token(self):
        from routers.github_authentication import validate_github_credentials

        with patch("routers.github_authentication.validate_github_token", return_value=True):
            result = validate_github_credentials("ghp_valid123")
        assert result["valid"] is True
        assert result["token_provided"] is True
        assert result["token_valid"] is True
        assert result["message"] == "GitHub credentials are valid"

    def test_invalid_token(self):
        from routers.github_authentication import validate_github_credentials

        with patch("routers.github_authentication.validate_github_token", return_value=False):
            result = validate_github_credentials("bad-token")
        assert result["valid"] is False
        assert result["token_provided"] is True
        assert result["token_valid"] is False
        assert result["message"] == "Invalid or expired GitHub access token"


# ── check_repository_access_permissions ────────────────────────────────


class TestCheckRepositoryAccessPermissions:
    def test_invalid_url_returns_not_accessible(self):
        from routers.github_authentication import check_repository_access_permissions

        result = check_repository_access_permissions("not-a-url")
        assert result["accessible"] is False
        assert result["message"] == "Invalid GitHub repository URL"

    def test_valid_url_with_valid_token(self):
        from routers.github_authentication import check_repository_access_permissions

        with patch("routers.github_authentication.validate_github_token", return_value=True):
            result = check_repository_access_permissions(
                "https://github.com/owner/repo", "ghp_valid"
            )
        assert result["accessible"] is True
        assert result["message"] == "Repository accessible with provided token"

    def test_valid_url_with_invalid_token(self):
        from routers.github_authentication import check_repository_access_permissions

        with patch("routers.github_authentication.validate_github_token", return_value=False):
            result = check_repository_access_permissions("https://github.com/owner/repo", "bad")
        assert result["accessible"] is True
        assert result["public_repo"] is True
        # token was provided but invalid — token_required stays False
        assert result["token_required"] is False

    def test_valid_url_without_token(self):
        from routers.github_authentication import check_repository_access_permissions

        result = check_repository_access_permissions("https://github.com/owner/repo")
        assert result["accessible"] is True
        assert result["public_repo"] is True
        assert result["token_required"] is True
        assert "token may be required" in result["message"]

    def test_valid_url_with_none_token(self):
        from routers.github_authentication import check_repository_access_permissions

        result = check_repository_access_permissions("https://github.com/owner/repo", None)
        assert result["accessible"] is True
        assert result["token_required"] is True

    def test_empty_url(self):
        from routers.github_authentication import check_repository_access_permissions

        result = check_repository_access_permissions("")
        assert result["accessible"] is False


# ── generate_secure_container_name ─────────────────────────────────────


class TestGenerateSecureContainerName:
    def test_empty_url(self):
        from routers.github_authentication import generate_secure_container_name

        assert generate_secure_container_name("") == ""

    def test_basic_repo_url(self):
        from routers.github_authentication import generate_secure_container_name

        result = generate_secure_container_name("https://github.com/owner/myrepo")
        assert result == "myrepo-main"

    def test_custom_branch(self):
        from routers.github_authentication import generate_secure_container_name

        result = generate_secure_container_name("https://github.com/owner/myrepo", "feature")
        assert result == "myrepo-feature"

    def test_git_suffix_stripped(self):
        from routers.github_authentication import generate_secure_container_name

        result = generate_secure_container_name("https://github.com/owner/myrepo.git")
        assert result == "myrepo-main"

    def test_trailing_slash_handled(self):
        from routers.github_authentication import generate_secure_container_name

        result = generate_secure_container_name("https://github.com/owner/myrepo/")
        assert result == "myrepo-main"

    def test_result_is_sanitized(self):
        from routers.github_authentication import generate_secure_container_name

        # Uppercase should be lowered, special chars removed
        result = generate_secure_container_name(
            "https://github.com/owner/My-Repo", "Feature/Branch"
        )
        # sanitize_container_name removes '/' so "my-repo-featurebranch"
        assert result == "my-repo-featurebranch"


# ── is_safe_branch_name ───────────────────────────────────────────────


class TestIsSafeBranchName:
    def test_safe_branch(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("feature/add-login") is True

    def test_empty_string(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("") is False

    def test_none_value(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name(None) is False

    def test_non_string(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name(123) is False

    def test_semicolon_injection(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main; rm -rf /") is False

    def test_ampersand(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main&echo hacked") is False

    def test_pipe(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main|cat /etc/passwd") is False

    def test_backtick(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main`whoami`") is False

    def test_dollar_sign(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main$HOME") is False

    def test_parentheses(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main()") is False

    def test_curly_braces(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main{}") is False

    def test_square_brackets(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main[]") is False

    def test_angle_brackets(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main<>") is False

    def test_double_ampersand_pattern(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main&&echo") is False

    def test_double_pipe_pattern(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main||echo") is False

    def test_command_substitution_dollar_paren(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main$(whoami)") is False

    def test_variable_expansion(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main${HOME}") is False

    def test_redirect_append(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main>>file") is False

    def test_redirect_heredoc(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("main<<EOF") is False

    def test_normal_branch_with_slashes(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("feature/my-branch") is True

    def test_normal_branch_with_dots(self):
        from routers.github_authentication import is_safe_branch_name

        assert is_safe_branch_name("release-1.0.0") is True


# ── validate_pr_request_fields ─────────────────────────────────────────


class TestValidatePrRequestFields:
    def _complete_fields(self, **overrides):
        fields = {
            "original_owner": "owner",
            "original_repo": "repo",
            "fork_owner": "fork-owner",
            "fork_repo": "fork-repo",
            "head_branch": "feature",
            "base_branch": "main",
            "title": "My PR Title",
            "message": "Description of changes",
        }
        fields.update(overrides)
        return fields

    def test_all_required_fields_present(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_required_field(self):
        from routers.github_authentication import validate_pr_request_fields

        fields = self._complete_fields()
        del fields["title"]
        result = validate_pr_request_fields(fields)
        assert result["valid"] is False
        assert any("title" in e for e in result["errors"])

    def test_empty_required_field(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(original_owner=""))
        assert result["valid"] is False
        assert any("original_owner" in e for e in result["errors"])

    def test_whitespace_only_required_field(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(title="   "))
        assert result["valid"] is False
        assert any("title" in e for e in result["errors"])

    def test_multiple_missing_fields(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields({})
        assert result["valid"] is False
        assert len(result["errors"]) == 8  # all 8 required fields

    def test_unsafe_head_branch(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(head_branch="main;rm -rf /"))
        assert result["valid"] is False
        assert any("head_branch" in e for e in result["errors"])

    def test_unsafe_base_branch(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(base_branch="main$(whoami)"))
        assert result["valid"] is False
        assert any("base_branch" in e for e in result["errors"])

    def test_long_title_generates_warning(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(title="x" * 201))
        # Warnings, but still valid
        assert result["valid"] is True
        assert any("title" in w for w in result["warnings"])

    def test_long_message_generates_warning(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(message="x" * 10001))
        assert result["valid"] is True
        assert any("message" in w for w in result["warnings"])

    def test_fields_within_max_length_no_warnings(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields())
        assert result["warnings"] == []

    def test_long_owner_field_generates_warning(self):
        from routers.github_authentication import validate_pr_request_fields

        result = validate_pr_request_fields(self._complete_fields(original_owner="o" * 101))
        assert any("original_owner" in w for w in result["warnings"])


# ── mask_sensitive_data ────────────────────────────────────────────────


class TestMaskSensitiveData:
    def test_non_dict_returned_as_is(self):
        from routers.github_authentication import mask_sensitive_data

        assert mask_sensitive_data("a string") == "a string"
        assert mask_sensitive_data(42) == 42
        assert mask_sensitive_data(None) is None

    def test_no_sensitive_fields(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"name": "test", "value": "ok"}
        result = mask_sensitive_data(data)
        assert result == data

    def test_long_token_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"access_token": "ghp_abcdefghijklmnop"}
        result = mask_sensitive_data(data)
        assert result["access_token"] == "ghp_...mnop"

    def test_short_token_fully_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": "short"}
        result = mask_sensitive_data(data)
        assert result["token"] == "***"

    def test_exactly_8_char_token_masked_as_stars(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": "12345678"}
        result = mask_sensitive_data(data)
        assert result["token"] == "***"

    def test_9_char_token_partially_shown(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": "123456789"}
        result = mask_sensitive_data(data)
        assert result["token"] == "1234...6789"

    def test_password_field_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"password": "mysecretpassword"}
        result = mask_sensitive_data(data)
        assert "..." in result["password"]

    def test_secret_field_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"secret": "topsecretvalue123"}
        result = mask_sensitive_data(data)
        assert "..." in result["secret"]

    def test_key_field_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"key": "my-api-key-value"}
        result = mask_sensitive_data(data)
        assert "..." in result["key"]

    def test_original_dict_not_mutated(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": "ghp_abcdefghijklmnop"}
        mask_sensitive_data(data)
        assert data["token"] == "ghp_abcdefghijklmnop"

    def test_empty_sensitive_field_not_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": ""}
        result = mask_sensitive_data(data)
        # empty/falsy values are not masked
        assert result["token"] == ""

    def test_none_sensitive_field_not_masked(self):
        from routers.github_authentication import mask_sensitive_data

        data = {"token": None}
        result = mask_sensitive_data(data)
        assert result["token"] is None


# ── create_validation_error ────────────────────────────────────────────


class TestCreateValidationError:
    def test_basic_error(self):
        from routers.github_authentication import create_validation_error

        result = create_validation_error("Something went wrong")
        assert result["error"] == "validation_error"
        assert result["message"] == "Something went wrong"
        assert "timestamp" in result
        assert "field" not in result

    def test_error_with_field(self):
        from routers.github_authentication import create_validation_error

        result = create_validation_error("Bad value", field="email")
        assert result["field"] == "email"
        assert result["error"] == "validation_error"

    def test_error_without_field(self):
        from routers.github_authentication import create_validation_error

        result = create_validation_error("General error", field=None)
        assert "field" not in result


# ── create_auth_error ──────────────────────────────────────────────────


class TestCreateAuthError:
    def test_basic_auth_error(self):
        from routers.github_authentication import create_auth_error

        result = create_auth_error("Unauthorized access")
        assert result["error"] == "authentication_error"
        assert result["message"] == "Unauthorized access"
        assert "timestamp" in result

    def test_auth_error_structure(self):
        from routers.github_authentication import create_auth_error

        result = create_auth_error("Token expired")
        assert set(result.keys()) == {"error", "message", "timestamp"}
