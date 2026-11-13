"""Extended tests for utils/github_utils.py – targeting uncovered lines.

Uncovered lines:
  38        – repo.endswith(".git") branch inside the regex match (already stripped by regex)
  74        – validate_github_token with empty string token
  92-94     – schemathesis token acceptance in SCHEMATHESIS_TESTING mode
  118-119   – test-pattern token acceptance in TEST_MODE with debug log
  126-131   – test-pattern acceptance outside test mode & valid-prefix fallback
"""

from __future__ import annotations

import os
from unittest.mock import patch

from utils.github_utils import format_github_url, parse_github_url, validate_github_token


class TestParseGitHubUrlExtended:
    """Extra coverage for parse_github_url edge cases."""

    def test_ssh_url(self):
        """SSH git@ URL should be parsed correctly."""
        owner, repo = parse_github_url("git@github.com:orgname/myrepo.git")
        assert owner == "orgname"
        assert repo == "myrepo"

    def test_ssh_url_without_git_suffix(self):
        """SSH URL without .git should still parse."""
        owner, repo = parse_github_url("git@github.com:orgname/myrepo")
        assert owner == "orgname"
        assert repo == "myrepo"

    def test_bare_domain_url(self):
        """URL starting with github.com (no scheme) should parse."""
        owner, repo = parse_github_url("github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_bare_domain_url_with_git(self):
        """github.com URL with .git suffix should parse."""
        owner, repo = parse_github_url("github.com/owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_bare_domain_url_trailing_slash(self):
        """github.com URL with trailing slash should parse."""
        owner, repo = parse_github_url("github.com/owner/repo/")
        assert owner == "owner"
        assert repo == "repo"

    def test_http_url(self):
        """Plain http:// GitHub URL should parse."""
        owner, repo = parse_github_url("http://github.com/myorg/myrepo")
        assert owner == "myorg"
        assert repo == "myrepo"

    def test_https_with_trailing_slash(self):
        """Trailing slash on HTTPS URL."""
        owner, repo = parse_github_url("https://github.com/a/b/")
        assert owner == "a"
        assert repo == "b"

    def test_none_like_input(self):
        """Empty string returns (None, None)."""
        assert parse_github_url("") == (None, None)

    def test_completely_unrelated_string(self):
        """Random string returns (None, None)."""
        assert parse_github_url("hello world") == (None, None)

    def test_repo_name_ending_with_git_after_regex_strip(self):
        """Line 37-38: If the regex already strips .git, the inner if should not trigger.

        We construct a URL where the regex captures repo WITH a trailing .git
        that isn't consumed by the non-greedy quantifier.  In practice the regex
        already handles it, so we verify the function returns correctly for
        a repo literally named 'something.git.git' (double .git).
        """
        # The regex's non-greedy match + optional \.git means the inner
        # endswith(".git") branch (line 37-38) would only fire if the captured
        # group still contains ".git".  This can happen with the bare-domain
        # pattern if the URL is slightly unusual.  Let's try to hit it by
        # passing a URL where the regex captures the name with .git still on.
        # Actually, looking at the patterns, (?:\.git)?$ is optional so
        # the non-greedy ([^/]+?) will consume as few characters as possible.
        # For "github.com/owner/repo.git", the captured repo is "repo" (non-greedy
        # leaves .git for the optional group).  The inner if is dead code for
        # normal inputs – but we still exercise line 38 indirectly.
        owner, repo = parse_github_url("https://github.com/owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"


class TestValidateGitHubTokenExtended:
    """Extra coverage for validate_github_token edge cases."""

    def test_empty_string_returns_false(self):
        """Line 74: Empty token should return False."""
        assert validate_github_token("") is False

    def test_whitespace_only_returns_false(self):
        """Token of only spaces, after strip, has length 0 → False."""
        assert validate_github_token("   ") is False

    def test_literal_none_string(self):
        """Token == 'none' (case-insensitive) should be invalid."""
        assert validate_github_token("none") is False
        assert validate_github_token("None") is False
        assert validate_github_token("NONE") is False

    def test_literal_null_string(self):
        """Token == 'null' should be invalid."""
        assert validate_github_token("null") is False
        assert validate_github_token("NULL") is False

    def test_literal_invalid_string(self):
        """Token == 'invalid' should be invalid."""
        assert validate_github_token("invalid") is False
        assert validate_github_token("INVALID") is False

    def test_schemathesis_token_accepted_in_schemathesis_mode(self):
        """Lines 92-94: schemathesis-prefixed tokens accepted when SCHEMATHESIS_TESTING=true."""
        env = {
            "SCHEMATHESIS_TESTING": "true",
            "TEST_MODE": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("schemathesis_abc123") is True

    def test_schemathesis_token_case_insensitive(self):
        """Lines 92-94: Schemathesis check is case-insensitive via .lower()."""
        env = {
            "SCHEMATHESIS_TESTING": "1",
            "TEST_MODE": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("SCHEMATHESIS_TOKEN_XYZ") is True

    def test_test_pattern_token_in_test_mode(self):
        """Lines 117-119: test-pattern tokens accepted in TEST_MODE."""
        env = {
            "TEST_MODE": "true",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("test_my_token") is True
            assert validate_github_token("mock_abc") is True
            assert validate_github_token("fake_token") is True
            assert validate_github_token("dummy_tok") is True

    def test_long_token_in_test_mode(self):
        """Lines 121-123: Long token accepted in TEST_MODE even without pattern match."""
        env = {
            "TEST_MODE": "yes",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            # 10+ chars, no test pattern, no valid prefix
            assert validate_github_token("randomlongtokenvalue") is True

    def test_test_pattern_outside_test_mode(self):
        """Lines 126-128: test-pattern tokens accepted even outside explicit test mode."""
        env = {
            "TEST_MODE": "",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("test_something") is True
            assert validate_github_token("1234567890") is True
            assert validate_github_token("abcdef") is True

    def test_valid_prefix_returns_true(self):
        """Lines 130-131: Tokens with valid GitHub prefixes pass."""
        env = {
            "TEST_MODE": "",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("ghp_abcdef1234567890") is True
            assert validate_github_token("gho_abcdef1234567890") is True
            assert validate_github_token("ghs_abcdef1234567890") is True
            assert validate_github_token("github_pat_abcdef1234567890") is True

    def test_no_match_returns_false(self):
        """Lines 130-131: Token with no valid prefix and no test pattern → False."""
        env = {
            "TEST_MODE": "",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("xyz_something") is False

    def test_pytest_current_test_env_activates_test_mode(self):
        """PYTEST_CURRENT_TEST env var should activate test mode."""
        env = {
            "TEST_MODE": "",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "",
            "PYTEST_CURRENT_TEST": "tests/unit/test_foo.py::test_bar",
        }
        with patch.dict(os.environ, env, clear=False):
            # Long token accepted via test_mode length check
            assert validate_github_token("anyrandomtoken12345") is True

    def test_schemathesis_test_env_activates_test_mode(self):
        """SCHEMATHESIS_TEST env var should activate test mode."""
        env = {
            "TEST_MODE": "",
            "SCHEMATHESIS_TESTING": "",
            "SCHEMATHESIS_TEST": "true",
            "PYTEST_CURRENT_TEST": "",
        }
        with patch.dict(os.environ, env, clear=False):
            assert validate_github_token("anyrandomtoken12345") is True


class TestParseGitHubUrlLine38:
    """Cover line 38: repo.endswith('.git') branch inside the match block."""

    def test_double_git_suffix_strips_inner_git(self):
        """Line 38: URL with double .git — regex captures 'repo.git', line 38 strips it."""
        owner, repo = parse_github_url("https://github.com/owner/repo.git.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_ssh_double_git_suffix(self):
        """Line 38: SSH URL with double .git suffix."""
        owner, repo = parse_github_url("git@github.com:owner/project.git.git")
        assert owner == "owner"
        assert repo == "project"


class TestFormatGitHubUrlExtended:
    """Minimal extra checks for format_github_url."""

    def test_special_characters_in_names(self):
        """Owner/repo with hyphens and dots."""
        url = format_github_url("my-org", "my.repo")
        assert url == "https://github.com/my-org/my.repo"
