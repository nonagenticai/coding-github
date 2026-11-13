"""Extended tests for validation utility functions in routers/github_validation_utils.py.

These are helper functions (not HTTP endpoints), so they are tested by direct invocation.
Tests cover security, performance, quality validation and README analysis.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure stubs
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

from routers.github_validation_utils import (  # noqa: E402
    basic_security_check,
    validate_pr_performance,
    validate_pr_quality,
    validate_pr_security,
    validate_readme_changes,
)


def _make_file(filename="main.py", patch="", additions=0, deletions=0, changes=0):
    return SimpleNamespace(
        filename=filename,
        patch=patch,
        additions=additions,
        deletions=deletions,
        changes=changes,
    )


# ===========================================================================
# Security: score calculation with multiple issues
# ===========================================================================


class TestSecurityScoreCalculation:
    async def test_multiple_high_issues_reduce_score_and_fail(self):
        """Multiple eval/exec calls produce high issues, reduce score, fail."""
        files = [
            _make_file(
                "src/app.py",
                patch="+eval(x)\n+exec(y)\n+eval(z)",
                additions=3,
            ),
        ]
        result = await validate_pr_security(files)
        assert result.score < 100
        assert len(result.issues) >= 3
        assert result.passed is False  # >2 high issues


class TestSecurityDefaultSuggestion:
    async def test_no_patch_adds_default_suggestion(self):
        """Files without patches still get a default suggestion."""
        files = [_make_file("src/app.py", patch="", additions=0)]
        result = await validate_pr_security(files)
        assert any("security" in s.lower() for s in result.suggestions)


# ===========================================================================
# basic_security_check
# ===========================================================================


class TestBasicSecurityCheckNoPatch:
    def test_no_patch_returns_immediately(self):
        """No patch (None) means no issues and no suggestions added."""
        file = _make_file("src/app.py", patch=None)
        issues = []
        suggestions = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) == 0


class TestBasicSecurityCheckPassword:
    def test_password_detected(self):
        """Hardcoded password pattern detected."""
        file = _make_file("config.py", patch="+password = 'secret123'")
        issues = []
        suggestions = []
        basic_security_check(file, issues, suggestions)
        assert len(issues) >= 1
        assert any("credential" in i.message.lower() for i in issues)


class TestBasicSecurityCheckSuggestions:
    def test_suggestions_added_when_empty(self):
        """Default suggestion added when suggestions list is empty."""
        file = _make_file("app.py", patch="+x = 1")
        issues = []
        suggestions = []
        basic_security_check(file, issues, suggestions)
        assert len(suggestions) >= 1
        assert any("static analysis" in s.lower() for s in suggestions)

    def test_suggestions_not_doubled_when_present(self):
        """No duplicate suggestion when one already exists."""
        file = _make_file("app.py", patch="+x = 1")
        issues = []
        suggestions = ["existing suggestion"]
        basic_security_check(file, issues, suggestions)
        assert len(suggestions) == 1


# ===========================================================================
# Performance: pattern detections
# ===========================================================================


class TestPerformancePatterns:
    async def test_range_len_pattern(self):
        files = [
            _make_file("src/app.py", patch="+for i in range(len(items)):", additions=1),
        ]
        result = await validate_pr_performance(files)
        assert any("inefficient loop" in i.message.lower() for i in result.issues)

    async def test_append_in_loop_pattern(self):
        files = [
            _make_file("src/app.py", patch="+for x in items: result.append(x * 2)", additions=1),
        ]
        result = await validate_pr_performance(files)
        assert any("list comprehension" in i.message.lower() for i in result.issues)

    async def test_string_concat_in_loop(self):
        files = [
            _make_file(
                "src/app.py",
                patch="+for item in data: result = result + str(item)",
                additions=1,
            ),
        ]
        result = await validate_pr_performance(files)
        assert any("string concatenation" in i.message.lower() for i in result.issues)

    async def test_non_code_file_skipped(self):
        """Non-code file extensions are skipped for performance checks."""
        files = [
            _make_file("README.md", patch="+for i in range(len(items)):", additions=1),
        ]
        result = await validate_pr_performance(files)
        assert len(result.issues) == 0
        assert result.passed is True

    async def test_default_suggestions_added(self):
        """Default performance suggestions always added."""
        files = [_make_file("src/app.py", patch="+return data", additions=1)]
        result = await validate_pr_performance(files)
        assert any("performance testing" in s.lower() for s in result.suggestions)
        assert any("profiling" in s.lower() for s in result.suggestions)


# ===========================================================================
# Quality: large PR, large file, README
# ===========================================================================


class TestQualityLargePR:
    async def test_large_pr_flagged(self):
        """PR with >500 additions flagged."""
        files = [_make_file("app.py", additions=600, deletions=0, changes=600)]
        result = await validate_pr_quality(files)
        assert any("large pr" in i.message.lower() for i in result.issues)


class TestQualityLargeFileChange:
    async def test_large_file_change_flagged(self):
        """File with >100 changes flagged."""
        files = [_make_file("big.py", additions=80, deletions=30, changes=110)]
        result = await validate_pr_quality(files)
        assert any("many changes" in i.message.lower() for i in result.issues)


class TestQualityReadmeValidation:
    async def test_readme_changes_trigger_validation(self):
        """README file changes trigger validate_readme_changes."""
        deleted_lines = "\n".join([f"-old line {i}" for i in range(60)])
        added_lines = "\n".join([f"+new line {i}" for i in range(5)])
        patch = deleted_lines + "\n" + added_lines
        files = [
            _make_file("readme.md", patch=patch, additions=5, deletions=60, changes=65),
        ]
        result = await validate_pr_quality(files)
        assert any("readme" in i.message.lower() for i in result.issues)


# ===========================================================================
# validate_readme_changes
# ===========================================================================


class TestReadmeChangesNoPatch:
    async def test_no_patch_returns_empty(self):
        file = _make_file("README.md", patch=None)
        result = await validate_readme_changes(file)
        assert result == []


class TestReadmeCompleteReplacement:
    async def test_complete_replacement_detected(self):
        """deletions > 50 and deletions > additions * 2 => flagged."""
        deleted_lines = "\n".join([f"-line {i}" for i in range(60)])
        added_lines = "\n".join([f"+new {i}" for i in range(10)])
        patch = deleted_lines + "\n" + added_lines
        file = _make_file("README.md", patch=patch)
        result = await validate_readme_changes(file)
        assert any("completely replaced" in i.message.lower() for i in result)


class TestReadmeSectionRemoval:
    async def test_important_section_removed(self):
        patch = (
            "-# Installation\n"
            "-Follow these steps to install\n"
            "-# License\n"
            "-MIT License\n"
            "+# New Section\n"
        )
        file = _make_file("README.md", patch=patch)
        result = await validate_readme_changes(file)
        assert any("remove important sections" in i.message.lower() for i in result)

    async def test_section_re_added_not_flagged(self):
        """If section word is in an added line, it is not flagged as removed."""
        patch = "-# Installation\n+installation guide updated\n"
        file = _make_file("README.md", patch=patch)
        result = await validate_readme_changes(file)
        removed_section_issues = [
            i for i in result if "remove important sections" in i.message.lower()
        ]
        assert len(removed_section_issues) == 0


class TestReadmeHighDeletionRatio:
    async def test_high_deletion_ratio(self):
        """deletions > 20 and ratio > 3 => flagged."""
        deleted_lines = "\n".join([f"-old content {i}" for i in range(25)])
        added_lines = "+single new line"
        patch = deleted_lines + "\n" + added_lines
        file = _make_file("README.md", patch=patch)
        result = await validate_readme_changes(file)
        assert any("high deletion" in i.message.lower() for i in result)


class TestReadmeCleanChanges:
    async def test_clean_readme_no_issues(self):
        patch = "+# Welcome\n+This is a great project\n-# Old Welcome"
        file = _make_file("README.md", patch=patch)
        result = await validate_readme_changes(file)
        assert len(result) == 0
