"""Tests for validation utility functions in routers/github_validation_utils.py.

These are helper functions (not HTTP endpoints), tested by direct invocation.
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
    validate_pr_performance,
    validate_pr_quality,
    validate_pr_security,
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
# Quality
# ===========================================================================


class TestQualityCleanFiles:
    async def test_quality_no_issues(self):
        """Clean files produce passing quality result."""
        files = [
            _make_file("src/main.py", patch="+print('hi')", additions=1, changes=1),
        ]
        result = await validate_pr_quality(files)
        assert result.passed is True
        assert result.score > 0
        assert result.validation_type == "quality"


# ===========================================================================
# Security
# ===========================================================================


class TestSecurityCleanCode:
    async def test_security_clean_code_passes(self):
        """Files without security anti-patterns produce passing result."""
        files = [
            _make_file("src/app.py", patch="+x = 1 + 2", additions=1),
        ]
        result = await validate_pr_security(files)
        assert result.passed is True
        assert result.score == 100
        assert result.validation_type == "security"


class TestSecurityDetectsEval:
    async def test_eval_detected(self):
        """eval() in patch flagged as security issue."""
        files = [
            _make_file("src/app.py", patch="+result = eval(user_input)", additions=1),
        ]
        result = await validate_pr_security(files)
        assert len(result.issues) >= 1
        assert any("injection" in i.message.lower() for i in result.issues)
        assert result.score < 100


# ===========================================================================
# Performance
# ===========================================================================


class TestPerformanceCleanCode:
    async def test_performance_clean_code_passes(self):
        """Files without performance anti-patterns produce passing result."""
        files = [
            _make_file("src/app.py", patch="+return data", additions=1),
        ]
        result = await validate_pr_performance(files)
        assert result.passed is True
        assert result.score == 100
        assert result.validation_type == "performance"


class TestPerformanceDetectsSleep:
    async def test_sleep_detected(self):
        """time.sleep() flagged as performance issue."""
        files = [
            _make_file(
                "src/worker.py",
                patch="+    time.sleep(5)\n+    time.sleep(10)",
                additions=2,
            ),
        ]
        result = await validate_pr_performance(files)
        assert len(result.issues) >= 1
        assert any("sleep" in i.message.lower() for i in result.issues)
        assert result.passed is False
