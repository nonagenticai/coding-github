"""Tests for utils/github_validation_helpers.py."""

from types import SimpleNamespace

from schemas.github_api_models import PRValidationResult
from utils.github_validation_helpers import (
    validate_pr_performance,
    validate_pr_quality,
    validate_pr_security,
    validate_readme_changes,
)


def _make_file(filename="app.py", patch="+x = 1"):
    return SimpleNamespace(filename=filename, patch=patch)


class TestGitHubValidationHelpers:
    """Tests for github_validation_helpers module."""

    async def test_validate_pr_security_clean(self):
        """Files with no security patterns should pass."""
        files = [_make_file(patch="+x = 1\n+y = 2")]
        result = await validate_pr_security(files)
        assert isinstance(result, PRValidationResult)
        assert result.validation_type == "security"
        assert result.passed is True
        assert result.score > 0

    async def test_validate_pr_security_detects_eval(self):
        """Files containing eval() should flag security issues."""
        files = [_make_file(patch="+result = eval(user_input)")]
        result = await validate_pr_security(files)
        assert len(result.issues) >= 1
        assert any(
            "injection" in i.message.lower() or "dangerous" in i.message.lower()
            for i in result.issues
        )

    async def test_validate_pr_performance(self):
        """Performance validator should detect sleep calls."""
        files = [_make_file(filename="app.py", patch="+import time\n+time.sleep(10)")]
        result = await validate_pr_performance(files)
        assert result.validation_type == "performance"
        assert isinstance(result.issues, list)

    async def test_validate_pr_quality(self):
        """Quality validator should detect TODOs."""
        files = [_make_file(patch="+# TODO: fix this later")]
        result = await validate_pr_quality(files)
        assert result.validation_type == "quality"
        assert any(
            "TODO" in i.message or "todo" in i.message.lower() or "unfinished" in i.message.lower()
            for i in result.issues
        )

    async def test_validate_readme_changes(self):
        """README changes removing important sections should be flagged."""
        file = _make_file(
            filename="README.md",
            patch="-## Installation\n-Steps to install\n+New content here",
        )
        issues = await validate_readme_changes(file)
        assert isinstance(issues, list)
        # Should detect removal of Installation section
        assert any("Installation" in i.message for i in issues)
