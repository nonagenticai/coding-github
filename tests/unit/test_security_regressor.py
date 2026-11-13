"""Unit tests for SecurityRegressionDetector."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.context.security_regressor import (
    SecurityRegressionDetector,
)


@pytest.fixture
def detector():
    return SecurityRegressionDetector()


@pytest.fixture
def codebase_ctx():
    return CodebaseContext(
        total_files=10,
        total_lines=5000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


# ---------------------------------------------------------------------------
# 1. No regressions – clean code
# ---------------------------------------------------------------------------
async def test_no_regressions(detector, codebase_ctx):
    code = "def greet(name):\n    return f'Hello, {name}'\n"
    changes = [
        FileChange(
            filename="src/greet.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    security_issues = [i for i in issues if "security_pattern" in i.type]
    assert security_issues == []


# ---------------------------------------------------------------------------
# 2. Authorization regression – removed authorize check
# ---------------------------------------------------------------------------
async def test_authz_regression(detector, codebase_ctx):
    patch = (
        "--- a/src/auth/views.py\n"
        "+++ b/src/auth/views.py\n"
        "-    authorize(user)\n"
        "+    pass  # authorization temporarily disabled\n"
    )
    changes = [
        FileChange(
            filename="src/auth/views.py",
            file_type=FileType.PYTHON,
            after_content="pass  # authorization temporarily disabled\n",
            patch=patch,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    removal_issues = [i for i in issues if i.type == "security_control_removal"]
    assert len(removal_issues) > 0


# ---------------------------------------------------------------------------
# 3. Authentication regression – commented-out auth check
# ---------------------------------------------------------------------------
async def test_authn_regression(detector, codebase_ctx):
    code = "# auth.check_session(request)\nresponse = get_protected_resource()\n"
    changes = [
        FileChange(
            filename="src/auth/middleware.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    auth_issues = [i for i in issues if "auth_bypass" in i.type or "auth_security" in i.type]
    assert len(auth_issues) > 0


# ---------------------------------------------------------------------------
# 4. Input validation regression – use of eval()
# ---------------------------------------------------------------------------
async def test_input_validation_regression(detector, codebase_ctx):
    code = "user_data = request.json\nresult = eval(user_data['expression'])\n"
    changes = [
        FileChange(
            filename="src/calc.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    eval_issues = [
        i for i in issues if "input_validation" in i.type or "unsafe_data_flow" in i.type
    ]
    assert len(eval_issues) > 0


# ---------------------------------------------------------------------------
# 5. Secret exposure – hardcoded password
# ---------------------------------------------------------------------------
async def test_secret_exposure(detector, codebase_ctx):
    code = 'password = "SuperSecretPassword123"\ndb.connect(password=password)\n'
    changes = [
        FileChange(
            filename="src/db_config.py",
            file_type=FileType.PYTHON,
            after_content=code,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    secret_issues = [i for i in issues if "hardcoded_secret" in i.type]
    assert len(secret_issues) > 0


# ---------------------------------------------------------------------------
# 6. False positive suppression – non-code file skipped
# ---------------------------------------------------------------------------
async def test_false_positive_suppression(detector, codebase_ctx):
    changes = [
        FileChange(
            filename="README.md",
            file_type=FileType.MARKDOWN,
            after_content="# Security notes\n\nWe use eval() for demonstration only.\n",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    assert issues == []


# ---------------------------------------------------------------------------
# 7. Empty diff – no content produces no issues
# ---------------------------------------------------------------------------
async def test_empty_diff(detector, codebase_ctx):
    changes = [
        FileChange(
            filename="src/empty.py",
            file_type=FileType.PYTHON,
            after_content=None,
            before_content=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    assert issues == []


# ---------------------------------------------------------------------------
# 8. Malformed changes – binary file is skipped
# ---------------------------------------------------------------------------
async def test_malformed_changes(detector, codebase_ctx):
    changes = [
        FileChange(
            filename="image.png",
            file_type=FileType.PYTHON,
            binary=True,
            after_content="binary data",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    issues = await detector.detect_security_regressions(changes, codebase_ctx)
    assert issues == []
