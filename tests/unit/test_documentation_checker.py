"""Unit tests for DocumentationSyncChecker."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

# DocumentationImpact is now defined in conftest with dual field names
from services.pr_validation.analyzers.impact.documentation_checker import (
    DocumentationSyncChecker,
)


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


@pytest.fixture
def checker(codebase_ctx):
    return DocumentationSyncChecker(codebase_context=codebase_ctx)


# ---------------------------------------------------------------------------
# 1. Docs required – code changes without doc changes -> missing docs
# ---------------------------------------------------------------------------
async def test_docs_required(checker, codebase_ctx):
    code = "def new_api_endpoint():\n    return {'status': 'ok'}\n"
    # NOTE: filename must not be "api.py" — the file classifier has a bug where
    # bare "api" in documentation_filenames matches "api.py" via splitext.
    changes = [
        FileChange(
            filename="src/routes.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.ADDED,
            after_content=code,
            additions=5,
            patch="+def new_api_endpoint():\n+    return {'status': 'ok'}\n",
            deletions=0,
        )
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)

    # No doc files were changed, so docs_affected must be empty
    assert result.docs_affected == []

    # The new public function should appear in new_docs_needed
    assert len(result.new_docs_needed) >= 1
    assert any("new_api_endpoint" in entry for entry in result.new_docs_needed)

    # No docs exist for the new code, so completeness must be 0.0
    assert result.doc_completeness_score == 0.0

    # No doc files to have outdated references
    assert result.outdated_docs == []


# ---------------------------------------------------------------------------
# 2. Docs not required – only doc files changed
# ---------------------------------------------------------------------------
async def test_docs_not_required(checker, codebase_ctx):
    changes = [
        FileChange(
            filename="README.md",
            file_type=FileType.MARKDOWN,
            change_type=ChangeType.MODIFIED,
            after_content="# Updated readme\n\nNew content here.\n",
            additions=2,
            deletions=0,
            patch="",
        )
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)

    # README.md is a doc file, should show up in docs_affected
    assert "README.md" in result.docs_affected

    # No code changes, so nothing needs new documentation
    assert result.new_docs_needed == []

    # No code changes means full completeness
    assert result.doc_completeness_score == 1.0

    # No code changes means nothing can be outdated
    assert result.outdated_docs == []


# ---------------------------------------------------------------------------
# 3. Docs updated – doc and code changed together
# ---------------------------------------------------------------------------
async def test_docs_updated(checker, codebase_ctx):
    code = "class Widget:\n    pass\n"
    doc = "# API Reference\n\n## Widget\n\nA widget component.\n"
    changes = [
        FileChange(
            filename="src/widget.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content=code,
            additions=2,
            patch="+class Widget:\n+    pass\n",
            deletions=0,
        ),
        FileChange(
            filename="docs/api.md",
            file_type=FileType.MARKDOWN,
            change_type=ChangeType.MODIFIED,
            after_content=doc,
            additions=3,
            deletions=0,
            patch="",
        ),
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)

    # docs/api.md should be in the affected docs list
    assert result.docs_affected == ["docs/api.md"]

    # The doc contains "## Widget" which matches the api_patterns, so Widget
    # is documented → completeness should be > 0.0 (actually 1.0 since the
    # only public symbol Widget is documented via the ## heading)
    assert result.doc_completeness_score > 0.0
    assert isinstance(result.doc_completeness_score, float)


# ---------------------------------------------------------------------------
# 4. Broken examples – doc references symbol that was changed
# ---------------------------------------------------------------------------
async def test_broken_examples(checker, codebase_ctx):
    doc_content = "# Guide\n\n```python\nfrom mylib import old_function\nold_function()\n```\n"
    code = "def old_function():\n    return 'new'\n"
    changes = [
        FileChange(
            filename="docs/guide.md",
            file_type=FileType.MARKDOWN,
            change_type=ChangeType.MODIFIED,
            after_content=doc_content,
            additions=3,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="src/mylib.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content=code,
            additions=1,
            patch="+def old_function():\n+    return 'new'\n",
            deletions=0,
        ),
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)

    # docs/guide.md should be in the affected docs list
    assert "docs/guide.md" in result.docs_affected

    # The doc's code example contains old_function which was modified in code,
    # so outdated_docs should flag the stale example referencing old_function
    assert len(result.outdated_docs) >= 1
    assert any("old_function" in entry for entry in result.outdated_docs)

    # Completeness score should be a float
    assert isinstance(result.doc_completeness_score, float)


# ---------------------------------------------------------------------------
# 5. Coverage ratio – completeness score is a float in [0, 1]
# ---------------------------------------------------------------------------
async def test_coverage_ratio(checker, codebase_ctx):
    code = "def public_func():\n    pass\n\nclass PublicClass:\n    pass\n"
    changes = [
        FileChange(
            filename="src/lib.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content=code,
            additions=5,
            patch="+def public_func():\n+    pass\n+class PublicClass:\n+    pass\n",
            deletions=0,
        ),
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)

    # Completeness score must be a float in [0.0, 1.0]
    assert isinstance(result.doc_completeness_score, float)
    assert 0.0 <= result.doc_completeness_score <= 1.0

    # No doc files changed → no docs for the new code → score should be 0.0
    assert result.doc_completeness_score == 0.0

    # Both public symbols should be listed as needing documentation
    assert len(result.new_docs_needed) >= 2
    needed_text = " ".join(result.new_docs_needed)
    assert "public_func" in needed_text
    assert "PublicClass" in needed_text
