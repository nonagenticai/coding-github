"""Edge-case and boundary tests for DiffParser."""

from __future__ import annotations

import logging

import pytest
from coding_client.models.pr_validation import ChangeType, FileType

from services.pr_validation.core.diff_parser import DiffHunk, DiffParser, ParsedDiff


@pytest.fixture
def parser():
    return DiffParser()


# ---------------------------------------------------------------------------
# Malformed / error inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    """Parse inputs that are malformed or missing required keys."""

    def test_github_file_missing_filename_key(self, parser):
        """KeyError on file_data['filename'] triggers fallback ParsedDiff."""
        bad = {"additions": 5, "deletions": 3}
        result = parser.parse_github_diff([bad])
        assert len(result) == 1
        assert result[0].filename == "unknown"
        assert result[0].binary is True
        assert result[0].additions == 5

    def test_empty_github_files_list(self, parser):
        assert parser.parse_github_diff([]) == []

    def test_empty_git_diff_text(self, parser):
        assert parser.parse_git_diff("") == []

    def test_git_diff_garbage_text(self, parser):
        """Non-diff text should produce no diffs."""
        result = parser.parse_git_diff("this is not a diff\njust random text\n")
        assert result == []

    def test_hunk_header_without_comma_counts(self, parser):
        """@@ -1 +1 @@ means old_count=1, new_count=1 (no comma)."""
        diff_text = "diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = parser.parse_git_diff(diff_text)
        assert result[0].hunks[0].old_count == 1
        assert result[0].hunks[0].new_count == 1


# ---------------------------------------------------------------------------
# Content provider edge cases
# ---------------------------------------------------------------------------


class TestContentProviderEdgeCases:
    """Content provider exceptions and skipped calls."""

    def test_content_provider_exception_is_caught(self, parser, caplog):
        parsed = ParsedDiff(
            filename="src/app.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            hunks=[],
        )

        def failing_provider(filename, ref):
            raise RuntimeError("Content not available")

        with caplog.at_level(logging.WARNING):
            changes = parser.create_file_changes([parsed], content_provider=failing_provider)
        assert len(changes) == 1
        assert changes[0].before_content is None
        assert changes[0].after_content is None

    def test_added_file_skips_before_content(self, parser):
        parsed = ParsedDiff(
            filename="new.py",
            old_filename=None,
            new_filename="new.py",
            change_type=ChangeType.ADDED,
            additions=1,
            deletions=0,
            hunks=[],
        )
        calls = []

        def provider(filename, ref):
            calls.append(ref)
            return f"content_{ref}"

        result = parser.create_file_changes([parsed], content_provider=provider)
        assert result[0].before_content is None
        assert result[0].after_content == "content_after"
        assert "before" not in calls

    def test_deleted_file_skips_after_content(self, parser):
        parsed = ParsedDiff(
            filename="old.py",
            old_filename="old.py",
            new_filename=None,
            change_type=ChangeType.DELETED,
            additions=0,
            deletions=1,
            hunks=[],
        )
        calls = []

        def provider(filename, ref):
            calls.append(ref)
            return f"content_{ref}"

        result = parser.create_file_changes([parsed], content_provider=provider)
        assert result[0].after_content is None
        assert result[0].before_content == "content_before"
        assert "after" not in calls


# ---------------------------------------------------------------------------
# FileChange creation error fallback
# ---------------------------------------------------------------------------


class TestFileChangeCreationFallback:
    """When file_classifier.classify_file raises, fallback FileChange is created."""

    def test_classify_error_produces_minimal_file_change(self, parser):
        parsed = ParsedDiff(
            filename="broken.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=3,
            deletions=2,
            hunks=[],
        )
        original = parser.file_classifier.classify_file
        parser.file_classifier.classify_file = lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("boom")
        )
        try:
            result = parser.create_file_changes([parsed])
        finally:
            parser.file_classifier.classify_file = original

        assert len(result) == 1
        assert result[0].file_type == FileType.UNKNOWN
        assert result[0].binary is True


# ---------------------------------------------------------------------------
# validate_diff — hunk count mismatches
# ---------------------------------------------------------------------------


class TestValidateDiffMismatches:
    """Hunk line-count validation."""

    def test_new_line_count_mismatch(self, parser):
        parsed = ParsedDiff(
            filename="test.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=1,
                    new_start=1,
                    new_count=99,  # deliberately wrong
                    lines=[" context", "+added"],
                ),
            ],
        )
        issues = parser.validate_diff(parsed)
        assert any("New line count mismatch" in i for i in issues)

    def test_old_line_count_mismatch(self, parser):
        parsed = ParsedDiff(
            filename="test.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=1,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=99,
                    new_start=1,
                    new_count=1,
                    lines=[" context", "-removed"],
                ),
            ],
        )
        issues = parser.validate_diff(parsed)
        assert any("Old line count mismatch" in i for i in issues)


# ---------------------------------------------------------------------------
# _reconstruct_patch — branch coverage
# ---------------------------------------------------------------------------


class TestReconstructPatchBranches:
    """Patch reconstruction for added/deleted/binary files."""

    def test_added_file_with_custom_mode(self, parser):
        parsed = ParsedDiff(
            filename="script.sh",
            old_filename=None,
            new_filename="script.sh",
            change_type=ChangeType.ADDED,
            additions=1,
            deletions=0,
            hunks=[
                DiffHunk(old_start=0, old_count=0, new_start=1, new_count=1, lines=["+#!/bin/bash"])
            ],
            new_mode="100755",
        )
        patch = parser._reconstruct_patch(parsed)
        assert "a//dev/null" in patch
        assert "new file mode 100755" in patch

    def test_deleted_file_default_mode(self, parser):
        parsed = ParsedDiff(
            filename="gone.py",
            old_filename="gone.py",
            new_filename=None,
            change_type=ChangeType.DELETED,
            additions=0,
            deletions=1,
            hunks=[],
        )
        patch = parser._reconstruct_patch(parsed)
        assert "b//dev/null" in patch
        assert "deleted file mode 100644" in patch

    def test_hunk_context_in_reconstructed_patch(self, parser):
        parsed = ParsedDiff(
            filename="mod.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[
                DiffHunk(
                    old_start=10,
                    old_count=3,
                    new_start=10,
                    new_count=4,
                    lines=[" ctx", "+new", " more", " end"],
                    context="def my_function():",
                ),
            ],
        )
        patch = parser._reconstruct_patch(parsed)
        assert "def my_function():" in patch


# ---------------------------------------------------------------------------
# Large inputs
# ---------------------------------------------------------------------------


class TestLargeInputs:
    """Boundary condition: many files or large diffs."""

    def test_many_github_files(self, parser):
        files = [
            {
                "filename": f"file_{i}.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": f"@@ -1 +1,2 @@\n line\n+add{i}\n",
            }
            for i in range(50)
        ]
        result = parser.parse_github_diff(files)
        assert len(result) == 50

    def test_diff_with_many_hunks(self, parser):
        hunks = ""
        for i in range(20):
            start = i * 10 + 1
            hunks += f"@@ -{start},2 +{start},3 @@\n ctx\n+add{i}\n end\n"
        diff_text = f"diff --git a/big.py b/big.py\n{hunks}"
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 1
        assert len(result[0].hunks) == 20


# ---------------------------------------------------------------------------
# Multiple hunks in parse_git_diff
# ---------------------------------------------------------------------------


class TestMultipleHunksInGitDiff:
    """Ensure the first hunk is appended when a second @@ appears."""

    def test_two_hunks_in_single_file(self, parser):
        diff_text = (
            "diff --git a/file.py b/file.py\n"
            "@@ -1,3 +1,4 @@\n"
            " line1\n"
            "+added\n"
            " line2\n"
            " line3\n"
            "@@ -10,2 +11,3 @@\n"
            " ctx\n"
            "+new\n"
            " end\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 1
        assert len(result[0].hunks) == 2


# ---------------------------------------------------------------------------
# Patch parsing — _parse_patch_hunks
# ---------------------------------------------------------------------------


class TestParsePatchHunks:
    """Parse hunk headers from patch strings."""

    def test_multiple_hunks(self, parser):
        patch = (
            "@@ -1,3 +1,4 @@\n"
            " unchanged\n"
            "+added\n"
            " more\n"
            "@@ -10,2 +11,3 @@\n"
            " context\n"
            "+another\n"
            " end\n"
        )
        hunks = parser._parse_patch_hunks(patch)
        assert len(hunks) == 2
        assert hunks[0].old_start == 1
        assert hunks[1].old_start == 10

    def test_empty_patch(self, parser):
        assert parser._parse_patch_hunks("") == []
