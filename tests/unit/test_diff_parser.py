"""Core tests for DiffParser — parse real diffs, verify real outputs."""

from __future__ import annotations

import pytest
from coding_client.models.pr_validation import ChangeType, FileType

from services.pr_validation.core.diff_parser import DiffHunk, DiffParser, ParsedDiff


@pytest.fixture
def parser():
    return DiffParser()


# ---------------------------------------------------------------------------
# parse_github_diff — parse GitHub API file dicts
# ---------------------------------------------------------------------------


class TestParseGithubDiff:
    """Parse GitHub API file dicts into ParsedDiff objects."""

    def test_single_modified_file(self, parser):
        files = [
            {
                "filename": "src/main.py",
                "status": "modified",
                "additions": 3,
                "deletions": 1,
                "patch": (
                    "@@ -10,4 +10,6 @@ def main():\n"
                    "     print('hello')\n"
                    "-    return 0\n"
                    "+    result = compute()\n"
                    "+    log(result)\n"
                    "+    return result\n"
                ),
            }
        ]
        result = parser.parse_github_diff(files)
        assert len(result) == 1
        diff = result[0]
        assert diff.filename == "src/main.py"
        assert diff.change_type == ChangeType.MODIFIED
        assert diff.additions == 3
        assert diff.deletions == 1
        assert len(diff.hunks) == 1
        assert diff.hunks[0].old_start == 10

    def test_added_file(self, parser):
        files = [
            {
                "filename": "src/new_module.py",
                "status": "added",
                "additions": 5,
                "deletions": 0,
                "patch": "@@ -0,0 +1,5 @@\n+import os\n+\n+def hello():\n+    print('hi')\n+    return True\n",
            }
        ]
        result = parser.parse_github_diff(files)
        assert result[0].change_type == ChangeType.ADDED
        assert result[0].additions == 5
        assert result[0].deletions == 0

    def test_deleted_file(self, parser):
        files = [
            {
                "filename": "src/old.py",
                "status": "removed",
                "additions": 0,
                "deletions": 10,
                "patch": "",
            }
        ]
        result = parser.parse_github_diff(files)
        assert result[0].change_type == ChangeType.DELETED

    def test_renamed_file(self, parser):
        files = [
            {
                "filename": "src/utils_v2.py",
                "status": "renamed",
                "previous_filename": "src/utils.py",
                "additions": 0,
                "deletions": 0,
                "patch": "",
            }
        ]
        result = parser.parse_github_diff(files)
        assert result[0].change_type == ChangeType.RENAMED
        assert result[0].old_filename == "src/utils.py"
        assert result[0].new_filename == "src/utils_v2.py"

    def test_binary_file_no_patch(self, parser):
        files = [
            {
                "filename": "assets/logo.png",
                "status": "modified",
                "additions": 0,
                "deletions": 0,
                "binary": True,
            }
        ]
        result = parser.parse_github_diff(files)
        assert result[0].binary is True
        assert result[0].hunks == []

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("added", ChangeType.ADDED),
            ("removed", ChangeType.DELETED),
            ("modified", ChangeType.MODIFIED),
            ("renamed", ChangeType.RENAMED),
            ("unknown_status", ChangeType.MODIFIED),
        ],
    )
    def test_status_to_change_type_mapping(self, parser, status, expected):
        files = [
            {"filename": "f.py", "status": status, "additions": 0, "deletions": 0, "patch": ""}
        ]
        result = parser.parse_github_diff(files)
        assert result[0].change_type == expected

    def test_multiple_files(self, parser):
        files = [
            {
                "filename": "a.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -1 +1,2 @@\n line\n+new\n",
            },
            {
                "filename": "b.py",
                "status": "added",
                "additions": 2,
                "deletions": 0,
                "patch": "@@ -0,0 +1,2 @@\n+x\n+y\n",
            },
        ]
        result = parser.parse_github_diff(files)
        assert len(result) == 2
        assert result[0].filename == "a.py"
        assert result[1].filename == "b.py"

    def test_multi_hunk_patch(self, parser):
        patch = (
            "@@ -1,3 +1,4 @@\n"
            " line1\n"
            "+added_line\n"
            " line2\n"
            " line3\n"
            "@@ -20,2 +21,3 @@ def func():\n"
            " existing\n"
            "+another_add\n"
            " end\n"
        )
        files = [
            {
                "filename": "multi.py",
                "status": "modified",
                "additions": 2,
                "deletions": 0,
                "patch": patch,
            }
        ]
        result = parser.parse_github_diff(files)
        assert len(result[0].hunks) == 2
        assert result[0].hunks[0].old_start == 1
        assert result[0].hunks[1].old_start == 20
        assert result[0].hunks[1].context == "def func():"


# ---------------------------------------------------------------------------
# parse_git_diff — parse raw git diff text
# ---------------------------------------------------------------------------


class TestParseGitDiff:
    """Parse raw git diff output."""

    def test_single_file_diff(self, parser):
        diff_text = (
            "diff --git a/hello.py b/hello.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import sys\n"
            "+import os\n"
            " def main():\n"
            "     pass\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 1
        assert result[0].filename == "hello.py"
        assert result[0].additions == 1
        assert result[0].deletions == 0

    def test_multiple_file_diff(self, parser):
        diff_text = (
            "diff --git a/a.py b/a.py\n"
            "@@ -1,2 +1,3 @@\n"
            " x = 1\n"
            "+y = 2\n"
            " z = 3\n"
            "diff --git a/b.py b/b.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            " keep\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert len(result) == 2
        assert result[0].filename == "a.py"
        assert result[0].additions == 1
        assert result[1].filename == "b.py"
        assert result[1].additions == 1
        assert result[1].deletions == 1

    def test_new_file_mode(self, parser):
        diff_text = (
            "diff --git a/new.py b/new.py\nnew file mode 100644\n@@ -0,0 +1,2 @@\n+line1\n+line2\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].change_type == ChangeType.ADDED
        assert result[0].mode_change is True
        assert result[0].new_mode == "100644"

    def test_deleted_file_mode(self, parser):
        diff_text = (
            "diff --git a/old.py b/old.py\n"
            "deleted file mode 100644\n"
            "@@ -1,2 +0,0 @@\n"
            "-line1\n"
            "-line2\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].change_type == ChangeType.DELETED
        assert result[0].old_mode == "100644"

    def test_rename_detection(self, parser):
        diff_text = (
            "diff --git a/old_name.py b/new_name.py\n"
            "rename from old_name.py\n"
            "rename to new_name.py\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].change_type == ChangeType.RENAMED
        assert result[0].old_filename == "old_name.py"
        assert result[0].new_filename == "new_name.py"

    def test_binary_file_detection(self, parser):
        diff_text = (
            "diff --git a/image.png b/image.png\nBinary files a/image.png and b/image.png differ\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].binary is True

    def test_hunk_with_context_function_name(self, parser):
        diff_text = (
            "diff --git a/mod.py b/mod.py\n"
            "@@ -5,3 +5,4 @@ def process():\n"
            "     x = compute()\n"
            "+    log(x)\n"
            "     return x\n"
            "     # end\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].hunks[0].context == "def process():"

    def test_counts_additions_and_deletions_correctly(self, parser):
        diff_text = (
            "diff --git a/count.py b/count.py\n"
            "@@ -1,5 +1,5 @@\n"
            " keep1\n"
            "-remove1\n"
            "-remove2\n"
            "+add1\n"
            "+add2\n"
            " keep2\n"
            " keep3\n"
        )
        result = parser.parse_git_diff(diff_text)
        assert result[0].additions == 2
        assert result[0].deletions == 2


# ---------------------------------------------------------------------------
# create_file_changes — convert ParsedDiff to FileChange
# ---------------------------------------------------------------------------


class TestCreateFileChanges:
    """Convert ParsedDiff objects into FileChange objects."""

    def test_basic_file_change_creation(self, parser):
        parsed = ParsedDiff(
            filename="src/app.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=3,
            deletions=1,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=3,
                    new_start=1,
                    new_count=5,
                    lines=[" ctx", "-old", "+new1", "+new2", "+new3"],
                )
            ],
        )
        changes = parser.create_file_changes([parsed])
        assert len(changes) == 1
        fc = changes[0]
        assert fc.filename == "src/app.py"
        assert fc.file_type == FileType.PYTHON
        assert fc.change_type == ChangeType.MODIFIED
        assert fc.additions == 3
        assert fc.deletions == 1

    def test_content_provider_called_for_modified(self, parser):
        parsed = ParsedDiff(
            filename="src/mod.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[],
        )
        calls = []

        def provider(filename, ref):
            calls.append((filename, ref))
            return f"content_{ref}"

        changes = parser.create_file_changes([parsed], content_provider=provider)
        assert changes[0].before_content == "content_before"
        assert changes[0].after_content == "content_after"
        refs = [ref for _, ref in calls]
        assert "before" in refs
        assert "after" in refs

    def test_no_content_provider(self, parser):
        parsed = ParsedDiff(
            filename="test.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[],
        )
        changes = parser.create_file_changes([parsed])
        assert changes[0].before_content is None
        assert changes[0].after_content is None


# ---------------------------------------------------------------------------
# create_pr_diff — assemble PRDiff from FileChanges
# ---------------------------------------------------------------------------


class TestCreatePRDiff:
    """Assemble PRDiff from FileChange list."""

    def test_creates_pr_diff_with_stats(self, parser):
        from coding_client.models.pr_validation import FileChange

        files = [
            FileChange(
                filename="a.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.ADDED,
                additions=10,
                deletions=0,
                patch="",
            ),
            FileChange(
                filename="b.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=5,
                deletions=3,
                patch="",
            ),
        ]
        pr_diff = parser.create_pr_diff(files, "base123", "head456", {"pr_number": 42})
        assert pr_diff.base_sha == "base123"
        assert pr_diff.head_sha == "head456"
        assert pr_diff.metadata["pr_number"] == 42
        assert pr_diff.stats.total_files == 2
        assert pr_diff.stats.total_additions == 15
        assert pr_diff.stats.total_deletions == 3
        assert pr_diff.stats.files_added == 1
        assert pr_diff.stats.files_modified == 1


# ---------------------------------------------------------------------------
# validate_diff — validate parsed diff consistency
# ---------------------------------------------------------------------------


class TestValidateDiff:
    """Validate parsed diff for consistency issues."""

    def test_valid_diff_no_issues(self, parser):
        parsed = ParsedDiff(
            filename="ok.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=1,
            hunks=[
                DiffHunk(
                    old_start=1,
                    old_count=2,
                    new_start=1,
                    new_count=2,
                    lines=[" context", "-removed", "+added", " end"],
                )
            ],
        )
        # old_count=2 should match deletions(1) + context(2) — actually old_count=2 is wrong here
        # Let's construct correctly: old_count = deletions + context_lines = 1 + 2 = 3
        # We need to fix counts: context lines starting with " " = 2, deletions = 1, additions = 1
        # old_count should be 3 (2 context + 1 deletion), new_count should be 3 (2 context + 1 addition)
        parsed.hunks[0].old_count = 3
        parsed.hunks[0].new_count = 3
        issues = parser.validate_diff(parsed)
        assert issues == []

    def test_missing_filename_flagged(self, parser):
        parsed = ParsedDiff(
            filename="",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[],
        )
        issues = parser.validate_diff(parsed)
        assert any("Missing filename" in i for i in issues)

    def test_no_changes_in_non_binary_flagged(self, parser):
        parsed = ParsedDiff(
            filename="empty.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            hunks=[],
            binary=False,
        )
        issues = parser.validate_diff(parsed)
        assert any("No changes detected" in i for i in issues)

    def test_binary_file_zero_changes_not_flagged(self, parser):
        parsed = ParsedDiff(
            filename="image.png",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            hunks=[],
            binary=True,
        )
        issues = parser.validate_diff(parsed)
        assert not any("No changes detected" in i for i in issues)


# ---------------------------------------------------------------------------
# _reconstruct_patch — round-trip reconstruction
# ---------------------------------------------------------------------------


class TestReconstructPatch:
    """Reconstruct patch text from ParsedDiff."""

    def test_modified_file_roundtrip(self, parser):
        parsed = ParsedDiff(
            filename="app.py",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            hunks=[
                DiffHunk(
                    old_start=5,
                    old_count=3,
                    new_start=5,
                    new_count=4,
                    lines=[" ctx", "+new", " more", " end"],
                )
            ],
        )
        patch = parser._reconstruct_patch(parsed)
        assert "diff --git" in patch
        assert "@@ -5,3 +5,4 @@" in patch
        assert "+new" in patch

    def test_binary_file_patch(self, parser):
        parsed = ParsedDiff(
            filename="data.bin",
            old_filename=None,
            new_filename=None,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            hunks=[],
            binary=True,
        )
        patch = parser._reconstruct_patch(parsed)
        assert "Binary file data.bin has changed" in patch
