"""Tests for git_diff_analyzer.py."""

from coding_client.models.pr_validation import (
    ChangeType,
    DiffAnalysisResult,
    DiffStats,
    FileChange,
    FileType,
    PRDiff,
)

from services.pr_validation.analyzers.diff.git_diff_analyzer import GitDiffAnalyzer


def _make_diff(files):
    return PRDiff(
        base_sha="aaa",
        head_sha="bbb",
        files_changed=files,
        stats=DiffStats.from_files(files),
        metadata={},
    )


class TestGitDiffAnalyzer:
    """Tests for GitDiffAnalyzer."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    async def test_fetch_compare_success(self):
        """Analyzing a normal diff should produce a valid DiffAnalysisResult."""
        diff = _make_diff(
            [
                FileChange(
                    filename="src/app.py",
                    file_type=FileType.PYTHON,
                    change_type=ChangeType.MODIFIED,
                    additions=10,
                    deletions=5,
                    patch="+def hello():\n+    return 'hi'\n-def old():\n-    pass",
                    before_content="def old():\n    pass\n",
                    after_content="def hello():\n    return 'hi'\n",
                )
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        assert isinstance(result, DiffAnalysisResult)
        assert len(result.file_analyses) == 1
        assert result.file_analyses[0].filename == "src/app.py"

    async def test_binary_file_handling(self):
        """Binary files should be flagged with BINARY_FILE_CHANGE risk."""
        diff = _make_diff(
            [
                FileChange(
                    filename="image.png",
                    file_type=FileType.UNKNOWN,
                    change_type=ChangeType.MODIFIED,
                    additions=0,
                    deletions=0,
                    patch="",
                    binary=True,
                )
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        assert len(result.file_analyses) == 1
        risks = result.file_analyses[0].risk_indicators
        assert "BINARY_FILE_CHANGE" in risks

    async def test_rename_detection(self):
        """Renamed files should be analyzable without error."""
        diff = _make_diff(
            [
                FileChange(
                    filename="new_name.py",
                    file_type=FileType.PYTHON,
                    change_type=ChangeType.RENAMED,
                    additions=0,
                    deletions=0,
                    patch="",
                )
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        assert len(result.file_analyses) == 1

    async def test_add_delete_modify(self):
        """Different change types (add, delete, modify) should be analyzed."""
        diff = _make_diff(
            [
                FileChange(
                    filename="added.py",
                    file_type=FileType.PYTHON,
                    change_type=ChangeType.ADDED,
                    additions=20,
                    deletions=0,
                    patch="+new code",
                    after_content="new code\n",
                ),
                FileChange(
                    filename="deleted.py",
                    file_type=FileType.PYTHON,
                    change_type=ChangeType.DELETED,
                    additions=0,
                    deletions=20,
                    patch="-old code",
                    before_content="old code\n",
                ),
                FileChange(
                    filename="modified.py",
                    file_type=FileType.PYTHON,
                    change_type=ChangeType.MODIFIED,
                    additions=5,
                    deletions=3,
                    patch="+changed\n-original",
                    before_content="original\n",
                    after_content="changed\n",
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        assert len(result.file_analyses) == 3

    async def test_empty_compare(self):
        """Empty diff (no files) should return valid result with zero complexity."""
        diff = _make_diff([])
        result = await self.analyzer.analyze_diff(diff)
        assert len(result.file_analyses) == 0
        assert result.change_complexity == 0.0

    async def test_github_api_failure_in_file(self):
        """If file analysis raises, minimal analysis with ANALYSIS_FAILED should be added."""
        bad_file = FileChange(
            filename="bad.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=5,
            deletions=5,
            patch="",
        )
        # Simulate error by giving before_content that will cause issues
        # Actually, let's patch _analyze_file_change to raise
        from unittest.mock import AsyncMock

        self.analyzer._analyze_file_change = AsyncMock(side_effect=RuntimeError("API error"))
        diff = _make_diff([bad_file])
        result = await self.analyzer.analyze_diff(diff)
        assert len(result.file_analyses) == 1
        assert "ANALYSIS_FAILED" in result.file_analyses[0].change_patterns

    async def test_large_diff_truncation(self):
        """Large diffs should still produce valid results with capped complexity."""
        files = [
            FileChange(
                filename=f"file_{i}.py",
                file_type=FileType.PYTHON,
                change_type=ChangeType.MODIFIED,
                additions=100,
                deletions=50,
                patch="+code\n" * 100,
                before_content="old\n" * 50,
                after_content="new\n" * 100,
            )
            for i in range(20)
        ]
        diff = _make_diff(files)
        result = await self.analyzer.analyze_diff(diff)
        assert result.change_complexity <= 10.0
        assert len(result.file_analyses) == 20
