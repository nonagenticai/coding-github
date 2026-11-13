"""Edge-case and boundary tests for PRValidationEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from coding_client.models.pr_validation import (
    BreakingChange,
    ChangeType,
    CodebaseContext,
    DiffAnalysisResult,
    DiffStats,
    DocumentationImpact,
    FileChange,
    FileType,
    PRAnalysisContext,
    PRDiff,
    SemanticAnalysisResult,
    ValidationIssue,
)

from services.pr_validation.core.pr_validation_engine import (
    PRValidationEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pr_diff(files=None, metadata=None):
    files = files or []
    return PRDiff(
        base_sha="aaa",
        head_sha="bbb",
        files_changed=files,
        stats=DiffStats.from_files(files),
        metadata=metadata or {},
    )


def _py_file(name="src/foo.py", additions=5, deletions=2):
    return FileChange(
        filename=name,
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=additions,
        deletions=deletions,
        patch="@@ -1 +1 @@\n-old\n+new",
        before_content="old",
        after_content="new",
    )


def _codebase_context():
    return CodebaseContext(
        total_files=100,
        total_lines=10000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


def _diff_analysis(**kw):
    kw.setdefault("file_analyses", [])
    kw.setdefault("overall_patterns", [])
    kw.setdefault("risk_factors", [])
    kw.setdefault("change_complexity", 0.0)
    kw.setdefault("requires_careful_review", False)
    return DiffAnalysisResult(**kw)


def _semantic_analysis(**kw):
    kw.setdefault("breaking_changes", [])
    kw.setdefault("logic_safety_issues", [])
    kw.setdefault("api_surface_changes", {})
    kw.setdefault("complexity_changes", {})
    return SemanticAnalysisResult(**kw)


def _doc_impact(**kw):
    kw.setdefault("docs_affected", [])
    kw.setdefault("outdated_docs", [])
    kw.setdefault("new_docs_needed", [])
    kw.setdefault("doc_completeness_score", 0.0)
    return DocumentationImpact(**kw)


# ---------------------------------------------------------------------------
# _parse_pr_diff — file_type classification fallback
# ---------------------------------------------------------------------------


class TestFileTypeClassificationFallback:
    """When file_change.file_type is None, file_classifier.classify_file is called."""

    @pytest.mark.asyncio
    async def test_file_type_none_triggers_classifier(self):
        engine = PRValidationEngine()
        pr_data = {
            "base": {"sha": "aaa"},
            "head": {"sha": "bbb"},
            "repository": {"full_name": "owner/repo"},
            "files": [
                {
                    "filename": "src/app.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 0,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
        }
        fc_none = FileChange(
            filename="src/app.py",
            file_type=None,
            change_type=ChangeType.MODIFIED,
            additions=1,
            deletions=0,
            patch="@@ -1 +1 @@\n-old\n+new",
        )
        with patch.object(engine.diff_parser, "parse_github_diff", return_value=[MagicMock()]):
            with patch.object(engine.diff_parser, "create_file_changes", return_value=[fc_none]):
                with patch.object(
                    engine.file_classifier, "classify_file", return_value=FileType.PYTHON
                ) as mock_cls:
                    with patch.object(
                        engine.diff_parser, "create_pr_diff", return_value=_pr_diff()
                    ):
                        await engine._parse_pr_diff(pr_data)
                        mock_cls.assert_called_once_with("src/app.py", None)

    @pytest.mark.asyncio
    async def test_file_type_set_skips_classifier(self):
        engine = PRValidationEngine()
        pr_data = {
            "base": {"sha": "a"},
            "head": {"sha": "b"},
            "repository": {"full_name": "o/r"},
            "files": [],
        }
        fc_set = FileChange(
            filename="app.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
        with patch.object(engine.diff_parser, "parse_github_diff", return_value=[MagicMock()]):
            with patch.object(engine.diff_parser, "create_file_changes", return_value=[fc_set]):
                with patch.object(engine.file_classifier, "classify_file") as mock_cls:
                    with patch.object(
                        engine.diff_parser, "create_pr_diff", return_value=_pr_diff()
                    ):
                        await engine._parse_pr_diff(pr_data)
                        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _create_content_provider
# ---------------------------------------------------------------------------


class TestContentProvider:
    """Content provider: GitHub API, local file, exceptions."""

    def test_github_service_returns_content(self):
        engine = PRValidationEngine()
        engine.github_service = MagicMock()
        engine.github_service.get_file_content.return_value = "file content"
        pr_data = {"repository": {"full_name": "o/r"}, "base": {"sha": "b"}, "head": {"sha": "h"}}
        provider = engine._create_content_provider(pr_data)
        result = provider("src/main.py", "head")
        assert result == "file content"
        engine.github_service.get_file_content.assert_called_once_with("o/r", "src/main.py", "h")

    def test_github_service_exception_falls_through(self):
        engine = PRValidationEngine()
        engine.github_service = MagicMock()
        engine.github_service.get_file_content.side_effect = RuntimeError("down")
        pr_data = {"repository": {"full_name": "o/r"}, "base": {"sha": "b"}, "head": {"sha": "h"}}
        provider = engine._create_content_provider(pr_data)
        result = provider("nonexistent.py", "head")
        assert result is None

    def test_missing_repo_info_returns_none(self):
        engine = PRValidationEngine()
        pr_data = {"repository": {}, "base": {}, "head": {}}
        provider = engine._create_content_provider(pr_data)
        result = provider("file.py", "head")
        assert result is None

    def test_local_utf8_file_read(self):
        engine = PRValidationEngine()
        repo_name = "_test_utf8_repo_"
        repo_dir = Path(f"/tmp/repos/{repo_name}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        target = repo_dir / "hello.py"
        try:
            target.write_text("print('hello')", encoding="utf-8")
            pr_data = {
                "repository": {"full_name": f"owner/{repo_name}"},
                "base": {"sha": "a"},
                "head": {"sha": "b"},
            }
            provider = engine._create_content_provider(pr_data)
            result = provider("hello.py", "head")
            assert result == "print('hello')"
        finally:
            target.unlink(missing_ok=True)
            repo_dir.rmdir()

    def test_local_latin1_fallback(self):
        engine = PRValidationEngine()
        repo_name = "_test_latin1_repo_"
        repo_dir = Path(f"/tmp/repos/{repo_name}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        target = repo_dir / "data.bin"
        try:
            target.write_bytes(b"\xff\xfe data with latin-1 chars")
            pr_data = {
                "repository": {"full_name": f"owner/{repo_name}"},
                "base": {"sha": "a"},
                "head": {"sha": "b"},
            }
            provider = engine._create_content_provider(pr_data)
            result = provider("data.bin", "head")
            assert result is not None
            assert "data with latin-1 chars" in result
        finally:
            target.unlink(missing_ok=True)
            repo_dir.rmdir()

    def test_outer_exception_returns_none(self):
        engine = PRValidationEngine()
        pr_data = {"repository": {"full_name": "o/r"}, "base": {"sha": "a"}, "head": {"sha": "b"}}
        provider = engine._create_content_provider(pr_data)
        with patch.dict(pr_data, {"repository": None}):
            result = provider("test.py", "head")
        assert result is None


# ---------------------------------------------------------------------------
# Semantic analysis — logic issues extend
# ---------------------------------------------------------------------------


class TestSemanticAnalysisIssues:
    """Logic safety issues from validator are included."""

    @pytest.mark.asyncio
    async def test_logic_issues_added(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        logic_issue = ValidationIssue(
            type="logic_inversion", severity="high", message="inv", file="src/foo.py", line=10
        )

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "ast_analysis":
                return _semantic_analysis()
            if name == "breaking_change_detection":
                return []
            if name == "logic_safety":
                return [logic_issue]
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_semantic_analysis(ctx)
        assert logic_issue in issues


# ---------------------------------------------------------------------------
# Impact analysis — coverage gaps and outdated docs
# ---------------------------------------------------------------------------


class TestImpactAnalysisIssues:
    """Coverage gaps and outdated docs become validation issues."""

    @pytest.mark.asyncio
    async def test_coverage_gaps_converted(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        mock_gap = MagicMock(
            severity="medium",
            symbol_name="uncovered",
            file_path="f.py",
            line_number=42,
            suggested_action="Add test",
        )
        mock_test_result = MagicMock()
        mock_test_result.coverage_gaps = [mock_gap]

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "test_coverage":
                return mock_test_result
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_impact_analysis(ctx)
        coverage_issues = [i for i in issues if i.type == "test_coverage_gap"]
        assert len(coverage_issues) == 1
        assert "uncovered" in coverage_issues[0].message

    @pytest.mark.asyncio
    async def test_outdated_docs_converted(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        mock_doc = _doc_impact(outdated_docs=["README outdated"])

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "documentation_sync":
                return mock_doc
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_impact_analysis(ctx)
        doc_issues = [i for i in issues if i.type == "documentation_sync"]
        assert len(doc_issues) == 1


# ---------------------------------------------------------------------------
# Context analysis — security/performance/maintainability issues extend
# ---------------------------------------------------------------------------


class TestContextAnalysisIssues:
    """Each analyzer's issues are extended into the result."""

    @pytest.mark.asyncio
    async def test_all_context_issues_combined(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )
        sec = ValidationIssue(type="sec", severity="high", message="s", file="a.py")
        perf = ValidationIssue(type="perf", severity="medium", message="p", file="a.py")
        maint = ValidationIssue(type="maint", severity="low", message="m", file="a.py")

        async def mock_analyzer(name, func, *args, **kwargs):
            if name == "security_analysis":
                return [sec]
            if name == "performance_analysis":
                return [perf]
            if name == "maintainability_analysis":
                return [maint]
            return None

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            issues = await engine._run_context_analysis(ctx)
        assert len(issues) == 3
        assert sec in issues and perf in issues and maint in issues


# ---------------------------------------------------------------------------
# _run_analyzer_with_timeout — timeout and error handling
# ---------------------------------------------------------------------------


class TestRunAnalyzerWithTimeout:
    """Timeout and error cases in analyzer execution."""

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        import asyncio

        engine = PRValidationEngine()
        engine.config.analyzer_timeout = 0.001

        async def slow_func():
            await asyncio.sleep(10)

        result = await engine._run_analyzer_with_timeout("slow", slow_func)
        assert result is None
        assert "slow_timeout" in engine.stats.errors_encountered

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        engine = PRValidationEngine()

        async def failing_func():
            raise ValueError("boom")

        result = await engine._run_analyzer_with_timeout("fail", failing_func)
        assert result is None
        assert any("fail_error" in e for e in engine.stats.errors_encountered)

    @pytest.mark.asyncio
    async def test_success_records_time(self):
        engine = PRValidationEngine()

        async def fast_func():
            return "ok"

        result = await engine._run_analyzer_with_timeout("fast", fast_func)
        assert result == "ok"
        assert "fast" in engine.stats.analyzer_times
        assert engine.stats.analyzer_times["fast"] >= 0


# ---------------------------------------------------------------------------
# _breaking_change_to_issue
# ---------------------------------------------------------------------------


class TestBreakingChangeToIssue:
    """Convert BreakingChange to ValidationIssue."""

    def test_conversion(self):
        engine = PRValidationEngine()
        bc = BreakingChange(
            type="FUNCTION_REMOVED",
            severity="HIGH",
            description="Function removed",
            file_path="api.py",
            affected_symbol="old_function",
            recommendation="Add deprecation notice",
            line_number=10,
        )
        issue = engine._breaking_change_to_issue(bc)
        assert issue.type == "function_removed"
        assert issue.severity == "high"
        assert issue.file == "api.py"
        assert issue.line == 10
        assert issue.rule == "breaking_change_detection"


# ---------------------------------------------------------------------------
# Full pipeline — all analyzers returning issues
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end pipeline with all analyzers returning issues."""

    @pytest.mark.asyncio
    async def test_all_issue_types_present(self):
        engine = PRValidationEngine()
        ctx = PRAnalysisContext(
            pr_diff=_pr_diff([_py_file()]), codebase_context=_codebase_context()
        )

        logic_issue = ValidationIssue(
            type="logic_inversion", severity="medium", message="inv", file="src/foo.py"
        )
        sec_issue = ValidationIssue(
            type="security_pattern_sqli", severity="high", message="sqli", file="src/foo.py"
        )
        perf_issue = ValidationIssue(
            type="perf", severity="medium", message="slow", file="src/foo.py"
        )
        maint_issue = ValidationIssue(
            type="complexity", severity="low", message="complex", file="src/foo.py"
        )

        mock_gap = MagicMock(
            severity="medium",
            symbol_name="untested",
            file_path="f.py",
            line_number=1,
            suggested_action="test",
        )
        mock_test_result = MagicMock()
        mock_test_result.coverage_gaps = [mock_gap]

        diff_result = _diff_analysis(change_complexity=1.0)

        async def mock_analyzer(name, func, *args, **kwargs):
            return {
                "diff_analysis": diff_result,
                "ast_analysis": _semantic_analysis(),
                "breaking_change_detection": [],
                "logic_safety": [logic_issue],
                "dependency_analysis": None,
                "test_coverage": mock_test_result,
                "documentation_sync": _doc_impact(outdated_docs=["README outdated"]),
                "security_analysis": [sec_issue],
                "performance_analysis": [perf_issue],
                "maintainability_analysis": [maint_issue],
            }.get(name)

        with patch.object(engine, "_run_analyzer_with_timeout", side_effect=mock_analyzer):
            result = await engine._execute_validation_pipeline(ctx)

        issue_types = {i.type for i in result.issues}
        assert "logic_inversion" in issue_types
        assert "security_pattern_sqli" in issue_types
        assert "test_coverage_gap" in issue_types
        assert "documentation_sync" in issue_types
        assert ctx.diff_analysis is diff_result
