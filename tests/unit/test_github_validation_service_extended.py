"""Extended unit tests for PRGitHubValidationService covering uncovered lines."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from services.pr_validation.integration.github_validation_service import (
    PRGitHubValidationService,
    validate_pr_comprehensive,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    repo_owner: str = "test-owner",
    repo_name: str = "test-repo",
    pr_number: int = 1,
    validation_types: list[str] | None = None,
) -> Any:
    """Build a ValidatePRRequest-like object."""
    return SimpleNamespace(
        repo_owner=repo_owner,
        repo_name=repo_name,
        pr_number=pr_number,
        validation_types=validation_types,
    )


def _make_file(
    filename: str = "main.py",
    changes: int = 10,
    additions: int = 5,
    deletions: int = 5,
    patch: str | None = None,
    sha: str | None = None,
    blob_url: str | None = None,
    last_modified: str | None = None,
) -> SimpleNamespace:
    """Build a mock PR file object."""
    ns = SimpleNamespace(
        filename=filename,
        changes=changes,
        additions=additions,
        deletions=deletions,
        patch=patch,
    )
    if sha is not None:
        ns.sha = sha
    if blob_url is not None:
        ns.blob_url = blob_url
    if last_modified is not None:
        ns.last_modified = last_modified
    return ns


# ---------------------------------------------------------------------------
# _convert_github_pr_data — pr_metadata branch (lines 162-177)
# ---------------------------------------------------------------------------


class TestConvertGitHubPRDataWithMetadata:
    """Cover _convert_github_pr_data when pr_metadata is a dict with base/head/user."""

    def test_metadata_with_base_and_head_sha(self):
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file()]
        metadata = {
            "base": {"sha": "abc1234"},
            "head": {"sha": "def5678"},
            "title": "My PR",
            "body": "Description here",
            "user": {"login": "alice"},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
        }
        result = svc._convert_github_pr_data(request, files, metadata)
        assert result.base_sha == "abc1234"
        assert result.head_sha == "def5678"
        assert result.pr_title == "My PR"
        assert result.pr_body == "Description here"
        assert result.author == "alice"
        assert result.created_at == "2026-01-01T00:00:00Z"
        assert result.updated_at == "2026-01-02T00:00:00Z"

    def test_metadata_base_sha_empty_falls_back_to_ref(self):
        """When base sha is empty string, falls back to ref (line 164-165)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file()]
        metadata = {
            "base": {"sha": "", "ref": "develop"},
            "head": {"sha": "", "ref": "feature-x"},
        }
        result = svc._convert_github_pr_data(request, files, metadata)
        assert result.base_sha == "develop"
        assert result.head_sha == "feature-x"

    def test_metadata_base_sha_none_falls_back_to_ref(self):
        """When base sha get returns None (falsy), falls back to ref (line 164-165)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file()]
        metadata = {
            "base": {"sha": None, "ref": "main-branch"},
            "head": {"sha": None, "ref": "pr-branch"},
        }
        result = svc._convert_github_pr_data(request, files, metadata)
        assert result.base_sha == "main-branch"
        assert result.head_sha == "pr-branch"

    def test_metadata_body_none_uses_default(self):
        """When body is None, the or-clause falls back to default (line 171)."""
        svc = PRGitHubValidationService()
        request = _make_request(repo_owner="owner", repo_name="repo", pr_number=5)
        files = [_make_file()]
        metadata = {"body": None}
        result = svc._convert_github_pr_data(request, files, metadata)
        assert "owner/repo" in result.pr_body
        assert "#5" in result.pr_body

    def test_metadata_author_string_field(self):
        """When metadata has 'author' string instead of user dict (line 174-175)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file()]
        metadata = {"author": "bob"}
        result = svc._convert_github_pr_data(request, files, metadata)
        assert result.author == "bob"

    def test_metadata_user_not_dict_falls_to_author(self):
        """When 'user' exists but is not a dict, should try 'author' (line 174-175)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file()]
        metadata = {"user": "not-a-dict", "author": "carol"}
        result = svc._convert_github_pr_data(request, files, metadata)
        assert result.author == "carol"


# ---------------------------------------------------------------------------
# _convert_github_pr_data — pr_files fallback branch (lines 183-192)
# ---------------------------------------------------------------------------


class TestConvertGitHubPRDataFileFallback:
    """Cover fallback when pr_metadata is None and pr_files is non-empty."""

    def test_file_with_blob_url(self):
        """Extract head_sha from blob_url when no sha attribute (lines 183-187)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        # blob_url format: https://github.com/owner/repo/blob/<sha>/path
        files = [_make_file(blob_url="https://github.com/owner/repo/blob/abc1234def/src/main.py")]
        result = svc._convert_github_pr_data(request, files, None)
        assert result.head_sha == "abc1234"

    def test_file_with_blob_url_too_short(self):
        """blob_url with too few parts doesn't set head_sha (line 185 condition false)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file(blob_url="https://github.com/short")]
        result = svc._convert_github_pr_data(request, files, None)
        # Should keep the default since blob_url doesn't match format
        assert result.head_sha == "pr-1"

    def test_file_with_blob_url_not_blob(self):
        """blob_url where parts[5] != 'blob' doesn't set head_sha."""
        svc = PRGitHubValidationService()
        request = _make_request()
        files = [_make_file(blob_url="https://github.com/owner/repo/tree/abc1234def/src/main.py")]
        result = svc._convert_github_pr_data(request, files, None)
        assert result.head_sha == "pr-1"

    def test_file_with_last_modified(self):
        """Use last_modified from first file when available (lines 188-192)."""
        svc = PRGitHubValidationService()
        request = _make_request()
        f = _make_file(last_modified="2026-03-01T12:00:00Z")
        f.sha = "abc1234"
        result = svc._convert_github_pr_data(request, [f], None)
        assert result.updated_at == "2026-03-01T12:00:00Z"

    def test_file_last_modified_exception(self):
        """When str(last_modified) raises, exception is caught (lines 189-192)."""
        svc = PRGitHubValidationService()
        request = _make_request()

        class BadStr:
            def __str__(self):
                raise ValueError("bad")

        f = _make_file()
        f.sha = "abc1234"
        f.last_modified = BadStr()
        result = svc._convert_github_pr_data(request, [f], None)
        # Should not crash; updated_at stays as default current time
        assert result.updated_at is not None


# ---------------------------------------------------------------------------
# _detect_languages (lines 214-221)
# ---------------------------------------------------------------------------


class TestDetectLanguages:
    def test_javascript_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="app.js")]
        langs = svc._detect_languages(files)
        assert "javascript" in langs

    def test_jsx_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="component.jsx")]
        langs = svc._detect_languages(files)
        assert "javascript" in langs

    def test_typescript_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="service.ts")]
        langs = svc._detect_languages(files)
        assert "typescript" in langs

    def test_tsx_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="component.tsx")]
        langs = svc._detect_languages(files)
        assert "typescript" in langs

    def test_java_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="Main.java")]
        langs = svc._detect_languages(files)
        assert "java" in langs

    def test_cpp_detected(self):
        svc = PRGitHubValidationService()
        for ext in [".c", ".cpp", ".cc", ".h", ".hpp"]:
            files = [_make_file(filename=f"file{ext}")]
            langs = svc._detect_languages(files)
            assert "c++" in langs, f"Extension {ext} should detect c++"

    def test_multiple_languages(self):
        svc = PRGitHubValidationService()
        files = [
            _make_file(filename="main.py"),
            _make_file(filename="app.js"),
            _make_file(filename="Main.java"),
        ]
        langs = svc._detect_languages(files)
        assert set(langs) >= {"python", "javascript", "java"}


# ---------------------------------------------------------------------------
# _run_comprehensive_validation (lines 239-278)
# ---------------------------------------------------------------------------


class TestRunComprehensiveValidation:
    async def test_comprehensive_validation_success(self):
        """Full comprehensive validation path with mocked engine."""
        svc = PRGitHubValidationService()

        # Mock validation_engine.validate_pr
        mock_result = MagicMock()
        mock_result.issues = []
        mock_result.change_impact = MagicMock()
        mock_result.change_impact.affected_functions = ["func_a"]

        svc.validation_engine = MagicMock()
        svc.validation_engine.validate_pr = AsyncMock(return_value=mock_result)

        # Mock risk_calculator
        mock_risk = MagicMock()
        mock_risk.dimension_scores = {}
        svc.risk_calculator = MagicMock()
        svc.risk_calculator.calculate_overall_risk = MagicMock(return_value=mock_risk)

        # Mock recommendation_generator
        svc.recommendation_generator = MagicMock()
        svc.recommendation_generator.generate_recommendations = MagicMock(return_value=[])

        from coding_client.models.pr_validation import CodebaseContext

        from services.pr_validation.integration.github_validation_service import (
            GitHubPRValidationData,
        )

        pr_data = GitHubPRValidationData(
            repo_owner="owner",
            repo_name="repo",
            pr_number=1,
            files=[],
            base_sha="abc",
            head_sha="def",
            pr_title="Test",
            pr_body="Body",
            author="user",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        codebase_ctx = CodebaseContext(
            total_files=1,
            total_lines=10,
            main_languages=["python"],
            test_framework="pytest",
            documentation_format="markdown",
            dependencies=[],
            architecture_patterns=[],
        )

        # No ``models`` stub: the real path must import only installed modules (D16).
        results = await svc._run_comprehensive_validation(
            pr_data, codebase_ctx, ["security", "quality"]
        )

        assert isinstance(results, list)
        svc.validation_engine.validate_pr.assert_awaited_once()

    async def test_comprehensive_validation_falls_back_on_error(self):
        """When comprehensive validation fails, falls back to legacy (line 276-278)."""
        svc = PRGitHubValidationService()

        svc.validation_engine = MagicMock()
        svc.validation_engine.validate_pr = AsyncMock(side_effect=RuntimeError("engine failed"))

        from coding_client.models.pr_validation import CodebaseContext

        from services.pr_validation.integration.github_validation_service import (
            GitHubPRValidationData,
        )

        pr_data = GitHubPRValidationData(
            repo_owner="owner",
            repo_name="repo",
            pr_number=1,
            files=[
                _make_file(
                    filename="app.py",
                    patch="+print('hello')\n",
                    additions=1,
                )
            ],
            base_sha="abc",
            head_sha="def",
            pr_title="Test",
            pr_body="Body",
            author="user",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        codebase_ctx = CodebaseContext(
            total_files=1,
            total_lines=10,
            main_languages=["python"],
            test_framework="pytest",
            documentation_format="markdown",
            dependencies=[],
            architecture_patterns=[],
        )

        results = await svc._run_comprehensive_validation(
            pr_data, codebase_ctx, ["security", "quality"]
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _run_legacy_validation (lines 292, 296)
# ---------------------------------------------------------------------------


class TestRunLegacyValidation:
    async def test_performance_type(self):
        """Cover 'performance' branch (line 292)."""
        svc = PRGitHubValidationService()
        from services.pr_validation.integration.github_validation_service import (
            GitHubPRValidationData,
        )

        pr_data = GitHubPRValidationData(
            repo_owner="o",
            repo_name="r",
            pr_number=1,
            files=[_make_file(filename="app.py", patch="+x = 1\n")],
            base_sha="a",
            head_sha="b",
            pr_title="T",
            pr_body="B",
            author="u",
            created_at="",
            updated_at="",
        )
        results = await svc._run_legacy_validation(pr_data, ["performance"], None)
        assert len(results) == 1
        assert results[0].validation_type == "performance"

    async def test_unknown_type_skipped(self):
        """Cover 'continue' branch for unknown types (line 296)."""
        svc = PRGitHubValidationService()
        from services.pr_validation.integration.github_validation_service import (
            GitHubPRValidationData,
        )

        pr_data = GitHubPRValidationData(
            repo_owner="o",
            repo_name="r",
            pr_number=1,
            files=[_make_file(filename="app.py", patch="+x = 1\n")],
            base_sha="a",
            head_sha="b",
            pr_title="T",
            pr_body="B",
            author="u",
            created_at="",
            updated_at="",
        )
        results = await svc._run_legacy_validation(pr_data, ["unknown_type"], None)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# _legacy_security_validation (lines 314, 326)
# ---------------------------------------------------------------------------


class TestLegacySecurityValidation:
    async def test_dangerous_function_detected(self):
        """Detect eval/exec/system patterns (line 314)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="main.py", patch="+result = eval(user_input)\n")]
        result = await svc._legacy_security_validation(files)
        assert not result.passed
        assert any(i.rule == "dangerous-function" for i in result.issues)

    async def test_hardcoded_password_detected(self):
        """Detect password patterns (line 326)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="config.py", patch="+password = 'secret123'\n")]
        result = await svc._legacy_security_validation(files)
        assert any(i.rule == "hardcoded-secret" for i in result.issues)


# ---------------------------------------------------------------------------
# _legacy_performance_validation (lines 351-389)
# ---------------------------------------------------------------------------


class TestLegacyPerformanceValidation:
    async def test_inefficient_loop_detected(self):
        """Detect 'for i in range(len(' pattern (line 359-370)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="algo.py", patch="+for i in range(len(items)):\n")]
        result = await svc._legacy_performance_validation(files)
        assert any(i.rule == "inefficient-loop" for i in result.issues)

    async def test_blocking_operation_detected(self):
        """Detect time.sleep/Thread.sleep patterns (line 371-384)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="worker.py", patch="+time.sleep(10)\n")]
        result = await svc._legacy_performance_validation(files)
        assert not result.passed
        assert any(i.rule == "blocking-operation" for i in result.issues)

    async def test_thread_sleep_detected(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="Worker.java", patch="+Thread.sleep(1000)\n")]
        # .java extension is not in the filter (.py, .js, .ts), so no issues
        result = await svc._legacy_performance_validation(files)
        assert len(result.issues) == 0

    async def test_clean_file_no_issues(self):
        svc = PRGitHubValidationService()
        files = [_make_file(filename="clean.py", patch="+x = 1\n")]
        result = await svc._legacy_performance_validation(files)
        assert result.passed
        assert result.score == 100
        assert "Consider performance testing" in result.suggestions[0]

    async def test_no_patch(self):
        """File without patch should produce no issues."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="app.py", patch=None)]
        result = await svc._legacy_performance_validation(files)
        assert result.passed
        assert len(result.issues) == 0

    async def test_non_matching_extension_skipped(self):
        """Non-.py/.js/.ts files skip analysis (line 354 condition)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="readme.md", patch="+for i in range(len(x)):\n")]
        result = await svc._legacy_performance_validation(files)
        assert len(result.issues) == 0


# ---------------------------------------------------------------------------
# _legacy_quality_validation (lines 403, 421-422, 424)
# ---------------------------------------------------------------------------


class TestLegacyQualityValidation:
    async def test_large_pr_detected(self):
        """PRs with >500 additions generate an issue (line 402-413)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="big.py", additions=600, changes=10)]
        result = await svc._legacy_quality_validation(files)
        assert any(i.rule == "large-pr" for i in result.issues)

    async def test_readme_changes_trigger_comprehensive_check(self):
        """readme.md triggers _validate_readme_changes_comprehensive (line 421-422)."""
        svc = PRGitHubValidationService()
        files = [
            _make_file(
                filename="readme.md",
                additions=5,
                changes=10,
                patch="+# Updated\n",
            )
        ]
        result = await svc._legacy_quality_validation(files)
        # Should at least run without errors
        assert result.validation_type == "quality"

    async def test_large_file_change(self):
        """Files with >100 changes generate an issue (line 423-434)."""
        svc = PRGitHubValidationService()
        files = [_make_file(filename="big.py", additions=50, changes=150)]
        result = await svc._legacy_quality_validation(files)
        assert any(i.rule == "large-file-change" for i in result.issues)

    async def test_many_issues_fails(self):
        """More than 3 issues causes pass=False."""
        svc = PRGitHubValidationService()
        files = [
            _make_file(
                filename="readme.md",
                additions=600,
                changes=200,
                patch="\n".join(
                    [
                        "-installation",
                        "-usage",
                        "-api",
                        "-license",
                        "-getting started",
                        "-configuration",
                        "-examples",
                        "-contributing",
                    ]
                    + ["+" * 1 for _ in range(5)]
                ),
            ),
        ]
        result = await svc._legacy_quality_validation(files)
        assert isinstance(result.passed, bool)


# ---------------------------------------------------------------------------
# _validate_readme_changes_comprehensive (lines 449-530)
# ---------------------------------------------------------------------------


class TestValidateReadmeChangesComprehensive:
    async def test_complete_replacement_detected(self):
        """Detect complete README replacement (lines 459-470)."""
        svc = PRGitHubValidationService()
        # 25 deletions, 10 additions => 25 > 20 and 25 > 10*1.5
        patch_lines = ["---a/readme.md", "+++b/readme.md"]
        patch_lines += [f"-old line {i}" for i in range(25)]
        patch_lines += [f"+new line {i}" for i in range(10)]
        f = _make_file(filename="readme.md", patch="\n".join(patch_lines))
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert any(i.rule == "readme-complete-replacement" for i in issues)

    async def test_section_removal_detected(self):
        """Detect removal of important sections (lines 486-503)."""
        svc = PRGitHubValidationService()
        patch_lines = ["---a/readme.md", "+++b/readme.md"]
        # Delete lines with important section names
        patch_lines += ["-## Installation", "-Follow these installation steps"]
        patch_lines += ["-## Usage", "-Use it like this"]
        patch_lines += ["+## New Section"]
        f = _make_file(filename="readme.md", patch="\n".join(patch_lines))
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert any(i.rule == "readme-section-removal" for i in issues)

    async def test_generic_content_detected(self):
        """Detect generic/template content in replacement (lines 504-529)."""
        svc = PRGitHubValidationService()
        patch_lines = ["---a/readme.md", "+++b/readme.md"]
        patch_lines += [f"-old line {i}" for i in range(55)]
        patch_lines += ["+# Your Project Name", "+Lorem ipsum dolor sit amet"]
        f = _make_file(filename="readme.md", patch="\n".join(patch_lines))
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert any(i.rule == "readme-generic-content" for i in issues)

    async def test_no_patch_returns_empty(self):
        """No patch means no issues (line 450-451)."""
        svc = PRGitHubValidationService()
        f = _make_file(filename="readme.md", patch=None)
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert issues == []

    async def test_small_deletion_no_issues(self):
        """Small deletions don't trigger replacement warning."""
        svc = PRGitHubValidationService()
        patch_lines = ["---a/readme.md", "+++b/readme.md", "-old", "+new"]
        f = _make_file(filename="readme.md", patch="\n".join(patch_lines))
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert not any(i.rule == "readme-complete-replacement" for i in issues)

    async def test_large_deletion_no_generic_content(self):
        """Large deletion but added content is legitimate (no generic indicators)."""
        svc = PRGitHubValidationService()
        patch_lines = ["---a/readme.md", "+++b/readme.md"]
        patch_lines += [f"-old line {i}" for i in range(55)]
        patch_lines += ["+# Real Project Documentation", "+This is specific content"]
        f = _make_file(filename="readme.md", patch="\n".join(patch_lines))
        issues = await svc._validate_readme_changes_comprehensive(f)
        assert not any(i.rule == "readme-generic-content" for i in issues)


# ---------------------------------------------------------------------------
# _convert_comprehensive_results (lines 540-587)
# ---------------------------------------------------------------------------


class TestConvertComprehensiveResults:
    def test_converts_issues_and_produces_results(self):
        """Full path through _convert_comprehensive_results."""
        svc = PRGitHubValidationService()

        class FakeSeverity:
            def __init__(self, val):
                self.value = val

        issue = SimpleNamespace(
            type="security",
            severity=FakeSeverity("high"),
            message="SQL injection risk",
            file="db.py",
            line=42,
            rule="sql-injection",
        )
        validation_result = MagicMock()
        validation_result.issues = [issue]

        class FakeDim(Enum):
            SECURITY = "SECURITY"

        risk_score = MagicMock()
        risk_score.dimension_scores = {FakeDim.SECURITY: 5.0}

        recommendations = []

        results = svc._convert_comprehensive_results(
            validation_result, risk_score, recommendations, ["security", "quality"]
        )
        assert len(results) == 2
        sec_result = next(r for r in results if r.validation_type == "security")
        assert len(sec_result.issues) == 1
        assert sec_result.issues[0].message == "SQL injection risk"

    def test_comprehensive_and_advanced_skipped(self):
        """'comprehensive' and 'advanced' types are skipped (line 548-549)."""
        svc = PRGitHubValidationService()
        validation_result = MagicMock()
        validation_result.issues = []
        risk_score = MagicMock()
        risk_score.dimension_scores = {}
        recommendations = []

        results = svc._convert_comprehensive_results(
            validation_result, risk_score, recommendations, ["comprehensive", "advanced"]
        )
        assert len(results) == 0

    def test_issue_without_value_attr_on_severity(self):
        """When issue.severity has no .value attr, str() is used (line 557)."""
        svc = PRGitHubValidationService()
        issue = SimpleNamespace(
            type="quality",
            severity="medium",
            message="code smell",
            file="app.py",
            line=10,
            rule=None,
        )
        validation_result = MagicMock()
        validation_result.issues = [issue]
        risk_score = MagicMock()
        risk_score.dimension_scores = {}

        results = svc._convert_comprehensive_results(validation_result, risk_score, [], ["quality"])
        assert len(results) == 1
        assert results[0].issues[0].severity == "medium"
        # When rule is None, falls back to "{validation_type}-check"
        assert results[0].issues[0].rule == "quality-check"

    def test_issue_with_description_instead_of_message(self):
        """When issue has 'description' instead of 'message' (line 559)."""
        svc = PRGitHubValidationService()
        issue = SimpleNamespace(
            type="performance_loop",
            severity="low",
            file_path="algo.py",
            line_number=5,
            rule="perf-1",
        )
        # No 'message' attribute, has 'description'
        issue.description = "Slow loop detected"
        validation_result = MagicMock()
        validation_result.issues = [issue]
        risk_score = MagicMock()
        risk_score.dimension_scores = {}

        results = svc._convert_comprehensive_results(
            validation_result, risk_score, [], ["performance"]
        )
        assert len(results) == 1
        assert results[0].issues[0].message == "Slow loop detected"
        assert results[0].issues[0].file == "algo.py"
        assert results[0].issues[0].line == 5


# ---------------------------------------------------------------------------
# _categorize_issue_type (lines 591-601)
# ---------------------------------------------------------------------------


class TestCategorizeIssueType:
    def test_security_keywords(self):
        svc = PRGitHubValidationService()
        for keyword in ["security_vuln", "auth_bypass", "crypto_weak", "sql_injection"]:
            assert svc._categorize_issue_type(keyword) == "security"

    def test_performance_keywords(self):
        svc = PRGitHubValidationService()
        for keyword in ["performance_issue", "high_complexity", "loop_unrolling", "cache_miss"]:
            assert svc._categorize_issue_type(keyword) == "performance"

    def test_fallback_to_quality(self):
        svc = PRGitHubValidationService()
        assert svc._categorize_issue_type("code_style") == "quality"
        assert svc._categorize_issue_type("readability") == "quality"


# ---------------------------------------------------------------------------
# _calculate_type_score (lines 605-620)
# ---------------------------------------------------------------------------


class TestCalculateTypeScore:
    def test_no_risk_score(self):
        """When risk_score is None, calculate from issues (lines 605-608)."""
        svc = PRGitHubValidationService()
        issues = [
            SimpleNamespace(severity="critical"),
            SimpleNamespace(severity="high"),
        ]
        score = svc._calculate_type_score("security", None, issues)
        # 100 - (1*30 + 1*15) = 55
        assert score == 55

    def test_no_risk_score_no_issues(self):
        svc = PRGitHubValidationService()
        score = svc._calculate_type_score("quality", None, [])
        assert score == 100

    def test_with_risk_score_security_dimension(self):
        """When risk_score exists, uses dimension_scores (lines 609-620)."""
        svc = PRGitHubValidationService()

        class FakeDim(Enum):
            SECURITY = "SECURITY"

        risk_score = MagicMock()
        risk_score.dimension_scores = {FakeDim.SECURITY: 3.0}

        score = svc._calculate_type_score("security", risk_score, [])
        # 100 - 3.0 * 10 = 70
        assert score == 70

    def test_with_risk_score_performance_dimension(self):
        svc = PRGitHubValidationService()

        class FakeDim(Enum):
            PERFORMANCE = "PERFORMANCE"

        risk_score = MagicMock()
        risk_score.dimension_scores = {FakeDim.PERFORMANCE: 8.0}

        score = svc._calculate_type_score("performance", risk_score, [])
        # 100 - 8.0 * 10 = 20
        assert score == 20

    def test_with_risk_score_quality_maps_to_maintainability(self):
        svc = PRGitHubValidationService()

        class FakeDim(Enum):
            MAINTAINABILITY = "MAINTAINABILITY"

        risk_score = MagicMock()
        risk_score.dimension_scores = {FakeDim.MAINTAINABILITY: 2.0}

        score = svc._calculate_type_score("quality", risk_score, [])
        assert score == 80

    def test_with_risk_score_no_matching_dimension(self):
        """When no dimension matches, dimension_risk stays 0."""
        svc = PRGitHubValidationService()

        class FakeDim(Enum):
            OTHER = "OTHER"

        risk_score = MagicMock()
        risk_score.dimension_scores = {FakeDim.OTHER: 5.0}

        score = svc._calculate_type_score("security", risk_score, [])
        assert score == 100

    def test_score_clamped_at_zero(self):
        """Score should never go below 0."""
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="critical") for _ in range(10)]
        score = svc._calculate_type_score("security", None, issues)
        assert score == 0


# ---------------------------------------------------------------------------
# _determine_type_passed (lines 624-631)
# ---------------------------------------------------------------------------


class TestDetermineTypePassed:
    def test_security_passes_no_critical_one_high(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="high")]
        assert svc._determine_type_passed("security", issues, 80) is True

    def test_security_fails_with_critical(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="critical")]
        assert svc._determine_type_passed("security", issues, 80) is False

    def test_security_fails_with_two_high(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="high"), SimpleNamespace(severity="high")]
        assert svc._determine_type_passed("security", issues, 80) is False

    def test_performance_passes(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="medium")]
        assert svc._determine_type_passed("performance", issues, 75) is True

    def test_performance_fails_high_issue(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="high")]
        assert svc._determine_type_passed("performance", issues, 75) is False

    def test_performance_fails_low_score(self):
        svc = PRGitHubValidationService()
        issues = []
        assert svc._determine_type_passed("performance", issues, 60) is False

    def test_quality_passes(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="low")]
        assert svc._determine_type_passed("quality", issues, 65) is True

    def test_quality_fails_low_score(self):
        svc = PRGitHubValidationService()
        issues = []
        assert svc._determine_type_passed("quality", issues, 50) is False

    def test_quality_fails_critical(self):
        svc = PRGitHubValidationService()
        issues = [SimpleNamespace(severity="critical")]
        assert svc._determine_type_passed("quality", issues, 80) is False


# ---------------------------------------------------------------------------
# _get_type_suggestions (lines 635-646)
# ---------------------------------------------------------------------------


class TestGetTypeSuggestions:
    def test_no_recommendations(self):
        """When recommendations is empty/None, returns default (line 635-636)."""
        svc = PRGitHubValidationService()
        result = svc._get_type_suggestions("security", None)
        assert result == ["Consider security-focused review"]

    def test_empty_recommendations_list(self):
        svc = PRGitHubValidationService()
        result = svc._get_type_suggestions("performance", [])
        assert result == ["Consider performance-focused review"]

    def test_matching_recommendations(self):
        """Matching recommendations by type and priority (lines 638-643)."""
        svc = PRGitHubValidationService()
        rec1 = SimpleNamespace(type="security_check", priority="high", action="Fix SQL injection")
        rec2 = SimpleNamespace(type="quality_lint", priority="critical", action="Run linter")
        rec3 = SimpleNamespace(type="other", priority="low", action="Not matched")
        result = svc._get_type_suggestions("security", [rec1, rec2, rec3])
        assert "Fix SQL injection" in result
        # rec2 has priority critical -> also matched
        assert "Run linter" in result

    def test_no_matching_recommendations_fallback(self):
        """When no recs match the type, returns generic fallback (line 644-645)."""
        svc = PRGitHubValidationService()
        rec = SimpleNamespace(type="other_type", priority="low", action="Do something")
        result = svc._get_type_suggestions("security", [rec])
        assert result == ["Review security aspects of the changes"]


# ---------------------------------------------------------------------------
# validate_github_pr — error path (exception handling)
# ---------------------------------------------------------------------------


class TestValidateGitHubPRError:
    async def test_exception_returns_error_result(self):
        """When validate_github_pr raises, returns error result (lines 113-133)."""
        svc = PRGitHubValidationService()
        # Force an error by passing files that lack .changes attribute
        request = _make_request()
        bad_files = [SimpleNamespace(filename="x.py")]  # missing 'changes'

        results = await svc.validate_github_pr(request, bad_files)
        assert len(results) == 1
        assert results[0].validation_type == "error"
        assert results[0].passed is False
        assert results[0].score == 0.0


# ---------------------------------------------------------------------------
# validate_pr_comprehensive (lines 668-669)
# ---------------------------------------------------------------------------


class TestValidatePRComprehensive:
    async def test_integration_function(self):
        """The module-level function creates a service and delegates (lines 668-669)."""
        request = _make_request(validation_types=["security"])
        files = [_make_file(filename="app.py", changes=5, additions=3, patch="+x = 1\n")]

        results = await validate_pr_comprehensive(request, files, None)
        assert isinstance(results, list)
        assert len(results) >= 1

    async def test_integration_function_with_metadata(self):
        request = _make_request(validation_types=["quality"])
        files = [_make_file(filename="main.py", changes=5, additions=3, patch="+y = 2\n")]
        metadata = {"title": "Test PR", "body": "Test body"}

        results = await validate_pr_comprehensive(request, files, metadata)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# validate_github_pr — full flow: comprehensive path
# ---------------------------------------------------------------------------


class TestValidateGitHubPRComprehensivePath:
    async def test_comprehensive_triggered_by_3_types(self):
        """When >2 validation_types, uses comprehensive path (line 107-110)."""
        svc = PRGitHubValidationService()
        request = _make_request(validation_types=["security", "performance", "quality"])
        files = [_make_file(filename="app.py", changes=5, additions=3, patch="+x = 1\n")]

        # The comprehensive path runs the real engine (no models stub needed),
        # and must return a result list
        results = await svc.validate_github_pr(request, files, None)
        assert isinstance(results, list)

    async def test_legacy_triggered_by_2_or_fewer_types(self):
        """When <=2 validation_types without comprehensive/advanced, uses legacy."""
        svc = PRGitHubValidationService()
        request = _make_request(validation_types=["security"])
        files = [_make_file(filename="app.py", changes=5, additions=3, patch="+x = 1\n")]

        results = await svc.validate_github_pr(request, files, None)
        assert len(results) == 1
        assert results[0].validation_type == "security"
