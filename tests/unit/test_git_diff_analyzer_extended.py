"""Extended tests for git_diff_analyzer.py targeting uncovered lines."""

from unittest.mock import patch

from coding_client.models.pr_validation import (
    ChangeSeverity,
    ChangeType,
    DiffStats,
    FileChange,
    FileChangeAnalysis,
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


def _fc(**kwargs):
    """Shortcut to build a FileChange with sensible defaults."""
    defaults = dict(
        filename="src/app.py",
        file_type=FileType.PYTHON,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    defaults.update(kwargs)
    return FileChange(**defaults)


class TestChangeSeverityBranches:
    """Cover lines 246, 250 – MAJOR and MODERATE severity returns."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_major_severity_via_line_change_ratio(self):
        """line_change_ratio > major_change_threshold (0.4) → MAJOR (line 246)."""
        severity = self.analyzer._determine_change_severity(
            line_change_ratio=0.5,
            content_similarity=0.8,
            file_change=_fc(additions=5, deletions=5),
        )
        assert severity == ChangeSeverity.MAJOR

    def test_major_severity_via_low_similarity(self):
        """content_similarity < 0.6 → MAJOR (line 246)."""
        severity = self.analyzer._determine_change_severity(
            line_change_ratio=0.2,
            content_similarity=0.5,
            file_change=_fc(additions=5, deletions=5),
        )
        assert severity == ChangeSeverity.MAJOR

    def test_moderate_severity_via_line_change_ratio(self):
        """line_change_ratio > 0.1 → MODERATE (line 250)."""
        severity = self.analyzer._determine_change_severity(
            line_change_ratio=0.15,
            content_similarity=0.8,
            file_change=_fc(additions=5, deletions=5),
        )
        assert severity == ChangeSeverity.MODERATE

    def test_moderate_severity_via_high_total_changes(self):
        """additions + deletions > 20 → MODERATE (line 250)."""
        severity = self.analyzer._determine_change_severity(
            line_change_ratio=0.05,
            content_similarity=0.8,
            file_change=_fc(additions=15, deletions=10),
        )
        assert severity == ChangeSeverity.MODERATE


class TestDetectChangePatterns:
    """Cover lines 286, 292, 295, 298, 301, 304, 307-311 – pattern detection."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_import_changes_pattern(self):
        """Patch with import lines → IMPORT_CHANGES (line 286, 327)."""
        fc = _fc(patch="+import os\n-import sys")
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "IMPORT_CHANGES" in patterns

    def test_logic_inversions_pattern(self):
        """Patch that flips == to != on similar lines → LOGIC_INVERSIONS (line 292, 375)."""
        fc = _fc(patch=("-    if x == 0:\n+    if x != 0:\n"))
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "LOGIC_INVERSIONS" in patterns

    def test_error_handling_changes_pattern(self):
        """Patch with try/except → ERROR_HANDLING_CHANGES (line 295, 398)."""
        fc = _fc(patch="+    try:\n+        pass\n+    except:\n+        pass")
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "ERROR_HANDLING_CHANGES" in patterns

    def test_security_sensitive_changes_pattern(self):
        """Patch with password references → SECURITY_SENSITIVE_CHANGES (line 298, 431)."""
        fc = _fc(patch="+password = 'secret123'")
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "SECURITY_SENSITIVE_CHANGES" in patterns

    def test_configuration_changes_pattern(self):
        """Config file with key=value → CONFIGURATION_CHANGES (line 301, 441-452)."""
        fc = _fc(
            filename="config.yaml",
            file_type=FileType.YAML,
            patch="+database: postgres",
        )
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "CONFIGURATION_CHANGES" in patterns

    def test_test_changes_pattern(self):
        """Test file with test function → TEST_CHANGES (line 304, 461-479)."""
        fc = _fc(
            filename="tests/test_foo.py",
            file_type=FileType.PYTHON,
            patch="+def test_something():\n+    assert True",
        )
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "TEST_CHANGES" in patterns

    def test_documentation_changes_pattern(self):
        """Doc file with added content → DOCUMENTATION_CHANGES (line 307, 489)."""
        fc = _fc(
            filename="README.md",
            file_type=FileType.MARKDOWN,
            patch="+# New Section\n+Some text",
        )
        patterns = self.analyzer._detect_change_patterns(fc)
        assert "DOCUMENTATION_CHANGES" in patterns

    def test_pattern_detection_error(self):
        """Exception during pattern detection → PATTERN_DETECTION_ERROR (lines 309-311)."""
        fc = _fc(patch="+some code")
        with patch.object(self.analyzer, "_has_import_changes", side_effect=RuntimeError("boom")):
            patterns = self.analyzer._detect_change_patterns(fc)
        assert "PATTERN_DETECTION_ERROR" in patterns


class TestLinesAreSimilar:
    """Cover lines 493-497 – _lines_are_similar."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_similar_lines(self):
        assert self.analyzer._lines_are_similar("if x == 0:", "if x != 0:") is True

    def test_dissimilar_lines(self):
        assert self.analyzer._lines_are_similar("import os", "return 42") is False

    def test_empty_line1(self):
        assert self.analyzer._lines_are_similar("", "something") is False

    def test_empty_line2(self):
        assert self.analyzer._lines_are_similar("something", "   ") is False


class TestRiskIndicators:
    """Cover lines 521-525, 529, 533-537, 541, 547, 550, 554, 566."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_high_impact_file_major_change(self):
        """High-impact file with MAJOR severity → HIGH_IMPACT_FILE_MAJOR_CHANGE (lines 521-525)."""
        fc = _fc(filename="setup.py", additions=50, deletions=50)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.MAJOR, [])
        assert "HIGH_IMPACT_FILE_MAJOR_CHANGE" in risks

    def test_high_impact_file_complete_rewrite(self):
        """High-impact file with COMPLETE_REWRITE → HIGH_IMPACT_FILE_MAJOR_CHANGE."""
        fc = _fc(filename="package.json", additions=100, deletions=100)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.COMPLETE_REWRITE, [])
        assert "HIGH_IMPACT_FILE_MAJOR_CHANGE" in risks

    def test_security_sensitive_file_modified(self):
        """File with security keyword in name → SECURITY_SENSITIVE_FILE_MODIFIED (line 529)."""
        fc = _fc(filename="src/auth_handler.py")
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.MINOR, [])
        assert "SECURITY_SENSITIVE_FILE_MODIFIED" in risks

    def test_config_file_major_change(self):
        """Config file with MAJOR severity → CONFIG_FILE_MAJOR_CHANGE (lines 533-537)."""
        fc = _fc(filename="settings.yaml", additions=30, deletions=20)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.MAJOR, [])
        assert "CONFIG_FILE_MAJOR_CHANGE" in risks

    def test_config_file_complete_rewrite(self):
        """Config file with COMPLETE_REWRITE → CONFIG_FILE_MAJOR_CHANGE."""
        fc = _fc(filename="app.json", additions=100, deletions=100)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.COMPLETE_REWRITE, [])
        assert "CONFIG_FILE_MAJOR_CHANGE" in risks

    def test_dangerous_logic_changes(self):
        """LOGIC_INVERSIONS pattern → DANGEROUS_LOGIC_CHANGES (line 541)."""
        fc = _fc()
        risks = self.analyzer._identify_risk_indicators(
            fc, ChangeSeverity.MINOR, ["LOGIC_INVERSIONS"]
        )
        assert "DANGEROUS_LOGIC_CHANGES" in risks

    def test_security_changes_detected(self):
        """SECURITY_SENSITIVE_CHANGES → SECURITY_CHANGES_DETECTED (line 547)."""
        fc = _fc()
        risks = self.analyzer._identify_risk_indicators(
            fc, ChangeSeverity.MINOR, ["SECURITY_SENSITIVE_CHANGES"]
        )
        assert "SECURITY_CHANGES_DETECTED" in risks

    def test_error_handling_modified(self):
        """ERROR_HANDLING_CHANGES → ERROR_HANDLING_MODIFIED (line 550)."""
        fc = _fc()
        risks = self.analyzer._identify_risk_indicators(
            fc, ChangeSeverity.MINOR, ["ERROR_HANDLING_CHANGES"]
        )
        assert "ERROR_HANDLING_MODIFIED" in risks

    def test_very_large_change(self):
        """total_changes > 500 → VERY_LARGE_CHANGE (line 554)."""
        fc = _fc(additions=300, deletions=250)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.MINOR, [])
        assert "VERY_LARGE_CHANGE" in risks

    def test_critical_file_complete_rewrite(self):
        """readme.md with COMPLETE_REWRITE → CRITICAL_FILE_COMPLETE_REWRITE (line 566)."""
        fc = _fc(filename="readme.md", additions=100, deletions=100)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.COMPLETE_REWRITE, [])
        assert "CRITICAL_FILE_COMPLETE_REWRITE" in risks

    def test_setup_py_critical_rewrite(self):
        fc = _fc(filename="setup.py", additions=100, deletions=100)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.COMPLETE_REWRITE, [])
        assert "CRITICAL_FILE_COMPLETE_REWRITE" in risks

    def test_package_json_critical_rewrite(self):
        fc = _fc(filename="package.json", additions=100, deletions=100)
        risks = self.analyzer._identify_risk_indicators(fc, ChangeSeverity.COMPLETE_REWRITE, [])
        assert "CRITICAL_FILE_COMPLETE_REWRITE" in risks


class TestAnalyzePRPatterns:
    """Cover lines 599, 608, 614."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def _fca(
        self, severity=ChangeSeverity.MINOR, patterns=None, risks=None, file_type=FileType.PYTHON
    ):
        return FileChangeAnalysis(
            filename="f.py",
            change_severity=severity,
            change_patterns=patterns or [],
            risk_indicators=risks or [],
            line_change_ratio=0.1,
            content_similarity=0.9,
            file_type=file_type,
        )

    def test_many_major_changes(self):
        """major_count > 5 → MANY_MAJOR_CHANGES (line 599)."""
        analyses = [self._fca(severity=ChangeSeverity.MAJOR) for _ in range(6)]
        patterns = self.analyzer._analyze_pr_patterns(analyses)
        assert "MANY_MAJOR_CHANGES" in patterns

    def test_multiple_complete_rewrites(self):
        """rewrite_count > 3 → MULTIPLE_COMPLETE_REWRITES."""
        analyses = [self._fca(severity=ChangeSeverity.COMPLETE_REWRITE) for _ in range(4)]
        patterns = self.analyzer._analyze_pr_patterns(analyses)
        assert "CONTAINS_COMPLETE_REWRITES" in patterns
        assert "MULTIPLE_COMPLETE_REWRITES" in patterns

    def test_coordinated_security_changes(self):
        """security_changes > 1 → COORDINATED_SECURITY_CHANGES (line 608)."""
        analyses = [
            self._fca(patterns=["SECURITY_SENSITIVE_CHANGES"]),
            self._fca(patterns=["SECURITY_SENSITIVE_CHANGES"]),
        ]
        patterns = self.analyzer._analyze_pr_patterns(analyses)
        assert "COORDINATED_SECURITY_CHANGES" in patterns

    def test_multiple_config_changes(self):
        """config_changes > 2 → MULTIPLE_CONFIG_CHANGES (line 614)."""
        analyses = [
            self._fca(patterns=["CONFIGURATION_CHANGES"]),
            self._fca(patterns=["CONFIGURATION_CHANGES"]),
            self._fca(patterns=["CONFIGURATION_CHANGES"]),
        ]
        patterns = self.analyzer._analyze_pr_patterns(analyses)
        assert "MULTIPLE_CONFIG_CHANGES" in patterns

    def test_code_changes_without_tests(self):
        """Code changes without test changes → CODE_CHANGES_WITHOUT_TESTS."""
        analyses = [
            self._fca(file_type=FileType.PYTHON),
        ]
        patterns = self.analyzer._analyze_pr_patterns(analyses)
        assert "CODE_CHANGES_WITHOUT_TESTS" in patterns


class TestRequiresCarefulReview:
    """Cover lines 709, 717."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def _fca(self, severity=ChangeSeverity.MINOR, patterns=None, risks=None):
        return FileChangeAnalysis(
            filename="f.py",
            change_severity=severity,
            change_patterns=patterns or [],
            risk_indicators=risks or [],
            line_change_ratio=0.1,
            content_similarity=0.9,
            file_type=FileType.PYTHON,
        )

    def test_many_major_changes_requires_review(self):
        """major_changes > 3 → requires careful review (line 709)."""
        analyses = [self._fca(severity=ChangeSeverity.MAJOR) for _ in range(4)]
        result = self.analyzer._requires_careful_review(analyses, [])
        assert result is True

    def test_high_total_complexity_requires_review(self):
        """total_complexity > 10 → requires careful review (line 717)."""
        # Each analysis contributes len(risk_indicators) + len(change_patterns) to complexity.
        # Need 0 risk_indicators (so is_high_risk stays False) and many change_patterns.
        # 6 analyses each with 2 patterns = 12 > 10
        analyses = [
            self._fca(patterns=["A", "B"]),
            self._fca(patterns=["C", "D"]),
            self._fca(patterns=["E", "F"]),
            self._fca(patterns=["G", "H"]),
            self._fca(patterns=["I", "J"]),
            self._fca(patterns=["K", "L"]),
        ]
        result = self.analyzer._requires_careful_review(analyses, [])
        assert result is True

    def test_no_review_needed_for_minor_changes(self):
        """Low risk, few patterns → no careful review needed."""
        analyses = [self._fca()]
        result = self.analyzer._requires_careful_review(analyses, [])
        assert result is False

    def test_high_risk_pattern_requires_review(self):
        """DANGEROUS_LOGIC_CHANGES in overall_patterns → review required."""
        analyses = [self._fca()]
        result = self.analyzer._requires_careful_review(analyses, ["DANGEROUS_LOGIC_CHANGES"])
        assert result is True


class TestAnalyzeDiffIntegration:
    """Integration-style tests exercising analyze_diff end-to-end for uncovered paths."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    async def test_logic_inversion_end_to_end(self):
        """Full diff with logic inversion triggers risk and pattern flags."""
        diff = _make_diff(
            [
                _fc(
                    filename="src/auth.py",
                    additions=1,
                    deletions=1,
                    patch="-    if user.active == True:\n+    if user.active != True:",
                    before_content="    if user.active == True:\n        grant()\n",
                    after_content="    if user.active != True:\n        grant()\n",
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert "LOGIC_INVERSIONS" in fa.change_patterns
        assert "DANGEROUS_LOGIC_CHANGES" in fa.risk_indicators
        # auth in filename → SECURITY_SENSITIVE_FILE_MODIFIED
        assert "SECURITY_SENSITIVE_FILE_MODIFIED" in fa.risk_indicators

    async def test_config_file_major_change_end_to_end(self):
        """Config file with high ratio → CONFIG_FILE_MAJOR_CHANGE risk."""
        before = "key: old\n" * 5
        after = "key: new\nother: val\n" * 30
        diff = _make_diff(
            [
                _fc(
                    filename="settings.yml",
                    file_type=FileType.YAML,
                    additions=60,
                    deletions=5,
                    patch="+key: new\n-key: old",
                    before_content=before,
                    after_content=after,
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert fa.change_severity in (ChangeSeverity.MAJOR, ChangeSeverity.COMPLETE_REWRITE)

    async def test_test_file_changes_end_to_end(self):
        """Test file patch triggers TEST_CHANGES pattern."""
        diff = _make_diff(
            [
                _fc(
                    filename="tests/test_main.py",
                    file_type=FileType.PYTHON,
                    additions=5,
                    deletions=0,
                    patch="+def test_new_feature():\n+    assert 1 == 1",
                    after_content="def test_new_feature():\n    assert 1 == 1\n",
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert "TEST_CHANGES" in fa.change_patterns

    async def test_documentation_changes_end_to_end(self):
        """Doc file triggers DOCUMENTATION_CHANGES pattern."""
        diff = _make_diff(
            [
                _fc(
                    filename="docs/guide.md",
                    file_type=FileType.MARKDOWN,
                    additions=3,
                    deletions=0,
                    patch="+## New Section\n+content here",
                    after_content="## New Section\ncontent here\n",
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert "DOCUMENTATION_CHANGES" in fa.change_patterns

    async def test_very_large_change_end_to_end(self):
        """File with > 500 total changes → VERY_LARGE_CHANGE."""
        diff = _make_diff(
            [
                _fc(
                    filename="big.py",
                    additions=300,
                    deletions=250,
                    patch="+x\n" * 300,
                    after_content="x\n" * 300,
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert "VERY_LARGE_CHANGE" in fa.risk_indicators

    async def test_critical_file_complete_rewrite_end_to_end(self):
        """readme.md complete rewrite → CRITICAL_FILE_COMPLETE_REWRITE."""
        diff = _make_diff(
            [
                _fc(
                    filename="readme.md",
                    file_type=FileType.MARKDOWN,
                    additions=100,
                    deletions=100,
                    patch="+new\n" * 100 + "-old\n" * 100,
                    before_content="old\n" * 100,
                    after_content="new\n" * 100,
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        fa = result.file_analyses[0]
        assert "CRITICAL_FILE_COMPLETE_REWRITE" in fa.risk_indicators

    async def test_coordinated_security_and_review(self):
        """Multiple security-sensitive files → COORDINATED_SECURITY_CHANGES + careful review."""
        diff = _make_diff(
            [
                _fc(
                    filename="src/service_a.py",
                    additions=2,
                    deletions=1,
                    patch="+token = get_token()",
                ),
                _fc(
                    filename="src/service_b.py",
                    additions=2,
                    deletions=1,
                    patch="+password = input()",
                ),
            ]
        )
        result = await self.analyzer.analyze_diff(diff)
        assert "COORDINATED_SECURITY_CHANGES" in result.overall_patterns

    async def test_multiple_config_changes_end_to_end(self):
        """3+ config files → MULTIPLE_CONFIG_CHANGES."""
        files = []
        for name in ["config.yaml", "settings.yml", "app.toml"]:
            files.append(
                _fc(
                    filename=name,
                    file_type=FileType.YAML,
                    additions=2,
                    deletions=1,
                    patch="+key: value",
                )
            )
        diff = _make_diff(files)
        result = await self.analyzer.analyze_diff(diff)
        assert "MULTIPLE_CONFIG_CHANGES" in result.overall_patterns


class TestHasImportChanges:
    """Dedicated tests for _has_import_changes (line 327)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_python_import(self):
        assert self.analyzer._has_import_changes(["import os"], []) is True

    def test_python_from_import(self):
        assert self.analyzer._has_import_changes(["from os import path"], []) is True

    def test_node_require(self):
        assert self.analyzer._has_import_changes(["require('fs')"], []) is True

    def test_es6_import(self):
        assert self.analyzer._has_import_changes(["import foo from 'bar'"], []) is True

    def test_no_import(self):
        assert self.analyzer._has_import_changes(["x = 1"], []) is False


class TestHasErrorHandlingChanges:
    """Dedicated tests for _has_error_handling_changes (line 398)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_python_try(self):
        assert self.analyzer._has_error_handling_changes(["try:"], []) is True

    def test_python_except(self):
        assert self.analyzer._has_error_handling_changes([], ["except:"]) is True

    def test_python_raise(self):
        assert self.analyzer._has_error_handling_changes(["raise ValueError"], []) is True

    def test_js_catch(self):
        assert self.analyzer._has_error_handling_changes(["catch(e)"], []) is True

    def test_no_error_handling(self):
        assert self.analyzer._has_error_handling_changes(["x = 1"], ["y = 2"]) is False


class TestHasSecuritySensitiveChanges:
    """Dedicated tests for _has_security_sensitive_changes (line 431)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_password(self):
        assert self.analyzer._has_security_sensitive_changes(["password = 'x'"], []) is True

    def test_token(self):
        assert self.analyzer._has_security_sensitive_changes([], ["token = 'abc'"]) is True

    def test_no_security(self):
        assert self.analyzer._has_security_sensitive_changes(["x = 1"], ["y = 2"]) is False


class TestHasConfigurationChanges:
    """Dedicated tests for _has_configuration_changes (lines 441-452)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_non_config_file_returns_false(self):
        assert self.analyzer._has_configuration_changes(["key: val"], [], "src/app.py") is False

    def test_config_key_value(self):
        assert (
            self.analyzer._has_configuration_changes(["database: postgres"], [], "config.yaml")
            is True
        )

    def test_config_yaml_list(self):
        assert (
            self.analyzer._has_configuration_changes(["- redis"], [], "docker-compose.yml") is True
        )

    def test_config_section_header(self):
        assert self.analyzer._has_configuration_changes(["[section]"], [], "setup.cfg") is True

    def test_config_no_match(self):
        # A config file but lines don't match patterns
        assert self.analyzer._has_configuration_changes([""], [], "config.yaml") is False


class TestHasTestChanges:
    """Dedicated tests for _has_test_changes (lines 461-479)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_non_test_file_returns_false(self):
        assert self.analyzer._has_test_changes(["def test_x():"], [], "src/app.py") is False

    def test_python_test_function(self):
        assert (
            self.analyzer._has_test_changes(["def test_something():"], [], "tests/test_foo.py")
            is True
        )

    def test_js_describe(self):
        assert (
            self.analyzer._has_test_changes(["describe('suite',"], [], "test/foo.spec.js") is True
        )

    def test_pytest_marker(self):
        assert (
            self.analyzer._has_test_changes(["@pytest.mark.parametrize"], [], "tests/test_bar.py")
            is True
        )

    def test_assert_line(self):
        assert (
            self.analyzer._has_test_changes(["assert result == 42"], [], "tests/test_check.py")
            is True
        )

    def test_no_match_in_test_file(self):
        assert self.analyzer._has_test_changes(["x = 1"], [], "tests/test_empty.py") is False


class TestHasDocumentationChanges:
    """Dedicated tests for _has_documentation_changes (line 489)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_non_doc_file(self):
        assert self.analyzer._has_documentation_changes(["# comment"], [], "src/app.py") is False

    def test_doc_file_with_additions(self):
        assert self.analyzer._has_documentation_changes(["new text"], [], "README.md") is True

    def test_doc_file_with_removals(self):
        assert self.analyzer._has_documentation_changes([], ["old text"], "CHANGELOG.md") is True

    def test_doc_file_empty_changes(self):
        assert self.analyzer._has_documentation_changes([], [], "README.md") is False


class TestHasLogicInversions:
    """Dedicated tests for _has_logic_inversions (line 375)."""

    def setup_method(self):
        self.analyzer = GitDiffAnalyzer()

    def test_and_to_or(self):
        removed = ["if a and b:"]
        added = ["if a or b:"]
        assert self.analyzer._has_logic_inversions(added, removed) is True

    def test_true_to_false(self):
        removed = ["enabled = True  # flag"]
        added = ["enabled = False  # flag"]
        assert self.analyzer._has_logic_inversions(added, removed) is True

    def test_no_inversion(self):
        removed = ["x = 1"]
        added = ["x = 2"]
        assert self.analyzer._has_logic_inversions(added, removed) is False

    def test_ampersand_to_pipe(self):
        removed = ["if (a && b)"]
        added = ["if (a || b)"]
        assert self.analyzer._has_logic_inversions(added, removed) is True
