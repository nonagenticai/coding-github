"""Extended tests for change_categorizer.py — covering uncovered lines."""

from coding_client.models.pr_validation import (
    ChangeType,
    FileChange,
    FileType,
)

from services.pr_validation.core.change_categorizer import (
    ChangeCategorizer,
    ChangeCategory,
    ChangeScope,
)


def _fc(filename="src/main.py", **kw):
    kw.setdefault("file_type", FileType.PYTHON)
    kw.setdefault("change_type", ChangeType.MODIFIED)
    kw.setdefault("additions", 10)
    kw.setdefault("deletions", 5)
    kw.setdefault("patch", "@@ -1,5 +1,10 @@")
    return FileChange(filename=filename, **kw)


class TestAnalyzeContentChangesUnknownFallback:
    """Line 406: _analyze_content_changes returns UNKNOWN when no patterns match."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_patch_with_no_matching_patterns_returns_unknown(self):
        """Content that matches no category file_patterns should fall back to UNKNOWN."""
        fc = _fc(
            "src/data.py",
            file_type=FileType.UNKNOWN,
            additions=2,
            deletions=1,
            # Lines that don't match any category file_patterns
            patch="+x = 42\n-y = 99\n+z = 0",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.UNKNOWN
        assert result.confidence <= 0.2


class TestMetadataCategoryAddedToCounts:
    """Line 332: metadata_category != UNKNOWN adds to category_counts."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_metadata_with_bug_fix_title_boosts_category(self):
        """PR metadata with 'resolve bug' in title should add BUG_FIX to counts."""
        files = [
            _fc(
                "src/foo.py",
                file_type=FileType.UNKNOWN,
                additions=1,
                deletions=0,
                patch="+x = 1",
            ),
        ]
        metadata = {"title": "resolve critical bug in parser"}
        profile = self.categorizer.categorize_pr_changes(files, pr_metadata=metadata)
        assert ChangeCategory.BUG_FIX in profile.change_types


class TestEmptyFileClassificationsProfile:
    """Lines 343-344: empty category_counts branch (no file_classifications)."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_empty_file_list_produces_unknown_profile(self):
        """categorize_pr_changes with empty file list should give UNKNOWN primary."""
        profile = self.categorizer.categorize_pr_changes([])
        assert profile.primary_category == ChangeCategory.UNKNOWN
        assert profile.secondary_categories == []


class TestDetectSpecificIndicators:
    """Lines 428, 449, 453-454: async function, imports, config indicators."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_async_function_indicator(self):
        """An added 'async de' line should add async_function_added indicator."""
        fc = _fc(
            "src/service.py",
            file_type=FileType.UNKNOWN,
            additions=5,
            deletions=0,
            patch="+    async def fetch_data(self):\n+        pass",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert "async_function_added" in result.indicators

    def test_imports_modified_indicator(self):
        """Lines containing 'import' should trigger imports_modified."""
        fc = _fc(
            "src/module.py",
            file_type=FileType.UNKNOWN,
            additions=2,
            deletions=1,
            patch="+import os\n-import sys\n+from pathlib import Path",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert "imports_modified" in result.indicators

    def test_configuration_modified_indicator_yaml(self):
        """YAML files with changes should get configuration_modified indicator."""
        fc = _fc(
            "settings.yaml",
            file_type=FileType.YAML,
            additions=2,
            deletions=1,
            patch="+key: new_value\n-key: old_value",
        )
        # YAML gets categorized as CONFIGURATION by file_type_categories
        # so _analyze_content_changes won't be called, but indicators come
        # from _detect_specific_change_indicators via the main flow.
        # We need to call the internal method to test this path.
        added = ["key: new_value"]
        removed = ["key: old_value"]
        indicators = self.categorizer._detect_specific_change_indicators(added, removed, fc)
        assert "configuration_modified" in indicators

    def test_configuration_modified_indicator_json(self):
        """JSON files with changes should get configuration_modified indicator."""
        fc = _fc(
            "package.json",
            file_type=FileType.JSON,
            additions=1,
            deletions=0,
            patch='+  "version": "2.0.0"',
        )
        added = ['"version": "2.0.0"']
        removed = []
        indicators = self.categorizer._detect_specific_change_indicators(added, removed, fc)
        assert "configuration_modified" in indicators


class TestDetermineChangeScopeEdgeCases:
    """Lines 479, 487: project-level and module-level scope detection."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_project_level_scope_for_dockerfile(self):
        """Files like Dockerfile should be PROJECT_LEVEL scope."""
        fc = _fc(
            "Dockerfile",
            file_type=FileType.DOCKERFILE,
            additions=5,
            deletions=0,
        )
        scope = self.categorizer._determine_change_scope(fc)
        assert scope == ChangeScope.PROJECT_LEVEL

    def test_project_level_scope_for_pyproject_toml(self):
        """pyproject.toml should be PROJECT_LEVEL scope."""
        fc = _fc(
            "pyproject.toml",
            file_type=FileType.UNKNOWN,
            additions=1,
            deletions=1,
        )
        scope = self.categorizer._determine_change_scope(fc)
        assert scope == ChangeScope.PROJECT_LEVEL

    def test_module_level_scope_for_api_file(self):
        """Files with 'api' in name should be MODULE_LEVEL."""
        fc = _fc(
            "src/api/endpoints.py",
            additions=5,
            deletions=5,
        )
        scope = self.categorizer._determine_change_scope(fc)
        assert scope == ChangeScope.MODULE_LEVEL

    def test_module_level_scope_for_interface_file(self):
        """Files with 'interface' in name should be MODULE_LEVEL."""
        fc = _fc(
            "src/interface/handler.py",
            additions=5,
            deletions=5,
        )
        scope = self.categorizer._determine_change_scope(fc)
        assert scope == ChangeScope.MODULE_LEVEL


class TestAssessImpactLevel:
    """Lines 531, 538: size_impact and CRITICAL threshold."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_large_change_over_500_adds_size_impact(self):
        """Changes > 500 should add size_impact = 2."""
        fc = _fc(
            "src/big.py",
            additions=300,
            deletions=250,
        )
        # SECURITY_FIX category (modifier=3) + PROJECT_LEVEL scope (base=4) + size(2) = 9 => CRITICAL
        impact = self.categorizer._assess_impact_level(
            fc, ChangeCategory.SECURITY_FIX, ChangeScope.PROJECT_LEVEL
        )
        assert impact == "CRITICAL"

    def test_critical_impact_level_threshold(self):
        """Score >= 7 should yield CRITICAL."""
        fc = _fc(
            "setup.py",
            additions=300,
            deletions=250,
        )
        # ECOSYSTEM scope(5) + SECURITY_FIX(3) + size>500(2) = 10 => CRITICAL
        impact = self.categorizer._assess_impact_level(
            fc, ChangeCategory.SECURITY_FIX, ChangeScope.ECOSYSTEM_LEVEL
        )
        assert impact == "CRITICAL"


class TestCalculateBreakingPotential:
    """Lines 571: API/interface/public/export keyword in filename."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_api_keyword_increases_breaking_potential(self):
        """Filename containing 'api' should increase breaking_potential by 0.2."""
        fc = _fc(
            "src/api/routes.py",
            additions=5,
            deletions=5,
        )
        potential = self.categorizer._calculate_breaking_potential(
            fc, ChangeCategory.FEATURE_ADDITION
        )
        # Base for FEATURE_ADDITION = 0.2, + 0.2 for 'api' keyword = 0.4
        assert potential >= 0.4

    def test_public_keyword_increases_breaking_potential(self):
        """Filename containing 'public' should increase breaking_potential."""
        fc = _fc(
            "src/public/widget.py",
            additions=5,
            deletions=5,
        )
        potential = self.categorizer._calculate_breaking_potential(fc, ChangeCategory.DOCUMENTATION)
        # Base for DOCUMENTATION = 0.0, + 0.2 for 'public' = 0.2
        assert potential >= 0.2


class TestAnalyzePRMetadata:
    """Lines 585, 595, 597, 600: PR metadata analysis branches."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_metadata_with_description(self):
        """Metadata with description should be analyzed."""
        metadata = {"title": "update readme", "description": "refactor the main module"}
        category = self.categorizer._analyze_pr_metadata(metadata)
        assert category == ChangeCategory.REFACTORING

    def test_metadata_matching_single_category(self):
        """Metadata matching a single category pattern should return it."""
        metadata = {"title": "security vulnerability fix"}
        category = self.categorizer._analyze_pr_metadata(metadata)
        assert category == ChangeCategory.SECURITY_FIX

    def test_metadata_matching_multiple_returns_highest(self):
        """Metadata matching multiple categories returns the one with highest score."""
        metadata = {"title": "security auth permission sanitize escape vulnerability exploit cve"}
        category = self.categorizer._analyze_pr_metadata(metadata)
        assert category == ChangeCategory.SECURITY_FIX

    def test_metadata_no_match_returns_unknown(self):
        """Metadata with no matching patterns should return UNKNOWN."""
        metadata = {"title": "zzz xyz nothing relevant here"}
        category = self.categorizer._analyze_pr_metadata(metadata)
        assert category == ChangeCategory.UNKNOWN


class TestDetermineOverallScope:
    """Lines 607, 623: empty scopes and >2 project-level triggering ecosystem."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_empty_scopes_returns_file_level(self):
        """Empty scope list should return FILE_LEVEL."""
        result = self.categorizer._determine_overall_scope([])
        assert result == ChangeScope.FILE_LEVEL

    def test_multiple_project_level_scopes_returns_ecosystem(self):
        """More than 2 PROJECT_LEVEL scopes should escalate to ECOSYSTEM_LEVEL."""
        scopes = [
            ChangeScope.PROJECT_LEVEL,
            ChangeScope.PROJECT_LEVEL,
            ChangeScope.PROJECT_LEVEL,
        ]
        result = self.categorizer._determine_overall_scope(scopes)
        assert result == ChangeScope.ECOSYSTEM_LEVEL


class TestIdentifyPRRiskFactors:
    """Lines 640, 647, 651, 655: risk factor branches."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def _make_classification(
        self, breaking=0.0, impact="LOW", confidence=0.9, category=ChangeCategory.FEATURE_ADDITION
    ):
        from services.pr_validation.core.change_categorizer import ChangeClassification

        return ChangeClassification(
            category=category,
            scope=ChangeScope.FILE_LEVEL,
            confidence=confidence,
            indicators=[],
            impact_level=impact,
            breaking_potential=breaking,
        )

    def test_high_breaking_potential_risk(self):
        """Average breaking_potential > 0.5 should add HIGH_BREAKING_POTENTIAL."""
        classifications = [
            self._make_classification(breaking=0.8),
            self._make_classification(breaking=0.9),
        ]
        risks = self.categorizer._identify_pr_risk_factors(
            classifications, {ChangeCategory.FEATURE_ADDITION: 2}
        )
        assert "HIGH_BREAKING_POTENTIAL" in risks

    def test_multiple_high_impact_changes_risk(self):
        """More than 3 HIGH/CRITICAL impact changes should add MULTIPLE_HIGH_IMPACT_CHANGES."""
        classifications = [
            self._make_classification(impact="HIGH"),
            self._make_classification(impact="CRITICAL"),
            self._make_classification(impact="HIGH"),
            self._make_classification(impact="HIGH"),
        ]
        risks = self.categorizer._identify_pr_risk_factors(
            classifications, {ChangeCategory.FEATURE_ADDITION: 4}
        )
        assert "MULTIPLE_HIGH_IMPACT_CHANGES" in risks

    def test_mixed_change_types_risk(self):
        """More than 4 different categories should add MIXED_CHANGE_TYPES."""
        counts = {
            ChangeCategory.FEATURE_ADDITION: 1,
            ChangeCategory.BUG_FIX: 1,
            ChangeCategory.REFACTORING: 1,
            ChangeCategory.DOCUMENTATION: 1,
            ChangeCategory.CONFIGURATION: 1,
        }
        classifications = [self._make_classification()] * 5
        risks = self.categorizer._identify_pr_risk_factors(classifications, counts)
        assert "MIXED_CHANGE_TYPES" in risks

    def test_security_mixed_with_other_changes_risk(self):
        """Security changes mixed with other categories should add the risk."""
        counts = {
            ChangeCategory.SECURITY_FIX: 1,
            ChangeCategory.FEATURE_ADDITION: 1,
        }
        classifications = [
            self._make_classification(category=ChangeCategory.SECURITY_FIX),
            self._make_classification(category=ChangeCategory.FEATURE_ADDITION),
        ]
        risks = self.categorizer._identify_pr_risk_factors(classifications, counts)
        assert "SECURITY_MIXED_WITH_OTHER_CHANGES" in risks


class TestCalculatePRComplexity:
    """Line 671: empty classifications returns 0.0."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_empty_classifications_returns_zero(self):
        """Empty classification list should return 0.0 complexity."""
        result = self.categorizer._calculate_pr_complexity([], {})
        assert result == 0.0
