"""Tests for change_categorizer.py."""

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


class TestChangeCategorizer:
    """Tests for ChangeCategorizer."""

    def setup_method(self):
        self.categorizer = ChangeCategorizer()

    def test_categorize_feature_addition(self):
        """File with new function definitions should be categorized as FEATURE_ADDITION."""
        fc = _fc(
            "src/new_feature.py",
            additions=50,
            deletions=0,
            patch="+def new_handler(request):\n+    return response\n+class NewService:\n+    pass",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.FEATURE_ADDITION

    def test_categorize_bugfix(self):
        """File with error handling patterns should suggest BUG_FIX."""
        fc = _fc(
            "src/handler.py",
            additions=10,
            deletions=5,
            patch="+    except ValueError as e:\n+        if error:\n+            return None\n-    return broken",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.BUG_FIX

    def test_categorize_refactor(self):
        """A Python file with refactoring patterns should be categorized appropriately."""
        fc = _fc(
            "src/utils.py",
            additions=20,
            deletions=20,
            patch="+    # extract method for clarity\n+    rename old_var\n+    split large_function",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.REFACTORING

    def test_categorize_documentation(self):
        """Markdown files should be categorized as DOCUMENTATION."""
        fc = _fc(
            "docs/guide.md",
            file_type=FileType.MARKDOWN,
            additions=30,
            deletions=5,
            patch="+# New Section\n+Some docs",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.DOCUMENTATION

    def test_categorize_test(self):
        """Test files detected by file classifier should indicate TEST_UPDATE in indicators."""
        fc = _fc(
            "tests/test_handler.py",
            additions=30,
            deletions=5,
            patch="+def test_new_feature():\n+    assert True",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert "tests_modified" in result.indicators

    def test_categorize_security(self):
        """Security-related patterns should be categorized as SECURITY_FIX."""
        fc = _fc(
            "src/auth.py",
            additions=15,
            deletions=10,
            patch="+    validate(user_input)\n+    sanitize(data)\n+    auth_check(token)\n+    permission_verify(role)",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.SECURITY_FIX

    def test_categorize_dependency(self):
        """YAML/JSON config files should be categorized as CONFIGURATION."""
        fc = _fc(
            "config.yaml",
            file_type=FileType.YAML,
            additions=5,
            deletions=2,
            patch="+key: value",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.CONFIGURATION

    def test_multi_file_aggregate(self):
        """categorize_pr_changes should aggregate across multiple files."""
        files = [
            _fc(
                "src/new.py",
                additions=50,
                deletions=0,
                patch="+def new_func():\n+    pass\n+class Foo:\n+    pass",
            ),
            _fc(
                "docs/readme.md",
                file_type=FileType.MARKDOWN,
                additions=20,
                deletions=0,
                patch="+# Docs",
            ),
            _fc(
                "src/another.py",
                additions=30,
                deletions=0,
                patch="+def another_func():\n+    return 1",
            ),
        ]
        profile = self.categorizer.categorize_pr_changes(files)
        assert profile.primary_category is not None
        assert isinstance(profile.change_types, dict)
        assert profile.complexity_score >= 0

    def test_ambiguous_fallback(self):
        """Files with no recognizable patterns should fall back to UNKNOWN."""
        fc = _fc(
            "data/blob.bin",
            file_type=FileType.UNKNOWN,
            additions=0,
            deletions=0,
            patch="",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.category == ChangeCategory.UNKNOWN

    def test_empty_diff(self):
        """Empty patch should not crash and should produce a classification."""
        fc = _fc(
            "empty.py",
            additions=0,
            deletions=0,
            patch="",
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result is not None
        assert isinstance(result.confidence, float)

    def test_large_diff_edge_case(self):
        """Large files should still be classified and have valid scope/impact."""
        fc = _fc(
            "src/big_module.py",
            additions=300,
            deletions=200,
            patch="\n".join([f"+def func_{i}():\n+    pass" for i in range(50)]),
        )
        result = self.categorizer.categorize_file_change(fc)
        assert result.scope in (
            ChangeScope.FILE_LEVEL,
            ChangeScope.MODULE_LEVEL,
            ChangeScope.PACKAGE_LEVEL,
            ChangeScope.PROJECT_LEVEL,
        )
        assert result.impact_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
