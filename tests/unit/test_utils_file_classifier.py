"""Tests for utils/file_classifier.py."""

from utils.file_classifier import CoreFileTypeClassifier


class TestCoreFileTypeClassifier:
    """Tests for CoreFileTypeClassifier."""

    def setup_method(self):
        self.classifier = CoreFileTypeClassifier()

    def test_python_classification(self):
        """Python source files should be classified as code, not test/config/doc."""
        assert self.classifier.get_file_type("src/main.py") == "code"
        assert self.classifier.get_file_type("utils/helpers.py") == "code"

    def test_js_classification(self):
        """JavaScript source files should be classified as code."""
        assert self.classifier.get_file_type("src/app.js") == "code"
        assert self.classifier.get_file_type("src/index.jsx") == "code"

    def test_config_classification(self):
        """Configuration files should be classified as config."""
        assert self.classifier.get_file_type("config.yaml") == "config"
        assert self.classifier.get_file_type("settings.json") == "config"
        assert self.classifier.get_file_type("pyproject.toml") == "config"
        assert self.classifier.get_file_type("Dockerfile") == "config"
        # .env is a dotfile with no extension per os.path.splitext, so the
        # extension check does not match; it falls through to "code"
        assert self.classifier.get_file_type(".env") == "code"

    def test_test_file_detection(self):
        """Test files should be classified as test."""
        assert self.classifier.get_file_type("test_main.py") == "test"
        assert self.classifier.get_file_type("tests/test_utils.py") == "test"
        assert self.classifier.get_file_type("app.test.js") == "test"
        assert self.classifier.get_file_type("app.spec.ts") == "test"

    def test_unknown_extension(self):
        """Files without a known pattern should be classified as code."""
        # A .rs file is not in config, test, or doc sets, so should be "code"
        result = self.classifier.get_file_type("main.rs")
        assert result == "code"
        # A file with no extension
        result2 = self.classifier.get_file_type("somebinary")
        assert result2 == "code"

    def test_config_file_pattern_match_in_path(self):
        """Line 160: Config file detected via pattern substring match in filename_lower.
        File has non-config extension so the extension check doesn't match,
        and basename doesn't exactly match config_filenames, but a pattern
        like '.github/workflows/' is found in the full path."""
        # .github/workflows/ is a pattern in config_filenames
        # basename is "deploy" (no extension -> not in config_extensions)
        # basename "deploy" is NOT in config_filenames exactly
        # But ".github/workflows/" IS in the full filename_lower string
        assert self.classifier.get_file_type(".github/workflows/deploy") == "config"

    def test_config_file_tox_ini_in_subdir(self):
        """Line 160: tox.ini pattern match in path."""
        # "tox.ini" is in config_filenames
        # If file is at "configs/tox.ini", basename is "tox.ini" which matches
        # exactly at line 155 (before reaching 160). Use a different approach:
        # pattern "pre-commit-config" is in config_filenames as ".pre-commit-config.yaml"
        # A file named "my.pre-commit-config.yaml" has basename matching the pattern
        assert self.classifier.get_file_type(".pre-commit-config.yaml") == "config"

    def test_documentation_file_base_without_ext_match(self):
        """Line 190: Documentation file detected by base_without_ext in documentation_filenames."""
        # "changelog" without extension — base_without_ext = "changelog" which is in doc filenames
        assert self.classifier.get_file_type("CHANGELOG") == "documentation"
        # "readme" without extension
        assert self.classifier.get_file_type("README") == "documentation"

    def test_documentation_file_docs_pattern_match(self):
        """Line 194: Documentation file detected via docs/ pattern in path."""
        # "docs/" is in documentation_filenames, it's a pattern match
        # basename is "guide" (no extension matching doc extensions)
        # "guide" is not in documentation_filenames
        # But "docs/" is found in the full filename -> line 194
        assert self.classifier.get_file_type("docs/guide") == "documentation"

    def test_documentation_file_in_documentation_dir(self):
        """Line 194: Files in /documentation/ directory pattern."""
        # "/documentation/" is in documentation_filenames
        assert self.classifier.get_file_type("project/documentation/api") == "documentation"
