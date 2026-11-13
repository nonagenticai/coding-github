"""Extended tests for file_classifier.py — covering uncovered lines."""

from coding_client.models.pr_validation import FileType

from services.pr_validation.core.file_classifier import PRFileTypeClassifier


class TestClassifyByContentUnknown:
    """Lines 227, 247: _classify_by_content returns UNKNOWN for empty/unmatched."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_empty_content_returns_unknown(self):
        """Empty or whitespace-only content should return UNKNOWN."""
        assert self.classifier._classify_by_content("") == FileType.UNKNOWN
        assert self.classifier._classify_by_content("   \n  \n  ") == FileType.UNKNOWN

    def test_none_content_returns_unknown(self):
        """None content should return UNKNOWN."""
        assert self.classifier._classify_by_content(None) == FileType.UNKNOWN

    def test_no_pattern_match_returns_unknown(self):
        """Content that matches no file type pattern should return UNKNOWN."""
        content = "this is just plain text with no code patterns whatsoever\nline two\nline three"
        assert self.classifier._classify_by_content(content) == FileType.UNKNOWN


class TestGetLanguageInfoBranches:
    """Lines 346-349: _get_language_specific_info branches for Python/JS/Markdown."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_python_language_info(self):
        """Python file_type should trigger _analyze_python_file."""
        info = self.classifier._get_language_specific_info(
            "main.py",
            "import os\ndef main():\n    pass\n",
            FileType.PYTHON,
        )
        assert "has_functions" in info
        assert "has_imports" in info

    def test_javascript_language_info(self):
        """JavaScript file_type should trigger _analyze_js_ts_file."""
        info = self.classifier._get_language_specific_info(
            "app.js",
            "function hello() { return 1; }\nmodule.exports = hello;\n",
            FileType.JAVASCRIPT,
        )
        assert "has_functions" in info
        assert "has_exports" in info

    def test_typescript_language_info(self):
        """TypeScript file_type should trigger _analyze_js_ts_file."""
        info = self.classifier._get_language_specific_info(
            "app.ts",
            "import { Foo } from 'bar';\nclass MyClass {}\nexport { MyClass };\n",
            FileType.TYPESCRIPT,
        )
        assert "has_classes" in info

    def test_markdown_language_info(self):
        """Markdown file_type should trigger _analyze_markdown_file."""
        info = self.classifier._get_language_specific_info(
            "README.md",
            "# Title\n\nSome text\n\n```python\ncode\n```\n",
            FileType.MARKDOWN,
        )
        assert "has_headers" in info
        assert "has_code_blocks" in info

    def test_unknown_type_returns_empty_info(self):
        """Unknown file_type should return empty dict."""
        info = self.classifier._get_language_specific_info(
            "data.bin",
            None,
            FileType.UNKNOWN,
        )
        assert info == {}


class TestAnalyzePythonComplexity:
    """Lines 391, 393: Python complexity estimation medium and high."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_high_complexity_python(self):
        """Python file with many constructs should be estimated as high complexity."""
        # >20 total complexity indicators
        content = "\n".join(
            [f"class Cls{i}:" for i in range(5)]
            + [f"def func{i}():" for i in range(5)]
            + [f"if cond{i}:" for i in range(6)]
            + [f"for x{i} in range(10):" for i in range(3)]
            + ["while True:", "while running:"]
            + ["try:", "try:"]
        )
        info = self.classifier._analyze_python_file(content)
        assert info["estimated_complexity"] == "high"
        assert info["has_classes"] is True
        assert info["has_functions"] is True

    def test_medium_complexity_python(self):
        """Python file with moderate constructs should be estimated as medium."""
        # Between 10 and 20 complexity indicators
        content = "\n".join(
            [f"class Cls{i}:" for i in range(2)]
            + [f"def func{i}():" for i in range(3)]
            + [f"if cond{i}:" for i in range(4)]
            + [f"for x{i} in range(10):" for i in range(2)]
        )
        info = self.classifier._analyze_python_file(content)
        assert info["estimated_complexity"] == "medium"

    def test_low_complexity_python(self):
        """Python file with few constructs should be estimated as low."""
        content = "x = 1\ny = 2\nprint(x + y)\n"
        info = self.classifier._analyze_python_file(content)
        assert info["estimated_complexity"] == "low"

    def test_none_content_python(self):
        """None content should return default info."""
        info = self.classifier._analyze_python_file(None)
        assert info["has_classes"] is False
        assert info["has_functions"] is False
        assert info["estimated_complexity"] == "low"

    def test_python_script_detection(self):
        """File with if __name__ == '__main__' should be detected as script."""
        content = "import sys\ndef main():\n    pass\nif __name__ == '__main__':\n    main()\n"
        info = self.classifier._analyze_python_file(content)
        assert info["is_script"] is True


class TestAnalyzeJsTsFile:
    """Line 416: _analyze_js_ts_file class detection and framework indicators."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_js_class_detection(self):
        """JS file with class keyword should set has_classes."""
        content = "class MyComponent extends React.Component {\n  render() {}\n}\n"
        info = self.classifier._analyze_js_ts_file(content)
        assert info["has_classes"] is True

    def test_js_none_content(self):
        """None content should return default info."""
        info = self.classifier._analyze_js_ts_file(None)
        assert info["has_functions"] is False
        assert info["has_classes"] is False
        assert info["framework_indicators"] == []

    def test_js_module_detection(self):
        """File with imports and exports should be detected as module."""
        content = "import { foo } from 'bar';\nexport { baz };\n"
        info = self.classifier._analyze_js_ts_file(content)
        assert info["is_module"] is True
        assert info["has_imports"] is True
        assert info["has_exports"] is True

    def test_react_framework_detection(self):
        """File with React patterns should include React indicator."""
        content = "import React from 'react';\nfunction App() { return <div/>; }\n"
        info = self.classifier._analyze_js_ts_file(content)
        assert "React" in info["framework_indicators"]

    def test_express_framework_detection(self):
        """File with Express patterns should include Express indicator."""
        content = "const express = require('express');\napp.get('/', handler);\n"
        info = self.classifier._analyze_js_ts_file(content)
        assert "Express" in info["framework_indicators"]

    def test_nodejs_framework_detection(self):
        """File with Node.js patterns should include Node.js indicator."""
        content = "const fs = require('fs');\nprocess.exit(0);\n"
        info = self.classifier._analyze_js_ts_file(content)
        assert "Node.js" in info["framework_indicators"]


class TestAnalyzeMarkdownDocumentTypes:
    """Lines 501-517: markdown document_type detection branches."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_readme_document_type(self):
        """Content with 'README' should be typed as readme."""
        content = "# README\n\nThis is the README for the project.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "readme"

    def test_changelog_document_type(self):
        """Content with 'changelog' should be typed as changelog."""
        content = "# Changelog\n\n## v2.0.0\n- Breaking changes\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "changelog"

    def test_contributing_document_type(self):
        """Content with 'contributing' should be typed as contributing."""
        content = "# Contributing Guidelines\n\nHow to contribute to this project.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "contributing"

    def test_license_document_type(self):
        """Content with 'license' should be typed as license."""
        content = "# License\n\nCopyright 2024. All rights reserved.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "license"

    def test_installation_document_type(self):
        """Content with 'installation' should be typed as installation."""
        content = "# Installation Guide\n\nRun `pip install mypackage` to install.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "installation"

    def test_api_documentation_type(self):
        """Content with 'api' or 'reference' should be typed as api_documentation."""
        content = "# API Reference\n\n## Endpoints\n\nGET /users\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "api_documentation"

    def test_general_document_type(self):
        """Content without specific keywords should remain general."""
        content = "# Something\n\nJust some notes.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["document_type"] == "general"

    def test_none_content_markdown(self):
        """None content should return default info."""
        info = self.classifier._analyze_markdown_file(None)
        assert info["has_headers"] is False
        assert info["document_type"] == "general"

    def test_long_markdown_length_estimation(self):
        """Markdown with >200 lines should be estimated as long."""
        content = "\n".join([f"Line {i}" for i in range(250)])
        info = self.classifier._analyze_markdown_file(content)
        assert info["estimated_length"] == "long"

    def test_medium_markdown_length_estimation(self):
        """Markdown with 50-200 lines should be estimated as medium."""
        content = "\n".join([f"Line {i}" for i in range(100)])
        info = self.classifier._analyze_markdown_file(content)
        assert info["estimated_length"] == "medium"

    def test_short_markdown_length_estimation(self):
        """Markdown with <50 lines should be estimated as short."""
        content = "# Title\n\nShort doc.\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["estimated_length"] == "short"

    def test_markdown_tables_detection(self):
        """Content with pipe-delimited tables should set has_tables."""
        content = "# Data\n\n| Col1 | Col2 |\n|------|------|\n| a    | b    |\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["has_tables"] is True

    def test_markdown_links_detection(self):
        """Content with markdown links should set has_links."""
        content = "# Links\n\n[Click here](https://example.com)\n"
        info = self.classifier._analyze_markdown_file(content)
        assert info["has_links"] is True


class TestGetFileInfoIntegration:
    """Integration test for get_file_info covering language-specific dispatch."""

    def setup_method(self):
        self.classifier = PRFileTypeClassifier()

    def test_get_file_info_python_with_content(self):
        """get_file_info for a Python file should include language_specific_info."""
        info = self.classifier.get_file_info(
            "main.py",
            content="import os\ndef main():\n    pass\n",
        )
        assert info["file_type"] == FileType.PYTHON
        assert info["is_test_file"] is False
        assert "has_functions" in info["language_specific_info"]

    def test_get_file_info_test_file(self):
        """Test file should be flagged is_test_file."""
        info = self.classifier.get_file_info("tests/test_main.py")
        assert info["is_test_file"] is True

    def test_get_file_info_config_file(self):
        """Config file should be flagged is_config_file."""
        info = self.classifier.get_file_info("config.yaml")
        assert info["is_config_file"] is True

    def test_get_file_info_documentation(self):
        """Documentation file should be flagged is_documentation."""
        info = self.classifier.get_file_info("docs/guide.md")
        assert info["is_documentation"] is True
