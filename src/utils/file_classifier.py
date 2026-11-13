"""
File type classifier for PR validation.

This module provides utilities to classify files by their type and purpose.
Extracted from monolith core/file_classifier.py.
"""

from __future__ import annotations

import os
import re


class CoreFileTypeClassifier:
    """
    Classifies files by their type and purpose.

    This classifier is used to identify configuration files, test files,
    documentation files, and other special file types for PR validation.
    """

    def __init__(self) -> None:
        self.config_extensions = {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".config",
            ".env",
            ".properties",
            ".cfg",
            ".xml",
        }

        self.config_filenames = {
            "dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "makefile",
            "cmake",
            "cmakelists.txt",
            "build.gradle",
            "pom.xml",
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "requirements.txt",
            "requirements-dev.txt",
            "poetry.lock",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "pipfile",
            "pipfile.lock",
            "conda.yaml",
            "environment.yml",
            "jenkinsfile",
            ".gitlab-ci.yml",
            ".github/workflows/",
            "tox.ini",
            "mypy.ini",
            "pytest.ini",
            "flake8.ini",
            ".flake8",
            ".mypy.ini",
            ".pre-commit-config.yaml",
        }

        self.test_patterns = {
            "test_",
            "_test.",
            "test/",
            "/test/",
            "/tests/",
            "spec_",
            "_spec.",
            "spec/",
            "/spec/",
            "/specs/",
            "integration_test",
            "unit_test",
            "e2e_test",
        }

        self.test_extensions = {
            "test.py",
            "test.js",
            "test.ts",
            "test.jsx",
            "test.tsx",
            "spec.py",
            "spec.js",
            "spec.ts",
            "spec.jsx",
            "spec.tsx",
        }

        self.documentation_extensions = {
            ".md",
            ".rst",
            ".txt",
            ".asciidoc",
            ".adoc",
            ".org",
        }

        self.documentation_filenames = {
            "readme",
            "readme.md",
            "readme.rst",
            "readme.txt",
            "changelog",
            "changelog.md",
            "changelog.rst",
            "changelog.txt",
            "contributing",
            "contributing.md",
            "contributing.rst",
            "license",
            "license.md",
            "license.rst",
            "license.txt",
            "authors",
            "authors.md",
            "authors.rst",
            "authors.txt",
            "install",
            "install.md",
            "install.rst",
            "install.txt",
            "usage",
            "usage.md",
            "usage.rst",
            "usage.txt",
            "api",
            "api.md",
            "api.rst",
            "api.txt",
            "docs/",
            "/docs/",
            "documentation/",
            "/documentation/",
        }

    def _is_config_file(self, filename: str) -> bool:
        """Check if a file is a configuration file."""
        filename_lower = filename.lower()
        basename = os.path.basename(filename_lower)

        _, ext = os.path.splitext(basename)
        if ext in self.config_extensions:
            return True

        if basename in self.config_filenames:
            return True

        for pattern in self.config_filenames:
            if self._matches_pattern_on_word_boundary(pattern, filename_lower):
                return True

        return False

    def _is_test_file(self, filename: str) -> bool:
        """Check if a file is a test file."""
        filename_lower = filename.lower()
        basename = os.path.basename(filename_lower)

        for ext_pattern in self.test_extensions:
            if basename.endswith(ext_pattern):
                return True

        for pattern in self.test_patterns:
            if self._matches_pattern_on_word_boundary(pattern, filename_lower):
                return True

        return False

    @staticmethod
    def _matches_pattern_on_word_boundary(pattern: str, filename: str) -> bool:
        """Match a test pattern on word boundaries, not as a bare substring.

        A pattern end that carries a separator character ("_", "/", or ".")
        is already bounded on that side; an end that does not must be anchored
        so it cannot be glued onto the adjacent word. This prevents e.g.
        "test_" matching inside "latest_" or "contest_".
        """
        sep = "_/."
        left_bounded = pattern[0] in sep
        right_bounded = pattern[-1] in sep
        regex = re.escape(pattern)
        if not left_bounded:
            regex = r"(?<![a-z0-9])" + regex
        if not right_bounded:
            regex = regex + r"(?![a-z0-9])"
        return re.search(regex, filename) is not None

    def _is_documentation_file(self, filename: str) -> bool:
        """Check if a file is a documentation file."""
        filename_lower = filename.lower()
        basename = os.path.basename(filename_lower)

        _, ext = os.path.splitext(basename)
        if ext in self.documentation_extensions:
            return True

        base_without_ext = os.path.splitext(basename)[0]
        if base_without_ext in self.documentation_filenames:
            return True

        for pattern in self.documentation_filenames:
            if self._matches_pattern_on_word_boundary(pattern, filename_lower):
                return True

        return False

    def get_file_type(self, filename: str) -> str:
        """Get the type of a file."""
        if self._is_test_file(filename):
            return "test"
        elif self._is_config_file(filename):
            return "config"
        elif self._is_documentation_file(filename):
            return "documentation"
        else:
            return "code"
