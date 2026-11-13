"""
File type classifier for PR validation.

This module provides intelligent file type detection that goes beyond
basic extension matching to include content-based classification.
"""

import re
from pathlib import Path
from typing import Any

from coding_client.models.pr_validation import FileType


class PRFileTypeClassifier:
    """
    Intelligent file type classifier that uses both filename patterns
    and content analysis to determine file types.
    """

    def __init__(self) -> None:
        self._extension_mappings = {
            ".py": FileType.PYTHON,
            ".pyx": FileType.PYTHON,
            ".pyi": FileType.PYTHON,
            ".js": FileType.JAVASCRIPT,
            ".jsx": FileType.JAVASCRIPT,
            ".mjs": FileType.JAVASCRIPT,
            ".ts": FileType.TYPESCRIPT,
            ".tsx": FileType.TYPESCRIPT,
            ".md": FileType.MARKDOWN,
            ".markdown": FileType.MARKDOWN,
            ".yml": FileType.YAML,
            ".yaml": FileType.YAML,
            ".json": FileType.JSON,
            ".html": FileType.HTML,
            ".htm": FileType.HTML,
            ".css": FileType.CSS,
            ".scss": FileType.CSS,
            ".sass": FileType.CSS,
            ".less": FileType.CSS,
        }

        # Special filename patterns
        self._filename_patterns = {
            FileType.DOCKERFILE: [
                r"Dockerfile$",
                r"Dockerfile\.[a-zA-Z0-9_-]+$",
                r"\.dockerfile$",
            ],
            FileType.PYTHON: [
                r"setup\.py$",
                r"__init__\.py$",
                r"conftest\.py$",
            ],
            FileType.YAML: [
                r"\.github/workflows/.*\.yml$",
                r"\.github/workflows/.*\.yaml$",
                r"docker-compose.*\.yml$",
                r"docker-compose.*\.yaml$",
            ],
            FileType.MARKDOWN: [
                r"README$",
                r"CHANGELOG$",
                r"CONTRIBUTING$",
                r"LICENSE$",
            ],
        }

        # Content-based detection patterns
        self._content_patterns = {
            FileType.PYTHON: [
                r"#!/usr/bin/env python",
                r"#!/usr/bin/python",
                r"# -\*- coding: utf-8 -\*-",
                r"from __future__ import",
                r"import \w+",
                r"def \w+\(",
                r"class \w+\(",
            ],
            FileType.JAVASCRIPT: [
                r"#!/usr/bin/env node",
                r"#!/usr/bin/node",
                r'require\([\'"]',
                r"module\.exports",
                r"exports\.",
                r"function\s+\w+\s*\(",
                r"const\s+\w+\s*=",
                r"let\s+\w+\s*=",
                r"var\s+\w+\s*=",
            ],
            FileType.TYPESCRIPT: [
                r"interface\s+\w+",
                r"type\s+\w+\s*=",
                r"enum\s+\w+",
                r":\s*(string|number|boolean)",
                r"export\s+(interface|type|enum)",
                r'import.*from\s+[\'"].*[\'"]',
            ],
            FileType.DOCKERFILE: [
                r"^FROM\s+\w+",
                r"^RUN\s+",
                r"^COPY\s+",
                r"^ADD\s+",
                r"^WORKDIR\s+",
                r"^EXPOSE\s+",
                r"^CMD\s+",
                r"^ENTRYPOINT\s+",
            ],
            FileType.YAML: [
                r"^\s*\w+:\s*$",
                r"^\s*-\s+\w+",
                r"^\s*\w+:\s+[\w\s]+$",
                r'version:\s*[\'"]?\d+',
                r"apiVersion:\s*",
                r"kind:\s*\w+",
            ],
            FileType.JSON: [
                r"^\s*\{",
                r"^\s*\[",
                r'\"[^"]+\":\s*[{\"\\[]',
            ],
            FileType.MARKDOWN: [
                r"^#\s+\w+",
                r"^\*\*\w+\*\*",
                r"^\*\s+\w+",
                r"^\d+\.\s+\w+",
                r"```\w*",
                r"\[.*\]\(.*\)",
            ],
        }

        # Binary file extensions
        self._binary_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gi",
            ".bmp",
            ".ico",
            ".svg",
            ".pd",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".rar",
            ".7z",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".wof",
            ".woff2",
            ".tt",
            ".ot",
            ".eot",
            ".mp3",
            ".wav",
            ".mp4",
            ".avi",
            ".mov",
            ".flv",
        }

    def classify_file(self, filename: str, content: str | None = None) -> FileType:
        """
        Classify a file based on filename and optionally content.

        Args:
            filename: The filename to classify
            content: Optional file content for content-based detection

        Returns:
            FileType enum value
        """
        # Check if it's a binary file first
        if self._is_binary_file(filename):
            return FileType.UNKNOWN

        # Try extension-based detection first
        file_type = self._classify_by_extension(filename)
        if file_type != FileType.UNKNOWN:
            return file_type

        # Try filename pattern matching
        file_type = self._classify_by_filename_pattern(filename)
        if file_type != FileType.UNKNOWN:
            return file_type

        # If content is available, try content-based detection
        if content:
            file_type = self._classify_by_content(content)
            if file_type != FileType.UNKNOWN:
                return file_type

        return FileType.UNKNOWN

    def _is_binary_file(self, filename: str) -> bool:
        """Check if file is likely binary based on extension."""
        ext = Path(filename).suffix.lower()
        return ext in self._binary_extensions

    def _classify_by_extension(self, filename: str) -> FileType:
        """Classify file by its extension."""
        ext = Path(filename).suffix.lower()
        return self._extension_mappings.get(ext, FileType.UNKNOWN)

    def _classify_by_filename_pattern(self, filename: str) -> FileType:
        """Classify file by filename patterns."""
        for file_type, patterns in self._filename_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return file_type
        return FileType.UNKNOWN

    def _classify_by_content(self, content: str) -> FileType:
        """Classify file by analyzing its content."""
        if not content or not content.strip():
            return FileType.UNKNOWN

        # Get first few lines for analysis
        lines = content.split("\n")[:20]
        content_sample = "\n".join(lines)

        # Score each file type based on pattern matches
        scores = {}
        for file_type, patterns in self._content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_sample, re.MULTILINE | re.IGNORECASE))
                score += matches
            if score > 0:
                scores[file_type] = score

        # Return the file type with the highest score
        if scores:
            return max(scores, key=lambda x: scores[x])

        return FileType.UNKNOWN

    def get_file_info(self, filename: str, content: str | None = None) -> dict[str, Any]:
        """
        Get comprehensive file information.

        Returns:
            Dictionary with file type and additional metadata
        """
        file_type = self.classify_file(filename, content)

        info = {
            "file_type": file_type,
            "extension": Path(filename).suffix.lower(),
            "is_binary": self._is_binary_file(filename),
            "is_test_file": self._is_test_file(filename),
            "is_config_file": self._is_config_file(filename),
            "is_documentation": self._is_documentation_file(filename),
            "language_specific_info": self._get_language_specific_info(
                filename, content, file_type
            ),
        }

        return info

    def _is_test_file(self, filename: str) -> bool:
        """Check if file is a test file."""
        test_patterns = [
            r"test_.*\.py$",
            r".*_test\.py$",
            r".*\.test\.js$",
            r".*\.spec\.js$",
            r".*\.test\.ts$",
            r".*\.spec\.ts$",
            r"tests?/.*",
            r"spec/.*",
            r"__tests__/.*",
        ]

        for pattern in test_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        return False

    def _is_config_file(self, filename: str) -> bool:
        """Check if file is a configuration file."""
        config_patterns = [
            r"\.yml$",
            r"\.yaml$",
            r"\.json$",
            r"\.toml$",
            r"\.ini$",
            r"\.conf$",
            r"config\..*",
            r"\.config\..*",
            r"configuration\..*",
            r"package\.json$",
            r"requirements\.txt$",
            r"setup\.py$",
            r"pyproject\.toml$",
            r"Dockerfile",
            r"docker-compose\..*",
            r"\.env$",
            r"\.env\..*",
        ]

        for pattern in config_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        return False

    def _is_documentation_file(self, filename: str) -> bool:
        """Check if file is documentation."""
        doc_patterns = [
            r"\.md$",
            r"\.markdown$",
            r"\.rst$",
            r"\.txt$",
            r"README",
            r"CHANGELOG",
            r"CONTRIBUTING",
            r"LICENSE",
            r"docs?/.*",
            r"documentation/.*",
        ]

        for pattern in doc_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        return False

    def _get_language_specific_info(
        self, filename: str, content: str | None, file_type: FileType
    ) -> dict[str, Any]:
        """Get language-specific information about the file."""
        info = {}

        if file_type == FileType.PYTHON:
            info.update(self._analyze_python_file(content))
        elif file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT]:
            info.update(self._analyze_js_ts_file(content))
        elif file_type == FileType.MARKDOWN:
            info.update(self._analyze_markdown_file(content))

        return info

    def _analyze_python_file(self, content: str | None) -> dict[str, Any]:
        """Analyze Python-specific file characteristics."""
        info = {
            "has_classes": False,
            "has_functions": False,
            "has_imports": False,
            "is_script": False,
            "estimated_complexity": "low",
        }

        if not content:
            return info

        # Check for various Python constructs
        if re.search(r"^class\s+\w+", content, re.MULTILINE):
            info["has_classes"] = True

        if re.search(r"^def\s+\w+", content, re.MULTILINE):
            info["has_functions"] = True

        if re.search(r"^(import|from)\s+\w+", content, re.MULTILINE):
            info["has_imports"] = True

        if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', content):
            info["is_script"] = True

        # Estimate complexity based on various factors
        complexity_indicators = [
            len(re.findall(r"^class\s+\w+", content, re.MULTILINE)),
            len(re.findall(r"^def\s+\w+", content, re.MULTILINE)),
            len(re.findall(r"if\s+", content)),
            len(re.findall(r"for\s+\w+\s+in", content)),
            len(re.findall(r"while\s+", content)),
            len(re.findall(r"try:", content)),
        ]

        total_complexity = sum(complexity_indicators)
        if total_complexity > 20:
            info["estimated_complexity"] = "high"
        elif total_complexity > 10:
            info["estimated_complexity"] = "medium"

        return info

    def _analyze_js_ts_file(self, content: str | None) -> dict[str, Any]:
        """Analyze JavaScript/TypeScript-specific file characteristics."""
        info: dict[str, Any] = {
            "has_functions": False,
            "has_classes": False,
            "has_imports": False,
            "has_exports": False,
            "is_module": False,
            "framework_indicators": [],
        }

        if not content:
            return info

        # Check for various JS/TS constructs
        if re.search(r"function\s+\w+", content):
            info["has_functions"] = True

        if re.search(r"class\s+\w+", content):
            info["has_classes"] = True

        if re.search(r"(import|require)\s*\(", content) or re.search(
            r"import\s+.*\s+from", content
        ):
            info["has_imports"] = True

        if re.search(r"(export|module\.exports)", content):
            info["has_exports"] = True

        if info["has_imports"] or info["has_exports"]:
            info["is_module"] = True

        # Check for framework indicators
        framework_patterns = {
            "React": [
                r"import.*React",
                r'from\s+[\'"]react[\'"]',
                r"jsx?",
                r"useState",
                r"useEffect",
            ],
            "Vue": [
                r"import.*Vue",
                r'from\s+[\'"]vue[\'"]',
                r"<template>",
                r"<script>",
                r"<style>",
            ],
            "Angular": [
                r"@Component",
                r"@Injectable",
                r"@NgModule",
                r'from\s+[\'"]@angular',
            ],
            "Express": [
                r'require\([\'"]express[\'"]',
                r"app\.get\(",
                r"app\.post\(",
            ],
            "Node.js": [
                r'require\([\'"]fs[\'"]',
                r'require\([\'"]path[\'"]',
                r"process\.",
            ],
        }

        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    info["framework_indicators"].append(framework)
                    break

        return info

    def _analyze_markdown_file(self, content: str | None) -> dict[str, Any]:
        """Analyze Markdown-specific file characteristics."""
        info = {
            "has_headers": False,
            "has_code_blocks": False,
            "has_links": False,
            "has_tables": False,
            "estimated_length": "short",
            "document_type": "general",
        }

        if not content:
            return info

        # Check for markdown constructs
        if re.search(r"^#+\s+", content, re.MULTILINE):
            info["has_headers"] = True

        if re.search(r"```", content):
            info["has_code_blocks"] = True

        if re.search(r"\[.*\]\(.*\)", content):
            info["has_links"] = True

        if re.search(r"^\|.*\|", content, re.MULTILINE):
            info["has_tables"] = True

        # Estimate document length
        lines = len(content.split("\n"))
        if lines > 200:
            info["estimated_length"] = "long"
        elif lines > 50:
            info["estimated_length"] = "medium"

        # Guess document type
        if re.search(r"readme", content, re.IGNORECASE) or "README" in content:
            info["document_type"] = "readme"
        elif re.search(r"changelog|change log", content, re.IGNORECASE):
            info["document_type"] = "changelog"
        elif re.search(r"contributing|contribution", content, re.IGNORECASE):
            info["document_type"] = "contributing"
        elif re.search(r"license|copyright", content, re.IGNORECASE):
            info["document_type"] = "license"
        elif re.search(r"installation|install|setup", content, re.IGNORECASE):
            info["document_type"] = "installation"
        elif re.search(r"api|reference|documentation", content, re.IGNORECASE):
            info["document_type"] = "api_documentation"

        return info
