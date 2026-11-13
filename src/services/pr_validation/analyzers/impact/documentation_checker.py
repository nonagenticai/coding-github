"""
Documentation sync checker for PR validation.

This module analyzes the synchronization between code changes and documentation,
identifying when documentation needs to be updated due to code modifications.
"""

import ast
import logging
import re
from dataclasses import dataclass, field

from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    DocumentationImpact,
    FileChange,
    FileType,
)

from utils.file_classifier import CoreFileTypeClassifier

logger = logging.getLogger(__name__)


@dataclass
class DocumentationFile:
    """Represents a documentation file and its characteristics."""

    file_path: str
    doc_type: str  # 'readme', 'api', 'user_guide', 'changelog', 'contributing', 'license'
    sections: list[str] = field(default_factory=list)
    code_references: list[str] = field(default_factory=list)
    api_references: list[str] = field(default_factory=list)
    example_code: list[str] = field(default_factory=list)
    outdated_indicators: list[str] = field(default_factory=list)


@dataclass
class CodeReference:
    """Represents a reference to code in documentation."""

    doc_file: str
    line_number: int
    reference_type: str  # 'function', 'class', 'module', 'example', 'config'
    referenced_symbol: str
    context: str
    confidence: float


@dataclass
class DocumentationGap:
    """Represents a gap in documentation coverage."""

    file_path: str
    gap_type: str  # 'missing_doc', 'outdated_re', 'broken_example', 'incomplete_api'
    symbol_name: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    suggested_action: str
    affected_docs: list[str] = field(default_factory=list)


class DocumentationSyncChecker:
    """
    Checks synchronization between code changes and documentation.

    This checker identifies:
    - Documentation that needs updates due to code changes
    - Broken code references in documentation
    - Missing documentation for new features
    - Outdated examples and API references
    - Inconsistencies between code and docs
    """

    def __init__(self, codebase_context: CodebaseContext | None = None):
        self.codebase_context = codebase_context
        self.file_classifier = CoreFileTypeClassifier()

        # Documentation file patterns
        self.doc_file_patterns = {
            "readme": [r"readme\.md", r"readme\.rst", r"readme\.txt"],
            "changelog": [r"changelog\.md", r"changes\.md", r"history\.md"],
            "contributing": [r"contributing\.md", r"contribute\.md"],
            "license": [r"license\.md", r"license\.txt", r"license"],
            "api": [r"api\.md", r"api_reference\.md", r"docs.*api"],
            "user_guide": [r"guide\.md", r"tutorial\.md", r"docs.*guide"],
        }

        # Code reference patterns in documentation
        self.code_ref_patterns = [
            # Function references
            r"`([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)`",  # `function(args)`
            r"`([a-zA-Z_][a-zA-Z0-9_]*)`",  # `function`
            # Class references
            r"`([A-Z][a-zA-Z0-9_]*)`",  # `ClassName`
            # Module references
            # `module.symbol`
            r"`([a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*)`",
            # File references
            r"`([a-zA-Z0-9_/.-]+\.(py|js|ts|jsx|tsx))`",
            # Configuration references
            r"`([A-Z_][A-Z0-9_]*)`",  # `CONFIG_VARIABLE`
        ]

        # Code block patterns
        self.code_block_patterns = [
            r"```python\n(.*?)\n```",
            r"```javascript\n(.*?)\n```",
            r"```typescript\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"```(\w+)\n(.*?)\n```",
        ]

        # API documentation patterns
        self.api_patterns = [
            r"## ([A-Z][a-zA-Z0-9_]*)",  # Class documentation
            r"### ([a-zA-Z_][a-zA-Z0-9_]*)",  # Method/function documentation
            r"\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*",  # Bold API references
        ]

    async def check_documentation_sync(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> DocumentationImpact:
        """
        Check documentation synchronization with code changes.

        Args:
            changes: List of file changes
            codebase_context: Context about the codebase

        Returns:
            DocumentationImpact with sync analysis
        """
        self.codebase_context = codebase_context

        try:
            # Separate documentation and code changes
            doc_changes = self._identify_doc_changes(changes)
            code_changes = self._identify_code_changes(changes)

            # Analyze existing documentation files
            doc_files = await self._analyze_documentation_files(doc_changes)

            # Find code references in documentation
            code_references = await self._find_code_references(doc_files)

            # Check for broken references due to code changes
            broken_refs = self._find_broken_references(code_references, code_changes)

            # Find missing documentation for new code
            missing_docs = self._find_missing_documentation(code_changes, doc_files)

            # Check for outdated examples
            outdated_examples = self._find_outdated_examples(doc_files, code_changes)

            # Calculate documentation completeness
            completeness_score = self._calculate_completeness_score(code_changes, doc_files)

            # Generate recommendations
            docs_affected = [doc.file_path for doc in doc_files]
            outdated_docs = broken_refs + outdated_examples
            new_docs_needed = missing_docs

            return DocumentationImpact(
                docs_affected=docs_affected,
                outdated_docs=outdated_docs,
                new_docs_needed=new_docs_needed,
                doc_completeness_score=completeness_score,
            )

        except Exception as e:
            logger.error(f"Error in documentation sync check: {e}")
            return DocumentationImpact(
                docs_affected=[],
                outdated_docs=[],
                new_docs_needed=[f"Error in documentation analysis: {str(e)}"],
                doc_completeness_score=0.0,
            )

    def _identify_doc_changes(self, changes: list[FileChange]) -> list[FileChange]:
        """Identify documentation file changes."""
        doc_changes = []

        for change in changes:
            if self.file_classifier._is_documentation_file(change.filename):
                doc_changes.append(change)

        return doc_changes

    def _identify_code_changes(self, changes: list[FileChange]) -> list[FileChange]:
        """Identify code file changes (non-documentation)."""
        code_changes = []

        for change in changes:
            if (
                change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
                and not self.file_classifier._is_documentation_file(change.filename)
                and not change.binary
            ):
                code_changes.append(change)

        return code_changes

    async def _analyze_documentation_files(
        self, doc_changes: list[FileChange]
    ) -> list[DocumentationFile]:
        """Analyze documentation files for structure and content."""
        doc_files = []

        for doc_change in doc_changes:
            try:
                doc_file = await self._analyze_doc_file(doc_change)
                if doc_file:
                    doc_files.append(doc_file)
            except Exception as e:
                logger.error(f"Error analyzing documentation file {doc_change.filename}: {e}")

        return doc_files

    async def _analyze_doc_file(self, doc_change: FileChange) -> DocumentationFile | None:
        """Analyze a single documentation file."""
        content = doc_change.after_content or doc_change.before_content
        if not content:
            return None

        doc_file = DocumentationFile(
            file_path=doc_change.filename,
            doc_type=self._determine_doc_type(doc_change.filename),
        )

        # Extract sections
        doc_file.sections = self._extract_sections(content)

        # Find code references
        doc_file.code_references = self._extract_code_references(content)

        # Find API references
        doc_file.api_references = self._extract_api_references(content)

        # Extract example code
        doc_file.example_code = self._extract_example_code(content)

        # Check for outdated indicators
        doc_file.outdated_indicators = self._find_outdated_indicators(content, doc_change.filename)

        return doc_file

    def _determine_doc_type(self, filename: str) -> str:
        """Determine the type of documentation file."""
        filename_lower = filename.lower()

        for doc_type, patterns in self.doc_file_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower):
                    return doc_type

        return "general"

    def _extract_sections(self, content: str) -> list[str]:
        """Extract section headers from markdown content."""
        sections = []

        # Find markdown headers
        header_patterns = [
            r"^# (.+)$",  # H1
            r"^## (.+)$",  # H2
            r"^### (.+)$",  # H3
        ]

        for line in content.split("\n"):
            for pattern in header_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    sections.append(match.group(1))

        return sections

    def _extract_code_references(self, content: str) -> list[str]:
        """Extract code references from documentation content."""
        references = []

        for pattern in self.code_ref_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if isinstance(matches[0], tuple) if matches else False:
                # Handle tuple matches (from patterns with multiple groups)
                references.extend([match[0] for match in matches])
            else:
                references.extend(matches)

        return list(set(references))  # Remove duplicates

    def _extract_api_references(self, content: str) -> list[str]:
        """Extract API references from documentation."""
        api_refs = []

        for pattern in self.api_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            api_refs.extend(matches)

        return list(set(api_refs))

    def _extract_example_code(self, content: str) -> list[str]:
        """Extract code examples from documentation."""
        examples = []

        for pattern in self.code_block_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle patterns with language specifier
                    examples.extend([match[-1] for match in matches])
                else:
                    examples.extend(matches)

        return examples

    def _find_outdated_indicators(self, content: str, filename: str) -> list[str]:
        """Find indicators that documentation might be outdated."""
        indicators = []

        # Check for old version references
        version_patterns = [
            r"version\s+(\d+\.\d+)",
            r"v(\d+\.\d+)",
            r"python\s+(\d+\.\d+)",
            r"node\s+(\d+\.\d+)",
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                indicators.append(f"Version reference: {match}")

        # Check for TODO/FIXME comments
        todo_patterns = [
            r"TODO:?\s*(.+)",
            r"FIXME:?\s*(.+)",
            r"DEPRECATED:?\s*(.+)",
            r"OUTDATED:?\s*(.+)",
        ]

        for pattern in todo_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            indicators.extend(matches)

        # Check for broken links (basic check)
        if "http" in content:
            broken_link_indicators = re.findall(r"\[.*\]\(http[^)]+\)", content)
            if broken_link_indicators:
                indicators.append(
                    f"Contains {len(broken_link_indicators)} external links to verify"
                )

        return indicators

    async def _find_code_references(
        self, doc_files: list[DocumentationFile]
    ) -> list[CodeReference]:
        """Find all code references in documentation files."""
        code_references = []

        for doc_file in doc_files:
            content = await self._get_file_content(doc_file.file_path)
            if content:
                refs = self._analyze_code_references_in_content(doc_file.file_path, content)
                code_references.extend(refs)

        return code_references

    def _analyze_code_references_in_content(
        self, doc_file_path: str, content: str
    ) -> list[CodeReference]:
        """Analyze code references in a single documentation file."""
        references = []

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern in self.code_ref_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    symbol = match.group(1)
                    ref_type = self._classify_code_reference(symbol)
                    confidence = self._calculate_reference_confidence(symbol, line)

                    references.append(
                        CodeReference(
                            doc_file=doc_file_path,
                            line_number=line_num,
                            reference_type=ref_type,
                            referenced_symbol=symbol,
                            context=line.strip(),
                            confidence=confidence,
                        )
                    )

        return references

    def _classify_code_reference(self, symbol: str) -> str:
        """Classify the type of code reference."""
        if "." in symbol:
            if symbol.endswith(".py") or symbol.endswith(".js"):
                return "file"
            else:
                return "module"
        elif symbol.isupper() and "_" in symbol:
            return "config"
        elif symbol[0].isupper():
            return "class"
        elif "(" in symbol or symbol.endswith("()"):
            return "function"
        else:
            return "variable"

    def _calculate_reference_confidence(self, symbol: str, context: str) -> float:
        """Calculate confidence that this is actually a code reference."""
        confidence = 0.5  # Base confidence

        # Increase confidence for certain patterns
        if symbol in ["`", '"', "'"]:
            confidence += 0.3

        if "function" in context.lower() or "method" in context.lower():
            confidence += 0.2

        if "class" in context.lower():
            confidence += 0.2

        if "import" in context.lower():
            confidence += 0.3

        return min(confidence, 1.0)

    def _find_broken_references(
        self,
        code_references: list[CodeReference],
        code_changes: list[FileChange],
    ) -> list[str]:
        """Find code references that may be broken due to code changes."""
        broken_refs = []

        # Create a map of changed/removed symbols
        changed_symbols = set()
        removed_symbols = set()

        for change in code_changes:
            symbols = self._extract_symbols_from_change(change)
            if change.change_type == ChangeType.DELETED:
                removed_symbols.update(symbols)
            else:
                changed_symbols.update(symbols)

        # Check each reference
        for ref in code_references:
            if ref.referenced_symbol in removed_symbols:
                broken_refs.append(
                    f"{ref.doc_file}:{ref.line_number} - References removed symbol '{ref.referenced_symbol}'"
                )
            elif ref.referenced_symbol in changed_symbols:
                broken_refs.append(
                    f"{ref.doc_file}:{ref.line_number} - References modified symbol '{ref.referenced_symbol}' (verify correctness)"
                )

        return broken_refs

    def _extract_symbols_from_change(self, change: FileChange) -> set[str]:
        """Extract symbols (functions, classes) from a code change."""
        symbols = set()

        if change.file_type == FileType.PYTHON and change.after_content:
            symbols.update(self._extract_python_symbols(change.after_content))
        elif (
            change.file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT] and change.after_content
        ):
            symbols.update(self._extract_javascript_symbols(change.after_content))

        return symbols

    def _extract_python_symbols(self, content: str) -> set[str]:
        """Extract Python symbols (functions, classes) from content."""
        symbols = set()

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    symbols.add(node.name)
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            symbols.update(re.findall(r"def\s+(\w+)", content))
            symbols.update(re.findall(r"class\s+(\w+)", content))

        return symbols

    def _extract_javascript_symbols(self, content: str) -> set[str]:
        """Extract JavaScript/TypeScript symbols from content."""
        symbols = set()

        # Function declarations
        symbols.update(re.findall(r"function\s+(\w+)", content))
        symbols.update(re.findall(r"const\s+(\w+)\s*=\s*\(", content))
        symbols.update(re.findall(r"(\w+)\s*:\s*function", content))

        # Class declarations
        symbols.update(re.findall(r"class\s+(\w+)", content))

        # Exports
        symbols.update(re.findall(r"export\s+(?:function\s+)?(\w+)", content))
        symbols.update(re.findall(r"exports\.(\w+)", content))

        return symbols

    def _find_missing_documentation(
        self,
        code_changes: list[FileChange],
        doc_files: list[DocumentationFile],
    ) -> list[str]:
        """Find code changes that need documentation."""
        missing_docs = []

        # Get documented symbols
        documented_symbols = set()
        for doc_file in doc_files:
            documented_symbols.update(doc_file.code_references)
            documented_symbols.update(doc_file.api_references)

        # Check each code change for new public symbols
        for change in code_changes:
            if change.change_type == ChangeType.ADDED or change.additions > 0:
                new_symbols = self._find_new_public_symbols(change)
                for symbol in new_symbols:
                    if symbol not in documented_symbols:
                        missing_docs.append(
                            f"New public symbol '{symbol}' in {change.filename} needs documentation"
                        )

        return missing_docs

    def _find_new_public_symbols(self, change: FileChange) -> set[str]:
        """Find new public symbols in a code change."""
        new_symbols: set[str] = set()

        if not change.patch:
            return new_symbols

        # Look for new function/class definitions in added lines
        added_lines = [
            line[1:]
            for line in change.patch.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]

        for line in added_lines:
            # Python patterns
            if change.file_type == FileType.PYTHON:
                func_match = re.search(r"def\s+(\w+)", line)
                if func_match and not func_match.group(1).startswith("_"):
                    new_symbols.add(func_match.group(1))

                class_match = re.search(r"class\s+(\w+)", line)
                if class_match and not class_match.group(1).startswith("_"):
                    new_symbols.add(class_match.group(1))

            # JavaScript patterns
            elif change.file_type in [
                FileType.JAVASCRIPT,
                FileType.TYPESCRIPT,
            ]:
                func_match = re.search(r"export\s+function\s+(\w+)", line)
                if func_match:
                    new_symbols.add(func_match.group(1))

                class_match = re.search(r"export\s+class\s+(\w+)", line)
                if class_match:
                    new_symbols.add(class_match.group(1))

        return new_symbols

    def _find_outdated_examples(
        self,
        doc_files: list[DocumentationFile],
        code_changes: list[FileChange],
    ) -> list[str]:
        """Find code examples in documentation that may be outdated."""
        outdated_examples = []

        # This is a streamlined check
        # In practice, you'd want to actually execute examples or do more sophisticated analysis

        for doc_file in doc_files:
            for example in doc_file.example_code:
                # Check if example contains symbols that were changed
                for change in code_changes:
                    changed_symbols = self._extract_symbols_from_change(change)
                    for symbol in changed_symbols:
                        if symbol in example:
                            outdated_examples.append(
                                f"Code example in {doc_file.file_path} may be outdated due to changes in '{symbol}'"
                            )

        return outdated_examples

    def _calculate_completeness_score(
        self,
        code_changes: list[FileChange],
        doc_files: list[DocumentationFile],
    ) -> float:
        """Calculate documentation completeness score."""
        if not code_changes:
            return 1.0

        # Count public symbols in code
        total_public_symbols = 0
        for change in code_changes:
            symbols = self._extract_symbols_from_change(change)
            public_symbols = {s for s in symbols if not s.startswith("_")}
            total_public_symbols += len(public_symbols)

        # Count documented symbols
        documented_symbols = set()
        for doc_file in doc_files:
            documented_symbols.update(doc_file.code_references)
            documented_symbols.update(doc_file.api_references)

        if total_public_symbols == 0:
            return 1.0

        # Calculate basic completeness
        basic_completeness = len(documented_symbols) / total_public_symbols

        # Adjust for documentation quality indicators
        quality_score = 1.0
        for doc_file in doc_files:
            if doc_file.outdated_indicators:
                quality_score -= 0.1  # Reduce score for outdated indicators

        return min(basic_completeness * quality_score, 1.0)

    async def _get_file_content(self, file_path: str) -> str | None:
        """Get content of a file from filesystem."""
        try:
            from pathlib import Path

            file_path_obj = Path(file_path)

            # Check if file exists and is readable
            if not file_path_obj.exists():
                logger.warning(f"File not found: {file_path}")
                return None

            if not file_path_obj.is_file():
                logger.warning(f"Path is not a file: {file_path}")
                return None

            # Read file content with encoding detection
            try:
                # Try UTF-8 first
                content = file_path_obj.read_text(encoding="utf-8")
                return content
            except UnicodeDecodeError:
                # Fallback to latin-1 for binary-like files
                try:
                    content = file_path_obj.read_text(encoding="latin-1")
                    return content
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path} with latin-1: {e}")
                    return None

        except Exception as e:
            logger.exception(f"Error reading file {file_path}: {e}")
            return None
