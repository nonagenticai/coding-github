"""Extended tests for DocumentationSyncChecker targeting uncovered lines.

Uncovered: 172-174, 216-217, 225, 287, 333, 349-351, 403, 411, 421, 424, 427, 430,
           449, 456, 460, 472-475, 490-493, 499-513, 546, 567-577, 628, 637, 654-655,
           662-673
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from coding_client.models.pr_validation import (
    ChangeType,
    CodebaseContext,
    FileChange,
    FileType,
)

from services.pr_validation.analyzers.impact.documentation_checker import (
    DocumentationFile,
    DocumentationSyncChecker,
)


@pytest.fixture
def codebase_ctx():
    return CodebaseContext(
        total_files=10,
        total_lines=5000,
        main_languages=["python"],
        test_framework="pytest",
        documentation_format="markdown",
        dependencies=[],
        architecture_patterns=[],
    )


@pytest.fixture
def checker(codebase_ctx):
    return DocumentationSyncChecker(codebase_context=codebase_ctx)


# ---------------------------------------------------------------------------
# Lines 172-174: check_documentation_sync exception handler
# ---------------------------------------------------------------------------
async def test_check_documentation_sync_exception_fallback(checker, codebase_ctx):
    """When an unexpected exception occurs, returns safe DocumentationImpact."""
    changes = [
        FileChange(
            filename="src/api.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.ADDED,
            after_content="def foo(): pass\n",
            additions=1,
            patch="+def foo(): pass\n",
            deletions=0,
        )
    ]

    with patch.object(checker, "_identify_doc_changes", side_effect=RuntimeError("boom")):
        result = await checker.check_documentation_sync(changes, codebase_ctx)

    assert result.docs_affected == []
    assert result.outdated_docs == []
    assert any("Error" in d for d in result.new_docs_needed)
    assert result.doc_completeness_score == 0.0


# ---------------------------------------------------------------------------
# Lines 216-217: _analyze_documentation_files — exception in single doc file
# ---------------------------------------------------------------------------
async def test_analyze_documentation_files_single_file_exception(checker):
    """When _analyze_doc_file raises, the file is skipped and others continue."""
    doc_ok = FileChange(
        filename="README.md",
        file_type=FileType.MARKDOWN,
        after_content="# Hello\n\nSome text\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    doc_bad = FileChange(
        filename="docs/broken.md",
        file_type=FileType.MARKDOWN,
        after_content="# Broken\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )

    original = checker._analyze_doc_file

    call_count = 0

    async def _side_effect(change):
        nonlocal call_count
        call_count += 1
        if change.filename == "docs/broken.md":
            raise ValueError("parse error")
        return await original(change)

    with patch.object(checker, "_analyze_doc_file", side_effect=_side_effect):
        result = await checker._analyze_documentation_files([doc_bad, doc_ok])

    # Only the OK file should survive
    assert len(result) == 1
    assert result[0].file_path == "README.md"


# ---------------------------------------------------------------------------
# Line 225: _analyze_doc_file returns None when no content
# ---------------------------------------------------------------------------
async def test_analyze_doc_file_no_content(checker):
    """When both before_content and after_content are None, returns None."""
    change = FileChange(
        filename="empty.md",
        file_type=FileType.MARKDOWN,
        before_content=None,
        after_content=None,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    result = await checker._analyze_doc_file(change)
    assert result is None


async def test_analyze_doc_file_uses_before_content_fallback(checker):
    """When after_content is None, uses before_content."""
    change = FileChange(
        filename="README.md",
        file_type=FileType.MARKDOWN,
        before_content="# Old readme\n## Section\n",
        after_content=None,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    result = await checker._analyze_doc_file(change)
    assert isinstance(result, DocumentationFile)
    assert result.file_path == "README.md"
    assert result.doc_type == "readme"
    assert "Old readme" in result.sections
    assert "Section" in result.sections
    assert isinstance(result.code_references, list)
    assert isinstance(result.api_references, list)
    assert isinstance(result.example_code, list)
    assert isinstance(result.outdated_indicators, list)


# ---------------------------------------------------------------------------
# Line 287: _extract_code_references — tuple matches
# ---------------------------------------------------------------------------
def test_extract_code_references_tuple_matches(checker):
    """Patterns producing tuple matches (multiple groups) return first group."""
    content = "Use `mymodule.py` in your project.\nAlso see `helper.js`.\n"
    refs = checker._extract_code_references(content)
    # Should contain file references like mymodule.py
    assert any("mymodule.py" in r for r in refs)


def test_extract_code_references_simple_matches(checker):
    """Simple backtick references are extracted."""
    content = "Call `process_data` to start.\n"
    refs = checker._extract_code_references(content)
    assert "process_data" in refs


# ---------------------------------------------------------------------------
# Line 333: _find_outdated_indicators — version references
# ---------------------------------------------------------------------------
def test_find_outdated_indicators_version_refs(checker):
    """Version patterns like 'python 3.10' are detected."""
    content = "Requires Python 3.10 or newer.\nUse node 18.0 runtime.\n"
    indicators = checker._find_outdated_indicators(content, "README.md")
    assert any("3.10" in i for i in indicators)
    assert any("18.0" in i for i in indicators)


# ---------------------------------------------------------------------------
# Lines 349-351: _find_outdated_indicators — broken links
# ---------------------------------------------------------------------------
def test_find_outdated_indicators_http_links(checker):
    """External http links are flagged for verification."""
    content = "See [docs](http://example.com/docs) and [api](https://api.example.com).\n"
    indicators = checker._find_outdated_indicators(content, "guide.md")
    link_indicators = [i for i in indicators if "external links" in i]
    assert len(link_indicators) == 1
    # Greedy regex matches the two markdown links as one span
    assert "1 external links to verify" in link_indicators[0]


def test_find_outdated_indicators_todo_fixme(checker):
    """TODO and FIXME markers are captured."""
    content = "TODO: update this section\nFIXME: wrong example\nDEPRECATED: old API\n"
    indicators = checker._find_outdated_indicators(content, "CHANGELOG.md")
    assert any("update this section" in i for i in indicators)
    assert any("wrong example" in i for i in indicators)


# ---------------------------------------------------------------------------
# Line 403: _classify_code_reference — file reference
# ---------------------------------------------------------------------------
def test_classify_code_reference_file(checker):
    """Symbol ending in .py or .js is classified as 'file'."""
    assert checker._classify_code_reference("utils.py") == "file"
    assert checker._classify_code_reference("app.js") == "file"


# ---------------------------------------------------------------------------
# Line 411: _classify_code_reference — function reference
# ---------------------------------------------------------------------------
def test_classify_code_reference_function(checker):
    """Symbol containing '(' is classified as 'function'."""
    assert checker._classify_code_reference("do_work(") == "function"
    assert checker._classify_code_reference("run()") == "function"


# ---------------------------------------------------------------------------
# Lines 403-413: _classify_code_reference — all branches
# ---------------------------------------------------------------------------
def test_classify_code_reference_module(checker):
    """Dotted symbol not ending in .py/.js is classified as 'module'."""
    assert checker._classify_code_reference("myapp.models") == "module"


def test_classify_code_reference_config(checker):
    """Uppercase underscore symbol is classified as 'config'."""
    assert checker._classify_code_reference("MAX_RETRIES") == "config"


def test_classify_code_reference_class(checker):
    """Capitalized symbol is classified as 'class'."""
    assert checker._classify_code_reference("UserManager") == "class"


def test_classify_code_reference_variable(checker):
    """Lowercase symbol without special patterns is classified as 'variable'."""
    assert checker._classify_code_reference("counter") == "variable"


# ---------------------------------------------------------------------------
# Lines 421, 424, 427, 430: _calculate_reference_confidence
# ---------------------------------------------------------------------------
def test_calculate_reference_confidence_function_context(checker):
    """Context with 'function' or 'method' increases confidence."""
    conf = checker._calculate_reference_confidence("helper", "This function does X")
    assert conf >= 0.7


def test_calculate_reference_confidence_class_context(checker):
    """Context with 'class' increases confidence."""
    conf = checker._calculate_reference_confidence("Widget", "The Widget class is...")
    assert conf >= 0.7


def test_calculate_reference_confidence_import_context(checker):
    """Context with 'import' increases confidence."""
    conf = checker._calculate_reference_confidence("utils", "import utils from ...")
    assert conf >= 0.8


def test_calculate_reference_confidence_base(checker):
    """Plain context gives base confidence only."""
    conf = checker._calculate_reference_confidence("x", "some random line")
    assert conf == 0.5


def test_calculate_reference_confidence_capped(checker):
    """Confidence is capped at 1.0."""
    conf = checker._calculate_reference_confidence("`", "import function class method")
    assert conf <= 1.0


# ---------------------------------------------------------------------------
# Lines 449, 456, 460: _find_broken_references
# ---------------------------------------------------------------------------
def test_find_broken_references_removed_symbol(checker):
    """Removed symbols are flagged as broken references."""
    from services.pr_validation.analyzers.impact.documentation_checker import CodeReference

    refs = [
        CodeReference(
            doc_file="docs/api.md",
            line_number=10,
            reference_type="function",
            referenced_symbol="old_func",
            context="Use `old_func()` to ...",
            confidence=0.9,
        ),
    ]
    code_changes = [
        FileChange(
            filename="src/lib.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.DELETED,
            after_content="def old_func():\n    pass\n",
            additions=0,
            deletions=0,
            patch="",
        ),
    ]
    broken = checker._find_broken_references(refs, code_changes)
    assert len(broken) == 1
    assert "removed" in broken[0].lower()
    assert "old_func" in broken[0]
    assert "docs/api.md" in broken[0]


def test_find_broken_references_modified_symbol(checker):
    """Modified symbols are flagged for verification."""
    from services.pr_validation.analyzers.impact.documentation_checker import CodeReference

    refs = [
        CodeReference(
            doc_file="docs/guide.md",
            line_number=5,
            reference_type="function",
            referenced_symbol="process",
            context="`process` does work",
            confidence=0.8,
        ),
    ]
    code_changes = [
        FileChange(
            filename="src/engine.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content="def process():\n    return True\n",
            additions=0,
            deletions=0,
            patch="",
        ),
    ]
    broken = checker._find_broken_references(refs, code_changes)
    assert len(broken) == 1
    assert "modified" in broken[0].lower() or "verify" in broken[0].lower()
    assert "process" in broken[0]
    assert "docs/guide.md" in broken[0]


# ---------------------------------------------------------------------------
# Lines 472-475: _extract_symbols_from_change — JS/TS branch
# ---------------------------------------------------------------------------
def test_extract_symbols_from_change_javascript(checker):
    """JavaScript symbols are extracted from after_content."""
    change = FileChange(
        filename="src/app.js",
        file_type=FileType.JAVASCRIPT,
        after_content="function processData() {}\nclass DataStore {}\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    symbols = checker._extract_symbols_from_change(change)
    assert "processData" in symbols
    assert "DataStore" in symbols


def test_extract_symbols_from_change_typescript(checker):
    """TypeScript symbols are extracted from after_content."""
    change = FileChange(
        filename="src/app.ts",
        file_type=FileType.TYPESCRIPT,
        after_content="function validate() {}\nexport class Validator {}\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    symbols = checker._extract_symbols_from_change(change)
    assert "validate" in symbols
    assert "Validator" in symbols


def test_extract_symbols_from_change_no_content(checker):
    """No after_content returns empty set."""
    change = FileChange(
        filename="src/deleted.py",
        file_type=FileType.PYTHON,
        after_content=None,
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
        patch="",
    )
    symbols = checker._extract_symbols_from_change(change)
    assert symbols == set()


# ---------------------------------------------------------------------------
# Lines 490-493: _extract_python_symbols — SyntaxError fallback
# ---------------------------------------------------------------------------
def test_extract_python_symbols_syntax_error_fallback(checker):
    """When AST parse fails, regex fallback extracts symbols."""
    bad_python = "def broken_func(:\n    pass\nclass BrokenClass(:\n    pass\n"
    symbols = checker._extract_python_symbols(bad_python)
    assert "broken_func" in symbols
    assert "BrokenClass" in symbols


def test_extract_python_symbols_valid(checker):
    """Valid Python code extracts via AST."""
    code = (
        "def good_func():\n    pass\n\nasync def async_func():\n    pass\n\nclass Good:\n    pass\n"
    )
    symbols = checker._extract_python_symbols(code)
    assert "good_func" in symbols
    assert "async_func" in symbols
    assert "Good" in symbols


# ---------------------------------------------------------------------------
# Lines 499-513: _extract_javascript_symbols
# ---------------------------------------------------------------------------
def test_extract_javascript_symbols_all_patterns(checker):
    """All JS symbol extraction patterns are covered."""
    content = (
        "function greet() {}\n"
        "const helper = (\n"
        "myHandler: function() {}\n"
        "class Service {}\n"
        "export function exportedFunc() {}\n"
        "exports.myExport = 1\n"
    )
    symbols = checker._extract_javascript_symbols(content)
    assert "greet" in symbols
    assert "helper" in symbols
    assert "myHandler" in symbols
    assert "Service" in symbols
    assert "exportedFunc" in symbols
    assert "myExport" in symbols


# ---------------------------------------------------------------------------
# Line 546: _find_new_public_symbols — no patch returns empty
# ---------------------------------------------------------------------------
def test_find_new_public_symbols_no_patch(checker):
    """No patch content returns empty set."""
    change = FileChange(
        filename="src/module.py",
        file_type=FileType.PYTHON,
        patch="",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
    )
    result = checker._find_new_public_symbols(change)
    assert result == set()


def test_find_new_public_symbols_python(checker):
    """Public Python functions and classes from added lines are found."""
    change = FileChange(
        filename="src/module.py",
        file_type=FileType.PYTHON,
        patch="+def new_public():\n+    pass\n+def _private():\n+    pass\n+class NewClass:\n+    pass\n+class _Hidden:\n+    pass\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
    )
    result = checker._find_new_public_symbols(change)
    assert "new_public" in result
    assert "NewClass" in result
    assert "_private" not in result
    assert "_Hidden" not in result


# ---------------------------------------------------------------------------
# Lines 567-577: _find_new_public_symbols — JS/TS branch
# ---------------------------------------------------------------------------
def test_find_new_public_symbols_javascript(checker):
    """JavaScript export patterns are detected."""
    change = FileChange(
        filename="src/module.js",
        file_type=FileType.JAVASCRIPT,
        patch="+export function handleRequest() {}\n+export class Handler {}\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
    )
    result = checker._find_new_public_symbols(change)
    assert "handleRequest" in result
    assert "Handler" in result


def test_find_new_public_symbols_typescript(checker):
    """TypeScript export patterns are detected."""
    change = FileChange(
        filename="src/module.ts",
        file_type=FileType.TYPESCRIPT,
        patch="+export function validate() {}\n+export class Schema {}\n",
        change_type=ChangeType.MODIFIED,
        additions=0,
        deletions=0,
    )
    result = checker._find_new_public_symbols(change)
    assert "validate" in result
    assert "Schema" in result


# ---------------------------------------------------------------------------
# Line 628: _calculate_completeness_score — no public symbols
# ---------------------------------------------------------------------------
def test_calculate_completeness_no_public_symbols(checker):
    """When there are no public symbols, score is 1.0."""
    code_changes = [
        FileChange(
            filename="src/internal.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
            after_content="x = 1\n",  # No functions/classes
        )
    ]
    score = checker._calculate_completeness_score(code_changes, [])
    assert score == 1.0


def test_calculate_completeness_no_code_changes(checker):
    """When there are no code changes, score is 1.0."""
    score = checker._calculate_completeness_score([], [])
    assert score == 1.0


# ---------------------------------------------------------------------------
# Line 637: _calculate_completeness_score — outdated indicators reduce quality
# ---------------------------------------------------------------------------
def test_calculate_completeness_outdated_indicators_reduce_score(checker):
    """Outdated indicators in doc files reduce the quality score."""
    code_changes = [
        FileChange(
            filename="src/api.py",
            file_type=FileType.PYTHON,
            after_content="def public_api():\n    pass\n\nclass PublicClass:\n    pass\n",
            change_type=ChangeType.MODIFIED,
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    doc_files = [
        DocumentationFile(
            file_path="docs/api.md",
            doc_type="api",
            code_references=["public_api", "PublicClass"],
            outdated_indicators=["Version reference: 1.0"],
        )
    ]
    score = checker._calculate_completeness_score(code_changes, doc_files)
    # With outdated indicators, quality_score is 0.9
    assert score < 1.0


# ---------------------------------------------------------------------------
# Lines 654-655: _get_file_content — file not found, not a file
# ---------------------------------------------------------------------------
async def test_get_file_content_file_not_found(checker):
    """Non-existent file returns None."""
    result = await checker._get_file_content("/nonexistent/path/to/file.md")
    assert result is None


async def test_get_file_content_path_is_directory(checker, tmp_path):
    """When path is a directory, returns None."""
    result = await checker._get_file_content(str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# Lines 662-673: _get_file_content — encoding handling
# ---------------------------------------------------------------------------
async def test_get_file_content_utf8_success(checker):
    """UTF-8 content is read successfully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Hello World\n\nSome content.\n")
        f.flush()
        result = await checker._get_file_content(f.name)

    assert isinstance(result, str)
    assert "# Hello World" in result
    assert "Some content." in result

    Path(f.name).unlink()


async def test_get_file_content_latin1_fallback(checker):
    """When UTF-8 fails, falls back to latin-1."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        # Write bytes that are valid latin-1 but invalid UTF-8
        f.write(b"# R\xe9sum\xe9\n")
        f.flush()
        result = await checker._get_file_content(f.name)

    assert isinstance(result, str)
    assert "sum" in result
    # Verify the latin-1 decoded content contains the accented character
    assert "\xe9" in result  # e-acute from R\xe9sum\xe9

    Path(f.name).unlink()


async def test_get_file_content_general_exception(checker):
    """General exceptions return None."""
    with patch("pathlib.Path.exists", side_effect=PermissionError("denied")):
        result = await checker._get_file_content("/some/protected/file.md")
    assert result is None


# ---------------------------------------------------------------------------
# Integration: full check_documentation_sync flow
# ---------------------------------------------------------------------------
async def test_full_sync_check_with_code_and_doc_changes(checker, codebase_ctx):
    """Full flow with both code and documentation changes."""
    doc_content = "# API Docs\n\n## UserService\n\n`get_user()` returns a user.\n\n```python\nfrom mylib import get_user\nresult = get_user()\n```\n"
    changes = [
        FileChange(
            filename="docs/api.md",
            file_type=FileType.MARKDOWN,
            change_type=ChangeType.MODIFIED,
            after_content=doc_content,
            additions=5,
            deletions=0,
            patch="",
        ),
        FileChange(
            filename="src/mylib.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content="def get_user():\n    return {'id': 1}\n",
            additions=2,
            patch="+def get_user():\n+    return {'id': 1}\n",
            deletions=0,
        ),
    ]
    result = await checker.check_documentation_sync(changes, codebase_ctx)
    assert isinstance(result.docs_affected, list)
    assert isinstance(result.outdated_docs, list)
    assert isinstance(result.new_docs_needed, list)
    assert isinstance(result.doc_completeness_score, float)
    assert result.doc_completeness_score >= 0.0
    assert result.doc_completeness_score <= 1.0
    # docs/api.md should be in docs_affected
    assert "docs/api.md" in result.docs_affected


# ---------------------------------------------------------------------------
# _find_outdated_examples — example contains changed symbol
# ---------------------------------------------------------------------------
def test_find_outdated_examples_with_changed_symbol(checker):
    """Examples containing changed symbols are flagged."""
    doc_files = [
        DocumentationFile(
            file_path="docs/guide.md",
            doc_type="user_guide",
            example_code=["from mylib import compute\nresult = compute(42)\n"],
        )
    ]
    code_changes = [
        FileChange(
            filename="src/mylib.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.MODIFIED,
            after_content="def compute(x):\n    return x * 2\n",
            additions=0,
            deletions=0,
            patch="",
        )
    ]
    outdated = checker._find_outdated_examples(doc_files, code_changes)
    assert len(outdated) == 1
    assert "compute" in outdated[0]
    assert "docs/guide.md" in outdated[0]
    assert "outdated" in outdated[0].lower()


# ---------------------------------------------------------------------------
# _find_missing_documentation — new public symbols not documented
# ---------------------------------------------------------------------------
def test_find_missing_documentation_new_symbols(checker):
    """New public symbols not in any doc are flagged."""
    code_changes = [
        FileChange(
            filename="src/new_module.py",
            file_type=FileType.PYTHON,
            change_type=ChangeType.ADDED,
            additions=10,
            patch="+def brand_new_api():\n+    pass\n+class BrandNewClass:\n+    pass\n",
            deletions=0,
        )
    ]
    doc_files = [
        DocumentationFile(
            file_path="docs/api.md",
            doc_type="api",
            code_references=["other_func"],
            api_references=["OtherClass"],
        )
    ]
    missing = checker._find_missing_documentation(code_changes, doc_files)
    assert len(missing) == 2
    assert any("brand_new_api" in m and "needs documentation" in m for m in missing)
    assert any("BrandNewClass" in m and "needs documentation" in m for m in missing)
    # Both should reference the source filename
    assert all("src/new_module.py" in m for m in missing)


# ---------------------------------------------------------------------------
# _extract_example_code — tuple matches
# ---------------------------------------------------------------------------
def test_extract_example_code_with_language(checker):
    """Code blocks with language specifiers extract content."""
    content = "```python\nprint('hello')\n```\n"
    examples = checker._extract_example_code(content)
    assert len(examples) >= 1
    assert any("print('hello')" in ex for ex in examples)


def test_extract_example_code_without_language(checker):
    """Code blocks without language specifiers extract content."""
    content = "```\nsome code here\n```\n"
    examples = checker._extract_example_code(content)
    assert len(examples) >= 1
    assert any("some code here" in ex for ex in examples)


# ---------------------------------------------------------------------------
# _extract_api_references
# ---------------------------------------------------------------------------
def test_extract_api_references(checker):
    """API headers and bold references are extracted."""
    content = "## UserManager\n\n### get_user\n\n**create_user** does stuff.\n"
    refs = checker._extract_api_references(content)
    assert "UserManager" in refs
    assert "get_user" in refs
    assert "create_user" in refs


# ---------------------------------------------------------------------------
# _determine_doc_type
# ---------------------------------------------------------------------------
def test_determine_doc_type_readme(checker):
    assert checker._determine_doc_type("README.md") == "readme"


def test_determine_doc_type_changelog(checker):
    assert checker._determine_doc_type("CHANGELOG.md") == "changelog"


def test_determine_doc_type_contributing(checker):
    assert checker._determine_doc_type("CONTRIBUTING.md") == "contributing"


def test_determine_doc_type_api(checker):
    assert checker._determine_doc_type("docs/api_reference.md") == "api"


def test_determine_doc_type_general(checker):
    assert checker._determine_doc_type("some_random_file.md") == "general"


# ---------------------------------------------------------------------------
# Lines 366-367, 375-397: _find_code_references / _analyze_code_references_in_content
# ---------------------------------------------------------------------------
async def test_find_code_references_with_content(checker):
    """_find_code_references delegates to _analyze_code_references_in_content when content exists."""
    doc_files = [
        DocumentationFile(
            file_path="docs/api.md",
            doc_type="api",
        )
    ]

    with patch.object(
        checker,
        "_get_file_content",
        new_callable=AsyncMock,
        return_value="# API\n\nUse `MyClass` and `helper_func()` for processing.\n",
    ):
        refs = await checker._find_code_references(doc_files)

    assert len(refs) >= 2
    symbols = [r.referenced_symbol for r in refs]
    assert "MyClass" in symbols
    assert "helper_func" in symbols
    # All refs should reference the correct doc file
    for ref in refs:
        assert ref.doc_file == "docs/api.md"
        assert ref.line_number >= 1
        assert ref.confidence >= 0.5


async def test_find_code_references_no_content(checker):
    """_find_code_references skips files where _get_file_content returns None."""
    doc_files = [DocumentationFile(file_path="docs/missing.md", doc_type="general")]

    with patch.object(
        checker,
        "_get_file_content",
        new_callable=AsyncMock,
        return_value=None,
    ):
        refs = await checker._find_code_references(doc_files)

    assert refs == []


def test_analyze_code_references_in_content_directly(checker):
    """_analyze_code_references_in_content returns CodeReference objects for matches."""
    content = "Use `process_data` to run.\nSee `UserManager` class.\n`CONFIG_VAR` is required.\n"
    refs = checker._analyze_code_references_in_content("docs/guide.md", content)
    assert len(refs) >= 3
    symbols = {ref.referenced_symbol for ref in refs}
    assert "process_data" in symbols
    assert "UserManager" in symbols
    assert "CONFIG_VAR" in symbols
    # Verify CodeReference fields
    for ref in refs:
        assert ref.doc_file == "docs/guide.md"
        assert ref.line_number >= 1
        assert ref.reference_type in ("function", "class", "config", "variable", "module", "file")
        assert ref.confidence >= 0.5
    # Verify specific classifications
    ref_by_symbol = {ref.referenced_symbol: ref for ref in refs}
    assert ref_by_symbol["UserManager"].reference_type == "class"
    assert ref_by_symbol["CONFIG_VAR"].reference_type == "config"
    # UserManager is on line 2 ("See `UserManager` class.")
    assert ref_by_symbol["UserManager"].line_number == 2
    # "class" in context should boost confidence
    assert ref_by_symbol["UserManager"].confidence >= 0.7


# ---------------------------------------------------------------------------
# Lines 667-669: _get_file_content — latin-1 fallback also fails
# ---------------------------------------------------------------------------
async def test_get_file_content_latin1_also_fails(checker):
    """When both UTF-8 and latin-1 reads fail, returns None."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"data")
        f.flush()
        fpath = f.name

    def _mock_read_text(encoding="utf-8"):
        if encoding == "utf-8":
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock")
        else:
            raise OSError("cannot read")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.read_text", side_effect=_mock_read_text),
    ):
        result = await checker._get_file_content(fpath)

    assert result is None

    Path(fpath).unlink(missing_ok=True)
