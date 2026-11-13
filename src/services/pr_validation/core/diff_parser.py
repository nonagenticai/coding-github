"""
Git diff parser for PR validation.

This module provides utilities to parse and process git diffs from various sources
including GitHub API responses, local git operations, and patch files.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from coding_client.models.pr_validation import (
    ChangeType,
    DiffStats,
    FileChange,
    FileType,
    PRDiff,
)

from services.pr_validation.core.file_classifier import PRFileTypeClassifier

logger = logging.getLogger(__name__)


@dataclass
class DiffHunk:
    """Represents a single hunk in a diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]
    context: str = ""


@dataclass
class ParsedDiff:
    """Represents a parsed diff for a single file."""

    filename: str
    old_filename: str | None
    new_filename: str | None
    change_type: ChangeType
    additions: int
    deletions: int
    hunks: list[DiffHunk]
    binary: bool = False
    mode_change: bool = False
    old_mode: str | None = None
    new_mode: str | None = None


class DiffParser:
    """
    Parses git diffs from various sources and converts them to structured data.

    This parser handles:
    - GitHub API diff responses
    - Standard git diff output
    - Patch files
    - Binary file detection
    - Mode changes
    """

    def __init__(self):
        self.file_classifier = PRFileTypeClassifier()

        # Regex patterns for parsing
        self.diff_header_pattern = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
        self.file_mode_pattern = re.compile(r"^(new|deleted) file mode (\d+)$")
        self.index_pattern = re.compile(r"^index ([a-f0-9]+)\.\.([a-f0-9]+)(?:\s+(\d+))?$")
        self.hunk_header_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
        self.binary_pattern = re.compile(r"^Binary files? .* differ$")
        self.rename_pattern = re.compile(r"^rename from (.*?)$|^rename to (.*?)$")
        self.copy_pattern = re.compile(r"^copy from (.*?)$|^copy to (.*?)$")

    def parse_github_diff(self, github_files: list[dict]) -> list[ParsedDiff]:
        """
        Parse diff from GitHub API response.

        Args:
            github_files: List of file dicts from GitHub API

        Returns:
            List of ParsedDiff objects
        """
        parsed_diffs = []

        for file_data in github_files:
            try:
                parsed_diff = self._parse_github_file(file_data)
                parsed_diffs.append(parsed_diff)
            except Exception as e:
                logger.error(
                    f"Error parsing GitHub file {file_data.get('filename', 'unknown')}: {e}"
                )
                # Create a minimal diff for failed parsing
                parsed_diffs.append(
                    ParsedDiff(
                        filename=file_data.get("filename", "unknown"),
                        old_filename=None,
                        new_filename=None,
                        change_type=ChangeType.MODIFIED,
                        additions=file_data.get("additions", 0),
                        deletions=file_data.get("deletions", 0),
                        hunks=[],
                        binary=True,  # Assume binary if parsing failed
                    )
                )

        return parsed_diffs

    def parse_git_diff(self, diff_text: str) -> list[ParsedDiff]:
        """
        Parse a git diff text output.

        Args:
            diff_text: Raw git diff output

        Returns:
            List of ParsedDiff objects
        """
        lines = diff_text.split("\n")
        parsed_diffs = []
        current_diff = None
        current_hunk = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Diff header
            if line.startswith("diff --git"):
                if current_diff:
                    if current_hunk:
                        current_diff.hunks.append(current_hunk)
                    parsed_diffs.append(current_diff)

                current_diff = self._parse_diff_header(line)
                current_hunk = None

            # File mode changes
            elif line.startswith(("new file mode", "deleted file mode")):
                if current_diff:
                    mode_match = self.file_mode_pattern.match(line)
                    if mode_match:
                        current_diff.mode_change = True
                        if mode_match.group(1) == "new":
                            current_diff.change_type = ChangeType.ADDED
                            current_diff.new_mode = mode_match.group(2)
                        else:
                            current_diff.change_type = ChangeType.DELETED
                            current_diff.old_mode = mode_match.group(2)

            # Rename/copy detection
            elif line.startswith(("rename from", "rename to", "copy from", "copy to")):
                if current_diff:
                    if line.startswith("rename"):
                        current_diff.change_type = ChangeType.RENAMED
                        if line.startswith("rename from"):
                            current_diff.old_filename = line.split(" ", 2)[2]
                        elif line.startswith("rename to"):
                            current_diff.new_filename = line.split(" ", 2)[2]
                    elif line.startswith("copy"):
                        current_diff.change_type = ChangeType.ADDED
                        if line.startswith("copy from"):
                            current_diff.old_filename = line.split(" ", 2)[2]
                        elif line.startswith("copy to"):
                            current_diff.new_filename = line.split(" ", 2)[2]

            # Binary file detection
            elif self.binary_pattern.match(line):
                if current_diff:
                    current_diff.binary = True

            # Hunk header
            elif line.startswith("@@"):
                if current_hunk and current_diff:
                    current_diff.hunks.append(current_hunk)
                current_hunk = self._parse_hunk_header(line)

            # Hunk content
            elif current_hunk and current_diff and (line.startswith(("+", "-", " ")) or line == ""):
                current_hunk.lines.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    current_diff.additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current_diff.deletions += 1

            i += 1

        # Add the last diff
        if current_diff:
            if current_hunk:
                current_diff.hunks.append(current_hunk)
            parsed_diffs.append(current_diff)

        return parsed_diffs

    def create_file_changes(
        self,
        parsed_diffs: list[ParsedDiff],
        content_provider: Callable | None = None,
    ) -> list[FileChange]:
        """
        Convert parsed diffs to FileChange objects.

        Args:
            parsed_diffs: List of ParsedDiff objects
            content_provider: Optional function to get file content (filename, ref) -> content

        Returns:
            List of FileChange objects
        """
        file_changes = []

        for parsed_diff in parsed_diffs:
            try:
                # Determine file type
                file_type = self.file_classifier.classify_file(parsed_diff.filename)

                # Get patch content
                patch = self._reconstruct_patch(parsed_diff)

                # Get file contents if provider available
                before_content = None
                after_content = None
                if content_provider:
                    try:
                        if parsed_diff.change_type != ChangeType.ADDED:
                            before_content = content_provider(
                                parsed_diff.old_filename or parsed_diff.filename,
                                "before",
                            )
                        if parsed_diff.change_type != ChangeType.DELETED:
                            after_content = content_provider(
                                parsed_diff.new_filename or parsed_diff.filename,
                                "after",
                            )
                    except Exception as e:
                        logger.warning(f"Could not get content for {parsed_diff.filename}: {e}")

                file_change = FileChange(
                    filename=parsed_diff.filename,
                    file_type=file_type,
                    change_type=parsed_diff.change_type,
                    additions=parsed_diff.additions,
                    deletions=parsed_diff.deletions,
                    patch=patch,
                    before_content=before_content,
                    after_content=after_content,
                    binary=parsed_diff.binary,
                )

                file_changes.append(file_change)

            except Exception as e:
                logger.error(f"Error creating FileChange for {parsed_diff.filename}: {e}")
                # Create minimal file change for error cases
                file_changes.append(
                    FileChange(
                        filename=parsed_diff.filename,
                        file_type=FileType.UNKNOWN,
                        change_type=ChangeType.MODIFIED,
                        additions=parsed_diff.additions,
                        deletions=parsed_diff.deletions,
                        patch="",
                        binary=True,
                    )
                )

        return file_changes

    def create_pr_diff(
        self,
        file_changes: list[FileChange],
        base_sha: str,
        head_sha: str,
        metadata: dict | None = None,
    ) -> PRDiff:
        """
        Create a PRDiff object from file changes.

        Args:
            file_changes: List of FileChange objects
            base_sha: Base commit SHA
            head_sha: Head commit SHA
            metadata: Optional metadata about the PR

        Returns:
            PRDiff object
        """
        stats = DiffStats.from_files(file_changes)

        return PRDiff(
            base_sha=base_sha,
            head_sha=head_sha,
            files_changed=file_changes,
            stats=stats,
            metadata=metadata or {},
        )

    def _parse_github_file(self, file_data: dict) -> ParsedDiff:
        """Parse a single file from GitHub API response."""
        filename = file_data["filename"]

        # Determine change type
        status = file_data.get("status", "modified")
        change_type_map = {
            "added": ChangeType.ADDED,
            "removed": ChangeType.DELETED,
            "modified": ChangeType.MODIFIED,
            "renamed": ChangeType.RENAMED,
        }
        change_type = change_type_map.get(status, ChangeType.MODIFIED)

        # Parse patch if available
        hunks = []
        patch = file_data.get("patch", "")
        if patch and not file_data.get("binary", False):
            hunks = self._parse_patch_hunks(patch)

        return ParsedDiff(
            filename=filename,
            old_filename=file_data.get("previous_filename"),
            new_filename=(filename if change_type == ChangeType.RENAMED else None),
            change_type=change_type,
            additions=file_data.get("additions", 0),
            deletions=file_data.get("deletions", 0),
            hunks=hunks,
            binary=file_data.get("binary", False),
        )

    def _parse_diff_header(self, line: str) -> ParsedDiff:
        """Parse a git diff header line."""
        match = self.diff_header_pattern.match(line)
        if not match:
            raise ValueError(f"Invalid diff header: {line}")

        old_filename = match.group(1)
        new_filename = match.group(2)

        # Determine change type based on filenames
        change_type = ChangeType.MODIFIED
        if old_filename == "/dev/null":
            change_type = ChangeType.ADDED
        elif new_filename == "/dev/null":
            change_type = ChangeType.DELETED

        return ParsedDiff(
            filename=(new_filename if new_filename != "/dev/null" else old_filename),
            old_filename=old_filename if old_filename != "/dev/null" else None,
            new_filename=new_filename if new_filename != "/dev/null" else None,
            change_type=change_type,
            additions=0,
            deletions=0,
            hunks=[],
        )

    def _parse_hunk_header(self, line: str) -> DiffHunk:
        """Parse a hunk header line."""
        match = self.hunk_header_pattern.match(line)
        if not match:
            raise ValueError(f"Invalid hunk header: {line}")

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1
        context = match.group(5).strip() if match.group(5) else ""

        return DiffHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=[],
            context=context,
        )

    def _parse_patch_hunks(self, patch: str) -> list[DiffHunk]:
        """Parse hunks from a patch string."""
        lines = patch.split("\n")
        hunks = []
        current_hunk = None

        for line in lines:
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = self._parse_hunk_header(line)
            elif current_hunk and (line.startswith(("+", "-", " ")) or line == ""):
                current_hunk.lines.append(line)

        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    def _reconstruct_patch(self, parsed_diff: ParsedDiff) -> str:
        """Reconstruct patch text from parsed diff."""
        if parsed_diff.binary:
            return f"Binary file {parsed_diff.filename} has changed"

        patch_lines = []

        # Add header
        old_file = parsed_diff.old_filename or parsed_diff.filename
        new_file = parsed_diff.new_filename or parsed_diff.filename

        if parsed_diff.change_type == ChangeType.ADDED:
            old_file = "/dev/null"
        elif parsed_diff.change_type == ChangeType.DELETED:
            new_file = "/dev/null"

        patch_lines.append(f"diff --git a/{old_file} b/{new_file}")

        if parsed_diff.change_type == ChangeType.ADDED:
            patch_lines.append(f"new file mode {parsed_diff.new_mode or '100644'}")
        elif parsed_diff.change_type == ChangeType.DELETED:
            patch_lines.append(f"deleted file mode {parsed_diff.old_mode or '100644'}")

        patch_lines.append(f"--- a/{old_file}")
        patch_lines.append(f"+++ b/{new_file}")

        # Add hunks
        for hunk in parsed_diff.hunks:
            hunk_header = (
                f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            )
            if hunk.context:
                hunk_header += f" {hunk.context}"
            patch_lines.append(hunk_header)
            patch_lines.extend(hunk.lines)

        return "\n".join(patch_lines)

    def validate_diff(self, parsed_diff: ParsedDiff) -> list[str]:
        """
        Validate a parsed diff and return any issues found.

        Args:
            parsed_diff: The parsed diff to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check for basic consistency
        if not parsed_diff.filename:
            issues.append("Missing filename")

        # Check hunk consistency
        for i, hunk in enumerate(parsed_diff.hunks):
            additions = sum(1 for line in hunk.lines if line.startswith("+"))
            deletions = sum(1 for line in hunk.lines if line.startswith("-"))

            # Validate hunk counts
            if hunk.new_count != additions + sum(1 for line in hunk.lines if line.startswith(" ")):
                issues.append(f"Hunk {i}: New line count mismatch")

            if hunk.old_count != deletions + sum(1 for line in hunk.lines if line.startswith(" ")):
                issues.append(f"Hunk {i}: Old line count mismatch")

        # Check for suspicious patterns
        if parsed_diff.additions == 0 and parsed_diff.deletions == 0 and not parsed_diff.binary:
            issues.append("No changes detected in non-binary file")

        return issues
