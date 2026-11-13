"""GitHub API Models for V2 API.

This module contains Pydantic models for GitHub integration endpoints in the V2 API.
These models support comprehensive GitHub repository operations including forking,
code changes, pull request management, and repository analysis.

The models provide:
    - Type-safe representations of GitHub operations
    - Repository provisioning and container management
    - Fork and pull request workflow support
    - Code search and file operations
    - PR validation and quality checks

Key features:
    - Integration with containerized repository environments
    - Support for batch operations on multiple files
    - Comprehensive PR validation with scoring
    - Tree-based repository navigation
    - Content search with pattern matching

Example:
    ```python
    from github_api_models import ForkRequest, CreatePRRequest

    # Fork a repository
    fork_req = ForkRequest(container_name="my-repo")

    # Create a pull request
    pr_req = CreatePRRequest(
        fork_owner="myuser",
        fork_repo="awesome-project",
        original_owner="original-owner",
        original_repo="awesome-project",
        head_branch="feature/new-feature",
        base_branch="main",
        title="Add new feature",
        message="This PR adds an awesome new feature"
    )
    ```

Note:
    All models integrate with the container management system for
    isolated repository operations and code analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

# TODO: verify import - ContainerInfo/ContainerStatus come from coding-client models
from coding_client.models.container import (  # TODO: verify import
    ContainerInfo,
    ContainerStatus,
    GitHubRepoRequest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# TODO: verify import - SAFE_STRING_PATTERN may need to come from coding-client or be defined locally
SAFE_STRING_PATTERN = r"^[\s\S]*$"  # Permissive pattern; override from coding-client if available


# TODO: verify import - AnalysisIssue may belong to coding-analysis
class AnalysisIssue(BaseModel):
    """Represents a single code issue discovered during analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "security",
                "severity": "high",
                "message": "Potential security risk: Using pickle to deserialize data from untrusted sources",
                "line": 6,
                "column": 16,
                "file": "data_loader.py",
                "rule": "B301",
            }
        }
    )

    type: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Issue type (e.g., 'syntax', 'security', 'performance')",
    )
    severity: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Issue severity ('low', 'medium', 'high', 'critical')",
    )
    message: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Human-readable issue description"
    )
    line: int = Field(..., description="Line number where issue occurs")
    column: int = Field(1, description="Column number where issue occurs")
    file: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="File path where issue was found"
    )
    rule: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Rule or check that identified the issue",
    )


class ProvisionRepositoryRequest(GitHubRepoRequest):
    """``GitHubRepoRequest`` plus coding-github-only provisioning options."""

    exact_name: bool = Field(
        default=False,
        description=(
            "When true and container_name is set, only a workspace with exactly that "
            "name is reused; repo-url reuse is skipped."
        ),
    )
    # Forwarded verbatim to coding-containers' create schema (``toolchain``,
    # Literal["python", "python+rust"], default "python", coding-containers#184).
    # ``None`` here, not "python": ``create_container`` dumps with
    # ``exclude_none=True``, so an omitted toolchain sends NO key and the
    # outgoing body stays byte-identical to the pre-toolchain request (R3).
    toolchain: Literal["python", "python+rust"] | None = Field(
        default=None,
        description=(
            "Workspace toolchain. Omit for the standard python workspace; "
            "`python+rust` provisions a Rust-capable workspace (cargo, clippy, "
            "rustfmt, sqlx-cli, loopback Postgres 18)."
        ),
    )


class ResponseStatus:
    """Response status constants."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Error response models for OpenAPI documentation
class _ErrorResponse(BaseModel):
    """Base error response for OpenAPI docs."""

    status: str = "error"
    message: str = ""
    detail: str | None = None


class Error400Response(_ErrorResponse):
    """Bad Request response."""

    pass


class Error401Response(_ErrorResponse):
    """Unauthorized response."""

    pass


class Error403Response(_ErrorResponse):
    """Forbidden response."""

    pass


class Error404Response(_ErrorResponse):
    """Not Found response."""

    pass


class Error406Response(_ErrorResponse):
    """Not Acceptable response."""

    pass


class Error408Response(_ErrorResponse):
    """Request Timeout response."""

    pass


class Error409Response(_ErrorResponse):
    """Conflict response."""

    pass


class Error422Response(_ErrorResponse):
    """Validation Error response."""

    pass


class Error429Response(_ErrorResponse):
    """Too Many Requests response."""

    pass


class Error500Response(_ErrorResponse):
    """Internal Server Error response."""

    pass


class ValidationErrorResponse(_ErrorResponse):
    """Validation error response with details."""

    errors: list[dict[str, Any]] | None = None


class BaseResponse(BaseModel):
    """Base response model providing standard fields for all API responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "timestamp": "2025-07-13T10:30:00Z",
                "request_id": "req_12345",
            }
        }
    )

    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message", pattern=SAFE_STRING_PATTERN)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Response timestamp",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=0,
        description="Unique request identifier",
    )


class ForkRequest(BaseModel):
    """Request model for forking a GitHub repository.

    This model represents a request to create a fork of a repository that
    is currently containerized within the system. The fork operation creates
    a personal copy of the repository under the authenticated user's account.

    Attributes:
        container_name (str): Name of the container holding the repository to fork

    Example:
        ```python
        fork_request = ForkRequest(container_name="awesome-opensource-project")
        ```

    Note:
        The container must exist and be accessible to the authenticated user
        before the fork operation can be performed.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"container_name": "my-github-repo"}})

    container_name: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Container name containing the repository",
    )


class ForkResponse(BaseResponse):
    """Response model for successful repository fork operations.

    This model provides details about the newly created fork, including
    URLs and ownership information for subsequent operations.

    Attributes:
        fork_url (str): Complete URL to the newly created fork
        fork_owner (str): GitHub username that owns the fork
        fork_name (str): Name of the forked repository
        original_url (str): URL of the original repository that was forked

    Example:
        ```python
        response = ForkResponse(
            status=ResponseStatus.SUCCESS,
            message="Repository forked successfully",
            fork_url="https://github.com/myuser/awesome-project",
            fork_owner="myuser",
            fork_name="awesome-project",
            original_url="https://github.com/original/awesome-project"
        )
        ```
    """

    fork_url: str = Field(..., pattern=SAFE_STRING_PATTERN, description="URL of the created fork")
    fork_owner: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Owner of the fork")
    fork_name: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Name of the forked repository"
    )
    original_url: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Original repository URL"
    )


class GitHubFileChangeOperation(BaseModel):
    """Model representing a single file change operation for GitHub API.

    This model describes a change to be applied to a file in a repository,
    supporting various change types including additions, modifications,
    deletions, and renames.

    Attributes:
        file_path (str): Path to the file relative to repository root
        content (Optional[str]): New content for the file. Required for
            "added"/"modified"/"renamed"; omitted for "deleted".
        change_type (str): Type of change ("added", "modified", "deleted", "renamed")
        old_file_path (Optional[str]): Original path for rename operations

    Example:
        ```python
        # Modify existing file
        change = FileChange(
            file_path="src/main.py",
            content="def main():\n    print('Hello, World!')\n",
            change_type="modified"
        )

        # Rename file
        rename_change = FileChange(
            file_path="src/new_main.py",
            content="def main():\n    print('Hello, World!')\n",
            change_type="renamed",
            old_file_path="src/main.py"
        )

        # Delete a file (no content)
        delete_change = FileChange(
            file_path="src/dead_code.py",
            change_type="deleted"
        )
        ```

    Note:
        For rename operations, both old_file_path and file_path must be provided.
        The content field should contain the final content after the rename.
        A delete carries no content; every other change_type requires it.

    ⛔ ``change_type`` is a CLOSED set, and must stay closed.
        ``GitHubIntegration._apply_changes`` branches on these four exact
        strings and ends in an ``else`` that treats anything unrecognised as a
        CONTENT WRITE. This field previously carried only
        ``pattern=SAFE_STRING_PATTERN`` — ``^[\\s\\S]*$``, which every string
        satisfies — so the singular ``"delete"`` validated, fell to that
        ``else``, and overwrote the file it was asked to remove, logging a
        warning and reporting success.

        That spelling is not hypothetical: the change spec is LLM-generated and
        ``AA_IMPLEMENTATION/doing_action_procedures.json`` forwards
        ``generate_code_change_spec.output.changes`` verbatim. n8n's worker
        prompt has to tell the model outright that ``"delete"``, ``"remove"``
        and ``"removed"`` are NOT recognised — which is exactly the evidence
        that a model reaches for them. Widening this back to ``str`` restores a
        silent data-destroying path; a typo must be a 422, never a write.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "src/main.py",
                "content": "def hello():\n    return 'Hello, World!'",
                "change_type": "modified",
                "old_file_path": None,
            }
        }
    )

    file_path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Path to the file")
    content: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="New file content (required except for change_type='deleted')",
    )
    change_type: Literal["added", "modified", "deleted", "renamed"] = Field(
        "modified",
        description="Type of change (added, modified, deleted, renamed)",
    )
    old_file_path: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Original file path (for renames)",
    )

    @model_validator(mode="after")
    def _check_required_fields_per_change_type(self) -> GitHubFileChangeOperation:
        """Enforce, at the boundary, exactly what ``_apply_changes`` requires.

        ``content`` used to be unconditionally required, which made a DELETE
        unrepresentable: the n8n ``generate-code-fix`` worker emits
        ``{file_path, change_type: "deleted"}`` with no content at all, so every
        delete the conveyor produced was rejected with a 422 before it reached
        the route body.

        Relaxing it to optional must not weaken the types that genuinely need
        it. ``GitHubIntegration._apply_changes`` raises ``ValueError`` for a
        content-less add/modify/rename and for a rename with no
        ``old_file_path``; the route's broad ``except Exception`` turns that
        into an opaque 500. Refuse it here instead, as a typed 422 that names
        the field.
        """
        if self.change_type != "deleted" and self.content is None:
            raise ValueError(f"content is required for change_type={self.change_type!r}")
        if self.change_type == "renamed" and not self.old_file_path:
            raise ValueError("old_file_path is required for change_type='renamed'")
        return self


class ApplyChangesRequest(BaseModel):
    """Request model for applying multiple code changes to a repository.

    This model represents a batch of file changes to be applied to a
    containerized repository, typically followed by a commit operation.

    Attributes:
        container_name (str): Name of the container holding the repository
        changes (List[FileChange]): List of file changes to apply
        branch_name (Optional[str]): Target branch name (auto-generated if not provided)
        commit_message (Optional[str]): Commit message for the changes

    Example:
        ```python
        request = ApplyChangesRequest(
            container_name="my-project",
            changes=[
                FileChange(
                    file_path="README.md",
                    content="# Updated README\n\nThis is the updated readme.",
                    change_type="modified"
                ),
                FileChange(
                    file_path="src/new_feature.py",
                    content="def new_feature():\n    pass\n",
                    change_type="added"
                )
            ],
            branch_name="feature/update-docs",
            commit_message="Update documentation and add new feature"
        )
        ```

    Note:
        If branch_name is not provided, a unique branch name will be generated.
        All changes are applied atomically within a single commit.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-github-repo",
                "changes": [
                    {
                        "file_path": "src/main.py",
                        "content": "def hello():\n    return 'Hello, World!'",
                        "change_type": "modified",
                    }
                ],
                "branch_name": "feature/update-hello",
                "commit_message": "Update hello function",
            }
        }
    )

    container_name: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Container name"
    )
    changes: list[GitHubFileChangeOperation] = Field(default=[], description="List of file changes")
    branch_name: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Branch name (auto-generated if not provided)",
    )
    commit_message: str | None = Field(
        "Applied code changes",
        pattern=SAFE_STRING_PATTERN,
        description="Commit message",
    )
    target_files: list[str] | None = Field(
        None,
        description=(
            "The card's deliverable paths. When supplied, every changed path "
            "(file_path, and old_file_path of a rename) must be in this list or the "
            "request is refused with 422 `path_not_in_target_files`; an empty list "
            "admits nothing. Independently of this field, process data (`.aa/`, "
            "`.claude/`, `plans/`, `traces/`, `run-state/`, `*_probe/`, `*.log`, "
            "`*.trace.json`) is ALWAYS refused with 422 `path_denied_process_data`, "
            "and absolute or `..` paths with 422 `path_invalid`."
        ),
        # Distinct n8n parameter name: coding-analysis's scan request also has a
        # `target_files` (files to SCAN); sharing one generated parameter would feed
        # a scan scope into the commit guard (cf. n8n-nodes-coding#42).
        json_schema_extra={"x-n8n-param-name": "commitTargetFiles"},
    )


class ApplyChangesResponse(BaseResponse):
    """Response model for successful code change application.

    This model provides details about the applied changes, including
    the branch where changes were made and commit information.

    Attributes:
        branch_name (str): Name of the branch where changes were applied
        commit_sha (Optional[str]): SHA hash of the commit containing the changes
        changes_applied (int): Number of file changes successfully applied
        fork_url (str): URL of the fork containing the changes

    Example:
        ```python
        response = ApplyChangesResponse(
            status=ResponseStatus.SUCCESS,
            message="Changes applied successfully",
            branch_name="feature/update-docs",
            commit_sha="abc123def456",
            changes_applied=3,
            fork_url="https://github.com/myuser/project"
        )
        ```
    """

    branch_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Branch where changes were applied",
    )
    commit_sha: str | None = Field(None, pattern=SAFE_STRING_PATTERN, description="Commit SHA")
    changes_applied: int = Field(..., description="Number of changes applied")
    no_changes: bool = Field(
        default=False,
        description=(
            "True when no file actually changed (every requested change was a "
            "no-op identical-content write), so zero commits were produced. The "
            "caller MUST treat this as benign 'nothing to commit' rather than a "
            "successful commit — there is no commit_sha and create-pull-request "
            "would otherwise fail with GitHub's 'No commits between' 422."
        ),
    )
    fork_url: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="URL of the fork with changes"
    )
    authoring_identity: str | None = Field(
        default=None,
        description=(
            "WHICH credential rung signed this write: 'app' (GitHub App "
            "installation — the platform identity for our own repos), 'byo' (a "
            "project/customer PAT from secrets-service), or 'env'/'pool'/"
            "'system' for a third-party owner where the App is not installed. "
            "Recorded because the ladder used to downgrade silently: GitHub's "
            "403 names the credential TYPE it ended up with and never says why "
            "a weaker one was chosen, which cost three sessions the diagnosis "
            "on 2026-08-15."
        ),
    )


class CreatePRRequest(BaseModel):
    """Request model for creating a GitHub pull request.

    This model represents a request to create a pull request from a fork
    branch to the original repository, including all necessary metadata.

    Attributes:
        fork_owner (str): GitHub username that owns the fork
        fork_repo (str): Name of the fork repository
        original_owner (str): GitHub username that owns the original repository
        original_repo (str): Name of the original repository
        head_branch (str): Branch containing the changes (in the fork)
        base_branch (str): Target branch for the pull request (default: "main")
        title (str): Pull request title
        message (str): Pull request description/body

    Example:
        ```python
        pr_request = CreatePRRequest(
            fork_owner="myusername",
            fork_repo="awesome-project",
            original_owner="original-owner",
            original_repo="awesome-project",
            head_branch="feature/new-feature",
            base_branch="main",
            title="Add awesome new feature",
            message="This PR adds an awesome new feature that improves performance by 50%"
        )
        ```

    Note:
        The head_branch must exist in the fork repository before creating the PR.
        The base_branch must exist in the original repository.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fork_owner": "myusername",
                "fork_repo": "awesome-project",
                "original_owner": "original-owner",
                "original_repo": "awesome-project",
                "head_branch": "feature/new-feature",
                "base_branch": "main",
                "title": "Add awesome new feature",
                "message": "This PR adds an awesome new feature that improves performance by 50%",
            }
        }
    )

    fork_owner: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Owner of the fork"
    )
    fork_repo: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Fork repository name",
    )
    original_owner: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Original repository owner",
    )
    original_repo: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Original repository name",
    )
    head_branch: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Branch with changes",
    )
    base_branch: str = Field("main", pattern=SAFE_STRING_PATTERN, description="Target branch")
    title: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Pull request title"
    )
    message: str = Field(
        ...,
        min_length=1,
        pattern=SAFE_STRING_PATTERN,
        description="Pull request description/message",
    )
    draft: bool = Field(
        False,
        description="Open the pull request as a draft (not ready for review). "
        "Used by the AA delivery pipeline so a PR can be opened during DOING and "
        "only published (marked ready) once QA is green.",
    )


class CreatePRResponse(BaseResponse):
    """Response model for successful pull request creation.

    This model provides details about the newly created pull request,
    including its number, URL, and current status.

    Attributes:
        pr_number (int): GitHub pull request number
        pr_url (str): Complete URL to the pull request
        pr_status (str): Current status of the pull request ("open", "closed", "merged")

    Example:
        ```python
        response = CreatePRResponse(
            status=ResponseStatus.SUCCESS,
            message="Pull request created successfully",
            pr_number=42,
            pr_url="https://github.com/original/awesome-project/pull/42",
            pr_status="open"
        )
        ```
    """

    # pr_number / pr_url are Optional only to carry the benign "no commits
    # between" no-op (no PR opened) without a sentinel; a real PR always
    # populates both. Existing callers reading an int/str on the success path
    # are unaffected.
    pr_number: int | None = Field(None, description="Pull request number")
    pr_url: str | None = Field(None, pattern=SAFE_STRING_PATTERN, description="Pull request URL")
    pr_status: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Pull request status")
    draft: bool = Field(False, description="Whether the PR was opened as a draft")
    no_changes: bool = Field(
        default=False,
        description=(
            "True when there were no commits to open a PR for (an empty-diff "
            "apply). The result is a benign no-op: status is success but no PR "
            "was opened (pr_number/pr_url are null). Lets the AA refactoring "
            "loop treat 'nothing to deliver' as benign instead of hard-failing "
            "on GitHub's 'No commits between' 422."
        ),
    )


class MarkPRReadyRequest(BaseModel):
    """Request model for marking a draft pull request ready for review.

    Flips a draft PR to ready-for-review via GitHub's GraphQL
    ``markPullRequestReadyForReview`` mutation (the REST ``pulls.edit`` route
    cannot change the draft flag). Idempotent: a no-op if the PR is already
    ready for review.
    """

    model_config = ConfigDict(extra="forbid")

    repo_owner: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository owner"
    )
    repo_name: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository name"
    )
    pr_number: int = Field(..., gt=0, description="Pull request number to mark ready")


class MarkPRReadyResponse(BaseResponse):
    """Response model for marking a pull request ready for review."""

    pr_number: int = Field(..., description="Pull request number")
    pr_url: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Pull request URL")
    pr_status: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Pull request status")
    draft: bool = Field(..., description="Draft state after the operation (False once ready)")
    already_ready: bool = Field(
        False, description="True when the PR was already ready for review (idempotent no-op)"
    )


class UpdatePRBodyRequest(BaseModel):
    """Request model for replacing an open pull request's description.

    Used by the bug-hunt flow, which opens ONE shared PR for several fixes and
    rewrites its description to list every fixed bug before asking a human to
    approve the merge. Idempotent: a no-op if the body is already identical.
    """

    model_config = ConfigDict(extra="forbid")

    repo_owner: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository owner"
    )
    repo_name: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository name"
    )
    pr_number: int = Field(..., gt=0, description="Pull request number to update")
    # Multi-line markdown, so no SAFE_STRING_PATTERN; 65536 is GitHub's PR body limit.
    body: str = Field(
        ..., min_length=1, max_length=65536, description="New pull request description"
    )


class UpdatePRBodyResponse(BaseResponse):
    """Response model for replacing a pull request's description."""

    pr_number: int = Field(..., description="Pull request number")
    pr_url: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Pull request URL")
    pr_status: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Pull request status")
    changed: bool = Field(
        ..., description="False when the body was already identical (idempotent no-op)"
    )


class ValidatePRRequest(BaseModel):
    """Request model for validating a GitHub pull request.

    This model represents a request to perform comprehensive validation
    on a pull request, including security, performance, and code quality checks.

    Attributes:
        repo_owner (str): GitHub username that owns the repository
        repo_name (str): Name of the repository
        pr_number (int): Pull request number to validate
        validation_types (Optional[List[str]]): Types of validation to perform

    Example:
        ```python
        validation_request = ValidatePRRequest(
            repo_owner="myorganization",
            repo_name="my-project",
            pr_number=42,
            validation_types=["security", "performance", "quality"]
        )
        ```

    Note:
        Available validation types include: "security", "performance", "quality",
        "tests", "documentation", "compatibility". If not specified, all types
        are performed by default.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repo_owner": "myorganization",
                "repo_name": "my-project",
                "pr_number": 42,
                "validation_types": ["security", "performance", "quality"],
            }
        }
    )

    repo_owner: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository owner"
    )
    repo_name: str = Field(
        ..., min_length=1, pattern=SAFE_STRING_PATTERN, description="Repository name"
    )
    pr_number: int = Field(..., ge=1, description="Pull request number")
    validation_types: list[str] | None = Field(
        ["security", "performance", "quality"],
        description="Types of validation to perform",
    )


class PRValidationResult(BaseModel):
    """Model representing the result of a specific PR validation check.

    This model contains the outcome of a single validation type performed
    on a pull request, including pass/fail status and detailed findings.

    Attributes:
        validation_type (str): Type of validation performed
        passed (bool): Whether the validation check passed
        score (float): Numeric score for the validation (0-100)
        issues (List[AnalysisIssue]): Specific issues found during validation
        suggestions (List[str]): Actionable suggestions for improvement

    Example:
        ```python
        result = PRValidationResult(
            validation_type="security",
            passed=False,
            score=75.0,
            issues=[
                AnalysisIssue(
                    type="security",
                    severity="high",
                    message="Potential SQL injection vulnerability",
                    line=42,
                    file="database.py"
                )
            ],
            suggestions=["Use parameterized queries", "Sanitize user input"]
        )
        ```

    Note:
        A score of 100 indicates perfect validation, while 0 indicates
        critical issues. The passed field provides a binary pass/fail result.
    """

    validation_type: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Type of validation")
    passed: bool = Field(..., description="Whether validation passed")
    score: float = Field(..., description="Validation score (0-100)")
    issues: list[AnalysisIssue] = Field(default_factory=list, description="Issues found")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions")


class ValidatePRResponse(BaseResponse):
    """Response model for pull request validation results.

    This model aggregates all validation results for a pull request,
    providing an overall assessment and individual validation details.

    Attributes:
        overall_score (float): Aggregate score across all validations (0-100)
        passed (bool): Whether the PR passed all validation checks
        validations (List[PRValidationResult]): Individual validation results
        recommendation (str): Final recommendation ("approve", "reject", "needs_work")

    Example:
        ```python
        response = ValidatePRResponse(
            status=ResponseStatus.SUCCESS,
            message="PR validation completed",
            overall_score=85.5,
            passed=True,
            validations=[
                PRValidationResult(validation_type="security", passed=True, score=90.0),
                PRValidationResult(validation_type="quality", passed=True, score=81.0)
            ],
            recommendation="approve"
        )
        ```

    Note:
        The overall_score is a weighted average of individual validation scores.
        The recommendation provides actionable guidance for code reviewers.
    """

    overall_score: float = Field(..., description="Overall validation score")
    passed: bool = Field(..., description="Whether PR passed all validations")
    validations: list[PRValidationResult] = Field(..., description="Individual validation results")
    recommendation: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Recommendation (approve/reject/needs_work)",
    )


class ProvisionResponse(BaseResponse):
    """Response model for GitHub repository provisioning operations.

    This model provides details about a successfully provisioned repository
    container, including container information and commit details.

    Attributes:
        status (str): Always SUCCESS for successful provisioning
        container (ContainerInfo): Complete information about the provisioned container
        request_id (str): Unique identifier for tracking the provisioning request
        commit_sha (Optional[str]): SHA hash of the latest commit on the provisioned branch

    Example:
        ```python
        response = ProvisionResponse(
            status=ResponseStatus.SUCCESS,
            message="Repository provisioned successfully",
            container=ContainerInfo(
                container_name="awesome-project-main",
                status=ContainerStatus.RUNNING,
                api_endpoint="http://coding-gateway:8687/api/workspaces/awesome-project-main",
                codebase_path="/workspace/awesome-project",
                repo_url="https://github.com/owner/awesome-project",
                branch="main"
            ),
            request_id="req_abc123",
            commit_sha="def456789"
        )
        ```

    Note:
        The container will be in RUNNING status when provisioning is complete.
        The commit_sha represents the HEAD commit of the provisioned branch.
    """

    status: str = ResponseStatus.SUCCESS
    container: ContainerInfo = Field(..., description="Provisioned container information")
    request_id: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Unique request identifier"
    )
    commit_sha: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="SHA of the latest commit on the provisioned branch",
    )


class RepositoryInfo(BaseModel):
    """Model representing information about a containerized repository.

    This model provides comprehensive details about a repository that has
    been provisioned into a container for analysis and operations.

    Attributes:
        container_name (str): Unique name of the container holding the repository
        repo_url (str): Complete GitHub repository URL
        branch (str): Currently checked out branch
        status (ContainerStatus): Current status of the container
        api_endpoint (str): Deprecated alias for the public endpoint
        internal_endpoint (Optional[str]): Runtime-reachable endpoint used by backend services
        public_endpoint (Optional[str]): Stable gateway-facing endpoint for clients
        description (Optional[str]): Human-readable description of the repository

    Example:
        ```python
        repo_info = RepositoryInfo(
            container_name="awesome-project-main",
            repo_url="https://github.com/owner/awesome-project",
            branch="main",
            status=ContainerStatus.RUNNING,
            api_endpoint="http://coding-gateway:8687/api/workspaces/awesome-project-main",
            internal_endpoint="http://coding-containers-awesome-project-main.code-graph-helper.svc.cluster.local:8700",
            public_endpoint="http://coding-gateway:8687/api/workspaces/awesome-project-main",
            description="A containerized instance of the awesome-project repository"
        )
        ```

    Note:
        The container_name must be unique across all provisioned repositories.
        Code graphs are accessed via the container name.
    """

    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")
    repo_url: str = Field(..., pattern=SAFE_STRING_PATTERN, description="GitHub repository URL")
    branch: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Repository branch")
    status: ContainerStatus = Field(..., description="Container status")
    api_endpoint: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Deprecated alias for public_endpoint"
    )
    internal_endpoint: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Direct runtime endpoint used for internal probes and exec routing",
    )
    public_endpoint: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Gateway-facing endpoint for client access",
    )
    code_graph_id: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="Code graph identifier"
    )
    description: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="Container description"
    )


class ListRepositoriesResponse(BaseResponse):
    """Response model for listing available containerized repositories.

    This model provides a list of all repositories that have been
    provisioned into containers and are available for operations.

    Attributes:
        status (str): Always SUCCESS for successful listing
        repositories (List[RepositoryInfo]): List of available repository containers

    Example:
        ```python
        response = ListRepositoriesResponse(
            status=ResponseStatus.SUCCESS,
            message="Repositories listed successfully",
            repositories=[
                RepositoryInfo(
                    container_name="project-a-main",
                    repo_url="https://github.com/owner/project-a",
                    branch="main",
                    status=ContainerStatus.RUNNING
                ),
                RepositoryInfo(
                    container_name="project-b-dev",
                    repo_url="https://github.com/owner/project-b",
                    branch="development",
                    status=ContainerStatus.STOPPED
                )
            ]
        )
        ```

    Note:
        The list includes repositories in all container states (running, stopped, etc.).
        Each repository entry contains full container and repository details.
    """

    status: str = ResponseStatus.SUCCESS
    repositories: list[RepositoryInfo] = Field(..., description="Available repositories")


class SearchRequest(BaseModel):
    """Request model for searching within a containerized repository.

    This model defines parameters for searching either file names or
    file contents within a repository, with support for pattern matching
    and filtering.

    Attributes:
        search_type (Literal["files", "content"]): Type of search to perform
        pattern (str): Search pattern or regular expression
        file_pattern (Optional[str]): File glob pattern filter (e.g., "*.py")
        case_sensitive (bool): Whether the search is case-sensitive
        max_results (int): Maximum number of results to return

    Example:
        ```python
        # Search for files containing "TODO" comments
        content_search = SearchRequest(
            search_type="content",
            pattern="TODO",
            file_pattern="*.py",
            case_sensitive=False,
            max_results=50
        )

        # Search for files with "test" in the name
        file_search = SearchRequest(
            search_type="files",
            pattern="test",
            case_sensitive=False,
            max_results=100
        )
        ```

    Note:
        File patterns support standard glob syntax (*, ?, []).
        Content search supports regular expressions for advanced pattern matching.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "search_type": "content",
                "pattern": "TODO",
                "file_pattern": "*.py",
                "case_sensitive": False,
                "max_results": 50,
            }
        }
    )

    search_type: Literal["files", "content"] = Field(..., description="Type of search to perform")
    pattern: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Search pattern")
    file_pattern: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="File pattern filter (e.g., '*.py')",
    )
    case_sensitive: bool = Field(False, description="Case sensitive search")
    max_results: int = Field(100, description="Maximum number of results")


class FileContentResponse(BaseResponse):
    """Response model for file content retrieval operations.

    This model provides complete file content along with metadata
    about the file, including size, modification time, and line counts.

    Attributes:
        content (str): Complete file content as a string
        file_path (str): Path to the file relative to repository root
        size (int): File size in bytes
        modified_time (int): Unix timestamp of last modification
        lines_total (int): Total number of lines in the file
        lines_returned (int): Number of lines returned (may be less if truncated)
        encoding (str): Character encoding of the file (default: "utf-8")

    Example:
        ```python
        response = FileContentResponse(
            status=ResponseStatus.SUCCESS,
            message="File content retrieved successfully",
            content="def main():\n    print('Hello, World!')\n",
            file_path="src/main.py",
            size=45,
            modified_time=1640995200,
            lines_total=2,
            lines_returned=2,
            encoding="utf-8"
        )
        ```

    Note:
        Binary files may have encoding issues; check the encoding field.
        Large files may be truncated; compare lines_total with lines_returned.
    """

    content: str = Field(..., pattern=SAFE_STRING_PATTERN, description="File content")
    file_path: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Path to the file")
    size: int = Field(..., description="File size in bytes")
    modified_time: int = Field(..., description="Last modified timestamp")
    lines_total: int = Field(..., description="Total lines in file")
    lines_returned: int = Field(..., description="Lines returned (after offset/length)")
    encoding: str = Field("utf-8", pattern=SAFE_STRING_PATTERN, description="File encoding")


class DirectoryItem(BaseModel):
    """Model representing a file or directory item in a repository.

    This model describes a single item (file or directory) within a
    repository directory listing, including metadata and permissions.

    Attributes:
        name (str): Name of the file or directory
        path (str): Full path relative to repository root
        type (Literal["file", "directory"]): Type of item
        size (Optional[int]): Size in bytes (files only)
        permissions (Optional[str]): File permissions in octal format

    Example:
        ```python
        file_item = DirectoryItem(
            name="main.py",
            path="src/main.py",
            type="file",
            size=1024,
            permissions="644"
        )

        dir_item = DirectoryItem(
            name="utils",
            path="src/utils",
            type="directory",
            size=None,
            permissions="755"
        )
        ```

    Note:
        Size is only applicable for files and will be None for directories.
        Permissions follow standard Unix octal notation (e.g., "644", "755").
    """

    name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Item name")
    path: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Relative path from repository root",
    )
    type: Literal["file", "directory"] = Field(..., description="Item type")
    size: int | None = Field(None, description="Size in bytes (files only)")
    permissions: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="File permissions"
    )


class DirectoryListResponse(BaseResponse):
    """Response model for directory listing operations.

    This model provides the contents of a directory within a containerized
    repository, including files and subdirectories with their metadata.

    Attributes:
        path (str): Directory path that was listed
        items (List[DirectoryItem]): List of files and directories in the path
        total (int): Total number of items in the directory

    Example:
        ```python
        response = DirectoryListResponse(
            status=ResponseStatus.SUCCESS,
            message="Directory contents retrieved successfully",
            path="src",
            items=[
                DirectoryItem(name="main.py", path="src/main.py", type="file", size=1024),
                DirectoryItem(name="utils", path="src/utils", type="directory")
            ],
            total=2
        )
        ```

    Note:
        Items are returned in alphabetical order with directories first.
        The total count includes all items, not just those returned.
    """

    path: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Directory path that was listed"
    )
    items: list[DirectoryItem] = Field(..., description="Directory contents")
    total: int = Field(..., description="Total number of items")


class TreeNode(BaseModel):
    """Tree node class."""

    model_config = ConfigDict(json_schema_extra={"title": "TreeNode"})
    """Model representing a node in a hierarchical repository tree structure.

    This model enables recursive representation of repository structure,
    where directories can contain child nodes (files and subdirectories).

    Attributes:
        type (Literal["file", "directory"]): Type of node
        name (str): Name of the file or directory
        children (Optional[Dict[str, TreeNode]]): Child nodes for directories

    Example:
        ```python
        tree = TreeNode(
            type="directory",
            name="src",
            children={
                "main.py": TreeNode(type="file", name="main.py", children=None),
                "utils": TreeNode(
                    type="directory",
                    name="utils",
                    children={
                        "helper.py": TreeNode(type="file", name="helper.py", children=None)
                    }
                )
            }
        )
        ```

    Note:
        The children dictionary uses file/directory names as keys.
        Leaf nodes (files) have children set to None.
    """

    type: Literal["file", "directory"] = Field(..., description="Node type")
    name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Node name")
    children: dict[str, TreeNode] | None = Field(None, description="Child nodes (directories only)")


# Enable forward references for recursive TreeNode model
TreeNode.model_rebuild()


class RepositoryTreeResponse(BaseResponse):
    """Repository tree response class."""

    model_config = ConfigDict(json_schema_extra={"title": "RepositoryTreeResponse"})
    """Response model for repository tree structure operations.

    This model provides a hierarchical view of the entire repository
    structure as a tree of nested nodes representing files and directories.

    Attributes:
        tree (TreeNode): Root node of the repository tree structure
        container_name (str): Name of the container containing the repository

    Example:
        ```python
        response = RepositoryTreeResponse(
            status=ResponseStatus.SUCCESS,
            message="Repository tree retrieved successfully",
            tree=TreeNode(
                type="directory",
                name=".",
                children={
                    "src": TreeNode(type="directory", name="src", children={...}),
                    "README.md": TreeNode(type="file", name="README.md", children=None)
                }
            ),
            container_name="awesome-project-main"
        )
        ```

    Note:
        The tree provides a complete recursive structure of the repository.
        Large repositories may have deep nesting levels.
    """

    tree: TreeNode = Field(..., description="Repository tree structure")
    container_name: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Container name")


class SearchMatch(BaseModel):
    """Model representing a single search result match.

    This model contains details about a specific match found during
    repository search operations, including location and context.

    Attributes:
        file (str): File path where the match was found
        line (int): Line number within the file (for content searches)
        content (str): The matching content or filename
        match_pattern (str): The pattern that was matched

    Example:
        ```python
        match = SearchMatch(
            file="src/main.py",
            line=42,
            content="# TODO: Implement error handling",
            match_pattern="TODO"
        )
        ```

    Note:
        For file name searches, the line number will be 0 and content
        will contain the filename. For content searches, line numbers
        are 1-based and content contains the matching line.
    """

    file: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="File path where match was found"
    )
    line: int = Field(..., description="Line number (for content search)")
    content: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Matching content or file name"
    )
    match_pattern: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Pattern that was matched"
    )


class SearchResponse(BaseResponse):
    """Response model for repository search operations.

    This model provides search results along with metadata about the
    search operation and total number of matches found.

    Attributes:
        results (List[Dict[str, Any]]): List of search result matches
        total_found (int): Total number of matches found
        search_type (Literal["files", "content"]): Type of search performed
        pattern (str): Search pattern that was used

    Example:
        ```python
        response = SearchResponse(
            status=ResponseStatus.SUCCESS,
            message="Search completed successfully",
            results=[
                {
                    "file": "src/main.py",
                    "line": 42,
                    "content": "# TODO: Implement error handling",
                    "match_pattern": "TODO"
                },
                {
                    "file": "tests/test_main.py",
                    "line": 15,
                    "content": "# TODO: Add more test cases",
                    "match_pattern": "TODO"
                }
            ],
            total_found=2,
            search_type="content",
            pattern="TODO"
        )
        ```

    Note:
        Results are returned as dictionaries for flexibility.
        ``total_found`` is the number of returned results (capped at
        ``max_results``). When ``truncated`` is True, more matches existed and
        were dropped — re-run with a higher ``max_results`` or a tighter pattern.
    """

    results: list[dict[str, Any]] = Field(..., description="Search results")  # More straightforward
    total_found: int = Field(..., description="Number of results returned (capped at max_results)")
    truncated: bool = Field(
        False,
        description="True when more matches existed than max_results and were dropped",
    )
    search_type: Literal["files", "content"] = Field(..., description="Type of search performed")
    pattern: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Search pattern used")


class PRActionsProcessedResponse(BaseResponse):
    """Response model for PR validation processing results.

    This model provides details about actions taken after processing PR validation results.

    Attributes:
        status (str): Operation status
        message (str): Human-readable message
        actions_taken (Dict[str, Any]): Details of actions performed (labels, comments, etc.)

    Example:
        ```python
        response = PRActionsProcessedResponse(
            status=ResponseStatus.SUCCESS,
            message="PR validation results processed successfully",
            actions_taken={
                "labels_added": True,
                "comment_posted": True,
                "changes_requested": False,
                "pr_closed": False
            }
        )
        ```
    """

    actions_taken: dict[str, Any] = Field(..., description="Details of actions performed on the PR")


class ValidationWebhookResponse(BaseResponse):
    """Response model for validation webhook processing.

    This model provides validation results and actions taken via webhook.

    Attributes:
        status (str): Operation status
        message (str): Human-readable message
        validation_score (float): Overall validation score (0-100)
        recommendation (str): Validation recommendation (approve/reject/needs_work)
        actions_taken (Dict[str, Any]): Details of actions performed

    Example:
        ```python
        response = ValidationWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="Webhook processed successfully",
            validation_score=85.5,
            recommendation="approve",
            actions_taken={
                "validation_performed": True,
                "labels_updated": True,
                "comment_posted": True,
                "pr_closed": False
            }
        )
        ```
    """

    validation_score: float = Field(..., description="Overall validation score (0-100)")
    recommendation: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Validation recommendation (approve/reject/needs_work)",
    )
    actions_taken: dict[str, Any] = Field(..., description="Details of actions performed")


class GitHubWebhookResponse(BaseResponse):
    """Response model for GitHub webhook processing.

    This model provides details about webhook event processing.

    Attributes:
        status (str): Always "success" for processed webhooks
        event_type (str): Type of GitHub event (e.g., "pull_request", "push")
        action (Optional[str]): Event action (e.g., "opened", "synchronize")
        processed (bool): Whether the webhook was successfully processed
        processing_time_ms (float): Processing time in milliseconds
        delivery_id (Optional[str]): GitHub webhook delivery ID

    Example:
        ```python
        response = GitHubWebhookResponse(
            status=ResponseStatus.SUCCESS,
            message="Webhook processed successfully",
            event_type="pull_request",
            action="opened",
            processed=True,
            processing_time_ms=125.5,
            delivery_id="abc123-def456"
        )
        ```
    """

    event_type: str = Field(..., pattern=SAFE_STRING_PATTERN, description="GitHub event type")
    action: str | None = Field(None, pattern=SAFE_STRING_PATTERN, description="Event action")
    processed: bool = Field(..., description="Whether webhook was processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    delivery_id: str | None = Field(
        None, pattern=SAFE_STRING_PATTERN, description="GitHub webhook delivery ID"
    )


# ---------------------------------------------------------------------------
# GitHub Issues
#
# The coding API exposed 20 routes covering repositories, pull requests, forks,
# files and webhooks, and NOT ONE issue route — while `GitHubService.create_issue`
# already existed and was reachable only from the webhook handler. These models
# open that surface. Verified 2026-08-26 with an unshimmed grep and a positive
# control, because a `grep -r .` in this environment silently skips ignore-listed
# paths and had already produced four false "does not exist" claims that day.
# ---------------------------------------------------------------------------


class IssueRef(BaseModel):
    """Repository coordinates shared by every issue operation."""

    owner: str = Field(..., description="Repository owner (user or org)")
    repo: str = Field(..., description="Repository name")

    def full_name(self) -> str:
        """PyGithub's get_repo takes 'owner/repo'."""
        return f"{self.owner}/{self.repo}"


class CreateIssueRequest(IssueRef):
    """Request model for opening a GitHub issue."""

    title: str = Field(..., min_length=1, description="Issue title")
    body: str = Field(default="", description="Issue body (markdown)")
    labels: list[str] | None = Field(default=None, description="Labels to apply")


class ListIssuesRequest(IssueRef):
    """Request model for listing issues.

    `state` mirrors GitHub's own vocabulary. Pull requests are always excluded —
    see GitHubService.list_issues for why that is not optional.
    """

    state: Literal["open", "closed", "all"] = Field(default="open")
    labels: list[str] | None = Field(default=None, description="Filter by labels")
    limit: int = Field(default=50, ge=1, le=200, description="Max issues to return")


class GetIssueRequest(IssueRef):
    """Request model for fetching one issue."""

    issue_number: int = Field(..., ge=1)


class CommentIssueRequest(IssueRef):
    """Request model for commenting on an issue."""

    issue_number: int = Field(..., ge=1)
    body: str = Field(..., min_length=1, description="Comment body (markdown)")


class CloseIssueRequest(IssueRef):
    """Request model for closing an issue."""

    issue_number: int = Field(..., ge=1)
    comment: str | None = Field(
        default=None, description="Optional comment posted BEFORE the issue is closed"
    )


class AddIssueLabelsRequest(IssueRef):
    """Request model for adding labels to an EXISTING issue.

    `CreateIssueRequest` already carries `labels`, but only at creation time — there was
    no way to label an issue that already existed. The AA conveyor's GitHub-issue intake
    needs exactly that, to stamp `conveyor:queued` / `conveyor:failed` on issues it did
    not open.

    Caps are GitHub's own, measured 2026-09-17 against the live API, not invented:
    a label name longer than **50** characters is refused by GitHub with
    `422 Validation Failed — "name is too long (maximum is 50 characters)"`. Validating
    it here turns that into a local 422 before a network round trip is spent.
    """

    issue_number: int = Field(..., ge=1)
    labels: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Label names to add. Must be non-empty — GitHub answers an empty array with "
            "422, so an empty list is a caller bug, never a silent no-op."
        ),
    )

    @field_validator("labels")
    @classmethod
    def _labels_are_non_blank(cls, v: list[str]) -> list[str]:
        """Strip each name and refuse blanks.

        A whitespace-only entry survives `min_length=1` on the LIST but is meaningless to
        GitHub, so it would be spent as a failed API call rather than caught here.
        """
        cleaned = [label.strip() for label in v]
        for label in cleaned:
            if not label:
                raise ValueError("label names must not be empty or whitespace-only")
            if len(label) > 50:
                raise ValueError(
                    f"label name exceeds GitHub's 50-character maximum: {len(label)} chars"
                )
        return cleaned


class IssueResponse(BaseModel):
    """A single issue, normalised."""

    number: int
    title: str
    state: str
    url: str
    body: str | None = None
    labels: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class ListIssuesResponse(BaseModel):
    """A page of issues. `count` is the number RETURNED, not the repository total."""

    issues: list[IssueResponse]
    count: int


class IssueCommentResponse(BaseModel):
    """The result of posting a comment."""

    issue_number: int
    comment_id: int
    url: str
    created_at: str | None = None


# ---------------------------------------------------------------------------
# PR checks + merge
#
# Measured 2026-08-26: `check_runs`, `combined_status`, `merge_method` and
# `merge_pull_request` returned ZERO hits across `coding-*/src` (unshimmed grep,
# with a positive control). `check_workflow_status()` existed at
# github_actions_trigger.py:190 with no route and no caller. So nothing in the
# stack could read CI state or merge a PR — the last two steps of the delivery
# loop had no implementation at all.
# ---------------------------------------------------------------------------


class PRChecksRequest(IssueRef):
    """Request model for reading a PR's CI state."""

    pr_number: int = Field(..., ge=1)
    expected_checks: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum number of check runs that must have conclusion 'success' before a "
            "'passed' verdict is possible. skipped/neutral runs do not count toward it. "
            "Guards the vacuous case where no workflow has been created yet."
        ),
    )


class PRCheckRun(BaseModel):
    """One check run."""

    name: str
    status: str
    conclusion: str | None = None


class PRChecksResponse(BaseModel):
    """CI state for a PR head commit.

    `verdict` is THREE-VALUED on purpose: `passed` | `failed` | `pending`. A caller must
    not infer 'passed' from `failed == 0`, because an unrun pipeline has no failures.
    """

    pr_number: int
    head_sha: str
    verdict: Literal["passed", "failed", "pending"]
    total: int
    completed: int
    succeeded: int = Field(
        default=0,
        description=(
            "Checks that completed with conclusion `success`. Only these count toward "
            "`expected_checks`; `skipped`/`neutral` neither fail nor pass."
        ),
    )
    failed: int
    failed_names: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(
        default_factory=list,
        description="`required_checks` names absent from the head SHA or still running.",
    )
    unsatisfied_required: list[str] = Field(
        default_factory=list,
        description=(
            "`required_checks` names whose latest run completed with a conclusion other "
            "than `success` (skipped and neutral included)."
        ),
    )
    checks: list[PRCheckRun] = Field(default_factory=list)


class GetPullRequestRequest(IssueRef):
    """Request model for reading one pull request."""

    pr_number: int = Field(..., ge=1)


class ListPullRequestsRequest(IssueRef):
    """Request model for listing pull requests (most recently updated first)."""

    state: Literal["open", "closed", "all"] = Field(default="open")
    limit: int = Field(default=30, ge=1, le=100)
    head: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional head-branch filter. A bare branch name is qualified with the "
            "repository owner (GitHub ignores an unqualified head filter); "
            "'owner:branch' is passed through for fork PRs."
        ),
    )


class GitHubRateLimit(BaseModel):
    """GitHub's ``X-RateLimit-*`` as of the LAST response this call made.

    ⛔ EVERY FIELD IS NULLABLE AND ``null`` MEANS *UNKNOWN*, NEVER ZERO. PyGithub
    seeds its counters at ``(-1, -1)`` / ``0`` before any response has been seen, so
    reporting the seed verbatim would tell a caller that backs off on
    ``remaining == 0`` that the quota is exhausted when in fact no header arrived.
    Unknown is reported as absence; a caller must treat `null` as "do not throttle
    on this", not as "throttle hardest".
    """

    remaining: int | None = Field(default=None, description="X-RateLimit-Remaining")
    limit: int | None = Field(default=None, description="X-RateLimit-Limit")
    reset_epoch: int | None = Field(default=None, description="X-RateLimit-Reset (unix seconds)")
    reset_at: str | None = Field(default=None, description="X-RateLimit-Reset as ISO-8601 UTC")


class PullRequestState(BaseModel):
    """Typed read shape of one pull request.

    `mergeable` is `None` when GitHub has not computed it yet, and ALWAYS `None` on
    list items (the list endpoint does not carry it) — `None` means unknown, never
    'not mergeable'.

    `merge_commit_sha`, `labels` and `base_default_branch` are all carried by the list
    endpoint, so they are populated on list items as well as single reads. All three are
    DEFAULTED: every existing consumer that does not ask for them is unaffected.
    """

    number: int
    state: str
    merged: bool
    merged_at: str | None = None
    mergeable: bool | None = None
    merge_commit_sha: str | None = Field(
        default=None,
        description=(
            "For a MERGED PR, the merge commit on the base branch. ⚠️ NON-NULL on an "
            "OPEN PR too, where GitHub reports the throwaway *test-merge* sha instead "
            "(measured: open PRs carry one while `merged_at` is null). It is NOT a "
            "commit on any branch, so a caller resolving workflow runs by sha MUST "
            "gate on `merged` first; this field alone cannot tell the two apart."
        ),
    )
    labels: list[str] = Field(
        default_factory=list,
        description=(
            "Label NAMES on the PR, in GitHub's order. Empty when unlabelled. "
            "⚠️ THIS IS A FLATTENED SHAPE AND IT SILENTLY BREAKS THE EXISTING CONSUMER. "
            "GitHub's own payload carries label OBJECTS, and n8n's "
            "workflows/observability/github-outcome-poll.json — the named Phase B "
            "consumer of this route — reads them as objects today: "
            "`prData.labels.map(l => String((l && l.name) || ''))`. Against this list "
            "of strings, `l.name` is `undefined`, EVERY element maps to '', and the "
            "`notJudged` suppression gate goes empty and FAILS OPEN. Concretely: a "
            "closed-unmerged PR deliberately labelled `superseded` / `not-a-rejection` "
            "stops being suppressed and is UPSERTed as outcome_kind='rejected', "
            "impact_score=-0.2 on `gh-closed-<repo>-<pr>` into the metering ledger — "
            "the exact row the label exists to prevent, poisoning the governor oracle, "
            "with no error anywhere. ⇒ THE MIGRATION OF THAT WORKFLOW MUST CHANGE "
            "`l.name` TO `l` IN THE SAME CHANGE that switches it to this route. "
            "Flattening is the better API; the hazard is the transition, not the shape."
        ),
    )
    head_sha: str
    base: str
    base_default_branch: str | None = Field(
        default=None,
        description=(
            "The DEFAULT BRANCH of the BASE repository — the only thing `base` can be "
            "compared against to decide whether this PR targets the repo's trunk. It "
            "exists so that `base != base_default_branch` is answerable from ONE typed "
            "call: without it a caller must also issue `github.getRepository` (the only "
            "other place `default_branch` lives), which is an extra node and an extra "
            "GitHub round trip per card. ⭐ COSTS NO EXTRA API CALL: GitHub already "
            "carries `base.repo.default_branch` inside the PR payload, so it is read "
            "off the object we have — measured, not assumed, in `TestNoLazyCompletion`. "
            "✅ POPULATED ON LIST ITEMS TOO, unlike `mergeable` and `rate_limit`: "
            "measured live 2026-09-24 on `GET /repos/{o}/{r}/pulls`, every "
            "`pull-request-simple` item carries `base.repo.default_branch`. "
            "⚠️ `None` MEANS UNKNOWN AND MUST NOT BE READ AS 'not the default branch'. "
            "It is `None` only when GitHub omits `base.repo` altogether (deleted or "
            "inaccessible base repository). The naive `base !== base_default_branch` "
            "turns that UNKNOWN into a positive mismatch, and the two named Phase B "
            "consumers then produce OPPOSITE wrong answers off the same null: "
            "`workflows/patterns/approval-callback.json :: Check Merge Base` refuses a "
            "perfectly ordinary trunk merge, while "
            "`workflows/system/delivery-sweep.json :: Observe Delivery State` records "
            "`merged_into_non_default` against a PR that DID land on trunk. ⇒ a caller "
            "must branch on `null` FIRST and refuse to decide, never fold it into the "
            "comparison. DEFAULTED, so every existing consumer is unaffected."
        ),
    )
    head: str
    title: str
    url: str
    draft: bool
    rate_limit: GitHubRateLimit = Field(
        default_factory=GitHubRateLimit,
        description=(
            "GitHub's `X-RateLimit-*` as of this call. DEFAULTED, so every existing "
            "consumer that does not ask for it is unaffected. ⚠️ Populated on the "
            "SINGLE-PR read (`get-pull-request`) only; on `list-pull-requests` items "
            "every field is `null`, which per `GitHubRateLimit` means UNKNOWN — not "
            "zero, and not 'the quota is fine'. Costs no extra API call: it is read "
            "off the requester's last response headers, never from the PR object."
        ),
    )


class ListPullRequestsResponse(BaseModel):
    """List of pull requests in the typed read shape."""

    items: list[PullRequestState] = Field(default_factory=list)


class MergePRRequest(IssueRef):
    """Request model for merging a pull request."""

    pr_number: int = Field(..., ge=1)
    merge_method: Literal["merge", "squash", "rebase"] = Field(default="squash")
    commit_title: str | None = None
    commit_message: str | None = None
    require_checks: bool = Field(
        default=True,
        description=(
            "Refuse to merge unless the check verdict is 'passed'. A 'pending' verdict "
            "is refused as firmly as a failure — on a single-node CI pool, unrun is the "
            "common case."
        ),
    )
    expected_checks: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum number of checks that must have concluded `success`. Skipped and "
            "neutral checks do NOT count toward it."
        ),
    )
    required_checks: list[str] | None = Field(
        default=None,
        description=(
            "Check-run names that must each exist on the head SHA with conclusion "
            "`success` (latest run per name). Missing or running => `checks_pending`; "
            "any other conclusion, skipped included => `checks_failed`. Omitted: only "
            "`expected_checks` applies."
        ),
    )
    sha: str | None = Field(
        default=None,
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-fA-F]{40}$",
        description=(
            "Pin the merge to this exact head commit (full 40-hex SHA). If the PR head is "
            "anything else the merge is refused with `reason: head_moved` and GitHub is "
            "never asked to merge; the SHA is also forwarded to GitHub, so a push that "
            "lands between the read and the merge is refused there (409 → `head_moved`). "
            "Omitted: legacy behaviour — whatever the head is at merge time is merged."
        ),
    )
    require_up_to_date: bool = Field(
        default=False,
        description=(
            "Refuse with `reason: base_moved` unless the merged head already contains the "
            "current base branch tip (GitHub compare `behind_by == 0`) at check time. "
            "Compares `sha` when given, else the PR head. GitHub does not pin the base, "
            "so a base commit landing between this check and the merge is not caught."
        ),
    )


class MergePRResponse(BaseModel):
    """Outcome of a merge attempt. `merged: false` carries a machine-readable `reason`.

    Reasons: `checks_pending`, `checks_failed`, `already_merged` (``sha`` is then the
    existing merge commit and ``head_sha`` the head that was merged, so a replay can
    confirm it is a no-op), `conflicts`, `head_moved`, `base_moved`, `merge_blocked`.
    """

    merged: bool
    pr_number: int
    reason: str | None = None
    sha: str | None = None
    head_sha: str | None = None
    behind_by: int | None = None
    message: str | None = None
    checks: PRChecksResponse | None = None
    head_branch: str | None = Field(default=None, description="The PR's head branch, when known.")
    head_branch_deleted: bool = Field(
        default=False,
        description=(
            "True only when the merge succeeded AND the head branch matched `aa/*` "
            "AND it was deleted. A non-`aa/*` or default branch is never deleted."
        ),
    )
    head_branch_delete_skipped: str | None = Field(
        default=None,
        description="Why the head branch was not deleted after a successful merge.",
    )


# ---------------------------------------------------------------------------
# Phase B read routes: repository metadata, PR changed files, Actions runs by SHA.
#
# WHY: six n8n workflows called api.github.com directly with a raw org PAT
# (``AA_GITHUB_TOKEN``) because these three reads had no tenant-scoped route.
# Every model below carries ``rate_limit`` (defined with ``PullRequestState``
# above, which carries it too) for the same reason — landing-reconciler reads
# GitHub's ``X-RateLimit-*`` headers today and would lose them the moment the
# call moved behind this service.
# ---------------------------------------------------------------------------


class GetRepositoryRequest(IssueRef):
    """Request model for reading one repository's metadata."""


class RepositoryPermissions(BaseModel):
    """What the ACTING credential may do on the repo.

    Each flag is tri-state: ``true`` / ``false`` / ``null``. ``null`` is "GitHub did
    not say" — an unauthenticated or metadata-only read omits the ``permissions``
    block entirely, and rendering that as five ``false`` flags would make "we know
    you may not push" indistinguishable from "we were not told".
    """

    admin: bool | None = None
    maintain: bool | None = None
    push: bool | None = None
    triage: bool | None = None
    pull: bool | None = None


class RepositoryStateResponse(BaseModel):
    """Typed read shape of one repository.

    ``archived`` and ``disabled`` are the two that silently break a delivery loop:
    a push to an archived repo fails with a 403 that names nothing, so the sweep
    reads it as an auth problem.
    """

    full_name: str
    name: str
    owner: str
    default_branch: str
    archived: bool
    disabled: bool
    private: bool
    visibility: str | None = Field(
        default=None,
        description="'public' | 'private' | 'internal'. Null when GitHub omits it.",
    )
    fork: bool
    description: str | None = None
    html_url: str
    permissions: RepositoryPermissions = Field(default_factory=RepositoryPermissions)
    rate_limit: GitHubRateLimit = Field(default_factory=GitHubRateLimit)


class ListPullRequestFilesRequest(IssueRef):
    """Request model for one PAGE of a pull request's changed files."""

    pr_number: int = Field(..., ge=1)
    page: int = Field(default=1, ge=1, description="1-based page number")
    per_page: int = Field(
        default=100,
        ge=1,
        le=100,
        description="GitHub caps this endpoint at 100 per page; 100 is the default here.",
    )


class PullRequestFile(BaseModel):
    """One changed file.

    ``patch`` is ``None`` — not ``""`` — whenever GitHub declines to supply one
    (binary files, and any diff over GitHub's size cap). An empty string would be
    indistinguishable from "this file changed but nothing was added or removed",
    which is a different fact.
    """

    filename: str
    status: str = Field(
        ...,
        description="added | removed | modified | renamed | copied | changed | unchanged",
    )
    additions: int
    deletions: int
    changes: int
    patch: str | None = None
    sha: str | None = None
    previous_filename: str | None = Field(
        default=None, description="Set only for status 'renamed'."
    )
    blob_url: str | None = None


class ListPullRequestFilesResponse(BaseModel):
    """One page of a PR's changed files."""

    pr_number: int
    page: int
    per_page: int
    count: int
    has_more: bool = Field(
        ...,
        description=(
            "A full page came back, so ANOTHER page MAY exist — it is not a promise "
            "that one does. GitHub's files endpoint carries no total, so the only "
            "honest test is `count == per_page`; the caller stops when a short page "
            "or an empty page arrives."
        ),
    )
    files: list[PullRequestFile] = Field(default_factory=list)
    rate_limit: GitHubRateLimit = Field(default_factory=GitHubRateLimit)


class ListWorkflowRunsRequest(IssueRef):
    """Request model for Actions runs keyed by head SHA.

    ``head_sha`` is validated as hex on purpose. GitHub's ``head_sha`` filter
    accepts any string and answers an unmatched one with an EMPTY list, so a branch
    name passed here would read as "this commit has no runs" — the vacuous-green
    shape that `PRChecksResponse.verdict` exists to refuse.
    """

    head_sha: str = Field(
        ...,
        min_length=7,
        max_length=40,
        pattern=r"^[0-9a-fA-F]+$",
        description="Full or abbreviated commit SHA (hex, 7..40 chars).",
    )
    per_page: int = Field(
        default=30,
        ge=1,
        le=100,
        # Distinct n8n parameter name: without it this merges with
        # listPullRequestFiles.per_page (default 100) in the generated node, so an
        # untouched listWorkflowRuns sends per_page=100 (n8n-nodes-coding#42).
        json_schema_extra={"x-n8n-param-name": "workflowRunsPerPage"},
    )


class WorkflowRunState(BaseModel):
    """One Actions workflow run.

    ``conclusion`` is ``None`` for every run that has not completed. ``status ==
    'completed'`` is what makes ``conclusion`` meaningful; a ``None`` conclusion is
    never a pass and never a failure.
    """

    id: int
    name: str | None = None
    status: str | None = None
    conclusion: str | None = None
    html_url: str | None = None
    head_sha: str
    head_branch: str | None = None
    event: str | None = None
    path: str | None = Field(
        default=None, description="Workflow file path, e.g. .github/workflows/tests.yml"
    )
    workflow_id: int | None = None
    run_number: int | None = None
    run_attempt: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ListWorkflowRunsResponse(BaseModel):
    """Actions runs for one head SHA.

    ⛔ ``count == 0`` IS NOT "GREEN" AND IS NOT "NO CI". It means GitHub has created
    no run for this SHA *yet* (the common case on a two-node pool), or the runs have
    aged out of Actions retention. Same property as ``PRChecksResponse.verdict``:
    the absence of a failure is not a pass.
    """

    head_sha: str
    count: int
    runs: list[WorkflowRunState] = Field(default_factory=list)
    rate_limit: GitHubRateLimit = Field(default_factory=GitHubRateLimit)


class GetBranchRequest(IssueRef):
    """Request model for reading one branch's HEAD commit.

    ``branch`` is the branch NAME (``feature/x`` is valid; the slash reaches GitHub
    as a path separator, which its branches endpoint accepts). Whitespace and ``..``
    are refused here because git refuses them in a ref name — a request carrying
    one can only ever 404 upstream, and answering 422 says why.
    """

    branch: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Branch name, e.g. `main` or `feature/x`.",
    )

    @field_validator("branch")
    @classmethod
    def _valid_ref_name(cls, value: str) -> str:
        if any(ch.isspace() for ch in value):
            raise ValueError("branch must not contain whitespace")
        if ".." in value:
            raise ValueError("branch must not contain '..'")
        return value


class BranchHeadResponse(BaseModel):
    """One branch and the commit its HEAD points at.

    ``sha`` is the full 40-hex commit SHA — never abbreviated, so a poller comparing
    it to the last-seen value cannot be fooled by a prefix collision.
    """

    owner: str
    repo: str
    branch: str
    sha: str = Field(..., pattern=r"^[0-9a-f]{40}$", description="HEAD commit SHA (40 hex).")
    protected: bool
    rate_limit: GitHubRateLimit = Field(default_factory=GitHubRateLimit)


def _valid_compare_ref(value: str, field: str) -> str:
    """Same refusals as ``GetBranchRequest.branch``. ``..`` also guards the compare
    URL itself: GitHub reads ``base...head`` as its separator, so a ref carrying dots
    could re-split the pair upstream instead of failing as one bad name."""
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{field} must not contain whitespace")
    if ".." in value:
        raise ValueError(f"{field} must not contain '..'")
    return value


class CompareCommitsRequest(IssueRef):
    """Request model for comparing two commits (``base`` vs ``head``).

    Each side is a commit SHA or a ref name (``main``, ``feature/x``). The question
    this answers for the conveyor is "is ``base`` already contained in ``head``?" —
    e.g. a child PR's merge commit against the run-branch HEAD.
    """

    base: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Base commit SHA or ref name, e.g. a merge commit SHA or `main`.",
    )
    head: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Head commit SHA or ref name, e.g. `feature/x`.",
    )

    @field_validator("base")
    @classmethod
    def _valid_base(cls, value: str) -> str:
        return _valid_compare_ref(value, "base")

    @field_validator("head")
    @classmethod
    def _valid_head(cls, value: str) -> str:
        return _valid_compare_ref(value, "head")


class CompareCommitsResponse(BaseModel):
    """GitHub's comparison of ``base`` against ``head``.

    ⛔ ``base_is_ancestor_of_head`` is DERIVED from ``status`` (``ahead`` or
    ``identical``), never from ``ahead_by``: a ``diverged`` pair also has
    ``ahead_by > 0`` and is NOT an ancestor relationship. An unknown sha or ref is a
    404, never a 200 carrying ``false`` — "not contained" and "not found" must not
    look alike to a caller gating a merge on this.
    """

    owner: str
    repo: str
    base: str
    head: str
    status: Literal["ahead", "behind", "identical", "diverged"]
    ahead_by: int = Field(..., ge=0)
    behind_by: int = Field(..., ge=0)
    merge_base_sha: str = Field(
        ..., pattern=r"^[0-9a-f]{40}$", description="Merge-base commit SHA (40 hex)."
    )
    base_is_ancestor_of_head: bool
    rate_limit: GitHubRateLimit = Field(default_factory=GitHubRateLimit)


class CompareFilesRequest(CompareCommitsRequest):
    """Request model for listing the files changed between ``base`` and ``head``.

    Identical fields and validation to :class:`CompareCommitsRequest` (subclassed so
    the two can never drift apart): whitespace and ``..`` are refused with 422.
    """


class CompareFile(BaseModel):
    """One changed file from GitHub's compare. ``previous_filename`` is set only when
    ``status`` is ``renamed``."""

    filename: str
    status: Literal["added", "modified", "removed", "renamed", "copied", "changed", "unchanged"]
    previous_filename: str | None = None


class CompareFilesResponse(CompareCommitsResponse):
    """The EXACT list of files changed between ``base`` and ``head``.

    ⛔ Never a possibly-truncated list: GitHub's compare returns at most 300 files with
    no truncation flag, so at 300 or more the route answers 422
    ``compare_files_truncated`` instead of this model. ``files`` is sorted by
    ``filename`` and ``files_count == len(files)``. ``base_is_ancestor_of_head`` is
    derived from ``status`` exactly as on ``/compare``.
    """

    files: list[CompareFile]
    files_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _count_matches(self) -> CompareFilesResponse:
        if self.files_count != len(self.files):
            raise ValueError("files_count must equal len(files)")
        return self


class CreateRustTargetRepoRequest(BaseModel):
    """Create the WRITABLE Rust target repo of a Python→Rust rewrite.

    The owner/name/source rules (``conveyor-coding`` only, name ends ``-rs``, every
    source PRIVATE under ``nonagenticai``/``conveyor-coding``) are enforced by
    ``services.rust_target`` and answered 422 with a machine ``error_code``.
    """

    target: str = Field(
        ...,
        min_length=3,
        max_length=140,
        description="Target 'owner/name-rs'; owner must be conveyor-coding",
        examples=["conveyor-coding/repo-jobs-rs"],
    )
    sources: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Python source repos 'owner/repo' (PRIVATE, nonagenticai or conveyor-coding)",
        examples=[["conveyor-coding/repo-jobs"]],
    )


class CreateRustTargetRepoResponse(BaseModel):
    """The created target: PRIVATE, one commit on ``default_branch``."""

    target: str = Field(..., description="'owner/name' as created")
    url: str = Field(..., description="GitHub html_url of the new repository")
    default_branch: str = Field(..., description="Always 'main'")
    head_sha: str = Field(..., description="The single initial commit pushed to main")
    template_sha: str = Field(
        ..., description="nonagenticai/template-rust-service commit the target was built from"
    )
