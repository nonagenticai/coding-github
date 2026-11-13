"""Unit tests for Pydantic models in schemas/github_api_models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.github_api_models import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    CreatePRRequest,
    CreatePRResponse,
    Error400Response,
    Error404Response,
    Error500Response,
    ForkRequest,
    ForkResponse,
    GitHubFileChangeOperation,
    ListRepositoriesResponse,
    ProvisionResponse,
    PRValidationResult,
    RepositoryInfo,
    ValidatePRRequest,
    ValidatePRResponse,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _container_info(**overrides):
    """Build a minimal ContainerInfo-compatible object."""
    from coding_client.models.container import ContainerInfo, ContainerStatus

    defaults = dict(
        container_name="ctr-1",
        api_endpoint="http://gw:8687/api/ctr-1",
        codebase_path="/app/code",
        repo_url="https://github.com/o/r",
        branch="main",
        status=ContainerStatus("running"),
    )
    defaults.update(overrides)
    return ContainerInfo(**defaults)


def _container_status(**overrides):
    from coding_client.models.container import ContainerStatus

    # ContainerStatus is a StrEnum; ignore extra kwargs and use 'status' value
    status_val = overrides.pop("status", "running")
    # Ignore any extra kwargs like 'healthy' that don't apply to StrEnum
    return ContainerStatus(status_val)


# ---------------------------------------------------------------------------
# 1. test_provision_request_valid  (ProvisionResponse used instead since
#    there is no ProvisionRequest model -- we test the closest request model
#    which is the GitHubRepoRequest from conftest / the fork/apply request)
# ---------------------------------------------------------------------------
def test_provision_request_valid():
    """ForkRequest with correct payload should instantiate."""
    req = ForkRequest(container_name="my-repo")
    assert req.container_name == "my-repo"


# ---------------------------------------------------------------------------
# 2. test_provision_request_invalid_url
# ---------------------------------------------------------------------------
def test_provision_request_invalid_url():
    """ForkRequest rejects empty container_name (min_length=1)."""
    with pytest.raises(ValidationError):
        ForkRequest(container_name="")


# ---------------------------------------------------------------------------
# 3. test_provision_response_shape
# ---------------------------------------------------------------------------
def test_provision_response_shape():
    """ProvisionResponse should instantiate with required fields."""
    resp = ProvisionResponse(
        status="success",
        message="Provisioned",
        container=_container_info(),
        request_id="req_abc12345",
        commit_sha="deadbeef",
    )
    assert resp.status == "success"
    assert resp.request_id == "req_abc12345"
    assert resp.commit_sha == "deadbeef"
    assert resp.container.container_name == "ctr-1"


# ---------------------------------------------------------------------------
# 4. test_repository_info_model
# ---------------------------------------------------------------------------
def test_repository_info_model():
    """RepositoryInfo should expose all expected fields."""
    info = RepositoryInfo(
        container_name="my-ctr",
        repo_url="https://github.com/o/r",
        branch="main",
        status=_container_status(),
        api_endpoint="http://gw:8687/api/ctr",
        internal_endpoint="http://ctr.svc:8700",
        public_endpoint="http://gw:8687/api/ctr",
        description="A repo",
    )
    assert info.container_name == "my-ctr"
    assert info.branch == "main"
    assert info.internal_endpoint == "http://ctr.svc:8700"
    assert info.public_endpoint == "http://gw:8687/api/ctr"
    assert info.description == "A repo"


# ---------------------------------------------------------------------------
# 5. test_list_repositories_response
# ---------------------------------------------------------------------------
def test_list_repositories_response():
    """ListRepositoriesResponse should hold a list of RepositoryInfo."""
    repo = RepositoryInfo(
        container_name="ctr-1",
        repo_url="https://github.com/o/r",
        branch="main",
        status=_container_status(),
        api_endpoint="http://gw/api/ctr-1",
    )
    resp = ListRepositoriesResponse(
        status="success",
        message="Listed",
        repositories=[repo],
    )
    assert len(resp.repositories) == 1
    assert resp.repositories[0].container_name == "ctr-1"


# ---------------------------------------------------------------------------
# 6. test_error_models
# ---------------------------------------------------------------------------
def test_error_models():
    """Error envelope models should instantiate with defaults."""
    e400 = Error400Response(message="bad request")
    assert e400.status == "error"
    assert e400.message == "bad request"

    e404 = Error404Response(message="not found")
    assert e404.status == "error"

    e500 = Error500Response(message="internal error")
    assert e500.status == "error"


# ---------------------------------------------------------------------------
# 7. test_internal_endpoint_field
# ---------------------------------------------------------------------------
def test_internal_endpoint_field():
    """RepositoryInfo.internal_endpoint should accept and validate a string."""
    info = RepositoryInfo(
        container_name="ctr",
        repo_url="https://github.com/o/r",
        branch="main",
        status=_container_status(),
        api_endpoint="http://gw/api/ctr",
        internal_endpoint="http://ctr.ns.svc.cluster.local:8700",
    )
    assert info.internal_endpoint == "http://ctr.ns.svc.cluster.local:8700"


# ---------------------------------------------------------------------------
# 8. test_public_endpoint_field
# ---------------------------------------------------------------------------
def test_public_endpoint_field():
    """RepositoryInfo.public_endpoint should accept and validate a string."""
    info = RepositoryInfo(
        container_name="ctr",
        repo_url="https://github.com/o/r",
        branch="main",
        status=_container_status(),
        api_endpoint="http://gw/api/ctr",
        public_endpoint="http://gw:8687/api/containers/ctr",
    )
    assert info.public_endpoint == "http://gw:8687/api/containers/ctr"


# ---------------------------------------------------------------------------
# 9. test_backward_compat_api_endpoint
# ---------------------------------------------------------------------------
def test_backward_compat_api_endpoint():
    """Old payloads that only supply api_endpoint (no internal/public) should still parse."""
    info = RepositoryInfo(
        container_name="ctr",
        repo_url="https://github.com/o/r",
        branch="main",
        status=_container_status(),
        api_endpoint="http://gw/api/ctr",
    )
    assert info.api_endpoint == "http://gw/api/ctr"
    assert info.internal_endpoint is None
    assert info.public_endpoint is None


# ---------------------------------------------------------------------------
# 10. test_pr_models_roundtrip
# ---------------------------------------------------------------------------
def test_pr_models_roundtrip():
    """CreatePRRequest -> model_dump -> model_validate should roundtrip."""
    req = CreatePRRequest(
        fork_owner="user",
        fork_repo="repo",
        original_owner="org",
        original_repo="repo",
        head_branch="feat/x",
        base_branch="main",
        title="Add feature",
        message="Detailed description",
    )
    data = req.model_dump()
    restored = CreatePRRequest.model_validate(data)
    assert restored.fork_owner == "user"
    assert restored.title == "Add feature"
    assert restored == req

    # Also test response roundtrip
    resp = CreatePRResponse(
        status="success",
        message="PR created",
        pr_number=42,
        pr_url="https://github.com/org/repo/pull/42",
        pr_status="open",
    )
    data2 = resp.model_dump()
    restored2 = CreatePRResponse.model_validate(data2)
    assert restored2.pr_number == 42
    assert restored2.pr_status == "open"


# ---------------------------------------------------------------------------
# 11. test_fork_models
# ---------------------------------------------------------------------------
def test_fork_models():
    """ForkRequest and ForkResponse should instantiate correctly."""
    req = ForkRequest(container_name="my-repo")
    assert req.container_name == "my-repo"

    resp = ForkResponse(
        status="success",
        message="Forked",
        fork_url="https://github.com/user/my-repo",
        fork_owner="user",
        fork_name="my-repo",
        original_url="https://github.com/org/my-repo",
    )
    assert resp.fork_owner == "user"
    assert resp.original_url == "https://github.com/org/my-repo"


# ---------------------------------------------------------------------------
# 12. test_apply_changes_models
# ---------------------------------------------------------------------------
def test_apply_changes_models():
    """ApplyChangesRequest and ApplyChangesResponse should instantiate."""
    change = GitHubFileChangeOperation(
        file_path="src/main.py",
        content="print('hello')",
        change_type="modified",
    )
    req = ApplyChangesRequest(
        container_name="ctr-1",
        changes=[change],
        branch_name="feat/update",
        commit_message="Update main",
    )
    assert len(req.changes) == 1
    assert req.changes[0].file_path == "src/main.py"
    assert req.branch_name == "feat/update"

    resp = ApplyChangesResponse(
        status="success",
        message="Applied",
        branch_name="feat/update",
        commit_sha="abc123",
        changes_applied=1,
        fork_url="https://github.com/user/repo",
    )
    assert resp.changes_applied == 1
    assert resp.commit_sha == "abc123"


# ---------------------------------------------------------------------------
# 13. test_validation_models
# ---------------------------------------------------------------------------
def test_validation_models():
    """ValidatePRRequest, PRValidationResult, ValidatePRResponse should instantiate."""
    req = ValidatePRRequest(
        repo_owner="org",
        repo_name="repo",
        pr_number=10,
        validation_types=["security", "quality"],
    )
    assert req.pr_number == 10
    assert "security" in req.validation_types

    result = PRValidationResult(
        validation_type="security",
        passed=True,
        score=95.0,
        issues=[],
        suggestions=["Add more tests"],
    )
    assert result.passed is True
    assert result.score == 95.0

    resp = ValidatePRResponse(
        status="success",
        message="Validated",
        overall_score=90.0,
        passed=True,
        validations=[result],
        recommendation="approve",
    )
    assert resp.overall_score == 90.0
    assert len(resp.validations) == 1
    assert resp.recommendation == "approve"

    # Negative case: pr_number must be >= 1
    with pytest.raises(ValidationError):
        ValidatePRRequest(
            repo_owner="org",
            repo_name="repo",
            pr_number=0,
        )
