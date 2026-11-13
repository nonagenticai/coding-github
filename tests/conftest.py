"""Shared test fixtures for coding-github.

Uses real coding_client imports (no sys.modules stubs).
Auth middleware is disabled via ENABLE_AUTH=false env var.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# 1. Add src/ to Python path so tests can import service-local modules
#    (routers, services, config, etc.)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# 2. Set test environment vars BEFORE any service imports.
#    - ENABLE_AUTH=false disables AuthMiddleware in main.py
#    - CODING_GITHUB_* vars override pydantic-settings defaults
# ---------------------------------------------------------------------------
os.environ["ENVIRONMENT"] = "test"
os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["CODING_GITHUB_ENVIRONMENT"] = "test"
os.environ["CODING_GITHUB_DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
# ⛔ The two PAT vars below map to NO setting any more (the PAT fallbacks were
# removed 2026-09-30). They stay set ON PURPOSE: a deployment that still carries
# them must boot (``extra: ignore``) and must never turn them into a credential —
# tests/unit/test_no_pat_fallback.py relies on them being present.
os.environ["CODING_GITHUB_GITHUB_TOKEN"] = "test-token"
os.environ["CODING_GITHUB_GITHUB_WEBHOOK_SECRET"] = "test-secret"
os.environ["CODING_GITHUB_SYSTEM_GITHUB_TOKEN"] = "test-token"
os.environ["CODING_GITHUB_SYSTEM_GITHUB_OWNER"] = "test-owner"
os.environ["CODING_GITHUB_GITHUB_APP_ID"] = ""
os.environ["CODING_GITHUB_GITHUB_APP_PRIVATE_KEY"] = ""
# fr's base URL. The andon brake (services.andon) refuses every mutating GitHub
# operation when this is unset — fail closed — so a value has to exist even though
# `_andon_released` below means no test actually dials it.
os.environ["CODING_GITHUB_FR_URL"] = "http://fr-backend.fr.svc:8888"

# Clear lru_cache so Settings() picks up the env vars above
from config.settings import get_settings

get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _neutralise_tenant_auth():
    """Resolve a test tenant for every route that depends on one.

    Replaces the `os.environ["SKIP_AUTH"] = "true"` that used to sit at module
    scope above. SKIP_AUTH was removed from coding-client in 1.16.0.

    ⚠️ Unlike coding-analysis, where the equivalent line turned out to be inert,
    this one is LOAD-BEARING here: deleting it and running bare gave
    `138 failed, 2922 passed`, all of them a real `401 != 200`. ENABLE_AUTH=false
    stops AuthMiddleware being registered, but it does not satisfy the
    `Depends(require_tenant)` on the routers themselves — that is what SKIP_AUTH
    was quietly doing.

    autouse because `main.app` is a module-level singleton that test modules
    import DIRECTLY, not only through the `client`/`async_client` fixtures. A
    per-fixture override would leave those files 401ing.

    ⚠️ FUNCTION scope, deliberately, and session scope does NOT work here.
    `test_github_repository_management_router.py:59` and
    `..._extended.py:56` end their teardown with
    `app.dependency_overrides.clear()`, which wipes every override on the shared
    singleton — including this one — for all tests that run afterwards. With
    session scope that showed up as 90 failures whose individual runs passed,
    the signature of an ordering bug rather than a real one. Re-installing per
    test is a dict write and immune to whatever any other teardown does.

    Uses the SDK's helper rather than a hand-rolled `dependency_overrides` dict
    so the key set cannot drift: it neutralises every identity dependency
    coding-client exposes, from one list, and gains any future one on the next
    pin bump. The alternative is how coding-testing's integration conftest ended
    up overriding three dependencies and missing `require_tenant`.
    """
    from coding_client.testing import override_tenant_auth

    from main import app

    undo = override_tenant_auth(app)
    yield
    undo()


@pytest.fixture(autouse=True)
def _andon_released(request, monkeypatch):
    """Suite-wide default posture: the andon cord is RELEASED.

    ``services.andon`` gates every MUTATING GitHub route (merge, PR create,
    apply-changes, fork, the four issue writes, the two pr-actions routes, the
    webhook receiver, push-scope token minting). It fails CLOSED, so without this
    every one of those pre-existing route tests would refuse with 503
    ``andon_check_failed`` — the guard would be "testing" nothing except its own
    inability to reach fr.

    ⚠️ It replaces exactly ONE function, ``read_andon_state`` — the HTTP read.
    ``assert_andon_released`` (the exemption allow-list, the refusal counter, the
    log line) still runs for real on every gated route in the whole suite.

    ⛔ It is NOT a way to disarm the brake. ``tests/unit/test_andon_chokepoint.py``
    carries ``@pytest.mark.andon_live``, which opts out of this fixture and
    exercises the real read path including every fail-closed branch. Deleting that
    marker would make those tests vacuous, which is why they assert on the HTTP
    calls they cause.
    """
    if request.node.get_closest_marker("andon_live"):
        return

    import services.andon as andon

    async def _released(operation: str) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(andon, "read_andon_state", _released)


# fr's EXECUTION tenant — a member of the default CODING_GITHUB_PLATFORM_TENANTS.
PLATFORM_TEST_TENANT = "0195f48b-d000-7000-8000-000000000003"


@pytest.fixture
def platform_tenant():
    """Resolve routes to a PLATFORM tenant (fr EXECUTION) instead of "test-tenant".

    Platform-owned repos are refused (403 ``platform_repo_forbidden``) to any
    tenant outside ``CODING_GITHUB_PLATFORM_TENANTS``; suites that exercise the
    platform-owner path as the conveyor does must run as a platform tenant.
    """
    from coding_client.testing import override_tenant_auth

    from main import app

    undo = override_tenant_auth(app, tenant_id=PLATFORM_TEST_TENANT)
    yield PLATFORM_TEST_TENANT
    undo()


@pytest.fixture
def enforce_platform_tenants(monkeypatch):
    """CODING_GITHUB_PLATFORM_TENANT_ENFORCE=true (default is observe-only)."""
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "platform_tenant_enforce", True)
    return True


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_github_client():
    """Mock PyGithub client."""
    client = MagicMock()
    repo = MagicMock()
    repo.full_name = "test-org/test-repo"
    repo.default_branch = "main"
    repo.clone_url = "https://github.com/test-org/test-repo.git"
    client.get_repo.return_value = repo

    user = MagicMock()
    user.login = "test-user"
    client.get_user.return_value = user

    return client


@pytest.fixture
async def async_client():
    """Async HTTP client for testing FastAPI endpoints."""
    from main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for inter-service calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def sample_webhook_payload():
    """Sample GitHub webhook push event payload."""
    return {
        "ref": "refs/heads/main",
        "repository": {
            "full_name": "test-org/test-repo",
            "name": "test-repo",
            "owner": {"login": "test-org"},
        },
        "sender": {"login": "test-user"},
        "commits": [],
    }


@pytest.fixture
def sample_pr_payload():
    """Sample GitHub PR event payload."""
    return {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "number": 1,
            "title": "Test PR",
            "body": "Test body",
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main"},
            "user": {"login": "test-user"},
        },
        "repository": {
            "full_name": "test-org/test-repo",
            "name": "test-repo",
        },
    }


@pytest.fixture
def sample_pr_review_payload():
    """Sample GitHub PR review event payload."""
    return {
        "action": "submitted",
        "review": {
            "state": "approved",
            "user": {"login": "reviewer-user"},
            "body": "LGTM",
        },
        "pull_request": {
            "number": 1,
            "title": "Test PR",
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main"},
        },
        "repository": {
            "full_name": "test-org/test-repo",
            "name": "test-repo",
        },
    }


@pytest.fixture
def sample_issue_comment_payload():
    """Sample GitHub issue comment event payload."""
    return {
        "action": "created",
        "issue": {
            "number": 1,
            "title": "Test Issue",
            "user": {"login": "test-user"},
        },
        "comment": {
            "body": "/run-tests",
            "user": {"login": "commenter"},
        },
        "repository": {
            "full_name": "test-org/test-repo",
            "name": "test-repo",
        },
    }


@pytest.fixture
def sample_file_changes():
    """Sample file changes for PR creation."""
    return [
        {"file_path": "src/main.py", "content": "print('Hello, World!')"},
        {"file_path": "README.md", "content": "# Test Project"},
        {"file_path": "tests/test_main.py", "content": "def test_main(): pass"},
    ]


@pytest.fixture
def mock_workspaces_client():
    """Mock CodingWorkspacesClient with the typed workspace methods."""
    client = AsyncMock()
    client.list_workspaces = AsyncMock(return_value=[])
    client.create_workspace = AsyncMock(return_value={"id": "ctr-1", "name": "test-container"})
    client.get_workspace = AsyncMock(return_value={})
    client.delete_workspace = AsyncMock(return_value=None)
    client.get_status = AsyncMock(return_value={})
    client.list_files = AsyncMock(return_value={})
    client.exec = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_security_client():
    """Mock CodingSecurityClient with standard methods."""
    client = AsyncMock()
    client.validate = AsyncMock(return_value={"valid": True})
    client.scan = AsyncMock(return_value={"issues": []})
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_token_manager():
    """Mock token manager for GitHub token operations."""
    mgr = AsyncMock()
    mgr.get_user_token = AsyncMock(return_value="ghp_user_test_token")
    return mgr


@pytest.fixture
def mock_settings():
    """Reusable test settings matching all Settings fields."""
    return SimpleNamespace(
        service_name="coding-github",
        environment="test",
        debug=False,
        database_url="postgresql+psycopg://test:test@localhost:5432/test",
        github_app_id="",
        github_app_private_key="",
        github_webhook_secret="test-webhook-secret",
        system_github_owner="test-owner",
        coding_gateway_url="http://coding-gateway:8687",
        coding_containers_url="http://coding-containers:8691",
        coding_security_url="http://coding-security:8693",
        coding_analysis_url="http://coding-analysis:8688",
        secrets_api_url="https://secrets-api:8543",
        ca_cert_path="/certs/ca-cert.pem",
    )


@pytest.fixture
def tenant_header():
    """Standard tenant header for request tests."""
    return {"X-Tenant-ID": "test-tenant"}


# ---------------------------------------------------------------------------
# Marker normalisation.
#
# The shared local-ci pre-push hook adds `-m unit` whenever ANY test in the
# repo carries the `unit` marker. Only 5 of ~95 test files did (254 of 3028
# tests), so the hook deselected 2774 tests while pyproject's
# `--cov-fail-under=85` still measured the whole of src/ — 28.24% coverage,
# exit 1, PUSH BLOCKED on a pristine main. The gate was unpassable and could
# only be got past with `--no-verify`, i.e. it ran nothing.
#
# Marking every non-integration/non-e2e test `unit` at collection time makes
# `-m unit` select exactly what CI's `-m 'not integration'` selects, so the
# local gate becomes as strict as CI instead of impossible.
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Tag every non-integration/non-e2e test as `unit`."""
    for item in items:
        names = {m.name for m in item.iter_markers()}
        if not names & {"integration", "e2e"}:
            item.add_marker(pytest.mark.unit)
