"""Unit test conftest — override async_client to load the real FastAPI app."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport


def _stub_missing_deps():
    """Stub out heavy dependencies that aren't installed in test env."""
    for mod_name in (
        "sqlalchemy",
        "sqlalchemy.ext",
        "sqlalchemy.ext.asyncio",
        # NOT opentelemetry. Those packages are declared dependencies of this
        # service and are installed by `uv sync`, so stubbing them replaced real
        # modules with empty ModuleTypes for the whole unit session. The effect
        # was that NO test in this repo could exercise real tracing -- a test
        # importing FastAPIInstrumentor got "cannot import name ... (unknown
        # location)" -- which is part of why a tracing block that exported zero
        # spans stayed green here for months.
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            # Provide minimal attributes so imports don't crash
            stub.__path__ = []
            sys.modules[mod_name] = stub

    # sqlalchemy specifics
    sa = sys.modules["sqlalchemy"]
    if not hasattr(sa, "text"):
        sa.text = lambda x: x

    sa_async = sys.modules["sqlalchemy.ext.asyncio"]
    if not hasattr(sa_async, "AsyncSession"):
        sa_async.AsyncSession = type("AsyncSession", (), {})
    if not hasattr(sa_async, "async_sessionmaker"):
        sa_async.async_sessionmaker = type(
            "async_sessionmaker", (), {"__class_getitem__": classmethod(lambda cls, x: cls)}
        )
    if not hasattr(sa_async, "create_async_engine"):
        sa_async.create_async_engine = lambda *a, **kw: None


@pytest.fixture(autouse=True)
def _disable_auto_index(monkeypatch):
    """Keep fire-and-forget scans from leaking out of unit tests unless a test opts in."""
    monkeypatch.setenv("AUTO_INDEX_ON_PROVISION", "false")


@pytest.fixture
def github_pr():
    """Patch PyGithub *inside the auto-closer* and hand back the fake PR object.

    Router tests must NOT mock `routers.pr_actions.PRAutoCloser` wholesale: a
    double that never touches GitHub cannot distinguish "the action happened"
    from "the action was merely configured", which is precisely the bug the
    `actions_taken` payload used to have. Patching only the GitHub client keeps
    the real PRAutoCloser in the loop, so the HTTP response is derived from real
    call outcomes and a raising `pr.*` method is a real failure.
    """
    with patch("services.pr_validation.actions.pr_auto_closer.Github") as mock_github_cls:
        pr = MagicMock()
        pr.number = 10
        pr.labels = []
        pr.get_commits.return_value = [MagicMock()]

        repo = MagicMock()
        repo.get_pull.return_value = pr

        client = MagicMock()
        client.get_repo.return_value = repo
        mock_github_cls.return_value = client

        pr.github_client = client
        yield pr


INTERNAL_APP_TOKEN = "ghs_internal_app_test"  # pragma: allowlist secret


@pytest.fixture
def internal_app(monkeypatch):
    """The internal GitHub App, configured and installed, minting a known token.

    Since 2026-09-30 there is no PAT fallback: a platform repo's credential is
    the internal App (``resolve_github_token``) or nothing. Suites that exercise
    the post-credential behaviour of a platform-repo route use this instead of
    the old ``GITHUB_ACCESS_TOKEN`` env var. Returns the token the App mints.
    """
    import services.github_app_auth as ga

    monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
    monkeypatch.setattr(ga, "get_installation_token_for", lambda _o, _r: INTERNAL_APP_TOKEN)
    monkeypatch.setattr(ga, "get_installation_token", lambda: INTERNAL_APP_TOKEN)
    return INTERNAL_APP_TOKEN


@pytest.fixture
async def async_client():
    """Async HTTP client that loads the real FastAPI app from src/main.py."""
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Stub out deps that might not be installed
    _stub_missing_deps()

    # Force-load the real main module from src/main.py
    main_mod = sys.modules.get("main")
    if main_mod is None or not hasattr(main_mod, "app"):
        spec = importlib.util.spec_from_file_location("main", Path(src_dir) / "main.py")
        real_main = importlib.util.module_from_spec(spec)
        # Preserve existing stub attributes (get_security_client, get_workspaces_client)
        old_main = sys.modules.get("main")
        sys.modules["main"] = real_main
        try:
            spec.loader.exec_module(real_main)
        except Exception:
            # If loading fails, restore old module and re-raise
            if old_main is not None:
                sys.modules["main"] = old_main
            raise
        # Copy over stub functions that tests may have set
        if old_main is not None:
            for attr in ("get_security_client", "get_workspaces_client"):
                if hasattr(old_main, attr) and not hasattr(real_main, attr):
                    setattr(real_main, attr, getattr(old_main, attr))
        app = real_main.app
    else:
        app = main_mod.app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
