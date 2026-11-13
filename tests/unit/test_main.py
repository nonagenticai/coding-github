"""Tests for main.py - FastAPI app, lifespan, exception handlers, endpoints.

Verifies: resource accessor guards, lifespan startup/shutdown with DB/client
init failures, DB URL normalization, health/root endpoints, exception handlers,
and router registration.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# Ensure stubs have all exceptions used by routers/utils
# ---------------------------------------------------------------------------


def _ensure_exception_stubs():
    """Add missing exception classes to the coding_client stubs."""
    exc_mod = sys.modules.get("coding_client.exceptions")
    if exc_mod is None:
        return

    for name in (
        "AuthenticationException",
        "RateLimitException",
        "ServiceUnavailableException",
        "ValidationException",
    ):
        if not hasattr(exc_mod, name):
            cls = type(name, (Exception,), {})
            setattr(exc_mod, name, cls)

    api_exc_mod = sys.modules.get("coding_client.exceptions.api_exceptions")
    if api_exc_mod:
        for name in (
            "AuthenticationException",
            "RateLimitException",
            "ServiceUnavailableException",
        ):
            if not hasattr(api_exc_mod, name):
                setattr(api_exc_mod, name, getattr(exc_mod, name))


_ensure_exception_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_main():
    """Re-import main module to get fresh globals."""
    _ensure_exception_stubs()
    for mod_name in list(sys.modules):
        if mod_name == "main":
            del sys.modules[mod_name]
    for mod_name in list(sys.modules):
        if mod_name.startswith("routers."):
            del sys.modules[mod_name]
    if "routers" in sys.modules:
        del sys.modules["routers"]

    import main

    return main


def _reset_main_globals(main_mod):
    """Reset module-level resources to None so lifespan starts clean."""
    main_mod._db_engine = None
    main_mod._db_session_factory = None
    main_mod._workspaces_client = None
    main_mod._security_client = None


def _lifespan_settings():
    """Settings object with all fields needed by lifespan."""
    from types import SimpleNamespace

    return SimpleNamespace(
        database_url="postgresql+psycopg://test:test@localhost:5432/test",
        # No CI-wait store: the wait manager stays off (main.lifespan reads it).
        ci_wait_dsn="",
        coding_containers_url="http://coding-containers:8691",
        coding_security_url="http://coding-security:8693",
    )


def _mock_engine():
    """Create a mock async DB engine with working connect() context manager."""
    engine = AsyncMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    engine.connect = MagicMock(return_value=conn)
    engine.dispose = AsyncMock()
    return engine


# ---------------------------------------------------------------------------
# Resource accessors raise before lifespan
# ---------------------------------------------------------------------------


class TestResourceAccessorsBeforeStartup:
    """Accessor functions raise RuntimeError before lifespan runs."""

    @pytest.mark.parametrize(
        ("attr", "func_name", "match"),
        [
            ("_db_session_factory", "get_db_session_factory", "DB not initialised"),
            ("_workspaces_client", "get_workspaces_client", "Containers client not initialised"),
            ("_security_client", "get_security_client", "Security client not initialised"),
        ],
    )
    def test_accessor_raises_before_startup(self, attr, func_name, match):
        main = _fresh_main()
        setattr(main, attr, None)
        with pytest.raises(RuntimeError, match=match):
            getattr(main, func_name)()


# ---------------------------------------------------------------------------
# Lifespan tests
# ---------------------------------------------------------------------------


class TestLifespanStartup:
    async def test_all_resources_initialised_on_success(self):
        """All resources are set during successful lifespan startup."""
        main = _fresh_main()

        _reset_main_globals(main)

        mock_engine = _mock_engine()
        mock_session_factory = MagicMock()
        mock_containers = AsyncMock()
        mock_containers.close = AsyncMock()
        mock_security = AsyncMock()
        mock_security.close = AsyncMock()

        with (
            patch("main.settings", _lifespan_settings()),
            patch("main.create_async_engine", return_value=mock_engine),
            patch("main.async_sessionmaker", return_value=mock_session_factory),
            patch(
                "coding_client.clients.CodingWorkspacesClient",
                return_value=mock_containers,
            ),
            patch(
                "coding_client.clients.coding_security.CodingSecurityClient",
                return_value=mock_security,
            ),
        ):
            app = MagicMock()
            async with main.lifespan(app):
                # No _redis assertion: the vestigial Redis resource is gone.
                assert not hasattr(main, "_redis")
                assert main._db_session_factory is mock_session_factory
                assert main._workspaces_client is mock_containers
                assert main._security_client is mock_security

        _reset_main_globals(main)

    async def test_db_failure_is_non_fatal(self):
        """A database failure during startup is non-fatal; the service continues.

        Was parametrised over (db, redis). The redis case is gone with the resource:
        nothing in `src` ever used it, so there was no startup step left to fail.
        """
        main = _fresh_main()

        _reset_main_globals(main)

        mock_session_factory = MagicMock()
        mock_containers = AsyncMock()
        mock_containers.close = AsyncMock()
        mock_security = AsyncMock()
        mock_security.close = AsyncMock()

        with (
            patch("main.settings", _lifespan_settings()),
            patch("main.create_async_engine", side_effect=Exception("db down")),
            patch("main.async_sessionmaker", return_value=mock_session_factory),
            patch(
                "coding_client.clients.CodingWorkspacesClient",
                return_value=mock_containers,
            ),
            patch(
                "coding_client.clients.coding_security.CodingSecurityClient",
                return_value=mock_security,
            ),
        ):
            app = MagicMock()
            async with main.lifespan(app):
                assert main._db_engine is None

        _reset_main_globals(main)

    async def test_containers_failure_continues(self):
        """Containers client init failure is non-fatal."""
        main = _fresh_main()

        _reset_main_globals(main)

        mock_engine = _mock_engine()
        mock_session_factory = MagicMock()
        mock_security = AsyncMock()
        mock_security.close = AsyncMock()

        with (
            patch("main.settings", _lifespan_settings()),
            patch("main.create_async_engine", return_value=mock_engine),
            patch("main.async_sessionmaker", return_value=mock_session_factory),
            patch(
                "coding_client.clients.CodingWorkspacesClient",
                side_effect=Exception("containers down"),
            ),
            patch(
                "coding_client.clients.coding_security.CodingSecurityClient",
                return_value=mock_security,
            ),
        ):
            app = MagicMock()
            async with main.lifespan(app):
                assert main._workspaces_client is None
                assert main._security_client is mock_security

        _reset_main_globals(main)


class TestDbUrlNormalization:
    async def test_postgresql_rewritten_to_psycopg(self):
        """postgresql:// should be rewritten to postgresql+psycopg://."""
        main = _fresh_main()

        _reset_main_globals(main)

        mock_engine = _mock_engine()
        mock_containers = AsyncMock()
        mock_containers.close = AsyncMock()
        mock_security = AsyncMock()
        mock_security.close = AsyncMock()

        orig_settings = main.settings
        main.settings = MagicMock()
        main.settings.database_url = "postgresql://user:pass@host:5432/db"
        main.settings.ci_wait_dsn = ""
        main.settings.coding_containers_url = "http://containers:8691"
        main.settings.coding_security_url = "http://security:8693"

        captured_url = None

        def capture_create_engine(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_engine

        try:
            with (
                patch("main.create_async_engine", side_effect=capture_create_engine),
                patch("main.async_sessionmaker", return_value=MagicMock()),
                patch(
                    "coding_client.clients.CodingWorkspacesClient",
                    return_value=mock_containers,
                ),
                patch(
                    "coding_client.clients.coding_security.CodingSecurityClient",
                    return_value=mock_security,
                ),
            ):
                app = MagicMock()
                async with main.lifespan(app):
                    pass

            assert captured_url is not None
            assert captured_url.startswith("postgresql+psycopg://")
        finally:
            main.settings = orig_settings
            _reset_main_globals(main)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    async def test_global_exception_returns_500(self):
        """Unhandled ValueError yields 500 with KAG-standard error envelope."""
        main = _fresh_main()

        @main.app.get("/test-500-global")
        async def _raise():
            raise ValueError("boom")

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-500-global")

        assert resp.status_code == 500
        body = resp.json()
        error = body["error"]
        assert error["code"] == "INTERNAL_SERVER_ERROR"
        assert error["message"] == "An unexpected error occurred"

    async def test_validation_error_returns_422(self):
        """Pydantic validation error yields 422 with KAG-standard error envelope."""
        main = _fresh_main()
        from pydantic import BaseModel

        class Payload(BaseModel):
            name: str
            count: int

        @main.app.post("/test-validation-handler")
        async def _validate(payload: Payload):
            return {"ok": True}

        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/test-validation-handler", json={"name": 123})

        assert resp.status_code == 422
        body = resp.json()
        error = body["error"]
        assert error["code"] == "HTTP_422"
        assert error["message"] == "Request validation failed"
        assert "details" in error
        assert len(error["details"]) > 0
        err = error["details"][0]
        assert "field" in err
        assert "message" in err
        assert "type" in err


# ---------------------------------------------------------------------------
# Health and root endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Health must follow a REAL local resource, not an unused one.

    These tests previously pinned the opposite contract — `degraded` whenever a
    Redis ping failed, including with the database up. Redis had no caller in
    `src` and no Service was ever deployed, so that contract made the endpoint
    permanently and uninformatively `degraded` while hiding the one resource that
    matters. They are rewritten, not deleted: the endpoint still reports, it just
    reports something true.
    """

    async def test_health_ok_when_the_database_is_connected(self):
        main = _fresh_main()
        main._db_engine = MagicMock()

        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["db"] == "connected"

    async def test_health_degraded_when_the_database_is_absent(self):
        """The case the old Redis-driven verdict could never report."""
        main = _fresh_main()
        main._db_engine = None

        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["db"] == "disconnected"

    async def test_health_no_longer_reports_a_redis_it_does_not_use(self):
        """No phantom dependency in the payload, and no accessor left behind."""
        main = _fresh_main()
        main._db_engine = MagicMock()

        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert "redis" not in resp.json()
        assert not hasattr(main, "get_redis")
        assert not hasattr(main, "_redis")


class TestStartupBannerReadsPrefixedSettings:
    """The startup banner must report the environment the service was CONFIGURED with.

    ``Settings`` carries ``env_prefix = "CODING_GITHUB_"``, so the deployed
    knob is ``CODING_GITHUB_ENVIRONMENT`` -- which is exactly what the CI smoke
    in ``.github/workflows/ci-cd.yaml`` sets. Reading a bare ``ENVIRONMENT``
    instead meant the banner fell through to the ``"production"`` default and
    every non-production boot logged ``(production)``.

    ``ENABLE_DOCS`` / ``ENABLE_AUTH`` / ``VERSION`` are deliberately NOT
    converted -- see ``TestDeliberatelyUnprefixedEnvVars`` below.
    """

    def test_environment_is_read_from_the_prefixed_setting(self):
        from config.settings import get_settings

        get_settings.cache_clear()
        try:
            # Prefixed says "staging", unprefixed says "production". The banner
            # must follow the prefixed one -- the only one gitops/CI can set.
            with patch.dict(
                "os.environ",
                {"CODING_GITHUB_ENVIRONMENT": "staging", "ENVIRONMENT": "production"},
            ):
                main = _fresh_main()
                assert main.ENVIRONMENT == "staging"
        finally:
            get_settings.cache_clear()
            _fresh_main()

    def test_service_name_is_read_from_the_prefixed_setting(self):
        from config.settings import get_settings

        get_settings.cache_clear()
        try:
            with patch.dict(
                "os.environ",
                {
                    "CODING_GITHUB_SERVICE_NAME": "coding-github-canary",
                    "SERVICE_NAME": "wrong-unprefixed-name",
                },
            ):
                main = _fresh_main()
                assert main.SERVICE_NAME == "coding-github-canary"
        finally:
            get_settings.cache_clear()
            _fresh_main()


class TestDeliberatelyUnprefixedEnvVars:
    """These three stay unprefixed ON PURPOSE -- do not "fix" them.

    * ``ENABLE_DOCS`` is set unprefixed in the live Deployment
      (``gitops-production/clusters/strato/services/coding-github/deployment.yaml``
      -> ``ENABLE_DOCS: "true"``). Reading a prefixed name instead would fall
      back to the ``"false"`` default and silently kill /docs in production.
    * ``ENABLE_AUTH`` is set unprefixed by the CI smoke
      (``docker run -e ENABLE_AUTH=false``) and by docker-compose. Switching it
      would turn auth back ON in the smoke and break the /health probe.
    * ``VERSION`` has no ``Settings`` field at all, so there is nothing
      prefixed to read.

    None of the three has a prefixed ``Settings`` equivalent, so converting
    them would be a behaviour change, not a bug fix.
    """

    def test_enable_docs_still_follows_the_unprefixed_env_var(self):
        with patch.dict("os.environ", {"ENABLE_DOCS": "true"}):
            main = _fresh_main()
            assert main.ENABLE_DOCS is True
        with patch.dict("os.environ", {"ENABLE_DOCS": "false"}):
            main = _fresh_main()
            assert main.ENABLE_DOCS is False

    def test_enable_auth_still_follows_the_unprefixed_env_var(self):
        with patch.dict("os.environ", {"ENABLE_AUTH": "false"}):
            main = _fresh_main()
            assert main.ENABLE_AUTH is False
        with patch.dict("os.environ", {"ENABLE_AUTH": "true"}):
            main = _fresh_main()
            assert main.ENABLE_AUTH is True

    def test_version_has_no_prefixed_settings_equivalent(self):
        from config.settings import Settings

        assert "version" not in Settings.model_fields
        with patch.dict("os.environ", {"VERSION": "9.9.9"}):
            main = _fresh_main()
            assert main.VERSION == "9.9.9"


class TestRootEndpoint:
    async def test_root_returns_service_info(self):
        """Root endpoint returns service name, version, docs and health links."""
        main = _fresh_main()

        transport = ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")

        assert resp.status_code == 200
        body = resp.json()
        assert "service" in body
        assert "version" in body
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def _route_paths(routes) -> list[str]:
    """All effective paths reachable from ``routes``.

    fastapi>=0.137 stopped flattening ``include_router()`` sub-routes into
    ``app.routes``: they are wrapped in a private ``_IncludedRouter`` that has
    no ``.path`` (https://github.com/fastapi/fastapi/releases/tag/0.137.0), so
    the old ``[r.path for r in app.routes]`` raises
    ``AttributeError: '_IncludedRouter' object has no attribute 'path'``.
    ``fastapi.routing.iter_route_contexts`` is the public accessor (added
    0.137.2) that resolves either shape to a flat sequence of contexts
    carrying the real, fully-prefixed ``.path``.
    """
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:  # fastapi < 0.137.2: app.routes is already flat
        return [r.path for r in routes if hasattr(r, "path")]
    return [ctx.path for ctx in iter_route_contexts(routes) if ctx.path is not None]


class TestRoutersRegistered:
    def test_github_and_pr_actions_routers_registered(self):
        """Both github and pr-actions routers are mounted."""
        main = _fresh_main()
        routes = _route_paths(main.app.routes)
        has_github = any("/api/repositories" in p for p in routes)
        has_pr_actions = any("/api/webhooks/pr-actions" in p for p in routes)
        assert has_github, f"github_router not found in routes: {routes}"
        assert has_pr_actions, f"pr_actions_router not found in routes: {routes}"


# ---------------------------------------------------------------------------
# Auth middleware installation must fail CLOSED
# ---------------------------------------------------------------------------


class TestInstallAuthMiddlewareFailsClosed:
    """A service that cannot load its auth middleware must not serve.

    The previous code caught ImportError, logged a warning, and continued with
    every route unauthenticated. coding-client has already moved a public name
    out of auth.tenant once (TRUSTED_NETWORKS, 1.7.0) -- exactly the shape of
    change that lands on this branch.
    """

    def test_import_error_raises_instead_of_serving_unauthenticated(self):
        import builtins

        from fastapi import FastAPI

        from main import install_auth_middleware

        real_import = builtins.__import__

        def _fail_auth_import(name, *args, **kwargs):
            if name in ("coding_client.auth.middleware", "coding_client.auth.tenant"):
                raise ImportError(f"simulated: cannot import {name}")
            return real_import(name, *args, **kwargs)

        target = FastAPI()
        with patch.object(builtins, "__import__", _fail_auth_import):
            with pytest.raises(RuntimeError, match="refusing to start unauthenticated"):
                install_auth_middleware(target)

        assert not any(
            m.cls.__name__ in ("AuthMiddleware", "TenantMiddleware") for m in target.user_middleware
        )

    def test_healthy_import_attaches_both_middlewares(self):
        """The success path still wires both -- so the test above cannot pass vacuously."""
        from fastapi import FastAPI

        from main import install_auth_middleware

        target = FastAPI()
        install_auth_middleware(target)

        from coding_client.auth.middleware import AuthMiddleware
        from coding_client.auth.tenant import TenantMiddleware

        attached = [m.cls for m in target.user_middleware]
        # AuthMiddleware is attached as its webhook-exempting subclass; the
        # exemption itself is pinned in test_webhook_auth_exemption.py.
        assert any(isinstance(c, type) and issubclass(c, AuthMiddleware) for c in attached)
        assert TenantMiddleware in attached
