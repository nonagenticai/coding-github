"""Extended tests for main.py -- covers lines missed by test_main.py.

Targets: accessor success paths, security client failure, OpenTelemetry init,
ENABLE_AUTH middleware, HTTPException handler branches (custom attrs, dict, non-str).
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_exception_stubs():
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
            setattr(exc_mod, name, type(name, (Exception,), {}))
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


def _fresh_main():
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
    main_mod._db_engine = None
    main_mod._db_session_factory = None
    main_mod._workspaces_client = None
    main_mod._security_client = None


def _lifespan_settings():
    from types import SimpleNamespace

    return SimpleNamespace(
        database_url="postgresql+psycopg://test:test@localhost:5432/test",
        # No CI-wait store: the wait manager stays off (main.lifespan reads it).
        ci_wait_dsn="",
        coding_containers_url="http://coding-containers:8691",
        coding_security_url="http://coding-security:8693",
    )


def _mock_engine():
    engine = AsyncMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    engine.connect = MagicMock(return_value=conn)
    engine.dispose = AsyncMock()
    return engine


# ===================================================================
# Accessor success paths
# ===================================================================


class TestAccessorSuccessPaths:
    """When resources are initialised, accessors return them."""

    @pytest.mark.parametrize(
        ("attr", "func_name"),
        [
            ("_db_session_factory", "get_db_session_factory"),
            ("_workspaces_client", "get_workspaces_client"),
            ("_security_client", "get_security_client"),
        ],
    )
    def test_accessor_returns_instance(self, attr, func_name):
        main = _fresh_main()
        sentinel = MagicMock()
        setattr(main, attr, sentinel)
        assert getattr(main, func_name)() is sentinel


# ===================================================================
# Security client init failure in lifespan
# ===================================================================


class TestLifespanSecurityClientFailure:
    async def test_security_client_failure_sets_none(self):
        """Security client init failure is non-fatal."""
        main = _fresh_main()

        _reset_main_globals(main)

        mock_engine = _mock_engine()
        mock_session_factory = MagicMock()
        mock_containers = AsyncMock()
        mock_containers.close = AsyncMock()

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
                side_effect=Exception("security init boom"),
            ),
        ):
            app = MagicMock()
            async with main.lifespan(app):
                assert main._security_client is None
                assert main._workspaces_client is mock_containers

        _reset_main_globals(main)


# ===================================================================
# OpenTelemetry init
# ===================================================================


class TestLifespanDoesNotInitialiseTracing:
    """Replaces TestOtelTracingEnabled / TestOtelImportError.

    Those two tests asserted the behaviour of a lifespan block that has been
    removed, and they were green for months while this service exported zero
    spans to Tempo. Both reasons are worth keeping on the record:

    - they passed ``app = MagicMock()`` and asserted
      ``instrument_app.assert_called_once_with(app)`` — i.e. that the CALL
      happened. That call is a silent no-op on a real Starlette app once the
      middleware stack is built, so asserting it was made proved nothing about
      whether a span was ever produced. A fake narrower than the real thing
      cannot fail the way the real thing does.
    - they set ``OTEL_EXPORTER_OTLP_ENDPOINT``, which this cluster does not
      provision (it sets ``OTLP_TRACES_ENDPOINT``), so they pinned the
      unprovisioned name as if it were correct.

    They also swapped ``opentelemetry.*`` entries in ``sys.modules`` for
    MagicMocks and a deliberately broken stub. The restore left global state
    poisoned, so the first test in this repo to import OpenTelemetry for real
    afterwards got an ImportError — which is how this was found.

    Tracing now lives in ``coding_client.telemetry.setup_tracing``, called at
    module scope; ``tests/unit/test_tracing_wiring.py`` asserts the call site by
    AST and drives a real request that produces a real span. What is left to
    assert here is the inverse: the lifespan must never instrument anything.
    """

    async def test_lifespan_never_instruments_the_app(self):
        """The removed block must not come back via the lifespan.

        Patches the real ``FastAPIInstrumentor`` rather than replacing modules in
        ``sys.modules``, so nothing global is left poisoned for later tests.
        """
        main = _fresh_main()
        _reset_main_globals(main)

        mock_engine = _mock_engine()
        mock_session_factory = MagicMock()
        mock_containers = AsyncMock()
        mock_containers.close = AsyncMock()
        mock_security = AsyncMock()
        mock_security.close = AsyncMock()

        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

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
            patch.object(FastAPIInstrumentor, "instrument_app") as spy,
            # Both endpoint variables set, so a restored block would fire on either.
            patch.dict(
                "os.environ",
                {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4318",
                    "OTLP_TRACES_ENDPOINT": "http://otel:4318/v1/traces",
                },
            ),
        ):
            app = MagicMock()
            async with main.lifespan(app):
                pass

        assert spy.call_count == 0, (
            "the lifespan instrumented the app — instrument_app() is a silent no-op "
            "there, so this would ship zero spans while logging success"
        )
        _reset_main_globals(main)


# ===================================================================
# ENABLE_AUTH middleware
# ===================================================================


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


class TestEnableAuthMiddleware:
    def test_auth_middleware_loaded_without_crash(self):
        """Auth middleware loads without crash when ENABLE_AUTH=true."""
        with patch.dict("os.environ", {"ENABLE_AUTH": "true"}):
            main = _fresh_main()
        routes = _route_paths(main.app.routes)
        assert any("/health" in p for p in routes)


# ===================================================================
# HTTPException handler branches
# ===================================================================


class TestHttpExceptionHandlerBranches:
    async def test_custom_error_code_and_details(self):
        """Exception with error_code attr still produces KAG-standard envelope."""
        main = _fresh_main()
        from fastapi import HTTPException

        class CustomHTTPException(HTTPException):
            def __init__(self):
                super().__init__(status_code=409, detail="conflict happened")
                self.error_code = "CONFLICT"
                self.error_details = {"resource": "repo", "id": "123"}

        @main.app.get("/test-custom-attrs")
        async def _raise():
            raise CustomHTTPException()

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-custom-attrs")

        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "HTTP_409"
        assert error["message"] == "conflict happened"

    async def test_dict_detail_with_error_code(self):
        """Exception with dict detail extracts message in KAG envelope."""
        main = _fresh_main()
        from fastapi import HTTPException

        @main.app.get("/test-dict-detail")
        async def _raise():
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "bad input",
                    "error_code": "INPUT_INVALID",
                    "details": {"field": "name"},
                },
            )

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-dict-detail")

        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["message"] == "bad input"
        assert error["code"] == "HTTP_400"
        assert error["details"] == {"field": "name"}

    async def test_dict_detail_without_error_code_falls_back(self):
        """Dict detail without error_code uses HTTP_<status> code."""
        main = _fresh_main()
        from fastapi import HTTPException

        @main.app.get("/test-dict-no-code")
        async def _raise():
            raise HTTPException(
                status_code=404,
                detail={"message": "gone", "details": {}},
            )

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-dict-no-code")

        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["message"] == "gone"
        assert error["code"] == "HTTP_404"

    async def test_string_detail(self):
        """String detail uses HTTP_<status> code in KAG envelope."""
        main = _fresh_main()
        from fastapi import HTTPException

        @main.app.get("/test-str-detail")
        async def _raise():
            raise HTTPException(status_code=403, detail="forbidden action")

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-str-detail")

        assert resp.status_code == 403
        error = resp.json()["error"]
        assert error["message"] == "forbidden action"
        assert error["code"] == "HTTP_403"

    async def test_non_string_non_dict_detail(self):
        """Non-str, non-dict detail (e.g., int) is stringified."""
        main = _fresh_main()
        from fastapi import HTTPException

        @main.app.get("/test-other-detail")
        async def _raise():
            raise HTTPException(status_code=500, detail=12345)

        transport = ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-other-detail")

        assert resp.status_code == 500
        error = resp.json()["error"]
        assert error["message"] == "12345"
        assert error["code"] == "HTTP_500"
