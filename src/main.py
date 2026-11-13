"""FastAPI application entry point for coding-github."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from coding_client.api.exception_handlers import register_exception_handlers
from coding_client.api.logging_setup import configure_logging
from coding_client.api.openapi_config import configure_openapi
from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware
from coding_client.telemetry import setup_tracing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings
from routers.github import router as github_router
from routers.github_internal import router as github_internal_router
from routers.github_repo_file_read import router as repo_file_read_router
from routers.pr_actions import router as pr_actions_router

configure_logging()

logger = logging.getLogger(__name__)

settings = get_settings()

# ``Settings`` carries env_prefix="CODING_GITHUB_", so these two are configured
# as CODING_GITHUB_SERVICE_NAME / CODING_GITHUB_ENVIRONMENT. Reading the bare
# names meant the startup banner ignored the prefixed value and fell through to
# the "production" default -- the CI smoke sets CODING_GITHUB_ENVIRONMENT=testing
# and still logged "(production)".
SERVICE_NAME = settings.service_name
ENVIRONMENT = settings.environment
# The three below stay on bare env vars ON PURPOSE -- none has a Settings field,
# and each is genuinely set unprefixed where it matters:
#   VERSION      -- no Settings equivalent exists.
#   ENABLE_DOCS  -- set unprefixed in the live Deployment (gitops-production
#                   clusters/strato/services/coding-github/deployment.yaml).
#   ENABLE_AUTH  -- set unprefixed by the CI smoke (docker run -e ENABLE_AUTH=false)
#                   and by docker-compose.
VERSION = os.getenv("VERSION", "1.0.0")
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Module-level resources (initialised in lifespan)
# ---------------------------------------------------------------------------
_db_engine = None
_db_session_factory: async_sessionmaker[AsyncSession] | None = None
_workspaces_client: Any = None
_security_client: Any = None


def get_db_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the async session factory (available after startup)."""
    if _db_session_factory is None:
        raise RuntimeError("DB not initialised -- called before lifespan startup?")
    return _db_session_factory


def get_workspaces_client() -> Any:
    """Return the shared CodingWorkspacesClient (available after startup)."""
    if _workspaces_client is None:
        raise RuntimeError("Containers client not initialised -- called before lifespan startup?")
    return _workspaces_client


def get_security_client() -> Any:
    """Return the shared CodingSecurityClient (available after startup)."""
    if _security_client is None:
        raise RuntimeError("Security client not initialised -- called before lifespan startup?")
    return _security_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown events."""
    global _db_engine, _db_session_factory, _workspaces_client, _security_client  # noqa: PLW0603

    logger.info("Starting %s v%s (%s)", SERVICE_NAME, VERSION, ENVIRONMENT)

    # -- Database (SQLAlchemy async engine via psycopg) ---------------------
    try:
        db_url = settings.database_url
        # Ensure async driver prefix
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

        _db_engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        _db_session_factory = async_sessionmaker(_db_engine, expire_on_commit=False)

        # Validate connectivity
        async with _db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection validated")
    except Exception as exc:
        logger.warning("DB init failed (service will run without database): %s", exc)
        _db_engine = None
        _db_session_factory = None

    # -- Service clients (coding-containers, coding-security) ----------------
    try:
        from coding_client.clients import CodingWorkspacesClient

        _workspaces_client = CodingWorkspacesClient(base_url=settings.coding_containers_url)
        logger.info("CodingWorkspacesClient initialised: %s", settings.coding_containers_url)
    except Exception as exc:
        logger.warning("Containers client init failed (non-fatal): %s", exc)
        _workspaces_client = None

    try:
        from coding_client.clients.coding_security import CodingSecurityClient

        _security_client = CodingSecurityClient(base_url=settings.coding_security_url)
        logger.info("CodingSecurityClient initialised: %s", settings.coding_security_url)
    except Exception as exc:
        logger.warning("Security client init failed (non-fatal): %s", exc)
        _security_client = None

    # -- CI-checks waits (coding_github DB) ---------------------------------
    # Inert when no wait-store DSN is configured: the wait routes then answer
    # 503 and webhook handling is exactly as before. The loop's first act is an
    # evaluation of EVERY open wait, so a delivery lost while no pod was up (or a
    # registration whose first evaluation died with its pod) is caught here.
    ci_wait_manager = None
    try:
        if settings.ci_wait_dsn:
            from services.ci_waits.manager import (
                CIWaitManager,
                make_owner_id,
                set_ci_wait_manager,
            )
            from services.ci_waits.store import CIWaitStore

            ci_wait_manager = CIWaitManager(
                CIWaitStore(),
                owner=make_owner_id(),
                stale_after_seconds=settings.ci_wait_stale_after_seconds,
            )
            set_ci_wait_manager(ci_wait_manager)
            ci_wait_manager.start(settings.ci_wait_sweep_interval_seconds)
            logger.info("CI-wait manager started (owner=%s)", ci_wait_manager.owner)
        else:
            logger.info("CI-checks waits disabled (no PGBOUNCER_URL)")
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("CI-wait manager not started (non-fatal): %s", exc)

    # Telemetry is NOT set up here. It used to be, and it exported nothing for
    # two independent reasons: it read OTEL_EXPORTER_OTLP_ENDPOINT, which is not
    # the variable this fleet provisions (OTLP_TRACES_ENDPOINT is, and it is set
    # on this workload), so it always took the else-branch -- at DEBUG, so the
    # miss was invisible; and instrument_app() inside a lifespan is a silent
    # no-op because Starlette has already built the middleware stack.
    # setup_tracing() is called at module scope below and raises if anyone moves
    # it back in here.

    yield

    # -- Shutdown -----------------------------------------------------------
    if ci_wait_manager is not None:
        try:
            from db.session import close_pool
            from services.ci_waits.manager import set_ci_wait_manager

            await ci_wait_manager.shutdown()
            set_ci_wait_manager(None)
            await close_pool()
        except Exception as exc:  # shutdown must continue to the other resources
            logger.warning("CI-wait shutdown error (non-fatal): %s", exc)
    if _workspaces_client is not None:
        await _workspaces_client.close()
        logger.info("Containers client closed")
    if _security_client is not None:
        await _security_client.close()
        logger.info("Security client closed")
    if _db_engine is not None:
        await _db_engine.dispose()
        logger.info("Database engine disposed")
    logger.info("Shutting down %s", SERVICE_NAME)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=SERVICE_NAME,
    description="""GitHub integration service for the Code Graph Helper platform.

**Auth:** `coding-client` AuthMiddleware + TenantMiddleware (license-token introspection via Istio).
All tenant-scoped routes require `X-Tenant-ID` header. `/health` and `/` are exempt.

**Endpoint groups:**
- `/api/repositories` -- GitHub-aware workspace import, remote forks, changes, pull requests, and validation
- `/api/repositories/webhooks` -- GitHub webhook receiver (push, PR, comments, workflow_run) with HMAC
  signature verification (constant-time compare; an empty configured secret or a missing/wrong signature
  is refused, fail closed). **Also NOT `require_tenant()`-guarded, for the same reason as `/api/internal`
  below** -- GitHub cannot send `X-Tenant-Id` or a licence token, so the HMAC signature is the only
  credential. Forwards PR-merged / PR-closed-without-merge / post-merge-CI outcomes to n8n (see
  `services/outcome_forward.py`), replacing a 30-minute n8n poll.
- `/api/webhooks/pr-actions` -- process validation results, validation webhook for CI/CD
- `/api/internal` -- service-to-service only (scoped clone-token vending for coding-containers).
  NOT proxied by coding-gateway (absent from its PROXY_ROUTES allowlist). **Authorised by an
  `X-Internal-Token` shared secret, NOT by `require_tenant()`** -- but read the next sentence
  before concluding the tenant plays no part. `TenantMiddleware`/`AuthMiddleware` are installed
  **globally** (see `install_auth_middleware` below); the ONLY exemption is `POST` on the exact
  GitHub webhook path, so `/api/internal`
  IS still traversed by them. A caller that sends BOTH its own licence and an `X-Tenant-Id`
  naming a DIFFERENT tenant than that licence gets a **403
  `TENANT_ASSERTION_NOT_PERMITTED`** from the middleware -- before the route, and before the
  shared secret is ever checked. The route reads no tenant, so the correct call omits
  `X-Tenant-Id` entirely.
- `/health`, `/` -- health check and service info (no auth)

**Key features:** dual-token architecture (user token + system @nonagenticai token),
hybrid PR validation (fast + comprehensive via GitHub Actions), webhook-driven code graph updates.

Repository checkout lifecycle and local file/exec/git operations are owned by the
canonical `/api/workspaces` API in coding-containers. Historical repository
provision/list/status/file routes remain callable here only as hidden migration aliases;
new GitHub-aware provisioning uses `POST /api/repositories/import`.
""",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,  # type: ignore[invalid-argument-type]
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Request logging middleware (KAG standard) ------------------------------
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
app.add_middleware(RequestLoggingMiddleware)  # type: ignore[invalid-argument-type]


# -- Auth middleware (from coding-client) -----------------------------------
#: The single route GitHub delivers to. Exempt from ``AuthMiddleware`` (see
#: ``install_auth_middleware``); its own HMAC signature check is its only gate.
GITHUB_WEBHOOK_PATH = "/api/repositories/webhooks/webhook"


def install_auth_middleware(target_app: FastAPI) -> None:
    """Attach the auth + tenant middleware, or refuse to start.

    Fails CLOSED. This used to log a warning and then serve every route
    unauthenticated -- the loudest possible security failure reduced to a line
    nobody reads. It is not hypothetical: coding-client has already moved a
    public name out of ``auth.tenant`` once (``TRUSTED_NETWORKS``, 1.7.0), which
    is exactly the shape of change that lands here. A service that cannot load
    its auth middleware must not serve.
    """
    try:
        from coding_client.auth.middleware import AuthMiddleware
        from coding_client.auth.tenant import TenantMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "ENABLE_AUTH is set but the auth middleware could not be imported; "
            "refusing to start unauthenticated"
        ) from exc

    class GitHubWebhookExemptAuthMiddleware(AuthMiddleware):
        """``AuthMiddleware`` minus exactly one route: GitHub's webhook delivery.

        GitHub cannot present a licence token or Istio identity headers, so the
        stock middleware 401'd every delivery -- signed ones included -- before the
        route's fail-closed HMAC check ever ran (measured in-pod after #133). The
        exemption is an exact (method, path) match on the decoded path Starlette
        routes on, so anything it lets through lands on the HMAC-gated handler and
        nowhere else; every other request, including GET on the same path, takes
        the unmodified parent path.
        """

        async def dispatch(self, request, call_next):  # type: ignore[override]
            if request.method == "POST" and request.url.path == GITHUB_WEBHOOK_PATH:
                return await call_next(request)
            return await super().dispatch(request, call_next)

    # TenantMiddleware runs after AuthMiddleware (added first = outermost)
    target_app.add_middleware(TenantMiddleware)  # type: ignore[invalid-argument-type]
    target_app.add_middleware(GitHubWebhookExemptAuthMiddleware)  # type: ignore[invalid-argument-type]
    logger.info("Auth + Tenant middleware enabled")


if ENABLE_AUTH:
    install_auth_middleware(app)

# --- Include Routers ---
app.include_router(github_router, prefix="/api/repositories", tags=["github"])
app.include_router(repo_file_read_router, prefix="/api/repositories", tags=["github"])
app.include_router(pr_actions_router, prefix="/api/webhooks/pr-actions", tags=["pr-actions"])
# Service-to-service only. NOT in coding-gateway's PROXY_ROUTES allowlist, so it
# is unreachable from the internet; guarded by an X-Internal-Token shared secret
# (see routers/github_internal.py — require_tenant cannot guard token vending).
app.include_router(github_internal_router, prefix="/api/internal", tags=["internal"])

# -- OpenAPI security config (KAG standard) --------------------------------
configure_openapi(
    app,
    protected_prefixes={"/api/github", "/api/repositories", "/api/webhooks", "/api/internal"},
    public_endpoints={"/", "/health", "/docs", "/redoc", "/openapi.json"},
)

# Tracing, at module scope -- before the app serves anything. setup_tracing()
# raises if it is ever called after startup, which is exactly what the removed
# lifespan block did silently. It also instruments outbound httpx, so this
# service becomes a source of service-graph edges and not only a target.
setup_tracing(app, SERVICE_NAME)

# -- Exception handlers (KAG standard) -------------------------------------
register_exception_handlers(app, debug=settings.debug)


@app.get("/health", operation_id="health_check")
async def health():
    """Liveness for the startup/readiness/liveness probes — and an HONEST one.

    This used to report ``status`` from a Redis ping ALONE. Redis had no caller
    anywhere in ``src`` (only a ``get_redis()`` accessor nothing invoked), and no
    ``coding-github-redis`` Service was ever deployed, so this endpoint reported
    ``degraded`` permanently, for a dependency the service does not have. Worse,
    the one local resource that IS wired — the database — could not influence the
    verdict at all: with Postgres down and Redis up it would still have said
    ``ok``. A signal that is always wrong in the same direction trains people to
    ignore it, so the Redis wiring is gone entirely and the verdict now follows
    something real.

    ``status`` is deliberately about THIS process, not about downstreams: the same
    payload backs the liveness probe, so folding coding-containers /
    coding-security / the GitHub API into it would let one of their outages
    restart-loop this service. The database is reported because it is a genuine
    local resource, and it is reported as ``degraded`` rather than fatal because
    startup already tolerates its absence by design (see the lifespan handler).
    """
    db_ok = _db_engine is not None
    return {
        "status": "ok" if db_ok else "degraded",
        "service": SERVICE_NAME,
        "version": VERSION,
        "db": "connected" if db_ok else "disconnected",
    }


@app.get("/", operation_id="get_root")
async def root():
    return {"service": SERVICE_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}
