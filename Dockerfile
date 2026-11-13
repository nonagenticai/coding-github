ARG BASE_REGISTRY=docker.io

# --- Stage 1: Builder ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

# git is required because coding-client is a pinned GIT dependency (a public repository, not a
# package on an index). uv shells out to the git CLI for those; the slim base does not ship it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Production ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# git: create_rust_target_repo clones the template, runs its scripts/rename.sh
# (bash) and pushes the new target's first commit (services/rust_target.py).
RUN apt-get update && apt-get install -y --no-install-recommends curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create appuser BEFORE the copies so each COPY can set ownership inline via
# --chown. A trailing `chown -R appuser:appuser /app` instead costs a
# recursive traversal of the populated .venv AND rewrites every file into a
# second, non-dedupable image layer that has to be uploaded whole — measured
# in-fleet at 92% of an otherwise-792s build (1942s of 2110s, 1115.3s of
# 1214s); a push without it is 0.2-5.2s. /app itself is chowned
# non-recursively so the directory keeps the ownership the old recursive
# chown gave it.
RUN useradd -m -u 1000 appuser && mkdir -p /app && chown appuser:appuser /app
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
# Migrations for the coding_github DB (CI-checks wait store); the Deployment's
# `migrate` initContainer runs `alembic upgrade head` from /app, as coding-containers.
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
USER appuser

EXPOSE 8692

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8692/health || exit 1

# --timeout-keep-alive 75: uvicorn's default 5s is shorter than the ingress
# gateway's upstream idle timeout, so envoy reused a pooled connection uvicorn had
# just closed and a GitHub webhook delivery got
#   503 "upstream connect error ... reset reason: connection termination"
# (2026-10-02, coding-github#131). The gitops DestinationRule coding-github-idle sets
# envoy's idleTimeout to 60s, so envoy gives up FIRST (60s < 75s). Same race and fix
# as core-app / admin-app. tests/unit/test_dockerfile_keepalive.py pins the value.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8692", "--timeout-keep-alive", "75"]
