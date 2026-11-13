"""Resolve a project's ``secret_config_id`` into its per-tenant GIT_TOKEN.

The dual-token upstream-PR flow needs the user's GitHub token to push the
winner branch and open the PR. That token is the source-of-truth in the
**secrets-service** (fr stores the project PAT there; coding-containers persists
only the opaque ``secret_config_id`` reference on the container). The previous
in-memory ``SecureTokenManager`` store never survived cross-service, so this
re-resolves the token from secrets-service — the same place the clone used —
keyed by ``secret_config_id`` + the caller's tenant.

Mirrors ``coding-containers/src/services/secret_resolver.py`` and
``GitHubProvisioningService._get_environment_variables``: a tenant-scoped GET
against the secrets service, reading ``GIT_TOKEN`` from the merged
``env_vars`` + ``secrets`` blob. All failures return ``None`` (never raise).
"""

from __future__ import annotations

import logging

import httpx

from config.settings import get_settings
from services.outbound import secrets_service_headers

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5.0


async def resolve_git_token(
    secret_config_id: str | None,
    tenant_id: str | None,
) -> str | None:
    """Return the GIT_TOKEN for ``secret_config_id`` from the secrets service.

    Returns ``None`` (never raises) when no id is given, the secrets URL is
    unset, the secret is missing, or the fetch fails.
    """
    if not secret_config_id:
        return None

    secrets_url = (get_settings().secrets_api_url or "").rstrip("/")
    if not secrets_url:
        logger.warning(
            "secrets_api_url is unset; cannot resolve secret_config_id=%s "
            "(upstream PR will fail without a token).",
            secret_config_id,
        )
        return None

    if not tenant_id:
        # Previously: ``tenant_id or "anonymous"``. That literal is not a null —
        # it is a valid partition key, so an unattributable fetch SUCCEEDED
        # against a partition no registry issued, and downstream it is
        # indistinguishable from a real tenant named "anonymous". Absent means
        # absent: omit the header and let secrets-service refuse. Same defect,
        # same fix as ``GitHubProvisioningService._get_environment_variables``.
        logger.warning(
            "No tenant for secret_config_id=%s; sending no tenant header. "
            "secrets-service will refuse the fetch, so no token is resolved.",
            secret_config_id,
        )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{secrets_url}/secrets/{secret_config_id}",
                headers=secrets_service_headers(tenant_id),
            )
        if response.status_code == 404:
            logger.warning(
                "Secret config %s not found in secrets-service for tenant=%s.",
                secret_config_id,
                tenant_id,
            )
            return None
        response.raise_for_status()
        payload = response.json()
        combined = {**(payload.get("env_vars") or {}), **(payload.get("secrets") or {})}
        token = combined.get("GIT_TOKEN")
        if token:
            logger.info(
                "Resolved secret_config_id=%s → GIT_TOKEN for upstream PR.",
                secret_config_id,
            )
            return token
        logger.info("Secret config %s had no GIT_TOKEN.", secret_config_id)
        return None
    except httpx.HTTPError as exc:
        logger.warning(
            "Secrets-service fetch failed for config_id=%s: %s — no token.",
            secret_config_id,
            exc,
        )
        return None
