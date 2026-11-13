"""
Secure GitHub webhook handler with signature verification and event processing.

This module provides secure webhook handling with HMAC signature verification,
comprehensive event routing, and security controls for GitHub webhook events.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from coding_client.types.core_types import GitHubEventType, GitHubEventTypeEnum
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebhookSecurityLevel(StrEnum):
    """Security levels for webhook processing."""

    STRICT = "strict"
    STANDARD = "standard"
    DEVELOPMENT = "development"


class WebhookProcessingResult(BaseModel):
    """Result of webhook event processing."""

    event_type: GitHubEventType
    action: str | None = None
    processed: bool = False
    error: str | None = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SecureWebhookHandler:
    """
    Secure GitHub webhook handler with signature verification and event routing.

    Provides comprehensive security controls, rate limiting, and event routing
    for GitHub webhooks with proper HMAC signature verification.
    """

    def __init__(
        self,
        webhook_secret: str,
        security_level: WebhookSecurityLevel = WebhookSecurityLevel.STANDARD,
    ):
        """
        Initialize secure webhook handler.

        Args:
            webhook_secret: GitHub webhook secret for signature verification
            security_level: Security level for webhook processing
        """
        self.webhook_secret = webhook_secret.encode()
        self.security_level = security_level
        self._event_handlers: dict[GitHubEventType, list[Callable]] = {}
        self._request_counts: dict[str, list[float]] = {}
        self._max_requests_per_minute = 60
        self._blocked_ips: set[str] = set()
        self._processed_events: list[WebhookProcessingResult] = []
        self._max_metrics_history = 1000
        # Track background tasks to prevent garbage collection
        self._background_tasks: set[asyncio.Task[Any]] = set()
        logger.info(f"Initialized secure webhook handler with {security_level} security")

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GitHub webhook signature using HMAC-SHA256.

        Args:
            payload: Raw webhook payload bytes
            signature: GitHub signature header (X-Hub-Signature-256)

        Returns:
            True if signature is valid
        """
        if not signature:
            logger.warning("Missing webhook signature")
            return False
        if signature.startswith("sha256="):
            signature = signature[7:]
        try:
            expected_signature = hmac.new(self.webhook_secret, payload, hashlib.sha256).hexdigest()
            is_valid = hmac.compare_digest(expected_signature, signature)
            if not is_valid:
                logger.warning("Invalid webhook signature")
            return is_valid
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def register_event_handler(
        self, event_type: GitHubEventType, handler: Callable[[dict[str, Any]], Any]
    ) -> None:
        """
        Register a handler for a specific GitHub event type.

        Args:
            event_type: GitHub event type to handle
            handler: Async function to handle the event
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type} events")

    async def handle_webhook(
        self,
        request: Request,
        payload: bytes,
        event_type: str,
        signature: str | None = None,
        delivery_id: str | None = None,
    ) -> WebhookProcessingResult:
        """
        Handle incoming GitHub webhook with security verification.

        Args:
            request: FastAPI request object
            payload: Raw webhook payload
            event_type: GitHub event type header
            signature: GitHub signature header
            delivery_id: GitHub delivery ID

        Returns:
            Processing result with success/failure status

        Raises:
            HTTPException: If security validation fails
        """
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        try:
            await self._validate_request_security(client_ip, payload, signature)
            github_event_type = self._parse_event_type(event_type)
            if not github_event_type:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported event type: {event_type}",
                )
            try:
                event_data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON payload: {e}",
                )
            action = event_data.get("action")
            # Carried as a dict key rather than a new handler-signature
            # parameter: `_process_event` invokes every registered handler the
            # same way (`handler(event_data)`), and changing that call
            # convention would touch push/issues handlers that have no use for
            # a delivery id. `.get(...)` on an absent key is `None`, which is
            # the correct "no delivery id" value for a handler called directly
            # in a test with a hand-built payload.
            event_data["_coding_github_delivery_id"] = delivery_id
            processed = await self._process_event(github_event_type, event_data)
            processing_time = (time.time() - start_time) * 1000
            result = WebhookProcessingResult(
                event_type=github_event_type,
                action=action,
                processed=processed,
                processing_time_ms=processing_time,
            )
            self._add_metrics(result)
            logger.info(
                f"Processed {github_event_type} webhook (action: {action}, delivery: {delivery_id}) in {processing_time:.2f}ms"
            )
            return result
        except HTTPException:
            raise
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            result = WebhookProcessingResult(
                event_type="push",
                processed=False,
                error=str(e),
                processing_time_ms=processing_time,
            )
            self._add_metrics(result)
            logger.error(f"Webhook processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Webhook processing failed: {e}",
            )

    async def verify_security(
        self, request: Request, payload: bytes, signature: str | None
    ) -> None:
        """Public wrapper around the same security gate ``handle_webhook`` runs.

        Exists so a caller outside this class -- a GitHub event type that is
        not a member of ``GitHubEventTypeEnum`` (e.g. ``workflow_run``; see
        ``routers.github_webhooks._handle_workflow_run_webhook``) and therefore
        cannot go through the enum-gated ``handle_webhook`` dispatch at all --
        still gets IDENTICAL authentication: IP block list, rate limit, and the
        HMAC signature check with its empty-secret fail-closed behaviour.
        Raises ``HTTPException`` on any refusal, exactly as ``handle_webhook``
        does.
        """
        client_ip = self._get_client_ip(request)
        await self._validate_request_security(client_ip, payload, signature)

    async def _validate_request_security(
        self, client_ip: str, payload: bytes, signature: str | None
    ) -> None:
        """Validate webhook request security."""
        if client_ip in self._blocked_ips:
            logger.warning(f"Blocked webhook request from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="IP address is blocked"
            )
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        if self.security_level != WebhookSecurityLevel.DEVELOPMENT:
            if not self.webhook_secret:
                # FAIL CLOSED: an empty configured secret must refuse every
                # delivery outright, never fall back to a guessable default.
                # (Until 2026-09-30 ``get_secure_webhook_handler`` below
                # substituted the hardcoded literal
                # "dev_secret_change_in_production" here, which made the
                # configured secret's value irrelevant in production -- see its
                # docstring.)
                logger.error(
                    "Webhook secret is not configured; refusing all webhook "
                    "deliveries (fail closed)."
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Webhook secret not configured",
                )
            if not signature:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing webhook signature",
                )
            if not self.verify_signature(payload, signature):
                logger.warning(f"Invalid signature from {client_ip}")
                self._block_ip_temporarily(client_ip)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature",
                )

    def _parse_event_type(self, event_type: str) -> GitHubEventType | None:
        """Parse GitHub event type string to literal type."""
        try:
            enum_value = GitHubEventTypeEnum(event_type)
            return enum_value.value
        except ValueError:
            logger.warning(f"Unsupported GitHub event type: {event_type}")
            return None

    async def _process_event(self, event_type: GitHubEventType, event_data: dict[str, Any]) -> bool:
        """Process GitHub event by calling registered handlers."""
        if event_type not in self._event_handlers:
            logger.debug(f"No handlers registered for {event_type}")
            return False
        handlers = self._event_handlers[event_type]
        results = []
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    result = await handler(event_data)
                else:
                    result = handler(event_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler failed for {event_type}: {e}")
                results.append(False)
        return any(results)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return str(request.client.host) if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client IP is within rate limits."""
        now = time.time()
        window_start = now - 60
        if client_ip not in self._request_counts:
            self._request_counts[client_ip] = []
        self._request_counts[client_ip] = [
            req_time for req_time in self._request_counts[client_ip] if req_time > window_start
        ]
        if len(self._request_counts[client_ip]) >= self._max_requests_per_minute:
            return False
        self._request_counts[client_ip].append(now)
        return True

    def _block_ip_temporarily(self, client_ip: str) -> None:
        """Temporarily block an IP address for suspicious activity."""
        self._blocked_ips.add(client_ip)

        async def unblock_later() -> None:
            await asyncio.sleep(600)
            self._blocked_ips.discard(client_ip)
            logger.info(f"Unblocked IP {client_ip}")

        # Store task reference to prevent garbage collection
        task = asyncio.create_task(unblock_later())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        logger.warning(f"Temporarily blocked IP {client_ip}")

    def _add_metrics(self, result: WebhookProcessingResult) -> None:
        """Add processing result to metrics history."""
        self._processed_events.append(result)
        if len(self._processed_events) > self._max_metrics_history:
            self._processed_events = self._processed_events[-self._max_metrics_history :]

    def get_processing_metrics(self) -> dict[str, Any]:
        """Get webhook processing metrics."""
        if not self._processed_events:
            return {"total_events": 0}
        total_events = len(self._processed_events)
        successful_events = sum(1 for e in self._processed_events if e.processed)
        avg_processing_time = (
            sum(e.processing_time_ms for e in self._processed_events) / total_events
        )
        event_type_counts: dict[GitHubEventType, int] = {}
        for event in self._processed_events:
            event_type = event.event_type
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "average_processing_time_ms": avg_processing_time,
            "event_type_breakdown": event_type_counts,
            "active_handlers": {
                event_type: len(handlers) for event_type, handlers in self._event_handlers.items()
            },
        }


def _platform_repo(repo_full_name: str | None) -> bool:
    """True only for a repo owned by ``system_github_owner``."""
    from services.github_integration import is_platform_owned

    return is_platform_owned((repo_full_name or "").split("/", 1)[0])


async def handle_pull_request_event(event_data: dict[str, Any]) -> bool:
    """Handle pull request webhook events."""
    action = event_data.get("action")
    pr = event_data.get("pull_request", {})
    repo = event_data.get("repository", {})
    repo_name = repo.get("full_name")
    pr_number = pr.get("number")
    logger.info(f"Processing PR {action} event for #{pr_number} in {repo_name}")
    if action in ["opened", "synchronize", "reopened"]:
        if not _platform_repo(repo_name):
            # ⛔ The analysis below reads the PR and posts a comment, which needs
            # a GitHub credential. A customer repo is tenant-bound public-App
            # only, a GitHub webhook carries no fr tenant — do not invent one —
            # and there is no PAT fallback. Skipping is the handled outcome.
            logger.info(
                "skipped: customer repo, no tenant context on webhook "
                "(PR #%s %s in %s); no PAT fallback",
                pr_number,
                action,
                repo_name,
            )
            return True
        try:
            from interfaces import WebhookEventHandlerRegistry

            try:
                handler = WebhookEventHandlerRegistry.get("github")
                await handler.analyze_pull_request_changes(repo_name, pr_number, action)
                return True
            except ValueError:
                # Nothing registers "github" today, so this is the live path.
                # It used to import ``api.endpoints.github_webhooks`` — a
                # pre-split monolith module that does not exist in this
                # service. The ModuleNotFoundError was laundered by the
                # ``except Exception`` below into ``return False``, making
                # every PR webhook a silent no-op. The real implementation is
                # in-repo; imported lazily because routers.github_webhooks
                # imports this module at module scope.
                from routers.github_webhooks import analyze_pull_request_changes

                await analyze_pull_request_changes(repo_name, pr_number, action)
                return True
        except Exception:
            # exception() not error(): a bare str(e) here hid a
            # ModuleNotFoundError for as long as this handler has existed.
            logger.exception("Failed to analyze PR #%s in %s", pr_number, repo_name)
            return False
    elif action == "closed":
        if pr.get("merged"):
            logger.info(f"PR #{pr_number} was merged in {repo_name}")
        else:
            logger.info(f"PR #{pr_number} was closed without merging in {repo_name}")
        # Push the outcome to n8n instead of it having to poll for it (George's
        # ruling: no timer may write on a clock). Lazy import: this module
        # avoids top-level imports of sibling `services.*` modules on purpose
        # (see the other lazy imports in this function/file) to dodge circular
        # imports; `services.outcome_forward` has no dependency back on this
        # module, but the convention is kept for consistency.
        from services.outcome_forward import build_pull_request_outcome, forward_outcome_event

        outcome = build_pull_request_outcome(
            event_data, event_data.get("_coding_github_delivery_id")
        )
        if outcome is not None:
            asyncio.create_task(forward_outcome_event(outcome))
        return True
    return False


async def handle_push_event(event_data: dict[str, Any]) -> bool:
    """Handle push webhook events."""
    repo = event_data.get("repository", {})
    ref = event_data.get("ref", "")
    commits = event_data.get("commits", [])
    repo_name = repo.get("full_name")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    logger.info(f"Processing push to {branch} in {repo_name} ({len(commits)} commits)")
    if branch in ["main", "master", "develop"]:
        try:
            # Was ``core.github_integration`` — a pre-split monolith module
            # absent from this service; the ModuleNotFoundError was swallowed
            # below, so every push to a protected branch was a no-op.
            #
            # This path works end to end as of #67. The hub delegates to
            # ``services.github.webhook_handler.handle_repository_update``,
            # which resolves the repo's container and asks coding-analysis to
            # re-scan it; that module's docstring records the sqlite /
            # ``Settings.get_database_manager()`` copy that was removed.
            # Failures below are logged WITH a traceback.
            #
            # This comment used to assert the opposite — that the path still
            # raised AttributeError — and it outlived the fix by three PRs. A
            # reviewer trusting it over the code reaches the wrong conclusion in
            # about a minute, which is why it is being corrected rather than
            # deleted: the trap is the stale claim, not the missing history.
            #
            # No tenant is passed: a GitHub webhook carries none, so the lookup
            # runs unscoped and coding-containers re-checks the tenant itself.
            # An unresolvable repo returns ``no_container``, it does not raise.
            from services.github_integration_core import GitHubIntegrationHub

            hub = GitHubIntegrationHub()
            # Extract changed files and commit SHAs from commits
            changed_files = []
            commit_shas = []
            for commit in commits:
                changed_files.extend(commit.get("modified", []))
                changed_files.extend(commit.get("added", []))
                changed_files.extend(commit.get("removed", []))
                if commit.get("id"):
                    commit_shas.append(commit["id"])
            await hub.handle_repository_update(repo_name, branch, changed_files, commit_shas)
            return True
        except Exception:
            logger.exception("Failed to process push to %s (%s)", repo_name, branch)
            return False
    return True


async def handle_issues_event(event_data: dict[str, Any]) -> bool:
    """Handle issues webhook events."""
    action = event_data.get("action")
    issue = event_data.get("issue", {})
    repo = event_data.get("repository", {})
    repo_name = repo.get("full_name")
    issue_number = issue.get("number")
    logger.info(f"Processing issue {action} event for #{issue_number} in {repo_name}")
    return True


_webhook_handler: SecureWebhookHandler | None = None


def get_secure_webhook_handler(
    webhook_secret: str | None = None,
) -> SecureWebhookHandler:
    """Get the global secure webhook handler instance.

    ⛔ FIXED 2026-09-30: this used to default to the UNPREFIXED env var
    ``GITHUB_WEBHOOK_SECRET`` -- never set anywhere in gitops; this service's
    typed ``Settings`` binds the PREFIXED ``CODING_GITHUB_GITHUB_WEBHOOK_SECRET``
    -- and fell back further to the HARDCODED literal
    ``"dev_secret_change_in_production"`` when that too was unset. In
    production, where only the prefixed var is ever set, this meant HMAC
    verification ran against a publicly-known constant: the configured secret
    had ZERO effect, and anyone who had read this source (or its own docstring)
    could forge a valid signature. It now reads
    ``Settings.github_webhook_secret`` -- the field that actually binds
    ``CODING_GITHUB_GITHUB_WEBHOOK_SECRET`` -- and an EMPTY secret is refused
    outright by ``_validate_request_security`` rather than silently
    substituted; see the fail-closed check there.
    """
    global _webhook_handler
    if _webhook_handler is None:
        if webhook_secret is None:
            from config.settings import get_settings

            webhook_secret = get_settings().github_webhook_secret
        _webhook_handler = SecureWebhookHandler(
            webhook_secret=webhook_secret, security_level=WebhookSecurityLevel.STANDARD
        )
        _webhook_handler.register_event_handler("pull_request", handle_pull_request_event)
        _webhook_handler.register_event_handler("push", handle_push_event)
        _webhook_handler.register_event_handler("issues", handle_issues_event)
    return _webhook_handler
