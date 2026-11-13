"""
GitHub error handling utilities.

This module centralizes GitHub API exception handling to reduce cyclomatic
complexity across services that interact with GitHub (github_validation_service,
github_integration_service, etc.).
"""

import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from coding_client.exceptions import (  # TODO: replace with coding-client exceptions package
    AuthenticationException,
    NotFoundException,
    PermissionException,
    RateLimitException,
    ServiceUnavailableException,
    ValidationException,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Try to import GithubException, fallback to base Exception if not available
try:
    from github.GithubException import GithubException, RateLimitExceededException
except ImportError:
    GithubException: type[Exception] = Exception
    RateLimitExceededException: type[Exception] = Exception


class GitHubErrorHandler:
    """
    Centralized handler for GitHub API exceptions.

    Maps GitHub API status codes to appropriate application exceptions,
    reducing the need for repeated if/elif chains across services.
    """

    # Map of HTTP status codes to exception classes and default messages
    # Using Any type for exception class to allow keyword arguments
    STATUS_MAP: dict[int, tuple[Any, str]] = {
        401: (AuthenticationException, "GitHub authentication failed"),
        403: (PermissionException, "Access forbidden to GitHub resource"),
        404: (NotFoundException, "GitHub resource not found"),
        409: (ValidationException, "GitHub resource conflict"),
        422: (ValidationException, "GitHub validation failed"),
        429: (RateLimitException, "GitHub API rate limit exceeded"),
        500: (ServiceUnavailableException, "GitHub server error"),
        502: (ServiceUnavailableException, "GitHub gateway error"),
        503: (ServiceUnavailableException, "GitHub service unavailable"),
    }

    @classmethod
    def handle_exception(
        cls,
        exception: Exception,
        resource_name: str | None = None,
        operation: str = "accessing",
        include_details: bool = True,
    ) -> Exception:
        """
        Convert a GitHub exception to an appropriate application exception.

        Args:
            exception: The caught GitHub exception
            resource_name: Name of the resource (e.g., "owner/repo")
            operation: Description of the operation (e.g., "accessing", "creating")
            include_details: Whether to include exception details in message

        Returns:
            Appropriate application exception to raise

        Example:
            try:
                repo = github.get_repo(repo_name)
            except GithubException as e:
                raise GitHubErrorHandler.handle_exception(
                    e, resource_name=repo_name, operation="fetching"
                )
        """
        status = getattr(exception, "status", None)
        data = getattr(exception, "data", {})

        # Get the error message from GitHub response
        github_message = ""
        if isinstance(data, dict):
            github_message = data.get("message", "")

        # Build context string
        context = f"{operation} {resource_name}" if resource_name else operation

        # Check for rate limit exception specifically
        if isinstance(exception, RateLimitExceededException):
            msg = f"GitHub rate limit exceeded while {context}"
            if github_message and include_details:
                msg += f": {github_message}"
            return RateLimitException(message=msg)

        # Look up status code in map
        if status and status in cls.STATUS_MAP:
            exc_class, default_msg = cls.STATUS_MAP[status]
            msg = f"{default_msg} while {context}"
            if github_message and include_details:
                msg += f": {github_message}"
            return exc_class(message=msg)

        # Default to ValidationException for unknown errors
        msg = f"GitHub API error while {context}"
        if include_details:
            msg += f": {exception}"
        return ValidationException(message=msg)

    @classmethod
    def is_not_found(cls, exception: Exception) -> bool:
        """Check if exception indicates resource not found."""
        return getattr(exception, "status", None) == 404

    @classmethod
    def is_rate_limited(cls, exception: Exception) -> bool:
        """Check if exception indicates rate limiting."""
        if isinstance(exception, RateLimitExceededException):
            return True
        return getattr(exception, "status", None) == 429

    @classmethod
    def is_auth_error(cls, exception: Exception) -> bool:
        """Check if exception indicates authentication/authorization error."""
        status = getattr(exception, "status", None)
        return status in (401, 403)

    @classmethod
    def is_server_error(cls, exception: Exception) -> bool:
        """Check if exception indicates a server-side error."""
        status = getattr(exception, "status", None)
        return status is not None and status >= 500

    @classmethod
    def is_secondary_rate_limited(cls, exception: Exception) -> bool:
        """Check for GitHub's *secondary* (abuse) rate limit.

        Distinct from the primary hourly quota (``is_rate_limited``): GitHub
        throttles bursts of mutating calls (PR/fork/comment creation) with an
        HTTP 403 (sometimes 429) whose body says "secondary rate limit" / "abuse
        detection mechanism", usually accompanied by a ``Retry-After`` header.
        This is the limit that bites a mass-PR campaign, and unlike a plain 403
        permission denial it IS retryable after a wait.
        """
        status = getattr(exception, "status", None)
        if status not in (403, 429):
            return False
        data = getattr(exception, "data", {}) or {}
        message = (data.get("message") if isinstance(data, dict) else "") or str(exception)
        low = message.lower()
        if (
            "secondary rate limit" in low
            or "exceeded a secondary" in low
            or "abuse detection" in low
        ):
            return True
        # A 403 carrying a Retry-After is GitHub asking us to back off, not a
        # permanent permission denial.
        headers = getattr(exception, "headers", {}) or {}
        if status == 403 and ("Retry-After" in headers or "retry-after" in headers):
            return True
        return False

    @classmethod
    def is_throttled(cls, exception: Exception) -> bool:
        """Primary OR secondary rate limit — retryable, but NOT generic 5xx.

        Use for calls (e.g. fork creation) that already have a non-retry
        fallback path for server errors, where only the rate-limit case
        warrants waiting and retrying.
        """
        return cls.is_rate_limited(exception) or cls.is_secondary_rate_limited(exception)

    @classmethod
    def is_retryable(cls, exception: Exception) -> bool:
        """Whether ``exception`` should be retried with backoff.

        Covers primary rate limit (429), secondary/abuse rate limit (403/429),
        and transient server errors (5xx). Plain 401/403/404/422 are NOT
        retryable — they are deterministic failures.
        """
        return (
            cls.is_rate_limited(exception)
            or cls.is_secondary_rate_limited(exception)
            or cls.is_server_error(exception)
        )

    @classmethod
    def get_retry_after(cls, exception: Exception) -> int | None:
        """
        Extract retry-after seconds from rate limit exception.

        Returns:
            Number of seconds to wait, or None if not available
        """
        headers = getattr(exception, "headers", {})
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    return int(retry_after)
                except (ValueError, TypeError):  # fmt: skip
                    pass

        # Try X-RateLimit-Reset header
        reset_time = headers.get("X-RateLimit-Reset") if headers else None
        if reset_time:
            try:
                import time

                return max(0, int(reset_time) - int(time.time()))
            except (ValueError, TypeError):  # fmt: skip
                pass

        return None


def retry_github_call[T](
    operation: Callable[[], T],
    *,
    op_name: str = "github call",
    retryable: Callable[[Exception], bool] = GitHubErrorHandler.is_retryable,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[float, float], float] = random.uniform,
) -> T:
    """Call ``operation()`` with exponential backoff + full jitter on retryable errors.

    Built for surviving GitHub's abuse/rate limits during high-volume PR/fork
    creation: when ``retryable(exc)`` is true and attempts remain, wait and
    retry. A server-provided ``Retry-After`` / ``X-RateLimit-Reset`` hint
    (via :meth:`GitHubErrorHandler.get_retry_after`) is honoured; otherwise the
    delay is ``random(0, min(max_delay, base_delay * 2**(attempt-1)))`` ("full
    jitter", which decorrelates concurrent workers far better than fixed
    backoff). Non-retryable exceptions, and the final attempt, are re-raised
    unchanged so existing error handling (idempotency, copy-repo fallback, etc.)
    still runs.

    The decision is delegated to ``retryable`` rather than an ``except`` type so
    callers can scope what they retry (e.g. throttle-only for fork creation,
    full for PR creation) and so it works regardless of which GitHub exception
    class is in play. ``sleep`` and ``rng`` are injectable for fast, deterministic
    tests.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 — retry decision delegated to `retryable`
            if attempt >= max_attempts or not retryable(exc):
                raise
            hint = GitHubErrorHandler.get_retry_after(exc)
            if hint is not None:
                delay = min(max_delay, float(hint)) + rng(0.0, 1.0)
            else:
                ceiling = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay = rng(0.0, ceiling)
            logger.warning(
                "GitHub %s failed (attempt %d/%d), retrying in %.2fs: %s",
                op_name,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            sleep(delay)


def handle_github_error(
    resource_name: str | None = None,
    operation: str = "accessing",
    reraise: bool = True,
    log_level: int = logging.WARNING,
) -> Callable:
    """
    Decorator for handling GitHub exceptions in functions.

    Args:
        resource_name: Name of the resource being accessed
        operation: Description of the operation
        reraise: Whether to reraise the converted exception
        log_level: Logging level for the error

    Returns:
        Decorated function

    Example:
        @handle_github_error(operation="fetching repository")
        async def get_repo(self, repo_name: str):
            return self.github.get_repo(repo_name)
    """
    from functools import wraps

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                result = func(*args, **kwargs)
                return await cast(Awaitable[T], result)
            except GithubException as e:
                converted = GitHubErrorHandler.handle_exception(
                    e, resource_name=resource_name, operation=operation
                )
                logger.log(log_level, str(converted))
                if reraise:
                    raise converted from e
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except GithubException as e:
                converted = GitHubErrorHandler.handle_exception(
                    e, resource_name=resource_name, operation=operation
                )
                logger.log(log_level, str(converted))
                if reraise:
                    raise converted from e
                raise

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
        return sync_wrapper

    return decorator
