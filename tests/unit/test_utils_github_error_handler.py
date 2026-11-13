"""Tests for utils/github_error_handler.py."""

import sys
import types


def _setup_exception_stubs():
    """Ensure coding_client.exceptions has all needed exception classes with keyword support."""
    exc_mod = sys.modules.get("coding_client.exceptions")
    if exc_mod is None:
        exc_mod = types.ModuleType("coding_client.exceptions")
        sys.modules["coding_client.exceptions"] = exc_mod

    for cls_name in (
        "AuthenticationException",
        "NotFoundException",
        "PermissionException",
        "ValidationException",
        "RateLimitException",
        "ServiceUnavailableException",
    ):
        existing = getattr(exc_mod, cls_name, None)
        if existing is None or not _accepts_message_kwarg(existing):
            exc_cls = type(
                cls_name,
                (Exception,),
                {
                    "__init__": lambda self, message="", **kw: (
                        super(type(self), self).__init__(message),
                        setattr(self, "message", message),
                    )[0]
                },
            )
            setattr(exc_mod, cls_name, exc_cls)


def _accepts_message_kwarg(cls):
    """Check if exception class accepts message keyword argument."""
    try:
        cls(message="test")
        return True
    except TypeError:
        return False


_setup_exception_stubs()

from github.GithubException import GithubException, RateLimitExceededException  # noqa: E402

from utils.github_error_handler import GitHubErrorHandler  # noqa: E402


class TestGitHubErrorHandler:
    """Tests for GitHubErrorHandler."""

    def _make_github_exc(self, status, data=None, headers=None):

        return GithubException(status=status, data=data or {}, headers=headers or {})

    def test_404_maps_to_not_found(self):
        """GitHub 404 should map to NotFoundException."""
        exc = self._make_github_exc(404, {"message": "Not Found"})
        result = GitHubErrorHandler.handle_exception(exc, resource_name="owner/repo")
        from coding_client.exceptions import NotFoundException

        assert isinstance(result, NotFoundException)
        assert "not found" in str(result).lower()

    def test_403_maps_to_permission(self):
        """GitHub 403 should map to PermissionException."""
        exc = self._make_github_exc(403, {"message": "Forbidden"})
        result = GitHubErrorHandler.handle_exception(exc, resource_name="owner/repo")
        from coding_client.exceptions import PermissionException

        assert isinstance(result, PermissionException)

    def test_422_maps_to_validation(self):
        """GitHub 422 should map to ValidationException."""
        exc = self._make_github_exc(422, {"message": "Validation Failed"})
        result = GitHubErrorHandler.handle_exception(exc)
        from coding_client.exceptions import ValidationException

        assert isinstance(result, ValidationException)

    def test_generic_unknown_status(self):
        """Unknown status should map to ValidationException (default)."""
        exc = self._make_github_exc(418, {"message": "I'm a teapot"})
        result = GitHubErrorHandler.handle_exception(exc)
        from coding_client.exceptions import ValidationException

        assert isinstance(result, ValidationException)
        assert "teapot" in str(result).lower()

    def test_rate_limit_error(self):
        """RateLimitExceededException should map to RateLimitException."""

        exc = RateLimitExceededException(status=429, data={"message": "rate limit"})
        result = GitHubErrorHandler.handle_exception(exc, operation="listing repos")
        from coding_client.exceptions import RateLimitException

        assert isinstance(result, RateLimitException)
        assert "rate limit" in str(result).lower()


from utils.github_error_handler import retry_github_call  # noqa: E402


def _exc(status, data=None, headers=None):
    return GithubException(status=status, data=data or {}, headers=headers or {})


class TestRetryabilityPredicates:
    """Predicates that decide what GitHub errors are worth retrying."""

    def test_secondary_rate_limit_by_message(self):
        exc = _exc(403, {"message": "You have exceeded a secondary rate limit"})
        assert GitHubErrorHandler.is_secondary_rate_limited(exc) is True
        assert GitHubErrorHandler.is_throttled(exc) is True
        assert GitHubErrorHandler.is_retryable(exc) is True

    def test_abuse_detection_message(self):
        exc = _exc(403, {"message": "abuse detection mechanism triggered"})
        assert GitHubErrorHandler.is_secondary_rate_limited(exc) is True

    def test_403_with_retry_after_header_is_secondary(self):
        exc = _exc(403, {"message": "Forbidden"}, headers={"Retry-After": "60"})
        assert GitHubErrorHandler.is_secondary_rate_limited(exc) is True

    def test_plain_403_permission_not_retryable(self):
        exc = _exc(403, {"message": "Forking disabled by repository owner"})
        assert GitHubErrorHandler.is_secondary_rate_limited(exc) is False
        assert GitHubErrorHandler.is_throttled(exc) is False
        assert GitHubErrorHandler.is_retryable(exc) is False

    def test_422_not_retryable(self):
        exc = _exc(422, {"message": "Validation Failed"})
        assert GitHubErrorHandler.is_retryable(exc) is False

    def test_429_is_throttled_and_retryable(self):
        exc = _exc(429, {"message": "rate limit"})
        assert GitHubErrorHandler.is_throttled(exc) is True
        assert GitHubErrorHandler.is_retryable(exc) is True

    def test_5xx_retryable_but_not_throttled(self):
        exc = _exc(502, {"message": "Bad gateway"})
        assert GitHubErrorHandler.is_server_error(exc) is True
        assert GitHubErrorHandler.is_retryable(exc) is True
        # 5xx is a transient server error, not a rate limit
        assert GitHubErrorHandler.is_throttled(exc) is False


class TestRetryGithubCall:
    """retry_github_call: backoff loop semantics."""

    def test_returns_immediately_on_success(self):
        calls = []
        out = retry_github_call(lambda: calls.append(1) or "ok", sleep=lambda _d: None)
        assert out == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        attempts = {"n": 0}
        slept = []

        def op():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _exc(502, {"message": "transient"})
            return "done"

        out = retry_github_call(
            op, max_attempts=5, base_delay=0.01, sleep=slept.append, rng=lambda _a, b: b
        )
        assert out == "done"
        assert attempts["n"] == 3
        assert len(slept) == 2  # slept before attempts 2 and 3

    def test_non_retryable_raises_immediately(self):
        attempts = {"n": 0}

        def op():
            attempts["n"] += 1
            raise _exc(422, {"message": "Validation Failed"})

        try:
            retry_github_call(op, sleep=lambda _d: None)
            raise AssertionError("expected GithubException")
        except GithubException:
            pass
        assert attempts["n"] == 1  # never retried

    def test_gives_up_after_max_attempts(self):
        attempts = {"n": 0}
        slept = []

        def op():
            attempts["n"] += 1
            raise _exc(503, {"message": "still down"})

        try:
            retry_github_call(
                op, max_attempts=3, base_delay=0.01, sleep=slept.append, rng=lambda _a, b: b
            )
            raise AssertionError("expected GithubException")
        except GithubException:
            pass
        assert attempts["n"] == 3
        assert len(slept) == 2  # one sleep between each of the 3 attempts

    def test_honours_retry_after_header(self):
        slept = []

        def op():
            raise _exc(403, {"message": "secondary rate limit"}, headers={"Retry-After": "7"})

        try:
            retry_github_call(
                op,
                max_attempts=2,
                sleep=slept.append,
                rng=lambda _a, _b: 0.0,  # no jitter
            )
            raise AssertionError("expected GithubException")
        except GithubException:
            pass
        assert slept == [7.0]  # honoured the server hint exactly (jitter zeroed)

    def test_throttle_only_predicate_skips_5xx(self):
        attempts = {"n": 0}

        def op():
            attempts["n"] += 1
            raise _exc(500, {"message": "server error"})

        try:
            retry_github_call(
                op,
                retryable=GitHubErrorHandler.is_throttled,
                sleep=lambda _d: None,
            )
            raise AssertionError("expected GithubException")
        except GithubException:
            pass
        assert attempts["n"] == 1  # is_throttled is False for 5xx → no retry
