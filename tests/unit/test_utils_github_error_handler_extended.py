"""Extended tests for utils/github_error_handler.py — covers uncovered lines."""

import logging
import sys
import time
import types


def _setup_exception_stubs():
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
    try:
        cls(message="test")
        return True
    except TypeError:
        return False


_setup_exception_stubs()

from github.GithubException import GithubException, RateLimitExceededException  # noqa: E402

from utils.github_error_handler import GitHubErrorHandler, handle_github_error  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────────


def _make_exc(status, data=None, headers=None):
    return GithubException(status=status, data=data or {}, headers=headers or {})


# ── Tests: is_not_found  (line 120) ───────────────────────────────────────


class TestIsNotFound:
    def test_returns_true_for_404(self):
        exc = _make_exc(404)
        assert GitHubErrorHandler.is_not_found(exc) is True

    def test_returns_false_for_other_status(self):
        exc = _make_exc(403)
        assert GitHubErrorHandler.is_not_found(exc) is False

    def test_returns_false_when_no_status(self):
        exc = Exception("plain error")
        assert GitHubErrorHandler.is_not_found(exc) is False


# ── Tests: is_rate_limited  (lines 125-127) ──────────────────────────────


class TestIsRateLimited:
    def test_returns_true_for_rate_limit_exception(self):
        exc = RateLimitExceededException(status=429, data={"message": "rate limit"})
        assert GitHubErrorHandler.is_rate_limited(exc) is True

    def test_returns_true_for_429_status(self):
        exc = _make_exc(429)
        assert GitHubErrorHandler.is_rate_limited(exc) is True

    def test_returns_false_for_other_status(self):
        exc = _make_exc(500)
        assert GitHubErrorHandler.is_rate_limited(exc) is False

    def test_returns_false_when_no_status(self):
        exc = Exception("plain error")
        assert GitHubErrorHandler.is_rate_limited(exc) is False


# ── Tests: is_auth_error  (lines 132-133) ────────────────────────────────


class TestIsAuthError:
    def test_returns_true_for_401(self):
        exc = _make_exc(401)
        assert GitHubErrorHandler.is_auth_error(exc) is True

    def test_returns_true_for_403(self):
        exc = _make_exc(403)
        assert GitHubErrorHandler.is_auth_error(exc) is True

    def test_returns_false_for_404(self):
        exc = _make_exc(404)
        assert GitHubErrorHandler.is_auth_error(exc) is False

    def test_returns_false_when_no_status(self):
        exc = Exception("no status attr")
        assert GitHubErrorHandler.is_auth_error(exc) is False


# ── Tests: is_server_error  (lines 138-139) ──────────────────────────────


class TestIsServerError:
    def test_returns_true_for_500(self):
        exc = _make_exc(500)
        assert GitHubErrorHandler.is_server_error(exc) is True

    def test_returns_true_for_502(self):
        exc = _make_exc(502)
        assert GitHubErrorHandler.is_server_error(exc) is True

    def test_returns_true_for_503(self):
        exc = _make_exc(503)
        assert GitHubErrorHandler.is_server_error(exc) is True

    def test_returns_false_for_404(self):
        exc = _make_exc(404)
        assert GitHubErrorHandler.is_server_error(exc) is False

    def test_returns_false_when_no_status(self):
        exc = Exception("no status")
        assert GitHubErrorHandler.is_server_error(exc) is False


# ── Tests: get_retry_after  (lines 149-168) ──────────────────────────────


class TestGetRetryAfter:
    def test_retry_after_header(self):
        exc = _make_exc(429, headers={"Retry-After": "120"})
        assert GitHubErrorHandler.get_retry_after(exc) == 120

    def test_retry_after_header_lowercase(self):
        exc = _make_exc(429, headers={"retry-after": "60"})
        assert GitHubErrorHandler.get_retry_after(exc) == 60

    def test_retry_after_invalid_value(self):
        exc = _make_exc(429, headers={"Retry-After": "not-a-number"})
        # Falls through to X-RateLimit-Reset check, which is absent → None
        assert GitHubErrorHandler.get_retry_after(exc) is None

    def test_ratelimit_reset_header(self):
        future = int(time.time()) + 300
        exc = _make_exc(429, headers={"X-RateLimit-Reset": str(future)})
        result = GitHubErrorHandler.get_retry_after(exc)
        assert result is not None
        # Should be approximately 300 seconds (allow 5s tolerance)
        assert 295 <= result <= 305

    def test_ratelimit_reset_in_past_returns_zero(self):
        past = int(time.time()) - 100
        exc = _make_exc(429, headers={"X-RateLimit-Reset": str(past)})
        result = GitHubErrorHandler.get_retry_after(exc)
        assert result == 0

    def test_ratelimit_reset_invalid_value(self):
        exc = _make_exc(429, headers={"X-RateLimit-Reset": "garbage"})
        assert GitHubErrorHandler.get_retry_after(exc) is None

    def test_no_headers_returns_none(self):
        exc = Exception("no headers attr")
        assert GitHubErrorHandler.get_retry_after(exc) is None

    def test_empty_headers_returns_none(self):
        exc = _make_exc(429, headers={})
        assert GitHubErrorHandler.get_retry_after(exc) is None


# ── Tests: handle_github_error decorator  (lines 194-230) ────────────────


class TestHandleGithubErrorDecorator:
    async def test_async_decorator_reraises_converted_exception(self):
        from coding_client.exceptions import ValidationException

        @handle_github_error(operation="fetching repo", reraise=True)
        async def failing_async():
            raise GithubException(status=422, data={"message": "Validation Failed"})

        try:
            await failing_async()
            raise AssertionError("Expected exception")
        except ValidationException as e:
            assert "fetching repo" in str(e)

    async def test_async_decorator_success(self):
        @handle_github_error(operation="listing repos")
        async def success_async():
            return "ok"

        result = await success_async()
        assert result == "ok"

    def test_sync_decorator_reraises_converted_exception(self):
        from coding_client.exceptions import NotFoundException

        @handle_github_error(operation="fetching file", reraise=True)
        def failing_sync():
            raise GithubException(status=404, data={"message": "Not Found"})

        try:
            failing_sync()
            raise AssertionError("Expected exception")
        except NotFoundException as e:
            assert "fetching file" in str(e)

    def test_sync_decorator_success(self):
        @handle_github_error(operation="reading")
        def success_sync():
            return 42

        assert success_sync() == 42

    async def test_async_decorator_logs_at_specified_level(self, caplog):
        @handle_github_error(operation="testing", log_level=logging.ERROR, reraise=True)
        async def failing():
            raise GithubException(status=500, data={"message": "oops"})

        with caplog.at_level(logging.ERROR):
            try:
                await failing()
            except Exception:
                pass

        assert any("oops" in r.message for r in caplog.records)

    def test_sync_decorator_logs_at_specified_level(self, caplog):
        @handle_github_error(operation="testing", log_level=logging.ERROR, reraise=True)
        def failing():
            raise GithubException(status=500, data={"message": "boom"})

        with caplog.at_level(logging.ERROR):
            try:
                failing()
            except Exception:
                pass

        assert any("boom" in r.message for r in caplog.records)

    async def test_async_decorator_with_resource_name(self):
        from coding_client.exceptions import PermissionException

        @handle_github_error(resource_name="owner/repo", operation="cloning", reraise=True)
        async def clone():
            raise GithubException(status=403, data={"message": "Forbidden"})

        try:
            await clone()
            raise AssertionError("Expected exception")
        except PermissionException as e:
            assert "cloning" in str(e)
            assert "owner/repo" in str(e)

    def test_sync_decorator_with_resource_name(self):
        from coding_client.exceptions import AuthenticationException

        @handle_github_error(resource_name="secret-repo", operation="accessing", reraise=True)
        def access():
            raise GithubException(status=401, data={"message": "Bad credentials"})

        try:
            access()
            raise AssertionError("Expected exception")
        except AuthenticationException as e:
            assert "accessing" in str(e)
            assert "secret-repo" in str(e)

    async def test_async_decorator_non_github_exception_propagates(self):
        """Non-GithubException errors should propagate unchanged."""

        @handle_github_error(operation="reading")
        async def raises_value_error():
            raise ValueError("unrelated")

        try:
            await raises_value_error()
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "unrelated" in str(e)

    def test_sync_decorator_non_github_exception_propagates(self):
        @handle_github_error(operation="reading")
        def raises_type_error():
            raise TypeError("wrong type")

        try:
            raises_type_error()
            raise AssertionError("Expected TypeError")
        except TypeError as e:
            assert "wrong type" in str(e)

    async def test_async_decorator_reraise_false_raises_original(self):
        """When reraise=False, the original GithubException should propagate."""

        @handle_github_error(operation="testing", reraise=False)
        async def failing():
            raise GithubException(status=500, data={"message": "server error"})

        try:
            await failing()
            raise AssertionError("Expected GithubException")
        except GithubException as e:
            assert e.status == 500

    def test_sync_decorator_reraise_false_raises_original(self):
        """When reraise=False, the original GithubException should propagate."""

        @handle_github_error(operation="testing", reraise=False)
        def failing():
            raise GithubException(status=404, data={"message": "Not Found"})

        try:
            failing()
            raise AssertionError("Expected GithubException")
        except GithubException as e:
            assert e.status == 404


# ── Tests: handle_exception edge cases (for completeness) ────────────────


class TestHandleExceptionEdgeCases:
    def test_include_details_false_hides_message(self):
        exc = _make_exc(404, {"message": "secret details"})
        result = GitHubErrorHandler.handle_exception(exc, include_details=False)
        assert "secret details" not in str(result)

    def test_unknown_status_include_details_false(self):
        exc = _make_exc(418, {"message": "teapot"})
        result = GitHubErrorHandler.handle_exception(exc, include_details=False)
        assert "teapot" not in str(result)
        assert "GitHub API error" in str(result)

    def test_status_401_maps_to_authentication(self):
        from coding_client.exceptions import AuthenticationException

        exc = _make_exc(401, {"message": "Bad credentials"})
        result = GitHubErrorHandler.handle_exception(exc)
        assert isinstance(result, AuthenticationException)

    def test_status_500_maps_to_service_unavailable(self):
        from coding_client.exceptions import ServiceUnavailableException

        exc = _make_exc(500, {"message": "Internal error"})
        result = GitHubErrorHandler.handle_exception(exc)
        assert isinstance(result, ServiceUnavailableException)

    def test_status_502_maps_to_service_unavailable(self):
        from coding_client.exceptions import ServiceUnavailableException

        exc = _make_exc(502)
        result = GitHubErrorHandler.handle_exception(exc)
        assert isinstance(result, ServiceUnavailableException)

    def test_status_503_maps_to_service_unavailable(self):
        from coding_client.exceptions import ServiceUnavailableException

        exc = _make_exc(503)
        result = GitHubErrorHandler.handle_exception(exc)
        assert isinstance(result, ServiceUnavailableException)

    def test_status_409_maps_to_validation(self):
        from coding_client.exceptions import ValidationException

        exc = _make_exc(409, {"message": "Conflict"})
        result = GitHubErrorHandler.handle_exception(exc)
        assert isinstance(result, ValidationException)

    def test_rate_limit_exception_without_include_details(self):
        exc = RateLimitExceededException(status=429, data={"message": "API rate limit exceeded"})
        result = GitHubErrorHandler.handle_exception(exc, include_details=False)
        from coding_client.exceptions import RateLimitException

        assert isinstance(result, RateLimitException)
        assert "API rate limit exceeded" not in str(result)

    def test_no_status_no_data(self):
        exc = Exception("generic")
        result = GitHubErrorHandler.handle_exception(exc)
        from coding_client.exceptions import ValidationException

        assert isinstance(result, ValidationException)
        assert "generic" in str(result)

    def test_data_is_not_dict(self):
        """When data is not a dict, github_message should stay empty."""
        exc = GithubException(status=404, data="string-data")
        result = GitHubErrorHandler.handle_exception(exc)
        from coding_client.exceptions import NotFoundException

        assert isinstance(result, NotFoundException)

    def test_rate_limit_exception_with_details(self):
        """RateLimitExceededException with include_details=True appends message."""
        exc = RateLimitExceededException(
            status=429, data={"message": "API rate limit exceeded for user"}
        )
        result = GitHubErrorHandler.handle_exception(
            exc, resource_name="org/repo", operation="listing", include_details=True
        )
        from coding_client.exceptions import RateLimitException

        assert isinstance(result, RateLimitException)
        assert "API rate limit exceeded for user" in str(result)
        assert "listing" in str(result)
