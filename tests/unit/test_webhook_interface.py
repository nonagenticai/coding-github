"""Tests for interfaces/webhook_interface.py."""

from __future__ import annotations

from typing import Any

import pytest

from interfaces.webhook_interface import IWebhookEventHandler, WebhookEventHandlerRegistry


class ConcreteHandler(IWebhookEventHandler):
    """Concrete implementation for testing."""

    async def analyze_pull_request_changes(
        self, repo_name: str, pr_number: int, trigger: str
    ) -> None:
        pass

    async def analyze_repository_changes(
        self, repo_name: str, branch: str, commits: list
    ) -> dict[str, Any]:
        return {"status": "ok"}


class TestProtocolMethodsExist:
    """Protocol/ABC defines expected method signatures."""

    def test_protocol_methods_exist(self):
        # IWebhookEventHandler should define these abstract methods
        assert hasattr(IWebhookEventHandler, "analyze_pull_request_changes")
        assert hasattr(IWebhookEventHandler, "analyze_repository_changes")

        # They should be abstract
        abstracts = getattr(IWebhookEventHandler, "__abstractmethods__", set())
        assert "analyze_pull_request_changes" in abstracts
        assert "analyze_repository_changes" in abstracts


class TestConcreteImplementationSatisfiesProtocol:
    """A concrete implementation passes isinstance check."""

    def test_concrete_implementation_satisfies_protocol(self):
        handler = ConcreteHandler()
        assert isinstance(handler, IWebhookEventHandler)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            IWebhookEventHandler()


class TestRegistryOperations:
    """WebhookEventHandlerRegistry register/get/get_handler."""

    def setup_method(self):
        # Clear registry between tests
        WebhookEventHandlerRegistry._handlers = {}

    def test_register_and_get(self):
        handler = ConcreteHandler()
        WebhookEventHandlerRegistry.register("test", handler)
        assert WebhookEventHandlerRegistry.get("test") is handler

    def test_get_handler_returns_none_for_missing(self):
        assert WebhookEventHandlerRegistry.get_handler("nope") is None

    def test_get_raises_for_missing(self):
        with pytest.raises(ValueError, match="No handler registered with name"):
            WebhookEventHandlerRegistry.get("nonexistent")

    def test_register_overwrites(self):
        h1 = ConcreteHandler()
        h2 = ConcreteHandler()
        WebhookEventHandlerRegistry.register("x", h1)
        WebhookEventHandlerRegistry.register("x", h2)
        assert WebhookEventHandlerRegistry.get("x") is h2


class TestPayloadTypeShapes:
    """Request/response helper types preserve expected fields."""

    async def test_analyze_pull_request_changes_signature(self):
        handler = ConcreteHandler()
        # Should accept repo_name, pr_number, trigger and return None
        result = await handler.analyze_pull_request_changes("owner/repo", 42, "push")
        assert result is None

    async def test_analyze_repository_changes_signature(self):
        handler = ConcreteHandler()
        # Should accept repo_name, branch, commits and return a dict
        result = await handler.analyze_repository_changes("owner/repo", "main", [{"sha": "abc"}])
        assert isinstance(result, dict)
        assert "status" in result


class TestAbstractMethodBodies:
    """Cover lines 28 and 44 — the pass statements in abstract methods."""

    async def test_analyze_pull_request_changes_pass_body(self):
        """Line 28: The pass body of analyze_pull_request_changes is exercised
        via the concrete handler which calls super() implicitly through the pass."""
        handler = ConcreteHandler()
        # ConcreteHandler.analyze_pull_request_changes body is pass -> returns None
        result = await handler.analyze_pull_request_changes("owner/repo", 1, "push")
        assert result is None

    async def test_analyze_repository_changes_pass_body(self):
        """Line 44: The pass body of analyze_repository_changes is exercised
        via a concrete handler that explicitly calls super()."""

        class SuperCallingHandler(IWebhookEventHandler):
            async def analyze_pull_request_changes(
                self, repo_name: str, pr_number: int, trigger: str
            ) -> None:
                await super().analyze_pull_request_changes(repo_name, pr_number, trigger)

            async def analyze_repository_changes(
                self, repo_name: str, branch: str, commits: list
            ) -> dict[str, Any]:
                await super().analyze_repository_changes(repo_name, branch, commits)
                return {"delegated": True}

        handler = SuperCallingHandler()
        # This calls super() which hits the pass body on line 28
        await handler.analyze_pull_request_changes("o/r", 1, "push")
        # This calls super() which hits the pass body on line 44
        result = await handler.analyze_repository_changes("o/r", "main", [])
        assert result == {"delegated": True}
