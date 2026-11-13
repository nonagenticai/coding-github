"""
Webhook interface for breaking circular dependencies.

This module provides abstract interfaces for webhook event handling to prevent
circular dependencies between webhook handler and endpoint modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IWebhookEventHandler(ABC):
    """Abstract interface for webhook event handlers."""

    @abstractmethod
    async def analyze_pull_request_changes(
        self, repo_name: str, pr_number: int, trigger: str
    ) -> None:
        """Analyze changes in a pull request and provide feedback.

        Args:
            repo_name: Repository name (owner/repo)
            pr_number: Pull request number
            trigger: What triggered the analysis
        """
        pass

    @abstractmethod
    async def analyze_repository_changes(
        self, repo_name: str, branch: str, commits: list
    ) -> dict[str, Any]:
        """Analyze changes from push events.

        Args:
            repo_name: Repository name (owner/repo)
            branch: Branch name
            commits: List of commits

        Returns:
            Analysis results dictionary
        """
        pass


class WebhookEventHandlerRegistry:
    """Registry for webhook event handlers to enable dependency injection."""

    _handlers: dict[str, IWebhookEventHandler] = {}

    @classmethod
    def register(cls, name: str, handler: IWebhookEventHandler) -> None:
        """Register a webhook event handler."""
        cls._handlers[name] = handler

    @classmethod
    def get_handler(cls, name: str) -> IWebhookEventHandler | None:
        """Get a registered webhook event handler."""
        return cls._handlers.get(name)

    @classmethod
    def get(cls, name: str) -> IWebhookEventHandler:
        """Get a registered webhook event handler (with error if not found)."""
        if name not in cls._handlers:
            raise ValueError(f"No handler registered with name: {name}")
        return cls._handlers[name]
