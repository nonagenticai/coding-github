"""Interfaces for breaking circular dependencies."""

from interfaces.webhook_interface import IWebhookEventHandler, WebhookEventHandlerRegistry

__all__ = ["IWebhookEventHandler", "WebhookEventHandlerRegistry"]
