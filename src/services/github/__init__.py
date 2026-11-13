"""GitHub integration package.

This package provides GitHub API integration functionality,
merging core GitHub operations and service-layer GitHub functionality.
"""

# Core GitHub modules
from .auth_manager import GitHubAuthManager
from .github_client import GitHubClient, PRAnalysisDict

# Service-layer GitHub modules
from .github_service import (
    GitHubConfig,
    GitHubService,
    RateLimitInfo,
    get_github_service,
)
from .repo_operations import RepositoryOperations
from .secure_token_manager import (
    SecureTokenManager,
    SecureTokenMetadata,
    TokenScope,
    TokenStatus,
    create_default_token,
    get_secure_token_manager,
)
from .secure_webhook_handler import (
    SecureWebhookHandler,
    WebhookProcessingResult,
    WebhookSecurityLevel,
    get_secure_webhook_handler,
    handle_issues_event,
    handle_pull_request_event,
    handle_push_event,
)
from .webhook_handler import GitHubWebhookHandler

__all__ = [
    # Core
    "GitHubAuthManager",
    "GitHubClient",
    "PRAnalysisDict",
    "RepositoryOperations",
    "GitHubWebhookHandler",
    # Service
    "GitHubConfig",
    "GitHubService",
    "RateLimitInfo",
    "get_github_service",
    "SecureTokenManager",
    "SecureTokenMetadata",
    "TokenScope",
    "TokenStatus",
    "create_default_token",
    "get_secure_token_manager",
    "SecureWebhookHandler",
    "WebhookProcessingResult",
    "WebhookSecurityLevel",
    "get_secure_webhook_handler",
    "handle_issues_event",
    "handle_pull_request_event",
    "handle_push_event",
]
