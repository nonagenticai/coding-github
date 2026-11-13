"""
V2 GitHub Integration Endpoint

Modular implementation of GitHub repository operations.
This file now serves as the main router that includes specialized modules for different GitHub operations.

Modules:
- github_repository_management: Repository provisioning, listing, and status
- github_issue_operations: Issue create/list/get/comment/add-labels/close
- github_pull_request_operations: PR creation, validation, fork, and code changes
- github_repo_read_operations: get-repository, list-pull-request-files, list-workflow-runs,
  get-branch, compare
- github_repository_ops: File operations (already existing)
- github_webhooks: Webhook handling (already existing)
- github_authentication: Authentication and validation utilities
- github_api_integration: Common API integration utilities
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from routers.ci_wait_operations import router as ci_wait_router
from routers.github_issue_operations import (
    router as issue_operations_router,
)
from routers.github_pr_status_operations import (
    router as pr_status_router,
)
from routers.github_pull_request_operations import (
    router as pr_operations_router,
)
from routers.github_repo_read_operations import (
    router as repo_read_router,
)

# Import specialized GitHub operation modules
from routers.github_repository_management import (
    router as repository_router,
)
from routers.github_repository_ops import router as repository_ops_router
from routers.github_rust_target import router as rust_target_router
from routers.github_webhooks import router as webhooks_router

logger = logging.getLogger(__name__)
router = APIRouter()

# Include all specialized routers
router.include_router(
    repository_router,
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

router.include_router(
    issue_operations_router,
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

router.include_router(
    pr_operations_router,
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

router.include_router(
    repository_ops_router,
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

router.include_router(
    webhooks_router,
    prefix="/webhooks",
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

# All endpoints have been moved to specialized modules:
# - Repository management (provision, list, status) -> github_repository_management.py
# - Pull request operations (fork, apply-changes, create-pr, validate-pr) -> github_pull_request_operations.py
# - File operations (read, tree, directory, search) -> github_repository_ops.py (already existing)
# - Webhook handling -> github_webhooks.py (already existing)

# This file now serves as the main router that aggregates all GitHub functionality
# The original 1651 lines have been split into focused modules under 500 lines each

# Vulture whitelist for FastAPI route handlers (not directly called)
_vulture_whitelist: list[str] = []  # No direct endpoints in this file anymore

router.include_router(
    pr_status_router,
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

# Phase B reads (get-repository / list-pull-request-files / list-workflow-runs).
# 401 and 403 are declared DELIBERATELY: these routes keep GitHub's "cannot
# authenticate" and "may not read" distinct from "does not exist", and the n8n
# callers they replace branch on exactly that distinction.
router.include_router(
    repo_read_router,
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

# CI-checks wait (wait_for_pr_checks / get_ci_wait_status): register an n8n
# Wait-node resume URL, POSTed once when the PR head's checks have all finished.
router.include_router(
    ci_wait_router,
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"},
    },
)

# create_rust_target_repo: the Python→Rust TARGET repo (George 2026-11-03).
router.include_router(
    rust_target_router,
    # Per-status descriptions (with the machine error codes) live on the route.
    responses={500: {"description": "Internal Server Error"}},
)
