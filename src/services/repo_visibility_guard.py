"""Public-repo guard: a PUBLIC repository is never a conveyor write target.

Operator ruling (George; n8n_custom_nodes AA_IMPLEMENTATION/TENANCY_AND_IMPROVEMENT.md
§8): conveyor output — commits, pull requests, merges — never lands in a public
repository. Seeded-defect fixtures and process data once sat on a public ``main`` for
months; a refusal at the service boundary is what makes that impossible rather than
merely discouraged.

ONE function decides: :func:`assert_repo_not_public`. It runs before ANY GitHub write on
``apply-code-changes``, ``create-pull-request``, ``create-upstream-pr`` and
``merge-pull-request``.

Two refusals, each with its own machine code (read off ``error.details.error_code``):

* ``public_repo_refused`` (403) — GitHub says ``private: false``.
* ``repo_visibility_unknown`` (503) — the lookup raised, or ``private`` is anything other
  than a real ``bool`` (absent, ``None``, a lazy attribute that failed to complete).
  ⛔ FAIL CLOSED: an unknown visibility is NEVER read as private. Treating "could not
  tell" as "allowed" is the dangerous direction of wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coding_client.models.responses import error_response
from fastapi import HTTPException

PUBLIC_REPO_REFUSED = "public_repo_refused"
REPO_VISIBILITY_UNKNOWN = "repo_visibility_unknown"


class RepoVisibilityRefusedError(Exception):
    """The target repo is public, or its visibility could not be established."""

    def __init__(self, error_code: str, repo: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.repo = repo
        self.status_code = status_code


def _unknown(repo: str, why: str) -> RepoVisibilityRefusedError:
    return RepoVisibilityRefusedError(
        REPO_VISIBILITY_UNKNOWN,
        repo,
        f"Could not establish the visibility of {repo} ({why}); refusing to write — "
        "a public repository is never a conveyor target.",
        503,
    )


def assert_repo_not_public(repo_obj: Any, repo: str) -> None:
    """Raise unless ``repo_obj.private`` is exactly ``True``.

    ``repo_obj`` is the repository metadata the caller already fetched (a PyGithub
    ``Repository``). Reading ``.private`` on a lazily-completed object can itself call
    GitHub, so a raise there is an unknown visibility, not a crash.
    """
    try:
        private = repo_obj.private
    except Exception as e:
        raise _unknown(repo, f"reading 'private' failed: {e}") from e
    if private is True:
        return
    if private is False:
        raise RepoVisibilityRefusedError(
            PUBLIC_REPO_REFUSED,
            repo,
            f"{repo} is a PUBLIC repository; a public repository is never a conveyor "
            "target (commits, pull requests and merges are refused).",
            403,
        )
    raise _unknown(repo, f"'private' is {private!r}, not a bool")


def assert_repo_not_public_lookup(fetch: Callable[[], Any], repo: str) -> None:
    """Fetch the repo metadata once via ``fetch`` and apply :func:`assert_repo_not_public`.

    Any error from the lookup is an unknown visibility (503), never a pass.
    """
    try:
        repo_obj = fetch()
    except Exception as e:
        raise _unknown(repo, f"lookup failed: {e}") from e
    assert_repo_not_public(repo_obj, repo)


def visibility_refused_http(e: RepoVisibilityRefusedError) -> HTTPException:
    """403/503 with the code repeated in ``details`` (coding-client's shared handler
    keeps only message + details on the wire)."""
    return error_response(
        status_code=e.status_code,
        message=str(e),
        error_code=e.error_code,
        details={"error_code": e.error_code, "repo": e.repo},
    )
