"""Intake capability probe: an under-privileged GitHub token must be visible at intake.

THE DEFECT (measured, 2026-08-15). ``validate_github_token`` is format-only -- prefix in
``ghp_/gho_/ghs_/github_pat_`` and it returns True. A well-formed but under-privileged PAT
therefore sailed through ``POST /api/projects``, and the system provisioned a board, six
lists, nine customFieldGroups and three tournament containers, ran the actual dependency
upgrade, and only THEN discovered the truth on its first write to GitHub::

    POST /repos/nonagenticai/enterprise_mcp_server/git/refs -> 403
    "Resource not accessible by personal access token"

Seven minutes and three containers late, with no record of why.

WHAT IS ASSERTED HERE. ``_record_token_capability`` runs on the repo object
``validate_repository_access`` has ALREADY fetched (no extra API call) and classifies the
token as write / read_only / unknown. It RECORDS, it does not reject -- provisioning serves
read-only analysis, and ``GitHubRepoRequest`` carries no authoring-intent field, so refusing
on ``push is False`` would block the majority case to catch the minority. The tests below pin
exactly that split, and pin that UNKNOWN never collapses into "fine".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.github_validation import CoreGitHubValidationService


@dataclass
class _RepoRequest:
    """Mirrors what the validation service reads off GitHubRepoRequest."""

    repo_url: str
    access_token: str | None = None
    branch: str = "main"
    pr_number: int | None = None


@pytest.fixture
def service() -> CoreGitHubValidationService:
    return CoreGitHubValidationService()


def _repo(permissions):
    """A stand-in for PyGithub's Repository, carrying only what the probe reads."""
    return SimpleNamespace(full_name="nonagenticai/enterprise_mcp_server", permissions=permissions)


def test_writable_token_is_classified_write(service, caplog):
    caplog.set_level(logging.INFO)
    result = service._record_token_capability(
        _repo(SimpleNamespace(push=True, pull=True)), "nonagenticai/enterprise_mcp_server"
    )
    assert result == CoreGitHubValidationService.CAPABILITY_WRITE
    assert "token_capability=write" in caplog.text
    # a healthy token must not emit the alarm
    assert "token_cannot_author" not in caplog.text


def test_read_only_token_is_named_at_intake(service, caplog):
    """THE DEFECT. push=False is exactly the 2026-08-15 canary token."""
    caplog.set_level(logging.INFO)
    result = service._record_token_capability(
        _repo(SimpleNamespace(push=False, pull=True)), "nonagenticai/enterprise_mcp_server"
    )
    assert result == CoreGitHubValidationService.CAPABILITY_READ_ONLY

    # the operator must be able to grep this, and it must name the repo and the 403 to come
    assert "token_cannot_author" in caplog.text
    assert "push=false" in caplog.text
    assert "nonagenticai/enterprise_mcp_server" in caplog.text
    assert "403" in caplog.text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize(
    "permissions",
    [
        None,  # no permissions block at all
        SimpleNamespace(pull=True),  # block present, push absent
        SimpleNamespace(push=None),  # push explicitly unknown
        object(),  # a bare object that answers nothing
    ],
    ids=["no-block", "push-absent", "push-none", "bare-object"],
)
def test_unverifiable_capability_is_unknown_never_fine(service, permissions, caplog):
    """An unreadable capability must NOT read as a healthy token.

    A stripped response or a degraded GitHub is not evidence of write access. Reporting
    these as ``write`` would be the silent-pass this probe exists to prevent.
    """
    caplog.set_level(logging.INFO)
    result = service._record_token_capability(_repo(permissions), "acme/widgets")

    assert result == CoreGitHubValidationService.CAPABILITY_UNKNOWN
    assert result != CoreGitHubValidationService.CAPABILITY_WRITE
    assert "token_capability=unknown" in caplog.text
    assert "token_capability=write" not in caplog.text


def test_probe_never_raises_and_never_blocks_provisioning(service, caplog):
    """A diagnostic that can break provisioning is worse than no diagnostic.

    The probe sits on the main provisioning path, so a hostile/broken repo object must
    degrade to UNKNOWN rather than propagate.
    """

    class _Exploding:
        @property
        def permissions(self):
            raise RuntimeError("github client blew up")

    caplog.set_level(logging.INFO)
    result = service._record_token_capability(_Exploding(), "acme/widgets")

    assert result == CoreGitHubValidationService.CAPABILITY_UNKNOWN
    assert "token_capability=unknown" in caplog.text


def test_read_only_does_not_reject_the_project(service):
    """Scope guard: this probe RECORDS, it does not raise.

    ``provision_repository`` provisions an ANALYSIS container and analysis needs read, not
    write. If someone later turns this into a rejection, they will break every legitimate
    read-only provision -- so pin the non-raising contract explicitly rather than leaving it
    implicit in the absence of a `pytest.raises`.
    """
    # would raise if the probe ever became a gate
    assert (
        service._record_token_capability(_repo(SimpleNamespace(push=False)), "acme/widgets")
        == CoreGitHubValidationService.CAPABILITY_READ_ONLY
    )


# ---------------------------------------------------------------------------
# WIRING. Everything above tests the helper in isolation, which means every one of
# those tests would still pass if the CALL SITE were deleted -- a check that answers
# a nearby question looks authoritative and proves nothing. This asserts the probe
# actually runs during validate_repository_access, on the repo object it already has.
# ---------------------------------------------------------------------------
async def test_probe_is_wired_into_validate_repository_access(caplog):
    """Delete the call in validate_repository_access and this test must fail."""
    mock_repo = MagicMock()
    mock_repo.full_name = "nonagenticai/enterprise_mcp_server"
    # the under-privileged PAT from the 2026-08-15 canary
    mock_repo.permissions = SimpleNamespace(push=False, pull=True)
    mock_repo.default_branch = "main"

    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.get_user.return_value = SimpleNamespace(login="georgiedekker")

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch.dict(os.environ, {}, clear=False),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
    ):
        for key in (
            "PYTEST_CURRENT_TEST",
            "TESTING",
            "TEST_MODE",
            "CI",
            "SCHEMATHESIS_TEST",
            "SCHEMATHESIS_TESTING",
        ):
            os.environ.pop(key, None)
        mock_auth.Token = MagicMock()

        svc = CoreGitHubValidationService()
        svc.test_mode = False

        caplog.set_level(logging.INFO)
        # This value is deliberately NOT a realistic PAT and deliberately carries no `ghp_`
        # prefix. detect-secrets flags a realistic one as a high-entropy string AND any
        # ghp_-prefixed one as a GitHub Token, reddening CI under a job name that says
        # nothing about secrets. The prefix is not needed here: this path calls
        # validate_repository_access directly, which never invokes the format-only
        # validate_github_token. It must also avoid the _is_test_token patterns
        # (test_/mock_/fake_/dummy_/...) -- those make _should_bypass_validation return
        # early, so get_repo never runs and the probe is never reached.
        request = _RepoRequest(
            repo_url="https://github.com/nonagenticai/enterprise_mcp_server",
            access_token="placeholder-pat-value-not-a-secret",
        )
        await svc.validate_repository_access(request)

    # provisioning still succeeded (the probe records, it does not gate) ...
    mock_g.get_repo.assert_called_once_with("nonagenticai/enterprise_mcp_server")
    # ... and the under-privileged token was named at intake
    assert "token_cannot_author" in caplog.text
    assert "nonagenticai/enterprise_mcp_server" in caplog.text


async def test_no_extra_github_api_call_for_the_probe():
    """The probe must read the ALREADY-FETCHED repo object, not re-query GitHub.

    Pins the cost claim: exactly one get_repo, and no additional client calls introduced
    by the capability check.
    """
    mock_repo = MagicMock()
    mock_repo.full_name = "acme/widgets"
    mock_repo.permissions = SimpleNamespace(push=True, pull=True)
    mock_repo.default_branch = "main"

    mock_g = MagicMock()
    mock_g.get_repo.return_value = mock_repo
    mock_g.get_user.return_value = SimpleNamespace(login="someone")

    with (
        patch("services.github_validation.load_feature_flags", return_value={}),
        patch.dict(os.environ, {}, clear=False),
        patch("services.github_validation.Github", return_value=mock_g),
        patch("services.github_validation.Auth") as mock_auth,
    ):
        for key in (
            "PYTEST_CURRENT_TEST",
            "TESTING",
            "TEST_MODE",
            "CI",
            "SCHEMATHESIS_TEST",
            "SCHEMATHESIS_TESTING",
        ):
            os.environ.pop(key, None)
        mock_auth.Token = MagicMock()

        svc = CoreGitHubValidationService()
        svc.test_mode = False
        await svc.validate_repository_access(
            _RepoRequest(
                repo_url="https://github.com/acme/widgets",
                access_token="placeholder-pat-value-not-a-secret",
            )
        )

    assert mock_g.get_repo.call_count == 1
