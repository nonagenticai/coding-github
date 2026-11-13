"""`actions_taken` must report what HAPPENED, not what was configured.

The old handler built `actions_taken` from the request's config flags
(`config.get("add_labels", True)` etc.) and ignored the auto-closer's result
entirely — `process_validation_result` returned None and swallowed every
exception. Two consequences, both tested here:

1. Even on a fully successful run the payload lied about which branch was
   taken: `changes_requested` and `pr_closed` were computed from
   `config AND recommendation == "reject"`, but the real code path is an
   if/elif — a PR that gets auto-closed never has changes requested, and a
   "reject" without critical issues is never closed.
2. A total no-op (GitHub unreachable, comment API down) still returned
   HTTP 200 / status "success" with every flag set to True.

Every test in this file drives the REAL PRAutoCloser via the `github_pr`
fixture (only PyGithub itself is faked), so the assertions are about genuine
call outcomes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from schemas.github_api_models import (
    AnalysisIssue,
    PRValidationResult,
    ValidatePRResponse,
)


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch, platform_tenant, internal_app):
    """The fixture owner "myorg" is declared the platform org, and the internal
    GitHub App is configured: since 2026-09-30 it is the ONLY credential for a
    platform repo (no PAT fallback). Customer owners are tenant-bound public-App
    only (tests/unit/test_tenant_repo_token.py).
    """
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "myorg")


BASE_URL = "/api/webhooks/pr-actions"


def _error_details(resp) -> dict:
    """Unwrap the service-wide error envelope: {"error": {..., "details": {...}}}."""
    error = resp.json()["error"]
    assert error["code"] == f"HTTP_{resp.status_code}"
    return error["details"]


# ---------------------------------------------------------------------------
# Builders — real schema objects, never a bare MagicMock. A double narrower
# than the real model cannot fail on a field the production code reads.
# ---------------------------------------------------------------------------


def _validation_payload(
    *,
    recommendation: str,
    overall_score: float,
    critical: bool = False,
) -> dict:
    """A wire-format validation_response that round-trips through the schema."""
    issues = []
    if critical:
        issues.append(
            AnalysisIssue(
                type="security",
                severity="critical",
                message="SQL injection",
                line=42,
                file="db.py",
            ).model_dump()
        )
    return {
        "status": "success",
        "message": "done",
        "overall_score": overall_score,
        "passed": recommendation == "approve",
        "validations": [
            PRValidationResult(
                validation_type="security",
                passed=not critical,
                score=overall_score,
                issues=[AnalysisIssue(**i) for i in issues],
                suggestions=["Use parameterized queries"],
            ).model_dump()
        ],
        "recommendation": recommendation,
    }


def _validation_response(*, recommendation: str, overall_score: float) -> ValidatePRResponse:
    """A real ValidatePRResponse for stubbing validate_pull_request."""
    return ValidatePRResponse(
        **_validation_payload(recommendation=recommendation, overall_score=overall_score)
    )


# ===========================================================================
# 1. Truthful reporting on the SUCCESS path (HTTP 200)
#
# These are the sharpest proofs: nothing fails, the status stays 200, but the
# old config-derived payload still disagreed with reality.
# ===========================================================================


class TestActionsTakenReflectsTheBranchActuallyTaken:
    async def test_closed_pr_does_not_claim_changes_requested(self, github_pr, async_client):
        """A PR that gets auto-closed never has changes requested (if/elif).

        OLD: changes_requested = config["request_changes"] and rec == "reject"
             => True, while create_review was never called.
        """
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="reject", overall_score=5.0, critical=True
                ),
                "config": {
                    "auto_close_enabled": True,
                    "add_labels": True,
                    "post_comment": True,
                    "request_changes": True,
                },
            },
        )

        assert resp.status_code == 200
        actions = resp.json()["actions_taken"]

        # Ground truth: the PR was closed and NO review was submitted.
        github_pr.edit.assert_called_once_with(state="closed")
        github_pr.create_review.assert_not_called()

        assert actions["pr_closed"] is True
        assert actions["changes_requested"] is False  # RED on old code (was True)

    async def test_reject_without_critical_issues_does_not_claim_pr_closed(
        self, github_pr, async_client
    ):
        """recommendation="reject" alone does not close a PR.

        _should_close_pr also requires a critical issue or score < 20. With
        score 50 and no critical issues the PR stays open and changes are
        requested instead.

        OLD: pr_closed = config["auto_close_enabled"] and rec == "reject"
             => True, while pr.edit was never called.
        """
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="reject", overall_score=50.0, critical=False
                ),
                "config": {
                    "auto_close_enabled": True,
                    "add_labels": True,
                    "post_comment": True,
                    "request_changes": True,
                },
            },
        )

        assert resp.status_code == 200
        actions = resp.json()["actions_taken"]

        # Ground truth: PR untouched by close, review WAS submitted.
        github_pr.edit.assert_not_called()
        github_pr.create_review.assert_called_once()

        assert actions["pr_closed"] is False  # RED on old code (was True)
        assert actions["changes_requested"] is True

    async def test_disabled_actions_are_reported_false_and_not_performed(
        self, github_pr, async_client
    ):
        """Actions switched off in config are neither performed nor claimed."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="approve", overall_score=90.0
                ),
                "config": {
                    "add_labels": False,
                    "post_comment": False,
                    "request_changes": False,
                    "auto_close_enabled": False,
                },
            },
        )

        assert resp.status_code == 200
        actions = resp.json()["actions_taken"]

        github_pr.add_to_labels.assert_not_called()
        github_pr.create_issue_comment.assert_not_called()

        assert actions == {
            "labels_added": False,
            "comment_posted": False,
            "changes_requested": False,
            "changes_request_degraded": None,
            "pr_closed": False,
        }


# ===========================================================================
# 2. A failed action must NOT be laundered into a 200
# ===========================================================================


class TestFailedActionsAreVisibleToTheCaller:
    async def test_comment_failure_returns_502_not_success(self, github_pr, async_client):
        """A GitHub comment failure used to be swallowed and reported as done.

        OLD: _post_validation_comment caught the exception, returned None, and
             the handler answered 200 with comment_posted=True.
        """
        github_pr.create_issue_comment.side_effect = RuntimeError("Comment API down")

        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="approve", overall_score=90.0
                ),
                "config": {"add_labels": True, "post_comment": True},
            },
        )

        assert resp.status_code == 502  # RED on old code (was 200)
        details = _error_details(resp)
        assert any("Comment API down" in e for e in details["errors"])
        # Partial success is still reported truthfully.
        assert details["actions_taken"]["comment_posted"] is False
        assert details["actions_taken"]["labels_added"] is True

    async def test_label_failure_returns_502_not_success(self, github_pr, async_client):
        """A GitHub label failure must not be reported as labels_added=True."""
        github_pr.add_to_labels.side_effect = RuntimeError("Label API down")

        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="approve", overall_score=90.0
                ),
                "config": {"add_labels": True, "post_comment": False},
            },
        )

        assert resp.status_code == 502  # RED on old code (was 200)
        details = _error_details(resp)
        assert details["actions_taken"]["labels_added"] is False
        assert any("Label API down" in e for e in details["errors"])

    async def test_total_no_op_is_not_reported_as_success(self, github_pr, async_client):
        """GitHub unreachable => nothing happened at all => must not be a 200.

        OLD: the outer `except Exception` logged and returned None, and the
             handler answered 200 with labels_added/comment_posted True.
        """
        github_pr.github_client.get_repo.side_effect = RuntimeError("GitHub 503")

        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="approve", overall_score=90.0
                ),
                "config": {"add_labels": True, "post_comment": True},
            },
        )

        assert resp.status_code == 502  # RED on old code (was 200)
        details = _error_details(resp)
        assert any("GitHub 503" in e for e in details["errors"])
        # Nothing reached GitHub, so every flag must be False.
        assert details["actions_taken"] == {
            "labels_added": False,
            "comment_posted": False,
            "changes_requested": False,
            "changes_request_degraded": None,
            "pr_closed": False,
        }


# ===========================================================================
# 3. Same contract on /validation-webhook
# ===========================================================================


class TestValidationWebhookReportsTruthfully:
    @pytest.fixture
    def _approve(self):
        with patch("routers.pr_actions.validate_pull_request", new_callable=AsyncMock) as mock:
            mock.return_value = _validation_response(recommendation="approve", overall_score=85.0)
            yield mock

    @pytest.fixture
    def _reject_no_critical(self):
        with patch("routers.pr_actions.validate_pull_request", new_callable=AsyncMock) as mock:
            mock.return_value = _validation_response(recommendation="reject", overall_score=50.0)
            yield mock

    async def test_label_failure_returns_502(self, _approve, github_pr, async_client):
        """OLD: labels_updated came straight from the hardcoded config => True."""
        github_pr.add_to_labels.side_effect = RuntimeError("Label API down")

        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "github_token": "ghp_test_token",
            },
        )

        assert resp.status_code == 502  # RED on old code (was 200)
        details = _error_details(resp)
        assert details["actions_taken"]["labels_updated"] is False
        assert details["actions_taken"]["validation_performed"] is True

    async def test_reject_without_critical_does_not_claim_pr_closed(
        self, _reject_no_critical, github_pr, async_client
    ):
        """OLD: pr_closed = auto_close_enabled and rec == "reject" => True."""
        resp = await async_client.post(
            f"{BASE_URL}/validation-webhook",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 5,
                "auto_close_enabled": True,
                "github_token": "ghp_test_token",
            },
        )

        assert resp.status_code == 200
        actions = resp.json()["actions_taken"]

        github_pr.edit.assert_not_called()
        github_pr.create_review.assert_called_once()

        assert actions["pr_closed"] is False  # RED on old code (was True)
        assert actions["changes_requested"] is True
        assert actions["labels_updated"] is True
        assert actions["comment_posted"] is True


# ===========================================================================
# 4. A partial config must not silently disable every action
# ===========================================================================


class TestPartialConfigDoesNotNoOp:
    async def test_partial_config_still_performs_defaults(self, github_pr, async_client):
        """`config` is arbitrary caller JSON; a missing key used to KeyError.

        OLD: PRAutoCloser stored the raw dict, so `self.config["add_labels"]`
             raised KeyError, the outer handler swallowed it, and the endpoint
             returned 200 having done NOTHING. Config is now merged over the
             defaults.
        """
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="approve", overall_score=90.0
                ),
                "config": {"post_comment": True},
            },
        )

        assert resp.status_code == 200
        # RED on old code: KeyError("add_labels") => nothing was ever called.
        github_pr.create_issue_comment.assert_called_once()
        github_pr.add_to_labels.assert_called_once()
        assert resp.json()["actions_taken"]["comment_posted"] is True


# ===========================================================================
# 5. The review body must contain the real score
# ===========================================================================


class TestRequestChangesBodyIsInterpolated:
    async def test_review_body_contains_actual_score(self, github_pr, async_client):
        """The review body was a plain string with an un-substituted placeholder."""
        await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(
                    recommendation="reject", overall_score=50.0
                ),
                "config": {
                    "add_labels": False,
                    "post_comment": False,
                    "request_changes": True,
                    "auto_close_enabled": False,
                },
            },
        )

        github_pr.create_review.assert_called_once()
        body = github_pr.create_review.call_args.kwargs["body"]
        assert "50.0/100" in body
        assert "{validation_response" not in body  # RED on old code


# ===========================================================================
# 5. The App must not REQUEST_CHANGES on its own PR (GitHub 422)
#
# Since 2026-09-30 /process-validation acts as the internal GitHub App, and the
# conveyor's PRs are App-authored. GitHub refuses a REQUEST_CHANGES review from
# a PR's own author with 422 "Can not request changes on your own pull request";
# that used to surface as a 502 PR_ACTIONS_FAILED after labels + comment landed.
# ===========================================================================

APP_BOT_LOGIN = "nonagentic-internal[bot]"
_REJECT_NO_CRITICAL = {"recommendation": "reject", "overall_score": 50.0, "critical": False}


def _own_pr_422():
    from github import GithubException

    return GithubException(
        422,
        {
            "message": "Unprocessable Entity",
            "errors": ["Can not request changes on your own pull request"],
        },
        None,
    )


def _author(github_pr, login: str, user_type: str) -> None:
    github_pr.user.login = login
    github_pr.user.type = user_type


def _comment_author(github_pr, login: str) -> None:
    """The validation comment we post comes back authored by the acting identity."""
    github_pr.create_issue_comment.return_value.user.login = login


class TestSelfAuthoredPRIsNotReviewedRequestChanges:
    async def _post(self, async_client, config: dict):
        return await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": _validation_payload(**_REJECT_NO_CRITICAL),
                "config": config,
            },
        )

    async def test_app_authored_pr_with_comment_skips_review(self, github_pr, async_client):
        _author(github_pr, APP_BOT_LOGIN, "Bot")
        _comment_author(github_pr, APP_BOT_LOGIN)

        resp = await self._post(async_client, {"post_comment": True, "request_changes": True})

        assert resp.status_code == 200
        github_pr.create_review.assert_not_called()
        actions = resp.json()["actions_taken"]
        assert actions["comment_posted"] is True
        assert actions["changes_requested"] is False
        assert actions["changes_request_degraded"] == (
            "self_authored_pr:review_skipped_comment_already_posted"
        )

    async def test_own_pr_without_comment_attempts_request_changes_then_degrades_on_422(
        self, github_pr, async_client
    ):
        """No comment → acting login unknown → nothing presumed. REQUEST_CHANGES is
        attempted; GitHub's 422 "own pull request" is what degrades it."""
        _author(github_pr, APP_BOT_LOGIN, "Bot")
        github_pr.create_review.side_effect = [_own_pr_422(), None]

        resp = await self._post(async_client, {"post_comment": False, "request_changes": True})

        assert resp.status_code == 200
        events = [c.kwargs["event"] for c in github_pr.create_review.call_args_list]
        assert events == ["REQUEST_CHANGES", "COMMENT"]
        assert "Validation Score:** 50.0/100" in github_pr.create_review.call_args.kwargs["body"]
        actions = resp.json()["actions_taken"]
        assert actions["changes_requested"] is False
        assert actions["changes_request_degraded"] == "own_pr_422:comment_review_posted"

    async def test_dependabot_pr_without_comment_gets_request_changes(
        self, github_pr, async_client
    ):
        """A [bot] author is NOT presumed to be us: dependabot's PR gets REQUEST_CHANGES."""
        _author(github_pr, "dependabot[bot]", "Bot")

        resp = await self._post(async_client, {"post_comment": False, "request_changes": True})

        assert resp.status_code == 200
        github_pr.create_review.assert_called_once()
        assert github_pr.create_review.call_args.kwargs["event"] == "REQUEST_CHANGES"
        actions = resp.json()["actions_taken"]
        assert actions["changes_requested"] is True
        assert actions["changes_request_degraded"] is None

    async def test_human_authored_pr_still_gets_request_changes(self, github_pr, async_client):
        _author(github_pr, "some-human", "User")
        _comment_author(github_pr, APP_BOT_LOGIN)

        resp = await self._post(async_client, {"post_comment": True, "request_changes": True})

        assert resp.status_code == 200
        github_pr.create_review.assert_called_once()
        assert github_pr.create_review.call_args.kwargs["event"] == "REQUEST_CHANGES"
        actions = resp.json()["actions_taken"]
        assert actions["changes_requested"] is True
        assert actions["changes_request_degraded"] is None

    async def test_other_bots_pr_gets_request_changes_when_our_login_is_known(
        self, github_pr, async_client
    ):
        """Our comment names us, so a DIFFERENT bot's PR is not mistaken for ours."""
        _author(github_pr, "dependabot[bot]", "Bot")
        _comment_author(github_pr, APP_BOT_LOGIN)

        resp = await self._post(async_client, {"post_comment": True, "request_changes": True})

        assert resp.status_code == 200
        assert github_pr.create_review.call_args.kwargs["event"] == "REQUEST_CHANGES"
        actions = resp.json()["actions_taken"]
        assert actions["changes_requested"] is True
        assert actions["changes_request_degraded"] is None

    async def test_own_pr_422_is_degraded_not_502(self, github_pr, async_client):
        """Detection missed it (author not recognisable) and GitHub said 422 own-PR."""
        _author(github_pr, "unrecognisable", "User")
        github_pr.create_review.side_effect = [_own_pr_422(), None]

        resp = await self._post(async_client, {"post_comment": False, "request_changes": True})

        assert resp.status_code == 200  # was 502 PR_ACTIONS_FAILED
        events = [c.kwargs["event"] for c in github_pr.create_review.call_args_list]
        assert events == ["REQUEST_CHANGES", "COMMENT"]
        actions = resp.json()["actions_taken"]
        assert actions["changes_requested"] is False
        assert actions["changes_request_degraded"] == "own_pr_422:comment_review_posted"

    async def test_other_422_is_still_a_502(self, github_pr, async_client):
        from github import GithubException

        _author(github_pr, "some-human", "User")
        github_pr.create_review.side_effect = GithubException(
            422, {"message": "Validation Failed"}, None
        )

        resp = await self._post(async_client, {"post_comment": False, "request_changes": True})

        assert resp.status_code == 502
        details = _error_details(resp)
        assert details["actions_taken"]["changes_requested"] is False
        assert details["actions_taken"]["changes_request_degraded"] is None
