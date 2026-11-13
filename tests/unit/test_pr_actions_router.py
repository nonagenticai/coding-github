"""Tests for /api/webhooks/pr-actions/ endpoints via real HTTP requests.

Covers: process-validation happy path, missing auth, downstream failure, validation error.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


# ===========================================================================
# process-validation — success
# ===========================================================================


class TestProcessPRValidationSuccess:
    async def test_success(self, github_pr, async_client):
        """Successful processing with valid validation_response.

        Drives the REAL PRAutoCloser (only PyGithub is faked). Previously this
        mocked PRAutoCloser wholesale and asserted `labels_added is True`, which
        the handler read straight off the config — so the assertion held even
        when nothing was done to the PR. It now asserts the label/comment calls
        actually reached GitHub.
        """
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {
                    "status": "success",
                    "message": "done",
                    "overall_score": 85.0,
                    "passed": True,
                    "validations": [],
                    "recommendation": "approve",
                },
                "config": {"add_labels": True, "post_comment": True},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        # Ground truth: these calls really happened.
        github_pr.add_to_labels.assert_called_once()
        github_pr.create_issue_comment.assert_called_once()
        assert body["actions_taken"]["labels_added"] is True
        assert body["actions_taken"]["comment_posted"] is True
        # approve => neither branch of the close/request-changes if/elif ran.
        assert body["actions_taken"]["changes_requested"] is False
        assert body["actions_taken"]["pr_closed"] is False


# ===========================================================================
# process-validation — missing auth
# ===========================================================================


_APPROVE_BODY = {
    "repo_owner": "myorg",
    "repo_name": "myrepo",
    "pr_number": 10,
    "validation_response": {
        "status": "success",
        "message": "done",
        "overall_score": 85.0,
        "passed": True,
        "validations": [],
        "recommendation": "approve",
    },
}
_LEGACY_PAT = "ghp_legacy_pat_must_not_be_used"  # pragma: allowlist secret


def _arm_legacy_pats(monkeypatch):
    for var in (
        "GITHUB_ACCESS_TOKEN",
        "GITHUB_TOKEN",
        "CODING_GITHUB_GITHUB_TOKEN",
        "CODING_GITHUB_SYSTEM_GITHUB_TOKEN",
    ):
        monkeypatch.setenv(var, _LEGACY_PAT)


class TestProcessPRValidationMissingAuth:
    @pytest.mark.parametrize("path", ["process-validation", "validation-webhook"])
    async def test_app_not_configured_is_401_and_no_pat_is_used(
        self, monkeypatch, async_client, path
    ):
        """⛔ No PAT fallback (owner ruling 2026-09-30): with every legacy PAT var
        armed and the internal App NOT configured, the route fails closed."""
        import services.github_app_auth as ga

        _arm_legacy_pats(monkeypatch)
        monkeypatch.setattr(ga, "app_auth_configured", lambda: False)
        closer = MagicMock()
        validate = AsyncMock()
        with (
            patch("routers.pr_actions.PRAutoCloser", closer),
            patch("routers.pr_actions.validate_pull_request", validate),
        ):
            resp = await async_client.post(f"{BASE_URL}/{path}", json=_APPROVE_BODY)
        assert resp.status_code == 401, resp.text
        assert "GitHub token" in resp.json()["error"]["message"]
        closer.assert_not_called()
        validate.assert_not_awaited()

    async def test_app_mint_failure_is_401_and_no_pat_is_used(self, monkeypatch, async_client):
        """App configured but not installed/mint fails: still no PAT, still 401."""
        import services.github_app_auth as ga

        _arm_legacy_pats(monkeypatch)

        def _not_installed(_o, _r):
            raise ga.GitHubAppNotInstalledError("not installed")

        monkeypatch.setattr(ga, "get_installation_token_for", _not_installed)
        closer = MagicMock()
        with patch("routers.pr_actions.PRAutoCloser", closer):
            resp = await async_client.post(f"{BASE_URL}/process-validation", json=_APPROVE_BODY)
        assert resp.status_code == 401, resp.text
        closer.assert_not_called()

    async def test_platform_tenant_gets_the_internal_app_token(
        self, monkeypatch, async_client, internal_app
    ):
        """Positive control: the App configured → its token reaches the closer,
        never the armed legacy PAT."""
        _arm_legacy_pats(monkeypatch)
        closer = MagicMock()
        closer.return_value.process_validation_result = AsyncMock(
            side_effect=RuntimeError("stop after token")
        )
        with patch("routers.pr_actions.PRAutoCloser", closer):
            await async_client.post(f"{BASE_URL}/process-validation", json=_APPROVE_BODY)
        assert closer.call_args.args[0] == internal_app
        assert _LEGACY_PAT not in closer.call_args.args


# ===========================================================================
# process-validation — downstream failure
# ===========================================================================


class TestProcessPRValidationDownstreamFailure:
    @patch("routers.pr_actions.PRAutoCloser")
    async def test_downstream_failure(self, mock_autocloser_cls, async_client):
        """When PRAutoCloser raises, returns 500."""
        mock_instance = MagicMock()
        mock_instance.process_validation_result = AsyncMock(
            side_effect=RuntimeError("GitHub API down")
        )
        mock_autocloser_cls.return_value = mock_instance

        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {
                    "status": "success",
                    "message": "done",
                    "overall_score": 85.0,
                    "passed": True,
                    "validations": [],
                    "recommendation": "approve",
                },
            },
        )
        assert resp.status_code == 500


# ===========================================================================
# process-validation — validation error (missing fields)
# ===========================================================================


class TestProcessPRValidationValidationError:
    async def test_validation_error_missing_fields(self, async_client):
        """validation_response missing required fields returns 400."""
        resp = await async_client.post(
            f"{BASE_URL}/process-validation",
            json={
                "repo_owner": "myorg",
                "repo_name": "myrepo",
                "pr_number": 10,
                "validation_response": {"some_field": "value"},
            },
        )
        assert resp.status_code == 400


class TestPlatformRepoForbiddenToCustomerTenant:
    @pytest.mark.parametrize("path", ["process-validation", "validation-webhook"])
    async def test_customer_tenant_gets_403_and_no_credential_is_used(
        self, async_client, monkeypatch, path, enforce_platform_tenants
    ):
        """MUTANT: drop the platform-tenant gate (``check_platform_repo_tenant`` in
        ``resolve_github_token``) -> the App token is minted, not 403."""
        from coding_client.testing import override_tenant_auth

        from main import app

        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_platform_pat")  # pragma: allowlist secret
        closer = MagicMock()
        validate = AsyncMock()
        undo = override_tenant_auth(app, tenant_id="11111111-1111-1111-1111-111111111111")
        try:
            with (
                patch("routers.pr_actions.PRAutoCloser", closer),
                patch("routers.pr_actions.validate_pull_request", validate),
            ):
                resp = await async_client.post(
                    f"{BASE_URL}/{path}",
                    json={
                        "repo_owner": "myorg",
                        "repo_name": "myrepo",
                        "pr_number": 10,
                        "validation_response": {
                            "status": "success",
                            "message": "done",
                            "overall_score": 85.0,
                            "passed": True,
                            "validations": [],
                            "recommendation": "approve",
                        },
                        "config": {},
                    },
                )
        finally:
            undo()
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "platform_repo_forbidden"
        closer.assert_not_called()
        # Refused at the route, before any downstream work.
        validate.assert_not_awaited()

    async def test_observe_mode_proceeds_and_logs_would_refuse(
        self, async_client, monkeypatch, caplog, internal_app
    ):
        from coding_client.testing import override_tenant_auth

        from main import app

        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_platform_pat")  # pragma: allowlist secret
        caplog.set_level("WARNING", logger="services.tenant_binding")
        closer = MagicMock()
        undo = override_tenant_auth(app, tenant_id="11111111-1111-1111-1111-111111111111")
        try:
            with patch("routers.pr_actions.PRAutoCloser", closer):
                await async_client.post(
                    f"{BASE_URL}/process-validation",
                    json={
                        "repo_owner": "myorg",
                        "repo_name": "myrepo",
                        "pr_number": 10,
                        "validation_response": {
                            "status": "success",
                            "message": "done",
                            "overall_score": 85.0,
                            "passed": True,
                            "validations": [],
                            "recommendation": "approve",
                        },
                        "config": {},
                    },
                )
        finally:
            undo()
        # Observe mode proceeds: the internal App token (never the armed env
        # PAT) reached the closer.
        assert closer.call_args.args[0] == internal_app
        assert "ghp_platform_pat" not in closer.call_args.args
        lines = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("platform_repo_tenant_gate ")
        ]
        assert len(lines) == 1
        assert "decision=would_refuse" in lines[0]
        assert "caller_path=pr_actions_process_validation" in lines[0]


# ===========================================================================
# CODING_GITHUB_PLATFORM_REPO_ALLOWLIST: the demo tenant on its ONE fixture repo
# gets a one-repo App token with PERMS_PR_ACTIONS — never the env PAT.
# ===========================================================================
DEMO_TENANT = "0195f48b-d000-7000-8000-000000000103"
FIXTURE = "aa-bugfix-capstone-fixture"
_PAT = "ghp_platform_pat_must_not_reach_demo"  # pragma: allowlist secret
_VALIDATION = {
    "status": "success",
    "message": "done",
    "overall_score": 85.0,
    "passed": True,
    "validations": [],
    "recommendation": "approve",
}


@pytest.fixture
def demo_scoped(monkeypatch, enforce_platform_tenants):
    """demo → myorg/<fixture> granted; the App mint mocked at the HTTP boundary."""
    from types import SimpleNamespace

    from coding_client.testing import override_tenant_auth

    import services.github_app_auth as ga
    import services.tenant_binding as tb
    from main import app

    real = tb._settings_module.get_settings()
    monkeypatch.setattr(
        real, "_platform_repo_grants", {DEMO_TENANT: frozenset({f"myorg/{FIXTURE}"})}
    )
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", _PAT)
    monkeypatch.setattr(ga, "_resolve_installation_id", lambda o, r, app="internal": 99)
    monkeypatch.setattr(
        ga, "_get_app_auth", lambda app="internal": SimpleNamespace(create_jwt=lambda: "jwt")
    )
    post = MagicMock()
    post.return_value.status_code = 201
    post.return_value.json.return_value = {
        "token": "ghs_demo_pr_actions",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    monkeypatch.setattr(ga.httpx, "post", post)
    ga.reset_cache()
    undo = override_tenant_auth(app, tenant_id=DEMO_TENANT)
    yield post
    undo()
    ga.reset_cache()


class TestPlatformRepoAllowlistPrActions:
    @pytest.mark.parametrize("path", ["process-validation", "validation-webhook"])
    async def test_demo_fixture_uses_one_repo_app_token_never_the_pat(
        self, async_client, demo_scoped, path
    ):
        import services.github_app_auth as ga

        closer = MagicMock()
        closer.return_value.process_validation_result = AsyncMock(
            side_effect=RuntimeError("stop after token")
        )
        validate = AsyncMock(side_effect=RuntimeError("stop after token"))
        with (
            patch("routers.pr_actions.PRAutoCloser", closer),
            patch("routers.pr_actions.validate_pull_request", validate),
        ):
            await async_client.post(
                f"{BASE_URL}/{path}",
                json={
                    "repo_owner": "myorg",
                    "repo_name": FIXTURE,
                    "pr_number": 10,
                    "validation_response": _VALIDATION,
                    "config": {},
                },
            )
        assert demo_scoped.call_count == 1
        assert demo_scoped.call_args.kwargs["json"] == {
            "permissions": {"pull_requests": "write", "issues": "write"},
            "repositories": [FIXTURE],
        }
        assert ga._token_by_id == {}
        if path == "process-validation":
            assert closer.call_args.args[0] == "ghs_demo_pr_actions"
        else:
            # webhook: the token was resolved (and accepted) before validation ran.
            validate.assert_awaited_once()
        for call in closer.call_args_list:
            assert _PAT not in call.args

    @pytest.mark.parametrize("path", ["process-validation", "validation-webhook"])
    async def test_demo_other_platform_repo_is_403(self, async_client, demo_scoped, path):
        closer = MagicMock()
        with patch("routers.pr_actions.PRAutoCloser", closer):
            resp = await async_client.post(
                f"{BASE_URL}/{path}",
                json={
                    "repo_owner": "myorg",
                    "repo_name": "myrepo",
                    "pr_number": 10,
                    "validation_response": _VALIDATION,
                    "config": {},
                },
            )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "platform_repo_forbidden"
        closer.assert_not_called()
        demo_scoped.assert_not_called()

    @pytest.mark.parametrize("path", ["process-validation", "validation-webhook"])
    async def test_demo_fixture_mint_failure_is_502_not_the_pat(
        self, async_client, demo_scoped, path
    ):
        demo_scoped.return_value.status_code = 422
        closer = MagicMock()
        validate = AsyncMock()
        with (
            patch("routers.pr_actions.PRAutoCloser", closer),
            patch("routers.pr_actions.validate_pull_request", validate),
        ):
            resp = await async_client.post(
                f"{BASE_URL}/{path}",
                json={
                    "repo_owner": "myorg",
                    "repo_name": FIXTURE,
                    "pr_number": 10,
                    "validation_response": _VALIDATION,
                    "config": {},
                },
            )
        assert resp.status_code == 502, resp.text
        assert resp.json()["error"]["details"]["error_code"] == "platform_repo_scoped_mint_failed"
        closer.assert_not_called()
        validate.assert_not_awaited()
