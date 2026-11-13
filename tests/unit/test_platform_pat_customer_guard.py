"""No PAT exists, so no PAT can touch any repo — customer or platform.

Owner ruling 2026-09-30: every GitHub personal-access-token fallback was removed.
GitHub access is the internal App (platform repos, platform TENANTS only) or the
tenant-bound public App (tests/unit/test_tenant_repo_token.py). The paths below
either carry no fr tenant at all (GitHub webhooks) or USED to fall back to the
PAT by construction; now they must SKIP or REFUSE before GitHub is contacted —
for a customer repo AND for our own org.

Every test arms the legacy env PATs, so a pass can never mean "there was simply
no credential to leak". Where a path still does real work once a credential is
supplied, a positive control stubs that credential in, so a guard that skipped
for some unrelated reason would also go red.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.github_integration as gi

PLATFORM = "nonagenticai"
PLATFORM_REPO = f"{PLATFORM}/coding-github"
CUSTOMER = "acme-corp"
CUSTOMER_REPO = f"{CUSTOMER}/widgets"
ENV_PAT = "ghp_ENV_PAT_must_not_leak"  # pragma: allowlist secret
TENANT_TOKEN = "ghs_tenant_scoped"  # pragma: allowlist secret
SKIP_LOG = "skipped: customer repo, no tenant context on webhook"
NO_CRED_LOG = "needs a GitHub credential; webhooks carry no tenant"
PR_ACTIONS_URL = "/api/webhooks/pr-actions"


@pytest.fixture(autouse=True)
def _platform_org_and_armed_pat(monkeypatch):
    monkeypatch.setattr(gi.settings, "system_github_owner", PLATFORM)
    for var in (
        "GITHUB_ACCESS_TOKEN",
        "GITHUB_TOKEN",
        "CODING_GITHUB_GITHUB_TOKEN",
        "CODING_GITHUB_SYSTEM_GITHUB_TOKEN",
    ):
        monkeypatch.setenv(var, ENV_PAT)


def _pr_event(repo: str, action: str = "opened") -> dict:
    return {"action": action, "pull_request": {"number": 7}, "repository": {"full_name": repo}}


# ---------------------------------------------------------------------------
# 1. LIVE path: secure webhook PR event -> analyze_pull_request_changes (PAT)
# ---------------------------------------------------------------------------
class TestSecureWebhookPullRequestEvent:
    @pytest.mark.parametrize("action", ["opened", "synchronize", "reopened"])
    async def test_customer_pr_event_is_skipped_before_github(self, action, caplog):
        from services.github.secure_webhook_handler import handle_pull_request_event

        analyzer = AsyncMock()
        pat_factory = AsyncMock()
        with (
            patch("routers.github_webhooks.analyze_pull_request_changes", analyzer),
            patch("services.github.github_client.get_github_service", pat_factory),
            caplog.at_level(logging.INFO),
        ):
            result = await handle_pull_request_event(_pr_event(CUSTOMER_REPO, action))

        assert result is True  # a deliberate skip, not a failure
        analyzer.assert_not_awaited()
        pat_factory.assert_not_called()
        assert SKIP_LOG in caplog.text
        assert CUSTOMER_REPO in caplog.text

    async def test_platform_pr_event_is_still_analyzed(self):
        """Positive control: the guard must not skip our own repos."""
        from services.github.secure_webhook_handler import handle_pull_request_event

        analyzer = AsyncMock()
        with patch("routers.github_webhooks.analyze_pull_request_changes", analyzer):
            result = await handle_pull_request_event(_pr_event(PLATFORM_REPO))

        assert result is True
        analyzer.assert_awaited_once_with(PLATFORM_REPO, 7, "opened")


# ---------------------------------------------------------------------------
# 1b. analyze_pull_request_changes itself (also reached by review comments)
# ---------------------------------------------------------------------------
class TestAnalyzePullRequestChanges:
    @pytest.mark.parametrize("repo", [CUSTOMER_REPO, PLATFORM_REPO])
    async def test_no_repo_builds_the_hub_without_a_credential(self, repo, caplog):
        """A webhook has no tenant for the App and no PAT exists: skip, for the
        platform org too (the env PATs are armed and must not be used)."""
        from routers import github_webhooks as gw

        hub_cls = MagicMock()
        post = AsyncMock()
        with (
            patch.object(gw, "GitHubIntegrationHub", hub_cls),
            patch.object(gw, "_post_pr_comment", post),
            caplog.at_level(logging.INFO),
        ):
            await gw.analyze_pull_request_changes(repo, 7, "comment_trigger")

        hub_cls.assert_not_called()
        post.assert_not_awaited()
        assert NO_CRED_LOG in caplog.text
        assert repo in caplog.text

    async def test_the_skip_is_the_credential_lookup(self):
        """Positive control: hand the path a credential and it analyzes — so the
        skip above is caused by the missing credential, not something else."""
        from routers import github_webhooks as gw

        hub_cls = MagicMock()
        hub_cls.return_value.analyze_pull_request = AsyncMock(
            return_value={"statistics": {}, "languages": {}}
        )
        post = AsyncMock()
        with (
            patch.object(gw, "_webhook_github_token", return_value="ghs_stub"),
            patch.object(gw, "GitHubIntegrationHub", hub_cls),
            patch.object(gw, "_post_pr_comment", post),
            patch.object(gw, "_generate_pr_analysis_summary", return_value="summary"),
        ):
            await gw.analyze_pull_request_changes(PLATFORM_REPO, 7, "opened")

        hub_cls.return_value.analyze_pull_request.assert_awaited_once_with(
            f"https://github.com/{PLATFORM_REPO}", 7
        )
        post.assert_awaited_once_with(PLATFORM_REPO, 7, "summary")

    @pytest.mark.parametrize("repo", [CUSTOMER_REPO, PLATFORM_REPO, None])
    def test_webhook_token_is_always_none(self, repo):
        from routers import github_webhooks as gw

        assert gw._webhook_github_token(repo, "any_feature") is None


# ---------------------------------------------------------------------------
# 1c. The sink: get_github_service refuses a customer repo, and has no
#     credential for a platform repo either
# ---------------------------------------------------------------------------
class TestGetGithubServiceSink:
    @pytest.mark.parametrize("repo_name", [CUSTOMER_REPO, "", "no-slash-owner-only"])
    async def test_non_platform_repo_is_refused_before_settings_are_read(self, repo_name):
        import services.github.github_service as gh

        settings = MagicMock(return_value=SimpleNamespace(is_production=False))
        service_cls = MagicMock()
        with (
            patch.object(gh, "get_settings", settings),
            patch.object(gh, "GitHubService", service_cls),
            pytest.raises(gh.PlatformPATRefusedError),
        ):
            await gh.get_github_service(repo_name)

        settings.assert_not_called()
        service_cls.assert_not_called()

    async def test_platform_repo_gets_no_pat(self, monkeypatch):
        """⛔ Was "platform repo gets the PAT". No PAT exists: with every legacy
        PAT armed, a platform repo on this tenantless path gets NO credential."""
        import services.github.github_service as gh

        monkeypatch.delenv("SCHEMATHESIS_TESTING", raising=False)
        service_cls = MagicMock()
        with (
            patch.object(gh, "get_settings", return_value=SimpleNamespace(is_production=False)),
            patch.object(gh, "GitHubService", service_cls),
            pytest.raises(ValueError, match="no PAT fallback"),
        ):
            await gh.get_github_service(PLATFORM_REPO)
        service_cls.assert_not_called()


# ---------------------------------------------------------------------------
# 1d. Push path: webhook_handler._process_commit_changes (PAT commit reads)
# ---------------------------------------------------------------------------
class TestProcessCommitChanges:
    async def test_customer_repo_is_skipped(self, caplog):
        from services.github.webhook_handler import GitHubWebhookHandler

        pat_factory = AsyncMock()
        with (
            patch("services.github.webhook_handler.get_github_service", pat_factory),
            caplog.at_level(logging.INFO),
        ):
            changes = await GitHubWebhookHandler()._process_commit_changes(CUSTOMER_REPO, ["sha1"])

        pat_factory.assert_not_called()
        assert changes == {"added": [], "modified": [], "deleted": []}
        assert SKIP_LOG in caplog.text

    async def test_platform_repo_reads_commits(self):
        from services.github.webhook_handler import GitHubWebhookHandler

        svc = AsyncMock()
        svc.get_commit_files = AsyncMock(return_value=[{"filename": "a.py", "status": "added"}])
        ctx = AsyncMock()
        ctx.__aenter__.return_value = svc
        pat_factory = AsyncMock(return_value=ctx)
        with patch("services.github.webhook_handler.get_github_service", pat_factory):
            changes = await GitHubWebhookHandler()._process_commit_changes(PLATFORM_REPO, ["sha1"])

        pat_factory.assert_awaited_once_with(PLATFORM_REPO)
        assert [str(p) for p in changes["added"]] == ["a.py"]


# ---------------------------------------------------------------------------
# 1e. Workflow dispatch: only a caller-passed token; no PAT fallback
# ---------------------------------------------------------------------------
class TestWorkflowTrigger:
    def _trigger(self, token=None):
        import services.github_actions_trigger as gat

        with patch.object(gat, "Github") as github_cls:
            return gat.GitHubActionsWorkflowTrigger(token), github_cls

    @pytest.mark.parametrize("owner", [CUSTOMER, PLATFORM])
    async def test_no_caller_token_refuses_every_dispatch(self, owner):
        """Was "PAT fallback refuses customer dispatch". There is no fallback: with
        the env PATs armed, a trigger without a caller token has no client at all
        and refuses for every owner, before GitHub is touched."""
        trigger, github_cls = self._trigger()
        github_cls.assert_not_called()
        assert trigger.github_token is None
        with pytest.raises(ValueError):
            await trigger.trigger_pr_comprehensive_validation(owner, "widgets", 7, "abc")
        with pytest.raises(ValueError):
            await trigger.trigger_workflow_via_api(owner, "widgets", "wf.yml", "main", {})

    def test_explicit_caller_token_is_the_only_credential(self):
        _, github_cls = self._trigger(token="ghs_caller_chosen")
        github_cls.assert_called_once_with("ghs_caller_chosen")


# ---------------------------------------------------------------------------
# 1f. Hub clone attaches NO credential, for any owner
# ---------------------------------------------------------------------------
class TestHubClone:
    @pytest.mark.parametrize(
        "url",
        [
            f"https://github.com/{CUSTOMER_REPO}",
            f"git@github.com:{CUSTOMER_REPO}.git",
            f"https://github.com/{PLATFORM_REPO}",
        ],
    )
    async def test_clone_never_attaches_a_token(self, url):
        auth = MagicMock()
        auth.get_token.return_value = ENV_PAT  # even if a token were somehow held
        repo_ops = MagicMock()
        repo_ops.clone_repository = AsyncMock()
        with (
            patch("services.github_integration_core.GitHubAuthManager", return_value=auth),
            patch("services.github_integration_core.RepositoryOperations", return_value=repo_ops),
        ):
            from services.github_integration_core import GitHubIntegrationHub

            await GitHubIntegrationHub().clone_repository(url)
        repo_ops.clone_repository.assert_awaited_once_with(url, None)


# ---------------------------------------------------------------------------
# 3c. pr_actions: both customer branches use ONLY the tenant-bound public App
# ---------------------------------------------------------------------------
_APPROVE = {
    "status": "success",
    "message": "done",
    "overall_score": 85.0,
    "passed": True,
    "validations": [],
    "recommendation": "approve",
}


def _approve_response():
    from schemas.github_api_models import ValidatePRResponse

    return ValidatePRResponse(**_APPROVE)


async def _post_customer(async_client, endpoint: str):
    if endpoint == "process-validation":
        body = {
            "repo_owner": CUSTOMER,
            "repo_name": "widgets",
            "pr_number": 10,
            "validation_response": _APPROVE,
            "config": {"add_labels": True, "post_comment": True},
        }
    else:  # validation-webhook, no github_token in the body
        body = {"repo_owner": CUSTOMER, "repo_name": "widgets", "pr_number": 10}
    return await async_client.post(f"{PR_ACTIONS_URL}/{endpoint}", json=body)


@pytest.fixture
def pr_actions_world(monkeypatch):
    import routers.pr_actions as pa

    with (
        patch("services.pr_validation.actions.pr_auto_closer.Github") as github_cls,
        patch.object(pa, "validate_pull_request", AsyncMock(return_value=_approve_response())),
    ):
        pr = MagicMock(number=10, labels=[])
        pr.get_commits.return_value = [MagicMock()]
        github_cls.return_value.get_repo.return_value.get_pull.return_value = pr
        yield SimpleNamespace(pa=pa, github_cls=github_cls)


class TestPrActionsCustomerBranches:
    @pytest.mark.parametrize("endpoint", ["process-validation", "validation-webhook"])
    async def test_customer_uses_the_tenant_bound_token_never_the_pat(
        self, endpoint, async_client, pr_actions_world
    ):
        from services.tenant_binding import PERMS_PR_ACTIONS

        resolve = MagicMock(return_value=SimpleNamespace(token=TENANT_TOKEN))
        with patch.object(pr_actions_world.pa, "resolve_customer_repo_token", resolve):
            resp = await _post_customer(async_client, endpoint)

        assert resp.status_code == 200, resp.text
        owner, repo, tenant, perms = resolve.call_args.args
        assert (owner, repo, perms) == (CUSTOMER, "widgets", PERMS_PR_ACTIONS)
        assert tenant  # the request's tenant, never None
        tokens = [c.args[0] for c in pr_actions_world.github_cls.call_args_list]
        assert tokens == [TENANT_TOKEN]
        assert ENV_PAT not in tokens

    @pytest.mark.parametrize("endpoint", ["process-validation", "validation-webhook"])
    async def test_customer_unbound_is_409_and_github_is_never_touched(
        self, endpoint, async_client, pr_actions_world
    ):
        from services.tenant_binding import NotInstalledForTenantError

        resolve = MagicMock(side_effect=NotInstalledForTenantError("not bound", owner=CUSTOMER))
        with patch.object(pr_actions_world.pa, "resolve_customer_repo_token", resolve):
            resp = await _post_customer(async_client, endpoint)

        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["details"]["owner"] == CUSTOMER
        pr_actions_world.github_cls.assert_not_called()
