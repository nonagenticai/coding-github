"""Extended tests for uncovered lines in github_webhooks.py.

Covers lines: 109-110, 163-165, 181, 318-319, 382, 405-413, 442-449, 689-690.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fixture_owner_is_the_platform_org(monkeypatch):
    """The fixture owner "owner" is declared the platform org here."""
    import services.github_integration as gi

    monkeypatch.setattr(gi.settings, "system_github_owner", "owner")


@pytest.fixture
def webhook_token(monkeypatch):
    """Stub the webhook credential lookup so the post-credential code is reachable.

    Production webhooks have NO credential since 2026-09-30 (no PAT exists and a
    webhook carries no tenant for the GitHub App), so ``_webhook_github_token``
    returns ``None`` and every feature skips; these tests cover what would run
    after it.
    """
    import routers.github_webhooks as gw

    monkeypatch.setattr(gw, "_webhook_github_token", lambda _repo, _feature: "tok")


os.environ["ENABLE_AUTH"] = "false"


def _make_settings(**overrides):
    """Create a settings mock with sensible defaults, applying overrides."""
    s = MagicMock()
    s.GITHUB_WEBHOOK_SECRET = "secret"
    s.coding_gateway_url = overrides.get("coding_gateway_url", "http://gw:8687")
    return s


def _make_mock_pr():
    """Create a fully stubbed mock PR object for PyGithub."""
    pr = MagicMock()
    pr.title = "T"
    pr.body = "B"
    pr.user = MagicMock(login="u")
    pr.base = MagicMock(sha="b", ref="main")
    pr.head = MagicMock(sha="h", ref="feat")
    pr.created_at = None
    pr.updated_at = None
    pr.get_files = MagicMock(return_value=[])
    pr.create_issue_comment = MagicMock()
    return pr


def _patch_github_client(mock_pr):
    """Return a patch context manager for github.Github returning a client with mock_pr."""
    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr
    mock_client = MagicMock()
    mock_client.get_repo.return_value = mock_repo
    return patch("github.Github", return_value=mock_client)


def _patch_orchestrator(return_value):
    """Return a patch context manager for the PR validation orchestrator."""
    orch = AsyncMock()
    orch.validate_pr_with_actions = AsyncMock(return_value=return_value)
    return patch(
        "services.pr_validation.pr_validation_orchestrator.get_pr_validation_orchestrator",
        return_value=orch,
    ), orch


def _pr_payload(action="opened", pr_number=1):
    return {
        "action": action,
        "pull_request": {
            "number": pr_number,
            "title": "T",
            "user": {"login": "u"},
            "base": {"sha": "b", "ref": "main"},
            "head": {"sha": "h", "ref": "feat"},
        },
        "repository": {"full_name": "owner/repo"},
    }


# =============================================================================
# No webhook credential exists: early return before GitHub
# =============================================================================


@pytest.mark.unit
class TestPREventNoTokenEarlyReturn:
    """Early return: a webhook has no GitHub credential (no PAT, no tenant)."""

    async def test_pr_opened_without_credential_returns_early(self, monkeypatch):
        """With the legacy env PATs armed, handle_pull_request_event still returns
        before GitHub or the orchestrator is touched."""
        from routers.github_webhooks import handle_pull_request_event

        monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "ghp_env_pat")
        monkeypatch.setenv("CODING_GITHUB_GITHUB_TOKEN", "ghp_env_pat")
        mock_pr = _make_mock_pr()

        with _patch_github_client(mock_pr) as github_cls:
            orch_patch, orch = _patch_orchestrator({})
            with orch_patch:
                await handle_pull_request_event(_pr_payload())
                orch.validate_pr_with_actions.assert_not_called()
        github_cls.assert_not_called()


# =============================================================================
# Lines 163-165: fast validation results iteration (passed/failed per result)
# =============================================================================


@pytest.mark.unit
class TestFastValidationResults:
    """Cover lines 163-165: iterating fast_validation results with passed/failed."""

    async def test_fast_validation_results_passed_and_failed(self, webhook_token):
        """When fast_validation has results, each is iterated and PASS/FAIL status is set."""
        from routers.github_webhooks import handle_pull_request_event

        passed_result = MagicMock(passed=True, validation_type="lint", score=95.0)
        failed_result = MagicMock(passed=False, validation_type="security", score=30.0)

        settings = _make_settings()
        mock_pr = _make_mock_pr()

        validation_return = {
            "validation_id": "v1",
            "fast_validation": {
                "passed": False,
                "score": 60,
                "results": [passed_result, failed_result],
            },
            "comprehensive_validation": {"status": "triggered", "message": "go"},
        }

        with patch("config.settings.get_settings", return_value=settings):
            with _patch_github_client(mock_pr):
                orch_patch, _ = _patch_orchestrator(validation_return)
                with orch_patch:
                    await handle_pull_request_event(_pr_payload())

                    posted = mock_pr.create_issue_comment.call_args.args[0]
                    assert "PASS **lint**" in posted
                    assert "FAIL **security**" in posted
                    assert "95.0/100" in posted
                    assert "30.0/100" in posted


# =============================================================================
# Line 181: comprehensive validation status "else" branch
# =============================================================================


@pytest.mark.unit
class TestComprehensiveValidationElse:
    """Cover line 181: comp_status not 'triggered' and not 'not_triggered'."""

    async def test_comprehensive_validation_unknown_status(self, webhook_token):
        """When comp_status is something unexpected, the else branch prints it."""
        from routers.github_webhooks import handle_pull_request_event

        settings = _make_settings()
        mock_pr = _make_mock_pr()

        validation_return = {
            "validation_id": "v2",
            "fast_validation": {"passed": True, "score": 100, "results": []},
            "comprehensive_validation": {"status": "error_state"},
        }

        with patch("config.settings.get_settings", return_value=settings):
            with _patch_github_client(mock_pr):
                orch_patch, _ = _patch_orchestrator(validation_return)
                with orch_patch:
                    await handle_pull_request_event(_pr_payload("synchronize", 5))

                    posted = mock_pr.create_issue_comment.call_args.args[0]
                    assert "error_state" in posted


# =============================================================================
# Lines 318-319: error comment posting failure in analyze_pull_request_changes
# =============================================================================


@pytest.mark.unit
class TestAnalyzePRErrorCommentFailure:
    """Cover lines 318-319: _post_pr_comment raises during error reporting."""

    async def test_error_comment_post_failure(self, webhook_token, caplog):
        """When analysis fails AND error comment posting also fails, both errors are logged."""
        import logging

        from routers.github_webhooks import analyze_pull_request_changes

        with patch("routers.github_webhooks.GitHubIntegrationHub") as mock_hub_cls:
            mock_hub = MagicMock()
            mock_hub.analyze_pull_request = AsyncMock(side_effect=ValueError("Analysis broke"))
            mock_hub_cls.return_value = mock_hub

            with patch("routers.github_webhooks._post_pr_comment") as mock_post:
                mock_post.side_effect = ConnectionError("Cannot post comment")

                with caplog.at_level(logging.ERROR):
                    await analyze_pull_request_changes("owner/repo", 42, "test")

                mock_post.assert_called_once()
                # The error comment should contain the original error message
                post_args = mock_post.call_args[0]
                assert post_args[0] == "owner/repo"
                assert post_args[1] == 42
                assert "Analysis broke" in post_args[2]
                assert "Failed to post error comment" in caplog.text


# =============================================================================
# Line 382: code graph update warning for non-success, non-no_graph status
# =============================================================================


@pytest.mark.unit
class TestCodeGraphUpdateWarning:
    """Cover line 382: code graph update returns unexpected status."""

    async def test_code_graph_update_unknown_status(self, caplog):
        """Anything other than scan_requested/no_container is an ERROR, and is logged as one.

        Previously an unrecognised status was logged at WARNING and the reason was
        dropped on the floor. A refresh that neither ran nor was skipped is a
        failure, so it is logged at ERROR with the underlying error surfaced.
        """
        import logging

        from routers.github_webhooks import _update_code_graph

        mock_hub = AsyncMock()
        mock_hub.handle_repository_update = AsyncMock(
            return_value={
                "status": "partial_failure",
                "update_stats": {"error": "analysis refused the kickoff"},
            }
        )

        with caplog.at_level(logging.ERROR):
            await _update_code_graph(mock_hub, "owner/repo", "main", ["file.py"], ["sha1"])

        mock_hub.handle_repository_update.assert_called_once_with(
            repo_name="owner/repo",
            branch="main",
            changed_files=["file.py"],
            commit_shas=["sha1"],
            tenant_id=None,
        )
        # The REASON must reach the log, not just the status word.
        assert "analysis refused the kickoff" in caplog.text


# =============================================================================
# Lines 405-413: _process_commits_analysis with file changes accumulation
# =============================================================================


@pytest.mark.unit
class TestProcessCommitsAnalysisFileAccumulation:
    """Cover lines 405-413: _process_commits_analysis accumulates file changes."""

    async def test_process_commits_with_files(self):
        """Test that _process_commits_analysis collects commit data and unique filenames."""
        from routers.github_webhooks import _process_commits_analysis

        mock_service = MagicMock()
        mock_service.get_commit_files = AsyncMock(
            side_effect=[
                [
                    {"filename": "a.py", "status": "modified", "additions": 5, "deletions": 2},
                    {"filename": "b.py", "status": "added", "additions": 10, "deletions": 0},
                ],
                [
                    {"filename": "a.py", "status": "modified", "additions": 3, "deletions": 1},
                    {"filename": "c.js", "status": "added", "additions": 20, "deletions": 0},
                ],
            ]
        )

        commits = [
            {"id": "sha1", "author": {"name": "A"}, "message": "c1", "timestamp": "t"},
            {"id": "sha2", "author": {"name": "B"}, "message": "c2", "timestamp": "t"},
        ]

        commit_analysis, all_changed_files = await _process_commits_analysis(
            mock_service, "owner/repo", commits
        )

        assert len(commit_analysis) == 2
        assert commit_analysis[0]["sha"] == "sha1"
        assert commit_analysis[1]["sha"] == "sha2"
        assert all_changed_files == {"a.py", "b.py", "c.js"}

    async def test_process_commits_skips_failed(self):
        """When _analyze_single_commit returns None, it's skipped."""
        from routers.github_webhooks import _process_commits_analysis

        mock_service = MagicMock()
        mock_service.get_commit_files = AsyncMock(
            side_effect=[
                [{"filename": "ok.py", "status": "modified", "additions": 1, "deletions": 0}],
                RuntimeError("API error"),
            ]
        )

        commits = [
            {"id": "s1", "author": {"name": "A"}, "message": "ok", "timestamp": "t"},
            {"id": "s2", "author": {"name": "B"}, "message": "fail", "timestamp": "t"},
        ]

        commit_analysis, all_changed_files = await _process_commits_analysis(
            mock_service, "owner/repo", commits
        )

        assert len(commit_analysis) == 1
        assert all_changed_files == {"ok.py"}

    async def test_process_commits_truncates_at_cap_and_warns(self, caplog):
        """A push larger than the cap analyses only the first N commits and logs a
        truncation warning, so partial coverage is observable (not silent)."""
        import logging

        from routers.github_webhooks import (
            _MAX_COMMITS_ANALYZED,
            _process_commits_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_commit_files = AsyncMock(
            return_value=[
                {"filename": "f.py", "status": "modified", "additions": 1, "deletions": 0}
            ]
        )
        commits = [
            {"id": f"sha{i}", "author": {"name": "A"}, "message": f"c{i}", "timestamp": "t"}
            for i in range(_MAX_COMMITS_ANALYZED + 5)
        ]

        with caplog.at_level(logging.WARNING):
            commit_analysis, _ = await _process_commits_analysis(
                mock_service, "owner/repo", commits
            )

        assert len(commit_analysis) == _MAX_COMMITS_ANALYZED
        assert mock_service.get_commit_files.await_count == _MAX_COMMITS_ANALYZED
        assert any("truncated" in r.getMessage().lower() for r in caplog.records)


# =============================================================================
# Lines 442-449: Issues analysis with critical issue notification
# =============================================================================


@pytest.mark.unit
class TestAnalyzeRepoWithCriticalIssues:
    """Cover lines 442-449: issues found with critical severity triggers notifications."""

    async def test_critical_issues_trigger_notification(self):
        """When _analyze_files_for_issues finds critical issues, _send_issue_notifications is called."""
        from routers.github_webhooks import analyze_repository_changes

        commits = [{"id": "sha1", "author": {"name": "Dev"}, "message": "m", "timestamp": "t"}]

        with patch("routers.github_webhooks._webhook_github_token", return_value="tok"):
            with patch("routers.github_webhooks.GitHubService"):
                with patch("routers.github_webhooks.GitHubIntegrationHub"):
                    with patch(
                        "routers.github_webhooks._process_commits_analysis",
                        return_value=(
                            [{"sha": "sha1", "total_additions": 5, "total_deletions": 0}],
                            {"secrets.py"},
                        ),
                    ):
                        with patch("routers.github_webhooks._update_code_graph"):
                            with patch(
                                "routers.github_webhooks._analyze_files_for_issues",
                                return_value=[
                                    {
                                        "type": "sensitive",
                                        "severity": "critical",
                                        "file": "s.py",
                                        "message": "bad",
                                    },
                                    {
                                        "type": "security",
                                        "severity": "high",
                                        "file": "o.py",
                                        "message": "meh",
                                    },
                                ],
                            ):
                                with patch(
                                    "routers.github_webhooks._send_issue_notifications"
                                ) as mock_notify:
                                    result = await analyze_repository_changes(
                                        "owner/repo", "main", commits
                                    )

                                    mock_notify.assert_called_once()
                                    critical = mock_notify.call_args[0][2]
                                    assert len(critical) == 1
                                    assert critical[0]["severity"] == "critical"
                                    assert result["repository"] == "owner/repo"

    async def test_issues_analysis_failure_caught(self):
        """When _analyze_files_for_issues raises ValueError, lines 448-449 catch it."""
        from routers.github_webhooks import analyze_repository_changes

        commits = [{"id": "sha1", "author": {"name": "D"}, "message": "m", "timestamp": "t"}]

        with patch("routers.github_webhooks._webhook_github_token", return_value="tok"):
            with patch("routers.github_webhooks.GitHubService"):
                with patch("routers.github_webhooks.GitHubIntegrationHub"):
                    with patch(
                        "routers.github_webhooks._process_commits_analysis",
                        return_value=(
                            [{"sha": "sha1", "total_additions": 1, "total_deletions": 0}],
                            {"f.py"},
                        ),
                    ):
                        with patch("routers.github_webhooks._update_code_graph"):
                            with patch(
                                "routers.github_webhooks._analyze_files_for_issues",
                                side_effect=ValueError("Issue analysis broke"),
                            ):
                                result = await analyze_repository_changes(
                                    "owner/repo", "main", commits
                                )
                                assert result["repository"] == "owner/repo"

    async def test_issues_found_no_critical(self):
        """When issues are found but none critical, no notification sent."""
        from routers.github_webhooks import analyze_repository_changes

        commits = [{"id": "s1", "author": {"name": "D"}, "message": "m", "timestamp": "t"}]

        with patch("routers.github_webhooks._webhook_github_token", return_value="tok"):
            with patch("routers.github_webhooks.GitHubService"):
                with patch("routers.github_webhooks.GitHubIntegrationHub"):
                    with patch(
                        "routers.github_webhooks._process_commits_analysis",
                        return_value=(
                            [{"sha": "s1", "total_additions": 1, "total_deletions": 0}],
                            {"f.py"},
                        ),
                    ):
                        with patch("routers.github_webhooks._update_code_graph"):
                            with patch(
                                "routers.github_webhooks._analyze_files_for_issues",
                                return_value=[
                                    {
                                        "type": "binary",
                                        "severity": "medium",
                                        "file": "x.zip",
                                        "message": "bin",
                                    },
                                ],
                            ):
                                with patch(
                                    "routers.github_webhooks._send_issue_notifications"
                                ) as mock_notify:
                                    result = await analyze_repository_changes(
                                        "owner/repo", "main", commits
                                    )
                                    mock_notify.assert_not_called()
                                    assert result["repository"] == "owner/repo"


# =============================================================================
# Lines 689-690: Exception in _send_issue_notifications
# =============================================================================


@pytest.mark.unit
class TestSendIssueNotificationsException:
    """Cover lines 689-690: outer exception handler in _send_issue_notifications."""

    async def test_send_notifications_raises_value_error(self, caplog):
        """When the credential lookup raises ValueError, the outer except catches and logs it."""
        import logging

        from routers.github_webhooks import _send_issue_notifications

        with patch(
            "routers.github_webhooks._webhook_github_token", side_effect=ValueError("broken")
        ):
            with caplog.at_level(logging.ERROR):
                await _send_issue_notifications(
                    "owner/repo", "main", [{"file": "x.py", "message": "bad"}]
                )

        assert "Error sending issue notifications" in caplog.text
        assert "broken" in caplog.text

    async def test_send_notifications_raises_runtime_error(self, caplog):
        """When GitHubConfig raises RuntimeError, the outer except catches and logs it."""
        import logging

        from routers.github_webhooks import _send_issue_notifications

        with patch("routers.github_webhooks._webhook_github_token", return_value="tok"):
            with patch(
                "routers.github_webhooks.GitHubConfig",
                side_effect=RuntimeError("Config init failed"),
            ):
                with caplog.at_level(logging.ERROR):
                    await _send_issue_notifications(
                        "owner/repo", "main", [{"file": "x.py", "message": "bad"}]
                    )

        assert "Error sending issue notifications" in caplog.text
        assert "Config init failed" in caplog.text

    async def test_send_notifications_raises_connection_error(self, caplog):
        """When _format_notification_message raises ConnectionError, outer except catches and logs it."""
        import logging

        from routers.github_webhooks import _send_issue_notifications

        with patch("routers.github_webhooks._webhook_github_token", return_value="tok"):
            with patch("routers.github_webhooks.GitHubConfig"):
                with patch("routers.github_webhooks.GitHubService"):
                    with patch(
                        "routers.github_webhooks._format_notification_message",
                        side_effect=ConnectionError("Network down"),
                    ):
                        with caplog.at_level(logging.ERROR):
                            await _send_issue_notifications(
                                "owner/repo", "main", [{"file": "x.py", "message": "bad"}]
                            )

        assert "Error sending issue notifications" in caplog.text
        assert "Network down" in caplog.text
