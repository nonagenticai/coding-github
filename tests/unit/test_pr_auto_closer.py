"""Tests for pr_auto_closer.py."""

import logging
from unittest.mock import MagicMock, patch


def _make_validation_response(
    overall_score=90.0,
    passed=True,
    recommendation="approve",
    validations=None,
):
    """Create a mock ValidatePRResponse."""
    resp = MagicMock()
    resp.overall_score = overall_score
    resp.passed = passed
    resp.recommendation = recommendation
    resp.validations = validations or []
    return resp


def _make_validation(vtype="security", passed=True, score=90, issues=None, suggestions=None):
    v = MagicMock()
    v.validation_type = vtype
    v.passed = passed
    v.score = score
    v.issues = issues or []
    v.suggestions = suggestions or []
    return v


def _make_issue(severity="low", message="test", file="test.py", line=1):
    i = MagicMock()
    i.severity = severity
    i.message = message
    i.file = file
    i.line = line
    return i


class TestPRAutoCloser:
    """Tests for PRAutoCloser."""

    def _make_closer(self, config=None):
        with patch("services.pr_validation.actions.pr_auto_closer.Github") as MockGithub:  # noqa: N806
            mock_github = MagicMock()
            MockGithub.return_value = mock_github
            from services.pr_validation.actions.pr_auto_closer import PRAutoCloser

            closer = PRAutoCloser("fake-token", config=config)
            closer.github = mock_github
        return closer

    async def test_close_on_critical(self):
        """PR with critical issues and reject recommendation should be closed."""
        config = {
            "auto_close_enabled": True,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": False,
            "request_changes": False,
            "labels": {},
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        critical_issue = _make_issue(severity="critical", message="SQL injection")
        validation = _make_validation(
            vtype="security",
            passed=False,
            score=10,
            issues=[critical_issue],
        )
        response = _make_validation_response(
            overall_score=10,
            passed=False,
            recommendation="reject",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.edit.assert_called_once_with(state="closed")

    async def test_no_close_on_minor(self):
        """PR with minor issues and approve recommendation should NOT be closed."""
        config = {
            "auto_close_enabled": True,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": False,
            "request_changes": False,
            "labels": {},
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=90,
            passed=True,
            recommendation="approve",
            validations=[_make_validation(passed=True, score=90)],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.edit.assert_not_called()

    async def test_comment_posted(self):
        """When post_comment is enabled, a comment should be created on the PR."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": True,
            "request_changes": False,
            "labels": {},
            "warning_message_template": "Score: {score}, Risk: {risk_level}, Rec: {recommendation}, Issues: {issues_summary}, Suggestions: {suggestions}",
            "close_message_template": "Closed: {score}, {risk_level}, {recommendation}, {critical_issues}, {issues_summary}, {suggestions}, {validation_summary}",
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.labels = []
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        low_issue = _make_issue(severity="low", message="minor thing")
        validation = _make_validation(
            passed=True, score=85, issues=[low_issue], suggestions=["Fix it"]
        )
        response = _make_validation_response(
            overall_score=85,
            passed=True,
            recommendation="approve",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.create_issue_comment.assert_called_once()
        comment_text = mock_pr.create_issue_comment.call_args[0][0]
        assert "Score: 85" in comment_text
        assert "APPROVE" in comment_text
        assert "approve" in comment_text
        assert "Fix it" in comment_text

    async def test_api_failure_handled(self, caplog):
        """If GitHub API fails, process_validation_result should log the error."""
        closer = self._make_closer()
        closer.github.get_repo.side_effect = Exception("API down")

        response = _make_validation_response(
            overall_score=50,
            passed=False,
            recommendation="reject",
            validations=[],
        )

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        assert any("API down" in record.message for record in caplog.records)
        assert any("PR #1" in record.message for record in caplog.records)
