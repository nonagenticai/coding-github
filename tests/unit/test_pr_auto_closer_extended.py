"""Extended tests for pr_auto_closer.py — covering uncovered lines."""

import logging
from unittest.mock import MagicMock, patch


def _make_validation_response(
    overall_score=90.0,
    passed=True,
    recommendation="approve",
    validations=None,
):
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


class TestPRAutoCloserExtended:
    """Extended tests covering uncovered lines in pr_auto_closer.py."""

    def _make_closer(self, config=None):
        with patch("services.pr_validation.actions.pr_auto_closer.Github") as MockGithub:  # noqa: N806
            mock_github = MagicMock()
            MockGithub.return_value = mock_github
            from services.pr_validation.actions.pr_auto_closer import PRAutoCloser

            closer = PRAutoCloser("fake-token", config=config)
            closer.github = mock_github
        return closer

    # --- Line 81: _should_close_pr with add_labels=True triggers _add_validation_labels ---

    async def test_process_with_add_labels_enabled(self):
        """When add_labels is True, _add_validation_labels should be called."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": False,
            "request_changes": False,
            "labels": {
                "critical": "validation-failed-critical",
                "high": "validation-failed-high",
                "medium": "validation-warning",
                "low": "validation-info",
            },
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.labels = []
        mock_pr.number = 1
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=85,
            recommendation="approve",
            validations=[_make_validation(passed=True, score=85)],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.add_to_labels.assert_called_once_with("validation-info")

    # --- Line 87: request_changes branch when recommendation is reject but auto_close disabled ---

    async def test_request_changes_on_reject_without_auto_close(self):
        """When recommendation is reject and auto_close is disabled, request_changes is called."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": False,
            "request_changes": True,
            "labels": {},
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_commits = MagicMock()
        mock_commit = MagicMock()
        mock_commits.__iter__ = MagicMock(return_value=iter([mock_commit]))
        mock_pr.get_commits.return_value = mock_commits
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        critical_issue = _make_issue(severity="critical", message="SQL injection")
        validation = _make_validation(passed=False, score=10, issues=[critical_issue])
        response = _make_validation_response(
            overall_score=25,  # Above close_on_score_below
            recommendation="reject",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        # Should have requested changes since recommendation is reject
        mock_pr.create_review.assert_called_once()
        call_kwargs = mock_pr.create_review.call_args.kwargs
        assert call_kwargs["event"] == "REQUEST_CHANGES"
        assert call_kwargs["commit"] is mock_commit
        assert "Changes Requested" in call_kwargs["body"]

    # --- Line 100: _should_close_pr returns True on low score ---

    def test_should_close_pr_low_score(self):
        """PR with score below threshold should be marked for closing."""
        closer = self._make_closer()
        response = _make_validation_response(
            overall_score=15,
            recommendation="needs_work",
            validations=[],
        )
        assert closer._should_close_pr(response) is True

    # --- Lines 107-123: _add_validation_labels with existing labels and different recommendations ---

    async def test_add_labels_removes_old_validation_labels(self):
        """Old validation labels should be removed before adding new ones."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": False,
            "request_changes": False,
            "labels": {
                "critical": "validation-failed-critical",
                "high": "validation-failed-high",
                "medium": "validation-warning",
                "low": "validation-info",
            },
        }
        closer = self._make_closer(config=config)

        # Existing labels that include a validation label
        old_label = MagicMock()
        old_label.name = "validation-failed-critical"
        mock_pr = MagicMock()
        mock_pr.labels = [old_label]
        mock_pr.number = 1
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=70,
            recommendation="needs_work",
            validations=[_make_validation(passed=False, score=50)],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.remove_from_labels.assert_called_with("validation-failed-critical")
        mock_pr.add_to_labels.assert_called_with("validation-failed-high")

    async def test_add_labels_reject_recommendation(self):
        """Reject recommendation should add critical label."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": False,
            "request_changes": False,
            "labels": {
                "critical": "validation-failed-critical",
                "high": "validation-failed-high",
                "medium": "validation-warning",
                "low": "validation-info",
            },
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.labels = []
        mock_pr.number = 1
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=30,
            recommendation="reject",
            validations=[_make_validation(passed=False, score=30)],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.add_to_labels.assert_called_with("validation-failed-critical")

    async def test_add_labels_exception_handled(self, caplog):
        """Exception during label addition should be logged, not raised."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": True,
            "post_comment": False,
            "request_changes": False,
            "labels": {
                "critical": "validation-failed-critical",
                "high": "validation-failed-high",
                "medium": "validation-warning",
                "low": "validation-info",
            },
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.labels = []
        mock_pr.number = 1
        mock_pr.add_to_labels.side_effect = Exception("Label API error")
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=85,
            recommendation="approve",
            validations=[],
        )

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        assert any("Label API error" in record.message for record in caplog.records)
        assert any("Error adding labels" in record.message for record in caplog.records)

    # --- Lines 167-168: _post_validation_comment exception ---

    async def test_post_comment_exception_handled(self, caplog):
        """Exception during comment posting should be logged, not raised."""
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
        mock_pr.number = 1
        mock_pr.create_issue_comment.side_effect = Exception("Comment API error")
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        response = _make_validation_response(
            overall_score=85,
            recommendation="approve",
            validations=[_make_validation(passed=True, score=85)],
        )

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        assert any("Comment API error" in record.message for record in caplog.records)
        assert any("Error posting comment" in record.message for record in caplog.records)

    # --- Lines 175-176: _close_pr exception ---

    async def test_close_pr_exception_handled(self, caplog):
        """Exception during PR close should be logged, not raised."""
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
        mock_pr.number = 1
        mock_pr.edit.side_effect = Exception("Close API error")
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        critical_issue = _make_issue(severity="critical", message="Critical flaw")
        validation = _make_validation(passed=False, score=5, issues=[critical_issue])
        response = _make_validation_response(
            overall_score=5,
            recommendation="reject",
            validations=[validation],
        )

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        mock_pr.edit.assert_called_once_with(state="closed")
        assert any("Close API error" in record.message for record in caplog.records)
        assert any("Error closing PR" in record.message for record in caplog.records)

    # --- Lines 182-189: _request_changes full path ---

    async def test_request_changes_exception_handled(self, caplog):
        """Exception during request_changes should be logged, not raised."""
        config = {
            "auto_close_enabled": False,
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": False,
            "request_changes": True,
            "labels": {},
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.get_commits.side_effect = Exception("Commits API error")
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        # Non-critical reject — doesn't trigger close, but triggers request_changes
        response = _make_validation_response(
            overall_score=30,
            recommendation="reject",
            validations=[_make_validation(passed=False, score=30)],
        )

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        mock_pr.get_commits.assert_called_once()
        assert any("Commits API error" in record.message for record in caplog.records)
        assert any("Error requesting changes" in record.message for record in caplog.records)

    # --- Lines 218-219: auto_process_pr_validation convenience function ---

    async def test_auto_process_pr_validation_function(self):
        """The convenience function should create a PRAutoCloser and call process_validation_result."""
        with patch("services.pr_validation.actions.pr_auto_closer.Github") as MockGithub:  # noqa: N806
            mock_github = MagicMock()
            MockGithub.return_value = mock_github

            from services.pr_validation.actions.pr_auto_closer import auto_process_pr_validation

            response = _make_validation_response(
                overall_score=90,
                recommendation="approve",
                validations=[],
            )

            await auto_process_pr_validation(
                "owner", "repo", 1, "fake-token", response, config=None
            )

            MockGithub.assert_called_once_with("fake-token")
            mock_github.get_repo.assert_called_once_with("owner/repo")

    async def test_auto_process_pr_validation_with_custom_config(self):
        """auto_process_pr_validation should accept and use a custom config."""
        with patch("services.pr_validation.actions.pr_auto_closer.Github") as MockGithub:  # noqa: N806
            mock_github = MagicMock()
            MockGithub.return_value = mock_github
            mock_repo = MagicMock()
            mock_pr = MagicMock()
            mock_pr.labels = []
            mock_pr.number = 1
            mock_repo.get_pull.return_value = mock_pr
            mock_github.get_repo.return_value = mock_repo

            from services.pr_validation.actions.pr_auto_closer import auto_process_pr_validation

            custom_config = {
                "auto_close_enabled": False,
                "close_on_risk_level": ["CRITICAL"],
                "close_on_score_below": 10,
                "add_labels": False,
                "post_comment": False,
                "request_changes": False,
                "labels": {},
            }

            response = _make_validation_response(
                overall_score=80,
                recommendation="approve",
                validations=[],
            )

            await auto_process_pr_validation(
                "owner", "repo", 1, "fake-token", response, config=custom_config
            )

    # --- Lines 88-89: outer exception handler in process_validation_result ---

    async def test_process_validation_result_outer_exception(self, caplog):
        """If _should_close_pr raises, the outer except should log the error."""
        closer = self._make_closer()
        mock_pr = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        # Make _should_close_pr raise by passing a response with broken recommendation attr
        response = MagicMock()
        response.recommendation = "reject"
        response.validations = MagicMock()
        response.validations.__iter__ = MagicMock(side_effect=RuntimeError("broken"))

        with caplog.at_level(logging.ERROR, logger="services.pr_validation.actions.pr_auto_closer"):
            await closer.process_validation_result("owner", "repo", 1, response)

        assert any("broken" in record.message for record in caplog.records)
        assert any(
            "Error processing validation result" in record.message for record in caplog.records
        )
        # PR should NOT have been closed since the error happened before that logic
        mock_pr.edit.assert_not_called()

    # --- Line 174: _close_pr success path ---

    async def test_close_pr_success_path(self):
        """When pr.edit succeeds, the info log line should be executed."""
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
        mock_pr.number = 42
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        critical_issue = _make_issue(severity="critical", message="Critical")
        validation = _make_validation(passed=False, score=5, issues=[critical_issue])
        response = _make_validation_response(
            overall_score=5,
            recommendation="reject",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 42, response)
        mock_pr.edit.assert_called_once_with(state="closed")

    # --- Post comment with close template (critical issues) ---

    async def test_post_comment_with_close_template(self):
        """When should_close is True, the close message template should be used."""
        config = {
            "auto_close_enabled": False,  # Don't actually close, just test comment
            "close_on_risk_level": ["CRITICAL"],
            "close_on_score_below": 20,
            "add_labels": False,
            "post_comment": True,
            "request_changes": False,
            "labels": {},
            "warning_message_template": "Warn: {score}, {risk_level}, {recommendation}, {issues_summary}, {suggestions}",
            "close_message_template": "Close: {score}, {risk_level}, {recommendation}, {critical_issues}, {issues_summary}, {suggestions}, {validation_summary}",
        }
        closer = self._make_closer(config=config)

        mock_pr = MagicMock()
        mock_pr.labels = []
        mock_pr.number = 1
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        critical_issue = _make_issue(severity="critical", message="SQL injection")
        validation = _make_validation(
            vtype="security",
            passed=False,
            score=5,
            issues=[critical_issue],
            suggestions=["Fix SQL injection"],
        )
        response = _make_validation_response(
            overall_score=5,
            recommendation="reject",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.create_issue_comment.assert_called_once()
        comment_text = mock_pr.create_issue_comment.call_args[0][0]
        assert "Close:" in comment_text
        assert "5" in comment_text  # score
        assert "REJECT" in comment_text  # risk_level
        assert "SQL injection" in comment_text  # critical issue message
        assert "Fix SQL injection" in comment_text  # suggestion

    # --- Multiple issue severities in comment ---

    async def test_post_comment_with_multiple_severities(self):
        """Comment should include issues of multiple severity levels."""
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
        mock_pr.number = 1
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        closer.github.get_repo.return_value = mock_repo

        issues = [
            _make_issue(severity="high", message="Performance issue"),
            _make_issue(severity="medium", message="Style issue"),
            _make_issue(severity="low", message="Minor thing"),
        ]
        validation = _make_validation(
            passed=True, score=70, issues=issues, suggestions=["Optimize", "Refactor"]
        )
        response = _make_validation_response(
            overall_score=70,
            recommendation="needs_work",
            validations=[validation],
        )

        await closer.process_validation_result("owner", "repo", 1, response)
        mock_pr.create_issue_comment.assert_called_once()
        comment_text = mock_pr.create_issue_comment.call_args[0][0]
        assert "Score: 70" in comment_text
        assert "NEEDS_WORK" in comment_text
        assert "High:" in comment_text
        assert "Medium:" in comment_text
        assert "Low:" in comment_text
        assert "Optimize" in comment_text
        assert "Refactor" in comment_text
