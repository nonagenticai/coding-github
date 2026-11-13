"""Extended tests for services/github_actions_trigger.py

Covers: singleton factory function get_workflow_trigger.
"""

from __future__ import annotations

from unittest.mock import patch

from services.github_actions_trigger import (
    GitHubActionsWorkflowTrigger,
    get_workflow_trigger,
)


class TestGetWorkflowTriggerSingleton:
    """Singleton factory function for GitHubActionsWorkflowTrigger."""

    def test_creates_instance_and_reuses_it(self):
        """First call creates, second call reuses the singleton."""
        with (
            patch("services.github_actions_trigger.Github"),
        ):
            import services.github_actions_trigger as mod

            mod._workflow_trigger = None
            t1 = get_workflow_trigger("ghp_test")
            assert isinstance(t1, GitHubActionsWorkflowTrigger)

            # Second call without token returns same instance
            t2 = get_workflow_trigger()
            assert t2 is t1

            mod._workflow_trigger = None

    def test_new_token_creates_new_instance(self):
        """Calling with a different token creates a new instance."""
        with (
            patch("services.github_actions_trigger.Github"),
        ):
            import services.github_actions_trigger as mod

            mod._workflow_trigger = None
            t1 = get_workflow_trigger("ghp_test")
            t2 = get_workflow_trigger("ghp_new")
            assert t2 is not t1
            mod._workflow_trigger = None
