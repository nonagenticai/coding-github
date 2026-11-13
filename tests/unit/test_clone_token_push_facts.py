"""Push-scope clone-token answers carry the repository facts coding-containers refuses on.

coding-containers' control pod has NO egress to api.github.com (by design: only
coding-github does). Its push route must still refuse a public repository and a push
to the default branch, and must range the pushed paths from the commit GitHub
already has. Before this contract it asked GitHub itself and so failed every push in
prod with 503 ``repo_visibility_unknown``. These tests pin that the internal
clone-token response for ``scope='push'`` carries ``private``, ``default_branch``
and (with ``branch``) the branch heads, read by THIS service with the minted token.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from services import github_app_auth as ga
from services.github_app_auth import RefHead, RepoPushMetadata, ScopedToken

_INTERNAL = "s3cret-internal-token"  # pragma: allowlist secret
_URL = "/api/internal/github/clone-token"
_SHA_B = "b" * 40
_SHA_D = "d" * 40


@pytest.fixture
def facts(monkeypatch):
    import services.github_integration as gi
    from routers import github_internal

    monkeypatch.setattr(gi.settings, "system_github_owner", "acme")
    metadata = MagicMock(return_value=RepoPushMetadata(True, "main"))
    heads = MagicMock(return_value=RefHead("present", _SHA_B))
    with (
        patch.object(
            github_internal,
            "get_settings",
            return_value=SimpleNamespace(internal_callers={"coding-containers": _INTERNAL}),
        ),
        patch.object(
            github_internal,
            "get_scoped_clone_token",
            MagicMock(return_value=ScopedToken("ghs_scoped", "2999-01-01T00:00:00Z")),
        ),
        patch.object(github_internal, "get_repo_push_metadata", metadata),
        patch.object(github_internal, "get_branch_head", heads),
        patch.object(github_internal, "get_repo_visibility", MagicMock(return_value="private")),
    ):
        yield SimpleNamespace(metadata=metadata, heads=heads)


async def _mint(client, scope="push", branch=None):
    body: dict[str, str] = {"repo_url": "https://github.com/acme/widgets", "scope": scope}
    if branch is not None:
        body["branch"] = branch
    return await client.post(_URL, json=body, headers={"X-Internal-Token": _INTERNAL})


@pytest.mark.unit
class TestPushFactsContract:
    async def test_push_answer_carries_private_and_default_branch(self, async_client, facts):
        resp = await _mint(async_client)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["token"] == "ghs_scoped"
        assert body["private"] is True
        assert body["default_branch"] == "main"
        # no branch asked ⇒ no head lookups (one GET, as before)
        assert body["branch_head"] is None and body["default_branch_head"] is None
        facts.heads.assert_not_called()
        facts.metadata.assert_called_once_with("acme", "widgets", "ghs_scoped")

    async def test_existing_branch_head_is_reported(self, async_client, facts):
        resp = await _mint(async_client, branch="aa/fix-1")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["branch_head"] == {"state": "present", "sha": _SHA_B}
        assert body["default_branch_head"] is None
        facts.heads.assert_called_once_with("acme", "widgets", "aa/fix-1", "ghs_scoped")

    async def test_absent_branch_reports_the_default_branch_head(self, async_client, facts):
        facts.heads.side_effect = [RefHead("absent"), RefHead("present", _SHA_D)]
        resp = await _mint(async_client, branch="aa/new")
        body = resp.json()
        assert body["branch_head"] == {"state": "absent", "sha": None}
        assert body["default_branch_head"] == {"state": "present", "sha": _SHA_D}
        assert facts.heads.call_args_list[1].args == ("acme", "widgets", "main", "ghs_scoped")

    async def test_unknown_branch_head_is_never_treated_as_absent(self, async_client, facts):
        """``unknown`` gets no default-branch base: a caller that ranged from it would
        use the wrong base whenever the branch in fact exists."""
        facts.heads.return_value = RefHead("unknown")
        body = (await _mint(async_client, branch="aa/fix-1")).json()
        assert body["branch_head"] == {"state": "unknown", "sha": None}
        assert body["default_branch_head"] is None
        facts.heads.assert_called_once_with("acme", "widgets", "aa/fix-1", "ghs_scoped")

    async def test_unknown_default_branch_is_reported_unknown_not_guessed(
        self, async_client, facts
    ):
        facts.metadata.return_value = RepoPushMetadata(True, None)
        facts.heads.return_value = RefHead("absent")
        body = (await _mint(async_client, branch="aa/new")).json()
        assert body["default_branch"] is None
        assert body["default_branch_head"] is None  # no base lookup on a guessed name
        assert facts.heads.call_count == 1

    async def test_public_repo_still_refused_and_no_facts_leak(self, async_client, facts):
        facts.metadata.return_value = RepoPushMetadata(False, "main")
        resp = await _mint(async_client, branch="aa/fix-1")
        assert resp.status_code == 403, resp.text
        assert "ghs_scoped" not in resp.text
        facts.heads.assert_not_called()

    async def test_clone_scope_carries_no_push_facts(self, async_client, facts):
        body = (await _mint(async_client, scope="clone", branch="aa/fix-1")).json()
        assert body["private"] is None and body["default_branch"] is None
        assert body["branch_head"] is None
        facts.metadata.assert_not_called()
        facts.heads.assert_not_called()

    @pytest.mark.parametrize("branch", ["../x", "a b", "-x", "x?y", ""])
    async def test_branch_is_validated(self, async_client, facts, branch):
        resp = await _mint(async_client, branch=branch)
        assert resp.status_code == 422, resp.text


def _resp(status, body):
    return SimpleNamespace(status_code=status, json=lambda: body)


@pytest.mark.unit
class TestRepoPushMetadata:
    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (200, {"private": True, "default_branch": "trunk"}, (True, "trunk")),
            (200, {"private": True, "default_branch": ""}, (True, None)),
            (200, {"private": True}, (True, None)),
            (200, {"default_branch": 7}, (None, None)),
            (404, {}, (None, None)),
        ],
    )
    def test_metadata(self, status, body, expected):
        with patch.object(ga.httpx, "get", MagicMock(return_value=_resp(status, body))):
            assert tuple(ga.get_repo_push_metadata("acme", "widgets", "ghs_x")) == expected

    def test_transport_error_raises(self):
        boom = MagicMock(side_effect=httpx.ConnectError("no route"))
        with patch.object(ga.httpx, "get", boom), pytest.raises(ga.GitHubAppAuthError):
            ga.get_repo_push_metadata("acme", "widgets", "ghs_x")


@pytest.mark.unit
class TestBranchHead:
    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (200, {"object": {"type": "commit", "sha": _SHA_B}}, RefHead("present", _SHA_B)),
            (200, {"object": {"type": "tag", "sha": _SHA_B}}, RefHead("unknown")),
            (200, {"object": {"type": "commit", "sha": "short"}}, RefHead("unknown")),
            (404, {}, RefHead("absent")),
            (409, {"message": "Git Repository is empty."}, RefHead("empty_repository")),
            (409, {"message": "conflict"}, RefHead("unknown")),
            (500, {}, RefHead("unknown")),
        ],
    )
    def test_head(self, status, body, expected):
        get = MagicMock(return_value=_resp(status, body))
        with patch.object(ga.httpx, "get", get):
            assert ga.get_branch_head("acme", "widgets", "aa/fix-1", "ghs_x") == expected
        assert get.call_args.args[0].endswith("/repos/acme/widgets/git/ref/heads/aa/fix-1")

    def test_transport_error_is_unknown_not_absent(self):
        boom = MagicMock(side_effect=httpx.ConnectError("no route"))
        with patch.object(ga.httpx, "get", boom):
            assert ga.get_branch_head("acme", "widgets", "main", "ghs_x") == RefHead("unknown")
