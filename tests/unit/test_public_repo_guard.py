"""A PUBLIC repository is never a conveyor target (TENANCY_AND_IMPROVEMENT.md §8).

apply-code-changes, create-pull-request, create-upstream-pr and merge-pull-request
refuse a public target with 403 ``public_repo_refused`` BEFORE any GitHub write, and
refuse with 503 ``repo_visibility_unknown`` when visibility cannot be established
(fail closed — unknown is never read as private).

Codes are read off ``error.details.error_code``: coding-client's shared handler
overwrites the top-level ``error.code`` with ``HTTP_<status>``.

MUTATION CONTROLS (``TestTheGuardIsWhatRefuses``): with the guard call neutralised at
each call site, the SAME public-repo request reaches the GitHub write. So the refusals
below demonstrably come from the guard, not from a mock that happened to fail.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.github_service import GitHubConfig, GitHubService
from services.github_app_auth import RepoPushMetadata
from services.repo_visibility_guard import (
    PUBLIC_REPO_REFUSED,
    REPO_VISIBILITY_UNKNOWN,
    RepoVisibilityRefusedError,
    assert_repo_not_public,
    assert_repo_not_public_lookup,
)


@pytest.fixture(autouse=True)
def _commit_seam(monkeypatch):
    """The fake repo hands out its commit through the seam production uses.

    Production never calls ``repo.get_commit`` here: that GET needs contents:read
    (see ``services.github.github_service.commit_handle``).
    """
    monkeypatch.setattr(
        "services.github.github_service.commit_handle", lambda repo, sha: repo.commit_for(sha)
    )


BASE = "/api/repositories"
MERGE_URL = f"{BASE}/merge-pull-request"


def _code(resp) -> str | None:
    body = resp.json()
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return (error.get("details") or {}).get("error_code")
    return None


# ===========================================================================
# 1. The decision function
# ===========================================================================
class TestAssertRepoNotPublic:
    def test_private_true_passes(self):
        assert_repo_not_public(SimpleNamespace(private=True), "o/r")

    def test_public_is_403_public_repo_refused(self):
        with pytest.raises(RepoVisibilityRefusedError) as exc:
            assert_repo_not_public(SimpleNamespace(private=False), "o/r")
        assert (exc.value.error_code, exc.value.status_code) == (PUBLIC_REPO_REFUSED, 403)

    @pytest.mark.parametrize(
        "repo_obj",
        [
            SimpleNamespace(),  # attribute absent
            SimpleNamespace(private=None),
            SimpleNamespace(private="false"),  # truthy string must not read as private
            SimpleNamespace(private=1),
            MagicMock(),  # an auto-attribute is not an answer
        ],
    )
    def test_anything_but_a_real_bool_is_unknown_503(self, repo_obj):
        with pytest.raises(RepoVisibilityRefusedError) as exc:
            assert_repo_not_public(repo_obj, "o/r")
        assert (exc.value.error_code, exc.value.status_code) == (REPO_VISIBILITY_UNKNOWN, 503)

    def test_a_raising_lazy_attribute_is_unknown_503(self):
        class _Lazy:
            @property
            def private(self):
                raise RuntimeError("completion call failed")

        with pytest.raises(RepoVisibilityRefusedError) as exc:
            assert_repo_not_public(_Lazy(), "o/r")
        assert exc.value.error_code == REPO_VISIBILITY_UNKNOWN

    def test_a_failed_lookup_is_unknown_503(self):
        def _boom():
            raise ConnectionError("api.github.com unreachable")

        with pytest.raises(RepoVisibilityRefusedError) as exc:
            assert_repo_not_public_lookup(_boom, "o/r")
        assert (exc.value.error_code, exc.value.status_code) == (REPO_VISIBILITY_UNKNOWN, 503)


# ===========================================================================
# 2. apply-code-changes
# ===========================================================================
def _workspaces_client():
    client = MagicMock()
    client.get_workspace = AsyncMock(
        return_value={
            "status": "success",
            "message": "ok",
            "container": {
                "container_name": "test-container",
                "repo_url": "https://github.com/owner/repo",
                "branch": "main",
                "status": "running",
                "github_access_token": None,
            },
        }
    )
    return client


def _fork_repo():
    repo = MagicMock()
    repo.default_branch = "main"
    repo.full_name = "owner/repo"
    repo.html_url = "https://github.com/owner/repo"
    repo.owner = SimpleNamespace(login="owner")
    repo.name = "repo"
    repo.private = True  # the resolved WRITE target is private too (#118)
    return repo


def _integration(private=None, lookup_error: Exception | None = None):
    inst = MagicMock()
    if lookup_error is not None:
        inst.github_client.get_repo.side_effect = lookup_error
    else:
        inst.github_client.get_repo.return_value = SimpleNamespace(private=private)
    inst._ensure_fork.return_value = SimpleNamespace(repository=_fork_repo())
    inst._create_branch.return_value = None
    inst._apply_changes.return_value = ("sha123", 1)
    inst.create_pull_request.return_value = {
        "success": True,
        "pr_number": 7,
        "pr_url": "https://github.com/owner/repo/pull/7",
    }
    return inst


def _assert_no_write(inst):
    inst._ensure_fork.assert_not_called()  # can itself create a fork
    inst._create_branch.assert_not_called()
    inst._apply_changes.assert_not_called()
    inst.create_pull_request.assert_not_called()


async def _apply(client):
    return await client.post(
        f"{BASE}/apply-changes",
        json={
            "container_name": "test-container",
            "changes": [{"file_path": "src/a.py", "content": "x"}],
            "branch_name": "aa/x",
        },
    )


class TestApplyChanges:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_public_target_is_refused_before_any_write(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=False)
        mock_gh_cls.return_value = inst

        resp = await _apply(async_client)

        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        inst.github_client.get_repo.assert_called_once_with("owner/repo")
        _assert_no_write(inst)

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_private_target_is_written(self, mock_gh_cls, mock_get_client, async_client):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=True)
        mock_gh_cls.return_value = inst

        resp = await _apply(async_client)

        assert resp.status_code == 200, resp.text
        inst._apply_changes.assert_called_once()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_lookup_failure_is_503_and_nothing_written(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(lookup_error=ConnectionError("github down"))
        mock_gh_cls.return_value = inst

        resp = await _apply(async_client)

        assert resp.status_code == 503, resp.text
        assert _code(resp) == REPO_VISIBILITY_UNKNOWN, resp.text
        _assert_no_write(inst)


# ===========================================================================
# 2b. The RESOLVED write target. `_ensure_fork` can redirect the write to a
#     different repo (the token user's fork/clone); a private upstream must not
#     launder a PUBLIC fork. Checked after `_ensure_fork`, before any write.
# ===========================================================================
def _public_fork():
    fork = _fork_repo()
    fork.full_name = "someuser/repo"
    fork.private = False
    return fork


class TestResolvedWriteTarget:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_private_upstream_public_fork_is_refused_before_any_write(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=True)  # upstream owner/repo is PRIVATE
        inst._ensure_fork.return_value = SimpleNamespace(repository=_public_fork())
        mock_gh_cls.return_value = inst

        resp = await _apply(async_client)

        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        assert "someuser/repo" in resp.text, resp.text  # refused the FORK, not upstream
        inst._ensure_fork.assert_called_once()
        inst._create_branch.assert_not_called()
        inst._apply_changes.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_private_upstream_private_fork_is_written(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=True)
        fork = _public_fork()
        fork.private = True
        inst._ensure_fork.return_value = SimpleNamespace(repository=fork)
        mock_gh_cls.return_value = inst

        resp = await _apply(async_client)

        assert resp.status_code == 200, resp.text
        inst._apply_changes.assert_called_once()

    @staticmethod
    def _svc_create_pr(fork):
        from services.github_integration import GitHubIntegration, GitHubRepoInfo

        with (
            patch("services.github_integration.HAS_GITHUB", True),
            patch("services.github_integration.Auth"),
            patch("services.github_integration.Github"),
            patch("services.github_integration.FixValidator"),
        ):
            integration = GitHubIntegration(github_token="test-token")
            ensure = MagicMock(
                return_value=GitHubRepoInfo(
                    repository=fork, is_fork=True, is_clone=False, original_url=None
                )
            )
            create_branch = MagicMock()
            object.__setattr__(integration, "_ensure_fork", ensure)
            object.__setattr__(integration, "_create_branch", create_branch)
            object.__setattr__(integration, "_update_files", MagicMock())
            fork.create_pull.return_value = SimpleNamespace(
                number=9, html_url="https://github.com/owner/repo/pull/9", draft=False
            )
            try:
                result = integration.create_pull_request(
                    repo_owner="owner",
                    repo_name="repo",
                    title="t",
                    message="m",
                    head_branch="aa/x",
                    base_branch="main",
                )
            except RepoVisibilityRefusedError as e:
                result = e
        return result, create_branch, fork

    def test_create_pr_public_fork_is_refused_before_branch_or_pr(self):
        result, create_branch, fork = self._svc_create_pr(_public_fork())
        assert isinstance(result, RepoVisibilityRefusedError), result
        assert (result.error_code, result.repo) == (PUBLIC_REPO_REFUSED, "someuser/repo")
        create_branch.assert_not_called()
        fork.create_pull.assert_not_called()

    def test_create_pr_private_fork_proceeds(self):
        fork = _public_fork()
        fork.private = True
        result, create_branch, _fork = self._svc_create_pr(fork)
        assert not isinstance(result, RepoVisibilityRefusedError), result
        create_branch.assert_called_once()


# ===========================================================================
# 2c. /fork route + the copy-repo fallback. A fork of a PUBLIC upstream is itself
#     public, and the fallback creates a repo and commits into it — both writes.
# ===========================================================================
async def _fork(client):
    return await client.post(f"{BASE}/fork", json={"container_name": "test-container"})


class TestForkRoute:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_of_public_upstream_is_refused_with_no_create_or_commit(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=False)  # upstream owner/repo is PUBLIC
        mock_gh_cls.return_value = inst

        resp = await _fork(async_client)

        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        inst.github_client.get_repo.assert_called_once_with("owner/repo")
        inst._ensure_fork.assert_not_called()  # the fork / clone-create / commits
        inst._create_user_repository_from_original.assert_not_called()

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_private_upstream_resolving_to_public_repo_is_refused(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=True)
        inst._ensure_fork.return_value = SimpleNamespace(repository=_public_fork())
        mock_gh_cls.return_value = inst

        resp = await _fork(async_client)

        assert resp.status_code == 403, resp.text
        assert "someuser/repo" in resp.text, resp.text

    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_fork_of_private_upstream_is_served(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=True)
        mock_gh_cls.return_value = inst

        resp = await _fork(async_client)

        assert resp.status_code == 200, resp.text
        inst._ensure_fork.assert_called_once_with("owner", "repo")


def _fallback(new_private):
    """Run the real copy-repo fallback; returns (result, user, copy_mock)."""
    from services.github_integration import GitHubIntegration

    with (
        patch("services.github_integration.HAS_GITHUB", True),
        patch("services.github_integration.Auth"),
        patch("services.github_integration.Github"),
        patch("services.github_integration.FixValidator"),
    ):
        integration = GitHubIntegration(github_token="test-token")
        client = MagicMock()
        client.get_repo.side_effect = _GhExc(404)
        object.__setattr__(integration, "github_client", client)
        copy = MagicMock()
        object.__setattr__(integration, "_copy_repository_content", copy)
        original = SimpleNamespace(name="repo", full_name="up/repo", private=False)
        user = MagicMock()
        user.login = "someuser"
        user.create_repo.return_value = SimpleNamespace(
            full_name="someuser/repo", private=new_private
        )
        try:
            result = integration._create_user_repository_from_original(original, user)
        except RepoVisibilityRefusedError as e:
            result = e
    return result, user, copy


def _GhExc(status):  # noqa: N802 — a factory for the real PyGithub exception
    from github import GithubException

    return GithubException(status, "nf", None)


class TestCopyRepoFallback:
    def test_fallback_repo_is_created_private_even_from_a_public_original(self):
        result, user, copy = _fallback(new_private=True)
        assert not isinstance(result, RepoVisibilityRefusedError), result
        assert user.create_repo.call_args.kwargs["private"] is True
        copy.assert_called_once()

    @pytest.mark.parametrize("new_private", [False, None], ids=["public", "unknown"])
    def test_fallback_repo_not_confirmed_private_gets_no_commits(self, new_private):
        result, _user, copy = _fallback(new_private=new_private)
        assert isinstance(result, RepoVisibilityRefusedError), result
        copy.assert_not_called()


# ===========================================================================
# 2d. Clone-token ``push`` scope — the chokepoint for container pushes, which
#     traverse no other guard. A write token only for a CONFIRMED-private repo.
# ===========================================================================
_INTERNAL = "s3cret-internal-token"  # pragma: allowlist secret


@pytest.fixture
def mint(monkeypatch):
    import services.github_integration as gi
    from routers import github_internal
    from services.github_app_auth import ScopedToken

    monkeypatch.setattr(gi.settings, "system_github_owner", "acme")
    scoped = MagicMock(return_value=ScopedToken("ghs_scoped", "2999-01-01T00:00:00Z"))
    flag = MagicMock(return_value=True)
    visibility = MagicMock(return_value="private")
    with (
        patch.object(
            github_internal,
            "get_settings",
            return_value=SimpleNamespace(internal_callers={"coding-containers": _INTERNAL}),
        ),
        patch.object(github_internal, "get_scoped_clone_token", scoped),
        patch.object(
            github_internal,
            "get_repo_push_metadata",
            MagicMock(side_effect=lambda o, r, t: RepoPushMetadata(flag(o, r, t), "main")),
        ),
        patch.object(github_internal, "get_repo_visibility", visibility),
    ):
        yield SimpleNamespace(scoped=scoped, flag=flag, visibility=visibility)


async def _mint(client, scope):
    return await client.post(
        "/api/internal/github/clone-token",
        json={"repo_url": "https://github.com/acme/widgets", "scope": scope},
        headers={"X-Internal-Token": _INTERNAL},
    )


class TestPushTokenMint:
    async def test_push_token_for_public_repo_is_403_and_withheld(self, async_client, mint):
        mint.flag.return_value = False
        resp = await _mint(async_client, "push")
        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        assert "ghs_scoped" not in resp.text
        mint.flag.assert_called_once_with("acme", "widgets", "ghs_scoped")

    async def test_push_token_for_unknown_visibility_is_refused(self, async_client, mint):
        mint.flag.return_value = None
        resp = await _mint(async_client, "push")
        assert resp.status_code == 503, resp.text
        assert _code(resp) == REPO_VISIBILITY_UNKNOWN, resp.text
        assert "ghs_scoped" not in resp.text

    async def test_read_token_for_public_repo_is_still_minted(self, async_client, mint):
        mint.visibility.return_value = "public"
        mint.flag.return_value = False
        resp = await _mint(async_client, "clone")
        assert resp.status_code == 200, resp.text
        assert resp.json()["token"] == "ghs_scoped"
        assert resp.json()["visibility"] == "public"
        mint.flag.assert_not_called()

    async def test_push_token_for_private_repo_is_minted(self, async_client, mint):
        resp = await _mint(async_client, "push")
        assert resp.status_code == 200, resp.text
        assert resp.json()["token"] == "ghs_scoped"
        assert resp.json()["visibility"] == "private"
        mint.scoped.assert_called_once_with("acme", "widgets", scope="push")


class TestStrictPrivateFlag:
    """``get_repo_private_flag`` must NOT fold "could not tell" into private (the
    clone-side ``get_repo_visibility`` does, deliberately)."""

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (200, {"private": True}, True),
            (200, {"private": False}, False),
            (200, {}, None),
            (404, {}, None),
        ],
        ids=["private", "public", "absent", "404"],
    )
    def test_raw_flag(self, status, body, expected):
        from services import github_app_auth as ga

        resp = SimpleNamespace(status_code=status, json=lambda: body)
        with patch.object(ga.httpx, "get", MagicMock(return_value=resp)):
            assert ga.get_repo_private_flag("acme", "widgets", "ghs_x") is expected


# ===========================================================================
# 3. create-pull-request
# ===========================================================================
async def _create_pr(client):
    return await client.post(
        f"{BASE}/create-pull-request",
        json={
            "original_owner": "owner",
            "original_repo": "repo",
            "fork_owner": "owner",
            "fork_repo": "repo",
            "head_branch": "aa/x",
            "base_branch": "main",
            "title": "t",
            "message": "m",
        },
    )


class TestCreatePullRequest:
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_public_base_repo_is_refused_before_any_write(self, mock_gh_cls, async_client):
        inst = _integration(private=False)
        mock_gh_cls.return_value = inst

        resp = await _create_pr(async_client)

        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        inst.github_client.get_repo.assert_called_once_with("owner/repo")
        _assert_no_write(inst)

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_private_base_repo_gets_its_pr(self, mock_gh_cls, async_client):
        inst = _integration(private=True)
        mock_gh_cls.return_value = inst

        resp = await _create_pr(async_client)

        assert resp.status_code == 200, resp.text
        inst.create_pull_request.assert_called_once()

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_lookup_failure_is_503_and_nothing_written(self, mock_gh_cls, async_client):
        inst = _integration(lookup_error=RuntimeError("rate limited"))
        mock_gh_cls.return_value = inst

        resp = await _create_pr(async_client)

        assert resp.status_code == 503, resp.text
        assert _code(resp) == REPO_VISIBILITY_UNKNOWN, resp.text
        _assert_no_write(inst)


# ===========================================================================
# 4. merge-pull-request (service + route)
# ===========================================================================
class _Run:
    name, status, conclusion = "ci", "completed", "success"


class _Commit:
    def get_check_runs(self):
        return [_Run()]


class _PR:
    def __init__(self) -> None:
        self.merged, self.mergeable, self.title = False, True, "t"
        self.head = SimpleNamespace(
            sha="deadbeef", ref="aa/fix-1", repo=SimpleNamespace(full_name="o/r")
        )
        self.base = SimpleNamespace(ref="main")
        self.merge_calls: list[dict] = []

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        return SimpleNamespace(merged=True, sha="abc", message="ok")


class _Repo:
    default_branch = "main"

    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)  # `private` only when given — absent is a case
        self.pr = _PR()
        self.deleted_refs: list[str] = []

    def get_pull(self, _n):
        return self.pr

    def commit_for(self, _sha):
        return _Commit()

    def get_git_ref(self, ref):
        repo = self
        return SimpleNamespace(delete=lambda: repo.deleted_refs.append(ref))


def _svc_merge(repo):
    s = GitHubService(GitHubConfig(token_id="t"))

    async def _gr(_name):
        return repo

    async def _ex(func, *a, **k):
        return func(*a, **k)

    s.get_repository = _gr  # type: ignore[method-assign]
    s._execute_with_retry = _ex  # type: ignore[method-assign]
    return asyncio.run(s.merge_pull_request("o/r", 1, require_checks=False))


class TestMergeService:
    def test_public_repo_is_refused_and_neither_merged_nor_branch_deleted(self):
        repo = _Repo(private=False)
        with pytest.raises(RepoVisibilityRefusedError) as exc:
            _svc_merge(repo)
        assert exc.value.error_code == PUBLIC_REPO_REFUSED
        assert repo.pr.merge_calls == []
        assert repo.deleted_refs == []

    def test_private_repo_merges(self):
        repo = _Repo(private=True)
        out = _svc_merge(repo)
        assert out["merged"] is True
        assert len(repo.pr.merge_calls) == 1

    def test_absent_visibility_is_unknown_and_nothing_merged(self):
        repo = _Repo()
        with pytest.raises(RepoVisibilityRefusedError) as exc:
            _svc_merge(repo)
        assert exc.value.error_code == REPO_VISIBILITY_UNKNOWN
        assert repo.pr.merge_calls == []


@pytest.fixture
def merge_repo(monkeypatch, platform_tenant):
    """The App rung armed and ``get_repository`` stubbed (mirrors test_andon_chokepoint)."""
    import services.github_app_auth as ga
    import services.github_integration as gi

    holder: dict = {}

    monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
    monkeypatch.setattr(ga, "get_installation_token_for", lambda _o, _r: "ghs_test")
    monkeypatch.setattr("services.github.github_service.HAS_GITHUB", True)

    async def _get_repo(_self, _full_name):
        return holder["repo"]

    monkeypatch.setattr(GitHubService, "get_repository", _get_repo)
    ga.reset_cache()
    yield holder
    ga.reset_cache()


async def _merge_route(client):
    return await client.post(
        MERGE_URL,
        json={
            "owner": "nonagenticai",
            "repo": "enterprise_mcp_server",
            "pr_number": 140,
            "require_checks": False,
            "sha": "deadbeef" + "0" * 32,
        },
    )


class TestMergeRoute:
    async def test_public_is_403_on_the_wire(self, async_client, merge_repo):
        merge_repo["repo"] = _Repo(private=False)
        resp = await _merge_route(async_client)
        assert resp.status_code == 403, resp.text
        assert _code(resp) == PUBLIC_REPO_REFUSED, resp.text
        assert merge_repo["repo"].pr.merge_calls == []

    async def test_unknown_is_503_on_the_wire(self, async_client, merge_repo):
        merge_repo["repo"] = _Repo(private=None)
        resp = await _merge_route(async_client)
        assert resp.status_code == 503, resp.text
        assert _code(resp) == REPO_VISIBILITY_UNKNOWN, resp.text
        assert merge_repo["repo"].pr.merge_calls == []


# ===========================================================================
# 5. create-upstream-pr (service)
# ===========================================================================
def _upstream(repo_private):
    """Mirrors test_github_integration.py::TestUpstreamPRCreation's harness."""
    from services.github_integration import GitHubIntegration

    original_repo = MagicMock()
    original_repo.private = repo_private
    original_repo.create_pull.return_value = SimpleNamespace(
        number=42, html_url="https://github.com/owner/repo/pull/42"
    )
    gh = MagicMock()
    gh.get_repo.return_value = original_repo
    containers = AsyncMock()
    containers.get_workspace = AsyncMock(
        return_value={
            "user_id": "u",
            "user_token_id": "token-123",
            "original_repo_owner": "owner",
            "original_repo_name": "repo",
            "system_fork_owner": "nonagenticai",
        }
    )
    tm = MagicMock()
    tm.get_user_token = AsyncMock(return_value="user-github-token")
    with (
        patch("services.github_integration.HAS_GITHUB", True),
        patch("services.github_integration.Auth"),
        patch("services.github_integration.Github", return_value=gh),
        patch("main.get_workspaces_client", return_value=containers),
        patch("services.github_token_manager.get_github_token_manager", return_value=tm),
    ):
        integration = GitHubIntegration(github_token="test-token")
        result = asyncio.run(
            integration.create_upstream_pull_request(
                container_name="c", title="t", message="m", head_branch="aa/x"
            )
        )
    return result, original_repo


class TestUpstreamPR:
    def test_public_original_repo_is_refused_before_create_pull(self):
        result, original_repo = _upstream(False)
        assert result["success"] is False, result
        assert (result["status_code"], result["error_code"]) == (403, PUBLIC_REPO_REFUSED)
        original_repo.create_pull.assert_not_called()

    def test_unknown_visibility_is_503(self):
        result, original_repo = _upstream(None)
        assert (result["status_code"], result["error_code"]) == (503, REPO_VISIBILITY_UNKNOWN)
        original_repo.create_pull.assert_not_called()


# ===========================================================================
# 6. MUTATION CONTROLS — neutralise the guard at each call site: the public request
#    must then reach the write. If these ever pass with the guard in place, or the
#    refusals above pass with it removed, the refusals are not coming from the guard.
# ===========================================================================
class TestTheGuardIsWhatRefuses:
    @patch("main.get_workspaces_client")
    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_apply_without_the_guard_writes_a_public_repo(
        self, mock_gh_cls, mock_get_client, async_client
    ):
        mock_get_client.return_value = _workspaces_client()
        inst = _integration(private=False)
        mock_gh_cls.return_value = inst
        with patch(
            "routers.github_pull_request_operations.assert_repo_not_public_lookup",
            lambda *_a, **_k: None,
        ):
            resp = await _apply(async_client)
        assert resp.status_code == 200, resp.text
        inst._apply_changes.assert_called_once()

    @patch("routers.github_pull_request_operations.GitHubIntegration")
    async def test_create_pr_without_the_guard_opens_on_a_public_repo(
        self, mock_gh_cls, async_client
    ):
        inst = _integration(private=False)
        mock_gh_cls.return_value = inst
        with patch(
            "routers.github_pull_request_operations.assert_repo_not_public_lookup",
            lambda *_a, **_k: None,
        ):
            resp = await _create_pr(async_client)
        assert resp.status_code == 200, resp.text
        inst.create_pull_request.assert_called_once()

    def test_merge_without_the_guard_merges_a_public_repo(self):
        repo = _Repo(private=False)
        with patch("services.github.github_service.assert_repo_not_public", lambda *_a: None):
            out = _svc_merge(repo)
        assert out["merged"] is True
        assert len(repo.pr.merge_calls) == 1

    def test_upstream_without_the_guard_opens_on_a_public_repo(self):
        with patch("services.github_integration.assert_repo_not_public", lambda *_a: None):
            _result, original_repo = _upstream(False)
        original_repo.create_pull.assert_called_once()
