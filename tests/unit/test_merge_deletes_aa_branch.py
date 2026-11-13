"""After a successful merge the head branch is deleted iff it is ``aa/*``.

``main``, a human's ``feature/x``, a fork's head, and a branch whose merge was refused
are never deleted. (The SHA pin itself is #117's — see test_github_pr_status_operations.)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from services.github.github_service import (
    GitHubConfig,
    GitHubService,
    aa_head_branch_deletable,
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


SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
REPO = "o/r"


class _Run:
    name, status, conclusion = "ci", "completed", "success"


class _Commit:
    def get_check_runs(self):
        return [_Run()]


class _PR:
    def __init__(self, head_sha=SHA, head_ref="aa/fix-1", head_repo=REPO, merge_error=None):
        self.merged, self.mergeable, self.title = False, True, "t"
        self.head = SimpleNamespace(
            sha=head_sha, ref=head_ref, repo=SimpleNamespace(full_name=head_repo)
        )
        self.base = SimpleNamespace(ref="main")
        self.merge_calls: list[dict] = []
        self._merge_error = merge_error

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        if self._merge_error is not None:
            raise self._merge_error
        return SimpleNamespace(merged=True, sha="m" * 40, message="ok")


class _Repo:
    private = True  # a private target repo (a public one is refused before merge)
    default_branch = "main"

    def __init__(self, pr):
        self.pr = pr
        self.deleted_refs: list[str] = []

    def get_pull(self, _n):
        return self.pr

    def commit_for(self, _sha):
        return _Commit()

    def get_git_ref(self, ref):
        repo = self
        return SimpleNamespace(delete=lambda: repo.deleted_refs.append(ref))


def _svc(repo) -> GitHubService:
    s = GitHubService(GitHubConfig(token_id="t"))

    async def _gr(_name):
        return repo

    async def _ex(func, *a, **k):
        return func(*a, **k)

    s.get_repository = _gr  # type: ignore[method-assign]
    s._execute_with_retry = _ex  # type: ignore[method-assign]
    return s


def _merge(repo, sha=SHA, require_checks=True):
    return asyncio.run(
        _svc(repo).merge_pull_request(REPO, 1, require_checks=require_checks, sha=sha)
    )


class TestBranchCleanup:
    def test_aa_branch_is_deleted_after_merge(self):
        repo = _Repo(_PR(head_ref="aa/fix-1"))
        out = _merge(repo)
        assert out["head_branch_deleted"] is True
        assert repo.deleted_refs == ["heads/aa/fix-1"]

    @pytest.mark.parametrize("ref", ["main", "feature/x", "aa", "aa/"])
    def test_non_aa_branch_is_never_deleted(self, ref):
        repo = _Repo(_PR(head_ref=ref))
        out = _merge(repo)
        assert out["merged"] is True
        assert out["head_branch_deleted"] is False
        assert repo.deleted_refs == []

    def test_fork_head_is_not_deleted(self):
        repo = _Repo(_PR(head_repo="someone/r"))
        out = _merge(repo)
        assert out["head_branch_delete_skipped"] == "head_on_other_repository"
        assert repo.deleted_refs == []

    def test_no_delete_when_the_merge_is_refused(self):
        repo = _Repo(_PR())
        repo.pr.mergeable = False
        out = _merge(repo)
        assert out["reason"] == "conflicts"
        assert repo.deleted_refs == []

    @pytest.mark.parametrize(
        ("ref", "default", "reason"),
        [
            ("aa/x", "main", None),
            ("aa/x", "aa/x", "is_default_or_base_branch"),
            ("aa/x", None, "default_branch_unknown"),
            (None, "main", "head_branch_unknown"),
            ("feature/x", "main", "not_aa_branch"),
        ],
    )
    def test_predicate_fails_closed(self, ref, default, reason):
        assert aa_head_branch_deletable(ref, default, "main") == reason
