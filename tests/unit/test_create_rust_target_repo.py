"""``create_rust_target_repo`` + the ``conveyor-coding/*-rs`` allowlist name pattern.

The op is exercised END TO END with REAL git and REAL bash: "GitHub" for git is a
directory of bare repositories reached over ``file://`` (``rust_target.GIT_BASE_URL``),
seeded with a template that carries its own ``scripts/rename.sh``. The REST API is an
``httpx.MockTransport`` whose ``POST /orgs/{org}/repos`` really creates the bare target —
so a push without a create fails exactly as it would on GitHub. Only the App token mints
(``services.github_app_auth``) are replaced, and every mint's permission set is asserted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from coding_client.auth.tenant import require_tenant

import services.github_app_auth as ga
import services.github_integration as gi
import services.rust_target as rt
import services.tenant_binding as tb
from config.settings import (
    get_settings,
    grant_covers,
    parse_platform_repo_allowlist,
    parse_platform_repo_allowlist_modes,
    parse_sandbox_github_owners,
)
from services.github_app_auth import AppInstallation, GitHubAppMintError, ScopedToken

PLATFORM = "nonagenticai"
SANDBOX = "conveyor-coding"
EXEC_TENANT = "0195f48b-d000-7000-8000-000000000003"  # platform tenant
CONVEYOR_TENANT = "5f868d42-03f8-51f0-a93c-3e83070dc196"  # allowlisted, NOT platform
CUSTOMER_TENANT = "11111111-1111-1111-1111-111111111111"
PATTERN = f"{SANDBOX}/*-rs"
URL = "/api/repositories/create-rust-target-repo"
TARGET = f"{SANDBOX}/repo-jobs-rs"
SOURCE = f"{SANDBOX}/repo-jobs"
PLATFORM_SOURCE = f"{PLATFORM}/repo-jobs"

RENAME_SH = r"""#!/usr/bin/env bash
set -euo pipefail
name="$1"; flag="${2:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
ex=.github/workflows/ci-self-contained.yml.example
if [[ "$flag" == --self-contained-ci ]]; then mv -f "$ex" .github/workflows/ci.yml; else rm -f "$ex"; fi
for f in $(git ls-files | grep -v '^scripts/rename.sh$'); do
  [[ -f "$f" ]] || continue
  sed -e "s/template-rust-service/${name}/g" -e "s/template_rust_service/${name//-/_}/g" "$f" > "$f.tmp"
  cat "$f.tmp" > "$f"; rm -f "$f.tmp"
done
"""
GITHUB_WORKFLOW_REFUSAL = (
    "refusing to allow a GitHub App to create or update workflow "
    "`.github/workflows/ci.yml` without `workflows` permission"
)


@pytest.fixture(autouse=True)
def _no_inherited_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every git call here against its own tmp repo, never the caller's.

    Inside a git hook (pre-commit/pre-push run this suite) git exports GIT_DIR,
    GIT_INDEX_FILE, … to the hook. Inherited, they point this file's `git init` /
    `git commit` at the CHECKOUT running the hook: measured 2026-11-03, it set
    core.bare=true in the shared .git/config and committed onto the branch.
    """
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# settings: the sandbox name pattern
# ---------------------------------------------------------------------------
def _parse(*entries: str, sandbox: frozenset[str] = frozenset({SANDBOX})):
    import json

    return parse_platform_repo_allowlist_modes(
        json.dumps({CONVEYOR_TENANT: list(entries)}),
        platform_owner=PLATFORM,
        platform_tenants=frozenset({EXEC_TENANT}),
        sandbox_owners=sandbox,
    )


class TestNamePattern:
    def test_pattern_is_a_write_grant_alongside_exact_entries(self):
        grants, read_only = _parse("Conveyor-Coding/*-RS", f"{SANDBOX}/py2rs-fixture-svc:read")
        assert grants == {CONVEYOR_TENANT: frozenset({PATTERN, f"{SANDBOX}/py2rs-fixture-svc"})}
        assert read_only == {CONVEYOR_TENANT: frozenset({f"{SANDBOX}/py2rs-fixture-svc"})}

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            (f"{PLATFORM}/*-rs", "only for a sandbox owner"),
            ("acme-corp/*-rs", "only for a sandbox owner"),
            ("*/repo-rs", "wildcard owner is refused"),
            (f"{SANDBOX}/*", "ONE leading"),
            (f"{SANDBOX}/repo*-rs", "ONE leading"),
            (f"{SANDBOX}/**-rs", "ONE leading"),
            (f"{SANDBOX}/*-rs*", "ONE leading"),
            (f"{PATTERN}:read", "always a write grant"),
        ],
    )
    def test_malformed_or_misplaced_patterns_fail_boot(self, entry, match):
        with pytest.raises(RuntimeError, match=match):
            _parse(entry)

    def test_pattern_needs_the_owner_to_be_a_configured_sandbox(self):
        with pytest.raises(RuntimeError, match="only for a sandbox owner"):
            _parse(PATTERN, sandbox=frozenset())

    def test_duplicate_pattern_fails_boot(self):
        with pytest.raises(RuntimeError, match="more than once"):
            _parse(PATTERN, "conveyor-coding/*-RS")

    @pytest.mark.parametrize(
        ("owner", "repo", "covered"),
        [
            (SANDBOX, "repo-jobs-rs", True),
            ("Conveyor-Coding", "Repo-Jobs-RS", True),
            (SANDBOX, "foo", False),  # the pattern grants *-rs, nothing else
            (SANDBOX, "repo-jobs-rs-old", False),
            (SANDBOX, "-rs", False),  # the suffix alone is not a name
            (PLATFORM, "repo-jobs-rs", False),
            ("acme-corp", "repo-jobs-rs", False),
            (SANDBOX, "a/b-rs", False),
        ],
    )
    def test_grant_covers(self, owner, repo, covered):
        assert grant_covers(frozenset({PATTERN}), owner, repo) is covered

    def test_exact_entries_still_match_exactly(self):
        entries = frozenset({f"{SANDBOX}/aa-bugfix-capstone-fixture"})
        assert grant_covers(entries, SANDBOX, "aa-bugfix-capstone-fixture")
        assert not grant_covers(entries, SANDBOX, "aa-bugfix-capstone-fixture-rs")

    def test_wrapper_returns_the_pattern(self):
        grants = parse_platform_repo_allowlist(
            f'{{"{CONVEYOR_TENANT}": ["{PATTERN}"]}}',
            platform_owner=PLATFORM,
            platform_tenants=frozenset(),
            sandbox_owners=frozenset({SANDBOX}),
        )
        assert grants == {CONVEYOR_TENANT: frozenset({PATTERN})}


# ---------------------------------------------------------------------------
# shared world: sandbox configured, conveyor tenant holds the pattern
# ---------------------------------------------------------------------------
def _settings_objects():
    real = get_settings()
    return (real,) if gi.settings is real else (real, gi.settings)


def _patch_all(monkeypatch, name, value):
    for obj in _settings_objects():
        monkeypatch.setattr(obj, name, value)


@pytest.fixture
def grants(monkeypatch):
    """Sandbox org + the conveyor tenant's allowlist; returns a setter for the entries."""
    owners = parse_sandbox_github_owners(SANDBOX, platform_owner=PLATFORM)
    _patch_all(monkeypatch, "system_github_owner", PLATFORM)
    _patch_all(monkeypatch, "_sandbox_owners", owners)

    def set_entries(*entries: str) -> None:
        full, read_only = _parse(*entries)
        _patch_all(monkeypatch, "_platform_repo_grants", full)
        _patch_all(monkeypatch, "_platform_repo_read_only_grants", read_only)

    # The conveyor tenant's real shape: the target pattern + each source as ':read'.
    set_entries(PATTERN, f"{SOURCE}:read", f"{PLATFORM_SOURCE}:read")
    return set_entries


class TestSandboxGrantHonoursPattern:
    def test_pattern_grants_new_target_and_nothing_else(self, grants):
        assert tb.sandbox_repo_grant(CONVEYOR_TENANT, SANDBOX, "brand-new-rs") == "scoped"
        assert tb.sandbox_repo_grant(CONVEYOR_TENANT, SANDBOX, "brand-new") == "refuse"
        assert tb.sandbox_repo_grant(CUSTOMER_TENANT, SANDBOX, "brand-new-rs") == "refuse"
        assert not tb.is_read_only_grant(CONVEYOR_TENANT, SANDBOX, "brand-new-rs")

    def test_pattern_never_widens_a_platform_repo(self, grants):
        # The internal-App path matches exactly; a sandbox pattern cannot reach it.
        assert tb.platform_repo_grant(CONVEYOR_TENANT, PLATFORM, "brand-new-rs") == "refuse"


# ---------------------------------------------------------------------------
# the op, end to end
# ---------------------------------------------------------------------------
class FakeGitHub:
    """GitHub REST for the op; ``POST /orgs/{o}/repos`` creates a real bare repo."""

    def __init__(self, root: Path):
        self.root = root
        self.repos: dict[str, dict[str, Any]] = {
            SOURCE: {"private": True, "owner": {"login": SANDBOX}},
            PLATFORM_SOURCE: {"private": True, "owner": {"login": PLATFORM}},
        }
        self.calls: list[tuple[str, str, str]] = []  # (method, path, bearer)
        self.create_bodies: list[dict[str, Any]] = []
        self.create_answer: tuple[int, dict[str, Any]] | None = None
        #: How ``DELETE /installation/token`` answers: a status, or an exception to raise.
        self.revoke_answer: int | str = 204

    def handler(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        path = request.url.path
        self.calls.append((request.method, path, bearer))
        if request.method == "GET" and path.startswith("/repos/"):
            full = path.removeprefix("/repos/")
            repo = self.repos.get(full)
            if repo is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"full_name": full, **repo})
        if request.method == "POST" and path.startswith("/orgs/"):
            org = path.split("/")[2]
            body = __import__("json").loads(request.content)
            self.create_bodies.append(body)
            if self.create_answer is not None:
                return httpx.Response(self.create_answer[0], json=self.create_answer[1])
            bare = self.root / org / f"{body['name']}.git"
            bare.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
            self.repos[f"{org}/{body['name']}"] = {"private": True, "owner": {"login": org}}
            return httpx.Response(
                201, json={"html_url": f"https://github.com/{org}/{body['name']}"}
            )
        if request.method == "DELETE" and path == "/installation/token":
            if self.revoke_answer == "raise":  # a transport error that echoes the credential
                raise httpx.ConnectError(f"down {bearer}")
            return httpx.Response(self.revoke_answer)
        raise AssertionError(f"unexpected GitHub call {request.method} {path}")

    @property
    def revokes(self) -> list[str]:
        """Bearer of every ``DELETE /installation/token``, in order."""
        return [b for m, p, b in self.calls if (m, p) == ("DELETE", "/installation/token")]

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=rt.GITHUB_API_BASE, transport=httpx.MockTransport(self.handler)
        )


@pytest.fixture
def template(tmp_path) -> tuple[Path, str]:
    """A bare ``nonagenticai/template-rust-service`` with history; returns (root, HEAD)."""
    root = tmp_path / "gh"
    work = tmp_path / "template-work"
    (work / "scripts").mkdir(parents=True)
    (work / ".github/workflows").mkdir(parents=True)
    (work / "scripts/rename.sh").write_text(RENAME_SH)
    (work / "Cargo.toml").write_text('[package]\nname = "template-rust-service"\n')
    (work / "src").mkdir()
    (work / "src/lib.rs").write_text("pub mod template_rust_service {}\n")
    (work / ".github/workflows/ci.yml").write_text(
        "uses: example-org/example-workflows/.github/workflows/rust-service.yml@main\n"
    )
    (work / ".github/workflows/ci-self-contained.yml.example").write_text(
        "runs-on: ubuntu-latest  # template-rust-service\n"
    )
    env = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    _git("init", "-q", "-b", "main", cwd=work)
    _git("add", "-A", cwd=work)
    _git(*env, "commit", "-q", "-m", "first", cwd=work)
    (work / "README.md").write_text("template-rust-service\n")
    _git("add", "-A", cwd=work)
    _git(*env, "commit", "-q", "-m", "second", cwd=work)
    bare = root / PLATFORM / "template-rust-service.git"
    bare.parent.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(work), str(bare), cwd=tmp_path)
    return root, _git("rev-parse", "HEAD", cwd=work)


class Mints:
    def __init__(self) -> None:
        self.org: list[tuple[int, dict[str, str]]] = []
        self.push: list[tuple[int, str, str, dict[str, str], str | None]] = []
        self.clone: list[tuple[str, str]] = []
        self.org_error: Exception | None = None
        #: Every token handed out, in mint order; each one must be revoked.
        self.issued: list[str] = []
        self.fresh: list[bool] = []
        self.push_error: Exception | None = None


@pytest.fixture
def world(monkeypatch, grants, template):
    root, template_sha = template
    gh = FakeGitHub(root)
    mints = Mints()

    def org_token(installation_id, permissions):
        mints.org.append((installation_id, dict(permissions)))
        if mints.org_error:
            raise mints.org_error
        mints.issued.append("ghs_org")
        return "ghs_org"

    def push_token(installation_id, owner, repo, permissions, *, expected_owner=None, fresh=False):
        mints.push.append((installation_id, owner, repo, dict(permissions), expected_owner))
        if mints.push_error:
            raise mints.push_error
        mints.issued.append("ghs_push")
        mints.fresh.append(fresh)
        return ScopedToken("ghs_push", "2999-01-01T00:00:00Z")

    def clone_token(owner, repo, scope="clone", *, fresh=False):
        mints.clone.append((owner, repo))
        mints.issued.append(f"ghs_clone_{repo}")
        mints.fresh.append(fresh)
        return ScopedToken(f"ghs_clone_{repo}", "2999-01-01T00:00:00Z")

    monkeypatch.setattr(rt, "GIT_BASE_URL", f"file://{root}")
    monkeypatch.setattr(rt, "_http_client", gh.client)
    monkeypatch.setattr(ga, "public_app_configured", lambda: True)
    monkeypatch.setattr(
        ga,
        "resolve_public_org_installation",
        lambda org: AppInstallation(555, SANDBOX, "Organization"),
    )
    monkeypatch.setattr(ga, "mint_public_org_token", org_token)
    monkeypatch.setattr(ga, "mint_public_repo_token", push_token)
    monkeypatch.setattr(ga, "get_scoped_clone_token", clone_token)
    gh.template_sha = template_sha  # type: ignore[attr-defined]
    gh.mints = mints  # type: ignore[attr-defined]
    return gh


@pytest.fixture
def as_tenant():
    from main import app

    def use(tenant: str) -> None:
        app.dependency_overrides[require_tenant] = lambda: tenant

    yield use
    app.dependency_overrides.pop(require_tenant, None)


def _bare(world, name: str = "repo-jobs-rs") -> Path:
    return world.root / SANDBOX / f"{name}.git"


class TestCreateRustTargetRoute:
    async def test_creates_private_renamed_single_commit_target(
        self, async_client, world, as_tenant
    ):
        as_tenant(CONVEYOR_TENANT)  # granted ONLY by the conveyor-coding/*-rs pattern
        resp = await async_client.post(
            URL, json={"target": TARGET, "sources": [SOURCE, PLATFORM_SOURCE]}
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        bare = _bare(world)
        head = _git("rev-parse", "refs/heads/main", cwd=bare)
        assert body == {
            "target": TARGET,
            "url": f"https://github.com/{TARGET}",
            "default_branch": "main",
            "head_sha": head,
            "template_sha": world.template_sha,
        }
        # ONE commit, no template history, the script's exact message.
        assert _git("rev-list", "--count", "main", cwd=bare) == "1"
        assert _git("log", "-1", "--format=%s", "main", cwd=bare) == (
            f"Initial commit from template-rust-service@{world.template_sha}"
        )
        assert _git("show", "main:Cargo.toml", cwd=bare) == '[package]\nname = "repo-jobs-rs"'
        assert _git("show", "main:src/lib.rs", cwd=bare) == "pub mod repo_jobs_rs {}"
        # --self-contained-ci: the GitHub-hosted workflow replaced the archetype caller.
        assert "ubuntu-latest" in _git("show", "main:.github/workflows/ci.yml", cwd=bare)
        files = _git("ls-tree", "-r", "--name-only", "main", cwd=bare).splitlines()
        assert ".github/workflows/ci-self-contained.yml.example" not in files
        # Created PRIVATE and empty; the org token carries ONLY administration.
        assert world.create_bodies == [
            {
                "name": "repo-jobs-rs",
                "private": True,
                "auto_init": False,
                "description": f"Rust rewrite of {SOURCE}, {PLATFORM_SOURCE}",
            }
        ]
        assert world.mints.org == [(555, {"administration": "write"})]
        assert world.mints.push == [
            (
                555,
                SANDBOX,
                "repo-jobs-rs",
                {"contents": "write", "workflows": "write"},
                SANDBOX,
            )
        ]
        # Template read with the internal App's one-repo token; the platform source's
        # privacy read with its own one-repo token, the sandbox source with the org's.
        assert (PLATFORM, "template-rust-service") in world.mints.clone
        bearer_by_path = {p: b for m, p, b in world.calls if m == "GET"}
        assert bearer_by_path[f"/repos/{PLATFORM_SOURCE}"] == "ghs_clone_repo-jobs"
        assert bearer_by_path[f"/repos/{SOURCE}"] == "ghs_org"
        # EVERY minted token is revoked exactly once (concurrently, so in any order); and
        # every scoped token was minted uncached (fresh), so revoking it cannot poison a cache.
        assert sorted(world.revokes) == sorted(
            ["ghs_push", "ghs_clone_template-rust-service", "ghs_clone_repo-jobs", "ghs_org"]
        )
        assert sorted(world.revokes) == sorted(world.mints.issued)
        assert world.mints.fresh and all(world.mints.fresh)
        assert world.calls[-1][:2] == ("DELETE", "/installation/token")

    @pytest.mark.parametrize(
        ("payload", "code"),
        [
            (
                {"target": f"{PLATFORM}/repo-jobs-rs", "sources": [SOURCE]},
                "target_owner_not_allowed",
            ),
            ({"target": f"{SANDBOX}/repo-jobs", "sources": [SOURCE]}, "target_name_not_rs"),
            ({"target": f"{SANDBOX}/Repo_Jobs-rs", "sources": [SOURCE]}, "target_name_invalid"),
            ({"target": TARGET, "sources": ["acme/repo-jobs"]}, "source_owner_not_allowed"),
            ({"target": TARGET, "sources": [TARGET]}, "source_is_target"),
            ({"target": TARGET, "sources": ["no-slash"]}, "invalid_source"),
            ({"target": TARGET, "sources": [f"{SANDBOX}/."]}, "invalid_source"),
            ({"target": TARGET, "sources": [f"{PLATFORM}/.."]}, "invalid_source"),
        ],
    )
    async def test_input_refusals_are_422_before_github(
        self, async_client, world, as_tenant, payload, code
    ):
        as_tenant(EXEC_TENANT)
        resp = await async_client.post(URL, json=payload)
        assert resp.status_code == 422, resp.text
        assert code in resp.text
        assert world.calls == [] and world.mints.org == []

    async def test_unusable_source_answers_identically_whatever_the_reason(
        self, async_client, world, as_tenant, monkeypatch
    ):
        """Missing, not visible, public, owned elsewhere: ONE answer, byte for byte, so a
        tenant cannot probe whether a PRIVATE nonagenticai repo exists."""
        as_tenant(EXEC_TENANT)
        payload = {"target": TARGET, "sources": [PLATFORM_SOURCE]}

        async def answer() -> tuple[int, Any]:
            resp = await async_client.post(URL, json=payload)
            error = dict(resp.json()["error"])
            error.pop("request_id"), error.pop("timestamp")  # per-request, not per-reason
            return resp.status_code, error

        world.repos[PLATFORM_SOURCE]["private"] = False  # exists, public
        public = await answer()
        world.repos[PLATFORM_SOURCE] = {"private": True, "owner": {"login": "elsewhere"}}
        moved = await answer()
        del world.repos[PLATFORM_SOURCE]  # GitHub 404 with a token that could see it
        missing = await answer()

        def not_installed(
            owner, repo, scope="clone", *, fresh=False
        ):  # App lookup 404: not visible
            raise ga.GitHubAppNotInstalledError("404")

        monkeypatch.setattr(ga, "get_scoped_clone_token", not_installed)
        invisible = await answer()
        assert public == moved == missing == invisible
        assert public[0] == 422 and "source_not_usable" in str(public[1])
        assert world.create_bodies == []
        # revoked on every refusal too: the source token (when one was minted) + org
        assert sorted(world.revokes) == sorted(["ghs_clone_repo-jobs", "ghs_org"] * 3 + ["ghs_org"])
        assert sorted(world.revokes) == sorted(world.mints.issued)

    async def test_existing_target_is_409_and_untouched(self, async_client, world, as_tenant):
        as_tenant(EXEC_TENANT)
        world.repos[TARGET] = {"private": True, "owner": {"login": SANDBOX}}
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 409 and "target_exists" in resp.text
        assert world.create_bodies == [] and world.mints.clone == []
        assert sorted(world.revokes) == sorted(["ghs_org"])
        assert sorted(world.revokes) == sorted(world.mints.issued)

    async def test_create_race_already_exists_is_409(self, async_client, world, as_tenant):
        as_tenant(EXEC_TENANT)
        world.create_answer = (
            422,
            {
                "message": "Repository creation failed.",
                "errors": [{"field": "name", "message": "name already exists on this account"}],
            },
        )
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 409 and "target_exists" in resp.text

    async def test_tenant_without_grant_is_403_before_github(self, async_client, world, as_tenant):
        as_tenant(CUSTOMER_TENANT)
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403 and "platform_repo_forbidden" in resp.text
        assert world.calls == [] and world.mints.org == []

    async def test_read_only_exact_grant_cannot_create(
        self, async_client, world, grants, as_tenant
    ):
        grants(f"{TARGET}:read")
        as_tenant(CONVEYOR_TENANT)
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403 and "platform_repo_read_only" in resp.text
        assert world.calls == []

    async def test_github_403_on_create_names_repository_administration(
        self, async_client, world, as_tenant
    ):
        as_tenant(EXEC_TENANT)
        world.create_answer = (403, {"message": "Resource not accessible by integration"})
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403
        assert "github_app_permission_missing" in resp.text
        assert "Repository permissions → Administration: write" in resp.json()["error"]["message"]
        assert sorted(world.revokes) == sorted(["ghs_clone_template-rust-service", "ghs_org"])
        assert sorted(world.revokes) == sorted(world.mints.issued)

    async def test_ungranted_org_mint_names_repository_administration(
        self, async_client, world, as_tenant
    ):
        as_tenant(EXEC_TENANT)
        world.mints.org_error = GitHubAppMintError("refused", status_code=422)
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403
        assert "Repository permissions → Administration: write" in resp.json()["error"]["message"]
        assert world.calls == []  # no token was minted, so there is nothing to revoke

    async def test_ungranted_push_mint_names_workflows_and_the_empty_repo(
        self, async_client, world, as_tenant
    ):
        as_tenant(EXEC_TENANT)
        world.mints.push_error = GitHubAppMintError("refused", status_code=422)
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403
        message = resp.json()["error"]["message"]
        assert "Repository permissions → Workflows: write" in message
        assert f"{TARGET} was CREATED but is EMPTY" in message
        assert sorted(world.revokes) == sorted(["ghs_clone_template-rust-service", "ghs_org"])
        assert sorted(world.revokes) == sorted(world.mints.issued)

    async def test_github_workflow_push_refusal_names_workflows(
        self, async_client, world, as_tenant
    ):
        """The push itself refused the way GitHub refuses it (a real pre-receive hook)."""
        as_tenant(EXEC_TENANT)
        real_handler = world.handler

        def create_with_hook(request):
            resp = real_handler(request)
            if request.method == "POST":
                hook = _bare(world) / "hooks/pre-receive"
                hook.write_text(f"#!/bin/sh\necho '{GITHUB_WORKFLOW_REFUSAL}' >&2\nexit 1\n")
                hook.chmod(0o755)
            return resp

        world.handler = create_with_hook
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 403, resp.text
        assert "Repository permissions → Workflows: write" in resp.json()["error"]["message"]
        with pytest.raises(subprocess.CalledProcessError):
            _git("rev-parse", "refs/heads/main", cwd=_bare(world))
        assert sorted(world.revokes) == sorted(
            ["ghs_push", "ghs_clone_template-rust-service", "ghs_org"]
        )

    async def test_wrong_org_installation_is_refused(
        self, async_client, world, as_tenant, monkeypatch
    ):
        as_tenant(EXEC_TENANT)
        monkeypatch.setattr(
            ga, "resolve_public_org_installation", lambda org: AppInstallation(9, "evil", "User")
        )
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 409 and "github_app_not_installed" in resp.text
        assert world.mints.org == []


class TestRevokeAndBudget:
    @pytest.mark.parametrize("answer", [500, "raise"])
    async def test_revoke_failure_never_fails_the_request_or_logs_the_token(
        self, async_client, world, as_tenant, caplog, answer
    ):
        as_tenant(EXEC_TENANT)
        world.revoke_answer = answer
        with caplog.at_level("WARNING"):
            resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 201, resp.text
        assert sorted(world.revokes) == sorted(
            ["ghs_push", "ghs_clone_template-rust-service", "ghs_org"]
        )
        assert sorted(world.revokes) == sorted(world.mints.issued)
        assert "token revoke" in caplog.text
        for token in world.mints.issued:
            assert token not in caplog.text

    async def test_spent_budget_is_504_before_any_call(
        self, async_client, world, as_tenant, monkeypatch
    ):
        """MUTATION: make ``_Deadline.left`` never raise -> this goes red (a 201)."""
        as_tenant(EXEC_TENANT)
        lookups: list[str] = []
        monkeypatch.setattr(
            ga,
            "resolve_public_org_installation",
            lambda org: lookups.append(org) or AppInstallation(555, SANDBOX, "Organization"),
        )
        popen = MagicMock(side_effect=AssertionError("no subprocess may start"))
        monkeypatch.setattr(rt, "subprocess", _subprocess_with(popen))
        monkeypatch.setattr(rt, "_TOTAL_BUDGET_SECONDS", 0.0)
        resp = await async_client.post(URL, json={"target": TARGET, "sources": [SOURCE]})
        assert resp.status_code == 504 and "deadline_exceeded" in resp.text
        assert world.calls == [] and lookups == [] and world.mints.org == []
        assert world.mints.clone == [] and world.mints.push == []
        popen.assert_not_called()

    @pytest.mark.parametrize("git_seconds", [0.5, 15.0])
    @pytest.mark.parametrize("http_seconds", [0.5 * k for k in range(1, 21)])
    @pytest.mark.parametrize("n_sources", [3, 20])  # 20 = the request schema's max_length
    def test_worst_case_answers_within_120s_on_a_fake_clock(
        self, world, monkeypatch, git_seconds, http_seconds, n_sources
    ):
        """Every App call takes its FULL timeout; every REST call ``http_seconds`` and
        every git step ``git_seconds`` (each capped by the timeout it was given). The
        sweep moves the moment the deadline is crossed across every step, including
        just before the 25 s template-token call. The answer must be back within 120 s
        whatever happens (the node waits 150 s).

        Revokes take their FULL timeout each; the phase costs (number of revokes that
        ran one AFTER another, measured from their real start/end times) x that timeout,
        so a sequential revoke loop over 23 tokens costs 115 s and a concurrent one 5 s.

        MUTATIONS that turn this red: a larger _TOTAL_BUDGET_SECONDS (95 s => ~123 s),
        any per-step cap that lets a controlled step run past the deadline, sequential
        revokes (20 sources => 23 revokes)."""
        from github import Consts

        clock = [0.0]
        monkeypatch.setattr(rt, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        lookup_s = float(Consts.DEFAULT_TIMEOUT)  # PyGithub GithubIntegration lookup
        mint_s = ga._HTTP_TIMEOUT_SECONDS  # raw httpx mint in github_app_auth

        def slow(fn, seconds):
            def wrapped(*a, **kw):
                clock[0] += seconds
                return fn(*a, **kw)

            return wrapped

        monkeypatch.setattr(
            ga,
            "resolve_public_org_installation",
            slow(ga.resolve_public_org_installation, lookup_s),
        )
        monkeypatch.setattr(ga, "mint_public_org_token", slow(ga.mint_public_org_token, mint_s))
        monkeypatch.setattr(
            ga, "get_scoped_clone_token", slow(ga.get_scoped_clone_token, lookup_s + mint_s)
        )
        monkeypatch.setattr(ga, "mint_public_repo_token", slow(ga.mint_public_repo_token, mint_s))
        real_handler = world.handler

        revoke_spans: list[tuple[float, float]] = []
        revoke_cap = [0.0]

        def slow_handler(request):
            cap = max(v for v in request.extensions["timeout"].values() if v)
            if request.method == "DELETE":
                # Concurrency is a REAL-time property: hold each revoke briefly and
                # record when it ran; the fake clock is charged after the op returns.
                started = time.monotonic()
                time.sleep(0.05)
                revoke_spans.append((started, time.monotonic()))
                revoke_cap[0] = max(revoke_cap[0], cap)
                return real_handler(request)
            clock[0] += min(http_seconds, cap)
            return real_handler(request)

        world.handler = slow_handler

        class SlowPopen:
            def __init__(self, args, **kw):
                self.args, self.pid, self.returncode = args, -1, None

            def communicate(self, timeout=None):
                if timeout is not None and git_seconds >= timeout:
                    clock[0] += timeout
                    raise subprocess.TimeoutExpired(self.args, timeout)
                clock[0] += git_seconds
                self.returncode = 0
                return "a" * 40, ""

        monkeypatch.setattr(rt, "subprocess", _subprocess_with(SlowPopen))
        monkeypatch.setattr(rt.os, "killpg", lambda pid, sig: None)
        # Every extra source is a nonagenticai repo: each mints (and revokes) its own token.
        sources = [SOURCE, PLATFORM_SOURCE] + [
            f"{PLATFORM}/repo-jobs-{i}" for i in range(n_sources - 2)
        ]
        for extra in sources[2:]:
            world.repos[extra] = {"private": True, "owner": {"login": PLATFORM}}
        try:
            rt.create_rust_target(TARGET, sources, EXEC_TENANT)
            outcome = 201
        except rt.RustTargetError as e:
            outcome = e.status
        # Longest chain of revokes that did NOT overlap = how many ran back to back.
        waves, last_end = 0, float("-inf")
        for started, ended in sorted(revoke_spans, key=lambda span: span[1]):
            if started >= last_end:
                waves, last_end = waves + 1, ended
        # A wave is charged the JOIN bound (per-call timeout + join slack), not the
        # per-call timeout alone: that is how long the op actually waits.
        assert revoke_cap[0] <= rt._REVOKE_JOIN_SECONDS
        total = clock[0] + waves * rt._REVOKE_JOIN_SECONDS
        assert total <= 120.0, f"answered after {total:.1f}s (outcome {outcome})"
        assert outcome in (201, 504)
        assert "ghs_org" in world.revokes
        assert sorted(world.revokes) == sorted(world.mints.issued)

    def test_a_hung_git_step_is_cut_at_its_timeout(self, tmp_path):
        with pytest.raises(rt.RustTargetError) as e:
            rt._run(
                ["bash", "-c", "sleep 5"],
                cwd=str(tmp_path),
                env=rt._git_env(str(tmp_path)),
                what="hang",
                timeout=0.2,
            )
        assert e.value.status == 504 and e.value.code == "deadline_exceeded"

    def test_timeout_kills_the_whole_process_group_including_grandchildren(
        self, tmp_path, monkeypatch
    ):
        """``git push`` forks git-remote-https (token in its env). A timeout must take
        the grandchild down too, not orphan it."""
        started: list[subprocess.Popen] = []
        real_popen = subprocess.Popen

        def recording_popen(*a, **kw):
            proc = real_popen(*a, **kw)
            started.append(proc)
            return proc

        monkeypatch.setattr(rt, "subprocess", _subprocess_with(recording_popen))
        t0 = time.monotonic()
        with pytest.raises(rt.RustTargetError) as e:
            rt._run(
                ["bash", "-c", "sleep 7 & wait"],  # the sleep is a GRANDCHILD of _run
                cwd=str(tmp_path),
                env=rt._git_env(str(tmp_path)),
                what="hang",
                timeout=0.3,
            )
        assert e.value.code == "deadline_exceeded"
        # Killing only the child leaves the sleep holding the pipes: communicate()
        # then blocks for the full 7 s. The group kill returns at once.
        assert time.monotonic() - t0 < 3.0
        pgid = started[0].pid  # start_new_session: the child leads its own group
        assert started[0].returncode is not None  # the child was reaped
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)  # anyone (the orphaned sleep) left in the group?
            except ProcessLookupError:
                return
            time.sleep(0.05)
        os.killpg(pgid, signal.SIGKILL)  # do not leak it from the test either
        pytest.fail(f"process group {pgid} still has members after the timeout")


class TestSourceGrants:
    async def test_ungranted_source_is_403_with_no_network_call(
        self, async_client, world, grants, as_tenant
    ):
        grants(PATTERN, f"{SOURCE}:read")
        as_tenant(CONVEYOR_TENANT)
        resp = await async_client.post(
            URL, json={"target": TARGET, "sources": [SOURCE, f"{PLATFORM}/fr"]}
        )
        assert resp.status_code == 403 and "source_not_granted" in resp.text
        assert world.calls == [] and world.mints.org == [] and world.mints.clone == []

    async def test_existing_target_is_answered_before_any_source_read(
        self, async_client, world, as_tenant
    ):
        as_tenant(EXEC_TENANT)
        world.repos[TARGET] = {"private": True, "owner": {"login": SANDBOX}}
        resp = await async_client.post(
            URL, json={"target": TARGET, "sources": [SOURCE, PLATFORM_SOURCE]}
        )
        assert resp.status_code == 409
        gets = [p for m, p, _ in world.calls if m == "GET"]
        assert gets == [f"/repos/{TARGET}"]
        assert world.mints.clone == []


def _subprocess_with(popen: Any) -> SimpleNamespace:
    """``subprocess`` as rust_target sees it, with ``Popen`` swapped (and nothing global)."""
    return SimpleNamespace(
        Popen=popen,
        PIPE=subprocess.PIPE,
        TimeoutExpired=subprocess.TimeoutExpired,
        CompletedProcess=subprocess.CompletedProcess,
    )


class TestGitCredentialHandling:
    def test_token_travels_in_env_never_argv_and_is_redacted(self, tmp_path):
        env = rt._git_env(str(tmp_path), "ghs_secret")
        assert "ghs_secret" not in str(env)  # base64 inside the header only
        assert env["GIT_CONFIG_KEY_0"] == f"http.{rt.GIT_BASE_URL}/.extraheader"
        with pytest.raises(rt.RustTargetError) as e:
            rt._run(
                ["git", "ls-remote", "https://ghs_secret@127.0.0.1:1/x.git"],
                cwd=str(tmp_path),
                env=env,
                what="probe",
                timeout=10,
                secret="ghs_secret",
            )
        assert "ghs_secret" not in str(e.value)


# ---------------------------------------------------------------------------
# github_app_auth: the org installation + uncached org token (public App only)
# ---------------------------------------------------------------------------
class TestPublicOrgToken:
    @pytest.fixture
    def app(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        integration = MagicMock()
        auth = MagicMock()
        auth.AppAuth.return_value.create_jwt.return_value = "app-jwt"
        post = MagicMock()
        settings = SimpleNamespace(
            github_app_id="2333879",
            github_app_installation_id="99",
            github_app_private_key="INTERNAL-KEY",  # pragma: allowlist secret
            github_app_private_key_path="",
            public_app_id="777000",
            public_app_private_key="PUBLIC-KEY",  # pragma: allowlist secret
            public_app_private_key_path="",
        )
        ga.reset_cache()
        with ExitStack() as stack:
            for p in (
                patch.object(ga, "get_settings", return_value=settings),
                patch.object(ga, "Auth", auth),
                patch.object(ga, "GithubIntegration", MagicMock(return_value=integration)),
                patch.object(ga, "HAS_GITHUB", True),
                patch.object(ga.httpx, "post", post),
            ):
                stack.enter_context(p)
            yield SimpleNamespace(integration=integration, auth=auth, post=post)
        ga.reset_cache()

    def test_org_installation_reports_the_account(self, app):
        from types import SimpleNamespace

        app.integration.get_org_installation.return_value = SimpleNamespace(
            id=555, raw_data={"account": {"login": SANDBOX, "type": "Organization"}}
        )
        assert ga.resolve_public_org_installation(SANDBOX) == AppInstallation(
            555, SANDBOX, "Organization"
        )
        app.integration.get_org_installation.assert_called_once_with(SANDBOX)
        app_ids = {c.args[0] for c in app.auth.AppAuth.call_args_list if c.args}
        assert app_ids == {"777000"}  # the PUBLIC App, never the internal one

    def test_org_installation_404_vs_failure(self, app):
        from github import GithubException, UnknownObjectException

        app.integration.get_org_installation.side_effect = UnknownObjectException(404, {}, {})
        with pytest.raises(ga.GitHubAppNotInstalledError):
            ga.resolve_public_org_installation(SANDBOX)
        app.integration.get_org_installation.side_effect = GithubException(502, {}, {})
        with pytest.raises(GitHubAppMintError):
            ga.resolve_public_org_installation(SANDBOX)

    def test_org_token_is_installation_wide_exact_and_uncached(self, app):
        from unittest.mock import MagicMock

        ok = MagicMock(status_code=201)
        ok.json.return_value = {"token": "ghs_org"}
        app.post.return_value = ok
        assert ga.mint_public_org_token(555, {"administration": "write"}) == "ghs_org"
        assert ga.mint_public_org_token(555, {"administration": "write"}) == "ghs_org"
        assert app.post.call_count == 2  # never served from a cache
        call = app.post.call_args
        assert call.args[0].endswith("/app/installations/555/access_tokens")
        assert call.kwargs["json"] == {"permissions": {"administration": "write"}}
        assert call.kwargs["headers"]["Authorization"] == "Bearer app-jwt"
        assert ga._token_by_id == {}

    def test_org_token_refusals(self, app):
        from unittest.mock import MagicMock

        refused = MagicMock(status_code=422)
        app.post.return_value = refused
        with pytest.raises(GitHubAppMintError) as e:
            ga.mint_public_org_token(555, {"administration": "write"})
        assert e.value.status_code == 422
        empty = MagicMock(status_code=201)
        empty.json.return_value = {"token": ""}
        app.post.return_value = empty
        with pytest.raises(GitHubAppMintError, match="empty"):
            ga.mint_public_org_token(555, {"administration": "write"})
        app.post.side_effect = httpx.ConnectError("down")
        with pytest.raises(GitHubAppMintError, match="reach GitHub"):
            ga.mint_public_org_token(555, {"administration": "write"})
        for bad in ({}, {"administration": "admin"}, {"bad name": "write"}):
            with pytest.raises(ValueError):
                ga.mint_public_org_token(555, bad)


class TestFreshScopedTokens:
    """``fresh=True`` (the tokens create_rust_target_repo revokes) never touch the cache."""

    @pytest.fixture
    def post(self, monkeypatch):
        ok = MagicMock(status_code=201)
        ok.json.return_value = {
            "token": "ghs_scoped",
            "expires_at": "2999-01-01T00:00:00Z",
            "repositories": [{"name": "r", "owner": {"login": SANDBOX}}],
        }
        post = MagicMock(return_value=ok)
        auth = MagicMock()
        auth.create_jwt.return_value = "app-jwt"
        monkeypatch.setattr(ga.httpx, "post", post)
        monkeypatch.setattr(ga, "_get_app_auth", lambda app=ga.INTERNAL_APP: auth)
        ga.reset_cache()
        yield post
        ga.reset_cache()

    def test_fresh_public_token_is_neither_served_from_nor_written_to_the_cache(self, post):
        perms = {"contents": "write"}
        ga.mint_public_repo_token(555, SANDBOX, "r", perms, expected_owner=SANDBOX)
        ga.mint_public_repo_token(555, SANDBOX, "r", perms, expected_owner=SANDBOX)
        assert post.call_count == 1  # the default path caches
        cached = dict(ga._public_scoped_token_by_repo)
        ga.mint_public_repo_token(555, SANDBOX, "r", perms, expected_owner=SANDBOX, fresh=True)
        ga.mint_public_repo_token(555, SANDBOX, "r", perms, expected_owner=SANDBOX, fresh=True)
        assert post.call_count == 3  # fresh: minted every time
        assert ga._public_scoped_token_by_repo == cached  # and never written

    def test_fresh_clone_token_skips_the_cache(self, post, monkeypatch):
        monkeypatch.setattr(ga, "_resolve_installation_id", lambda o, r, app=ga.INTERNAL_APP: 9)
        ga.get_scoped_clone_token(PLATFORM, "r", fresh=True)
        ga.get_scoped_clone_token(PLATFORM, "r", fresh=True)
        assert post.call_count == 2
        assert ga._scoped_token_by_repo == {}
