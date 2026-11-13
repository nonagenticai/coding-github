"""Tests for ``POST /api/repositories/get-repository-file``.

Two seams, deliberately:

* the ``repo`` fixture swaps ``routers.github_repo_file_read._github`` for a service
  wired to a fake repository, so the router -> response-shape path runs over HTTP;
* ``TestTenancyAndCredential`` swaps only ``GitHubService`` inside
  ``routers.github_issue_operations``, so the REAL ``_github`` ->
  ``resolve_github_token`` -> ``check_platform_repo_tenant`` /
  ``resolve_customer_repo_token`` -> ``get_installation_token_for`` chain runs.

Properties pinned, each with the mutant that must turn it red:

- an absent path is ``200 {exists: false}``, never a 404 (mutant: map the contents
  404 onto ``repository_not_found``);
- a missing repo / ref is 404 with ``repository_not_found`` / ``ref_not_found``;
- directory / binary / >1 MiB / unsafe path are refused with their own ``code``;
- every error carries a TOP-LEVEL ``code`` (mutant: raise HTTPException -> the shared
  handler renders ``code: HTTP_<status>`` and the machine code is lost);
- a non-platform tenant reading a platform repo gets 403 ``platform_repo_forbidden``
  and GitHub is never opened (MUTATION CONTROL, measured: delete the
  ``check_platform_repo_tenant`` call in ``services/github_integration.py`` ->
  ``test_non_platform_tenant_cannot_read_a_platform_repo`` goes red with a 200);
- the credential is the App installation token, never the configured PAT, and an
  unusable App fails closed (there is no PAT setting to fall back to).
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from typing import Any

import pytest
from github import BadCredentialsException, GithubException, UnknownObjectException

URL = "/api/repositories/get-repository-file"
SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
BLOB = "fedcba9876543210fedcba9876543210fedcba98"  # pragma: allowlist secret
OTHER_TENANT = "11111111-1111-1111-1111-111111111111"


def _body(**over: Any) -> dict[str, Any]:
    return {"owner": "o", "repo": "r", "ref": "main", "path": "pyproject.toml", **over}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------
class _Content:
    def __init__(
        self,
        data: bytes,
        *,
        type_: str = "file",
        size: int | None = None,
        encoding: str = "base64",
    ) -> None:
        self.type = type_
        self.size = len(data) if size is None else size
        self.sha = BLOB
        self.encoding = encoding
        self.content = base64.b64encode(data).decode() if encoding == "base64" else ""


class _Commit:
    def __init__(self, sha: str) -> None:
        self.raw_data = {"sha": sha}


class _Repo:
    def __init__(self) -> None:
        self.contents: Any = _Content(b'[project]\nname = "x"\n')
        self.contents_raises: Exception | None = None
        self.commit_raises: Exception | None = None
        self.calls: list[tuple[str, ...]] = []

    def get_commit(self, ref: str) -> _Commit:
        self.calls.append(("get_commit", ref))
        if self.commit_raises is not None:
            raise self.commit_raises
        return _Commit(SHA)

    def get_contents(self, path: str, ref: str) -> Any:
        self.calls.append(("get_contents", path, ref))
        if self.contents_raises is not None:
            raise self.contents_raises
        return self.contents


class _Service:
    """Stands in for GitHubService: async context, get_repository, _execute_with_retry."""

    instances: list[_Service] = []

    def __init__(self, config: Any = None, *, repo: _Repo | None = None, boom=None) -> None:
        self.config = config
        self.repo = repo or _Repo()
        self.boom = boom
        _Service.instances.append(self)

    async def __aenter__(self) -> _Service:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_repository(self, _name: str) -> _Repo:
        if self.boom is not None:
            raise self.boom
        return self.repo

    async def _execute_with_retry(self, func, *a, **k):
        return func(*a, **k)


@pytest.fixture
def repo(monkeypatch):
    holder: dict[str, Any] = {"repo": _Repo(), "boom": None, "asked": []}

    @asynccontextmanager
    async def _fake_github(owner=None, repo=None, tenant_id=None, permissions=None):
        holder["asked"].append((owner, repo, tenant_id, permissions))
        yield _Service(repo=holder["repo"], boom=holder["boom"])

    monkeypatch.setattr("routers.github_repo_file_read._github", _fake_github)
    return holder


def _assert_coded(resp, status: int, code: str) -> None:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["code"] == code, body
    # The established envelope carries the same machine code.
    assert body["error"]["details"]["error_code"] == code, body


# ---------------------------------------------------------------------------
# Success / absent
# ---------------------------------------------------------------------------
class TestRead:
    async def test_existing_file_returns_text_blob_sha_and_resolved_ref(
        self, async_client, repo
    ) -> None:
        resp = await async_client.post(URL, json=_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "exists": True,
            "ref_sha": SHA,
            "path": "pyproject.toml",
            "content": '[project]\nname = "x"\n',
            "sha": BLOB,
            "size": 21,
        }

    async def test_content_is_read_at_the_resolved_sha_not_the_ref_name(
        self, async_client, repo
    ) -> None:
        """MUTANT: pass ``ref`` instead of ``ref_sha`` to get_contents -> a push between
        the two calls returns a file from a commit ``ref_sha`` does not name."""
        await async_client.post(URL, json=_body(ref="feat/x"))
        assert repo["repo"].calls == [
            ("get_commit", "feat/x"),
            ("get_contents", "pyproject.toml", SHA),
        ]

    async def test_absent_file_is_200_data_not_an_error(self, async_client, repo) -> None:
        """MUTANT: let the contents 404 fall through to repository_not_found."""
        repo["repo"].contents_raises = UnknownObjectException(404, {"message": "Not Found"}, None)
        resp = await async_client.post(URL, json=_body(path="package.json"))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "exists": False,
            "ref_sha": SHA,
            "path": "package.json",
            "content": None,
            "sha": None,
            "size": None,
        }

    async def test_asks_for_contents_read_only(self, async_client, repo) -> None:
        from services.tenant_binding import PERMS_REPO_READ

        await async_client.post(URL, json=_body())
        assert repo["asked"][0][3] == PERMS_REPO_READ == {"contents": "read"}


# ---------------------------------------------------------------------------
# Refusals with their own code
# ---------------------------------------------------------------------------
class TestRefusals:
    async def test_directory_is_not_a_file(self, async_client, repo) -> None:
        repo["repo"].contents = [_Content(b"a"), _Content(b"b")]
        _assert_coded(await async_client.post(URL, json=_body(path="src")), 422, "not_a_file")

    async def test_submodule_is_not_a_file(self, async_client, repo) -> None:
        repo["repo"].contents = _Content(b"", type_="submodule")
        _assert_coded(await async_client.post(URL, json=_body(path="vendor")), 422, "not_a_file")

    @pytest.mark.parametrize("data", [b"\x89PNG\r\n\x1a\n\x00\x00", b"\xff\xfe\xfa bad utf8"])
    async def test_binary_file_is_refused(self, async_client, repo, data) -> None:
        repo["repo"].contents = _Content(data)
        _assert_coded(await async_client.post(URL, json=_body(path="logo.png")), 422, "binary_file")

    async def test_over_one_mib_is_file_too_large(self, async_client, repo) -> None:
        repo["repo"].contents = _Content(b"x", size=1024 * 1024 + 1)
        resp = await async_client.post(URL, json=_body(path="big.lock"))
        _assert_coded(resp, 413, "file_too_large")
        assert resp.json()["error"]["details"]["limit"] == 1024 * 1024

    async def test_exactly_one_mib_is_allowed(self, async_client, repo) -> None:
        """Boundary control for the test above: the limit is ``>``, not ``>=``."""
        repo["repo"].contents = _Content(b"a" * (1024 * 1024))
        resp = await async_client.post(URL, json=_body(path="big.txt"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["size"] == 1024 * 1024

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",
            "../x",
            "a/../../b",
            "a/..",
            "\\abs",
            "a\\..\\b",
            "a\x00b",
            "",
            "./x",
            "a//b",
        ],
    )
    async def test_unsafe_path_is_invalid_path_and_github_is_not_called(
        self, async_client, repo, path
    ) -> None:
        resp = await async_client.post(URL, json=_body(path=path))
        _assert_coded(resp, 422, "invalid_path")
        assert repo["asked"] == []

    async def test_safe_nested_path_is_accepted(self, async_client, repo) -> None:
        """Positive control for the path guard: it must not refuse normal paths."""
        resp = await async_client.post(URL, json=_body(path="services/api/package.json"))
        assert resp.status_code == 200, resp.text

    @pytest.mark.parametrize(
        "exc",
        [
            UnknownObjectException(404, {"message": "Not Found"}, None),
            GithubException(422, {"message": "No commit found for SHA: deadbeef"}, None),
        ],
    )
    async def test_unknown_ref_is_404_ref_not_found(self, async_client, repo, exc) -> None:
        repo["repo"].commit_raises = exc
        resp = await async_client.post(URL, json=_body(ref="nope"))
        _assert_coded(resp, 404, "ref_not_found")

    async def test_missing_repository_is_404_repository_not_found(self, async_client, repo) -> None:
        repo["boom"] = UnknownObjectException(404, {"message": "Not Found"}, None)
        _assert_coded(await async_client.post(URL, json=_body()), 404, "repository_not_found")

    async def test_ref_404_and_repo_404_do_not_share_a_code(self, async_client, repo) -> None:
        repo["repo"].commit_raises = UnknownObjectException(404, {}, None)
        a = (await async_client.post(URL, json=_body())).json()["code"]
        repo["boom"] = UnknownObjectException(404, {}, None)
        b = (await async_client.post(URL, json=_body())).json()["code"]
        assert (a, b) == ("ref_not_found", "repository_not_found")

    async def test_github_403_is_github_forbidden(self, async_client, repo) -> None:
        repo["repo"].contents_raises = GithubException(403, {"message": "Forbidden"}, None)
        _assert_coded(await async_client.post(URL, json=_body()), 403, "github_forbidden")

    async def test_github_401_is_github_unauthorized(self, async_client, repo) -> None:
        repo["boom"] = BadCredentialsException(401, {"message": "Bad"}, None)
        _assert_coded(await async_client.post(URL, json=_body()), 401, "github_unauthorized")

    async def test_unexpected_error_is_coded_500(self, async_client, repo) -> None:
        repo["repo"].contents_raises = RuntimeError("boom")
        _assert_coded(await async_client.post(URL, json=_body()), 500, "internal_error")

    async def test_not_gated_by_the_andon_brake(self, async_client, repo) -> None:
        """A read must not be coupled to fr's health (see test_andon_chokepoint's
        read-side census, which this route is not yet listed in)."""
        from fastapi.routing import APIRoute, iter_route_contexts

        import services.andon as andon
        from main import app

        # fastapi>=0.137 wraps included routers; iter_route_contexts yields the
        # effective (prefixed) route, and forwards attribute reads to it.
        route = next(
            r
            for r in iter_route_contexts(app.routes)
            if isinstance(r.original_route, APIRoute) and r.path == URL
        )
        deps = [d.call for d in route.dependant.dependencies]
        assert andon.require_andon_released not in deps
        assert route.operation_id == "get_repository_file"
        assert route.methods == {"POST"}


# ---------------------------------------------------------------------------
# Against REAL PyGithub (no network)
# ---------------------------------------------------------------------------
class TestAgainstRealPyGithub:
    """MUTANT: read ``repo.get_commit(ref).sha`` instead of ``raw_data["sha"]`` — a
    lazy Commit can answer without the GET, and ``ref_sha`` would not be a SHA. This
    drives real PyGithub objects over a stubbed requester and asserts the requests."""

    @staticmethod
    def _real_repo(answers: dict[str, Any]):
        from github import Auth, Github

        repo = Github(auth=Auth.Token("t"), lazy=True).get_repo("o/r")
        seen: list[tuple[str, dict]] = []

        def _rjac(verb, url, parameters=None, headers=None, input=None, **kw):  # noqa: A002
            seen.append((url, dict(parameters or {})))
            for suffix, answer in answers.items():
                if url.endswith(suffix):
                    if isinstance(answer, Exception):
                        raise answer
                    return {}, answer
            raise AssertionError(f"unexpected request {url}")

        repo._requester.requestJsonAndCheck = _rjac  # type: ignore[method-assign]
        return repo, seen

    async def test_ref_resolves_through_a_real_get_and_contents_pin_to_it(self) -> None:
        from routers.github_repo_file_read import _read_blob, _resolve_ref_sha

        text = b'{"scripts": {"test": "vitest"}}'
        repo, seen = self._real_repo(
            {
                "/commits/main": {"sha": SHA, "url": "https://api.github.com/repos/o/r/commits/x"},
                "/contents/package.json": {
                    "type": "file",
                    "encoding": "base64",
                    "content": base64.b64encode(text).decode(),
                    "size": len(text),
                    "sha": BLOB,
                    "name": "package.json",
                    "path": "package.json",
                },
            }
        )
        ref_sha = _resolve_ref_sha(repo, "main")
        blob = _read_blob(repo, "package.json", ref_sha)
        assert ref_sha == SHA
        assert blob == {"content": text.decode(), "sha": BLOB, "size": len(text)}
        assert seen[-1][1] == {"ref": SHA}

    async def test_real_404_on_contents_is_absent(self) -> None:
        from routers.github_repo_file_read import _read_blob

        repo, _ = self._real_repo(
            {"/contents/nope.toml": UnknownObjectException(404, {"message": "Not Found"}, None)}
        )
        assert _read_blob(repo, "nope.toml", SHA) is None


# ---------------------------------------------------------------------------
# Tenancy + credential, through the REAL _github seam
# ---------------------------------------------------------------------------
@pytest.fixture
def real_seam(monkeypatch):
    """Real _github/resolve_github_token; GitHubService replaced by a recorder; the
    App is configured and mints a recognisable token."""
    import services.github_app_auth as app_auth

    _Service.instances = []
    minted: list[tuple[str, str]] = []

    def _mint(owner: str, repo: str) -> str:
        minted.append((owner, repo))
        return "ghs_app_installation_token"

    monkeypatch.setattr("routers.github_issue_operations.GitHubService", _Service)
    monkeypatch.setattr(app_auth, "app_auth_configured", lambda: True)
    monkeypatch.setattr(app_auth, "get_installation_token_for", _mint)
    return {"minted": minted}


class TestTenancyAndCredential:
    async def test_non_platform_tenant_cannot_read_a_platform_repo(
        self, async_client, real_seam, enforce_platform_tenants
    ) -> None:
        """⛔ THE forbidden test. MUTATION CONTROL: delete the
        ``check_platform_repo_tenant(...)`` call in ``resolve_github_token`` -> the App
        mints for this tenant and the file is served (200): red."""
        from coding_client.testing import override_tenant_auth

        from main import app

        undo = override_tenant_auth(app, tenant_id=OTHER_TENANT)
        try:
            resp = await async_client.post(
                URL, json=_body(owner="test-owner", repo="coding-github")
            )
        finally:
            undo()
        _assert_coded(resp, 403, "platform_repo_forbidden")
        assert resp.json()["error"]["details"]["owner"] == "test-owner"
        assert real_seam["minted"] == []
        assert _Service.instances == []

    async def test_platform_tenant_reads_with_the_app_token_never_the_pat(
        self, async_client, real_seam, platform_tenant, monkeypatch
    ) -> None:
        """Positive control for the forbidden test AND the App-only property: the
        same request from a platform tenant succeeds, on the App's token — with
        every retired PAT variable armed, so 'not used' means something."""
        legacy = "legacy-pat-sentinel-not-a-credential"  # pragma: allowlist secret
        for var in (
            "GITHUB_ACCESS_TOKEN",
            "GITHUB_TOKEN",
            "CODING_GITHUB_GITHUB_TOKEN",
            "CODING_GITHUB_SYSTEM_GITHUB_TOKEN",
        ):
            monkeypatch.setenv(var, legacy)
        pats = {legacy}

        resp = await async_client.post(URL, json=_body(owner="test-owner", repo="coding-github"))
        assert resp.status_code == 200, resp.text
        assert real_seam["minted"] == [("test-owner", "coding-github")]
        assert [svc.config.token for svc in _Service.instances] == ["ghs_app_installation_token"]
        assert not pats & {svc.config.token for svc in _Service.instances}

    async def test_unusable_app_fails_closed_without_a_pat_fallback(
        self, async_client, real_seam, platform_tenant, monkeypatch
    ) -> None:
        import services.github_app_auth as app_auth

        monkeypatch.setattr(app_auth, "app_auth_configured", lambda: False)
        resp = await async_client.post(URL, json=_body(owner="test-owner", repo="coding-github"))
        _assert_coded(resp, 412, "github_auth_failed")
        assert _Service.instances == []

    async def test_tenant_cannot_read_a_customer_repo_bound_to_another_tenant(
        self, async_client, real_seam, monkeypatch
    ) -> None:
        """Customer owner ``acme`` is bound to OTHER_TENANT only; the default test
        tenant asks for it and is refused before any credential is minted."""
        import services.tenant_binding as tb

        asked: list[tuple[str, str]] = []

        def _lookup(tenant_id: str, owner: str) -> int | None:
            asked.append((tenant_id, owner))
            return 4242 if tenant_id == OTHER_TENANT else None

        monkeypatch.setattr(tb, "lookup_binding", _lookup)
        resp = await async_client.post(URL, json=_body(owner="acme", repo="secret"))
        _assert_coded(resp, 409, "github_app_not_installed")
        assert asked and asked[0][0] != OTHER_TENANT
        assert _Service.instances == []
        assert real_seam["minted"] == []
