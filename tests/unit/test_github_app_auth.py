"""Unit tests for the GitHub App installation-token auth module.

Covers: configured/unconfigured detection, file-vs-inline PEM loading,
installation-token minting (mocked HTTP via PyGithub's Auth), and cache-once.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services import github_app_auth
from services.github_app_auth import GitHubAppAuthError, GitHubAppNotInstalledError


def _github_error(status: int) -> Exception:
    """A PyGithub-shaped HTTP error (it carries ``.status``)."""
    from github import GithubException

    return GithubException(status, {"message": "boom"}, {})


def _github_404() -> Exception:
    """What PyGithub actually raises when the App is not installed on a repo."""
    from github import UnknownObjectException

    return UnknownObjectException(404, {"message": "Not Found"}, {})


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "github_app_id": "123456",
        "github_app_installation_id": "987654",
        "github_app_private_key": "",
        "github_app_private_key_path": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_cache():
    github_app_auth.reset_cache()
    yield
    github_app_auth.reset_cache()


# ---------------------------------------------------------------------------
# app_auth_configured
# ---------------------------------------------------------------------------
class TestAppAuthConfigured:
    def test_unconfigured_missing_ids(self):
        s = _settings(github_app_id="", github_app_private_key="x")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth.app_auth_configured() is False

    def test_unconfigured_no_key(self):
        s = _settings(github_app_private_key="", github_app_private_key_path="")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth.app_auth_configured() is False

    def test_configured_with_inline_key(self):
        s = _settings(github_app_private_key="-----BEGIN-----\nk\n-----END-----")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth.app_auth_configured() is True

    def test_configured_with_key_file(self, tmp_path):
        pem = tmp_path / "key.pem"
        pem.write_text("-----BEGIN-----\nk\n-----END-----")
        s = _settings(github_app_private_key_path=str(pem))
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth.app_auth_configured() is True

    def test_unconfigured_when_pygithub_absent(self):
        s = _settings(github_app_private_key="k")
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "HAS_GITHUB", False),
        ):
            assert github_app_auth.app_auth_configured() is False


# ---------------------------------------------------------------------------
# _load_private_key — file preferred over inline
# ---------------------------------------------------------------------------
class TestLoadPrivateKey:
    def test_prefers_file_over_inline(self, tmp_path):
        pem = tmp_path / "key.pem"
        pem.write_text("FILE-KEY")
        s = _settings(github_app_private_key_path=str(pem), github_app_private_key="INLINE-KEY")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth._load_private_key() == "FILE-KEY"

    def test_falls_back_to_inline_when_no_path(self):
        s = _settings(github_app_private_key="INLINE-KEY")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth._load_private_key() == "INLINE-KEY"

    def test_falls_back_to_inline_when_file_empty(self, tmp_path):
        pem = tmp_path / "key.pem"
        pem.write_text("   \n")
        s = _settings(github_app_private_key_path=str(pem), github_app_private_key="INLINE-KEY")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            assert github_app_auth._load_private_key() == "INLINE-KEY"

    def test_raises_when_file_unreadable(self, tmp_path):
        s = _settings(github_app_private_key_path=str(tmp_path / "missing.pem"))
        with patch.object(github_app_auth, "get_settings", return_value=s):
            with pytest.raises(GitHubAppAuthError, match="Cannot read"):
                github_app_auth._load_private_key()

    def test_raises_when_nothing_configured(self):
        s = _settings(github_app_private_key="", github_app_private_key_path="")
        with patch.object(github_app_auth, "get_settings", return_value=s):
            with pytest.raises(GitHubAppAuthError, match="No GitHub App private key"):
                github_app_auth._load_private_key()


# ---------------------------------------------------------------------------
# get_installation_token + cache-once
# ---------------------------------------------------------------------------
class TestInstallationToken:
    def _mock_app(self, token="ghs_installtoken", install_id=987654):
        authz = MagicMock()
        authz.token = token
        integration = MagicMock()
        integration.get_access_token.return_value = authz
        integration.get_repo_installation.return_value = SimpleNamespace(id=install_id)
        mock_gi = MagicMock(return_value=integration)
        mock_auth = MagicMock()  # Auth.AppAuth(...) — fed to GithubIntegration(auth=...)
        return mock_auth, mock_gi, integration

    def _ctx(self, s, mock_auth, mock_gi):
        import contextlib

        stack = contextlib.ExitStack()
        for p in (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", mock_auth),
            patch.object(github_app_auth, "GithubIntegration", mock_gi),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            stack.enter_context(p)
        return stack

    def test_mints_token(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mock_app()
        with self._ctx(s, mock_auth, mock_gi):
            token = github_app_auth.get_installation_token()
        assert token == "ghs_installtoken"
        mock_auth.AppAuth.assert_called_once_with("123456", "KEY")
        integration.get_access_token.assert_called_once_with(987654)

    def test_cache_once(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mock_app()
        with self._ctx(s, mock_auth, mock_gi):
            github_app_auth.get_installation_token()
            github_app_auth.get_installation_token()
        # TTL cache: token minted once, app integration (JWT signer) built once.
        integration.get_access_token.assert_called_once()
        mock_gi.assert_called_once()

    def test_empty_token_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._mock_app(token="")
        with self._ctx(s, mock_auth, mock_gi):
            with pytest.raises(GitHubAppAuthError, match="empty"):
                github_app_auth.get_installation_token()

    def test_unconfigured_ids_raise(self):
        s = _settings(github_app_id="", github_app_private_key="KEY")
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            with pytest.raises(GitHubAppAuthError, match="not configured"):
                github_app_auth.get_installation_token()


# ---------------------------------------------------------------------------
# get_installation_token_for — per-owner multi-installation resolution (D3)
# ---------------------------------------------------------------------------
class TestPerOwnerInstallation:
    def _ctx(self, s, mock_auth, mock_gi):
        import contextlib

        stack = contextlib.ExitStack()
        for p in (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", mock_auth),
            patch.object(github_app_auth, "GithubIntegration", mock_gi),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            stack.enter_context(p)
        return stack

    def _mocks(self, install_id=131797507, token="ghs_perowner"):
        authz = MagicMock()
        authz.token = token
        integration = MagicMock()
        integration.get_access_token.return_value = authz
        integration.get_repo_installation.return_value = SimpleNamespace(id=install_id)
        mock_gi = MagicMock(return_value=integration)
        mock_auth = MagicMock()
        return mock_auth, mock_gi, integration

    def test_resolves_per_owner_token(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mocks()
        with self._ctx(s, mock_auth, mock_gi):
            token = github_app_auth.get_installation_token_for("acme", "widgets")
        assert token == "ghs_perowner"
        integration.get_repo_installation.assert_called_once_with("acme", "widgets")
        integration.get_access_token.assert_called_once_with(131797507)

    def test_not_installed_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mocks()
        integration.get_repo_installation.side_effect = _github_404()
        with self._ctx(s, mock_auth, mock_gi):
            with pytest.raises(GitHubAppNotInstalledError, match="not installed"):
                github_app_auth.get_installation_token_for("ext-org", "repo")

    def test_non_404_installation_lookup_failure_is_not_reported_as_not_installed(self):
        """A 500/timeout means the question was never answered, not "no install".

        The old code raised the identical error for both, so an api.github.com
        outage read downstream as "the App is not installed on that owner".
        """
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mocks()
        integration.get_repo_installation.side_effect = _github_error(500)
        with self._ctx(s, mock_auth, mock_gi):
            with pytest.raises(github_app_auth.GitHubAppMintError) as exc:
                github_app_auth.get_installation_token_for("acme", "widgets")
        assert not isinstance(exc.value, GitHubAppNotInstalledError)
        assert "lookup itself failed" in str(exc.value)

    def test_installation_lookup_transport_error_is_not_not_installed(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._mocks()
        integration.get_repo_installation.side_effect = OSError("connection reset")
        with self._ctx(s, mock_auth, mock_gi):
            with pytest.raises(github_app_auth.GitHubAppMintError):
                github_app_auth.get_installation_token_for("acme", "widgets")

    def test_per_installation_cache_and_single_integration(self):
        s = _settings(github_app_private_key="KEY")
        integration = MagicMock()
        integration.get_access_token.side_effect = lambda i: MagicMock(
            token="ghs_a" if i == 111 else "ghs_b"
        )
        integration.get_repo_installation.side_effect = lambda o, r: SimpleNamespace(
            id=111 if o == "orgA" else 222
        )
        mock_gi = MagicMock(return_value=integration)
        mock_auth = MagicMock()
        with self._ctx(s, mock_auth, mock_gi):
            ta = github_app_auth.get_installation_token_for("orgA", "x")
            tb = github_app_auth.get_installation_token_for("orgB", "y")
            ta2 = github_app_auth.get_installation_token_for("orgA", "z")  # cache hit on id 111
        assert (ta, tb, ta2) == ("ghs_a", "ghs_b", "ghs_a")
        mock_gi.assert_called_once()  # app integration built once (JWT signer cached)
        assert integration.get_access_token.call_count == 2  # one mint per distinct id


# ---------------------------------------------------------------------------
# get_scoped_clone_token — repository-scoped, read-only mint (raw HTTP)
# ---------------------------------------------------------------------------
def _mock_response(status_code=201, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


class _ScopedFixtures:
    """Shared wiring: a configured App whose PyGithub layer is fully mocked."""

    def _app(self, install_id=131797507, pr_token="ghs_full_pr_token"):
        authz = MagicMock()
        authz.token = pr_token
        integration = MagicMock()
        integration.get_access_token.return_value = authz
        integration.get_repo_installation.return_value = SimpleNamespace(id=install_id)
        mock_gi = MagicMock(return_value=integration)
        mock_auth = MagicMock()
        mock_auth.AppAuth.return_value.create_jwt.return_value = "jwt-abc"
        return mock_auth, mock_gi, integration

    def _ctx(self, s, mock_auth, mock_gi):
        import contextlib

        stack = contextlib.ExitStack()
        for p in (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", mock_auth),
            patch.object(github_app_auth, "GithubIntegration", mock_gi),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            stack.enter_context(p)
        return stack


class TestScopedCloneToken(_ScopedFixtures):
    def test_mint_is_scoped_to_one_repo_read_only(self):
        """The raw POST must carry contents:read AND repositories=[repo].

        PyGithub cannot express repository-scoping (get_access_token hardcodes
        body={"permissions": ...}), so a PyGithub mint would grant contents:read
        across EVERY repo in the installation. This pins the raw body.
        """
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_scoped", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            scoped = github_app_auth.get_scoped_clone_token("acme", "widgets")

        assert scoped.token == "ghs_scoped"
        assert scoped.expires_at == "2999-01-01T00:00:00Z"
        post.assert_called_once()
        args, kwargs = post.call_args
        assert args[0] == ("https://api.github.com/app/installations/131797507/access_tokens")
        assert kwargs["json"] == {
            "permissions": {"contents": "read"},
            "repositories": ["widgets"],
        }
        # Authenticated with the App JWT that PyGithub's AppAuth signs.
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-abc"

    def test_scoped_token_is_cached_per_repo(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_scoped", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            first = github_app_auth.get_scoped_clone_token("acme", "widgets")
            second = github_app_auth.get_scoped_clone_token("acme", "widgets")
            other = github_app_auth.get_scoped_clone_token("acme", "gadgets")

        assert first == second
        assert other.token == "ghs_scoped"
        # Same repo ⇒ 1 mint (cache hit). Different repo ⇒ its own mint: a token
        # scoped to `widgets` cannot clone `gadgets`.
        assert post.call_count == 2

    def test_pr_read_scope_mints_contents_and_pull_requests_read(self):
        """scope="pr-read" adds pull_requests:read — GET /pulls/{n}/files 403s without it."""
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_pr_read", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            scoped = github_app_auth.get_scoped_clone_token("acme", "widgets", scope="pr-read")

        assert scoped.token == "ghs_pr_read"
        post.assert_called_once()
        assert post.call_args.kwargs["json"] == {
            "permissions": {"contents": "read", "pull_requests": "read"},
            "repositories": ["widgets"],
        }

    def test_unknown_scope_raises_before_any_mint(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock()
        with (
            self._ctx(s, mock_auth, mock_gi),
            patch.object(github_app_auth.httpx, "post", post),
            pytest.raises(ValueError, match="Unknown scoped-token scope"),
        ):
            github_app_auth.get_scoped_clone_token("acme", "widgets", scope="admin")  # type: ignore[arg-type]
        post.assert_not_called()

    def test_cache_is_keyed_by_scope(self):
        """A cached clone token must never answer a pr-read request (and vice versa).

        The clone token lacks pull_requests:read, so serving it for pr-read would
        re-create the exact 403 pr-read exists to fix.
        """
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(
            side_effect=[
                _mock_response(201, {"token": "ghs_clone", "expires_at": "2999-01-01T00:00:00Z"}),
                _mock_response(201, {"token": "ghs_pr_read", "expires_at": "2999-01-01T00:00:00Z"}),
            ]
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            clone = github_app_auth.get_scoped_clone_token("acme", "widgets")
            pr_read = github_app_auth.get_scoped_clone_token("acme", "widgets", scope="pr-read")
            clone_again = github_app_auth.get_scoped_clone_token("acme", "widgets", scope="clone")
            pr_read_again = github_app_auth.get_scoped_clone_token(
                "acme", "widgets", scope="pr-read"
            )

        assert clone.token == "ghs_clone"
        assert pr_read.token == "ghs_pr_read"  # minted anew, NOT the cached clone token
        assert clone_again.token == "ghs_clone"
        assert pr_read_again.token == "ghs_pr_read"
        assert post.call_count == 2  # one mint per scope, then cache hits
        assert [c.kwargs["json"]["permissions"] for c in post.call_args_list] == [
            {"contents": "read"},
            {"contents": "read", "pull_requests": "read"},
        ]

    def test_expired_token_is_not_served_from_cache(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_scoped", "expires_at": "2000-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            github_app_auth.get_scoped_clone_token("acme", "widgets")
            github_app_auth.get_scoped_clone_token("acme", "widgets")
        assert post.call_count == 2  # already-expired ⇒ never cached

    def test_github_refusal_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(return_value=_mock_response(422, {"message": "nope"}))
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(GitHubAppAuthError, match="refused"):
                github_app_auth.get_scoped_clone_token("acme", "widgets")

    def test_github_refusal_carries_the_status_code(self):
        """A refused mint exposes GitHub's status: 422 is how an installation that was
        never granted a requested permission answers, and /close-issue keys its one
        narrow fallback on exactly that value."""
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(return_value=_mock_response(422, {"message": "nope"}))
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(github_app_auth.GitHubAppMintError) as ei:
                github_app_auth.get_scoped_clone_token("acme", "widgets")
        assert ei.value.status_code == 422

    def test_transport_error_carries_no_status_code(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(side_effect=github_app_auth.httpx.ConnectError("down"))
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(github_app_auth.GitHubAppMintError) as ei:
                github_app_auth.get_scoped_clone_token("acme", "widgets")
        assert ei.value.status_code is None

    def test_transport_error_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(side_effect=github_app_auth.httpx.ConnectError("down"))
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(GitHubAppAuthError, match="Failed to reach GitHub"):
                github_app_auth.get_scoped_clone_token("acme", "widgets")

    def test_empty_token_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, _ = self._app()
        post = MagicMock(return_value=_mock_response(201, {"token": ""}))
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(GitHubAppAuthError, match="empty"):
                github_app_auth.get_scoped_clone_token("acme", "widgets")

    def test_app_not_installed_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._app()
        integration.get_repo_installation.side_effect = _github_404()
        post = MagicMock()
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            with pytest.raises(GitHubAppNotInstalledError, match="not installed"):
                github_app_auth.get_scoped_clone_token("ext-org", "repo")
        post.assert_not_called()  # no mint attempted without an installation


# ---------------------------------------------------------------------------
# ⚠️ THE INVARIANT: scoped tokens must never reach the PR-opening token cache
# ---------------------------------------------------------------------------
class TestScopedCacheIsolation(_ScopedFixtures):
    def test_scoped_mint_does_not_poison_pr_token(self):
        """A scoped clone mint must not leak into ``_token_by_id``.

        ``_token_by_id`` is keyed on installation_id ALONE and backs the tokens
        that OPEN PULL REQUESTS. If the scoped (contents:read, single-repo) mint
        wrote there, the very next ``get_installation_token_for()`` call in the
        PR-opening flow would be handed a crippled token from cache and every AA
        pull request would silently fail until the TTL expired.
        """
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._app(pr_token="ghs_full_pr_token")
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_scoped_clone", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            scoped = github_app_auth.get_scoped_clone_token("acme", "widgets")
            pr_token = github_app_auth.get_installation_token_for("acme", "widgets")

        assert scoped.token == "ghs_scoped_clone"
        assert pr_token == "ghs_full_pr_token"  # NOT the scoped token
        assert scoped.token != pr_token

        # Two DISTINCT mints via two distinct paths — neither served the other.
        post.assert_called_once()  # scoped: raw HTTP
        integration.get_access_token.assert_called_once_with(131797507)  # PR: PyGithub

        # The caches share nothing.
        assert github_app_auth._token_by_id[131797507][0] == "ghs_full_pr_token"
        assert "ghs_scoped_clone" not in [t for t, _ in github_app_auth._token_by_id.values()]
        assert (
            github_app_auth._scoped_token_by_repo[(131797507, "acme", "widgets", "contents:read")][
                0
            ]
            == "ghs_scoped_clone"
        )

    def test_pr_token_does_not_satisfy_a_scoped_request(self):
        """The reverse direction: a cached PR token must not be served as a clone token."""
        s = _settings(github_app_private_key="KEY")
        mock_auth, mock_gi, integration = self._app(pr_token="ghs_full_pr_token")
        post = MagicMock(
            return_value=_mock_response(
                201, {"token": "ghs_scoped_clone", "expires_at": "2999-01-01T00:00:00Z"}
            )
        )
        with self._ctx(s, mock_auth, mock_gi), patch.object(github_app_auth.httpx, "post", post):
            pr_token = github_app_auth.get_installation_token_for("acme", "widgets")
            scoped = github_app_auth.get_scoped_clone_token("acme", "widgets")

        assert pr_token == "ghs_full_pr_token"
        assert scoped.token == "ghs_scoped_clone"  # freshly minted, not the cached PR token
        post.assert_called_once()


# ---------------------------------------------------------------------------
# get_repo_visibility
# ---------------------------------------------------------------------------
class TestRepoVisibility:
    def test_public_repo(self):
        get = MagicMock(return_value=_mock_response(200, {"private": False}))
        with patch.object(github_app_auth.httpx, "get", get):
            assert github_app_auth.get_repo_visibility("acme", "widgets") == "public"
        args, kwargs = get.call_args
        assert args[0] == "https://api.github.com/repos/acme/widgets"
        assert "Authorization" not in kwargs["headers"]  # unauthenticated probe

    def test_private_repo(self):
        get = MagicMock(return_value=_mock_response(200, {"private": True}))
        with patch.object(github_app_auth.httpx, "get", get):
            visibility = github_app_auth.get_repo_visibility("acme", "widgets", "ghs_scoped")
        assert visibility == "private"
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer ghs_scoped"

    def test_404_is_reported_private(self):
        """Invisible to this credential ⇒ 'private' (fail safe: creds required)."""
        get = MagicMock(return_value=_mock_response(404, {}))
        with patch.object(github_app_auth.httpx, "get", get):
            assert github_app_auth.get_repo_visibility("acme", "secret") == "private"

    def test_missing_private_field_is_reported_private(self):
        get = MagicMock(return_value=_mock_response(200, {}))
        with patch.object(github_app_auth.httpx, "get", get):
            assert github_app_auth.get_repo_visibility("acme", "widgets") == "private"

    def test_unexpected_status_raises(self):
        get = MagicMock(return_value=_mock_response(500, {}))
        with patch.object(github_app_auth.httpx, "get", get):
            with pytest.raises(GitHubAppAuthError, match="Failed to resolve visibility"):
                github_app_auth.get_repo_visibility("acme", "widgets")

    def test_transport_error_raises(self):
        get = MagicMock(side_effect=github_app_auth.httpx.ConnectError("down"))
        with patch.object(github_app_auth.httpx, "get", get):
            with pytest.raises(GitHubAppAuthError, match="Failed to reach GitHub"):
                github_app_auth.get_repo_visibility("acme", "widgets")


# ---------------------------------------------------------------------------
# _cache_ttl_for
# ---------------------------------------------------------------------------
class TestCacheTTL:
    def test_far_future_expiry_is_capped_at_standard_ttl(self):
        assert github_app_auth._cache_ttl_for("2999-01-01T00:00:00Z") == (
            github_app_auth._TOKEN_TTL_SECONDS
        )

    def test_past_expiry_is_zero(self):
        assert github_app_auth._cache_ttl_for("2000-01-01T00:00:00Z") == 0.0

    def test_unparseable_expiry_falls_back_to_default(self):
        assert github_app_auth._cache_ttl_for("not-a-date") == github_app_auth._TOKEN_TTL_SECONDS

    def test_missing_expiry_falls_back_to_default(self):
        assert github_app_auth._cache_ttl_for("") == github_app_auth._TOKEN_TTL_SECONDS

    def test_naive_expiry_is_treated_as_utc(self):
        """GitHub sends a 'Z' suffix, but a tz-naive stamp must not crash the TTL math."""
        assert github_app_auth._cache_ttl_for("2999-01-01T00:00:00") == (
            github_app_auth._TOKEN_TTL_SECONDS
        )
        assert github_app_auth._cache_ttl_for("2000-01-01T00:00:00") == 0.0


# ---------------------------------------------------------------------------
# _get_app_auth — the shared JWT signer (error rungs fall through, never crash)
# ---------------------------------------------------------------------------
class TestGetAppAuth:
    def test_raises_when_pygithub_absent(self):
        with patch.object(github_app_auth, "HAS_GITHUB", False):
            with pytest.raises(GitHubAppAuthError, match="PyGithub is not available"):
                github_app_auth._get_app_auth()

    def test_raises_when_app_id_missing(self):
        s = _settings(github_app_id="", github_app_private_key="KEY")
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            with pytest.raises(GitHubAppAuthError, match="App ID not configured"):
                github_app_auth._get_app_auth()

    def test_raises_when_key_is_unusable(self):
        s = _settings(github_app_private_key="NOT-A-PEM")
        mock_auth = MagicMock()
        mock_auth.AppAuth.side_effect = ValueError("bad key")
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", mock_auth),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            with pytest.raises(GitHubAppAuthError, match="Failed to build GitHub App auth"):
                github_app_auth._get_app_auth()

    def test_integration_build_failure_raises(self):
        s = _settings(github_app_private_key="KEY")
        mock_auth = MagicMock()
        mock_gi = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", mock_auth),
            patch.object(github_app_auth, "GithubIntegration", mock_gi),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            with pytest.raises(GitHubAppAuthError, match="Failed to build GitHub App integration"):
                github_app_auth._get_app_integration()

    def test_installation_mint_failure_raises(self):
        s = _settings(github_app_private_key="KEY")
        integration = MagicMock()
        integration.get_access_token.side_effect = RuntimeError("500")
        mock_gi = MagicMock(return_value=integration)
        with (
            patch.object(github_app_auth, "get_settings", return_value=s),
            patch.object(github_app_auth, "Auth", MagicMock()),
            patch.object(github_app_auth, "GithubIntegration", mock_gi),
            patch.object(github_app_auth, "HAS_GITHUB", True),
        ):
            with pytest.raises(GitHubAppAuthError, match="Failed to mint"):
                github_app_auth.get_installation_token()
