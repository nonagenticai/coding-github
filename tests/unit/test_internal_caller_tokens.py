"""Per-caller secrets on ``/api/internal`` (``CODING_GITHUB_INTERNAL_CALLER_TOKENS``).

Invariant: one service = one secret. ``CODING_GITHUB_INTERNAL_SERVICE_TOKEN``
keeps authenticating coding-containers unchanged; every other caller (fr, ...)
gets its own entry in the JSON caller map, and startup refuses any config that
would let two callers share a credential.

These build REAL ``Settings`` objects (not SimpleNamespace) so the startup
validation and the router guard are exercised together.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings, parse_internal_caller_tokens
from routers import github_internal
from services.github_app_auth import ScopedToken

CLONE_TOKEN_URL = "/api/internal/github/clone-token"
LEGACY = "legacy-containers-secret-AAAA"  # pragma: allowlist secret
FR = "fr-own-secret-BBBB"  # pragma: allowlist secret
OTHER = "other-own-secret-CCCC"  # pragma: allowlist secret
BODY = {"repo_url": "https://github.com/acme/widgets"}


def _make(legacy: str = "", callers: str = "") -> Settings:
    # environment=test sidesteps the unrelated required-in-production URL checks.
    return Settings(
        _env_file=None,
        environment="test",
        internal_service_token=legacy,
        internal_caller_tokens=callers,
    )


@pytest.fixture
def mock_github():
    scoped = MagicMock(return_value=ScopedToken("ghs_scoped", "2999-01-01T00:00:00Z"))
    visibility = MagicMock(return_value="private")
    with (
        patch.object(github_internal, "get_scoped_clone_token", scoped),
        patch.object(github_internal, "get_repo_visibility", visibility),
    ):
        yield scoped


async def _post(async_client, settings: Settings, token: str | None):
    headers = {} if token is None else {"X-Internal-Token": token}
    with patch.object(github_internal, "get_settings", return_value=settings):
        return await async_client.post(CLONE_TOKEN_URL, json=BODY, headers=headers)


def _served_lines(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "Clone token request served" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# Accepted map
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestAcceptedCallers:
    def test_legacy_only_is_coding_containers(self):
        assert _make(legacy=LEGACY).internal_callers == {"coding-containers": LEGACY}

    def test_union_of_legacy_and_map(self):
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}", "other": "{OTHER}"}}')
        assert s.internal_callers == {"coding-containers": LEGACY, "fr": FR, "other": OTHER}

    def test_map_only_without_legacy(self):
        assert _make(callers=f'{{"fr": "{FR}"}}').internal_callers == {"fr": FR}

    def test_nothing_configured_is_empty(self):
        assert _make().internal_callers == {}

    def test_reads_env_var_names(self, monkeypatch):
        monkeypatch.setenv("CODING_GITHUB_INTERNAL_SERVICE_TOKEN", LEGACY)
        monkeypatch.setenv("CODING_GITHUB_INTERNAL_CALLER_TOKENS", f'{{"fr": "{FR}"}}')
        s = Settings(_env_file=None, environment="test")
        assert s.internal_callers == {"coding-containers": LEGACY, "fr": FR}


# ---------------------------------------------------------------------------
# Startup validation — fail fast, name the key, never the token
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestStartupValidation:
    @pytest.mark.parametrize(
        ("raw", "fragment"),
        [
            (f'{{"fr": "{FR}"', "not valid JSON"),
            (f'["{FR}"]', "must be a JSON object"),
            ('{"fr": 12345}', "'fr'] must be a string"),
            ('{"fr": ""}', "'fr'] is an empty token"),
            ('{"fr": "   "}', "'fr'] is an empty token"),
            (f'{{"": "{FR}"}}', "empty caller name"),
            (f'{{"fr": "{FR}", "other": "{FR}"}}', "share the same token"),
        ],
    )
    def test_invalid_map_crashes_startup(self, raw, fragment):
        with pytest.raises(RuntimeError) as exc:
            _make(callers=raw)
        msg = str(exc.value)
        assert "CODING_GITHUB_INTERNAL_CALLER_TOKENS" in msg
        assert fragment in msg
        assert FR not in msg
        # A JSON decode error would otherwise chain the raw document in.
        assert exc.value.__cause__ is None

    def test_reusing_the_legacy_token_crashes_startup(self):
        """fr must NOT be handed coding-containers' secret."""
        with pytest.raises(RuntimeError) as exc:
            _make(legacy=LEGACY, callers=f'{{"fr": "{LEGACY}"}}')
        assert "'fr'" in str(exc.value)
        assert LEGACY not in str(exc.value)

    def test_coding_containers_in_map_while_legacy_set_crashes(self):
        with pytest.raises(RuntimeError) as exc:
            _make(legacy=LEGACY, callers=f'{{"coding-containers": "{OTHER}"}}')
        assert "coding-containers" in str(exc.value)
        assert OTHER not in str(exc.value)

    def test_blank_map_is_empty_not_an_error(self):
        assert parse_internal_caller_tokens("   ") == {}


# ---------------------------------------------------------------------------
# The guard, end to end through the router
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGuardPerCaller:
    async def test_legacy_token_still_accepted_as_coding_containers(
        self, async_client, mock_github, caplog
    ):
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}"}}')
        with caplog.at_level(logging.DEBUG):
            r = await _post(async_client, s, LEGACY)
        assert r.status_code == 200
        assert any("caller=coding-containers" in m for m in _served_lines(caplog))

    async def test_map_token_accepted_with_its_caller(self, async_client, mock_github, caplog):
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}", "other": "{OTHER}"}}')
        with caplog.at_level(logging.DEBUG):
            r = await _post(async_client, s, FR)
        assert r.status_code == 200
        lines = _served_lines(caplog)
        assert any("caller=fr," in m for m in lines)
        assert not any("caller=coding-containers" in m for m in lines)

    async def test_wrong_token_403_and_nothing_minted(self, async_client, mock_github):
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}"}}')
        r = await _post(async_client, s, "not-any-callers-token")
        assert r.status_code == 403
        mock_github.assert_not_called()

    async def test_missing_header_403(self, async_client, mock_github):
        r = await _post(async_client, _make(callers=f'{{"fr": "{FR}"}}'), None)
        assert r.status_code == 403

    @pytest.mark.parametrize("presented", ["", "anything", LEGACY])
    async def test_empty_config_403(self, async_client, mock_github, presented):
        r = await _post(async_client, _make(), presented)
        assert r.status_code == 403
        mock_github.assert_not_called()

    async def test_compares_against_every_entry_no_short_circuit(self, async_client, mock_github):
        """A match on the FIRST entry must still compare the rest (timing)."""
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}", "other": "{OTHER}"}}')
        real = github_internal.hmac.compare_digest
        calls = MagicMock(side_effect=real)
        with patch.object(github_internal.hmac, "compare_digest", calls):
            r = await _post(async_client, s, LEGACY)
        assert r.status_code == 200
        assert calls.call_count == 3

    async def test_no_token_ever_logged(self, async_client, mock_github, caplog):
        s = _make(legacy=LEGACY, callers=f'{{"fr": "{FR}", "other": "{OTHER}"}}')
        with caplog.at_level(logging.DEBUG):
            for presented in (LEGACY, FR, OTHER, "wrong-presented-value", None):
                await _post(async_client, s, presented)
            await _post(async_client, _make(), FR)
        text = "\n".join(r.getMessage() for r in caplog.records) + caplog.text
        # Positive control: the logs we are scanning are the endpoint's own.
        assert "caller=fr," in text
        assert "Rejected internal request" in text
        for secret in (LEGACY, FR, OTHER, "wrong-presented-value"):
            assert secret not in text
