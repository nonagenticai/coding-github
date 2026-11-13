"""The andon brake at coding-github's mutating chokepoint (fr #1195).

THE PROPERTY: while fr's andon cord is engaged — or while its state cannot be
READ — this service performs no GitHub write. The scenario the brake exists for is
concrete: the cord is pulled, a mass-deactivation partially fails on
delivery-sweep, the 15-minute tick fires anyway, and ``delivery-sweep.json ::
Self-Merge PR`` merges a PR sitting at the ``:4`` check floor onto a repo's
``main``.

⭐ THE HARD HALF IS THE DIRECTION, NOT THE CHECK. fr's own ``readAndon()`` is
deliberately fail-OPEN for its other consumers. A caller that copies that
direction gets a brake that releases itself the moment fr is unreachable — which
is precisely when the estate is already in trouble. So the four "cannot read"
tests below (unreadable / uninitialised / unreachable / malformed) matter more
than the engaged one, and they share a single killing mutation:

    MUTATION (kills 4 tests at once): rewrite ``services.andon._verdict`` to
    proceed unless explicitly engaged --

        if engaged: raise ...
        return

    -- i.e. drop the reason ladder. ``test_unreadable_*``,
    ``test_uninitialised_*``, ``test_malformed_*`` go red. Combine with dropping
    the ``httpx.HTTPError`` handler in ``read_andon_state`` and
    ``test_unreachable_*`` goes red too.

Sections 9-11 close the three non-blocking findings an adversarial audit of this
branch raised. Each names its own killing mutation, and each was measured:

    9.  FINDING 1 — a read failure httpx does not classify (raw ``OSError``,
        ``ssl.SSLError``, bare ``TimeoutError``, ``httpx.InvalidURL``) escaped the
        module and produced an UNCODED 500. Never a hole — the write did not
        happen — but "the brake is broken" became indistinguishable from "the
        brake is pulled". MUTATION (measured): remove the terminal
        ``except Exception`` in ``read_andon_state`` -> 4 red of 52.
    10. FINDING 3 — fr fail-CLOSES a read error to ``engaged: true``, so
        ``{"engaged": true, "reason": "unreadable"}`` was reported as
        ``andon_engaged``: a halt attributed to a human who did not pull anything.
        MUTATION (measured): revert ``_verdict`` to checking ``engaged`` alone
        -> 2 red of 52.
    11. FINDING 2 — removing ``/webhook``'s gate killed ONLY the hardcoded
        ``EXPECTED_GATED`` census (1 of 40), so gate + census could be deleted in
        one hunk and stay green. MUTATION (measured): remove that route's
        ``dependencies=[...]`` -> 4 red of 52, three of them behavioural and
        independent of the census; the GitHub comment write was observed actually
        happening on the seam.

Every test here carries ``andon_live``, which opts OUT of the suite-wide
``_andon_released`` default in ``tests/conftest.py``, so the real read path runs.
That marker is NOT a skip: these tests always execute, and they assert on the HTTP
calls they cause, so a fixture that silently stopped the read from happening would
turn them red rather than green.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import ssl

import httpx
import pytest

import services.andon as andon
from routers import github_internal

pytestmark = pytest.mark.andon_live

MERGE_URL = "/api/repositories/merge-pull-request"
PR_CHECKS_URL = "/api/repositories/pr-checks"
CLONE_TOKEN_URL = "/api/internal/github/clone-token"
INTERNAL_TOKEN = "s3cret-internal-token"  # pragma: allowlist secret

RELEASED = {"engaged": False, "reason": "released"}
ENGAGED = {"engaged": True, "reason": "engaged"}
UNREADABLE = {"engaged": False, "reason": "unreadable"}
UNINITIALISED = {"engaged": False, "reason": "uninitialised"}


# ---------------------------------------------------------------------------
# A fake httpx client, so the andon read is exercised for real up to the socket.
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int, payload=None, non_json: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._non_json = non_json

    def json(self):
        if self._non_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _FakeAndonClient:
    """Records every andon GET. Class state so the test can count calls."""

    calls: list[tuple[str, float | None]] = []
    response: _FakeResponse | None = None
    error: Exception | None = None

    def __init__(self, timeout=None, **_kw) -> None:
        self._timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, url, headers=None):  # noqa: ARG002
        type(self).calls.append((url, self._timeout))
        if type(self).error is not None:
            raise type(self).error
        assert type(self).response is not None, "test did not arm an andon response"
        return type(self).response


@pytest.fixture
def andon_http(monkeypatch):
    """Install the fake client and reset the refusal counter.

    Patches ``AsyncClient`` on the httpx module ``services.andon`` actually holds,
    so ``read_andon_state`` runs unmodified — URL building, status handling, JSON
    decoding and the verdict ladder are all real.
    """
    _FakeAndonClient.calls = []
    _FakeAndonClient.response = None
    _FakeAndonClient.error = None
    andon.reset_andon_refusal_counts()
    monkeypatch.setattr(andon.httpx, "AsyncClient", _FakeAndonClient)
    yield _FakeAndonClient
    andon.reset_andon_refusal_counts()


def _arm(state: dict | None, *, status: int = 200, non_json: bool = False) -> None:
    _FakeAndonClient.response = _FakeResponse(status, state, non_json=non_json)
    _FakeAndonClient.error = None


def _arm_transport_error(exc: Exception) -> None:
    _FakeAndonClient.response = None
    _FakeAndonClient.error = exc


# ---------------------------------------------------------------------------
# A GitHub double good enough for BOTH merge and pr-checks, so the "released"
# control can reach a real 200 and the read-only control can too.
# ---------------------------------------------------------------------------
class _Run:
    def __init__(self, name: str, status: str, conclusion: str | None) -> None:
        self.name, self.status, self.conclusion = name, status, conclusion


class _Commit:
    def get_check_runs(self):
        return [_Run("ci", "completed", "success")]


class _Head:
    sha = "deadbeef"


class _PR:
    merged = False
    mergeable = True
    title = "t"

    def __init__(self) -> None:
        self.head = _Head()
        self.merge_calls: list[dict] = []

    def merge(self, **kwargs):
        self.merge_calls.append(kwargs)
        return type("R", (), {"merged": True, "sha": "abc123", "message": "ok"})()


class _Repo:
    private = True  # a private target repo (a public one is refused before merge)

    def __init__(self) -> None:
        self.pr = _PR()

    def get_pull(self, _n):
        return self.pr

    def commit_for(self, _sha):
        return _Commit()


@pytest.fixture
def github(monkeypatch, platform_tenant):
    """Arm the App credential rung and stub PyGithub, as production has it.

    Mirrors ``tests/unit/test_merge_uses_the_github_app.py::_app_ladder``: the env
    PAT is EMPTY in production, so the App is the only rung that can author.
    """
    import services.github_app_auth as ga
    import services.github_integration as gi
    from services.github.github_service import GitHubService

    repo = _Repo()

    monkeypatch.setattr(gi.settings, "system_github_owner", "nonagenticai")
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(ga, "app_auth_configured", lambda: True)
    monkeypatch.setattr(ga, "get_installation_token_for", lambda _o, _r: "ghs_test")
    monkeypatch.setattr("services.github.github_service.HAS_GITHUB", True)
    monkeypatch.setattr(
        "services.github.github_service.commit_handle", lambda r, sha: r.commit_for(sha)
    )

    async def _get_repo(_self, _full_name):
        return repo

    monkeypatch.setattr(GitHubService, "get_repository", _get_repo)

    ga.reset_cache()
    yield repo
    ga.reset_cache()


async def _merge(client, *, require_checks: bool = False):
    return await client.post(
        MERGE_URL,
        json={
            "owner": "nonagenticai",
            "repo": "enterprise_mcp_server",
            "pr_number": 140,
            "require_checks": require_checks,
        },
    )


def _andon(resp) -> str | None:
    """The refusal's machine-readable code, read off the wire as a caller sees it.

    ⚠️ Read via ``error.details.error_code``, NOT ``error.code``. coding-client's
    shared handler (``api/exception_handlers.py:_handle_http``) OVERWRITES the top
    level code with ``f"HTTP_{status}"`` and keeps only ``message`` + ``details``
    from the raised ``HTTPException``. So ``error.code`` is always ``"HTTP_503"``
    and asserting on it would pass for every one of the five reasons alike —
    which is the exact confusion this issue is about. The duplication of
    ``error_code`` into ``details`` in ``andon_error_http_exception`` is what makes
    the distinction survive the wire, and this helper is what proves it does.
    """
    body = resp.json()
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        details = error.get("details") or {}
        if isinstance(details, dict):
            return details.get("error_code") or details.get("andon")
    # Fallback for a raw FastAPI HTTPException body, in case the shared handler
    # is ever not installed — the tests must not silently pass in that case.
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return detail.get("error_code") or (detail.get("details") or {}).get("error_code")
    return None


# ===========================================================================
# 1. ENGAGED — the refusal the issue names
# ===========================================================================
class TestEngaged:
    async def test_engaged_refuses_the_merge_with_andon_engaged(
        self, async_client, andon_http, github
    ):
        """MUTATION: invert the predicate in ``_verdict`` (``if not engaged:``
        raise / ``if engaged:`` return) -> this goes red AND
        ``test_released_lets_the_merge_through`` goes red with it."""
        _arm(ENGAGED)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_engaged", resp.text
        # ⭐ The write did not happen. A 503 that still merged would be the worst
        # possible outcome and the status code alone cannot rule it out.
        assert github.pr.merge_calls == []

    async def test_the_refusal_says_the_cord_is_engaged_not_that_we_are_broken(
        self, async_client, andon_http, github
    ):
        """The caller must be able to tell "brake pulled" from "service broken".

        MUTATION: collapse every refusal to one code (or to a bare 500) -> red,
        because ``andon_engaged`` and ``andon_unreadable`` stop differing.
        """
        _arm(ENGAGED)
        engaged_code = _andon(await _merge(async_client))
        _arm(UNREADABLE)
        unreadable_code = _andon(await _merge(async_client))

        assert engaged_code == "andon_engaged"
        assert unreadable_code == "andon_unreadable"
        assert engaged_code != unreadable_code


# ===========================================================================
# 2. ⭐ THE FOUR "CANNOT READ" CASES — the whole difficulty of the issue
# ===========================================================================
class TestFailsClosedWhenUnreadable:
    async def test_unreadable_refuses(self, async_client, andon_http, github):
        """fr can be reached and says it cannot read its own cord state.

        MUTATION (the shared one in this module's docstring): proceed-unless-
        explicitly-engaged -> 200 here, and the brake is off exactly when fr is
        already unwell.
        """
        _arm(UNREADABLE)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_unreadable", resp.text
        assert github.pr.merge_calls == []

    async def test_uninitialised_refuses(self, async_client, andon_http, github):
        """No cord state has ever been written. Absence is not release.

        MUTATION: proceed-unless-explicitly-engaged -> 200. Also killed by
        treating ``uninitialised`` as a synonym for ``released``, which is the
        tempting reading of a fresh deployment.
        """
        _arm(UNINITIALISED)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_uninitialised", resp.text
        assert github.pr.merge_calls == []

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("name or service not known"),
            httpx.RemoteProtocolError("peer closed connection"),
        ],
        ids=["connect-timeout", "read-timeout", "dns/refused", "reset"],
    )
    async def test_unreachable_refuses(self, async_client, andon_http, github, exc):
        """fr cannot be reached at all — the case a mesh AuthorizationPolicy
        denying egress produces, and the one a fail-open brake sleeps through.

        MUTATION: delete the ``except httpx.HTTPError`` clause in
        ``read_andon_state`` -> ``_andon(resp)`` is no longer
        ``andon_unreachable`` -> red.

        ⚠️ WHAT IT BECOMES CHANGED, and the note is corrected rather than deleted:
        before the terminal ``except Exception`` was added (finding 1, section 9
        below) the error ESCAPED the module and the route's generic
        ``except Exception`` rendered an uncoded 500 ``Failed to merge PR``. Today
        it falls into the terminal handler and becomes a 503
        ``andon_check_failed`` — still red here, because ``andon_unreachable`` is
        the code this case owns, but red on the CODE rather than on the status.
        """
        _arm_transport_error(exc)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_unreachable", resp.text
        assert github.pr.merge_calls == []

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"state": None, "non_json": True}, "body is not JSON at all"),
            ({"state": ["released"]}, "body is a list, not an object"),
            ({"state": {"engaged": False}}, "'reason' key missing"),
            ({"state": {"reason": "released"}}, "'engaged' key missing"),
            ({"state": {"engaged": "false", "reason": "released"}}, "engaged is a STRING"),
            ({"state": {"engaged": 0, "reason": "released"}}, "engaged is an int"),
            ({"state": {"engaged": None, "reason": "released"}}, "engaged is null"),
            ({"state": {"engaged": False, "reason": None}}, "reason is null"),
            ({"state": {"engaged": False, "reason": "engaged"}}, "self-contradictory body"),
            ({"state": {"engaged": False, "reason": "probably fine"}}, "unknown reason"),
            ({"state": RELEASED, "status": 500}, "fr answered HTTP 500"),
            ({"state": RELEASED, "status": 404}, "andon route absent (old fr)"),
        ],
    )
    async def test_malformed_or_untrustworthy_answer_refuses(
        self, async_client, andon_http, github, kwargs, why
    ):
        """Anything we cannot positively read as released is a refusal.

        ⛔ ``{"engaged": "false"}`` is the one that would have bitten: the string
        "false" is TRUTHY, so a naive ``if payload["engaged"]:`` reads it as
        engaged and happens to refuse — but ``if not payload["engaged"]:`` reads
        it as RELEASED and merges. The ``isinstance(..., bool)`` check is what
        makes the answer independent of which way that comparison is written.

        MUTATION: drop the ``isinstance`` checks and the final unconditional
        ``raise`` in ``_verdict`` -> the string/int/null rows and both unknown-
        reason rows go red. MUTATION: drop the ``status_code != 200`` check ->
        the two HTTP rows go red.
        """
        state = kwargs.pop("state")
        _arm(state, **kwargs)

        resp = await _merge(async_client)

        assert resp.status_code == 503, f"{why}: {resp.text}"
        assert _andon(resp) == "andon_check_failed", f"{why}: {resp.text}"
        assert github.pr.merge_calls == [], why

    async def test_an_unconfigured_fr_url_refuses_rather_than_proceeding(
        self, async_client, andon_http, github, monkeypatch
    ):
        """No ``CODING_GITHUB_FR_URL`` ⇒ no brake ⇒ refuse. Fail closed on config too.

        MUTATION: make ``_andon_url`` return early / skip the check when ``fr_url``
        is empty -> 200, and a service deployed without that env var would have a
        silently disabled brake, which is the worst of the available outcomes.
        """
        from config.settings import get_settings

        monkeypatch.setattr(get_settings(), "fr_url", "")

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_check_failed", resp.text
        assert github.pr.merge_calls == []
        # Nothing was even dialled — there was nowhere to dial.
        assert andon_http.calls == []


# ===========================================================================
# 3. THE CONTROL — the guard must not be a brick
# ===========================================================================
class TestReleased:
    async def test_released_lets_the_merge_through(self, async_client, andon_http, github):
        """MUTATION: refuse-always (drop the ``return`` from ``_verdict``'s
        released branch) -> red. Without this test every other test in this file
        passes for a guard that blocks the whole service."""
        _arm(RELEASED)

        resp = await _merge(async_client)

        assert resp.status_code == 200, resp.text
        assert resp.json()["merged"] is True
        assert len(github.pr.merge_calls) == 1

    async def test_the_brake_consults_fr_on_the_right_url_with_a_short_timeout(
        self, async_client, andon_http, github
    ):
        """MUTATION: point the read at any other path, or drop the timeout -> red.

        A guard that dials the wrong URL would 404 and (correctly) refuse, so
        every refusal test above would still pass while the brake was in fact
        broken. This is the test that says the refusals are earned.
        """
        _arm(RELEASED)

        await _merge(async_client)

        assert len(andon_http.calls) == 1
        url, timeout = andon_http.calls[0]
        assert url.endswith("/api/internal/andon"), url
        assert timeout == 3.0

    async def test_the_verdict_is_never_cached(self, async_client, andon_http, github):
        """⛔ A cache is a window in which a pulled cord still reads as released.

        MUTATION: add any TTL/memoisation to ``read_andon_state`` (or an
        ``@lru_cache`` on ``_andon_url``'s verdict) -> the second call is served
        from the cache, ``calls`` stays at 1, red. The second half is the part
        that matters: the SECOND request sees the cord pulled and must refuse.
        """
        _arm(RELEASED)
        first = await _merge(async_client)
        _arm(ENGAGED)
        second = await _merge(async_client)

        assert first.status_code == 200, first.text
        assert second.status_code == 503, second.text
        assert _andon(second) == "andon_engaged"
        assert len(andon_http.calls) == 2, "the andon state was not re-read"


# ===========================================================================
# 4. READ-ONLY OPERATIONS MUST KEEP WORKING WHILE THE CORD IS PULLED
# ===========================================================================
class TestReadsAreNotGated:
    async def test_pr_checks_still_answers_while_the_cord_is_engaged(
        self, async_client, andon_http, github
    ):
        """⛔ A brake that blocks observation makes its own incident undiagnosable.

        MUTATION: add ``dependencies=[Depends(require_andon_released)]`` to
        ``get_pull_request_checks`` -> 503, red. That is the exact over-reach this
        test exists to prevent, and it is as much a defect as a missing gate.
        """
        _arm(ENGAGED)

        resp = await async_client.post(
            PR_CHECKS_URL,
            json={"owner": "nonagenticai", "repo": "enterprise_mcp_server", "pr_number": 140},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["verdict"] == "passed"
        # ⭐ And it did not even ASK: a read must not be coupled to fr's health.
        assert andon_http.calls == []

    async def test_no_read_only_route_carries_the_guard(self, async_client):
        """Census of the read side.

        MUTATION: gate any one of these -> red, naming the route. Keeps a future
        "gate everything, it is safer" edit from silently blinding the operator.
        """
        gated = _gated_operations()
        read_only = {
            "list_issues",
            "get_issue",
            "get_pull_request_checks",
            "get_pull_request_state",
            "list_pull_request_states",
            "get_repository",
            "list_pull_request_files",
            "list_workflow_runs",
            "get_branch",
            "compare_commits",
            "compare_files",
            "list_repositories",
            "get_provisioning_status",
            "read_repository_file",
            "get_repository_tree",
            "list_directory_contents",
            "search_repository",
            "validate_pull_request",
        }
        assert gated & read_only == set(), f"read-only routes were gated: {gated & read_only}"


# ===========================================================================
# 5. THE BLAST RADIUS, PINNED — which routes carry the guard
# ===========================================================================
def _gated_operations() -> set[str]:
    """Handler names of every route carrying ``require_andon_released``."""
    from fastapi.routing import APIRoute, iter_route_contexts

    from main import app

    gated: set[str] = set()
    # fastapi>=0.137 no longer flattens include_router() routes into app.routes;
    # iter_route_contexts is the public accessor for the effective, prefixed routes.
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        deps = [d.dependency for d in (route.dependencies or [])]
        deps += [d.call for d in route.dependant.dependencies]
        if andon.require_andon_released in deps:
            gated.add(route.endpoint.__name__)
    return gated


EXPECTED_GATED = {
    # --- GitHub writes on the tenant-facing surface ---------------------------
    "merge_pull_request",  # the failure scenario in fr #1195
    "create_pull_request",
    "apply_code_changes",  # creates a branch + commits + pushes
    "fork_repository",
    "mark_pull_request_ready",  # draft -> ready is a PR state change
    "update_pull_request_body",  # rewrites a PR description on GitHub
    "create_upstream_pull_request",
    "create_issue",
    "comment_on_issue",
    "add_issue_labels",
    "close_issue",
    "create_rust_target_repo",  # creates a GitHub repo + pushes its first commit
    # --- PR automation: labels, comments, REQUEST_CHANGES reviews, close ------
    "process_pr_validation",
    "validation_webhook",  # operation_id trigger_validation_webhook
    # --- inbound webhook whose live path posts a PR comment -------------------
    "github_webhook_handler",  # operation_id handle_github_webhook
}


class TestBlastRadius:
    async def test_the_gated_set_is_exactly_the_mutating_set(self, async_client):
        """MUTATION: remove the guard from any route (or add it to one not listed)
        -> red, naming the difference. This is the tripwire for someone "tidying"
        a dependency away, which is otherwise invisible: the route keeps working.
        """
        assert _gated_operations() == EXPECTED_GATED

    async def test_provisioning_is_deliberately_not_gated(self, async_client):
        """``provision_repository`` (``/import`` + ``/provision``) creates a
        WORKSPACE in coding-containers and writes nothing to GitHub — verified by
        grep: no ``create_*``/``update_*``/``merge``/``_ensure_fork`` call reaches
        it. It stays open because provisioning a workspace to read a repository is
        how an incident gets diagnosed while the cord is pulled.

        MUTATION: gate it -> red. If provisioning is ever given a GitHub write,
        this test is the one that must be changed FIRST, deliberately.
        """
        assert "provision_repository" not in _gated_operations()


# ===========================================================================
# 6. TOKEN MINTING — gated on the SCOPE, not on the route
# ===========================================================================
class TestPushScopeMinting:
    """⚠️ LOUD, as required: this DOES gate part of ``/api/internal/github/clone-token``.

    ``scope='push'`` mints ``contents:write``. That credential leaves the service
    and is used to push commits without traversing any other chokepoint here, so a
    pulled cord that did not cover it would still permit a push. The READ scopes
    are untouched, because gating them would stop every clone — including the ones
    an operator needs while the cord is pulled.
    """

    @pytest.fixture(autouse=True)
    def _internal(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import services.github_integration as gi
        from services.github_app_auth import RepoPushMetadata, ScopedToken

        monkeypatch.setattr(gi.settings, "system_github_owner", "acme")
        scoped = MagicMock(return_value=ScopedToken("ghs_scoped", "2999-01-01T00:00:00Z"))
        with (
            patch.object(
                github_internal,
                "get_settings",
                return_value=SimpleNamespace(
                    internal_callers={"coding-containers": INTERNAL_TOKEN}
                ),
            ),
            patch.object(github_internal, "get_scoped_clone_token", scoped),
            patch.object(github_internal, "get_repo_visibility", MagicMock(return_value="private")),
            # push scope requires a CONFIRMED-private repo (#118).
            patch.object(
                github_internal,
                "get_repo_push_metadata",
                MagicMock(return_value=RepoPushMetadata(True, "main")),
            ),
        ):
            yield scoped

    async def _mint(self, client, scope):
        body = {"repo_url": "https://github.com/acme/widgets"}
        if scope is not None:
            body["scope"] = scope
        return await client.post(
            CLONE_TOKEN_URL, json=body, headers={"X-Internal-Token": INTERNAL_TOKEN}
        )

    async def test_push_scope_is_refused_while_engaged_and_mints_nothing(
        self, async_client, andon_http, _internal
    ):
        """MUTATION: drop the ``if request.scope == "push"`` gate -> 200 and a
        ``contents:write`` credential is handed out under a pulled cord."""
        _arm(ENGAGED)

        resp = await self._mint(async_client, "push")

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_engaged", resp.text
        _internal.assert_not_called()

    async def test_push_scope_fails_closed_when_the_state_is_unreadable(
        self, async_client, andon_http, _internal
    ):
        """MUTATION: proceed-unless-engaged -> a write credential is minted while
        fr cannot say whether the cord is pulled."""
        _arm(UNREADABLE)

        resp = await self._mint(async_client, "push")

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_unreadable", resp.text
        _internal.assert_not_called()

    async def test_push_scope_succeeds_when_released(self, async_client, andon_http, _internal):
        """Control: the gate is conditional, not a block on push scope.

        MUTATION: refuse push scope unconditionally -> red.
        """
        _arm(RELEASED)

        resp = await self._mint(async_client, "push")

        assert resp.status_code == 200, resp.text
        _internal.assert_called_once_with("acme", "widgets", scope="push")

    @pytest.mark.parametrize(
        "scope", [None, "clone", "pr-read"], ids=["default", "clone", "pr-read"]
    )
    async def test_read_scopes_are_not_gated_even_while_engaged(
        self, async_client, andon_http, _internal, scope
    ):
        """⛔ A read-only credential must stay mintable under a pulled cord —
        otherwise coding-containers cannot clone a repo to look at it.

        MUTATION: gate the route unconditionally (a route-level
        ``dependencies=[...]``, or ``if request.scope in (...)`` widened) -> 503,
        red on all three rows.
        """
        _arm(ENGAGED)

        resp = await self._mint(async_client, scope)

        assert resp.status_code == 200, resp.text
        # And the brake was never consulted for a read-only mint.
        assert andon_http.calls == []


# ===========================================================================
# 7. OBSERVABILITY — a refusal is never silent
# ===========================================================================
class TestRefusalIsObservable:
    async def test_each_refusal_is_counted_by_surface_and_outcome(
        self, async_client, andon_http, github
    ):
        """MUTATION: remove the ``_record_refusal`` call from
        ``assert_andon_released`` -> the counter stays empty, red. A brake nobody
        can see firing is indistinguishable from a conveyor that simply stopped.
        """
        _arm(ENGAGED)
        await _merge(async_client)
        await _merge(async_client)
        _arm(UNREADABLE)
        await _merge(async_client)

        counts = andon.andon_refusal_counts()
        assert counts[("merge_pull_request", "andon_engaged")] == 2
        assert counts[("merge_pull_request", "andon_unreadable")] == 1

    async def test_the_refusal_is_logged_with_the_metric_label_shape(
        self, async_client, andon_http, github, caplog
    ):
        """The labels must be IN THE MESSAGE, not in ``extra=``.

        ``coding_client.api.logging_setup`` installs the format
        ``"%(asctime)s %(levelname)s [%(name)s] %(message)s"``, which renders
        nothing passed via ``extra=`` — so a label outside the message never
        reaches Loki.

        MUTATION: move ``surface``/``outcome`` into ``extra={...}`` -> the
        formatted record no longer contains them, red.
        """
        import logging

        _arm(ENGAGED)
        with caplog.at_level(logging.WARNING, logger="services.andon"):
            await _merge(async_client)

        rendered = [r.getMessage() for r in caplog.records]
        assert any(
            "andon_brake_refusal" in m
            and "surface=merge_pull_request" in m
            and "outcome=andon_engaged" in m
            and andon.ANDON_METRIC_NAME in m
            for m in rendered
        ), rendered

    async def test_a_permitted_operation_records_no_refusal(self, async_client, andon_http, github):
        """The negative control for the counter: an unchanged sibling.

        MUTATION: count on every check rather than on refusals -> red, and the
        counter would have said "the brake fired" on every merge ever.
        """
        _arm(RELEASED)
        await _merge(async_client)
        assert andon.andon_refusal_counts() == {}


# ===========================================================================
# 8. THE ESCAPE HATCH
# ===========================================================================
class TestEscapeHatch:
    async def test_the_shipped_allow_list_is_empty(self):
        """⭐ THE DECISION: no operation is exempt, because releasing the cord
        needs nothing from this service — the cord lives in fr and is released
        through fr, and every diagnostic read stays ungated.

        MUTATION: add any entry -> red. That is the point: an exemption cannot be
        added without a reviewer seeing this assertion change, and it can never be
        added by an environment variable, because there is no such variable.
        """
        assert andon.ANDON_EXEMPT_OPERATIONS == frozenset()

    async def test_the_hatch_mechanism_actually_works(
        self, async_client, andon_http, github, monkeypatch
    ):
        """An untested hatch is a hatch that will not open when it is needed.

        The allow-list ships EMPTY, so this proves the mechanism rather than an
        entry: with ``merge_pull_request`` exempted, an ENGAGED cord lets it
        through. If a future incident does require an exemption, this is the test
        that says adding one line to ``ANDON_EXEMPT_OPERATIONS`` will be enough.

        MUTATION: delete the ``if operation in ANDON_EXEMPT_OPERATIONS`` branch ->
        503, red. MUTATION: check the allow-list AFTER reading the state -> the
        andon assertion below fails, because an exempt operation must not depend
        on fr being reachable at all.
        """
        monkeypatch.setattr(andon, "ANDON_EXEMPT_OPERATIONS", frozenset({"merge_pull_request"}))
        _arm(ENGAGED)

        resp = await _merge(async_client)

        assert resp.status_code == 200, resp.text
        assert len(github.pr.merge_calls) == 1
        assert andon_http.calls == [], "an exempt operation must not need fr at all"
        assert andon.andon_refusal_counts() == {}

    async def test_an_exemption_does_not_leak_to_other_operations(
        self, async_client, andon_http, github, monkeypatch
    ):
        """MUTATION: make the exemption check truthy for anything (e.g. ``if
        ANDON_EXEMPT_OPERATIONS:``) -> the issue writes would ride along, red."""
        monkeypatch.setattr(andon, "ANDON_EXEMPT_OPERATIONS", frozenset({"merge_pull_request"}))
        _arm(ENGAGED)

        resp = await async_client.post(
            "/api/repositories/close-issue",
            json={"owner": "nonagenticai", "repo": "enterprise_mcp_server", "issue_number": 1},
        )

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_engaged", resp.text


# ===========================================================================
# 9. FINDING 1 — a read failure httpx does NOT classify must still carry a code
#
# The auditor measured that ``read_andon_state`` caught only
# ``httpx.HTTPError``, so a raw ``OSError``, an ``ssl.SSLError``, a bare
# ``asyncio.TimeoutError`` and ``httpx.InvalidURL`` escaped the module untouched.
# The write never happened — so this was never a HOLE — but the caller got a 500
# with NO ``error_code``, i.e. exactly the "brake pulled" vs "brake broken"
# confusion the whole design exists to prevent, and which
# ``test_unreachable_refuses`` above already calls insufficient in prose.
# ===========================================================================
class TestUnclassifiedReadFailuresStillCarryACode:
    @pytest.mark.parametrize(
        "exc",
        [
            OSError(104, "Connection reset by peer"),
            ssl.SSLError("WRONG_VERSION_NUMBER"),
            TimeoutError(),
            httpx.InvalidURL("No host included in URL"),
        ],
        # ⚠️ ``asyncio.TimeoutError`` IS the builtin ``TimeoutError`` from 3.11 on
        # (a plain alias), so the builtin is what is spelled here.
        ids=["raw-oserror", "ssl-error", "bare-asyncio-timeout", "invalid-url"],
    )
    async def test_an_unclassified_read_failure_refuses_with_andon_check_failed(
        self, async_client, andon_http, github, exc
    ):
        """MUTATION: delete the terminal ``except Exception`` clause in
        ``read_andon_state`` -> the exception escapes ``services.andon``, the
        route's own ``except Exception`` renders a generic 500 ``Failed to merge
        PR``, ``resp.status_code`` is 500 and ``_andon(resp)`` is ``None`` -> all
        four rows red.

        ⛔ None of these is an ``httpx.HTTPError``: ``ssl.SSLError`` and
        ``TimeoutError`` are ``OSError`` subclasses raised by the socket layer,
        and ``httpx.InvalidURL`` inherits straight from ``Exception`` (it is what
        ``client.get()`` raises for a malformed ``fr_url``). So ordering the new
        clause after ``except httpx.HTTPError`` cannot steal ``andon_unreachable``
        from the cases that own it — ``test_unreachable_refuses`` above is the
        unchanged sibling that proves it did not.
        """
        _arm_transport_error(exc)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_check_failed", resp.text
        # ⭐ Same assertion every refusal test here makes: the write did not happen.
        assert github.pr.merge_calls == []

    async def test_the_terminal_handler_cannot_swallow_the_guards_own_refusal(
        self, async_client, andon_http, github
    ):
        """``AndonRefusedError`` IS an ``Exception``, so only CLAUSE ORDER protects it.

        This drives a refusal raised INSIDE the guard (an ENGAGED cord, decided by
        ``_verdict``) and asserts it kept its own code. MUTATION: move the terminal
        ``except Exception`` so it can see an ``AndonRefusedError`` (e.g. wrap the
        whole body of ``read_andon_state``, including ``_verdict``, in the try) ->
        every distinct code collapses to ``andon_check_failed`` and this goes red.
        """
        _arm(ENGAGED)
        assert _andon(await _merge(async_client)) == "andon_engaged"
        _arm(UNINITIALISED)
        assert _andon(await _merge(async_client)) == "andon_uninitialised"
        assert github.pr.merge_calls == []


# ===========================================================================
# 10. FINDING 3 — a definite ``engaged`` must halt, but the CODE must be honest
#
# fr's own gate fail-CLOSES a read error to ``engaged: true``, so fr emits
# ``{"engaged": true, "reason": "unreadable"}``. Reporting that as
# ``andon_engaged`` tells the operator a human pulled the cord when in truth fr
# could not read its own state — and ``error_code`` is the field callers branch
# on, so the ``(reason: ...)`` text in the message is not enough.
#
# ⚠️ NOTE FOR A REVIEWER LOOKING FOR THE TEST THIS CHANGED: there isn't one. The
# ``UNREADABLE``/``UNINITIALISED`` fixtures at the top of this module both carry
# ``engaged: False``, so no existing assertion covered the ``engaged: true``
# shape fr actually sends — which is precisely why the defect survived.
# ===========================================================================
ENGAGED_UNREADABLE = {"engaged": True, "reason": "unreadable"}
ENGAGED_UNINITIALISED = {"engaged": True, "reason": "uninitialised"}
ENGAGED_CONTRADICTORY = {"engaged": True, "reason": "released"}


class TestTheHaltCodeIsHonestAboutWhy:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (ENGAGED_UNREADABLE, "andon_unreadable"),
            (ENGAGED_UNINITIALISED, "andon_uninitialised"),
        ],
        ids=["engaged+unreadable", "engaged+uninitialised"],
    )
    async def test_a_diagnostic_reason_is_not_reported_as_a_pulled_cord(
        self, async_client, andon_http, github, state, expected
    ):
        """MUTATION: revert ``_verdict`` to checking ``engaged`` alone (raise
        ``andon_engaged`` and return before the reason ladder) -> both rows report
        ``andon_engaged`` -> red.

        ⛔ The halt itself is NOT what this test relaxes: the 503 and the empty
        ``merge_calls`` are asserted on the same rows, so a "fix" that made an
        unreadable state PROCEED in order to stop calling it engaged would go red
        here too.
        """
        _arm(state)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == expected, resp.text
        assert github.pr.merge_calls == []

    async def test_engaged_still_wins_over_an_unrecognised_reason(
        self, async_client, andon_http, github
    ):
        """⭐ THE INVARIANT THE HONESTY FIX MUST NOT COST.

        ``{"engaged": true, "reason": "released"}`` is self-contradictory. It must
        still refuse, and as ``andon_engaged`` — the ``engaged`` bit wins, and the
        unrecognised reason is not permitted to route it into a
        ``check_failed``/proceed branch.

        MUTATION: move the ``engaged`` test BELOW the reason ladder (i.e. decide
        on ``reason`` first) -> this row becomes ``andon_check_failed`` or, worse,
        reaches the released branch -> red.
        """
        _arm(ENGAGED_CONTRADICTORY)

        resp = await _merge(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_engaged", resp.text
        assert github.pr.merge_calls == []

    async def test_the_engaged_false_shapes_are_unchanged(self, async_client, andon_http, github):
        """The unchanged sibling: fr's ``engaged: false`` diagnostic shapes still
        map to the same two codes they always did, so the fix above widened the
        codes' meaning rather than moving it."""
        _arm(UNREADABLE)
        assert _andon(await _merge(async_client)) == "andon_unreadable"
        _arm(UNINITIALISED)
        assert _andon(await _merge(async_client)) == "andon_uninitialised"
        assert github.pr.merge_calls == []


# ===========================================================================
# 11. FINDING 2 — the webhook route, driven BEHAVIOURALLY
#
# The auditor mutated ``dependencies=[...]`` off the ``/webhook`` route and killed
# exactly 1 of 40 tests: ``test_the_gated_set_is_exactly_the_mutating_set``, a
# hardcoded census. Versus 2/4/2/3 for every other mutation. So an edit that
# removed the gate AND deleted ``github_webhook_handler`` from ``EXPECTED_GATED``
# — one hunk, both halves plausibly "tidying" — was GREEN, and this is the route
# whose GitHub write is the least obvious of the thirteen:
#
#   POST /api/repositories/webhooks/webhook
#     -> SecureWebhookHandler.handle_webhook            (signature + routing)
#     -> secure_webhook_handler.handle_pull_request_event
#     -> routers.github_webhooks.analyze_pull_request_changes   (:365-367)
#     -> routers.github_webhooks._post_pr_comment
#     -> GitHubService.add_pull_request_comment -> create_issue_comment
#
# THE SEAM asserted on below is ``routers.github_webhooks._post_pr_comment``: the
# single funnel through which this route's live path reaches GitHub (it is called
# on BOTH the success and the analysis-error branch of
# ``analyze_pull_request_changes``, so neither branch can slip past it). It is the
# webhook's equivalent of ``github.pr.merge_calls``.
# ===========================================================================
WEBHOOK_URL = "/api/repositories/webhooks/webhook"
_WEBHOOK_SECRET = "andon-chokepoint-test-secret"  # pragma: allowlist secret

# Platform-owned on purpose: ``secure_webhook_handler._platform_repo`` short-
# circuits a CUSTOMER repo before GitHub is touched, so a customer-repo payload
# would make the seam un-fireable and the refusal assertion vacuous.
_PR_OPENED_EVENT = {
    "action": "opened",
    "number": 140,
    "pull_request": {"number": 140, "title": "t"},
    "repository": {
        "full_name": "nonagenticai/enterprise_mcp_server",
        "name": "enterprise_mcp_server",
        "owner": {"login": "nonagenticai"},
    },
    "sender": {"login": "octocat"},
}

# The minimum ``PRAnalysisDict`` ``_generate_pr_analysis_summary`` can render.
_EMPTY_ANALYSIS = {
    "statistics": {"total": 0, "added": 0, "modified": 0, "removed": 0},
    "total_additions": 0,
    "total_deletions": 0,
    "languages": {},
    "files": [],
}


@pytest.fixture
def webhook_writes(monkeypatch, github):
    """Wire the webhook's live PR path and return the recorded GitHub writes.

    ``github`` is pulled in for its ``system_github_owner='nonagenticai'`` and its
    platform tenant override, so ``_platform_repo`` admits the payload above.

    Production webhooks have NO GitHub credential (no PAT exists and a webhook
    carries no tenant for the App, owner ruling 2026-09-30), so
    ``_webhook_github_token`` returns ``None`` and the PR path skips. To reach the
    comment seam at all, this fixture stubs that credential lookup — it sits
    DOWNSTREAM of the andon gate like everything else replaced here.

    Everything replaced here is DOWNSTREAM of the andon gate, so nothing in this
    fixture can hide a missing gate — and the released-cord control below proves
    the wiring actually reaches the seam.
    """
    import services.github.secure_webhook_handler as swh
    from routers import github_webhooks as gw

    comments: list[tuple[str, int]] = []

    async def _record(repo_name, pr_number, comment):  # noqa: ARG001
        comments.append((repo_name, pr_number))

    monkeypatch.setattr(gw, "_post_pr_comment", _record)
    monkeypatch.setattr(gw, "_webhook_github_token", lambda _repo, _feature: "ghs_test")

    class _Hub:
        async def analyze_pull_request(self, _repo_url, _pr_number):
            return dict(_EMPTY_ANALYSIS)

    monkeypatch.setattr(gw, "GitHubIntegrationHub", _Hub)

    # A fresh handler with a secret we know, registered exactly as
    # ``get_secure_webhook_handler`` registers the production one.
    handler = swh.SecureWebhookHandler(
        webhook_secret=_WEBHOOK_SECRET,
        security_level=swh.WebhookSecurityLevel.STANDARD,
    )
    handler.register_event_handler("pull_request", swh.handle_pull_request_event)
    monkeypatch.setattr(swh, "_webhook_handler", handler)
    return comments


async def _deliver_pr_opened(client):
    """POST a signature-valid ``pull_request``/``opened`` delivery."""
    body = json.dumps(_PR_OPENED_EVENT).encode()
    digest = hmac.new(_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return await client.post(
        WEBHOOK_URL,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": f"sha256={digest}",
            "X-GitHub-Delivery": "andon-test-delivery",
        },
    )


class TestTheWebhookIsGatedBehaviourally:
    @pytest.fixture(autouse=True)
    def _legacy_all_webhook_mode(self, monkeypatch):
        """These deliveries exercise the PR-validation WRITE path, which since
        coding-github#131 only runs with CODING_GITHUB_WEBHOOK_MODE=all (the code
        default, outcome_only, answers 202 ignored before any handler). The andon
        dependency sits on the route itself, so it gates both modes."""
        from config.settings import get_settings

        monkeypatch.setattr(get_settings(), "webhook_mode", "all")

    async def test_the_seam_fires_when_the_cord_is_released(
        self, async_client, andon_http, webhook_writes
    ):
        """⭐ THE POSITIVE CONTROL, and it comes first on purpose.

        Without it the refusal test below would pass for a delivery that never
        reaches GitHub for some unrelated reason (a bad signature, an unparsed
        event type, a customer-repo skip) — a gate that cannot fail. This says the
        route really does write, so the empty list in the next test is earned.
        """
        _arm(RELEASED)

        resp = await _deliver_pr_opened(async_client)

        assert resp.status_code == 200, resp.text
        assert resp.json()["processed"] is True, resp.text
        assert webhook_writes == [("nonagenticai/enterprise_mcp_server", 140)]
        assert len(andon_http.calls) == 1, "the webhook did not consult the brake"

    async def test_an_engaged_cord_refuses_the_webhook_and_writes_nothing(
        self, async_client, andon_http, webhook_writes
    ):
        """MUTATION: delete ``dependencies=[Depends(require_andon_released)]`` from
        the ``/webhook`` route -> 200, ``webhook_writes`` gains a PR comment, and
        this goes red ON ITS OWN, without ``EXPECTED_GATED``. That is the whole
        point of this class: the census pinned the WIRING, nothing pinned the
        BEHAVIOUR, so the census could be edited in the same hunk as the gate.
        """
        _arm(ENGAGED)

        resp = await _deliver_pr_opened(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_engaged", resp.text
        # ⭐ No GitHub write. The seam the live path funnels through stayed empty.
        assert webhook_writes == []

    async def test_the_webhook_also_fails_closed_when_the_state_is_unreadable(
        self, async_client, andon_http, webhook_writes
    ):
        """The direction that matters most, on the least obvious route: fr cannot
        say whether the cord is pulled, and the webhook still writes nothing.

        MUTATION: proceed-unless-explicitly-engaged in ``_verdict`` -> 200 and a
        comment is posted while fr is unwell -> red.
        """
        _arm(UNREADABLE)

        resp = await _deliver_pr_opened(async_client)

        assert resp.status_code == 503, resp.text
        assert _andon(resp) == "andon_unreadable", resp.text
        assert webhook_writes == []
