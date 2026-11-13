"""The andon brake: coding-github refuses to WRITE to GitHub while the cord is pulled.

fr owns the cord. This module is the read path, and it is deliberately the ONLY
place the verdict is computed (fr issue #1195).

⛔ THE PROPERTY THIS MODULE EXISTS TO PROTECT

The cord is pulled during a residual window — a mass-deactivation that partially
failed, an operator stopping the conveyor mid-flight. In exactly that window a
scheduled sweep tick fires (``AA Maintenance - Delivery Sweep`` every 15 min) and
a PR sitting at the ``:4`` check floor gets merged to a repo's ``main``. The three
sweeps (delivery / blocked / canary) each hold an unconsulted mutating call site:
``delivery-sweep.json :: Self-Merge PR`` (``coding.mergePullRequest``),
``blocked-sweep :: Re-mint Approval``, ``canary-sweep :: Delete Worker Canary``.

The brake therefore lives HERE, at the chokepoint, and not in the workflow nodes:
a per-workflow check covers the three callers somebody has already written, and
none of the ones nobody has written yet.

⛔ FAIL CLOSED, AND THAT IS THE WHOLE DIFFICULTY

``fr``'s own internal ``readAndon()`` is deliberately fail-OPEN for its other
consumers. That direction must NOT travel across the wire into this caller. Here,
an andon state we cannot READ is treated exactly like an andon state that says
ENGAGED. We proceed on one condition only:

    HTTP 200, a well-formed body, ``engaged is False`` AND ``reason == "released"``

Everything else refuses: a non-200, a non-JSON body, a body of the wrong shape, an
``engaged`` that is not a real bool, an unrecognised ``reason``, a timeout, a DNS
failure, an unconfigured ``fr_url``, and — via the terminal handler in
:func:`read_andon_state` — ANY other exception the read can raise, including the
ones httpx does not classify (raw ``OSError``, ``ssl.SSLError``, bare
``asyncio.TimeoutError``, ``httpx.InvalidURL``). Each refusal carries its OWN
machine-readable code so a caller can tell "the brake is pulled" from "the brake
is broken" — returning a generic 500 for the second case is what makes an incident
undiagnosable, and an exception escaping this module uncoded produces exactly that
500 even though the write is correctly prevented.

⚠️ AND THE CODE MUST BE HONEST, not merely present. fr's own gate fail-closes a
read error to ``{"engaged": true, "reason": "unreadable"}``. Any definite
``engaged`` halts — that is not negotiable — but a halt for THAT reason is
reported as ``andon_unreadable`` (and ``uninitialised`` as
``andon_uninitialised``), never as ``andon_engaged``: telling an operator a human
pulled the cord when fr merely could not read it sends them to the wrong place.
``engaged`` still wins over an *unrecognised* reason, so the self-contradictory
``{"engaged": true, "reason": "released"}`` refuses with ``andon_engaged``.

⛔ NO CACHE. NO MEMOISATION. NO TTL.

Every mutating operation re-reads the state. A cache is a tolerance window in
which a pulled cord is still reported as released, and the operator rejected that
explicitly. The cost is one in-cluster GET per mutating operation, and these
operations are low-frequency by construction (a sweep tick, a card transition) —
not a hot path. ``tests/unit/test_andon_chokepoint.py`` pins the absence of a
cache by counting HTTP calls.

⛔ NO KILL SWITCH. There is deliberately no environment variable that disables
this brake, because an env var disables it for EVERY operation at once and does so
invisibly. The only way to exempt an operation is
:data:`ANDON_EXEMPT_OPERATIONS` — an explicit, empty, reviewable allow-list.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from coding_client.models.responses import error_response
from fastapi import HTTPException, Request

from config.settings import get_settings

logger = logging.getLogger(__name__)

ANDON_PATH = "/api/internal/andon"

# 3 seconds. Justified rather than picked:
#
# * These are LOW-FREQUENCY operations (a 15-minute sweep tick, a card
#   transition), so the added latency is irrelevant to throughput.
# * The hop is in-cluster and plaintext (``fr-backend.fr.svc:8888``) against a
#   state read, so a healthy fr answers in single-digit milliseconds. 3s is ~2
#   orders of magnitude of headroom.
# * The asymmetry that sets the FLOOR: a timeout REFUSES. Too short and a merely
#   slow-but-healthy fr brakes the conveyor for no reason (a false refusal, which
#   looks exactly like a real one). Too long only delays a refusal that is
#   already correct. So the value is chosen to be generous, not tight.
# * ``tenant_binding`` uses 5.0s for the binding lookup; this is deliberately
#   shorter because it is on EVERY mutating operation rather than on the
#   customer-repo path alone.
_ANDON_TIMEOUT_SECONDS = 3.0

# The only (engaged, reason) pair that permits a mutating operation.
_RELEASED_REASON = "released"

# fr's documented reason vocabulary for GET /api/internal/andon.
_REASON_UNREADABLE = "unreadable"
_REASON_UNINITIALISED = "uninitialised"
_REASON_ENGAGED = "engaged"

# ---------------------------------------------------------------------------
# THE ESCAPE HATCH.
#
# ⭐ It is EMPTY, and that is a decision, not an omission.
#
# A brake that can lock out its own recovery is worse than the failure it
# prevents, so the hatch has to exist. It is empty because releasing the cord
# needs NOTHING from this service:
#
#   * The cord's state lives in fr and is released through fr (the same
#     ``/api/internal/andon`` surface this module reads, plus fr's own UI/API).
#     No coding-github route participates in engaging or releasing it.
#   * Recovering from the named failure scenario — a mass-deactivation that
#     partially failed — means deactivating the remaining workers in fr and then
#     releasing the cord in fr. Again, no GitHub write.
#   * Everything an operator needs in order to DIAGNOSE while the cord is pulled
#     is a READ, and no read is gated (see the classification in the PR body):
#     PR checks, PR state, PR files, workflow runs, repository reads, the file /
#     tree / directory / search routes, and ``validate_pull_request`` (pure
#     analysis, ``PERMS_PR_READ``, writes nothing) all stay available.
#
# ⚠️ CRITERION for ever adding an entry — all three, not any:
#   1. The operation is REQUIRED to release the cord or to stop the bleeding.
#   2. It cannot be performed any other way (by hand on github.com, or via fr).
#   3. Its blast radius while the cord is pulled is smaller than the cost of not
#      having it.
# Each entry carries its reason inline. Adding one is a one-line diff in this
# file and therefore reviewable; that is the entire point of a constant rather
# than a config value.
# ---------------------------------------------------------------------------
ANDON_EXEMPT_OPERATIONS: frozenset[str] = frozenset()


class AndonRefusedError(Exception):
    """A mutating operation is refused because of the andon state.

    ``code`` is the machine-readable reason and is the field callers branch on.
    ``http_status`` is 503 for every case ON PURPOSE: the status says "refusing,
    try later", which is true of a pulled cord and of an unreadable one alike, and
    a uniform status forces the caller to read ``code`` instead of guessing from
    the number. The five codes are:

    ``andon_engaged``       fr says the cord IS pulled, for a reason that means a
                            human (or fr's own policy) pulled it. The intended
                            refusal.
    ``andon_unreadable``    fr could not read its own andon state. ⚠️ fr reports
                            this with ``engaged`` EITHER way: its internal gate
                            fail-closes a read error to ``engaged: true, reason:
                            "unreadable"``. Both shapes map here, because the
                            code must say WHY we are refusing and "fr cannot read
                            its own cord" is not "somebody pulled the cord".
    ``andon_uninitialised`` fr has no andon state at all — again with ``engaged``
                            either way, for the same reason.
    ``andon_unreachable``   the HTTP call to fr failed and httpx classified it
                            (timeout, DNS, reset, refused connection).
    ``andon_check_failed``  fr answered, but not with something we can trust
                            (non-200, non-JSON, wrong shape, unknown reason) —
                            or this service has no ``fr_url`` configured — or the
                            read raised something httpx does NOT classify as an
                            ``httpx.HTTPError`` (a raw ``OSError``,
                            ``ssl.SSLError``, ``asyncio.TimeoutError``,
                            ``httpx.InvalidURL``). ⛔ That last clause exists
                            because such an exception used to escape this module
                            entirely: the write still did not happen, but the
                            caller got a 500 with NO ``error_code``, which is
                            exactly the "brake pulled" / "brake broken"
                            confusion this class exists to prevent.
    """

    http_status = 503

    def __init__(self, code: str, message: str, operation: str) -> None:
        super().__init__(message)
        self.code = code
        self.operation = operation


# ---------------------------------------------------------------------------
# Refusal accounting.
#
# ⚠️ THIS SERVICE HAS NO PROMETHEUS REGISTRY, and this comment is here so nobody
# concludes one was overlooked. ``prometheus_client`` is not a dependency (it is
# absent from ``pyproject.toml`` and ``uv.lock``), there is no ``/metrics`` route
# in ``main.py``, and ``coding_client.api.metrics`` says so in its own docstring:
# "Uses structured-log timing rather than Prometheus counters". Inventing a
# registry here would mean a new dependency, a new route and a new ServiceMonitor
# — a separate change.
#
# So a refusal is made observable the way this service ALREADY makes a policy
# decision observable: a single WARNING whose labels are key=value pairs IN THE
# MESSAGE STRING. That is not a stylistic choice — ``configure_logging`` installs
# the format ``"%(asctime)s %(levelname)s [%(name)s] %(message)s"``, which renders
# NOTHING passed via ``extra=``, so labels outside the message would be invisible
# in Loki. The existing precedent is ``platform_repo_tenant_gate
# decision=would_refuse`` in ``tenant_binding``.
#
# The in-process counter below is the machine-readable half: it is what the tests
# assert on, and it is what a future ``/metrics`` route would export without
# having to re-derive the label shape. Label names mirror fr's
# ``fr_andon_brake_total{surface,outcome}``.
# ---------------------------------------------------------------------------
ANDON_METRIC_NAME = "coding_github_andon_brake_total"

_refusals: dict[tuple[str, str], int] = {}


def andon_refusal_counts() -> dict[tuple[str, str], int]:
    """``{(surface, outcome): count}`` for every refusal this process has made."""
    return dict(_refusals)


def reset_andon_refusal_counts() -> None:
    """Test helper. Never called by the service."""
    _refusals.clear()


def _record_refusal(operation: str, code: str, message: str) -> None:
    """Count the refusal and say so out loud. A silent brake is an unfindable one."""
    key = (operation, code)
    count = _refusals.get(key, 0) + 1
    _refusals[key] = count
    logger.warning(
        "andon_brake_refusal metric=%s surface=%s outcome=%s count=%d detail=%s",
        ANDON_METRIC_NAME,
        operation,
        code,
        count,
        message,
    )


def andon_error_http_exception(e: AndonRefusedError) -> HTTPException:
    """Map a refusal to an HTTP error whose machine code survives the wire.

    ``error_code`` is repeated inside ``details`` for the same reason
    ``tenant_binding.tenant_error_http_exception`` repeats it: coding-client's
    shared handler renders the wire ``error.code`` as ``HTTP_<status>`` and keeps
    only ``message`` + ``details``, so without this the code never reaches the
    caller — and the caller's entire job here is to distinguish "brake pulled"
    from "brake broken".
    """
    return error_response(
        status_code=e.http_status,
        message=str(e),
        error_code=e.code,
        details={"error_code": e.code, "andon": e.code, "operation": e.operation},
    )


def _andon_url(operation: str) -> str:
    settings = get_settings()
    fr_url = (getattr(settings, "fr_url", "") or "").rstrip("/")
    if not fr_url:
        # Fail CLOSED on missing configuration. An unconfigured brake is an
        # unreadable brake, and an unreadable brake refuses.
        raise AndonRefusedError(
            "andon_check_failed",
            "The andon state cannot be read: CODING_GITHUB_FR_URL is not configured on "
            "this coding-github, so a mutating GitHub operation is refused (fail closed).",
            operation,
        )
    return f"{fr_url}{ANDON_PATH}"


def _verdict(payload: Any, operation: str) -> None:
    """Return normally ONLY for a definite released state; otherwise raise.

    ⛔ Read the shape of this function, not just its branches: the final
    ``raise`` is unconditional. There is no code path that falls off the end and
    proceeds. Inverting any single check cannot turn this into fail-open, because
    an unrecognised state has to pass the whole ladder to reach a return that
    does not exist.
    """
    if not isinstance(payload, dict):
        raise AndonRefusedError(
            "andon_check_failed",
            f"fr's andon endpoint returned {type(payload).__name__}, not an object; refusing.",
            operation,
        )
    if "engaged" not in payload or "reason" not in payload:
        raise AndonRefusedError(
            "andon_check_failed",
            "fr's andon endpoint returned an unexpected shape "
            "(expected {'engaged': bool, 'reason': str}); refusing.",
            operation,
        )

    engaged = payload["engaged"]
    reason = payload["reason"]

    # A truthy non-bool (the string "false", 0, 1, None) must never be read as a
    # released cord. Only a real bool counts.
    if not isinstance(engaged, bool):
        raise AndonRefusedError(
            "andon_check_failed",
            f"fr's andon endpoint returned a non-boolean 'engaged' ({engaged!r}); refusing.",
            operation,
        )
    if not isinstance(reason, str):
        raise AndonRefusedError(
            "andon_check_failed",
            f"fr's andon endpoint returned a non-string 'reason' ({reason!r}); refusing.",
            operation,
        )

    # ENGAGED is checked first and still WINS: any definite `engaged is True`
    # refuses, and nothing below can turn that into a proceed.
    #
    # ⭐ But the CODE must be truthful about WHY. fr's own gate fail-closes a read
    # error to `{"engaged": true, "reason": "unreadable"}`, so reporting that as
    # `andon_engaged` tells the operator a human pulled the cord when in fact fr
    # could not read its own state — and `error_code` is the field callers branch
    # on, so the honest text in the message is not enough. The two diagnostic
    # reasons are therefore split out FIRST, without changing the halt.
    #
    # ⚠️ The order inside this block is load-bearing in the other direction too:
    # the fallthrough keeps `andon_engaged` for an UNRECOGNISED reason, so a
    # self-contradictory `{"engaged": true, "reason": "released"}` still refuses
    # with `andon_engaged` rather than being treated as released.
    if engaged:
        if reason == _REASON_UNREADABLE:
            raise AndonRefusedError(
                "andon_unreadable",
                "fr reports its andon state as UNREADABLE (engaged=true, fail-closed by fr's "
                f"own gate); refusing '{operation}'. Nobody pulled the cord — fr cannot read it.",
                operation,
            )
        if reason == _REASON_UNINITIALISED:
            raise AndonRefusedError(
                "andon_uninitialised",
                "fr reports its andon state as UNINITIALISED (engaged=true) — there is no cord "
                f"state to trust, so it is treated as engaged. Refusing '{operation}'.",
                operation,
            )
        raise AndonRefusedError(
            "andon_engaged",
            f"The andon cord is ENGAGED (reason: {reason}); refusing '{operation}'. "
            "Release the cord in fr before retrying.",
            operation,
        )

    if reason == _REASON_UNREADABLE:
        raise AndonRefusedError(
            "andon_unreadable",
            "fr reports its andon state as UNREADABLE. fr's own readAndon() is fail-open "
            f"for its other consumers; this caller is not. Refusing '{operation}'.",
            operation,
        )
    if reason == _REASON_UNINITIALISED:
        raise AndonRefusedError(
            "andon_uninitialised",
            "fr reports its andon state as UNINITIALISED — there is no cord state to "
            f"trust, so it is treated as engaged. Refusing '{operation}'.",
            operation,
        )
    if reason == _RELEASED_REASON:
        return  # ⭐ The ONLY return in this function.

    # `engaged is False` with a reason we do not recognise — including fr's own
    # "engaged" string, which would be self-contradictory. Refuse; do not guess.
    raise AndonRefusedError(
        "andon_check_failed",
        f"fr's andon endpoint reported engaged=false with an unrecognised reason "
        f"{reason!r}; refusing '{operation}' rather than guessing.",
        operation,
    )


async def read_andon_state(operation: str) -> None:
    """Return normally iff the cord is definitely released. Otherwise raise.

    ⛔ NO CACHE. The GET happens on every call, by design.
    """
    url = _andon_url(operation)
    try:
        async with httpx.AsyncClient(timeout=_ANDON_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers={"Accept": "application/json"})
    except AndonRefusedError:
        # ⛔ ORDERING, NOT DECORATION. This clause is what proves the terminal
        # ``except Exception`` below cannot swallow the guard's own refusal:
        # ``AndonRefusedError`` IS a subclass of ``Exception``, so the subclass
        # relation gives no protection and only clause order does. Nothing inside
        # the ``try`` raises it today (``_andon_url`` runs BEFORE the ``try``), so
        # this is unreachable as written — it is here so that it stays true if
        # anything andon-aware is ever moved inside the block.
        raise
    except httpx.HTTPError as e:
        # Covers timeouts, DNS failures, connection resets and refused
        # connections — including a mesh AuthorizationPolicy denying this pod
        # egress to fr, which is exactly the shape of failure a fail-OPEN brake
        # would sleep through.
        raise AndonRefusedError(
            "andon_unreachable",
            f"Could not reach fr's andon endpoint ({type(e).__name__}); refusing "
            f"'{operation}' (fail closed).",
            operation,
        ) from e
    except Exception as e:
        # ⭐ THE TERMINAL HANDLER, and it must stay LAST.
        #
        # httpx does not wrap everything the socket layer can throw. A raw
        # ``OSError``, an ``ssl.SSLError``, a bare ``asyncio.TimeoutError`` (e.g.
        # from an outer cancel-scope) and ``httpx.InvalidURL`` (raised by
        # ``client.get()`` for a malformed ``fr_url``, and NOT an
        # ``httpx.HTTPError``) all escaped this module before this clause existed.
        # The write still did not happen — the exception propagated and the route
        # aborted — so this was never a hole. What it WAS is an uncoded 500: the
        # caller learned nothing, which is the one outcome this whole module is
        # built to avoid.
        #
        # It is ``andon_check_failed`` rather than ``andon_unreachable`` on
        # purpose: ``andon_unreachable`` means "httpx told us the hop failed", and
        # widening it to cover anything at all would make it meaningless. This
        # bucket is "the check could not be completed", which is exactly true.
        raise AndonRefusedError(
            "andon_check_failed",
            f"The andon check itself failed with an unclassified "
            f"{type(e).__name__} ({e}); refusing '{operation}' (fail closed).",
            operation,
        ) from e

    if response.status_code != 200:
        raise AndonRefusedError(
            "andon_check_failed",
            f"fr's andon endpoint answered HTTP {response.status_code}; refusing "
            f"'{operation}' (fail closed).",
            operation,
        )
    try:
        payload: Any = response.json()
    except ValueError as e:
        raise AndonRefusedError(
            "andon_check_failed",
            f"fr's andon endpoint returned a non-JSON body; refusing '{operation}'.",
            operation,
        ) from e

    _verdict(payload, operation)


async def assert_andon_released(operation: str) -> None:
    """The brake. Raises :class:`AndonRefusedError` unless the cord is released.

    Records + logs every refusal before re-raising, so no refusal is silent.
    """
    if operation in ANDON_EXEMPT_OPERATIONS:
        # Exempt by the explicit allow-list above (empty as shipped). Logged at
        # INFO so an exemption being USED is visible, not merely declared.
        logger.info(
            "andon_brake_exempt metric=%s surface=%s outcome=exempt",
            ANDON_METRIC_NAME,
            operation,
        )
        return
    try:
        await read_andon_state(operation)
    except AndonRefusedError as e:
        _record_refusal(operation, e.code, str(e))
        raise


async def enforce_andon(operation: str) -> None:
    """:func:`assert_andon_released`, but raising the HTTP refusal.

    Used by the dependency below and by the one call site that must gate on a
    request FIELD rather than on the route (``mint_clone_token``, push scope).
    """
    try:
        await assert_andon_released(operation)
    except AndonRefusedError as e:
        raise andon_error_http_exception(e) from e


async def require_andon_released(request: Request) -> None:
    """FastAPI dependency: attach to a MUTATING route with ``dependencies=[...]``.

    Matches this service's existing cross-cutting-guard idiom —
    ``dependencies=[Depends(require_internal_token)]`` on
    ``routers/github_internal.py`` — rather than a decorator or middleware.
    Middleware was rejected on purpose: it would have to re-derive "is this route
    mutating?" from the path on every request, which is the per-caller
    enumeration this issue exists to get away from.

    ⚠️ ONE function, not a ``Depends(factory("name"))`` per route, because
    ``app.dependency_overrides`` is keyed by the callable: a factory would mint a
    distinct key per route and there would be no single handle to neutralise (or
    to assert on) in tests.

    The ``surface`` label is the HANDLER FUNCTION's name, taken from
    ``scope["endpoint"]`` (Starlette sets it; it does NOT set ``scope["route"]``).
    For most routes that equals the ``operation_id``; for two it does not —
    ``validation_webhook`` (operation_id ``trigger_validation_webhook``) and
    ``github_webhook_handler`` (operation_id ``handle_github_webhook``). The
    handler name is used because it is what a grep of ``src/`` finds.
    """
    endpoint = request.scope.get("endpoint")
    operation = getattr(endpoint, "__name__", None) or request.url.path
    await enforce_andon(operation)
