"""Credential redaction for anything derived from a git/GitHub subprocess.

⛔ WHY THIS LIVES IN ``utils`` AND NOT IN A SERVICE MODULE. There are TWO clone
paths in this service and both embed a credential in a clone URL:

* ``services.github_container_provisioner._clone_repository`` (``subprocess.run``;
  that module was dead code and was deleted on 2026-09-30 with the PAT fallbacks)
* ``services.github.repo_operations.clone_repository``
  (``asyncio.create_subprocess_exec``) — the one with a production caller, via
  ``services.github_integration_core.GitHubIntegrationHub.clone_repository``

The first one grew this helper as a private function. Importing it from the
second would have made a leaf module (``services.github.repo_operations``, which
``services.github.__init__`` imports) depend on a top-level service module that
itself imports ``services.github_validation`` — a cycle waiting to happen and
the wrong direction besides. ``utils`` imports nothing from ``services``, so
both services can import from here and neither imports the other.

⚠️ KNOWN RESIDUALS — read these before trusting this module as a boundary:

* **A TRUNCATED credential is NOT redacted.** Both passes are exact: the regex
  needs a ``…@host`` URL shape and the literal pass needs the whole secret. When
  git (or an upstream proxy) prints only a PREFIX of the token outside a URL —
  ``remote: Bad credentials (ghs_scopedContent…)`` — neither pass fires and the
  prefix survives. A prefix is not directly usable, but it is not nothing
  either. Closing this means substring matching against every window of the
  secret, which would redact innocent text; it is recorded here rather than
  implied away.
* Only the values PASSED IN as ``secrets`` are matched literally. A credential
  this process never held (e.g. one baked into a submodule URL) is caught only
  if it appears in URL userinfo.
* ``redact_exception``'s generic branch reads ``str(exc)`` only. An exception
  type that hides a credential in an attribute it does not render is not
  covered; the two subprocess types that DO (``TimeoutExpired``,
  ``CalledProcessError``) are handled explicitly.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

__all__ = ["URL_CREDENTIAL_RE", "redact_credentials", "redact_exception"]

# ``https://<userinfo>@host`` — the userinfo half of a clone URL is exactly
# where a credential ends up, and git echoes the whole URL back in its own
# error text.
URL_CREDENTIAL_RE = re.compile(r"(https?://)[^/\s@]+@", re.IGNORECASE)


def redact_credentials(text: str | None, *secrets: str | None) -> str:
    """Return ``text`` with clone-URL credentials replaced by ``***``.

    ⛔ EVERYTHING derived from a clone attempt must pass through this before it
    is logged, RETURNED TO A CALLER, or raised. ``git clone`` puts the FULL url
    — userinfo included — into its own stderr (``fatal: repository
    'https://<token>@github.com/o/r/' not found``) and ``subprocess`` puts the
    whole argv into ``TimeoutExpired``'s message, so both were writing a live
    GitHub credential into the log stream verbatim before this existed.

    Two passes, deliberately, and BOTH are load-bearing:

    * the regex catches a credential this process never saw (anything in a
      URL's userinfo, including the ``x-access-token:…`` form);
    * the literal replacement catches the token we DID embed even when git
      reported it OUTSIDE a URL — ``remote: Bad credentials for <token>`` has no
      ``@host`` for the regex to anchor on, so the regex pass alone leaks it.

    See the module docstring for the truncated-token residual.
    """
    redacted = URL_CREDENTIAL_RE.sub(r"\1***@", text or "")
    for secret in secrets:
        candidate = (secret or "").strip()
        if candidate:
            redacted = redacted.replace(candidate, "***")
    return redacted


def _redact_value(value: Any, secrets: tuple[str | None, ...]) -> Any:
    """Redact a str / bytes / list / tuple while preserving its shape.

    ``Any`` in and ``Any`` out deliberately: the values flowing through here are
    ``subprocess`` exception fields whose declared types (``_CMD``, ``str |
    bytes | None``) the shape-preserving recursion cannot express, and the
    alternative is an ``object`` return that every call site has to cast.
    """
    if isinstance(value, str):
        return redact_credentials(value, *secrets)
    if isinstance(value, bytes):
        return redact_credentials(value.decode("utf-8", "replace"), *secrets).encode("utf-8")
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, secrets) for item in value)
    return value


def redact_exception(exc: BaseException, *secrets: str | None) -> BaseException:
    """Return an exception equivalent to ``exc`` with every credential redacted.

    ⛔ WHY A BARE ``raise`` IS NOT ENOUGH. A bare ``raise`` inside an ``except``
    re-raises the ORIGINAL object, so redacting only the ``logger.error`` call
    beside it fixes the log and leaves the credential in ``str(exc)`` — where
    the next frame up logs it, or FastAPI renders it into a 500 body. Measured
    on the clone path: the log line read ``https://***@github.com/…`` while
    ``str(reraised)`` still read ``https://<real token>@github.com/…``.

    The two subprocess exception types are rebuilt field by field because
    neither renders from ``args``:

    * ``TimeoutExpired.__str__`` interpolates ``self.cmd`` — the WHOLE argv,
      auth URL included — and ``args`` is empty, so any args-based sanitiser is
      a no-op on it.
    * ``CalledProcessError`` renders ``self.cmd`` too and additionally carries
      ``.output`` / ``.stderr``, which is where git's own echo of the URL lands.

    The type is preserved for both, so a caller that branches on
    ``except subprocess.TimeoutExpired`` keeps working. For any other exception
    the same type is reconstructed from the redacted message when that type
    accepts a single argument, and only if that fails does the type change (to
    ``RuntimeError``) — losing the type is strictly better than leaking.

    Returns ``exc`` ITSELF, unchanged and identical, when there was nothing to
    redact. That is the signal for the caller to use a plain ``raise`` and keep
    the original object and its context intact.
    """
    if isinstance(exc, subprocess.CalledProcessError):
        return subprocess.CalledProcessError(
            returncode=exc.returncode,
            cmd=_redact_value(exc.cmd, secrets),
            output=_redact_value(exc.output, secrets),
            stderr=_redact_value(exc.stderr, secrets),
        )

    if isinstance(exc, subprocess.TimeoutExpired):
        return subprocess.TimeoutExpired(
            cmd=_redact_value(exc.cmd, secrets),
            timeout=exc.timeout,
            output=_redact_value(exc.output, secrets),
            stderr=_redact_value(exc.stderr, secrets),
        )

    rendered = str(exc)
    safe_text = redact_credentials(rendered, *secrets)
    if safe_text == rendered:
        # Nothing to hide — hand back the SAME object so the caller can re-raise
        # it untouched rather than swapping it for a lookalike.
        return exc

    try:
        rebuilt = type(exc)(safe_text)
    except Exception:  # noqa: BLE001 - any constructor signature may refuse this
        return RuntimeError(safe_text)

    # A type whose ``__str__`` ignores its argument would silently reintroduce
    # the credential. Verify rather than assume.
    if redact_credentials(str(rebuilt), *secrets) != str(rebuilt):
        return RuntimeError(safe_text)
    return rebuilt
