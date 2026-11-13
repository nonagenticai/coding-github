"""Commit-path guard: a user's repository receives card deliverables, never process data.

ONE function decides whether a set of paths may be written by ``apply-code-changes``:
:func:`assert_commit_paths_allowed`. It runs before ANY GitHub write on that route
(before ``_ensure_fork``, which can itself create a fork), so a refused path leaves no
artefact at all.

Three refusals, each with its own machine code (read off ``error.details.error_code``):

* ``path_invalid`` — empty, absolute (``/x``, ``C:\\x``, ``\\x``) or containing a ``..``
  segment. ``SAFE_STRING_PATTERN`` is ``^[\\s\\S]*$`` and accepts all of these.
* ``path_denied_process_data`` — matches :data:`PROCESS_DATA_DENYLIST`. ALWAYS applied,
  with or without ``target_files``: no caller can opt a path back in.
* ``path_not_in_target_files`` — the caller supplied ``target_files`` and the path is
  not in it. An EMPTY list is a real list: it admits nothing.

Matching is ``fnmatchcase`` against the LOWERCASED normalised path (``./`` and ``.``
segments removed), so ``.AA/x`` and ``X.LOG`` are refused like ``.aa/x`` and ``x.log``;
denylist entries are lowercase. ``*`` in fnmatch crosses ``/``, so ``*.log`` is any
depth. A bare root FILE named exactly like a denied root directory (``.aa``,
``.claude``, ``plans``, ``traces``, ``run-state``) is refused too. Error messages
quote the caller's original path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from fnmatch import fnmatchcase

# ⛔ The process-data denylist. Emptying or narrowing this lets conveyor state land in
# a customer repository — `test_denylist_constant_is_what_refuses` goes red if it is
# emptied. Dot-directories are denied at ANY depth (never a deliverable); the plain
# names ``plans/``, ``traces/``, ``run-state/`` only at the repo ROOT, because
# ``src/billing/plans/`` is an ordinary domain module in a user's repo.
PROCESS_DATA_DENYLIST: tuple[str, ...] = (
    ".aa/*",
    "*/.aa/*",
    ".claude/*",
    "*/.claude/*",
    "plans/*",
    "traces/*",
    "run-state/*",
    "*_probe/*",
    "*.log",
    "*.trace.json",
)

# Bare root files named exactly like a denied root directory (no slash): a file called
# ``plans`` at the root is not a deliverable either. Lowercase, compared lowercased.
PROCESS_DATA_ROOT_NAMES: frozenset[str] = frozenset(
    {".aa", ".claude", "plans", "traces", "run-state"}
)

PATH_INVALID = "path_invalid"
PATH_DENIED_PROCESS_DATA = "path_denied_process_data"
PATH_NOT_IN_TARGET_FILES = "path_not_in_target_files"

_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:)?[\\/]")


class CommitPathRefusedError(Exception):
    """A path the apply route must not write. Carries a machine ``error_code``."""

    def __init__(self, error_code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.path = path


def _normalise(path: str) -> str:
    """Return the canonical relative form, or raise ``path_invalid``."""
    if not isinstance(path, str) or not path.strip():
        raise CommitPathRefusedError(PATH_INVALID, str(path), "Empty file path")
    if path.startswith("/") or _WINDOWS_ABSOLUTE.match(path):
        raise CommitPathRefusedError(
            PATH_INVALID, path, f"Absolute path refused: {path!r}. Paths are repo-relative."
        )
    segments = path.replace("\\", "/").split("/")
    if ".." in segments:
        raise CommitPathRefusedError(PATH_INVALID, path, f"'..' segment refused: {path!r}")
    kept = [s for s in segments if s not in ("", ".")]
    if not kept:
        raise CommitPathRefusedError(PATH_INVALID, path, f"Path names no file: {path!r}")
    return "/".join(kept)


def assert_commit_paths_allowed(
    paths: Iterable[str | None], target_files: list[str] | None = None
) -> None:
    """Raise :class:`CommitPathRefusedError` for the first path that may not be written.

    ``None`` entries are skipped (an absent ``old_file_path``). ``target_files`` of
    ``None`` means "not supplied"; ``[]`` means "nothing is a target".
    """
    allowed: set[str] | None = None
    if target_files is not None:
        allowed = set()
        for target in target_files:
            try:
                allowed.add(_normalise(target))
            except CommitPathRefusedError:
                # An unusable target admits nothing; it must not widen the set.
                continue

    for raw in paths:
        if raw is None:
            continue
        path = _normalise(raw)
        folded = path.lower()
        if folded in PROCESS_DATA_ROOT_NAMES:
            raise CommitPathRefusedError(
                PATH_DENIED_PROCESS_DATA,
                raw,
                f"Path {raw!r} is process data (a root entry named {folded!r}); a "
                "user's repository receives card deliverables only.",
            )
        for pattern in PROCESS_DATA_DENYLIST:
            if fnmatchcase(folded, pattern):
                raise CommitPathRefusedError(
                    PATH_DENIED_PROCESS_DATA,
                    raw,
                    f"Path {raw!r} is process data (matches {pattern!r}); a user's "
                    "repository receives card deliverables only.",
                )
        if allowed is not None and path not in allowed:
            raise CommitPathRefusedError(
                PATH_NOT_IN_TARGET_FILES,
                raw,
                f"Path {raw!r} is not in the request's target_files.",
            )
