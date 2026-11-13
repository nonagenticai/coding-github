#!/usr/bin/env python3
# ⛔ CONVEYOR FIXTURE: the fix is deliberately absent; an AA conveyor card drives at
# this defect. Do not fix it by hand. Hardening this checker, or wiring it into CI,
# is welcome and does not consume the target.
#
# MEASURED DEFECT — src/services/pr_validation/core/diff_parser.py:157-165.
# The rename/copy branch is entered by PREFIX (line 158 uses str.startswith) but then
# discriminates by SUBSTRING membership: `"rename" in line` (:160) and `"from" in line`
# (:162). A substring test reads the whole line, filenames included, so the FILENAME
# votes on the classification: `copy to renamed_utils.py` contains "rename" and is
# classified RENAMED, and `rename to from_utils.py` contains "from" and overwrites
# old_filename instead of new_filename. The same substring confusion is why a genuine
# `copy from`/`copy to` pair falls through assigning nothing at all, leaving the copy
# MODIFIED and making `self.copy_pattern` (:77) dead code. Root cause: only the first
# two tokens of a git extended-header line are grammar; everything after them is data.

"""Oracle for the diff_parser rename/copy misclassification.

Run from the repository root:

    python -m checks.diff_parser_rename_copy

Exit codes:
    0 — every case classified correctly (the defect is fixed).
    1 — at least one defect case still misclassified (expected on today's main).
    2 — the harness itself is broken: the positive control failed, so no verdict
        about the defect can be drawn from this run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
# pyproject sets pythonpath = ["src"] for pytest only; a bare module run gets nothing.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coding_client.models.pr_validation import ChangeType  # noqa: E402

from services.pr_validation.core.diff_parser import DiffParser  # noqa: E402

SITE = "src/services/pr_validation/core/diff_parser.py:157-165"

# `ChangeType` is `StaticAnalysisChangeType` re-exported from the external
# `coding-client` package (measured at 1.19.1: ADDED, DELETED, MODIFIED, RENAMED —
# there is NO COPIED member). A fix confined to this repo therefore cannot mint
# ChangeType.COPIED, so a copy is accepted as COPIED when the enum grows one and as
# ADDED otherwise (a copy does create a new file). It is never MODIFIED, never RENAMED.
ACCEPTABLE_COPY_TYPES = {"COPIED", "ADDED"}

# ---------------------------------------------------------------------------
# Fixtures. Shape verified against the parser: parse_git_diff() keys every file on a
# `diff --git a/X b/Y` header (_parse_diff_header, :330) and only the branch at :158
# can ever set ChangeType.RENAMED — the header alone yields MODIFIED. So the positive
# control's RENAMED verdict is itself the proof that these fixtures reach the branch
# under test, and a fixture too malformed to get there cannot read as green.
# ---------------------------------------------------------------------------

CONTROL_RENAME = """diff --git a/a.py b/b.py
similarity index 100%
rename from a.py
rename to b.py
"""

# Bug 1: `copy to renamed_utils.py` — the filename contains "rename".
COPY_WITH_RENAME_IN_NAME = """diff --git a/src_utils.py b/renamed_utils.py
similarity index 100%
copy from src_utils.py
copy to renamed_utils.py
"""

# Bug 2: `rename to from_utils.py` — the filename contains "from".
RENAME_WITH_FROM_IN_NAME = """diff --git a/old_utils.py b/from_utils.py
similarity index 96%
rename from old_utils.py
rename to from_utils.py
index 1111111..2222222 100644
--- a/old_utils.py
+++ b/from_utils.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""

# Bug 3: an unambiguous copy pair — no token anywhere contains "rename".
PLAIN_COPY = """diff --git a/orig.py b/dup.py
similarity index 100%
copy from orig.py
copy to dup.py
"""


def _parse_one(name: str, diff_text: str):
    """Parse a single-file diff, failing loudly rather than vacuously."""
    parsed = DiffParser().parse_git_diff(diff_text)
    if len(parsed) != 1:
        raise RuntimeError(
            f"fixture {name!r} produced {len(parsed)} ParsedDiff objects, expected exactly 1; "
            "the fixture never reached the parser branch under test"
        )
    return parsed[0]


def positive_control() -> list[str]:
    """A plain rename must be RENAMED with the right old/new filenames.

    If this fails the harness is broken — the fixtures no longer reach
    diff_parser.py:158 at all — and nothing can be concluded about the defect.
    """
    problems: list[str] = []
    d = _parse_one("CONTROL_RENAME", CONTROL_RENAME)
    if d.change_type != ChangeType.RENAMED:
        problems.append(
            f"control: change_type expected RENAMED, got {d.change_type.name} "
            f"(only {SITE} can set RENAMED, so the branch was never reached)"
        )
    if d.old_filename != "a.py":
        problems.append(f"control: old_filename expected 'a.py', got {d.old_filename!r}")
    if d.new_filename != "b.py":
        problems.append(f"control: new_filename expected 'b.py', got {d.new_filename!r}")
    return problems


def assert_copy_is_not_renamed_by_its_filename() -> list[str]:
    """Bug 1 — `copy to renamed_utils.py` must not be classified RENAMED."""
    problems: list[str] = []
    d = _parse_one("COPY_WITH_RENAME_IN_NAME", COPY_WITH_RENAME_IN_NAME)
    if d.change_type.name not in ACCEPTABLE_COPY_TYPES:
        problems.append(
            "copy_misread_as_rename: a `copy from`/`copy to` pair whose destination is "
            "'renamed_utils.py' — expected change_type in "
            f"{sorted(ACCEPTABLE_COPY_TYPES)}, got {d.change_type.name}. "
            f"{SITE}: `\"rename\" in line` matches the substring 'rename' inside the "
            "FILENAME, so the copy is classified as a rename. Must test the line PREFIX "
            "('rename from'/'rename to'), not substring membership."
        )
    if d.old_filename != "src_utils.py":
        problems.append(
            "copy_misread_as_rename: old_filename expected 'src_utils.py', "
            f"got {d.old_filename!r} ({SITE})"
        )
    if d.new_filename != "renamed_utils.py":
        problems.append(
            "copy_misread_as_rename: new_filename expected 'renamed_utils.py', "
            f"got {d.new_filename!r} ({SITE})"
        )
    return problems


def assert_rename_to_writes_new_filename() -> list[str]:
    """Bug 2 — `rename to from_utils.py` must set new_filename, not old_filename."""
    problems: list[str] = []
    d = _parse_one("RENAME_WITH_FROM_IN_NAME", RENAME_WITH_FROM_IN_NAME)
    if d.change_type != ChangeType.RENAMED:
        problems.append(
            "rename_to_with_from_in_name: change_type expected RENAMED, "
            f"got {d.change_type.name} ({SITE})"
        )
    if d.old_filename != "old_utils.py":
        problems.append(
            "rename_to_with_from_in_name: old_filename expected 'old_utils.py', got "
            f"{d.old_filename!r}. {SITE}: on the line `rename to from_utils.py`, "
            "`\"from\" in line` matches the substring 'from' inside the FILENAME, so the "
            "'to' branch writes old_filename and destroys the value set by `rename from`. "
            "Must branch on the line PREFIX 'rename from' vs 'rename to'."
        )
    if d.new_filename != "from_utils.py":
        problems.append(
            "rename_to_with_from_in_name: new_filename expected 'from_utils.py', "
            f"got {d.new_filename!r} ({SITE})"
        )
    return problems


def assert_plain_copy_is_classified() -> list[str]:
    """Bug 3 — a genuine `copy from`/`copy to` pair must be classified at all."""
    problems: list[str] = []
    d = _parse_one("PLAIN_COPY", PLAIN_COPY)
    if d.change_type.name not in ACCEPTABLE_COPY_TYPES:
        problems.append(
            "plain_copy_unclassified: an unambiguous `copy from orig.py` / `copy to dup.py` "
            f"pair — expected change_type in {sorted(ACCEPTABLE_COPY_TYPES)}, got "
            f"{d.change_type.name}. {SITE}: the branch is entered but the body only handles "
            '`if "rename" in line`, so a copy assigns NOTHING and keeps the header\'s '
            "default; `self.copy_pattern` (diff_parser.py:77) is dead code."
        )
    if d.old_filename != "orig.py":
        problems.append(
            f"plain_copy_unclassified: old_filename expected 'orig.py', got {d.old_filename!r} ({SITE})"
        )
    if d.new_filename != "dup.py":
        problems.append(
            f"plain_copy_unclassified: new_filename expected 'dup.py', got {d.new_filename!r} ({SITE})"
        )
    return problems


DEFECT_CASES = (
    assert_copy_is_not_renamed_by_its_filename,
    assert_rename_to_writes_new_filename,
    assert_plain_copy_is_classified,
)


def main() -> int:
    try:
        control_problems = positive_control()
    except Exception as exc:  # noqa: BLE001 — a crash here is a harness failure too
        print(
            f"harness broken: positive control raised {type(exc).__name__}: {exc}", file=sys.stderr
        )
        print(
            "harness broken: no verdict about the defect can be drawn from this run.",
            file=sys.stderr,
        )
        return 2
    if control_problems:
        for problem in control_problems:
            print(f"HARNESS: {problem}", file=sys.stderr)
        print(
            "harness broken: the positive control (a plain rename must parse as RENAMED) "
            "failed, so the fixtures never reach the code under test. This is NOT the "
            "defect firing — fix the harness before reading any verdict.",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for case in DEFECT_CASES:
        try:
            failures.extend(case())
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case.__name__}: raised {type(exc).__name__}: {exc}")

    total_cases = 1 + len(DEFECT_CASES)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            f"\n{len(failures)} assertion(s) failed across {len(DEFECT_CASES)} defect case(s); "
            "positive control PASSED, so these are real misclassifications.",
            file=sys.stderr,
        )
        return 1

    print(f"ok: all {total_cases} case(s) classified correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
