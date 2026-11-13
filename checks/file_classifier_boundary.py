#!/usr/bin/env python3
# ⛔ CONVEYOR FIXTURE: the fix is deliberately absent; an AA conveyor card drives at
# this defect. Do not fix it by hand. Hardening this checker, or wiring it into CI,
# is welcome and does not consume the target.
#
# MEASURED DEFECT — src/utils/file_classifier.py:169-181,
# CoreFileTypeClassifier._is_test_file. The loop over `self.test_patterns`
# tests SUBSTRING membership anywhere in the lowercased filename:
#
#     for pattern in self.test_patterns:      # {"test_", "_test.", "spec_", ...}
#         if pattern in filename_lower:
#             return True
#
# "test_" and "spec_" carry no left boundary, so any filename whose word merely
# ENDS in "...test_" or "...spec_" is misread as a test file, even though the
# match is entirely inside an unrelated word:
#
#     latest_utils.py      -> "test"  (correct: "code")  — "test_" inside "latest_"
#     contest_results.py   -> "test"  (correct: "code")  — "test_" inside "contest_"
#     protest_handler.py   -> "test"  (correct: "code")  — "test_" inside "protest_"
#
# Called from 4 real sites: git_diff_analyzer.py, documentation_checker.py,
# pr_change_detector.py, test_coverage_correlator.py.
#
# This is a BEHAVIOURAL check: it imports and calls the real
# CoreFileTypeClassifier.get_file_type(), never re-implements the pattern
# matching. ⚠️ CORRECTED: an earlier version of this file claimed "a fix that
# only special-cases the three names above does not pass" — that was FALSE.
# An adversarial reviewer wrote exactly that hack (hardcode the three
# literals, no boundary logic at all) and it went fully green. The three
# named filenames are ENUMERABLE from the bug report above, so hardcoding
# them defeats a check that only tests those three.
#
# The fix: MUST_BE_CODE now also carries (a) held-out cases using different
# words than the bug report, and (b) cases generated at runtime from the
# classifier's own `test_patterns` set (see _generate_boundary_defect_cases
# below) — gluing extra letters onto whichever end of each pattern lacks a
# separator character, which is exactly the end the substring-membership bug
# leaves unchecked. Because (b) is derived from the code under test rather
# than typed in advance, it cannot be defeated by enumerating literals from
# the bug report.
#
# ⚠️ CORRECTED AGAIN: (b) used a FIXED glue literal ("q9z"), visible in this
# checked-in source, which the automated fixer can read as context. A
# reviewer defeated it with `if "q9z" in filename_lower: return False` bolted
# onto the same literal-enumeration hack above — no boundary logic at all,
# clean exit 0, while latest_utils.py/contest_results.py/etc. (real call
# sites, not test fixtures) stayed misclassified. Any fixed token in this
# file is a hardcodable key, full stop.
#
# The fix for THAT: the glue is generated fresh at runtime (see
# _generate_glue) — a short random lowercase-alphanumeric string, different
# every invocation, validated to contain no substring of the classifier's
# own test_patterns/test_extensions so it can never accidentally spell a
# genuine test-file convention. There is no constant left to special-case;
# a hardcoded "q9z" check (or any other single literal) now fails on the
# next run. This still cannot prove no enumeration whatsoever could survive
# every run — see the honesty note in the module docstring below for what
# actually remains open.

r"""Oracle for the file_classifier test/code boundary misclassification.

Run from the repository root:

    uv run python -m checks.file_classifier_boundary

Exit codes:
    0 — every case classified correctly (the defect is fixed, no regression,
        and the held-out / generated cases below also pass, against a
        boundary glue that is regenerated at random THIS run). Neither the
        three literals from the bug report, nor a hardcoded check for any
        previously-seen glue value, will reach 0.
    1 — at least one case still misclassified (expected on today's main).
    2 — the harness itself is broken (import failure, or the glue generator
        could not find a safe token); no verdict can be drawn from this run.
        "This check measured nothing; do not read it as a pass."

⚠️ HONEST LIMIT (do not overclaim a third time): randomizing the glue closes
the "hardcode the fixed literal" hole, and only that hole. It is a black-box
BEHAVIOURAL check — it calls the real classifier and reads its return value.
It has no defence against a classifier that inspects *how* it is being
called rather than *what* it is being asked: e.g. walking the call stack
(`inspect.stack()`/`sys._getframe`) to detect that the caller is this
module, and branching to correct-looking logic only then. That is a generic
limit of any behavioural oracle running in its own identifiable process,
not something per-run data randomization can close. See the worktree notes
for a reproduced example; no attempt is made here to defend against it.
"""

from __future__ import annotations

import random
import string
import sys

SITE = "src/utils/file_classifier.py:169-181 (CoreFileTypeClassifier._is_test_file)"

# Separator characters that, when they anchor an end of a test_patterns
# literal, make that end of the pattern a real word boundary (a directory
# segment bounded by "/", or a "_"/"." delimited token). An end that is NOT
# one of these is the end the substring-membership bug leaves unchecked.
_SEPARATOR_CHARS = ("_", "/", ".")

# Runtime glue generation: a short lowercase-alphanumeric token, freshly
# random on every invocation, used to glue extra letters onto an unbounded
# pattern end (see _generate_boundary_defect_cases). There is deliberately
# no fixed literal here for a fixer to read and special-case — the previous
# version's "_GLUE = 'q9z'" constant was exactly that, and an adversarial
# fix defeated it with `if "q9z" in filename_lower: return False` bolted
# onto the literal-enumeration hack, while the real defect (real call sites
# outside this oracle's fixture names) stayed broken.
_GLUE_LENGTH = 6
_GLUE_ALPHABET = string.ascii_lowercase + string.digits
_MAX_GLUE_ATTEMPTS = 200


def _generate_glue(needles: frozenset[str]) -> str:
    """Return a fresh random lowercase-alphanumeric token, regenerating
    until it is safe to glue onto any pattern end.

    "Safe" means: no `needle` (a live test_patterns or test_extensions
    entry) appears as a substring of the candidate, in either direction.
    Without this check a random draw could — extremely rarely, but not
    with probability zero — itself spell "test" or "spec" or a whole
    pattern, which would smuggle a genuine test-file convention into a case
    this oracle expects to classify as "code" (inverting the expected
    answer rather than testing it). Raises if no safe candidate turns up in
    `_MAX_GLUE_ATTEMPTS` tries, which the caller must treat as a broken
    harness (exit 2), never as a pass.
    """
    for _ in range(_MAX_GLUE_ATTEMPTS):
        candidate = "".join(random.choice(_GLUE_ALPHABET) for _ in range(_GLUE_LENGTH))
        if any(needle in candidate or candidate in needle for needle in needles):
            continue
        return candidate
    raise RuntimeError(
        f"could not draw a glue token safe against {sorted(needles)!r} "
        f"after {_MAX_GLUE_ATTEMPTS} attempts"
    )


# Snapshots of the patterns/extensions as they exist on today's main, used
# ONLY as a fallback if a fix changes `test_patterns`/`test_extensions` into
# something this generator cannot introspect (e.g. a different attribute
# name or a compiled-regex list) — so a legitimate refactor can't turn into
# an import/attribute crash. When the live attribute IS readable, it is
# used instead (see _classifier_test_patterns / _classifier_test_extensions),
# which is what keeps this check derived from the code under test.
_FALLBACK_TEST_PATTERNS = frozenset(
    {
        "test_",
        "_test.",
        "test/",
        "/test/",
        "/tests/",
        "spec_",
        "_spec.",
        "spec/",
        "/spec/",
        "/specs/",
        "integration_test",
        "unit_test",
        "e2e_test",
    }
)

_FALLBACK_TEST_EXTENSIONS = frozenset(
    {
        "test.py",
        "test.js",
        "test.ts",
        "test.jsx",
        "test.tsx",
        "spec.py",
        "spec.js",
        "spec.ts",
        "spec.jsx",
        "spec.tsx",
    }
)

# ---------------------------------------------------------------------------
# MUST_BE_TEST — genuine test files. These already classify correctly as
# "test" on today's main: they are POSITIVE CONTROLS held in the same run as
# the defect cases, so a broken harness (everything reads "code") is
# distinguishable from the real defect (only specific words misread).
#
# They double as the OVER-CORRECTION GUARD: a fix that anchors so hard it
# starts rejecting real test files must fail these too, and is reported as a
# regression rather than folded into the defect count.
# ---------------------------------------------------------------------------
MUST_BE_TEST: tuple[tuple[str, str], ...] = (
    ("prefix pattern", "test_foo.py"),
    ("suffix pattern", "foo_test.py"),
    ("spec prefix", "spec_helper.py"),
    ("spec suffix (extension)", "my_module_spec.py"),
    ("tests directory segment", "src/tests/helper.py"),
    ("integration_test compound", "integration_test_login.py"),
    ("unit_test compound", "unit_test_math.py"),
    ("e2e_test compound", "e2e_test_flow.py"),
    # "_test." is bounded on BOTH ends (leading "_", trailing "."), so this
    # is the same real "<word>_test.py" suffix convention as "foo_test.py"
    # above, not a boundary-bug case — checked by executing a genuine
    # boundary-aware reference fix against it before asserting this: the
    # token "test" appears standing alone between "_" and ".", so it is
    # correctly a test file regardless of what "manifest" is.
    ("_test. suffix, unrelated stem", "manifest_test.py"),
)

# ---------------------------------------------------------------------------
# MUST_BE_CODE — the measured defect, plus held-out cases.
#
# The first three are the exact filenames from the bug report (kept as a
# baseline regression check). The rest are NOT named in the bug report:
# "held-out" hand-picked words using the same "test_" boundary bug, and a
# batch generated straight from the classifier's own `test_patterns` set at
# import time (see _generate_boundary_defect_cases) so the case set is
# derived from the code under test, not enumerable in advance.
# ---------------------------------------------------------------------------
_NAMED_IN_BUG_REPORT: tuple[tuple[str, str], ...] = (
    ("'test_' glued onto 'la-' (bug report)", "latest_utils.py"),
    ("'test_' glued onto 'con-' (bug report)", "contest_results.py"),
    ("'test_' glued onto 'pro-' (bug report)", "protest_handler.py"),
)

_HELD_OUT_HANDPICKED: tuple[tuple[str, str], ...] = (
    ("'test_' in the middle of a word (held out)", "unittest_helper.py"),
    ("'test_' at the end of a word (held out)", "fastest_route.py"),
)


def _classifier_test_patterns(classifier: object) -> frozenset[str]:
    """Read `test_patterns` off the live classifier when possible, so the
    generated cases below track whatever a fix actually ships. Falls back to
    a frozen snapshot only if the attribute is missing or not a plain set of
    strings, so a legitimate refactor of the internal representation can't
    turn this oracle into an AttributeError."""
    patterns = getattr(classifier, "test_patterns", None)
    if isinstance(patterns, (set, frozenset)) and all(isinstance(p, str) and p for p in patterns):
        return frozenset(patterns)
    return _FALLBACK_TEST_PATTERNS


def _classifier_test_extensions(classifier: object) -> frozenset[str]:
    """Read `test_extensions` off the live classifier when possible, for
    the same reason and with the same fallback discipline as
    `_classifier_test_patterns` above — used only to validate a candidate
    glue token, never to build a case."""
    extensions = getattr(classifier, "test_extensions", None)
    if isinstance(extensions, (set, frozenset)) and all(
        isinstance(e, str) and e for e in extensions
    ):
        return frozenset(extensions)
    return _FALLBACK_TEST_EXTENSIONS


def _generate_boundary_defect_cases(
    patterns: frozenset[str],
    glue: str,
) -> tuple[tuple[str, str], ...]:
    """For each pattern, glue synthetic letters onto whichever end lacks a
    separator character in the pattern's own literal text — precisely the
    end substring-membership matching leaves unchecked.

    `glue` is generated fresh per run by `_generate_glue` (see above) — a
    random token, not a fixed literal, and validated before this function
    ever sees it.

    A pattern bounded by a separator on BOTH ends (e.g. "_test.", "/test/",
    "/tests/") encodes a real filename convention — a "<word>_test.py"
    suffix, or a directory literally named "test" — and is left alone; only
    a genuinely unbounded end produces a case, and every case built here
    expects "code" because the glue text turns the matched word into
    something that is not "test" or "spec" on its own — true for ANY safe
    glue value, which is what makes this deterministic in outcome despite
    being random in input (see module-level honesty note on what "random"
    does and does not defend against).

    For a pattern unbounded on BOTH ends (e.g. "integration_test", which has
    no separator on either side of its literal text), only the right end is
    glued: right-glue is unambiguous (the trailing token becomes a different
    word, e.g. "test<glue>" != "test"), while left-gluing a multi-word
    compound is the same kind of question "manifest_test.py" raises for
    "_test." — genuinely ambiguous — and is deliberately not asserted here
    rather than risk a wrong expectation.
    """
    cases: list[tuple[str, str]] = []
    for pattern in sorted(patterns):
        left_open = pattern[0] not in _SEPARATOR_CHARS
        right_open = pattern[-1] not in _SEPARATOR_CHARS
        if left_open and not right_open:
            filename = f"{glue}{pattern}module.py"
            cases.append((f"generated left-glue on {pattern!r}", filename))
        elif right_open:
            # Covers both "right-open only" and "open on both ends".
            filename = f"{pattern}{glue}.py"
            cases.append((f"generated right-glue on {pattern!r}", filename))
        # else: bounded on both ends -> real convention, no case generated.
    return tuple(cases)


def main() -> int:
    try:
        from src.utils.file_classifier import CoreFileTypeClassifier
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        print(f"FAIL: could not import CoreFileTypeClassifier: {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    try:
        classifier = CoreFileTypeClassifier()
    except Exception as exc:
        print(f"FAIL: CoreFileTypeClassifier() raised {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    patterns = _classifier_test_patterns(classifier)
    extensions = _classifier_test_extensions(classifier)

    try:
        glue = _generate_glue(patterns | extensions)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        print("This check measured nothing; do not read it as a pass.")
        return 2
    print(f"glue used this run: {glue!r}")

    generated_cases = _generate_boundary_defect_cases(patterns, glue)
    must_be_code = _NAMED_IN_BUG_REPORT + _HELD_OUT_HANDPICKED + generated_cases

    # Held-out / generated filenames must not collide with the bug-report
    # literals or each other -- otherwise "checked" double-counts a name.
    seen: set[str] = set()
    for _, filename in must_be_code:
        if filename in seen:
            print(f"FAIL: duplicate MUST_BE_CODE filename {filename!r}")
            return 2
        seen.add(filename)

    regressions: list[tuple[str, str, str]] = []
    defects: list[tuple[str, str, str]] = []

    for label, filename in MUST_BE_TEST:
        try:
            got = classifier.get_file_type(filename)
        except Exception as exc:
            print(f"FAIL: get_file_type({filename!r}) raised {exc!r}")
            return 2
        if got != "test":
            regressions.append((label, filename, got))

    for label, filename in must_be_code:
        try:
            got = classifier.get_file_type(filename)
        except Exception as exc:
            print(f"FAIL: get_file_type({filename!r}) raised {exc!r}")
            return 2
        if got != "code":
            defects.append((label, filename, got))

    checked = len(MUST_BE_TEST) + len(must_be_code)
    print(
        f"filenames checked: {checked} "
        f"({len(must_be_code)} MUST_BE_CODE, of which {len(generated_cases)} "
        "generated from the classifier's own test_patterns)"
    )

    if regressions or defects:
        if defects:
            print(f"\nFAIL: {len(defects)} file(s) still MISCLASSIFIED (the live defect):")
            for label, filename, got in defects:
                print(f"  [{label}] {filename!r} -> {got!r}, expected 'code'")
            print(
                f"\n{SITE}\n"
                "tests substring membership anywhere in the filename; a pattern "
                "with no separator on one end has no boundary on that side, so a "
                "word that merely glues onto that end is misread."
            )
        if regressions:
            print(
                f"\nFAIL: {len(regressions)} genuine test file(s) were REJECTED "
                "(over-correction regression):"
            )
            for label, filename, got in regressions:
                print(f"  [{label}] {filename!r} -> {got!r}, expected 'test'")
            print(
                "\nThe fix has been anchored too aggressively and now rejects real "
                "test files; this is a regression, not progress on the defect."
            )
        return 1

    print(f"\nok: all {checked} filename(s) classified correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
