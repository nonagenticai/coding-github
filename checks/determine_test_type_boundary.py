#!/usr/bin/env python3
# ⛔ CONVEYOR FIXTURE: the fix is deliberately absent; an AA conveyor card drives at
# this defect. Do not fix it by hand. Hardening this checker, or wiring it into CI,
# is welcome and does not consume the target.
#
# MEASURED DEFECT — src/services/pr_validation/analyzers/impact/test_coverage_correlator.py
# :264-273, TestCoverageCorrelator._determine_test_type. On main @076b7e8:
#
#   def _determine_test_type(self, filename: str) -> str:
#       filename_lower = filename.lower()
#       if "unit" in filename_lower or "test_" in filename_lower:   # bare substring
#           return "unit"
#       elif "integration" in filename_lower or "e2e" in filename_lower:
#           return "integration"
#       ...
#
# Both literals in the FIRST branch are matched with bare `in`, so a filename that
# merely CONTAINS them inside an unrelated word is classified "unit" and the
# integration branch below is never reached. MEASURED, 5/5 on main:
#
#     latest_config.py              -> "unit"   ("test_" inside "latest_")   [should be "unknown"]
#     opportunity_check.py          -> "unit"   ("unit"  inside "opportunity") [should be "unknown"]
#     latest_integration.py         -> "unit"   ("test_" inside "latest_")
#     contest_e2e_flow.py           -> "unit"   ("test_" inside "contest_")
#     protest_integration_check.py  -> "unit"   ("test_" inside "protest_")
#     opportunity_integration.py    -> "unit"   ("unit"  inside "opportunity")
#     greatest_e2e.py               -> "unit"   ("test_" inside "greatest_")
#
# ⭐ NOTE THE SECOND CARRIER. Earlier analysis named only `test_`. The literal
# `unit` is a carrier too — `"unit" in "opportunity"` is True — so a fix that
# guards ONLY `test_` leaves the defect live. `opportunity_integration.py` is the
# case that discriminates a complete fix from a partial one, and it is the reason
# this oracle refuses to pass on a one-literal repair.
#
# This is the same defect CLASS already fixed twice in src/utils/file_classifier.py
# (cg#97, cg#99) via CoreFileTypeClassifier._matches_pattern_on_word_boundary.
# This is a SEPARATE inline reimplementation in a different module — not a call
# site of the fixed classifier — so those fixes did not reach it.
#
# ⚠️ The correlator ALREADY holds the helper: __init__ sets
# `self.file_classifier = CoreFileTypeClassifier()`, whose
# `_matches_pattern_on_word_boundary` is the existing, tested remedy.
#
# REACHED FROM REAL CODE, not only from tests:
#   test_coverage_correlator.py:244 — `test_type=self._determine_test_type(...)`
#
# ⛔ SECOND SURVIVED ATTACK: a component-PREFIX rule (base.startswith("test_") /
# filename.startswith("unit/")) passed an earlier version 9/9 and 7/7 while
# regressing foo_unit_test.py, api_unit_test.py and tests/payment_unit_test.py from
# "unit" to "unknown" -- all three of which DO reach this method (_is_test_file is
# True for each). Cause: every positive control carried its marker at a path-component
# START, so the control SET was satisfiable without any boundary rule. Fixed by the
# two mid-filename controls in POSITIVE_CONTROLS.
# ⛔ THIRD SURVIVED ATTACK: boundary-matching ONLY branch 1 (`unit`/`test_`) and
# leaving branches 2-3 (`integration`/`e2e`, `spec`) as bare `in` scored 9/9 and 9/9,
# exit 0 -- nothing pinned those branches. And a prefix rule with ONE extra literal
# (`or "_unit_" in filename`) defeated the two mid-filename controls while regressing
# `unit_test.py` itself. Closed by Group 3 and by the `unit_test.py` control.
# ⚠️ Three independent reviewers found three DIFFERENT routes to a false green on this
# one file. Treat any future "simplification" of these case lists as reopening one.
# ⭐ THE GENERAL LAW, learned THREE TIMES on this one file: a case that can pass via two
#   sufficient signals tests the DISJUNCTION, not the property you meant -- and
#   mutation controls do NOT catch it, because deleting the fix still reds on a case
#   that was never isolating the property.
#
# ⛔ ATTACK THIS ORACLE ALREADY SURVIVES (found by an adversarial pre-push review,
# which is why Group 2 below exists): reordering the branches so integration/e2e is
# tested FIRST, with every bare `in` left intact, made an earlier version of this
# oracle report 5/5 and 7/7 and exit 0 while the defect was completely live.
# ⇒ Do not "simplify" DEFECT_CASES. Group 2 distinguishes a fix from a REORDERING;
#   Group 3 distinguishes a WHOLE fix from one that repairs only the first branch.
#
# ⚠️ NO EXISTING TEST LOCKS IN THE BUG. tests/unit/test_test_coverage_correlator_extended.py
# covers this method with 5 cases, all non-colliding filenames. So this is a bug
# fix, not a spec change — see the docstring.

r"""Oracle for the _determine_test_type filename-boundary misclassification.

Run from the repository root:

    uv run python -m checks.determine_test_type_boundary

Exit codes:
    0 - every case classified correctly: BOTH carriers (`test_` and `unit`) now
        match on a word boundary AND every positive control still classifies.
    1 - at least one case misclassified (expected on today's main), OR a positive
        control was broken by an over-correction. The two are reported
        SEPARATELY and never merged into one count -- a fix that repairs the
        defect by breaking legitimate classification is not a fix.
    2 - the harness measured nothing (import failure, or the method is absent).
        "This check measured nothing; do not read it as a pass."

THE VERDICT ALWAYS COMES FROM CALLING THE REAL METHOD on a real
TestCoverageCorrelator instance. This oracle never reimplements the
classification logic, so it cannot go green against a reimplementation that
diverges from shipped behaviour.
"""

import random
import sys

EXIT_FIXED, EXIT_DEFECT, EXIT_NO_MEASUREMENT = 0, 1, 2

# (filename, expected) -- each collides a first-branch literal with a real
# integration marker. A correct boundary match yields "integration".
DEFECT_CASES = [
    # --- Group 1: collision PLUS a genuine integration marker. ---
    ("latest_integration.py", "integration"),
    ("contest_e2e_flow.py", "integration"),
    ("protest_integration_check.py", "integration"),
    ("opportunity_integration.py", "integration"),  # `unit` carrier -- see header
    ("greatest_e2e.py", "integration"),
    # --- Group 2: THE COLLISION IS THE ONLY SIGNAL. These are what make the
    # oracle mean something. Group 1 alone can be satisfied by REORDERING the
    # branches while leaving every bare `in` intact -- the integration marker is
    # then reached by ordering, not by boundary matching, and the defect ships.
    # Here there is no integration marker at all, so the ONLY way to return
    # "unknown" is for the first branch to stop matching mid-word.
    #   pristine main -> "unit" (red) | reorder-only -> "unit" (STILL red)
    #   real boundary fix -> "unknown" (green)
    ("latest_config.py", "unknown"),  # "test_" inside "latest_"
    ("contest_helper.py", "unknown"),  # "test_" inside "contest_"
    ("opportunity_check.py", "unknown"),  # "unit"  inside "opportunity"
    ("unittest_runner.py", "unknown"),  # "unit"/"test_" glued mid-word
    ("latest_test.py", "unknown"),  # reachable: _is_test_file is True for this
    # --- Group 3: BRANCHES 2 AND 3 are bare `in` too, and nothing else pins them.
    # Boundary-matching only branch 1 scores a full 9/9 + 9/9 and exit 0 while
    # shipping the identical defect here. Both reach the method (_is_test_file True).
    ("reintegration_helper_test.py", "unknown"),  # "integration" inside "reintegration"
    ("specialist_test.py", "unknown"),  # "spec" inside "specialist" -- branch 3
    ("e2etest_flow_test.py", "unknown"),  # "e2e" glued right -- branch 2's SECOND literal
]

# Must KEEP working. These catch an over-correction that fixes the collisions by
# breaking ordinary classification.
POSITIVE_CONTROLS = [
    # ⛔ MID-FILENAME controls. Every other control below carries its marker at a
    # PATH-COMPONENT START ("test_foo.py", "unit/...", "integration/..."), so the
    # set as a whole was satisfiable by a component-PREFIX rule -- which is not a
    # boundary rule. A second adversarial review passed the entire oracle 9/9 7/7
    # with `base.startswith("test_") or filename.startswith("unit/")`, while
    # regressing foo_unit_test.py from "unit" to "unknown". These two cases carry
    # the marker separator-bounded MID-name, so only a real boundary match
    # satisfies them. They are load-bearing -- do not "tidy" them away.
    ("unit_test.py", "unit"),  # marker at string START, bounded only on the RIGHT --
    # the one shape no other control covers. `"_unit_" in filename` added to a prefix
    # rule defeated the two cases below; this one refuses that too. "unit_test" is in
    # the classifier's own test_patterns, so it is the most canonical name here.
    ("foo_unit_test.py", "unit"),
    ("checkout_e2e_test.py", "integration"),
    ("test_foo.py", "unit"),
    ("unit/test_helpers.py", "unit"),
    ("integration_check.py", "integration"),
    ("integration/api_check.py", "integration"),
    ("e2e/flow_check.py", "integration"),
    ("test.spec.js", "unit"),
    ("random_file.py", "unknown"),
]


# --- GENERATED CASES: the hand-picked lists above cannot close this ---------
#
# ⛔ WHY THIS SECTION EXISTS. Four adversarial reviews found five DIFFERENT
# non-fixes that scored a perfect hand-picked sweep and exited 0. That was not
# bad luck, it is arithmetic: the obligation is 5 literals x their non-separator
# ends = 9 ANCHOR SLOTS, and each hand-written round closed one or two while
# leaving others open, so every round leaked a new one. Enumeration cannot win a
# race against a rule tuned to the enumeration.
# ⇒ This sweep closes ALL NINE at once, and also defeats a denylist that simply
#   strips the known-bad prefixes, because the tokens are drawn fresh each run.
# Same technique as checks/file_classifier_filename_boundary.py in this directory.

_PATTERNS = ("unit", "test_", "integration", "e2e", "spec")
_SEPARATORS = "_/."
_GLUE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_GLUE_LEN = (4, 7)
_MAX_GLUE_ATTEMPTS = 200


def _safe_glue() -> str:
    """A fresh lowercase token containing no pattern literal, and contained in none.

    Without this a random draw could itself spell "spec" or "unit", which would
    make the ORACLE wrong rather than strict. Raises -> exit 2, never a pass.
    """
    for _ in range(_MAX_GLUE_ATTEMPTS):
        cand = "".join(random.choice(_GLUE_ALPHABET) for _ in range(random.randint(*_GLUE_LEN)))
        if any(n in cand or cand in n for n in _PATTERNS):
            continue
        return cand
    raise RuntimeError(f"no safe glue token in {_MAX_GLUE_ATTEMPTS} attempts")


def _boundary_match(pattern: str, filename: str) -> bool:
    """True if `pattern` occurs in `filename` on a word boundary. Used ONLY to
    decide which generated cases are FAIR to ask -- never to grade one."""
    left_bounded, right_bounded = pattern[0] in _SEPARATORS, pattern[-1] in _SEPARATORS
    i = filename.find(pattern)
    while i != -1:
        before_ok = left_bounded or i == 0 or not filename[i - 1].isalnum()
        end = i + len(pattern)
        after_ok = right_bounded or end == len(filename) or not filename[end].isalnum()
        if before_ok and after_ok:
            return True
        i = filename.find(pattern, i + 1)
    return False


def generate_boundary_cases() -> list[tuple[str, str, str]]:
    """(filename, expected, slot) for every (pattern, gluable end) slot.

    Each filename carries its pattern ONLY mid-word, and carries no OTHER
    pattern on a boundary -- so a correct boundary rule must return "unknown".
    """
    cases: list[tuple[str, str, str]] = []
    for pattern in _PATTERNS:
        for side in ("left", "right"):
            end = pattern[0] if side == "left" else pattern[-1]
            if end in _SEPARATORS:
                continue  # that end is already bounded; nothing to glue onto
            for _ in range(_MAX_GLUE_ATTEMPTS):
                glue = _safe_glue()
                stem = glue + pattern if side == "left" else pattern + glue
                filename = f"{stem}.py"
                if any(_boundary_match(p, filename) for p in _PATTERNS):
                    continue  # unfair: some pattern legitimately matches
                cases.append((filename, "unknown", f"{pattern}-{side}"))
                break
            else:
                raise RuntimeError(f"could not build a fair case for {pattern}-{side}")
    return cases


def main() -> int:
    try:
        from services.pr_validation.analyzers.impact.test_coverage_correlator import (
            TestCoverageCorrelator,
        )
    except Exception as exc:  # pragma: no cover - harness failure path
        print(f"HARNESS: import failed: {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return EXIT_NO_MEASUREMENT

    try:
        correlator = TestCoverageCorrelator()
    except Exception as exc:  # pragma: no cover
        print(f"HARNESS: could not construct TestCoverageCorrelator: {exc!r}")
        print("This check measured nothing; do not read it as a pass.")
        return EXIT_NO_MEASUREMENT

    if not hasattr(correlator, "_determine_test_type"):
        print("HARNESS: _determine_test_type is absent -- renamed or removed.")
        print("This check measured nothing; do not read it as a pass.")
        return EXIT_NO_MEASUREMENT

    misclassified, controls_broken = [], []

    for filename, expected in DEFECT_CASES:
        actual = correlator._determine_test_type(filename)
        if actual != expected:
            misclassified.append((filename, expected, actual))

    for filename, expected in POSITIVE_CONTROLS:
        actual = correlator._determine_test_type(filename)
        if actual != expected:
            controls_broken.append((filename, expected, actual))

    try:
        generated = generate_boundary_cases()
    except RuntimeError as exc:
        print(f"HARNESS: could not generate boundary cases: {exc}")
        print("This check measured nothing; do not read it as a pass.")
        return EXIT_NO_MEASUREMENT
    if len(generated) != 9:
        print(f"HARNESS: expected 9 anchor slots, generated {len(generated)}.")
        print("This check measured nothing; do not read it as a pass.")
        return EXIT_NO_MEASUREMENT
    for filename, expected, slot in generated:
        actual = correlator._determine_test_type(filename)
        if actual != expected:
            misclassified.append((f"{filename} [{slot}]", expected, actual))

    print(
        f"defect cases:      {len(DEFECT_CASES) + len(generated) - len(misclassified)}/{len(DEFECT_CASES) + len(generated)} correct"
        f"  ({len(DEFECT_CASES)} fixed + {len(generated)} generated = all 9 anchor slots)"
    )
    print(
        f"positive controls: {len(POSITIVE_CONTROLS) - len(controls_broken)}/{len(POSITIVE_CONTROLS)} correct"
    )

    if misclassified:
        print("\nMISCLASSIFIED (the defect):")
        for filename, expected, actual in misclassified:
            print(f"  {filename:32s} expected {expected:12s} got {actual}")

    if controls_broken:
        print("\n⛔ POSITIVE CONTROL BROKEN (over-correction -- NOT a fix):")
        for filename, expected, actual in controls_broken:
            print(f"  {filename:32s} expected {expected:12s} got {actual}")

    if misclassified or controls_broken:
        return EXIT_DEFECT

    print("\nAll cases correct: both carriers match on a word boundary.")
    return EXIT_FIXED


if __name__ == "__main__":
    sys.exit(main())
