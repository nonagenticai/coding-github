#!/usr/bin/env python3
# ⛔ CONVEYOR FIXTURE: the fix is deliberately absent; an AA conveyor card drives at
# this defect. Do not fix it by hand. Hardening this checker, or wiring it into CI,
# is welcome and does not consume the target.
#
# MEASURED DEFECT — src/utils/file_classifier.py, TWO remaining sites in
# CoreFileTypeClassifier. On main @957e6e83 they are:
#
#   :159-161  _is_config_file
#       for pattern in self.config_filenames:
#           if pattern in filename_lower:      # bare substring membership
#               return True
#
#   :212-214  _is_documentation_file
#       for pattern in self.documentation_filenames:
#           if pattern in filename_lower:      # bare substring membership
#               return True
#
# A third site, `_is_test_file` (:174), was fixed earlier by routing
# `self.test_patterns` through the boundary helper
# `CoreFileTypeClassifier._matches_pattern_on_word_boundary` (:183-198). These
# two loops were left on raw `in`, so any filename that merely CONTAINS one of
# the literals anywhere — inside an unrelated word — is misread:
#
#     src/loaders/rapid_ingest.py       -> "documentation"  ("api" inside "rapid")
#     src/finance/capital_gains.py      -> "documentation"  ("api" inside "capital")
#     src/build/makefiles_registry.py   -> "config"         ("makefile" inside "makefiles")
#     src/services/mydockerfile_linter.py -> "config"       ("dockerfile" glued to "my")
#
# Both predicates are called from real code, not only from get_file_type():
#   git_diff_analyzer.py:438 (_is_config_file), :485 (_is_documentation_file);
#   documentation_checker.py:186,198; pr_change_detector.py:90,274.
#
# ⚠️ ATTRIBUTE NAMES: the lists are `config_filenames` and
# `documentation_filenames`. There are no attributes named `config_patterns` or
# `doc_patterns` on this class; if a brief says otherwise, the brief is wrong.
#
# This is a BEHAVIOURAL check: the VERDICT always comes from calling the real
# CoreFileTypeClassifier.get_file_type(). See the docstring for the one place
# where a reference reimplementation IS used (choosing which questions are fair
# to ask) and why that cannot turn a live defect green.

r"""Oracle for the config/documentation filename-boundary misclassification.

Run from the repository root:

    uv run python -m checks.file_classifier_filename_boundary

Exit codes:
    0 — every case classified correctly: both loops now match on a word
        boundary, the loop-only positive controls still classify, and the
        cases generated fresh from the classifier's own filename lists THIS
        run all pass.
    1 — at least one case misclassified (expected on today's main), or a
        positive control was broken by an over-correction. The two are
        reported separately and never merged into one count.
    2 — the harness measured nothing (import failure, no safe random token,
        or too few fair cases could be generated). "This check measured
        nothing; do not read it as a pass."

WHAT THIS GUARANTEES
    Only this: for each filename it asks about, the real classifier returned
    the type stated here. Cases are built at runtime from the classifier's own
    `config_filenames` / `documentation_filenames`, glued to random per-run
    tokens with a per-case random extension, so there is no fixed string in
    this file that a fixer could special-case to go green. Hardcoding the
    handpicked literals below (they are enumerable — they are printed right
    there in the header) does not pass, because the generated cases are not
    known until the run starts.

WHAT IT CANNOT SEE
    * It is black-box. A classifier that detects it is being observed — e.g.
      walking the call stack with `inspect.stack()` / `sys._getframe` and
      branching to correct-looking logic only when this module is the caller
      — defeats it completely. That is a generic limit of any behavioural
      oracle running in an identifiable process; randomizing the DATA cannot
      close it, and no attempt is made here to defend against it.
    * "Random per run" is not "unguessable". It raises the cost of a
      literal-enumeration hack from one line to "must implement the real
      rule"; it does not prove no hack whatsoever could survive.
    * The generated filenames are SYNTHETIC — a pattern with random text glued
      to it — and therefore structurally unlike hand-written filenames. A hack
      keyed on that shape rather than on the literals is the residual hole.
      Two mitigations are in place, both measured: glue length is drawn per
      token (no fixed-width run to key on), and MUST_PRESERVE carries long,
      digit-bearing, loop-only real paths so that "reject long basenames" or
      "reject names containing digits" breaks a control. Neither closes the
      class; a sufficiently clever shape heuristic is not excluded.
    * It says nothing about the four non-loop branches of either predicate
      (extension membership, exact basename membership) beyond the handful of
      positive controls held below.
    * `_predict_code()` is a reference reimplementation of the classifier's
      own rules. It NEVER grades. It is used only to decide whether a
      generated filename is a FAIR question (i.e. whether "code" is the
      defensible answer); a candidate it cannot vouch for is dropped, not
      asserted. If it were sabotaged into rejecting everything, this check
      exits 2 (measured nothing), never 0. It prefers the live
      `_matches_pattern_on_word_boundary` and only falls back to a local copy
      if that helper is gone.
"""

from __future__ import annotations

import os
import random
import string
import sys

SITE_CONFIG = "src/utils/file_classifier.py:159-161 (CoreFileTypeClassifier._is_config_file)"
SITE_DOC = "src/utils/file_classifier.py:212-214 (CoreFileTypeClassifier._is_documentation_file)"

# A pattern end carrying one of these is already a real word boundary (a path
# segment bounded by "/", or a "_"/"."-delimited token). An end that is NOT one
# of these is the end bare substring membership leaves unchecked. Same set the
# shipped `_matches_pattern_on_word_boundary` uses.
_SEPARATOR_CHARS = ("_", "/", ".")

# Glue length is drawn per token, not fixed, so that a hack keyed on the SHAPE
# of the generated names ("a 6-character alphanumeric run next to a pattern")
# has no constant to key on either. See the long positive controls in
# MUST_PRESERVE for the other half of that defence.
_GLUE_LENGTH_RANGE = (3, 7)
_GLUE_ALPHABET = string.ascii_lowercase + string.digits
_MAX_GLUE_ATTEMPTS = 200

# Extension candidates for generated filenames. One is drawn per case so that a
# blanket "skip the loop for .py files" hack cannot pass: it would have to hold
# for every extension drawn, including the empty one. Filtered at runtime
# against the classifier's live extension sets before use.
_EXTENSION_CANDIDATES = (".py", ".go", ".rs", ".sh", ".java", ".rb", ".kt", "")

# Minimum fair generated cases per site. Below this the run is not a
# measurement of anything and exits 2 rather than reporting a pass.
_MIN_GENERATED_PER_SITE = 8

# Frozen snapshots of main @957e6e83, used ONLY when the live attribute is
# missing or is not a plain set of non-empty strings — so a legitimate
# refactor of the internal representation degrades into a weaker check rather
# than an AttributeError. When the live attribute is readable it is used, which
# is what keeps the generated cases derived from the code under test.
_FALLBACK_CONFIG_FILENAMES = frozenset(
    {
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
        "cmake",
        "cmakelists.txt",
        "build.gradle",
        "pom.xml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "poetry.lock",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pipfile",
        "pipfile.lock",
        "conda.yaml",
        "environment.yml",
        "jenkinsfile",
        ".gitlab-ci.yml",
        ".github/workflows/",
        "tox.ini",
        "mypy.ini",
        "pytest.ini",
        "flake8.ini",
        ".flake8",
        ".mypy.ini",
        ".pre-commit-config.yaml",
    }
)

_FALLBACK_DOC_FILENAMES = frozenset(
    {
        "readme",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "changelog",
        "changelog.md",
        "changelog.rst",
        "changelog.txt",
        "contributing",
        "contributing.md",
        "contributing.rst",
        "license",
        "license.md",
        "license.rst",
        "license.txt",
        "authors",
        "authors.md",
        "authors.rst",
        "authors.txt",
        "install",
        "install.md",
        "install.rst",
        "install.txt",
        "usage",
        "usage.md",
        "usage.rst",
        "usage.txt",
        "api",
        "api.md",
        "api.rst",
        "api.txt",
        "docs/",
        "/docs/",
        "documentation/",
        "/documentation/",
    }
)

_FALLBACK_CONFIG_EXTENSIONS = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".conf",
        ".config",
        ".env",
        ".properties",
        ".cfg",
        ".xml",
    }
)

_FALLBACK_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt", ".asciidoc", ".adoc", ".org"})

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
# MUST_PRESERVE — files that classify CORRECTLY on today's main and must keep
# doing so. They are positive controls (so "the harness is broken / everything
# reads code" is distinguishable from "the boundary cases leak") AND the
# over-correction guard: deleting the loops outright, or anchoring them so hard
# that real path-segment conventions stop matching, breaks these and is
# reported as a REGRESSION, separately from the defect count.
#
# The three marked "loop-only" are reachable ONLY through the loop under
# repair — their extension is not in any extension set and their basename is
# not an exact member of any filename set — so they are the ones a blanket
# rejection kills.
# ---------------------------------------------------------------------------
MUST_PRESERVE: tuple[tuple[str, str, str], ...] = (
    (
        "config, loop-only: .github/workflows/ segment",
        "deploy/.github/workflows/build.sh",
        "config",
    ),
    ("config, exact basename membership", "requirements.txt", "config"),
    ("config, extension membership", "src/config/settings.yaml", "config"),
    ("config, Dockerfile basename", "services/web/Dockerfile", "config"),
    ("doc, loop-only: docs/ segment", "project/docs/overview.html", "documentation"),
    ("doc, loop-only: documentation/ segment", "site/documentation/index.html", "documentation"),
    ("doc, extension membership", "README.md", "documentation"),
    ("doc, stem membership", "AUTHORS.pdf", "documentation"),
    # Long, digit-bearing, loop-only. The generated cases below are synthetic
    # and therefore structurally distinguishable from hand-written filenames;
    # these two exist so that the cheapest hacks keyed on that shape — "reject
    # anything with a long basename", "reject anything containing digits" —
    # break a control instead of going green.
    (
        "config, loop-only: long digit-bearing path",
        "infra/.github/workflows/nightly-release-verify-v2.sh",
        "config",
    ),
    (
        "doc, loop-only: long digit-bearing path",
        "project/docs/release-notes-2026-architecture.html",
        "documentation",
    ),
    # Cross-site guard: the already-fixed _is_test_file must not be collateral
    # damage of a fix to the other two loops.
    ("test site, already fixed, must stay fixed", "src/foo_test.py", "test"),
)

# ---------------------------------------------------------------------------
# Handpicked held-out defect cases. ⚠️ These ARE enumerable from the header
# above — a fixer that hardcodes exactly these nine strings would pass this
# tuple. It would still fail the generated cases, which is the point of
# generating them. They are kept because they are what the defect looks like in
# real source trees, which the synthetic names do not convey.
# ---------------------------------------------------------------------------
_HANDPICKED_CONFIG: tuple[tuple[str, str], ...] = (
    ("'dockerfile' glued onto 'my'", "src/services/mydockerfile_linter.py"),
    ("'makefile' inside 'makefiles'", "src/build/makefiles_registry.py"),
    ("'pipfile' inside 'pipfiles'", "src/packaging/pipfiles_audit.py"),
    ("'cmake' inside 'cmakey'", "src/tools/cmakey_shim.py"),
)

_HANDPICKED_DOC: tuple[tuple[str, str], ...] = (
    ("'api' inside 'rapid'", "src/loaders/rapid_ingest.py"),
    ("'api' inside 'capital'", "src/finance/capital_gains.py"),
    ("'usage' inside 'usages'", "src/metrics/usages_report.py"),
    ("'install' inside 'installer'", "src/packaging/installer_utils.py"),
    ("'license' inside 'licensed'", "src/legal/licensed_seats.py"),
)


def _read_str_set(classifier: object, attribute: str, fallback: frozenset[str]) -> frozenset[str]:
    """Read a set-of-strings attribute off the live classifier, falling back to
    a frozen snapshot if it is absent or has been refactored into something
    this generator cannot introspect."""
    value = getattr(classifier, attribute, None)
    if (
        isinstance(value, (set, frozenset))
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return frozenset(value)
    return fallback


def _generate_glue(needles: frozenset[str]) -> str:
    """Return a fresh random lowercase-alphanumeric token, safe to glue onto a
    pattern end.

    "Safe" means no live pattern literal appears inside the candidate, and the
    candidate does not appear inside a literal. Without this a random draw
    could — rarely, but not with probability zero — itself spell "api" or
    "docs", smuggling a genuine convention into a filename this oracle is
    about to assert is code, which would make the ORACLE wrong rather than
    strict. Raises if no safe candidate appears in _MAX_GLUE_ATTEMPTS tries;
    the caller must treat that as a broken harness (exit 2), never a pass.
    """
    for _ in range(_MAX_GLUE_ATTEMPTS):
        length = random.randint(*_GLUE_LENGTH_RANGE)
        candidate = "".join(random.choice(_GLUE_ALPHABET) for _ in range(length))
        if any(needle in candidate or candidate in needle for needle in needles):
            continue
        return candidate
    raise RuntimeError(
        f"could not draw a glue token safe against {len(needles)} pattern literals "
        f"after {_MAX_GLUE_ATTEMPTS} attempts"
    )


def _boundary_match(classifier: object, pattern: str, filename_lower: str) -> bool:
    """True if `pattern` occurs in `filename_lower` on a word boundary.

    Prefers the classifier's own shipped helper so the fairness filter tracks
    whatever a fix actually lands. Falls back to a local copy of the same rule
    only if that helper has been removed or renamed — in which case this check
    is weaker, not wrong, because it still never grades with this function.
    """
    helper = getattr(classifier, "_matches_pattern_on_word_boundary", None)
    if callable(helper):
        try:
            return bool(helper(pattern, filename_lower))
        except Exception:
            pass
    left_bounded = pattern[0] in _SEPARATOR_CHARS
    right_bounded = pattern[-1] in _SEPARATOR_CHARS
    index = filename_lower.find(pattern)
    while index != -1:
        before_ok = left_bounded or index == 0 or not filename_lower[index - 1].isalnum()
        end = index + len(pattern)
        after_ok = right_bounded or end == len(filename_lower) or not filename_lower[end].isalnum()
        if before_ok and after_ok:
            return True
        index = filename_lower.find(pattern, index + 1)
    return False


def _predict_code(classifier: object, filename: str, sets: dict[str, frozenset[str]]) -> bool:
    """Reference prediction: would a boundary-correct classifier call this
    filename "code"?

    ⚠️ FAIRNESS FILTER ONLY — never a verdict. A candidate this returns False
    for is DROPPED from the case list, never asserted against. It reimplements
    every branch of the three predicates (extension membership, exact
    basename/stem membership, test-extension suffix, and boundary pattern
    matching) so that a generated name is only asked about when "code" is the
    defensible answer under the classifier's own rules.
    """
    lowered = filename.lower()
    basename = os.path.basename(lowered)
    _, ext = os.path.splitext(basename)
    stem = os.path.splitext(basename)[0]

    if ext in sets["config_extensions"] or ext in sets["doc_extensions"]:
        return False
    if basename in sets["config_filenames"]:
        return False
    if stem in sets["doc_filenames"]:
        return False
    if any(basename.endswith(suffix) for suffix in sets["test_extensions"]):
        return False
    for key in ("config_filenames", "doc_filenames", "test_patterns"):
        if any(_boundary_match(classifier, pattern, lowered) for pattern in sets[key]):
            return False
    return True


def _generate_cases(
    classifier: object,
    patterns: frozenset[str],
    sets: dict[str, frozenset[str]],
    glue_left: str,
    glue_right: str,
    extensions: tuple[str, ...],
    rng: random.Random,
) -> tuple[tuple[str, str], ...]:
    """Build one boundary case per pattern by gluing random text onto whichever
    end of the pattern literal lacks a separator character — precisely the end
    bare substring membership leaves unchecked.

    A pattern bounded on BOTH ends (".github/workflows/", "/docs/") encodes a
    real path convention and yields no case. Unlike the sibling test-pattern
    oracle, a pattern open on both ends is glued on both ends here: these lists
    hold whole filenames and whole words, not affix conventions, so
    "<junk>dockerfile<junk>" is unambiguously not a Dockerfile, whereas
    left-gluing a compound like "integration_test" would have been arguable.

    Every candidate is passed through `_predict_code` before it is kept, so a
    candidate whose expected answer is not defensible is dropped rather than
    asserted. The caller treats "too few survivors" as exit 2.
    """
    cases: list[tuple[str, str]] = []
    for pattern in sorted(patterns):
        left_open = pattern[0] not in _SEPARATOR_CHARS
        right_open = pattern[-1] not in _SEPARATOR_CHARS
        ext = rng.choice(extensions)
        if left_open:
            # Open on the left, and possibly on the right too: gluing both ends
            # is safe here and strictly stronger than gluing one.
            filename = f"{glue_left}{pattern}{glue_right}{ext}"
        elif right_open:
            filename = f"{pattern}{glue_right}{ext}"
        else:
            continue
        if not _predict_code(classifier, filename, sets):
            continue
        cases.append((f"generated boundary case for {pattern!r}", filename))
    return tuple(cases)


def _classify(classifier: object, filename: str) -> tuple[str | None, str | None]:
    """Call the real classifier. Returns (result, error)."""
    try:
        return str(classifier.get_file_type(filename)), None  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        return None, f"get_file_type({filename!r}) raised {exc!r}"


def main() -> int:  # noqa: C901 - linear report, split would obscure the flow
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

    sets = {
        "config_filenames": _read_str_set(
            classifier, "config_filenames", _FALLBACK_CONFIG_FILENAMES
        ),
        "doc_filenames": _read_str_set(
            classifier, "documentation_filenames", _FALLBACK_DOC_FILENAMES
        ),
        "config_extensions": _read_str_set(
            classifier, "config_extensions", _FALLBACK_CONFIG_EXTENSIONS
        ),
        "doc_extensions": _read_str_set(
            classifier, "documentation_extensions", _FALLBACK_DOC_EXTENSIONS
        ),
        "test_patterns": _read_str_set(classifier, "test_patterns", _FALLBACK_TEST_PATTERNS),
        "test_extensions": _read_str_set(classifier, "test_extensions", _FALLBACK_TEST_EXTENSIONS),
    }
    all_literals = frozenset().union(*sets.values())

    try:
        glue_left = _generate_glue(all_literals)
        glue_right = _generate_glue(all_literals)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    extensions = tuple(
        ext
        for ext in _EXTENSION_CANDIDATES
        if ext not in sets["config_extensions"] and ext not in sets["doc_extensions"]
    )
    if not extensions:
        print("FAIL: every candidate extension collides with a live extension set")
        print("This check measured nothing; do not read it as a pass.")
        return 2

    rng = random.Random()
    print(f"glue tokens this run: {glue_left!r} / {glue_right!r}")

    generated_config = _generate_cases(
        classifier, sets["config_filenames"], sets, glue_left, glue_right, extensions, rng
    )
    generated_doc = _generate_cases(
        classifier, sets["doc_filenames"], sets, glue_left, glue_right, extensions, rng
    )
    for label, generated in (
        ("config_filenames", generated_config),
        ("documentation_filenames", generated_doc),
    ):
        if len(generated) < _MIN_GENERATED_PER_SITE:
            print(
                f"FAIL: only {len(generated)} fair case(s) generated from {label}, "
                f"need {_MIN_GENERATED_PER_SITE}"
            )
            print("This check measured nothing; do not read it as a pass.")
            return 2

    must_be_code = _HANDPICKED_CONFIG + _HANDPICKED_DOC + generated_config + generated_doc

    # A handpicked literal whose expected answer is not defensible under the
    # classifier's own rules is an ORACLE bug, not a defect: exit 2 rather than
    # report a red we cannot stand behind.
    for label, filename in _HANDPICKED_CONFIG + _HANDPICKED_DOC:
        if not _predict_code(classifier, filename, sets):
            print(f"FAIL: handpicked case [{label}] {filename!r} is not defensibly 'code'")
            print("This check measured nothing; do not read it as a pass.")
            return 2

    seen: set[str] = set()
    for _, filename in must_be_code:
        if filename in seen:
            print(f"FAIL: duplicate case filename {filename!r}")
            print("This check measured nothing; do not read it as a pass.")
            return 2
        seen.add(filename)

    regressions: list[tuple[str, str, str, str]] = []
    defects: list[tuple[str, str, str]] = []

    for label, filename, expected in MUST_PRESERVE:
        got, error = _classify(classifier, filename)
        if error is not None:
            print(f"FAIL: {error}")
            print("This check measured nothing; do not read it as a pass.")
            return 2
        if got != expected:
            regressions.append((label, filename, str(got), expected))

    for label, filename in must_be_code:
        got, error = _classify(classifier, filename)
        if error is not None:
            print(f"FAIL: {error}")
            print("This check measured nothing; do not read it as a pass.")
            return 2
        if got != "code":
            defects.append((label, filename, str(got)))

    checked = len(MUST_PRESERVE) + len(must_be_code)
    print(
        f"filenames checked: {checked} ({len(must_be_code)} must classify as 'code', "
        f"of which {len(generated_config)} generated from config_filenames and "
        f"{len(generated_doc)} from documentation_filenames; "
        f"{len(MUST_PRESERVE)} positive controls)"
    )

    if not defects and not regressions:
        print(f"\nok: all {checked} filename(s) classified correctly.")
        return 0

    if defects:
        # Attribute by what the classifier RETURNED, not by which list the case
        # came from: the returned type names the live loop directly, so a fix to
        # one site narrows the report to the other on the next run.
        by_site = {
            "config": (SITE_CONFIG, [d for d in defects if d[2] == "config"]),
            "documentation": (SITE_DOC, [d for d in defects if d[2] == "documentation"]),
        }
        other = [d for d in defects if d[2] not in ("config", "documentation")]
        print(f"\nFAIL: {len(defects)} file(s) MISCLASSIFIED (the live defect):")
        for site, group in by_site.values():
            if not group:
                continue
            print(f"\n  {site} — {len(group)} case(s):")
            for label, filename, got in group:
                print(f"    [{label}] {filename!r} -> {got!r}, expected 'code'")
        if other:
            print(f"\n  UNEXPECTED classification, not attributable to either loop — {len(other)}:")
            for label, filename, got in other:
                print(f"    [{label}] {filename!r} -> {got!r}, expected 'code'")
        live = [name for name, (_, group) in by_site.items() if group]
        print(
            f"\nLive site(s): {', '.join(live) if live else 'none attributable'}. "
            "Each loop tests substring membership anywhere in the lowercased "
            "filename, so a literal with no separator on one end has no boundary "
            "on that side and matches inside an unrelated word."
        )

    if regressions:
        print(f"\nFAIL: {len(regressions)} correctly-classified file(s) BROKEN (over-correction):")
        for label, filename, got, expected in regressions:
            print(f"  [{label}] {filename!r} -> {got!r}, expected {expected!r}")
        print(
            "\nThese classify correctly on main. Rejecting them is a regression, "
            "not progress on the defect — a loop that has been deleted or "
            "anchored too hard no longer recognises real path conventions."
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
