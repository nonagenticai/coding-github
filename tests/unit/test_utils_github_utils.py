"""Tests for utils/github_utils.py."""

from datetime import UTC, datetime

import pytest

from utils.github_utils import (
    format_github_url,
    parse_github_url,
    parse_rate_limit_reset,
    validate_github_token,
)


class TestGitHubUtils:
    """Tests for github_utils module."""

    def test_format_github_url(self):
        """format_github_url should produce a standard HTTPS URL."""
        url = format_github_url("myorg", "myrepo")
        assert url == "https://github.com/myorg/myrepo"

    def test_parse_github_url_https(self):
        """parse_github_url should extract owner and repo from HTTPS URLs."""
        owner, repo = parse_github_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

        # With .git suffix
        owner2, repo2 = parse_github_url("https://github.com/org/project.git")
        assert owner2 == "org"
        assert repo2 == "project"

    def test_validate_github_token_valid_prefix(self):
        """Tokens with valid GitHub prefixes should pass."""
        assert validate_github_token("ghp_abc123def456ghi789jkl") is True
        assert validate_github_token("gho_abc123def456ghi789jkl") is True
        assert validate_github_token("ghs_abc123def456ghi789jkl") is True
        assert validate_github_token("github_pat_abc123def456") is True

    def test_invalid_url_formats(self):
        """Invalid URLs should return (None, None)."""
        assert parse_github_url("") == (None, None)
        assert parse_github_url("not-a-url") == (None, None)
        assert parse_github_url("https://gitlab.com/owner/repo") == (None, None)
        assert parse_github_url("ftp://github.com/owner/repo") == (None, None)


class TestParseRateLimitReset:
    """The SHARED total conversion for ``X-RateLimit-Reset``.

    ⛔ THE REGRESSION THIS GROUP PINS: both call sites used to run
    ``datetime.fromtimestamp(reset_epoch, UTC)`` OUTSIDE the try/except that guarded
    the ``int()`` parse. A reset arriving in MILLISECONDS (edge proxies and GHE
    fronting layers do send one) raised ``ValueError: year must be in 1..9999, not
    58692``:

    * error path — it escaped the route's own ``except Exception`` clause, so a 403
      that must be told apart from a permission denial arrived as an opaque 500 with
      NO ``error_code``;
    * success path (shipped live in #110) — it was swallowed by
      ``except ValueError -> not_found``, so ``get-repository`` answered **404 for a
      repository that exists**.

    MUTANTS this group must refuse, each verified RED:
    1. delete the plausible-range guard -> ``test_milliseconds_*`` goes red
       (``reset_epoch`` comes back as 1.79e12 even if ``reset_at`` does not);
    2. ``return None, None`` unconditionally -> ``test_plain_seconds_*`` (the positive
       control) goes red. ⭐ WITHOUT that control every other row here passes on a
       function that always answers None, which is a gate that cannot fail.
    """

    # 2026-09-21T14:13:20+00:00 — the value the route tests use.
    SECONDS = 1_790_000_000
    MILLIS = 1_790_000_000_000

    def test_plain_seconds_round_trip(self) -> None:
        """⭐ THE POSITIVE CONTROL. Without it, an always-None helper passes the rest."""
        assert parse_rate_limit_reset(self.SECONDS) == (
            self.SECONDS,
            datetime.fromtimestamp(self.SECONDS, UTC).isoformat(),
        )

    def test_plain_seconds_as_a_string_round_trips_too(self) -> None:
        """Headers arrive as strings; the router hands the raw header value straight in."""
        assert parse_rate_limit_reset(str(self.SECONDS)) == (
            self.SECONDS,
            datetime.fromtimestamp(self.SECONDS, UTC).isoformat(),
        )

    @pytest.mark.parametrize("raw", [MILLIS, str(MILLIS), MILLIS * 1000])
    def test_milliseconds_are_unknown_not_a_fabricated_date(self, raw) -> None:
        """⭐ THE REGRESSION ITSELF. The ms value must NOT raise, must NOT become a
        date, and must NOT be handed on as an epoch either — a consumer computing
        ``new Date(reset_epoch * 1000)`` would otherwise reproduce the defect
        downstream. ⛔ NOT divided by 1000: nothing in the header says whether it is
        ms, µs or corruption, so the unit is undecidable and the answer is None."""
        assert parse_rate_limit_reset(raw) == (None, None)

    @pytest.mark.parametrize("raw", [-1, "-1", -1_790_000_000])
    def test_negative_is_unknown_not_1969(self, raw) -> None:
        """``-1`` used to render ``1969-12-31T23:59:59+00:00`` — wrong, and not fatal,
        which is exactly why it survived: a plausible-looking date nobody checked."""
        assert parse_rate_limit_reset(raw) == (None, None)

    @pytest.mark.parametrize("raw", [0, "0"])
    def test_zero_is_unknown(self, raw) -> None:
        """PyGithub's pre-response seed. Unknown, never "the window resets at 1970"."""
        assert parse_rate_limit_reset(raw) == (None, None)

    @pytest.mark.parametrize(
        "raw", ["", "  ", "soon", "1790000000.5", "1e12", {"a": 1}, [1], object()]
    )
    def test_non_numeric_is_unknown(self, raw) -> None:
        assert parse_rate_limit_reset(raw) == (None, None)

    def test_absent_is_unknown(self) -> None:
        """``dict.get`` on a missing header, and ``getattr`` on a requester that has
        not been given one, both arrive here as ``None``."""
        assert parse_rate_limit_reset(None) == (None, None)

    @pytest.mark.parametrize(
        ("raw", "expected_epoch"),
        [
            (999_999_999, None),  # just below the floor (2001-09-09)
            (1_000_000_000, 1_000_000_000),  # the floor itself is accepted
            (4_000_000_000, 4_000_000_000),  # the ceiling itself is accepted
            (4_000_000_001, None),  # just above the ceiling (2096-10-02)
        ],
    )
    def test_the_plausible_seconds_window_is_inclusive_at_both_ends(
        self, raw, expected_epoch
    ) -> None:
        """A range guard that is off by one at either end is a silent behaviour change,
        so both edges are pinned rather than described."""
        epoch, iso = parse_rate_limit_reset(raw)
        assert epoch == expected_epoch
        assert (iso is None) is (expected_epoch is None)

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "nope",
            0,
            -1,
            1_790_000_000,
            1_790_000_000_000,
            10**30,
            -(10**30),
            True,
            1.5,
            float("nan"),
            float("inf"),
            {"x": 1},
            (),
        ],
    )
    def test_it_is_total_and_never_raises(self, raw) -> None:
        """Requirement 1, stated as one property: for ANY input this returns a pair and
        never propagates an exception. ``fromtimestamp``'s ValueError, ``int()``'s
        TypeError and an OverflowError on a huge float all had to be caught for this."""
        epoch, iso = parse_rate_limit_reset(raw)
        assert epoch is None or isinstance(epoch, int)
        assert iso is None or isinstance(iso, str)
