"""Unit tests for ``utils.credential_redaction``.

⛔ WHAT THESE EXIST TO PREVENT. Redacting the ``logger.error`` call beside a
bare ``raise`` fixes the log and leaves the credential in ``str(exc)``. Measured
on the clone path before this module existed::

    log line      : https://***@github.com/…          (redacted)
    str(reraised) : https://<real token>@github.com/…  (NOT redacted)

The tests below drive the helper directly. The clone path that uses it is
covered separately in ``test_repo_operations.py`` (the second, dormant path in
``github_container_provisioner`` was deleted 2026-09-30) — a helper test cannot
prove a call site passes its secrets in, so neither file substitutes for the other.
"""

from __future__ import annotations

import subprocess

from utils.credential_redaction import redact_credentials, redact_exception

_TOKEN = "ghp_redactionUnitTestCredential00000000"


class TestRedactCredentials:
    def test_url_userinfo_is_masked_without_being_told_the_secret(self):
        """Pass 1. Catches a credential this process never held."""
        text = "fatal: repository 'https://someone-elses-token@github.com/o/r/' not found"
        assert (
            redact_credentials(text) == "fatal: repository 'https://***@github.com/o/r/' not found"
        )

    def test_the_x_access_token_userinfo_form_is_masked_whole(self):
        """``x-access-token:<token>`` is userinfo too — the colon must not split it."""
        redacted = redact_credentials(f"https://x-access-token:{_TOKEN}@github.com/o/r")
        assert redacted == "https://***@github.com/o/r"
        assert _TOKEN not in redacted

    def test_a_secret_outside_a_url_needs_the_literal_pass(self):
        """⛔ Pass 2, and it is LOAD-BEARING.

        Deleting the literal loop left 54 tests passing because every one of
        them fed URL-shaped text. This is the shape that separates the passes:
        no ``@host``, so the regex is a no-op and only the literal replacement
        can mask the token.
        """
        text = f"remote: Bad credentials for {_TOKEN}"
        assert redact_credentials(text) == text  # regex alone: no change at all
        assert redact_credentials(text, _TOKEN) == "remote: Bad credentials for ***"

    def test_none_and_empty_are_safe(self):
        assert redact_credentials(None) == ""
        assert redact_credentials("", _TOKEN) == ""

    def test_an_empty_or_whitespace_secret_does_not_mask_everything(self):
        """A blank secret must not become a ``str.replace("", "***")`` bomb."""
        assert redact_credentials("plain text", "", None, "   ") == "plain text"

    def test_a_truncated_secret_is_a_known_residual_not_a_covered_case(self):
        """⚠️ RECORDED, NOT FIXED — asserting the leak so nobody claims coverage.

        Both passes are exact: the regex needs a ``…@host`` shape, the literal
        pass needs the WHOLE secret. A prefix printed outside a URL survives
        both. This test documents that honestly rather than leaving a reader to
        assume the helper is a boundary. If someone closes it, this test fails
        and should be rewritten — that is the intended signal, not a regression.
        """
        truncated = f"remote: Bad credentials ({_TOKEN[:12]}…)"
        assert redact_credentials(truncated, _TOKEN) == truncated
        assert _TOKEN[:12] in redact_credentials(truncated, _TOKEN)


class TestRedactException:
    def test_timeout_expired_keeps_its_type_and_loses_the_credential(self):
        """``TimeoutExpired.__str__`` interpolates the WHOLE argv.

        Its ``args`` is empty, so any args-based sanitiser is a no-op on it —
        which is why this type is rebuilt field by field.
        """
        argv = ["git", "clone", f"https://{_TOKEN}@github.com/o/r", "/tmp/x"]
        original = subprocess.TimeoutExpired(cmd=argv, timeout=1200)
        assert _TOKEN in str(original)  # instrument control

        safe = redact_exception(original, _TOKEN)

        assert isinstance(safe, subprocess.TimeoutExpired)
        assert _TOKEN not in str(safe)
        assert "***@github.com" in str(safe)
        assert safe.timeout == 1200

    def test_called_process_error_redacts_cmd_output_and_stderr(self):
        """``CalledProcessError`` carries caller-read attributes, not just a message."""
        argv = ["git", "clone", f"https://{_TOKEN}@github.com/o/r"]
        original = subprocess.CalledProcessError(
            returncode=128,
            cmd=argv,
            output=f"stdout mentions {_TOKEN}",
            stderr=f"fatal: repository 'https://{_TOKEN}@github.com/o/r/' not found",
        )

        safe = redact_exception(original, _TOKEN)

        assert isinstance(safe, subprocess.CalledProcessError)
        assert safe.returncode == 128
        assert _TOKEN not in str(safe)
        assert _TOKEN not in (safe.output or "")
        assert _TOKEN not in (safe.stderr or "")
        assert "***@github.com" in (safe.stderr or "")

    def test_bytes_output_stays_bytes(self):
        """``subprocess`` without ``text=True`` yields bytes; shape must survive."""
        original = subprocess.CalledProcessError(
            returncode=1, cmd=["git"], stderr=f"https://{_TOKEN}@github.com".encode()
        )
        safe = redact_exception(original, _TOKEN)
        assert isinstance(safe.stderr, bytes)
        assert safe.stderr == b"https://***@github.com"

    def test_a_clean_exception_is_returned_unchanged_and_identical(self):
        """Identity is the contract: it tells the caller a plain ``raise`` is safe.

        ``_clone_repository`` relies on this — its own already-redacted
        ``ValueError`` must be re-raised as the SAME object so existing callers
        and tests keep the exception they expect.
        """
        original = ValueError("Failed to clone repository: fatal: not found")
        assert redact_exception(original, _TOKEN) is original

    def test_a_generic_exception_keeps_its_type_when_it_can(self):
        original = ValueError(f"cloning https://{_TOKEN}@github.com/o/r failed")
        safe = redact_exception(original, _TOKEN)
        assert type(safe) is ValueError
        assert safe is not original
        assert _TOKEN not in str(safe)

    def test_a_type_that_cannot_be_rebuilt_degrades_to_runtime_error(self):
        """Losing the TYPE is strictly better than leaking the credential."""

        class _AwkwardError(Exception):
            def __init__(self, a, b):  # refuses a single-argument rebuild
                super().__init__(f"{a}/{b}")

        original = _AwkwardError(f"https://{_TOKEN}@github.com", "boom")
        safe = redact_exception(original, _TOKEN)

        assert isinstance(safe, RuntimeError)
        assert _TOKEN not in str(safe)
        assert "***@github.com" in str(safe)

    def test_a_type_whose_str_ignores_its_argument_degrades_too(self):
        """Verify the rebuild, do not assume it. A ``__str__`` that hard-codes
        the message would reintroduce the credential through the back door."""

        class _StubbornError(Exception):
            def __str__(self):
                return f"always https://{_TOKEN}@github.com"

        safe = redact_exception(_StubbornError("ignored"), _TOKEN)

        assert isinstance(safe, RuntimeError)
        assert _TOKEN not in str(safe)
