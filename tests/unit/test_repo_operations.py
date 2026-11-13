"""Tests for services/github/repo_operations.py"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from services.github.repo_operations import RepositoryOperations

# 40 chars, the length measured on the real GITHUB_ACCESS_TOKEN in the pod.
_LIVE_CLONE_CREDENTIAL = "ghp_liveClonePathCredential000000000000"

# One distinct literal per redaction site, deliberately: a shared constant
# would let a test pass because a DIFFERENT site redacted the same string, and
# these tests exist precisely because each site was found undefended on its own.
_CHECKOUT_STDERR_CREDENTIAL = "ghp_checkoutStderrCredential00000000000"
_BRANCH_RAISE_CREDENTIAL = "ghp_branchCreateRaiseCredential00000000"
_CLONE_HANDLER_CREDENTIAL = "ghp_cloneExceptHandlerCredential0000000"
_SYNC_HANDLER_CREDENTIAL = "ghp_syncExceptHandlerCredential00000000"
_AUTH_URL_CREDENTIAL = "ghp_authUrlBuilderCredential00000000000"
_CHECKOUT_RAISE_CREDENTIAL = "ghp_checkoutHandlerRaiseCredential0000"
_CLONE_URL_ARG_CREDENTIAL = "ghp_cloneUrlArgumentCredential000000000"


class TestRepositoryOperations:
    """Tests for RepositoryOperations."""

    def _make_ops(self, tmp_path: Path | None = None):
        with patch("services.github.repo_operations.settings") as mock_settings:
            workspace = str(tmp_path) if tmp_path else "/tmp/test_workspaces"
            mock_settings.WORKSPACE_BASE_PATH = workspace
            ops = RepositoryOperations()
            ops.workspace_base = workspace
        return ops

    # ------------------------------------------------------------------ #
    # Clone repository success
    # ------------------------------------------------------------------ #
    async def test_clone_repo_success(self, tmp_path):
        """Test clone_repository returns success on git clone success."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.clone_repository("https://github.com/owner/repo")
            assert result.success is True
            assert "owner-repo" in result.repo_id

    # ------------------------------------------------------------------ #
    # Clone repository failure
    # ------------------------------------------------------------------ #
    async def test_clone_repo_failure(self, tmp_path):
        """Test clone_repository returns failure when git clone fails."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", b"fatal: repo not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.clone_repository("https://github.com/owner/bad-repo")
            assert result.success is False
            assert "fatal" in (result.error or "")

    # ------------------------------------------------------------------ #
    # ⛔ Credential redaction on the clone path that HAS a production caller
    #
    # THE DEFECT THESE PIN. ``clone_repository`` builds
    # ``https://<token>@github.com/o/r`` and hands it to git. git echoes the
    # WHOLE url back in its own stderr, and this function used to
    #   (a) ``logger.error("Git clone failed: %s", error_msg)`` with it, and
    #   (b) RETURN it as ``CloneResult.error``.
    # (b) is the worse of the two and is why this is not merely a logging nit: a
    # returned field is caller-visible, so it gets persisted, echoed into a
    # card, or rendered in a UI. This is the only clone path left (the dormant
    # ``github_container_provisioner`` one was deleted 2026-09-30); a caller that
    # passes a token here (e.g. a BYO tenant token) must never see it echoed.
    # ------------------------------------------------------------------ #
    async def test_clone_failure_redacts_the_credential_from_log_and_returned_error(
        self, tmp_path, caplog
    ):
        """The token appears in neither the log record nor ``CloneResult.error``.

        Kills the mutant "return ``stderr.decode()`` unredacted as
        ``CloneResult.error``" and, independently, "redact only the log line":
        the two assertions fail separately, so neither half can carry the other.
        """
        ops = self._make_ops(tmp_path)
        leaked = (
            "remote: Repository not found.\n"
            f"fatal: repository 'https://{_LIVE_CLONE_CREDENTIAL}@github.com/owner/repo/' "
            "not found"
        ).encode()

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", leaked))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token=_LIVE_CLONE_CREDENTIAL
            )

        assert result.success is False
        returned = result.error or ""
        assert "***@github.com" in returned  # instrument control: redaction ran
        assert _LIVE_CLONE_CREDENTIAL not in returned
        # The pre-existing contract still holds: git's own diagnosis survives.
        assert "fatal" in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _LIVE_CLONE_CREDENTIAL not in message

    async def test_clone_failure_redacts_a_credential_echoed_outside_a_url(self, tmp_path, caplog):
        """The literal pass, on the live path, with no URL to anchor on.

        Kills the mutant "delete the literal-secret loop from
        ``redact_credentials``" as observed through THIS caller: the URL regex is
        a no-op on this stderr, so only the literal pass can mask the token
        before it lands in ``CloneResult.error``.
        """
        ops = self._make_ops(tmp_path)
        leaked = (
            f"remote: Bad credentials for {_LIVE_CLONE_CREDENTIAL}\nfatal: Authentication failed"
        ).encode()

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", leaked))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token=_LIVE_CLONE_CREDENTIAL
            )

        returned = result.error or ""
        assert "Bad credentials for ***" in returned  # instrument control
        assert _LIVE_CLONE_CREDENTIAL not in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _LIVE_CLONE_CREDENTIAL not in message

    async def test_fetch_failure_redacts_the_credential_persisted_in_git_config(
        self, tmp_path, caplog
    ):
        """⚠️ THE CLONE CREDENTIAL OUTLIVES THE CLONE, and this is where it resurfaces.

        ``git clone https://<token>@github.com/o/r`` writes that url — credential
        included — into the workspace's ``.git/config`` as
        ``remote.origin.url``. So a later ``git fetch origin`` in that tree
        echoes the token back in ITS stderr even though ``sync_repository`` was
        never handed a token and has no secret to match literally. Only the URL
        pass can save this one, which is why ``sync_repository`` redacts with no
        secrets argument at all.

        Kills the mutant "redact only ``clone_repository``, leave
        ``sync_repository`` alone": ``SyncResult.error`` then carries the token
        to the caller.
        """
        ops = self._make_ops(tmp_path)
        workspace = AsyncMock()
        workspace.local_path = str(tmp_path)
        leaked = (
            f"fatal: could not read Username for 'https://{_LIVE_CLONE_CREDENTIAL}"
            "@github.com': No such device or address"
        ).encode()

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", leaked))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ops.sync_repository(workspace)

        assert result.success is False
        returned = result.error or ""
        assert "***@github.com" in returned  # instrument control
        assert _LIVE_CLONE_CREDENTIAL not in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _LIVE_CLONE_CREDENTIAL not in message

    # ------------------------------------------------------------------ #
    # Fork / authenticated clone
    # ------------------------------------------------------------------ #
    async def test_clone_with_token(self, tmp_path):
        """Test clone uses authenticated URL when token is provided."""
        ops = self._make_ops(tmp_path)

        called_args = []
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        async def capture_exec(*args, **kwargs):
            called_args.extend(args)
            return mock_process

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token="ghp_mytoken"
            )
            assert result.success is True
            # The clone URL should contain the token
            clone_url = called_args[2]  # 'git', 'clone', <url>, ...
            assert "ghp_mytoken@" in clone_url

    # ------------------------------------------------------------------ #
    # URL parsing
    # ------------------------------------------------------------------ #
    def test_parse_repo_url_variants(self):
        """Test _parse_repo_url with various URL formats."""
        ops = self._make_ops()

        assert ops._parse_repo_url("https://github.com/a/b") == {"owner": "a", "name": "b"}
        assert ops._parse_repo_url("https://github.com/a/b.git") == {"owner": "a", "name": "b"}
        assert ops._parse_repo_url("git@github.com:a/b.git") == {"owner": "a", "name": "b"}

        with pytest.raises(ValueError):
            ops._parse_repo_url("https://github.com/lonely")

    # ------------------------------------------------------------------ #
    # Workspace existence
    # ------------------------------------------------------------------ #
    def test_workspace_exists(self, tmp_path):
        """Test workspace_exists checks directory."""
        ops = self._make_ops(tmp_path)

        assert ops.workspace_exists("nonexistent") is False

        (tmp_path / "my-repo").mkdir()
        assert ops.workspace_exists("my-repo") is True

    # ------------------------------------------------------------------ #
    # ⛔ THE FOUR REDACTION SITES ON THIS MODULE THAT WERE MUTATION
    # SURVIVORS. Each of the tests below was written against a measured
    # survivor: with the named site reverted to its pre-PR (leaky) form the
    # WHOLE suite stayed green, so the redaction was correct and undefended.
    #
    # ⚠️ READ THIS BEFORE ADDING A "TOKEN OUTSIDE A URL" CASE HERE.
    # ``redact_credentials`` has two passes and only the URL regex is
    # available at three of these sites: ``checkout_branch`` (:233),
    # ``create_and_checkout_branch`` (:256) and ``sync_repository``'s handler
    # (:206) are all called with NO ``secrets`` argument, because none of
    # those functions is handed a token — the credential they can echo is the
    # one ``git clone`` persisted into ``.git/config`` as
    # ``remote.origin.url``, which is URL-shaped by construction. A bare
    # ``remote: Bad credentials for ghp_…`` at those three sites is NOT
    # redacted and cannot be: there is no literal to match. That residual is
    # recorded here rather than papered over with a test that would have to
    # assert the leak. The two sites that DO receive a secret —
    # ``clone_repository``'s handler and ``_build_authenticated_url`` — carry
    # the non-URL cases below.
    # ------------------------------------------------------------------ #

    async def test_checkout_branch_failure_redacts_the_credential_in_its_warning(
        self, tmp_path, caplog
    ):
        """``checkout_branch``'s warning is built from ``git checkout`` stderr.

        ``git checkout <branch>`` can contact the remote (a branch that only
        exists upstream, a lazy-fetch clone), and the remote it contacts is the
        ``remote.origin.url`` ``git clone`` persisted WITH its credential. So
        this stderr carries a live token even though ``checkout_branch`` is
        never handed one.

        Kills the mutant "log ``stderr.decode()`` instead of
        ``redact_credentials(stderr.decode())``" at this site: the warning then
        carries the token and the loop below fails.
        """
        ops = self._make_ops(tmp_path)
        leaked = (
            f"fatal: unable to access 'https://{_CHECKOUT_STDERR_CREDENTIAL}@github.com/o/r/': "
            "The requested URL returned error: 403"
        ).encode()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", leaked))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            await ops.checkout_branch(tmp_path, "feature/needs-remote")

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _CHECKOUT_STDERR_CREDENTIAL not in message
        # Instrument control, asserted AFTER the leak check so a real leak names
        # itself rather than being reported as a vacuous-capture problem: the
        # redaction ran over text that really did carry the credential, so the
        # loop above is not passing against a message that simply went missing.
        assert any("***@github.com" in message for message in messages), (
            "the redacted stderr never reached a log record - the loop above is vacuous"
        )

    async def test_checkout_branch_exception_handler_redacts_the_credential(self, tmp_path, caplog):
        """``checkout_branch``'s ``except`` handler logged a BARE ``e``.

        Distinct site from the stderr warning above: that one fires on a
        non-zero returncode, this one on an ``OSError``/``RuntimeError`` raised
        by the subprocess machinery itself. ``OSError`` from git carries the
        argv, and this workspace's ``.git/config`` holds the credential, so the
        exception text is URL-shaped.

        Kills the mutant "log bare ``e``" at this site.
        """
        ops = self._make_ops(tmp_path)
        boom = OSError(
            f"fatal: unable to access 'https://{_CHECKOUT_RAISE_CREDENTIAL}@github.com/o/r/'"
        )

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", side_effect=boom),
        ):
            await ops.checkout_branch(tmp_path, "feature/x")

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _CHECKOUT_RAISE_CREDENTIAL not in message
        assert any("***@github.com" in message for message in messages), (
            "the redacted exception text never reached a log record"
        )

    async def test_clone_logs_redact_a_credential_handed_in_via_repo_url(self, tmp_path, caplog):
        """Defence in depth on the two ``repo_url`` log lines.

        Today's callers pass a PLAIN url here - the authenticated one is the
        separate ``clone_url`` built inside ``clone_repository`` - so this is
        not a live leak. It is pinned anyway because the parameter is public and
        a future caller handing in userinfo would otherwise log it verbatim on
        BOTH the entry line and the success line.

        Kills the mutants "log bare ``repo_url``" at either site.
        """
        ops = self._make_ops(tmp_path)
        authed = f"https://{_CLONE_URL_ARG_CREDENTIAL}@github.com/o/r"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            await ops.clone_repository(authed)

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _CLONE_URL_ARG_CREDENTIAL not in message
        assert any("***@github.com" in message for message in messages), (
            "neither repo_url log line survived redaction - the loop above is vacuous"
        )

    async def test_create_branch_failure_raises_a_message_with_no_credential_in_it(self, tmp_path):
        """⭐ THE HIGHEST-VALUE SITE: this one RAISES, and a raised message travels.

        ``create_and_checkout_branch`` turns ``git checkout -b`` stderr into a
        ``RuntimeError`` message. Nothing on this path logs it and then stops —
        it propagates to whatever frame called it, which logs it, returns it in
        a 500 body, or writes it onto a card. The workspace it runs in is the
        one ``clone_repository`` created, so ``.git/config`` holds
        ``https://<token>@github.com/o/r`` and a ``checkout -b`` that touches
        the remote echoes it back.

        Kills the mutant ``raise RuntimeError(f"Failed to create branch:
        {stderr.decode()}")``.

        ⚠️ ``__cause__`` and ``__context__`` are asserted too: the previous
        audit of this PR found a leak surviving in ``__context__`` on the other
        clone path, where the redacted object was raised INSIDE the ``except``
        block and the raise machinery re-attached the unredacted original. This
        ``raise`` is not inside a handler, so both must be ``None`` — and if
        someone later moves it into one, this assertion is what reddens.
        """
        ops = self._make_ops(tmp_path)
        workspace = AsyncMock()
        workspace.local_path = str(tmp_path)
        leaked = (
            f"fatal: unable to access 'https://{_BRANCH_RAISE_CREDENTIAL}@github.com/o/r/': "
            "The requested URL returned error: 403"
        ).encode()

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", leaked))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Failed to create branch") as excinfo:
                await ops.create_and_checkout_branch(workspace, "feature/x")

        raised = str(excinfo.value)
        assert "***@github.com" in raised  # instrument control: redaction ran
        assert _BRANCH_RAISE_CREDENTIAL not in raised
        # git's own diagnosis must survive the redaction, or this trades a leak
        # for an undebuggable error.
        assert "403" in raised
        # ⛔ No path back to an unredacted original through the chain.
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None

    async def test_clone_handler_redacts_the_credential_from_log_and_returned_error(
        self, tmp_path, caplog
    ):
        """``clone_repository``'s ``except`` handler — a DIFFERENT site from its
        ``returncode != 0`` branch, and separately undefended.

        The returncode branch is pinned by
        ``test_clone_failure_redacts_the_credential_from_log_and_returned_error``
        above. This one fires when the subprocess machinery itself raises — the
        git binary is missing, the fork fails, the workspace is unwritable —
        and the message ``asyncio`` renders can quote the argv, auth URL
        included. It logs AND returns, so both are asserted.

        Kills the mutant ``error_msg = str(e)`` at this site.
        """
        ops = self._make_ops(tmp_path)
        boom = OSError(
            "[Errno 13] Permission denied while spawning: "
            f"git clone https://{_CLONE_HANDLER_CREDENTIAL}@github.com/owner/repo"
        )

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", side_effect=boom),
        ):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token=_CLONE_HANDLER_CREDENTIAL
            )

        assert result.success is False
        returned = result.error or ""
        assert "***@github.com" in returned  # instrument control: redaction ran
        assert _CLONE_HANDLER_CREDENTIAL not in returned
        assert "Permission denied" in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _CLONE_HANDLER_CREDENTIAL not in message

    async def test_clone_handler_redacts_a_credential_raised_outside_a_url(self, tmp_path, caplog):
        """The literal pass, at the handler site, with no URL to anchor on.

        ``clone_repository``'s handler is one of only two sites in this module
        that is handed the secret, so it is one of only two where a credential
        rendered OUTSIDE a URL can be masked at all. Kills the mutant "drop the
        ``github_token`` argument from this call" — which the URL-shaped test
        above cannot see, because the regex alone already handles that text.
        """
        ops = self._make_ops(tmp_path)
        boom = RuntimeError(
            f"git-credential helper refused: Bad credentials for {_CLONE_HANDLER_CREDENTIAL}"
        )

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", side_effect=boom),
        ):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token=_CLONE_HANDLER_CREDENTIAL
            )

        returned = result.error or ""
        # No "***@github.com" to look for — the premise is that no URL is
        # present — so the control is the bare mask where the token was.
        assert "Bad credentials for ***" in returned  # instrument control
        assert _CLONE_HANDLER_CREDENTIAL not in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        assert any("Bad credentials for ***" in message for message in messages), (
            "the redacted message never reached a log record - the loop below is vacuous"
        )
        for message in messages:
            assert _CLONE_HANDLER_CREDENTIAL not in message

    async def test_sync_handler_redacts_the_credential_from_log_and_returned_error(
        self, tmp_path, caplog
    ):
        """``sync_repository``'s ``except`` handler — distinct from its
        ``returncode != 0`` branch, which
        ``test_fetch_failure_redacts_the_credential_persisted_in_git_config``
        pins.

        Same credential provenance as that test: ``sync_repository`` is never
        handed a token, so the only token it can see is the one ``git clone``
        wrote into ``.git/config``, and the only pass that can mask it is the
        URL regex. This handler both logs and RETURNS the message as
        ``SyncResult.error``.

        Kills the mutant ``error_msg = str(e)`` at this site.
        """
        ops = self._make_ops(tmp_path)
        workspace = AsyncMock()
        workspace.local_path = str(tmp_path)
        boom = OSError(
            "[Errno 24] Too many open files while fetching "
            f"https://{_SYNC_HANDLER_CREDENTIAL}@github.com/owner/repo"
        )

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("asyncio.create_subprocess_exec", side_effect=boom),
        ):
            result = await ops.sync_repository(workspace)

        assert result.success is False
        returned = result.error or ""
        assert "***@github.com" in returned  # instrument control: redaction ran
        assert _SYNC_HANDLER_CREDENTIAL not in returned
        assert "Too many open files" in returned

        messages = [record.getMessage() for record in caplog.records]
        assert messages, "no log records captured - the loop below would pass vacuously"
        for message in messages:
            assert _SYNC_HANDLER_CREDENTIAL not in message

    async def test_authenticated_url_builder_redacts_a_credential_it_was_handed(
        self, tmp_path, caplog
    ):
        """``_build_authenticated_url``'s warning, reached through the PUBLIC caller.

        This site is reached only when ``urlparse`` refuses the repo URL, which
        is why it is driven here through ``clone_repository`` with ``urlparse``
        made to raise on its SECOND call: the first call belongs to
        ``_parse_repo_url``, which must succeed or the outer handler (a
        different site) swallows the run and this test would silently grade
        nothing. ``calls["n"] == 2`` is therefore load-bearing, not incidental.

        The credential here is the one the caller HANDED IN, so this site gets
        the secret and the literal pass is live. The message shape is the
        non-URL one deliberately — ``urlparse``'s own ``ValueError`` text
        ("Invalid IPv6 URL") does not quote the URL, so the realistic leak at
        this site is a credential rendered bare by a helper, not a clone URL.

        Kills the mutant ``logger.warning("Failed to build authenticated URL:
        %s", str(e))`` at this site.
        """
        ops = self._make_ops(tmp_path)
        real_urlparse = urllib.parse.urlparse
        calls = {"n": 0}

        def urlparse_then_refuse(url, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_urlparse(url, *args, **kwargs)
            raise ValueError(f"credential helper rejected token {_AUTH_URL_CREDENTIAL}")

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with (
            caplog.at_level(logging.DEBUG, logger="services.github.repo_operations"),
            patch("urllib.parse.urlparse", side_effect=urlparse_then_refuse),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
        ):
            result = await ops.clone_repository(
                "https://github.com/owner/repo", github_token=_AUTH_URL_CREDENTIAL
            )

        assert calls["n"] >= 2, "urlparse was never reached a second time - the site did not run"
        # The fallback contract: a refused parse must not break the clone.
        assert result.success is True

        warnings = [
            record.getMessage()
            for record in caplog.records
            if "Failed to build authenticated URL" in record.getMessage()
        ]
        assert warnings, "the _build_authenticated_url warning never fired - nothing was graded"
        for message in warnings:
            assert "rejected token ***" in message  # instrument control
            assert _AUTH_URL_CREDENTIAL not in message

        for record in caplog.records:
            assert _AUTH_URL_CREDENTIAL not in record.getMessage()
