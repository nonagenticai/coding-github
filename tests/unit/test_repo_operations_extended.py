"""Extended tests for services/github/repo_operations.py — covering uncovered lines."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.github.repo_operations import RepositoryOperations


class TestRepositoryOperationsExtended:
    """Additional tests targeting uncovered lines in repo_operations.py."""

    def _make_ops(self, tmp_path: Path | None = None):
        with patch("services.github.repo_operations.settings") as mock_settings:
            workspace = str(tmp_path) if tmp_path else "/tmp/test_workspaces_ext"
            mock_settings.WORKSPACE_BASE_PATH = workspace
            ops = RepositoryOperations()
            ops.workspace_base = workspace
        return ops

    # ------------------------------------------------------------------ #
    # Line 58: clone_repository when local_path already exists
    # ------------------------------------------------------------------ #
    async def test_clone_removes_existing_directory(self, tmp_path):
        """When the target dir already exists, _remove_directory is called before cloning."""
        ops = self._make_ops(tmp_path)

        # Pre-create the target directory so os.path.exists returns True
        existing_dir = tmp_path / "owner-repo"
        existing_dir.mkdir()
        (existing_dir / "old_file.txt").write_text("stale")

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.clone_repository("https://github.com/owner/repo")

        assert result.success is True
        # The old directory should have been removed
        assert not (existing_dir / "old_file.txt").exists()

    # ------------------------------------------------------------------ #
    # Lines 93-95: clone_repository exception path
    # ------------------------------------------------------------------ #
    async def test_clone_repository_exception(self, tmp_path):
        """When _parse_repo_url raises ValueError, clone returns failure."""
        ops = self._make_ops(tmp_path)
        result = await ops.clone_repository("not-a-valid-url")
        assert result.success is False
        assert result.repo_id == ""
        assert result.error is not None

    async def test_clone_repository_oserror(self, tmp_path):
        """When an OSError occurs during clone, it is caught and returns failure."""
        ops = self._make_ops(tmp_path)
        with patch.object(ops, "_parse_repo_url", side_effect=OSError("disk full")):
            result = await ops.clone_repository("https://github.com/owner/repo")
        assert result.success is False
        assert "disk full" in result.error

    # ------------------------------------------------------------------ #
    # Lines 99-167: sync_repository — success with updates
    # ------------------------------------------------------------------ #
    async def test_sync_repository_with_updates(self, tmp_path):
        """sync_repository detects commits behind and pulls updates."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            proc = AsyncMock()
            if args[1] == "fetch":
                # git fetch origin succeeds
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif args[1] == "rev-list":
                # First branch (main) shows 3 commits behind
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"3\n", b""))
            elif args[1] == "pull":
                # git pull succeeds
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            call_count += 1
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is True
        assert result.commits_behind == 3
        assert result.has_updates is True

    async def test_sync_repository_no_updates(self, tmp_path):
        """sync_repository with zero commits behind returns has_updates=False."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            if args[1] == "fetch":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif args[1] == "rev-list":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"0\n", b""))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is True
        assert result.commits_behind == 0
        assert result.has_updates is False

    # ------------------------------------------------------------------ #
    # Lines 114-122: sync_repository — fetch failure
    # ------------------------------------------------------------------ #
    async def test_sync_repository_fetch_failure(self, tmp_path):
        """sync_repository returns failure when git fetch fails."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.returncode = 128
            proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repo"))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is False
        assert "fatal" in (result.error or "")

    # ------------------------------------------------------------------ #
    # Lines 156-157: sync_repository — pull fails after finding updates
    # ------------------------------------------------------------------ #
    async def test_sync_repository_pull_fails(self, tmp_path):
        """sync_repository still succeeds if pull fails but fetch succeeded."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            if args[1] == "fetch":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif args[1] == "rev-list":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"2\n", b""))
            elif args[1] == "pull":
                proc.returncode = 1
                proc.communicate = AsyncMock(return_value=(b"", b"merge conflict"))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        # Still success because fetch worked
        assert result.success is True
        assert result.commits_behind == 2
        assert result.has_updates is True

    # ------------------------------------------------------------------ #
    # Line 167: sync_repository — no main/master branch found
    # ------------------------------------------------------------------ #
    async def test_sync_repository_no_main_or_master(self, tmp_path):
        """sync_repository falls through both branches and returns no updates."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            if args[1] == "fetch":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif args[1] == "rev-list":
                # Both main and master fail
                proc.returncode = 128
                proc.communicate = AsyncMock(return_value=(b"", b"unknown revision"))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is True
        assert result.commits_behind == 0
        assert result.has_updates is False

    # ------------------------------------------------------------------ #
    # Lines 169-176: sync_repository — exception path
    # ------------------------------------------------------------------ #
    async def test_sync_repository_exception(self, tmp_path):
        """sync_repository catches OSError and returns failure."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no git")):
            result = await ops.sync_repository(workspace)

        assert result.success is False
        assert "no git" in result.error

    # ------------------------------------------------------------------ #
    # Lines 180-201: checkout_branch
    # ------------------------------------------------------------------ #
    async def test_checkout_branch_non_main(self, tmp_path):
        """checkout_branch runs git checkout for non-main branches."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await ops.checkout_branch(tmp_path, "feature-xyz")

        # Should have called git checkout
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0]
        assert "checkout" in call_args
        assert "feature-xyz" in call_args

    async def test_checkout_branch_main_skipped(self, tmp_path):
        """checkout_branch does nothing for 'main' branch."""
        ops = self._make_ops(tmp_path)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            await ops.checkout_branch(tmp_path, "main")

        mock_exec.assert_not_called()

    async def test_checkout_branch_master_skipped(self, tmp_path):
        """checkout_branch does nothing for 'master' branch."""
        ops = self._make_ops(tmp_path)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            await ops.checkout_branch(tmp_path, "master")

        mock_exec.assert_not_called()

    async def test_checkout_branch_empty_string_skipped(self, tmp_path):
        """checkout_branch does nothing for empty string branch."""
        ops = self._make_ops(tmp_path)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            await ops.checkout_branch(tmp_path, "")

        mock_exec.assert_not_called()

    async def test_checkout_branch_failure(self, tmp_path):
        """checkout_branch logs warning on failure but does not raise."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error: pathspec not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Should not raise
            await ops.checkout_branch(tmp_path, "nonexistent-branch")

    async def test_checkout_branch_exception(self, tmp_path):
        """checkout_branch catches OSError and logs warning."""
        ops = self._make_ops(tmp_path)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no git binary")):
            # Should not raise
            await ops.checkout_branch(tmp_path, "some-branch")

    # ------------------------------------------------------------------ #
    # Lines 205-217: create_and_checkout_branch
    # ------------------------------------------------------------------ #
    async def test_create_and_checkout_branch_success(self, tmp_path):
        """create_and_checkout_branch runs git checkout -b successfully."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await ops.create_and_checkout_branch(workspace, "new-feature")

        call_args = mock_exec.call_args[0]
        assert "checkout" in call_args
        assert "-b" in call_args
        assert "new-feature" in call_args

    async def test_create_and_checkout_branch_failure(self, tmp_path):
        """create_and_checkout_branch raises RuntimeError on failure."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"fatal: branch already exists"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Failed to create branch"):
                await ops.create_and_checkout_branch(workspace, "existing-branch")

    # ------------------------------------------------------------------ #
    # Line 251: _build_authenticated_url with SSH URL + token
    # ------------------------------------------------------------------ #
    def test_build_authenticated_url_ssh_with_token(self):
        """SSH URLs are returned as-is even when a token is provided."""
        ops = self._make_ops()
        result = ops._build_authenticated_url("git@github.com:owner/repo.git", "ghp_token123")
        assert result == "git@github.com:owner/repo.git"

    # ------------------------------------------------------------------ #
    # Lines 261-263: _build_authenticated_url exception fallback
    # ------------------------------------------------------------------ #
    def test_build_authenticated_url_parse_error(self):
        """On URL parse error, original URL is returned."""
        ops = self._make_ops()
        # Patch urlparse to raise ValueError
        with patch("services.github.repo_operations.RepositoryOperations._build_authenticated_url"):
            # Actually test the real method — we need to trigger the except
            pass

        # Use a real approach: patch urlparse inside the method to raise
        with patch("urllib.parse.urlparse", side_effect=ValueError("bad url")):
            result = ops._build_authenticated_url("https://github.com/owner/repo", "ghp_tok")
        assert result == "https://github.com/owner/repo"

    # ------------------------------------------------------------------ #
    # Lines 267-272: _remove_directory
    # ------------------------------------------------------------------ #
    async def test_remove_directory_success(self, tmp_path):
        """_remove_directory removes the directory."""
        ops = self._make_ops(tmp_path)
        target = tmp_path / "to_remove"
        target.mkdir()
        (target / "file.txt").write_text("data")

        await ops._remove_directory(str(target))
        assert not target.exists()

    async def test_remove_directory_nonexistent(self, tmp_path):
        """_remove_directory does nothing for nonexistent path."""
        ops = self._make_ops(tmp_path)
        await ops._remove_directory(str(tmp_path / "nonexistent"))
        # Should not raise

    async def test_remove_directory_permission_error(self, tmp_path):
        """_remove_directory catches PermissionError and continues."""
        ops = self._make_ops(tmp_path)
        with patch("shutil.rmtree", side_effect=PermissionError("denied")):
            with patch("os.path.exists", return_value=True):
                # Should not raise
                await ops._remove_directory("/some/protected/path")

    # ------------------------------------------------------------------ #
    # Lines 277-299: get_changed_files_from_git
    # ------------------------------------------------------------------ #
    async def test_get_changed_files_success(self, tmp_path):
        """get_changed_files_from_git returns list of changed files."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"src/main.py\ntests/test_main.py\n", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.get_changed_files_from_git(tmp_path)

        assert result == ["src/main.py", "tests/test_main.py"]

    async def test_get_changed_files_empty(self, tmp_path):
        """get_changed_files_from_git returns empty list when no changes."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.get_changed_files_from_git(tmp_path)

        assert result == []

    async def test_get_changed_files_git_failure(self, tmp_path):
        """get_changed_files_from_git returns empty list on git failure."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 128
        mock_process.communicate = AsyncMock(return_value=(b"", b"fatal: bad revision"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.get_changed_files_from_git(tmp_path)

        assert result == []

    async def test_get_changed_files_exception(self, tmp_path):
        """get_changed_files_from_git catches OSError and returns empty list."""
        ops = self._make_ops(tmp_path)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no git")):
            result = await ops.get_changed_files_from_git(tmp_path)

        assert result == []

    async def test_get_changed_files_filters_empty_strings(self, tmp_path):
        """get_changed_files_from_git filters out empty strings from split."""
        ops = self._make_ops(tmp_path)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"file1.py\n\nfile2.py\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await ops.get_changed_files_from_git(tmp_path)

        assert result == ["file1.py", "file2.py"]

    # ------------------------------------------------------------------ #
    # Line 139: sync_repository — empty stdout from rev-list
    # ------------------------------------------------------------------ #
    async def test_sync_repository_empty_rev_list_stdout(self, tmp_path):
        """sync_repository handles empty stdout from rev-list (0 commits behind)."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            if args[1] == "fetch":
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif args[1] == "rev-list":
                # Return empty string from stdout
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is True
        assert result.commits_behind == 0
        assert result.has_updates is False

    # ------------------------------------------------------------------ #
    # Line 115: sync_repository — fetch fails with empty stderr
    # ------------------------------------------------------------------ #
    async def test_sync_repository_fetch_fails_empty_stderr(self, tmp_path):
        """sync_repository uses default message when stderr is empty on fetch failure."""
        ops = self._make_ops(tmp_path)
        workspace = MagicMock()
        workspace.local_path = str(tmp_path)

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await ops.sync_repository(workspace)

        assert result.success is False
        assert result.error == "Git fetch failed"
