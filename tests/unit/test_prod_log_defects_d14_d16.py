"""Regression tests for three defects read off the live coding-github logs (2026-10-12).

Every test here runs over REAL PyGithub objects (``Github`` with only
``Requester.requestJsonAndCheck`` replaced, or ``github.File.File`` built with a
requester that refuses any use), never a ``MagicMock(core=...)``: the mocks in the
older suites had the shape the code assumed, which is exactly why these shipped.

* D14 ``Failed to check rate limit, assuming exhausted: 'RateLimitOverview' object
  has no attribute 'core'`` — and the ~1 s sleep it caused on every GitHub call.
* D15 ``PR validation failed: 'File' object has no attribute 'get'``.
* D16 ``Comprehensive validation failed: No module named 'models'``.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import pathlib
import time
from typing import Any
from unittest.mock import patch

from github import Auth, Github
from github.File import File
from github.Requester import Requester

from services.github.github_service import GitHubConfig, GitHubService

RESET_EPOCH = 4070908800  # 2099-01-01T00:00:00Z

# GitHub's documented GET /rate_limit body (trimmed to the buckets we read).
RATE_LIMIT_BODY = {
    "resources": {
        "core": {"limit": 5000, "used": 1, "remaining": 4999, "reset": RESET_EPOCH},
        "search": {"limit": 30, "used": 0, "remaining": 30, "reset": RESET_EPOCH},
    },
    "rate": {"limit": 5000, "used": 1, "remaining": 4999, "reset": RESET_EPOCH},
}


def _service_over_real_pygithub() -> GitHubService:
    svc = GitHubService(GitHubConfig(token_id="d14:test", token="not-a-real-credential"))
    svc._github = Github(auth=Auth.Token("not-a-real-credential"), retry=None)
    return svc


def _answer(verb: str, url: str) -> tuple[dict[str, Any], Any]:
    if url.endswith("/rate_limit"):
        return {}, RATE_LIMIT_BODY
    raise AssertionError(f"unexpected GitHub call: {verb} {url}")


def _fake_rest(self, verb, url, parameters=None, headers=None, input=None, **kw):
    return _answer(verb, url)


# --------------------------------------------------------------------------- D14


class TestD14RateLimitProbe:
    async def test_probe_reads_core_off_the_real_overview(self):
        svc = _service_over_real_pygithub()
        with patch.object(Requester, "requestJsonAndCheck", new=_fake_rest):
            info = await svc.check_rate_limit()
        assert info.probe_failed is False
        assert (info.remaining, info.limit, info.used) == (4999, 5000, 1)
        assert int(info.reset_at.timestamp()) == RESET_EPOCH

    async def test_healthy_probe_adds_no_latency_to_a_github_call(self):
        """Before the fix this took ~1.0 s: the probe raised, returned remaining=0
        with reset_at≈now, and ``_wait_for_rate_limit`` slept ``wait_time + 1``."""
        svc = _service_over_real_pygithub()
        with patch.object(Requester, "requestJsonAndCheck", new=_fake_rest):
            t0 = time.perf_counter()
            result = await svc._execute_with_retry(lambda: 42)
            elapsed = time.perf_counter() - t0
        assert result == 42
        assert elapsed < 0.25, f"GitHub call took {elapsed:.3f}s; the rate-limit probe slept"

    async def test_failed_probe_never_sleeps(self, caplog):
        """Any probe failure (here: a PyGithub shape we do not understand) must cost
        no latency — sleeping on an unknown budget is pure delay."""

        class _Overview:  # the pre-fix assumption's mirror image: no .resources
            core = None

        svc = _service_over_real_pygithub()
        sleeps: list[float] = []

        async def _record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with (
            patch.object(svc._github, "get_rate_limit", return_value=_Overview()),
            patch("services.github.github_service.asyncio.sleep", new=_record_sleep),
            caplog.at_level(logging.WARNING, logger="services.github.github_service"),
        ):
            info = await svc.check_rate_limit()
            svc._last_rate_check = svc._last_rate_check.replace(year=2000)
            result = await svc._execute_with_retry(lambda: "ok")

        assert info.probe_failed is True
        assert info.remaining == 0  # still conservative for any reader of the numbers
        assert result == "ok"
        assert sleeps == []
        assert "Rate-limit probe failed (AttributeError" in caplog.text


# --------------------------------------------------------------------------- D15


class _RefusingRequester:
    """Any network use by a File (lazy completion) fails the test loudly."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"PyGithub File tried to use its requester: {name}")


def _real_file(**attrs: Any) -> File:
    base = {
        "sha": "b" * 40,
        "filename": "app/service.py",
        "status": "modified",
        "additions": 2,
        "deletions": 1,
        "changes": 3,
        "patch": "@@ -1,2 +1,3 @@\n import os\n-x = 1\n+x = 2\n+y = 3\n",
        "blob_url": "https://github.com/o/r/blob/" + "b" * 40 + "/app/service.py",
    }
    base.update(attrs)
    return File(_RefusingRequester(), {}, base)  # type: ignore[arg-type]


class TestD15PyGithubFilesReachTheEngine:
    def test_file_to_dict_reads_real_pygithub_files(self):
        from services.pr_validation.integration.github_validation_service import (
            github_file_to_dict,
        )

        renamed = _real_file(
            status="renamed", filename="new.py", previous_filename="old.py", patch=None
        )
        d = github_file_to_dict(renamed)
        assert d["filename"] == "new.py"
        assert d["status"] == "renamed"
        assert d["previous_filename"] == "old.py"
        assert d["patch"] is None
        plain = {"filename": "x.py"}
        assert github_file_to_dict(plain) is plain

    async def test_comprehensive_validation_parses_real_files(self, caplog):
        from schemas.github_api_models import ValidatePRRequest
        from services.pr_validation.integration.github_validation_service import (
            PRGitHubValidationService,
        )

        svc = PRGitHubValidationService()
        seen: list[Any] = []
        real_validate = svc.validation_engine.validate_pr

        async def _spy(pr_api_data, ctx):
            seen.append(pr_api_data["files"])
            return await real_validate(pr_api_data, ctx)

        svc.validation_engine.validate_pr = _spy  # type: ignore[method-assign]
        request = ValidatePRRequest(
            repo_owner="o",
            repo_name="r",
            pr_number=7,
            validation_types=["security", "performance", "quality"],
        )
        files = [_real_file(), _real_file(filename="README.md", patch="@@ -0,0 +1 @@\n+hi\n")]
        with caplog.at_level(logging.ERROR):
            results = await svc.validate_github_pr(request, files, None)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not any("has no attribute 'get'" in m for m in errors), errors
        assert not any("Comprehensive validation failed" in m for m in errors), errors
        assert not any("PR validation failed" in m for m in errors), errors
        assert seen and all(isinstance(f, dict) for f in seen[0])
        assert results and all(r.validation_type != "error" for r in results)


# --------------------------------------------------------------------------- D16


def _without_models_modules() -> dict[str, Any]:
    """``sys.modules`` minus any ``models`` / ``models.*`` a sibling test stubbed in,
    so these tests prove the real import path regardless of test order."""
    import sys

    return {k: v for k, v in sys.modules.items() if k != "models" and not k.startswith("models.")}


class TestD16NoForeignSysPathImports:
    async def test_comprehensive_path_needs_no_models_package(self, caplog):
        """No ``sys.modules`` stub for ``models``: the old tests stubbed it, which is
        the only reason ``from models.pr_validation import PRDiff`` ever passed."""
        from coding_client.models.pr_validation import CodebaseContext

        from services.pr_validation.integration.github_validation_service import (
            GitHubPRValidationData,
            PRGitHubValidationService,
        )

        svc = PRGitHubValidationService()
        pr_data = GitHubPRValidationData(
            repo_owner="o",
            repo_name="r",
            pr_number=1,
            files=[_real_file()],
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="t",
            pr_body="b",
            author="u",
            created_at="2026-10-12T00:00:00Z",
            updated_at="2026-10-12T00:00:00Z",
        )
        ctx = CodebaseContext(
            total_files=1,
            total_lines=3,
            main_languages=["python"],
            test_framework=None,
            documentation_format=None,
            dependencies=[],
            architecture_patterns=[],
        )
        with (
            patch.dict("sys.modules", _without_models_modules(), clear=True),
            caplog.at_level(logging.ERROR),
        ):
            results = await svc._run_comprehensive_validation(
                pr_data, ctx, ["security", "performance", "quality"]
            )
        assert "No module named" not in caplog.text
        assert "Comprehensive validation failed" not in caplog.text
        assert results

    async def test_orchestrator_fast_path_runs_without_a_models_stub(self):
        from services.pr_validation.pr_validation_orchestrator import PRValidationOrchestrator

        orch = PRValidationOrchestrator()
        with patch.dict("sys.modules", _without_models_modules(), clear=True):
            out = await orch.validate_pr_with_actions(
                repo_owner="o",
                repo_name="r",
                pr_number=1,
                pr_files=[_real_file()],
                pr_metadata=None,
                trigger_comprehensive=False,
            )
        assert "No module named" not in str(out)
        assert out.get("status") != "error", out

    def test_no_src_import_resolves_only_under_a_foreign_sys_path(self):
        """Every absolute import in ``src`` must resolve from this service's own
        packages or an installed distribution. ``git`` and ``chardet`` are third-party
        names that are simply not dependencies; they are listed so this test catches
        a NEW unresolved name, and are reported separately."""
        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        known_missing_third_party = {"git", "chardet"}
        unresolved: list[str] = []
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in known_missing_third_party:
                        continue
                    if importlib.util.find_spec(top) is None:
                        unresolved.append(f"{path.relative_to(src)}:{node.lineno} {name}")
        assert unresolved == []
