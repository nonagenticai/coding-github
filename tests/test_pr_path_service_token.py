"""The PR-creation path must present coding-github's OWN licence, not a bare tenant.

#72 converted coding-github's sibling calls to `optional_sibling_headers` — but only
the webhook, provisioning and secret-resolver callers. The two on the **PR-creation**
path kept their hand-rolled `{"X-Tenant-ID": tenant_id}`:

    src/routers/github_pull_request_operations.py  ::  _resolve_container
    src/services/github_integration.py             ::  create_pull_request

Those are exactly the ones a dependency-upgrade card needs, so the gap was invisible
to #72's own tests and fatal to the only workflow that exercises it.

MEASURED LIVE 2026-08-14, driving `development_chore — IT Flow1 — bump redis to 8.1.0`
against `nonagenticai/enterprise_mcp_server`:

    coding-github  [error] service_error
      base_url=http://coding-containers…:8691 status_code=401
      body={'detail': 'Authentication required'}
      path=/api/containers/tour-1841261397355267865-c0
    Unexpected error creating upstream PR: … returned 401
    POST /api/repositories/create-upstream-pr → 500

    coding-security  RuntimeError: apply-changes failed:
      coding-github HTTP 500: … returned 401

The card reached BLOCKED with `completed 1/3 typed steps` and `pr_url: null`. It had
already done the real work — it found 8 CVE records and provisioned a container — and
then could not open the pull request.

coding-containers runs `TRUST_TENANT_HEADER=false`, so a header-only call is no longer
an identity: it 401s. `svc-coding-github` (`019ff971-40dc-…`) IS on that service's
`CODING_ONBEHALF_SUBJECTS`, so presenting the licence is all that was ever needed.

Assertions are on what reached the wire, and always on BOTH halves at once:
`Authorization` carries OUR identity, `X-Tenant-Id` still carries the CUSTOMER's.
Folding the licence into the tenant header would make the callee resolve the tenant
from our token and collapse every customer into svc-coding-github's own tenant.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PR_PATH_FILES = (
    SRC / "routers" / "github_pull_request_operations.py",
    SRC / "services" / "github_integration.py",
)


class TestNoBareTenantHeaderSurvives:
    """The defect, stated so it cannot come back by copy-paste.

    A source-level assertion rather than a behavioural one, deliberately: the two
    call sites live inside large async handlers whose collaborators (`main`, the
    token manager, a live containers client) would have to be stood up wholesale to
    reach the one line that matters. What must not regress is the LINE, and a grep is
    an honest guard for that — it fails the moment anyone re-hand-rolls the dict.
    """

    @pytest.mark.parametrize("path", PR_PATH_FILES, ids=lambda p: p.name)
    def test_no_hand_rolled_tenant_only_headers(self, path: Path) -> None:
        src = path.read_text()
        offenders = [
            line.strip()
            for line in src.splitlines()
            if '{"X-Tenant-ID"' in line or "{'X-Tenant-ID'" in line
        ]
        assert not offenders, (
            f"{path.name} hand-rolls a tenant-only header dict: {offenders}. "
            "coding-containers runs TRUST_TENANT_HEADER=false, so this 401s and the "
            "PR path dies. Use services.outbound.optional_sibling_headers(tenant_id)."
        )

    @pytest.mark.parametrize("path", PR_PATH_FILES, ids=lambda p: p.name)
    def test_uses_the_shared_helper(self, path: Path) -> None:
        src = path.read_text()
        assert "optional_sibling_headers" in src, (
            f"{path.name} must build sibling headers with the #72 helper so the "
            "licence and the tenant assertion stay together and stay separate."
        )


class TestHelperContract:
    """Pin what the helper must keep doing, since both call sites now depend on it."""

    def test_helper_sends_both_halves(self) -> None:
        import os

        from services.outbound import optional_sibling_headers, service_token

        os.environ["CODING_SERVICE_TOKEN"] = "test-licence"
        try:
            headers = optional_sibling_headers("0195f48b-d000-7000-8000-000000000003")
        finally:
            os.environ.pop("CODING_SERVICE_TOKEN", None)

        assert headers.get("Authorization") == "Bearer test-licence", (
            "the licence must ride in Authorization"
        )
        assert (
            headers.get("X-Tenant-Id") == "0195f48b-d000-7000-8000-000000000003"
            or headers.get("X-Tenant-ID") == "0195f48b-d000-7000-8000-000000000003"
        ), "the CUSTOMER's tenant must still ride in the tenant header"
        assert "0195f48b" not in headers.get("Authorization", ""), (
            "folding the tenant into the credential would collapse every customer "
            "into svc-coding-github's own tenant"
        )
        assert service_token.__doc__, "helper must stay documented"

    def test_helper_reads_the_token_at_call_time(self) -> None:
        """A module-level constant would pin whatever was set at first import and
        survive a secret rotation."""
        from services.outbound import service_token

        source = inspect.getsource(service_token)
        assert "getenv" in source, "token must be read from the environment per call"
