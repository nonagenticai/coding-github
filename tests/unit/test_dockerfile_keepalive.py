"""Pin uvicorn's keep-alive above the gateway's upstream idle timeout.

With uvicorn's default 5s keep-alive, envoy (idle timeout 1h by default, 60s via
the gitops DestinationRule coding-github-idle) reuses a pooled connection that
uvicorn already closed, and the caller gets 503 "connection termination"
(observed on a GitHub webhook delivery, 2026-10-02, #131). The upstream must
outlive envoy's idle timeout, so the flag must stay strictly above 60.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"
GATEWAY_IDLE_TIMEOUT_S = 60


def _cmd_argv() -> list[str]:
    cmds = [ln for ln in DOCKERFILE.read_text().splitlines() if re.match(r"^CMD\s+\[", ln)]
    assert len(cmds) == 1, f"expected exactly one exec-form CMD, found {cmds}"
    return json.loads(cmds[0].split("CMD", 1)[1])


def test_uvicorn_keep_alive_outlives_gateway_idle_timeout():
    argv = _cmd_argv()
    assert argv[0] == "uvicorn"
    assert "--timeout-keep-alive" in argv, "uvicorn would fall back to its 5s default"
    value = int(argv[argv.index("--timeout-keep-alive") + 1])
    assert value > GATEWAY_IDLE_TIMEOUT_S
