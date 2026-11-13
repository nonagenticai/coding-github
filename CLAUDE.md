# coding-github — project instructions

Part of the `coding/` microservice bundle (see `../CLAUDE.md`). API routes are
`/api/<resource>` — **never** `/api/v1/`. Service port 8692.

## ⛔ `checks/` holds CONVEYOR FIXTURES — do not fix them by hand

`checks/` contains standalone oracles for defects the **AA conveyor** drives at.
The defect each one detects is **deliberately still present in `src/`**. A card in
the conveyor targets it, provisions a workspace, runs the oracle red, generates a
fix and opens a PR. Fixing the defect by hand discharges the card and destroys the
only evidence that the machine can do the work.

This is not hypothetical. On 2026-09-15 a sibling session hand-fixed
`enterprise_mcp_server/checks/http_status_binding.py`'s defect ten minutes after the
oracle merged, explicitly closing the conveyor's card. The work itself was good — it
also wired the oracles into CI, which a PR audit had flagged as a gap — but it proved
nothing about the machine, which is what the card existed to test.

**Hardening an oracle, or wiring `checks/` into CI, is welcome and does not consume
the target.** Only editing the defective code in `src/` does.

If you believe a fixture must be fixed by hand — a real incident, a security
exposure — say so explicitly in the PR body and name the card you are discharging.

### Why `checks/` and not `tests/`

`pyproject.toml` sets `testpaths = ["tests"]`, so pytest never collects `checks/`.
A deliberately-failing oracle under `tests/` would turn `main` red for everyone;
under `checks/` it is invisible to CI and executable on demand. `addopts` also
carries `--cov-fail-under=85`, so any pytest-based oracle must pass `--no-cov`.

Run one directly:

```bash
.venv/bin/python -m checks.diff_parser_rename_copy    # exit 1 while the defect stands
```

Each oracle establishes a **positive control** first and exits **2** if the control
fails, so "the defect is present" and "the oracle could not run" can never look alike.

## Current fixtures

| oracle | defect | file |
|---|---|---|
| `checks/diff_parser_rename_copy.py` | rename/copy classification tests substring membership (`"rename" in line`, `"from" in line`) instead of the line prefix | `src/services/pr_validation/core/diff_parser.py:157-165` |
