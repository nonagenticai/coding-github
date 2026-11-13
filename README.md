# coding-github

[![CI](https://github.com/nonagenticai/coding-github/actions/workflows/ci.yml/badge.svg)](https://github.com/nonagenticai/coding-github/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/coding-github)](https://github.com/nonagenticai/coding-github/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

GitHub integration service for the Code Graph Helper platform. Owns remote GitHub concerns: repository import and access validation, credentials, forks, commits to forks, pull requests, webhooks, and automated PR validation. Checked-out repository lifecycle and local code operations belong to `coding-containers`' canonical `/api/workspaces` API.

**Port:** 8692

## Why This Service Exists

The Code Graph Helper platform analyzes codebases in isolated repository workspaces. This service isolates the GitHub API dependency and handles remote-provider operations; `coding-containers` owns cloning and workspace lifecycle.

## Architecture

```
coding-github (8692)
  |
  |-- coding-containers (8691)  -- container lifecycle (create, list, delete)
  |-- coding-security (8693)    -- vulnerability scanning for webhook file analysis
  |-- coding-analysis (8688)    -- code quality analysis (TODO: not yet wired)
  |-- PostgreSQL (via PgBouncer 6432) -- persistent state
  |-- GitHub API (PyGithub)     -- repository operations, webhooks, PRs
```

## Authentication

Auth is provided by `coding-client` middleware, NOT Keycloak/JWT.

- **AuthMiddleware** (`coding_client.auth.middleware`) -- license-token introspection via Istio
- **TenantMiddleware** (`coding_client.auth.tenant`) -- extracts and validates `X-Tenant-ID` header
- All tenant-scoped routes use `Depends(require_tenant)` to enforce tenant context
- Controlled by `ENABLE_AUTH` env var (default: `true`)
- `/health` and `/` are exempt from auth
- CORS handled by FastAPI CORSMiddleware (Istio handles mesh-level CORS in production)

### Dual-Token Architecture

The service supports a dual-token workflow for private repositories:

1. **User token** -- provided during provisioning to validate access to private repos. Stored encrypted via `SecureTokenManager`. Used later to create upstream PRs back to the user's repo.
2. **System token** (`CODING_GITHUB_SYSTEM_GITHUB_TOKEN`) -- used for @nonagenticai organization operations: forking, committing, pushing, triggering GitHub Actions.

## API Routes

All routes are tenant-scoped (require `X-Tenant-ID` header via auth middleware).

### Remote Repository Operations (`/api/repositories`)

| Method | Path | Summary |
|--------|------|---------|
| `POST` | `/api/repositories/import` | Validate GitHub access and credentials, resolve branch/PR commit metadata, and create or rebuild a canonical workspace. |
| `POST` | `/api/repositories/create-upstream-pr` | Create a PR from the system fork (@nonagenticai) back to the user's private repo using stored encrypted token. Returns 201. |

### Pull Request Operations (`/api/repositories`)

| Method | Path | Summary |
|--------|------|---------|
| `POST` | `/api/repositories/fork` | Fork the repository associated with a provisioned container. |
| `POST` | `/api/repositories/apply-changes` | Apply file changes to a forked repo. Creates branch, commits atomically, pushes. Max 50 files. |
| `POST` | `/api/repositories/create-pull-request` | Create a PR from fork to original repo. |
| `POST` | `/api/repositories/validate-pull-request` | Validate a PR for security, performance, and quality. Returns score (0-100), pass/fail, recommendation (approve/needs_work/reject). Supports `comprehensive` and `advanced` types. |

### Issue Operations (`/api/repositories`)

Every route declares an explicit `operation_id`, because the generated
`@nonagentic/n8n-nodes-coding` node derives its operation names from exactly that field —
a route without one is reachable by curl and invisible to the AA conveyor.

| Method | Path | Summary |
|--------|------|---------|
| `POST` | `/api/repositories/create-issue` | Open an issue on the given repository, optionally labelled. |
| `POST` | `/api/repositories/list-issues` | List issues. **Pull requests are excluded** — GitHub models a PR as an issue, so an unfiltered listing answers with the open PRs. `count` is the number returned, not the repository total. |
| `POST` | `/api/repositories/get-issue` | Fetch one issue by number. Returns 400 if the number refers to a pull request. |
| `POST` | `/api/repositories/comment-on-issue` | Post a markdown comment on an existing issue. |
| `POST` | `/api/repositories/add-issue-labels` | Add labels to an **existing** issue; returns the issue with its resulting label set. Body: `{owner, repo, issue_number, labels: [str]}` — 1–20 names, each ≤ 50 chars. |
| `POST` | `/api/repositories/close-issue` | Close an issue, optionally commenting first. The comment is posted BEFORE the state change, so a failed comment leaves the issue open and the call retryable. |

`add-issue-labels` exists because `create-issue` accepts `labels` only at CREATION, which
left the conveyor's GitHub-issue intake stamping `conveyor:queued` / `conveyor:failed`
through a raw `POST /issues/{n}/labels` with a PAT holding no Issues permission at all.
These routes authenticate through the App-first ladder in `_github()`, and the App holds
`issues: write`.

Two behaviours of that route were **measured against the live GitHub API on 2026-09-17**,
because the intake re-runs and both decide whether a re-run is safe:

- **Re-adding a label the issue already carries is a 200 no-op** — the response is the
  label list, unchanged, with no duplicate and no error. The call is idempotent.
- **Adding a label the repository does not define CREATES it** on the repository (grey
  `ededed`, null description) and then attaches it. A typo'd name does not fail; it
  permanently grows the repo's label set. Callers wanting attach-only-if-defined must
  check `GET /repos/{o}/{r}/labels` themselves.

### Legacy Workspace Compatibility

The historical provision/list/status/file/tree/directory/search routes remain
callable for compatibility but are excluded from OpenAPI so generated clients
see one workspace surface. New callers use:

- `POST /api/repositories/import` when GitHub validation, secret resolution,
  branch/PR commit lookup, idempotency, or auto-indexing is required.
- `POST /api/workspaces/` for direct low-level workspace creation.
- `GET /api/workspaces/` and `GET /api/workspaces/{container_name}/status` to list and inspect it.
- `GET /api/workspaces/{container_name}/files` for both directory listing and file reads.
- `/api/workspaces/{container_name}/git/*`, `/exec`, dependency, and agent routes for local operations.

### Webhooks (`/api/repositories/webhooks`)

| Method | Path | Summary |
|--------|------|---------|
| `POST` | `/api/repositories/webhooks/webhook` | Receive GitHub webhook events (push, pull_request, issue_comment, review_comment, workflow_run). Verifies `X-Hub-Signature-256` HMAC-SHA256 with a constant-time compare; an EMPTY configured secret or a missing/wrong signature is refused with 401 (fail closed). On PR open/synchronize: runs hybrid validation (fast + triggers GitHub Actions). On push to main/master/develop: analyzes commits, updates code graph, checks for security issues. On PR closed and on `workflow_run` completed (default branch only): forwards a normalised outcome event to n8n — see "GitHub outcome forwarding" below. ⛔ NOT tenant-scoped: GitHub cannot send `X-Tenant-Id` or a licence token, so this route carries no `require_tenant` dependency (unlike every other route in this service) — the HMAC signature is the only credential. |

#### GitHub outcome forwarding (replaces "AA Observability - GitHub Outcome Poll")

George's ruling: no timer may write on a clock. A 30-minute n8n poll for
PR-merged / PR-closed-without-merge / post-merge CI results is replaced by
push: a verified webhook delivery forwards a normalised JSON event to n8n's
`aa-github-outcome` webhook the moment GitHub reports it.

* `pull_request` `action=closed` → `pr_merged` (if `merged`) or
  `pr_closed_unmerged`.
* `workflow_run` `action=completed` **on the repository's default branch
  only** → `ci_completed` (carries `conclusion`, `head_sha`).

Configured via `CODING_GITHUB_OUTCOME_FORWARD_URL` and
`CODING_GITHUB_OUTCOME_FORWARD_TOKEN` (sent as the `x-aa-github-outcome-token`
header, matching n8n's own `AA_GITHUB_OUTCOME_WEBHOOK_TOKEN` check on that
trigger). EITHER empty (both default empty) disables forwarding entirely.
The forward is best-effort and fire-and-forget (5s timeout, no retry — GitHub
already redelivers): n8n being down, slow, or erroring NEVER changes the 2xx
coding-github sends back to GitHub. See `src/services/outcome_forward.py`.

#### Webhook mode (`CODING_GITHUB_WEBHOOK_MODE`)

Default `outcome_only`: after HMAC verification (unchanged, 401 on failure) the
receiver acts ONLY on `pull_request` `closed` and `workflow_run` `completed`
(outcome forwarding above). Every other event/action (pull_request
opened/synchronize, push, issue_comment, workflow_job, check_run, check_suite,
ping, ...) returns `202 {"status":"ignored","reason":"webhook_mode=outcome_only"}`
and wakes no handler. `all` restores the legacy full dispatch (PR validation,
push analysis, comments); it must be set explicitly and reviewed. Unset, empty
or any unrecognised value (including `ALL`) means `outcome_only`.

### PR Actions (`/api/webhooks/pr-actions`)

| Method | Path | Summary |
|--------|------|---------|
| `POST` | `/api/webhooks/pr-actions/process-validation` | Process PR validation results: add labels, post comments, request changes, auto-close. |
| `POST` | `/api/webhooks/pr-actions/validation-webhook` | Webhook endpoint for CI/CD systems to trigger validation + automated actions on a PR. |

### Health and Info

| Method | Path | Summary |
|--------|------|---------|
| `GET` | `/health` | Returns service status (ok/degraded) and DB connection state. Exempt from auth. |
| `GET` | `/` | Service info: name, version, links to /docs and /health. Exempt from auth. |

## Configuration

All environment variables use the `CODING_GITHUB_` prefix (via pydantic-settings `env_prefix`).

| Variable | Default | Description |
|----------|---------|-------------|
| `CODING_GITHUB_DATABASE_URL` | `postgresql+psycopg://coding_github:pass@pgbouncer...:6432/coding_github` | PostgreSQL connection string (async psycopg) |
| `CODING_GITHUB_GITHUB_TOKEN` | (empty) | GitHub PAT for API calls |
| `CODING_GITHUB_GITHUB_APP_ID` | (empty) | GitHub App ID (alternative to PAT) |
| `CODING_GITHUB_GITHUB_APP_PRIVATE_KEY` | (empty) | GitHub App private key |
| `CODING_GITHUB_GITHUB_WEBHOOK_SECRET` | (empty) | Secret for webhook signature verification |
| `CODING_GITHUB_SYSTEM_GITHUB_TOKEN` | (empty) | System token for @nonagenticai operations |
| `CODING_GITHUB_SYSTEM_GITHUB_OWNER` | `nonagenticai` | System GitHub org/account |
| `CODING_GITHUB_CODING_CONTAINERS_URL` | `http://coding-containers:8691` | coding-containers service URL |
| `CODING_GITHUB_CODING_SECURITY_URL` | `http://coding-security:8693` | coding-security service URL |
| `CODING_GITHUB_CODING_ANALYSIS_URL` | `http://coding-analysis:8688` | coding-analysis service URL |
| `CODING_GITHUB_SECRETS_API_URL` | `https://secrets-api:8543` | Secrets API URL |
| `CODING_GITHUB_CA_CERT_PATH` | `/certs/ca-cert.pem` | CA cert for secrets API TLS |
| `CODING_GITHUB_WEBHOOK_MODE` | `outcome_only` | `outcome_only` or `all` (exact, lowercase); anything else = `outcome_only`. See "Webhook mode". |
| `CODING_GITHUB_OUTCOME_FORWARD_URL` | (empty) | n8n `aa-github-outcome` webhook URL. Empty disables outcome forwarding. |
| `CODING_GITHUB_OUTCOME_FORWARD_TOKEN` | (empty) | Token sent as the `x-aa-github-outcome-token` header to the URL above. Empty also disables forwarding (mirrors an empty URL). |
| `SERVICE_NAME` | `coding-github` | Service name for telemetry |
| `VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `production` | Environment name |
| `ENABLE_AUTH` | `true` | Enable auth + tenant middleware (from coding-client) |
| `ENABLE_DOCS` | `false` | Enable /docs and /redoc |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | OpenTelemetry collector endpoint |
| `CODING_GITHUB_FEATURE_*` | (unset) | Feature flags (boolean env vars) |

## Directory Structure

```
src/
  main.py                          -- FastAPI app, lifespan, error handlers
  config/
    settings.py                    -- Pydantic settings (CODING_GITHUB_ prefix), feature flags
  routers/
    github.py                      -- Aggregator router: includes all sub-routers under /api/repositories
    github_repository_management.py -- GitHub-aware import, hidden workspace façades, upstream PR
    github_pull_request_operations.py -- fork, apply-changes, create-pull-request, validate-pull-request
    github_repository_ops.py       -- hidden compatibility façades over /api/workspaces
    github_webhooks.py             -- GitHub webhook receiver with signature verification
    pr_actions.py                  -- process-validation, validation-webhook (/api/webhooks/pr-actions)
    github_authentication.py       -- Token/URL validation utilities (no routes)
    github_api_integration.py      -- Container lookup, PR validation helpers (no routes)
    github_validation_utils.py     -- Security/performance/quality PR validators (no routes)
  services/
    github_integration.py          -- PyGithub wrapper: fork, branch, file CRUD, PR creation (dual-token)
    github_provisioning.py         -- GitHub-aware import orchestration + workspace HTTP adapter
    github_container_provisioner.py -- Dual-token provisioning: validates repo, stores user token
    github_validation.py           -- Repository/branch/PR access validation via GitHub API
    github_token_manager.py        -- Dual-token architecture: system + per-user encrypted tokens
    github_integration_core.py     -- Facade hub: clone, sync, PR analysis, webhook handling
    github_actions_trigger.py      -- Triggers GitHub Actions workflows (comprehensive PR validation)
    github/
      github_client.py             -- Low-level GitHub API client
      github_service.py            -- Async GitHub service (config, PR comments, commit files)
      secure_webhook_handler.py    -- HMAC-verified webhook handler
      webhook_handler.py           -- Base webhook handler
      auth_manager.py              -- GitHub auth management
      secure_token_manager.py      -- Encrypted token storage
      repo_operations.py           -- Repository operations
    pr_validation/
      pr_validation_orchestrator.py -- Two-phase validation orchestration (fast + comprehensive)
      actions/                     -- PR auto-closer, label management
      analyzers/                   -- Security, performance, quality analyzers
      config/                      -- Validation configuration
      core/                        -- Core validation logic
      integration/                 -- GitHub validation service integration
      scoring/                     -- Validation scoring
  schemas/
    github_api_models.py           -- All Pydantic request/response models
  interfaces/
    webhook_interface.py           -- Webhook event handler registry
  utils/
    github_utils.py                -- URL parsing, token validation
    github_error_handling.py       -- Error-to-HTTP-status mapping
    github_validation_helpers.py   -- Container-tenant validation
    github_error_handler.py        -- Error handler utilities
    file_classifier.py             -- File type classification
```

## Dependencies

### Internal Services
- **coding-containers** (8691) -- canonical repository workspace lifecycle and local operations
- **coding-security** (8693) -- vulnerability scanning of changed files during webhook processing
- **coding-analysis** (8688) -- code quality analysis (wiring incomplete, marked TODO)
- **coding-client** (pip package) -- shared models (`ContainerInfo`, `GitHubRepoRequest`), auth middleware, HTTP clients, exceptions

### External
- **GitHub API** via PyGithub -- repository operations, PR management, webhook processing
- **PostgreSQL** -- persistent state (via SQLAlchemy async + psycopg, through PgBouncer)

### Python Dependencies (key)
- `fastapi`, `uvicorn`, `gunicorn` -- web framework
- `PyGithub` -- GitHub API client
- `sqlalchemy[asyncio]`, `psycopg[binary,pool]` -- database
- `httpx`, `aiohttp` -- async HTTP
- `pydantic`, `pydantic-settings` -- data validation and config
- `opentelemetry-*` -- distributed tracing (optional)
- `coding-client` -- shared client library (pinned git dependency)

## Development

```bash
# Install dependencies
uv sync

# Run locally
uv run uvicorn src.main:app --host 0.0.0.0 --port 8692 --reload

# Run tests
uv run pytest

# Lint and type check
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/

# Docker
docker compose up -d
```

**Dev tools:** ruff (lint + format) + ty (type checking). See `pyproject.toml` for config.
**Python:** 3.14+ required.

## CI/CD

GitHub Actions workflow: `ruff + ty → build amd64 → smoke test → push`

Image pushed to `192.168.1.114:5000/coding-github:sha-<commit>`, promoted to `:latest` on success. Flux auto-deploys to K3s.

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

## Contributing

Issues and pull requests are welcome. See
[`CONTRIBUTING.md`](https://github.com/nonagenticai/.github/blob/main/CONTRIBUTING.md) for the
organisation-wide rules; setup, test and lint commands are in this README. Commits carry DCO
sign-off.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
