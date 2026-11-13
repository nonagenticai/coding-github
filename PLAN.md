# coding-github — Extraction Plan

## Purpose
Most isolated domain. Owns all GitHub integration: repository operations, webhooks, PR validation, provisioning.

## Port: 8692

---

## Endpoints

| Route Prefix | Source Router | Source Endpoint File |
|---|---|---|
| `/api/repositories` | `github` | `src/app/api/endpoints/github.py` |
| `/api/webhooks/pr-actions` | `pr_actions` | `src/app/api/endpoints/pr_actions.py` |

## Services to Extract

| Monolith Service | Target Location |
|---|---|
| `services/github_integration_service.py` | `src/services/github_integration.py` |
| `services/github_provisioning_service.py` | `src/services/github_provisioning.py` |
| `services/github_webhook_service.py` | `src/services/github_webhook.py` |
| `github/*` | `src/services/github/` |
| `pr_validation/*` | `src/services/pr_validation/` |
| `core/github_container_provisioner.py` | `src/services/github_container_provisioner.py` |

## Duplication Resolution
- **Provisioning**: This service owns both `github_provisioning_service.py` AND `core/github_container_provisioner.py`. The container provisioner is GitHub-centric (creates repos, forks) so it belongs here. coding-containers calls this service's API.

## Dependencies
- coding-client (shared models, auth, DB)
- PyGithub (GitHub API)
