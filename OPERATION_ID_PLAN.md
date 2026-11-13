# Operation ID Plan: coding-github

## Note
`coding-client` was originally requested but it is a shared library with no routes.
This plan covers `coding-github` instead.

## Files to Edit
- `src/main.py` (2 app-level routes)
- `src/routers/github_repository_management.py` (4 routes, 1 stacked decorator = 5 decorators)
- `src/routers/github_pull_request_operations.py` (4 routes)
- `src/routers/github_repository_ops.py` (4 routes)
- `src/routers/github_webhooks.py` (1 route)
- `src/routers/pr_actions.py` (2 routes)

## Mapping Table

### main.py (app-level, no prefix)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/health` | `health` | `health_check` |
| GET | `/` | `root` | `get_root` |

### github_repository_management.py (prefix `/api/repositories` via github.py -> main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/repositories/provision` | `provision_repository` | `provision_repository` |
| GET | `/api/repositories/` | `list_repositories` | `list_repositories_root` |
| GET | `/api/repositories/repositories` | `list_repositories` | `list_repositories` |
| GET | `/api/repositories/status/{container_name}` | `get_provisioning_status` | `get_provisioning_status` |
| POST | `/api/repositories/create-upstream-pr` | `create_upstream_pull_request` | `create_upstream_pull_request` |

### github_pull_request_operations.py (prefix `/api/repositories` via github.py -> main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/repositories/fork` | `fork_repository` | `fork_repository` |
| POST | `/api/repositories/apply-changes` | `apply_code_changes` | `apply_code_changes` |
| POST | `/api/repositories/create-pull-request` | `create_pull_request` | `create_pull_request` |
| POST | `/api/repositories/validate-pull-request` | `validate_pull_request` | `validate_pull_request` |

### github_repository_ops.py (prefix `/api/repositories` via github.py -> main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/repositories/{container_name}/file` | `read_repository_file` | `read_repository_file` |
| GET | `/api/repositories/{container_name}/tree` | `get_repository_tree` | `get_repository_tree` |
| GET | `/api/repositories/{container_name}/directory` | `list_directory_contents` | `list_directory_contents` |
| POST | `/api/repositories/{container_name}/search` | `search_repository` | `search_repository` |

### github_webhooks.py (prefix `/api/repositories/webhooks` via github.py -> main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/repositories/webhooks/webhook` | `github_webhook_handler` | `handle_github_webhook` |

### pr_actions.py (prefix `/api/webhooks/pr-actions` via main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/webhooks/pr-actions/process-validation` | `process_pr_validation` | `process_pr_validation` |
| POST | `/api/webhooks/pr-actions/validation-webhook` | `validation_webhook` | `trigger_validation_webhook` |

## Uniqueness Check

All 17 operation_ids are unique:
`health_check`, `get_root`, `provision_repository`, `list_repositories_root`,
`list_repositories`, `get_provisioning_status`, `create_upstream_pull_request`,
`fork_repository`, `apply_code_changes`, `create_pull_request`,
`validate_pull_request`, `read_repository_file`, `get_repository_tree`,
`list_directory_contents`, `search_repository`, `handle_github_webhook`,
`process_pr_validation`, `trigger_validation_webhook`

## Execution Steps
```bash
cd /Users/mini1/Movies/code_graph_helper/coding-github && git pull --rebase
```
1. Add `operation_id=` to every `@router` / `@app` decorator in the files listed above
2. For the stacked decorator on `list_repositories`, add `operation_id` to both `@router.get` lines
3. Validate uniqueness (no duplicates across all files)
4. `pre-commit run --all-files`
5. `pytest`
6. `git commit` and `push`
