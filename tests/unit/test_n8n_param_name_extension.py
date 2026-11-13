"""``x-n8n-param-name`` reaches the published OpenAPI spec for list-workflow-runs.

The n8n node generator (n8n_module_scripts generate-node.ts) flattens every body
field of a resource into ONE parameter namespace, and the first op seen supplies
the definition. ``ListPullRequestFilesRequest.per_page`` (default 100) and
``ListWorkflowRunsRequest.per_page`` (default 30) therefore became one shared
``perPage`` defaulting to 100, so an untouched listWorkflowRuns sent per_page=100
(n8n-nodes-coding#42). The extension must survive into ``app.openapi()``, which is
what the aggregator and generator read; asserting on the model alone would not
prove that.
"""

from __future__ import annotations

from typing import Any


def _spec() -> dict[str, Any]:
    from main import app

    app.openapi_schema = None  # force a fresh build, never a cached spec
    return app.openapi()


def _request_schema(spec: dict[str, Any], operation_id: str) -> dict[str, Any]:
    for path_item in spec["paths"].values():
        for op in path_item.values():
            if isinstance(op, dict) and op.get("operationId") == operation_id:
                ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
                assert ref.startswith("#/components/schemas/"), ref
                return spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    raise AssertionError(f"operationId {operation_id!r} not in the spec")


def test_list_workflow_runs_per_page_carries_n8n_param_name() -> None:
    prop = _request_schema(_spec(), "list_workflow_runs")["properties"]["per_page"]
    assert prop.get("x-n8n-param-name") == "workflowRunsPerPage"
    # The wire contract is unchanged: same key, same default and bounds.
    assert prop["default"] == 30
    assert prop["minimum"] == 1
    assert prop["maximum"] == 100


def test_list_pull_request_files_per_page_is_untouched() -> None:
    """Control: files' ``per_page`` keeps its plain name, or the rename is global."""
    prop = _request_schema(_spec(), "list_pull_request_files")["properties"]["per_page"]
    assert "x-n8n-param-name" not in prop
    assert prop["default"] == 100


def test_apply_changes_target_files_carries_n8n_param_name() -> None:
    """coding-analysis's scan request also has ``target_files``; keep them apart."""
    schema = _request_schema(_spec(), "apply_code_changes")
    assert "target_files" not in schema.get("required", [])
    prop = schema["properties"]["target_files"]
    assert prop.get("x-n8n-param-name") == "commitTargetFiles"
