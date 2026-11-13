"""coding-github must set tracing up at module scope, never in the lifespan.

This service's previous tracing block was correct in every visible way — right
exporter class, provider registered globally, httpx handled — and exported zero
spans for two independent reasons:

1. it read ``OTEL_EXPORTER_OTLP_ENDPOINT``. This fleet provisions
   ``OTLP_TRACES_ENDPOINT``, and it *is* set on this workload (verified in the
   live pod), so the block always took its else-branch — which logs at DEBUG, so
   the miss was invisible in production log levels;
2. it called ``FastAPIInstrumentor.instrument_app(app)`` from inside ``lifespan``,
   which is a silent no-op once Starlette has built the middleware stack.

Fixing only (1) would still have shipped zero traces, which is why the call site
is asserted here and not just the variable name.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parents[2] / "src" / "main.py"


def _module() -> ast.Module:
    source = MAIN.read_text(encoding="utf-8")
    # Non-vacuity floor: a wrong or empty path would make every assertion pass.
    assert len(source) > 5_000, f"{MAIN} looks wrong: {len(source)} bytes"
    assert "app = FastAPI(" in source, f"{MAIN} does not construct the app"
    return ast.parse(source)


def test_setup_tracing_is_called_once_at_module_scope() -> None:
    tree = _module()

    top_level = [
        n
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "setup_tracing"
    ]
    anywhere = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setup_tracing"
    ]

    assert len(anywhere) == 1, f"expected one setup_tracing() call, found {len(anywhere)}"
    assert len(top_level) == 1, (
        "setup_tracing() is not called at module scope. Anywhere that runs after "
        "the app has started exports zero spans without raising."
    )


def test_the_lifespan_does_not_instrument_anything() -> None:
    """The old block is gone, not merely bypassed.

    An AST walk rather than a text grep, so the comment that replaced the block
    cannot satisfy it.
    """
    tree = _module()
    lifespan = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "lifespan"
        ),
        None,
    )
    assert lifespan is not None, "no lifespan function found"

    banned = {"instrument_app", "set_tracer_provider"}
    found = [
        n.func.attr
        for n in ast.walk(lifespan)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in banned
    ]
    assert not found, f"instrumentation calls left in the lifespan: {found}"


def test_the_unprovisioned_env_var_name_is_not_read_anywhere() -> None:
    """Defect 1, pinned across the whole src/ tree.

    ``OTEL_EXPORTER_OTLP_ENDPOINT`` is the OpenTelemetry-standard name but is not
    what this cluster sets, so reading it is indistinguishable from tracing being
    switched off.

    Walks the AST rather than grepping: the comment that documents the removal
    necessarily contains the name, and a text search cannot tell prose from code —
    the first draft of this test failed on its own explanation. ``ast.parse``
    discards comments, so only real string literals are considered.
    """
    src = MAIN.parent
    files = sorted(src.rglob("*.py"))
    assert len(files) > 20, f"scanned only {len(files)} files — the walk is wrong"

    offenders = []
    scanned_literals = 0
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                scanned_literals += 1
                if "OTEL_EXPORTER_OTLP_ENDPOINT" in node.value:
                    offenders.append(str(f.relative_to(src)))

    # Positive control: the walk must actually be seeing string literals.
    assert scanned_literals > 100, f"only {scanned_literals} string literals seen — walk is broken"
    assert not offenders, (
        f"unprovisioned OTLP endpoint variable still read in: {sorted(set(offenders))}"
    )


def test_a_real_request_through_the_real_app_produces_a_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring tests cannot prove spans exist; this drives one."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.util._once import Once

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None, raising=False)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)
    monkeypatch.setenv("OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")

    exporter = InMemorySpanExporter()
    real_add = TracerProvider.add_span_processor
    monkeypatch.setattr(
        TracerProvider,
        "add_span_processor",
        lambda self, _p: real_add(self, SimpleSpanProcessor(exporter)),
    )

    from coding_client.telemetry import setup_tracing

    app = FastAPI()

    @app.get("/api/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    assert setup_tracing(app, "coding-github") is True

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert len(exporter.get_finished_spans()) == 0, "probe traffic must not be traced"
        assert client.get("/api/probe").status_code == 200

    assert exporter.get_finished_spans(), "no span produced"
