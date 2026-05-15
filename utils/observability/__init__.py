"""Observability primitives — traces, spans, LLM usage, and cost.

Why this package exists
-----------------------
Build #1 (the eval harness) tells you *if* the agent passed a case.
This package tells you *why* — what each step did, how long it took,
how many tokens it burned, and how much it cost.

Public API
----------
::

    from utils.observability import (
        TraceContext, current_trace, use_trace,    # tracing
        LLMUsage, cost_usd_for, get_price,         # cost
        render_trace, dump_trace_json,             # reporting
    )

Typical usage::

    ctx = TraceContext.start(session_id="sess_1")
    with use_trace(ctx):
        with ctx.span("intent.classify") as span:
            intent = classify(question)
            span["intent"] = intent.name

        with ctx.span("llm.narrate"):
            agent.generate_insight(...)        # auto-records LLMUsage

    print(render_trace(ctx))

Design constraints
------------------
* **No-op friendly** — code that calls ``current_trace()`` outside an
  active trace gets ``None`` and skips the recording cleanly. Adding
  observability never breaks an un-instrumented call site.
* **Zero extra deps** — pure stdlib + the project's existing loguru.
* **Cheap when off** — span objects are not allocated unless a trace
  is active. The LLM call cost path is one ``ContextVar.get()`` plus
  one ``None`` check.
"""

from .pricing import (
    DEFAULT_PRICE,
    ModelPrice,
    PRICING,
    cost_usd_for,
    get_price,
)
from .reporter import dump_trace_json, render_trace
from .trace import (
    LLMUsage,
    Span,
    TraceContext,
    current_trace,
    use_trace,
)

__all__ = [
    "DEFAULT_PRICE",
    "LLMUsage",
    "ModelPrice",
    "PRICING",
    "Span",
    "TraceContext",
    "cost_usd_for",
    "current_trace",
    "dump_trace_json",
    "get_price",
    "render_trace",
    "use_trace",
]
