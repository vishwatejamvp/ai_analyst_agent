"""Trace reporting — pretty terminal tree + JSON dump.

Two outputs, mirroring the eval harness:

* :func:`render_trace` — human-friendly tree-shaped report. ANSI colour
  by default; pass ``color=False`` for CI logs.
* :func:`dump_trace_json` — typed JSON dump for archival and diffing.

The terminal tree shows nesting via indentation (parent → child) and
attaches LLM usage to the span it occurred under, so a glance answers
"where did the time and money go?".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .trace import LLMUsage, Span, TraceContext


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade to plain when color=False)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


def _c(text: str, code: str, *, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_trace(ctx: TraceContext, *, color: bool = True) -> str:
    """Render ``ctx`` as a multi-line terminal report.

    Layout::

        TRACE req_abc12345  total 2453.1 ms · in 1245 tok · out 312 tok · $0.005010
          ├─ request.answer                          2453.1 ms
          │   question="average occupancy in 2024"
          │   ├─ router.decide                         312.4 ms
          │   │   target=occupancy_facts route=ANALYTICAL
          │   ├─ data.query                            432.1 ms
          │   │   rows=156
          │   └─ llm.narrate                          1248.3 ms
          │       LLM claude-3-5-sonnet · in 1245 / out 312 · $0.005010
    """
    lines: list[str] = []
    head = (
        f"TRACE {ctx.request_id}  "
        f"total {ctx.total_duration_ms:.1f} ms · "
        f"in {ctx.total_input_tokens} tok · "
        f"out {ctx.total_output_tokens} tok · "
        f"${ctx.total_cost_usd:.6f}"
    )
    lines.append(_c(head, _BOLD + _CYAN, color=color))
    if not ctx.spans:
        lines.append(_c("  (no spans recorded)", _DIM, color=color))
        return "\n".join(lines)

    children_of: dict[str | None, list[Span]] = defaultdict(list)
    for s in ctx.spans:
        children_of[s.parent].append(s)

    usages_by_span: dict[str | None, list[LLMUsage]] = defaultdict(list)
    for u in ctx.llm_usages:
        usages_by_span[u.span_name].append(u)

    roots = children_of.get(None, [])
    for i, root in enumerate(roots):
        _render_span(root, "", i == len(roots) - 1, lines, children_of, usages_by_span, color=color)
    return "\n".join(lines)


def dump_trace_json(path: Path, ctx: TraceContext) -> None:
    """Write the full trace as a JSON file. Diff-friendly, stable shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _render_span(
    span: Span,
    indent: str,
    is_last: bool,
    lines: list[str],
    children_of: dict[str | None, list[Span]],
    usages_by_span: dict[str | None, list[LLMUsage]],
    *,
    color: bool,
) -> None:
    branch = "└─" if is_last else "├─"
    next_indent = indent + ("    " if is_last else "│   ")

    duration = f"{span.duration_ms:>8.1f} ms" if span.duration_ms is not None else "    ?    "
    err_marker = (
        " " + _c(f"[error: {span.error}]", _RED, color=color) if span.error else ""
    )
    name_col = _c(f"{span.name:<32}", _BOLD, color=color)
    lines.append(f"  {indent}{branch} {name_col} {duration}{err_marker}")

    for k, v in span.attrs.items():
        lines.append(
            "  " + next_indent + _c(f"{k}={_fmt_val(v)}", _DIM, color=color)
        )

    for usage in usages_by_span.get(span.name, []):
        lat = f"{usage.latency_ms:.1f} ms" if usage.latency_ms is not None else "?"
        lines.append(
            "  " + next_indent + _c(
                f"LLM {usage.model} · in {usage.input_tokens} / out {usage.output_tokens} "
                f"· ${usage.cost_usd:.6f} · {lat}",
                _MAGENTA,
                color=color,
            )
        )

    children = children_of.get(span.name, [])
    for i, child in enumerate(children):
        _render_span(
            child, next_indent, i == len(children) - 1,
            lines, children_of, usages_by_span, color=color,
        )


def _fmt_val(v: Any) -> str:
    """Compact string for an attribute value — quotes strings, truncates long."""
    if isinstance(v, str):
        s = v if len(v) <= 60 else v[:57] + "..."
        return f'"{s}"'
    return str(v)
