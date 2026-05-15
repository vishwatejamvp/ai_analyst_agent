"""Trace context, spans, and LLM-usage records.

The mental model
----------------
A **TraceContext** is one request's story. It owns:

* a flat list of :class:`Span` objects (one per timed unit of work),
* a flat list of :class:`LLMUsage` records (one per Claude call),
* a parent stack so nested spans know who created them.

A **Span** is a single timed unit of work. You enter it with
``with ctx.span("name"): ...`` — entry stamps ``started_at``, exit
fills ``duration_ms`` and (if the block raised) ``error``.

A **LLMUsage** is one Claude call's metering: model, tokens in/out,
cost in USD, latency, and the span it was recorded under.

The contextvar pattern
----------------------
We expose the *current* trace via a :class:`contextvars.ContextVar`.
This means deep code (e.g. ``agent_service._call_claude``) can record
LLM usage by calling ``current_trace()`` — no need to thread the trace
through every function signature. When no trace is active (e.g. unit
test, ad-hoc script) ``current_trace()`` returns ``None`` and the
recording is silently skipped. **Adding observability never breaks an
un-instrumented call site.** This is the key safety property.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .pricing import cost_usd_for


# ---------------------------------------------------------------------------
# ContextVar for the active trace
# ---------------------------------------------------------------------------
_current_trace: ContextVar["TraceContext | None"] = ContextVar(
    "current_trace", default=None
)


def current_trace() -> "TraceContext | None":
    """Return the active :class:`TraceContext`, or ``None`` if none is set.

    Code that wants to be observability-aware should::

        trace = current_trace()
        if trace is not None:
            trace.record_llm(...)

    The ``None``-check is the contract: never assume a trace is active.
    """
    return _current_trace.get()


@contextmanager
def use_trace(ctx: "TraceContext") -> Iterator["TraceContext"]:
    """Activate ``ctx`` as the current trace for the scope of the ``with``.

    Restores the prior trace (or ``None``) on exit, so nested ``use_trace``
    blocks compose cleanly — useful when an outer caller starts a trace
    and an inner test wants to start its own.
    """
    token: Token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        _current_trace.reset(token)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class LLMUsage:
    """One Claude call's accounting record."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float | None = None
    span_name: str | None = None


@dataclass
class Span:
    """One timed unit of work.

    Use ``span["key"] = value`` to attach attributes during the block —
    these end up in the trace dump and the pretty report.
    """

    name: str
    started_at: float
    duration_ms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent: str | None = None

    # Dict-like access so call sites read like ``span["target"] = ...``
    def __setitem__(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.attrs[key]

    def set(self, **kwargs: Any) -> None:
        """Set multiple attributes at once: ``span.set(target="x", rows=5)``."""
        self.attrs.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": (
                round(self.duration_ms, 2) if self.duration_ms is not None else None
            ),
            "parent": self.parent,
            "attrs": self.attrs,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------
@dataclass
class TraceContext:
    """One request's trace — spans + LLM usage + totals."""

    request_id: str
    session_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    llm_usages: list[LLMUsage] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def start(
        cls,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> "TraceContext":
        """Create a fresh trace. Generates a short ``req_<hex>`` id by default."""
        return cls(
            request_id=request_id or f"req_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
        )

    # ---------------------------------------------------------------
    # Span management
    # ---------------------------------------------------------------
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Open a timed span. Use as ``with ctx.span("router.decide"): ...``.

        Initial attrs can be passed as keyword args; more can be added
        inside the block via ``span["..."] = ...`` or ``span.set(...)``.

        Re-raises any exception raised inside the block AFTER recording
        the error on the span — diagnostics survive even on failure paths.
        """
        span = Span(
            name=name,
            started_at=time.perf_counter(),
            attrs=dict(attrs),
            parent=self._stack[-1] if self._stack else None,
        )
        self._stack.append(name)
        t0 = time.perf_counter()
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 — record then re-raise
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - t0) * 1000.0
            self._stack.pop()
            self.spans.append(span)

    # ---------------------------------------------------------------
    # LLM usage
    # ---------------------------------------------------------------
    def record_llm(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float | None = None,
        span_name: str | None = None,
    ) -> LLMUsage:
        """Record one LLM call's tokens + cost. Returns the stored usage."""
        usage = LLMUsage(
            model=model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cost_usd=cost_usd_for(model, input_tokens, output_tokens),
            latency_ms=latency_ms,
            span_name=span_name or (self._stack[-1] if self._stack else None),
        )
        self.llm_usages.append(usage)
        return usage

    # ---------------------------------------------------------------
    # Aggregates
    # ---------------------------------------------------------------
    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.llm_usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.llm_usages)

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.llm_usages)

    @property
    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    # ---------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Compact dict suitable for response.meta even when details=False."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "duration_ms": round(self.total_duration_ms, 2),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "llm_calls": len(self.llm_usages),
            "spans": len(self.spans),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full trace dump — spans + LLM usages + totals."""
        return {
            **self.summary(),
            "spans": [s.to_dict() for s in self.spans],
            "llm_usages": [asdict(u) for u in self.llm_usages],
        }
