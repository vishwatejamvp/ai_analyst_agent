"""LLM-driven chart planner — function calling done correctly.

The mental model
----------------
The existing :class:`ChartService` knows *how* to render a chart given
a typed spec. This module decides *whether* and *how* to render — the
"chart selection" decision that previously lived in
:func:`select_chart_type` as deterministic rules.

Two tools, the model picks one:

* ``render_chart`` — visualization adds value; the model returns a
  full :class:`RenderChartArgs` (chart_type, x, y, title, reason).
* ``skip_chart``   — text-only answer is more honest; the model
  returns just a :class:`SkipChartArgs` (reason).

Why a separate "skip" tool?
---------------------------
Giving the model an explicit "do nothing, here's why" option makes the
decision auditable and forces it to commit. Without an explicit skip
tool, models tend to render mediocre charts just to feel useful.

Safety boundary (Layer 3 of the agent stack)
--------------------------------------------
The model only *requests* a chart spec. The runtime executes it via
:class:`ChartService.render`, which is unchanged. If the model picks a
field that doesn't exist on the data, we **fall back to rule-based
selection** rather than crashing — the agent's job is to be useful,
not to be correct at every step.

Cost discipline
---------------
This adds **one extra Claude call per analytical answer**. The build
#2 trace context will report it; if the cost-per-question becomes
material, switch ``chart_decider`` back to ``rule``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from models.enums import ChartType
from utils.config import settings
from utils.exceptions import LLMError
from utils.logger import logger
from utils.observability import current_trace


# ---------------------------------------------------------------------------
# Tool argument schemas (Pydantic → JSON Schema → Anthropic tool input)
# ---------------------------------------------------------------------------
class RenderChartArgs(BaseModel):
    """Arguments the LLM passes when it chooses ``render_chart``.

    Field descriptions are intentionally prompt-shaped — they are part
    of the JSON schema sent to Claude and steer how the model fills
    each field.
    """

    chart_type: Literal["bar", "line", "pie", "kpi"] = Field(
        ...,
        description=(
            "bar = compare categories or small group counts; "
            "line = trends over time; "
            "pie = parts of a single whole (rare; use only when categories are <=6 and sum is meaningful); "
            "kpi = a single headline number."
        ),
    )
    x_field: str = Field(
        ...,
        description=(
            "Name of the column to plot on the horizontal axis. Must be one "
            "of the column names listed under DATA COLUMNS in the prompt. "
            "Use 'label' for the canonical period/category column."
        ),
    )
    y_field: str = Field(
        ...,
        description=(
            "Name of the numeric column for the vertical axis. Must be one "
            "of the columns listed. Use 'value' for the canonical metric column."
        ),
    )
    title: str = Field(
        ...,
        max_length=80,
        description="Short, business-readable chart title (no metric ids).",
    )
    reason: str = Field(
        ...,
        max_length=200,
        description="One sentence: why this chart type fits the question + data.",
    )


class SkipChartArgs(BaseModel):
    """Arguments the LLM passes when it chooses ``skip_chart``."""

    reason: str = Field(
        ...,
        max_length=200,
        description=(
            "One sentence: why no chart adds value here. Examples: "
            "'Question asks for an explanation; numbers alone are not the answer.', "
            "'Single value already shown in narrative.'"
        ),
    )


# ---------------------------------------------------------------------------
# Decision returned to the orchestrator
# ---------------------------------------------------------------------------
class ChartToolDecision(BaseModel):
    """Unified result the orchestrator consumes."""

    action: Literal["render", "skip", "fallback"]
    chart_type: ChartType | None = None
    x_field: str | None = None
    y_field: str | None = None
    title: str | None = None
    reason: str = ""

    @classmethod
    def render(cls, args: RenderChartArgs) -> "ChartToolDecision":
        return cls(
            action="render",
            chart_type=ChartType(args.chart_type),
            x_field=args.x_field,
            y_field=args.y_field,
            title=args.title,
            reason=args.reason,
        )

    @classmethod
    def skip(cls, reason: str) -> "ChartToolDecision":
        return cls(action="skip", reason=reason)

    @classmethod
    def fallback(cls, reason: str) -> "ChartToolDecision":
        """Returned when the LLM call fails — caller falls back to rule-based."""
        return cls(action="fallback", reason=reason)


# ---------------------------------------------------------------------------
# Tool schemas in Anthropic's expected shape
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "render_chart",
        "description": (
            "Render a chart for the structured rows shown in the prompt. "
            "Choose this when a visual would meaningfully aid understanding "
            "(trends over time, comparisons across categories, distributions). "
            "x_field and y_field MUST be column names that appear under "
            "'DATA COLUMNS' in the prompt; do not invent fields."
        ),
        "input_schema": RenderChartArgs.model_json_schema(),
    },
    {
        "name": "skip_chart",
        "description": (
            "Skip the chart entirely when no visualization adds value: the "
            "user asked for an explanation, the result is a single number "
            "already stated in the narrative, the data is too sparse, or "
            "the question is purely qualitative."
        ),
        "input_schema": SkipChartArgs.model_json_schema(),
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a chart-selection assistant for a business analyst app. "
    "You will receive a user question, a short summary of the analytical "
    "plan that produced the data, and the column list + a small sample "
    "of rows.\n\n"
    "Your ONLY job is to decide whether to render a chart and, if so, "
    "what kind. You do this by calling exactly ONE of two tools:\n"
    "  - render_chart(...) — when a chart would help the reader.\n"
    "  - skip_chart(reason) — when text alone is more honest.\n\n"
    "Decision rules (apply in order):\n"
    "1. If the question is purely qualitative ('why', 'explain', "
    "'describe'), call skip_chart.\n"
    "2. If the data has 1 row and the answer is one number, prefer kpi "
    "(or skip_chart when the narrative already states the number).\n"
    "3. If the data has a time dimension, prefer line; bar is acceptable "
    "for short series (<=6 buckets).\n"
    "4. If the data is grouped by a categorical dimension (no time), "
    "prefer bar.\n"
    "5. Pie ONLY when the categories sum to a meaningful whole AND there "
    "are 2-6 slices. Otherwise prefer bar.\n"
    "6. Never invent column names. Use 'label' for the canonical x and "
    "'value' for the canonical y unless the data clearly uses other names.\n\n"
    "You MUST call exactly one tool. Do not return free-text — the runtime "
    "ignores text and only consumes tool_use blocks."
)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class LLMChartPlanner:
    """One Claude call → one tool decision → one ChartToolDecision."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ------------------------------------------------------------------
    # Lazy client (mirrors agent_service to avoid re-init cost)
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured.")
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            raise LLMError(f"Failed to initialise Anthropic client: {exc}") from exc
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self,
        *,
        question: str,
        rows: list[dict[str, Any]],
        plan_summary: str,
    ) -> ChartToolDecision:
        """Ask Claude which chart (if any) fits.

        Returns :class:`ChartToolDecision` with ``action`` in
        ``render | skip | fallback``. ``fallback`` means "the LLM call
        failed — caller should use rule-based selection instead". The
        planner never raises into the orchestrator; observability is
        free because LLM tokens flow into the active trace via
        ``agent_service``-style ``current_trace()`` recording.
        """
        if not rows:
            return ChartToolDecision.skip("No data rows to plot.")

        prompt = self._compose_user_prompt(question, rows, plan_summary)

        try:
            decision = self._call_claude(prompt)
        except LLMError as exc:
            logger.warning(f"Chart-tool LLM failed; falling back: {exc}")
            return ChartToolDecision.fallback(str(exc))
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            # The orchestrator must never crash on chart selection — any
            # unexpected failure (timeout, JSON parse, schema drift) downgrades
            # to rule-based selection rather than failing the whole answer.
            logger.warning(f"Chart-tool unexpected error; falling back: {exc}")
            return ChartToolDecision.fallback(f"{type(exc).__name__}: {exc}")
        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _call_claude(self, user_prompt: str) -> ChartToolDecision:
        client = self._get_client()
        t0 = time.perf_counter()

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Cost capture — always before any return path so failed-parse
        # cases still appear in the trace as "we paid for this call".
        usage = getattr(message, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        trace = current_trace()
        if trace is not None:
            trace.record_llm(
                model=self.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                span_name="chart.plan.llm",
            )

        return self._parse_tool_use(message)

    @staticmethod
    def _parse_tool_use(message: Any) -> ChartToolDecision:
        """Pull the first tool_use block out of a Claude response.

        Validates the input dict against the matching Pydantic model so
        any schema drift (Anthropic API quirks, model hallucinating an
        extra field) is caught here, not three layers deeper in the
        renderer.
        """
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            raw = getattr(block, "input", None) or {}
            if name == "render_chart":
                try:
                    args = RenderChartArgs.model_validate(raw)
                except ValidationError as exc:
                    logger.warning(
                        f"render_chart args invalid; falling back: {exc}"
                    )
                    return ChartToolDecision.fallback(
                        f"render_chart arg validation: {exc.errors()[:1]}"
                    )
                return ChartToolDecision.render(args)
            if name == "skip_chart":
                try:
                    args = SkipChartArgs.model_validate(raw)
                except ValidationError as exc:
                    return ChartToolDecision.skip(f"(invalid skip args: {exc})")
                return ChartToolDecision.skip(args.reason)
            logger.warning(f"Unknown tool_use name: {name!r}")
        return ChartToolDecision.fallback("no tool_use block in response")

    @staticmethod
    def _compose_user_prompt(
        question: str,
        rows: list[dict[str, Any]],
        plan_summary: str,
    ) -> str:
        sample = rows[: min(5, len(rows))]
        cols: list[str] = sorted({k for r in rows for k in r.keys()})
        return (
            f"USER QUESTION:\n{question}\n\n"
            f"PLAN SUMMARY:\n{plan_summary}\n\n"
            f"DATA SHAPE:\n"
            f"- rows: {len(rows)}\n"
            f"- columns: {cols}\n\n"
            f"DATA SAMPLE (up to 5 rows):\n"
            f"{json.dumps(sample, indent=2, default=str)}"
        )


llm_chart_planner = LLMChartPlanner()
