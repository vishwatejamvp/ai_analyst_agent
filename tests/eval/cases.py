"""Typed schemas for evaluation cases, expectations, scores, and results.

Design notes
------------
* ``Case`` is the *input* — what we want to test, declared in JSONL.
* ``Expect`` is the *contract* — every field is optional so a case only
  asserts what it cares about. A pure intent case sets ``intent``; a
  routing case sets ``target`` / ``operation`` / etc.
* ``Score`` is one (expected, actual) comparison from one scorer.
* ``Result`` aggregates all scores from running one case.

Keeping these as Pydantic models gives us:
* JSONL parsing for free (``Case.model_validate_json``)
* Schema validation when authors add malformed cases
* Self-documenting field descriptions (visible in IDEs)
"""

from __future__ import annotations

from enum import Enum
from typing import Any  # noqa: F401 — re-exported via dict[str, Any] in Result.trace

from pydantic import BaseModel, Field


class EvalLevel(str, Enum):
    """Which layer of the agent stack a case exercises.

    Cheaper levels first. The runner dispatches each case to the
    matching handler; ``--level`` filters which levels run.
    """

    INTENT = "intent"
    ROUTING = "routing"
    E2E = "e2e"


class Expect(BaseModel):
    """Per-case expectations. All fields optional — assert only what matters."""

    intent: str | None = Field(
        default=None,
        description="QuestionIntent value: DISCOVERY|OUT_OF_SCOPE|COMPARISON|ANALYTICAL.",
    )
    route: str | None = Field(
        default=None,
        description="QueryRoute value: ANALYTICAL|SEMANTIC|HYBRID|DISCOVERY|OUT_OF_SCOPE.",
    )
    target: str | None = Field(
        default=None,
        description="Mongo collection / MySQL table the router must pick.",
    )
    operation: str | None = Field(
        default=None,
        description="Aggregation operation: sum|avg|count|min|max.",
    )
    metric: str | None = Field(
        default=None,
        description="Aggregation metric column name.",
    )
    metric_in: list[str] | None = Field(
        default=None,
        description="Acceptable set of metric values (any one matches).",
    )
    group_by: str | None = None
    time_present: bool | None = Field(
        default=None,
        description="True/False: whether aggregation.time should be set.",
    )
    text_contains: list[str] | None = Field(
        default=None,
        description="Substrings that must appear in `insight` (case-insensitive).",
    )
    text_not_contains: list[str] | None = Field(
        default=None,
        description="Substrings that must NOT appear in `insight`.",
    )
    warning_codes_include: list[str] | None = Field(
        default=None,
        description="WarningCode values that must be present in response.warnings.",
    )
    warning_codes_exclude: list[str] | None = Field(
        default=None,
        description="WarningCode values that must NOT be present.",
    )
    data_nonempty: bool | None = Field(
        default=None,
        description="True: structured_data must have rows. False: must be empty.",
    )
    chart_type: str | None = Field(
        default=None,
        description="Expected ChartType value (BAR|LINE|PIE|KPI) on response.chart, "
        "or the literal 'SKIP' when no chart should be rendered.",
    )
    llm_calls_min: int | None = Field(
        default=None,
        description="Minimum number of LLM calls observed via the trace context. "
        "Useful when verifying that an extra tool-use call happened "
        "(e.g. chart_decider=llm should add 1 to the baseline).",
    )
    llm_calls_max: int | None = Field(
        default=None,
        description="Maximum allowed LLM calls. Useful for cost-budget regressions.",
    )
    reranked: bool | None = Field(
        default=None,
        description="True: at least one vector_context hit must carry a non-null "
        "rerank_score (proves the cross-encoder ran). False: no hit may "
        "carry one (proves the reranker stayed off).",
    )
    vector_hits_min: int | None = Field(
        default=None,
        description="Minimum number of vector_context hits that must come back. "
        "Useful for semantic / hybrid cases that should always retrieve "
        "something.",
    )
    span_names_include: list[str] | None = Field(
        default=None,
        description="Span names that must appear at least once in the case's "
        "trace. Useful to prove that an opt-in subsystem actually ran "
        "(e.g. 'session.summarise', 'vector.rerank').",
    )
    routing_refined: bool | None = Field(
        default=None,
        description="Build #6: whether the LLM router fallback actually patched "
        "the rule decision. True asserts at least one 'llm-refined:' "
        "marker appears in matched_keywords; False asserts none do "
        "(use as a negative control for cases where no refinement is "
        "expected).",
    )
    refined_fields_include: list[str] | None = Field(
        default=None,
        description="Build #6: subset of fields the LLM must have patched. "
        "Names must match the markers emitted by router_tool._apply: "
        "'target', 'operation', 'metric', 'group_by', 'route'.",
    )
    critic_action: str | None = Field(
        default=None,
        description="Build #7: expected critic verdict on the analyst's draft. "
        "One of 'approve' (no issues), 'flag' (issues found), or "
        "'fallback' (critic itself failed). Read from "
        "response.meta['critic']['action'].",
    )
    critic_max_blocking_issues: int | None = Field(
        default=None,
        description="Build #7: maximum allowed medium/high-severity issues. "
        "Useful as a false-positive guard on clean analytical cases — "
        "set to 0 to assert the critic did not flag any blocking "
        "issues on a draft we expect to be faithful.",
    )
    span_names_exclude: list[str] | None = Field(
        default=None,
        description="Span names that must NOT appear in the case's trace. "
        "Useful as a negative control (e.g. asserting that the "
        "summariser stayed off when the buffer was below threshold).",
    )
    intent_source: str | None = Field(
        default=None,
        description="Build #8: expected origin of the intent decision. "
        "One of 'rule' (deterministic classifier), 'distiller' "
        "(student model overrode the rule), or 'distiller-agree' "
        "(student concurred with the rule). Read from "
        "IntentResult.source. Use to assert distillation actually "
        "fired (or stayed dormant) on a given question.",
    )
    intent_source_in: list[str] | None = Field(
        default=None,
        description="Build #8: set of acceptable intent_source values. "
        "Use this when you want 'either rule or distiller-agree is "
        "fine, but never distiller-override'. Mutually inclusive with "
        "intent_source — both can be set, both must hold.",
    )


class Case(BaseModel):
    """One evaluation case, parsed from one JSONL line."""

    id: str = Field(..., description="Stable, human-readable id, e.g. 'intent-greet-001'.")
    level: EvalLevel = Field(..., description="Which agent layer to invoke.")
    question: str
    expect: Expect = Field(default_factory=Expect)
    session_id: str | None = Field(
        default=None,
        description="Set on follow-up cases to exercise session-patch behaviour.",
    )
    previous_question: str | None = Field(
        default=None,
        description="For follow-up cases (e2e only): the question to send first "
        "to seed session state, before sending `question`.",
    )
    previous_questions: list[str] | None = Field(
        default=None,
        description="Build #5 multi-turn seeding (e2e only). When set, the "
        "runner sends each question in order before the main `question`, "
        "all under the same `session_id`. Trace + scoring still apply "
        "ONLY to the final main question; seed turns are discarded so "
        "the case's reported tokens / latency reflect the turn under "
        "test, not the warm-up. Mutually compatible with "
        "`previous_question`: if both are set, the singular field is "
        "appended to this list.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering: 'follow-up', 'arabic', 'edge'.",
    )


class Score(BaseModel):
    """One (expected, actual) comparison from one scorer."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""


class Result(BaseModel):
    """Aggregated outcome of running one case."""

    case_id: str
    level: EvalLevel
    passed: bool = Field(
        ..., description="True iff every Score in `scores` passed."
    )
    latency_ms: float
    scores: list[Score] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="Set when the case failed to execute (infra missing, exception).",
    )
    skipped: bool = Field(
        default=False,
        description="True when the case could not run (e.g. e2e but no DB) "
        "and was neither passed nor failed.",
    )
    # ------------------------------------------------------------------
    # Observability — populated by the runner when a TraceContext was
    # active for the case. Always-present fields use 0/None defaults so
    # cases that ran without tracing (intent-only) still serialize cleanly.
    # ------------------------------------------------------------------
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    trace: dict[str, Any] | None = Field(
        default=None,
        description="Full TraceContext.to_dict() — only populated when the "
        "runner was started with --traces (kept off by default to keep "
        "saved JSON small).",
    )
