"""Eval harness runner.

Dispatches each :class:`Case` to the appropriate handler based on
``case.level``, collects scores + latency, and returns :class:`Result`
records that the report layer renders.

Three handlers, ordered cheapest first:

* ``_run_intent``  — calls ``question_intent.classify``. Pure Python;
  needs no DB and no LLM key. This is the layer you'll iterate on most.
* ``_run_routing`` — calls ``routing_service.decide``. Needs Mongo to
  probe schema. If Mongo is unreachable, the case is *skipped* (not
  failed) so the cheap layer above still produces a clean report.
* ``_run_e2e``     — calls ``analyst_orchestrator.answer``. Full pipeline.
  Needs Mongo + Anthropic key. Same skip-on-missing-infra rule.

CLI::

    python -m tests.eval.runner
    python -m tests.eval.runner --level intent
    python -m tests.eval.runner --level intent,routing --tag follow-up
    python -m tests.eval.runner --strict --save tests/eval/results.json
    python -m tests.eval.runner --golden tests/eval/golden.jsonl

``--strict`` exits non-zero if any (non-skipped) case fails — wire this
into CI to block regressions.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from utils.observability import TraceContext, use_trace

from .cases import Case, EvalLevel, Result, Score
from . import scorers as S
from .report import render_terminal, write_json


# ---------------------------------------------------------------------------
# Tracing scope per case
# ---------------------------------------------------------------------------
# Module-level toggle flipped on by --traces. When False we still create
# a TraceContext (so token/cost are recorded by agent_service) but skip
# saving the full trace dict on the Result to keep saved JSON small.
_KEEP_FULL_TRACE = False


@contextmanager
def _case_trace(case: Case) -> Iterator[TraceContext]:
    """Run a case inside a fresh TraceContext.

    Each case gets its own trace so the runner can attribute tokens
    and cost back to a specific case_id. Using ``use_trace`` activates
    the contextvar so deep code (notably ``agent_service._call_claude``)
    finds it via ``current_trace()``.
    """
    ctx = TraceContext.start(
        request_id=f"eval_{case.id}",
        session_id=case.session_id,
    )
    with use_trace(ctx):
        yield ctx


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------
def load_cases(path: Path) -> list[Case]:
    """Load cases from a JSONL file. Skips blank lines and ``# `` comments.

    JSONL is git-diffable and lets non-engineers add cases with no Python
    knowledge. The runner validates each line via Pydantic, so a malformed
    case fails loudly at load time, not silently at run time.
    """
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(Case.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001 — re-raised below with context
                raise ValueError(
                    f"{path}:{line_no} — invalid case JSON: {exc}\n  line: {line[:200]}"
                ) from exc
    return cases


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_cases(
    cases: list[Case],
    levels: set[EvalLevel] | None,
    tags: set[str] | None,
    only_id: str | None,
) -> list[Case]:
    out = cases
    if levels:
        out = [c for c in out if c.level in levels]
    if tags:
        out = [c for c in out if tags.intersection(c.tags)]
    if only_id:
        out = [c for c in out if c.id == only_id]
    return out


# ---------------------------------------------------------------------------
# Handlers (one per level)
# ---------------------------------------------------------------------------
def _run_intent(case: Case) -> Result:
    """Execute an intent-level case.

    Pure Python — no DB, no LLM. The fastest layer to iterate on, and the
    one most likely to short-circuit cost upstream of everything else.
    Wrapped in a TraceContext for uniformity, even though no LLM cost is
    incurred — the trace summary then shows just the timing.
    """
    from services.question_intent import classify as classify_intent

    t0 = time.perf_counter()
    with _case_trace(case) as ctx:
        with ctx.span("intent.classify", question=case.question[:80]):
            try:
                intent_result = classify_intent(case.question)
            except Exception as exc:  # noqa: BLE001 — surface, don't crash the run
                return Result(
                    case_id=case.id,
                    level=case.level,
                    passed=False,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )

        actual_intent = intent_result.intent.name
        actual_intent_source = getattr(intent_result, "source", "rule")
        scores = _collect([
            S.score_intent(case.expect.intent, actual_intent),
            S.score_intent_source(
                case.expect.intent_source, actual_intent_source,
            ),
            S.score_intent_source_in(
                case.expect.intent_source_in, actual_intent_source,
            ),
        ])
        return _finalize(case, scores, t0, ctx)


def _run_routing(case: Case) -> Result:
    """Execute a routing-level case.

    Calls ``routing_service.decide``, which internally probes Mongo for
    schema. We catch infra failures and mark the case ``skipped`` so a
    missing local DB doesn't poison the report.
    """
    t0 = time.perf_counter()
    with _case_trace(case) as ctx:
        try:
            from services.routing_service import routing_service

            with ctx.span("router.decide", question=case.question[:80]) as span:
                decision = routing_service.decide(case.question)
                span.set(
                    target=decision.target,
                    route=decision.route.name,
                    matched=",".join(decision.matched_keywords[:5]),
                )
        except Exception as exc:  # noqa: BLE001 — infra failure → skip, not fail
            return _skip(case, t0, f"routing infra unavailable: {type(exc).__name__}: {exc}", ctx)

        spec = decision.aggregation
        actual_route = decision.route.name
        actual_target = decision.target
        actual_op = spec.operation if spec else None
        actual_metric = spec.metric if spec else None
        actual_group_by = spec.group_by if spec else None
        actual_time = bool(spec and spec.time)
        actual_keywords = list(decision.matched_keywords or [])
        actual_span_names = [s.name for s in ctx.spans]

        scores = _collect([
            S.score_route(case.expect.route, actual_route),
            S.score_target(case.expect.target, actual_target),
            S.score_operation(case.expect.operation, actual_op),
            S.score_metric(case.expect.metric, case.expect.metric_in, actual_metric),
            S.score_group_by(case.expect.group_by, actual_group_by),
            S.score_time_present(case.expect.time_present, actual_time),
            S.score_routing_refined(case.expect.routing_refined, actual_keywords),
            S.score_refined_fields_include(
                case.expect.refined_fields_include, actual_keywords
            ),
            S.score_span_names_include(case.expect.span_names_include, actual_span_names),
            S.score_span_names_exclude(case.expect.span_names_exclude, actual_span_names),
        ])
        return _finalize(case, scores, t0, ctx)


def _seed_questions(case: Case) -> list[str]:
    """Resolve the list of seed questions to send before the main question.

    Two compatible inputs are merged:

    * ``previous_question`` (singular, legacy) — kept for the single-turn
      follow-up cases written in earlier builds.
    * ``previous_questions`` (list, Build #5) — used by multi-turn
      conversation-memory cases.

    If both are set, the singular field is appended to the list, in the
    order it would naturally play out (older context first).
    """
    seeds: list[str] = []
    if case.previous_questions:
        seeds.extend(q for q in case.previous_questions if q and q.strip())
    if case.previous_question and case.previous_question.strip():
        seeds.append(case.previous_question.strip())
    return seeds


def _run_e2e(case: Case) -> Result:
    """Execute an end-to-end case through the orchestrator.

    Needs the full stack (Mongo + Anthropic). Seed turns (from
    ``previous_question`` and/or ``previous_questions``) fire first
    under the same ``session_id`` so session-patch and conversation
    memory are both exercised. Scoring still applies ONLY to the final
    main question — the case's reported tokens / cost / latency
    reflect the turn under test, not the warm-up.

    The whole call runs inside ``_case_trace`` — the orchestrator's own
    ``request.answer`` span and ``agent_service``'s LLM-cost recording
    both detect this trace and contribute to it.
    """
    from models.schemas import QueryRequest

    t0 = time.perf_counter()
    with _case_trace(case) as ctx:
        try:
            from services.analyst_service import analyst_orchestrator
        except Exception as exc:  # noqa: BLE001 — import-time infra issues are skips
            return _skip(case, t0, f"orchestrator import failed: {exc}", ctx)

        sid = case.session_id

        for i, seed_q in enumerate(_seed_questions(case)):
            try:
                with ctx.span(f"seed.turn[{i}]", question=seed_q[:80]):
                    analyst_orchestrator.answer(
                        QueryRequest(question=seed_q, session_id=sid)
                    )
            except Exception as exc:  # noqa: BLE001 — seeding failure → skip dependent case
                return _skip(case, t0, f"seeding turn {i} failed: {exc}", ctx)

        try:
            resp = analyst_orchestrator.answer(
                QueryRequest(question=case.question, session_id=sid)
            )
        except Exception as exc:  # noqa: BLE001 — any pipeline failure → skip
            return _skip(case, t0, f"e2e infra unavailable: {type(exc).__name__}: {exc}", ctx)

        spec = resp.routing.aggregation
        actual_route = resp.routing.route.name
        actual_target = resp.routing.target
        actual_op = spec.operation if spec else None
        actual_metric = spec.metric if spec else None
        actual_group_by = spec.group_by if spec else None
        actual_time = bool(spec and spec.time)
        actual_warnings = [w.code.value for w in (resp.warnings or [])]

        actual_chart = (
            resp.chart.chart_type.value.upper() if resp.chart else "SKIP"
        )
        actual_llm_calls = len(ctx.llm_usages)
        actual_hits = list(resp.vector_context or [])
        actual_span_names = [s.name for s in ctx.spans]
        actual_keywords = list(resp.routing.matched_keywords or [])
        actual_critic = (resp.meta or {}).get("critic")

        scores = _collect([
            S.score_route(case.expect.route, actual_route),
            S.score_target(case.expect.target, actual_target),
            S.score_operation(case.expect.operation, actual_op),
            S.score_metric(case.expect.metric, case.expect.metric_in, actual_metric),
            S.score_group_by(case.expect.group_by, actual_group_by),
            S.score_time_present(case.expect.time_present, actual_time),
            S.score_text_contains(case.expect.text_contains, resp.insight),
            S.score_text_not_contains(case.expect.text_not_contains, resp.insight),
            S.score_warnings_include(case.expect.warning_codes_include, actual_warnings),
            S.score_warnings_exclude(case.expect.warning_codes_exclude, actual_warnings),
            S.score_data_nonempty(case.expect.data_nonempty, len(resp.structured_data)),
            S.score_chart_type(case.expect.chart_type, actual_chart),
            S.score_llm_calls(
                case.expect.llm_calls_min,
                case.expect.llm_calls_max,
                actual_llm_calls,
            ),
            S.score_reranked(case.expect.reranked, actual_hits),
            S.score_vector_hits_min(case.expect.vector_hits_min, len(actual_hits)),
            S.score_span_names_include(case.expect.span_names_include, actual_span_names),
            S.score_span_names_exclude(case.expect.span_names_exclude, actual_span_names),
            S.score_routing_refined(case.expect.routing_refined, actual_keywords),
            S.score_refined_fields_include(
                case.expect.refined_fields_include, actual_keywords
            ),
            S.score_critic_action(case.expect.critic_action, actual_critic),
            S.score_critic_blocking_issues(
                case.expect.critic_max_blocking_issues, actual_critic
            ),
        ])
        return _finalize(case, scores, t0, ctx)


_HANDLERS = {
    EvalLevel.INTENT: _run_intent,
    EvalLevel.ROUTING: _run_routing,
    EvalLevel.E2E: _run_e2e,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _collect(scores: list[Score | None]) -> list[Score]:
    """Drop ``None`` scores (asserts the case didn't make)."""
    return [s for s in scores if s is not None]


def _trace_fields(ctx: TraceContext | None) -> dict:
    """Pull token / cost / latency out of a TraceContext into Result fields.

    Empty trace (no LLM calls) → all zeros, which is correct: an
    intent-only case really did spend $0 and zero tokens.
    """
    if ctx is None:
        return {}
    fields = {
        "input_tokens": ctx.total_input_tokens,
        "output_tokens": ctx.total_output_tokens,
        "cost_usd": round(ctx.total_cost_usd, 6),
        "llm_calls": len(ctx.llm_usages),
    }
    if _KEEP_FULL_TRACE:
        fields["trace"] = ctx.to_dict()
    return fields


def _finalize(
    case: Case, scores: list[Score], t0: float, ctx: TraceContext | None = None
) -> Result:
    passed = all(s.passed for s in scores) if scores else True
    return Result(
        case_id=case.id,
        level=case.level,
        passed=passed,
        latency_ms=(time.perf_counter() - t0) * 1000,
        scores=scores,
        **_trace_fields(ctx),
    )


def _skip(
    case: Case, t0: float, reason: str, ctx: TraceContext | None = None
) -> Result:
    return Result(
        case_id=case.id,
        level=case.level,
        passed=False,
        skipped=True,
        latency_ms=(time.perf_counter() - t0) * 1000,
        error=reason,
        **_trace_fields(ctx),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run(cases: list[Case]) -> list[Result]:
    """Execute every case through its handler. Order preserved."""
    results: list[Result] = []
    for case in cases:
        handler = _HANDLERS[case.level]
        results.append(handler(case))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_GOLDEN = Path(__file__).parent / "golden.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AI Analyst Agent eval harness.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to the golden JSONL file.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Comma-separated levels to run: intent,routing,e2e (default: all).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Comma-separated tags. A case must have at least one to be included.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run a single case by id.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any non-skipped case fails.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write JSON results to this path (in addition to terminal output).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in the terminal report.",
    )
    parser.add_argument(
        "--traces",
        action="store_true",
        help="Include full trace dicts on each Result (large; off by default). "
        "Token/cost summary is always captured.",
    )
    parser.add_argument(
        "--chart-decider",
        choices=["rule", "llm"],
        default=None,
        help="Override settings.chart_decider for this run. "
        "'llm' activates the function-calling chart planner "
        "(adds one Claude call per analytical answer).",
    )
    parser.add_argument(
        "--reranker",
        choices=["on", "off"],
        default=None,
        help="Override settings.reranker_enabled for this run. "
        "'on' activates the cross-encoder rerank stage in vector_service "
        "(adds local model latency, no LLM cost).",
    )
    parser.add_argument(
        "--session-summary",
        choices=["on", "off"],
        default=None,
        help="Override settings.session_summary_enabled for this run. "
        "'on' threads a multi-turn running summary into the analyst "
        "prompt; the summariser fires when the buffer exceeds the "
        "trigger threshold (see --summary-trigger / --summary-keep-last).",
    )
    parser.add_argument(
        "--summary-trigger",
        type=int,
        default=None,
        help="Override settings.session_summary_trigger_at. Compress the "
        "buffer once it holds this many turns. Lower this (e.g. 3) when "
        "running short multi-turn cases that should still trip the "
        "summariser.",
    )
    parser.add_argument(
        "--summary-keep-last",
        type=int,
        default=None,
        help="Override settings.session_summary_keep_last. After "
        "compression, keep this many recent turns verbatim and fold the "
        "rest into the running summary.",
    )
    parser.add_argument(
        "--llm-router",
        choices=["on", "off"],
        default=None,
        help="Override settings.router_llm_fallback_enabled for this run. "
        "'on' lets RoutingService.decide consult an LLM second-opinion "
        "ONLY when its uncertainty flags fire (no_target, "
        "missing_metric_for_op, group_by_intent_unmet, etc.). The LLM "
        "can patch target / op / metric / group_by / route, validated "
        "against schema columns and candidate targets.",
    )
    parser.add_argument(
        "--critic",
        choices=["on", "off"],
        default=None,
        help="Override settings.critic_enabled for this run. "
        "'on' runs a verification critic over the analyst's draft "
        "narrative; medium/high-severity findings trigger one bounded "
        "revise round (unless --critic-revise off). Adds 1 LLM call "
        "per analytical answer (2 if revise fires).",
    )
    parser.add_argument(
        "--critic-revise",
        choices=["on", "off"],
        default=None,
        help="Override settings.critic_revise_on_flag for this run. "
        "'off' puts the critic in shadow / annotate-only mode: "
        "findings are recorded in response.meta and the trace, but no "
        "second generator call fires. Useful to measure hallucination "
        "rate before paying for revision.",
    )
    parser.add_argument(
        "--distill",
        choices=["on", "off"],
        default=None,
        help="Override settings.intent_distiller_enabled for this run. "
        "'on' lets question_intent.classify consult the trained TF-IDF "
        "+ LogReg student model. The student can ONLY override the rule "
        "when its confidence >= --distill-threshold AND it disagrees "
        "with the rule. Falls back silently if the model artifact is "
        "missing.",
    )
    parser.add_argument(
        "--distill-threshold",
        type=float,
        default=None,
        help="Override settings.intent_distiller_confidence_threshold "
        "(0.0-1.0). Cases that need a low-confidence override to fire "
        "(e.g. exploratory paraphrases) can lower this temporarily.",
    )
    args = parser.parse_args()

    global _KEEP_FULL_TRACE
    _KEEP_FULL_TRACE = bool(args.traces)

    # Optional setting overrides — mutate the cached singleton for this
    # process. Safe because each runner invocation is its own process.
    if args.chart_decider is not None:
        from models.config import get_settings
        get_settings().chart_decider = args.chart_decider
        print(f"[runner] chart_decider overridden to: {args.chart_decider}")
    if args.reranker is not None:
        from models.config import get_settings
        get_settings().reranker_enabled = (args.reranker == "on")
        print(f"[runner] reranker_enabled overridden to: {args.reranker == 'on'}")
    if args.session_summary is not None:
        from models.config import get_settings
        get_settings().session_summary_enabled = (args.session_summary == "on")
        print(
            f"[runner] session_summary_enabled overridden to: "
            f"{args.session_summary == 'on'}"
        )
    if args.summary_trigger is not None:
        from models.config import get_settings
        get_settings().session_summary_trigger_at = args.summary_trigger
        print(f"[runner] session_summary_trigger_at overridden to: {args.summary_trigger}")
    if args.summary_keep_last is not None:
        from models.config import get_settings
        get_settings().session_summary_keep_last = args.summary_keep_last
        print(f"[runner] session_summary_keep_last overridden to: {args.summary_keep_last}")
    if args.llm_router is not None:
        from models.config import get_settings
        get_settings().router_llm_fallback_enabled = (args.llm_router == "on")
        print(
            f"[runner] router_llm_fallback_enabled overridden to: "
            f"{args.llm_router == 'on'}"
        )
    if args.critic is not None:
        from models.config import get_settings
        get_settings().critic_enabled = (args.critic == "on")
        print(f"[runner] critic_enabled overridden to: {args.critic == 'on'}")
    if args.critic_revise is not None:
        from models.config import get_settings
        get_settings().critic_revise_on_flag = (args.critic_revise == "on")
        print(
            f"[runner] critic_revise_on_flag overridden to: "
            f"{args.critic_revise == 'on'}"
        )
    if args.distill is not None:
        from models.config import get_settings
        get_settings().intent_distiller_enabled = (args.distill == "on")
        print(
            f"[runner] intent_distiller_enabled overridden to: "
            f"{args.distill == 'on'}"
        )
    if args.distill_threshold is not None:
        from models.config import get_settings
        get_settings().intent_distiller_confidence_threshold = (
            float(args.distill_threshold)
        )
        print(
            f"[runner] intent_distiller_confidence_threshold "
            f"overridden to: {args.distill_threshold}"
        )

    cases = load_cases(args.golden)

    levels: set[EvalLevel] | None = None
    if args.level:
        try:
            levels = {EvalLevel(v.strip().lower()) for v in args.level.split(",") if v.strip()}
        except ValueError as exc:
            print(f"error: invalid --level: {exc}", file=sys.stderr)
            return 2

    tags: set[str] | None = None
    if args.tag:
        tags = {t.strip() for t in args.tag.split(",") if t.strip()}

    selected = filter_cases(cases, levels, tags, args.only)
    if not selected:
        print("No cases matched the filters.", file=sys.stderr)
        return 0

    results = run(selected)

    print(render_terminal(results, color=not args.no_color))
    if args.save:
        write_json(args.save, results)
        print(f"\nResults written to: {args.save}")

    if args.strict:
        any_failed = any((not r.passed) and (not r.skipped) for r in results)
        return 1 if any_failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
