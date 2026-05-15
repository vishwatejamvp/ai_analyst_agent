"""LLM-router fallback (Build #6).

The mental model
----------------
The deterministic :class:`services.routing_service.RoutingService` is
fast, cheap, and right ~80% of the time. The remaining ~20% are the
inherently ambiguous queries: phrasings that don't match any keyword
table, target collections whose names share no tokens with the
question, group-by intents expressed in unusual word order, etc.

For those, calling an LLM is genuinely useful — but only as a
*targeted second opinion*, never as a replacement for the rule path.
This module is that second opinion.

How it differs from "just send everything to the LLM"
-----------------------------------------------------
1. **Rule-first.** ``RoutingService.decide()`` always produces a
   complete decision before this module is asked anything. The LLM
   is consulted only when ``_uncertainty_flags()`` reports at least
   one concrete concern.
2. **Patch, don't replace.** The LLM returns a small "refinement"
   object whose every field is optional. The refinement is then
   *applied on top of* the rule decision: the LLM can fill or
   correct fields it's confident about; everything else stays as the
   rules computed it. This is cheaper than re-deriving the spec and
   safer than letting the LLM rewrite.
3. **Allowed-set validation.** Every patched value must be in a
   known-good set:
   * ``target`` must be in the rule decision's ``target_candidates``.
   * ``metric`` and ``group_by`` must be in the chosen target's
     actual columns.
   * ``operation`` must be one of {sum, avg, count, min, max}.
   * ``route`` must be one of {ANALYTICAL, SEMANTIC, HYBRID}.
   Hallucinated values are silently rejected and the rule field stands
   — better to ship the rule answer than a confidently-wrong LLM one.
4. **One round trip.** No retry on bad output, no agent loop. One
   ``client.messages.create`` with ``tool_choice="any"``, then validate
   and apply.

What the LLM sees
-----------------
* The user question.
* The current rule decision (target, op, metric, group_by, route).
* The list of uncertainty flags that triggered the call.
* The candidate target collections (only allowed picks).
* The chosen target's column list (only allowed metric / group_by picks).

What the LLM returns (via tool use)
-----------------------------------
A single ``refine_routing`` tool call with these optional fields:
``target, operation, metric, group_by, route, reasoning``. Anything
left ``None`` means "I'm not confident; keep what the rules decided".

Cost discipline
---------------
This adds **at most one extra Claude call**, fired only when the rule
router was uncertain. The trace records its tokens just like any
other LLM call; if the cost-per-question becomes material, set
``ROUTER_LLM_FALLBACK_ENABLED=false`` and the module disappears
entirely.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from models.enums import DataSource, QueryRoute
from models.schemas import AggregationSpec, RoutingDecision
from utils.config import settings
from utils.exceptions import LLMError
from utils.logger import logger
from utils.observability import current_trace


# ---------------------------------------------------------------------------
# Tool argument schema (one tool, all-optional fields = "patch")
# ---------------------------------------------------------------------------
_VALID_OPERATIONS = ("sum", "avg", "count", "min", "max")
_VALID_ROUTES = ("ANALYTICAL", "SEMANTIC", "HYBRID")


class RefineRoutingArgs(BaseModel):
    """Arguments the LLM passes to the ``refine_routing`` tool.

    Every field is optional. Filling a field means "I am confident
    enough to override the rule decision for this dimension". Leaving
    a field ``None`` means "I'm not confident; keep what the rules
    decided". The model is explicitly instructed (in SYSTEM_PROMPT)
    not to over-reach by setting fields it doesn't have evidence for.
    """

    target: str | None = Field(
        default=None,
        description=(
            "Collection / table that should answer this question. "
            "MUST be one of the names listed in CANDIDATE TARGETS. "
            "Leave null if the current target looks correct."
        ),
    )
    operation: Literal["sum", "avg", "count", "min", "max"] | None = Field(
        default=None,
        description=(
            "Aggregation operation. Use 'count' when the user wants a "
            "row count (no metric needed). Leave null to keep the rule's "
            "operation."
        ),
    )
    metric: str | None = Field(
        default=None,
        description=(
            "Numeric column to aggregate. MUST be one of the names listed "
            "in TARGET COLUMNS. Leave null if no metric column is needed "
            "(count operation) or if the rule already picked the right one."
        ),
    )
    group_by: str | None = Field(
        default=None,
        description=(
            "Categorical column to group results by. MUST be one of the "
            "names listed in TARGET COLUMNS. Leave null when the user "
            "didn't ask for a breakdown."
        ),
    )
    route: Literal["ANALYTICAL", "SEMANTIC", "HYBRID"] | None = Field(
        default=None,
        description=(
            "Overall query path. ANALYTICAL = numeric DB query only, "
            "SEMANTIC = vector search only, HYBRID = both. Leave null "
            "to keep the rule's route."
        ),
    )
    reasoning: str = Field(
        ...,
        max_length=300,
        description=(
            "One short sentence explaining what you changed and why. "
            "Mandatory — visible in the trace."
        ),
    )


# ---------------------------------------------------------------------------
# Tool schema in Anthropic's expected shape
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "refine_routing",
        "description": (
            "Patch the rule-based routing decision when you can improve it. "
            "Set only the fields you are confident about — leave others "
            "null to keep the rule's value. Never invent target / metric / "
            "group_by names: they MUST appear in CANDIDATE TARGETS or "
            "TARGET COLUMNS in the prompt."
        ),
        "input_schema": RefineRoutingArgs.model_json_schema(),
    }
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a routing-decision reviewer for an analytics agent. A "
    "deterministic rule-based router has produced a draft decision and "
    "flagged specific concerns about it. Your job is to PATCH that "
    "decision — confirm what's right, fix what's wrong, leave the rest "
    "alone.\n\n"
    "You do this by calling exactly ONE tool: refine_routing. Set ONLY "
    "the fields you are confident about; leave the others null. Setting "
    "a field overrides the rule's value for that field; null leaves the "
    "rule's value in place.\n\n"
    "Strict rules:\n"
    "1. NEVER invent a target, metric, or group_by name. They MUST be "
    "in CANDIDATE TARGETS / TARGET COLUMNS as listed.\n"
    "2. CONSERVATIVE BIAS — leave a field null whenever you are not "
    "*more* confident than the rule. The rule is right ~80% of the time; "
    "your job is to fix the rest, not second-guess every decision.\n"
    "3. ROUTE CHANGES ARE RARE. Do NOT change the route from HYBRID to "
    "SEMANTIC just because the question begins with 'why' or 'explain'. "
    "HYBRID is the correct route when BOTH semantic verbs ('why', "
    "'explain', 'describe') AND analytical signals (a year mention, an "
    "aggregation keyword, a metric column name) are present. Only set "
    "route='SEMANTIC' when there is NO analytical signal at all (no "
    "year, no metric, no aggregation word).\n"
    "4. Prefer a metric whose name shares vocabulary with the question. "
    "If multiple plausible metric columns exist, pick the most specific "
    "one. If none clearly match, leave metric=null rather than guess.\n"
    "5. If the question contains 'by X' / 'per X' / 'for each X' and X "
    "matches a column name, set group_by=X.\n"
    "6. If the rule's target shares no tokens with the question, look "
    "for a CANDIDATE TARGET whose name does — but only override if you "
    "are confident it's the right one.\n"
    "7. Output ONLY the tool_use block. Free-text is ignored.\n"
    "8. The reasoning string is mandatory and must explain what you "
    "changed (or 'no changes needed' if you confirm the rule decision)."
)


# ---------------------------------------------------------------------------
# Decision returned to the routing service
# ---------------------------------------------------------------------------
class RouterRefinement(BaseModel):
    """Internal record of what the refiner did, for tracing + tests."""

    applied: bool = Field(
        default=False,
        description="True iff at least one rule field was actually patched.",
    )
    fallback: bool = Field(
        default=False,
        description="True iff the LLM call failed — rule decision returned untouched.",
    )
    fields_changed: list[str] = Field(
        default_factory=list,
        description="Names of decision fields the LLM successfully patched.",
    )
    fields_rejected: list[str] = Field(
        default_factory=list,
        description="Names of fields the LLM tried to patch but the value "
        "was not in the allowed set (target not in candidates, "
        "metric not in columns, etc.).",
    )
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------
class LLMRouterRefiner:
    """One Claude call → one tool decision → patched RoutingDecision.

    Stateless across calls. Holds a lazy Anthropic client so the same
    instance can be reused across requests without re-initialising.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refine(
        self,
        *,
        question: str,
        decision: RoutingDecision,
        target_columns: list[str],
        flags: list[str],
    ) -> tuple[RoutingDecision, RouterRefinement]:
        """Return ``(possibly-patched-decision, audit-record)``.

        Never raises into the routing service: any LLM error degrades
        to "fallback=True" and the original decision is returned
        unchanged. Tracing happens via ``current_trace()`` so this
        works whether or not an outer trace is active.
        """
        trace = current_trace()
        if trace is None:
            return self._do_refine(
                question, decision, target_columns, flags
            )
        with trace.span(
            "router.llm_fallback",
            flags=",".join(flags),
            target=decision.target or "<none>",
        ) as span:
            new_decision, refinement = self._do_refine(
                question, decision, target_columns, flags
            )
            span.set(
                changed=",".join(refinement.fields_changed) or "<none>",
                rejected=",".join(refinement.fields_rejected) or "<none>",
                fallback=refinement.fallback,
            )
            return new_decision, refinement

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _do_refine(
        self,
        question: str,
        decision: RoutingDecision,
        target_columns: list[str],
        flags: list[str],
    ) -> tuple[RoutingDecision, RouterRefinement]:
        prompt = self._compose_user_prompt(
            question, decision, target_columns, flags
        )
        try:
            args = self._call_claude(prompt)
        except LLMError as exc:
            logger.warning(
                f"Router LLM fallback unavailable; using rule decision: {exc}"
            )
            return decision, RouterRefinement(
                fallback=True, reasoning=f"LLM error: {exc}"
            )
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            logger.warning(
                f"Router LLM fallback unexpected error; using rule decision: "
                f"{type(exc).__name__}: {exc}"
            )
            return decision, RouterRefinement(
                fallback=True,
                reasoning=f"unexpected: {type(exc).__name__}: {exc}",
            )

        return self._apply(decision, args, target_columns)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured.")
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"Failed to initialise Anthropic client: {exc}") from exc
        return self._client

    def _call_claude(self, user_prompt: str) -> RefineRoutingArgs:
        client = self._get_client()
        t0 = time.perf_counter()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Router-refiner Claude call failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
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
            )

        return self._parse_tool_use(message)

    @staticmethod
    def _parse_tool_use(message: Any) -> RefineRoutingArgs:
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                name = getattr(block, "name", None)
                if name != "refine_routing":
                    raise LLMError(f"unexpected tool: {name!r}")
                raw = getattr(block, "input", None) or {}
                try:
                    return RefineRoutingArgs.model_validate(raw)
                except ValidationError as exc:
                    raise LLMError(f"tool input failed validation: {exc}") from exc
        raise LLMError("no tool_use block in Claude response")

    # ------------------------------------------------------------------
    # Patching with allowed-set validation
    # ------------------------------------------------------------------
    @staticmethod
    def _apply(
        decision: RoutingDecision,
        args: RefineRoutingArgs,
        target_columns: list[str],
    ) -> tuple[RoutingDecision, RouterRefinement]:
        """Apply only the patches that pass allowed-set validation.

        We mutate a shallow copy of the decision rather than the
        original to keep the rule decision available for the trace
        (callers can compare ``before`` vs ``after`` by holding both).
        """
        changed: list[str] = []
        rejected: list[str] = []

        # Snapshot AggregationSpec separately so we can rebuild it
        # without partial mutation if no aggregation field is patched.
        spec: AggregationSpec | None = decision.aggregation
        new_target = decision.target
        new_route = decision.route
        new_data_source = decision.data_source

        # ---- target ----
        if args.target is not None and args.target != decision.target:
            allowed = set(decision.target_candidates) | (
                {decision.target} if decision.target else set()
            )
            if args.target in allowed:
                new_target = args.target
                changed.append("target")
                # If the chosen target is a Mongo collection (matches the
                # AWQAF naming convention), keep data_source on Mongo.
                # Otherwise leave the rule's source intact.
                if new_target.startswith("awqaf_"):
                    new_data_source = DataSource.MONGO
            else:
                rejected.append("target")

        # ---- route ----
        if args.route is not None:
            try:
                proposed_route = QueryRoute(args.route.lower())
            except ValueError:
                proposed_route = None
            if proposed_route is not None and proposed_route != decision.route:
                new_route = proposed_route
                changed.append("route")
            elif proposed_route is None:
                rejected.append("route")

        # ---- aggregation: operation / metric / group_by ----
        # Build a working copy only if there's anything to patch on
        # the spec — that way pure SEMANTIC routes don't gain an empty
        # AggregationSpec by accident.
        wants_spec_patch = (
            args.operation is not None
            or args.metric is not None
            or args.group_by is not None
        )
        if wants_spec_patch:
            if spec is None:
                # The rule path didn't produce a spec (e.g. SEMANTIC).
                # The LLM is asking us to promote it to ANALYTICAL — but
                # only if it provided a valid operation. Without that,
                # any metric / group_by it volunteered is meaningless.
                if args.operation is None:
                    rejected.append("aggregation_promotion")
                else:
                    spec = AggregationSpec(operation=args.operation)
                    changed.append("operation")
            # ``spec`` may still be None if we rejected the promotion.
            if spec is not None:
                if (
                    args.operation is not None
                    and args.operation in _VALID_OPERATIONS
                    and args.operation != spec.operation
                ):
                    spec = spec.model_copy(update={"operation": args.operation})
                    if "operation" not in changed:
                        changed.append("operation")
                if args.metric is not None:
                    if args.metric in target_columns:
                        if args.metric != spec.metric:
                            spec = spec.model_copy(update={"metric": args.metric})
                            changed.append("metric")
                    else:
                        rejected.append("metric")
                if args.group_by is not None:
                    if args.group_by in target_columns:
                        if args.group_by != spec.group_by:
                            spec = spec.model_copy(update={"group_by": args.group_by})
                            changed.append("group_by")
                    else:
                        rejected.append("group_by")

        # If route was promoted to ANALYTICAL/HYBRID but no spec
        # exists, nothing useful happened — restore SEMANTIC and reject.
        if (
            new_route in (QueryRoute.ANALYTICAL, QueryRoute.HYBRID)
            and spec is None
        ):
            new_route = decision.route
            if "route" in changed:
                changed.remove("route")
                rejected.append("route_without_spec")

        if not changed:
            # No patches applied — return the original decision
            # unchanged (preserve identity for downstream caching).
            return decision, RouterRefinement(
                applied=False,
                fields_changed=[],
                fields_rejected=rejected,
                reasoning=args.reasoning,
            )

        # Build the patched decision. We tag the matched_keywords with
        # an audit marker so the trust panel / logs make it obvious
        # the LLM got involved.
        marker = f"llm-refined:{','.join(changed)}"
        new_matched = list(decision.matched_keywords) + [marker]
        new_reason = decision.reason
        if args.reasoning:
            new_reason = f"{decision.reason} [LLM: {args.reasoning.strip()}]"

        new_decision = decision.model_copy(
            update={
                "target": new_target,
                "route": new_route,
                "data_source": new_data_source,
                "aggregation": spec,
                "matched_keywords": new_matched,
                "reason": new_reason,
            }
        )
        return new_decision, RouterRefinement(
            applied=True,
            fields_changed=changed,
            fields_rejected=rejected,
            reasoning=args.reasoning,
        )

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------
    @staticmethod
    def _compose_user_prompt(
        question: str,
        decision: RoutingDecision,
        target_columns: list[str],
        flags: list[str],
    ) -> str:
        """Build the single user message Claude sees."""
        spec = decision.aggregation
        rule_summary = {
            "target": decision.target,
            "route": decision.route.value,
            "operation": spec.operation if spec else None,
            "metric": spec.metric if spec else None,
            "group_by": spec.group_by if spec else None,
            "matched_keywords": decision.matched_keywords[:8],
        }
        # Cap candidates / columns to keep the prompt small. The model
        # rarely needs more than ~30 of each to choose well.
        candidates = list(decision.target_candidates)[:25]
        columns = list(target_columns)[:60]

        return (
            f"QUESTION:\n{question.strip()}\n\n"
            f"UNCERTAINTY FLAGS (concerns about the rule decision):\n"
            f"{', '.join(flags) or '(none — confirmation pass)'}\n\n"
            f"RULE-BASED DECISION (your starting point):\n"
            f"{json.dumps(rule_summary, indent=2, default=str)}\n\n"
            f"CANDIDATE TARGETS (choose only from this list when "
            f"setting `target`):\n{json.dumps(candidates, indent=2)}\n\n"
            f"TARGET COLUMNS (choose only from this list when setting "
            f"`metric` or `group_by`):\n{json.dumps(columns, indent=2)}\n\n"
            f"Call refine_routing now. Set only the fields you are "
            f"confident about; leave others null."
        )


# Module-level singleton, mirroring agent_service.
router_refiner = LLMRouterRefiner()
