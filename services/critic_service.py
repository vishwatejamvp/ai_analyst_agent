"""Insight critic — verifier loop over the analyst's narrative (Build #7).

Why this module exists
----------------------
The :class:`services.agent_service.AgentService` system prompt forbids
the model from recomputing numbers, fabricating metrics, contradicting
the trust panel, or burying quality warnings. Those rules push the
hallucination rate *down* but not to zero. In production:

* The model sometimes quotes a number that's CLOSE to but not IN the
  STRUCTURED DATA (e.g. 76% when the row says 72%).
* The model sometimes invents a period or scope ("from 2020 to 2024"
  when the rows are 2024 only).
* The model sometimes mentions an entity (an emirate, a category) that
  isn't in the rows it was given.
* The model sometimes paraphrases the trust panel into something that
  reads more confident than it should ("data is current" when the
  trust panel actually says "data through 2024-12").

A second, narrower LLM call — the critic — reads the *same context the
analyst saw* plus the *draft the analyst produced*, and answers one
question only: "does this draft contain any claim that the data does
not support?". Its output is structured (typed issue list) so a small
deterministic policy can decide what to do.

The pattern (verifier / critic / reflexion-style self-check) is the
canonical way to add a quality floor to LLM output without giving the
LLM more authority. The critic cannot rewrite the draft; it can only
flag specific claims for revision or banner annotation. The policy in
:mod:`services.agent_service` decides whether to spend a second
generator call to fix them.

Cost discipline
---------------
* One critic call per analytical answer (~500 in / 150 out tokens =
  ~$0.003).
* If the critic flags blocking issues, ONE revise call (full
  generator regeneration with critic findings appended).
* Off by default. With CRITIC_REVISE_ON_FLAG=False the critic runs
  but never triggers a second generator call — useful for measuring
  hallucination rates in shadow mode before paying for revision.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from models.config import settings
from utils.exceptions import LLMError
from utils.logger import logger
from utils.observability import current_trace


# ---------------------------------------------------------------------------
# Issue taxonomy
# ---------------------------------------------------------------------------
# These five categories map 1:1 to the constraints in
# ``agent_service.SYSTEM_PROMPT``. Every flag the critic raises must
# map cleanly to one of them — no free-text "miscellaneous" so the
# eval can score critic behaviour over time.
IssueType = Literal[
    "fabricated_number",
    "wrong_period",
    "wrong_scope",
    "contradicts_trust",
    "contradicts_warning",
    "other",
]
Severity = Literal["high", "medium", "low"]


class IssueFound(BaseModel):
    """A single discrepancy between the draft and the supplied data."""

    severity: Severity = Field(
        ...,
        description=(
            "high   — quoted a number / period / row that does NOT appear "
            "in STRUCTURED DATA, or directly contradicts the TRUST PANEL.\n"
            "medium — paraphrases the data inaccurately (e.g. says 'most' "
            "when the actual share is 30%) or suppresses a QUALITY WARNING.\n"
            "low    — minor stylistic drift; data is supported but the "
            "phrasing implies more / less than the data shows."
        ),
    )
    type: IssueType = Field(
        ...,
        description=(
            "Category of discrepancy. Choose 'other' only when none of the "
            "named types fit; do not invent new categories."
        ),
    )
    quote: str = Field(
        ...,
        max_length=300,
        description=(
            "The EXACT substring from the draft that is unsupported. "
            "Quote it verbatim — do not paraphrase. The orchestrator may "
            "use this string to highlight the offending text."
        ),
    )
    evidence: str = Field(
        ...,
        max_length=400,
        description=(
            "Short, factual reference to what the data actually says. "
            "Cite the row(s) or trust-panel field that contradicts the "
            "quote. No analyst-style commentary."
        ),
    )
    suggested_fix: str = Field(
        ...,
        max_length=300,
        description=(
            "A short rewrite the analyst could ship instead. One sentence."
        ),
    )


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------
class ApproveArgs(BaseModel):
    """The critic vouches for the draft — no issues found."""

    reasoning: str = Field(
        ...,
        max_length=300,
        description=(
            "One sentence: why the draft is fully supported by STRUCTURED "
            "DATA / TRUST PANEL / QUALITY WARNINGS. Visible in the trace."
        ),
    )


class FlagArgs(BaseModel):
    """The critic found one or more discrepancies.

    ``issues`` is intentionally NOT marked as required at the schema
    level even though we expect at least one — Anthropic's
    ``tool_choice={"type": "any"}`` mode sometimes coerces the model
    to pick a tool with empty / partial input when it has "nothing
    confident to say". We accept that signal gracefully (in
    :meth:`InsightCritic._parse_tool_use`) by reinterpreting an empty
    ``flag_issues`` call as an approval.
    """

    issues: list[IssueFound] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Concrete discrepancies. Order from most to least severe. "
            "Cap at 8 — if there are more, the draft should be regenerated "
            "wholesale, not patched. Leave empty ONLY if you decided "
            "nothing needs flagging (in which case prefer approve_draft)."
        ),
    )
    summary: str = Field(
        default="",
        max_length=200,
        description=(
            "One sentence overview the orchestrator can show to the user "
            "or include in the verification banner."
        ),
    )


# ---------------------------------------------------------------------------
# Tool schemas in Anthropic's expected shape
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "approve_draft",
        "description": (
            "Call this when EVERY concrete claim in the draft (numbers, "
            "periods, entities, freshness statements) is directly "
            "supported by STRUCTURED DATA, TRUST PANEL, or QUALITY "
            "WARNINGS. Do not approve drafts that quietly drop an "
            "important warning — that's a 'contradicts_warning' flag."
        ),
        "input_schema": ApproveArgs.model_json_schema(),
    },
    {
        "name": "flag_issues",
        "description": (
            "Call this when the draft contains at least one claim that "
            "is unsupported, contradicted, or invented. List each "
            "discrepancy as one IssueFound; quote the offending text "
            "verbatim and cite the row or trust-panel field that "
            "contradicts it."
        ),
        "input_schema": FlagArgs.model_json_schema(),
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a verification critic for an analytics agent. A senior "
    "analyst has produced a DRAFT narrative based on STRUCTURED DATA, "
    "a TRUST PANEL, and QUALITY WARNINGS. Your single job is to flag "
    "every claim in the draft that the data does NOT support.\n\n"
    "You do this by calling exactly ONE tool:\n"
    "  - approve_draft(reasoning) — every concrete claim is grounded.\n"
    "  - flag_issues(issues=[...], summary) — at least one claim is "
    "unsupported, contradicted, or invented.\n\n"
    "How to verify, step by step (do NOT verbalise this; just do it):\n"
    "1. Extract every concrete claim from the draft: numbers, percentages, "
    "ranks, periods (year / month / quarter), entity names (emirates, "
    "categories), and freshness / scope statements.\n"
    "2. For each claim, locate the supporting row or field in STRUCTURED "
    "DATA / TRUST PANEL. If the claim is paraphrased (e.g. 'most' or "
    "'a majority'), confirm the underlying share is consistent.\n"
    "3. If a number quoted in the draft does not appear in STRUCTURED DATA "
    "(within typical rounding), flag it as 'fabricated_number' with the "
    "quote verbatim.\n"
    "4. If a period (year / month / range) is mentioned but absent from "
    "STRUCTURED DATA rows, flag 'wrong_period'.\n"
    "5. If an entity (emirate, category, group) is named but no row in "
    "STRUCTURED DATA carries that value, flag 'wrong_scope'.\n"
    "6. If the draft asserts freshness, source, or scope that contradicts "
    "the TRUST PANEL, flag 'contradicts_trust'.\n"
    "7. If a QUALITY WARNING is silently dropped or contradicted by the "
    "draft, flag 'contradicts_warning'.\n\n"
    "Conservative bias:\n"
    "* Do NOT flag stylistic phrasing as an issue. The draft is allowed "
    "to summarise, contextualise, and recommend.\n"
    "* Do NOT flag rounded numbers (12.34% vs 12.3%) as fabricated.\n"
    "* Do NOT flag claims supported by VECTOR CONTEXT — that block is "
    "informal color and the analyst is allowed to repeat it.\n"
    "* When in genuine doubt, lean toward approve_draft. False positives "
    "trigger costly revise loops; the goal is to catch real errors, not "
    "to second-guess the analyst.\n\n"
    "You MUST call exactly one tool. Free-text replies are ignored."
)


# ---------------------------------------------------------------------------
# Decision returned to the agent
# ---------------------------------------------------------------------------
class CriticDecision(BaseModel):
    """Unified result the agent consumes."""

    action: Literal["approve", "flag", "fallback"] = Field(
        ...,
        description=(
            "approve  — ship the draft as-is.\n"
            "flag     — at least one issue found; agent decides whether "
            "to revise or annotate.\n"
            "fallback — critic itself failed (LLM error, parse error). "
            "Treat the draft as best-effort and ship it; the failure is "
            "logged but never crashes the answer."
        ),
    )
    issues: list[IssueFound] = Field(default_factory=list)
    summary: str = ""
    reasoning: str = ""

    @property
    def has_blocking_issues(self) -> bool:
        """True iff at least one issue is medium or high severity.

        Low-severity findings are recorded for observability but do
        not gate the response — over-eager revisions waste tokens and
        give the user unstable phrasing across reloads.
        """
        # pylint: disable=not-an-iterable  # Pydantic list field
        return any(i.severity in ("high", "medium") for i in self.issues)

    @classmethod
    def approve(cls, reasoning: str) -> "CriticDecision":
        return cls(action="approve", reasoning=reasoning)

    @classmethod
    def flag(cls, args: FlagArgs) -> "CriticDecision":
        return cls(
            action="flag",
            issues=list(args.issues),
            summary=args.summary,
        )

    @classmethod
    def fallback(cls, reason: str) -> "CriticDecision":
        return cls(action="fallback", reasoning=reason)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class InsightCritic:
    """One Claude call → one tool decision → one CriticDecision.

    Stateless across calls. Holds a lazy Anthropic client so the same
    instance can be reused across requests without re-initialising.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> None:
        # Default to the same model as the analyst so a "critic-of-self"
        # pass uses comparable reasoning power. Callers can override
        # with a cheaper model in cost-sensitive deployments.
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def review(
        self,
        *,
        question: str,
        context: str,
        draft: str,
    ) -> CriticDecision:
        """Verify ``draft`` against ``context``; return a structured decision.

        Never raises into the agent: any LLM error degrades to a
        ``fallback`` decision so the draft still ships. Tracing happens
        via ``current_trace()``; works whether or not an outer trace
        is active.
        """
        if not (draft or "").strip():
            return CriticDecision.approve("empty draft — nothing to verify")

        trace = current_trace()
        if trace is None:
            return self._do_review(question, context, draft)
        with trace.span(
            "critic.review",
            draft_chars=len(draft),
        ) as span:
            decision = self._do_review(question, context, draft)
            span.set(
                action=decision.action,
                issue_count=len(decision.issues),
                has_blocking=decision.has_blocking_issues,
            )
            return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _do_review(
        self, question: str, context: str, draft: str
    ) -> CriticDecision:
        prompt = self._compose_user_prompt(question, context, draft)
        try:
            return self._call_claude(prompt)
        except LLMError as exc:
            logger.warning(f"Critic LLM unavailable; shipping draft as-is: {exc}")
            return CriticDecision.fallback(str(exc))
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            logger.warning(
                f"Critic unexpected error; shipping draft as-is: "
                f"{type(exc).__name__}: {exc}"
            )
            return CriticDecision.fallback(f"{type(exc).__name__}: {exc}")

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

    def _call_claude(self, user_prompt: str) -> CriticDecision:
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
                f"Critic Claude call failed "
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
    def _parse_tool_use(message: Any) -> CriticDecision:
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", None)
            raw = getattr(block, "input", None) or {}
            if name == "approve_draft":
                try:
                    args = ApproveArgs.model_validate(raw)
                except ValidationError as exc:
                    raise LLMError(
                        f"approve_draft input failed validation: {exc}"
                    ) from exc
                return CriticDecision.approve(args.reasoning)
            if name == "flag_issues":
                try:
                    args = FlagArgs.model_validate(raw)
                except ValidationError as exc:
                    raise LLMError(
                        f"flag_issues input failed validation: {exc}"
                    ) from exc
                # Self-healing: an empty flag_issues call is the
                # model's way of saying "I had to pick a tool but
                # there's nothing to flag". Treat as approval rather
                # than dropping a meaningless ``flag`` decision into
                # the pipeline.
                if not args.issues:
                    return CriticDecision.approve(
                        args.summary or "no issues found"
                    )
                return CriticDecision.flag(args)
            raise LLMError(f"unexpected tool: {name!r}")
        raise LLMError("no tool_use block in Claude response")

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------
    @staticmethod
    def _compose_user_prompt(question: str, context: str, draft: str) -> str:
        # Cap the draft to keep the prompt bounded. A 4 k-char draft is
        # already an unusually long analyst response; in practice they
        # land at 800-2000 chars.
        capped_draft = draft if len(draft) <= 6000 else draft[:5990] + "…"
        return (
            "QUESTION the analyst was answering:\n"
            f"{question.strip()}\n\n"
            "CONTEXT the analyst was given (TRUST PANEL + STRUCTURED DATA "
            "+ QUALITY WARNINGS + VECTOR CONTEXT):\n"
            f"{context.strip()}\n\n"
            "DRAFT NARRATIVE the analyst produced:\n"
            "----- BEGIN DRAFT -----\n"
            f"{capped_draft.strip()}\n"
            "----- END DRAFT -----\n\n"
            "Verify the draft now. Call exactly one tool."
        )


# ---------------------------------------------------------------------------
# Helpers used by the agent's revise loop
# ---------------------------------------------------------------------------
def format_findings_for_revision(decision: CriticDecision) -> str:
    """Render critic findings as an extra context block for the regenerator.

    The agent appends this block to the original context and runs the
    generator again. The block is intentionally directive — the goal
    is "fix these specific issues", not "freely rewrite".
    """
    if not decision.issues:
        return ""
    bullets = []
    for i, issue in enumerate(decision.issues, start=1):
        bullets.append(
            f"{i}. [{issue.severity.upper()} / {issue.type}] "
            f"Quote: \"{issue.quote.strip()}\"\n"
            f"   Why it's wrong: {issue.evidence.strip()}\n"
            f"   Suggested fix: {issue.suggested_fix.strip()}"
        )
    return (
        "CRITIC FINDINGS (you produced an earlier draft; a verifier "
        "found these specific issues — produce a revised narrative "
        "that fixes EACH one without changing anything else, and "
        "without re-introducing the flagged claims):\n"
        + "\n".join(bullets)
    )


def format_findings_for_banner(decision: CriticDecision) -> str:
    """Render a short verification banner to prepend to a flagged draft.

    Used in *annotate-only* mode (revise disabled) so the user sees a
    visible signal that the agent flagged its own answer.
    """
    if not decision.issues:
        return ""
    blocking = [i for i in decision.issues if i.severity in ("high", "medium")]
    headline = (
        decision.summary
        or f"{len(blocking)} potential issue(s) flagged by self-verification."
    )
    return (
        "> **Verification notice:** "
        f"{headline} See `meta.critic_findings` in the response for "
        "details.\n\n"
    )


# Module-level singleton, mirroring agent_service.
critic_service = InsightCritic()
