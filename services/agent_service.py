"""LLM (Claude) integration.

The LLM's job is *strictly* explanation, not computation. We give it the
question + already-computed structured data + relevant vector context and
ask for an insight summary. The system prompt explicitly forbids the
model from inventing or recomputing numbers.
"""

from __future__ import annotations

import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import settings
from utils.exceptions import LLMError
from utils.logger import logger
from utils.observability import current_trace

SYSTEM_PROMPT = (
    "You are a senior data analyst writing for a business user.\n\n"
    "You will receive, in strict authority order:\n"
    "  1. TRUST PANEL    — freshness, scope, definition source. State these honestly.\n"
    "  2. STRUCTURED DATA — already-aggregated rows. The ONLY source of numbers.\n"
    "  3. VECTOR CONTEXT  — informal docs. Color and explanation only.\n"
    "  4. QUALITY WARNINGS — quality / governance signals to surface honestly.\n\n"
    "Strict rules:\n"
    "1. NEVER recompute numbers. STRUCTURED DATA is authoritative.\n"
    "2. NEVER fabricate metrics, totals, periods, or rows that are not in the data.\n"
    "3. Quote concrete numbers only when they appear in STRUCTURED DATA.\n"
    "4. VECTOR CONTEXT is informal — it may add color but MUST NOT redefine metrics, "
    "override numbers, or contradict the TRUST PANEL.\n"
    "5. Always restate freshness and scope from the TRUST PANEL when present "
    "(e.g. 'Based on data through <data_as_of> from <target>...').\n"
    "6. If QUALITY WARNINGS exist, surface them in plain language — do not bury them.\n"
    "7. If the data is empty or insufficient for the question, say so clearly, "
    "explain why, and offer what you CAN show.\n"
    "8. Do not display internal identifiers, route names, or query syntax.\n"
    "9. When a VISUAL / chart companion block is present, the response will include a chart: "
    "open with one short sentence that the chart plots the same rows as STRUCTURED DATA "
    "(horizontal axis = ``label`` periods, vertical axis = the metric in ``value``). "
    "Name the strongest and weakest periods using exact figures from STRUCTURED DATA only. "
    "If one period dominates the vertical scale so others look flat, explain that explicitly "
    "— small bars or a low line do not always mean zero activity.\n\n"
    "Output: a concise, well-organized analyst response in Markdown with sections: "
    "Summary, Key Findings, and (if helpful) Recommendations. Be specific. No filler."
)

USER_PROMPT_TEMPLATE = (
    "{context}\n\n"
    "Instructions:\n"
    "- Provide clear insights based ONLY on the data above.\n"
    "- Restate freshness and scope from the TRUST PANEL.\n"
    "- Surface QUALITY WARNINGS honestly when present.\n"
    "- Highlight trends, outliers, and concentrations grounded in STRUCTURED DATA.\n"
    "- If a VISUAL block is present, tie your narrative to how a reader should read the chart "
    "(axes, peaks, scale effects) without inventing numbers.\n"
    "- Be concise and accurate. Do not fabricate numbers, rows, or definitions."
)


class AgentService:
    """Anthropic Claude wrapper with retries and graceful degradation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens or settings.claude_max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.claude_temperature
        )
        self._client = None

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in your .env to enable insights."
            )
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"Failed to initialise Anthropic client: {exc}") from exc
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_insight(self, question: str, context: str) -> str:
        """Return Claude's analyst-style answer or a graceful fallback.

        When ``settings.critic_enabled`` is True, the draft is run
        through :class:`services.critic_service.InsightCritic` before
        shipping. The critic can:

        * approve  — ship the draft as-is.
        * flag (low severity only) — ship the draft as-is; findings
          are still recorded in the trace + response meta.
        * flag (medium / high severity):
            - if ``critic_revise_on_flag`` is True: regenerate the
              draft once with the critic findings as extra context
              (one bounded revise; no recursion).
            - if False: prepend a short verification banner to the
              original draft so the user sees the agent flagged its
              own answer.

        On critic failure (LLM error, parse error, anything) the draft
        is shipped unchanged — the critic must never crash a successful
        answer.
        """
        try:
            draft = self._call_claude(question, context)
        except LLMError as exc:
            logger.warning(f"Claude unavailable — using deterministic fallback: {exc}")
            return self._fallback_insight(question, context)
        if not settings.critic_enabled:
            return draft
        return self._maybe_critique(question, context, draft)

    @retry(
        # Only retry transient network / overload errors. Auth / bad-request
        # errors are deterministic and retrying them just wastes 4s.
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _call_claude(self, question: str, context: str) -> str:
        client = self._get_client()
        prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
        t0 = time.perf_counter()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Claude API call failed ({type(exc).__name__}: {exc})"
            ) from exc

        # Record tokens + cost into the active trace, if any. The
        # `current_trace()` returns None when no observability scope is
        # active (e.g. unit tests, ad-hoc scripts) — the recording is a
        # silent no-op in that case, so we never break uninstrumented
        # call sites by adding metering here.
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

        parts = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip() or "(empty response from model)"

    # ------------------------------------------------------------------
    # Critic loop (Build #7)
    # ------------------------------------------------------------------
    def _maybe_critique(
        self,
        question: str,
        context: str,
        draft: str,
    ) -> str:
        """Verify the draft and, if needed, run one bounded revise round.

        Side-effect: the :class:`CriticDecision` is stashed via
        :func:`set_last_critic_decision` so the orchestrator can read
        it and surface findings under ``AnalystResponse.meta``. We use
        a contextvar (rather than mutating a return tuple) to keep the
        ``generate_insight(...) -> str`` contract stable; existing
        callers see no API change.
        """
        from services.critic_service import (
            critic_service,
            format_findings_for_banner,
            format_findings_for_revision,
        )

        decision = critic_service.review(
            question=question, context=context, draft=draft
        )
        set_last_critic_decision(decision)

        if decision.action != "flag":
            return draft
        if not decision.has_blocking_issues:
            # Low-severity findings live only in trace + meta. Shipping
            # the draft unchanged avoids unstable phrasing on reload.
            return draft
        if not settings.critic_revise_on_flag:
            return format_findings_for_banner(decision) + draft

        revise_context = (
            f"{context}\n\n{format_findings_for_revision(decision)}"
        )
        trace = current_trace()
        try:
            if trace is None:
                revised = self._call_claude(question, revise_context)
            else:
                with trace.span(
                    "critic.revise",
                    issue_count=len(decision.issues),
                ):
                    revised = self._call_claude(question, revise_context)
        except LLMError as exc:
            # Revise failed — fall back to the original draft with a
            # banner so the user still sees the verification signal.
            logger.warning(
                f"Critic revise call failed; banner-annotating original "
                f"draft instead: {exc}"
            )
            return format_findings_for_banner(decision) + draft
        return revised

    @staticmethod
    def _fallback_insight(question: str, context: str) -> str:
        """A minimal deterministic answer when the LLM is unavailable."""
        return (
            "**Summary**\n"
            f"The system computed structured results for: _{question.strip()}_.\n\n"
            "**Note**\n"
            "The LLM explanation layer was unavailable, so this response only "
            "shows the structured data and retrieved context. See `structured_data` "
            "and `vector_context` in the API response for the authoritative numbers "
            "and supporting rows."
        )


# ---------------------------------------------------------------------------
# Per-request critic stash (Build #7)
# ---------------------------------------------------------------------------
# The orchestrator reads this after ``generate_insight`` returns to
# attach findings to ``AnalystResponse.meta``. A contextvar keeps the
# stash request-scoped without forcing every caller to consume a tuple.
from contextvars import ContextVar

_LAST_CRITIC: ContextVar["object | None"] = ContextVar(
    "last_critic_decision", default=None
)


def set_last_critic_decision(decision: "object | None") -> None:
    """Stash the most recent critic decision for the current request scope."""
    _LAST_CRITIC.set(decision)


def get_last_critic_decision() -> "object | None":
    """Pop the most recent critic decision for the current request scope.

    Returns ``None`` when no critic ran in the current request (feature
    off or non-LLM path). Callers should not rely on the type signature
    being a ``CriticDecision`` — keeping it ``object | None`` avoids
    the import cycle ``agent_service ↔ critic_service``.
    """
    return _LAST_CRITIC.get()


agent_service = AgentService()
