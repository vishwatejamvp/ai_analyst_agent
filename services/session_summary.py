"""Conversation summarisation (Build #5).

Why this module exists
----------------------

The orchestrator already remembers the *last* (question, routing-decision)
pair per session for short-form follow-ups (``services.session_patch``).
What it does **not** remember is the full multi-turn conversation: which
metrics the user already asked about, what the LLM already explained,
what trends were called out, what choices were made.

When a user has a long thread ("show 2024 occupancy → why is March low?
→ compare to 2023 → which months drove the gap?"), each LLM call today
sees only the *current* question. The model has no way to say "as you
saw earlier, March was an outlier driven by ...".

The naive fix is to stuff every prior turn into the prompt. That breaks
quickly:

* Token cost grows linearly with turn count.
* The prompt drifts away from the system instructions as it lengthens
  (lost-in-the-middle effect).
* Slow turns get slower because input tokens dominate latency.

The standard fix — and what this module implements — is the
*ConversationSummaryBuffer* pattern:

1. **Buffer** the most recent ``keep_last`` turns verbatim.
2. **Summarise** older turns, on demand, into a running paragraph.
3. The next prompt receives ``summary + last few turns`` instead of
   the entire history.

The summary itself is produced by a small Claude call with a strict
system prompt: facts only, no speculation, no fabricated numbers.

Costs
-----

The summariser is invoked at most once per
``trigger_at - keep_last`` turns. Its input is small (older turns are
already short narratives), so a typical call is a few hundred input
tokens and ~150 output tokens — pennies. The trace context records its
tokens just like any other LLM call.

This is intentionally orthogonal to the existing patcher: the patcher
helps the *router* keep the right collection / target, while this
module helps the *narrator* keep its memory of what was said.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from utils.config import settings
from utils.exceptions import LLMError
from utils.logger import logger
from utils.observability import current_trace


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One question / insight pair in the conversation history.

    We deliberately store only the *narrative insight* — not the raw
    structured rows or vector hits — because the summariser cares about
    "what was said" not "what was queried". Keeping turn payloads small
    also keeps summariser prompts cheap.
    """

    question: str
    insight: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def short_insight(self, max_chars: int = 600) -> str:
        """Trim the insight for the summariser prompt.

        We cap each turn at ~600 chars so a buffer of N turns stays
        bounded even if one answer was unusually long.
        """
        text = (self.insight or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# System prompt for the summariser
# ---------------------------------------------------------------------------

# The prompt is intentionally narrow: "summarise what was discussed",
# nothing else. We do NOT ask the model to draw conclusions, predict
# next steps, or speculate. That keeps the running summary safe to
# inject back into the analyst prompt.
_SYSTEM_PROMPT = (
    "You compress a multi-turn analytics conversation into a short, "
    "factual running summary that a senior analyst will read before "
    "answering the next question.\n\n"
    "Strict rules:\n"
    "1. Summarise ONLY what the user asked and what was answered. "
    "Do not invent topics, numbers, or motivations.\n"
    "2. Preserve concrete entities mentioned: dataset names, metrics, "
    "time periods, dimensions (emirate, year, month).\n"
    "3. If the new turns refine or correct an earlier turn, say so "
    "(e.g. 'user narrowed the scope to 2024 only').\n"
    "4. If a previous summary is provided, MERGE the new turns into "
    "it — keep the existing facts, fold in the new ones, drop nothing "
    "that is still relevant.\n"
    "5. Output 3-6 short bullet points or 1 short paragraph. No "
    "headings, no markdown formatting, no analyst tone.\n"
    "6. Never include phrases like 'in this conversation' or 'so far' "
    "— write it as a neutral set of facts.\n"
)


_USER_TEMPLATE = (
    "{prior_block}"
    "TURNS TO FOLD IN (oldest → newest):\n"
    "{turns_block}\n\n"
    "Produce the updated running summary now."
)


def _format_turns(turns: list[Turn]) -> str:
    """Render turns as a compact transcript for the summariser prompt."""
    chunks: list[str] = []
    for i, t in enumerate(turns, start=1):
        chunks.append(
            f"[Turn {i}]\n"
            f"USER: {t.question.strip()}\n"
            f"ANALYST: {t.short_insight()}"
        )
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# The summariser
# ---------------------------------------------------------------------------


class ConversationSummariser:
    """Compress old turns into a running summary, on demand.

    Lifecycle, given a session with ``trigger_at=6`` and ``keep_last=2``:

        turn count   action
        ----------   --------------------------------------------
        1..5         no-op (buffer below trigger)
        6            compress turns[0:4] → summary
                     keep turns[4:6] verbatim (the "fresh tail")
        7..11        no-op (buffer below trigger again)
        12           compress (existing summary + turns[0:4])
                     keep turns[4:6] verbatim
        ...

    The method is **stateless** — the caller (``SessionService``) holds
    the buffer and the summary string; this class just turns N old
    turns into M < N tokens of context.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> None:
        # We reuse the same Claude credentials and model family as the
        # main analyst by default so the summariser inherits the same
        # cost / quality profile. Callers can override per-instance for
        # cheaper-model experiments.
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    @staticmethod
    def should_compress(
        buffer_size: int,
        *,
        trigger_at: int,
    ) -> bool:
        """Pure function: trip the compressor when the buffer is full.

        Kept static so callers (eval cases, unit tests, etc.) can ask
        the same question without instantiating a Claude-backed object.
        """
        return buffer_size >= max(trigger_at, 1)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def compress(
        self,
        *,
        old_turns: list[Turn],
        prior_summary: str | None,
    ) -> str:
        """Produce the new running summary.

        ``old_turns`` are the turns that should be folded INTO the
        summary (i.e. the buffer minus the fresh tail the caller wants
        to keep verbatim). ``prior_summary`` is whatever the previous
        compression produced, or ``None`` on the first compression.

        Returns the new summary string. On any LLM failure we degrade
        to a deterministic concatenation so the rest of the system can
        still benefit from "some" memory rather than crashing.
        """
        if not old_turns:
            return (prior_summary or "").strip()

        trace = current_trace()
        if trace is None:
            return self._do_compress(old_turns, prior_summary)
        with trace.span(
            "session.summarise",
            n_turns=len(old_turns),
            had_prior_summary=bool(prior_summary),
        ):
            return self._do_compress(old_turns, prior_summary)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _do_compress(
        self,
        old_turns: list[Turn],
        prior_summary: str | None,
    ) -> str:
        try:
            new_summary = self._call_claude(old_turns, prior_summary)
        except LLMError as exc:
            logger.warning(
                f"Summariser unavailable — falling back to deterministic "
                f"merge ({exc})"
            )
            return self._fallback_summary(old_turns, prior_summary)

        return new_summary.strip() or self._fallback_summary(
            old_turns, prior_summary
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not configured — summariser disabled"
            )
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"Failed to initialise Anthropic client: {exc}") from exc
        return self._client

    def _call_claude(
        self,
        old_turns: list[Turn],
        prior_summary: str | None,
    ) -> str:
        client = self._get_client()
        prior_block = ""
        if prior_summary:
            prior_block = (
                "EXISTING SUMMARY (merge into this; do not discard facts):\n"
                f"{prior_summary.strip()}\n\n"
            )
        prompt = _USER_TEMPLATE.format(
            prior_block=prior_block,
            turns_block=_format_turns(old_turns),
        )
        t0 = time.perf_counter()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Summariser Claude call failed "
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

        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _fallback_summary(
        old_turns: list[Turn],
        prior_summary: str | None,
    ) -> str:
        """Deterministic fold used when the LLM is unavailable.

        Far less useful than a real LLM summary, but it preserves the
        invariant "memory monotonically grows" so downstream callers
        don't see the summary disappear under transient API failures.
        """
        bullets = [
            f"- The user asked: {t.question.strip()}"
            for t in old_turns
            if t.question.strip()
        ]
        body = "\n".join(bullets) if bullets else "- (no prior questions recorded)"
        if prior_summary:
            return f"{prior_summary.strip()}\n{body}"
        return body


# Module-level singleton, mirroring ``agent_service``.
summariser = ConversationSummariser()
