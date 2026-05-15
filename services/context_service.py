"""Context builder.

Combines, in this strict authority order:

1. **TRUST PANEL** — freshness, scope, definition source. Always restated.
2. **STRUCTURED DATA** — already-aggregated rows (Mongo / MySQL). Authoritative.
3. **VECTOR CONTEXT** — informal docs / wiki snippets. Color & explanation only.
4. **WARNINGS** — typed quality / governance signals.

…into a single well-formatted block of text that's ready to drop into
the LLM prompt. The LLM is told (and structurally only able) to *explain*
these numbers, never to recompute them. Vector context can NEVER override
either trust panel facts or structured numbers.
"""

from __future__ import annotations

import json
from typing import Any

from models.schemas import AnalystWarning, TrustPanel, VectorHit
from utils.config import settings


def _trim(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[:limit] if limit and len(rows) > limit else rows


class ContextService:
    """Format trust + DB + vector results into an LLM-ready context block."""

    def __init__(self, max_rows_for_llm: int | None = None) -> None:
        self.max_rows_for_llm = max_rows_for_llm or settings.max_rows_for_llm

    def build(
        self,
        *,
        question: str,
        structured_data: list[dict[str, Any]] | None = None,
        vector_hits: list[VectorHit] | None = None,
        routing_reason: str = "",
        trust: TrustPanel | None = None,
        warnings: list[AnalystWarning] | None = None,
        chart_companion: str | None = None,
        dataset_definitions: str | None = None,
        conversation_memory: str | None = None,
    ) -> str:
        """Return a single string fit for the LLM prompt.

        ``conversation_memory`` (Build #5) is a pre-formatted block
        containing the running summary of older turns plus any recent
        verbatim turns the orchestrator chose to keep. We slot it in
        AFTER the question (so the model knows what was just asked)
        but BEFORE the trust panel and structured data (so the model
        treats it as background, not as a source of numbers). The
        block itself is labelled with a strict guard rail so a
        misremembered figure cannot leak into the new answer.
        """
        sections: list[str] = []

        if routing_reason:
            sections.append(f"ROUTING: {routing_reason}")

        sections.append(f"QUESTION:\n{question.strip()}")

        if conversation_memory:
            sections.append(
                "CONVERSATION MEMORY (background only; what was discussed in "
                "earlier turns of this session). Use it to keep the narrative "
                "coherent and to refer back to prior topics. NEVER quote a "
                "number from this block — STRUCTURED DATA below is the only "
                "source of figures.\n"
                + conversation_memory.strip()
            )

        if dataset_definitions:
            sections.append(
                "DATASET DEFINITIONS (authoritative meanings; use to ground "
                "narrative, never to change numbers):\n" + dataset_definitions
            )

        if trust is not None:
            sections.append(
                "TRUST PANEL (must be reflected in the answer; do not change these facts):\n"
                + json.dumps(trust.model_dump(mode="json"), indent=2, default=str)
            )

        if structured_data:
            trimmed = _trim(structured_data, self.max_rows_for_llm)
            sections.append(
                "STRUCTURED DATA (authoritative; computed by the database):\n"
                + json.dumps(trimmed, indent=2, default=str)
            )
        else:
            sections.append("STRUCTURED DATA: (none)")

        if chart_companion:
            sections.append(chart_companion)

        if vector_hits:
            lines = [
                f"- [score={hit.score:.3f} | "
                f"source={hit.collection}#{hit.document_id}] {hit.text}"
                for hit in vector_hits
            ]
            sections.append(
                "VECTOR CONTEXT (informal; for color/explanation only — "
                "MUST NOT redefine metrics or override numbers):\n"
                + "\n".join(lines)
            )
        else:
            sections.append("VECTOR CONTEXT: (none)")

        if warnings:
            warn_lines = [f"- [{w.code.value}] {w.message}" for w in warnings]
            sections.append(
                "QUALITY WARNINGS (mention these honestly when relevant):\n"
                + "\n".join(warn_lines)
            )

        return "\n\n".join(sections)


context_service = ContextService()
