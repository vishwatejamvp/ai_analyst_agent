"""Multi-turn critic enhancement for cross-turn contradiction detection.

Extends the existing critic service to check for contradictions across
conversation history, not just within a single turn.

Examples of cross-turn contradictions:
- Turn 1: "Dubai had the highest revenue"
- Turn 3: "Abu Dhabi had the highest revenue"

- Turn 2: "Total was 1.5M in 2024"
- Turn 4: "Total was 2.1M in 2024"

Architecture:
- Builds conversation context from recent turns
- Checks current draft against prior claims
- Flags contradictions with specific turn references
- Integrates with existing InsightCritic
"""

from __future__ import annotations

from typing import Any

from services.session_summary import Turn
from utils.logger import logger


class HistoricalCriticEnhancement:
    """Enhances critic with cross-turn contradiction detection."""

    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns

    def build_history_context(
        self,
        prior_turns: list[Turn],
    ) -> str:
        """Build conversation history context for critic.

        Args:
            prior_turns: Previous conversation turns

        Returns:
            Formatted history string for critic prompt
        """
        if not prior_turns:
            return ""

        # Use only recent turns to avoid token bloat
        recent_turns = prior_turns[-self.max_history_turns :]

        history_lines = ["CONVERSATION HISTORY (for contradiction checking):"]
        history_lines.append("")

        for i, turn in enumerate(recent_turns, 1):
            turn_num = len(prior_turns) - len(recent_turns) + i
            history_lines.append(f"Turn {turn_num}:")
            history_lines.append(f"Q: {turn.question}")
            history_lines.append(f"A: {turn.insight[:500]}...")  # Truncate long insights
            history_lines.append("")

        return "\n".join(history_lines)

    def enhance_critic_prompt(
        self,
        base_prompt: str,
        prior_turns: list[Turn],
    ) -> str:
        """Enhance critic prompt with history context.

        Args:
            base_prompt: Original critic prompt
            prior_turns: Previous conversation turns

        Returns:
            Enhanced prompt with history context
        """
        if not prior_turns:
            return base_prompt

        history_context = self.build_history_context(prior_turns)

        enhanced_prompt = f"""
{history_context}

{base_prompt}

ADDITIONAL CHECK - Cross-Turn Contradictions:
Review the CONVERSATION HISTORY above. Does the current draft contradict
any claim made in prior turns?

Examples of contradictions:
- Different values for the same metric in the same period
- Different rankings (e.g., "Dubai highest" vs "Abu Dhabi highest")
- Different trends (e.g., "increasing" vs "decreasing")
- Different time ranges (e.g., "2024 only" vs "2023-2024")

If you find a contradiction with a prior turn, flag it as:
- severity: "high"
- type: "contradicts_history"
- quote: The contradicting statement from the current draft
- explanation: "Contradicts Turn X which stated: [prior claim]"
"""

        return enhanced_prompt

    def extract_key_claims(self, insight: str) -> list[dict[str, Any]]:
        """Extract key factual claims from an insight.

        This is a simple heuristic-based extractor. A production version
        could use NER or LLM-based extraction.

        Args:
            insight: Insight text

        Returns:
            List of claims with metadata
        """
        claims = []

        # Extract numeric claims (e.g., "1.5M", "75%", "2024")
        import re

        # Pattern: number + optional unit
        numeric_pattern = r"\b(\d+(?:\.\d+)?(?:[KMB])?)\s*(%|AED|USD|million|thousand|billion)?\b"

        for match in re.finditer(numeric_pattern, insight, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2) or ""

            # Get surrounding context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(insight), match.end() + 20)
            context = insight[start:end]

            claims.append(
                {
                    "type": "numeric",
                    "value": value,
                    "unit": unit,
                    "context": context,
                }
            )

        # Extract ranking claims (e.g., "highest", "lowest", "top")
        ranking_pattern = r"\b(highest|lowest|top|bottom|first|last|leading)\b"

        for match in re.finditer(ranking_pattern, insight, re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(insight), match.end() + 30)
            context = insight[start:end]

            claims.append(
                {
                    "type": "ranking",
                    "value": match.group(1),
                    "context": context,
                }
            )

        # Extract trend claims (e.g., "increasing", "decreasing", "stable")
        trend_pattern = r"\b(increas(?:ing|ed)|decreas(?:ing|ed)|ris(?:ing|e)|fall(?:ing|en)|stable|steady|growing|declining)\b"

        for match in re.finditer(trend_pattern, insight, re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(insight), match.end() + 30)
            context = insight[start:end]

            claims.append(
                {
                    "type": "trend",
                    "value": match.group(1),
                    "context": context,
                }
            )

        return claims

    def check_contradictions(
        self,
        current_draft: str,
        prior_turns: list[Turn],
    ) -> list[dict[str, Any]]:
        """Check for contradictions between current draft and prior turns.

        This is a heuristic-based checker. The LLM critic is still the
        primary contradiction detector; this is a fast pre-check.

        Args:
            current_draft: Current draft insight
            prior_turns: Previous conversation turns

        Returns:
            List of potential contradictions
        """
        if not prior_turns:
            return []

        contradictions = []

        # Extract claims from current draft
        current_claims = self.extract_key_claims(current_draft)

        # Extract claims from recent prior turns
        recent_turns = prior_turns[-self.max_history_turns :]

        for turn_idx, turn in enumerate(recent_turns):
            turn_num = len(prior_turns) - len(recent_turns) + turn_idx + 1
            prior_claims = self.extract_key_claims(turn.insight)

            # Check for numeric contradictions
            for curr_claim in current_claims:
                if curr_claim["type"] != "numeric":
                    continue

                for prior_claim in prior_claims:
                    if prior_claim["type"] != "numeric":
                        continue

                    # Same unit but different value in similar context
                    if (
                        curr_claim["unit"] == prior_claim["unit"]
                        and curr_claim["value"] != prior_claim["value"]
                    ):
                        # Check if contexts are similar (simple heuristic)
                        if self._contexts_similar(
                            curr_claim["context"], prior_claim["context"]
                        ):
                            contradictions.append(
                                {
                                    "type": "numeric_mismatch",
                                    "current": curr_claim,
                                    "prior": prior_claim,
                                    "prior_turn": turn_num,
                                    "severity": "medium",
                                }
                            )

            # Check for ranking contradictions
            for curr_claim in current_claims:
                if curr_claim["type"] != "ranking":
                    continue

                for prior_claim in prior_claims:
                    if prior_claim["type"] != "ranking":
                        continue

                    # Different rankings in similar context
                    if self._contexts_similar(
                        curr_claim["context"], prior_claim["context"]
                    ):
                        contradictions.append(
                            {
                                "type": "ranking_mismatch",
                                "current": curr_claim,
                                "prior": prior_claim,
                                "prior_turn": turn_num,
                                "severity": "high",
                            }
                        )

        if contradictions:
            logger.warning(
                f"Found {len(contradictions)} potential cross-turn contradictions"
            )

        return contradictions

    @staticmethod
    def _contexts_similar(context1: str, context2: str) -> bool:
        """Check if two contexts are similar (simple heuristic).

        A production version would use embeddings or fuzzy matching.
        """
        # Normalize
        c1 = context1.lower().strip()
        c2 = context2.lower().strip()

        # Check for common keywords
        keywords1 = set(c1.split())
        keywords2 = set(c2.split())

        # Jaccard similarity
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        if not union:
            return False

        similarity = len(intersection) / len(union)

        # Threshold: 30% overlap
        return similarity > 0.3


# Module-level singleton
historical_critic = HistoricalCriticEnhancement()
