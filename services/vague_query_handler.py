"""Vague query detection and helpful suggestion generation.

Catches queries that are too vague to execute analytically and provides
context-aware suggestions to guide users toward specific, executable questions.

This is part of Phase 1: Semantic Foundation - improving user experience
through intelligent query understanding rather than returning empty results.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.semantic_intent_enricher import semantic_intent_enricher
from utils.logger import logger


@dataclass(frozen=True)
class VagueQueryCheck:
    """Result of vague query detection."""
    
    is_vague: bool
    confidence: float  # How confident we are it's vague (0.0-1.0)
    reasoning: str
    suggested_questions: list[str]
    inferred_operation: str | None  # What we think they want


class VagueQueryHandler:
    """Detect vague queries and generate helpful suggestions."""
    
    def __init__(self):
        self.enricher = semantic_intent_enricher
    
    def check(
        self, 
        question: str, 
        target: str | None = None
    ) -> VagueQueryCheck:
        """
        Check if a query is too vague to execute analytically.
        
        A query is considered vague when:
        1. No explicit aggregation keywords detected
        2. Semantic enrichment has low confidence
        3. User intent is unclear
        
        Args:
            question: User's natural language question
            target: Target collection/dataset (if known)
            
        Returns:
            VagueQueryCheck with detection results and suggestions
        """
        if not question or not question.strip():
            return VagueQueryCheck(
                is_vague=True,
                confidence=1.0,
                reasoning="Empty question",
                suggested_questions=["Try asking: 'What datasets are available?'"],
                inferred_operation=None
            )
        
        # Use semantic enricher to analyze intent
        enriched = self.enricher.enrich(question, target)
        
        # Determine if query is vague based on confidence
        is_vague = not enriched.is_confident
        
        # Build reasoning
        if is_vague:
            reasoning = (
                f"Query is too vague - {enriched.reasoning}. "
                f"Confidence: {enriched.confidence:.2f} "
                f"(threshold: 0.5)"
            )
        else:
            reasoning = (
                f"Query is specific enough - {enriched.reasoning}"
            )
        
        return VagueQueryCheck(
            is_vague=is_vague,
            confidence=enriched.confidence,
            reasoning=reasoning,
            suggested_questions=enriched.suggested_questions,
            inferred_operation=enriched.inferred_operation
        )
    
    def should_suggest_instead_of_execute(
        self, 
        question: str, 
        target: str | None,
        has_aggregation_spec: bool
    ) -> bool:
        """
        Determine if we should show suggestions instead of executing query.
        
        Args:
            question: User's question
            target: Target collection
            has_aggregation_spec: Whether routing service created an aggregation spec
            
        Returns:
            True if we should show suggestions instead of executing
        """
        # If aggregation spec exists, routing was successful - don't interfere
        if has_aggregation_spec:
            return False
        
        # Check if query is vague
        check = self.check(question, target)
        
        # Suggest if vague and we have a target (dataset-specific vague query)
        return check.is_vague and target is not None
    
    def format_suggestion_response(
        self, 
        question: str, 
        check: VagueQueryCheck,
        target: str | None = None
    ) -> str:
        """
        Format a helpful response with suggestions.
        
        Args:
            question: Original question
            check: Vague query check result
            target: Target dataset (if known)
            
        Returns:
            Formatted markdown response with suggestions
        """
        response_parts = [
            "**Your question needs more detail**\n",
            f"I understand you're asking about **{self._humanize_target(target)}**, "
            "but I need more specifics to give you accurate data.\n"
        ]
        
        if check.inferred_operation:
            response_parts.append(
                f"\n💡 I think you want to see **{check.inferred_operation}** "
                f"(confidence: {check.confidence:.0%}), but I'm not certain enough "
                "to execute automatically.\n"
            )
        
        response_parts.append("\n**Try one of these specific questions:**\n")
        
        for i, suggestion in enumerate(check.suggested_questions, 1):
            response_parts.append(f"{i}. {suggestion}\n")
        
        response_parts.append(
            "\n**Why this matters:**\n"
            "Specific questions help me:\n"
            "- Query the right data fields\n"
            "- Apply correct calculations\n"
            "- Return accurate results\n"
            "\n_Tip: Include what metric you want (e.g., 'total transactions'), "
            "the time period (e.g., '2025'), and the operation (e.g., 'monthly trend')._"
        )
        
        return "".join(response_parts)
    
    def _humanize_target(self, target: str | None) -> str:
        """Convert target collection name to human-readable form."""
        if not target:
            return "the data"
        
        # Remove prefixes and suffixes
        clean = target.replace("awqaf_", "").replace("_facts", "")
        
        # Replace underscores with spaces and title case
        human = clean.replace("_", " ").title()
        
        return human


# Singleton instance
vague_query_handler = VagueQueryHandler()
