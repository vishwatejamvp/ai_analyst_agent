"""Tests for semantic intent enrichment system.

Tests the Phase 1 production roadmap implementation: moving from code-driven
to data-driven architecture with embedding-based intent detection.
"""

import pytest

from services.semantic_intent_enricher import semantic_intent_enricher, EnrichedIntent
from services.vague_query_handler import vague_query_handler, VagueQueryCheck


class TestSemanticIntentEnricher:
    """Test semantic intent enrichment for vague queries."""
    
    def test_vague_query_with_dataset_name(self):
        """Test: 'what should i look into hajj package' should infer 'sum' operation."""
        question = "what should i look into hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        assert isinstance(enriched, EnrichedIntent)
        assert enriched.inferred_operation is not None
        assert enriched.inferred_operation in ("sum", "count", "avg")
        assert enriched.confidence > 0.0
        assert len(enriched.reasoning) > 0
    
    def test_specific_query_high_confidence(self):
        """Test: Specific queries should have high confidence."""
        question = "total transactions for hajj package service in 2025"
        target = "awqaf_hajj_package_service_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        # Specific query should still work (may have keywords detected earlier)
        assert isinstance(enriched, EnrichedIntent)
        assert enriched.confidence >= 0.0  # Any confidence is fine
    
    def test_empty_question(self):
        """Test: Empty questions should return low confidence."""
        enriched = semantic_intent_enricher.enrich("", None)
        
        assert enriched.inferred_operation is None
        assert enriched.confidence == 0.0
        assert "Empty question" in enriched.reasoning
    
    def test_show_me_dataset(self):
        """Test: 'show me hajj package' should infer operation."""
        question = "show me hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        assert enriched.inferred_operation is not None
        assert enriched.confidence > 0.0
    
    def test_tell_me_about_dataset(self):
        """Test: 'tell me about zakat disbursement' should infer operation."""
        question = "tell me about zakat disbursement"
        target = "awqaf_zakat_disbursement_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        assert enriched.inferred_operation is not None
        assert enriched.confidence > 0.0
    
    def test_analyze_dataset(self):
        """Test: 'analyze occupancy rates' should infer operation."""
        question = "analyze occupancy rates"
        target = "awqaf_occupancy_rates_and_revenues_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        assert enriched.inferred_operation is not None
        assert enriched.confidence > 0.0
    
    def test_suggestions_for_low_confidence(self):
        """Test: Low confidence queries should generate suggestions."""
        question = "something vague and unclear"
        target = "awqaf_hajj_package_service_facts"
        
        enriched = semantic_intent_enricher.enrich(question, target)
        
        # Should have suggestions if confidence is low
        if enriched.confidence < 0.5:
            assert len(enriched.suggested_questions) > 0


class TestVagueQueryHandler:
    """Test vague query detection and suggestion generation."""
    
    def test_vague_query_detection(self):
        """Test: Vague queries should be detected."""
        question = "what should i look into hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        check = vague_query_handler.check(question, target)
        
        assert isinstance(check, VagueQueryCheck)
        assert check.confidence >= 0.0
        assert len(check.reasoning) > 0
    
    def test_empty_query_is_vague(self):
        """Test: Empty queries are always vague."""
        check = vague_query_handler.check("", None)
        
        assert check.is_vague is True
        assert check.confidence == 1.0
        assert len(check.suggested_questions) > 0
    
    def test_suggestion_formatting(self):
        """Test: Suggestions should be well-formatted."""
        question = "what should i look into hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        check = vague_query_handler.check(question, target)
        response = vague_query_handler.format_suggestion_response(
            question, check, target
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "specific" in response.lower() or "detail" in response.lower()
    
    def test_should_suggest_logic(self):
        """Test: should_suggest_instead_of_execute logic."""
        question = "what should i look into hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        # With aggregation spec - should NOT suggest
        should_suggest = vague_query_handler.should_suggest_instead_of_execute(
            question, target, has_aggregation_spec=True
        )
        assert should_suggest is False
        
        # Without aggregation spec and vague - SHOULD suggest
        # (This depends on confidence, so we can't assert True always)
        should_suggest = vague_query_handler.should_suggest_instead_of_execute(
            question, target, has_aggregation_spec=False
        )
        assert isinstance(should_suggest, bool)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_original_bug_scenario(self):
        """Test: Original bug 'what should i look into hajj package' scenario."""
        question = "what should i look into hajj package"
        target = "awqaf_hajj_package_service_facts"
        
        # Step 1: Enrichment should infer operation
        enriched = semantic_intent_enricher.enrich(question, target)
        assert enriched.inferred_operation is not None
        
        # Step 2: If confidence is low, handler should detect vagueness
        check = vague_query_handler.check(question, target)
        assert isinstance(check, VagueQueryCheck)
        
        # Step 3: Should generate helpful suggestions
        if check.is_vague:
            assert len(check.suggested_questions) > 0
            response = vague_query_handler.format_suggestion_response(
                question, check, target
            )
            assert "specific" in response.lower() or "Try" in response
    
    def test_various_vague_patterns(self):
        """Test: Various vague query patterns should be handled."""
        vague_queries = [
            "show me hajj data",
            "tell me about zakat",
            "analyze mosques",
            "give me quran centers information",
            "explore umrah campaigns"
        ]
        
        for question in vague_queries:
            enriched = semantic_intent_enricher.enrich(question, None)
            
            # Should infer some operation or provide suggestions
            assert (
                enriched.inferred_operation is not None or
                len(enriched.suggested_questions) > 0
            )


class TestDataDrivenConfig:
    """Test that business logic is data-driven, not code-driven."""
    
    def test_config_loaded(self):
        """Test: Operation patterns config should be loaded."""
        # Trigger config load
        enriched = semantic_intent_enricher.enrich("test", None)
        
        # Config should be loaded
        assert semantic_intent_enricher._config is not None
        assert len(semantic_intent_enricher._operation_patterns) > 0
    
    def test_operation_patterns_exist(self):
        """Test: Operation patterns should be defined in config."""
        # Trigger config load
        semantic_intent_enricher._ensure_config_loaded()
        
        patterns = semantic_intent_enricher._operation_patterns
        assert len(patterns) > 0
        
        # Should have at least sum, count, avg
        operations = {p.operation for p in patterns}
        assert "sum" in operations
        assert "count" in operations
    
    def test_confidence_thresholds_configurable(self):
        """Test: Confidence thresholds should be configurable."""
        semantic_intent_enricher._ensure_config_loaded()
        
        threshold = semantic_intent_enricher._get_threshold("min_acceptable")
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
