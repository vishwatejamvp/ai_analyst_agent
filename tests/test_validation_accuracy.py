"""Test suite for data accuracy validation.

Tests the new validation system to ensure it catches errors before execution
and provides helpful error messages.
"""

import pytest
from models.schemas import QueryRequest
from services.analyst_service import AnalystOrchestrator


class TestValidationAccuracy:
    """Test suite for validation accuracy"""
    
    def setup_method(self):
        self.orchestrator = AnalystOrchestrator()
    
    def test_schema_validation_catches_wrong_metric(self):
        """System should catch non-existent metrics"""
        request = QueryRequest(
            question="total nonexistent_metric for hajj-permit-service in 2024"
        )
        response = self.orchestrator.answer(request)
        
        # Should have warning about metric not found
        assert any('not found' in w.message.lower() or 'metric' in w.message.lower() 
                  for w in response.warnings), \
            f"Expected metric warning, got: {[w.message for w in response.warnings]}"
    
    def test_coverage_validation_catches_future_year(self):
        """System should catch queries for years without data"""
        request = QueryRequest(
            question="total transactions for hajj-permit-service in 2030"
        )
        response = self.orchestrator.answer(request)
        
        # Should have warning about year not available
        has_year_warning = any(
            '2030' in w.message or 'available' in w.message.lower() or 'year' in w.message.lower()
            for w in response.warnings
        )
        assert has_year_warning, \
            f"Expected year warning, got: {[w.message for w in response.warnings]}"
    
    def test_semantic_matching_picks_correct_collection(self):
        """System should use semantic matching for collection selection"""
        request = QueryRequest(
            question="how many pilgrimage permits were issued in 2024"
        )
        response = self.orchestrator.answer(request)
        
        # Should route to hajj_permit_service, not other collections
        assert response.routing.target is not None, "No target selected"
        assert 'hajj' in response.routing.target.lower() or 'permit' in response.routing.target.lower(), \
            f"Expected hajj/permit collection, got: {response.routing.target}"
    
    def test_zero_results_provides_helpful_message(self):
        """System should explain why no results were found"""
        request = QueryRequest(
            question="total transactions for hajj-permit-service in 1990"
        )
        response = self.orchestrator.answer(request)
        
        # Should have helpful message about available years
        if len(response.structured_data) == 0:
            has_helpful_warning = any(
                'available' in w.message.lower() or 'year' in w.message.lower()
                for w in response.warnings
            )
            assert has_helpful_warning, \
                f"Expected helpful warning for zero results, got: {[w.message for w in response.warnings]}"
    
    def test_auto_correction_fixes_similar_names(self):
        """System should auto-correct similar field names"""
        request = QueryRequest(
            question="total transactons for hajj-permit-service in 2024"  # typo
        )
        response = self.orchestrator.answer(request)
        
        # Should either auto-correct or warn about the typo
        if len(response.structured_data) > 0:
            # Auto-correction worked
            assert True
        else:
            # Should have warning about the field
            has_field_warning = any(
                'metric' in w.message.lower() or 'field' in w.message.lower()
                for w in response.warnings
            )
            assert has_field_warning, \
                f"Expected field warning, got: {[w.message for w in response.warnings]}"
    
    def test_validation_allows_valid_queries(self):
        """System should not block valid queries"""
        request = QueryRequest(
            question="total transactions for hajj-permit-service in 2024"
        )
        response = self.orchestrator.answer(request)
        
        # Should execute successfully
        assert response.routing.target is not None, "Valid query was blocked"
        # May or may not have data, but should not have validation errors
        validation_errors = [w for w in response.warnings 
                           if 'not found' in w.message.lower() or 'does not exist' in w.message.lower()]
        assert len(validation_errors) == 0, \
            f"Valid query had validation errors: {[w.message for w in validation_errors]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
