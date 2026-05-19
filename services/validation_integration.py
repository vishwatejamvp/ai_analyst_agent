"""Integration layer for validation services.

This module provides a unified interface for validating routing decisions
before execution, combining schema, coverage, and semantic validation.
"""

from typing import Tuple, List, Optional
from models.schemas import RoutingDecision, AnalystWarning
from models.enums import WarningCode
from services.schema_validator import schema_validator
from services.coverage_validator import coverage_validator
from services.query_previewer import query_previewer
from utils.logger import logger


class ValidationIntegration:
    """Unified validation interface for routing decisions"""
    
    def __init__(self):
        self.schema_validator = schema_validator
        self.coverage_validator = coverage_validator
        self.query_previewer = query_previewer
    
    def validate_and_correct(
        self, 
        decision: RoutingDecision,
        question: str
    ) -> Tuple[RoutingDecision, List[AnalystWarning], bool]:
        """
        Validate routing decision and attempt auto-correction.
        
        Returns:
            (corrected_decision, warnings, is_valid)
        """
        warnings = []
        is_valid = True
        
        # Skip validation for non-analytical routes
        if not decision.aggregation or not decision.target:
            return decision, warnings, True
        
        # Step 1: Schema validation
        schema_valid, schema_errors = self.schema_validator.validate_decision(decision)
        
        if not schema_valid:
            logger.warning(f"Schema validation failed: {schema_errors}")
            
            # Try auto-correction
            suggestions = self.schema_validator.suggest_correction(decision, schema_errors)
            if suggestions:
                logger.info(f"Auto-correcting decision: {suggestions}")
                
                # Apply corrections
                if 'metric' in suggestions:
                    old_metric = decision.aggregation.metric
                    decision.aggregation.metric = suggestions['metric']
                    warnings.append(AnalystWarning(
                        code=WarningCode.SCHEMA_MISMATCH,
                        message=f"Auto-corrected metric '{old_metric}' → '{suggestions['metric']}'"
                    ))
                
                if 'group_by' in suggestions:
                    old_group = decision.aggregation.group_by
                    decision.aggregation.group_by = suggestions['group_by']
                    warnings.append(AnalystWarning(
                        code=WarningCode.SCHEMA_MISMATCH,
                        message=f"Auto-corrected group field '{old_group}' → '{suggestions['group_by']}'"
                    ))
                
                if 'time_field' in suggestions and decision.aggregation.time:
                    old_field = decision.aggregation.time.field
                    decision.aggregation.time.field = suggestions['time_field']
                    warnings.append(AnalystWarning(
                        code=WarningCode.SCHEMA_MISMATCH,
                        message=f"Auto-corrected time field '{old_field}' → '{suggestions['time_field']}'"
                    ))
                
                # Re-validate after correction
                schema_valid, schema_errors = self.schema_validator.validate_decision(decision)
            
            # If still invalid, add error warnings
            if not schema_valid:
                is_valid = False
                for error in schema_errors:
                    warnings.append(AnalystWarning(
                        code=WarningCode.SCHEMA_MISMATCH,
                        message=error
                    ))
        
        # Step 2: Coverage validation (only if schema is valid)
        if schema_valid and decision.aggregation.time:
            has_data, coverage_msg = self.coverage_validator.validate_time_range(
                decision.target,
                decision.aggregation.time
            )
            
            if not has_data:
                logger.warning(f"Coverage validation failed: {coverage_msg}")
                warnings.append(AnalystWarning(
                    code=WarningCode.PARTIAL_PERIOD,
                    message=coverage_msg
                ))
                
                # Try to suggest alternative
                alt_suggestion = self.coverage_validator.suggest_alternative_range(
                    decision.target,
                    decision.aggregation.time.field
                )
                if alt_suggestion:
                    warnings[-1].message += f". {alt_suggestion}"
        
        # Step 3: Query preview (only if schema and coverage are valid)
        if schema_valid and is_valid:
            estimated_count, preview_warning = self.query_previewer.preview_count(
                decision.target,
                decision.aggregation
            )
            
            if estimated_count == 0:
                logger.warning(f"Query preview: 0 results. {preview_warning}")
                warnings.append(AnalystWarning(
                    code=WarningCode.NO_DATA,
                    message=preview_warning or "Query will return no results"
                ))
                # Don't mark as invalid - let it execute and return empty with explanation
            elif estimated_count > 0:
                logger.info(f"Query preview: ~{estimated_count} matching records")
        
        return decision, warnings, is_valid
    
    def invalidate_caches(self, collection: str = None):
        """Invalidate all validation caches (useful after data ingestion)"""
        self.schema_validator.invalidate_cache(collection)
        self.coverage_validator.invalidate_cache(collection)
        logger.info(f"Validation caches invalidated for {collection or 'all collections'}")


# Singleton instance
validation_integration = ValidationIntegration()
