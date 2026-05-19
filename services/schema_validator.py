"""Schema validation service to prevent querying non-existent fields.

Validates routing decisions against actual database schema before execution,
eliminating "field not found" errors and hallucinated metrics.
"""

from typing import Dict, List, Optional, Tuple
from difflib import get_close_matches

from services.mongo_service import mongo_service
from utils.logger import logger


class SchemaValidator:
    """Validate routing decisions against actual database schema"""
    
    def __init__(self):
        self.mongo = mongo_service
        self._schema_cache: Dict[str, Dict] = {}
    
    def validate_decision(self, decision) -> Tuple[bool, List[str]]:
        """
        Validate routing decision against actual schema.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        if not decision.target:
            errors.append("No target collection selected")
            return False, errors
        
        # Get actual schema
        schema = self._get_schema(decision.target)
        if not schema:
            errors.append(f"Collection '{decision.target}' does not exist")
            return False, errors
        
        # Validate metric exists
        if decision.aggregation and decision.aggregation.metric:
            metric = decision.aggregation.metric
            if metric not in schema['fields']:
                # Find similar fields
                similar = self._find_similar_fields(metric, schema['numeric_fields'])
                if similar:
                    errors.append(
                        f"Metric '{metric}' not found in {decision.target}. "
                        f"Did you mean: {', '.join(similar[:3])}?"
                    )
                else:
                    errors.append(
                        f"Metric '{metric}' not found. "
                        f"Available numeric fields: {', '.join(schema['numeric_fields'][:5])}"
                    )
        
        # Validate group_by exists
        if decision.aggregation and decision.aggregation.group_by:
            group_by = decision.aggregation.group_by
            if group_by not in schema['fields']:
                similar = self._find_similar_fields(group_by, schema['fields'])
                if similar:
                    errors.append(
                        f"Group field '{group_by}' not found. "
                        f"Did you mean: {', '.join(similar[:3])}?"
                    )
                else:
                    errors.append(
                        f"Group field '{group_by}' not found in schema"
                    )
        
        # Validate time field exists
        if decision.aggregation and decision.aggregation.time:
            time_field = decision.aggregation.time.field
            if time_field not in schema['fields']:
                time_fields = schema.get('time_fields', [])
                if time_fields:
                    errors.append(
                        f"Time field '{time_field}' not found. "
                        f"Available time fields: {', '.join(time_fields)}"
                    )
                else:
                    errors.append(
                        f"Time field '{time_field}' not found. "
                        f"Collection has no time fields."
                    )
        
        return len(errors) == 0, errors
    
    def _get_schema(self, collection: str) -> Optional[Dict]:
        """Get schema for collection (cached)"""
        if collection in self._schema_cache:
            return self._schema_cache[collection]
        
        try:
            # Sample documents to infer schema
            sample = self.mongo.find(collection, limit=50)
            if not sample:
                logger.warning(f"Collection '{collection}' is empty")
                return None
            
            # Extract all field names and types
            fields = set()
            time_fields = []
            numeric_fields = []
            categorical_fields = []
            
            for doc in sample:
                for key, value in doc.items():
                    if key == '_id':
                        continue
                    fields.add(key)
                    
                    # Identify time fields
                    if any(hint in key.lower() for hint in ['date', 'time', 'period', 'year', 'month', '_at']):
                        if key not in time_fields:
                            time_fields.append(key)
                    
                    # Identify numeric fields
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key not in numeric_fields:
                            numeric_fields.append(key)
                    
                    # Identify categorical fields
                    elif isinstance(value, str) and len(value) < 100:
                        if key not in categorical_fields:
                            categorical_fields.append(key)
            
            schema = {
                'fields': sorted(list(fields)),
                'time_fields': sorted(time_fields),
                'numeric_fields': sorted(numeric_fields),
                'categorical_fields': sorted(categorical_fields),
                'sample_count': len(sample)
            }
            
            self._schema_cache[collection] = schema
            logger.info(
                f"Schema cached for {collection}: "
                f"{len(schema['fields'])} fields, "
                f"{len(schema['numeric_fields'])} numeric"
            )
            return schema
            
        except Exception as e:
            logger.error(f"Schema validation failed for {collection}: {e}")
            return None
    
    def _find_similar_fields(self, target: str, available: List[str]) -> List[str]:
        """Find similar field names using fuzzy matching"""
        if not available:
            return []
        return get_close_matches(target, available, n=5, cutoff=0.6)
    
    def suggest_correction(self, decision, errors: List[str]) -> Optional[Dict]:
        """Suggest corrections for invalid decision"""
        if not decision.target:
            return None
        
        schema = self._get_schema(decision.target)
        if not schema:
            return None
        
        suggestions = {}
        
        # Suggest metric correction
        if decision.aggregation and decision.aggregation.metric:
            metric = decision.aggregation.metric
            if metric not in schema['fields']:
                similar = self._find_similar_fields(metric, schema['numeric_fields'])
                if similar:
                    suggestions['metric'] = similar[0]
                    logger.info(f"Auto-correcting metric '{metric}' → '{similar[0]}'")
        
        # Suggest group_by correction
        if decision.aggregation and decision.aggregation.group_by:
            group_by = decision.aggregation.group_by
            if group_by not in schema['fields']:
                similar = self._find_similar_fields(group_by, schema['categorical_fields'])
                if similar:
                    suggestions['group_by'] = similar[0]
                    logger.info(f"Auto-correcting group_by '{group_by}' → '{similar[0]}'")
        
        # Suggest time field correction
        if decision.aggregation and decision.aggregation.time:
            time_field = decision.aggregation.time.field
            if time_field not in schema['fields']:
                time_fields = schema.get('time_fields', [])
                if time_fields:
                    suggestions['time_field'] = time_fields[0]
                    logger.info(f"Auto-correcting time field '{time_field}' → '{time_fields[0]}'")
        
        return suggestions if suggestions else None
    
    def invalidate_cache(self, collection: str = None):
        """Invalidate schema cache (useful after data ingestion)"""
        if collection:
            self._schema_cache.pop(collection, None)
            logger.info(f"Schema cache invalidated for {collection}")
        else:
            self._schema_cache.clear()
            logger.info("Schema cache cleared")


# Singleton instance
schema_validator = SchemaValidator()
