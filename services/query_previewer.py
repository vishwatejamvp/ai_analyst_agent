"""Query result preview service.

Previews query result counts before full execution to catch zero-result
queries early and provide helpful diagnostic messages.
"""

from typing import Dict, Optional, Tuple, Any

from services.mongo_service import mongo_service
from utils.logger import logger


class QueryPreviewer:
    """Preview query results before full execution"""
    
    def __init__(self):
        self.mongo = mongo_service
    
    def preview_count(
        self, 
        collection: str, 
        spec
    ) -> Tuple[int, Optional[str]]:
        """
        Get approximate result count without full aggregation.
        
        Returns:
            (estimated_count, warning_message)
        """
        try:
            # Build match stage only (no grouping)
            match_stage = self._build_match_stage(spec)
            
            # Count matching documents
            pipeline = [match_stage, {"$count": "total"}]
            result = self.mongo.aggregate(collection, pipeline)
            
            count = result[0]['total'] if result else 0
            
            # Generate warning if count is 0
            warning = None
            if count == 0:
                warning = self._diagnose_zero_results(collection, spec)
            
            logger.info(f"Query preview for {collection}: ~{count} matching documents")
            return count, warning
            
        except Exception as e:
            logger.error(f"Preview failed for {collection}: {e}")
            return -1, None  # -1 = unknown
    
    def _build_match_stage(self, spec) -> Dict[str, Any]:
        """Build MongoDB $match stage from spec"""
        match: Dict[str, Any] = {}
        
        # Add filters
        if spec.filters:
            match.update(spec.filters)
        
        # Add time range
        if spec.time:
            time_match: Dict[str, Any] = {}
            
            if spec.time.field == "year":
                # Integer year field
                if spec.time.range_from:
                    time_match['$gte'] = spec.time.range_from.year
                if spec.time.range_to:
                    time_match['$lte'] = spec.time.range_to.year
            elif spec.time.field == "period":
                # YYYY-MM string field
                if spec.time.range_from:
                    time_match['$gte'] = spec.time.range_from.strftime('%Y-%m')
                if spec.time.range_to:
                    time_match['$lte'] = spec.time.range_to.strftime('%Y-%m')
            else:
                # Datetime field
                if spec.time.range_from:
                    time_match['$gte'] = spec.time.range_from
                if spec.time.range_to:
                    time_match['$lte'] = spec.time.range_to
            
            if time_match:
                match[spec.time.field] = time_match
        
        return {"$match": match} if match else {"$match": {}}
    
    def _diagnose_zero_results(self, collection: str, spec) -> str:
        """Diagnose why query returned 0 results"""
        # Check if collection has any data
        try:
            total_docs = len(self.mongo.find(collection, limit=1))
            if total_docs == 0:
                return f"Collection '{collection}' is empty"
        except:
            return "Could not access collection"
        
        # Check if time filter is too restrictive
        if spec.time:
            # Try without time filter
            try:
                spec_copy = spec.model_copy(update={'time': None})
                count_no_time, _ = self.preview_count(collection, spec_copy)
                
                if count_no_time > 0:
                    # Time filter is the issue
                    if spec.time.years:
                        return (
                            f"No data for years: {', '.join(map(str, spec.time.years))}. "
                            f"Collection has {count_no_time} records in other periods."
                        )
                    else:
                        return (
                            f"No data for requested time period. "
                            f"Collection has {count_no_time} records in other periods."
                        )
            except:
                pass
        
        # Check if filters are too restrictive
        if spec.filters:
            try:
                spec_copy = spec.model_copy(update={'filters': {}})
                count_no_filters, _ = self.preview_count(collection, spec_copy)
                
                if count_no_filters > 0:
                    return (
                        f"No data matching filters: {spec.filters}. "
                        f"Collection has {count_no_filters} records without filters."
                    )
            except:
                pass
        
        # Check if metric field exists but has no values
        if spec.metric:
            try:
                # Check if metric field has any non-null values
                pipeline = [
                    {"$match": {spec.metric: {"$exists": True, "$ne": None}}},
                    {"$limit": 1}
                ]
                result = self.mongo.aggregate(collection, pipeline)
                if not result:
                    return (
                        f"Metric '{spec.metric}' exists but has no values in this collection"
                    )
            except:
                pass
        
        return "No matching records found. Try broadening your query criteria."
    
    def check_metric_availability(
        self, 
        collection: str, 
        metric: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if a metric field has usable data"""
        try:
            # Check if field exists and has non-null numeric values
            pipeline = [
                {
                    "$match": {
                        metric: {
                            "$exists": True,
                            "$ne": None,
                            "$type": ["int", "double", "long", "decimal"]
                        }
                    }
                },
                {"$limit": 1}
            ]
            result = self.mongo.aggregate(collection, pipeline)
            
            if result:
                return True, None
            else:
                return False, f"Metric '{metric}' has no numeric values"
                
        except Exception as e:
            logger.error(f"Metric availability check failed: {e}")
            return False, f"Could not check metric '{metric}'"


# Singleton instance
query_previewer = QueryPreviewer()
