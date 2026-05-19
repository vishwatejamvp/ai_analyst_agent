"""Data coverage validation service.

Validates that requested time ranges actually have data in the database,
preventing "no records found" errors by checking coverage before execution.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

from services.mongo_service import mongo_service
from utils.logger import logger


class CoverageValidator:
    """Validate queries against actual data coverage"""
    
    def __init__(self):
        self.mongo = mongo_service
        self._coverage_cache: Dict[str, Dict] = {}
    
    def validate_time_range(self, collection: str, time_spec) -> Tuple[bool, Optional[str]]:
        """
        Check if requested time range has data.
        
        Returns:
            (has_data, suggestion_message)
        """
        if not time_spec:
            return True, None
        
        # Get actual coverage
        coverage = self._get_coverage(collection, time_spec.field)
        if not coverage:
            return False, f"No time data found in collection '{collection}'"
        
        # Check if requested range overlaps with actual data
        requested_from = time_spec.range_from
        requested_to = time_spec.range_to
        
        if requested_from and requested_to:
            # Check overlap
            has_overlap = (
                requested_from <= coverage['max_date'] and
                requested_to >= coverage['min_date']
            )
            
            if not has_overlap:
                suggestion = (
                    f"No data for requested period. "
                    f"Available data: {self._format_date(coverage['min_date'])} to "
                    f"{self._format_date(coverage['max_date'])}"
                )
                return False, suggestion
        
        # Check if requested years exist
        if time_spec.years:
            available_years = coverage.get('years', [])
            missing_years = [y for y in time_spec.years if y not in available_years]
            
            if missing_years:
                suggestion = (
                    f"No data for years: {', '.join(map(str, missing_years))}. "
                    f"Available years: {', '.join(map(str, sorted(available_years)))}"
                )
                return False, suggestion
        
        return True, None
    
    def _get_coverage(self, collection: str, time_field: str) -> Optional[Dict]:
        """Get actual time coverage for collection"""
        cache_key = f"{collection}:{time_field}"
        if cache_key in self._coverage_cache:
            return self._coverage_cache[cache_key]
        
        try:
            # Get min/max dates
            min_val = self.mongo.earliest_value(collection, time_field)
            max_val = self.mongo.latest_value(collection, time_field)
            
            if not min_val or not max_val:
                logger.warning(f"No time data found in {collection}.{time_field}")
                return None
            
            # Convert to datetime if needed
            min_date = self._to_datetime(min_val)
            max_date = self._to_datetime(max_val)
            
            if not min_date or not max_date:
                logger.warning(f"Could not parse dates from {collection}.{time_field}")
                return None
            
            # Get distinct years
            years = self.mongo.distinct_years(collection, time_field)
            
            coverage = {
                'min_date': min_date,
                'max_date': max_date,
                'years': sorted(years),
                'field': time_field,
                'collection': collection
            }
            
            self._coverage_cache[cache_key] = coverage
            logger.info(
                f"Coverage for {collection}.{time_field}: "
                f"{len(years)} years ({min(years) if years else 'N/A'}-{max(years) if years else 'N/A'})"
            )
            return coverage
            
        except Exception as e:
            logger.error(f"Coverage check failed for {collection}.{time_field}: {e}")
            return None
    
    def _to_datetime(self, value) -> Optional[datetime]:
        """Convert various date formats to datetime"""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, int):
            # Assume it's a year
            return datetime(value, 1, 1, tzinfo=timezone.utc)
        elif isinstance(value, str):
            # Try to parse YYYY-MM format
            try:
                if len(value) == 7 and value[4] == '-':  # YYYY-MM
                    year, month = value.split('-')
                    return datetime(int(year), int(month), 1, tzinfo=timezone.utc)
                elif len(value) == 4:  # YYYY
                    return datetime(int(value), 1, 1, tzinfo=timezone.utc)
            except:
                pass
        return None
    
    def _format_date(self, dt: datetime) -> str:
        """Format datetime for display"""
        if dt.month == 1 and dt.day == 1:
            return str(dt.year)
        return dt.strftime('%Y-%m')
    
    def suggest_alternative_range(self, collection: str, time_field: str) -> Optional[str]:
        """Suggest alternative time range with actual data"""
        coverage = self._get_coverage(collection, time_field)
        if not coverage:
            return None
        
        years = coverage.get('years', [])
        if not years:
            return None
        
        # Suggest most recent year
        latest_year = max(years)
        return f"Try querying {latest_year} instead (most recent year with data)"
    
    def get_available_years(self, collection: str, time_field: str) -> List[int]:
        """Get list of years with actual data"""
        coverage = self._get_coverage(collection, time_field)
        if not coverage:
            return []
        return coverage.get('years', [])
    
    def invalidate_cache(self, collection: str = None):
        """Invalidate coverage cache (useful after data ingestion)"""
        if collection:
            # Remove all cache entries for this collection
            keys_to_remove = [k for k in self._coverage_cache if k.startswith(f"{collection}:")]
            for key in keys_to_remove:
                self._coverage_cache.pop(key, None)
            logger.info(f"Coverage cache invalidated for {collection}")
        else:
            self._coverage_cache.clear()
            logger.info("Coverage cache cleared")


# Singleton instance
coverage_validator = CoverageValidator()
