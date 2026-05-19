"""Redis-based aggregation result cache.

Caches MongoDB aggregation results to avoid repeated queries.
Automatically invalidates on data ingestion.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from models.schemas import AggregationSpec
from models.config import settings
from utils.logger import logger


class RedisAggregationCache:
    """Cache MongoDB aggregation results in Redis."""
    
    def __init__(self) -> None:
        self._redis_client = None
        self._redis_available = None
        
        # Metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "invalidations": 0,
        }
    
    def _get_redis_client(self):
        """Lazy-load Redis client with availability check."""
        if self._redis_client is not None:
            return self._redis_client
        
        if not getattr(settings, "redis_url", None):
            self._redis_available = False
            return None
        
        if not getattr(settings, "aggregation_cache_enabled", True):
            self._redis_available = False
            return None
        
        try:
            import redis
            
            self._redis_client = redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
            )
            
            # Test connection
            self._redis_client.ping()
            self._redis_available = True
            logger.info(f"Redis aggregation cache connected: {settings.redis_url}")
            return self._redis_client
            
        except Exception as exc:  # noqa: BLE001
            self._redis_available = False
            logger.warning(
                f"Redis aggregation cache unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to direct MongoDB queries"
            )
            return None
    
    def _build_cache_key(
        self,
        collection: str,
        spec: AggregationSpec,
    ) -> str:
        """Build deterministic cache key from query parameters.
        
        Key format: "agg:{collection}:{spec_hash}"
        
        Example: "agg:awqaf_hajj_package_service_facts:a3f2b1c4d5e6"
        """
        # Serialize spec to stable JSON
        spec_dict = spec.model_dump(mode="json", exclude_none=True)
        spec_json = json.dumps(spec_dict, sort_keys=True)
        
        # Hash for compact key
        spec_hash = hashlib.sha256(spec_json.encode()).hexdigest()[:12]
        
        return f"agg:{collection}:{spec_hash}"
    
    def _get_ttl(self, spec: AggregationSpec) -> int:
        """Determine TTL based on query characteristics.
        
        Strategy:
        - Historical data (>1 year old): 15 minutes (stable)
        - Current year data: 3 minutes (frequently updated)
        - Default: 5 minutes
        """
        # Check if query has time range
        if spec.time and spec.time.range_to:
            current_year = datetime.now().year
            query_year = spec.time.range_to.year
            
            # Historical data - longer TTL
            if query_year < current_year:
                return getattr(settings, "aggregation_cache_ttl_historical", 900)
            
            # Current year - shorter TTL
            if query_year == current_year:
                return getattr(settings, "aggregation_cache_ttl_current", 180)
        
        # Default TTL
        return getattr(settings, "aggregation_cache_ttl", 300)
    
    def get(
        self,
        collection: str,
        spec: AggregationSpec,
    ) -> list[dict[str, Any]] | None:
        """Get cached aggregation result.
        
        Returns:
            Cached result if found, None otherwise
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return None
        
        try:
            cache_key = self._build_cache_key(collection, spec)
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                self.metrics["hits"] += 1
                hit_rate = self.hit_rate
                logger.info(
                    f"Aggregation cache HIT: {cache_key[:50]}... "
                    f"(hit_rate={hit_rate:.1%})"
                )
                return json.loads(cached_data)
            
            self.metrics["misses"] += 1
            logger.debug(f"Aggregation cache MISS: {cache_key[:50]}...")
            return None
            
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis aggregation cache read error: {exc}")
            return None
    
    def set(
        self,
        collection: str,
        spec: AggregationSpec,
        result: list[dict[str, Any]],
    ) -> bool:
        """Store aggregation result in cache.
        
        Returns:
            True if cached successfully, False otherwise
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return False
        
        try:
            cache_key = self._build_cache_key(collection, spec)
            cached_data = json.dumps(result, default=str)
            ttl = self._get_ttl(spec)
            
            redis_client.setex(cache_key, ttl, cached_data)
            
            self.metrics["sets"] += 1
            logger.debug(
                f"Aggregation cache SET: {cache_key[:50]}... "
                f"({len(result)} rows, TTL={ttl}s)"
            )
            return True
            
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis aggregation cache write error: {exc}")
            return False
    
    def invalidate(self, collection: str) -> int:
        """Invalidate all cached results for a collection.
        
        Useful after data ingestion.
        
        Returns:
            Number of keys deleted
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return 0
        
        try:
            pattern = f"agg:{collection}:*"
            keys = list(redis_client.scan_iter(match=pattern, count=100))
            
            if keys:
                deleted = redis_client.delete(*keys)
                self.metrics["invalidations"] += 1
                logger.info(
                    f"Invalidated {deleted} aggregation cache entries for {collection}"
                )
                return deleted
            
            return 0
            
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis aggregation cache invalidation error: {exc}")
            return 0
    
    def invalidate_all(self) -> int:
        """Invalidate all cached aggregation results.
        
        Returns:
            Number of keys deleted
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return 0
        
        try:
            pattern = "agg:*"
            keys = list(redis_client.scan_iter(match=pattern, count=100))
            
            if keys:
                deleted = redis_client.delete(*keys)
                self.metrics["invalidations"] += 1
                logger.info(f"Invalidated all {deleted} aggregation cache entries")
                return deleted
            
            return 0
            
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis aggregation cache invalidation error: {exc}")
            return 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.metrics["hits"] + self.metrics["misses"]
        return self.metrics["hits"] / total if total > 0 else 0.0
    
    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics for monitoring."""
        return {
            **self.metrics,
            "hit_rate": self.hit_rate,
            "enabled": getattr(settings, "aggregation_cache_enabled", True),
            "redis_available": self._redis_available,
        }


# Module-level singleton
redis_aggregation_cache = RedisAggregationCache()
