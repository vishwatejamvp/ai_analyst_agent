"""Redis-based metadata cache.

Caches collection names and schemas globally (shared across all users).
"""

from __future__ import annotations

import json
from typing import Any

from models.config import settings
from utils.logger import logger


class RedisMetadataCache:
    """Cache collection metadata in Redis."""
    
    def __init__(self) -> None:
        self._redis_client = None
        self._redis_available = None
        
        # Metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
        }
    
    def _get_redis_client(self):
        """Lazy-load Redis client with availability check."""
        if self._redis_client is not None:
            return self._redis_client
        
        if not getattr(settings, "redis_url", None):
            self._redis_available = False
            return None
        
        if not getattr(settings, "metadata_cache_enabled", True):
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
            logger.info(f"Redis metadata cache connected: {settings.redis_url}")
            return self._redis_client
            
        except Exception as exc:  # noqa: BLE001
            self._redis_available = False
            logger.warning(
                f"Redis metadata cache unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to direct MongoDB queries"
            )
            return None
    
    def get_collections(self) -> list[str] | None:
        """Get cached collection names.
        
        Returns:
            List of collection names if cached, None if miss
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return None
        
        try:
            cached = redis_client.get("metadata:collections")
            if cached:
                self.metrics["hits"] += 1
                logger.debug("Metadata cache HIT: collections")
                return json.loads(cached)
            
            self.metrics["misses"] += 1
            logger.debug("Metadata cache MISS: collections")
            return None
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache read error: {exc}")
            return None
    
    def set_collections(self, collections: list[str]) -> bool:
        """Cache collection names globally.
        
        Args:
            collections: List of collection names
            
        Returns:
            True if cached successfully
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return False
        
        try:
            ttl = getattr(settings, "metadata_cache_ttl", 300)
            redis_client.setex(
                "metadata:collections",
                ttl,
                json.dumps(collections)
            )
            self.metrics["sets"] += 1
            logger.debug(f"Cached {len(collections)} collection names (TTL={ttl}s)")
            return True
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache write error: {exc}")
            return False
    
    def get_schema(self, collection: str) -> dict[str, Any] | None:
        """Get cached schema for a collection.
        
        Returns:
            Schema dict if cached, None if miss
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return None
        
        try:
            cached = redis_client.get(f"metadata:schema:{collection}")
            if cached:
                self.metrics["hits"] += 1
                logger.debug(f"Metadata cache HIT: schema:{collection}")
                return json.loads(cached)
            
            self.metrics["misses"] += 1
            logger.debug(f"Metadata cache MISS: schema:{collection}")
            return None
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache read error: {exc}")
            return None
    
    def set_schema(self, collection: str, schema: dict[str, Any]) -> bool:
        """Cache schema for a collection.
        
        Args:
            collection: Collection name
            schema: Schema dict with columns and sample
            
        Returns:
            True if cached successfully
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return False
        
        try:
            ttl = getattr(settings, "metadata_cache_ttl", 300)
            redis_client.setex(
                f"metadata:schema:{collection}",
                ttl,
                json.dumps(schema, default=str)
            )
            self.metrics["sets"] += 1
            logger.debug(f"Cached schema for {collection} (TTL={ttl}s)")
            return True
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache write error: {exc}")
            return False
    
    def invalidate_collection(self, collection: str) -> int:
        """Invalidate cached schema for a collection.
        
        Returns:
            Number of keys deleted
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return 0
        
        try:
            deleted = redis_client.delete(f"metadata:schema:{collection}")
            if deleted:
                logger.info(f"Invalidated metadata cache for {collection}")
            return deleted
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache invalidation error: {exc}")
            return 0
    
    def invalidate_all(self) -> int:
        """Invalidate all cached metadata.
        
        Returns:
            Number of keys deleted
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            return 0
        
        try:
            pattern = "metadata:*"
            keys = list(redis_client.scan_iter(match=pattern, count=100))
            
            if keys:
                deleted = redis_client.delete(*keys)
                logger.info(f"Invalidated all {deleted} metadata cache entries")
                return deleted
            
            return 0
        except Exception as exc:  # noqa: BLE001
            self.metrics["errors"] += 1
            logger.warning(f"Redis metadata cache invalidation error: {exc}")
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
            "enabled": getattr(settings, "metadata_cache_enabled", True),
            "redis_available": self._redis_available,
        }


# Module-level singleton
redis_metadata_cache = RedisMetadataCache()
