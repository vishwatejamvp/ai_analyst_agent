"""Hybrid multi-tier session storage with automatic failover.

Architecture:
- L1: In-memory cache (fastest, limited capacity, LRU eviction)
- L2: Redis (persistent, shared across instances)
- L3: MongoDB (long-term archival, unlimited capacity)

Read Strategy (cache-aside):
- Try L1 → L2 → L3 in order
- Promote hits to higher tiers

Write Strategy (write-through):
- Write to L1 synchronously
- Write to L2 asynchronously (non-blocking)
- Write to L3 asynchronously (archival)

Failover:
- If Redis is unavailable, falls back to MongoDB
- If both are unavailable, uses in-memory only
- Circuit breaker pattern prevents cascading failures
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from models.schemas import RoutingDecision
from services.session_service import SessionRecord
from models.config import settings
from utils.logger import logger


class HybridSessionService:
    """Multi-tier session storage with automatic failover."""

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_ttl_seconds: int = 1800,  # 30 minutes
        l3_ttl_days: int = 7,
    ) -> None:
        self.l1_max_size = l1_max_size
        self.l2_ttl_seconds = l2_ttl_seconds
        self.l3_ttl_days = l3_ttl_days

        # L1: In-memory cache
        self.l1_cache: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

        # L2: Redis client (lazy-loaded)
        self._redis_client = None
        self._redis_available = None
        self._redis_last_check = datetime.min.replace(tzinfo=timezone.utc)
        self._redis_check_interval = timedelta(seconds=30)

        # L3: MongoDB (lazy-loaded)
        self._mongo_service = None

        # Metrics
        self.metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "l1_evictions": 0,
            "redis_failures": 0,
            "mongo_failures": 0,
        }

    def _get_mongo_service(self):
        """Lazy-load MongoDB service."""
        if self._mongo_service is None:
            from services.mongo_service import mongo_service

            self._mongo_service = mongo_service
        return self._mongo_service

    def _get_redis_client(self):
        """Lazy-load Redis client with circuit breaker."""
        # Circuit breaker: don't retry too frequently
        now = datetime.now(timezone.utc)
        if (
            self._redis_available is False
            and now - self._redis_last_check < self._redis_check_interval
        ):
            return None

        if self._redis_client is not None:
            return self._redis_client

        if not settings.redis_url:
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
            self._redis_last_check = now

            logger.info(f"Connected to Redis at {settings.redis_url}")
            return self._redis_client

        except Exception as exc:  # noqa: BLE001
            self._redis_available = False
            self._redis_last_check = now
            logger.warning(
                f"Redis unavailable ({type(exc).__name__}: {exc}); "
                f"using in-memory + MongoDB fallback"
            )
            return None

    def get(self, session_id: str | None) -> SessionRecord | None:
        """Retrieve session from multi-tier storage.

        Read path: L1 → L2 → L3
        Promotes hits to higher tiers.
        """
        if not session_id:
            return None

        # L1: In-memory (fastest)
        with self._lock:
            if session_id in self.l1_cache:
                record = self.l1_cache[session_id]
                # Update last_used_at
                record.last_used_at = datetime.now(timezone.utc)
                self.metrics["l1_hits"] += 1
                logger.debug(f"Session {session_id[:8]} hit L1 cache")
                return record

        # L2: Redis (shared, persistent)
        redis_client = self._get_redis_client()
        if redis_client is not None:
            try:
                data = redis_client.get(f"session:{session_id}")
                if data:
                    record = self._deserialize_session(data)
                    if record:
                        # Promote to L1
                        self._set_l1(session_id, record)
                        self.metrics["l2_hits"] += 1
                        logger.debug(f"Session {session_id[:8]} hit L2 (Redis)")
                        return record
            except Exception as exc:  # noqa: BLE001
                self.metrics["redis_failures"] += 1
                logger.warning(
                    f"Redis read failed for {session_id[:8]}: {exc}, "
                    f"falling back to L3"
                )

        # L3: MongoDB (archival, slowest)
        mongo = self._get_mongo_service()
        try:
            doc = mongo.collection("sessions").find_one(
                {"session_id": session_id}, {"_id": 0}
            )
            if doc:
                record = SessionRecord(**doc)

                # Check if expired
                if record.is_expired(timedelta(days=self.l3_ttl_days)):
                    # Clean up expired session
                    self._delete_from_l3(session_id)
                    self.metrics["misses"] += 1
                    return None

                # Promote to L2 and L1
                self._set_l2_async(session_id, record)
                self._set_l1(session_id, record)
                self.metrics["l3_hits"] += 1
                logger.debug(f"Session {session_id[:8]} hit L3 (MongoDB)")
                return record

        except Exception as exc:  # noqa: BLE001
            self.metrics["mongo_failures"] += 1
            logger.error(f"MongoDB read failed for {session_id[:8]}: {exc}")

        self.metrics["misses"] += 1
        return None

    def put(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
    ) -> None:
        """Store session in multi-tier storage.

        Write path: L1 (sync) + L2 (async) + L3 (async)
        """
        if not session_id:
            return

        now = datetime.now(timezone.utc)

        # Get existing record or create new
        existing = self.get(session_id)
        created_at = existing.created_at if existing else now

        # Build session record (same logic as original session_service)
        from models.enums import QueryRoute

        is_analytical = decision is not None and (
            decision.route == QueryRoute.ANALYTICAL
        )

        if is_analytical:
            last_analytical_question = question
            last_analytical_decision = decision
        elif existing is not None:
            last_analytical_question = existing.last_analytical_question
            last_analytical_decision = existing.last_analytical_decision
        else:
            last_analytical_question = None
            last_analytical_decision = None

        record = SessionRecord(
            session_id=session_id,
            last_question=question,
            last_decision=decision,
            created_at=created_at,
            last_used_at=now,
            last_analytical_question=last_analytical_question,
            last_analytical_decision=last_analytical_decision,
            turns=existing.turns if existing else [],
            summary=existing.summary if existing else None,
        )

        # Write to L1 (synchronous)
        self._set_l1(session_id, record)

        # Write to L2 (asynchronous)
        self._set_l2_async(session_id, record)

        # Write to L3 (asynchronous, archival)
        self._set_l3_async(session_id, record)

    def _set_l1(self, session_id: str, record: SessionRecord) -> None:
        """Set in L1 cache with LRU eviction."""
        with self._lock:
            # LRU eviction if cache is full
            if len(self.l1_cache) >= self.l1_max_size:
                # Find oldest session
                oldest_id = min(
                    self.l1_cache.keys(),
                    key=lambda k: self.l1_cache[k].last_used_at,
                )
                del self.l1_cache[oldest_id]
                self.metrics["l1_evictions"] += 1
                logger.debug(f"Evicted session {oldest_id[:8]} from L1 cache")

            self.l1_cache[session_id] = record

    def _set_l2_async(self, session_id: str, record: SessionRecord) -> None:
        """Set in L2 (Redis) asynchronously."""
        redis_client = self._get_redis_client()
        if redis_client is None:
            return

        def _write():
            try:
                data = self._serialize_session(record)
                redis_client.setex(
                    f"session:{session_id}",
                    self.l2_ttl_seconds,
                    data,
                )
                logger.debug(f"Wrote session {session_id[:8]} to L2 (Redis)")
            except Exception as exc:  # noqa: BLE001
                self.metrics["redis_failures"] += 1
                logger.warning(f"Redis write failed for {session_id[:8]}: {exc}")

        # Run in background thread
        threading.Thread(target=_write, daemon=True).start()

    def _set_l3_async(self, session_id: str, record: SessionRecord) -> None:
        """Set in L3 (MongoDB) asynchronously for archival."""
        mongo = self._get_mongo_service()

        def _write():
            try:
                # Convert to dict
                doc = asdict(record)

                # Upsert
                mongo.collection("sessions").update_one(
                    {"session_id": session_id},
                    {"$set": doc},
                    upsert=True,
                )
                logger.debug(f"Wrote session {session_id[:8]} to L3 (MongoDB)")
            except Exception as exc:  # noqa: BLE001
                self.metrics["mongo_failures"] += 1
                logger.warning(f"MongoDB write failed for {session_id[:8]}: {exc}")

        # Run in background thread
        threading.Thread(target=_write, daemon=True).start()

    def _delete_from_l3(self, session_id: str) -> None:
        """Delete expired session from MongoDB."""
        mongo = self._get_mongo_service()
        try:
            mongo.collection("sessions").delete_one({"session_id": session_id})
            logger.debug(f"Deleted expired session {session_id[:8]} from L3")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to delete session {session_id[:8]}: {exc}")

    @staticmethod
    def _serialize_session(record: SessionRecord) -> str:
        """Serialize session record to JSON string for Redis."""
        import json

        # Convert to dict
        data = asdict(record)

        # Convert datetime objects to ISO strings
        data["created_at"] = data["created_at"].isoformat()
        data["last_used_at"] = data["last_used_at"].isoformat()

        # Convert RoutingDecision to dict
        if data["last_decision"]:
            data["last_decision"] = data["last_decision"].model_dump()
        if data["last_analytical_decision"]:
            data["last_analytical_decision"] = data[
                "last_analytical_decision"
            ].model_dump()

        # Convert turns
        if data["turns"]:
            data["turns"] = [
                {
                    "question": t.question,
                    "insight": t.insight,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in record.turns
            ]

        return json.dumps(data)

    @staticmethod
    def _deserialize_session(data: str) -> SessionRecord | None:
        """Deserialize session record from JSON string."""
        import json

        try:
            obj = json.loads(data)

            # Convert ISO strings back to datetime
            obj["created_at"] = datetime.fromisoformat(obj["created_at"])
            obj["last_used_at"] = datetime.fromisoformat(obj["last_used_at"])

            # Convert dicts back to RoutingDecision
            if obj["last_decision"]:
                obj["last_decision"] = RoutingDecision(**obj["last_decision"])
            if obj["last_analytical_decision"]:
                obj["last_analytical_decision"] = RoutingDecision(
                    **obj["last_analytical_decision"]
                )

            # Convert turns
            from services.session_summary import Turn

            if obj["turns"]:
                obj["turns"] = [
                    Turn(
                        question=t["question"],
                        insight=t["insight"],
                        timestamp=datetime.fromisoformat(t["timestamp"]),
                    )
                    for t in obj["turns"]
                ]

            return SessionRecord(**obj)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to deserialize session: {exc}")
            return None

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics for monitoring."""
        total_requests = sum(
            [
                self.metrics["l1_hits"],
                self.metrics["l2_hits"],
                self.metrics["l3_hits"],
                self.metrics["misses"],
            ]
        )

        return {
            "l1_size": len(self.l1_cache),
            "l1_max_size": self.l1_max_size,
            "l1_hit_rate": (
                self.metrics["l1_hits"] / total_requests if total_requests > 0 else 0
            ),
            "l2_hit_rate": (
                self.metrics["l2_hits"] / total_requests if total_requests > 0 else 0
            ),
            "l3_hit_rate": (
                self.metrics["l3_hits"] / total_requests if total_requests > 0 else 0
            ),
            "miss_rate": (
                self.metrics["misses"] / total_requests if total_requests > 0 else 0
            ),
            "redis_available": self._redis_available,
            **self.metrics,
        }

    def clear_l1(self) -> int:
        """Clear L1 cache. Returns number of sessions cleared."""
        with self._lock:
            count = len(self.l1_cache)
            self.l1_cache.clear()
            return count


# Module-level singleton
hybrid_session_service = HybridSessionService()
