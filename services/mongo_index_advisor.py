"""Real-time MongoDB index advisor based on query patterns.

Analyzes actual query execution patterns and recommends/creates indexes
automatically. Avoids over-indexing by only creating indexes for frequently
slow queries.

Architecture:
- Logs all aggregation queries with execution time
- Analyzes slow queries (>100ms) every N queries
- Extracts fields from $match, $sort stages
- Checks if covering index exists
- Auto-creates indexes in background (non-blocking)
- Tracks index usage and drops unused indexes
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from models.config import settings
from utils.logger import logger


@dataclass
class QueryPattern:
    """Represents a query pattern for index analysis."""

    collection: str
    match_fields: list[str]
    sort_fields: list[tuple[str, int]]  # (field, direction)
    exec_time_ms: float
    timestamp: datetime
    count: int = 1


@dataclass
class IndexRecommendation:
    """Index recommendation with justification."""

    collection: str
    fields: list[tuple[str, int]]  # (field, direction)
    reason: str
    avg_exec_time_ms: float
    query_count: int
    created: bool = False
    created_at: datetime | None = None


class MongoIndexAdvisor:
    """Runtime index recommendation and auto-creation based on query patterns."""

    def __init__(
        self,
        mongo_service: Any,  # MongoService
        analysis_interval: int = 100,
        slow_query_threshold_ms: float = 100.0,
    ) -> None:
        self.mongo = mongo_service
        self.analysis_interval = analysis_interval
        self.slow_query_threshold_ms = slow_query_threshold_ms

        # Query log for pattern analysis
        self.query_log: list[dict[str, Any]] = []

        # Index cache: collection -> list of index specs
        self.index_cache: dict[str, list[dict[str, Any]]] = {}

        # Pattern aggregation: (collection, fields) -> QueryPattern
        self.patterns: dict[tuple[str, str], QueryPattern] = {}

        # Recommendations history
        self.recommendations: list[IndexRecommendation] = []

        # Index usage tracking
        self.index_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def log_query(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
        exec_time_ms: float,
    ) -> None:
        """Track query patterns for index recommendations.

        Args:
            collection: Collection name
            pipeline: Aggregation pipeline
            exec_time_ms: Execution time in milliseconds
        """
        self.query_log.append(
            {
                "collection": collection,
                "pipeline": pipeline,
                "exec_time_ms": exec_time_ms,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        # Analyze every N queries
        if len(self.query_log) >= self.analysis_interval:
            self._analyze_and_recommend()

    def _analyze_and_recommend(self) -> None:
        """Analyze slow queries and recommend/create indexes."""
        slow_queries = [
            q
            for q in self.query_log
            if q["exec_time_ms"] > self.slow_query_threshold_ms
        ]

        if not slow_queries:
            self.query_log.clear()
            return

        logger.info(
            f"Analyzing {len(slow_queries)} slow queries "
            f"(>{self.slow_query_threshold_ms}ms)"
        )

        for query in slow_queries:
            # Extract fields from pipeline
            match_fields = self._extract_match_fields(query["pipeline"])
            sort_fields = self._extract_sort_fields(query["pipeline"])

            if not match_fields and not sort_fields:
                continue

            # Aggregate pattern
            pattern_key = (
                query["collection"],
                ",".join(match_fields + [f[0] for f in sort_fields]),
            )

            if pattern_key in self.patterns:
                # Update existing pattern
                pattern = self.patterns[pattern_key]
                pattern.count += 1
                pattern.exec_time_ms = (
                    pattern.exec_time_ms * (pattern.count - 1)
                    + query["exec_time_ms"]
                ) / pattern.count
            else:
                # New pattern
                self.patterns[pattern_key] = QueryPattern(
                    collection=query["collection"],
                    match_fields=match_fields,
                    sort_fields=sort_fields,
                    exec_time_ms=query["exec_time_ms"],
                    timestamp=query["timestamp"],
                )

        # Recommend indexes for frequent slow patterns
        self._recommend_indexes()

        # Clear log
        self.query_log.clear()

    def _extract_match_fields(self, pipeline: list[dict[str, Any]]) -> list[str]:
        """Extract fields from $match stages."""
        fields = []

        for stage in pipeline:
            if "$match" in stage:
                match_doc = stage["$match"]
                fields.extend(self._extract_fields_from_doc(match_doc))

        return list(dict.fromkeys(fields))  # Deduplicate, preserve order

    def _extract_sort_fields(
        self, pipeline: list[dict[str, Any]]
    ) -> list[tuple[str, int]]:
        """Extract fields from $sort stages."""
        for stage in pipeline:
            if "$sort" in stage:
                sort_doc = stage["$sort"]
                return [(field, direction) for field, direction in sort_doc.items()]

        return []

    def _extract_fields_from_doc(self, doc: dict[str, Any]) -> list[str]:
        """Recursively extract field names from a query document."""
        fields = []

        for key, value in doc.items():
            if key.startswith("$"):
                # Operator, recurse into value
                if isinstance(value, dict):
                    fields.extend(self._extract_fields_from_doc(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            fields.extend(self._extract_fields_from_doc(item))
            else:
                # Field name
                fields.append(key)

        return fields

    def _recommend_indexes(self) -> None:
        """Generate index recommendations from patterns."""
        for pattern_key, pattern in self.patterns.items():
            # Only recommend if pattern appears frequently and is slow
            if pattern.count < 3:
                continue

            # Build index spec
            index_fields = []

            # Match fields first (for filtering)
            for field in pattern.match_fields:
                index_fields.append((field, ASCENDING))

            # Sort fields next (for sorting)
            for field, direction in pattern.sort_fields:
                if field not in pattern.match_fields:
                    index_fields.append((field, direction))

            if not index_fields:
                continue

            # Check if covering index exists
            if self._has_covering_index(pattern.collection, index_fields):
                logger.debug(
                    f"Covering index exists for {pattern.collection}: {index_fields}"
                )
                continue

            # Create recommendation
            recommendation = IndexRecommendation(
                collection=pattern.collection,
                fields=index_fields,
                reason=f"Frequent slow queries ({pattern.count}x, avg {pattern.exec_time_ms:.0f}ms)",
                avg_exec_time_ms=pattern.exec_time_ms,
                query_count=pattern.count,
            )

            self.recommendations.append(recommendation)

            logger.info(
                f"Index recommendation: {pattern.collection} on {index_fields} "
                f"({pattern.count} queries, avg {pattern.exec_time_ms:.0f}ms)"
            )

            # Auto-create if enabled
            if settings.auto_create_indexes:
                self._create_index(recommendation)

    def _has_covering_index(
        self, collection: str, fields: list[tuple[str, int]]
    ) -> bool:
        """Check if a covering index exists for the given fields.

        An index covers a query if it starts with the query's fields in order.
        """
        # Refresh index cache if needed
        if collection not in self.index_cache:
            self._refresh_index_cache(collection)

        existing_indexes = self.index_cache.get(collection, [])

        # Extract field names from requested index
        requested_fields = [f[0] for f in fields]

        for index in existing_indexes:
            # Skip special indexes
            if index.get("name") == "_id_":
                continue

            # Get index key spec
            index_key = index.get("key", {})
            index_fields = list(index_key.keys())

            # Check if index covers requested fields
            # Index covers if it starts with requested fields in order
            if len(index_fields) >= len(requested_fields):
                if index_fields[: len(requested_fields)] == requested_fields:
                    return True

        return False

    def _refresh_index_cache(self, collection: str) -> None:
        """Refresh index cache for a collection."""
        try:
            indexes = list(self.mongo.collection(collection).list_indexes())
            self.index_cache[collection] = indexes
            logger.debug(f"Refreshed index cache for {collection}: {len(indexes)} indexes")
        except PyMongoError as exc:
            logger.warning(f"Failed to list indexes for {collection}: {exc}")
            self.index_cache[collection] = []

    def _create_index(self, recommendation: IndexRecommendation) -> None:
        """Create index with background=True to avoid blocking writes."""
        try:
            # Generate index name
            field_names = "_".join([f[0] for f in recommendation.fields])
            index_name = f"auto_{field_names}"

            # Create index in background
            self.mongo.collection(recommendation.collection).create_index(
                recommendation.fields,
                background=True,  # Non-blocking
                name=index_name,
            )

            recommendation.created = True
            recommendation.created_at = datetime.now(timezone.utc)

            logger.info(
                f"Created index on {recommendation.collection}: "
                f"{recommendation.fields} (name={index_name})"
            )

            # Refresh cache
            self._refresh_index_cache(recommendation.collection)

        except PyMongoError as exc:
            logger.error(
                f"Failed to create index on {recommendation.collection}: {exc}"
            )

    def get_recommendations(
        self, collection: str | None = None
    ) -> list[IndexRecommendation]:
        """Get index recommendations, optionally filtered by collection."""
        if collection:
            return [r for r in self.recommendations if r.collection == collection]
        return list(self.recommendations)

    def get_index_stats(self, collection: str) -> dict[str, Any]:
        """Get index statistics for a collection."""
        try:
            # Refresh cache
            self._refresh_index_cache(collection)

            indexes = self.index_cache.get(collection, [])

            # Get index stats from MongoDB
            stats = self.mongo.db.command("collStats", collection)

            return {
                "collection": collection,
                "index_count": len(indexes),
                "indexes": [
                    {
                        "name": idx.get("name"),
                        "key": idx.get("key"),
                        "size_bytes": stats.get("indexSizes", {}).get(
                            idx.get("name"), 0
                        ),
                    }
                    for idx in indexes
                ],
                "total_index_size_bytes": stats.get("totalIndexSize", 0),
            }

        except PyMongoError as exc:
            logger.error(f"Failed to get index stats for {collection}: {exc}")
            return {"error": str(exc)}

    def drop_unused_indexes(
        self, collection: str, min_age_days: int = 7
    ) -> list[str]:
        """Drop indexes that haven't been used in min_age_days.

        WARNING: Use with caution in production.

        Args:
            collection: Collection name
            min_age_days: Minimum age in days before dropping

        Returns:
            List of dropped index names
        """
        if not settings.auto_drop_unused_indexes:
            logger.warning("Auto-drop unused indexes is disabled")
            return []

        dropped = []

        try:
            # Get index stats
            index_stats = self.mongo.db.command("indexStats", collection)

            cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)

            for stat in index_stats.get("indexStats", []):
                index_name = stat.get("name")

                # Never drop _id index
                if index_name == "_id_":
                    continue

                # Check last access time
                accesses = stat.get("accesses", {})
                last_access = accesses.get("since")

                if last_access and last_access < cutoff:
                    logger.info(
                        f"Dropping unused index {index_name} on {collection} "
                        f"(last access: {last_access})"
                    )

                    self.mongo.collection(collection).drop_index(index_name)
                    dropped.append(index_name)

            # Refresh cache
            if dropped:
                self._refresh_index_cache(collection)

        except PyMongoError as exc:
            logger.error(f"Failed to drop unused indexes on {collection}: {exc}")

        return dropped


# Note: Singleton will be created in mongo_service.py to avoid circular imports
