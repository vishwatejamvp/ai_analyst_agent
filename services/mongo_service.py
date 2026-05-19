"""MongoDB service: storage + aggregations.

The agent uses MongoDB as the primary store for ingested Excel/JSON rows
and for running aggregation pipelines (sum, avg, count, group-by, filter).
The LLM never computes business metrics — this service does.
"""

from __future__ import annotations

from typing import Any, Iterable

import certifi
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from models.enums import TimeBucket
from models.schemas import AggregationSpec, TimeSpec
from models.config import settings
from utils.exceptions import DatabaseError
from utils.logger import logger

def _mongo_uses_tls(uri: str) -> bool:
    u = uri.lower()
    return uri.startswith("mongodb+srv://") or "tls=true" in u or "ssl=true" in u


def _mongo_tls_ca_path() -> str | None:
    """CA bundle for server certificate verification (Atlas on macOS)."""
    explicit = settings.mongo_tls_ca_file
    if explicit:
        return explicit
    return certifi.where()


def _redact_mongo_uri(uri: str) -> str:
    """Avoid logging raw credentials."""
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    if "@" not in rest:
        return uri
    _userinfo, hostpath = rest.rsplit("@", 1)
    return f"{scheme}://***:***@{hostpath}"


_OP_TO_MONGO = {
    "sum": "$sum",
    "avg": "$avg",
    "average": "$avg",
    "mean": "$avg",
    "min": "$min",
    "max": "$max",
    "count": "$sum",  # implemented via {"$sum": 1}
}


class MongoService:
    """Thin wrapper around PyMongo with aggregation helpers."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
    ) -> None:
        self.uri = uri or settings.effective_mongo_uri
        self.db_name = db_name or settings.mongo_db
        self._client: MongoClient | None = None
        self.index_advisor = None  # Lazy-loaded to avoid circular imports

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    @property
    def client(self) -> MongoClient:
        if self._client is None:
            try:
                opts: dict[str, Any] = {
                    # Atlas is occasionally slow during idle reconnects; the
                    # default of 5s caused spurious target=None routing.
                    "serverSelectionTimeoutMS": 15000,
                    "connectTimeoutMS": 15000,
                    "retryReads": True,
                }
                if _mongo_uses_tls(self.uri):
                    ca = _mongo_tls_ca_path()
                    if ca:
                        opts["tlsCAFile"] = ca
                self._client = MongoClient(self.uri, **opts)
                self._client.admin.command("ping")
                logger.info(f"Connected to MongoDB at {_redact_mongo_uri(self.uri)}")
            except PyMongoError as exc:
                raise DatabaseError(
                    f"MongoDB connection to {_redact_mongo_uri(self.uri)} "
                    f"failed ({type(exc).__name__}: {exc})"
                ) from exc
        return self._client

    @property
    def db(self) -> Database:
        return self.client[self.db_name]

    def collection(self, name: str) -> Collection:
        return self.db[name]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def insert_many(self, collection: str, documents: list[dict[str, Any]]) -> list[str]:
        if not documents:
            return []
        try:
            result = self.collection(collection).insert_many(documents, ordered=False)
            return [str(_id) for _id in result.inserted_ids]
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo insert failed: {exc}") from exc

    def drop_collection(self, collection: str) -> None:
        try:
            self.db.drop_collection(collection)
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo drop failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def list_collections(self) -> list[str]:
        try:
            return sorted(self.db.list_collection_names())
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo list_collections failed: {exc}") from exc

    def find(
        self,
        collection: str,
        filter_: dict[str, Any] | None = None,
        limit: int = 100,
        projection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            cursor = self.collection(collection).find(
                filter_ or {}, projection or {"_id": 0}
            ).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo find failed: {exc}") from exc

    def fetch_by_ids(
        self, collection: str, ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        from bson import ObjectId

        object_ids = []
        for _id in ids:
            try:
                object_ids.append(ObjectId(_id))
            except Exception:  # noqa: BLE001
                continue
        if not object_ids:
            return []
        try:
            cursor = self.collection(collection).find(
                {"_id": {"$in": object_ids}}, {"_id": 0}
            )
            return list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo fetch_by_ids failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def aggregate(
        self, collection: str, pipeline: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        import time
        
        start = time.time()
        
        try:
            result = list(self.collection(collection).aggregate(pipeline))
            
            # Log query for index advisor (if enabled)
            exec_time_ms = (time.time() - start) * 1000
            if self.index_advisor is not None:
                self.index_advisor.log_query(collection, pipeline, exec_time_ms)
            
            return result
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo aggregate failed: {exc}") from exc

    def build_pipeline(self, spec: AggregationSpec) -> list[dict[str, Any]]:
        """Translate an :class:`AggregationSpec` into a Mongo pipeline.

        Pure: does not execute. Useful both for ``run_aggregation`` and for
        provenance previews exposed to analysts.
        """
        op = (spec.operation or "").lower().strip()
        if op not in _OP_TO_MONGO:
            raise DatabaseError(f"Unsupported aggregation operation: {spec.operation}")

        pipeline: list[dict[str, Any]] = []

        match: dict[str, Any] = {}
        if spec.filters:
            match.update(_build_match(spec.filters))
        if spec.time is not None:
            time_match = _time_range_match(spec.time)
            if time_match:
                if spec.time.field == "year":
                    yf = (
                        spec.time.range_from.year
                        if spec.time.range_from is not None
                        else None
                    )
                    yt = (
                        spec.time.range_to.year
                        if spec.time.range_to is not None
                        else None
                    )
                    if yf is not None and yt is not None:
                        match["year"] = {"$gte": yf, "$lte": yt}
                elif spec.time.field == "period":
                    pf = (
                        spec.time.range_from.strftime("%Y-%m")
                        if spec.time.range_from is not None
                        else None
                    )
                    pt = (
                        spec.time.range_to.strftime("%Y-%m")
                        if spec.time.range_to is not None
                        else None
                    )
                    if pf is not None and pt is not None:
                        match["period"] = {"$gte": pf, "$lte": pt}
                else:
                    match.setdefault(spec.time.field, {}).update(time_match)
        if match:
            pipeline.append({"$match": match})

        # The "_id" of the $group stage is what defines the row shape.
        # Three cases:
        #   1. time only          → _id = bucket   → label only       (single series)
        #   2. group_by only      → _id = group    → label only       (categorical)
        #   3. time + group_by    → _id = {bucket, group}             (MULTI-SERIES)
        # Case 3 used to silently drop the group_by, which made
        # "metric over time by category" charts impossible. We now keep
        # both dimensions and project them out as ``label`` (time bucket)
        # and ``series`` (group value).
        multi_series = spec.time is not None and bool(spec.group_by)
        if multi_series:
            group_id: Any = {
                "bucket": _time_bucket_expr(spec.time),
                "series": f"${spec.group_by}",
            }
        elif spec.time is not None:
            group_id = _time_bucket_expr(spec.time)
        elif spec.group_by:
            group_id = f"${spec.group_by}"
        else:
            group_id = None

        if op == "count":
            value_expr: dict[str, Any] = {"$sum": 1}
        else:
            if not spec.metric:
                raise DatabaseError(
                    f"Aggregation '{op}' requires a metric field"
                )
            value_expr = {_OP_TO_MONGO[op]: f"${spec.metric}"}

        pipeline.append(
            {
                "$group": {
                    "_id": group_id,
                    "value": value_expr,
                }
            }
        )

        # Sort: time-aware paths sort by bucket first (so series lines
        # are chronological); pure-categorical sorts by value DESC for
        # ranked bars; explicit order_by wins when the caller supplied one.
        if multi_series:
            pipeline.append({"$sort": {"_id.bucket": 1, "_id.series": 1}})
        elif spec.time is not None:
            pipeline.append({"$sort": {"_id": 1}})
        else:
            sort_field = spec.order_by or "value"
            sort_dir = -1 if sort_field == "value" else 1
            pipeline.append({"$sort": {sort_field: sort_dir}})

        if spec.limit and spec.time is None:
            pipeline.append({"$limit": int(spec.limit)})

        if multi_series:
            pipeline.append(
                {
                    "$project": {
                        "_id": 0,
                        "label": {"$ifNull": ["$_id.bucket", "(all)"]},
                        "series": {"$ifNull": ["$_id.series", "(unknown)"]},
                        "value": 1,
                    }
                }
            )
        else:
            pipeline.append(
                {
                    "$project": {
                        "_id": 0,
                        "label": {"$ifNull": ["$_id", "(all)"]},
                        "value": 1,
                    }
                }
            )
        return pipeline

    def run_aggregation(
        self,
        collection: str,
        spec: AggregationSpec,
    ) -> list[dict[str, Any]]:
        """Translate an :class:`AggregationSpec` into a Mongo pipeline and run it.

        Row contract:

        *  ``time`` only      → ``{label, value}``        single time series
        *  ``group_by`` only  → ``{label, value}``        ranked categories
        *  ``time + group_by``→ ``{label, series, value}`` MULTI-series time
                                  (one row per bucket × group value)

        Time-bucketed paths are sorted ascending by bucket so the chart
        layer can plot lines in chronological order without resorting.
        
        Enhanced with dynamic freshness detection: adds metadata to indicate
        data completeness based on temporal patterns.
        
        Enhanced with Redis caching: checks cache before running aggregation,
        stores result in cache after execution.
        """
        # Try Redis cache first
        from services.redis_aggregation_cache import redis_aggregation_cache
        
        cached_result = redis_aggregation_cache.get(collection, spec)
        if cached_result is not None:
            logger.info(
                f"Returning cached aggregation for {collection} "
                f"({len(cached_result)} rows)"
            )
            return cached_result
        
        # Cache miss - run MongoDB aggregation
        pipeline = self.build_pipeline(spec)
        logger.debug(f"Mongo pipeline on {collection}: {pipeline}")
        rows = self.aggregate(collection, pipeline)
        
        # Infer data freshness from temporal patterns (time-series only)
        if spec.time and rows:
            rows = self._infer_data_freshness(rows)
        
        # Store in Redis cache
        redis_aggregation_cache.set(collection, spec, rows)
        
        return rows
    
    def _infer_data_freshness(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Infer data completeness from temporal patterns.
        
        Adds metadata fields to each row:
        - _data_complete: True if this period likely has complete data
        - _likely_missing: True if this period likely has missing/incomplete data
        
        Detection logic:
        - Find the last period with non-zero values
        - Mark all periods after that as likely missing
        """
        if not rows:
            return rows
        
        def _safe_number(val: Any) -> float:
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            try:
                return float(str(val).replace(",", ""))
            except (TypeError, ValueError):
                return 0.0
        
        # Find last non-zero period
        last_non_zero_idx = None
        for i in range(len(rows) - 1, -1, -1):
            if _safe_number(rows[i].get("value")) > 0:
                last_non_zero_idx = i
                break
        
        # Add metadata to each row
        enhanced_rows = []
        for i, row in enumerate(rows):
            enhanced_row = dict(row)
            if last_non_zero_idx is not None:
                enhanced_row["_data_complete"] = i <= last_non_zero_idx
                enhanced_row["_likely_missing"] = i > last_non_zero_idx
            else:
                # All zeros - mark all as potentially missing
                enhanced_row["_data_complete"] = False
                enhanced_row["_likely_missing"] = True
            enhanced_rows.append(enhanced_row)
        
        return enhanced_rows

    def latest_value(self, collection: str, field: str) -> Any:
        """Return the maximum ``field`` value in ``collection`` (None if empty)."""
        try:
            doc = self.collection(collection).find_one(
                {field: {"$exists": True, "$ne": None}},
                projection={field: 1, "_id": 0},
                sort=[(field, -1)],
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo latest_value failed: {exc}") from exc
        if not doc:
            return None
        return doc.get(field)

    def earliest_value(self, collection: str, field: str) -> Any:
        """Return the minimum ``field`` value in ``collection`` (None if empty).

        Mirrors :meth:`latest_value` but ascending. Used by the
        "what years do you actually have?" probe so the zero-rows
        branch can tell the user the real coverage span instead of a
        generic "try a different period".
        """
        try:
            doc = self.collection(collection).find_one(
                {field: {"$exists": True, "$ne": None}},
                projection={field: 1, "_id": 0},
                sort=[(field, 1)],
            )
        except PyMongoError as exc:
            raise DatabaseError(f"Mongo earliest_value failed: {exc}") from exc
        if not doc:
            return None
        return doc.get(field)

    def distinct_years(self, collection: str, field: str) -> list[int]:
        """Return sorted distinct calendar years present in ``collection``.

        Handles the three time-field shapes we see in the AWQAF facts
        collections:

        * ``year``   — already an integer year (most ``*_facts``)
        * ``period`` — ``YYYY-MM`` string; take the first 4 chars
        * anything else — assume a datetime / date and use ``$year``

        Capped at 50 distinct values (no AWQAF dataset spans more than
        a decade); fails *closed* (returns ``[]``) so a flaky probe
        never surfaces a wrong "what's available" message. The point
        of this method is **live truth**, not metadata — the caller
        relies on it instead of trusting the catalog's ``years``
        array, which can drift if ingest fails silently.
        """
        if field == "year":
            year_expr: Any = "$year"
        elif field == "period":
            year_expr = {
                "$convert": {
                    "input": {"$substr": ["$period", 0, 4]},
                    "to": "int",
                    "onError": None,
                    "onNull": None,
                }
            }
        else:
            year_expr = {"$year": f"${field}"}

        pipeline: list[dict[str, Any]] = [
            {"$match": {field: {"$exists": True, "$ne": None}}},
            {"$group": {"_id": year_expr}},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"_id": 1}},
            {"$limit": 50},
        ]
        try:
            rows = self.aggregate(collection, pipeline)
        except DatabaseError as exc:
            logger.warning(
                f"Mongo distinct_years on {collection}.{field} failed "
                f"({type(exc).__name__}: {exc}); returning empty."
            )
            return []
        years: list[int] = []
        for row in rows:
            val = row.get("_id")
            try:
                years.append(int(val))
            except (TypeError, ValueError):
                continue
        return years


_BUCKET_FORMAT = {
    TimeBucket.DAY: "%Y-%m-%d",
    TimeBucket.MONTH: "%Y-%m",
    TimeBucket.YEAR: "%Y",
    TimeBucket.WEEK: "%G-W%V",
}

_MONTH_NAME_TO_NUM: tuple[tuple[str, str], ...] = (
    ("january", "01"),
    ("february", "02"),
    ("march", "03"),
    ("april", "04"),
    ("may", "05"),
    ("june", "06"),
    ("july", "07"),
    ("august", "08"),
    ("september", "09"),
    ("october", "10"),
    ("november", "11"),
    ("december", "12"),
)


def _awqaf_year_month_bucket_expr() -> dict[str, Any]:
    """``YYYY-MM`` label from integer ``year`` + English ``month`` string (AWQAF)."""
    mfield = "$month"
    branches = [
        {"case": {"$eq": [{"$toLower": mfield}, name]}, "then": num}
        for name, num in _MONTH_NAME_TO_NUM
    ]
    return {
        "$concat": [
            {"$toString": "$year"},
            "-",
            {"$switch": {"branches": branches, "default": "00"}},
        ]
    }


def _time_bucket_expr(time: TimeSpec) -> dict[str, Any]:
    """Mongo aggregation expression that turns ``time.field`` into a bucket label.

    Output values are strings like ``2026-04``, ``2026-Q2``, etc., suitable for
    the ``label`` axis of the standard ``{label, value}`` row contract.
    """
    field = f"${time.field}"
    bucket = time.bucket

    if time.field == "period":
        # Already a ``YYYY-MM`` string at ingest time — no transform needed for
        # monthly trends. For yearly buckets, slice off the month suffix.
        if bucket == TimeBucket.YEAR:
            return {"$substr": [field, 0, 4]}
        return field

    if time.field == "year":
        if bucket == TimeBucket.MONTH:
            return _awqaf_year_month_bucket_expr()
        return {"$toString": field}

    if bucket == TimeBucket.QUARTER:
        return {
            "$concat": [
                {"$toString": {"$year": field}},
                "-Q",
                {
                    "$toString": {
                        "$ceil": {"$divide": [{"$month": field}, 3]}
                    }
                },
            ]
        }

    fmt = _BUCKET_FORMAT.get(bucket, "%Y-%m")
    return {"$dateToString": {"format": fmt, "date": field}}


def _time_range_match(time: TimeSpec) -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    if time.range_from is not None:
        bounds["$gte"] = time.range_from
    if time.range_to is not None:
        bounds["$lte"] = time.range_to
    return bounds


def _build_match(filters: dict[str, Any]) -> dict[str, Any]:
    """Normalize a flat filter dict into a Mongo ``$match`` stage."""
    match: dict[str, Any] = {}
    for key, value in filters.items():
        if isinstance(value, dict):
            mongo_clause: dict[str, Any] = {}
            for k, v in value.items():
                k_lower = k.lower()
                op = {
                    ">": "$gt",
                    ">=": "$gte",
                    "<": "$lt",
                    "<=": "$lte",
                    "!=": "$ne",
                    "in": "$in",
                    "nin": "$nin",
                    "like": "$regex",
                }.get(k_lower, k_lower)
                mongo_clause[op] = v
            match[key] = mongo_clause
        else:
            match[key] = value
    return match


mongo_service = MongoService()

# Initialize index advisor if auto-indexing is enabled
if settings.auto_create_indexes or settings.auto_drop_unused_indexes:
    from services.mongo_index_advisor import MongoIndexAdvisor
    
    mongo_service.index_advisor = MongoIndexAdvisor(
        mongo_service=mongo_service,
        analysis_interval=100,
        slow_query_threshold_ms=100.0,
    )
    logger.info("MongoDB index advisor initialized")
