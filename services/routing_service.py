"""Query routing layer.

Inspects a natural-language question and decides:

* whether it needs an analytical (DB) path, semantic (vector) path,
  or a hybrid of both;
* if analytical: which operation (sum/avg/count/min/max), which metric,
  optional group-by, optional time bucket / range, and which target
  collection or table the query should hit;
* if semantic: it falls through to vector search.

The output is a fully-typed :class:`RoutingDecision`. Downstream services
do not re-interpret natural language — they consume this decision.

Rule-based on purpose: fast, deterministic, debuggable, and keeps numeric
calculations far from the LLM. An LLM-assisted planner is the Sprint 2
upgrade that will replace pieces of this.
"""

from __future__ import annotations

# Import semantic collection matcher
from services.semantic_collection_matcher import semantic_matcher

# Import semantic collection matcher
from services.semantic_collection_matcher import semantic_matcher

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from models.enums import ComparisonMode, DataSource, MetricStatus, QueryRoute, TimeBucket
from models.schemas import (
    AggregationSpec,
    GlossaryMatch,
    MetricFormula,
    RoutingDecision,
    TargetResolution,
    TimeSpec,
)
from services.catalog_routing_service import (
    catalog_routing_service,
    extract_group_by_before_by,
)
from models.config import settings
from services.knowledge_base_service import (
    KnowledgeBaseService,
    knowledge_base_service,
)
from services.mongo_service import MongoService, mongo_service
from services.mysql_service import MySQLService, mysql_service
from services.visualization.metric_registry import metric_semantic
from utils.logger import logger

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------
_AGG_KEYWORDS: dict[str, list[str]] = {
    "sum": ["sum", "total", "totals", "revenue", "sales", "summed", "aggregate"],
    "avg": ["average", "avg", "mean"],
    "count": ["count", "number of", "tally"],  # Removed "how many" - handled by transaction intent
    "min": ["min", "minimum", "lowest", "smallest"],
    "max": ["max", "maximum", "highest", "largest", "top"],
}

# Transaction verbs that indicate the user wants to SUM transaction counts, not COUNT documents
_TRANSACTION_VERBS = {
    "signed", "signed up", "registered", "enrolled", "applied",
    "submitted", "completed", "transacted", "booked", "purchased",
    "paid", "donated", "contributed", "renewed", "issued"
}

# Metric patterns that indicate transaction/count fields (should use SUM, not COUNT)
_TRANSACTION_METRIC_PATTERNS = [
    r"_transactions?$",
    r"_registrations?$",
    r"_applications?$",
    r"_permits?$",
    r"_requests?$",
    r"_submissions?$",
    r"_bookings?$",
    r"_payments?$",
    r"_recipients?$",
    r"_renewals?$",
]

_GROUP_KEYWORDS = ["by", "per", "group by", "grouped by", "for each", "across"]

_SEMANTIC_KEYWORDS = [
    "why", "explain", "insight", "insights",
    "anomaly", "anomalies", "summary", "summarize", "describe",
    "story", "pattern", "qualitative",
    "feedback", "comment", "comments", "review", "reviews",
]

# Single source of truth for "this question implies a temporal axis".
# Used by both ``decide()`` (to set ``time_intent``) and
# ``_pick_date_field`` (to choose between ``period`` / ``year`` / a date
# column). Substring matching is intentional — ``"year" in "years"`` is
# True, ``"month" in "monthly"`` is True, etc. — so the list does not
# need to enumerate every plural / inflection.
_TIME_INTENT_KEYWORDS = [
    "trend", "trends", "trending", "over time", "by month",
    "month", "monthly",
    "year", "yearly", "annual", "annually", "year over year", "yoy",
    "week", "weekly", "day", "daily", "quarter", "quarterly",
]

_TOP_K_RE = re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE)
_LAST_N_RE = re.compile(
    r"\blast\s+(\d+)\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _extract_all_years(q: str) -> list[int]:
    """Extract all 4-digit years (20XX) mentioned in the question.
    
    Returns years in descending order (most recent first).
    
    Examples:
        "trends for 2026 and 2025" → [2026, 2025]
        "compare 2024 vs 2023" → [2024, 2023]
        "between 2022 and 2025" → [2025, 2024, 2023, 2022]
    """
    matches = _YEAR_RE.findall(q)
    if not matches:
        return []
    years = sorted([int(y) for y in matches], reverse=True)
    return years

# Catalog-style collections that are never the right target for numeric trends.
_AWQAF_METADATA_COLLECTION = "awqaf_datasets_metadata"
_AWQAF_GLOSSARY_COLLECTION = "awqaf_datasets_glossary"
_AWQAF_NON_FACT_COLLECTIONS = frozenset(
    {_AWQAF_METADATA_COLLECTION, _AWQAF_GLOSSARY_COLLECTION}
)

# Phrases that explicitly request the catalog / metadata view.
# Matched as **whole-word phrases** (``\bphrase\b``), not substrings, so
# wording like "all available data years" — which contains the bare word
# "available" — does NOT count as catalog intent.
_CATALOG_INTENT_PHRASES: tuple[str, ...] = (
    "data catalog",
    "dataset catalog",
    "list datasets",
    "list all datasets",
    "list of datasets",
    "show catalog",
    "show the catalog",
    "show datasets",
    "show all datasets",
    "show me datasets",
    "browse catalog",
    "browse datasets",
    "what datasets",
    "available datasets",
    "all datasets",
)

_DATE_FIELD_HINTS = (
    "date", "_at", "time", "timestamp", "created", "updated", "occurred", "happened"
)


# ---------------------------------------------------------------------------
# Uncertainty flags (Build #6 — LLM router fallback)
# ---------------------------------------------------------------------------
# Names emitted in :func:`_uncertainty_flags` — kept as module-level
# constants so the LLM-router prompt and the eval harness can reference
# the same identifiers without string drift.
FLAG_NO_TARGET = "no_target"
FLAG_LOW_TARGET_OVERLAP = "low_target_overlap"
FLAG_MISSING_METRIC_FOR_OP = "missing_metric_for_op"
FLAG_GROUP_BY_INTENT_UNMET = "group_by_intent_unmet"
FLAG_OP_VIA_DEFAULT = "op_via_default"
FLAG_PURE_SEMANTIC_WITH_YEAR = "pure_semantic_with_year"

# Operations that cannot be evaluated without an explicit metric
# column. ``count`` is excluded — counting rows works without a metric.
_OPS_REQUIRING_METRIC = frozenset({"sum", "avg", "min", "max"})


def _uncertainty_flags(
    decision: RoutingDecision,
    question: str,
) -> list[str]:
    """List the *named* concerns about this rule-based decision.

    A non-empty list means the deterministic router is unsure enough
    that an LLM second opinion would likely improve the answer. The
    list is what the orchestrator (or an eval scorer) checks; the
    actual values are documented in the FLAG_* constants above.

    Pure function — takes only the decision + question and returns a
    list. No I/O, no dependencies. That keeps it cheap to compute on
    every routing call (so we always have the signal in the trace,
    even when the LLM-router fallback is disabled).
    """
    q = (question or "").lower()
    flags: list[str] = []
    spec = decision.aggregation

    # 1. Couldn't pick a collection at all — the bluntest signal.
    if decision.target is None:
        flags.append(FLAG_NO_TARGET)

    # 2. Picked a collection but with zero token overlap — basically
    # a coin-flip pick. We re-derive the overlap here (cheap) so the
    # check is independent of any internal scoring state.
    elif decision.target_candidates:
        q_tokens = set(_tokenize(question))
        chosen_tokens = set(_tokenize(decision.target.replace("_", " ")))
        if q_tokens and not (q_tokens & chosen_tokens):
            flags.append(FLAG_LOW_TARGET_OVERLAP)

    # 3. Operation needs a metric and we couldn't infer one.
    if (
        spec is not None
        and spec.operation in _OPS_REQUIRING_METRIC
        and not spec.metric
    ):
        flags.append(FLAG_MISSING_METRIC_FOR_OP)

    # 4. Question explicitly asks for a breakdown but no group_by came
    # out. This is exactly the ``route-top-gap-001`` failure mode the
    # eval harness has been carrying as a known-gap case.
    if (
        spec is not None
        and any(kw in q for kw in _GROUP_KEYWORDS)
        and not spec.group_by
    ):
        flags.append(FLAG_GROUP_BY_INTENT_UNMET)

    # 5. Operation was set via ``trend->sum`` rather than an explicit
    # aggregation keyword. The user may genuinely have wanted ``avg``,
    # ``count``, etc. — worth a second opinion.
    if "trend->sum" in decision.matched_keywords:
        flags.append(FLAG_OP_VIA_DEFAULT)

    # 6. Pure SEMANTIC route but the question mentions a year — the
    # rule path defaults to semantic when no aggregation keyword fires,
    # but a year mention often indicates the user wants a number too.
    if (
        decision.route == QueryRoute.SEMANTIC
        and _YEAR_RE.search(q)
    ):
        flags.append(FLAG_PURE_SEMANTIC_WITH_YEAR)

    return flags


@dataclass
class _SchemaSnapshot:
    columns: list[str]
    is_mongo: bool
    sample: list[dict] = field(default_factory=list)


class RoutingService:
    """Decide how to answer a question and emit an :class:`AggregationSpec`."""

    # Cache the live collection / table list so a transient Atlas hiccup on
    # one request does not silently strip the routing target on the next.
    _LISTING_TTL_SECONDS = 60

    def __init__(
        self,
        mongo: MongoService | None = None,
        mysql: MySQLService | None = None,
        knowledge_base: KnowledgeBaseService | None = None,
        refiner: Any | None = None,
    ) -> None:
        self.mongo = mongo or mongo_service
        self.mysql = mysql or mysql_service
        self.knowledge_base = knowledge_base or knowledge_base_service
        # Build #6: opt-in LLM second-opinion. When ``refiner`` is None
        # (the default), :meth:`_maybe_refine` lazily imports the
        # module-level :data:`router_refiner` singleton on first need —
        # but only when ``settings.router_llm_fallback_enabled`` is
        # True. This keeps the routing service self-contained: tests
        # can inject a mock by passing ``refiner=...``, production
        # toggles via env, and uses that don't need it never pull in
        # the Anthropic SDK.
        self.refiner = refiner
        self._mongo_cols_cache: tuple[float, list[str]] | None = None
        self._mysql_tables_cache: tuple[float, list[str]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(
        self,
        question: str,
        *,
        collection: str | None = None,
        data_source: DataSource = DataSource.AUTO,
    ) -> RoutingDecision:
        q = (question or "").lower().strip()
        matched: list[str] = []

        op = self._detect_operation(q, matched)
        is_semantic = any(kw in q for kw in _SEMANTIC_KEYWORDS)
        if is_semantic:
            matched.append("semantic")

        time_intent = (
            any(kw in q for kw in _TIME_INTENT_KEYWORDS)
            or _LAST_N_RE.search(q)
            or bool(_YEAR_RE.search(q))
        )
        if time_intent and op is None:
            op = "sum"
            matched.append("trend->sum")

        # Resolve target up-front so glossary scoping can use it.
        chosen_source, target, candidates, schema, resolution = self._resolve_target(
            data_source, collection, q
        )
        
        # CRITICAL FIX: Transaction intent detection
        # When user asks "how many signed/registered/applied", they want to SUM
        # the transaction count field, NOT count documents.
        # This overrides the "count" operation detected from "how many".
        transaction_intent = self._detect_transaction_intent(q, schema.columns)
        if transaction_intent:
            if op == "count":
                logger.info(
                    f"Transaction intent detected (verb: '{transaction_intent['verb']}', "
                    f"metric: '{transaction_intent['metric']}') — overriding COUNT with SUM"
                )
                op = "sum"
                matched.append(f"transaction-intent:{transaction_intent['verb']}")
            elif op is None:
                # "how many" without explicit count keyword
                logger.info(
                    f"Transaction intent detected (verb: '{transaction_intent['verb']}', "
                    f"metric: '{transaction_intent['metric']}') — setting operation to SUM"
                )
                op = "sum"
                matched.append(f"transaction-intent:{transaction_intent['verb']}")

        # Glossary lookup — surface even on semantic-only paths so the trust
        # panel can mention an applicable definition.
        glossary_match = self._lookup_glossary(question, target)
        if glossary_match is not None:
            matched.append(_glossary_keyword(glossary_match))

            # An approved formula with an operation can promote a question
            # that had no aggregation keyword into the analytical path.
            if (
                op is None
                and glossary_match.definition.status == MetricStatus.APPROVED
                and glossary_match.definition.formula.operation is not None
            ):
                op = glossary_match.definition.formula.operation
                matched.append(f"glossary-op:{op}")

        if op is None:
            # ✅ NEW: Try semantic intent enrichment before defaulting to SEMANTIC route
            # This catches vague analytical queries like "what should i look into hajj package"
            # and either infers the operation or suggests specific questions.
            try:
                from services.semantic_intent_enricher import semantic_intent_enricher
                from models.config import settings
                
                enriched = semantic_intent_enricher.enrich(question, target)
                
                # If confidence is high enough, use the inferred operation
                enrichment_threshold = getattr(settings, 'intent_enrichment_threshold', 0.5)
                if enriched.confidence >= enrichment_threshold and enriched.inferred_operation:
                    op = enriched.inferred_operation
                    matched.append(f"semantic-enriched:{op}")
                    logger.info(
                        f"Semantic enrichment: inferred operation '{op}' "
                        f"(confidence: {enriched.confidence:.2f}) - {enriched.reasoning}"
                    )
                    # Continue to build aggregation spec below
                else:
                    # Low confidence - return semantic decision with enrichment metadata
                    # The analyst service will check this and potentially show suggestions
                    semantic_decision = RoutingDecision(
                        route=QueryRoute.SEMANTIC,
                        data_source=DataSource.AUTO,
                        target=target,
                        target_candidates=candidates,
                        resolution=resolution,
                        reason=(
                            f"No aggregation keywords detected. "
                            f"Semantic enrichment confidence too low ({enriched.confidence:.2f} < {enrichment_threshold:.2f}). "
                            f"{enriched.reasoning}"
                        ),
                        matched_keywords=matched,
                        definition=glossary_match,
                    )
                    return self._maybe_refine(semantic_decision, question, schema)
                    
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Semantic intent enrichment failed ({type(exc).__name__}: {exc}); "
                    f"falling back to semantic route"
                )
                semantic_decision = RoutingDecision(
                    route=QueryRoute.SEMANTIC,
                    data_source=DataSource.AUTO,
                    target=target,
                    target_candidates=candidates,
                    resolution=resolution,
                    reason=(
                        "No aggregation keywords; pure semantic query."
                        if is_semantic
                        else "No aggregation keywords detected; defaulting to semantic search."
                    ),
                    matched_keywords=matched,
                    definition=glossary_match,
                )
                return self._maybe_refine(semantic_decision, question, schema)

        spec = self._build_aggregation_spec(q, op, schema, transaction_intent)
        spec.limit = spec.limit or self._extract_top_k(q)

        if time_intent and schema.columns:
            time_spec = self._build_time_spec(q, schema)
            if time_spec is not None:
                spec.time = time_spec
                matched.append(f"time:{time_spec.bucket.value}")

        # Approved glossary entries change query construction; draft entries
        # are surfaced only and do NOT modify the spec.
        if (
            glossary_match is not None
            and glossary_match.definition.status == MetricStatus.APPROVED
        ):
            _merge_formula_into_spec(spec, glossary_match.definition.formula)
            glossary_match.applied_to_query = True
            matched.append("glossary-applied")

        route = QueryRoute.HYBRID if is_semantic else QueryRoute.ANALYTICAL
        reason_bits = [
            f"Detected aggregation '{op}' on {chosen_source.value}"
            + (f" / {target}" if target else ""),
            "with time axis." if spec.time else ".",
        ]
        if is_semantic:
            reason_bits.append("Also semantic context requested.")
        if glossary_match is not None:
            applied = "applied" if glossary_match.applied_to_query else "surfaced (unverified)"
            reason_bits.append(
                f"Glossary definition '{glossary_match.definition.term}' {applied}."
            )

        logger.debug(f"Routing decision: {route} ({chosen_source}/{target}) - {spec}")
        analytical_decision = RoutingDecision(
            route=route,
            data_source=chosen_source,
            aggregation=spec,
            target=target,
            target_candidates=candidates,
            resolution=resolution,
            reason=" ".join(reason_bits),
            matched_keywords=matched,
            definition=glossary_match,
        )
        return self._maybe_refine(analytical_decision, question, schema)

    # ------------------------------------------------------------------
    # LLM fallback (Build #6)
    # ------------------------------------------------------------------
    def _maybe_refine(
        self,
        decision: RoutingDecision,
        question: str,
        schema: "_SchemaSnapshot",
    ) -> RoutingDecision:
        """Consult the LLM only when the rule decision looks shaky.

        Gating order matters here:

        1. Compute uncertainty flags (always cheap, always traced).
        2. If no flags, return the rule decision untouched (~80% of
           queries; the LLM is never invoked).
        3. If flags AND ``settings.router_llm_fallback_enabled``, send
           one Claude call with the rule decision as the prior + the
           schema columns as the allowed set. Apply only validated
           patches.
        4. If flags but feature is OFF, return the rule decision
           anyway — the flags themselves are still informative for
           anyone reading the trace / logs.
        """
        flags = _uncertainty_flags(decision, question)
        if not flags:
            return decision
        # Lazy import + lazy resolve of the module-level singleton so
        # we don't pull in the Anthropic SDK when the feature is off.
        from models.config import settings  # local to avoid import cycles

        if not settings.router_llm_fallback_enabled:
            logger.debug(
                f"Routing uncertainty flags fired but LLM fallback "
                f"disabled: {flags}"
            )
            return decision
        refiner = self.refiner
        if refiner is None:
            from services.router_tool import router_refiner

            refiner = router_refiner
        new_decision, refinement = refiner.refine(
            question=question,
            decision=decision,
            target_columns=list(schema.columns),
            flags=flags,
        )
        if refinement.applied:
            logger.info(
                f"LLM refined routing: changed={refinement.fields_changed} "
                f"rejected={refinement.fields_rejected} "
                f"flags={flags}"
            )
        elif refinement.fallback:
            logger.warning(
                f"LLM router fallback degraded to rule decision: "
                f"{refinement.reasoning}"
            )
        return new_decision

    # ------------------------------------------------------------------
    # Glossary lookup
    # ------------------------------------------------------------------
    def _lookup_glossary(
        self, question: str, target: str | None
    ) -> GlossaryMatch | None:
        try:
            return self.knowledge_base.lookup(question, target=target)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Glossary lookup failed (ignoring): {exc}")
            return None

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_operation(q: str, matched: list[str]) -> str | None:
        for op, words in _AGG_KEYWORDS.items():
            for w in words:
                if re.search(rf"\b{re.escape(w)}\b", q):
                    matched.append(w)
                    return op
        return None
    
    @staticmethod
    def _detect_transaction_intent(q: str, columns: list[str]) -> dict[str, str] | None:
        """Detect if query is asking for transaction counts (should use SUM, not COUNT).
        
        Returns:
            dict with 'verb' and 'metric' if transaction intent detected, None otherwise
            
        Examples:
            "how many signed through the website" → {"verb": "signed", "metric": "website_transactions"}
            "how many registered for hajj" → {"verb": "registered", "metric": "total_transactions"}
        """
        # Check for transaction verbs
        matched_verb = None
        for verb in _TRANSACTION_VERBS:
            if verb in q:
                matched_verb = verb
                break
        
        if not matched_verb:
            return None
        
        # Find transaction-related metrics in the schema
        transaction_metrics = []
        for col in columns:
            col_lower = col.lower()
            # Check if column matches transaction patterns
            for pattern in _TRANSACTION_METRIC_PATTERNS:
                if re.search(pattern, col_lower):
                    transaction_metrics.append(col)
                    break
        
        if not transaction_metrics:
            return None
        
        # Try to match specific channel/type mentioned in query
        # e.g., "website" → "website_transactions"
        for metric in transaction_metrics:
            metric_lower = metric.lower()
            # Extract channel/type from metric name (e.g., "website" from "website_transactions")
            metric_parts = metric_lower.replace("_", " ").split()
            for part in metric_parts:
                if len(part) > 3 and part in q:  # Avoid matching short words like "app"
                    return {"verb": matched_verb, "metric": metric}
        
        # Fallback: return the first transaction metric found
        return {"verb": matched_verb, "metric": transaction_metrics[0]}

    @staticmethod
    def _extract_top_k(q: str) -> int | None:
        m = _TOP_K_RE.search(q)
        return int(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # Schema + target resolution
    # ------------------------------------------------------------------
    def _resolve_target(
        self,
        data_source: DataSource,
        collection: str | None,
        question: str,
    ) -> tuple[DataSource, str | None, list[str], _SchemaSnapshot, TargetResolution | None]:
        """Pick a data source + concrete target + snapshot the schema."""
        if data_source == DataSource.MYSQL:
            tables = self._list_mysql_tables()
            chosen, resolution = self._pick_target(question, tables, collection)
            return (
                DataSource.MYSQL,
                chosen,
                [t for t in tables if t != chosen],
                self._mysql_schema(chosen),
                resolution,
            )

        if data_source == DataSource.MONGO:
            cols = self._list_mongo_collections()
            chosen, resolution = self._pick_target(question, cols, collection)
            return (
                DataSource.MONGO,
                chosen,
                [c for c in cols if c != chosen],
                self._mongo_schema(chosen),
                resolution,
            )

        cols = self._list_mongo_collections()
        if cols:
            chosen, resolution = self._pick_target(question, cols, collection)
            return (
                DataSource.MONGO,
                chosen,
                [c for c in cols if c != chosen],
                self._mongo_schema(chosen),
                resolution,
            )

        tables = self._list_mysql_tables()
        if tables:
            chosen, resolution = self._pick_target(question, tables, collection)
            return (
                DataSource.MYSQL,
                chosen,
                [t for t in tables if t != chosen],
                self._mysql_schema(chosen),
                resolution,
            )

        logger.warning(
            "Routing could not find any Mongo collection or MySQL table — "
            "target will be None. Check the database connection."
        )
        return DataSource.MONGO, None, [], _SchemaSnapshot(columns=[], is_mongo=True), None

    def _pick_target(
        self,
        question: str,
        candidates: list[str],
        explicit: str | None,
    ) -> tuple[str | None, TargetResolution | None]:
        """Choose one target from ``candidates`` with catalog-first logic."""
        if not candidates:
            return None, None

        if explicit:
            return explicit, TargetResolution(
                method="explicit",
                score=1.0,
                top_scores={explicit: 1.0},
            )

        # Catalog-first (AWQAF facts collections only).
        if settings.catalog_routing_enabled:
            cat = catalog_routing_service.score_question(
                question, available_collections=candidates
            )
            if cat.method == "catalog" and cat.facts_collection:
                logger.info(
                    f"Catalog routing: {cat.facts_collection} "
                    f"(slug={cat.slug}, score={cat.score:.1f})"
                )
                return cat.facts_collection, TargetResolution(
                    method="catalog",
                    score=cat.score,
                    catalog_slug=cat.slug,
                    top_scores=cat.top_scores,
                )

        # ✅ NEW: Use semantic matching for better accuracy
        try:
            collection, confidence, reasoning = semantic_matcher.find_best_collection(
                question, candidates
            )
            
            if collection and confidence > 0.5:
                logger.info(f"Semantic collection match: {reasoning}")
                return collection, TargetResolution(
                    method="semantic",
                    score=confidence,
                    top_scores={collection: confidence}
                )
        except Exception as e:
            logger.warning(f"Semantic matching failed, falling back to token overlap: {e}")

        # Fallback to token overlap
        chosen, overlap_scores = _score_target_with_scores(question, candidates)
        best_overlap = float(overlap_scores.get(chosen or "", 0)) if chosen else 0.0
        method = "token_overlap" if best_overlap > 0 else "default"
        return chosen, TargetResolution(
            method=method,
            score=best_overlap,
            top_scores={
                k: float(v) for k, v in sorted(
                    overlap_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]
            },
        )

    def _mongo_schema(self, collection: str | None) -> _SchemaSnapshot:
        if not collection:
            return _SchemaSnapshot(columns=[], is_mongo=True)
        try:
            sample = self.mongo.find(collection, limit=20)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Mongo schema probe on '{collection}' failed "
                f"({type(exc).__name__}: {exc}); routing falls back to empty schema."
            )
            return _SchemaSnapshot(columns=[], is_mongo=True)
        cols = sorted({k for doc in sample for k in doc.keys()})
        return _SchemaSnapshot(columns=cols, is_mongo=True, sample=sample)

    def _mysql_schema(self, table: str | None) -> _SchemaSnapshot:
        if not table:
            return _SchemaSnapshot(columns=[], is_mongo=False)
        try:
            rows = self.mysql.run_sql(f"SELECT * FROM `{table}`", limit=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"MySQL schema probe on '{table}' failed "
                f"({type(exc).__name__}: {exc}); routing falls back to empty schema."
            )
            return _SchemaSnapshot(columns=[], is_mongo=False)
        cols = list(rows[0].keys()) if rows else []
        return _SchemaSnapshot(columns=cols, is_mongo=False, sample=rows)

    # ------------------------------------------------------------------
    # Cached database discovery (transient-error tolerant)
    # ------------------------------------------------------------------
    def _list_mongo_collections(self) -> list[str]:
        """Return Mongo collection names with a short TTL cache.

        On a transient error we fall back to the last successful list rather
        than collapsing to ``[]`` (which would make the router pick no target
        and force every downstream call to act as if the DB were empty).
        """
        return self._cached_listing(
            label="mongo.list_collections",
            fetch=self.mongo.list_collections,
            slot="_mongo_cols_cache",
        )

    def _list_mysql_tables(self) -> list[str]:
        return self._cached_listing(
            label="mysql.list_tables",
            fetch=self.mysql.list_tables,
            slot="_mysql_tables_cache",
        )

    def _cached_listing(self, *, label: str, fetch, slot: str) -> list[str]:
        """Fetch collection/table listing with 3-tier caching.
        
        Cache hierarchy:
        1. In-memory cache (60s TTL) - fastest
        2. Redis cache (300s TTL) - shared across instances
        3. Database query - slowest
        """
        now = time.monotonic()
        
        # Check in-memory cache first (existing logic)
        cached: tuple[float, list[str]] | None = getattr(self, slot)
        if cached is not None and (now - cached[0]) < self._LISTING_TTL_SECONDS:
            return list(cached[1])
        
        # Check Redis cache (NEW - only for MongoDB collections)
        if label == "mongo.list_collections":
            from services.redis_metadata_cache import redis_metadata_cache
            
            redis_cached = redis_metadata_cache.get_collections()
            if redis_cached is not None:
                # Promote to in-memory cache
                setattr(self, slot, (now, redis_cached))
                logger.debug(f"{label}: Redis cache hit ({len(redis_cached)} collections)")
                return list(redis_cached)
        
        # Cache miss - query database
        try:
            value = list(fetch() or [])
        except Exception as exc:  # noqa: BLE001
            if cached is not None:
                logger.warning(
                    f"{label} failed ({type(exc).__name__}: {exc}); "
                    f"using cached list of {len(cached[1])} entries from "
                    f"{int(now - cached[0])}s ago."
                )
                return list(cached[1])
            logger.error(
                f"{label} failed and no prior list is cached "
                f"({type(exc).__name__}: {exc}); routing will have no target."
            )
            return []
        
        # Store in both caches
        setattr(self, slot, (now, value))
        
        if label == "mongo.list_collections":
            from services.redis_metadata_cache import redis_metadata_cache
            redis_metadata_cache.set_collections(value)
        
        return value

    # ------------------------------------------------------------------
    # Aggregation spec inference
    # ------------------------------------------------------------------
    def _build_aggregation_spec(
        self,
        q: str,
        op: str,
        schema: _SchemaSnapshot,
        transaction_intent: dict[str, str] | None = None,
    ) -> AggregationSpec:
        metric = None
        group_by = None
        if schema.columns:
            # If transaction intent detected, use that metric directly
            if transaction_intent and transaction_intent.get("metric"):
                metric = transaction_intent["metric"]
                logger.info(
                    f"Using transaction metric '{metric}' from intent detection "
                    f"(verb: '{transaction_intent.get('verb')}')"
                )
            else:
                metric = self._guess_metric(q, schema.columns, op)
            group_by = self._guess_group_by(q, schema.columns)


        # intelligent fallback grouping
        # when user asks for trend but
        # dataset only has sparse time density
        if (
            "trend" in q.lower()
            and group_by is None
        ):
            auto_group = self._auto_dimension_field(
                schema.columns
            )

            if auto_group:
                logger.info(
                    f"Auto-selected fallback dimension "
                    f"`{auto_group}` for sparse trend analysis."
                )

                group_by = auto_group


        semantic = metric_semantic(metric)

        # intelligent aggregation override
        if semantic and op == "sum":
            op = semantic.default_aggregation

        return AggregationSpec(operation=op, metric=metric, group_by=group_by)


    def _auto_dimension_field(
    self,
    columns: list[str],
    ) -> str | None:

        preferred = [
            "dimension",
            "emirate",
            "category",
            "region",
            "department",
            "city",
            "type",
        ]

        lower_map = {
            c.lower(): c
            for c in columns
        }

        for field in preferred:
            if field in lower_map:
                return lower_map[field]

        return None

    @staticmethod
    def _guess_metric(q: str, columns: list[str], op: str) -> str | None:
        if op == "count":
            return None

        if "transaction" in q and "total_transactions" in columns:
            return "total_transactions"

        if "trend" in q and "total_transactions" in columns:
            return "total_transactions"

        numeric_hints = {
            "amount", "price", "total", "revenue", "sales", "cost",
            "value", "quantity", "qty", "score", "profit", "spend",
        }

        # Match column names tolerantly: ``smart_app_transactions`` should
        # also match "Smart App Transactions" or "smart-app-transactions" the
        # user typed (i.e. the same words with spaces or hyphens instead of
        # underscores). Without this, copy-paste of the suggested follow-up
        # questions failed even though the user named the column exactly.
        for col in columns:
            pattern = re.escape(col).replace("_", r"[\s_\-]")
            if re.search(rf"\b{pattern}\b", q):
                return col

        for col in columns:
            for hint in numeric_hints:
                if hint in col:
                    return col

        # We deliberately do NOT silently substitute a similar column here.
        # Returning ``None`` lets ``analyst_service._compose_insight`` honestly
        # tell the user "this metric is not in the schema; here is what IS
        # available" — never produce a confident chart against a different
        # field than the one asked for.
        if "transaction" in q and any(c.endswith("_transactions") for c in columns):
            logger.info(
                "Question mentions transactions but collection has no "
                "`total_transactions` column; deferring to analyst layer to "
                "surface available channel fields."
            )

        return None

    @staticmethod
    def _guess_group_by(q: str, columns: list[str]) -> str | None:
        # "top 5 emirates by total revenue" — dimension precedes "by".
        before_by = extract_group_by_before_by(q, columns)
        if before_by:
            return before_by

        if not any(kw in q for kw in _GROUP_KEYWORDS):
            return None

        for kw in (" by ", " per ", " for each "):
            idx = q.find(kw)
            if idx == -1:
                continue
            tail = q[idx + len(kw) :].strip()
            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", tail)
            for token in tokens[:3]:
                if token in columns:
                    return token

        for col in columns:
            if col in q and any(
                hint in col
                for hint in ("region", "category", "department", "type", "city",
                             "country", "month", "year", "product")
            ):
                return col
        return None

    # ------------------------------------------------------------------
    # Time intent
    # ------------------------------------------------------------------
    @staticmethod
    def _build_time_spec(q: str, schema: _SchemaSnapshot) -> TimeSpec | None:
        time_field = _pick_date_field(schema, q)
        if not time_field:
            return None

        bucket = _pick_bucket(q)
        if time_field == "year" and bucket == TimeBucket.MONTH:
            if not any(
                kw in q
                for kw in (
                    "trend",
                    "month",
                    "monthly",
                    "week",
                    "weekly",
                    "day",
                    "daily",
                    "quarter",
                    "quarterly",
                )
            ):
                bucket = TimeBucket.YEAR
        range_from, range_to, years = _pick_range(q)
        compare = ComparisonMode.NONE
        
        # Set comparison mode for multi-year queries
        if years and len(years) >= 2:
            compare = ComparisonMode.YOY
        elif (
            "year over year" in q
            or "yoy" in q
            or "vs last year" in q
            or "compared to last year" in q
        ):
            compare = ComparisonMode.YOY
        elif "vs last" in q or "compared to last" in q or "previous period" in q:
            compare = ComparisonMode.PREV_PERIOD

        return TimeSpec(
            field=time_field,
            bucket=bucket,
            range_from=range_from,
            range_to=range_to,
            compare=compare,
            years=years,
        )


# ---------------------------------------------------------------------------
# Glossary helpers
# ---------------------------------------------------------------------------
def _glossary_keyword(match: GlossaryMatch) -> str:
    status = match.definition.status.value
    return f"glossary[{status}]:{match.definition.id}"


def _merge_formula_into_spec(spec: AggregationSpec, formula: MetricFormula) -> None:
    """Merge an approved glossary formula into the analytical spec.

    Rules (v1):
    * ``operation`` from the formula wins (so a glossary that pins ``count``
      cannot be subverted by a vague question).
    * ``metric`` from the formula wins.
    * ``group_by`` only fills if the question didn't already specify one.
    * ``filters`` are merged with the formula winning on key collision —
      glossary filters define eligibility for the metric.
    """
    if formula.operation:
        spec.operation = formula.operation
    if formula.metric:
        spec.metric = formula.metric
    if formula.group_by and not spec.group_by:
        spec.group_by = formula.group_by
    if formula.filters:
        merged = dict(spec.filters or {})
        merged.update(formula.filters)
        spec.filters = merged


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "by", "with", "and", "or",
    "show", "give", "tell", "me", "us", "what", "is", "are", "this", "that",
    "from", "over", "all", "my", "our", "your", "please", "report",
}


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
        if t not in _STOPWORDS
    ]


def _question_targets_service_metrics(q: str) -> bool:
    """True when the user is asking for numbers/trends, not catalog browsing."""
    ql = q.lower()
    # Whole-word phrase match against the canonical catalog intents.
    # Earlier versions used a bare ``"catalog" in ql`` substring check
    # which mis-classified phrasing like "catalog of services". Matching
    # against the same phrase list as `_score_target` keeps the two in
    # sync — a question is a catalog browse only if it explicitly says so.
    if any(
        re.search(rf"\b{re.escape(p)}\b", ql) for p in _CATALOG_INTENT_PHRASES
    ):
        return False
    if _YEAR_RE.search(ql):
        return True
    if "how many" in ql:
        return True
    for kw in (
        "trend",
        "trends",
        "transaction",
        "transactions",
        "monthly",
        "statistics",
        "average",
        "avg",
        "mean",
        "count",
    ):
        if re.search(rf"\b{re.escape(kw)}\b", ql):
            return True
    if re.search(r"\btotal\b", ql) or re.search(r"\bsum\b", ql):
        return True
    return False


def _score_target(question: str, candidates: list[str]) -> str | None:
    """Pick the candidate whose name best matches the question tokens."""
    chosen, _ = _score_target_with_scores(question, candidates)
    return chosen


def _score_target_with_scores(
    question: str, candidates: list[str]
) -> tuple[str | None, dict[str, int]]:
    """Return best target and per-candidate token-overlap scores."""
    if not candidates:
        return None, {}
    if len(candidates) == 1:
        return candidates[0], {candidates[0]: 0}

    q_tokens = set(_tokenize(question))
    active = list(candidates)
    scores: dict[str, int] = {}

    if _question_targets_service_metrics(question):
        trimmed = [c for c in active if c not in _AWQAF_NON_FACT_COLLECTIONS]
        if trimmed:
            active = trimmed
    else:
        ql = question.lower()
        looks_catalog = any(
            re.search(rf"\b{re.escape(p)}\b", ql)
            for p in _CATALOG_INTENT_PHRASES
        )
        if looks_catalog and _AWQAF_METADATA_COLLECTION in active:
            facts = [
                c for c in active if c not in _AWQAF_NON_FACT_COLLECTIONS
            ]
            facts_overlap = any(
                len(q_tokens & set(_tokenize(c.replace("_", " ")))) > 0
                for c in facts
            )
            if not facts_overlap:
                return _AWQAF_METADATA_COLLECTION, {_AWQAF_METADATA_COLLECTION: 0}

    if not q_tokens:
        return active[0], {active[0]: 0}

    best, best_score = active[0], -1
    for name in active:
        name_tokens = set(_tokenize(name.replace("_", " ")))
        score = len(q_tokens & name_tokens)
        scores[name] = score
        if score > best_score:
            best_score, best = score, name
    return best, scores


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _pick_date_field(schema: _SchemaSnapshot, q: str = "") -> str | None:
    """Pick a column to drive time filtering / bucketing.

    Order of preference:

    1. The denormalized ``period`` (``YYYY-MM``) on AWQAF facts collections —
       direct grouping with no string parsing.
    2. The integer ``year`` column when the question names a calendar year.
    3. A column that looks like a date (created_at, etc.).
    4. A sample value that looks like an ISO date.

    Leading-underscore fields (``_ingested_at``) are skipped so date hints do
    not latch onto provenance columns.
    """
    has_period = "period" in schema.columns
    has_year = "year" in schema.columns

    # Substring match against the canonical ``_TIME_INTENT_KEYWORDS`` list
    # so this layer cannot drift out of sync with ``decide()``. Substring
    # also gracefully handles plurals — "year" matches "years", "month"
    # matches "months", etc. — without enumerating every inflection.
    has_temporal_token = bool(_YEAR_RE.search(q)) or any(
        kw in q for kw in _TIME_INTENT_KEYWORDS
    )

    if has_period and has_temporal_token:
        return "period"

    if has_year and has_temporal_token:
        return "year"

    # Fallback: pick a column whose NAME hints at a date AND whose sample
    # is actually a real ``datetime`` instance. Strings on date-named
    # columns (e.g. ``ingested_at`` stored as an ISO string) are rejected
    # here because Mongo's ``$dateToString`` only accepts BSON Dates;
    # picking such a field would deterministically crash the pipeline.
    for col in schema.columns:
        if col.startswith("_"):
            continue
        col_l = col.lower()
        if any(hint in col_l for hint in _DATE_FIELD_HINTS) and _column_has_datetime_sample(col, schema.sample):
            return col

    for row in schema.sample[:5]:
        for col, value in row.items():
            if col.startswith("_"):
                continue
            if _looks_like_date(value):
                return col
    return None


def _column_has_datetime_sample(col: str, sample: list[dict]) -> bool:
    """True iff at least one sample doc carries a real ``datetime`` for ``col``."""
    for doc in sample:
        if isinstance(doc.get(col), datetime):
            return True
    return False


def _looks_like_date(value) -> bool:
    """True only for real ``datetime`` instances.

    Earlier versions also accepted ISO-looking strings (``"2024-01-15"``,
    ISO datetimes), but Mongo's ``$dateToString`` rejects strings with
    error code ``Location4997901``. Picking such a field as the time
    axis was therefore a guaranteed crash. The narrow ``period`` /
    ``year`` paths above already handle the AWQAF string/integer cases
    by name; this fallback is reserved for collections that store an
    actual BSON Date.
    """
    return isinstance(value, datetime)


def _pick_bucket(q: str) -> TimeBucket:
    if "daily" in q or "day" in q:
        return TimeBucket.DAY
    if "weekly" in q or "week" in q:
        return TimeBucket.WEEK
    if "quarterly" in q or "quarter" in q:
        return TimeBucket.QUARTER
    if (
        "yearly" in q or "annually" in q or "annual" in q
        or "year over year" in q or "yoy" in q
    ):
        return TimeBucket.YEAR
    return TimeBucket.MONTH


def _pick_range(q: str) -> tuple[datetime | None, datetime | None, list[int] | None]:
    """Extract time range from question.
    
    Returns:
        (range_from, range_to, years) where years is populated for multi-year queries
    """
    now = datetime.now(timezone.utc)
    m = _LAST_N_RE.search(q)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = _delta_for_unit(n, unit)
        return now - delta, now, None

    if "last year" in q:
        year = now.year - 1
        return (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            None,
        )

    if "this year" in q:
        return datetime(now.year, 1, 1, tzinfo=timezone.utc), now, None

    # NEW: Check for multiple years
    all_years = _extract_all_years(q)
    
    # Multi-year comparison detected
    if len(all_years) >= 2:
        # Check for comparison keywords
        q_lower = q.lower()
        has_compare = any(kw in q_lower for kw in [
            "compare", "comparison", "vs", "versus", "and", "between"
        ])
        if has_compare:
            # Return None for range_from/range_to to signal multi-year mode
            return None, None, all_years
    
    # Single year (existing logic)
    if all_years:
        y = all_years[0]
        return (
            datetime(y, 1, 1, tzinfo=timezone.utc),
            datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            None,
        )

    return None, None, None


def _delta_for_unit(n: int, unit: str) -> timedelta:
    if unit.startswith("day"):
        return timedelta(days=n)
    if unit.startswith("week"):
        return timedelta(weeks=n)
    if unit.startswith("month"):
        return timedelta(days=30 * n)
    if unit.startswith("quarter"):
        return timedelta(days=90 * n)
    if unit.startswith("year"):
        return timedelta(days=365 * n)
    return timedelta(days=n)


routing_service = RoutingService()
