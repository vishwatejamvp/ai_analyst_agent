"""Enumerations used by routing, charts, and data source selection."""

from __future__ import annotations

from enum import Enum


class QueryRoute(str, Enum):
    """How an incoming user query should be answered."""

    ANALYTICAL = "analytical"   # Pure DB aggregation (sum/avg/group by/...)
    SEMANTIC = "semantic"       # Vector search only (insights, trends, "why")
    HYBRID = "hybrid"           # Both DB + vector search
    DISCOVERY = "discovery"     # Catalog summary + executable lead suggestions
    OUT_OF_SCOPE = "out_of_scope"  # Polite redirect; no data work performed


class DataSource(str, Enum):
    """Underlying data store to consult for analytical queries."""

    MONGO = "mongo"
    MYSQL = "mysql"
    AUTO = "auto"


class ChartType(str, Enum):
    """Supported chart types."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    KPI = "kpi"
    NONE = "none"


class TimeBucket(str, Enum):
    """Calendar bucket used when an analytical query has a time dimension."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class ComparisonMode(str, Enum):
    """Period-over-period comparison style for trend queries."""

    NONE = "none"
    PREV_PERIOD = "prev_period"   # equal-length window immediately prior
    YOY = "yoy"                    # same window, one year earlier


class MetricStatus(str, Enum):
    """Lifecycle status of a curated metric definition in the glossary."""

    APPROVED = "approved"     # canonical, may change query construction
    DRAFT = "draft"           # surfaced as 'unverified', does NOT change queries
    DEPRECATED = "deprecated" # excluded from matching


class DefinitionSource(str, Enum):
    """Where the metric definition used for an answer came from."""

    NONE = "none"
    NAIVE = "naive"               # routing's default heuristic, no glossary
    GLOSSARY = "glossary"         # curated, approved entry
    USER_CLARIFIED = "user_clarified"
    UNVERIFIED_DOC = "unverified_doc"  # informal doc surfaced, NOT applied to query


class WarningCode(str, Enum):
    """Typed, machine-stable warning codes surfaced in the trust panel."""

    STALE_DATA = "stale_data"
    PARTIAL_PERIOD = "partial_period"
    SPARSE_DATA = "sparse_data"
    INSUFFICIENT_POINTS = "insufficient_points"
    HIGH_MISSINGNESS = "high_missingness"
    ALL_ZERO_OR_FLAT = "all_zero_or_flat"
    TOP_N_TRUNCATED = "top_n_truncated"
    SERIES_CAP_APPLIED = "series_cap_applied"
    CHART_TYPE_REDIRECTED = "chart_type_redirected"
    DEFINITION_UNVERIFIED = "definition_unverified"
    TARGET_AMBIGUOUS = "target_ambiguous"
    EMPTY_RESULT = "empty_result"
    METRIC_EMPTY = "metric_empty"  # Field exists in schema but has no values for the requested scope.
