"""Pydantic schemas for requests, responses, and internal contracts.

These models are the public contract of the API as well as the
typed payloads that flow between services. Keeping them in one place
keeps services framework-agnostic and easy to reason about.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from models.enums import (
    ChartType,
    ComparisonMode,
    DataSource,
    DefinitionSource,
    MetricStatus,
    QueryRoute,
    TimeBucket,
    WarningCode,
)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestResponse(BaseModel):
    collection: str
    rows_ingested: int
    vectors_indexed: int
    sample: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Natural language question.")
    collection: str | None = Field(
        default=None,
        description="Optional MongoDB collection / MySQL table to scope the query.",
    )
    data_source: DataSource = Field(default=DataSource.AUTO)
    top_k: int | None = Field(default=None, ge=1, le=50)
    chart_type: ChartType | None = Field(default=None)
    session_id: str | None = Field(
        default=None,
        description=(
            "Opaque session id. When provided, short follow-up questions "
            "('by region', 'exclude refunds', 'vs last year') are merged with "
            "the previous question's plan instead of starting fresh."
        ),
    )
    include_details: bool = Field(
        default=False,
        description="When true, include provenance (generated pipeline / SQL, "
        "tables touched, query fingerprint) in the response. Off by default "
        "to keep the trust panel concise for business users.",
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class TimeSpec(BaseModel):
    """Time dimension for an analytical query.

    Populated by the routing layer when a question implies a temporal axis
    ("trend", "monthly", "last 6 months", "compare to last year"). The DB
    layer is responsible for honest bucketing — no interpolation of gaps.
    """

    field: str = Field(..., description="Date/datetime field on the target entity.")
    bucket: TimeBucket = Field(default=TimeBucket.MONTH)
    range_from: datetime | None = Field(
        default=None,
        description="Inclusive lower bound; None means 'unbounded / use available history'.",
    )
    range_to: datetime | None = Field(
        default=None,
        description="Inclusive upper bound; None means 'as of latest data'.",
    )
    compare: ComparisonMode = Field(default=ComparisonMode.NONE)
    years: list[int] | None = Field(
        default=None,
        description="Multiple years for multi-year comparison queries (e.g., '2026 and 2025').",
    )


class AggregationSpec(BaseModel):
    """A normalized description of an analytical operation."""

    operation: str = Field(..., description="sum | avg | count | min | max")
    metric: str | None = Field(default=None, description="Field/column being aggregated.")
    group_by: str | None = Field(default=None, description="Field/column to group by.")
    filters: dict[str, Any] = Field(default_factory=dict)
    order_by: str | None = None
    limit: int | None = None
    time: TimeSpec | None = Field(
        default=None,
        description="Optional temporal axis. When set, the executor produces "
        "bucketed series suitable for trend/comparison analysis.",
    )


class SQLSpec(BaseModel):
    """A vetted, parameter-free read-only SQL query for MySQL."""

    sql: str
    description: str | None = None


# ---------------------------------------------------------------------------
# Knowledge base / glossary
# ---------------------------------------------------------------------------
class MetricFormula(BaseModel):
    """Structured formula spec, applied as an :class:`AggregationSpec` template.

    Kept deliberately small for v1 — every field is optional so a glossary
    entry can pin only what it cares about (e.g. "Active Customer = filter
    by status='active'", letting the natural-language question still pick
    the metric / group_by).
    """

    operation: Literal["sum", "avg", "count", "min", "max"] | None = None
    metric: str | None = None
    group_by: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class MetricDefinition(BaseModel):
    """A curated metric / business term in the glossary."""

    id: str = Field(..., description="Stable internal id, e.g. 'active_customer'.")
    term: str = Field(..., description="Canonical display name, e.g. 'Active customer'.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other phrasings users might say.",
    )
    description: str | None = None
    formula: MetricFormula = Field(default_factory=MetricFormula)
    applies_to_targets: list[str] = Field(
        default_factory=list,
        description="Collections/tables this definition is valid on. Empty = any.",
    )
    status: MetricStatus = Field(default=MetricStatus.DRAFT)
    owner: str | None = None
    updated_at: datetime | None = None


class GlossaryMatch(BaseModel):
    """A glossary lookup result attached to a routing decision."""

    definition: MetricDefinition
    matched_alias: str
    confidence: float = Field(ge=0.0, le=1.0)
    applied_to_query: bool = Field(
        default=False,
        description="True if the formula was used to override query construction.",
    )


class RoutingDecision(BaseModel):
    route: QueryRoute
    data_source: DataSource
    aggregation: AggregationSpec | None = None
    sql: SQLSpec | None = None
    target: str | None = Field(
        default=None,
        description="Concrete collection/table chosen for the analytical path.",
    )
    target_candidates: list[str] = Field(
        default_factory=list,
        description="Other reasonable targets considered during scoring.",
    )
    reason: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    definition: GlossaryMatch | None = Field(
        default=None,
        description=(
            "Curated glossary match that drove (or annotated) this plan. "
            "Approved matches change query construction; draft matches are "
            "surfaced as 'unverified' and do NOT change the query."
        ),
    )


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
class VectorHit(BaseModel):
    score: float = Field(
        ...,
        description="Effective relevance score. With reranking enabled this is "
        "the cross-encoder score; otherwise it's the bi-encoder cosine.",
    )
    text: str
    collection: str
    document_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder rerank score, when reranking ran on this hit. "
        "None means the hit returned untouched from the bi-encoder stage. "
        "Useful in evals to assert the reranker actually executed.",
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
class ChartSpec(BaseModel):
    chart_type: ChartType
    x: str
    y: str
    title: str | None = None


class ChartPayload(BaseModel):
    chart_type: ChartType
    title: str | None
    image_base64: str | None = None
    image_path: str | None = None
    plotly_json: dict[str, Any] | None = None
    echarts_option: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Apache ECharts ``option`` spec (https://echarts.apache.org/). "
            "Optional companion to ``plotly_json`` — when present, the "
            "frontend prefers ECharts because it has dashboard-friendlier "
            "defaults (auto-wrap titles, scrollable legends, smarter axis "
            "tick formatting, faster Canvas rendering). Plotly stays the "
            "fallback so older clients keep working."
        ),
    )
    x_field: str | None = None
    y_field: str | None = None
    series_count: int = 1
    partial_latest: bool = False
    requested_type: ChartType | None = Field(
        default=None,
        description="Original chart type the user asked for, when different from the rendered type.",
    )
    chart_id: str | None = Field(
        default=None,
        description=(
            "Stable id for this view inside a chart panel "
            "(e.g. 'trend.line', 'trend.bar', 'distribution.donut'). "
            "Used by the frontend to remember the user's last-selected tab."
        ),
    )
    view_label: str | None = Field(
        default=None,
        description=(
            "Short tab/button label for this chart view, e.g. 'Trend', "
            "'Bars', 'Distribution', 'Total'. Empty when this is the only view."
        ),
    )
    view_description: str | None = Field(
        default=None,
        description=(
            "One-line tooltip explaining what this chart view shows "
            "(why a user would pick it). Optional."
        ),
    )
    is_primary: bool = Field(
        default=False,
        description=(
            "True for the chart view selected as the default visualization "
            "for this answer. Exactly one chart in a panel is primary."
        ),
    )


# ---------------------------------------------------------------------------
# Trust panel & warnings
# ---------------------------------------------------------------------------
class AnalystWarning(BaseModel):
    code: WarningCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class TrustPanel(BaseModel):
    """Lightweight, business-shaped trust signal for the answer.

    Default-visible fields only. Technical details (generated pipeline / SQL,
    row counts at each stage, internal route ids) belong in an opt-in
    ``provenance`` block, not here.
    """

    data_as_of: datetime | None = Field(
        default=None,
        description="Latest event/record timestamp observed on the target.",
    )
    target: str | None = Field(
        default=None, description="Collection/table the answer is computed from."
    )
    data_source: DataSource | None = None
    rows_analyzed: int = 0
    time_window: str | None = Field(
        default=None,
        description="Human-readable window the answer covers, e.g. 'Jan – Apr 2026'.",
    )
    definition_used: str | None = Field(
        default=None,
        description="Display label of the metric definition applied (e.g. glossary term).",
    )
    definition_source: DefinitionSource = Field(
        default=DefinitionSource.NONE,
        description="Where the metric definition came from.",
    )
    definition_id: str | None = Field(
        default=None,
        description="Stable id of the glossary entry, when applicable.",
    )
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final response
# ---------------------------------------------------------------------------
class Provenance(BaseModel):
    """Opt-in technical details for analysts and admins.

    Returned only when ``QueryRequest.include_details`` is true. Never shown
    in the default trust panel — kept here so business users see clean output
    while analysts can audit the methodology.
    """

    fingerprint: str | None = Field(
        default=None, description="Stable hash of the executed plan."
    )
    target: str | None = None
    data_source: DataSource | None = None
    mongo_pipeline: list[dict[str, Any]] | None = None
    sql: str | None = None
    columns_sampled: list[str] = Field(default_factory=list)
    glossary_definition_id: str | None = None
    session_id: str | None = None


class AnalystResponse(BaseModel):
    question: str
    routing: RoutingDecision
    structured_data: list[dict[str, Any]] = Field(default_factory=list)
    vector_context: list[VectorHit] = Field(default_factory=list)
    insight: str = ""
    chart: ChartPayload | None = Field(
        default=None,
        description=(
            "Primary (default) chart for this answer. Same object as the "
            "first ``charts[]`` entry whose ``is_primary`` is true. Kept as "
            "a top-level field for backward compatibility with clients that "
            "render a single visualization."
        ),
    )
    charts: list[ChartPayload] = Field(
        default_factory=list,
        description=(
            "Full chart panel — every visualization the agent built for this "
            "answer (e.g. trend line, bar comparison, distribution donut, "
            "total KPI). Frontends should render this as a tab/segment "
            "control so users can switch between views without re-asking."
        ),
    )
    trust: TrustPanel | None = None
    warnings: list[AnalystWarning] = Field(default_factory=list)
    provenance: Provenance | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
