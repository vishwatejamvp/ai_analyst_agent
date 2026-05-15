"""Assembles the lightweight :class:`TrustPanel` for the response.

Trust panel is the *default-visible* signal for end users:
freshness, scope, and which definition (if any) was applied. Anything
technical (generated pipeline, SQL, internal route ids) belongs in an
opt-in provenance block, not here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from models.enums import DataSource, DefinitionSource, MetricStatus
from models.schemas import (
    AggregationSpec,
    GlossaryMatch,
    RoutingDecision,
    TimeSpec,
    TrustPanel,
)


class TrustService:
    """Build a :class:`TrustPanel` from a routing decision + executed rows."""

    def build(
        self,
        *,
        decision: RoutingDecision,
        rows: list[dict[str, Any]],
        as_of: datetime | None = None,
        notes: list[str] | None = None,
    ) -> TrustPanel:
        spec = decision.aggregation
        defn_used, defn_source, defn_id = _definition_fields(decision.definition, spec)

        return TrustPanel(
            data_as_of=as_of,
            target=decision.target,
            data_source=(
                decision.data_source
                if decision.data_source != DataSource.AUTO
                else None
            ),
            rows_analyzed=len(rows),
            time_window=_describe_window(spec, rows),
            definition_used=defn_used,
            definition_source=defn_source,
            definition_id=defn_id,
            notes=list(notes or []),
        )


def _definition_fields(
    match: GlossaryMatch | None, spec: AggregationSpec | None
) -> tuple[str | None, DefinitionSource, str | None]:
    if match is None:
        return (None, DefinitionSource.NAIVE if spec else DefinitionSource.NONE, None)

    defn = match.definition
    if defn.status == MetricStatus.APPROVED and match.applied_to_query:
        return (defn.term, DefinitionSource.GLOSSARY, defn.id)
    # Draft / surfaced-only matches: still tell the user a definition exists,
    # but flag that it was NOT applied to query construction.
    return (defn.term, DefinitionSource.UNVERIFIED_DOC, defn.id)


def _describe_window(
    spec: AggregationSpec | None, rows: list[dict[str, Any]]
) -> str | None:
    if spec is None:
        return None

    time = spec.time
    if time is None:
        return None

    explicit = _format_explicit_window(time)
    if explicit:
        return explicit

    labels = [r.get("label") for r in rows if isinstance(r.get("label"), str)]
    if labels:
        if len(labels) == 1:
            return str(labels[0])
        return f"{labels[0]} – {labels[-1]}"

    return None


def _format_explicit_window(time: TimeSpec) -> str | None:
    if not (time.range_from or time.range_to):
        return None
    a = time.range_from.date().isoformat() if time.range_from else "earliest"
    b = time.range_to.date().isoformat() if time.range_to else "latest"
    return f"{a} – {b}"


trust_service = TrustService()
