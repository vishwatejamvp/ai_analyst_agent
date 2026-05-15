"""Chart-panel builder — turn one analytical result into many chart views.

The single-chart era
--------------------
Before this module the analyst returned exactly one :class:`ChartPayload`
per answer. Choosing between line / bar / donut / KPI was a runtime
heuristic the user couldn't override without re-asking the question.

The panel era
-------------
This builder takes the same structured rows the analyst already
computed and renders **multiple chart views** of them — a trend line, a
bar chart, a distribution donut, a headline KPI — each carrying a
short ``view_label`` so the frontend can render them as a tab strip
(or segmented control). Exactly one view is marked ``is_primary``;
that's also surfaced via the legacy ``response.chart`` field for clients
that haven't adopted the panel yet.

Decision shape
--------------
The choice of which views to include is **data-shape driven**, not
question-driven. The same data shape always produces the same panel,
so users learn what to expect:

* ``KPI``                                — single row in the result.
* ``LINE + BAR + KPI``                   — single-metric time series.
* ``LINE + BAR + DONUT + KPI``           — single-metric time series with
   2–12 buckets (a donut over months reads as a distribution).
* ``LINE multi + BAR multi + DONUT + KPI`` — multi-series time series
   (e.g. website vs smart-app monthly).
* ``BAR + DONUT + KPI``                  — categorical (no time axis).

Adding a new view kind
----------------------
1. Add a constant id to :data:`_VIEW_IDS`.
2. Add a helper method to :class:`ChartPanelBuilder` that returns a
   :class:`ChartPayload` annotated with ``chart_id``, ``view_label``,
   and ``view_description``.
3. Wire it into :meth:`ChartPanelBuilder.build` under the appropriate
   data-shape branch.

Cost discipline
---------------
Every view goes through the existing :meth:`ChartService.render`. The
LLM chart planner is **not** consulted per view — the panel selection
itself is deterministic. This keeps the per-answer cost bounded
(Plotly + matplotlib rendering only; no extra LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.enums import ChartType
from models.schemas import ChartPayload
from services.chart_service import ChartService
from services.metric_profile import (
    MetricProfile,
    MultiSeriesProfile,
)
from utils.logger import logger


# Stable view ids — keep these strings frozen so frontends can persist
# "user last selected the trend.line tab" across sessions.
_VIEW_IDS: dict[str, str] = {
    "TREND_LINE": "trend.line",
    "TREND_BAR": "trend.bar",
    "TREND_LINE_MULTI": "trend.line.multi",
    "TREND_BAR_MULTI": "trend.bar.multi",
    "CATEGORY_BAR": "category.bar",
    "DISTRIBUTION_DONUT": "distribution.donut",
    "TOTAL_KPI": "total.kpi",
}


@dataclass(frozen=True)
class _ViewSpec:
    """Internal description of one view to render."""

    chart_id: str
    chart_type: ChartType
    label: str
    description: str
    rows: list[dict[str, Any]]
    has_time: bool
    title_suffix: str = ""  # appended to the base title (e.g. " — Distribution")


class ChartPanelBuilder:
    """Compose a panel of chart views from a single analytical result.

    Pure orchestration: the heavy lifting (matplotlib + Plotly rendering)
    stays in :class:`ChartService`. This builder knows the **policy** —
    what views are appropriate for which data shape — but never decides
    *how* to draw a chart.
    """

    # Bucket-count thresholds for "donut over time makes sense".
    _DONUT_MIN_BUCKETS: int = 2
    _DONUT_MAX_BUCKETS: int = 12

    def __init__(self, charts: ChartService) -> None:
        self.charts = charts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        rows: list[dict[str, Any]],
        *,
        primary_chart_type: ChartType,
        base_title: str,
        has_time: bool,
        partial_latest: bool,
        metric_profile: MetricProfile | None = None,
        multi_series_profile: MultiSeriesProfile | None = None,
    ) -> list[ChartPayload]:
        """Render every chart view appropriate for ``rows``.

        ``primary_chart_type`` decides which of the rendered views is
        flagged ``is_primary=True`` (and therefore mirrored to
        ``response.chart`` for legacy clients). The order of the
        returned list is also the recommended tab order — primary
        always first.
        """
        if not rows:
            return []

        is_multi_series = any("series" in row for row in rows)
        n_rows = len(rows)
        single_row = n_rows == 1

        try:
            specs = self._plan_views(
                rows=rows,
                is_multi_series=is_multi_series,
                has_time=has_time,
                single_row=single_row,
            )
        except Exception as exc:  # noqa: BLE001 — planner failure must never crash the answer
            logger.warning(f"Chart panel planner failed; falling back to single view: {exc}")
            specs = []

        if not specs:
            # Fallback: always at least render the requested chart so the
            # caller never gets an empty panel for non-empty data.
            specs = [
                _ViewSpec(
                    chart_id=_VIEW_IDS["TOTAL_KPI"]
                    if single_row
                    else _VIEW_IDS["TREND_LINE" if has_time else "CATEGORY_BAR"],
                    chart_type=primary_chart_type,
                    label="Chart",
                    description="Default visualization",
                    rows=rows,
                    has_time=has_time,
                ),
            ]

        # Promote the requested chart_type to first position when
        # possible, so the user's explicit ?chart_type=line still wins
        # over the default "trend line first" ordering.
        if primary_chart_type and primary_chart_type != ChartType.NONE:
            specs = self._reorder_for_primary(specs, primary_chart_type)

        rendered: list[ChartPayload] = []
        for idx, spec in enumerate(specs):
            payload = self._render_one(
                spec=spec,
                base_title=base_title,
                partial_latest=partial_latest,
                metric_profile=metric_profile,
                multi_series_profile=multi_series_profile,
            )
            if payload is None:
                continue
            payload.chart_id = spec.chart_id
            payload.view_label = spec.label
            payload.view_description = spec.description
            payload.is_primary = idx == 0
            rendered.append(payload)
        return rendered

    # ------------------------------------------------------------------
    # View planning
    # ------------------------------------------------------------------
    def _plan_views(
        self,
        *,
        rows: list[dict[str, Any]],
        is_multi_series: bool,
        has_time: bool,
        single_row: bool,
    ) -> list[_ViewSpec]:
        """Pick the set of views to render for this data shape."""
        if single_row and not is_multi_series:
            return [self._kpi_view(rows)]

        if is_multi_series:
            return self._multi_series_views(rows, has_time)
        return self._single_series_views(rows, has_time)

    def _single_series_views(
        self, rows: list[dict[str, Any]], has_time: bool
    ) -> list[_ViewSpec]:
        views: list[_ViewSpec] = []

        if has_time:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TREND_LINE"],
                    chart_type=ChartType.LINE,
                    label="Trend",
                    description="Line chart over time — see the shape and direction.",
                    rows=rows,
                    has_time=True,
                )
            )
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TREND_BAR"],
                    chart_type=ChartType.BAR,
                    label="Bars",
                    description="Bar chart over time — compare each period side by side.",
                    rows=rows,
                    has_time=True,
                )
            )
            if self._is_donut_friendly(rows):
                views.append(
                    _ViewSpec(
                        chart_id=_VIEW_IDS["DISTRIBUTION_DONUT"],
                        chart_type=ChartType.PIE,
                        label="Distribution",
                        description="Donut — share of the total each bucket contributes.",
                        rows=self._rows_for_donut(rows),
                        has_time=False,
                        title_suffix=" — Distribution",
                    )
                )
        else:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["CATEGORY_BAR"],
                    chart_type=ChartType.BAR,
                    label="Bars",
                    description="Bar chart — compare each category.",
                    rows=rows,
                    has_time=False,
                )
            )
            if self._is_donut_friendly(rows, max_buckets=8):
                views.append(
                    _ViewSpec(
                        chart_id=_VIEW_IDS["DISTRIBUTION_DONUT"],
                        chart_type=ChartType.PIE,
                        label="Distribution",
                        description="Donut — share of the total each category contributes.",
                        rows=self._rows_for_donut(rows),
                        has_time=False,
                        title_suffix=" — Distribution",
                    )
                )

        kpi_rows, kpi_label = self._kpi_rows_from_series(rows)
        if kpi_rows is not None:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TOTAL_KPI"],
                    chart_type=ChartType.KPI,
                    label="Total",
                    description="Headline number — the sum across the visible buckets.",
                    rows=kpi_rows,
                    has_time=False,
                    title_suffix=f" — {kpi_label}",
                )
            )
        return views

    def _multi_series_views(
        self, rows: list[dict[str, Any]], has_time: bool
    ) -> list[_ViewSpec]:
        views: list[_ViewSpec] = []

        if has_time:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TREND_LINE_MULTI"],
                    chart_type=ChartType.LINE,
                    label="Trend",
                    description="One line per metric / group — compare shapes over time.",
                    rows=rows,
                    has_time=True,
                )
            )
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TREND_BAR_MULTI"],
                    chart_type=ChartType.BAR,
                    label="Bars",
                    description="Side-by-side bars per period — see absolute differences.",
                    rows=rows,
                    has_time=True,
                )
            )
        else:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TREND_BAR_MULTI"],
                    chart_type=ChartType.BAR,
                    label="Bars",
                    description="Side-by-side bars per category.",
                    rows=rows,
                    has_time=False,
                )
            )

        donut_rows = self._aggregate_series_for_donut(rows)
        if donut_rows is not None and len(donut_rows) >= 2:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["DISTRIBUTION_DONUT"],
                    chart_type=ChartType.PIE,
                    label="Distribution",
                    description="Donut — share of the total each metric / group contributes.",
                    rows=donut_rows,
                    has_time=False,
                    title_suffix=" — Distribution",
                )
            )

        kpi_rows, kpi_label = self._kpi_rows_from_multi(rows)
        if kpi_rows is not None:
            views.append(
                _ViewSpec(
                    chart_id=_VIEW_IDS["TOTAL_KPI"],
                    chart_type=ChartType.KPI,
                    label="Total",
                    description="Headline number — the grand total across all series.",
                    rows=kpi_rows,
                    has_time=False,
                    title_suffix=f" — {kpi_label}",
                )
            )
        return views

    def _kpi_view(self, rows: list[dict[str, Any]]) -> _ViewSpec:
        return _ViewSpec(
            chart_id=_VIEW_IDS["TOTAL_KPI"],
            chart_type=ChartType.KPI,
            label="Total",
            description="Headline number.",
            rows=rows,
            has_time=False,
        )

    # ------------------------------------------------------------------
    # Data shape helpers
    # ------------------------------------------------------------------
    def _is_donut_friendly(
        self,
        rows: list[dict[str, Any]],
        *,
        max_buckets: int | None = None,
    ) -> bool:
        """A donut over single-series rows reads honestly only when:

        *  the bucket count is between :data:`_DONUT_MIN_BUCKETS` and
           :data:`_DONUT_MAX_BUCKETS` (inclusive),
        *  every value is non-negative (a donut can't show negatives),
        *  the total is strictly positive (an all-zero donut is just a
           blank ring).
        """
        upper = max_buckets if max_buckets is not None else self._DONUT_MAX_BUCKETS
        if not (self._DONUT_MIN_BUCKETS <= len(rows) <= upper):
            return False
        total = 0.0
        for row in rows:
            v = self._coerce_number(row.get("value"))
            if v is None or v < 0:
                return False
            total += v
        return total > 0

    def _rows_for_donut(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-positive buckets so the donut doesn't show empty slices."""
        out: list[dict[str, Any]] = []
        for row in rows:
            v = self._coerce_number(row.get("value"))
            if v is None or v <= 0:
                continue
            out.append({"label": str(row.get("label", "")), "value": v})
        return out

    def _aggregate_series_for_donut(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Sum ``value`` per ``series`` so multi-series rows render as a donut.

        Skips negative or zero totals — same honesty rule as
        :meth:`_is_donut_friendly`. Also detects and removes "parent"
        series whose total equals the sum of the other series (e.g. the
        dataset includes both ``total_transactions`` and the ``smart_app``
        + ``website`` channels that compose it). Without this filter the
        donut would show 50% for the parent and 50% for the children put
        together — a textbook double-count that misleads the reader.
        """
        agg: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("series")
            if key is None:
                continue
            key = str(key)
            v = self._coerce_number(row.get("value"))
            if v is None or v < 0:
                continue
            label = str(row.get("series_label") or key)
            bucket = agg.setdefault(key, {"label": label, "value": 0.0})
            bucket["value"] += v
        if not agg:
            return None

        parent_key = self._detect_parent_series(
            {k: bucket["value"] for k, bucket in agg.items()}
        )
        if parent_key is not None:
            agg.pop(parent_key, None)

        out = [r for r in agg.values() if r["value"] > 0]
        if len(out) < 2:
            return None
        return out

    def _kpi_rows_from_series(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Sum every value to build a single-row KPI payload."""
        total = 0.0
        seen = False
        for row in rows:
            v = self._coerce_number(row.get("value"))
            if v is None:
                continue
            seen = True
            total += v
        if not seen:
            return None, "Total"
        return [{"label": "(all)", "value": total}], "Total"

    def _kpi_rows_from_multi(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Sum every value (across every series) for a grand-total KPI.

        When one series is the sum of the others (a "parent" aggregate
        like ``total_transactions`` alongside its channel breakdown) we
        treat **only that parent** as the grand total — adding the
        children on top would double-count the same transactions and
        produce a number that doesn't match anything the user asked for.
        Falls back to a plain sum when no parent / child relationship
        is detected.
        """
        per_series: dict[str, float] = {}
        for row in rows:
            v = self._coerce_number(row.get("value"))
            if v is None:
                continue
            key = row.get("series")
            key = str(key) if key is not None else "_"
            per_series[key] = per_series.get(key, 0.0) + v
        if not per_series:
            return None, "Grand Total"

        parent_key = self._detect_parent_series(per_series)
        if parent_key is not None:
            total = per_series[parent_key]
        else:
            total = sum(per_series.values())
        return [{"label": "(all)", "value": total}], "Grand Total"

    @staticmethod
    def _detect_parent_series(
        per_series_totals: dict[str, float],
    ) -> str | None:
        """Return the series key whose total ≈ sum of every other series.

        Used by donut + KPI builders to avoid double-counting when the
        result mixes an aggregate metric (e.g. ``total_transactions``)
        with its component channels (``smart_app``, ``website``, …).
        Tolerance is 2% of the larger value so legitimate rounding
        (e.g. one channel reports in cents while the parent reports in
        whole AED) doesn't mask the relationship.

        Returns ``None`` when no series fits the sum-of-others pattern,
        in which case the caller should treat every series as
        independent and sum them naively.
        """
        if len(per_series_totals) < 3:
            # Need at least one parent + two children for the
            # double-count pattern to be possible. Two-series donuts
            # are by definition independent slices.
            return None
        positive = {k: v for k, v in per_series_totals.items() if v > 0}
        if len(positive) < 3:
            return None
        grand = sum(positive.values())
        for key, val in positive.items():
            others = grand - val
            if others <= 0:
                continue
            denom = max(val, others)
            if denom <= 0:
                continue
            if abs(val - others) / denom <= 0.02:
                return key
        return None

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Render dispatch
    # ------------------------------------------------------------------
    def _reorder_for_primary(
        self, specs: list[_ViewSpec], requested: ChartType
    ) -> list[_ViewSpec]:
        """If the user requested a specific chart_type, surface it first.

        The user's chart_type request is a *preference*, not a hard
        contract — the panel always renders all sensible views; we just
        adjust which one becomes ``is_primary``.
        """
        for i, spec in enumerate(specs):
            if spec.chart_type == requested and i != 0:
                return [specs[i]] + specs[:i] + specs[i + 1 :]
        return specs

    def _render_one(
        self,
        *,
        spec: _ViewSpec,
        base_title: str,
        partial_latest: bool,
        metric_profile: MetricProfile | None,
        multi_series_profile: MultiSeriesProfile | None,
    ) -> ChartPayload | None:
        title = f"{base_title}{spec.title_suffix}" if base_title else spec.label
        try:
            return self.charts.render(
                spec.rows,
                chart_type=spec.chart_type,
                x="label",
                y="value",
                title=title,
                has_time=spec.has_time,
                partial_latest=partial_latest if spec.has_time else False,
                requested=spec.chart_type,
                metric_profile=metric_profile,
                multi_series_profile=multi_series_profile if spec.has_time else None,
            )
        except Exception as exc:  # noqa: BLE001 — one bad view must not break the whole panel
            logger.warning(
                f"Chart panel skipped view {spec.chart_id} "
                f"({spec.chart_type.value}): {exc}"
            )
            return None
