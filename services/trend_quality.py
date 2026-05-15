"""Trend / chart quality guardrails.

Pure-policy module: looks at the executed rows and the plan that produced
them, and decides:

* whether the data is honest enough to draw a trend chart,
* whether the latest period is partial,
* whether to truncate to top-N + "other" to keep things readable,
* whether to refuse a chart entirely (with a typed warning explaining why).

It does NOT recompute numbers or change values. It only:
* attaches partial-period labels,
* drops or coalesces excess series,
* emits :class:`AnalystWarning` records the trust panel can render.

Rule of thumb: prefer a refusal + alternative over a misleading chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from models.enums import ChartType, TimeBucket, WarningCode
from models.schemas import AggregationSpec, AnalystWarning, TimeSpec


# ---------------------------------------------------------------------------
# Tunables (sane v1 defaults; promote to settings later if needed)
# ---------------------------------------------------------------------------
MIN_TREND_POINTS = 3
MIN_TOTAL_RECORDS_FOR_TREND = 30
MAX_MISSING_FRACTION = 0.30
MAX_SERIES_FOR_LINE = 8
TOP_N_BEFORE_OTHER = 7   # bar charts when categories explode

# Suspicious placeholder values that often indicate test/broken data
SUSPICIOUS_PLACEHOLDERS = {0, 1, -1, 99, 999, 9999}


@dataclass
class TrendQualityResult:
    """Outcome of inspecting structured rows for trend integrity."""

    rows: list[dict[str, Any]]
    warnings: list[AnalystWarning] = field(default_factory=list)
    suppress_chart: bool = False
    forced_chart_type: ChartType | None = None
    partial_latest: bool = False
    series_count: int = 1
    has_data_quality_issue: bool = False
    suggested_alternatives: list[str] = field(default_factory=list)


class TrendQualityChecker:
    """Decide whether a chart is honest and shape rows for safe rendering."""

    def assess(
        self,
        rows: list[dict[str, Any]],
        spec: AggregationSpec | None,
        *,
        as_of: datetime | None = None,
    ) -> TrendQualityResult:
        result = TrendQualityResult(rows=list(rows))
        if not rows:
            result.warnings.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message="No data matched the request.",
                )
            )
            result.suppress_chart = True
            return result

        # Check for multi-metric uniform data quality issues
        self._detect_multi_metric_uniformity(result, rows)

        time_spec = spec.time if spec else None

        if time_spec is not None:
            self._assess_time_series(result, time_spec, as_of=as_of)
        else:
            self._assess_categorical(result)

        return result

    # ------------------------------------------------------------------
    # Multi-metric data quality detection
    # ------------------------------------------------------------------
    def _detect_multi_metric_uniformity(
        self, result: TrendQualityResult, rows: list[dict[str, Any]]
    ) -> None:
        """Detect when ALL metrics across ALL periods show the same suspicious value.
        
        This catches data pipeline errors where every metric (e.g., transactions,
        recipients, revenues) all show value=1 for every time period, which is
        statistically impossible in real operational data.
        """
        # Group values by metric
        metrics_values: dict[str, set[float]] = {}
        for row in rows:
            metric = row.get("metric") or row.get("metric_label") or "default"
            value = _safe_number(row.get("value"))
            metrics_values.setdefault(metric, set()).add(value)
        
        # If we have multiple metrics, check if they all have uniform suspicious values
        if len(metrics_values) >= 2:
            # Check if every metric has only one unique value
            all_uniform = all(len(vals) == 1 for vals in metrics_values.values())
            
            if all_uniform:
                # Get the uniform values for each metric
                uniform_vals = {metric: list(vals)[0] for metric, vals in metrics_values.items()}
                
                # Check if all metrics share the same suspicious placeholder value
                unique_uniform_vals = set(uniform_vals.values())
                if len(unique_uniform_vals) == 1:
                    shared_value = list(unique_uniform_vals)[0]
                    
                    if shared_value in SUSPICIOUS_PLACEHOLDERS:
                        metric_names = list(metrics_values.keys())
                        result.warnings.append(
                            AnalystWarning(
                                code=WarningCode.ALL_ZERO_OR_FLAT,
                                message=(
                                    f"Data quality issue detected: All {len(metric_names)} metrics "
                                    f"({', '.join(metric_names[:3])}{'...' if len(metric_names) > 3 else ''}) "
                                    f"show a uniform value of {int(shared_value)} across all periods. "
                                    "This strongly indicates a data pipeline or aggregation error. "
                                    "Contact the data owner to verify the source before using these figures."
                                ),
                            )
                        )
                        result.has_data_quality_issue = True

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------
    def _assess_time_series(
        self,
        result: TrendQualityResult,
        time_spec: TimeSpec,
        *,
        as_of: datetime | None,
    ) -> None:
        rows = result.rows
        n = len(rows)

        if n < MIN_TREND_POINTS:
            result.warnings.append(
                AnalystWarning(
                    code=WarningCode.INSUFFICIENT_POINTS,
                    message=(
                        f"Only {n} data point(s) available — not enough to draw a trend. "
                        "Showing as a comparison instead."
                    ),
                    details={"points": n, "minimum": MIN_TREND_POINTS},
                )
            )
            result.forced_chart_type = ChartType.KPI if n == 1 else ChartType.BAR

        total_records = sum(_safe_int(r.get("value")) for r in rows)
        if total_records and total_records < MIN_TOTAL_RECORDS_FOR_TREND:
            result.warnings.append(
                AnalystWarning(
                    code=WarningCode.SPARSE_DATA,
                    message=(
                        f"Only {total_records} record(s) across {n} period(s) — patterns "
                        "in this view are likely noise rather than signal."
                    ),
                    details={"total_records": total_records, "periods": n},
                )
            )

        values = [_safe_number(r.get("value")) for r in rows]
        if values and all(v == 0 for v in values):
            result.warnings.append(
                AnalystWarning(
                    code=WarningCode.ALL_ZERO_OR_FLAT,
                    message="Series is all zero — no trend to display.",
                )
            )
            result.suppress_chart = True
            return
        
        # Enhanced detection for suspicious uniform values
        if values and len(set(values)) == 1:
            uniform_value = values[0]
            
            # Check if the uniform value is a known placeholder/suspicious value
            if uniform_value in SUSPICIOUS_PLACEHOLDERS:
                result.warnings.append(
                    AnalystWarning(
                        code=WarningCode.ALL_ZERO_OR_FLAT,
                        message=(
                            f"All periods show a value of exactly {int(uniform_value)}. "
                            "This typically indicates a data pipeline or aggregation error "
                            "rather than real operational data. Verify the source data before "
                            "using these figures for reporting."
                        ),
                    )
                )
                result.has_data_quality_issue = True
                # Don't suppress chart - let user see the issue visually
            else:
                # Non-suspicious but still flat
                result.warnings.append(
                    AnalystWarning(
                        code=WarningCode.ALL_ZERO_OR_FLAT,
                        message="Every period has the same value — trend is flat.",
                    )
                )

        if as_of is not None and n > 0:
            label = rows[-1].get("label")
            if isinstance(label, str) and _is_partial_period(label, time_spec.bucket, as_of):
                result.partial_latest = True
                rows[-1] = {**rows[-1], "partial": True}
                result.warnings.append(
                    AnalystWarning(
                        code=WarningCode.PARTIAL_PERIOD,
                        message=f"Latest period ({label}) is partial through {as_of.date().isoformat()}.",
                        details={"as_of": as_of.isoformat(), "period_label": label},
                    )
                )

        result.series_count = 1

    # ------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------
    def _assess_categorical(self, result: TrendQualityResult) -> None:
        rows = result.rows
        n = len(rows)

        if n == 1:
            result.forced_chart_type = ChartType.KPI
            return

        if n > TOP_N_BEFORE_OTHER + 1:
            top = rows[:TOP_N_BEFORE_OTHER]
            tail = rows[TOP_N_BEFORE_OTHER:]
            other_value = sum(_safe_number(r.get("value")) for r in tail)
            top.append({"label": "Other", "value": other_value, "_aggregated": True})
            result.rows = top
            result.warnings.append(
                AnalystWarning(
                    code=WarningCode.TOP_N_TRUNCATED,
                    message=(
                        f"Showing top {TOP_N_BEFORE_OTHER} of {n} categories. "
                        f"Remaining {len(tail)} are grouped as 'Other'."
                    ),
                    details={"shown": TOP_N_BEFORE_OTHER, "total": n},
                )
            )

        result.series_count = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return int(_safe_number(value))


def _is_partial_period(label: str, bucket: TimeBucket, as_of: datetime) -> bool:
    """Heuristic: does a YYYY / YYYY-MM / YYYY-Qn / YYYY-Wn label include
    ``as_of`` but extend beyond it?"""
    as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)

    if bucket == TimeBucket.YEAR and len(label) == 4 and label.isdigit():
        return int(label) == as_of.year and (as_of.month, as_of.day) < (12, 31)

    if bucket == TimeBucket.MONTH and len(label) == 7 and label[4] == "-":
        try:
            y, m = int(label[:4]), int(label[5:7])
        except ValueError:
            return False
        return (y, m) == (as_of.year, as_of.month)

    if bucket == TimeBucket.QUARTER and "-Q" in label:
        try:
            y, q = label.split("-Q")
            y_i, q_i = int(y), int(q)
        except ValueError:
            return False
        cur_q = (as_of.month - 1) // 3 + 1
        return (y_i, q_i) == (as_of.year, cur_q)

    if bucket == TimeBucket.WEEK and "-W" in label:
        try:
            y, w = label.split("-W")
            y_i, w_i = int(y), int(w)
        except ValueError:
            return False
        return (y_i, w_i) == as_of.isocalendar()[:2]

    if bucket == TimeBucket.DAY and len(label) == 10 and label.count("-") == 2:
        return label == as_of.date().isoformat()

    return False


trend_quality_checker = TrendQualityChecker()
