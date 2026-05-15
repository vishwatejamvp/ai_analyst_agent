"""Chart generation from already-aggregated structured data.

Outputs (kept stable for the existing frontend):
* base64-encoded PNG (matplotlib) for easy embedding in JSON responses,
* a saved file path on disk,
* an interactive Plotly JSON spec for frontends that prefer it,
* metadata: ``x_field``, ``y_field``, ``series_count``, ``partial_latest``,
  ``requested_type`` (so the UI can show "we redirected pie->line because…").

Charts are *only* generated from DB-computed structured data, never from
LLM output, so numbers always match the source of truth.
"""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from models.enums import ChartType
from models.schemas import ChartPayload
from services import chart_echarts
from services.metric_profile import (
    MetricProfile,
    MultiSeriesProfile,
    SERIES_PALETTE,
    SeriesProfile,
    Unit,
)
from utils.config import settings
from utils.logger import logger

_YM_LABEL_CORE = re.compile(r"^(\d{4})-(\d{2})$")
_PARTIAL_SUFFIX = " (partial)"

# Time-series line: calm teal accent on cool neutrals (readable in light UI)
_LINE_ACCENT = "#0d9488"
_LINE_ACCENT_SOFT = "#5eead4"
_LINE_FILL = "#99f6e4"
_GRID = "#e2e8f0"
_AXIS = "#64748b"
_TITLE = "#0f172a"


def _core_axis_label(lab: str) -> str:
    if lab.endswith(_PARTIAL_SUFFIX):
        return lab[: -len(_PARTIAL_SUFFIX)].strip()
    return lab.strip()


def _labels_are_year_month(labels: list[str]) -> bool:
    if not labels:
        return False
    for lab in labels:
        if not _YM_LABEL_CORE.match(_core_axis_label(lab)):
            return False
    return True


def _friendly_month_axis_labels(labels: list[str]) -> list[str]:
    """Turn ``2026-01`` … into ``Jan``, ``Feb`` … (append `` (partial)`` if present)."""
    cores_y: list[tuple[int, int]] = []
    for lab in labels:
        c = _core_axis_label(lab)
        m = _YM_LABEL_CORE.match(c)
        if not m:
            return list(labels)
        cores_y.append((int(m.group(1)), int(m.group(2))))
    years = {y for y, _ in cores_y}
    single_year = len(years) == 1
    out: list[str] = []
    for lab, (y, mo) in zip(labels, cores_y, strict=True):
        try:
            dt = datetime(y, mo, 1)
            base = dt.strftime("%b") if single_year else dt.strftime("%b %Y")
        except ValueError:
            base = _core_axis_label(lab)
        if lab.endswith(_PARTIAL_SUFFIX):
            base += _PARTIAL_SUFFIX
        out.append(base)
    return out


def _parse_yearmonth_dates(labels: list[str]) -> list[datetime] | None:
    """Convert ``YYYY-MM`` (and ``YYYY-MM (partial)``) labels to ``datetime``.

    Returns ``None`` if ANY label fails to parse — callers should fall
    back to categorical rendering rather than risk a misaligned axis.
    Each datetime is anchored to the first day of the month (the
    canonical bucket start) so renderers can space ticks consistently.
    """
    out: list[datetime] = []
    for lab in labels:
        core = _core_axis_label(lab)
        m = _YM_LABEL_CORE.match(core)
        if not m:
            return None
        try:
            out.append(datetime(int(m.group(1)), int(m.group(2)), 1))
        except ValueError:
            return None
    return out


def _adaptive_month_tick_interval(n_months: int) -> int:
    """Pick a tick interval (months) that keeps ~6-12 ticks on screen.

    Tick density rules of thumb:
      *  ≤ 12 months  → every month
      *  ≤ 24 months  → every 2 months
      *  ≤ 36 months  → every 3 months (quarterly)
      *  ≤ 60 months  → every 6 months (semi-annual)
      *  > 60 months  → every 12 months (yearly)
    """
    if n_months <= 12:
        return 1
    if n_months <= 24:
        return 2
    if n_months <= 36:
        return 3
    if n_months <= 60:
        return 6
    return 12


def _y_axis_uses_si_prefix(values: list[float]) -> bool:
    """Use compact SI tick labels (1.5M, 3.2K) when values are large."""
    if not values:
        return False
    return max(abs(v) for v in values) >= 10_000


def _wrap_long_title(title: str, *, soft_limit: int = 64) -> str:
    """Insert a Plotly ``<br>`` wrap into long titles so they read on two lines.

    Plotly titles do not auto-wrap — a 120-character title just runs off
    the right edge of the chart card and gets clipped (`"Smart App
    Transac…"`). We split on the most natural separator we can find
    (``" — "`` first, then ``" vs "``) so the break lands between
    semantic units, not mid-word.

    Returns the original string untouched when it fits within
    ``soft_limit`` or when no break point exists. The render layer
    pairs this with ``title.automargin=True`` so the figure reserves
    whatever vertical space the wrapped title needs.
    """
    if not title or len(title) <= soft_limit:
        return title
    # Prefer the em-dash separator (matches our title composer's
    # ``head — body`` shape) so the bucket adverb / op-word stays on
    # line one and the metric list moves to line two.
    if " — " in title:
        head, _, tail = title.partition(" — ")
        if 0 < len(head) <= soft_limit:
            return f"{head}<br><span style='font-size:0.85em;color:#475569'>{tail}</span>"
    # Otherwise break before the second " vs " (so two metrics fit on
    # the first line and the rest spill onto the second).
    if " vs " in title:
        parts = title.split(" vs ")
        if len(parts) >= 3:
            line1 = " vs ".join(parts[:2])
            line2 = " vs ".join(parts[2:])
            return f"{line1}<br><span style='font-size:0.85em;color:#475569'>vs {line2}</span>"
    return title


def _matplotlib_si_formatter(v: float, _pos: int) -> str:
    """Matplotlib SI tick formatter: 3,500,000 → '3.5M'."""
    abs_v = abs(v)
    if abs_v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.0f}K"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _single_calendar_year_from_labels(labels: list[str]) -> str | None:
    if not _labels_are_year_month(labels):
        return None
    ys: set[int] = set()
    for lab in labels:
        m = _YM_LABEL_CORE.match(_core_axis_label(lab))
        if m:
            ys.add(int(m.group(1)))
    if len(ys) == 1:
        return str(next(iter(ys)))
    return None


@dataclass
class _SeriesTrace:
    """One series ready for the chart layer.

    Holds the per-bucket values keyed by label so we can re-align traces
    that have different bucket coverage (one emirate may not have data
    for January; we fill its slot with ``None`` so the line breaks
    instead of dragging to zero).
    """

    key: str
    label: str
    profile: MetricProfile
    color: str
    points: dict[str, float]

    def values_for_labels(self, labels: list[str]) -> list[float | None]:
        return [self.points.get(lab) for lab in labels]

    def values_for_labels_with_nan(self, labels: list[str]) -> list[float]:
        # matplotlib renders np.nan as a gap, same UX as Plotly's None
        return [
            float("nan") if self.points.get(lab) is None else float(self.points[lab])
            for lab in labels
        ]


def _group_rows_by_series(
    rows: list[dict[str, Any]],
    multi: MultiSeriesProfile,
) -> list[_SeriesTrace]:
    """Bucket rows by ``series`` and align them to the multi profile.

    Series in ``multi.series`` define the order — keys not present in
    ``multi`` are grouped under their raw key with a fallback profile
    so an unrecognized series still renders (the alternative is dropping
    real data).
    """
    by_key: dict[str, dict[str, float]] = {}
    for row in rows:
        key = row.get("series")
        if key is None:
            continue
        key = str(key)
        label = row.get("label")
        value = _to_number(row.get("value"))
        if label is None or value is None:
            continue
        by_key.setdefault(key, {})[str(label)] = value

    result: list[_SeriesTrace] = []
    profile_by_key = {sp.key: sp for sp in multi.series}
    seen: set[str] = set()
    for sp in multi.series:
        if sp.key not in by_key:
            continue
        result.append(
            _SeriesTrace(
                key=sp.key,
                label=sp.label,
                profile=sp.profile,
                color=sp.color,
                points=by_key[sp.key],
            )
        )
        seen.add(sp.key)
    # Append any extra keys present in the data but missing from the
    # profile — happens when a group_by query returns a value the
    # caller didn't enumerate (e.g. a new emirate appears).
    for idx, key in enumerate(by_key):
        if key in seen:
            continue
        fallback = (
            profile_by_key[key].profile
            if key in profile_by_key
            else multi.series[0].profile
        )
        color = SERIES_PALETTE[(len(result) + idx) % len(SERIES_PALETTE)]
        result.append(
            _SeriesTrace(
                key=key, label=key, profile=fallback, color=color,
                points=by_key[key],
            )
        )
    return result


def _select_multi_chart_type(
    requested: ChartType | None,
    has_time: bool,
    n_series: int,
) -> tuple[ChartType, ChartType | None]:
    """Pick a chart type for multi-series rows.

    PIE / KPI never make sense across multiple series (one slice per
    series is just a stacked total in disguise; a KPI can show one
    number, not many) — both redirect to LINE for time data and BAR
    for categorical. With many series, BAR turns into clutter so we
    bias toward LINE.
    """
    if requested == ChartType.NONE:
        return ChartType.NONE, None
    if requested in (ChartType.PIE, ChartType.KPI):
        target = ChartType.LINE if has_time else ChartType.BAR
        return target, requested
    if has_time:
        return requested or ChartType.LINE, None
    return requested or ChartType.BAR, None


def _synthesize_multi_profile(
    rows: list[dict[str, Any]],
    metric_profile: MetricProfile | None,
) -> MultiSeriesProfile:
    """Last-resort profile when the caller passed multi-series rows but
    no :class:`MultiSeriesProfile`. Borrows the single-metric profile
    (when present) for every series so the chart still renders.
    """
    from services.metric_profile import classify_series  # avoid cycle

    seen: list[tuple[str, str | None, str]] = []
    used: set[str] = set()
    for row in rows:
        key = row.get("series")
        if key is None or str(key) in used:
            continue
        key = str(key)
        used.add(key)
        label = str(row.get("series_label") or key)
        metric_field = (
            metric_profile.raw_metric if metric_profile else None
        )
        seen.append((key, metric_field, label))

    if not seen:
        # Should never hit (caller already detected a series field).
        return classify_series([("series", None, "Series")])
    return classify_series(seen)


def _union_labels_sorted(
    traces: list[_SeriesTrace], has_time: bool
) -> list[str]:
    """Sorted union of all bucket labels across traces.

    For time data we sort by parsed ``YYYY-MM`` (chronological); for
    categorical we sort alphabetically as a stable, predictable default.
    """
    union: set[str] = set()
    for tr in traces:
        union.update(tr.points.keys())
    labels = list(union)
    if has_time and _labels_are_year_month(labels):
        labels.sort(key=lambda lab: _core_axis_label(lab))
    else:
        labels.sort()
    return labels


def _primary_profile_for_axis(multi: MultiSeriesProfile) -> MetricProfile:
    """First non-bounded profile, falling back to the first series."""
    for sp in multi.series:
        if not sp.profile.is_bounded_0_100:
            return sp.profile
    return multi.series[0].profile


def _secondary_profile_for_axis(multi: MultiSeriesProfile) -> MetricProfile:
    """First bounded profile (only meaningful when ``dual_axis`` is True)."""
    for sp in multi.series:
        if sp.profile.is_bounded_0_100:
            return sp.profile
    return multi.series[0].profile


def _plotly_yaxis_from_profile(profile: MetricProfile) -> dict[str, Any]:
    """Build a Plotly y-axis kwargs dict from a profile (no title attached)."""
    kwargs: dict[str, Any] = dict(gridcolor="#e2e8f0", zeroline=False)
    if profile.plotly_tickformat:
        kwargs["tickformat"] = profile.plotly_tickformat
    else:
        kwargs["separatethousands"] = True
    if profile.yaxis_ticksuffix:
        kwargs["ticksuffix"] = profile.yaxis_ticksuffix
    if profile.is_bounded_0_100:
        kwargs["range"] = [0, 100]
    return kwargs


def _flatten_values(
    traces: list[_SeriesTrace], *, primary: bool, multi: MultiSeriesProfile
) -> list[float]:
    """Flatten values for one axis (used to pick matplotlib y-limits)."""
    bucket = []
    for tr in traces:
        on_secondary = multi.dual_axis and tr.profile.is_bounded_0_100
        if primary == (not on_secondary):
            bucket.extend(v for v in tr.points.values())
    return bucket


def _apply_matplotlib_y_axis(
    ax: Any,
    values: list[float],
    profile: MetricProfile | None,
) -> None:
    """Apply tick formatter and bounds to ``ax`` from the profile (if any).

    Centralised so the BAR and LINE branches can't drift apart on number
    formatting. Without a profile this falls back to the legacy
    magnitude-based SI heuristic, preserving behaviour for callers that
    don't yet thread a profile through.
    """
    if profile is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(profile.matplotlib_value_formatter))
        if profile.is_bounded_0_100:
            ax.set_ylim(bottom=0, top=100)
            return
    elif _y_axis_uses_si_prefix(values):
        ax.yaxis.set_major_formatter(FuncFormatter(_matplotlib_si_formatter))
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda v, _p: (
                    f"{int(v):,}" if float(v) == int(v) else f"{v:,.1f}"
                )
            )
        )
    ymax = max(values) if values else 0.0
    ax.set_ylim(bottom=0, top=max(ymax * 1.14, 1.0))


def _plotly_indicator_value_format(profile: MetricProfile) -> dict[str, Any] | None:
    """Build a Plotly ``Indicator.number`` spec from a metric profile.

    Returns ``None`` for the generic fallback so Plotly's default kicks
    in (callers should leave the indicator alone in that case).
    """
    if profile.unit == Unit.PERCENT:
        return {"valueformat": ".1f", "suffix": "%"}
    if profile.unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR):
        return {"valueformat": ".3s", "suffix": f" {profile.unit_label}"}
    if profile.unit == Unit.COUNT:
        return {"valueformat": ".3s"}
    if profile.unit == Unit.DURATION_MS:
        return {"valueformat": ",.0f", "suffix": " ms"}
    if profile.unit == Unit.DURATION_SECONDS:
        return {"valueformat": ",.1f", "suffix": " s"}
    return None


def _format_data_axis_label(field_key: str, display: str | None) -> str:
    """Human-readable axis title (avoid raw ``value`` when we know the metric)."""
    if display and display.strip():
        return display.strip()
    if field_key == "value":
        return "Count / amount"
    return field_key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Inline value-label formatters (data labels printed *on* the chart)
# ---------------------------------------------------------------------------
# Maximum visible inline labels before we hide them all (cap chosen so a
# 6-series × 4-bucket dashboard panel stays legible while still allowing
# 12 monthly buckets of sparse data — the common Awqaf shape).
_MAX_INLINE_LABELS: int = 30


def _meaningful_count(*value_lists: list[float | None]) -> int:
    """Count non-null, non-zero, finite values across one or more series.

    A "meaningful" point is one a user would *want* labelled. Zeros and
    nulls are noise on a sparse chart (the line/bar already shows they
    are zero). NaNs and infinities are skipped because they represent
    missing buckets, not data.
    """
    total = 0
    for values in value_lists:
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == 0:
                continue
            if fv != fv:  # NaN
                continue
            if fv == float("inf") or fv == float("-inf"):
                continue
            total += 1
    return total


def _format_inline_value(
    value: float | None, profile: MetricProfile | None
) -> str:
    """Format a single number for an inline data label.

    Priority order:
      1. Use the metric profile's KPI formatter when available (handles
         currency / percent / SI suffixes consistently with the y-axis
         tick formatter so the bar label and tick label never disagree).
      2. Fall back to a compact SI-style formatter for large counts so
         "3,500,000" doesn't crowd the bar top.
      3. Plain comma-separated integer/float for everything else.
    """
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if profile is not None:
        try:
            return profile.kpi_value_formatter(v)
        except Exception:  # noqa: BLE001 - never let a label crash the chart
            pass
    if abs(v) >= 10_000:
        return _format_kpi_value(v)
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _annotate_bar_values(
    ax: Any,
    xs: list[Any],
    values: list[float],
    profile: MetricProfile | None,
    *,
    color: str = "#0f172a",
) -> None:
    """Print the numeric value above every bar.

    The label is placed slightly above the bar top using a fraction of
    the data range as offset so very small charts and tall charts both
    look balanced. ``None`` / NaN values are silently skipped — they
    represent missing buckets and shouldn't render as "0".
    """
    if not values:
        return
    finite = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if not finite:
        return
    span = max(finite) - min(0.0, min(finite))
    pad = span * 0.025 if span > 0 else 0.5
    for x, v in zip(xs, values, strict=False):
        if v is None or (isinstance(v, float) and v != v):
            continue
        # Skip zero-value bars: drawing "0" on every empty bucket of a
        # sparse series clutters the chart with redundant labels (the
        # zero-height bar already conveys the absence).
        if float(v) == 0:
            continue
        ax.annotate(
            _format_inline_value(v, profile),
            xy=(x, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="600",
            color=color,
            clip_on=False,
        )
    # Reserve a little extra headroom so the topmost label isn't clipped
    # by the y-axis upper bound.
    cur_bottom, cur_top = ax.get_ylim()
    needed_top = max(finite) + pad * 4
    if needed_top > cur_top:
        ax.set_ylim(bottom=cur_bottom, top=needed_top)


def _annotate_line_values(
    ax: Any,
    xs: list[Any],
    values: list[float],
    profile: MetricProfile | None,
    *,
    color: str = "#0f172a",
) -> None:
    """Print the numeric value above every line marker.

    Only call this when the series is sparse enough that the labels
    don't collide — the caller decides the threshold (~24 points is a
    comfortable upper bound for monthly data).
    """
    if not values:
        return
    finite = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if not finite:
        return
    for x, v in zip(xs, values, strict=False):
        if v is None or (isinstance(v, float) and v != v):
            continue
        # Skip zero-value markers (matches the bar-annotator policy):
        # zeros on a sparse trend are visual noise — the line already
        # touches the baseline.
        if float(v) == 0:
            continue
        ax.annotate(
            _format_inline_value(v, profile),
            xy=(x, v),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="600",
            color=color,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="#cbd5e1",
                linewidth=0.6,
                alpha=0.92,
            ),
        )


def _pick_xy(
    rows: list[dict[str, Any]],
    x: str | None,
    y: str | None,
) -> tuple[str, str]:
    if not rows:
        raise ValueError("Cannot infer chart axes from empty data")
    keys = list(rows[0].keys())
    if x and y:
        return x, y
    if "label" in keys and "value" in keys:
        return "label", "value"
    if len(keys) >= 2:
        return keys[0], keys[1]
    raise ValueError("Need at least two columns to draw a chart")


def select_chart_type(
    rows: list[dict[str, Any]],
    *,
    requested: ChartType | None,
    has_time: bool,
    values: list[float] | None = None,
) -> tuple[ChartType, ChartType | None]:
    """Pick a chart type that is honest for the data shape.

    Returns ``(rendered_type, requested_type_if_redirected)`` so callers can
    surface "we redirected pie->line because the question is over time".
    """
    n = len(rows)
    if n == 0:
        return ChartType.NONE, None

    if requested == ChartType.NONE:
        return ChartType.NONE, None

    if n == 1:
        return ChartType.KPI, requested if requested not in (None, ChartType.KPI) else None

    if has_time:
        if requested in (ChartType.LINE, ChartType.BAR, None):
            return requested or ChartType.LINE, None
        return ChartType.LINE, requested

    if values and any(v < 0 for v in values):
        if requested == ChartType.PIE:
            return ChartType.BAR, ChartType.PIE
        return requested or ChartType.BAR, None

    if requested in (ChartType.BAR, ChartType.LINE, ChartType.PIE, ChartType.KPI):
        return requested, None

    if n <= 6:
        return ChartType.BAR, None
    return ChartType.BAR, None


class ChartService:
    """Render charts from structured aggregation rows."""

    def __init__(self, chart_dir: str | None = None) -> None:
        self.chart_dir = Path(chart_dir or settings.chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(
        self,
        rows: list[dict[str, Any]],
        *,
        chart_type: ChartType | None = None,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
        has_time: bool = False,
        partial_latest: bool = False,
        requested: ChartType | None = None,
        y_axis_label: str | None = None,
        metric_profile: MetricProfile | None = None,
        multi_series_profile: MultiSeriesProfile | None = None,
    ) -> ChartPayload | None:
        """Render a chart from already-aggregated ``rows``.

        ``metric_profile`` (when supplied) replaces every ad-hoc display
        decision — y-axis label, tick formatter, y-range, hover spec, and
        KPI value formatting — with the profile's typed instructions.

        ``multi_series_profile`` (when supplied AND the rows carry a
        ``series`` field) makes the chart multi-trace: one line / bar
        group per series, distinct colors from
        :data:`metric_profile.SERIES_PALETTE`, optional dual-axis when
        units mix (e.g. % alongside AED). Pass ``None`` to keep the
        legacy single-series behaviour.
        """
        if not rows:
            return None

        # Multi-series detection. We accept the profile but can also
        # synthesize a minimal one inline if the caller forgot — the
        # alternative is silently dropping every comparison/grouped
        # chart, which is exactly the bug we're fixing.
        if any("series" in row for row in rows):
            return self._render_multi_series(
                rows,
                chart_type=chart_type,
                title=title,
                has_time=has_time,
                requested=requested,
                multi=multi_series_profile or _synthesize_multi_profile(
                    rows, metric_profile
                ),
            )

        try:
            x_key, y_key = _pick_xy(rows, x, y)
        except ValueError as exc:
            logger.warning(f"Cannot render chart: {exc}")
            return None

        labels = [str(r.get(x_key)) for r in rows]
        values = [_to_number(r.get(y_key)) for r in rows]
        if all(v is None for v in values):
            logger.warning("Chart skipped: y-axis has no numeric values")
            return None
        values = [0.0 if v is None else v for v in values]

        partial_flags = [bool(r.get("partial")) for r in rows]
        if partial_latest and partial_flags and partial_flags[-1]:
            labels[-1] = f"{labels[-1]} (partial)"

        rendered_type, redirected_from = select_chart_type(
            rows,
            requested=chart_type if chart_type is not None else requested,
            has_time=has_time,
            values=values,
        )
        if rendered_type == ChartType.NONE:
            return None

        # Profile drives the y-axis title; fall back to the legacy
        # field-key humanizer when no profile was supplied.
        y_display = (
            metric_profile.axis_title
            if metric_profile is not None
            else _format_data_axis_label(y_key, y_axis_label)
        )
        png_b64, path = self._render_matplotlib(
            rendered_type,
            labels,
            values,
            x_key,
            y_key,
            title,
            partial_flags,
            has_time=has_time,
            y_axis_label=y_display,
            metric_profile=metric_profile,
        )
        plotly_json = self._render_plotly(
            rendered_type,
            labels,
            values,
            x_key,
            y_key,
            title,
            has_time=has_time,
            y_axis_label=y_display,
            metric_profile=metric_profile,
        )
        # ECharts companion payload — same data, dashboard-friendlier
        # defaults. Frontend prefers it when present; falls back to
        # Plotly otherwise so this migration is fully incremental.
        echarts_option = self._render_echarts_single(
            rendered_type,
            labels,
            values,
            title,
            has_time=has_time,
            y_axis_label=y_display,
            metric_profile=metric_profile,
        )

        return ChartPayload(
            chart_type=rendered_type,
            title=title,
            image_base64=png_b64,
            image_path=str(path),
            plotly_json=plotly_json,
            echarts_option=echarts_option,
            x_field=x_key,
            y_field=y_key,
            series_count=1,
            partial_latest=partial_latest,
            requested_type=redirected_from,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _matplotlib_line_chart(
        ax: Any,
        labels: list[str],
        values: list[float],
        x_key: str,
        y_key: str,
        title: str | None,
        partial_flags: list[bool],
        has_time: bool,
        y_axis_label: str,
        metric_profile: MetricProfile | None = None,
    ) -> None:
        fig = ax.figure
        fig.patch.set_facecolor("#e8edf5")
        ax.set_facecolor("#f8fafc")

        # Date-axis path: parse YYYY-MM labels to datetimes and let
        # matplotlib's date locators choose tick spacing. This avoids the
        # 37-overlapping-labels failure mode of the old categorical axis.
        date_x = (
            _parse_yearmonth_dates(labels)
            if has_time and _labels_are_year_month(labels)
            else None
        )

        if date_x is not None:
            x_plot: Any = date_x
        else:
            x_plot = np.arange(len(labels))

        ax.fill_between(x_plot, values, color=_LINE_FILL, alpha=0.55, linewidth=0, zorder=1)
        (line,) = ax.plot(
            x_plot,
            values,
            color=_LINE_ACCENT,
            linewidth=3,
            marker="o",
            markersize=8,
            markerfacecolor=_LINE_ACCENT,
            markeredgecolor="white",
            markeredgewidth=2.0,
            zorder=4,
            clip_on=False,
        )
        if partial_flags and partial_flags[-1] and len(values) > 0:
            line.set_alpha(0.92)
            last_x = x_plot[-1] if date_x is not None else x_plot[-1]
            ax.scatter(
                [last_x],
                [values[-1]],
                s=200,
                zorder=6,
                facecolors="white",
                edgecolors=_LINE_ACCENT,
                linewidths=2.8,
            )

        if date_x is not None:
            interval = _adaptive_month_tick_interval(len(date_x))
            if interval >= 12:
                ax.xaxis.set_major_locator(YearLocator())
                ax.xaxis.set_major_formatter(DateFormatter("%Y"))
            else:
                ax.xaxis.set_major_locator(MonthLocator(interval=interval))
                ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
            for lbl in ax.get_xticklabels():
                lbl.set_fontsize(10)
                lbl.set_color(_AXIS)
            # Year-separator guide lines for multi-year series.
            years = sorted({d.year for d in date_x})
            if len(years) > 1:
                for yr in years[1:]:
                    ax.axvline(
                        datetime(yr, 1, 1),
                        color="#cbd5e1",
                        linestyle=":",
                        linewidth=1,
                        zorder=0,
                    )
        else:
            ax.set_xticks(x_plot)
            ax.set_xticklabels(list(labels), fontsize=10, color=_AXIS)
            if len(labels) > 6:
                plt.setp(ax.get_xticklabels(), rotation=32, ha="right")

        year_note = (
            _single_calendar_year_from_labels(labels) if date_x is not None else None
        )

        xlab = "Month" if date_x is not None else x_key.replace("_", " ").title()
        if year_note:
            xlab = f"{xlab} ({year_note})"
        ax.set_xlabel(xlab, color=_AXIS, fontsize=11, labelpad=10)
        ax.set_ylabel(y_axis_label, color=_AXIS, fontsize=11, labelpad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors=_AXIS, which="both")
        ax.grid(True, axis="y", color=_GRID, linestyle="-", linewidth=1, alpha=0.95)
        ax.set_axisbelow(True)
        _apply_matplotlib_y_axis(ax, values, metric_profile)

        # Value labels above each marker — readable without hover (mirrors
        # the Awqaf statistics chart style). The density gate counts only
        # meaningful (non-null, non-zero) points so a 12-month series
        # where 10 buckets are zero is still labelled (only ~2 visible
        # labels — well below the cap).
        if _meaningful_count(values) <= _MAX_INLINE_LABELS:
            _annotate_line_values(ax, list(x_plot), values, metric_profile)

        if title:
            ax.set_title(
                title,
                loc="left",
                color=_TITLE,
                fontsize=14.5,
                fontweight="700",
                pad=18,
            )

    def _render_matplotlib(
        self,
        chart_type: ChartType,
        labels: list[str],
        values: list[float],
        x_key: str,
        y_key: str,
        title: str | None,
        partial_flags: list[bool],
        *,
        has_time: bool = False,
        y_axis_label: str = "Value",
        metric_profile: MetricProfile | None = None,
    ) -> tuple[str, Path]:
        fig, ax = plt.subplots(figsize=(10, 5.8), dpi=140)
        try:
            if chart_type == ChartType.BAR:
                colors = [
                    "#93c5fd" if partial else "#3b82f6"
                    for partial in (partial_flags or [False] * len(labels))
                ]
                date_x = (
                    _parse_yearmonth_dates(labels)
                    if has_time and _labels_are_year_month(labels)
                    else None
                )
                if date_x is not None:
                    width_days = 25  # ~one month bar width
                    ax.bar(
                        date_x,
                        values,
                        color=colors,
                        width=width_days,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                    interval = _adaptive_month_tick_interval(len(date_x))
                    if interval >= 12:
                        ax.xaxis.set_major_locator(YearLocator())
                        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
                    else:
                        ax.xaxis.set_major_locator(MonthLocator(interval=interval))
                        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
                    ax.set_xlabel("Month", color=_AXIS, fontsize=11)
                    bar_xs: list[Any] = list(date_x)
                else:
                    x_idx = np.arange(len(labels))
                    ax.bar(x_idx, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
                    ax.set_xticks(x_idx)
                    ax.set_xticklabels(list(labels))
                    ax.set_xlabel(x_key, color=_AXIS, fontsize=11)
                    if len(labels) > 6:
                        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
                    bar_xs = list(x_idx)
                ax.set_ylabel(y_axis_label, color=_AXIS, fontsize=11)
                _apply_matplotlib_y_axis(ax, values, metric_profile)
                # Print the value on top of each bar so the chart reads
                # like the Awqaf reference (numbers visible without
                # squinting at the y-axis). Density gate counts only
                # *meaningful* points (non-null, non-zero) — sparse
                # series with mostly-zero buckets stay labelled because
                # they only have a handful of real values.
                if _meaningful_count(values) <= _MAX_INLINE_LABELS:
                    _annotate_bar_values(ax, bar_xs, values, metric_profile)
            elif chart_type == ChartType.LINE:
                self._matplotlib_line_chart(
                    ax,
                    labels,
                    values,
                    x_key,
                    y_key,
                    title,
                    partial_flags,
                    has_time,
                    y_axis_label,
                    metric_profile,
                )
            elif chart_type == ChartType.PIE:
                # Donut style with center "Total" — matches the Awqaf
                # statistics chart visual language. Each slice is
                # annotated outside the ring with the absolute value
                # (so the reader gets both proportion AND magnitude
                # without hovering).
                pie_palette = (
                    "#11649e",  # deep navy (Awqaf primary)
                    "#10b981",  # emerald
                    "#f59e0b",  # amber
                    "#8b5cf6",  # violet
                    "#ec4899",  # pink
                    "#14b8a6",  # teal
                    "#ef4444",  # red
                    "#a855f7",  # purple
                )
                slice_colors = [
                    pie_palette[i % len(pie_palette)] for i in range(len(values))
                ]
                total_value = sum(v for v in values if v is not None)

                def _slice_pct_label(pct: float) -> str:
                    # Hide labels for very small slices so the donut
                    # doesn't end up with overlapping "0.1%" text.
                    return f"{pct:.1f}%" if pct >= 3.0 else ""

                wedges, _texts, autotexts = ax.pie(
                    values,
                    labels=None,
                    autopct=_slice_pct_label,
                    startangle=90,
                    counterclock=False,
                    colors=slice_colors,
                    pctdistance=0.81,
                    wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2),
                    textprops=dict(color="white", fontsize=12, fontweight="700"),
                )
                ax.axis("equal")
                # Per-slice absolute values live in the legend below
                # ("Website — 57") rather than as floating annotations
                # outside the donut — that matches the Awqaf reference
                # and avoids collisions with the title and legend.
                value_fmt = (
                    metric_profile.kpi_value_formatter
                    if metric_profile is not None
                    else _format_kpi_value
                )
                # Center total — the headline number, just like the
                # reference donut. We use the metric profile's KPI
                # formatter so 3,500,000 reads as "3.5M" the same way
                # the y-axis would format it.
                center_value = value_fmt(float(total_value))
                ax.text(
                    0, 0.10, "Total",
                    ha="center", va="center",
                    fontsize=13, color="#64748b", fontweight="500",
                )
                ax.text(
                    0, -0.12, center_value,
                    ha="center", va="center",
                    fontsize=30, color="#0f172a", fontweight="800",
                )
                # Legend below: colored swatch + label + absolute value
                # so users can read "Website 57" / "Smart Application 5"
                # at a glance.
                legend_labels = [
                    f"{lab} — {value_fmt(float(v))}"
                    if v is not None else lab
                    for lab, v in zip(labels, values, strict=True)
                ]
                ax.legend(
                    wedges, legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(len(legend_labels), 3),
                    frameon=False,
                    fontsize=10,
                )
            elif chart_type == ChartType.KPI:
                ax.axis("off")
                value = values[0] if values else 0
                label = labels[0] if labels else ""
                value_text = (
                    metric_profile.kpi_value_formatter(value)
                    if metric_profile is not None
                    else _format_kpi_value(value)
                )
                ax.text(
                    0.5, 0.55, value_text,
                    ha="center", va="center", fontsize=42, fontweight="bold",
                    color="#0f172a", transform=ax.transAxes,
                )
                # Hide the synthetic ``(all)`` placeholder — when there's
                # no real grouping, the title above already names the
                # metric and a redundant subtitle just adds visual noise.
                subtitle = "" if label.strip() in {"", "(all)"} else label
                if subtitle:
                    ax.text(
                        0.5, 0.30, subtitle, ha="center", va="center",
                        fontsize=14, color="#475569", transform=ax.transAxes,
                    )
            else:
                raise ValueError(f"Unsupported chart_type: {chart_type}")

            if title and chart_type != ChartType.KPI:
                if chart_type != ChartType.LINE:
                    ax.set_title(title, color=_TITLE, fontsize=13, fontweight="600", pad=12)
            elif title and chart_type == ChartType.KPI:
                ax.text(
                    0.5, 0.85, title, ha="center", va="center",
                    fontsize=12, color="#64748b", transform=ax.transAxes,
                )
            fig.tight_layout(pad=1.2)

            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="png",
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            buffer.seek(0)
            png_b64 = base64.b64encode(buffer.read()).decode("ascii")

            filename = f"chart_{int(time.time() * 1000)}_{chart_type.value}.png"
            path = self.chart_dir / filename
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(png_b64))
            return png_b64, path
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Multi-series renderer (one trace per ``series``)
    # ------------------------------------------------------------------
    def _render_multi_series(
        self,
        rows: list[dict[str, Any]],
        *,
        chart_type: ChartType | None,
        title: str | None,
        has_time: bool,
        requested: ChartType | None,
        multi: MultiSeriesProfile,
    ) -> ChartPayload | None:
        """Render rows of shape ``{label, value, series}`` as one trace per series.

        The decision tree mirrors the single-series path:
        *  pick a chart type honest for the shape (PIE / KPI redirected
           to LINE for multi-series — they don't compose with multiple
           traces);
        *  for time-series, build a date-x once so every trace shares
           the same axis;
        *  optionally drive a secondary y-axis when units mix.
        """
        traces = _group_rows_by_series(rows, multi)
        if not traces:
            logger.warning("Multi-series chart skipped: no usable series rows")
            return None

        # PIE / KPI don't make sense across multiple series — collapse
        # to LINE for time data and BAR otherwise. Matches the spirit
        # of select_chart_type for the single-series path.
        rendered_type, redirected_from = _select_multi_chart_type(
            chart_type or requested, has_time, len(traces)
        )
        if rendered_type == ChartType.NONE:
            return None

        png_b64, path = self._matplotlib_multi_series(
            rendered_type, traces, title, has_time, multi
        )
        plotly_json = self._plotly_multi_series(
            rendered_type, traces, title, has_time, multi
        )
        echarts_option = self._echarts_multi_series(
            rendered_type, traces, title, has_time, multi
        )

        return ChartPayload(
            chart_type=rendered_type,
            title=title,
            image_base64=png_b64,
            image_path=str(path),
            plotly_json=plotly_json,
            echarts_option=echarts_option,
            x_field="label",
            y_field="value",
            series_count=len(traces),
            partial_latest=False,
            requested_type=redirected_from,
        )

    def _plotly_multi_series(
        self,
        chart_type: ChartType,
        traces: list[_SeriesTrace],
        title: str | None,
        has_time: bool,
        multi: MultiSeriesProfile,
    ) -> dict[str, Any]:
        # Build one trace per series. When the bucket axis is a time
        # axis we share a single ``date_x`` across all traces so they
        # align even when some series have missing buckets — Plotly
        # treats nulls as gaps which reads better than fake zeros for
        # absent observations.
        all_labels = _union_labels_sorted(traces, has_time)
        date_x = (
            _parse_yearmonth_dates(all_labels)
            if has_time and _labels_are_year_month(all_labels)
            else None
        )

        # Inline label gating for multi-series: count only meaningful
        # (non-null, non-zero) points across every series. Sparse data
        # (most months are zero) keeps full labels because the visible
        # count is small; dense data correctly suppresses labels to
        # avoid clutter.
        per_series_values = [tr.values_for_labels(all_labels) for tr in traces]
        show_inline_labels = _meaningful_count(*per_series_values) <= _MAX_INLINE_LABELS

        plotly_traces: list[Any] = []
        for tr in traces:
            x_axis = date_x if date_x is not None else all_labels
            y_values = tr.values_for_labels(all_labels)
            yref = "y2" if (multi.dual_axis and tr.profile.is_bounded_0_100) else "y"
            hover_value_spec = tr.profile.plotly_hover_value_format
            ticksuffix = tr.profile.yaxis_ticksuffix
            if date_x is not None:
                hover = (
                    "<b>%{x|%B %Y}</b><br>"
                    f"{tr.label}: %{{y:{hover_value_spec}}}{ticksuffix}<extra></extra>"
                )
            else:
                hover = (
                    "<b>%{x}</b><br>"
                    f"{tr.label}: %{{y:{hover_value_spec}}}{ticksuffix}<extra></extra>"
                )

            # Blank zero-valued labels (matches the matplotlib annotator
            # policy and prevents a wall of "0" markers across empty
            # months on a sparse multi-series chart).
            text_values = [
                _format_inline_value(v, tr.profile) if (v not in (None, 0)) else ""
                for v in y_values
            ]

            common: dict[str, Any] = dict(
                x=x_axis,
                y=y_values,
                name=tr.label,
                hovertemplate=hover,
            )
            if multi.dual_axis:
                common["yaxis"] = yref

            if chart_type == ChartType.BAR:
                bar_extra: dict[str, Any] = dict(
                    marker=dict(color=tr.color, line=dict(color="white", width=0.6)),
                )
                if show_inline_labels:
                    bar_extra["text"] = text_values
                    bar_extra["textposition"] = "outside"
                    bar_extra["textfont"] = dict(
                        color=tr.color, size=11, family="system-ui, sans-serif",
                    )
                    bar_extra["cliponaxis"] = False
                plotly_traces.append(go.Bar(**common, **bar_extra))
            else:  # LINE
                line_extra: dict[str, Any] = dict(
                    mode="lines+markers+text" if show_inline_labels else "lines+markers",
                    line=dict(color=tr.color, width=2.5, shape="linear"),
                    marker=dict(
                        size=7,
                        color=tr.color,
                        line=dict(color="white", width=1.5),
                    ),
                )
                if show_inline_labels:
                    line_extra["text"] = text_values
                    line_extra["textposition"] = "top center"
                    line_extra["textfont"] = dict(
                        color=tr.color, size=10, family="system-ui, sans-serif",
                    )
                    line_extra["cliponaxis"] = False
                plotly_traces.append(go.Scatter(**common, **line_extra))

        fig = go.Figure(data=plotly_traces)

        # Y-axis (primary): driven by the first non-bounded series profile
        # in the dual-axis case, or the only profile otherwise.
        primary_profile = _primary_profile_for_axis(multi)
        yaxis_kwargs = _plotly_yaxis_from_profile(primary_profile)
        layout_kwargs: dict[str, Any] = dict(yaxis=yaxis_kwargs)

        if multi.dual_axis:
            secondary_profile = _secondary_profile_for_axis(multi)
            yaxis2_kwargs = _plotly_yaxis_from_profile(secondary_profile)
            yaxis2_kwargs["overlaying"] = "y"
            yaxis2_kwargs["side"] = "right"
            yaxis2_kwargs["showgrid"] = False
            layout_kwargs["yaxis2"] = yaxis2_kwargs
            layout_kwargs["yaxis"] = {
                **yaxis_kwargs,
                "title": dict(text=multi.primary_axis_label),
            }
            layout_kwargs["yaxis2"] = {
                **yaxis2_kwargs,
                "title": dict(text=multi.secondary_axis_label or ""),
            }
        else:
            layout_kwargs["yaxis"] = {
                **yaxis_kwargs,
                "title": dict(text=multi.primary_axis_label),
            }

        # X-axis: same date / categorical logic as single-series.
        xaxis_kwargs: dict[str, Any] = dict(
            showgrid=False, showline=True, linewidth=1, linecolor="#cbd5e1",
        )
        if date_x is not None:
            xaxis_kwargs["type"] = "date"
            xaxis_kwargs["tickformat"] = "%b %Y"
            xaxis_kwargs["nticks"] = 8
            span_months = (
                (date_x[-1].year - date_x[0].year) * 12
                + (date_x[-1].month - date_x[0].month)
                + 1
            )
            if span_months >= 12:
                xaxis_kwargs["rangeslider"] = dict(visible=True, thickness=0.06)
                # Rangeselector pinned to the **bottom-left** (just above
                # the rangeslider) instead of the top-left where it used
                # to overlap the title. Same buttons, much less crowding
                # in the title bar.
                xaxis_kwargs["rangeselector"] = dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#f1f5f9", activecolor="#0d9488",
                    bordercolor="#cbd5e1", borderwidth=1,
                    font=dict(size=11, color="#334155"),
                    x=0, y=-0.04, xanchor="left", yanchor="top",
                )
        elif len(all_labels) > 8:
            xaxis_kwargs["tickangle"] = -32

        shapes: list[dict[str, Any]] = []
        if date_x is not None:
            years = sorted({d.year for d in date_x})
            if len(years) > 1:
                for yr in years[1:]:
                    shapes.append(dict(
                        type="line", xref="x", yref="paper",
                        x0=datetime(yr, 1, 1), x1=datetime(yr, 1, 1),
                        y0=0, y1=1,
                        line=dict(color="#cbd5e1", width=1, dash="dot"),
                        layer="below",
                    ))

        # Title sizing: long multi-series titles overflow the card. Drop
        # the font 1-2 points and let Plotly's ``automargin`` reserve
        # whatever vertical space it needs so the title is never cut off
        # mid-word ("Smart App Transac…"). The hard wrap at ~64 chars is
        # belt-and-braces for the worst cases.
        title_text = _wrap_long_title(title or "")
        title_font_size = 16 if len(title_text) > 48 else 17
        # Legend layout: above 4 series the horizontal bar wraps and
        # eats real chart area. Push it further down with a generous
        # bottom margin so the legend never sits on top of the bars
        # / lines, and centre it under the chart for symmetry.
        n_series = len(plotly_traces)
        # Range slider eats ~6% of the chart; plus the rangeselector
        # buttons (now bottom-pinned) take another ~6%; legend needs
        # ~16% to avoid bar collision with 4-series labels. Total
        # bottom budget grows with the season.
        bottom_margin = 150 if (date_x is not None) else 130
        legend_y = -0.42 if date_x is not None else -0.30
        # Per-series isolation is handled by a native HTML <select>
        # rendered above the chart by the frontend (see
        # ``buildSeriesSelector`` in ``static/index.html``). We keep
        # the Plotly figure free of in-SVG updatemenus because:
        #   * the SVG dropdown is easy to miss in the corner,
        #   * it competes for space with the rangeselector + title,
        #   * a real <select> reads natively on mobile and to a11y tools.
        # The frontend reads trace ``name`` straight off the figure
        # data, so no extra metadata is needed in the payload.
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=title_font_size, color="#0f172a", family="system-ui, sans-serif"),
                x=0, xanchor="left", pad=dict(t=8, b=12),
                automargin=True,
            ),
            xaxis=xaxis_kwargs,
            xaxis_title="Month" if date_x is not None else "",
            template="plotly_white",
            paper_bgcolor="#eef2f7",
            plot_bgcolor="#f8fafc",
            font=dict(color="#475569", size=12, family="system-ui, sans-serif"),
            barmode="group" if chart_type == ChartType.BAR else None,
            shapes=shapes,
            legend=dict(
                orientation="h", yanchor="top", y=legend_y,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.0)",
                bordercolor="#cbd5e1", borderwidth=0,
                font=dict(size=11 if n_series <= 4 else 10, color="#334155"),
                itemwidth=30,
                tracegroupgap=8,
            ),
            hoverlabel=dict(
                bgcolor="white",
                font=dict(family="system-ui, sans-serif", size=12, color="#0f172a"),
                bordercolor="#cbd5e1",
            ),
            margin=dict(l=64, r=64, t=70, b=bottom_margin),
            **layout_kwargs,
        )
        return fig.to_plotly_json()

    def _matplotlib_multi_series(
        self,
        chart_type: ChartType,
        traces: list[_SeriesTrace],
        title: str | None,
        has_time: bool,
        multi: MultiSeriesProfile,
    ) -> tuple[str, Path]:
        all_labels = _union_labels_sorted(traces, has_time)
        date_x = (
            _parse_yearmonth_dates(all_labels)
            if has_time and _labels_are_year_month(all_labels)
            else None
        )

        fig, ax = plt.subplots(figsize=(11, 5.8), dpi=140)
        try:
            fig.patch.set_facecolor("#e8edf5")
            ax.set_facecolor("#f8fafc")

            ax2 = ax.twinx() if multi.dual_axis else None
            if ax2 is not None:
                ax2.set_facecolor("none")

            n = len(traces)
            x_index_to_pos = (
                date_x if date_x is not None else np.arange(len(all_labels))
            )

            # Inline-label gate: count only meaningful (non-null,
            # non-zero) points across every series. Sparse data (most
            # months are zero) still gets full labels — there are only
            # a handful of real values to draw. Dense charts (every
            # series ×every bucket = real number) get suppressed.
            all_meaningful = _meaningful_count(
                *[tr.values_for_labels_with_nan(all_labels) for tr in traces]
            )
            label_density_ok = all_meaningful <= _MAX_INLINE_LABELS

            for idx, tr in enumerate(traces):
                target_ax = ax2 if (multi.dual_axis and tr.profile.is_bounded_0_100) else ax
                y_values = tr.values_for_labels_with_nan(all_labels)
                if chart_type == ChartType.BAR and date_x is None:
                    # Side-by-side bars: split each x slot into N sub-bars.
                    width = 0.8 / max(n, 1)
                    offsets = np.arange(len(all_labels)) - 0.4 + width * (idx + 0.5)
                    bar_values = [0.0 if v is None else v for v in y_values]
                    target_ax.bar(
                        offsets,
                        bar_values,
                        width=width,
                        color=tr.color,
                        edgecolor="white",
                        linewidth=0.6,
                        label=tr.label,
                    )
                    if label_density_ok:
                        _annotate_bar_values(
                            target_ax, list(offsets), bar_values, tr.profile,
                            color=tr.color,
                        )
                else:
                    target_ax.plot(
                        x_index_to_pos,
                        y_values,
                        color=tr.color,
                        linewidth=2.4,
                        marker="o",
                        markersize=5.5,
                        markerfacecolor=tr.color,
                        markeredgecolor="white",
                        markeredgewidth=1.4,
                        label=tr.label,
                    )
                    # Annotate every plotted line (this branch also
                    # handles the BAR-on-time-axis fallback where bars
                    # can't be grouped cleanly, so we draw them as
                    # lines — the user still expects the values).
                    if label_density_ok:
                        _annotate_line_values(
                            target_ax,
                            list(x_index_to_pos),
                            y_values,
                            tr.profile,
                            color=tr.color,
                        )

            if date_x is not None:
                interval = _adaptive_month_tick_interval(len(date_x))
                if interval >= 12:
                    ax.xaxis.set_major_locator(YearLocator())
                    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
                else:
                    ax.xaxis.set_major_locator(MonthLocator(interval=interval))
                    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
                years = sorted({d.year for d in date_x})
                if len(years) > 1:
                    for yr in years[1:]:
                        ax.axvline(datetime(yr, 1, 1), color="#cbd5e1",
                                   linestyle=":", linewidth=1, zorder=0)
            elif chart_type != ChartType.BAR:
                ax.set_xticks(np.arange(len(all_labels)))
                ax.set_xticklabels(list(all_labels))
                if len(all_labels) > 6:
                    plt.setp(ax.get_xticklabels(), rotation=32, ha="right")
            else:
                ax.set_xticks(np.arange(len(all_labels)))
                ax.set_xticklabels(list(all_labels))
                if len(all_labels) > 6:
                    plt.setp(ax.get_xticklabels(), rotation=32, ha="right")

            primary_profile = _primary_profile_for_axis(multi)
            ax.set_ylabel(multi.primary_axis_label, color=_AXIS, fontsize=11, labelpad=10)
            _apply_matplotlib_y_axis(ax, _flatten_values(traces, primary=True, multi=multi), primary_profile)
            if ax2 is not None:
                secondary_profile = _secondary_profile_for_axis(multi)
                ax2.set_ylabel(
                    multi.secondary_axis_label or "",
                    color=_AXIS, fontsize=11, labelpad=10,
                )
                _apply_matplotlib_y_axis(
                    ax2,
                    _flatten_values(traces, primary=False, multi=multi),
                    secondary_profile,
                )
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_color("#cbd5e1")
                ax2.tick_params(colors=_AXIS, which="both")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(ax2 is not None)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")
            ax.tick_params(colors=_AXIS, which="both")
            ax.grid(True, axis="y", color=_GRID, linestyle="-", linewidth=1, alpha=0.95)
            ax.set_axisbelow(True)

            handles, labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                labels += l2
            if handles:
                ax.legend(
                    handles, labels,
                    loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=min(len(handles), 4),
                    frameon=True, fontsize=10,
                    edgecolor="#cbd5e1",
                )

            if title:
                ax.set_title(
                    title, loc="left", color=_TITLE,
                    fontsize=14.5, fontweight="700", pad=18,
                )

            fig.tight_layout(pad=1.2)

            buffer = io.BytesIO()
            fig.savefig(
                buffer, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none",
            )
            buffer.seek(0)
            png_b64 = base64.b64encode(buffer.read()).decode("ascii")
            filename = f"chart_{int(time.time() * 1000)}_multi_{chart_type.value}.png"
            path = self.chart_dir / filename
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(png_b64))
            return png_b64, path
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # ECharts payloads (Apache 2.0). Companion to Plotly — the frontend
    # prefers these when present because their dashboard defaults
    # (auto-wrap titles, scrollable legends, smarter axis formatters,
    # Canvas rendering) eliminate the layout bugs Plotly's defaults
    # produce inside a card.
    # ------------------------------------------------------------------
    @staticmethod
    def _render_echarts_single(
        chart_type: ChartType,
        labels: list[str],
        values: list[float],
        title: str | None,
        *,
        has_time: bool,
        y_axis_label: str,
        metric_profile: MetricProfile | None,
    ) -> dict[str, Any] | None:
        """ECharts ``option`` for a single-series chart. ``None`` → fall back to Plotly."""
        date_x = (
            _parse_yearmonth_dates(labels)
            if has_time and _labels_are_year_month(labels)
            else None
        )
        iso_labels = chart_echarts.to_iso_labels(date_x)
        if chart_type in (ChartType.LINE, ChartType.BAR):
            return chart_echarts.build_single_series_option(
                chart_type,
                title=title,
                labels=labels,
                values=values,
                has_time=has_time,
                date_iso_labels=iso_labels,
                profile=metric_profile,
                y_axis_label=y_axis_label,
            )
        if chart_type == ChartType.PIE:
            return chart_echarts.build_donut_option(
                title=title,
                labels=labels,
                values=values,
                profile=metric_profile,
            )
        if chart_type == ChartType.KPI:
            value = values[0] if values else 0.0
            label = labels[0] if labels else ""
            return chart_echarts.build_kpi_option(
                title=title,
                label=label,
                value=float(value),
                profile=metric_profile,
            )
        return None  # NONE / unsupported → Plotly handles it

    @staticmethod
    def _echarts_multi_series(
        chart_type: ChartType,
        traces: list[_SeriesTrace],
        title: str | None,
        has_time: bool,
        multi: MultiSeriesProfile,
    ) -> dict[str, Any] | None:
        """ECharts ``option`` for a multi-series chart. ``None`` → fall back to Plotly."""
        if chart_type not in (ChartType.LINE, ChartType.BAR):
            return None  # multi-series PIE/KPI already redirect upstream
        all_labels = _union_labels_sorted(traces, has_time)
        date_x = (
            _parse_yearmonth_dates(all_labels)
            if has_time and _labels_are_year_month(all_labels)
            else None
        )
        iso_labels = chart_echarts.to_iso_labels(date_x)
        # Serialise traces to plain dicts so chart_echarts stays free
        # of internal types.
        series_data = [
            {
                "label": tr.label,
                "color": tr.color,
                "is_bounded_0_100": tr.profile.is_bounded_0_100,
                "points": dict(tr.points),
            }
            for tr in traces
        ]
        return chart_echarts.build_multi_series_option(
            chart_type,
            title=title,
            series_data=series_data,
            multi=multi,
            has_time=has_time,
            all_labels=all_labels,
            date_iso_labels=iso_labels,
        )

    @staticmethod
    def _render_plotly(
        chart_type: ChartType,
        labels: list[str],
        values: list[float],
        x_key: str,
        y_key: str,
        title: str | None,
        *,
        has_time: bool = False,
        y_axis_label: str = "Value",
        metric_profile: MetricProfile | None = None,
    ) -> dict[str, Any]:
        # Single source of truth for y-value formatting: hovers, tick
        # labels, KPI numbers all read from the profile when present.
        hover_value_spec = (
            metric_profile.plotly_hover_value_format
            if metric_profile is not None
            else ",.0f"
        )
        date_hover = (
            "<b>%{x|%B %Y}</b><br>%{y:" + hover_value_spec + "}"
            + (metric_profile.yaxis_ticksuffix if metric_profile else "")
            + "<extra></extra>"
        )
        cat_hover = (
            "%{x}<br>%{y:" + hover_value_spec + "}"
            + (metric_profile.yaxis_ticksuffix if metric_profile else "")
            + "<extra></extra>"
        )
        # Time-series: parse labels to actual datetimes so Plotly drives a
        # proper date axis (auto tick spacing, zoom, range selector). The
        # categorical-string fallback is reserved for non-parsable labels.
        date_x: list[datetime] | None = None
        if (
            chart_type in (ChartType.LINE, ChartType.BAR)
            and has_time
            and _labels_are_year_month(labels)
        ):
            date_x = _parse_yearmonth_dates(labels)

        # Inline value labels (printed on the chart). These mirror the
        # Awqaf statistics widget so the reader sees actual numbers
        # without hovering. The density gate counts only meaningful
        # (non-null, non-zero) values; zeros are blanked out below so
        # sparse series stay readable without "0" plastered on every
        # empty bucket.
        show_inline_labels = _meaningful_count(values) <= _MAX_INLINE_LABELS
        text_values = [
            _format_inline_value(v, metric_profile) if (v not in (None, 0)) else ""
            for v in values
        ]

        if chart_type == ChartType.BAR:
            bar_kwargs: dict[str, Any] = dict(
                y=values,
                marker=dict(color="#3b82f6", line=dict(color="white", width=1)),
            )
            if date_x is not None:
                bar_kwargs["x"] = date_x
                bar_kwargs["hovertemplate"] = date_hover
            else:
                bar_kwargs["x"] = list(labels)
                bar_kwargs["hovertemplate"] = cat_hover
            if show_inline_labels:
                bar_kwargs["text"] = text_values
                bar_kwargs["textposition"] = "outside"
                bar_kwargs["textfont"] = dict(
                    color="#0f172a", size=12, family="system-ui, sans-serif",
                )
                bar_kwargs["cliponaxis"] = False
            fig = go.Figure(data=[go.Bar(**bar_kwargs)])
        elif chart_type == ChartType.LINE:
            scatter_kwargs: dict[str, Any] = dict(
                y=values,
                mode="lines+markers+text" if show_inline_labels else "lines+markers",
                line=dict(color=_LINE_ACCENT, width=3, shape="linear"),
                marker=dict(
                    size=8,
                    color=_LINE_ACCENT,
                    line=dict(color="white", width=2),
                ),
                fill="tozeroy",
                fillcolor="rgba(153, 246, 228, 0.45)",
            )
            if show_inline_labels:
                scatter_kwargs["text"] = text_values
                scatter_kwargs["textposition"] = "top center"
                scatter_kwargs["textfont"] = dict(
                    color="#0f172a", size=11, family="system-ui, sans-serif",
                )
                scatter_kwargs["cliponaxis"] = False
            if date_x is not None:
                scatter_kwargs["x"] = date_x
                scatter_kwargs["hovertemplate"] = date_hover
            else:
                scatter_kwargs["x"] = list(labels)
                scatter_kwargs["hovertemplate"] = cat_hover
            fig = go.Figure(data=[go.Scatter(**scatter_kwargs)])
        elif chart_type == ChartType.PIE:
            # Donut style — visual language matches the Awqaf "Statistics"
            # widget: deep-navy primary slice, emerald secondary, amber
            # tertiary; large headline value in the center; per-slice
            # hover that the frontend can wire to swap the center text
            # via `customdata`.
            #
            # The annotations carry stable ``name`` ids ("center.title" /
            # "center.value") so a frontend Plotly hover handler can
            # target them by name without DOM scraping. See
            # ``donut_hover_center_snippet`` at the bottom of this module
            # for the JS recipe.
            pie_palette = [
                "#11649e",  # deep navy (Awqaf primary)
                "#10b981",  # emerald
                "#f59e0b",  # amber
                "#8b5cf6",  # violet
                "#ec4899",  # pink
                "#14b8a6",  # teal
                "#ef4444",  # red
                "#a855f7",  # purple
            ]
            slice_colors = [
                pie_palette[i % len(pie_palette)] for i in range(len(values))
            ]
            value_fmt = (
                metric_profile.kpi_value_formatter
                if metric_profile is not None
                else _format_kpi_value
            )
            total_value = sum(v for v in values if v is not None)
            center_value_text = value_fmt(float(total_value)) if total_value else "0"

            # customdata payload per slice — index 0=label, 1=formatted value,
            # 2=raw numeric value, 3=slice color. The frontend hover
            # handler reads these to update the center annotations and
            # match the Awqaf "hover swaps center" behaviour exactly.
            customdata = [
                [lab, value_fmt(float(v)) if v is not None else "—",
                 v if v is not None else 0,
                 slice_colors[i]]
                for i, (lab, v) in enumerate(zip(labels, values, strict=True))
            ]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.62,
                        sort=False,
                        direction="clockwise",
                        marker=dict(
                            colors=slice_colors,
                            line=dict(color="white", width=2),
                        ),
                        textinfo="percent",
                        texttemplate="%{percent:.1%}",
                        textposition="inside",
                        insidetextorientation="horizontal",
                        textfont=dict(
                            color="white", size=14, family="system-ui, sans-serif",
                        ),
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Value: %{customdata[1]}<br>"
                            "Share: %{percent}<extra></extra>"
                        ),
                        hoverlabel=dict(
                            bgcolor=slice_colors,
                            bordercolor="white",
                            font=dict(
                                color="white", size=12,
                                family="system-ui, sans-serif",
                            ),
                        ),
                        # Pull the hovered slice slightly outward so the
                        # focus state reads cleanly (matches the Awqaf
                        # widget where the active slice nudges out).
                        pull=[0.02] * len(values),
                    )
                ]
            )
            donut_title_text = _wrap_long_title(title or "")
            donut_title_size = 16 if len(donut_title_text) > 48 else 17
            fig.update_layout(
                title=dict(
                    text=donut_title_text,
                    font=dict(size=donut_title_size, color="#0f172a", family="system-ui, sans-serif"),
                    x=0,
                    xanchor="left",
                    pad=dict(t=8, b=12),
                    automargin=True,
                ),
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#475569", size=12, family="system-ui, sans-serif"),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color="#334155"),
                    itemsizing="constant",
                ),
                annotations=[
                    dict(
                        name="center.title",
                        text="Total",
                        x=0.5, y=0.575,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=14, color="#64748b",
                                  family="system-ui, sans-serif"),
                    ),
                    dict(
                        name="center.value",
                        text=center_value_text,
                        x=0.5, y=0.44,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=36, color="#0f172a",
                                  family="system-ui, sans-serif"),
                    ),
                ],
                margin=dict(l=24, r=24, t=80, b=72),
            )
            # Capture the default center state so the frontend hover
            # handler can restore it on mouseout. Stored in a custom
            # ``meta`` field — Plotly JSON-roundtrips it untouched.
            plotly_dict = fig.to_plotly_json()
            plotly_dict.setdefault("layout", {})["meta"] = {
                "donut_center_default": {
                    "title": "Total",
                    "value": center_value_text,
                    "color": "#0f172a",
                },
            }
            return plotly_dict
        elif chart_type == ChartType.KPI:
            value = values[0] if values else 0
            label = labels[0] if labels else ""
            # KPI titles inherit the multi-series chart title which can
            # be very long ("Monthly Comparison — Total Transactions
            # vs …"). Plotly's Indicator title doesn't wrap — it just
            # clips off the right edge. Force a wrap and shrink the
            # font when it's long so the headline number stays the
            # focus.
            kpi_title_text = _wrap_long_title(title or label or "", soft_limit=44)
            kpi_title_size = 13 if len(kpi_title_text) > 44 else 15
            indicator_kwargs: dict[str, Any] = dict(
                mode="number",
                value=value,
                title={
                    "text": kpi_title_text,
                    "font": {"size": kpi_title_size, "color": "#475569"},
                    "align": "center",
                },
            )
            # Profile-driven number format (currency, percent, count, …)
            # so KPIs read like a finance/ops dashboard instead of raw
            # JSON.
            if metric_profile is not None:
                fmt = _plotly_indicator_value_format(metric_profile)
                if fmt is not None:
                    indicator_kwargs["number"] = fmt
            fig = go.Figure(data=[go.Indicator(**indicator_kwargs)])
            # Reserve enough top margin for a wrapped two-line title so
            # the number doesn't shove against the title.
            fig.update_layout(margin=dict(l=24, r=24, t=80, b=24))
            return fig.to_plotly_json()
        else:
            return {}

        # Axis title: "Month" for time charts, otherwise the field name.
        if date_x is not None:
            yn = _single_calendar_year_from_labels(labels)
            xlab_plot = "Month" if yn else "Period"
            if yn:
                xlab_plot = f"Month ({yn})"
        else:
            xlab_plot = x_key.replace("_", " ").title()

        # Y-axis: profile dictates tick format, suffix, and bounds when
        # supplied. Without a profile we fall back to the magnitude-based
        # SI heuristic — same behaviour as before.
        yaxis_kwargs: dict[str, Any] = dict(
            gridcolor="#e2e8f0",
            zeroline=False,
        )
        if metric_profile is not None:
            tickfmt = metric_profile.plotly_tickformat
            if tickfmt:
                yaxis_kwargs["tickformat"] = tickfmt
            else:
                yaxis_kwargs["separatethousands"] = True
            if metric_profile.yaxis_ticksuffix:
                yaxis_kwargs["ticksuffix"] = metric_profile.yaxis_ticksuffix
            if metric_profile.is_bounded_0_100:
                yaxis_kwargs["range"] = [0, 100]
        elif _y_axis_uses_si_prefix(values):
            yaxis_kwargs["tickformat"] = "~s"
        else:
            yaxis_kwargs["separatethousands"] = True

        # X-axis: when on a date axis, hand control to Plotly's date logic
        # (auto tick spacing across zoom levels). When on a string axis,
        # rotate labels if there are many of them.
        xaxis_kwargs: dict[str, Any] = dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#cbd5e1",
        )
        if date_x is not None:
            xaxis_kwargs["type"] = "date"
            xaxis_kwargs["tickformat"] = "%b %Y"
            xaxis_kwargs["tickangle"] = 0
            xaxis_kwargs["nticks"] = 8
            # Rangeslider + selector make multi-year series navigable
            # without dumping every month label on the user.
            span_months = (
                (date_x[-1].year - date_x[0].year) * 12
                + (date_x[-1].month - date_x[0].month)
                + 1
            )
            if span_months >= 12:
                xaxis_kwargs["rangeslider"] = dict(visible=True, thickness=0.06)
                # Pin to the bottom-left, just above the rangeslider —
                # the old top-left position collided with the chart
                # title every time.
                xaxis_kwargs["rangeselector"] = dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#f1f5f9",
                    activecolor="#0d9488",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                    font=dict(size=11, color="#334155"),
                    x=0,
                    y=-0.04,
                    xanchor="left",
                    yanchor="top",
                )
        else:
            xaxis_kwargs["tickangle"] = 0
            if len(labels) > 8:
                xaxis_kwargs["tickangle"] = -32

        # Year-separator vertical guides for multi-year time series. Drawn
        # in muted grey so they group bars/markers visually without
        # competing with the data.
        shapes: list[dict[str, Any]] = []
        if date_x is not None:
            years = sorted({d.year for d in date_x})
            if len(years) > 1:
                for y in years[1:]:
                    shapes.append(
                        dict(
                            type="line",
                            xref="x",
                            yref="paper",
                            x0=datetime(y, 1, 1),
                            x1=datetime(y, 1, 1),
                            y0=0,
                            y1=1,
                            line=dict(color="#cbd5e1", width=1, dash="dot"),
                            layer="below",
                        )
                    )

        title_text = _wrap_long_title(title or "")
        title_font_size = 16 if len(title_text) > 48 else 17
        # Single-series time charts also reserve more bottom margin
        # when the rangeslider + bottom-pinned rangeselector are in
        # play, so neither the slider nor the buttons land on top of
        # the x-axis label.
        bottom_margin = 110 if (date_x is not None) else 72
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=title_font_size, color="#0f172a", family="system-ui, sans-serif"),
                x=0,
                xanchor="left",
                pad=dict(t=8, b=12),
                automargin=True,
            ),
            xaxis_title=xlab_plot,
            yaxis_title=y_axis_label,
            template="plotly_white",
            paper_bgcolor="#eef2f7",
            plot_bgcolor="#f8fafc",
            font=dict(color="#475569", size=12, family="system-ui, sans-serif"),
            xaxis=xaxis_kwargs,
            yaxis=yaxis_kwargs,
            shapes=shapes,
            hoverlabel=dict(
                bgcolor="white",
                font=dict(family="system-ui, sans-serif", size=12, color="#0f172a"),
                bordercolor="#cbd5e1",
            ),
            margin=dict(l=64, r=28, t=70, b=bottom_margin),
        )
        return fig.to_plotly_json()


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _format_kpi_value(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


chart_service = ChartService()


# ---------------------------------------------------------------------------
# Frontend integration helper — donut hover-center behaviour
# ---------------------------------------------------------------------------
# The donut Plotly spec exposes per-slice ``customdata`` (label, formatted
# value, raw value, color) and two named annotations (``center.title`` and
# ``center.value``). A frontend can wire the Awqaf-style "hover swaps the
# centre text" behaviour with the snippet below — drop it next to the
# ``Plotly.newPlot`` call:
#
#   const meta = (figure.layout && figure.layout.meta) || {};
#   const fallback = meta.donut_center_default || {title: 'Total', value: ''};
#   const centerTitleIdx = (figure.layout.annotations || []).findIndex(
#     a => a && a.name === 'center.title'
#   );
#   const centerValueIdx = (figure.layout.annotations || []).findIndex(
#     a => a && a.name === 'center.value'
#   );
#
#   chartEl.on('plotly_hover', (ev) => {
#     const pt = ev.points && ev.points[0];
#     if (!pt || !pt.customdata) return;
#     const [label, formatted, , color] = pt.customdata;
#     Plotly.relayout(chartEl, {
#       [`annotations[${centerTitleIdx}].text`]: label,
#       [`annotations[${centerTitleIdx}].font.color`]: color,
#       [`annotations[${centerValueIdx}].text`]: formatted,
#     });
#   });
#
#   chartEl.on('plotly_unhover', () => {
#     Plotly.relayout(chartEl, {
#       [`annotations[${centerTitleIdx}].text`]: fallback.title,
#       [`annotations[${centerTitleIdx}].font.color`]: '#64748b',
#       [`annotations[${centerValueIdx}].text`]: fallback.value,
#     });
#   });
#
# Exported here so any caller (e.g. a code-splitter that bundles the JS
# alongside the chart payload) can grab it without re-deriving the
# annotation indexes.
DONUT_HOVER_CENTER_SNIPPET: str = (
    "const meta = (figure.layout && figure.layout.meta) || {};\n"
    "const fallback = meta.donut_center_default || {title:'Total', value:''};\n"
    "const ann = figure.layout.annotations || [];\n"
    "const titleIdx = ann.findIndex(a => a && a.name === 'center.title');\n"
    "const valueIdx = ann.findIndex(a => a && a.name === 'center.value');\n"
    "chartEl.on('plotly_hover', (ev) => {\n"
    "  const pt = ev.points && ev.points[0];\n"
    "  if (!pt || !pt.customdata) return;\n"
    "  const [label, formatted, , color] = pt.customdata;\n"
    "  Plotly.relayout(chartEl, {\n"
    "    [`annotations[${titleIdx}].text`]: label,\n"
    "    [`annotations[${titleIdx}].font.color`]: color,\n"
    "    [`annotations[${valueIdx}].text`]: formatted,\n"
    "  });\n"
    "});\n"
    "chartEl.on('plotly_unhover', () => {\n"
    "  Plotly.relayout(chartEl, {\n"
    "    [`annotations[${titleIdx}].text`]: fallback.title,\n"
    "    [`annotations[${titleIdx}].font.color`]: '#64748b',\n"
    "    [`annotations[${valueIdx}].text`]: fallback.value,\n"
    "  });\n"
    "});\n"
)
