"""Apache ECharts ``option`` builders — production-grade alternative to Plotly.

Why ECharts
-----------
Plotly's defaults were designed for Jupyter notebooks: scientific tick
formatters, an always-on modebar, no auto-wrap on titles or legends,
SVG-only rendering. Inside a constrained dashboard card those defaults
produce the visual bugs you saw in the screenshots — clipped titles,
legends sitting on top of bars, ``-500m`` y-axis ticks on zero data,
modebars overlapping the title.

ECharts (Apache 2.0, ~1MB gzipped) is purpose-built for embedded
dashboards. The same data renders correctly with one-tenth of the
hand-tuning:

* Titles auto-wrap and reserve their own vertical space (``overflow:
  'breakAll'`` + ``rich`` text);
* Legends are scrollable when there are too many series — never
  collide with bars;
* The toolbox (camera / zoom / restore) is opt-in and floats below the
  title, not on top of it;
* ``valueFormatter`` cleanly returns "62", "1.2K", "3.5M" without the
  ``-500m`` SI bug;
* Canvas rendering scales to 50K+ points without browser jank;
* ``dataZoom`` replaces Plotly's awkward rangeselector + rangeslider
  pair with a single, cleaner control.

Architecture
------------
This module is **pure data transformation** — same input as
``chart_service`` (rows + metric profiles) → ECharts ``option`` dict.
The dict travels in :class:`ChartPayload.echarts_option` alongside
:class:`ChartPayload.plotly_json`; the frontend prefers ECharts when
present, Plotly otherwise. That keeps the migration incremental: any
chart kind not yet ported here keeps working through Plotly.

To add a new chart kind:
1. Add a ``def build_<kind>_option(...)`` helper here.
2. Wire it into :func:`build_option` (the dispatch entry point).
3. Surface it in ``chart_service.ChartService`` so renders fill in
   ``echarts_option`` on the payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from models.enums import ChartType
from services.metric_profile import (
    MetricProfile,
    MultiSeriesProfile,
    Unit,
)


# ---------------------------------------------------------------------------
# Theme — single source of truth so every chart kind reads the same colors,
# fonts, and spacing as the rest of the app's CSS variables.
# ---------------------------------------------------------------------------
_FAMILY = "system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
_TEXT = "#0f172a"
_TEXT_MUTED = "#475569"
_AXIS = "#64748b"
_GRID = "#e2e8f0"
_BORDER = "#cbd5e1"
_BG_CARD = "#ffffff"
_BG_PLOT = "#f8fafc"
_ACCENT = "#0d9488"
_ACCENT_FILL = "rgba(13, 148, 136, 0.12)"

# Same palette as the matplotlib/Plotly side so colors stay stable
# across a session even if the renderer flips per chart kind.
_PALETTE = (
    "#0d9488", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899",
    "#10b981", "#ef4444", "#14b8a6", "#a855f7", "#f97316",
)

# Donut-specific palette — deeper, more saturated than the line palette
# because pie slices need to be individually distinguishable at a
# glance (Awqaf-style "navy / emerald / amber" leading hues).
_PIE_PALETTE = (
    "#11649e", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
    "#14b8a6", "#ef4444", "#a855f7", "#f97316", "#0ea5e9",
)


# ---------------------------------------------------------------------------
# Number formatting — JS expressions because ECharts evaluates
# ``valueFormatter`` / ``axisLabel.formatter`` in the browser.
# ---------------------------------------------------------------------------
def _smart_value_formatter_js(unit: Unit | None) -> str:
    """Return a JS function source for ``valueFormatter`` / axisLabel.

    Decisions:
    * ``CURRENCY_*`` — comma-separate, append unit label ("AED", "USD").
    * ``PERCENT``    — one decimal + "%" suffix.
    * ``COUNT``      — comma-separate when small, SI prefix when large.
                       Crucially: never SI for values < 1000, so a zero
                       series doesn't render "500m".
    * ``DURATION_*`` — append " ms" or " s".
    * Otherwise      — comma-separate integers, 1-decimal floats.

    Returned as a JS source string (frontend ``eval``s into a function).
    """
    if unit == Unit.PERCENT:
        return (
            "function(v){"
            "  if(v==null||isNaN(v))return '';"
            "  return (Math.round(v*10)/10).toFixed(1)+'%';"
            "}"
        )
    if unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR):
        suffix = {
            Unit.CURRENCY_AED: " AED",
            Unit.CURRENCY_USD: " USD",
            Unit.CURRENCY_EUR: " EUR",
        }[unit]
        return (
            "function(v){"
            "  if(v==null||isNaN(v))return '';"
            "  var a=Math.abs(v);"
            "  if(a>=1e9)return (v/1e9).toFixed(1)+'B" + suffix + "';"
            "  if(a>=1e6)return (v/1e6).toFixed(1)+'M" + suffix + "';"
            "  if(a>=1e3)return (v/1e3).toFixed(1)+'K" + suffix + "';"
            "  return v.toLocaleString()+'" + suffix + "';"
            "}"
        )
    if unit == Unit.DURATION_MS:
        return (
            "function(v){"
            "  if(v==null||isNaN(v))return '';"
            "  return v.toLocaleString()+' ms';"
            "}"
        )
    if unit == Unit.DURATION_SECONDS:
        return (
            "function(v){"
            "  if(v==null||isNaN(v))return '';"
            "  return v.toFixed(1)+' s';"
            "}"
        )
    # COUNT and unknown — smart compact: integer below 1000, SI above.
    return (
        "function(v){"
        "  if(v==null||isNaN(v))return '';"
        "  var a=Math.abs(v);"
        "  if(a>=1e9)return (v/1e9).toFixed(1)+'B';"
        "  if(a>=1e6)return (v/1e6).toFixed(1)+'M';"
        "  if(a>=1e4)return (v/1e3).toFixed(1)+'K';"
        "  return Math.round(v).toLocaleString();"
        "}"
    )


def _js(expr: str) -> dict[str, str]:
    """Wrap a JS source string so the frontend can ``new Function`` it.

    ECharts ``option`` is a JSON-serialisable dict. JS callbacks can't
    travel across JSON, so we encode them as ``{"__js__": "function(){...}"}``
    and the frontend recursively rehydrates them into real functions
    before passing the option to ``chart.setOption``. Keeps the backend
    100% serialisable.
    """
    return {"__js__": expr}


# ---------------------------------------------------------------------------
# Title — single helper so every chart kind handles long titles the same way.
# ---------------------------------------------------------------------------
def _title_block(title: str | None, subtitle: str | None = None) -> dict[str, Any]:
    """Top-left title that auto-wraps and reserves its own vertical space.

    ``overflow: 'break'`` + ``width`` makes ECharts break long titles
    onto multiple lines instead of clipping — directly fixes the
    ``…Smart App Transac`` truncation we hit with Plotly.
    """
    block: dict[str, Any] = {
        "text": (title or "").strip(),
        "left": 16,
        "top": 12,
        "textStyle": {
            "color": _TEXT,
            "fontSize": 16,
            "fontFamily": _FAMILY,
            "fontWeight": 600,
            "overflow": "break",
            "width": 600,
            "lineHeight": 22,
        },
    }
    if subtitle:
        block["subtext"] = subtitle
        block["subtextStyle"] = {
            "color": _TEXT_MUTED,
            "fontSize": 12,
            "fontFamily": _FAMILY,
            "overflow": "break",
            "width": 600,
            "lineHeight": 18,
        }
    return block


# ---------------------------------------------------------------------------
# Axis builders
# ---------------------------------------------------------------------------
def _x_axis_for_dates(_date_iso_labels: list[str]) -> dict[str, Any]:
    """Time-typed x-axis with auto-rotated tick labels and clean grid.

    ECharts auto-handles tick density (no ``MonthLocator`` gymnastics)
    and respects the underlying datetime so zooming/panning keeps the
    labels in sync. The labels argument is accepted (and unused inside
    the spec) only so callers can mirror the categorical builder's
    signature; ECharts derives ticks from the series ``[iso, value]``
    pairs themselves.
    """
    return {
        "type": "time",
        "axisLine": {"lineStyle": {"color": _BORDER}},
        "axisLabel": {
            "color": _AXIS,
            "fontFamily": _FAMILY,
            "fontSize": 11,
            "hideOverlap": True,
            "formatter": _js(
                "function(v){"
                "  var d=new Date(v);"
                "  var m=['Jan','Feb','Mar','Apr','May','Jun',"
                "         'Jul','Aug','Sep','Oct','Nov','Dec'];"
                "  return m[d.getUTCMonth()]+' '+String(d.getUTCFullYear()).slice(-2);"
                "}"
            ),
        },
        "axisTick": {"alignWithLabel": True, "lineStyle": {"color": _BORDER}},
        "splitLine": {"show": False},
    }


def _x_axis_for_categories(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "category",
        "data": labels,
        "boundaryGap": True,
        "axisLine": {"lineStyle": {"color": _BORDER}},
        "axisTick": {"alignWithLabel": True, "lineStyle": {"color": _BORDER}},
        "axisLabel": {
            "color": _AXIS,
            "fontFamily": _FAMILY,
            "fontSize": 11,
            "interval": 0 if len(labels) <= 8 else "auto",
            "rotate": 0 if len(labels) <= 8 else 30,
            "hideOverlap": True,
        },
        "splitLine": {"show": False},
    }


def _y_axis_for_profile(
    profile: MetricProfile | None,
    *,
    name: str = "",
) -> dict[str, Any]:
    """Y-axis driven by a metric profile (currency/percent/count/...)."""
    unit = profile.unit if profile is not None else None
    block: dict[str, Any] = {
        "type": "value",
        "name": name,
        "nameTextStyle": {
            "color": _AXIS,
            "fontFamily": _FAMILY,
            "fontSize": 11,
            "padding": [0, 0, 8, 0],
        },
        "axisLine": {"show": False},
        "axisTick": {"show": False},
        "splitLine": {"lineStyle": {"color": _GRID, "type": "solid"}},
        "axisLabel": {
            "color": _AXIS,
            "fontFamily": _FAMILY,
            "fontSize": 11,
            "formatter": _js(_smart_value_formatter_js(unit)),
        },
    }
    if profile is not None and profile.is_bounded_0_100:
        block["min"] = 0
        block["max"] = 100
    return block


# ---------------------------------------------------------------------------
# Tooltip — same look-and-feel across every chart kind.
# ---------------------------------------------------------------------------
def _tooltip_block(profile: MetricProfile | None = None) -> dict[str, Any]:
    return {
        "trigger": "axis",
        "axisPointer": {"type": "shadow"},
        "confine": False,  # ← KEY FIX: Allow tooltip to render outside chart container
        "appendToBody": True,  # ← KEY FIX: Render tooltip in document body, not chart div
        "backgroundColor": _BG_CARD,
        "borderColor": _BORDER,
        "borderWidth": 1,
        "padding": [8, 12],
        "textStyle": {
            "color": _TEXT,
            "fontFamily": _FAMILY,
            "fontSize": 12,
        },
        "valueFormatter": _js(
            _smart_value_formatter_js(profile.unit if profile else None)
        ),
    }


# ---------------------------------------------------------------------------
# Legend — scrollable, never overlaps the chart.
# ---------------------------------------------------------------------------
def _legend_block(names: list[str]) -> dict[str, Any]:
    """Bottom-centred horizontal legend that scrolls instead of wrapping.

    ``type: 'scroll'`` is the key fix — Plotly's horizontal legend
    overflows into the plot area when names + colors don't fit.
    ECharts shows arrows on either side and the user pages through.
    """
    return {
        "data": names,
        "type": "scroll",
        "orient": "horizontal",
        "left": "center",
        "bottom": 8,
        "icon": "roundRect",
        "itemWidth": 12,
        "itemHeight": 12,
        "itemGap": 16,
        "textStyle": {
            "color": _TEXT_MUTED,
            "fontFamily": _FAMILY,
            "fontSize": 11,
        },
        "pageIconColor": _ACCENT,
        "pageTextStyle": {"color": _TEXT_MUTED},
    }


# ---------------------------------------------------------------------------
# Toolbox — same affordances as Plotly's modebar but visually quiet.
# ---------------------------------------------------------------------------
def _toolbox_block() -> dict[str, Any]:
    return {
        "show": True,
        "right": 12,
        "top": 8,
        "itemSize": 14,
        "itemGap": 8,
        "iconStyle": {"borderColor": _AXIS},
        "emphasis": {"iconStyle": {"borderColor": _ACCENT}},
        "feature": {
            "saveAsImage": {
                "show": True,
                "title": "Save as PNG",
                "type": "png",
                "pixelRatio": 2,
                "backgroundColor": _BG_CARD,
            },
            "dataView": {
                "show": True,
                "title": "View data",
                "readOnly": True,
                "buttonColor": _ACCENT,
            },
            "restore": {"show": True, "title": "Reset"},
        },
    }


# ---------------------------------------------------------------------------
# Dimension extraction for dynamic filtering
# ---------------------------------------------------------------------------
def _extract_dimensions_from_series(
    series_data: list[dict[str, Any]],
    all_labels: list[str],
) -> dict[str, list[str]]:
    """Extract unique dimension values from series data for filtering.
    
    Analyzes series names and labels to identify filterable dimensions:
    - Years (e.g., "2024", "2025", "2026")
    - Months (e.g., "Jan", "Feb", "2026-01")
    - Metrics (e.g., "Total Transactions", "Smart App")
    - Categories (e.g., "Dubai", "Abu Dhabi")
    
    Returns a dict mapping dimension names to their unique values.
    """
    dimensions: dict[str, set[str]] = {}
    
    # Extract from series names (e.g., "Total Transactions (2024)")
    for sd in series_data:
        name = sd.get("label", "")
        
        # Check for year in parentheses: "Metric (2024)"
        import re
        year_match = re.search(r'\((\d{4})\)', name)
        if year_match:
            dimensions.setdefault("year", set()).add(year_match.group(1))
        
        # Check for year in name: "2024 Total Transactions"
        year_prefix = re.match(r'^(\d{4})\s+', name)
        if year_prefix:
            dimensions.setdefault("year", set()).add(year_prefix.group(1))
        
        # NEW: Extract metric from parentheses: "Channel (Metric Name)"
        # Matches patterns like "BANK-MOB (Funds Collected Aed)"
        metric_match = re.search(r'\(([^)]+)\)$', name)
        if metric_match:
            metric_text = metric_match.group(1)
            # Only add if it's not a year (already handled above)
            if not re.match(r'^\d{4}$', metric_text):
                dimensions.setdefault("metric", set()).add(metric_text)
    
    # Extract from labels (x-axis values)
    for label in all_labels:
        # Check for YYYY-MM format (months)
        if re.match(r'^\d{4}-\d{2}', str(label)):
            dimensions.setdefault("month", set()).add(str(label))
        
        # Check for year-only labels
        if re.match(r'^\d{4}$', str(label)):
            dimensions.setdefault("year", set()).add(str(label))
    
    # Convert sets to sorted lists
    result: dict[str, list[str]] = {}
    for dim, values in dimensions.items():
        result[dim] = sorted(list(values))
    
    return result


# ---------------------------------------------------------------------------
# Multi-series LINE + BAR — the chart that drove your screenshots.
# ---------------------------------------------------------------------------
def build_multi_series_option(
    chart_type: ChartType,
    *,
    title: str | None,
    series_data: list[dict[str, Any]],
    multi: MultiSeriesProfile,
    has_time: bool,
    all_labels: list[str],
    date_iso_labels: list[str] | None,
) -> dict[str, Any]:
    """Build an ECharts option for a multi-series LINE or BAR chart.

    ``series_data`` is a list of ``{name, color, points: {label: value}}``
    — same shape ``chart_service._SeriesTrace`` already produces, just
    serialised to dicts so this module stays free of internal types.
    """
    # Build series. For time-typed x we emit ``[isoDate, value]`` pairs so
    # ECharts handles missing months as gaps (matches Plotly behaviour).
    chart_kind = "line" if chart_type == ChartType.LINE else "bar"
    series: list[dict[str, Any]] = []
    
    # Extract dimension metadata for filtering
    import re
    
    for idx, sd in enumerate(series_data):
        on_secondary = bool(
            multi.dual_axis
            and sd.get("is_bounded_0_100")
        )
        data_points: list[Any] = []
        if has_time and date_iso_labels is not None:
            for lab, iso in zip(all_labels, date_iso_labels, strict=True):
                v = sd["points"].get(lab)
                # ECharts treats null as a gap (line breaks instead of
                # dragging to zero) — same honest behaviour as Plotly.
                data_points.append([iso, v if v is not None else None])
        else:
            for lab in all_labels:
                v = sd["points"].get(lab)
                data_points.append(v if v is not None else None)

        # Extract dimensions from series name for filtering
        name = sd["label"]
        series_dims: dict[str, str] = {"index": str(idx)}
        
        # Extract year from name (e.g., "Metric (2024)" or "2024 Metric")
        year_match = re.search(r'\((\d{4})\)', name)
        if year_match:
            series_dims["year"] = year_match.group(1)
        else:
            year_prefix = re.match(r'^(\d{4})\s+', name)
            if year_prefix:
                series_dims["year"] = year_prefix.group(1)
        
        # Extract metric from parentheses: "Channel (Metric Name)"
        # This handles multi-metric + multi-dimension cases like:
        # "BANK-MOB (Funds Collected Aed)" → metric = "Funds Collected Aed"
        metric_in_parens = re.search(r'\(([^)]+)\)$', name)
        if metric_in_parens:
            potential_metric = metric_in_parens.group(1)
            # Only use it as metric if it's not a year (already handled above)
            if not re.match(r'^\d{4}$', potential_metric):
                series_dims["metric"] = potential_metric
        else:
            # Fallback: use the full name cleaned of year suffixes
            metric_name = re.sub(r'\s*\(\d{4}\)', '', name).strip()
            metric_name = re.sub(r'^\d{4}\s+', '', metric_name).strip()
            if metric_name:
                series_dims["metric"] = metric_name

        s: dict[str, Any] = {
            "name": sd["label"],
            "type": chart_kind,
            "yAxisIndex": 1 if on_secondary else 0,
            "data": data_points,
            "itemStyle": {"color": sd["color"]},
            "emphasis": {"focus": "series", "blurScope": "coordinateSystem"},
            "_dimensions": series_dims,  # Dimension tags for filtering
        }
        if chart_kind == "line":
            s["smooth"] = False
            s["symbol"] = "circle"
            s["symbolSize"] = 7
            s["lineStyle"] = {"width": 2.5, "color": sd["color"]}
            # Subtle area fill only when there's a single non-bounded
            # series — too many overlapping fills would muddy the chart.
            if len(series_data) == 1:
                s["areaStyle"] = {"color": _ACCENT_FILL}
        else:
            s["barMaxWidth"] = 28
            s["itemStyle"] = {
                "color": sd["color"],
                "borderRadius": [4, 4, 0, 0],
            }
        series.append(s)

    # X-axis
    x_axis = (
        _x_axis_for_dates(date_iso_labels)
        if (has_time and date_iso_labels is not None)
        else _x_axis_for_categories(all_labels)
    )

    # Y-axes — single, or dual when units mix (% alongside AED).
    primary_y = _y_axis_for_profile(
        next(
            (sp.profile for sp in multi.series if not sp.profile.is_bounded_0_100),
            multi.series[0].profile if multi.series else None,
        ),
        name=multi.primary_axis_label or "",
    )
    primary_y["nameGap"] = 40

    y_axes: list[dict[str, Any]] = [primary_y]
    if multi.dual_axis:
        secondary_y = _y_axis_for_profile(
            next(
                (sp.profile for sp in multi.series if sp.profile.is_bounded_0_100),
                multi.series[0].profile,
            ),
            name=multi.secondary_axis_label or "",
        )
        secondary_y["splitLine"] = {"show": False}
        secondary_y["position"] = "right"
        secondary_y["nameGap"] = 40
        y_axes.append(secondary_y)

    # DataZoom — replaces Plotly's rangeselector + rangeslider with a
    # single slider only for time charts that span 12+ months.
    data_zoom: list[dict[str, Any]] = []
    if has_time and date_iso_labels and len(date_iso_labels) >= 12:
        data_zoom = [
            {
                "type": "slider",
                "show": True,
                "xAxisIndex": 0,
                "height": 20,
                "bottom": 56,
                "borderColor": _BORDER,
                "fillerColor": "rgba(13, 148, 136, 0.10)",
                "handleStyle": {"color": _ACCENT},
                "moveHandleStyle": {"color": _ACCENT},
                "selectedDataBackground": {
                    "lineStyle": {"color": _ACCENT, "width": 1},
                    "areaStyle": {"color": _ACCENT_FILL},
                },
                "dataBackground": {
                    "lineStyle": {"color": _BORDER, "width": 1},
                    "areaStyle": {"color": "rgba(0,0,0,0.04)"},
                },
                "textStyle": {"color": _AXIS, "fontFamily": _FAMILY, "fontSize": 10},
            },
            {"type": "inside", "xAxisIndex": 0},
        ]

    # Grid — leaves room for the dataZoom slider AND the legend without
    # ever letting them collide with the bars.
    # Dynamic top spacing based on title length to prevent overlap
    grid_bottom = 110 if data_zoom else 64
    title_len = len(title or "")
    title_lines = max(1, (title_len // 50) + 1)  # Estimate wrapped lines
    grid_top = max(100, 70 + (title_lines * 24))  # 24px per line + base spacing
    
    # Extract available dimensions for dynamic filtering
    dimensions = _extract_dimensions_from_series(series_data, all_labels)
    
    # Build dimension values map for frontend filtering
    dimension_values: dict[str, list[str]] = {}
    dimension_roles: dict[str, str] = {}
    
    # Collect unique values for each dimension across all series
    for s in series:
        s_dims = s.get("_dimensions", {})
        for dim_key, dim_val in s_dims.items():
            if dim_key != "index":  # Skip internal index
                if dim_key not in dimension_values:
                    dimension_values[dim_key] = []
                if dim_val not in dimension_values[dim_key]:
                    dimension_values[dim_key].append(dim_val)
    
    # Add extracted dimensions from labels (years, months)
    for dim_key, dim_vals in dimensions.items():
        if dim_key not in dimension_values:
            dimension_values[dim_key] = dim_vals
        else:
            # Merge and deduplicate
            for val in dim_vals:
                if val not in dimension_values[dim_key]:
                    dimension_values[dim_key].append(val)
    
    # Sort dimension values
    for dim_key in dimension_values:
        dimension_values[dim_key] = sorted(dimension_values[dim_key])
    
    # Determine dimension roles (X_AXIS, SERIES, or FILTER)
    # This tells the frontend which dimensions should have dropdowns
    
    # Month/time on x-axis → role is X_AXIS (no dropdown needed)
    if has_time and date_iso_labels:
        dimension_roles["month"] = "X_AXIS"
    
    # Metrics in series → role is SERIES (show dropdown if 2+)
    if len(series) >= 2:
        # Check if series have different metrics
        metrics_in_series = set()
        for s in series:
            s_dims = s.get("_dimensions", {})
            if "metric" in s_dims:
                metrics_in_series.add(s_dims["metric"])
        if len(metrics_in_series) >= 2:
            dimension_roles["metric"] = "SERIES"
    
    # Years in series names → role is SERIES (show dropdown if 2+)
    if "year" in dimension_values and len(dimension_values["year"]) >= 2:
        # Check if years are in series (not x-axis)
        years_in_series = any("year" in s.get("_dimensions", {}) for s in series)
        if years_in_series:
            dimension_roles["year"] = "SERIES"
    
    option: dict[str, Any] = {
        "title": _title_block(title),
        "tooltip": _tooltip_block(
            next((sp.profile for sp in multi.series), None)
        ),
        "legend": _legend_block([s["name"] for s in series]),
        "toolbox": _toolbox_block(),
        "grid": {
            "left": 56,
            "right": 24 if not multi.dual_axis else 56,
            "top": grid_top,
            "bottom": grid_bottom,
            "containLabel": True,
        },
        "xAxis": x_axis,
        "yAxis": y_axes,
        "series": series,
        "dataZoom": data_zoom,
        "color": [s["itemStyle"]["color"] for s in series],
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": _FAMILY, "color": _TEXT_MUTED},
        "animationDuration": 400,
        # Side-channel metadata the frontend uses for the metric picker.
        # Mirrored from Plotly side so the existing dropdown logic
        # works identically.
        "_metricNames": [s["name"] for s in series],
        # NEW: Dimension metadata for dynamic filtering
        "_dimensions": dimension_values,
        "_dimension_roles": dimension_roles,
    }
    return option


# ---------------------------------------------------------------------------
# Single-series LINE + BAR
# ---------------------------------------------------------------------------
def build_single_series_option(
    chart_type: ChartType,
    *,
    title: str | None,
    labels: list[str],
    values: list[float],
    has_time: bool,
    date_iso_labels: list[str] | None,
    profile: MetricProfile | None,
    y_axis_label: str,
) -> dict[str, Any]:
    """Build an ECharts option for a single-series LINE or BAR chart."""
    chart_kind = "line" if chart_type == ChartType.LINE else "bar"

    if has_time and date_iso_labels is not None:
        data_points = [
            [iso, v if v is not None else None]
            for iso, v in zip(date_iso_labels, values, strict=True)
        ]
    else:
        data_points = list(values)

    series_block: dict[str, Any] = {
        "name": y_axis_label,
        "type": chart_kind,
        "data": data_points,
        "itemStyle": {"color": _ACCENT, "borderRadius": [4, 4, 0, 0]},
        "emphasis": {"focus": "series"},
    }
    if chart_kind == "line":
        series_block["smooth"] = False
        series_block["symbol"] = "circle"
        series_block["symbolSize"] = 8
        series_block["lineStyle"] = {"width": 3, "color": _ACCENT}
        series_block["areaStyle"] = {"color": _ACCENT_FILL}
    else:
        series_block["barMaxWidth"] = 36

    x_axis = (
        _x_axis_for_dates(date_iso_labels)
        if (has_time and date_iso_labels is not None)
        else _x_axis_for_categories(labels)
    )
    y_axis = _y_axis_for_profile(profile, name=y_axis_label)

    # Ensure Y-axis label spacing (critical fix)
    if isinstance(y_axis, dict):
        y_axis["nameGap"] = 40

    data_zoom: list[dict[str, Any]] = []
    if has_time and date_iso_labels and len(date_iso_labels) >= 12:
        data_zoom = [
            {
                "type": "slider",
                "show": True,
                "xAxisIndex": 0,
                "height": 20,
                "bottom": 28,
                "borderColor": _BORDER,
                "fillerColor": "rgba(13, 148, 136, 0.10)",
                "handleStyle": {"color": _ACCENT},
                "moveHandleStyle": {"color": _ACCENT},
            },
            {"type": "inside", "xAxisIndex": 0},
        ]
    grid_bottom = 80 if data_zoom else 56

    # -------------------------------
    # Dynamic spacing to prevent title/y-axis overlap
    # -------------------------------
    title_len = len(title or "")
    title_lines = max(1, (title_len // 50) + 1)  # Estimate wrapped lines
    grid_top = max(100, 70 + (title_lines * 24))  # 24px per line + base spacing

    return {
        "title": _title_block(title),
        "tooltip": _tooltip_block(profile),
        "toolbox": _toolbox_block(),
        "grid": {
            "left": 56,
            "right": 32,
            "top": grid_top,
            "bottom": grid_bottom,
            "containLabel": True,
        },
        "xAxis": x_axis,
        "yAxis": y_axis,
        "series": [series_block],
        "dataZoom": data_zoom,
        "color": [_ACCENT],
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": _FAMILY, "color": _TEXT_MUTED},
        "animationDuration": 400,
    }


# ---------------------------------------------------------------------------
# Donut (PIE) — with center "Total" label and hover-swap behaviour
# ---------------------------------------------------------------------------
def build_donut_option(
    *,
    title: str | None,
    labels: list[str],
    values: list[float],
    profile: MetricProfile | None,
) -> dict[str, Any]:
    """Build an ECharts donut option with Awqaf-style center number.

    The center "Total" + value uses ``graphic`` elements (not chart
    annotations) so the frontend can mutate them on hover via
    ``setOption({graphic:[...]})``.
    """
    palette = list(_PIE_PALETTE)
    colored: list[dict[str, Any]] = []
    total = 0.0
    for i, (lab, val) in enumerate(zip(labels, values, strict=True)):
        v = float(val) if val is not None else 0.0
        total += v
        colored.append({
            "name": lab,
            "value": v,
            "itemStyle": {"color": palette[i % len(palette)]},
        })

    fmt_js = _smart_value_formatter_js(profile.unit if profile else None)
    center_value_text = total  # frontend formats via _value_formatter

    return {
        "title": _title_block(title),
        "tooltip": {
            "trigger": "item",
            "backgroundColor": _BG_CARD,
            "borderColor": _BORDER,
            "borderWidth": 1,
            "padding": [8, 12],
            "textStyle": {
                "color": _TEXT,
                "fontFamily": _FAMILY,
                "fontSize": 12,
            },
            "formatter": _js(
                "function(p){"
                "  var fmt=" + fmt_js + ";"
                "  return '<b>'+p.name+'</b><br/>'"
                "    +fmt(p.value)+' &middot; '"
                "    +p.percent.toFixed(1)+'%';"
                "}"
            ),
        },
        "legend": _legend_block(labels),
        "toolbox": _toolbox_block(),
        "color": palette,
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": _FAMILY, "color": _TEXT_MUTED},
        "graphic": [
            # Center label (mutable by frontend on hover).
            {
                "id": "centerTitle",
                "type": "text",
                "left": "center",
                "top": "47%",
                "style": {
                    "text": "Total",
                    "fontSize": 13,
                    "fontFamily": _FAMILY,
                    "fill": _TEXT_MUTED,
                    "textAlign": "center",
                },
            },
            {
                "id": "centerValue",
                "type": "text",
                "left": "center",
                "top": "53%",
                "style": {
                    "text": "",  # filled in by frontend (uses fmt fn)
                    "fontSize": 28,
                    "fontFamily": _FAMILY,
                    "fontWeight": 700,
                    "fill": _TEXT,
                    "textAlign": "center",
                },
            },
        ],
        "series": [
            {
                "type": "pie",
                "radius": ["52%", "72%"],
                "center": ["50%", "52%"],
                "avoidLabelOverlap": True,
                "padAngle": 2,
                "itemStyle": {
                    "borderRadius": 4,
                    "borderColor": _BG_CARD,
                    "borderWidth": 2,
                },
                "label": {
                    "show": True,
                    "position": "outside",
                    "formatter": "{b}\n{d}%",
                    "color": _TEXT_MUTED,
                    "fontFamily": _FAMILY,
                    "fontSize": 11,
                },
                "labelLine": {"show": True, "smooth": True, "lineStyle": {"color": _BORDER}},
                "emphasis": {
                    "scale": True,
                    "scaleSize": 6,
                    "label": {"fontSize": 12, "fontWeight": 600, "color": _TEXT},
                },
                "data": colored,
            }
        ],
        # Side-channel for the frontend hover handler — gives it the
        # JS formatter and the "default center" text so it can restore
        # on mouseout.
        "_donut": {
            "centerDefault": {
                "title": "Total",
                "value": center_value_text,
            },
            "valueFormatter": _js(fmt_js),
        },
    }


# ---------------------------------------------------------------------------
# KPI — large headline number with a small subtitle.
# ---------------------------------------------------------------------------
def build_kpi_option(
    *,
    title: str | None,
    label: str,
    value: float,
    profile: MetricProfile | None,
) -> dict[str, Any]:
    """A KPI is just a centred ``graphic.text`` — no axis or series.

    Plotly's ``Indicator`` doesn't wrap titles and overflowed both
    edges of the card in the screenshots. Here we fully control
    typography and use ECharts ``graphic`` so the title wraps and the
    value sits underneath at a fixed proportion.
    """
    fmt_js = _smart_value_formatter_js(profile.unit if profile else None)
    subtitle = "" if (label or "").strip() in {"", "(all)"} else label
    return {
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": _FAMILY, "color": _TEXT_MUTED},
        "graphic": [
            {
                "type": "text",
                "left": "center",
                "top": "20%",
                "style": {
                    "text": (title or "").strip(),
                    "fontSize": 13,
                    "fontFamily": _FAMILY,
                    "fill": _TEXT_MUTED,
                    "textAlign": "center",
                    "width": 360,
                    "overflow": "break",
                    "lineHeight": 18,
                },
            },
            {
                "id": "kpiValue",
                "type": "text",
                "left": "center",
                "top": "44%",
                "style": {
                    "text": "",  # frontend writes formatted value here
                    "fontSize": 56,
                    "fontFamily": _FAMILY,
                    "fontWeight": 700,
                    "fill": _TEXT,
                    "textAlign": "center",
                },
            },
            {
                "type": "text",
                "left": "center",
                "top": "70%",
                "style": {
                    "text": subtitle,
                    "fontSize": 13,
                    "fontFamily": _FAMILY,
                    "fill": _TEXT_MUTED,
                    "textAlign": "center",
                },
            },
        ],
        # Side-channel: numeric value + JS formatter, frontend hydrates
        # the ``kpiValue`` graphic by formatting the value once.
        "_kpi": {
            "value": float(value),
            "valueFormatter": _js(fmt_js),
        },
    }


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def to_iso_labels(date_x: list[datetime] | None) -> list[str] | None:
    """Convert datetime list → ISO strings ECharts ``time`` axis can parse."""
    if date_x is None:
        return None
    return [d.strftime("%Y-%m-%dT00:00:00") for d in date_x]
