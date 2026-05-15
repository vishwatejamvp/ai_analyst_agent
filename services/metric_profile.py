"""Metric display profile — single source of truth for how a metric should
be **named, labelled, and formatted** in any visualization or narrative.

Why this module exists
----------------------
Before this module, three layers each made independent decisions about a
metric:

* ``analyst_service._chart_title`` built titles like
  ``"SUM of revenues_collected_aed by month"`` from raw spec components.
* ``analyst_service._metric_display_title`` produced y-axis labels by
  title-casing the column name → ``"Revenues Collected Aed"``.
* ``chart_service`` decided independently whether to use SI tick formatting,
  using a value-magnitude heuristic.

Each layer needed its own patch every time a new unit (``_aed``, ``_pct``,
``_kwh``, …) showed up. This module replaces all three with a typed
:class:`MetricProfile` derived once from the metric name + glossary +
operation, and consumed by every renderer.

Add a new unit
--------------
Drop a new entry in :data:`_UNIT_RULES`. All charts (Plotly + matplotlib),
all titles, all axis labels, all hover tooltips pick it up automatically.
That is the whole point: **no per-metric branching anywhere downstream.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from models.enums import TimeBucket


# ---------------------------------------------------------------------------
# Unit taxonomy
# ---------------------------------------------------------------------------
class Unit(str, Enum):
    """Semantic unit of a metric.

    Drives axis labels, tick formatters, summability, and y-range bounding.
    Add new variants only when a downstream renderer must do something
    *different* — purely cosmetic differences belong in ``unit_label``.
    """

    CURRENCY_AED = "currency_aed"
    CURRENCY_USD = "currency_usd"
    CURRENCY_EUR = "currency_eur"
    PERCENT = "percent"          # 0-100 scale; never sum
    RATE = "rate"                # generic rate; never sum
    RATIO = "ratio"              # per-unit ratio; never sum
    COUNT = "count"              # integer-ish; sum is meaningful
    DURATION_MS = "duration_ms"
    DURATION_SECONDS = "duration_seconds"
    NONE = "none"


# Ordering matters: longest / most specific suffix first so ``_aed`` wins
# over a generic catch-all and ``_per_capita`` wins over ``_capita``.
_UNIT_RULES: tuple[tuple[str, Unit], ...] = (
    ("_per_capita", Unit.RATIO),
    ("_per_user", Unit.RATIO),
    ("_per_request", Unit.RATIO),
    ("_per_pilgrim", Unit.RATIO),
    ("_milliseconds", Unit.DURATION_MS),
    ("_seconds", Unit.DURATION_SECONDS),
    ("_minutes", Unit.DURATION_SECONDS),
    ("_aed", Unit.CURRENCY_AED),
    ("_usd", Unit.CURRENCY_USD),
    ("_eur", Unit.CURRENCY_EUR),
    ("_pct", Unit.PERCENT),
    ("_percent", Unit.PERCENT),
    ("_percentage", Unit.PERCENT),
    ("_rate", Unit.RATE),
    ("_count", Unit.COUNT),
)

# Tokens that mark a metric as count-ish even without a ``_count`` suffix.
# Used as a fallback when no rule matched.
_COUNT_TOKENS: frozenset[str] = frozenset(
    {
        "count", "transactions", "registrations", "requests",
        "recipients", "donations", "fatwas", "pilgrims", "applications",
        "permits", "campaigns", "members", "students", "downloads",
        "visits", "logins", "sessions", "events",
    }
)

# Acronyms that should keep their canonical capitalization in display
# names (otherwise ``UAE`` becomes ``Uae``). Add freely.
_ACRONYMS: frozenset[str] = frozenset(
    {"AED", "USD", "EUR", "GBP", "UAE", "AI", "ML", "KPI", "SLA",
     "ID", "URL", "API", "CSV", "PDF", "GCC"}
)

# Small words kept lowercase in title case (matches the existing
# ``_humanize_dataset`` convention so titles read naturally).
_SMALL_WORDS: frozenset[str] = frozenset(
    {"and", "or", "of", "for", "in", "the", "to", "a", "an", "by", "with",
     "from", "on", "at", "per"}
)

_OP_WORDS: dict[str, str] = {
    "sum": "Total",
    "avg": "Average",
    "average": "Average",
    "mean": "Average",
    "count": "Total",
    "min": "Minimum",
    "max": "Maximum",
}

# First-token aliases that signal "the metric name already carries this
# op-word, don't repeat it in the title". E.g. ``total_transactions``
# under SUM should render as "Total Transactions", not "Total Total
# Transactions". Keys are the normalized operation, values are first
# tokens of the metric display name that we treat as equivalent.
_OP_FIRST_TOKEN_ALIASES: dict[str, frozenset[str]] = {
    "sum": frozenset({"total", "sum"}),
    "count": frozenset({"total", "count"}),
    "avg": frozenset({"average", "avg", "mean"}),
    "average": frozenset({"average", "avg", "mean"}),
    "mean": frozenset({"average", "avg", "mean"}),
    "min": frozenset({"minimum", "min", "lowest"}),
    "max": frozenset({"maximum", "max", "highest", "peak"}),
}

_BUCKET_ADVERB: dict[TimeBucket, str] = {
    TimeBucket.DAY: "Daily",
    TimeBucket.WEEK: "Weekly",
    TimeBucket.MONTH: "Monthly",
    TimeBucket.QUARTER: "Quarterly",
    TimeBucket.YEAR: "Yearly",
}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricProfile:
    """Everything every renderer / narrator needs to talk about one metric.

    The profile is **renderer-agnostic**: ``chart_service`` reads it,
    ``analyst_service`` reads it, future report exports could read it.
    All formatting rules live here so a unit added once propagates
    everywhere.
    """

    raw_metric: str | None
    display_name: str             # e.g. "Revenues Collected"
    unit: Unit
    unit_label: str               # e.g. "AED", "%", "ms" or ""
    operation: str                # "sum" / "avg" / ...
    op_word: str                  # "Total" / "Average" / ...
    is_summable: bool
    is_bounded_0_100: bool

    # Rendering hints
    plotly_tickformat: str | None
    plotly_hover_value_format: str
    matplotlib_value_formatter: Callable[[float, int], str]
    axis_title: str               # e.g. "Revenues Collected (AED)"
    yaxis_ticksuffix: str         # e.g. "%" for percent metrics

    # Summary cards (KPI) need a compact value formatter
    kpi_value_formatter: Callable[[float], str] = field(repr=False)


# ---------------------------------------------------------------------------
# Multi-series support
# ---------------------------------------------------------------------------

# Categorical palette used when a chart shows multiple series. Picked for
# (a) good separation in light *and* dark UI, (b) colour-blind safety
# (no red/green clash), (c) calm hues that don't out-shout the title.
# Add freely — the renderer cycles through and never indexes by name.
SERIES_PALETTE: tuple[str, ...] = (
    "#0d9488",  # teal     — primary, matches single-series accent
    "#3b82f6",  # blue
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#10b981",  # emerald
    "#ef4444",  # red
    "#14b8a6",  # cyan
    "#a855f7",  # purple
    "#f97316",  # orange
)


@dataclass(frozen=True)
class SeriesProfile:
    """One series in a multi-series chart.

    A series is a (label, profile, color) triple. ``key`` is the raw
    series identifier as it appears in the rows (e.g. the metric name
    for compared-metrics, or the group value for ``group_by`` series);
    ``label`` is the human display name shown in the legend.
    """

    key: str
    label: str
    profile: MetricProfile
    color: str


@dataclass(frozen=True)
class MultiSeriesProfile:
    """Collection of :class:`SeriesProfile` plus chart-level decisions.

    ``dual_axis`` is True when at least one series is bounded (percent /
    rate) and at least one is not — drawing them on a single y-axis
    would either flatten the percent line to invisibility or stretch
    the absolute line off the chart. The renderer maps the bounded
    series to ``yaxis2`` (right side) and the absolute series to the
    primary axis.

    ``shared_unit`` carries the y-axis label when every series shares
    a unit (e.g. all currencies in AED) — in that case there's no
    ambiguity and the legend names alone communicate which series is
    which.
    """

    series: tuple[SeriesProfile, ...]
    dual_axis: bool
    primary_axis_label: str
    secondary_axis_label: str | None  # populated only when dual_axis

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.series)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify(
    metric: str | None,
    *,
    operation: str = "sum",
    glossary_term: str | None = None,
) -> MetricProfile:
    """Build a :class:`MetricProfile` for ``metric`` under ``operation``.

    ``glossary_term`` (when present) overrides the inferred display name
    so curated definitions win over heuristics. ``operation`` decides the
    op-word ("Total" / "Average" / "Maximum" / …) and is used by the
    title composer.
    """
    metric = (metric or "").strip()
    op_norm = (operation or "sum").lower()

    unit, unit_label = _detect_unit(metric)
    base_name = _strip_unit_suffix(metric, unit)
    display_name = (
        glossary_term.strip()
        if glossary_term
        else _humanize_metric_name(base_name) or "Records"
    )

    is_summable = unit not in (Unit.PERCENT, Unit.RATE, Unit.RATIO)
    is_bounded_0_100 = unit == Unit.PERCENT
    op_word = _OP_WORDS.get(op_norm, op_norm.title())

    axis_title = display_name if not unit_label else f"{display_name} ({unit_label})"
    yaxis_ticksuffix = "%" if unit == Unit.PERCENT else ""

    plotly_tickformat = _plotly_tickformat(unit)
    plotly_hover_value_format = _plotly_hover_format(unit)
    matplotlib_value_formatter = _matplotlib_formatter(unit)
    kpi_value_formatter = _kpi_formatter(unit)

    return MetricProfile(
        raw_metric=metric or None,
        display_name=display_name,
        unit=unit,
        unit_label=unit_label,
        operation=op_norm,
        op_word=op_word,
        is_summable=is_summable,
        is_bounded_0_100=is_bounded_0_100,
        plotly_tickformat=plotly_tickformat,
        plotly_hover_value_format=plotly_hover_value_format,
        matplotlib_value_formatter=matplotlib_value_formatter,
        axis_title=axis_title,
        yaxis_ticksuffix=yaxis_ticksuffix,
        kpi_value_formatter=kpi_value_formatter,
    )


def compose_chart_title(
    profile: MetricProfile,
    *,
    time_bucket: TimeBucket | None = None,
    group_by: str | None = None,
) -> str:
    """Produce a clean, business-readable chart title from ``profile``.

    Layout rules (compact, dataset name lives in the trust panel):

    *  Time + group_by  →  ``Monthly Total Revenues Collected (AED) by Emirate``
    *  Time only        →  ``Monthly Total Revenues Collected (AED)``
    *  Group_by only    →  ``Total Revenues Collected (AED) by Emirate``
    *  KPI / no time    →  ``Total Revenues Collected (AED)``

    For percent / rate metrics under "sum" the op-word is silently
    swapped to "Average" — summing percentages is a category error and
    a misleading title would just hide it.
    """
    operation = profile.operation
    op_word = profile.op_word
    if not profile.is_summable and operation == "sum":
        op_word = "Average"
        operation = "avg"

    # Don't double up: "Total" + "Total Transactions" should render as
    # "Total Transactions", not "Total Total Transactions". Same for
    # "Avg Response (s)" under AVG.
    show_op_word = not _display_name_carries_op(profile.display_name, operation)

    head_parts: list[str] = []
    if time_bucket is not None:
        head_parts.append(_BUCKET_ADVERB.get(time_bucket, ""))
    if show_op_word:
        head_parts.append(op_word)
    head_parts.append(profile.display_name)
    head = " ".join(p for p in head_parts if p)

    if profile.unit_label:
        head = f"{head} ({profile.unit_label})"

    if group_by:
        return f"{head} by {_humanize_metric_name(group_by)}"
    return head


def classify_series(
    pairs: list[tuple[str, str | None, str]],
    *,
    operation: str = "sum",
    glossary_terms: dict[str, str] | None = None,
) -> MultiSeriesProfile:
    """Build a :class:`MultiSeriesProfile` from raw (key, metric, label) tuples.

    Each tuple is ``(series_key, metric_field, display_label)``:

    *  ``series_key``    — value found in the row's ``series`` field;
                            cycled through :data:`SERIES_PALETTE` for color.
    *  ``metric_field``  — column the series represents. For
                            compared-metrics the key *is* the metric, so
                            both arguments are equal. For ``group_by``
                            series the metric is the underlying numeric
                            column (e.g. ``revenues_collected_aed``)
                            shared by all groups.
    *  ``display_label`` — what the legend shows (already humanized).

    The dual-axis decision is unit-driven: any mix of bounded (percent /
    rate / ratio) with absolute (currency / count) triggers the second
    axis. Same units across the board → single axis with a shared label.
    """
    glossary_terms = glossary_terms or {}
    series: list[SeriesProfile] = []
    for idx, (key, metric, label) in enumerate(pairs):
        prof = classify(
            metric,
            operation=operation,
            glossary_term=glossary_terms.get(key),
        )
        series.append(
            SeriesProfile(
                key=key,
                label=label or prof.display_name,
                profile=prof,
                color=SERIES_PALETTE[idx % len(SERIES_PALETTE)],
            )
        )

    bounded = [s for s in series if s.profile.is_bounded_0_100]
    absolute = [s for s in series if not s.profile.is_bounded_0_100]
    dual_axis = bool(bounded and absolute)

    if dual_axis:
        primary = absolute[0].profile.axis_title
        secondary = bounded[0].profile.axis_title
    else:
        units = {s.profile.axis_title for s in series}
        if len(units) == 1:
            primary = next(iter(units))
        else:
            # Mixed absolute units (rare: AED + count) — drop the
            # ambiguous parenthesis and let the legend disambiguate.
            primary = "Value"
        secondary = None

    return MultiSeriesProfile(
        series=tuple(series),
        dual_axis=dual_axis,
        primary_axis_label=primary,
        secondary_axis_label=secondary,
    )


def compose_multi_chart_title(
    multi: MultiSeriesProfile,
    *,
    time_bucket: TimeBucket | None = None,
    group_by_label: str | None = None,
) -> str:
    """Title for a multi-series chart.

    Two flavours:

    *  ``group_by_label`` set  →  data is one numeric column split across
        the values of ``group_by`` (e.g. occupancy by emirate). The
        title reads ``Monthly Average Occupancy Rate (%) by Emirate``.
    *  ``group_by_label`` unset → series are different metrics being
        compared head-to-head. The title is generic
        (``Comparison — Monthly``) since each metric carries its own
        unit; the legend communicates which line is which.
    """
    bucket_word = _BUCKET_ADVERB.get(time_bucket, "") if time_bucket else ""

    if group_by_label:
        # Single metric, split across groups → use the first profile's
        # title and append "by <group>".
        head_profile = multi.series[0].profile
        return compose_chart_title(
            head_profile,
            time_bucket=time_bucket,
            group_by=group_by_label,
        )

    # Multi-metric comparison: no single profile owns the title.
    parts = [bucket_word] if bucket_word else []
    parts.append("Comparison")
    head = " ".join(p for p in parts if p)
    if not multi.series:
        return head
    # Listing every metric in the title (``A vs B vs C vs D``) overflows
    # the chart card the moment 3+ series get involved — and the legend
    # below already names every series. Cap at two; collapse the rest
    # into a count so the title stays a single line on a phone-width
    # card.
    names = [s.label for s in multi.series]
    if len(names) <= 2:
        listing = " vs ".join(names)
    else:
        listing = f"{len(names)} metrics"
    return f"{head} — {listing}"


# ---------------------------------------------------------------------------
# Internals — unit detection, naming
# ---------------------------------------------------------------------------
def _display_name_carries_op(display_name: str, operation: str) -> bool:
    """Return True iff ``display_name`` already starts with the op-word.

    Used by :func:`compose_chart_title` to suppress duplicated leading
    words like "Total Total Transactions" or "Avg Average Response".
    """
    if not display_name:
        return False
    first_token = display_name.strip().split(maxsplit=1)[0].lower()
    aliases = _OP_FIRST_TOKEN_ALIASES.get((operation or "").lower(), frozenset())
    return first_token in aliases


def _detect_unit(metric: str) -> tuple[Unit, str]:
    """Return ``(Unit, unit_label)`` for ``metric``.

    Matching is suffix-first so ``revenues_collected_aed`` resolves to
    ``CURRENCY_AED`` even though the bulk of the name is generic. Falls
    back to a token scan to catch count-style names without a ``_count``
    suffix (e.g. ``total_transactions``).
    """
    m = metric.lower()
    for suffix, unit in _UNIT_RULES:
        if m.endswith(suffix):
            return unit, _unit_label_for(unit)

    tokens = set(re.split(r"[_\s\-]+", m))
    if tokens & _COUNT_TOKENS:
        return Unit.COUNT, ""
    return Unit.NONE, ""


def _unit_label_for(unit: Unit) -> str:
    return {
        Unit.CURRENCY_AED: "AED",
        Unit.CURRENCY_USD: "USD",
        Unit.CURRENCY_EUR: "EUR",
        Unit.PERCENT: "%",
        Unit.RATE: "",
        Unit.RATIO: "",
        Unit.COUNT: "",
        Unit.DURATION_MS: "ms",
        Unit.DURATION_SECONDS: "s",
        Unit.NONE: "",
    }[unit]


def _strip_unit_suffix(metric: str, unit: Unit) -> str:
    """Strip the unit suffix from a metric name for display purposes.

    Currency / percent / duration suffixes are pure type tags
    (``_aed`` / ``_pct`` / ``_seconds``) and stripping them gives a clean
    noun. ``_count`` is special: it's both a type tag *and* the noun
    itself ("campaign count", "uae pilgrim count"), so we keep it — the
    op-word + count read naturally ("Total Campaign Count") and avoid
    awkward singularizations like "Total Campaign".
    """
    if not metric:
        return ""
    m = metric.lower()
    for suffix, u in _UNIT_RULES:
        if u == unit and m.endswith(suffix):
            if suffix == "_count":
                return metric  # keep the noun
            return metric[: -len(suffix)]
    return metric


def _humanize_metric_name(raw: str) -> str:
    """``revenues_collected`` → ``Revenues Collected``.

    Title-cases each token, keeps small words lowercase except at the
    start, and preserves known acronyms (``UAE``, ``KPI``, ``ID``, …)
    in their canonical form.
    """
    if not raw:
        return ""
    tokens = [t for t in re.split(r"[_\-\s]+", raw.strip()) if t]
    if not tokens:
        return ""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        upper = tok.upper()
        if upper in _ACRONYMS:
            out.append(upper)
            continue
        low = tok.lower()
        if i > 0 and low in _SMALL_WORDS:
            out.append(low)
        else:
            out.append(low.capitalize())
    return " ".join(out)


# ---------------------------------------------------------------------------
# Internals — formatters
# ---------------------------------------------------------------------------
def _plotly_tickformat(unit: Unit) -> str | None:
    """Plotly ``tickformat`` string for the y-axis.

    * Currency / count → SI prefix (``"~s"``) — e.g. ``3.5M``.
    * Percent          → ``".0f"`` so 67.4 stays as ``67`` (suffix is ``%``).
    * Duration         → fall through to default; let the unit label do
                          the work for now.
    """
    if unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR, Unit.COUNT):
        return "~s"
    if unit == Unit.PERCENT:
        return ".0f"
    return None  # default plotly formatting


def _plotly_hover_format(unit: Unit) -> str:
    """Plotly ``hovertemplate`` value spec, applied as ``%{y:<spec>}``."""
    if unit == Unit.PERCENT:
        return ".1f"
    if unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR):
        return ",.0f"
    if unit == Unit.COUNT:
        return ",.0f"
    return ",.2f"


def _matplotlib_formatter(unit: Unit) -> Callable[[float, int], str]:
    if unit == Unit.PERCENT:
        return lambda v, _p: f"{v:.0f}"
    if unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR, Unit.COUNT):
        return _si_formatter
    return _plain_formatter


def _kpi_formatter(unit: Unit) -> Callable[[float], str]:
    if unit == Unit.PERCENT:
        return lambda v: f"{v:.1f}%"
    if unit in (Unit.CURRENCY_AED, Unit.CURRENCY_USD, Unit.CURRENCY_EUR):
        return _si_compact
    if unit == Unit.COUNT:
        return _si_compact
    if unit == Unit.DURATION_MS:
        return lambda v: f"{v:,.0f} ms"
    if unit == Unit.DURATION_SECONDS:
        return lambda v: f"{v:,.1f}s"
    return lambda v: _si_compact(v) if abs(v) >= 10_000 else _plain_value(v)


def _si_formatter(v: float, _pos: int) -> str:
    """Matplotlib FuncFormatter for SI tick labels (3,500,000 → '3.5M')."""
    return _si_compact(v)


def _si_compact(v: float) -> str:
    abs_v = abs(v)
    if abs_v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return _plain_value(v)


def _plain_formatter(v: float, _pos: int) -> str:
    return _plain_value(v)


def _plain_value(v: float) -> str:
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"
