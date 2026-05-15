"""High-level orchestrator that ties every service together.

Pipeline for a single user question:

    routing  → DB query (Mongo/MySQL) → vector search →
    trend-quality gates → trust panel → context build →
    Claude insight → chart render → AnalystResponse

All numeric work is delegated to the database. The LLM is given only
already-computed results plus retrieved text rows, never raw arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from models.enums import (
    ChartType,
    ComparisonMode,
    DataSource,
    QueryRoute,
    TimeBucket,
    WarningCode,
)
from models.schemas import (
    AggregationSpec,
    AnalystResponse,
    AnalystWarning,
    ChartPayload,
    Provenance,
    QueryRequest,
    RoutingDecision,
    TimeSpec,
    VectorHit,
)
from services.agent_service import AgentService, agent_service
from services.chart_panel import ChartPanelBuilder
from services.chart_service import ChartService, chart_service
from services.context_service import ContextService, context_service
from services.mongo_service import MongoService, mongo_service
from services.mysql_service import MySQLService, mysql_service
from services.plan_validator import PlanValidator, plan_validator
from services.metric_profile import (
    MetricProfile,
    MultiSeriesProfile,
    classify as classify_metric,
    classify_series,
    compose_chart_title,
    compose_multi_chart_title,
)
from services.question_intent import (
    QuestionIntent,
    classify as classify_intent,
    extract_compared_columns,
    extract_known_columns,
    tokenize as tokenize_question,
)
from services.routing_service import RoutingService, routing_service
from services.session_patch import (
    FollowUpAnalysis,
    analyze_followup,
    apply_patch,
    mask_refinement_patterns,
    resolve_followup,
    should_reuse_prior_collection,
)
from services.session_service import SessionService, session_service
from services.session_summary import (
    ConversationSummariser,
    Turn,
    summariser as conversation_summariser,
)
from services.trend_quality import TrendQualityChecker, trend_quality_checker
from services.trust_service import TrustService, trust_service
from services.vector_service import VectorService, vector_service
from utils.config import settings
from utils.exceptions import AIAnalystError, DatabaseError
from utils.logger import logger
from services.visualization.shape_analysis import infer_shape
from services.visualization.visualization_planner import VisualizationPlanner


def _build_metric_profile(decision: RoutingDecision) -> MetricProfile | None:
    """Build the single :class:`MetricProfile` used by titles, axes, ticks.

    Returns ``None`` when there's no aggregation (e.g. semantic-only
    answers) — callers fall back to the legacy display path in that
    case.
    """
    spec = decision.aggregation
    if spec is None:
        return None
    glossary_term = (
        decision.definition.definition.term
        if decision.definition is not None
        else None
    )
    return classify_metric(
        spec.metric,
        operation=(spec.operation or "sum"),
        glossary_term=glossary_term,
    )


def _metric_display_title(spec: AggregationSpec | None) -> str | None:
    """Legacy helper retained as a defensive fallback for tests / scripts.

    Production code paths now go through :func:`_build_metric_profile`.
    """
    if not spec or not spec.metric:
        return None
    return spec.metric.replace("_", " ").title()


_AWQAF_FACTS_SUFFIX = "_facts"
_AWQAF_GLOSSARY_COLLECTION = "awqaf_datasets_glossary"
_AWQAF_METADATA_COLLECTION = "awqaf_datasets_metadata"

# Cache TTLs (seconds) for the analytical pipeline. The catalog used by
# the discovery handler is intentionally NOT cached — every "hi" / "list
# datasets" reads ``awqaf_datasets_metadata`` and probes facts collections
# live, so a re-ingest shows up in the very next response with no TTL
# wait. See ``_handle_discovery`` for the trade-off.
_SCHEMA_CACHE_TTL = 60.0
_GLOSSARY_CACHE_TTL = 300.0
_CATALOG_ANCHOR_CACHE_TTL = 60.0
_CATALOG_ANCHOR_MIN_TOKEN_LEN = 6
# Coverage = "what years/periods does this target actually contain right
# now?". Short TTL so re-ingest is reflected quickly, but long enough
# that a follow-up question on the same target doesn't pay the probe
# cost twice in a row. Cheap to recompute either way (single $group).
_COVERAGE_CACHE_TTL = 60.0

# Minimum rows a (slug, year) pair must have before that year is good
# enough to anchor a discovery lead suggestion. Below this we walk back
# to an earlier year on the same dataset rather than proposing a query
# that would chart a 1- or 2-point series and call it a "trend" or an
# "annual total". A row is one (period × dimension) record, so for a
# monthly time-series with 6 emirates this corresponds to ~half a month.
_MIN_FACTS_ROWS_FOR_LEAD = 3


class _TTLCache:
    """Tiny per-orchestrator TTL cache with explicit miss handling.

    Avoids pulling in a heavier dependency for what is a few-key lookup.
    Not thread-safe in the strict sense; a concurrent miss may fetch the
    same key twice, but that's safe because the fetch is idempotent.
    """

    __slots__ = ("_ttl", "_data")

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}

    def get_or_set(self, key: str, fetch):
        now = time.monotonic()
        cached = self._data.get(key)
        if cached is not None and (now - cached[0]) < self._ttl:
            return cached[1]
        value = fetch()
        self._data[key] = (now, value)
        return value

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._data.clear()
        else:
            self._data.pop(key, None)

# Envelope fields aren't user-meaningful metrics — never offer them as
# alternatives when the user's requested metric is missing.
_ENVELOPE_FIELDS = frozenset(
    {
        "_id",
        "dataset",
        "source_file",
        "ingested_at",
        "year",
        "month",
        "month_num",
        "period",
        "dimension",
    }
)

# Single source of truth for "tokens that are structural noise rather than
# real metric names". Used both for question→column keyword extraction and
# for the partial-relevance unrecognized-term warning. Kept generous: a
# false-positive structural classification just means we don't warn about
# a token; a false-negative would flag conversational words and look
# broken to the user. Both call sites pay the cost of one frozenset
# membership check per question token, which is O(1).
_STRUCTURAL_TOKENS: frozenset[str] = frozenset(
    {
        # Articles / connectors / common prepositions
        "the", "of", "and", "or", "for", "from", "with", "in", "on",
        "by", "to", "per", "into", "onto", "about", "around", "across",
        "during", "between",
        # Set / comparison words
        "vs", "versus", "both", "side", "either",
        "compare", "comparison", "difference",
        # Question pronouns and conversational filler
        "what", "which", "when", "where", "why", "how", "who", "whose",
        "this", "that", "these", "those", "any", "all",
        "can", "could", "would", "should", "may", "might", "will",
        "shall", "do", "does", "did", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had",
        "please", "kindly", "thanks", "thank",
        "me", "us", "we", "you", "your", "yours", "my", "our", "ours",
        "i", "myself", "yourself", "themselves",
        # Directives that map to "show me data" (not metrics)
        "show", "give", "list", "find", "tell", "explain", "describe",
        "display", "present", "render", "draw", "plot", "summarize",
        "summarise", "report", "reports", "reporting",
        "directly", "specifically", "exactly", "really", "actually",
        # Meta-words about asking / answering
        "question", "questions", "ask", "asked", "answer", "answers",
        # Chart / visualization vocabulary (medium, not a metric)
        "chart", "charts", "graph", "graphs", "plots",
        "visualization", "visualizations", "visualisation",
        "visualisations", "diagram", "diagrams", "table", "tables",
        "trend", "trends", "trending", "spike", "drop",
        # Time cadence words (not metric names)
        "monthly", "yearly", "annual", "annually", "weekly", "daily",
        "quarter", "quarterly", "hourly", "today", "yesterday",
        "tomorrow", "now",
        # Domain-generic words that are part of the slug or label
        "service", "services", "data", "dataset", "datasets",
        "value", "values", "number", "numbers", "amount", "amounts",
        # Aggregation verbs
        "total", "totals", "sum", "average", "avg", "mean", "median",
        "min", "minimum", "max", "maximum", "count", "counts",
    }
)

# Calendar month names — handled separately so we never have to update
# this list when adding new structural tokens.
_MONTH_TOKENS: frozenset[str] = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    }
)


# ---------------------------------------------------------------------------
# Metric classification (used by the discovery lead-suggestion picker)
#
# These are deliberately simple substring/suffix lists rather than a learned
# classifier — the metric vocabulary in `awqaf_datasets_metadata.key_metrics`
# is small and stable. Each new ingestion can easily be audited against the
# lists below; if a domain word is missing we drop the metric from the
# corresponding bucket rather than fabricate a suggestion.
# ---------------------------------------------------------------------------
_NUMERIC_METRIC_HINTS: frozenset[str] = frozenset(
    {
        "transaction", "transactions", "count", "counts",
        "recipient", "recipients", "registration", "registrations",
        "payer", "payers", "payment", "payments",
        "_pct", "percent", "percentage", "_aed",
        "rate", "rates", "amount", "amounts",
        "total", "totals", "revenue", "revenues",
        "student", "students", "family", "families",
        "request", "requests", "disbursed", "collected",
        "donation", "donations", "fatwa", "fatwas",
        "occupancy", "calculation", "calculations",
        "eligible",
    }
)

_CHANNEL_METRIC_HINTS: tuple[str, ...] = (
    "website", "smart_app", "smartphone_app", "smartphone",
    "ipad", "branch_visit", "mobile_website", "mobile",
    "kiosk", "bank", "in_person", "online",
)

_DIMENSIONAL_METRIC_SUFFIXES: tuple[str, ...] = (
    "_type", "_types", "_branch", "_branches",
    "_emirate", "_emirates", "_category", "_categories",
    "_name", "_names", "activity_type",
)

# Metrics that are numeric (chartable) but not meaningfully summable —
# adding two months of "occupancy_rate_pct" yields 95% + 88% = 183%, which
# is a category error, not a real total. The Trend bucket is unaffected
# (monthly views show the value as-is); only the Annual-total bucket uses
# this filter.
_NON_SUMMABLE_METRIC_HINTS: tuple[str, ...] = (
    "_pct", "percent", "percentage", "_rate", "rates",
)

# Phrases that opt the user into the long department-grouped catalog
# instead of the default compact summary.
_FULL_CATALOG_PHRASES: tuple[str, ...] = (
    "show full catalog", "show the full catalog", "full catalog",
    "complete catalog", "entire catalog",
    "show all datasets", "list all datasets", "list every dataset",
    "show every dataset", "all datasets",
    "show me everything", "everything you have",
)


def _question_keywords(question: str) -> list[str]:
    """Return content words from the question, dropping stopwords."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z_]+", question.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STRUCTURAL_TOKENS]


def _metric_is_numeric(name: str) -> bool:
    """True if ``name`` looks like a measurable metric (sum / chartable)."""
    if not isinstance(name, str) or not name:
        return False
    n = name.lower()
    if any(n.endswith(suf) for suf in _DIMENSIONAL_METRIC_SUFFIXES):
        return False
    return any(hint in n for hint in _NUMERIC_METRIC_HINTS)


def _metric_is_channel(name: str) -> bool:
    """True if ``name`` is a service-delivery channel (used by Compare bucket)."""
    if not isinstance(name, str) or not name:
        return False
    n = name.lower()
    return any(n.startswith(h) or h in n for h in _CHANNEL_METRIC_HINTS)


def _metric_is_summable(name: str) -> bool:
    """True if ``name`` is numeric AND survives ``sum`` aggregation.

    Excludes rates / percentages where summing across periods produces a
    nonsense value (e.g. ``occupancy_rate_pct``).
    """
    if not _metric_is_numeric(name):
        return False
    n = name.lower()
    return not any(h in n for h in _NON_SUMMABLE_METRIC_HINTS)


def _normalize_years(row: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for y in row.get("years") or []:
        try:
            out.append(int(y))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _max_year(row: dict[str, Any]) -> int | None:
    ys = _normalize_years(row)
    return ys[-1] if ys else None


def _row_slug(row: dict[str, Any]) -> str:
    """Return the user-facing dataset slug (hyphenated)."""
    return str(row.get("dataset") or "").replace("_", "-")


# Title-Case helpers used by the lead-suggestion renderer. The router
# tolerates both the storage form (``hajj_package_service``,
# ``total_transactions``) and the human form (``Hajj Package Service``,
# ``total transactions``) — see ``_score_target`` and ``_guess_metric``
# in routing_service.py — so these two functions only affect the visible
# text. Suggestions remain executable when copy-pasted.
_DATASET_SMALL_WORDS: frozenset[str] = frozenset(
    {"and", "or", "of", "for", "in", "the", "to", "a", "an", "by", "with"}
)


def _humanize_dataset(slug: str) -> str:
    """``hajj-package-service`` → ``Hajj Package Service``.

    Title Case with English small-word handling so connector words don't
    get awkwardly capitalized (e.g. ``Occupancy Rates and Revenues``,
    not ``Occupancy Rates And Revenues``).
    """
    if not slug:
        return ""
    tokens = [t for t in re.split(r"[-_]+", str(slug).strip()) if t]
    if not tokens:
        return ""
    out = [tokens[0].capitalize()]
    for t in tokens[1:]:
        if t.lower() in _DATASET_SMALL_WORDS:
            out.append(t.lower())
        else:
            out.append(t.capitalize())
    return " ".join(out)


def _humanize_metric(name: str) -> str:
    """``total_transactions`` → ``total transactions``.

    Kept lowercase: rendered inline inside a sentence (``monthly
    {metric} for {dataset}``), so leading capitalization would jar.
    Underscores and hyphens both collapse to single spaces so the
    output stays parseable by the router's metric regex.
    """
    if not name:
        return ""
    return re.sub(r"[_-]+", " ", str(name).strip()).lower()


def _wants_full_catalog(question: str) -> bool:
    """User explicitly asked for the long department-grouped catalog."""
    q = (question or "").lower().strip()
    if not q:
        return False
    for phrase in _FULL_CATALOG_PHRASES:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
            return True
    return False


def _summarize_catalog(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the one-line scope statistics rendered in the summary header."""
    departments_seen: dict[str, int] = {}
    for row in catalog:
        dept = (row.get("department") or "").strip()
        if dept:
            departments_seen[dept] = departments_seen.get(dept, 0) + 1
    departments = sorted(
        departments_seen, key=lambda d: (-departments_seen[d], d)
    )

    all_years: list[int] = []
    for row in catalog:
        all_years.extend(_normalize_years(row))

    n_time_series = sum(
        1 for r in catalog if (r.get("data_type") or "").lower() == "time_series"
    )
    n_directories = sum(
        1 for r in catalog if (r.get("data_type") or "").lower() == "directory"
    )

    return {
        "n_datasets": len(catalog),
        "departments": departments,
        "n_departments": len(departments),
        "year_min": min(all_years) if all_years else None,
        "year_max": max(all_years) if all_years else None,
        "n_time_series": n_time_series,
        "n_directories": n_directories,
    }


def _pick_lead_suggestions(
    catalog: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return up to 5 ``(label, executable_query)`` pairs from real metadata.

    Each pair is built from a real ``(slug, metric, year)`` triple, so a
    user copying the suggested line gets a query the analytical pipeline
    can actually execute against the ingested facts collection.

    Diversity rule: a dataset can occupy at most one bucket per render —
    five suggestions never collapse to the same domain. Buckets without
    an eligible dataset are silently dropped (we never fabricate a
    suggestion).
    """
    if not catalog:
        return []

    by_recency = sorted(
        catalog,
        key=lambda r: (-(_max_year(r) or 0), str(r.get("dataset") or "")),
    )
    used: set[str] = set()
    leads: list[tuple[str, str]] = []

    def _ts(row: dict[str, Any]) -> bool:
        return (row.get("data_type") or "").lower() == "time_series"

    def _dir(row: dict[str, Any]) -> bool:
        return (row.get("data_type") or "").lower() == "directory"

    def _year_suffix(row: dict[str, Any]) -> str:
        y = _max_year(row)
        return f" in {y}" if y else ""

    def _ds(row: dict[str, Any]) -> str:
        """Human-form dataset label for inline rendering."""
        return _humanize_dataset(_row_slug(row))

    # 1. TREND — monthly time-series with a clear total / numeric metric.
    for row in by_recency:
        slug = _row_slug(row)
        if not slug or slug in used or not _ts(row):
            continue
        nums = [
            m for m in (row.get("key_metrics") or []) if _metric_is_numeric(m)
        ]
        if not nums:
            continue
        # Prefer a "total_*" metric if one exists; this gives the cleanest
        # single-number-per-month chart.
        metric = next((m for m in nums if "total" in m.lower()), nums[0])
        leads.append(
            (
                "Trend",
                f"monthly {_humanize_metric(metric)} for {_ds(row)}{_year_suffix(row)}",
            )
        )
        used.add(slug)
        break

    # 2. COMPARE channels — at least 2 channel-style metrics in one dataset.
    for row in by_recency:
        slug = _row_slug(row)
        if not slug or slug in used:
            continue
        channels = [
            m for m in (row.get("key_metrics") or []) if _metric_is_channel(m)
        ]
        if len(channels) < 2:
            continue
        m1, m2 = channels[0], channels[1]
        leads.append(
            (
                "Compare channels",
                f"compare {_humanize_metric(m1)} and {_humanize_metric(m2)} "
                f"for {_ds(row)}{_year_suffix(row)}",
            )
        )
        used.add(slug)
        break

    # 3. ANNUAL TOTAL — different time-series dataset; prefer a non-total
    # metric so the wording stays varied across buckets. Uses the strict
    # ``_metric_is_summable`` filter so we never suggest summing a rate.
    for row in by_recency:
        slug = _row_slug(row)
        if not slug or slug in used or not _ts(row):
            continue
        nums = [
            m for m in (row.get("key_metrics") or []) if _metric_is_summable(m)
        ]
        if not nums:
            continue
        non_total = [m for m in nums if "total" not in m.lower()]
        metric = non_total[0] if non_total else nums[0]
        leads.append(
            (
                "Annual total",
                f"total {_humanize_metric(metric)} for {_ds(row)}{_year_suffix(row)}",
            )
        )
        used.add(slug)
        break

    # 4. DEFINE — any dataset with a metric we can teach about. The
    # phrasing anchors the metric to its host dataset (``explain X for
    # Y``) so the router can't grab the wrong target on token overlap
    # and doesn't mis-parse ``mean`` as the ``avg`` aggregation keyword.
    for row in catalog:
        slug = _row_slug(row)
        if not slug or slug in used:
            continue
        metrics = [
            m for m in (row.get("key_metrics") or [])
            if isinstance(m, str) and m
        ]
        if not metrics:
            continue
        # Prefer a numeric metric — definitions are most useful for
        # measurable indicators rather than pure category dimensions.
        metric = next((m for m in metrics if _metric_is_numeric(m)), metrics[0])
        leads.append(
            (
                "Define a metric",
                f"explain {_humanize_metric(metric)} for {_ds(row)}",
            )
        )
        used.add(slug)
        break

    # 5. REGISTRY — directory-style dataset. ``how many <X> are there
    # in <year>`` routes to ``count`` (no metric required) and works on
    # directory collections; the older ``list datasets in <X>`` phrasing
    # forced a ``sum`` aggregation that always failed.
    for row in by_recency:
        slug = _row_slug(row)
        if not slug or slug in used or not _dir(row):
            continue
        leads.append(
            (
                "Browse a registry",
                f"how many {_ds(row).lower()} are there{_year_suffix(row)}?",
            )
        )
        used.add(slug)
        break

    return leads


def _render_discovery_summary(
    catalog: list[dict[str, Any]],
    leads: list[tuple[str, str]] | None = None,
) -> str:
    """Compact catalog summary + a few executable lead questions.

    The opposite of dumping the entire catalog: the user gets a one-line
    sense of scope and a short list of concrete questions, all computed
    from ``catalog`` so nothing here goes stale when datasets are added
    or removed. The full department-grouped listing is still available
    via ``show full catalog`` (handled by ``_render_discovery_markdown``).

    ``leads`` is normally pre-computed by the orchestrator so each
    suggestion can be validated against the live facts collection
    before being shown. When ``None``, this falls back to the
    unvalidated picker — convenient for tests but not the production
    path.
    """
    if not catalog:
        return (
            "**No datasets are ingested yet.**\n\n"
            "Run `python scripts/ingest_awqaf.py --drop-all` to load the "
            "AWQAF data, then ask me again."
        )

    s = _summarize_catalog(catalog)
    departments = s["departments"]
    if not departments:
        dept_inline = ""
    elif len(departments) <= 5:
        dept_inline = ", ".join(departments)
    else:
        dept_inline = (
            ", ".join(departments[:5]) + f", + {len(departments) - 5} more"
        )

    year_min, year_max = s["year_min"], s["year_max"]
    if year_min and year_max:
        year_phrase = (
            f"{year_min} – {year_max}" if year_min != year_max else str(year_min)
        )
    else:
        year_phrase = "various periods"

    n_ts, n_dir = s["n_time_series"], s["n_directories"]
    if n_ts and n_dir:
        shape_note = (
            f" Of these, **{n_ts}** are monthly/time-series and "
            f"**{n_dir}** are registries."
        )
    elif n_ts:
        shape_note = " Most are monthly/time-series statistics."
    elif n_dir:
        shape_note = " Most are registries / directories."
    else:
        shape_note = ""

    header = (
        f"**AWQAF data warehouse** — I have **{s['n_departments']} departments**"
    )
    if dept_inline:
        header += f" ({dept_inline})"
    header += f", covering **{year_phrase}**.{shape_note}"

    if leads is None:
        leads = _pick_lead_suggestions(catalog)
    lines: list[str] = [header, ""]
    if leads:
        lines.append("**A few questions to get you started**")
        lines.append("")
        for label, query in leads:
            lines.append(f"- **{label}** — `{query}`")
        lines.append("")
        lines.append(
            "Want a different angle? Just describe what you'd like to "
            "know — name a dataset and (optionally) a metric and a year. "
            "Type _show full catalog_ to see every dataset."
        )
    else:
        lines.append(
            "Ask any question about a specific dataset and I'll route it. "
            "Type _show full catalog_ to see every dataset."
        )
    return "\n".join(lines)


def _render_discovery_markdown(catalog: list[dict[str, Any]]) -> str:
    """Render the long department-grouped catalog with per-dataset starters.

    This is the opt-in "give me everything" view, reached by phrases like
    ``show full catalog`` or ``list every dataset``. The default discovery
    response is the much shorter ``_render_discovery_summary``.
    """
    if not catalog:
        return (
            "**No datasets are ingested yet.**\n\n"
            "Run `python scripts/ingest_awqaf.py --drop-all` to load the "
            "AWQAF data, then ask me again."
        )

    by_dept: dict[str, list[dict[str, Any]]] = {}
    for row in catalog:
        dept = (row.get("department") or "Other").strip() or "Other"
        by_dept.setdefault(dept, []).append(row)

    lines: list[str] = [
        "**AWQAF Data Catalog**",
        "",
        "I have data on the AWQAF datasets below. Pick any starter question "
        "or write your own — name a dataset and (optionally) a metric and a "
        "year and I'll do the rest.",
        "",
    ]
    for dept in sorted(by_dept.keys()):
        lines.append(f"**{dept}**")
        for row in sorted(by_dept[dept], key=lambda r: (r.get("dataset_name") or "")):
            name = row.get("dataset_name") or row.get("dataset") or "unknown"
            slug = (row.get("dataset") or "").replace("_", "-")
            purpose = (row.get("purpose") or "").strip()
            years = row.get("years") or []
            metrics = row.get("key_metrics") or []
            data_type = (row.get("data_type") or "").lower()

            year_part = ""
            if years:
                ys = sorted({int(y) for y in years if isinstance(y, (int, str)) and str(y).isdigit()})
                if ys:
                    year_part = f" — {ys[0]}–{ys[-1]}" if len(ys) > 1 else f" — {ys[0]}"

            metric_hint = ""
            if metrics:
                pick = metrics[0]
                if isinstance(pick, dict):
                    pick = pick.get("name") or pick.get("metric") or ""
                metric_hint = str(pick or "").strip()

            short_purpose = purpose
            if len(short_purpose) > 120:
                short_purpose = short_purpose[:117].rstrip() + "…"

            lines.append(f"- **{name}**{year_part}. {short_purpose}")
            starter = _starter_question(slug, metric_hint, data_type, years)
            if starter:
                lines.append(f"  - _Try: {starter}_")
        lines.append("")

    lines += [
        "**Other things to try**",
        "- _What can I ask?_  → re-shows this catalog",
        "- _Compare two metrics_ — e.g. `compare smart app and website "
        "transactions for hajj-permit-service in 2024`",
        "- _Define a term_ — e.g. `what does total_transactions mean?`",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Empty-metric detection (post-aggregation)
#
# Distinct from "missing metric" (column doesn't exist): here the column
# IS in the schema, the aggregation ran, but every row's value is null
# or zero. That pattern almost always means the field wasn't populated
# for the requested scope — and we owe the user an explicit explanation
# instead of a chart full of zeros.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _EmptyMetricFinding:
    """Per-metric outcome of the post-aggregation data-availability scan.

    ``has_series`` distinguishes the two response shapes:

    * False → single-metric query; ``empty`` (if non-empty) means we
      should short-circuit the LLM with a deterministic "what IS
      available" answer.
    * True  → comparison fan-out; ``empty`` is the subset of compared
      metrics that returned no usable data, ``populated`` is the subset
      that did. We keep the LLM in the loop but prepend an explicit
      data-availability notice so the response leads with the gap.
    """

    requested: list[str]
    empty: list[str]
    populated: list[str]
    has_series: bool

    @property
    def all_empty(self) -> bool:
        """True iff every requested metric came back empty."""
        return bool(self.requested) and not self.populated


@dataclass(frozen=True)
class _TargetCoverage:
    """Live + metadata view of "what time periods does this target hold?".

    Produced by :meth:`AnalystService._probe_target_coverage` and cached
    per (target, time_field) for the duration of a request. The
    zero-rows branch consults this instead of returning a generic
    "try a different period" so the user sees concrete years that
    actually have data.

    Fields:

    * ``live_years``   — sorted distinct calendar years present in the
      facts collection right now (source of truth).
    * ``earliest`` / ``latest`` — bounds of the time field (datetime
      when the field is a real date; int when it's a ``year`` column;
      ``YYYY-MM`` string when it's a ``period`` column). Mostly used
      to render a human-readable span.
    * ``catalog_years`` — years the metadata catalog *claims* exist
      for this dataset. Surfaces ingest gaps when it disagrees with
      ``live_years``.
    * ``probed`` — False when the live probe failed entirely (network
      error, unknown time field). Callers fall back to a generic
      message rather than asserting "this dataset has no data" on a
      flaky probe.
    """

    target: str
    time_field: str | None
    live_years: list[int]
    earliest: Any
    latest: Any
    catalog_years: list[int]
    probed: bool

    @property
    def has_any_data(self) -> bool:
        """True iff the live probe found at least one year of data."""
        return bool(self.live_years) or self.earliest is not None

    def excludes_year(self, year: int) -> bool:
        """True iff the live probe confirms ``year`` has no data here."""
        return self.probed and bool(self.live_years) and year not in self.live_years

    def gap_years(self) -> list[int]:
        """Catalog claims these years but the live probe sees none.

        Used to flag ingest drift in the trust panel — the catalog
        should be the contract, not a wish list. Empty when either
        side is unknown.
        """
        if not self.catalog_years or not self.live_years:
            return []
        live = set(self.live_years)
        return sorted(y for y in self.catalog_years if y not in live)


def _is_empty_value(value: Any) -> bool:
    """True iff ``value`` contributes nothing to the user's question.

    ``None`` is unambiguously "not collected". ``0`` is ambiguous (could
    be "collected and was zero" or "field never populated"); for the
    empty-metric warning we treat it as empty when *every* row is zero
    — that pattern almost never occurs in genuinely populated data.
    Booleans are excluded so a literal ``False`` doesn't get classified
    as a metric value.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _starter_question(
    slug: str,
    metric_hint: str,
    data_type: str,
    years: list[Any],
) -> str | None:
    """Pick the most useful starter query for a dataset row.

    Renders in the human form (Title-Case dataset, lowercase metric with
    spaces) rather than the storage slug — both forms are accepted by
    the router, but only the human form belongs in user-facing text.
    """
    if not slug:
        return None
    year = ""
    if years:
        ys = [int(y) for y in years if isinstance(y, (int, str)) and str(y).isdigit()]
        if ys:
            year = f" in {max(ys)}"
    dataset_label = _humanize_dataset(slug)
    if data_type in {"directory", "registry"} or not metric_hint:
        return f"list datasets in {dataset_label}{year}".strip()
    return (
        f"monthly {_humanize_metric(metric_hint)} for {dataset_label}{year}".strip()
    )


def _dataset_slug_from_target(target: str | None) -> str | None:
    """``awqaf_hajj_package_service_facts`` → ``hajj_package_service``."""
    if not target or not target.startswith("awqaf_") or not target.endswith(_AWQAF_FACTS_SUFFIX):
        return None
    return target[len("awqaf_") : -len(_AWQAF_FACTS_SUFFIX)]


def _format_definitions_text(
    doc: dict[str, Any] | None,
    spec_metric: str | None,
) -> str | None:
    """Format a glossary doc + the picked metric into a short prompt block.

    Pure formatter: no I/O. Kept module-level so it stays trivially
    testable and so the orchestrator's cached fetch can wrap it.
    """
    if not doc:
        return None
    lines: list[str] = []
    dataset_name = doc.get("dataset_name")
    indicator = doc.get("indicator") or {}
    if dataset_name:
        line = f"Dataset: {dataset_name}"
        unit = indicator.get("unit")
        if unit:
            line += f" (unit: {unit})"
        lines.append(line)
    if indicator.get("definition"):
        lines.append(f"Indicator: {indicator['definition']}")

    fields = doc.get("fields") or {}
    if isinstance(fields, dict):
        if spec_metric and spec_metric in fields:
            relevant = [spec_metric]
        else:
            relevant = list(fields.keys())[:6]
        for fk in relevant:
            meta = fields.get(fk) or {}
            desc = (meta or {}).get("description")
            if not desc:
                continue
            lines.append(f"- {fk}: {desc}")

    return "\n".join(lines) if lines else None


def _chart_companion_for_llm(
    decision: RoutingDecision,
    rows: list[dict[str, Any]],
    *,
    will_render_chart: bool,
) -> str | None:
    """Tell the model a chart exists and how it maps to STRUCTURED DATA (axes, scale caveats)."""
    if not will_render_chart or len(rows) < 2:
        return None
    spec = decision.aggregation
    if spec is None:
        return None
    metric = _metric_display_title(spec) or "the metric in field ``value``"
    bucket = spec.time.bucket.value if spec.time else "category"
    return (
        "VISUAL (a chart is included with this answer; it is drawn from the same rows as "
        "STRUCTURED DATA above):\n"
        "- Horizontal axis: each ``label`` (time period or category).\n"
        f"- Vertical axis: «{metric}» — use only the numeric ``value`` field (authoritative).\n"
        f"- Plan context: aggregation bucket / grouping = «{bucket}».\n"
        "In your narrative, briefly teach the reader how to read the chart: identify the highest "
        "and lowest periods with exact figures from STRUCTURED DATA. If one period is vastly larger "
        "than the others, state explicitly that smaller values can appear near the baseline on a "
        "shared scale even when they are not zero."
    )


class AnalystOrchestrator:
    def __init__(
        self,
        router: RoutingService | None = None,
        mongo: MongoService | None = None,
        mysql: MySQLService | None = None,
        vector: VectorService | None = None,
        context: ContextService | None = None,
        agent: AgentService | None = None,
        charts: ChartService | None = None,
        trend_quality: TrendQualityChecker | None = None,
        trust: TrustService | None = None,
        sessions: SessionService | None = None,
        validator: PlanValidator | None = None,
        summariser: ConversationSummariser | None = None,
    ) -> None:
        self.router = router or routing_service
        self.mongo = mongo or mongo_service
        self.mysql = mysql or mysql_service
        self.vector = vector or vector_service
        self.context = context or context_service
        self.agent = agent or agent_service
        self.charts = charts or chart_service
        self._panel = ChartPanelBuilder(self.charts)
        self.trend_quality = trend_quality or trend_quality_checker
        self.trust = trust or trust_service
        self.sessions = sessions or session_service
        self.validator = validator or plan_validator
        # Build #5: opt-in conversation memory. The summariser is held
        # at orchestrator level (rather than re-instantiated per call)
        # so it can keep its lazy Anthropic client warm across requests.
        self.summariser = summariser or conversation_summariser

        # Local caches collapse repeated reads of the same target / dataset
        # within a single analytical request. TTLs are short so freshly
        # ingested data shows up quickly. No catalog cache: the discovery
        # handler reads `awqaf_datasets_metadata` and probes facts
        # collections live on every request (Option A: freshness over
        # ~700 ms latency).
        self._schema_cache = _TTLCache(_SCHEMA_CACHE_TTL)
        self._glossary_cache = _TTLCache(_GLOSSARY_CACHE_TTL)
        # Anchors derived from `awqaf_datasets_metadata` (slugs, dataset
        # names, key-metric names + their salient sub-tokens). Used by the
        # `answer()` second-chance guard to avoid sending real questions
        # into the discovery / out-of-scope short-circuit just because the
        # wording brushes a phrase list.
        self._catalog_anchor_cache = _TTLCache(_CATALOG_ANCHOR_CACHE_TTL)
        # Schema snapshot (columns + small sample) per target. Used by the
        # plan validator to check that a spec is consistent with the live
        # collection BEFORE the database is asked to execute it. Same TTL
        # as `_schema_cache` so a fresh ingest is reflected within ~60 s.
        self._target_schema_cache = _TTLCache(_SCHEMA_CACHE_TTL)
        # Coverage probe (earliest/latest + distinct years) per target.
        # Consulted by the zero-rows branch so the user is told what
        # the dataset actually contains today instead of a generic
        # "try a different period". Live truth — never trusts the
        # catalog alone.
        self._coverage_cache = _TTLCache(_COVERAGE_CACHE_TTL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(self, request: QueryRequest) -> AnalystResponse:
        """Public entry point — wraps the impl in a trace-aware top-level span.

        Behaviour:

        * If a trace is already active (e.g. the eval harness or a test
          fixture started one), we *extend* it by adding a child span
          rather than starting a competing trace.
        * Otherwise we start a fresh :class:`TraceContext`, run the
          pipeline inside it, and attach a compact summary to
          ``response.meta["trace"]`` (and the full trace under
          ``trace_full`` when ``request.include_details`` is true).

        The actual pipeline lives in ``_answer_impl``. Keeping the
        wrapper tiny means observability concerns stay separated from
        analytical concerns, and ``_answer_impl`` can be called directly
        in tests without paying any tracing cost.
        """
        from utils.observability import TraceContext, current_trace, use_trace

        question_attr = (request.question or "")[:80]

        existing = current_trace()
        if existing is not None:
            with existing.span("request.answer", question=question_attr):
                return self._answer_impl(request)

        ctx = TraceContext.start(session_id=request.session_id)
        with use_trace(ctx), ctx.span("request.answer", question=question_attr):
            response = self._answer_impl(request)
        response.meta = {**(response.meta or {}), "trace": ctx.summary()}
        if request.include_details:
            response.meta["trace_full"] = ctx.to_dict()
        return response

    def _answer_impl(self, request: QueryRequest) -> AnalystResponse:
        t0 = time.perf_counter()

        # Step 0a — Stateful follow-up co-reference check. This runs
        # BEFORE the stateless intent classifier because short
        # follow-ups ("compare with 2025", "by emirate", "exclude
        # refunds") look like vague-discovery to a stateless rule but
        # are clearly analytical in context. We use the catalog
        # anchor probe + the prior session decision + a structural
        # refinement-shape detector — no keyword prefix list.
        followup = self._resolve_followup(request)
        if followup is not None:
            decision, patched_from_session = followup
            comparison_intent = self._followup_implies_comparison(decision)
            logger.info(
                f"Follow-up resolved from session — patched prior decision "
                f"target=`{decision.target}` "
                f"reason={decision.reason!r}"
            )
        else:
            # Step 0b — Cheap intent gate. Discovery and out-of-scope
            # questions are answered without touching Mongo, FAISS, or
            # Claude. Comparison is allowed to fall through; the
            # orchestrator picks up multi-metric extraction during
            # the analytical run.
            #
            # Second-chance guard: even if `classify_intent` says
            # DISCOVERY or OUT_OF_SCOPE, we re-check the question
            # against the LIVE catalog. If the user named a real
            # dataset, slug, or key metric, we let the router run.
            intent_result = classify_intent(request.question)
            if intent_result.intent in (
                QuestionIntent.DISCOVERY,
                QuestionIntent.OUT_OF_SCOPE,
            ):
                anchor = self._catalog_anchor_in_question(request.question)
                if anchor is not None:
                    logger.info(
                        f"Intent={intent_result.intent.value.upper()} "
                        f"({intent_result.reason}) but catalog anchor "
                        f"`{anchor}` matched — continuing to router."
                    )
                elif intent_result.intent == QuestionIntent.DISCOVERY:
                    logger.info(
                        f"Intent=DISCOVERY ({intent_result.reason}); "
                        f"skipping router."
                    )
                    return self._handle_discovery(request, t0)
                else:
                    logger.info(
                        f"Intent=OUT_OF_SCOPE ({intent_result.reason}); "
                        f"skipping router."
                    )
                    return self._handle_out_of_scope(request, t0)

            comparison_intent = intent_result.intent == QuestionIntent.COMPARISON
            decision, patched_from_session = self._decide_with_session(request)

        structured_data, exec_warnings = self._run_analytical(
            decision,
            question=request.question,
            comparison=comparison_intent,
        )
        vector_hits = self._run_semantic(decision, request)

        as_of = self._probe_freshness(decision)

        quality = self.trend_quality.assess(
            structured_data,
            decision.aggregation,
            as_of=as_of,
        )
        structured_data = quality.rows
        warnings: list[AnalystWarning] = list(exec_warnings) + list(quality.warnings)

        defn_warning = _draft_definition_warning(decision)
        if defn_warning is not None:
            warnings.append(defn_warning)

        partial_warning = self._partial_relevance_warning(request.question, decision)
        if partial_warning is not None:
            warnings.append(partial_warning)

        # Empty-metric scan (after quality.assess so we see post-processed
        # rows). Two outcomes drive downstream behaviour:
        #
        #   * Comparison fan-out where one or more requested metrics are
        #     empty → emit a structured METRIC_EMPTY warning AND prepend
        #     a prominent "DATA AVAILABILITY" notice to the LLM context
        #     so the model leads with the data gap instead of burying it.
        #
        #   * Single-metric query where the requested metric is empty →
        #     handled later in ``_compose_insight``: short-circuit with a
        #     deterministic "what IS available" response (no LLM call,
        #     no chart of zeros).
        empty_finding = self._detect_empty_metric_data(structured_data, decision)
        if empty_finding.has_series and empty_finding.empty:
            warnings.insert(
                0,
                self._empty_metric_warning(empty_finding, decision.target or ""),
            )

        # Data quality issue detection — when suspicious uniform values are detected,
        # automatically fall back to the most recent valid year
        if (
            quality.has_data_quality_issue
            and decision.target
            and decision.aggregation is not None
            and decision.aggregation.time is not None
        ):
            coverage = self._probe_target_coverage(decision)
            if coverage is not None:
                requested_year = self._requested_year(decision.aggregation)
                # Find alternative years with potentially valid data
                alternative_years = [
                    y for y in coverage.live_years
                    if y != requested_year and y < 2026  # Exclude future years
                ]
                if alternative_years:
                    # Sort by recency and pick the most recent valid year
                    alternative_years.sort(reverse=True)
                    fallback_year = alternative_years[0]
                    
                    logger.info(
                        f"Data quality issue detected in {requested_year}. "
                        f"Automatically falling back to {fallback_year}."
                    )
                    
                    # Re-run aggregation with fallback year
                    fallback_spec = decision.aggregation.model_copy(
                        update={
                            "time": TimeSpec(
                                field=decision.aggregation.time.field,
                                bucket=decision.aggregation.time.bucket,
                                range_from=datetime(fallback_year, 1, 1, tzinfo=timezone.utc),
                                range_to=datetime(fallback_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                                compare=ComparisonMode.NONE,
                                years=None,
                            )
                        }
                    )
                    
                    try:
                        # Re-execute with valid year
                        fallback_rows = self.mongo.run_aggregation(decision.target, fallback_spec)
                        
                        # Re-assess quality of fallback data
                        fallback_quality = self.trend_quality.assess(
                            fallback_rows,
                            fallback_spec,
                            as_of=as_of,
                        )
                        
                        # Only use fallback if it doesn't also have quality issues
                        if not fallback_quality.has_data_quality_issue and fallback_rows:
                            structured_data = fallback_quality.rows
                            warnings = list(exec_warnings) + list(fallback_quality.warnings)
                            
                            # Add informative warning about the fallback
                            warnings.insert(
                                0,
                                AnalystWarning(
                                    code=WarningCode.PARTIAL_PERIOD,
                                    message=(
                                        f"{requested_year} data contains errors (all values = 1). "
                                        f"Automatically showing {fallback_year} data instead. "
                                        f"Other available years: {', '.join(map(str, alternative_years[1:4]))}."
                                    ),
                                )
                            )
                            
                            logger.info(
                                f"Successfully fell back to {fallback_year}: "
                                f"{len(structured_data)} rows retrieved."
                            )
                        else:
                            # Fallback also has issues, keep original with warning
                            logger.warning(
                                f"Fallback year {fallback_year} also has data quality issues. "
                                f"Keeping original {requested_year} data with warnings."
                            )
                            warnings.append(
                                AnalystWarning(
                                    code=WarningCode.PARTIAL_PERIOD,
                                    message=(
                                        f"Try querying {fallback_year} instead — "
                                        f"available years: {', '.join(map(str, alternative_years[:3]))}."
                                    ),
                                )
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            f"Failed to fall back to {fallback_year}: {exc}. "
                            f"Keeping original {requested_year} data."
                        )
                        warnings.append(
                            AnalystWarning(
                                code=WarningCode.PARTIAL_PERIOD,
                                message=(
                                    f"Try querying {fallback_year} instead — "
                                    f"available years: {', '.join(map(str, alternative_years[:3]))}."
                                ),
                            )
                        )
        
        # Coverage probe — for the zero-rows / "wrong period" case the
        # trust panel should reflect what the dataset *does* contain
        # and any catalog ↔ live drift. Cached, so the response-text
        # branch in ``_zero_rows_response`` re-uses the same probe
        # without re-querying Mongo.
        if (
            not structured_data
            and decision.target
            and decision.aggregation is not None
            and decision.aggregation.time is not None
        ):
            coverage = self._probe_target_coverage(decision)
            if coverage is not None:
                requested_year = self._requested_year(decision.aggregation)
                warnings.extend(
                    self._coverage_warnings(coverage, requested_year)
                )

        trust_panel = self.trust.build(
            decision=decision,
            rows=structured_data,
            as_of=as_of,
        )

        will_render_chart = (
            len(structured_data) > 0
            and not quality.suppress_chart
            and request.chart_type != ChartType.NONE
        )
        chart_companion = _chart_companion_for_llm(
            decision,
            structured_data,
            will_render_chart=will_render_chart,
        )

        dataset_definitions = self._dataset_definitions_text(decision)

        conversation_memory = self._load_conversation_memory(request)

        context_block = self.context.build(
            question=request.question,
            structured_data=structured_data,
            vector_hits=vector_hits,
            routing_reason=decision.reason,
            trust=trust_panel,
            warnings=warnings,
            chart_companion=chart_companion,
            dataset_definitions=dataset_definitions,
            conversation_memory=conversation_memory,
        )

        if empty_finding.has_series and empty_finding.empty:
            # Pin the notice at the very top so the LLM cannot miss it.
            notice = self._format_empty_metric_notice(
                empty_finding, decision.target or ""
            )
            context_block = f"{notice}\n\n{context_block}"

        insight = self._compose_insight(
            request=request,
            decision=decision,
            context_block=context_block,
            structured_data=structured_data,
            warnings=warnings,
            empty_finding=empty_finding,
        )
        chart_panel = self._maybe_chart_panel(
            structured_data, request, decision, quality
        )
        primary_chart = next(
            (c for c in chart_panel if c.is_primary), chart_panel[0] if chart_panel else None
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"Answered question in {elapsed_ms}ms — route={decision.route.value} "
            f"source={decision.data_source.value} target={decision.target} "
            f"rows={len(structured_data)} hits={len(vector_hits)} "
            f"warnings={len(warnings)} "
            f"session_patch={patched_from_session}"
        )

        provenance = self._build_provenance(request, decision) if request.include_details else None

        if request.session_id:
            self.sessions.put(request.session_id, request.question, decision)
            self._persist_conversation_turn(request, insight)

        meta: dict[str, Any] = {
            "elapsed_ms": elapsed_ms,
            "rows": len(structured_data),
            "vector_hits": len(vector_hits),
            "warnings": len(warnings),
            "session_id": request.session_id,
            "patched_from_session": patched_from_session,
        }
        critic_meta = self._collect_critic_meta()
        if critic_meta is not None:
            meta["critic"] = critic_meta

        return AnalystResponse(
            question=request.question,
            routing=decision,
            structured_data=structured_data,
            vector_context=vector_hits,
            insight=insight,
            chart=primary_chart,
            charts=chart_panel,
            trust=trust_panel,
            warnings=warnings,
            provenance=provenance,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Intent-level short-circuits (discovery / out-of-scope)
    # ------------------------------------------------------------------
    def _handle_discovery(
        self, request: QueryRequest, t0: float
    ) -> AnalystResponse:
        """Return a deterministic catalog response when the user is exploring.

        Default = compact summary + up to 5 executable lead questions,
        each one validated against the live facts collection so we never
        suggest a query that would return zero rows.

        Every request is served live: one read of
        ``awqaf_datasets_metadata`` plus one existence probe per bucket
        candidate. No cache, no precomputed snapshot — a re-ingest is
        reflected in the very next response. Trade-off: ~700-1100 ms
        per request (mostly the 5 facts probes against Atlas).

        If the user explicitly asks for the full department-grouped
        listing (``show full catalog``, ``list every dataset``…) we
        render the longer view instead — same live read, no per-row
        validation.

        No router, no LLM, no vector search.
        """
        catalog = self._fetch_catalog()
        if _wants_full_catalog(request.question):
            insight = _render_discovery_markdown(catalog)
            reason = "discovery: full catalog (explicit)"
        else:
            leads = self._pick_validated_lead_suggestions(catalog)
            insight = _render_discovery_summary(catalog, leads)
            reason = "discovery: summary + executable leads"
        return self._build_static_response(
            request=request,
            insight=insight,
            t0=t0,
            reason=reason,
            route=QueryRoute.DISCOVERY,
            data_source=DataSource.MONGO,
        )

    @staticmethod
    def _facts_collection_for(slug: str) -> str:
        """``hajj-package-service`` → ``awqaf_hajj_package_service_facts``."""
        return "awqaf_" + slug.replace("-", "_") + "_facts"

    def _facts_row_count(self, slug: str, year: int | None = None) -> int:
        """Best-effort row count for ``slug``'s facts collection.

        With ``year`` set, restricts the count to that calendar year.
        Capped via ``$limit`` so a directory with thousands of records
        doesn't pay a full collection scan just to confirm "yes, it's
        non-empty enough for a discovery suggestion". Fails *closed*
        (returns 0) so a flaky Mongo connection never surfaces an
        unverifiable suggestion.
        """
        if not slug:
            return 0
        coll = self._facts_collection_for(slug)
        match: dict[str, Any] = {"year": int(year)} if year is not None else {}
        pipeline: list[dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline.extend([
            {"$limit": _MIN_FACTS_ROWS_FOR_LEAD * 4},
            {"$count": "n"},
        ])
        try:
            rows = self.mongo.aggregate(coll, pipeline)
            return int(rows[0]["n"]) if rows else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Lead validation: facts count probe on {coll} year={year} "
                f"failed ({type(exc).__name__}: {exc}); treating as empty."
            )
            return 0

    def _facts_distinct_months(self, slug: str, year: int) -> int:
        """How many distinct months in ``year`` have at least one row.

        Raw row counts are misleading for time-series datasets sliced by
        emirate / channel / dimension — a single month with six emirates
        already produces six rows, so a strict ``>=3 rows`` density gate
        passes even when only January is populated. The discovery lead
        for ``Occupancy Rates and Revenues`` hit exactly that trap: the
        ``2026`` annual-total suggestion summed six January rows and
        called it a "year", which the user reasonably flagged as
        misleading.

        Counting *distinct months* instead measures what the suggestion
        actually advertises (a Trend / Annual total over a year of
        activity). Fails *closed* (returns 0) on any error.
        """
        if not slug or year is None:
            return 0
        coll = self._facts_collection_for(slug)
        pipeline: list[dict[str, Any]] = [
            {"$match": {"year": int(year)}},
            {"$group": {"_id": "$month_num"}},
            {"$count": "n"},
        ]
        try:
            rows = self.mongo.aggregate(coll, pipeline)
            return int(rows[0]["n"]) if rows else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Lead validation: distinct-month probe on {coll} "
                f"year={year} failed ({type(exc).__name__}: {exc})."
            )
            return 0

    def _facts_has_data(self, slug: str, year: int | None = None) -> bool:
        """True iff the facts collection for ``slug`` has at least one row."""
        return self._facts_row_count(slug, year) >= 1

    def _pick_data_dense_year(
        self,
        slug: str,
        candidate_years: list[int],
        *,
        require_monthly_spread: bool,
    ) -> int | None:
        """Pick the most recent year on ``slug`` that has *enough* data.

        Two probe strategies depending on the bucket:

        * ``require_monthly_spread=True`` (Trend / Compare / Annual
          total): we need at least
          :data:`_MIN_FACTS_ROWS_FOR_LEAD` *distinct months* in the
          chosen year, not just rows. This is the fix for the "most
          recent year is one month deep" bug — an Occupancy dataset
          that lists 2026 in metadata but only has January 2026
          ingested looks dense by row count (six emirates → six rows)
          yet produces a misleading "annual total" of one month. The
          monthly-spread gate skips 2026 in that case and falls back
          to 2025.
        * ``require_monthly_spread=False`` (Registry / directory): raw
          row count is the right measure — directories don't have a
          monthly axis.

        Returns ``None`` only when every candidate year is empty — the
        caller drops the lead silently rather than fabricate one.
        """
        ys = sorted({int(y) for y in (candidate_years or []) if y}, reverse=True)
        if not ys:
            return None

        def _is_dense(year: int) -> bool:
            if require_monthly_spread:
                return (
                    self._facts_distinct_months(slug, year)
                    >= _MIN_FACTS_ROWS_FOR_LEAD
                )
            return self._facts_row_count(slug, year) >= _MIN_FACTS_ROWS_FOR_LEAD

        # First pass: a year that is genuinely dense enough.
        for year in ys:
            if _is_dense(year):
                return year
        # Second pass: any non-empty year. Better an imperfect lead than
        # a missing bucket — we'd rather suggest a sparse year than drop
        # the dataset entirely.
        for year in ys:
            if self._facts_row_count(slug, year) >= 1:
                return year
        return None

    def _pick_validated_lead_suggestions(
        self, catalog: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Same buckets as ``_pick_lead_suggestions`` plus per-candidate
        quality validation against the live facts collection.

        Two-axis validation:

          1. **Coverage** — does the dataset's facts collection actually
             have rows? Catches metadata drift (dataset removed but
             still listed) and ingest gaps.
          2. **Density** — for time-series buckets (Trend, Compare,
             Annual total), does the picked year have enough rows to
             make the resulting chart / total meaningful? When the most
             recent year is sparse, we walk back through the dataset's
             ``years`` list rather than show a misleading near-empty
             answer.

        Phrasing is also intent-aware so each lead routes cleanly:

          * **Define** uses ``explain <metric> for <dataset>`` rather
            than ``what does <metric> mean?`` — the former binds the
            metric to the right dataset and avoids ``mean`` being
            mis-parsed as the ``avg`` aggregation keyword.
          * **Registry** uses ``how many <dataset> are there in <year>``
            rather than ``list datasets in <dataset>`` — the former
            routes to a row-count operation that works on directory
            collections, while the latter forced a ``sum`` aggregation
            against a directory that has no summable metric.

        Walk-forward semantics: if a bucket's top-recency candidate
        fails validation, we move to the next eligible candidate. If no
        candidate validates, the bucket is silently dropped — we never
        fabricate a suggestion just to fill a slot.
        """
        if not catalog:
            return []

        by_recency = sorted(
            catalog,
            key=lambda r: (-(_max_year(r) or 0), str(r.get("dataset") or "")),
        )
        used: set[str] = set()
        leads: list[tuple[str, str]] = []

        def _ts(row: dict[str, Any]) -> bool:
            return (row.get("data_type") or "").lower() == "time_series"

        def _dir(row: dict[str, Any]) -> bool:
            return (row.get("data_type") or "").lower() == "directory"

        def _ds(row: dict[str, Any]) -> str:
            return _humanize_dataset(_row_slug(row))

        # 1. TREND — monthly time-series with a numeric metric AND a
        # year that has data spread across multiple months (not just
        # any year listed in metadata).
        for row in by_recency:
            slug = _row_slug(row)
            if not slug or slug in used or not _ts(row):
                continue
            nums = [
                m for m in (row.get("key_metrics") or [])
                if _metric_is_numeric(m)
            ]
            if not nums:
                continue
            year = self._pick_data_dense_year(
                slug, _normalize_years(row), require_monthly_spread=True
            )
            if year is None:
                continue
            metric = next((m for m in nums if "total" in m.lower()), nums[0])
            leads.append(
                (
                    "Trend",
                    f"monthly {_humanize_metric(metric)} for {_ds(row)} in {year}",
                )
            )
            used.add(slug)
            break

        # 2. COMPARE channels — at least 2 channel-style metrics AND a
        # year with data spread across multiple months.
        for row in by_recency:
            slug = _row_slug(row)
            if not slug or slug in used:
                continue
            channels = [
                m for m in (row.get("key_metrics") or [])
                if _metric_is_channel(m)
            ]
            if len(channels) < 2:
                continue
            year = self._pick_data_dense_year(
                slug, _normalize_years(row), require_monthly_spread=True
            )
            if year is None:
                continue
            m1, m2 = channels[0], channels[1]
            leads.append(
                (
                    "Compare channels",
                    f"compare {_humanize_metric(m1)} and {_humanize_metric(m2)} "
                    f"for {_ds(row)} in {year}",
                )
            )
            used.add(slug)
            break

        # 3. ANNUAL TOTAL — different time-series dataset; summable
        # metric only, with multi-month coverage in the picked year.
        # ``Annual`` is in the name; rolling up a single month and
        # calling it the annual total is exactly the misleading answer
        # the density gate is here to prevent.
        for row in by_recency:
            slug = _row_slug(row)
            if not slug or slug in used or not _ts(row):
                continue
            nums = [
                m for m in (row.get("key_metrics") or [])
                if _metric_is_summable(m)
            ]
            if not nums:
                continue
            year = self._pick_data_dense_year(
                slug, _normalize_years(row), require_monthly_spread=True
            )
            if year is None:
                continue
            non_total = [m for m in nums if "total" not in m.lower()]
            metric = non_total[0] if non_total else nums[0]
            leads.append(
                (
                    "Annual total",
                    f"total {_humanize_metric(metric)} for {_ds(row)} in {year}",
                )
            )
            used.add(slug)
            break

        # 4. DEFINE — anchor the metric to its host dataset using the
        # ``explain ... for ...`` phrasing. ``what does X mean?`` looks
        # natural in English but the router treats ``mean`` as the
        # ``avg`` keyword (see ``_AGG_KEYWORDS``) and, with no dataset
        # anchor, picks the wrong target by token overlap. ``explain``
        # is a semantic keyword, so the question lands on the glossary
        # lookup scoped to the named dataset.
        for row in catalog:
            slug = _row_slug(row)
            if not slug or slug in used:
                continue
            metrics = [
                m for m in (row.get("key_metrics") or [])
                if isinstance(m, str) and m
            ]
            if not metrics:
                continue
            if not self._facts_has_data(slug):
                continue
            metric = next(
                (m for m in metrics if _metric_is_numeric(m)), metrics[0]
            )
            leads.append(
                (
                    "Define a metric",
                    f"explain {_humanize_metric(metric)} for {_ds(row)}",
                )
            )
            used.add(slug)
            break

        # 5. REGISTRY — directory dataset, asked as a row-count question
        # so it executes cleanly. The old ``list datasets in <X> in
        # <year>`` phrasing collided with the catalog-intent matcher and
        # forced a ``sum`` aggregation against a collection with no
        # summable metric, producing the ``Aggregation 'sum' requires a
        # metric field`` failure. ``how many <X> are there in <year>``
        # maps to ``count`` (no metric required) and answers the natural
        # browsing question.
        for row in by_recency:
            slug = _row_slug(row)
            if not slug or slug in used or not _dir(row):
                continue
            year = self._pick_data_dense_year(
                slug, _normalize_years(row), require_monthly_spread=False
            )
            if year is None:
                continue
            leads.append(
                (
                    "Browse a registry",
                    f"how many {_ds(row).lower()} are there in {year}?",
                )
            )
            used.add(slug)
            break

        return leads

    def _fetch_catalog(self) -> list[dict[str, Any]]:
        """One-shot Mongo read of ``awqaf_datasets_metadata``.

        Returned rows feed both the compact summary and (on opt-in) the
        long department-grouped catalog. Caching the *rows* rather than
        the rendered markdown keeps the two renderers in sync without
        needing per-mode cache keys.
        """
        try:
            return self.mongo.find(
                _AWQAF_METADATA_COLLECTION,
                projection={
                    "_id": 0,
                    "dataset": 1,
                    "dataset_name": 1,
                    "department": 1,
                    "purpose": 1,
                    "key_metrics": 1,
                    "years": 1,
                    "data_type": 1,
                    "granularity": 1,
                    "coverage": 1,
                },
                limit=200,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Discovery: could not read {_AWQAF_METADATA_COLLECTION} "
                f"({type(exc).__name__}: {exc})"
            )
            return []

    # ------------------------------------------------------------------
    # Catalog-anchor probe (used by the second-chance guard in answer())
    # ------------------------------------------------------------------
    def _build_catalog_anchors(self) -> list[str]:
        """Return a deduplicated list of substrings that anchor a question
        to a real dataset / metric in ``awqaf_datasets_metadata``.

        Anchors come in three layers, listed longest-first so the matcher
        prefers the most specific anchor:

        * dataset slugs (``occupancy-rates-and-revenues`` and the
          underscore form), dataset display names (``occupancy rates and
          revenues``), and full key-metric names in both ``-`` / ``_`` /
          space forms (``total_revenues_collected_aed``);
        * 2-token sub-windows of metric names (``occupancy rate`` from
          ``occupancy_rate_pct``) so a user does not have to know the
          full schema field name;
        * single tokens of length ≥ ``_CATALOG_ANCHOR_MIN_TOKEN_LEN``
          that are not structural noise — catches words like
          ``occupancy``, ``revenues``, ``transactions``, ``disbursement``.
        """
        anchors: set[str] = set()

        catalog = self._fetch_catalog()

        def _add_token_windows(tokens: list[str]) -> None:
            tokens = [t for t in tokens if t]
            if not tokens:
                return
            # Full phrase, in space and underscore forms.
            anchors.add(" ".join(tokens))
            anchors.add("_".join(tokens))
            anchors.add("-".join(tokens))
            # 2+ token windows.
            for size in range(2, len(tokens)):
                for i in range(0, len(tokens) - size + 1):
                    window = tokens[i : i + size]
                    anchors.add(" ".join(window))
                    anchors.add("_".join(window))
            # Single tokens, length-gated and structural-filtered.
            for tok in tokens:
                if (
                    len(tok) >= _CATALOG_ANCHOR_MIN_TOKEN_LEN
                    and tok not in _STRUCTURAL_TOKENS
                ):
                    anchors.add(tok)

        for row in catalog or []:
            slug = str(row.get("dataset") or "").strip().lower()
            if slug:
                anchors.add(slug)
                anchors.add(slug.replace("_", "-"))
                anchors.add(slug.replace("-", "_"))
                _add_token_windows(re.split(r"[-_]+", slug))

            name = str(row.get("dataset_name") or "").strip().lower()
            if name:
                anchors.add(name)
                _add_token_windows(re.split(r"\s+", name))

            for metric in row.get("key_metrics") or []:
                if isinstance(metric, dict):
                    metric_name = (
                        metric.get("name") or metric.get("metric") or ""
                    )
                else:
                    metric_name = metric
                metric_name = str(metric_name or "").strip().lower()
                if not metric_name:
                    continue
                anchors.add(metric_name)
                anchors.add(metric_name.replace("_", " "))
                anchors.add(metric_name.replace("-", " "))
                _add_token_windows(re.split(r"[-_\s]+", metric_name))

        # Drop anything trivially short — short anchors collide with
        # common English words ("data", "year") and would defeat the
        # whole point of the probe.
        return sorted(
            (a for a in anchors if len(a) >= _CATALOG_ANCHOR_MIN_TOKEN_LEN),
            key=lambda s: (-len(s), s),
        )

    def _catalog_anchor_in_question(self, question: str) -> str | None:
        """Return the first catalog anchor present in ``question``.

        Substring match is case-insensitive against a precomputed (TTL
        cached) anchor list. The longest anchor is checked first so that
        ``occupancy rates and revenues`` wins over ``occupancy``.
        """
        q = (question or "").lower()
        if not q.strip():
            return None
        try:
            anchors = self._catalog_anchor_cache.get_or_set(
                "anchors", self._build_catalog_anchors
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Catalog-anchor probe: could not build anchors "
                f"({type(exc).__name__}: {exc})"
            )
            return None
        for anchor in anchors:
            if anchor and anchor in q:
                return anchor
        return None

    def _handle_out_of_scope(
        self, request: QueryRequest, t0: float
    ) -> AnalystResponse:
        """Politely redirect chatter that has nothing to do with the data."""
        insight = (
            "**Summary**\n"
            "I'm the AWQAF data analyst — I answer questions about the "
            "AWQAF datasets (Hajj services, Zakat, mosques, Quran centers, "
            "Umrah campaigns, fatwas, and related directories). The "
            "question you asked doesn't look like it's about any of those, "
            "so I'd rather not guess.\n\n"
            "**What I can answer**\n"
            "- _What can I ask?_  → I'll list the available datasets.\n"
            "- _Monthly total transactions for hajj-package-service in 2025_\n"
            "- _Compare smart app and website transactions for "
            "hajj-permit-service in 2024_\n"
            "- _Zakat disbursement by category in 2024_"
        )
        return self._build_static_response(
            request=request,
            insight=insight,
            t0=t0,
            reason="out-of-scope: no domain tokens",
            route=QueryRoute.OUT_OF_SCOPE,
            data_source=None,
        )

    def _build_static_response(
        self,
        *,
        request: QueryRequest,
        insight: str,
        t0: float,
        reason: str,
        route: QueryRoute,
        data_source: DataSource | None = None,
    ) -> AnalystResponse:
        """Wrap a deterministic insight in an empty-but-valid AnalystResponse.

        Each short-circuit handler stamps its own ``route`` (and optional
        ``data_source``) so the UI badge tells the truth. The default
        ``data_source=None`` is used by handlers that touch no store.
        """
        decision = RoutingDecision(
            route=route,
            data_source=data_source if data_source is not None else DataSource.MONGO,
            target=None,
            reason=reason,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # Short-circuit responses (discovery / out-of-scope) bypass the
        # router and the LLM. Discovery still reads `awqaf_datasets_metadata`
        # and probes facts collections for lead validation, so claiming
        # "no DB" here would be a lie when the elapsed time goes up.
        logger.info(
            f"Answered question in {elapsed_ms}ms — {reason} "
            f"(no router, no LLM)."
        )
        return AnalystResponse(
            question=request.question,
            routing=decision,
            structured_data=[],
            vector_context=[],
            insight=insight,
            chart=None,
            trust=None,
            warnings=[],
            provenance=None,
            meta={
                "elapsed_ms": elapsed_ms,
                "rows": 0,
                "vector_hits": 0,
                "warnings": 0,
                "session_id": request.session_id,
                "intent": reason.split(":", 1)[0],
            },
        )

    # ------------------------------------------------------------------
    # Critic findings → response meta (Build #7)
    # ------------------------------------------------------------------
    def _collect_critic_meta(self) -> dict[str, Any] | None:
        """Pull the most recent critic decision off the contextvar.

        Returns ``None`` when no critic ran in this request scope (the
        feature is off, or the LLM was short-circuited by no_target /
        zero_rows / missing_metric branches). Returns a small dict
        otherwise so the user can see "the agent self-verified and
        flagged N issue(s)" without us bloating the response when
        there's nothing to report.
        """
        from services.agent_service import get_last_critic_decision

        decision = get_last_critic_decision()
        if decision is None:
            return None
        # Reset for the next request — the contextvar is request-scoped
        # via the trace context, but explicit clear avoids surprise if
        # a caller invokes the orchestrator twice in the same scope.
        from services.agent_service import set_last_critic_decision

        set_last_critic_decision(None)

        # Convert to plain dicts so the response stays JSON-serialisable
        # without forcing the orchestrator to import critic_service.
        action = getattr(decision, "action", "approve")
        issues = getattr(decision, "issues", []) or []
        out = {
            "action": action,
            "issue_count": len(issues),
            "summary": getattr(decision, "summary", "")
            or getattr(decision, "reasoning", ""),
        }
        if issues:
            out["issues"] = [
                {
                    "severity": getattr(i, "severity", "low"),
                    "type": getattr(i, "type", "other"),
                    "quote": getattr(i, "quote", ""),
                    "evidence": getattr(i, "evidence", ""),
                    "suggested_fix": getattr(i, "suggested_fix", ""),
                }
                for i in issues
            ]
        return out

    # ------------------------------------------------------------------
    # Session-aware routing
    # ------------------------------------------------------------------
    def _load_conversation_memory(
        self, request: QueryRequest
    ) -> str | None:
        """Render the running summary + recent verbatim turns for the prompt.

        Returns ``None`` when:
        * the feature is disabled, OR
        * the request has no ``session_id``, OR
        * the session has no recorded turns yet.

        The returned string mixes the compressed summary ("everything
        older than the last few turns") with the verbatim recent turns
        ("what was just discussed"), which is the standard
        ConversationSummaryBuffer shape. The model thus gets gist +
        details together.
        """
        if not settings.session_summary_enabled:
            return None
        if not request.session_id:
            return None
        summary, turns = self.sessions.get_history(request.session_id)
        if not summary and not turns:
            return None

        sections: list[str] = []
        if summary:
            sections.append(
                "Summary of older turns:\n" + summary.strip()
            )
        if turns:
            tail_lines = []
            for i, t in enumerate(turns, start=1):
                tail_lines.append(
                    f"[Recent turn {i}]\n"
                    f"USER: {t.question.strip()}\n"
                    f"ANALYST: {t.short_insight()}"
                )
            sections.append(
                "Most recent turns (verbatim):\n" + "\n\n".join(tail_lines)
            )
        return "\n\n".join(sections)

    def _persist_conversation_turn(
        self,
        request: QueryRequest,
        insight: str,
    ) -> None:
        """Append the (question, insight) pair and compress when full.

        Compression policy lives here (not in :class:`SessionService`)
        because it depends on settings the storage layer should not
        know about. The trace context is already active at this point
        — the summariser will record its tokens and timing under a
        ``session.summarise`` span automatically.
        """
        if not settings.session_summary_enabled:
            return
        if not request.session_id:
            return
        size = self.sessions.add_turn(
            request.session_id, request.question, insight
        )
        trigger_at = settings.session_summary_trigger_at
        if not ConversationSummariser.should_compress(
            size, trigger_at=trigger_at
        ):
            return
        keep_last = settings.session_summary_keep_last
        prior_summary, all_turns = self.sessions.get_history(
            request.session_id
        )
        if not all_turns:
            return
        # Fold everything except the fresh tail into the summary. If
        # ``keep_last`` is 0, the entire buffer becomes summary fodder.
        if keep_last >= len(all_turns):
            old_turns: list[Turn] = []
        elif keep_last == 0:
            old_turns = list(all_turns)
        else:
            old_turns = list(all_turns[:-keep_last])
        if not old_turns:
            return
        try:
            new_summary = self.summariser.compress(
                old_turns=old_turns,
                prior_summary=prior_summary,
            )
        except Exception as exc:  # noqa: BLE001
            # The summariser already degrades gracefully internally,
            # but we still guard against unexpected failures here so a
            # broken compressor never poisons a successful answer.
            logger.warning(
                f"Conversation summariser raised unexpectedly "
                f"({type(exc).__name__}: {exc}); leaving prior summary intact."
            )
            return
        self.sessions.set_summary(
            request.session_id, new_summary, keep_last=keep_last
        )
        logger.info(
            f"Compressed {len(old_turns)} turns into running summary "
            f"(session={request.session_id!s}, kept_last={keep_last})"
        )

    def _resolve_followup(
        self, request: QueryRequest
    ) -> tuple[RoutingDecision, bool] | None:
        """Stateful, structural follow-up resolver.

        Two preconditions for a follow-up route:

        1. An active session has a *prior analytical decision* worth
           patching (semantic-only or discovery sessions are skipped).
        2. The structural classifier
           (:func:`services.session_patch.analyze_followup`) reports
           refinement signals on the new utterance AND the
           catalog-anchor probe does NOT match — i.e. the user did
           not name a fresh dataset or metric.

        On a positive verdict we return ``(patched_decision, True)``
        so the caller short-circuits intent classification entirely.
        ``None`` means "no follow-up signal; route fresh".
        """
        prior = self.sessions.get(request.session_id)
        if prior is None:
            return None
        # Always reach for the most recent *analytical* decision in
        # the session — not just the immediate prior turn. A user can
        # ask one discovery / semantic question mid-thread without
        # losing the analytical context the follow-up depends on.
        # When no analytical decision exists yet we fall back to the
        # immediate prior so the first follow-up after an analytical
        # turn still works.
        prior_decision = prior.last_analytical_decision or prior.last_decision
        if prior_decision is None or prior_decision.route != QueryRoute.ANALYTICAL:
            return None
        if prior_decision.aggregation is None:
            return None

        # Pull the prior target's numeric column list once. The
        # snapshot has two consumers in this method:
        #
        #   * ``analyze_followup`` uses it to elevate METRIC swap from
        #     "any 1-3 word noun" to "a noun that names a real
        #     column on the prior target".
        #   * ``resolve_followup`` uses it to gate the metric-swap
        #     patcher so we never point the executor at a
        #     nonexistent column.
        #
        # Failure is silent: if the probe errors we treat the schema
        # as unknown and skip metric-swap patching, but every other
        # refinement still applies.
        try:
            schema_columns: tuple[str, ...] = tuple(
                self._sample_numeric_columns(prior_decision.target or "")
            )
        except Exception:  # noqa: BLE001
            schema_columns = ()

        # Run the catalog-anchor probe on the question with refinement
        # payloads masked. Without masking, ``by emirate`` looks like a
        # standalone reference to a catalog metric named ``emirate``;
        # masked, the probe sees ``by`` and correctly returns no
        # anchor — preserving the follow-up shape. Passing
        # ``schema_columns`` also masks known column names so a
        # METRIC-swap follow-up like ``show hajj package recipients``
        # doesn't get blocked by the column sharing tokens with the
        # dataset slug. This is what makes the resolver structural
        # rather than keyword-driven.
        anchor_probe_text = mask_refinement_patterns(
            request.question, schema_columns=schema_columns or None
        )
        anchor = self._catalog_anchor_in_question(anchor_probe_text)
        analysis: FollowUpAnalysis = analyze_followup(
            request.question,
            has_catalog_anchor=anchor is not None,
            has_prior_decision=True,
            schema_columns=schema_columns or None,
        )
        if not analysis.is_followup:
            return None

        patched = resolve_followup(
            question=request.question,
            prior_decision=prior_decision,
            analysis=analysis,
            schema_columns=schema_columns,
            llm_resolver=self._followup_llm_resolver(),
        )
        logger.info(
            f"Follow-up detected ({analysis.reason}, "
            f"confidence={analysis.confidence:.2f}); "
            f"patched target=`{patched.target}` "
            f"compare={patched.aggregation.time.compare.value if patched.aggregation and patched.aggregation.time else 'n/a'}"
        )
        return patched, True

    def _followup_llm_resolver(self):
        """Return an LLM-driven follow-up resolver, or ``None`` when off.

        Layer 2 of the production design: when the deterministic
        structural patcher is uncertain (low confidence), an LLM
        co-reference resolver can take over. Default is off — the
        Layer 1 patcher covers every refinement shape we ship today
        and stays sub-millisecond. Wiring is here so the hook can be
        enabled without touching the orchestrator.
        """
        if not getattr(settings, "followup_llm_enabled", False):
            return None
        # Intentional placeholder — production wiring would invoke a
        # small structured-output LLM call (e.g. Claude Haiku) with
        # the prior decision (as JSON), the new utterance, and a
        # schema that returns a patched ``AggregationSpec``. The
        # patcher would then merge the LLM patch onto the prior
        # decision. Kept as a hook here so the orchestrator does not
        # need to change when the resolver lands.
        return None

    @staticmethod
    def _followup_implies_comparison(decision: RoutingDecision) -> bool:
        """True when the patched decision carries an explicit comparison axis.

        Used to set the ``comparison_intent`` flag that drives the
        multi-metric / multi-window executor. We trust the patched
        spec rather than re-classifying the original utterance: by
        construction the patcher only sets ``compare=YOY/PREV_PERIOD``
        when the follow-up actually contained a comparison verb or
        target year.
        """
        spec = decision.aggregation
        if spec is None or spec.time is None:
            return False
        return spec.time.compare != ComparisonMode.NONE

    def _decide_with_session(
        self, request: QueryRequest
    ) -> tuple[RoutingDecision, bool]:
        prior = self.sessions.get(request.session_id)
        if (
            prior is not None
            and not request.collection
            and should_reuse_prior_collection(
                request.question,
                prior_target=prior.last_decision.target,
                prior_route=prior.last_decision.route,
            )
        ):
            d0 = prior.last_decision
            ds = d0.data_source
            if ds == DataSource.AUTO:
                t = d0.target or ""
                ds = DataSource.MONGO if t.startswith("awqaf_") else DataSource.MYSQL
            decision = self.router.decide(
                request.question,
                collection=d0.target,
                data_source=ds,
            )
            decision.matched_keywords = list(decision.matched_keywords) + [
                "session-continue-target",
            ]
            decision.reason = (
                (decision.reason or "")
                + f" Scoped to prior target `{d0.target}` from session context."
            ).strip()
            return decision, True
        decision = self.router.decide(
            request.question,
            collection=request.collection,
            data_source=request.data_source,
        )
        return decision, False

    # ------------------------------------------------------------------
    # Pipeline pieces
    # ------------------------------------------------------------------
    def _run_analytical(
        self,
        decision: RoutingDecision,
        *,
        question: str = "",
        comparison: bool = False,
    ) -> tuple[list[dict[str, Any]], list[AnalystWarning]]:
        if decision.route == QueryRoute.SEMANTIC or decision.aggregation is None:
            return [], []

        target = decision.target
        if not target:
            logger.warning(
                "Analytical route chosen but no target collection/table available"
            )
            return [], [
                AnalystWarning(
                    code=WarningCode.TARGET_AMBIGUOUS,
                    message=(
                        "No data target could be selected. Please specify a "
                        "collection/table, or ingest data first."
                    ),
                )
            ]

        # Plan-validation gate (Mongo only — MySQL has its own SQL builder
        # and richer error reporting). The validator catches:
        #   * targets that look analytical but are catalog/glossary
        #     collections (refuse cleanly with a helpful message);
        #   * specs whose metric / group_by / time field is not on the
        #     target schema (downgrade or refuse before Mongo crashes
        #     with an opaque ``$dateToString`` / similar pipeline error).
        # The decision is replaced with the validator's (possibly
        # downgraded) version so all downstream code — chart titling,
        # trust panel, LLM context — sees the spec we are actually
        # going to execute.
        validation_warnings: list[AnalystWarning] = []
        if decision.data_source != DataSource.MYSQL:
            columns, sample = self._target_schema_snapshot(target)
            verdict = self.validator.validate(
                decision, columns=columns, sample=sample
            )
            for note in verdict.notes:
                logger.info(f"Plan validator [{target}]: {note}")
            validation_warnings.extend(verdict.warnings)
            if not verdict.should_execute:
                return [], validation_warnings
            if verdict.decision is not decision:
                # Mutate the caller's decision in place so downstream
                # services (trust panel, chart titling, partial-relevance
                # warning) operate on the executed plan, not the original.
                decision.aggregation = verdict.decision.aggregation

        # Multi-year comparison: when the question mentions multiple years
        # ("2026 and 2025"), run one aggregation per year and tag rows with
        # the year in the series field so they can be visualized separately.
        spec = decision.aggregation
        if (
            spec
            and spec.time
            and spec.time.years
            and len(spec.time.years) >= 2
            and decision.data_source != DataSource.MYSQL
        ):
            multi_year_rows, multi_year_warnings = self._run_multi_year_comparison(
                target, spec, spec.time.years
            )
            if multi_year_rows is not None:
                return multi_year_rows, validation_warnings + multi_year_warnings

        # Multi-metric comparison fan-out: when the question explicitly
        # asks for a comparison and names 2+ available metrics, run one
        # aggregation per metric and tag rows with ``series`` so the LLM
        # (and a future multi-series chart layer) can present them together.
        if comparison and decision.data_source != DataSource.MYSQL:
            multi_rows, multi_warnings = self._run_multi_metric_comparison(
                target, decision.aggregation, question
            )
            if multi_rows is not None:
                return multi_rows, validation_warnings + multi_warnings

        # Vague-metric expansion: when the question names a dataset/service
        # ("trend for hajj package service") but no specific metric, the
        # router falls back to its default metric pick. Without this
        # branch the user only sees that one metric; with it, the agent
        # fans across every numeric metric on the target so the chart
        # panel can render all of them as series.
        if (
            not comparison
            and decision.data_source != DataSource.MYSQL
            and self._is_vague_metric_question(
                question, target, decision.aggregation
            )
        ):
            fanned_rows, fan_warnings = self._run_all_metrics_fan_out(
                target, decision.aggregation, question
            )
            if fanned_rows:
                logger.info(
                    f"Vague-metric question detected; fanned out across "
                    f"all numeric metrics on {target}."
                )
                return fanned_rows, validation_warnings + fan_warnings

        try:
            if decision.data_source == DataSource.MYSQL:
                rows = self.mysql.run_aggregation(target, decision.aggregation)
            else:
                rows = self.mongo.run_aggregation(target, decision.aggregation)
            return rows, validation_warnings
        except DatabaseError as exc:
            logger.error(f"Aggregation failed: {exc}")
        except AIAnalystError as exc:
            logger.error(f"Aggregation error: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected aggregation error: {exc}")

        return [], validation_warnings + [
            AnalystWarning(
                code=WarningCode.EMPTY_RESULT,
                message=(
                    "The analytical query failed to execute. "
                    "Showing the qualitative context only."
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Vague-metric fan-out
    # ------------------------------------------------------------------
    # Cap on how many series to fan out across at once. Above this we
    # prefer the default single-metric path — wide multi-series charts
    # (8+ traces) are unreadable and the panel donut becomes a wheel of
    # confetti slices. Six metrics covers every dataset we ship today
    # (Hajj campaigns, Hajj package service, fatwa, …) with headroom.
    _VAGUE_FAN_OUT_MAX_SERIES: int = 6

    def _is_vague_metric_question(
        self,
        question: str,
        target: str,
        spec: AggregationSpec | None,
    ) -> bool:
        """True when the user named a dataset but no specific metric.

        The router still has to pick *some* metric to execute against
        — typically the most generic numeric column ("total_…", or the
        first numeric field). When that pick happens silently the user
        only ever sees one of N available metrics, which is what
        prompted this expansion.

        We detect "vague" by running the same metric-name extractor the
        comparison fan-out uses and returning True when it finds zero
        matches. We also gate on having 2+ numeric columns available —
        a single-metric dataset has nothing to fan across.
        """
        if spec is None:
            return False
        columns = self._sample_numeric_columns(target)
        if len(columns) < 2:
            return False
        named = extract_known_columns(question, columns)
        if not named:
            target_slug = _dataset_slug_from_target(target) or target
            named = extract_compared_columns(
                question, columns, target_slug=target_slug
            )
        return len(named) == 0

    def _run_all_metrics_fan_out(
        self,
        target: str,
        spec: AggregationSpec,
        question: str,
    ) -> tuple[list[dict[str, Any]], list[AnalystWarning]]:
        """
        Fan an aggregation out across every numeric metric on ``target``.

        Same row shape as :meth:`_run_multi_metric_comparison` (each
        result row carries ``series`` and ``series_label``) so the
        downstream multi-series chart code path is reused unchanged.
        
        IMPORTANT: Preserves dimensional information from Mongo aggregation
        by NOT overwriting the ``series`` field when dimensions exist.
        """
        columns = self._sample_numeric_columns(target)
        if len(columns) < 2:
            return [], []
        
        # Cap to keep multi-series charts readable. The router's
        # default metric is kept first so the primary view still
        # matches what the user would have seen pre-fan-out.
        default_metric = spec.metric
        ordered: list[str] = []
        if default_metric and default_metric in columns:
            ordered.append(default_metric)
        for col in columns:
            if col not in ordered:
                ordered.append(col)
        ordered = ordered[: self._VAGUE_FAN_OUT_MAX_SERIES]

        all_rows: list[dict[str, Any]] = []
        warnings: list[AnalystWarning] = []
        
        for metric in ordered:
            metric_spec = spec.model_copy(update={"metric": metric})
            
            # Auto-select group_by for sparse temporal datasets
            # (same logic as multi-metric comparison)
            if metric_spec.group_by is None and metric_spec.time is not None:
                schema_columns = self._target_schema_snapshot(target)[0]
                auto_group = self._auto_dimension_field(schema_columns)
                if auto_group:
                    logger.info(
                        f"Vague fan-out: auto-selected grouping `{auto_group}` "
                        f"for metric `{metric}`"
                    )
                    metric_spec = metric_spec.model_copy(update={"group_by": auto_group})
            
            try:
                rows = self.mongo.run_aggregation(target, metric_spec)
            except (DatabaseError, AIAnalystError) as exc:
                logger.warning(
                    f"Fan-out series for `{metric}` skipped "
                    f"({type(exc).__name__}: {exc})"
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Fan-out series for `{metric}` skipped due to "
                    f"unexpected error ({type(exc).__name__}: {exc})"
                )
                continue
            
            for row in rows:
                row = dict(row)
                
                # ─────────────────────────────────────────────────────────
                # PRESERVE DIMENSIONAL INFORMATION
                #
                # Same fix as _run_multi_metric_comparison:
                # - Preserve Mongo's series field (contains dimensions)
                # - Add metric metadata separately
                # - Let transformation logic decide final row structure
                # ─────────────────────────────────────────────────────────
                existing_series = row.get("series")
                if existing_series is not None:
                    row["dimension"] = existing_series
                
                # Add metric metadata
                row["metric"] = metric
                row["metric_label"] = self._human_label(metric).title()
                
                # CRITICAL FIX: When there are no meaningful dimensions
                # (series is "(unknown)"), use the metric as the series
                # so multi-metric visualization works correctly
                if existing_series in (None, "(unknown)", "(all)"):
                    row["series"] = metric
                    row["series_label"] = row["metric_label"]
                else:
                    # When there are multiple metrics AND dimensions,
                    # combine both in the series label for clarity
                    if len(ordered) > 1:
                        # Multi-metric + multi-dimension: "Channel (Metric)"
                        row["series"] = f"{existing_series} ({row['metric_label']})"
                        row["series_label"] = row["series"]
                    else:
                        # Single metric, multiple dimensions: just use dimension name
                        row["series_label"] = row["series"]
                
                all_rows.append(row)

        if not all_rows:
            warnings.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message=(
                        "Could not retrieve data for any of the available "
                        "metrics on this dataset."
                    ),
                )
            )
            return [], warnings
        
        logger.info(
            f"Vague fan-out across {len(ordered)} metrics on {target} "
            f"produced {len(all_rows)} rows."
        )
        return all_rows, warnings

    def _run_multi_metric_comparison(
        self,
        target: str,
        spec: AggregationSpec,
        question: str,
    ) -> tuple[list[dict[str, Any]] | None, list[AnalystWarning]]:
        """Fan an aggregation out across the metrics named in the question.

        Returns ``(rows, warnings)`` on success and ``(None, [])`` when fewer
        than two distinct metrics were named — in which case the caller
        falls back to the regular single-metric path.
        """

        columns = self._sample_numeric_columns(target)

        # First try exact phrase match.
        named = extract_known_columns(question, columns)

        # Fallback relaxed matcher.
        if len(named) < 2:
            target_slug = _dataset_slug_from_target(target) or target

            named = extract_compared_columns(
                question,
                columns,
                target_slug=target_slug,
            )

        if len(named) < 2:
            return None, []

        logger.info(
            f"Multi-metric comparison detected on {target}: "
            f"{named}"
        )

        all_rows: list[dict[str, Any]] = []
        warnings: list[AnalystWarning] = []

        for metric in named:

            metric_spec = spec.model_copy(
                update={
                    "metric": metric,
                }
            )

            # ─────────────────────────────────────────────
            # Sparse trend intelligence:
            #
            # If user asked for trend/time analysis but
            # no explicit group_by exists, automatically
            # preserve available categorical dimensions.
            #
            # Example:
            #   "trend for occupancy and revenue in 2026"
            #
            # becomes:
            #
            #   time = period
            #   group_by = emirate/dimension
            #
            # so charts don't collapse into meaningless
            # single-bucket trend bars.
            # ─────────────────────────────────────────────
            if (
                metric_spec.group_by is None
                and metric_spec.time is not None
            ):

                schema_columns = self._target_schema_snapshot(
                    target
                )[0]

                auto_group = self._auto_dimension_field(
                    schema_columns
                )

                if auto_group:

                    logger.info(
                        f"Auto-selected grouping "
                        f"`{auto_group}` "
                        f"for comparison metric `{metric}`"
                    )

                    metric_spec = metric_spec.model_copy(
                        update={
                            "group_by": auto_group
                        }
                    )

            logger.info(
                f"Running comparison aggregation: "
                f"metric={metric_spec.metric}, "
                f"group_by={metric_spec.group_by}, "
                f"time={metric_spec.time}"
            )

            try:
                rows = self.mongo.run_aggregation(
                    target,
                    metric_spec,
                )

            except (DatabaseError, AIAnalystError) as exc:

                logger.warning(
                    f"Comparison series for `{metric}` skipped "
                    f"({type(exc).__name__}: {exc})"
                )

                continue

            except Exception as exc:  # noqa: BLE001

                logger.warning(
                    f"Comparison series for `{metric}` skipped due to "
                    f"unexpected error ({type(exc).__name__}: {exc})"
                )

                continue

            logger.info(
                f"Comparison aggregation returned "
                f"{len(rows)} rows for metric `{metric}`"
            )

            for row in rows:

                row = dict(row)

                # ─────────────────────────────────────────────────────────
                # PRESERVE DIMENSIONAL INFORMATION
                #
                # Mongo multi-series aggregation (time + group_by) returns:
                #
                # {
                #   "label": "2026-01",      # time bucket
                #   "series": "Fujairah",    # dimensional group (emirate)
                #   "value": 480681.67
                # }
                #
                # We MUST preserve "Fujairah" as the dimensional grouping
                # axis for proper categorical visualization.
                # ─────────────────────────────────────────────────────────
                existing_series = row.get("series")
                if existing_series is not None:
                    row["dimension"] = existing_series

                # ─────────────────────────────────────────────────────────
                # ADD METRIC METADATA FOR MULTI-METRIC COMPARISON
                #
                # Store metric information in dedicated fields so the
                # transformation layer can decide whether to use dimension
                # or metric as the chart grouping axis based on analytical
                # shape (sparse temporal vs. strong trend).
                # ─────────────────────────────────────────────────────────
                row["metric"] = metric
                row["metric_label"] = self._human_label(metric).title()

                # Note: The "series" field will be set by the transformation
                # layer based on whether this is a sparse temporal dataset
                # (dimension → label, metric → series) or a strong trend
                # (time → label, dimension → series, metric → metadata).

                all_rows.append(row)

        if not all_rows:

            warnings.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message="All comparison series returned no data.",
                )
            )

            return [], warnings

        logger.info(
            f"Comparison fan-out across {len(named)} metrics "
            f"on {target} produced {len(all_rows)} rows."
        )

        logger.info(
            f"Comparison rows sample: {all_rows[:5]}"
        )

        return all_rows, warnings

    def _run_multi_year_comparison(
        self,
        target: str,
        spec: AggregationSpec,
        years: list[int],
    ) -> tuple[list[dict[str, Any]] | None, list[AnalystWarning]]:
        """Execute aggregation for multiple years and tag each row with its year.
        
        When a user asks "trends for 2026 and 2025", this method:
        1. Queries each year separately
        2. Tags rows with year in the series field (e.g., "Total Transactions (2026)")
        3. Combines all results for multi-year visualization
        
        Returns:
            (rows, warnings) on success, (None, []) if no data for any year
        """
        all_rows: list[dict[str, Any]] = []
        warnings: list[AnalystWarning] = []
        available_years: list[int] = []
        missing_years: list[int] = []
        
        # Check which years actually have data
        catalog_years = self._catalog_years_for_target(target)
        for year in years:
            if year in catalog_years:
                available_years.append(year)
            else:
                missing_years.append(year)
        
        if not available_years:
            # No requested years have data
            warnings.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message=(
                        f"No data available for the requested years: {', '.join(map(str, years))}. "
                        f"Available years: {', '.join(map(str, catalog_years)) if catalog_years else 'none'}."
                    ),
                )
            )
            return None, warnings
        
        if missing_years:
            # Some years are missing
            warnings.append(
                AnalystWarning(
                    code=WarningCode.PARTIAL_PERIOD,
                    message=(
                        f"Data not available for: {', '.join(map(str, missing_years))}. "
                        f"Showing data for: {', '.join(map(str, available_years))}."
                    ),
                )
            )
        
        logger.info(
            f"Multi-year comparison on {target}: querying years {available_years}"
        )
        
        # Get all numeric metrics for multi-metric fan-out
        all_metrics = self._sample_numeric_columns(target)
        
        # Fan out across all metrics for multi-year queries
        metrics_to_query = all_metrics if len(all_metrics) > 1 else [spec.metric]
        
        logger.info(
            f"Fanning out across {len(metrics_to_query)} metrics for each year"
        )
        
        # Query each year × each metric
        for year in available_years:
            for metric in metrics_to_query:
                # Create year-specific, metric-specific spec
                year_metric_spec = spec.model_copy(
                    update={
                        "metric": metric,
                        "time": TimeSpec(
                            field=spec.time.field if spec.time else "period",
                            bucket=spec.time.bucket if spec.time else TimeBucket.MONTH,
                            range_from=datetime(year, 1, 1, tzinfo=timezone.utc),
                            range_to=datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                            compare=ComparisonMode.NONE,
                            years=None,  # Clear multi-year flag for individual query
                        )
                    }
                )
                
                try:
                    rows = self.mongo.run_aggregation(target, year_metric_spec)
                    
                    if not rows:
                        logger.debug(f"No data for {metric} in year {year}")
                        continue
                    
                    # Tag each row with metric and year
                    metric_label = _humanize_metric(metric)
                    
                    for row in rows:
                        # Set series field to include metric and year
                        row["series"] = f"{metric_label} ({year})"
                        row["series_label"] = row["series"]
                        row["year"] = year  # Add year as dimension
                        row["metric"] = metric
                        row["metric_label"] = metric_label.title()
                        all_rows.append(row)
                    
                    logger.info(f"Year {year}, metric {metric}: retrieved {len(rows)} rows")
                    
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to query {metric} for year {year}: {exc}")
                    # Don't add warning for individual metric failures
                    continue
        
        if not all_rows:
            warnings.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message="No data returned for any of the requested years.",
                )
            )
            return [], warnings
        
        logger.info(
            f"Multi-year comparison produced {len(all_rows)} total rows across "
            f"{len(available_years)} years"
        )
        
        return all_rows, warnings

    # ------------------------------------------------------------------
    # Partial-relevance detection
    # ------------------------------------------------------------------
    def _partial_relevance_warning(
        self, question: str, decision: RoutingDecision
    ) -> AnalystWarning | None:
        """Flag terms in the question that are not present in the routed dataset.

        We answer the available parts faithfully, but we also tell the user
        clearly what *parts* of the question we ignored. Catches the
        "transactions and revenue for hajj-permit in 2024" pattern, where
        ``revenue`` simply does not exist in this collection.
        """
        if decision.target is None:
            return None

        tokens = tokenize_question(question)
        if not tokens:
            return None

        target_tokens = set(decision.target.lower().split("_"))
        if decision.target.startswith("awqaf_"):
            target_tokens.discard("awqaf")
            target_tokens.discard("facts")

        spec_tokens: set[str] = set()
        if decision.aggregation:
            if decision.aggregation.metric:
                spec_tokens.update(decision.aggregation.metric.lower().split("_"))
            if decision.aggregation.group_by:
                spec_tokens.update(decision.aggregation.group_by.lower().split("_"))
            if decision.aggregation.time and decision.aggregation.time.field:
                spec_tokens.update(decision.aggregation.time.field.lower().split("_"))

        try:
            columns = self._sample_numeric_columns(decision.target)
        except Exception:  # noqa: BLE001
            columns = []
        column_tokens: set[str] = set()
        for col in columns:
            column_tokens.update(col.lower().split("_"))

        unrecognised: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            t = tok.lower()
            if t in seen or len(t) <= 2:
                continue
            seen.add(t)
            if t in _STRUCTURAL_TOKENS or t in _MONTH_TOKENS:
                continue
            if t.isdigit():
                continue
            if t in target_tokens or t in spec_tokens or t in column_tokens:
                continue
            unrecognised.append(t)

        if not unrecognised:
            return None

        return AnalystWarning(
            code=WarningCode.TARGET_AMBIGUOUS,
            message=(
                "Some terms in your question are not part of "
                f"`{decision.target}`: "
                + ", ".join(f"`{t}`" for t in unrecognised[:6])
                + ". The answer covers only the parts of the question that "
                "this dataset can serve."
            ),
        )

    def _compose_insight(
        self,
        *,
        request: QueryRequest,
        decision: RoutingDecision,
        context_block: str,
        structured_data: list[dict[str, Any]],
        warnings: list[AnalystWarning],
        empty_finding: _EmptyMetricFinding | None = None,
    ) -> str:
        """Short-circuit the LLM when we have no real data to ground on.

        Four short-circuit cases (in priority order):

        1. **Missing metric** — router located the collection but couldn't
           resolve a metric field (op is sum/avg/min/max but ``metric=None``).
           This is a *schema-mismatch* between user vocabulary and stored
           columns. We sample the collection, list the closest available
           numeric fields, and suggest concrete follow-up questions.
        2. **No target** — router couldn't even pick a collection. Tell the
           user honestly and ask them to name the dataset.
        3. **Empty metric (single-metric query)** — the requested metric IS
           in the schema but has no values for the requested scope (every
           row null/zero). Distinct from #1: the column exists; the data
           doesn't. We list related populated fields and suggest concrete
           follow-ups instead of charting a flat-zero series. Comparison
           queries are NOT short-circuited here — those keep the LLM in
           the loop with a prepended data-availability notice (handled in
           ``answer()``) so the populated metric still gets analyzed.
        4. **Zero rows** — query executed but returned nothing for the
           requested scope (typically a year/period filter with no data).

        In all four cases we skip the LLM call: it has no data to ground
        on, which is exactly when it tends to hallucinate.
        """
        analytical = decision.route in (QueryRoute.ANALYTICAL, QueryRoute.HYBRID)
        if not analytical:
            return self.agent.generate_insight(request.question, context_block)

        spec = decision.aggregation
        op = (spec.operation or "").lower() if spec else ""
        needs_metric = op in {"sum", "avg", "average", "min", "max"}
        # Treat "missing metric" as a *failure* state only when we also
        # have no rows. A successful comparison fan-out can produce rows
        # while leaving ``decision.aggregation.metric`` as ``None`` (each
        # series carries its own metric in the ``series`` column instead).
        missing_metric = bool(
            spec is not None
            and needs_metric
            and not spec.metric
            and decision.target
            and not structured_data
        )
        no_target = decision.target is None and spec is not None
        no_rows = not structured_data
        single_metric_empty = bool(
            empty_finding is not None
            and empty_finding.empty
            and not empty_finding.has_series
            and not empty_finding.populated
            and decision.target
        )

        if missing_metric:
            return self._missing_metric_response(request, decision, warnings)
        if no_target:
            return self._no_target_response(request, warnings)
        if single_metric_empty:
            assert empty_finding is not None  # narrow for the type checker
            return self._empty_metric_response(
                request, decision, empty_finding, warnings
            )
        if no_rows:
            return self._zero_rows_response(request, decision, warnings)

        return self.agent.generate_insight(request.question, context_block)

    # ------------------------------------------------------------------
    # Deterministic "what IS available" responses
    # ------------------------------------------------------------------
    def _sample_numeric_columns(self, target: str) -> list[str]:
        """Return the numeric metric columns currently stored in ``target``.

        Cached for ``_SCHEMA_CACHE_TTL`` seconds: within one analytical
        request the same target was previously sampled up to three times
        (comparison fan-out, partial-relevance check, missing-metric
        response). The cache collapses all of those to one Mongo round-trip
        and keeps memory bounded — one small ``list[str]`` per collection.
        """
        return self._schema_cache.get_or_set(
            target, lambda: self._fetch_numeric_columns(target)
        )

    def _dataset_definitions_text(
        self, decision: RoutingDecision
    ) -> str | None:
        """Cached glossary lookup for the routed dataset.

        Repeated questions on the same dataset previously caused one extra
        Mongo ``find_one`` per request, even though the glossary doc almost
        never changes between ingest cycles. Cached for
        ``_GLOSSARY_CACHE_TTL`` seconds keyed by ``(slug, picked_metric)``.
        """
        slug = _dataset_slug_from_target(decision.target)
        if not slug:
            return None
        spec_metric = (
            decision.aggregation.metric if decision.aggregation else None
        )
        key = f"{slug}|{spec_metric or ''}"
        return self._glossary_cache.get_or_set(
            key, lambda: self._fetch_definitions_text(slug, spec_metric)
        )

    def _fetch_definitions_text(
        self, slug: str, spec_metric: str | None
    ) -> str | None:
        try:
            doc = self.mongo.collection(_AWQAF_GLOSSARY_COLLECTION).find_one(
                {"dataset": slug}, {"_id": 0}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Glossary lookup for `{slug}` failed "
                f"({type(exc).__name__}: {exc})"
            )
            return None
        return _format_definitions_text(doc, spec_metric)

    def _fetch_numeric_columns(self, target: str) -> list[str]:
        try:
            sample = self.mongo.find(target, limit=20)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not sample {target} for alternatives "
                f"({type(exc).__name__}: {exc})"
            )
            return []
        numeric: dict[str, bool] = {}
        for doc in sample:
            for key, value in doc.items():
                if key in _ENVELOPE_FIELDS:
                    continue
                if key in numeric:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric[key] = True
        return sorted(numeric.keys())

    # ------------------------------------------------------------------
    # Schema snapshot (used by the plan validator)
    # ------------------------------------------------------------------
    def _target_schema_snapshot(
        self, target: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Return ``(columns, sample)`` for ``target`` (cached, TTL-bounded).

        Distinct from ``_sample_numeric_columns`` because the validator
        needs *all* columns plus the actual sample documents (to check
        whether a candidate time field is date-coercible). Both helpers
        sample the same Mongo collection, but cache separately so the
        validator's needs don't change the existing return shape used by
        comparison fan-out and partial-relevance detection.
        """
        return self._target_schema_cache.get_or_set(
            target, lambda: self._fetch_target_schema(target)
        )

    def _fetch_target_schema(
        self, target: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            sample = self.mongo.find(target, limit=20)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not sample {target} for plan validation "
                f"({type(exc).__name__}: {exc})"
            )
            return [], []
        cols = sorted({k for doc in sample for k in doc.keys()})
        return cols, sample

    @staticmethod
    def _rank_alternatives(
        columns: list[str], question: str, *, limit: int = 6
    ) -> list[str]:
        """Order ``columns`` by how well their tokens overlap the question.

        Token comparison is intentionally lenient: ``transaction`` (singular
        from the question) still matches ``transactions`` (plural in the
        column) via prefix/substring. Without this, "total transaction"
        would score zero against ``website_transactions`` and we'd fall back
        to alphabetical order, which is unhelpful.
        """
        if not columns:
            return []
        keywords = _question_keywords(question)
        if not keywords:
            return columns[:limit]
        scored: list[tuple[int, int, str]] = []
        for col in columns:
            col_tokens = col.lower().split("_")
            last_token = col_tokens[-1] if col_tokens else ""
            overlap = 0
            suffix_bonus = 0
            for kw in keywords:
                for ct in col_tokens:
                    if kw == ct or kw in ct or ct in kw:
                        overlap += 1
                        # Columns whose *suffix* (last token) matches a
                        # question keyword are usually the metric name (e.g.
                        # "transactions" → ``website_transactions``).
                        if ct is last_token and (
                            kw == last_token
                            or kw in last_token
                            or last_token in kw
                        ):
                            suffix_bonus = 1
                        break
            scored.append((overlap + suffix_bonus, suffix_bonus, col))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        related = [c for s, _, c in scored if s > 0][:limit]
        if related:
            return related
        return columns[:limit]

    @staticmethod
    def _human_label(col: str) -> str:
        return col.replace("_", " ")

    @staticmethod
    def _warning_block(warnings: list[AnalystWarning]) -> str:
        msgs = [f"- {w.message}" for w in warnings if w.message]
        if not msgs:
            return ""
        return "\n\n**Warnings**\n" + "\n".join(msgs)

    def _missing_metric_response(
        self,
        request: QueryRequest,
        decision: RoutingDecision,
        warnings: list[AnalystWarning],
    ) -> str:
        """User asked for a metric the collection does not store.

        The response has three sections:

        * **Summary** — name the missing field honestly.
        * **Available related fields** — full ranked list of related numeric
          columns sampled from the live schema.
        * **Try asking** — concrete, copy-paste-ready follow-up questions.
          The "sum of …" suggestion only combines columns that share the
          user's metric noun (e.g. for "total transactions" we only sum
          ``*_transactions`` columns, never lump in unrelated recipient or
          campaign counts).
        """
        target = decision.target or ""
        columns = self._sample_numeric_columns(target)
        alternatives = self._rank_alternatives(columns, request.question)
        metric_noun = self._guess_intent_word(request.question)
        same_metric = (
            self._columns_sharing_suffix(alternatives, metric_noun)
            if metric_noun
            else []
        )

        intent_phrase = (
            f"a `{metric_noun}` field" if metric_noun else "the metric you asked for"
        )

        lines = [
            "**Summary**",
            f"I couldn't find {intent_phrase} in `{target}`, "
            "so I'm not going to invent a number. Here's what the dataset "
            "actually contains and what I can answer instead.",
        ]

        if alternatives:
            bullet_lines = [
                f"- `{c}` — {self._human_label(c)}" for c in alternatives
            ]
            lines += [
                "",
                "**Available related fields**",
                *bullet_lines,
            ]
            suggestions = self._followup_suggestions(
                request.question,
                target,
                alternatives=alternatives,
                same_metric=same_metric,
            )
            if suggestions:
                lines += ["", "**Try asking**", *suggestions]
        else:
            lines += [
                "",
                "**What I do have**",
                "- No numeric metric columns are visible in this collection.",
                "- It may not yet be ingested for the period you asked about.",
            ]

        warning_block = self._warning_block(warnings)
        if warning_block:
            lines.append(warning_block)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Empty-metric detection + response (post-aggregation)
    # ------------------------------------------------------------------
    def _detect_empty_metric_data(
        self,
        rows: list[dict[str, Any]],
        decision: RoutingDecision,
    ) -> _EmptyMetricFinding:
        """Classify which user-requested metrics returned no usable values.

        Reads from two sources of truth, never re-parses the question:

          * Multi-metric comparison rows already carry a ``series`` field
            stamped by ``_run_multi_metric_comparison``. Each unique
            series value is a metric the user asked to compare.
          * Single-metric rows have no ``series`` field; the requested
            metric is whatever ``decision.aggregation.metric`` resolved
            to during routing.

        The detector never fires on routing failures (no target / no
        aggregation / no rows) — those have their own short-circuits.
        """
        empty: list[str] = []
        populated: list[str] = []
        requested: list[str] = []

        if not decision.target or not rows:
            return _EmptyMetricFinding(requested, empty, populated, False)

        has_series = any("series" in row for row in rows)

        if has_series:
            series_values: dict[str, list[Any]] = {}
            for row in rows:
                s = row.get("series")
                if s is None:
                    continue
                series_values.setdefault(str(s), []).append(row.get("value"))
            for metric, values in series_values.items():
                requested.append(metric)
                if all(_is_empty_value(v) for v in values):
                    empty.append(metric)
                else:
                    populated.append(metric)
            return _EmptyMetricFinding(
                requested, empty, populated, has_series=True
            )

        spec = decision.aggregation
        if not (spec and spec.metric):
            # Single-metric path needs a known metric to evaluate;
            # routing failures (no metric) are caught elsewhere.
            return _EmptyMetricFinding(requested, empty, populated, False)
        requested = [spec.metric]
        values = [r.get("value") for r in rows]
        if values and all(_is_empty_value(v) for v in values):
            empty = [spec.metric]
        else:
            populated = [spec.metric]
        return _EmptyMetricFinding(
            requested, empty, populated, has_series=False
        )

    def _empty_metric_warning(
        self, finding: _EmptyMetricFinding, target: str
    ) -> AnalystWarning:
        """Structured warning surfaced in the trust panel (comparison case)."""
        empty_list = ", ".join(f"`{m}`" for m in finding.empty)
        return AnalystWarning(
            code=WarningCode.METRIC_EMPTY,
            message=(
                f"{empty_list} returned no values for the requested period "
                f"(every row checked is null or zero). Other comparison "
                f"metric(s) had data and are analyzed below."
            ),
            details={
                "target": target,
                "empty_metrics": list(finding.empty),
                "populated_metrics": list(finding.populated),
            },
        )

    @staticmethod
    def _format_empty_metric_notice(
        finding: _EmptyMetricFinding, target: str
    ) -> str:
        """Prominent notice prepended to the LLM context for partial comparisons.

        Forces the LLM to lead with the data-availability gap instead of
        burying it in the prose. Pure formatter; no IO.
        """
        empty_list = ", ".join(f"`{m}`" for m in finding.empty)
        populated_list = ", ".join(f"`{m}`" for m in finding.populated) or "none"
        return (
            "DATA AVAILABILITY NOTICE (lead the response with this):\n"
            f"- Empty (no values for the requested period in `{target}`): "
            f"{empty_list}\n"
            f"- Populated (analyzed below): {populated_list}\n"
            "Tell the user explicitly which requested metric is empty and "
            "which one is being analyzed. Do not invent values for the "
            "empty metric."
        )

    def _empty_metric_response(
        self,
        request: QueryRequest,
        decision: RoutingDecision,
        finding: _EmptyMetricFinding,
        warnings: list[AnalystWarning],
    ) -> str:
        """User asked for a metric that exists in the schema but has no values.

        Distinct from ``_missing_metric_response`` (column doesn't exist
        at all). Here the column IS in the schema; it just wasn't
        populated for the period the user asked about. Response shape
        deliberately mirrors the missing-metric one so the trust panel,
        warnings block, and follow-up suggestions stay consistent.
        """
        target = decision.target or ""
        empty_metrics = list(finding.empty)
        # Rank ALL numeric columns minus the empty ones — those are the
        # candidates we can honestly point the user to.
        all_columns = self._sample_numeric_columns(target)
        alternatives = self._rank_alternatives(
            [c for c in all_columns if c not in empty_metrics],
            request.question,
        )
        # Cull obviously-empty alternatives so we don't bait-and-switch.
        # Cheap one-shot existence probe per alternative; usually 0-3
        # alternatives are checked, and the schema cache absorbs misses.
        populated_alts = [
            c for c in alternatives if self._field_has_any_value(target, c)
        ]
        if not populated_alts:
            populated_alts = alternatives  # fall back rather than show nothing

        if len(empty_metrics) == 1:
            intro = (
                f"`{empty_metrics[0]}` exists in `{target}` but has no values "
                "recorded for the period you asked about (every row checked "
                "is null or zero). I'm not going to invent a number."
            )
        else:
            field_list = ", ".join(f"`{m}`" for m in empty_metrics)
            intro = (
                f"The fields {field_list} all exist in `{target}` but have "
                "no values recorded for the period you asked about (every "
                "row checked is null or zero). I'm not going to invent "
                "numbers."
            )

        lines = ["**Data not available**", "", intro]

        if populated_alts:
            bullets = [
                f"- `{c}` — {self._human_label(c)}" for c in populated_alts
            ]
            lines += ["", "**Related fields with data in this dataset**", *bullets]
            suggestions = self._followup_suggestions(
                request.question,
                target,
                alternatives=populated_alts,
                same_metric=[],
            )
            if suggestions:
                lines += ["", "**Try asking**", *suggestions]
        else:
            lines += [
                "",
                "**What I do have**",
                "- No other populated numeric fields are visible in this "
                "collection.",
                "- The dataset may not yet be fully ingested for this period.",
            ]

        warning_block = self._warning_block(warnings)
        if warning_block:
            lines.append(warning_block)

        return "\n".join(lines)

    def _field_has_any_value(self, target: str, field_name: str) -> bool:
        """Cheap existence probe: at least one row where ``field_name`` is non-empty.

        Used by the empty-metric response to filter "alternatives" so we
        only suggest fields that actually have values today. Fails open
        (returns True on Mongo errors) so a flaky probe doesn't suppress
        a useful suggestion.
        """
        try:
            rows = self.mongo.find(
                target,
                {
                    field_name: {
                        "$exists": True, "$nin": [None, 0],
                    }
                },
                limit=1,
            )
            return len(rows) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Empty-metric alternative probe on {target}.{field_name} "
                f"failed ({type(exc).__name__}: {exc}); keeping the suggestion."
            )
            return True

    @staticmethod
    def _columns_sharing_suffix(
        columns: list[str], metric_noun: str
    ) -> list[str]:
        """Return columns whose last underscore-token matches ``metric_noun``.

        Used to scope the "sum of …" suggestion so we only combine fields
        that mean the same thing (``website_transactions`` +
        ``smart_app_transactions``), never adjacent-but-different counters
        like ``hajj_package_recipients``.
        """
        if not metric_noun:
            return []
        noun = metric_noun.lower()
        out: list[str] = []
        for col in columns:
            tokens = col.lower().split("_")
            if not tokens:
                continue
            last = tokens[-1]
            if last == noun or last in noun or noun in last:
                out.append(col)
        return out

    def _no_target_response(
        self, request: QueryRequest, warnings: list[AnalystWarning]
    ) -> str:
        return (
            "**Summary**\n"
            f"I couldn't match _{request.question.strip()}_ to any dataset "
            "in the catalog.\n\n"
            "**Next step**\n"
            "Name the dataset explicitly — for example "
            "`total transactions for hajj-permit-service in 2024` "
            "or `monthly recipients for zakat-disbursement in 2024`."
            + self._warning_block(warnings)
        )

    def _zero_rows_response(
        self,
        request: QueryRequest,
        decision: RoutingDecision,
        warnings: list[AnalystWarning],
    ) -> str:
        """User's filter matched zero rows on a real dataset.

        Previously this was a generic "try a different period" stub.
        The user can't act on that — they don't know which period the
        dataset actually covers. We now probe the live data
        (:meth:`_probe_target_coverage`) so the response leads with
        what IS available, with copy-paste follow-ups bound to a year
        that actually has data. Falls back to the generic message
        when the probe can't run or the dataset is truly empty.
        """
        target = decision.target or "the selected collection"
        spec = decision.aggregation

        requested_year = self._requested_year(spec)
        scope_label = self._requested_scope_label(spec)

        # Probe is cached — _answer_impl already ran it and pushed any
        # coverage warnings into the trust panel before short-circuiting
        # here. We re-fetch only to render the response text.
        coverage = self._probe_target_coverage(decision)

        if coverage is not None and coverage.has_any_data:
            return self._zero_rows_with_coverage(
                request=request,
                decision=decision,
                coverage=coverage,
                requested_year=requested_year,
                scope_label=scope_label,
                warnings=warnings,
            )

        # No probe (no time field) or probe ran but dataset is genuinely
        # empty — preserve the old, honest message.
        scope = scope_label or "the requested scope"
        intro = (
            f"`{target}` is reachable but has no rows matching {scope}."
        )
        if coverage is not None and coverage.probed and not coverage.has_any_data:
            intro = (
                f"`{target}` is reachable but appears to have no ingested "
                "data at all."
            )
        return (
            "**Summary**\n"
            f"{intro}\n\n"
            "**Next step**\n"
            "Try a different period, broaden the filters, or confirm the "
            "data for that range has been ingested."
            + self._warning_block(warnings)
        )

    def _zero_rows_with_coverage(
        self,
        *,
        request: QueryRequest,
        decision: RoutingDecision,
        coverage: _TargetCoverage,
        requested_year: int | None,
        scope_label: str,
        warnings: list[AnalystWarning],
    ) -> str:
        """Honest "what IS available?" message, shaped like _empty_metric_response.

        Mirrors the empty-metric branch deliberately: a clear "Data not
        available" headline, a "What I do have" section grounded in the
        live probe, and copy-paste follow-ups that swap the requested
        year for an available one. Keeps the LLM out of the loop —
        there's nothing to ground on, so a deterministic response is
        safer than a generated one.
        """
        target = decision.target or ""
        spec = decision.aggregation
        suggest_year = self._pick_suggested_year(coverage, requested_year)

        scope_phrase = scope_label or "the requested scope"
        if requested_year is not None:
            intro = (
                f"I couldn't find any rows in `{target}` for "
                f"**{requested_year}**. I'm not going to invent numbers."
            )
        else:
            intro = (
                f"I couldn't find any rows in `{target}` for "
                f"{scope_phrase}. I'm not going to invent numbers."
            )

        lines = ["**Data not available**", "", intro, "", "**What I do have**"]
        span = self._format_coverage_span(coverage)
        if span:
            lines.append(f"- Live coverage: **{span}**")
        if coverage.live_years:
            year_list = ", ".join(str(y) for y in coverage.live_years)
            lines.append(f"- Years with data: **{year_list}**")
        gap = coverage.gap_years()
        if gap:
            gap_list = ", ".join(str(y) for y in gap)
            lines.append(
                f"- Catalog also lists {gap_list}, but the live "
                "collection has nothing ingested for "
                f"{'that year' if len(gap) == 1 else 'those years'} yet."
            )

        suggestions = self._followups_for_coverage(
            request_question=request.question,
            target=target,
            spec=spec,
            suggest_year=suggest_year,
        )
        if suggestions:
            lines += ["", "**Try asking**", *suggestions]

        warning_block = self._warning_block(warnings)
        if warning_block:
            lines.append(warning_block)
        return "\n".join(lines)

    @staticmethod
    def _requested_year(spec: AggregationSpec | None) -> int | None:
        """Pull the calendar year the user asked about (best-effort).

        Returns ``None`` when the spec has no time filter or the
        filter spans more than one year — in which case the
        suggestion logic falls back to "the most recent available
        year" instead of pretending we know the user's exact intent.
        """
        if spec is None or spec.time is None:
            return None
        rf = spec.time.range_from
        rt = spec.time.range_to
        if rf is None or rt is None:
            return None
        if not isinstance(rf, datetime) or not isinstance(rt, datetime):
            return None
        if rf.year == rt.year:
            return rf.year
        return None

    @staticmethod
    def _requested_scope_label(spec: AggregationSpec | None) -> str:
        """Render the user's time + filter scope as a short human phrase."""
        if spec is None:
            return ""
        bits: list[str] = []
        time = spec.time
        if time and time.range_from and time.range_to:
            rf = time.range_from
            rt = time.range_to
            if (
                isinstance(rf, datetime)
                and isinstance(rt, datetime)
                and rf.year == rt.year
            ):
                bits.append(str(rf.year))
            else:
                bits.append(f"between {rf} and {rt}")
        if spec.filters:
            bits.extend(f"{k}={v}" for k, v in spec.filters.items())
        return " ".join(bits)

    @staticmethod
    def _format_coverage_span(coverage: _TargetCoverage) -> str:
        """Compact ``Jan 2024 – Dec 2025`` / ``2024 – 2025`` / ``2025``."""

        def _fmt(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.strftime("%b %Y")
            if isinstance(value, int):
                return str(value)
            if isinstance(value, str):
                # AWQAF ``period`` is ``YYYY-MM``; show as-is when it
                # looks like one, otherwise raw string.
                return value
            return str(value)

        a = _fmt(coverage.earliest)
        b = _fmt(coverage.latest)
        if a and b and a != b:
            return f"{a} – {b}"
        return a or b or ""

    @staticmethod
    def _pick_suggested_year(
        coverage: _TargetCoverage, requested_year: int | None
    ) -> int | None:
        """Pick the most useful year to suggest in copy-paste follow-ups.

        Preference order:

        1. The latest year strictly *before* the requested one (so
           "trend for 2026 → try 2025" feels natural).
        2. Otherwise, the latest available year overall.
        3. ``None`` when the live probe found no usable years.
        """
        years = coverage.live_years
        if not years:
            return None
        if requested_year is not None:
            earlier = [y for y in years if y < requested_year]
            if earlier:
                return earlier[-1]
        return years[-1]

    def _followups_for_coverage(
        self,
        *,
        request_question: str,
        target: str,
        spec: AggregationSpec | None,
        suggest_year: int | None,
    ) -> list[str]:
        """Build 1-2 copy-paste follow-up questions bound to an available year.

        Prefers the metric the user actually asked for (when known)
        and rewrites just the year. Falls back to ranking sibling
        numeric columns when the original spec didn't carry a metric.
        """
        if suggest_year is None or not target:
            return []
        slug = target
        if slug.startswith("awqaf_") and slug.endswith(_AWQAF_FACTS_SUFFIX):
            slug = slug[len("awqaf_") : -len(_AWQAF_FACTS_SUFFIX)]
        slug = slug.replace("_", "-")

        suggestions: list[str] = []
        requested_metric = (spec.metric if spec else None) or None
        if requested_metric:
            label = self._human_label(requested_metric).title()
            verb = "monthly" if (spec and spec.time) else "total"
            suggestions.append(
                f"- _{verb} {label} for {slug} in {suggest_year}_"
            )

        try:
            columns = self._sample_numeric_columns(target)
        except Exception:  # noqa: BLE001
            columns = []
        ranked = self._rank_alternatives(
            [c for c in columns if c != requested_metric], request_question
        )
        for col in ranked[:2]:
            suggestions.append(
                f"- _{self._human_label(col).title()} for "
                f"{slug} in {suggest_year}_"
            )
        return suggestions[:3]

    @staticmethod
    def _coverage_warnings(
        coverage: _TargetCoverage, requested_year: int | None
    ) -> list[AnalystWarning]:
        """Surface ingest drift so the trust panel doesn't silently agree.

        Two warnings can come out of a coverage probe:

        * **catalog gap** — the metadata catalog claims a year that
          the live collection doesn't have. Almost always points at a
          half-finished ingest; worth flagging operationally even
          when the user didn't ask about that year.
        * **year explicitly empty** — the year the user *did* ask
          about is confirmed-empty by the live probe (vs "we just
          don't know"). The user already sees the headline; the
          warning is for machine-readable provenance.
        """
        out: list[AnalystWarning] = []
        gap = coverage.gap_years()
        if gap:
            gap_list = ", ".join(str(y) for y in gap)
            out.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message=(
                        f"Catalog lists {gap_list} for `{coverage.target}`, "
                        "but the live collection has no rows for "
                        f"{'that year' if len(gap) == 1 else 'those years'}."
                    ),
                    details={"target": coverage.target, "gap_years": gap},
                )
            )
        if (
            requested_year is not None
            and coverage.excludes_year(requested_year)
        ):
            out.append(
                AnalystWarning(
                    code=WarningCode.EMPTY_RESULT,
                    message=(
                        f"No rows for {requested_year} in "
                        f"`{coverage.target}` (confirmed by live probe). "
                        f"Available years: {coverage.live_years}."
                    ),
                    details={
                        "target": coverage.target,
                        "requested_year": requested_year,
                        "available_years": coverage.live_years,
                    },
                )
            )
        return out

    @staticmethod
    def _guess_intent_word(question: str) -> str | None:
        """Pick the most metric-like content word from the question."""
        keywords = _question_keywords(question)
        for kw in keywords:
            if kw.endswith(("s", "ion", "ions", "ment", "ments")):
                return kw
        return keywords[0] if keywords else None

    @staticmethod
    def _followup_suggestions(
        question: str,
        target: str,
        *,
        alternatives: list[str],
        same_metric: list[str],
    ) -> list[str]:
        """Build concrete copy-paste follow-up questions.

        Only emits suggestions the engine can actually execute today (single
        metric, single dataset, optional year). Multi-field arithmetic like
        "Sum of A + B" is intentionally avoided because the router parses
        one metric per query — surfacing it would just dead-end the user.
        """
        slug = target
        if slug.startswith("awqaf_") and slug.endswith(_AWQAF_FACTS_SUFFIX):
            slug = slug[len("awqaf_") : -len(_AWQAF_FACTS_SUFFIX)]
        slug = slug.replace("_", "-")
        year_match = re.search(r"\b(20\d{2})\b", question)
        when = year_match.group(1) if year_match else "2024"

        out: list[str] = []
        for col in (same_metric or alternatives)[:3]:
            out.append(
                f"- _{AnalystOrchestrator._human_label(col).title()} "
                f"for {slug} in {when}_"
            )
        if len(same_metric) >= 2:
            out.append(
                "- _(These channels are stored separately. Each query above "
                "returns one channel — combine the results to get the total, "
                "or re-ingest the dataset to publish a unified "
                "`total_transactions` field.)_"
            )
        return out

    def _run_semantic(
        self,
        decision: RoutingDecision,
        request: QueryRequest,
    ) -> list[VectorHit]:
        if decision.route == QueryRoute.ANALYTICAL:
            return []
        try:
            return self.vector.search(request.question, top_k=request.top_k)
        except AIAnalystError as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

    def _probe_freshness(self, decision: RoutingDecision) -> datetime | None:
        spec = decision.aggregation
        if spec is None or spec.time is None or not decision.target:
            return None
        try:
            if decision.data_source == DataSource.MYSQL:
                value = self.mysql.latest_value(decision.target, spec.time.field)
            else:
                value = self.mongo.latest_value(decision.target, spec.time.field)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Freshness probe on {decision.target}.{spec.time.field} "
                f"failed ({type(exc).__name__}: {exc}); 'as of' will be unset."
            )
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _probe_target_coverage(
        self, decision: RoutingDecision
    ) -> _TargetCoverage | None:
        """Return what years/periods the routed target actually contains.

        Order of truth:

        1. **Live** distinct-years probe on the facts collection. This
           is the contract — the catalog can drift.
        2. **Live** earliest/latest bounds for human-readable spans
           (``Jan 2024 – Dec 2025``).
        3. **Catalog** ``years`` array from
           ``awqaf_datasets_metadata``, only used for the gap-warning
           (catalog claims a year that the live probe doesn't see).

        Returns ``None`` when we can't run the probe at all (no
        target, no time field). Returns a coverage object with
        ``probed=False`` on probe error so callers can still
        gracefully degrade to the generic "try another period"
        message.
        """
        spec = decision.aggregation
        if spec is None or spec.time is None or not decision.target:
            return None

        target = decision.target
        time_field = spec.time.field
        cache_key = f"{decision.data_source.value}|{target}|{time_field}"

        def _fetch() -> _TargetCoverage:
            live_years: list[int] = []
            earliest: Any = None
            latest: Any = None
            probed = False
            try:
                if decision.data_source == DataSource.MYSQL:
                    live_years = self.mysql.distinct_years(target, time_field)
                    earliest = self.mysql.earliest_value(target, time_field)
                    latest = self.mysql.latest_value(target, time_field)
                else:
                    live_years = self.mongo.distinct_years(target, time_field)
                    earliest = self.mongo.earliest_value(target, time_field)
                    latest = self.mongo.latest_value(target, time_field)
                probed = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Coverage probe on {target}.{time_field} failed "
                    f"({type(exc).__name__}: {exc}); coverage unknown."
                )

            catalog_years = self._catalog_years_for_target(target)

            return _TargetCoverage(
                target=target,
                time_field=time_field,
                live_years=sorted(set(live_years)),
                earliest=earliest,
                latest=latest,
                catalog_years=sorted(set(catalog_years)),
                probed=probed,
            )

        return self._coverage_cache.get_or_set(cache_key, _fetch)

    def _catalog_years_for_target(self, target: str) -> list[int]:
        """Pull the ``years`` array from ``awqaf_datasets_metadata`` for a target.

        Best-effort: returns ``[]`` on any failure or when the target
        isn't an AWQAF facts collection (catalog only knows AWQAF
        datasets). Used to flag ingest drift, never as a primary
        source of truth.
        """
        slug = _dataset_slug_from_target(target)
        if not slug:
            return []
        try:
            row = self.mongo.collection(_AWQAF_METADATA_COLLECTION).find_one(
                {"dataset": slug.replace("_", "-")},
                projection={"_id": 0, "years": 1},
            )
            if not row:
                row = self.mongo.collection(_AWQAF_METADATA_COLLECTION).find_one(
                    {"dataset": slug},
                    projection={"_id": 0, "years": 1},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Catalog years lookup for {slug} failed "
                f"({type(exc).__name__}: {exc})."
            )
            return []
        years = (row or {}).get("years") or []
        out: list[int] = []
        for y in years:
            try:
                out.append(int(y))
            except (TypeError, ValueError):
                continue
        return out

    def _transform_rows_for_visualization(
        self,
        rows: list[dict[str, Any]],
        shape,
        chart_type: ChartType,
    ) -> list[dict[str, Any]]:
        """
        Transform row schema to match visualization requirements.
        
        Key transformations:
        1. Sparse temporal pivot: dimension → label, metric → series
        2. Multi-metric grouping: preserve dimension in label, metric in series
        3. Pass-through: rows already in correct shape
        
        Args:
            rows: Raw aggregation rows
            shape: AnalyticalShape with dimensional analysis
            chart_type: Selected chart type
            
        Returns:
            Transformed rows ready for chart rendering
        """
        
        # ─────────────────────────────────────────────────────────────
        # SPARSE TEMPORAL PIVOT
        #
        # Input:  {"label": "2026-01", "series": "Fujairah", "metric": "revenues", ...}
        # Output: {"label": "Fujairah", "series": "revenues", ...}
        #
        # When: Only 1 time bucket but multiple dimensions exist
        # Why: User asked for "trend" but data doesn't support it
        # Solution: Pivot dimensions to x-axis, metrics to series
        # ─────────────────────────────────────────────────────────────
        if shape.sparse_time_series and shape.categorical_comparison:
            transformed = []
            for row in rows:
                dimension = row.get("dimension")
                metric = row.get("metric")
                
                if not dimension:
                    # No dimension to pivot; keep as-is
                    transformed.append(row)
                    continue
                
                # CRITICAL FIX: Ensure series field is set for multi-metric visualization
                # When we have multiple metrics, each metric becomes a series
                series_value = metric or row.get("series")
                series_label = row.get("metric_label") or row.get("series_label") or series_value
                
                new_row = {
                    "label": dimension,                    # Dimension becomes x-axis
                    "series": series_value,                # Metric becomes grouping
                    "series_label": series_label,          # Human-readable series name
                    "dimension": dimension,                # Preserve for metadata
                    "metric": metric,                      # Preserve for metadata
                    "value": row.get("value"),
                    "partial": row.get("partial"),
                }
                transformed.append(new_row)
            
            logger.info(
                f"Pivoted {len(rows)} rows for sparse temporal visualization: "
                f"dimension → label, metric → series. Sample: {transformed[:2]}"
            )
            return transformed
        
        # ─────────────────────────────────────────────────────────────
        # MULTI-METRIC CATEGORICAL (no time dimension)
        #
        # When: Comparing multiple metrics across categories
        # Action: Ensure series field contains metric for grouping
        # ─────────────────────────────────────────────────────────────
        if shape.multi_metric_comparison and shape.categorical_comparison:
            transformed = []
            for row in rows:
                metric = row.get("metric")
                dimension = row.get("dimension")
                
                # CRITICAL FIX: For multi-metric comparisons, ensure series field
                # is properly set to enable multi-series visualization
                if metric and not row.get("series"):
                    # Metric exists but series is empty; use metric as series
                    new_row = dict(row)
                    new_row["series"] = metric
                    new_row["series_label"] = row.get("metric_label") or metric
                    
                    # If we have dimensions, use them as labels for categorical x-axis
                    if dimension and not new_row.get("label"):
                        new_row["label"] = dimension
                    
                    transformed.append(new_row)
                else:
                    transformed.append(row)
            
            if transformed != rows:
                logger.info(
                    f"Adjusted {len(rows)} rows for multi-metric categorical: "
                    f"metric → series. Sample: {transformed[:2]}"
                )
            return transformed
        
        # ─────────────────────────────────────────────────────────────
        # PASS-THROUGH
        #
        # Rows are already in correct shape for visualization.
        # This includes:
        # - Strong time trends (multi-series or single)
        # - Simple categorical comparisons
        # - KPI single values
        # ─────────────────────────────────────────────────────────────
        return rows

    def _maybe_chart(
        self,
        rows: list[dict[str, Any]],
        request: QueryRequest,
        decision: RoutingDecision,
        quality,
    ) -> ChartPayload | None:
        """Backward-compat shim — returns the panel's primary view only.

        Kept so any external caller still calling ``_maybe_chart``
        (tests, scripts) gets the same single :class:`ChartPayload`
        they used to. Internally the orchestrator now uses
        :meth:`_maybe_chart_panel` so it can populate both the primary
        ``chart`` and the full ``charts`` list on the response.
        """
        panel = self._maybe_chart_panel(rows, request, decision, quality)
        if not panel:
            return None
        return next((c for c in panel if c.is_primary), panel[0])

    def _maybe_chart_panel(
        self,
        rows: list[dict[str, Any]],
        request: QueryRequest,
        decision: RoutingDecision,
        quality,
    ) -> list[ChartPayload]:
        """
        Build the full chart panel for this answer.
        
        Pipeline:
        1. Infer analytical shape from raw rows
        2. Choose visualization type based on shape
        3. Transform rows to match visualization requirements
        4. Recompute shape after transformation
        5. Build chart panel with transformed rows
        
        Returns an empty list when no chart should render (no data, the
        caller suppressed charts, or every view raised). Otherwise
        returns one or more :class:`ChartPayload` objects with stable
        ``chart_id`` / ``view_label`` annotations the frontend uses to
        render a tab strip. Exactly one entry has ``is_primary=True``.
        """
        if not rows or quality.suppress_chart:
            return []
        if request.chart_type == ChartType.NONE:
            return []

        has_time = bool(decision.aggregation and decision.aggregation.time)

        # ─────────────────────────────────────────────────────────────
        # STEP 1: INFER ANALYTICAL SHAPE
        #
        # Analyze raw rows to understand:
        # - Temporal density (how many time buckets?)
        # - Categorical richness (how many dimensions?)
        # - Metric comparison (single or multi-metric?)
        # - Series structure (what's the grouping axis?)
        # ─────────────────────────────────────────────────────────────
        shape = infer_shape(rows)

        logger.info(
            "Analytical shape: "
            f"time_points={shape.distinct_time_points} "
            f"groups={shape.distinct_groups} "
            f"series={shape.distinct_series} "
            f"metrics={shape.distinct_metrics} "
            f"rows={shape.row_count} "
            f"has_dimensions={shape.has_dimensions} "
            f"has_metrics={shape.has_metrics}"
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 2: CHOOSE VISUALIZATION TYPE
        #
        # Adaptive decision based on analytical shape:
        # - Sparse temporal → Grouped bars
        # - Strong trend → Line chart
        # - Categorical → Bars
        # - Single value → KPI
        # ─────────────────────────────────────────────────────────────
        chart_type, reason = VisualizationPlanner.choose(
            shape,
            has_time=has_time,
        )

        logger.info(f"Visualization decision: {chart_type.value} — {reason}")

        # ─────────────────────────────────────────────────────────────
        # STEP 3: TRANSFORM ROWS FOR VISUALIZATION
        #
        # Adapt row schema to match visualization requirements:
        # - Sparse temporal: pivot dimension → label, metric → series
        # - Multi-metric categorical: ensure metric → series
        # - Pass-through: rows already correct
        # ─────────────────────────────────────────────────────────────
        transformed_rows = self._transform_rows_for_visualization(
            rows, shape, chart_type
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 4: RECOMPUTE SHAPE AFTER TRANSFORMATION
        #
        # Row structure may have changed (e.g., dimension became label).
        # Recompute shape so chart rendering sees the final structure.
        # ─────────────────────────────────────────────────────────────
        final_shape = infer_shape(transformed_rows)

        # Update has_time flag after transformation
        # (sparse temporal pivot removes time dimension)
        if shape.sparse_time_series and shape.categorical_comparison:
            has_time = False

        logger.info(
            f"Final shape after transformation: "
            f"time_points={final_shape.distinct_time_points} "
            f"groups={final_shape.distinct_groups} "
            f"series={final_shape.distinct_series}"
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 5: BUILD CHART PANEL
        #
        # Multi-series detection. Two flavors:
        #   *  rows tagged with ``series`` from _run_multi_metric_comparison
        #      (one series per metric — different units possible)
        #   *  rows tagged with ``series`` from the time+group_by Mongo
        #      path (one series per group value — same metric/unit
        #      everywhere)
        # ─────────────────────────────────────────────────────────────
        has_series = any("series" in row for row in transformed_rows)
        logger.info(
            f"Multi-series detection: has_series={has_series}, "
            f"row_count={len(transformed_rows)}, "
            f"sample_row={transformed_rows[0] if transformed_rows else None}"
        )
        
        if has_series:
            multi = self._build_multi_series_profile(transformed_rows, decision)
            logger.info(
                f"Built multi-series profile: {len(multi.series)} series, "
                f"dual_axis={multi.dual_axis}, "
                f"series_names={[s.label for s in multi.series]}"
            )
            title = self._multi_chart_title(decision, multi)
            return self._panel.build(
                transformed_rows,
                primary_chart_type=chart_type,
                base_title=title,
                has_time=(
                    has_time
                    and final_shape.strong_trend
                ),
                partial_latest=quality.partial_latest,
                multi_series_profile=multi,
            )

        profile = _build_metric_profile(decision)
        title = self._chart_title(decision, profile=profile)

        return self._panel.build(
            transformed_rows,
            primary_chart_type=chart_type,
            base_title=title,
            has_time=(
                has_time
                and final_shape.strong_trend
            ),
            partial_latest=quality.partial_latest,
            metric_profile=profile,
        )



    def _auto_dimension_field(
    self,
    schema_columns: list[str],
    ) -> str | None:

        preferred = [
            "dimension",
            "emirate",
            "category",
            "region",
            "department",
        ]

        for col in preferred:
            if col in schema_columns:
                return col

        return None

    def _plan_chart_with_llm(
        self,
        rows: list[dict[str, Any]],
        request: QueryRequest,
        decision: RoutingDecision,
    ):
        """Compose a small prompt context and ask the LLM chart planner.

        Kept on the orchestrator (not the planner) because composing the
        plan summary from a :class:`RoutingDecision` is orchestrator
        knowledge — the planner stays decoupled from our schema.
        """
        from services.chart_tool import llm_chart_planner

        spec = decision.aggregation
        plan_summary_bits: list[str] = []
        if spec is not None:
            plan_summary_bits.append(
                f"operation={spec.operation} metric={spec.metric or '-'}"
            )
            if spec.group_by:
                plan_summary_bits.append(f"group_by={spec.group_by}")
            if spec.time:
                plan_summary_bits.append(
                    f"time_bucket={spec.time.bucket.value}"
                )
        if decision.target:
            plan_summary_bits.append(f"target={decision.target}")
        plan_summary = " | ".join(plan_summary_bits) or "(no plan)"

        return llm_chart_planner.plan(
            question=request.question,
            rows=rows,
            plan_summary=plan_summary,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chart_title(
        decision: RoutingDecision,
        *,
        profile: MetricProfile | None = None,
    ) -> str:
        """Compose the chart title from the metric profile.

        Falls back to the raw spec rendering only when there is no
        aggregation at all (e.g. a semantic-only answer that still asked
        for a chart) — in normal analytical flow ``profile`` is always
        present and the title comes from
        :func:`services.metric_profile.compose_chart_title`.
        """
        spec = decision.aggregation
        if not spec:
            return ""
        prof = profile or _build_metric_profile(decision)
        if prof is None:
            return ""
        return compose_chart_title(
            prof,
            time_bucket=spec.time.bucket if spec.time is not None else None,
            group_by=spec.group_by,
        )

    @staticmethod
    def _build_multi_series_profile(
        rows: list[dict[str, Any]],
        decision: RoutingDecision,
    ) -> MultiSeriesProfile:
        """Build a :class:`MultiSeriesProfile` from multi-series rows.

        Two distinct shapes flow through the same chart layer:

        *  Compared-metrics rows — produced by
           ``_run_multi_metric_comparison``. ``series`` IS a column name
           (e.g. ``smart_app_transactions``). Each series may have a
           different unit, so we classify each one independently and
           let the renderer decide on dual-axis.
        *  Grouped time-series rows — produced by the Mongo
           ``time + group_by`` path. ``series`` is a category value
           (e.g. an emirate name), but every series shares the SAME
           underlying metric (``decision.aggregation.metric``). All
           series get the same unit/profile, only the legend label and
           color differ.
        """
        spec = decision.aggregation
        op = (spec.operation if spec else None) or "sum"
        seen: list[tuple[str, str | None, str]] = []
        used: set[str] = set()
        for row in rows:
            key = row.get("series")
            if key is None:
                continue
            key = str(key)
            if key in used:
                continue
            used.add(key)
            label = str(row.get("series_label") or key)

            # Compared-metrics path: the series key IS the metric column.
            # Grouped path: the series key is a group value, the metric
            # is the spec's metric and shared across series.
            if spec and spec.group_by and key != (spec.metric or ""):
                metric_field = spec.metric
            else:
                metric_field = key
            seen.append((key, metric_field, label))

        return classify_series(seen, operation=op)

    @staticmethod
    def _multi_chart_title(
        decision: RoutingDecision,
        multi: MultiSeriesProfile,
    ) -> str:
        spec = decision.aggregation
        bucket = spec.time.bucket if spec and spec.time is not None else None
        # Group_by present → a single metric split across categories.
        # Use the by-clause flavour ("Monthly Avg Occupancy Rate (%) by
        # Emirate"). Otherwise it's a head-to-head metric comparison.
        if spec and spec.group_by:
            return compose_multi_chart_title(
                multi,
                time_bucket=bucket,
                group_by_label=spec.group_by,
            )
        return compose_multi_chart_title(multi, time_bucket=bucket)


    def _build_provenance(
        self, request: QueryRequest, decision: RoutingDecision
    ) -> Provenance:
        """Compile an opt-in provenance block for analyst-mode responses."""
        target = decision.target
        ds = decision.data_source
        spec = decision.aggregation

        mongo_pipeline: list[dict[str, Any]] | None = None
        sql: str | None = None

        if spec is not None and target:
            try:
                if ds == DataSource.MONGO:
                    mongo_pipeline = self.mongo.build_pipeline(spec)
                elif ds == DataSource.MYSQL:
                    sql, _ = self.mysql.build_sql(target, spec)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Provenance preview unavailable: {exc}")

        return Provenance(
            fingerprint=_plan_fingerprint(decision),
            target=target,
            data_source=ds if ds != DataSource.AUTO else None,
            mongo_pipeline=mongo_pipeline,
            sql=sql,
            columns_sampled=_columns_from_spec(spec),
            glossary_definition_id=(
                decision.definition.definition.id if decision.definition else None
            ),
            session_id=request.session_id,
        )


def _plan_fingerprint(decision: RoutingDecision) -> str:
    """Stable hash of the executable parts of the plan."""
    payload = {
        "data_source": decision.data_source.value,
        "target": decision.target,
        "aggregation": (
            decision.aggregation.model_dump(mode="json")
            if decision.aggregation
            else None
        ),
        "definition_id": (
            decision.definition.definition.id if decision.definition else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _columns_from_spec(spec: AggregationSpec | None) -> list[str]:
    if spec is None:
        return []
    cols: list[str] = []
    if spec.metric:
        cols.append(spec.metric)
    if spec.group_by:
        cols.append(spec.group_by)
    if spec.time is not None and spec.time.field:
        cols.append(spec.time.field)
    cols.extend(spec.filters.keys())
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def _draft_definition_warning(decision: RoutingDecision) -> AnalystWarning | None:
    """Surface a typed warning when a non-applied glossary match was found."""
    match = decision.definition
    if match is None or match.applied_to_query:
        return None
    return AnalystWarning(
        code=WarningCode.DEFINITION_UNVERIFIED,
        message=(
            f"A definition for '{match.definition.term}' exists in the glossary "
            "but was not applied to the query (status: "
            f"{match.definition.status.value}). The numbers above use the "
            "system's default interpretation."
        ),
        details={
            "definition_id": match.definition.id,
            "status": match.definition.status.value,
            "matched_alias": match.matched_alias,
        },
    )


analyst_orchestrator = AnalystOrchestrator()

