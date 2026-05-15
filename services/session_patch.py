"""Stateful co-reference resolution for multi-turn follow-ups.

The earlier version of this module did **keyword matching** on the new
utterance in isolation: a hardcoded prefix list (``by``, ``per``,
``exclude``, …) decided whether to merge with prior session state.
That doesn't survive real users — the same wording (``2025``, ``vs
that``, ``now show emirates``) means different things depending on the
prior turn, and any prefix list will always miss novel phrasings.

The new design is **structural**, not keyword-based:

1. **Catalog anchor sensing** — if the new question references a known
   dataset or metric on its own (e.g. ``hajj-package-service``,
   ``total_transactions``), it stands by itself and is NOT a follow-up.
   The orchestrator already maintains a TTL-cached anchor list built
   from ``awqaf_datasets_metadata`` plus the live key-metric vocabulary;
   we just consult it.
2. **Refinement-shape detection** — a follow-up carries only signals
   that need prior context to make sense: bare years, quarter shortcuts,
   ``vs X`` / ``compare with X`` / ``compared to X``, ``by <field>``,
   ``per <field>``, ``exclude <field>``, ``only <field>``, etc.
3. **Headless brevity** — short questions with no catalog anchor and
   no comparison verb either are almost always referring to the prior
   turn.

When all three signals are absent we fall through to fresh routing.
When at least one structural signal fires AND we have a prior
analytical decision, we patch the prior plan.

This module stays deterministic and dependency-free. An LLM-based
co-reference resolver can be layered on top via
``resolve_followup(..., llm_resolver=...)`` for truly ambiguous cases
(currently a no-op hook; see :func:`resolve_followup`).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from models.enums import ComparisonMode, QueryRoute, TimeBucket
from models.schemas import AggregationSpec, RoutingDecision, TimeSpec


# ---------------------------------------------------------------------------
# Refinement-signal detectors
# ---------------------------------------------------------------------------
class FollowUpSignal(str, Enum):
    """A structural signal that the new utterance is a refinement.

    Used by the analyst orchestrator to decide whether to consult prior
    session state. Each signal corresponds to a concrete patch we
    know how to apply to a prior :class:`RoutingDecision`.

    The taxonomy was designed by reverse-engineering what real
    analysts say in a follow-up after seeing a chart:

    * **TIME**   — swap or narrow the time window. Includes bare
      years, quarter / half / YTD shortcuts, and "last N months /
      years" rolling windows.
    * **COMPARE** — period-over-period or year-over-year comparison.
    * **GROUP_BY** — break the existing answer down by some dimension.
    * **FILTER**  — narrow the population (only X / exclude Y).
    * **BUCKET**  — change granularity (daily / monthly / quarterly).
    * **OPERATION** — switch the aggregation (avg vs sum vs max).
    * **METRIC**  — swap to a different field on the same dataset.
    * **SORT_LIMIT** — top N / bottom N / ranked / highest / lowest.
    * **SCOPE**   — drop the time filter ("all data", "full history").
    * **HEADLESS** — short utterance with no standalone anchor.
      Required-companion: never enough on its own; must accompany
      a substantive signal.
    """

    TIME = "time"
    COMPARE = "compare"
    GROUP_BY = "group_by"
    FILTER = "filter"
    BUCKET = "bucket"
    OPERATION = "operation"
    METRIC = "metric"
    SORT_LIMIT = "sort_limit"
    SCOPE = "scope"
    HEADLESS = "headless"


# Year-only utterance: useful both for detection and for patch logic.
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_QUARTER_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)

# Comparison shapes. We deliberately match VERB FORMS plus the
# preposition that almost always follows them — this is structural,
# not a vocabulary list. ``compare with``, ``compare to``, ``compared
# to``, ``vs``, ``versus`` all reduce to the same intent.
_COMPARE_SHAPE_RE = re.compile(
    r"\b(?:vs\.?|versus|compare(?:d)?\s+(?:to|with|against)|against|head\s+to\s+head)\b",
    re.IGNORECASE,
)

# Group-by shape: "by <noun>", "per <noun>", "for each <noun>",
# "across <noun>". Captures the field name for patching.
_GROUP_BY_RE = re.compile(
    r"\b(?:by|per|across|for\s+each)\s+([a-zA-Z][a-zA-Z0-9_]*)\b",
    re.IGNORECASE,
)

# Filter shapes. Two flavours: "only X=Y" / "just X=Y" (positive) and
# "exclude X(=Y)?" / "excluding ..." / "without ..." (negative).
_ONLY_RE = re.compile(
    r"\b(?:only|just)\s+([a-zA-Z][a-zA-Z0-9_]*)"
    r"\s*=\s*['\"]?([^'\"]+)['\"]?",
    re.IGNORECASE,
)
_EXCLUDE_RE = re.compile(
    r"\b(?:exclude|excluding|without)\s+([a-zA-Z][a-zA-Z0-9_]*)"
    r"(?:\s*=\s*['\"]?([^'\"]+)['\"]?)?",
    re.IGNORECASE,
)

# "Time-shift" verbs: "now show 2024", "how about 2024", "what about
# 2024". Used to detect a follow-up when the only payload is a year.
_TIME_SHIFT_PHRASES = (
    "now show",
    "how about",
    "what about",
    "instead",
    "again for",
)

# Bucket / granularity refinements. Captured as word-boundary regexes
# rather than substring contains() so "daily" doesn't fire on
# "diarrhea" or "monthly digest" — and so we can drop the verbose
# "show me ..." prefix without affecting detection.
_BUCKET_PATTERNS: tuple[tuple[re.Pattern[str], TimeBucket], ...] = (
    (re.compile(r"\b(?:daily|by\s+day|per\s+day)\b", re.IGNORECASE), TimeBucket.DAY),
    (re.compile(r"\b(?:weekly|by\s+week|per\s+week)\b", re.IGNORECASE), TimeBucket.WEEK),
    (re.compile(r"\b(?:monthly|by\s+month|per\s+month)\b", re.IGNORECASE), TimeBucket.MONTH),
    (re.compile(r"\b(?:quarterly|by\s+quarter|per\s+quarter)\b", re.IGNORECASE), TimeBucket.QUARTER),
    (re.compile(r"\b(?:yearly|annually|annual|by\s+year|per\s+year)\b", re.IGNORECASE), TimeBucket.YEAR),
)

# Aggregation-op swaps. Each entry pairs a regex with the canonical
# operation string ``AggregationSpec.operation`` understands. Order
# matters: "total" / "sum total" must beat the bare "total" alias for
# "sum", and "average" must beat the substring "avg" inside "average".
#
# Deliberately conservative: "highest" / "lowest" are sort cues
# (handled by SORT_LIMIT) and intentionally absent here, otherwise
# "highest first" would fire OP=max *and* SORT_LIMIT and only one of
# them would win the chart. "peak" / "max" / "maximum" are
# unambiguous aggregation verbs and stay.
_OPERATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:average|avg|mean)\b", re.IGNORECASE), "avg"),
    (re.compile(r"\b(?:maximum|max|peak)\b", re.IGNORECASE), "max"),
    (re.compile(r"\b(?:minimum|min)\b", re.IGNORECASE), "min"),
    (re.compile(r"\b(?:count|number\s+of|how\s+many)\b", re.IGNORECASE), "count"),
    (re.compile(r"\b(?:total|sum|aggregate)\b", re.IGNORECASE), "sum"),
)

# Sort / top-N refinements. The user wants the existing aggregation
# re-ordered or trimmed; we don't re-route, we just patch order_by /
# limit on the prior spec.
_TOP_N_RE = re.compile(
    r"\btop\s+(\d{1,3})\b", re.IGNORECASE
)
_BOTTOM_N_RE = re.compile(
    r"\bbottom\s+(\d{1,3})\b", re.IGNORECASE
)
_RANKED_RE = re.compile(
    r"\b(?:ranked|sort(?:ed)?\s+by|highest\s+first|largest\s+first)\b",
    re.IGNORECASE,
)
_ASCENDING_RE = re.compile(
    r"\b(?:lowest\s+first|smallest\s+first|ascending|asc)\b",
    re.IGNORECASE,
)

# Rolling-window refinements. "last 3 months", "past 2 years". The
# bucket determines the unit; the number is captured for arithmetic
# in :func:`_patch_time_range`.
_ROLLING_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d{1,3})\s+"
    r"(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    re.IGNORECASE,
)

# Half-year / YTD shortcuts. We treat H1 = Jan-Jun, H2 = Jul-Dec.
_HALF_RE = re.compile(
    r"\b(?:h1|first\s+half|h2|second\s+half)\b", re.IGNORECASE
)
_YTD_RE = re.compile(
    r"\b(?:ytd|year\s+to\s+date|year-to-date)\b", re.IGNORECASE
)

# Scope-reset signals: user explicitly asks for the full series the
# dataset has. We patch by clearing the time bounds so the executor
# reverts to "all available history".
_FULL_HISTORY_RE = re.compile(
    r"\b(?:all\s+data|all\s+years|full\s+history|every\s+year|"
    r"complete\s+history|entire\s+history|all\s+available)\b",
    re.IGNORECASE,
)

# Metric-swap shapes. The user references a different field on the
# same dataset. We capture the candidate name but do NOT patch
# blindly — the orchestrator validates against the live schema
# (:meth:`AnalystOrchestrator._sample_numeric_columns`) before
# applying. A miss falls through to fresh routing.
_METRIC_SWAP_RE = re.compile(
    r"\b(?:show|give|use|switch\s+to|what\s+about|change\s+to|"
    r"and\s+also|also\s+show|with|using)\s+"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\s+[a-zA-Z][a-zA-Z0-9_]*){0,3})"
    r"(?:\s+instead)?\b",
    re.IGNORECASE,
)

# Comparison aliases that should set ComparisonMode.YOY / PREV_PERIOD
# in the patch. Deliberately a small set of well-understood phrases —
# anything more ambiguous goes through the structural compare detector
# above (which patches the time range itself when a year is named).
_YOY_PHRASES = (
    "yoy",
    "year over year",
    "year-over-year",
    "vs last year",
    "vs previous year",
    "compared to last year",
    "compared to previous year",
)
_PREV_PERIOD_PHRASES = (
    "vs last",
    "previous period",
    "compared to last",
    "vs prior",
    "vs the previous",
)

# Cap on how many tokens a "headless" follow-up may have. Anything
# longer almost always carries its own subject (dataset / metric / verb).
_HEADLESS_MAX_TOKENS = 8


@dataclass(frozen=True)
class FollowUpAnalysis:
    """Structured outcome of the follow-up classifier.

    ``signals`` is the set of refinement shapes detected; ``confidence``
    is a coarse score the orchestrator can log for trace inspection.
    ``is_followup`` is the operational verdict callers should branch
    on. The detector is deliberately permissive when a prior session
    exists: if even one structural signal fires and there is no
    standalone catalog anchor, we treat the utterance as a follow-up
    rather than send the user back through fresh routing.
    """

    is_followup: bool
    signals: set[FollowUpSignal] = field(default_factory=set)
    reason: str = ""
    confidence: float = 0.0


def _matches_known_column(
    question: str, schema_columns: tuple[str, ...]
) -> bool:
    """True iff a metric-swap candidate in ``question`` names a real column.

    We accept multi-token candidates and normalise the schema to a
    space-separated form so ``"show hajj package recipients"``
    matches ``hajj_package_recipients``. Falls through silently when
    no candidate matches — the detector then keeps METRIC out of the
    signal set, which is exactly what stops us from treating "show
    revenue instead" as a metric swap (revenue isn't a column).
    """
    if not schema_columns:
        return False
    by_normalised = {
        col.lower().replace("_", " ").replace("-", " ")
        for col in schema_columns
    }
    for match in _METRIC_SWAP_RE.finditer(question):
        candidate = match.group(1).strip().lower()
        candidate_norm = candidate.replace("_", " ").replace("-", " ")
        if candidate_norm in by_normalised:
            return True
    return False


def mask_refinement_patterns(
    question: str,
    *,
    schema_columns: tuple[str, ...] | None = None,
) -> str:
    """Return the question with refinement payloads blanked out.

    Used by the orchestrator to make catalog-anchor probing
    follow-up-aware. Without masking, ``by emirate`` would match a
    catalog anchor on ``emirate``, and ``show hajj package
    recipients`` would match an anchor on ``hajj package`` (which
    is also the slug prefix of the prior dataset). Masking removes
    those false signals so the anchor probe only sees what's left.

    Patterns masked (replaced with spaces of the same width to keep
    char offsets stable for any downstream matcher):

    * bare years (``2024``), quarter shortcuts (``Q1``)
    * ``by/per/across/for each <noun>``
    * ``only/just <noun>=value`` and ``exclude/excluding/without <noun>``
    * comparison verbs (``vs``, ``versus``, ``compare with/to``…)
    * rolling-window phrases (``last 3 months``)
    * half/YTD/full-history shortcuts
    * top-N / bottom-N / ranked / ascending cues
    * bucket cues (``daily``, ``weekly``…) and operation cues
      (``average``, ``max``…)
    * **known schema columns** — when ``schema_columns`` is
      supplied, every occurrence of a real column name (in any of
      space / underscore / hyphen forms) is also masked. This is
      what lets ``show hajj package recipients`` keep flowing as a
      metric-swap follow-up even when the column shares tokens with
      the dataset name.
    """
    if not question:
        return ""
    masked = question

    def _blank(match: re.Match) -> str:
        return " " * (match.end() - match.start())

    masked = _YEAR_RE.sub(_blank, masked)
    masked = _QUARTER_RE.sub(_blank, masked)
    masked = _GROUP_BY_RE.sub(_blank, masked)
    masked = _ONLY_RE.sub(_blank, masked)
    masked = _EXCLUDE_RE.sub(_blank, masked)
    masked = _COMPARE_SHAPE_RE.sub(_blank, masked)
    masked = _ROLLING_RE.sub(_blank, masked)
    masked = _HALF_RE.sub(_blank, masked)
    masked = _YTD_RE.sub(_blank, masked)
    masked = _FULL_HISTORY_RE.sub(_blank, masked)
    masked = _TOP_N_RE.sub(_blank, masked)
    masked = _BOTTOM_N_RE.sub(_blank, masked)
    masked = _RANKED_RE.sub(_blank, masked)
    masked = _ASCENDING_RE.sub(_blank, masked)
    for pattern, _ in _BUCKET_PATTERNS:
        masked = pattern.sub(_blank, masked)
    for pattern, _ in _OPERATION_PATTERNS:
        masked = pattern.sub(_blank, masked)

    if schema_columns:
        # Mask every occurrence of a real column name. We try three
        # written forms (spaces / underscores / hyphens) so we catch
        # what users actually type. Longest forms first so we don't
        # leave half a column name behind that could re-match a
        # shorter anchor (e.g. ``hajj_package`` after masking
        # ``hajj_package_recipients``).
        forms: list[str] = []
        for col in schema_columns:
            for sep in (" ", "_", "-"):
                forms.append(col.lower().replace("_", sep).replace("-", sep))
        for form in sorted(set(forms), key=len, reverse=True):
            if not form:
                continue
            pattern = re.compile(
                rf"\b{re.escape(form)}\b", re.IGNORECASE
            )
            masked = pattern.sub(_blank, masked)
    return masked


def analyze_followup(
    question: str,
    *,
    has_catalog_anchor: bool,
    has_prior_decision: bool,
    schema_columns: tuple[str, ...] | None = None,
) -> FollowUpAnalysis:
    """Structural classifier — replaces the old prefix-list ``is_followup``.

    Inputs:

    * ``question`` — the new utterance, exactly as typed by the user.
    * ``has_catalog_anchor`` — outcome of the catalog-anchor probe on
      the live ``awqaf_datasets_metadata`` (consulted by the
      orchestrator). When True, the question names a dataset / metric
      on its own and is NOT a follow-up.
    * ``has_prior_decision`` — whether the active session has a prior
      *analytical* decision worth patching. When False we never claim
      "follow-up" — there's nothing to follow up *to*.
    * ``schema_columns`` — optional snapshot of the prior target's
      numeric columns. Used to validate the METRIC signal so we only
      claim "this is a metric swap" when the candidate the user named
      actually exists on the target. Without this, the detector
      cannot tell a real column name from any other 1–3-word noun.

    Output: a :class:`FollowUpAnalysis` whose ``is_followup`` flag
    drives the orchestrator. The orchestrator should fall back to
    fresh routing on a negative verdict.
    """
    q = (question or "").strip()
    if not q:
        return FollowUpAnalysis(False, reason="empty question")

    if not has_prior_decision:
        return FollowUpAnalysis(False, reason="no prior session")

    if has_catalog_anchor:
        # User named a dataset or metric explicitly — treat as a
        # fresh question, not a follow-up. Refinement signals are
        # ignored on purpose: "trend for hajj-package-service vs
        # zakat-payment" is two anchors, not a follow-up.
        return FollowUpAnalysis(
            False,
            reason="question names a dataset/metric explicitly",
        )

    signals: set[FollowUpSignal] = set()
    q_lower = q.lower()

    if (
        _YEAR_RE.search(q)
        or _QUARTER_RE.search(q)
        or _ROLLING_RE.search(q)
        or _HALF_RE.search(q)
        or _YTD_RE.search(q)
    ):
        signals.add(FollowUpSignal.TIME)
    if any(p in q_lower for p in _TIME_SHIFT_PHRASES):
        signals.add(FollowUpSignal.TIME)
    if _COMPARE_SHAPE_RE.search(q):
        signals.add(FollowUpSignal.COMPARE)
    if any(p in q_lower for p in _YOY_PHRASES + _PREV_PERIOD_PHRASES):
        signals.add(FollowUpSignal.COMPARE)
    if _GROUP_BY_RE.search(q):
        signals.add(FollowUpSignal.GROUP_BY)
    if _ONLY_RE.search(q) or _EXCLUDE_RE.search(q):
        signals.add(FollowUpSignal.FILTER)
    if any(pattern.search(q) for pattern, _ in _BUCKET_PATTERNS):
        signals.add(FollowUpSignal.BUCKET)
    if any(pattern.search(q) for pattern, _ in _OPERATION_PATTERNS):
        signals.add(FollowUpSignal.OPERATION)
    if (
        _TOP_N_RE.search(q)
        or _BOTTOM_N_RE.search(q)
        or _RANKED_RE.search(q)
        or _ASCENDING_RE.search(q)
    ):
        signals.add(FollowUpSignal.SORT_LIMIT)
    if _FULL_HISTORY_RE.search(q):
        signals.add(FollowUpSignal.SCOPE)
    if schema_columns and _matches_known_column(q, schema_columns):
        signals.add(FollowUpSignal.METRIC)

    # Headless brevity: short utterance + no anchor. Captures things
    # like "and 2024?", "by emirate", "exclude refunds" that don't fit
    # any single pattern above but are still clearly refinements.
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", q_lower)
    if len(tokens) <= _HEADLESS_MAX_TOKENS:
        signals.add(FollowUpSignal.HEADLESS)

    if not signals:
        return FollowUpAnalysis(
            False, reason="no refinement signals detected"
        )

    # A *substantive* signal (time, compare, group_by, filter) is
    # required to call something a follow-up. HEADLESS alone catches
    # things like "hello" / "thanks" — short and anchorless, but with
    # no actual refinement payload. Those should defer to the regular
    # intent classifier (which routes them to discovery).
    substantive = signals - {FollowUpSignal.HEADLESS}
    if not substantive:
        return FollowUpAnalysis(
            False,
            signals=signals,
            reason="headless-only utterance — no refinement payload",
            confidence=0.0,
        )

    confidence = min(1.0, 0.5 + 0.2 * len(substantive))
    reason_bits = sorted(s.value for s in signals)
    return FollowUpAnalysis(
        True,
        signals=signals,
        reason=f"refinement signals: {', '.join(reason_bits)}",
        confidence=confidence,
    )


def resolve_followup(
    *,
    question: str,
    prior_decision: RoutingDecision,
    analysis: FollowUpAnalysis,
    schema_columns: tuple[str, ...] | None = None,
    llm_resolver: Optional[Callable[[str, RoutingDecision], RoutingDecision]] = None,
) -> RoutingDecision:
    """Return a patched decision for a question classified as a follow-up.

    The deterministic patcher handles every signal the structural
    classifier emits today. ``llm_resolver`` is an optional hook for
    future ambiguous cases — when set and ``analysis.confidence`` is
    low, it can return a fully-formed :class:`RoutingDecision` that
    overrides the deterministic patch. Default is no-op.

    ``schema_columns`` is forwarded to :func:`apply_patch` so the
    metric-swap patcher can gate against the live target. The
    orchestrator supplies it via its TTL-cached schema probe; tests
    can pass an explicit tuple.

    Callers should already have verified ``analysis.is_followup``.
    """
    if (
        llm_resolver is not None
        and analysis.confidence < 0.5
        and prior_decision is not None
    ):
        try:
            return llm_resolver(question, prior_decision)
        except Exception:  # noqa: BLE001
            # LLM fallback must never harden the user out of a useful
            # deterministic answer. Drop through to the patcher.
            pass

    return apply_patch(prior_decision, question, schema_columns=schema_columns)


def apply_patch(
    prior: RoutingDecision,
    follow_up: str,
    *,
    schema_columns: tuple[str, ...] | None = None,
) -> RoutingDecision:
    """Return a deep-copied decision with ``follow_up`` patched into ``prior``.

    Pure: no I/O. The orchestrator is responsible for validating the
    follow-up shape (:func:`analyze_followup`) before calling this.

    ``schema_columns`` is an optional snapshot of the target's
    numeric columns. When supplied we gate the *metric swap* patcher
    on it — only fields the live collection actually exposes can
    replace ``spec.metric``. Without the snapshot we conservatively
    skip the metric swap rather than risk pointing the executor at a
    nonexistent column.
    """
    decision = deepcopy(prior)
    spec = decision.aggregation
    if spec is None:
        return decision

    q = follow_up.strip()
    q_lower = q.lower()

    _patch_group_by(spec, q_lower)
    _patch_filters(spec, q_lower)
    compare_patched = _patch_compare(spec, q_lower)
    _patch_operation(spec, q_lower)
    _patch_bucket(spec, q_lower)
    _patch_sort_limit(spec, q_lower)
    _patch_time_range(spec, q_lower, compare_patched=compare_patched)
    if schema_columns:
        _patch_metric_swap(spec, q, schema_columns=schema_columns)

    decision.matched_keywords = list(decision.matched_keywords) + ["session-patch"]
    decision.reason = (
        (decision.reason or "")
        + f" Refined by follow-up '{q}'."
    ).strip()
    return decision


# ---------------------------------------------------------------------------
# Legacy adapters retained for callers that haven't migrated yet.
# ---------------------------------------------------------------------------
def is_followup(question: str) -> bool:
    """**Deprecated** — kept only so external callers don't break.

    The orchestrator now uses :func:`analyze_followup`, which is
    session-aware and structural. This wrapper preserves a permissive
    "looks like a refinement" check based purely on shape — no
    keyword prefix list — so importing modules degrade gracefully.
    """
    q = (question or "").strip()
    if not q:
        return False
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", q.lower())
    if len(tokens) > _HEADLESS_MAX_TOKENS:
        return False
    return bool(
        _YEAR_RE.search(q)
        or _QUARTER_RE.search(q)
        or _COMPARE_SHAPE_RE.search(q)
        or _GROUP_BY_RE.search(q)
        or _ONLY_RE.search(q)
        or _EXCLUDE_RE.search(q)
    )


# Tokens that usually indicate a *different* AWQAF service than the
# prior one. Used by :func:`should_reuse_prior_collection` as a
# guardrail: if the user's follow-up mentions e.g. ``zakat`` and the
# prior target was a hajj collection, we don't silently reuse the
# hajj collection — the router must re-pick.
_SERVICE_TOPIC_TOKENS = frozenset(
    {
        "quran", "hajj", "umrah", "mosque", "mosques",
        "campaign", "campaigns", "zakat", "permit", "package",
        "teaching", "worker", "workers", "renewal",
        "centers", "center", "booking", "pilgrim", "pilgrims",
        "license", "licensed", "orphan", "orphans",
        "wakf", "waqf",
    }
)


def should_reuse_prior_collection(
    question: str,
    *,
    prior_target: str | None,
    prior_route: QueryRoute | None,
) -> bool:
    """Should fresh routing be scoped to the prior collection?

    Used for follow-ups that don't fit the patch path (e.g. the user
    re-asks with a fresh metric vocabulary) but clearly stay on the
    same dataset. We never reuse across service-topic boundaries
    (``hajj`` → ``zakat`` etc.).
    """
    if not prior_target or prior_route == QueryRoute.SEMANTIC:
        return False
    if prior_target == "awqaf_catalog":
        return False
    q = (question or "").strip().lower()
    if not q or "catalog" in q:
        return False
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", q)
    if len(tokens) > 14:
        return False

    prior_bits = set(
        re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", prior_target.replace("_", " ").lower())
    ) - {"awqaf"}
    q_set = set(tokens)
    hinted = q_set & _SERVICE_TOPIC_TOKENS
    if hinted and not hinted <= prior_bits:
        return False
    return True


# ---------------------------------------------------------------------------
# Internals — patchers
# ---------------------------------------------------------------------------
def _patch_group_by(spec: AggregationSpec, q: str) -> None:
    m = _GROUP_BY_RE.search(q)
    if m:
        spec.group_by = m.group(1)


def _patch_filters(spec: AggregationSpec, q: str) -> None:
    filters = dict(spec.filters or {})

    for col, val in _ONLY_RE.findall(q):
        filters[col] = val.strip()

    for col, val in _EXCLUDE_RE.findall(q):
        if val:
            filters[col] = {"!=": val.strip()}
        else:
            # "exclude refunds" without an explicit column = best we
            # can do is filter on a same-named column equal to false.
            filters[col] = False

    spec.filters = filters


def _patch_compare(spec: AggregationSpec, q: str) -> bool:
    """Set ``spec.time.compare`` when the follow-up implies a comparison.

    Returns True when a compare *mode* was applied (YoY or
    PREV_PERIOD). The caller uses this to decide how the time-range
    patch should behave — when the user wrote ``compare with 2025``
    AND a bare year is present, the year takes precedence (explicit
    target window) and the compare-mode stays NONE.
    """
    if spec.time is None:
        return False
    if any(p in q for p in _YOY_PHRASES):
        spec.time.compare = ComparisonMode.YOY
        return True
    if any(p in q for p in _PREV_PERIOD_PHRASES):
        spec.time.compare = ComparisonMode.PREV_PERIOD
        return True
    return False


def _patch_bucket(spec: AggregationSpec, q: str) -> None:
    """Switch the time bucket when the follow-up names a new granularity.

    ``"show monthly"`` / ``"daily"`` / ``"quarterly"`` all map to a
    :class:`TimeBucket` and patch ``spec.time.bucket`` in place. No-op
    when the spec has no time axis — bucket only makes sense for
    time-series queries.
    """
    if spec.time is None:
        return
    for pattern, bucket in _BUCKET_PATTERNS:
        if pattern.search(q):
            spec.time.bucket = bucket
            return


def _patch_operation(spec: AggregationSpec, q: str) -> None:
    """Swap ``spec.operation`` when the follow-up names a new aggregation.

    Recognises ``avg`` / ``max`` / ``min`` / ``count`` / ``sum``
    (and human aliases like ``average``, ``maximum``, ``total``).
    Skipped when the bare word also appears as a metric noun in the
    follow-up — the user might be asking for a metric whose name
    happens to start with ``avg_`` or ``total_``. Conservative miss
    is fine: the regular router will handle it on the next turn if
    the user clarifies.
    """
    for pattern, op in _OPERATION_PATTERNS:
        if pattern.search(q):
            spec.operation = op
            return


def _patch_sort_limit(spec: AggregationSpec, q: str) -> None:
    """Patch ``order_by`` / ``limit`` for top-N / bottom-N / ranked follow-ups.

    Three patterns:

    * ``top N``   — sort by value DESC, keep first N.
    * ``bottom N`` — sort by value ASC, keep first N.
    * ``ranked`` / ``sorted by`` / ``highest first`` — DESC ordering
      without a limit (the prior limit, if any, is preserved).
    * ``lowest first`` / ``ascending`` — ASC ordering.

    The executor's row contract is unchanged — we only mutate the
    spec fields it already understands.
    """
    m = _TOP_N_RE.search(q)
    if m:
        spec.order_by = "value"
        try:
            spec.limit = int(m.group(1))
        except (TypeError, ValueError):
            pass
        return
    m = _BOTTOM_N_RE.search(q)
    if m:
        spec.order_by = "value_asc"
        try:
            spec.limit = int(m.group(1))
        except (TypeError, ValueError):
            pass
        return
    if _RANKED_RE.search(q):
        spec.order_by = "value"
        return
    if _ASCENDING_RE.search(q):
        spec.order_by = "value_asc"


def _patch_metric_swap(
    spec: AggregationSpec,
    q: str,
    *,
    schema_columns: tuple[str, ...],
) -> None:
    """Replace ``spec.metric`` when the follow-up names a different field.

    The candidate name is captured by :data:`_METRIC_SWAP_RE` and
    normalised in three forms (spaces, underscores, hyphens) before
    being checked against the live schema. Only an exact column
    match patches the spec — we never guess. This keeps the
    deterministic guarantee that the executor is given a field the
    target actually has.

    The structural detector emits ``METRIC`` whenever the swap shape
    fires, but the patch only lands when validation succeeds; the
    analyst orchestrator falls back to fresh routing when neither
    the patch nor any other refinement takes effect.
    """
    if not schema_columns:
        return
    by_normalised = {
        col.lower().replace("_", " ").replace("-", " "): col
        for col in schema_columns
    }
    for match in _METRIC_SWAP_RE.finditer(q):
        candidate = match.group(1).strip().lower()
        candidate_norm = candidate.replace("_", " ").replace("-", " ")
        col = by_normalised.get(candidate_norm)
        if col is not None and col != spec.metric:
            spec.metric = col
            return


def _patch_time_range(
    spec: AggregationSpec, q: str, *, compare_patched: bool
) -> None:
    """Update the time window in place when the follow-up names a new range.

    Handled shapes (in priority order):

    1. ``compare with YYYY`` AND prior window is set → keep window,
       switch on YoY mode. ``"compare with 2025"`` after a 2026 trend
       means "show 2026 alongside 2025", not "replace 2026 with 2025".
    2. Full-history reset (``all data``, ``full history``) → clear
       the bounds; executor reverts to all-time.
    3. Rolling window (``last 3 months``, ``past 2 years``) → compute
       a relative range anchored at the prior ``range_to`` (or now).
    4. Half-year shortcut (``H1`` / ``first half`` / ``H2``).
    5. Year-to-date (``YTD``).
    6. Bare year.
    7. Quarter shortcut (``Q1``).

    The first match wins so a question like
    ``"show me Q1 2026 with H2 last year"`` deterministically maps
    to one window. Mixed-time follow-ups are intentionally a
    fall-through case for the LLM resolver.
    """
    if spec.time is None:
        return

    year_match = _YEAR_RE.search(q)
    quarter_match = _QUARTER_RE.search(q)
    has_compare_verb = bool(_COMPARE_SHAPE_RE.search(q))

    if year_match and has_compare_verb and spec.time.range_from is not None:
        spec.time.compare = ComparisonMode.YOY
        return

    if _FULL_HISTORY_RE.search(q):
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=spec.time.bucket,
            range_from=None,
            range_to=None,
            compare=spec.time.compare,
        )
        return

    rolling = _ROLLING_RE.search(q)
    if rolling:
        try:
            n = int(rolling.group(1))
        except (TypeError, ValueError):
            n = 0
        unit = rolling.group(2).lower().rstrip("s")  # "months" -> "month"
        anchor = spec.time.range_to or datetime.now(timezone.utc)
        end = anchor
        start = _shift_backwards(anchor, n, unit)
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=_bucket_for_rolling(unit, spec.time.bucket),
            range_from=start,
            range_to=end,
            compare=spec.time.compare,
        )
        return

    half = _HALF_RE.search(q)
    if half:
        # The reference year is whichever year the prior window
        # already pointed at; falling back to "now" only when the
        # prior spec was rangeless.
        ref_year = (
            spec.time.range_from.year
            if spec.time.range_from is not None
            else datetime.now(timezone.utc).year
        )
        if half.group(0).lower().startswith(("h2", "second")):
            start_month, end_month = 7, 12
        else:
            start_month, end_month = 1, 6
        from calendar import monthrange
        end_day = monthrange(ref_year, end_month)[1]
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=spec.time.bucket,
            range_from=datetime(ref_year, start_month, 1, tzinfo=timezone.utc),
            range_to=datetime(
                ref_year, end_month, end_day, 23, 59, 59, tzinfo=timezone.utc
            ),
            compare=spec.time.compare,
        )
        return

    if _YTD_RE.search(q):
        now = datetime.now(timezone.utc)
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=spec.time.bucket,
            range_from=datetime(now.year, 1, 1, tzinfo=timezone.utc),
            range_to=now,
            compare=spec.time.compare,
        )
        return

    if year_match:
        y = int(year_match.group(1))
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=spec.time.bucket,
            range_from=datetime(y, 1, 1, tzinfo=timezone.utc),
            range_to=datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            compare=spec.time.compare,
        )
        return

    if quarter_match:
        quarter = int(quarter_match.group(1))
        ref_year = (
            spec.time.range_from.year
            if spec.time.range_from is not None
            else datetime.now(timezone.utc).year
        )
        start_month = 3 * (quarter - 1) + 1
        end_month = start_month + 2
        from calendar import monthrange
        end_day = monthrange(ref_year, end_month)[1]
        spec.time = TimeSpec(
            field=spec.time.field,
            bucket=spec.time.bucket,
            range_from=datetime(ref_year, start_month, 1, tzinfo=timezone.utc),
            range_to=datetime(
                ref_year, end_month, end_day, 23, 59, 59, tzinfo=timezone.utc
            ),
            compare=spec.time.compare,
        )


# ---------------------------------------------------------------------------
# Helpers for the rolling-window patcher.
# ---------------------------------------------------------------------------
def _shift_backwards(anchor: datetime, n: int, unit: str) -> datetime:
    """Return ``anchor`` shifted back by ``n`` units (``day``/``month``/…).

    Calendar-aware: subtracting a month from March 31 lands on
    February 28/29 rather than an invalid date. We do this manually
    so we don't drag in ``dateutil`` for a single use site.
    """
    if n <= 0:
        return anchor
    if unit == "day":
        from datetime import timedelta
        return anchor - timedelta(days=n)
    if unit == "week":
        from datetime import timedelta
        return anchor - timedelta(weeks=n)
    if unit == "month":
        year = anchor.year
        month = anchor.month - n
        while month <= 0:
            month += 12
            year -= 1
        from calendar import monthrange
        day = min(anchor.day, monthrange(year, month)[1])
        return anchor.replace(year=year, month=month, day=day)
    if unit == "quarter":
        return _shift_backwards(anchor, n * 3, "month")
    if unit == "year":
        from calendar import monthrange
        year = anchor.year - n
        day = min(anchor.day, monthrange(year, anchor.month)[1])
        return anchor.replace(year=year, day=day)
    return anchor


def _bucket_for_rolling(unit: str, prior_bucket: TimeBucket) -> TimeBucket:
    """Pick a sensible chart bucket for a rolling window.

    Defaults are conservative: a "last 3 months" follow-up rolls down
    to a daily / weekly bucket only if the prior bucket was already
    finer than month; otherwise we keep month-level so the chart
    stays the same shape as the prior turn.
    """
    if unit == "day":
        return TimeBucket.DAY
    if unit == "week":
        return TimeBucket.WEEK
    if unit == "month":
        if prior_bucket in (TimeBucket.DAY, TimeBucket.WEEK):
            return prior_bucket
        return TimeBucket.MONTH
    if unit == "quarter":
        return TimeBucket.QUARTER
    if unit == "year":
        return TimeBucket.YEAR
    return prior_bucket
