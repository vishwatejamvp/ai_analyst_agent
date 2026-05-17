"""Question intent classification (pre-router, deterministic).

A small, fast, dependency-free classifier that decides — *before* we touch
Mongo or the LLM — what kind of response a question deserves. The goal is
to short-circuit on cases where the regular analytical pipeline would
either waste cycles (out-of-scope chatter) or produce a confusingly empty
answer (a new user asking "what can I ask?" against a column-level router).

Returned intents:

* ``DISCOVERY``    — user is exploring; show a catalog of datasets.
* ``OUT_OF_SCOPE`` — clearly not about the AWQAF datasets at all.
* ``COMPARISON``   — explicit metric-vs-metric or side-by-side intent.
* ``ANALYTICAL``   — the default; let the existing pipeline run.

The classifier never blocks a real question: any uncertainty falls through
to ``ANALYTICAL`` so the downstream router (and its "what IS available"
short-circuits) remains the source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from models.config import settings
from utils.logger import logger


class QuestionIntent(str, Enum):
    DISCOVERY = "discovery"
    OUT_OF_SCOPE = "out_of_scope"
    COMPARISON = "comparison"
    ANALYTICAL = "analytical"


# ---------------------------------------------------------------------------
# Signal vocabularies
# ---------------------------------------------------------------------------
_DISCOVERY_PHRASES: tuple[str, ...] = (
    # Greetings — short, friendly, no real data ask
    "hi",
    "hello",
    "hey",
    "yo",
    "howdy",
    "hiya",
    "good morning",
    "good afternoon",
    "good evening",
    "salam",
    "salaam",
    "assalamualaikum",
    "as-salamu alaykum",
    # Explicit "what can I do here" questions
    "what can i ask",
    "what should i ask",
    "what can ask",
    "what should ask",
    "what to ask",
    "what question",
    "what questions",
    "give question",
    "give me question",
    "give me a question",
    "give me questions",
    "suggest a question",
    "suggest questions",
    "starter question",
    "starter questions",
    "what do you have",
    "what can you do",
    "what data",
    "what datasets",
    "what is available",
    "what's available",
    "list datasets",
    "list all datasets",
    "show datasets",
    "show me datasets",
    "available data",
    "available datasets",
    "catalog",
    "data catalog",
    "help",
    "help me",
    "how do i use",
    "how to use",
    "getting started",
    "where do i start",
    "where to start",
    "i don't know what to ask",
    "i dont know what to ask",
)

_COMPARISON_PHRASES: tuple[str, ...] = (
    " vs ",
    " vs.",
    " versus ",
    "compare ",
    "comparison",
    "difference between",
    "side by side",
    "head to head",
    "both ",
    " against ",
)

# Tokens that strongly suggest the question is *not* about AWQAF data.
# Expanded list covers common off-topic categories. False positives would
# block legitimate questions, so we keep domain anchor protection active.
_OOS_TOKENS: tuple[str, ...] = (
    # Entertainment
    "weather",
    "joke",
    "movie",
    "song",
    "lyrics",
    "celebrity",
    "horoscope",
    
    # Finance (non-AWQAF)
    "stock price",
    "crypto",
    "bitcoin",
    "ethereum",
    "cryptocurrency",
    "forex",
    "trading",
    
    # Sports
    "football score",
    "cricket score",
    "sports",
    "game",
    "match",
    "tournament",
    
    # Politics
    "election",
    "president",
    "prime minister",
    "government",
    "politics",
    "politician",
    
    # Technology/Programming
    "code",
    "programming",
    "python",
    "javascript",
    "react",
    "angular",
    "software",
    "algorithm",
    
    # Food/Travel
    "recipe",
    "restaurant",
    "hotel",
    "travel",
    "flight",
    "booking",
    "vacation",
    
    # Health/Medical
    "medical",
    "doctor",
    "medicine",
    "health",
    "disease",
    "symptom",
    
    # News/Current Events
    "news",
    "breaking",
    "latest news",
    "current events",
)

# Tokens that anchor a question to the AWQAF domain. If any of these appear,
# we treat the question as in-scope even when other signals look weak.
_DOMAIN_TOKENS: tuple[str, ...] = (
    "awqaf",
    "hajj",
    "umrah",
    "zakat",
    "mosque",
    "mosques",
    "quran",
    "permit",
    "pilgrim",
    "pilgrimage",
    "transaction",
    "transactions",
    "campaign",
    "campaigns",
    "recipient",
    "recipients",
    "disbursement",
    "donation",
    "donations",
    "fatwa",
    "fatwas",
    "emirate",
    "uae",
)


_TOKENISER = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


@dataclass(frozen=True)
class IntentResult:
    """Outcome of intent classification with a short human-readable reason.

    ``source`` records WHO produced the result:

    * ``"rule"``       — the deterministic keyword classifier
    * ``"distiller"``  — the trained student model overrode the rule
                         (Build #8). Only happens when the student is
                         confident AND disagrees with the rule.
    * ``"distiller-agree"`` — the student agreed with the rule; the
                         rule's verdict is shipped but the trace
                         records that the model concurred.

    The default is ``"rule"`` so existing call sites continue to work.
    """

    intent: QuestionIntent
    reason: str
    source: str = "rule"


def classify(question: str) -> IntentResult:
    """Public entry: multi-layer intent classification.

    Classification layers (in order):
    
    1. **Rule classifier** (always runs) — Fast, deterministic keyword matching
    2. **Semantic OOS** (Build #9, optional) — Embedding-based OOS detection
    3. **Intent distiller** (Build #8, optional) — ML student model
    
    The deterministic rule classifier (see :func:`_rule_classify`) is
    always run because it is fast, dependency-free, and right most of
    the time. When ``settings.semantic_oos_enabled`` is True, we check
    for out-of-scope questions using sentence embeddings before consulting
    the distiller. When ``settings.intent_distiller_enabled`` is True we
    *additionally* consult the trained student model and apply a
    strict policy:

      * student unavailable (no artifact / load failed)  → ship rule
      * student.confidence < threshold                   → ship rule
      * student.intent == rule.intent                    → ship rule (with source="distiller-agree")
      * student.intent != rule.intent AND confident      → ship student (source="distiller")

    The student therefore can ONLY fix rule misses; it can never make
    them worse without first proving high confidence in a different
    answer. If the user later disables distillation, behaviour
    instantly reverts to the deterministic rule path with no code
    change.
    """
    rule = _rule_classify(question)

    # Layer 2: Semantic OOS detection (Build #9)
    # Check for out-of-scope questions using embeddings BEFORE distiller.
    # This catches novel phrasings that don't match static keywords.
    if settings.semantic_oos_enabled and rule.intent != QuestionIntent.OUT_OF_SCOPE:
        try:
            from services.semantic_oos import semantic_oos_detector  # pylint: disable=import-outside-toplevel
            
            is_oos, confidence = semantic_oos_detector.is_oos(question)
            if is_oos:
                return IntentResult(
                    QuestionIntent.OUT_OF_SCOPE,
                    f"semantic OOS detection (confidence={confidence:.2f})",
                    source="semantic",
                )
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.debug(
                f"Semantic OOS check failed; continuing with rule: "
                f"{type(exc).__name__}: {exc}"
            )

    # Layer 3: Intent distiller (Build #8)
    # Cheap exit when distillation is disabled — keeps existing tests
    # and benchmarks unchanged.
    if not settings.intent_distiller_enabled:
        return rule

    # Lazy import: importing intent_distiller pulls joblib which we
    # don't want on the cold path when the feature is off.
    try:
        from services.intent_distiller import intent_distiller  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        logger.debug(
            f"Intent distiller import failed; using rule classifier: "
            f"{type(exc).__name__}: {exc}"
        )
        return rule

    prediction = intent_distiller.predict(question)
    if prediction is None:
        # Model artifact not loaded (probably not trained yet) — quietly
        # use the rule. ``IntentDistiller._get_pipeline`` already logged
        # the reason on first call.
        return rule

    threshold = float(settings.intent_distiller_confidence_threshold)
    if prediction.confidence < threshold:
        return IntentResult(
            intent=rule.intent,
            reason=(
                f"{rule.reason} (distiller {prediction.intent.value}@"
                f"{prediction.confidence:.2f} below threshold {threshold:.2f})"
            ),
            source="rule",
        )

    if prediction.intent == rule.intent:
        return IntentResult(
            intent=rule.intent,
            reason=(
                f"{rule.reason} (distiller agreed @"
                f"{prediction.confidence:.2f})"
            ),
            source="distiller-agree",
        )

    # High-confidence disagreement: trust the student. We log so this
    # is visible in the trace — these are the cases worth eyeballing.
    logger.info(
        f"Intent distiller overrode rule on {question!r}: "
        f"rule={rule.intent.value} → "
        f"distiller={prediction.intent.value} "
        f"(conf={prediction.confidence:.2f})"
    )
    return IntentResult(
        intent=prediction.intent,
        reason=(
            f"distiller override (rule said {rule.intent.value}, "
            f"distiller {prediction.intent.value}@{prediction.confidence:.2f})"
        ),
        source="distiller",
    )


def _rule_classify(question: str) -> IntentResult:
    """The original deterministic rule classifier.

    Ordering:
      1. empty / explicit discovery phrase → DISCOVERY
      2. vague question without any domain anchor → DISCOVERY
         (so "hi", "what should I ask?", "help me start" never burn 12 s
         going through the semantic pipeline)
      3. out-of-scope chatter → OUT_OF_SCOPE
      4. explicit comparison phrase → COMPARISON
      5. else → ANALYTICAL

    Step 1 uses **whole-message equality** (after normalising trailing
    punctuation), not substring matching. Earlier versions matched any
    discovery phrase as a substring, so a real analytical question like
    "graph on occupancy rate and revenue with all available data years"
    was short-circuited to DISCOVERY because it happened to contain
    ``available data``. A real catalog-anchor probe still runs in
    ``analyst_service.answer`` as a second guard.
    """
    q = (question or "").lower().strip()
    if not q:
        return IntentResult(QuestionIntent.DISCOVERY, "empty question")

    if _is_explicit_discovery_phrase(q):
        return IntentResult(QuestionIntent.DISCOVERY, "explicit discovery phrase")

    if _looks_like_discovery(q):
        return IntentResult(
            QuestionIntent.DISCOVERY,
            "vague question with no domain anchor",
        )

    if _is_out_of_scope(q):
        return IntentResult(QuestionIntent.OUT_OF_SCOPE, "no domain tokens; off-topic")

    if _matches_any(q, _COMPARISON_PHRASES):
        return IntentResult(QuestionIntent.COMPARISON, "explicit comparison phrase")

    return IntentResult(QuestionIntent.ANALYTICAL, "default analytical")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
_TRAILING_PUNCT = ".,!?;:\"')]}"


def _normalize_for_phrase_equality(text: str) -> str:
    """Lower-case, strip whitespace and trailing/leading punctuation.

    Used by :func:`_is_explicit_discovery_phrase` so that ``"hi!"``,
    ``"What can I ask?"``, and ``"  hello.  "`` all collapse to the
    canonical phrase form before equality comparison.
    """
    s = (text or "").lower().strip()
    s = s.strip(_TRAILING_PUNCT + " \t\n")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_explicit_discovery_phrase(text: str) -> bool:
    """Whole-message equality match against :data:`_DISCOVERY_PHRASES`.

    Substring matching (the previous behaviour) caused false positives
    on real analytical questions whose wording happened to contain a
    catalog phrase — e.g. ``"available data"`` inside
    ``"all available data years"``. Whole-message equality keeps short
    greetings ("hi", "hello") and meta-questions ("what can I ask?")
    flowing into the discovery handler while letting longer questions
    fall through to the router.
    """
    norm = _normalize_for_phrase_equality(text)
    if not norm:
        return False
    for phrase in _DISCOVERY_PHRASES:
        if norm == phrase.strip().lower():
            return True
    return False


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Case-insensitive word-boundary match against any phrase.

    Substring matching would mis-fire on short words like ``hi`` (matches
    ``this``, ``history``). Word boundaries pin the phrase to whole-word
    occurrences, including at the start / end of the string.

    Leading/trailing whitespace in a phrase is stripped before matching
    so authors can write ``"compare "`` (visually emphasising "compare
    is followed by something") without the lookarounds failing on a
    space-then-word boundary. The ``(?<!\\w)…(?!\\w)`` pair already
    enforces whole-word matching, making the inline spaces redundant.
    """
    for p in phrases:
        if not p:
            continue
        token = p.strip()
        if not token:
            continue
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text):
            return True
    return False


def _has_domain_anchor(q: str) -> bool:
    return _matches_any(q, _DOMAIN_TOKENS)


def _is_out_of_scope(q: str) -> bool:
    """True when the question has off-topic markers and no domain anchor.

    Domain anchors win: "tell me about Hajj weather" is still in-scope even
    though it mentions weather, because ``hajj`` is a domain token.
    """
    if _has_domain_anchor(q):
        return False
    return _matches_any(q, _OOS_TOKENS)


def _looks_like_discovery(q: str) -> bool:
    """Heuristic for "user hasn't named anything specific" questions.

    Catches:
      * single-token greetings/utterances not in the phrase list ("yo!")
      * meta questions with no dataset anchor ("can you give me a question")
      * very short prompts with no domain word ("anything else?")
    """
    if _has_domain_anchor(q):
        return False

    tokens = [t for t in _TOKENISER.findall(q)]
    if not tokens:
        return True
    if len(tokens) <= 2:
        return True
    # Meta-asking pattern: a question pronoun *and* a meta verb anywhere.
    pronouns = {"what", "which", "how", "where", "can", "could", "should", "would"}
    meta_verbs = {
        "ask", "asked", "asking", "question", "questions", "answer",
        "answers", "do", "use", "start", "begin", "explore", "explain",
        "tell", "show", "give", "suggest",
    }
    tset = set(tokens)
    if tset & pronouns and tset & meta_verbs:
        return True
    return False


def extract_known_columns(question: str, columns: list[str]) -> list[str]:
    """Return the subset of ``columns`` whose name appears in ``question``.

    Matching is tolerant of underscores vs. spaces vs. hyphens so that the
    follow-up suggestions surfaced by the analyst layer (which display the
    human form ``Smart App Transactions``) can be copy-pasted back without
    failing the router's metric regex.
    """
    q = (question or "").lower()
    found: list[str] = []
    for col in columns:
        pattern = re.escape(col).replace("_", r"[\s_\-]")
        if re.search(rf"\b{pattern}\b", q):
            found.append(col)
    return found


def tokenize(question: str) -> list[str]:
    """Light tokenizer used by partial-relevance detection."""
    return [m.group(0).lower() for m in _TOKENISER.finditer(question)]


def extract_compared_columns(
    question: str,
    columns: list[str],
    *,
    target_slug: str | None = None,
) -> list[str]:
    """Return the columns the user is comparing, even when not contiguous.

    ``extract_known_columns`` only matches when a column's exact phrase
    appears in the question. That misses natural comparison wording like
    "compare smart app and website transactions", where the joiner "and"
    breaks the contiguous phrase. This helper looks for *distinguishing*
    tokens (each column's prefix, e.g. ``smart_app`` from
    ``smart_app_transactions``) so the comparison fan-out has something to
    work with.

    Target-slug tokens are masked first so a question naming the dataset
    (``hajj-permit-service``) doesn't accidentally match an unrelated
    column whose first token happens to be ``hajj``.
    """
    q = (question or "").lower()
    if target_slug:
        for tok in target_slug.split("_"):
            if tok:
                q = re.sub(rf"\b{re.escape(tok)}\b", " ", q)

    matched: list[str] = []
    for col in columns:
        tokens = col.lower().split("_")
        if len(tokens) <= 1:
            # Single-token columns have no prefix to disambiguate them.
            if re.search(rf"\b{re.escape(tokens[0])}\b", q):
                matched.append(col)
            continue
        prefix = tokens[:-1]
        if any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in prefix):
            matched.append(col)
    return matched
