"""Catalog-first dataset resolution (Build #9).

Resolves a natural-language question to an AWQAF facts collection by
searching ``awqaf_datasets_metadata`` (dataset slug, display name,
purpose, department, key metrics) instead of relying only on token
overlap against Mongo collection names.

Falls back to the legacy :func:`routing_service._score_target` path when
catalog confidence is low or the facts collection is not present.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from models.config import settings
from services.awqaf_normalize import (
    DATASETS_METADATA_COLLECTION,
    collection_name_from_service,
)
from services.mongo_service import MongoService, mongo_service
from utils.logger import logger

# Shared stopwords with routing_service — keep in sync.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "by", "with", "and", "or",
    "show", "give", "tell", "me", "us", "what", "is", "are", "this", "that",
    "from", "over", "all", "my", "our", "your", "please", "report",
}

_TOP_N_BY_RE = re.compile(
    r"\btop\s+\d+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+by\b",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", (text or "").lower())
        if t not in _STOPWORDS
    ]


@dataclass(frozen=True)
class CatalogDataset:
    """One row from ``awqaf_datasets_metadata`` prepared for scoring."""

    slug: str
    dataset_name: str
    facts_collection: str
    purpose: str = ""
    department: str = ""
    metric_phrases: tuple[str, ...] = ()


@dataclass
class CatalogMatchResult:
    """Outcome of catalog resolution."""

    facts_collection: str | None = None
    slug: str | None = None
    score: float = 0.0
    method: str = "none"  # catalog | none
    top_scores: dict[str, float] = field(default_factory=dict)


class CatalogRoutingService:
    """Score questions against the AWQAF dataset catalog."""

    _CACHE_TTL_SECONDS = 120

    def __init__(self, mongo: MongoService | None = None) -> None:
        self.mongo = mongo or mongo_service
        self._cache: tuple[float, list[CatalogDataset]] | None = None

    def load_catalog(self) -> list[CatalogDataset]:
        now = time.monotonic()
        if self._cache is not None and (now - self._cache[0]) < self._CACHE_TTL_SECONDS:
            return list(self._cache[1])

        rows: list[dict[str, Any]] = []
        try:
            rows = self.mongo.find(
                DATASETS_METADATA_COLLECTION,
                projection={
                    "_id": 0,
                    "dataset": 1,
                    "dataset_name": 1,
                    "purpose": 1,
                    "department": 1,
                    "key_metrics": 1,
                },
                limit=500,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Catalog routing: could not read {DATASETS_METADATA_COLLECTION} "
                f"({type(exc).__name__}: {exc})"
            )

        entries: list[CatalogDataset] = []
        for row in rows or []:
            slug = str(row.get("dataset") or "").strip().lower()
            if not slug:
                continue
            metrics: list[str] = []
            for m in row.get("key_metrics") or []:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("metric") or ""
                else:
                    name = m
                name = str(name or "").strip()
                if name:
                    metrics.append(name)
            entries.append(
                CatalogDataset(
                    slug=slug,
                    dataset_name=str(row.get("dataset_name") or "").strip(),
                    facts_collection=collection_name_from_service(slug),
                    purpose=str(row.get("purpose") or "").strip(),
                    department=str(row.get("department") or "").strip(),
                    metric_phrases=tuple(metrics),
                )
            )

        self._cache = (now, entries)
        return list(entries)

    def score_question(
        self,
        question: str,
        *,
        available_collections: list[str] | None = None,
    ) -> CatalogMatchResult:
        """Return the best catalog-backed facts collection for ``question``."""
        q = (question or "").strip()
        if not q:
            return CatalogMatchResult()

        entries = self.load_catalog()
        if not entries:
            return CatalogMatchResult()

        q_lower = q.lower()
        q_tokens = set(_tokenize(q))
        available = set(available_collections or [])

        scored: list[tuple[CatalogDataset, float]] = []
        for entry in entries:
            if available and entry.facts_collection not in available:
                continue
            score = self._score_entry(entry, q_lower, q_tokens)
            if score > 0:
                scored.append((entry, score))

        if not scored:
            return CatalogMatchResult()

        scored.sort(key=lambda x: x[1], reverse=True)
        best_entry, best_score = scored[0]
        min_score = settings.catalog_routing_min_score

        top_scores = {
            e.facts_collection: round(s, 2)
            for e, s in scored[:5]
        }

        if best_score < min_score:
            return CatalogMatchResult(
                score=best_score,
                method="none",
                top_scores=top_scores,
            )

        return CatalogMatchResult(
            facts_collection=best_entry.facts_collection,
            slug=best_entry.slug,
            score=best_score,
            method="catalog",
            top_scores=top_scores,
        )

    @staticmethod
    def _score_entry(
        entry: CatalogDataset,
        q_lower: str,
        q_tokens: set[str],
    ) -> float:
        score = 0.0

        # Strong: full display name or slug appears in the question.
        name_l = entry.dataset_name.lower()
        slug_l = entry.slug.lower()
        slug_spaced = slug_l.replace("-", " ").replace("_", " ")
        if name_l and len(name_l) >= 4 and name_l in q_lower:
            score += 8.0
        if slug_l and slug_l in q_lower:
            score += 7.0
        if slug_spaced and slug_spaced in q_lower:
            score += 6.0

        # Token overlap per field (weighted).
        def _overlap(text: str, weight: float) -> float:
            tokens = set(_tokenize(text.replace("-", " ").replace("_", " ")))
            if not tokens or not q_tokens:
                return 0.0
            return weight * len(q_tokens & tokens)

        score += _overlap(entry.dataset_name, 2.0)
        score += _overlap(entry.slug, 2.5)
        score += _overlap(entry.purpose, 1.0)
        score += _overlap(entry.department, 0.5)
        for phrase in entry.metric_phrases:
            score += _overlap(phrase, 1.5)
            pl = phrase.lower().replace("_", " ")
            if len(pl) >= 6 and pl in q_lower:
                score += 3.0

        return score


catalog_routing_service = CatalogRoutingService()


def extract_group_by_before_by(question: str, columns: list[str]) -> str | None:
    """``top 5 emirates by total revenue`` → ``emirate`` when column exists.

    Fixes the word-order gap in ``_guess_group_by`` which only inspects
    tokens *after* ``by``.
    """
    if not columns:
        return None
    col_set = {c.lower(): c for c in columns}
    m = _TOP_N_BY_RE.search(question or "")
    if not m:
        return None
    raw = m.group(1).lower()
    if raw in col_set:
        return col_set[raw]
    # Plural → singular heuristic (emirates → emirate).
    if raw.endswith("s") and len(raw) > 3:
        singular = raw[:-1]
        if singular in col_set:
            return col_set[singular]
    return None
