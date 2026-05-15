"""Curated metric glossary service.

The glossary is the **organizational ground truth** for what a metric or
business term means. Approved entries change query construction; draft
entries are surfaced as "unverified" but do NOT modify the query (per the
governance hierarchy decided in v1):

    1. Curated glossary (approved)  →  changes the query
    2. User clarification in chat   →  changes the query (session-scoped)
    3. Structured DB output         →  authoritative numbers
    4. Informal vector hits          →  color and explanation only

Storage: a dedicated MongoDB collection (``glossary`` by default) so
admins can edit it via the API and the data survives restarts. We never
auto-promote informal docs into this store — promotion is an explicit
admin action.

Lookup is deterministic and fast: word-boundary match against the
canonical ``term`` and any ``aliases``, scoped to ``applies_to_targets``.
The LLM-assisted planner in a later sprint can replace this with a
semantic match while keeping the same :class:`GlossaryMatch` contract.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import PyMongoError

from models.enums import MetricStatus
from models.schemas import GlossaryMatch, MetricDefinition, MetricFormula
from services.mongo_service import MongoService, mongo_service
from utils.exceptions import DatabaseError
from utils.logger import logger

GLOSSARY_COLLECTION = "glossary"


class KnowledgeBaseService:
    """CRUD + lookup for curated metric definitions."""

    def __init__(
        self,
        mongo: MongoService | None = None,
        collection: str = GLOSSARY_COLLECTION,
    ) -> None:
        self.mongo = mongo or mongo_service
        self.collection_name = collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def list(self, status: MetricStatus | None = None) -> list[MetricDefinition]:
        try:
            filter_: dict[str, Any] = {}
            if status is not None:
                filter_["status"] = status.value
            docs = self.mongo.find(self.collection_name, filter_, limit=500)
        except DatabaseError as exc:
            logger.warning(f"Glossary list unavailable: {exc}")
            return []
        return [_doc_to_definition(d) for d in docs if d]

    def get(self, definition_id: str) -> MetricDefinition | None:
        try:
            docs = self.mongo.find(
                self.collection_name, {"id": definition_id}, limit=1
            )
        except DatabaseError as exc:
            logger.warning(f"Glossary get unavailable: {exc}")
            return None
        return _doc_to_definition(docs[0]) if docs else None

    def upsert(self, definition: MetricDefinition) -> MetricDefinition:
        """Insert or replace a glossary entry by ``id``."""
        doc = definition.model_dump(mode="json")
        doc["status"] = definition.status.value
        doc["updated_at"] = (
            (definition.updated_at or datetime.now(timezone.utc))
            .astimezone(timezone.utc)
            .isoformat()
        )
        try:
            coll = self.mongo.collection(self.collection_name)
            coll.replace_one({"id": definition.id}, doc, upsert=True)
        except PyMongoError as exc:
            raise DatabaseError(f"Glossary upsert failed: {exc}") from exc
        logger.info(f"Glossary upserted: {definition.id} ({definition.status.value})")
        return _doc_to_definition(doc)

    def delete(self, definition_id: str) -> bool:
        try:
            coll = self.mongo.collection(self.collection_name)
            result = coll.delete_one({"id": definition_id})
        except PyMongoError as exc:
            raise DatabaseError(f"Glossary delete failed: {exc}") from exc
        deleted = bool(result.deleted_count)
        if deleted:
            logger.info(f"Glossary deleted: {definition_id}")
        return deleted

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(
        self,
        question: str,
        *,
        target: str | None = None,
    ) -> GlossaryMatch | None:
        """Return the best match for ``question`` or None if nothing matches.

        The match is scoped to ``target`` when ``applies_to_targets`` is set.
        Only :data:`MetricStatus.APPROVED` and :data:`MetricStatus.DRAFT`
        entries are considered; deprecated entries are excluded. Draft
        matches still return so the caller can surface "unverified" — they
        will set ``applied_to_query=False`` regardless of confidence.
        """
        if not question:
            return None
        q = question.lower()

        try:
            docs = self.mongo.find(self.collection_name, limit=500)
        except DatabaseError as exc:
            logger.debug(f"Glossary lookup unavailable: {exc}")
            return None

        best: GlossaryMatch | None = None
        for doc in docs:
            try:
                definition = _doc_to_definition(doc)
            except (ValueError, TypeError) as exc:
                logger.debug(f"Skipping malformed glossary doc: {exc}")
                continue

            if definition.status == MetricStatus.DEPRECATED:
                continue
            applies_to = list(definition.applies_to_targets or [])
            if applies_to and target is not None and target not in applies_to:
                continue

            phrases = [definition.term, *definition.aliases]
            for phrase in phrases:
                if not phrase:
                    continue
                if _word_boundary_in(phrase.lower(), q):
                    confidence = _phrase_confidence(phrase, q)
                    if best is None or confidence > best.confidence:
                        best = GlossaryMatch(
                            definition=definition,
                            matched_alias=phrase,
                            confidence=confidence,
                            applied_to_query=False,
                        )
        return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _doc_to_definition(doc: dict[str, Any]) -> MetricDefinition:
    """Translate a Mongo doc into a typed :class:`MetricDefinition`."""
    raw = {k: v for k, v in doc.items() if k != "_id"}
    formula = raw.get("formula") or {}
    if not isinstance(formula, dict):
        formula = {}
    raw["formula"] = MetricFormula(**formula)
    status = raw.get("status")
    if isinstance(status, str):
        try:
            raw["status"] = MetricStatus(status)
        except ValueError:
            raw["status"] = MetricStatus.DRAFT
    updated_at = raw.get("updated_at")
    if isinstance(updated_at, str):
        try:
            raw["updated_at"] = datetime.fromisoformat(
                updated_at.replace("Z", "+00:00")
            )
        except ValueError:
            raw["updated_at"] = None
    return MetricDefinition(**raw)


def _word_boundary_in(needle: str, haystack: str) -> bool:
    """True if ``needle`` appears in ``haystack`` on word boundaries."""
    pattern = rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])"
    return bool(re.search(pattern, haystack, re.IGNORECASE))


def _phrase_confidence(phrase: str, question: str) -> float:
    """Rough confidence: longer phrases that take up more of the question score higher."""
    if not question:
        return 0.0
    p = phrase.strip().lower()
    coverage = min(1.0, len(p) / max(len(question), 1))
    word_bonus = min(0.3, 0.1 * len(p.split()))
    return min(1.0, 0.6 + coverage * 0.3 + word_bonus)


knowledge_base_service = KnowledgeBaseService()
