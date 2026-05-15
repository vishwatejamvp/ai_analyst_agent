"""AWQAF rows → MongoDB + FAISS writer.

Three target collection families:

* ``awqaf_<service>_facts``        — only flattened measurement rows
* ``awqaf_datasets_metadata``      — one doc per dataset (from contents.json)
* ``awqaf_datasets_glossary``      — one doc per dataset (from glossary.json)

Each ingest function builds the appropriate Mongo indexes after writing and
adds natural-language sentences to FAISS so semantic queries get good recall.
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from services.awqaf_normalize import (
    DATASETS_GLOSSARY_COLLECTION,
    DATASETS_METADATA_COLLECTION,
    collection_name_from_service,
    flatten_contents,
    flatten_dataset_glossary,
    flatten_facts,
    service_slug,
)
from services.embedding_service import embedding_service
from services.mongo_service import mongo_service
from services.vector_service import vector_service
from utils.logger import logger

_BATCH = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _chunks(seq: list[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _ensure_facts_indexes(collection: str) -> None:
    """Secondary indexes for the common filter patterns.

    Idempotency is provided by ``--replace`` / ``--drop-all`` at ingest time,
    not by a unique key — AWQAF datasets are heterogeneous (some are pure
    time series, some are registries with many rows per ``(year, slice)``)
    and a one-size-fits-all unique constraint would block legitimate rows.
    """
    coll = mongo_service.collection(collection)
    try:
        coll.create_index(
            [("dataset", ASCENDING), ("period", ASCENDING)],
            name="ix_dataset_period",
        )
        coll.create_index(
            [("dataset", ASCENDING), ("year", ASCENDING)],
            name="ix_dataset_year",
        )
        coll.create_index(
            [("dataset", ASCENDING), ("dimension", ASCENDING)],
            name="ix_dataset_dimension",
        )
    except PyMongoError as exc:
        logger.warning(f"[{collection}] index creation failed: {exc}")


def _ensure_metadata_indexes() -> None:
    coll = mongo_service.collection(DATASETS_METADATA_COLLECTION)
    try:
        coll.create_index([("dataset", ASCENDING)], name="ux_dataset", unique=True)
    except PyMongoError as exc:
        logger.warning(f"[{DATASETS_METADATA_COLLECTION}] index creation failed: {exc}")


def _ensure_glossary_indexes() -> None:
    coll = mongo_service.collection(DATASETS_GLOSSARY_COLLECTION)
    try:
        coll.create_index([("dataset", ASCENDING)], name="ux_dataset", unique=True)
    except PyMongoError as exc:
        logger.warning(f"[{DATASETS_GLOSSARY_COLLECTION}] index creation failed: {exc}")


def _embed_facts(
    collection: str,
    rows: list[dict[str, Any]],
    ids: list[str],
    *,
    dataset_label: str | None,
) -> int:
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    for row, _id in zip(rows, ids):
        sentence = embedding_service.fact_to_sentence(row, dataset_label=dataset_label)
        if not sentence:
            continue
        texts.append(sentence)
        metas.append(
            {
                "kind": "fact",
                "collection": collection,
                "document_id": _id,
                "dataset": row.get("dataset"),
                "period": row.get("period"),
                "year": row.get("year"),
                "month": row.get("month"),
            }
        )
    if not texts:
        return 0
    return vector_service.add(texts, metas)


def _embed_glossary_doc(doc: dict[str, Any]) -> int:
    """One vector per term + one per field description; all link to the same Mongo doc."""
    dataset = doc.get("dataset") or "dataset"
    label = doc.get("dataset_name") or dataset
    texts: list[str] = []
    metas: list[dict[str, Any]] = []

    for term in doc.get("terms") or []:
        if not isinstance(term, dict):
            continue
        sentence = embedding_service.glossary_term_to_sentence(term, dataset_label=label)
        if not sentence:
            continue
        texts.append(sentence)
        metas.append(
            {
                "kind": "glossary_term",
                "collection": DATASETS_GLOSSARY_COLLECTION,
                "dataset": dataset,
                "term": term.get("term"),
            }
        )

    fields = doc.get("fields") or {}
    if isinstance(fields, dict):
        for fk, meta in fields.items():
            if not isinstance(meta, dict):
                continue
            sentence = embedding_service.glossary_field_to_sentence(
                str(fk), meta, dataset_label=label
            )
            if not sentence:
                continue
            texts.append(sentence)
            metas.append(
                {
                    "kind": "glossary_field",
                    "collection": DATASETS_GLOSSARY_COLLECTION,
                    "dataset": dataset,
                    "field_key": str(fk),
                }
            )

    if not texts:
        return 0
    return vector_service.add(texts, metas)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def insert_facts(
    service: str,
    records: list[Any],
    *,
    replace: bool = False,
    embed: bool = True,
    dataset_label: str | None = None,
) -> dict[str, Any]:
    """Flatten ``all_records.json`` for one service and write into Mongo + FAISS."""
    collection = collection_name_from_service(service)
    rows = flatten_facts(records, service=service)
    summary: dict[str, Any] = {
        "collection": collection,
        "rows_normalized": len(rows),
        "documents_inserted": 0,
        "vectors_indexed": 0,
        "replaced": False,
    }

    if not rows:
        logger.info(f"[{collection}] no fact rows")
        return summary

    if replace:
        mongo_service.drop_collection(collection)
        summary["replaced"] = True

    for chunk in _chunks(rows, _BATCH):
        ids = mongo_service.insert_many(collection, chunk)
        summary["documents_inserted"] += len(ids)
        if embed:
            summary["vectors_indexed"] += _embed_facts(
                collection, chunk, ids, dataset_label=dataset_label
            )

    _ensure_facts_indexes(collection)
    logger.info(
        f"[{collection}] inserted={summary['documents_inserted']} "
        f"vectors={summary['vectors_indexed']}"
    )
    return summary


def insert_dataset_glossary(
    service: str,
    data: dict[str, Any],
    *,
    embed: bool = True,
) -> dict[str, Any]:
    """Upsert one ``awqaf_datasets_glossary`` doc for ``service``."""
    doc = flatten_dataset_glossary(data, service=service)
    summary: dict[str, Any] = {
        "collection": DATASETS_GLOSSARY_COLLECTION,
        "documents_inserted": 0,
        "vectors_indexed": 0,
    }
    if doc is None:
        logger.info(f"[{DATASETS_GLOSSARY_COLLECTION}] no doc for {service}")
        return summary

    coll = mongo_service.collection(DATASETS_GLOSSARY_COLLECTION)
    try:
        coll.replace_one({"dataset": doc["dataset"]}, doc, upsert=True)
        summary["documents_inserted"] = 1
    except PyMongoError as exc:
        logger.error(f"glossary upsert failed for {service}: {exc}")
        return summary

    _ensure_glossary_indexes()

    if embed:
        summary["vectors_indexed"] = _embed_glossary_doc(doc)
    logger.info(
        f"[{DATASETS_GLOSSARY_COLLECTION}] upserted {service} "
        f"(vectors={summary['vectors_indexed']})"
    )
    return summary


def insert_datasets_metadata(
    contents_payload: Any,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Upsert one row per dataset from ``contents.json``."""
    rows = flatten_contents(contents_payload)
    summary: dict[str, Any] = {
        "collection": DATASETS_METADATA_COLLECTION,
        "rows_normalized": len(rows),
        "documents_inserted": 0,
        "replaced": False,
    }
    if not rows:
        return summary

    if replace:
        mongo_service.drop_collection(DATASETS_METADATA_COLLECTION)
        summary["replaced"] = True

    coll = mongo_service.collection(DATASETS_METADATA_COLLECTION)
    try:
        for row in rows:
            coll.replace_one({"dataset": row["dataset"]}, row, upsert=True)
            summary["documents_inserted"] += 1
    except PyMongoError as exc:
        logger.error(f"datasets_metadata upsert failed: {exc}")

    _ensure_metadata_indexes()
    logger.info(
        f"[{DATASETS_METADATA_COLLECTION}] upserted={summary['documents_inserted']}"
    )
    return summary


def dataset_label_for(service: str) -> str:
    """Human-readable label for sentence prefixes; falls back to slug."""
    try:
        coll = mongo_service.collection(DATASETS_METADATA_COLLECTION)
        doc = coll.find_one({"dataset": service_slug(service)}, {"dataset_name": 1, "_id": 0})
        if doc and doc.get("dataset_name"):
            return str(doc["dataset_name"])
    except PyMongoError:
        pass
    return service_slug(service).replace("_", " ")


# ---------------------------------------------------------------------------
# HTTP-friendly single-payload entry
# ---------------------------------------------------------------------------
def ingest_payload(
    payload: Any,
    *,
    kind: str,
    service: str | None = None,
    replace: bool = False,
    embed: bool = True,
) -> dict[str, Any]:
    """Single-payload entry point for the HTTP endpoint.

    ``kind`` is one of ``"facts"``, ``"glossary"``, ``"contents"``.
    ``service`` is required for ``facts`` and ``glossary``.
    """
    if kind == "facts":
        if not service:
            raise ValueError("ingest_payload(kind='facts') requires service")
        if not isinstance(payload, list):
            raise ValueError("facts payload must be a JSON array (like all_records.json)")
        return insert_facts(
            service,
            payload,
            replace=replace,
            embed=embed,
            dataset_label=dataset_label_for(service),
        )

    if kind == "glossary":
        if not service:
            raise ValueError("ingest_payload(kind='glossary') requires service")
        if not isinstance(payload, dict):
            raise ValueError("glossary payload must be a JSON object")
        return insert_dataset_glossary(service, payload, embed=embed)

    if kind == "contents":
        return insert_datasets_metadata(payload, replace=replace)

    raise ValueError(f"Unknown ingest kind: {kind!r}")
