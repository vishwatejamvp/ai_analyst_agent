"""End-to-end ingestion: Excel/JSON → Mongo → embeddings → FAISS.

Flow per upload:

1. Read Excel/CSV/JSON with pandas (multiple sheets supported).
2. Normalize column names and rows to JSON-safe dicts.
3. Insert into a MongoDB collection (one document per row).
4. Build a stable text representation of each row.
5. Compute embeddings via :class:`EmbeddingService`.
6. Add vectors + metadata (collection name + Mongo ``_id``) to FAISS.

This guarantees the critical rule: every vector links back to a real
Mongo document, so retrieval can always cite the source row.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from models.schemas import IngestResponse
from services.embedding_service import EmbeddingService, embedding_service
from services.mongo_service import MongoService, mongo_service
from services.vector_service import VectorService, vector_service
from models.config import settings
from utils.exceptions import IngestionError
from utils.logger import logger

_NAME_CLEAN_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _normalize_collection_name(name: str) -> str:
    cleaned = _NAME_CLEAN_RE.sub("_", name.strip().lower()).strip("_")
    return cleaned or "default"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        _NAME_CLEAN_RE.sub("_", str(c).strip().lower()).strip("_") or f"col_{i}"
        for i, c in enumerate(df.columns)
    ]
    return df


def _to_json_safe(value: Any) -> Any:
    """Convert pandas/NumPy scalars into JSON/BSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _row_to_doc(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _to_json_safe(v) for k, v in row.items()}


class IngestionService:
    """Orchestrates Excel/CSV/JSON → Mongo → FAISS ingestion."""

    def __init__(
        self,
        mongo: MongoService | None = None,
        vector: VectorService | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.mongo = mongo or mongo_service
        self.vector = vector or vector_service
        self.embedder = embedder or embedding_service

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------
    def ingest_file(
        self,
        path: str | Path,
        collection: str | None = None,
        replace: bool = False,
    ) -> IngestResponse:
        """Ingest a local Excel/CSV/JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(file_path, sheet_name=None)
        elif suffix == ".csv":
            sheets = {file_path.stem: pd.read_csv(file_path)}
        elif suffix == ".json":
            sheets = {file_path.stem: pd.read_json(file_path)}
        else:
            raise IngestionError(f"Unsupported file type: {suffix}")

        return self._ingest_sheets(
            sheets,
            base_collection=collection or file_path.stem,
            replace=replace,
        )

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        collection: str,
        replace: bool = False,
    ) -> IngestResponse:
        """Ingest an in-memory DataFrame."""
        return self._ingest_sheets(
            {collection: df}, base_collection=collection, replace=replace
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ingest_sheets(
        self,
        sheets: dict[str, pd.DataFrame],
        base_collection: str,
        replace: bool,
    ) -> IngestResponse:
        total_rows = 0
        total_vectors = 0
        last_collection = ""
        sample: list[dict[str, Any]] = []

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                logger.info(f"Skipping empty sheet '{sheet_name}'")
                continue

            df = _normalize_columns(df)
            collection_name = _normalize_collection_name(
                base_collection if len(sheets) == 1 else f"{base_collection}_{sheet_name}"
            )
            last_collection = collection_name

            if replace:
                self.mongo.drop_collection(collection_name)
                logger.info(f"Dropped collection '{collection_name}' before reingest")

            documents = [_row_to_doc(rec) for rec in df.to_dict(orient="records")]
            if not documents:
                continue

            inserted_ids = self.mongo.insert_many(collection_name, documents)
            total_rows += len(inserted_ids)
            if not sample:
                sample = documents[:3]

            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for doc, _id in zip(documents, inserted_ids):
                text = self.embedder.row_to_text(doc, source=collection_name)
                if not text:
                    continue
                texts.append(text)
                metadatas.append(
                    {
                        "collection": collection_name,
                        "document_id": _id,
                        "payload": doc,
                    }
                )

            added = self.vector.add(texts, metadatas)
            total_vectors += added
            logger.info(
                f"Ingested {len(inserted_ids)} rows and {added} vectors "
                f"into '{collection_name}'"
            )
            
            # Invalidate Redis caches for this collection
            self._invalidate_caches(collection_name)

        if total_rows == 0:
            raise IngestionError("No rows were ingested (file empty or unreadable)")

        return IngestResponse(
            collection=last_collection,
            rows_ingested=total_rows,
            vectors_indexed=total_vectors,
            sample=sample,
        )
    
    def _invalidate_caches(self, collection: str) -> None:
        """Invalidate Redis caches after data ingestion.
        
        Ensures users get fresh data after ingestion by clearing:
        1. Aggregation cache - all cached query results for this collection
        2. Metadata cache - schema and collection list
        """
        try:
            from services.redis_aggregation_cache import redis_aggregation_cache
            from services.redis_metadata_cache import redis_metadata_cache
            
            # Invalidate aggregation cache
            deleted_agg = redis_aggregation_cache.invalidate(collection)
            
            # Invalidate metadata cache
            deleted_meta = redis_metadata_cache.invalidate_collection(collection)
            
            if deleted_agg > 0 or deleted_meta > 0:
                logger.info(
                    f"Invalidated caches for {collection}: "
                    f"{deleted_agg} aggregation entries, {deleted_meta} metadata entries"
                )
        except Exception as exc:  # noqa: BLE001
            # Cache invalidation failure should not break ingestion
            logger.warning(f"Cache invalidation failed (non-critical): {exc}")


ingestion_service = IngestionService()
