"""FAISS-backed vector store with sidecar JSON metadata.

Each vector index position ``i`` corresponds to ``meta[i]``, which links the
vector back to the source MongoDB collection + document id. The metadata is
the source of truth for surfacing the underlying row when answering a query.

Index type: ``IndexFlatIP`` over L2-normalized embeddings → exact cosine.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from bson import ObjectId

import numpy as np

from models.schemas import VectorHit
from services.embedding_service import EmbeddingService, embedding_service
from utils.config import settings
from utils.exceptions import VectorStoreError
from utils.logger import logger


def _json_default(o: Any) -> Any:
    """BSON / datetime values embedded in FAISS metadata (e.g. Mongo ``_id``)."""
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class VectorService:
    """Persistent FAISS index with row-level metadata."""

    def __init__(
        self,
        index_path: str | None = None,
        meta_path: str | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.meta_path = Path(meta_path or settings.faiss_meta_path)
        self.embedder = embedder or embedding_service
        self._index = None  # Loaded lazily on first use.
        self._metadata: list[dict[str, Any]] = []
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        with self._lock:
            if self._index is not None:
                return
            try:
                import faiss  # noqa: F401  (imported here to keep cold imports cheap)
            except ImportError as exc:  # pragma: no cover
                raise VectorStoreError(f"faiss is not installed: {exc}") from exc

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            if self.index_path.exists() and self.meta_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    self._metadata = json.load(fh)
                logger.info(
                    f"Loaded FAISS index ({self._index.ntotal} vectors) "
                    f"from {self.index_path}"
                )
            else:
                self._index = faiss.IndexFlatIP(self.embedder.embedding_dim)
                self._metadata = []
                logger.info(
                    f"Created new FAISS index (dim={self.embedder.embedding_dim})"
                )

    def _persist(self) -> None:
        import faiss

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a per-process tmp filename so two concurrent ingest processes
        # cannot clobber each other's atomic rename. This made a previous
        # ``--replace`` run race with an in-flight ``--drop-all``.
        pid = os.getpid()
        tmp_index = self.index_path.with_suffix(
            self.index_path.suffix + f".tmp.{pid}"
        )
        tmp_meta = self.meta_path.with_suffix(
            self.meta_path.suffix + f".tmp.{pid}"
        )

        faiss.write_index(self._index, str(tmp_index))
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(
                self._metadata,
                fh,
                ensure_ascii=False,
                indent=0,
                default=_json_default,
            )

        os.replace(tmp_index, self.index_path)
        os.replace(tmp_meta, self.meta_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        self._ensure_loaded()
        return int(self._index.ntotal)

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Embed and add ``texts`` with parallel ``metadatas``. Returns count added."""
        if len(texts) != len(metadatas):
            raise VectorStoreError(
                "texts and metadatas must have the same length",
                details={"texts": len(texts), "metadatas": len(metadatas)},
            )
        if not texts:
            return 0

        self._ensure_loaded()
        vectors = self.embedder.encode(texts)
        if vectors.shape[1] != self.embedder.embedding_dim:
            raise VectorStoreError(
                f"Embedding dim mismatch: got {vectors.shape[1]}, "
                f"expected {self.embedder.embedding_dim}"
            )

        with self._lock:
            self._index.add(vectors.astype("float32"))
            for text, meta in zip(texts, metadatas):
                self._metadata.append({"text": text, **meta})
            self._persist()
        logger.info(f"Indexed {len(texts)} vectors (total={self._index.ntotal})")
        return len(texts)

    def search(self, query: str, top_k: int | None = None) -> list[VectorHit]:
        """Return up to ``top_k`` nearest hits for the query.

        Two-stage retrieval when ``settings.reranker_enabled`` is True:

        1. **Recall** — fetch ``reranker_fan_out × top_k`` from FAISS
           (cheap, broad).
        2. **Precision** — cross-encoder rerank to ``top_k``
           (slower but much more accurate).

        When the reranker is off (the default), behaviour is unchanged:
        single-stage bi-encoder cosine, top-K straight from FAISS.
        """
        self._ensure_loaded()
        if self._index.ntotal == 0:
            return []

        final_k = min(top_k or settings.vector_top_k, self._index.ntotal)

        # Stage 1: choose how many hits to pull from FAISS. With reranking
        # we pull a larger candidate pool so the cross-encoder has room to
        # promote a near-miss above an earlier false-positive.
        if settings.reranker_enabled:
            fetch_k = min(final_k * settings.reranker_fan_out, self._index.ntotal)
        else:
            fetch_k = final_k

        vector = self.embedder.encode_one(query).reshape(1, -1).astype("float32")
        scores, indices = self._index.search(vector, fetch_k)

        hits: list[VectorHit] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            hits.append(
                VectorHit(
                    score=float(score),
                    text=meta.get("text", ""),
                    collection=meta.get("collection", ""),
                    document_id=str(meta.get("document_id", "")),
                    payload=meta.get("payload", {}),
                )
            )

        # Stage 2: optional rerank. Imported lazily so the heavy
        # ``sentence-transformers`` cross-encoder model isn't loaded
        # on processes that never opt in.
        if settings.reranker_enabled and hits:
            from services.reranker import reranker
            return reranker.rerank(query, hits, top_k=final_k)

        return hits[:final_k]

    def reset(self) -> None:
        """Drop the index and metadata. Useful for re-ingestion or tests."""
        import faiss

        with self._lock:
            self._index = faiss.IndexFlatIP(self.embedder.embedding_dim)
            self._metadata = []
            self._persist()
        logger.warning("FAISS index reset")


vector_service = VectorService()
