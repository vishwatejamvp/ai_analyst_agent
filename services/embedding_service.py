"""Local embedding service backed by ``sentence-transformers``.

The model is lazy-loaded on first use so importing this module is cheap.
All vectors are L2-normalized so cosine similarity reduces to inner product,
which lets FAISS use ``IndexFlatIP`` for exact cosine search.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Iterable

import numpy as np

from models.config import settings
from utils.exceptions import VectorStoreError
from utils.logger import logger

# Suppress the "checking for model updates" HEAD requests that try to reach
# huggingface.co on every load. Once the model is cached locally we never
# need network access for inference. Setting this at import time avoids the
# ~160s retry storm seen on offline / firewalled machines.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class EmbeddingService:
    """Wraps a SentenceTransformer model with thread-safe lazy loading."""

    def __init__(
        self,
        model_name: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self._model = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - install issue
                raise VectorStoreError(
                    f"sentence-transformers not installed: {exc}"
                ) from exc

            # Try the local HF cache first so we never reach out to
            # huggingface.co on machines where DNS / network is restricted.
            try:
                logger.info(
                    f"Loading embedding model '{self.model_name}' (offline cache)"
                )
                self._model = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
                return self._model
            except (OSError, ValueError) as cache_miss:
                logger.warning(
                    f"Embedding model not in local cache "
                    f"({type(cache_miss).__name__}); attempting one-shot "
                    "download. Subsequent runs will use the cache."
                )

            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"Failed to load embedding model '{self.model_name}' "
                    f"({type(exc).__name__}: {exc}). If you are offline, "
                    "pre-download the model on a network-connected machine."
                ) from exc
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32, L2-normalized matrix of embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype="float32")

        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience helper returning a single ``(dim,)`` vector."""
        return self.encode([text])[0]

    @staticmethod
    def row_to_text(row: dict, *, source: str | None = None) -> str:
        """Stable ``key: value | ...`` representation (legacy fallback)."""
        parts = [
            f"{key}: {value}"
            for key, value in sorted(row.items())
            if value is not None and value != ""
        ]
        body = " | ".join(parts)
        return f"[{source}] {body}" if source else body

    # ------------------------------------------------------------------
    # Natural-language sentences (preferred for AWQAF facts + glossary)
    # ------------------------------------------------------------------
    _ENVELOPE_KEYS = frozenset(
        {
            "dataset",
            "source_file",
            "ingested_at",
            "year",
            "month",
            "month_num",
            "period",
            "dimension",
            "_id",
        }
    )

    @classmethod
    def fact_to_sentence(
        cls,
        row: dict,
        *,
        dataset_label: str | None = None,
    ) -> str:
        """Render a fact row as a short human-readable sentence.

        Sentence embeddings (with verbs, units, dates, magnitudes) give the
        retriever much better recall on questions like "why September spike"
        than the legacy ``key: value | ...`` form.
        """
        label = dataset_label or row.get("dataset") or "dataset"
        time_part = ""
        period = row.get("period")
        month = row.get("month")
        year = row.get("year")
        if period:
            time_part = f"in {period}"
        elif month and year:
            time_part = f"in {str(month).title()} {year}"
        elif year:
            time_part = f"in {year}"

        metrics: list[str] = []
        for key, value in row.items():
            if key in cls._ENVELOPE_KEYS:
                continue
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)):
                metrics.append(f"{key.replace('_', ' ')} {value}")
            else:
                metrics.append(f"{key.replace('_', ' ')}: {value}")

        body = ", ".join(metrics) if metrics else "(no metrics)"
        prefix = f"{label} {time_part}".strip()
        return f"{prefix} — {body}".strip(" —")

    @staticmethod
    def glossary_term_to_sentence(term: dict, *, dataset_label: str) -> str:
        name = term.get("term") or "term"
        definition = term.get("definition") or ""
        unit = term.get("unit")
        suffix = f" Unit: {unit}." if unit else ""
        return f"In {dataset_label}, '{name}' means: {definition}.{suffix}".strip()

    @staticmethod
    def glossary_field_to_sentence(
        field_key: str, meta: dict, *, dataset_label: str
    ) -> str:
        description = meta.get("description") or ""
        original = meta.get("original_name")
        original_part = f" (original: {original})" if original else ""
        return (
            f"Field '{field_key}'{original_part} in {dataset_label}: "
            f"{description}".strip()
        )


# Module-level singleton (cheap; the model itself is lazy).
embedding_service = EmbeddingService()
