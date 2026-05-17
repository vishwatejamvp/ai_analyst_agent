"""Cross-encoder reranker — the precision stage of the RAG pipeline.

Why this module exists
----------------------
``vector_service`` does **bi-encoder** retrieval: query and documents
are embedded *separately* and compared by cosine. Fast, but coarse —
the model never sees the query and the document together, so subtle
relevance signals (e.g. "this document is about Hajj 2024 specifically,
not Hajj in general") are lost.

A **cross-encoder** scores ``(query, doc)`` pairs *jointly*. The model
attends across both texts and outputs a single relevance score. Roughly
5–10× more accurate than bi-encoder cosine on out-of-domain queries
but ~10–50× more expensive *per pair*. Reranking only the top-N from
recall means total cost stays small while quality jumps.

Standard pipeline
-----------------
::

    query
      │
      ▼
    ┌──────────────┐  bi-encoder
    │ vector store │──────────────► top-N (e.g. 20)   ← RECALL
    └──────────────┘  fast, broad
                                 │
                                 ▼
                          ┌──────────────┐  cross-encoder
                          │   Reranker   │──────────────► top-K (e.g. 5)   ← PRECISION
                          └──────────────┘  slower, accurate

Cost discipline
---------------
The cross-encoder runs locally (no API tokens), so the cost is
**latency**, not dollars. Build #2's trace context records each call
under a ``vector.rerank`` span so you can A/B latency in evals
exactly the way you A/B'd cost in build #3.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING

from models.config import settings
from utils.exceptions import VectorStoreError
from utils.logger import logger
from utils.observability import current_trace

if TYPE_CHECKING:  # pragma: no cover — import-cycle avoidance
    from models.schemas import VectorHit


# Module-level lock so two concurrent first-time loaders don't both
# download / instantiate the model.
_LOAD_LOCK = Lock()


class Reranker:
    """Lazy-loaded cross-encoder reranker over :class:`VectorHit` lists."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        with _LOAD_LOCK:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise VectorStoreError(
                    f"sentence-transformers not installed: {exc}"
                ) from exc

            # Match embedding_service's offline-first behaviour: try the
            # local HuggingFace cache first so we never reach the network
            # on machines where it's restricted. Cross-encoders are small
            # (~22 MB for MiniLM) so the one-shot download path is fine.
            #
            # The cache-miss catch is intentionally broad because
            # ``transformers`` does not expose a single canonical
            # exception for "not cached" — depending on the version it
            # raises OSError, ValueError, or AttributeError on a NoneType
            # checkpoint reference. Any error here is safe to treat as a
            # miss because the next branch tries the network path and
            # surfaces a clean ``VectorStoreError`` if THAT fails.
            t0 = time.perf_counter()
            try:
                logger.info(
                    f"Loading reranker '{self.model_name}' (offline cache first)"
                )
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as cache_miss:  # noqa: BLE001  pylint: disable=broad-except
                logger.warning(
                    f"Reranker not in local cache "
                    f"({type(cache_miss).__name__}); attempting one-shot "
                    "download. Subsequent runs will use the cache."
                )
                try:
                    self._model = CrossEncoder(self.model_name)
                except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
                    raise VectorStoreError(
                        f"Failed to load reranker '{self.model_name}' "
                        f"({type(exc).__name__}: {exc})."
                    ) from exc
            logger.info(
                f"Reranker '{self.model_name}' ready "
                f"in {(time.perf_counter() - t0) * 1000:.0f} ms"
            )
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        hits: list["VectorHit"],
        top_k: int,
    ) -> list["VectorHit"]:
        """Re-score ``hits`` with the cross-encoder; return top-K.

        Each returned :class:`VectorHit` has its ``score`` replaced by
        the cross-encoder score and its ``rerank_score`` set so eval
        assertions can verify reranking actually ran. The original
        ``hits`` list is **not** mutated — we use ``model_copy``.

        Behaviour:

        * Empty ``hits`` → returned unchanged (no model load triggered).
        * ``top_k`` larger than ``len(hits)`` → all hits returned, sorted.
        * Model load / inference failure → log + return the original
          (un-reranked) hits truncated to ``top_k``. Reranking should
          never crash the answer pipeline.
        """
        if not hits:
            return hits

        trace = current_trace()
        if trace is not None:
            with trace.span(
                "vector.rerank",
                model=self.model_name,
                n_in=len(hits),
                top_k=top_k,
            ) as span:
                out = self._do_rerank(query, hits, top_k)
                span["n_out"] = len(out)
                if out:
                    span["top_score"] = round(out[0].score, 4)
                return out
        return self._do_rerank(query, hits, top_k)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _do_rerank(
        self,
        query: str,
        hits: list["VectorHit"],
        top_k: int,
    ) -> list["VectorHit"]:
        try:
            model = self._load()
        except VectorStoreError as exc:
            logger.warning(
                f"Reranker unavailable; passing through bi-encoder hits: {exc}"
            )
            return list(hits[:top_k])

        pairs = [(query, h.text or "") for h in hits]
        try:
            scores = model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            logger.warning(
                f"Reranker prediction failed; passing through bi-encoder "
                f"hits ({type(exc).__name__}: {exc})"
            )
            return list(hits[:top_k])

        rescored = [
            h.model_copy(update={
                "score": float(s),
                "rerank_score": float(s),
            })
            for h, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


reranker = Reranker()
