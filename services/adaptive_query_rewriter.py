"""Adaptive query rewriter with self-learning from user feedback.

Handles:
- Typos and misspellings (fuzzy matching)
- Synonyms and semantic similarity
- User corrections (learns from feedback)
- Context-aware disambiguation

Architecture:
- Fuzzy matching using rapidfuzz (Levenshtein distance)
- Semantic matching using sentence embeddings
- User correction history (persistent learning)
- Dynamic vocabulary from actual data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from models.config import settings
from utils.logger import logger


class AdaptiveQueryRewriter:
    """Self-learning query rewriter with user feedback loop."""

    # Transaction verbs that should map to SUM of transaction metrics, not COUNT
    TRANSACTION_VERBS = {
        "signed", "signed up", "registered", "enrolled", "applied",
        "submitted", "completed", "transacted", "booked", "purchased",
        "paid", "donated", "contributed"
    }
    
    # Metric suffixes that indicate transaction/count fields
    TRANSACTION_SUFFIXES = {
        "_transactions", "_registrations", "_applications", "_permits",
        "_requests", "_submissions", "_bookings", "_payments"
    }

    def __init__(
        self,
        corrections_path: str | Path | None = None,
        fuzzy_threshold: float = 0.85,
        semantic_threshold: float = 0.75,
    ) -> None:
        self.corrections_path = Path(
            corrections_path or "data/query_corrections.json"
        )
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold

        # User corrections: original_query → corrected_query
        self.corrections: dict[str, str] = {}

        # Vocabulary from actual data
        self.collections: list[str] = []
        self.metrics: list[str] = []
        self.all_terms: list[str] = []

        # Embeddings cache
        self.embeddings_cache: dict[str, np.ndarray] = {}
        self._model = None
        self._load_attempted = False

        # Load corrections
        self._load_corrections()

        # Load vocabulary
        self._load_vocabulary()

    def _get_model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None
        self._load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded query rewriter model: {settings.embedding_model}")

            # Pre-compute embeddings for vocabulary
            if self.all_terms:
                self._compute_embeddings()

            return self._model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to load query rewriter model ({type(exc).__name__}: {exc}); "
                f"semantic matching disabled, using fuzzy only."
            )
            return None

    def _load_corrections(self) -> None:
        """Load user corrections from JSON file."""
        if self.corrections_path.exists():
            try:
                with self.corrections_path.open("r", encoding="utf-8") as f:
                    self.corrections = json.load(f)
                logger.info(
                    f"Loaded {len(self.corrections)} query corrections "
                    f"from {self.corrections_path}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Failed to load corrections from {self.corrections_path} "
                    f"({type(exc).__name__}: {exc})"
                )
                self.corrections = {}
        else:
            logger.debug(
                f"Corrections file not found at {self.corrections_path}; "
                f"starting fresh."
            )
            self.corrections = {}

    def _load_vocabulary(self) -> None:
        """Build vocabulary from actual MongoDB collections and metrics."""
        try:
            # Import here to avoid circular dependency
            from services.mongo_service import mongo_service

            # Collections
            self.collections = mongo_service.list_collections()

            # Metrics - get from knowledge base or use empty list
            try:
                from services.knowledge_base_service import knowledge_base_service
                
                # Get all metric definitions
                all_metrics = knowledge_base_service.list_metrics()
                self.metrics = [m.name for m in all_metrics]
            except Exception:  # noqa: BLE001
                # Fallback: use common metric names
                self.metrics = [
                    "total", "count", "average", "sum", "revenue",
                    "transactions", "registrations", "permits", "campaigns"
                ]

            # Combine all terms
            self.all_terms = self.collections + self.metrics

            logger.info(
                f"Loaded vocabulary: {len(self.collections)} collections, "
                f"{len(self.metrics)} metrics"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to load vocabulary ({type(exc).__name__}: {exc}); "
                f"query rewriting may be limited."
            )
            self.all_terms = []

    def _compute_embeddings(self) -> None:
        """Pre-compute embeddings for all vocabulary terms."""
        model = self._get_model()
        if model is None or not self.all_terms:
            return

        try:
            embeddings = model.encode(
                self.all_terms,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            self.embeddings_cache = {
                term: embeddings[i] for i, term in enumerate(self.all_terms)
            }

            logger.debug(
                f"Computed embeddings for {len(self.embeddings_cache)} terms"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to compute embeddings ({type(exc).__name__}: {exc})"
            )

    def detect_transaction_intent(self, query: str) -> dict[str, Any]:
        """Detect if query is asking for transaction counts (should use SUM, not COUNT).
        
        Returns:
            {
                "is_transaction_query": bool,
                "suggested_operation": str,  # "sum" or None
                "matched_verb": str | None,
                "suggested_metrics": list[str]  # Metrics that match transaction pattern
            }
        """
        query_lower = query.lower()
        
        # Check for transaction verbs
        matched_verb = None
        for verb in self.TRANSACTION_VERBS:
            if verb in query_lower:
                matched_verb = verb
                break
        
        if not matched_verb:
            return {
                "is_transaction_query": False,
                "suggested_operation": None,
                "matched_verb": None,
                "suggested_metrics": []
            }
        
        # Find transaction-related metrics in vocabulary
        suggested_metrics = []
        for metric in self.metrics:
            metric_lower = metric.lower()
            # Check if metric ends with transaction suffix
            if any(metric_lower.endswith(suffix) for suffix in self.TRANSACTION_SUFFIXES):
                suggested_metrics.append(metric)
            # Or contains transaction-related words
            elif any(word in metric_lower for word in ["transaction", "registration", "application", "permit"]):
                suggested_metrics.append(metric)
        
        return {
            "is_transaction_query": True,
            "suggested_operation": "sum",
            "matched_verb": matched_verb,
            "suggested_metrics": suggested_metrics
        }

    def rewrite(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rewrite query with fuzzy matching + semantic similarity.

        Args:
            query: Original user query
            context: Optional context (session, prior queries, etc.)

        Returns:
            {
                "original": str,
                "rewritten": str,
                "corrections": list[dict],  # What was fixed
                "confidence": float,
                "strategy": str,  # "learned", "fuzzy", "semantic", "none"
                "transaction_intent": dict | None  # Transaction detection result
            }
        """
        # Detect transaction intent
        transaction_intent = self.detect_transaction_intent(query)
        # Check user correction history first (highest confidence)
        query_lower = query.lower().strip()
        if query_lower in self.corrections:
            return {
                "original": query,
                "rewritten": self.corrections[query_lower],
                "corrections": [
                    {
                        "original": query,
                        "corrected": self.corrections[query_lower],
                        "type": "learned",
                        "confidence": 1.0,
                    }
                ],
                "confidence": 1.0,
                "strategy": "learned",
            }

        # Extract potential entity mentions (simple tokenization)
        corrections = []
        rewritten = query
        tokens = query.split()

        for token in tokens:
            token_clean = token.lower().strip(".,!?;:")

            if len(token_clean) < 3:  # Skip very short tokens
                continue

            # Try fuzzy matching first (faster)
            fuzzy_match, fuzzy_score = self._fuzzy_match(token_clean)

            if fuzzy_score >= self.fuzzy_threshold:
                # High confidence fuzzy match
                rewritten = rewritten.replace(token, fuzzy_match)
                corrections.append(
                    {
                        "original": token,
                        "corrected": fuzzy_match,
                        "type": "fuzzy",
                        "confidence": fuzzy_score,
                    }
                )
                continue

            # Try semantic matching (slower, but handles synonyms)
            semantic_match, semantic_score = self._semantic_match(token_clean)

            if semantic_score >= self.semantic_threshold:
                rewritten = rewritten.replace(token, semantic_match)
                corrections.append(
                    {
                        "original": token,
                        "corrected": semantic_match,
                        "type": "semantic",
                        "confidence": semantic_score,
                    }
                )

        # Calculate overall confidence
        if corrections:
            avg_confidence = np.mean([c["confidence"] for c in corrections])
            strategy = corrections[0]["type"]
        else:
            avg_confidence = 1.0
            strategy = "none"

        return {
            "original": query,
            "rewritten": rewritten,
            "corrections": corrections,
            "confidence": float(avg_confidence),
            "strategy": strategy,
            "transaction_intent": transaction_intent if transaction_intent["is_transaction_query"] else None,
        }

    def _fuzzy_match(self, token: str) -> tuple[str, float]:
        """Find best fuzzy match using rapidfuzz."""
        if not self.all_terms:
            return token, 0.0

        try:
            from rapidfuzz import fuzz

            scores = [
                (term, fuzz.ratio(token, term.lower()) / 100.0)
                for term in self.all_terms
            ]
            best_term, best_score = max(scores, key=lambda x: x[1])

            return best_term, best_score

        except ImportError:
            logger.warning(
                "rapidfuzz not installed; fuzzy matching disabled. "
                "Install with: pip install rapidfuzz"
            )
            return token, 0.0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Fuzzy matching failed: {exc}")
            return token, 0.0

    def _semantic_match(self, token: str) -> tuple[str, float]:
        """Find best semantic match using embeddings."""
        model = self._get_model()
        if model is None or not self.embeddings_cache:
            return token, 0.0

        try:
            # Encode token
            token_emb = model.encode(token, convert_to_numpy=True)

            # Find best match
            best_term = token
            best_score = 0.0

            for term, term_emb in self.embeddings_cache.items():
                score = self._cosine_similarity(token_emb, term_emb)
                if score > best_score:
                    best_score = score
                    best_term = term

            return best_term, float(best_score)

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Semantic matching failed: {exc}")
            return token, 0.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def learn_from_feedback(
        self,
        original: str,
        corrected: str,
    ) -> dict[str, Any]:
        """User explicitly corrects a query → learn for future.

        Args:
            original: Original query that was wrong
            corrected: Corrected query from user

        Returns:
            Status dict
        """
        original_lower = original.lower().strip()

        # Store correction
        self.corrections[original_lower] = corrected

        # Persist to disk
        self._save_corrections()

        logger.info(f"Learned correction: '{original}' → '{corrected}'")

        return {
            "status": "success",
            "original": original,
            "corrected": corrected,
            "total_corrections": len(self.corrections),
        }

    def _save_corrections(self) -> None:
        """Save corrections to JSON file."""
        try:
            self.corrections_path.parent.mkdir(parents=True, exist_ok=True)
            with self.corrections_path.open("w", encoding="utf-8") as f:
                json.dump(
                    self.corrections,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.debug(f"Saved {len(self.corrections)} corrections to {self.corrections_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to save corrections to {self.corrections_path} "
                f"({type(exc).__name__}: {exc})"
            )

    def refresh_vocabulary(self) -> dict[str, Any]:
        """Refresh vocabulary from current data.

        Call this after ingesting new data or adding new metrics.
        """
        old_count = len(self.all_terms)

        self._load_vocabulary()

        # Re-compute embeddings if model is loaded
        if self._model is not None:
            self._compute_embeddings()

        new_count = len(self.all_terms)

        logger.info(
            f"Refreshed vocabulary: {old_count} → {new_count} terms "
            f"({new_count - old_count:+d})"
        )

        return {
            "status": "success",
            "old_count": old_count,
            "new_count": new_count,
            "collections": len(self.collections),
            "metrics": len(self.metrics),
        }

    def get_corrections(self) -> dict[str, str]:
        """Get all learned corrections for inspection."""
        return dict(self.corrections)

    def get_stats(self) -> dict[str, Any]:
        """Get rewriter statistics."""
        return {
            "corrections_count": len(self.corrections),
            "vocabulary_size": len(self.all_terms),
            "collections_count": len(self.collections),
            "metrics_count": len(self.metrics),
            "embeddings_cached": len(self.embeddings_cache),
            "model_loaded": self._model is not None,
            "fuzzy_threshold": self.fuzzy_threshold,
            "semantic_threshold": self.semantic_threshold,
        }


# Module-level singleton
adaptive_query_rewriter = AdaptiveQueryRewriter()
