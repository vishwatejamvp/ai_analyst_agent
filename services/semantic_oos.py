"""Semantic out-of-scope detection using sentence embeddings.

Real-time OOS detection that adapts to new patterns without retraining.
Uses cosine similarity between question embeddings and domain exemplars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from models.config import settings
from utils.logger import logger


class SemanticOOSDetector:
    """Real-time semantic OOS detection using sentence embeddings.
    
    Architecture:
    - Uses sentence-transformers for embedding generation
    - Compares question against in-scope and OOS exemplars
    - Supports dynamic exemplar updates without retraining
    - Lazy-loads model to avoid import-time overhead
    
    Exemplars are stored in data/oos_exemplars.json and can be updated
    at runtime via the admin API.
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        exemplars_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name or settings.semantic_oos_model
        self.exemplars_path = Path(exemplars_path or "data/oos_exemplars.json")
        self._model = None
        self._in_scope_exemplars: list[str] = []
        self._oos_exemplars: list[str] = []
        self._in_scope_embeddings: np.ndarray | None = None
        self._oos_embeddings: np.ndarray | None = None
        self._load_attempted = False
    
    def _get_model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None
        self._load_attempted = True
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded semantic OOS model: {self.model_name}")
            
            # Load exemplars
            self._load_exemplars()
            
            return self._model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to load semantic OOS model ({type(exc).__name__}: {exc}); "
                f"semantic OOS detection disabled."
            )
            return None
    
    def _load_exemplars(self) -> None:
        """Load exemplars from JSON file or use defaults."""
        if self.exemplars_path.exists():
            try:
                with self.exemplars_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._in_scope_exemplars = data.get("in_scope", [])
                self._oos_exemplars = data.get("oos", [])
                logger.info(
                    f"Loaded {len(self._in_scope_exemplars)} in-scope and "
                    f"{len(self._oos_exemplars)} OOS exemplars from {self.exemplars_path}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Failed to load exemplars from {self.exemplars_path} "
                    f"({type(exc).__name__}: {exc}); using defaults."
                )
                self._use_default_exemplars()
        else:
            logger.info(
                f"Exemplars file not found at {self.exemplars_path}; using defaults."
            )
            self._use_default_exemplars()
        
        # Pre-compute embeddings
        if self._model is not None:
            self._compute_embeddings()
    
    def _use_default_exemplars(self) -> None:
        """Set default exemplars for AWQAF domain."""
        self._in_scope_exemplars = [
            "show me hajj transactions by month",
            "zakat disbursement trends in 2024",
            "compare mosque registrations across emirates",
            "total revenues for umrah campaigns",
            "how many quran centers are there",
            "petition requests by category",
            "occupancy rates for hajj packages",
            "zakat payment statistics",
            "hajj permit service data",
            "umrah worker permits",
        ]
        
        self._oos_exemplars = [
            "who is the president of UAE",
            "write python code to calculate average",
            "tell me a funny joke",
            "what's the weather in Dubai today",
            "how do I cook biryani",
            "latest football scores",
            "stock price of Apple",
            "movie recommendations",
            "best restaurants in Abu Dhabi",
            "how to fix my computer",
            "what is React Native",
            "cryptocurrency prices",
        ]
    
    def _compute_embeddings(self) -> None:
        """Pre-compute embeddings for all exemplars."""
        if not self._model:
            return
        
        try:
            if self._in_scope_exemplars:
                self._in_scope_embeddings = self._model.encode(
                    self._in_scope_exemplars,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            
            if self._oos_exemplars:
                self._oos_embeddings = self._model.encode(
                    self._oos_exemplars,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            
            logger.debug(
                f"Computed embeddings: in_scope={self._in_scope_embeddings.shape if self._in_scope_embeddings is not None else None}, "
                f"oos={self._oos_embeddings.shape if self._oos_embeddings is not None else None}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to compute exemplar embeddings "
                f"({type(exc).__name__}: {exc})"
            )
    
    def is_oos(
        self,
        question: str,
        threshold: float | None = None,
    ) -> tuple[bool, float]:
        """Check if question is out-of-scope using semantic similarity.
        
        Args:
            question: User question to classify
            threshold: Minimum similarity to OOS exemplars (default from settings)
        
        Returns:
            (is_oos, confidence) where confidence is the similarity score
        
        Algorithm:
            1. Encode question
            2. Compute cosine similarity to in-scope and OOS exemplars
            3. If max(OOS similarity) > max(in-scope similarity) AND
               max(OOS similarity) > threshold → OOS
        """
        model = self._get_model()
        if model is None:
            # Model unavailable → fall back to rule-based
            return False, 0.0
        
        if not self._in_scope_embeddings or not self._oos_embeddings:
            logger.warning("No exemplar embeddings available; skipping semantic OOS check.")
            return False, 0.0
        
        threshold = threshold if threshold is not None else settings.semantic_oos_threshold
        
        try:
            # Encode question
            q_embedding = model.encode(
                question,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            
            # Cosine similarity to in-scope exemplars
            in_scope_sims = self._cosine_similarity(q_embedding, self._in_scope_embeddings)
            max_in_scope = float(np.max(in_scope_sims))
            
            # Cosine similarity to OOS exemplars
            oos_sims = self._cosine_similarity(q_embedding, self._oos_embeddings)
            max_oos = float(np.max(oos_sims))
            
            # Decision: OOS if closer to OOS exemplars AND above threshold
            is_oos = max_oos > max_in_scope and max_oos > threshold
            confidence = max_oos if is_oos else (1.0 - max_oos)
            
            logger.debug(
                f"Semantic OOS check: question='{question[:50]}...' "
                f"in_scope_sim={max_in_scope:.3f} oos_sim={max_oos:.3f} "
                f"is_oos={is_oos} confidence={confidence:.3f}"
            )
            
            return is_oos, confidence
        
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Semantic OOS check failed ({type(exc).__name__}: {exc}); "
                f"falling back to rule-based."
            )
            return False, 0.0
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vector a and matrix b."""
        # Normalize
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Dot product
        return np.dot(b_norm, a_norm)
    
    def update_exemplars(
        self,
        new_in_scope: list[str] | None = None,
        new_oos: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add new exemplars and re-compute embeddings.
        
        Real-time update: No retraining needed, just re-encode.
        Takes ~100ms for 20 exemplars.
        
        Args:
            new_in_scope: New in-scope exemplars to add
            new_oos: New OOS exemplars to add
        
        Returns:
            Status dict with counts
        """
        model = self._get_model()
        if model is None:
            return {
                "status": "error",
                "message": "Model not available",
            }
        
        added_in_scope = 0
        added_oos = 0
        
        if new_in_scope:
            self._in_scope_exemplars.extend(new_in_scope)
            added_in_scope = len(new_in_scope)
        
        if new_oos:
            self._oos_exemplars.extend(new_oos)
            added_oos = len(new_oos)
        
        # Re-compute embeddings
        self._compute_embeddings()
        
        # Save to file
        self._save_exemplars()
        
        return {
            "status": "success",
            "added_in_scope": added_in_scope,
            "added_oos": added_oos,
            "total_in_scope": len(self._in_scope_exemplars),
            "total_oos": len(self._oos_exemplars),
        }
    
    def _save_exemplars(self) -> None:
        """Save current exemplars to JSON file."""
        try:
            self.exemplars_path.parent.mkdir(parents=True, exist_ok=True)
            with self.exemplars_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "in_scope": self._in_scope_exemplars,
                        "oos": self._oos_exemplars,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(f"Saved exemplars to {self.exemplars_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to save exemplars to {self.exemplars_path} "
                f"({type(exc).__name__}: {exc})"
            )
    
    def get_exemplars(self) -> dict[str, list[str]]:
        """Return current exemplars for inspection."""
        return {
            "in_scope": list(self._in_scope_exemplars),
            "oos": list(self._oos_exemplars),
        }


# Module-level singleton
semantic_oos_detector = SemanticOOSDetector()
