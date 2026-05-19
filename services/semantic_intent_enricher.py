"""Semantic intent enrichment for vague analytical queries.

Uses embeddings to detect analytical intent even when no explicit aggregation
keywords are present. This is Phase 1 of the production roadmap: moving from
code-driven (regex/keywords) to data-driven (embeddings + config) architecture.

Key Principles:
- Business logic as DATA (config/operation_patterns.json), not CODE
- Confidence scoring on all decisions
- Embedding-based semantic matching (not regex)
- Zero new technical debt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from services.embedding_service import embedding_service
from services.mongo_service import mongo_service
from utils.logger import logger


@dataclass(frozen=True)
class EnrichedIntent:
    """Result of semantic intent enrichment with confidence scoring."""
    
    original_question: str
    inferred_operation: str | None  # "sum", "count", "avg", etc.
    confidence: float  # 0.0-1.0
    reasoning: str
    suggested_questions: list[str]  # If confidence < threshold
    matched_pattern: str | None  # Which pattern matched
    
    @property
    def is_confident(self) -> bool:
        """True if confidence exceeds minimum acceptable threshold."""
        return self.confidence >= 0.5


@dataclass
class OperationPattern:
    """Data-driven operation pattern from config."""
    
    operation: str
    description: str
    semantic_patterns: list[str]
    default_metric_hints: list[str]
    confidence_boost: float
    priority: int
    
    # Cached embeddings for semantic patterns
    _pattern_embeddings: list[np.ndarray] | None = None


class SemanticIntentEnricher:
    """Enrich vague queries with inferred analytical operations using embeddings."""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.mongo = mongo_service
        self._config: dict[str, Any] | None = None
        self._operation_patterns: list[OperationPattern] = []
        self._pattern_embeddings: dict[str, list[np.ndarray]] = {}
        
    def enrich(
        self, 
        question: str, 
        target: str | None = None
    ) -> EnrichedIntent:
        """
        Detect analytical intent using semantic similarity.
        
        Args:
            question: User's natural language question
            target: Target collection/dataset (if known)
            
        Returns:
            EnrichedIntent with inferred operation and confidence score
        """
        if not question or not question.strip():
            return EnrichedIntent(
                original_question=question,
                inferred_operation=None,
                confidence=0.0,
                reasoning="Empty question",
                suggested_questions=[],
                matched_pattern=None
            )
        
        # Load config and patterns
        self._ensure_config_loaded()
        
        # Embed the question
        try:
            q_embedding = self.embedding_service.embed(question)
        except Exception as e:
            logger.warning(f"Failed to embed question for intent enrichment: {e}")
            return self._fallback_enrichment(question, target)
        
        # Score against all operation patterns
        best_operation = None
        best_confidence = 0.0
        best_pattern = None
        best_reasoning = ""
        
        for pattern in self._operation_patterns:
            score, matched_pattern = self._score_pattern(
                question, q_embedding, pattern, target
            )
            
            if score > best_confidence:
                best_confidence = score
                best_operation = pattern.operation
                best_pattern = matched_pattern
                best_reasoning = (
                    f"Semantic match to '{pattern.operation}' pattern "
                    f"(confidence: {score:.2f})"
                )
        
        # Generate suggestions if confidence is low
        suggested_questions = []
        if best_confidence < self._get_threshold("min_acceptable"):
            suggested_questions = self._generate_suggestions(question, target)
        
        return EnrichedIntent(
            original_question=question,
            inferred_operation=best_operation,
            confidence=best_confidence,
            reasoning=best_reasoning,
            suggested_questions=suggested_questions,
            matched_pattern=best_pattern
        )
    
    def _score_pattern(
        self,
        question: str,
        q_embedding: np.ndarray,
        pattern: OperationPattern,
        target: str | None
    ) -> tuple[float, str | None]:
        """
        Score how well a question matches an operation pattern.
        
        Returns:
            (confidence_score, matched_pattern_text)
        """
        # Ensure pattern embeddings are computed
        if pattern.operation not in self._pattern_embeddings:
            self._compute_pattern_embeddings(pattern)
        
        pattern_embeds = self._pattern_embeddings.get(pattern.operation, [])
        if not pattern_embeds:
            return 0.0, None
        
        # Compute semantic similarity to each pattern
        max_similarity = 0.0
        best_pattern_text = None
        
        for i, p_embedding in enumerate(pattern_embeds):
            similarity = float(
                np.dot(q_embedding, p_embedding) / 
                (np.linalg.norm(q_embedding) * np.linalg.norm(p_embedding))
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                if i < len(pattern.semantic_patterns):
                    best_pattern_text = pattern.semantic_patterns[i]
        
        # Apply confidence boost from config
        confidence = max_similarity + pattern.confidence_boost
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        # Boost if target dataset is mentioned in question
        if target and self._target_mentioned_in_question(question, target):
            confidence += 0.1
            confidence = min(confidence, 1.0)
        
        return confidence, best_pattern_text
    
    def _compute_pattern_embeddings(self, pattern: OperationPattern):
        """Compute and cache embeddings for pattern's semantic patterns."""
        embeddings = []
        
        for semantic_pattern in pattern.semantic_patterns:
            try:
                # Replace {dataset} placeholder with generic term for embedding
                normalized = semantic_pattern.replace("{dataset}", "the dataset")
                embedding = self.embedding_service.embed(normalized)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(
                    f"Failed to embed pattern '{semantic_pattern}': {e}"
                )
        
        self._pattern_embeddings[pattern.operation] = embeddings
    
    def _target_mentioned_in_question(self, question: str, target: str) -> bool:
        """Check if target dataset is mentioned in question."""
        q_lower = question.lower()
        
        # Clean target name
        target_clean = target.replace("awqaf_", "").replace("_facts", "")
        target_tokens = target_clean.replace("_", " ").split()
        
        # Check if any significant token from target appears in question
        for token in target_tokens:
            if len(token) > 3 and token in q_lower:
                return True
        
        return False
    
    def _generate_suggestions(
        self, 
        question: str, 
        target: str | None
    ) -> list[str]:
        """Generate helpful specific questions when confidence is low."""
        if not target:
            return [
                "Try asking: 'What datasets are available?'",
                "Or be more specific: 'total transactions for hajj package service in 2025'"
            ]
        
        suggestions = []
        templates = self._config.get("vague_query_templates", {})
        
        # Get dataset metadata
        dataset_info = self._get_dataset_info(target)
        
        # Detect dataset type
        dataset_type = self._detect_dataset_type(target, dataset_info)
        
        # Generate suggestions based on dataset type
        if dataset_type == "time_series":
            template_info = templates.get("time_series", {})
            template = template_info.get("template", "")
            
            if template and dataset_info:
                metric = dataset_info.get("default_metric", "total_transactions")
                dataset_name = dataset_info.get("name", target)
                year = dataset_info.get("latest_year", 2025)
                
                suggestion = template.format(
                    metric=metric,
                    dataset=dataset_name,
                    year=year
                )
                suggestions.append(f"Try: '{suggestion}'")
        
        elif dataset_type == "directory":
            template_info = templates.get("directory", {})
            template = template_info.get("template", "")
            
            if template and dataset_info:
                dataset_name = dataset_info.get("name", target)
                year = dataset_info.get("latest_year", 2023)
                
                suggestion = template.format(
                    dataset=dataset_name,
                    year=year
                )
                suggestions.append(f"Try: '{suggestion}'")
        
        # Add trend suggestion
        if dataset_type == "time_series" and dataset_info:
            template_info = templates.get("trend", {})
            template = template_info.get("template", "")
            
            if template:
                metric = dataset_info.get("default_metric", "total_transactions")
                dataset_name = dataset_info.get("name", target)
                year = dataset_info.get("latest_year", 2025)
                
                suggestion = template.format(
                    metric=metric,
                    dataset=dataset_name,
                    year=year
                )
                suggestions.append(f"Or: '{suggestion}'")
        
        # Fallback suggestions
        if not suggestions:
            suggestions = [
                f"Try being more specific about what you want to analyze",
                f"Example: 'total transactions for {target} in 2025'",
                f"Or: 'monthly trend for {target} in 2024'"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _get_dataset_info(self, target: str) -> dict[str, Any]:
        """Get metadata about target dataset from catalog."""
        try:
            # Query awqaf_datasets_metadata
            slug = target.replace("awqaf_", "").replace("_facts", "").replace("_", "-")
            
            result = self.mongo.find(
                "awqaf_datasets_metadata",
                filter={"dataset": slug},
                limit=1
            )
            
            if result:
                metadata = result[0]
                
                # Extract useful info
                years = metadata.get("years", [])
                latest_year = max(years) if years else 2025
                
                key_metrics = metadata.get("key_metrics", [])
                default_metric = "total_transactions"
                if key_metrics:
                    # Prefer metrics with "total" in name
                    total_metrics = [m for m in key_metrics if "total" in str(m).lower()]
                    default_metric = total_metrics[0] if total_metrics else key_metrics[0]
                
                return {
                    "name": metadata.get("dataset_name", slug),
                    "slug": slug,
                    "latest_year": latest_year,
                    "years": years,
                    "default_metric": default_metric,
                    "key_metrics": key_metrics,
                    "data_type": metadata.get("data_type", "time_series")
                }
        except Exception as e:
            logger.debug(f"Could not fetch dataset info for {target}: {e}")
        
        return {}
    
    def _detect_dataset_type(
        self, 
        target: str, 
        dataset_info: dict[str, Any]
    ) -> str:
        """Detect if dataset is time_series, directory, or service."""
        # Check metadata first
        if dataset_info:
            data_type = dataset_info.get("data_type", "").lower()
            if data_type in ("time_series", "directory", "service"):
                return data_type
        
        # Fallback to name-based detection
        target_lower = target.lower()
        
        indicators = self._config.get("dataset_type_detection", {})
        
        # Check directory indicators
        dir_indicators = indicators.get("directory_indicators", [])
        if any(ind in target_lower for ind in dir_indicators):
            return "directory"
        
        # Check service indicators
        service_indicators = indicators.get("service_indicators", [])
        if any(ind in target_lower for ind in service_indicators):
            return "time_series"  # Services are usually time-series
        
        # Default to time_series
        return "time_series"
    
    def _fallback_enrichment(
        self, 
        question: str, 
        target: str | None
    ) -> EnrichedIntent:
        """Fallback when embedding fails - use simple heuristics."""
        q_lower = question.lower()
        
        # Simple keyword detection as fallback
        if any(kw in q_lower for kw in ["total", "sum", "revenue"]):
            operation = "sum"
            confidence = 0.6
        elif any(kw in q_lower for kw in ["how many", "count", "number"]):
            operation = "count"
            confidence = 0.6
        elif any(kw in q_lower for kw in ["average", "mean", "avg"]):
            operation = "avg"
            confidence = 0.6
        else:
            operation = "sum"  # Default to sum for vague queries
            confidence = 0.4
        
        return EnrichedIntent(
            original_question=question,
            inferred_operation=operation,
            confidence=confidence,
            reasoning=f"Fallback heuristic detected '{operation}' (confidence: {confidence:.2f})",
            suggested_questions=self._generate_suggestions(question, target),
            matched_pattern=None
        )
    
    def _ensure_config_loaded(self):
        """Load operation patterns config if not already loaded."""
        if self._config is not None:
            return
        
        config_path = Path(__file__).parent.parent / "config" / "operation_patterns.json"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            
            # Parse operation patterns
            patterns_data = self._config.get("operation_patterns", [])
            self._operation_patterns = [
                OperationPattern(
                    operation=p["operation"],
                    description=p["description"],
                    semantic_patterns=p["semantic_patterns"],
                    default_metric_hints=p["default_metric_hints"],
                    confidence_boost=p["confidence_boost"],
                    priority=p["priority"]
                )
                for p in patterns_data
            ]
            
            # Sort by priority
            self._operation_patterns.sort(key=lambda p: p.priority)
            
            logger.info(
                f"Loaded {len(self._operation_patterns)} operation patterns "
                f"from {config_path}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load operation patterns config: {e}")
            self._config = {}
            self._operation_patterns = []
    
    def _get_threshold(self, threshold_name: str) -> float:
        """Get confidence threshold from config."""
        if not self._config:
            return 0.5
        
        thresholds = self._config.get("confidence_thresholds", {})
        return thresholds.get(threshold_name, 0.5)
    
    def invalidate_cache(self):
        """Invalidate cached embeddings (useful after config changes)."""
        self._pattern_embeddings.clear()
        logger.info("Semantic intent enricher cache invalidated")


# Singleton instance
semantic_intent_enricher = SemanticIntentEnricher()
