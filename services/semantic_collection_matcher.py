"""Semantic collection matching using embeddings.

Matches natural language questions to database collections using semantic
similarity instead of simple token overlap, improving accuracy significantly.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from services.embedding_service import embedding_service
from services.mongo_service import mongo_service
from utils.logger import logger


class SemanticCollectionMatcher:
    """Match questions to collections using semantic similarity"""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.mongo = mongo_service
        self._collection_embeddings: Dict[str, np.ndarray] = {}
        self._collection_metadata: Dict[str, Dict] = {}
    
    def find_best_collection(
        self, 
        question: str, 
        candidates: List[str]
    ) -> Tuple[Optional[str], float, str]:
        """
        Find best matching collection using semantic similarity.
        
        Returns:
            (collection_name, confidence_score, reasoning)
        """
        if not candidates:
            return None, 0.0, "No collections available"
        
        if len(candidates) == 1:
            return candidates[0], 0.8, f"Only collection available: {candidates[0]}"
        
        # Ensure embeddings are computed
        self._ensure_embeddings(candidates)
        
        # Embed question
        q_embedding = self.embedding_service.embed(question)
        
        # Score each candidate
        scores = []
        for collection in candidates:
            score = self._score_collection(question, q_embedding, collection)
            scores.append((collection, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        best_collection, best_score = scores[0]
        
        # Generate reasoning
        reasoning = self._explain_choice(question, best_collection, scores)
        
        logger.info(f"Semantic match: {best_collection} (score: {best_score:.3f})")
        return best_collection, best_score, reasoning
    
    def _ensure_embeddings(self, collections: List[str]):
        """Compute embeddings for collections that don't have them"""
        for collection in collections:
            if collection not in self._collection_embeddings:
                # Get collection metadata
                metadata = self._get_collection_metadata(collection)
                
                # Create semantic description
                description = self._create_description(collection, metadata)
                
                # Compute embedding
                embedding = self.embedding_service.embed(description)
                
                self._collection_embeddings[collection] = embedding
                self._collection_metadata[collection] = metadata
    
    def _get_collection_metadata(self, collection: str) -> Dict[str, Any]:
        """Get metadata about collection"""
        try:
            # Sample documents
            sample = self.mongo.find(collection, limit=20)
            if not sample:
                return {'fields': [], 'sample_values': {}}
            
            # Extract field names and sample values
            fields = list(sample[0].keys())
            sample_values = {}
            
            for field in fields:
                if field == '_id':
                    continue
                # Get unique values for categorical fields
                values = []
                for doc in sample[:10]:
                    val = doc.get(field, '')
                    if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                        values.append(str(val))
                
                unique_values = list(set(values))
                sample_values[field] = unique_values[:5]  # Top 5 values
            
            return {
                'fields': fields,
                'sample_values': sample_values,
                'row_count': len(sample)
            }
        except Exception as e:
            logger.warning(f"Could not get metadata for {collection}: {e}")
            return {'fields': [], 'sample_values': {}}
    
    def _create_description(self, collection: str, metadata: Dict) -> str:
        """Create semantic description of collection"""
        # Clean collection name
        name = collection.replace('awqaf_', '').replace('_facts', '').replace('_', ' ')
        
        # Add field names
        fields = [f for f in metadata.get('fields', []) if f != '_id']
        fields_text = ' '.join(fields[:10])  # First 10 fields
        
        # Add sample values for context
        sample_text = []
        for field, values in list(metadata.get('sample_values', {}).items())[:5]:
            if values:
                sample_text.append(f"{field}: {', '.join(str(v) for v in values[:3])}")
        
        description = (
            f"{name} dataset with fields: {fields_text}. "
            f"Sample data: {' | '.join(sample_text)}"
        )
        return description
    
    def _score_collection(
        self, 
        question: str, 
        q_embedding: np.ndarray, 
        collection: str
    ) -> float:
        """Score collection match using multiple signals"""
        c_embedding = self._collection_embeddings.get(collection)
        if c_embedding is None:
            return 0.0
        
        # Semantic similarity (primary signal)
        semantic_score = float(
            np.dot(q_embedding, c_embedding) / 
            (np.linalg.norm(q_embedding) * np.linalg.norm(c_embedding))
        )
        
        # Token overlap (secondary signal)
        q_tokens = set(question.lower().split())
        c_tokens = set(collection.lower().replace('_', ' ').split())
        token_overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        
        # Metadata relevance (tertiary signal)
        metadata = self._collection_metadata.get(collection, {})
        metadata_score = self._score_metadata_relevance(question, metadata)
        
        # Weighted combination
        final_score = (
            0.6 * semantic_score +
            0.25 * token_overlap +
            0.15 * metadata_score
        )
        
        return final_score
    
    def _score_metadata_relevance(self, question: str, metadata: Dict) -> float:
        """Score how relevant collection metadata is to question"""
        q_lower = question.lower()
        
        # Check if question mentions any field names
        fields = metadata.get('fields', [])
        field_matches = sum(1 for f in fields if f.lower() in q_lower)
        
        # Check if question mentions any sample values
        sample_values = metadata.get('sample_values', {})
        value_matches = 0
        for values in sample_values.values():
            value_matches += sum(1 for v in values if str(v).lower() in q_lower)
        
        # Normalize
        total_matches = field_matches + value_matches
        max_possible = len(fields) + sum(len(v) for v in sample_values.values())
        
        return total_matches / max(max_possible, 1) if max_possible > 0 else 0.0
    
    def _explain_choice(
        self, 
        question: str, 
        chosen: str, 
        all_scores: List[Tuple[str, float]]
    ) -> str:
        """Generate human-readable explanation of choice"""
        chosen_score = next(s for c, s in all_scores if c == chosen)
        
        if chosen_score > 0.8:
            confidence = "high"
        elif chosen_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Show alternatives
        alternatives = [c for c, s in all_scores[1:3] if s > 0.3]
        
        reasoning = (
            f"Selected '{chosen}' with {confidence} confidence "
            f"(semantic score: {chosen_score:.2f})"
        )
        if alternatives:
            reasoning += f". Alternatives: {', '.join(alternatives)}"
        
        return reasoning
    
    def invalidate_cache(self, collection: str = None):
        """Invalidate embedding cache (useful after schema changes)"""
        if collection:
            self._collection_embeddings.pop(collection, None)
            self._collection_metadata.pop(collection, None)
            logger.info(f"Semantic cache invalidated for {collection}")
        else:
            self._collection_embeddings.clear()
            self._collection_metadata.clear()
            logger.info("Semantic cache cleared")


# Singleton instance
semantic_matcher = SemanticCollectionMatcher()
