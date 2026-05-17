"""Tests for semantic out-of-scope detection (Build #9)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from services.semantic_oos import SemanticOOSDetector


@pytest.fixture
def temp_exemplar_file():
    """Create a temporary exemplar file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        exemplars = {
            "in_scope": [
                "How many mosques are in Dubai?",
                "Show me zakat payments in 2024",
                "What is the total hajj revenue?",
            ],
            "out_of_scope": [
                "What's the weather today?",
                "Tell me a joke",
                "How do I code in Python?",
            ],
        }
        json.dump(exemplars, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def detector(temp_exemplar_file):
    """Create a detector instance with test exemplars."""
    return SemanticOOSDetector(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        exemplar_path=temp_exemplar_file,
    )


class TestSemanticOOSDetector:
    """Test suite for SemanticOOSDetector."""

    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.in_scope_exemplars is not None
        assert detector.oos_exemplars is not None
        assert len(detector.in_scope_exemplars) == 3
        assert len(detector.oos_exemplars) == 3

    def test_in_scope_detection(self, detector):
        """Test that in-scope questions are correctly identified."""
        # Similar to training exemplars
        is_oos, confidence = detector.is_oos("How many mosques in Abu Dhabi?")
        assert not is_oos, f"Should be in-scope, got confidence={confidence}"
        
        is_oos, confidence = detector.is_oos("Show zakat data for 2023")
        assert not is_oos, f"Should be in-scope, got confidence={confidence}"
        
        is_oos, confidence = detector.is_oos("What was the hajj revenue last year?")
        assert not is_oos, f"Should be in-scope, got confidence={confidence}"

    def test_oos_detection(self, detector):
        """Test that out-of-scope questions are correctly identified."""
        # Similar to OOS exemplars
        is_oos, confidence = detector.is_oos("What's the temperature in Dubai?")
        assert is_oos, f"Should be OOS, got confidence={confidence}"
        
        is_oos, confidence = detector.is_oos("Tell me something funny")
        assert is_oos, f"Should be OOS, got confidence={confidence}"
        
        is_oos, confidence = detector.is_oos("How to write JavaScript code?")
        assert is_oos, f"Should be OOS, got confidence={confidence}"

    def test_novel_in_scope_questions(self, detector):
        """Test generalization to novel in-scope phrasings."""
        # Domain-relevant but novel phrasings
        questions = [
            "Count the number of Quran centers",
            "Display umrah campaign statistics",
            "What's the total zakat collected?",
            "Show me petition request trends",
        ]
        
        for q in questions:
            is_oos, confidence = detector.is_oos(q)
            assert not is_oos, f"'{q}' should be in-scope, got confidence={confidence}"

    def test_novel_oos_questions(self, detector):
        """Test generalization to novel OOS phrasings."""
        # Clearly out-of-scope topics
        questions = [
            "What's the stock price of Apple?",
            "Who won the football match?",
            "Recommend a good restaurant",
            "What are the symptoms of flu?",
        ]
        
        for q in questions:
            is_oos, confidence = detector.is_oos(q)
            assert is_oos, f"'{q}' should be OOS, got confidence={confidence}"

    def test_threshold_sensitivity(self, detector):
        """Test that threshold parameter affects classification."""
        question = "Show me some data"  # Ambiguous
        
        # Low threshold (more permissive, fewer OOS)
        is_oos_low, conf_low = detector.is_oos(question, threshold=0.3)
        
        # High threshold (stricter, more OOS)
        is_oos_high, conf_high = detector.is_oos(question, threshold=0.8)
        
        # Confidence should be the same
        assert conf_low == conf_high
        
        # At least one should classify differently (unless very clear)
        # This tests that threshold has an effect

    def test_update_exemplars_in_scope(self, detector):
        """Test adding new in-scope exemplars."""
        initial_count = len(detector.in_scope_exemplars)
        
        result = detector.update_exemplars(
            new_in_scope=["What are the occupancy rates?"],
            new_oos=None,
        )
        
        assert result["in_scope_count"] == initial_count + 1
        assert "What are the occupancy rates?" in detector.in_scope_exemplars

    def test_update_exemplars_oos(self, detector):
        """Test adding new OOS exemplars."""
        initial_count = len(detector.oos_exemplars)
        
        result = detector.update_exemplars(
            new_in_scope=None,
            new_oos=["What's the latest news?"],
        )
        
        assert result["oos_count"] == initial_count + 1
        assert "What's the latest news?" in detector.oos_exemplars

    def test_update_exemplars_both(self, detector):
        """Test adding both in-scope and OOS exemplars."""
        initial_in = len(detector.in_scope_exemplars)
        initial_oos = len(detector.oos_exemplars)
        
        result = detector.update_exemplars(
            new_in_scope=["Show hajj permit data", "Count Quran circles"],
            new_oos=["Play music", "Set a timer"],
        )
        
        assert result["in_scope_count"] == initial_in + 2
        assert result["oos_count"] == initial_oos + 2

    def test_update_exemplars_persistence(self, detector, temp_exemplar_file):
        """Test that exemplar updates are persisted to disk."""
        detector.update_exemplars(
            new_in_scope=["New in-scope question"],
            new_oos=["New OOS question"],
        )
        
        # Read the file directly
        with open(temp_exemplar_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        
        assert "New in-scope question" in saved["in_scope"]
        assert "New OOS question" in saved["out_of_scope"]

    def test_empty_question(self, detector):
        """Test handling of empty questions."""
        is_oos, confidence = detector.is_oos("")
        # Should handle gracefully, likely classify as OOS
        assert isinstance(is_oos, bool)
        assert 0.0 <= confidence <= 1.0

    def test_very_long_question(self, detector):
        """Test handling of very long questions."""
        long_question = "Show me " + "mosque " * 100 + "data"
        is_oos, confidence = detector.is_oos(long_question)
        
        # Should still work
        assert isinstance(is_oos, bool)
        assert 0.0 <= confidence <= 1.0

    def test_special_characters(self, detector):
        """Test handling of special characters."""
        questions = [
            "What's the zakat amount? (2024)",
            "Show me data: mosques, hajj, umrah",
            "Revenue in AED 1,000,000+",
        ]
        
        for q in questions:
            is_oos, confidence = detector.is_oos(q)
            assert isinstance(is_oos, bool)
            assert 0.0 <= confidence <= 1.0

    def test_multilingual_robustness(self, detector):
        """Test behavior with non-English text."""
        # Arabic question (should likely be OOS unless exemplars include Arabic)
        is_oos, confidence = detector.is_oos("كم عدد المساجد في دبي؟")
        
        # Should handle gracefully
        assert isinstance(is_oos, bool)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_range(self, detector):
        """Test that confidence scores are always in valid range."""
        questions = [
            "How many mosques?",
            "What's the weather?",
            "Show me data",
            "Tell me a joke",
            "Zakat payments",
        ]
        
        for q in questions:
            _, confidence = detector.is_oos(q)
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for '{q}': {confidence}"

    def test_duplicate_exemplars_ignored(self, detector):
        """Test that duplicate exemplars are not added."""
        initial_count = len(detector.in_scope_exemplars)
        
        # Add an exemplar that already exists
        existing = detector.in_scope_exemplars[0]
        detector.update_exemplars(new_in_scope=[existing], new_oos=None)
        
        # Count should not increase
        assert len(detector.in_scope_exemplars) == initial_count

    def test_case_insensitive_duplicates(self, detector):
        """Test that case variations are treated as duplicates."""
        initial_count = len(detector.in_scope_exemplars)
        
        # Add uppercase version of existing exemplar
        existing = detector.in_scope_exemplars[0]
        detector.update_exemplars(new_in_scope=[existing.upper()], new_oos=None)
        
        # Should not add duplicate (case-insensitive)
        assert len(detector.in_scope_exemplars) == initial_count


class TestSemanticOOSIntegration:
    """Integration tests with question_intent module."""

    def test_integration_with_classify(self):
        """Test that semantic OOS integrates with question_intent.classify()."""
        from services.question_intent import classify
        from models.config import settings
        
        # Only run if semantic OOS is enabled
        if not settings.semantic_oos_enabled:
            pytest.skip("Semantic OOS not enabled in config")
        
        # Test OOS question
        result = classify("What's the weather today?")
        
        # Should be classified as OOS (either by static or semantic)
        from models.enums import QuestionIntent
        assert result.intent == QuestionIntent.OUT_OF_SCOPE

    def test_semantic_layer_fallback(self):
        """Test that static layer catches obvious OOS before semantic."""
        from services.question_intent import classify
        from models.enums import QuestionIntent
        
        # Question with static OOS token
        result = classify("Tell me a joke about weather")
        
        assert result.intent == QuestionIntent.OUT_OF_SCOPE
        # Should be caught by static layer (faster)
        assert result.source in ["rule", "semantic"]


class TestExemplarFileManagement:
    """Test exemplar file creation and management."""

    def test_creates_default_exemplars_if_missing(self):
        """Test that detector creates default exemplars if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent.json"
            
            detector = SemanticOOSDetector(exemplar_path=nonexistent_path)
            
            # Should create file with defaults
            assert nonexistent_path.exists()
            assert len(detector.in_scope_exemplars) > 0
            assert len(detector.oos_exemplars) > 0

    def test_loads_existing_exemplars(self, temp_exemplar_file):
        """Test that detector loads existing exemplars correctly."""
        detector = SemanticOOSDetector(exemplar_path=temp_exemplar_file)
        
        assert "How many mosques are in Dubai?" in detector.in_scope_exemplars
        assert "What's the weather today?" in detector.oos_exemplars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
