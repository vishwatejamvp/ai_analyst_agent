"""API routes for OOS detection management and feedback collection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from services.semantic_oos import semantic_oos_detector
from utils.logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class OOSFeedbackRequest(BaseModel):
    """User feedback on OOS classification."""
    
    question: str = Field(..., min_length=1, max_length=500)
    predicted_oos: bool = Field(
        ...,
        description="What the system predicted (true = OOS, false = in-scope)",
    )
    user_confirmed_oos: bool = Field(
        ...,
        description="What the user confirmed (true = OOS, false = in-scope)",
    )


class AddExemplarRequest(BaseModel):
    """Request to add new exemplars to semantic OOS detector."""
    
    exemplar: str = Field(..., min_length=1, max_length=500)
    is_oos: bool = Field(
        ...,
        description="True if this is an OOS exemplar, False if in-scope",
    )


class ExemplarBatchRequest(BaseModel):
    """Request to add multiple exemplars at once."""
    
    in_scope: list[str] = Field(default_factory=list)
    oos: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Feedback Collection
# ---------------------------------------------------------------------------
@router.post("/feedback", tags=["oos"])
def submit_oos_feedback(request: OOSFeedbackRequest = Body(...)) -> dict[str, Any]:
    """Submit user feedback on OOS classification.
    
    This endpoint collects feedback to improve OOS detection over time.
    When a user disagrees with the prediction, the question is automatically
    added as an exemplar to the semantic OOS detector.
    
    Example:
        User asks: "who is the president"
        System predicts: OOS (predicted_oos=true)
        User confirms: Yes, it's OOS (user_confirmed_oos=true)
        → No action needed (agreement)
        
        User asks: "hajj transactions"
        System predicts: In-scope (predicted_oos=false)
        User confirms: Yes, in-scope (user_confirmed_oos=false)
        → No action needed (agreement)
        
        User asks: "weather in Dubai"
        System predicts: In-scope (predicted_oos=false) [WRONG]
        User confirms: No, it's OOS (user_confirmed_oos=true)
        → Add "weather in Dubai" to OOS exemplars
    """
    try:
        # Log feedback for analytics
        logger.info(
            f"OOS feedback: question='{request.question}' "
            f"predicted={request.predicted_oos} "
            f"confirmed={request.user_confirmed_oos}"
        )
        
        # If user disagreed with prediction, update exemplars
        if request.predicted_oos != request.user_confirmed_oos:
            if request.user_confirmed_oos:
                # User says it IS OOS (but we predicted in-scope)
                result = semantic_oos_detector.update_exemplars(
                    new_in_scope=None,
                    new_oos=[request.question],
                )
                logger.info(
                    f"Added OOS exemplar from feedback: '{request.question}'"
                )
            else:
                # User says it's NOT OOS (but we predicted OOS)
                result = semantic_oos_detector.update_exemplars(
                    new_in_scope=[request.question],
                    new_oos=None,
                )
                logger.info(
                    f"Added in-scope exemplar from feedback: '{request.question}'"
                )
            
            return {
                "status": "feedback_applied",
                "message": "Exemplars updated based on your feedback",
                "update_result": result,
            }
        
        # Agreement — just log it
        return {
            "status": "feedback_recorded",
            "message": "Thank you for your feedback",
        }
    
    except Exception as exc:
        logger.error(f"Failed to process OOS feedback: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process feedback: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Exemplar Management (Admin)
# ---------------------------------------------------------------------------
@router.post("/exemplars", tags=["oos", "admin"])
def add_exemplar(request: AddExemplarRequest = Body(...)) -> dict[str, Any]:
    """Add a single exemplar to the semantic OOS detector (admin only).
    
    This endpoint allows administrators to manually add exemplars without
    waiting for user feedback. Useful for bootstrapping or fixing
    specific misclassifications.
    """
    try:
        if request.is_oos:
            result = semantic_oos_detector.update_exemplars(
                new_in_scope=None,
                new_oos=[request.exemplar],
            )
        else:
            result = semantic_oos_detector.update_exemplars(
                new_in_scope=[request.exemplar],
                new_oos=None,
            )
        
        return {
            "status": "success",
            "message": f"Added {'OOS' if request.is_oos else 'in-scope'} exemplar",
            "exemplar": request.exemplar,
            "update_result": result,
        }
    
    except Exception as exc:
        logger.error(f"Failed to add exemplar: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add exemplar: {str(exc)}",
        ) from exc


@router.post("/exemplars/batch", tags=["oos", "admin"])
def add_exemplars_batch(request: ExemplarBatchRequest = Body(...)) -> dict[str, Any]:
    """Add multiple exemplars at once (admin only).
    
    Useful for bulk updates or importing exemplars from a file.
    """
    try:
        result = semantic_oos_detector.update_exemplars(
            new_in_scope=request.in_scope if request.in_scope else None,
            new_oos=request.oos if request.oos else None,
        )
        
        return {
            "status": "success",
            "message": "Batch exemplars added",
            "update_result": result,
        }
    
    except Exception as exc:
        logger.error(f"Failed to add batch exemplars: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add batch exemplars: {str(exc)}",
        ) from exc


@router.get("/exemplars", tags=["oos", "admin"])
def get_exemplars() -> dict[str, list[str]]:
    """Get current exemplars (admin only).
    
    Returns all in-scope and OOS exemplars currently used by the
    semantic OOS detector.
    """
    try:
        return semantic_oos_detector.get_exemplars()
    except Exception as exc:
        logger.error(f"Failed to get exemplars: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get exemplars: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Testing/Debugging
# ---------------------------------------------------------------------------
@router.post("/test", tags=["oos", "debug"])
def test_oos_detection(
    question: str = Body(..., embed=True),
) -> dict[str, Any]:
    """Test OOS detection on a question (debug endpoint).
    
    Returns the classification result with confidence scores.
    """
    try:
        is_oos, confidence = semantic_oos_detector.is_oos(question)
        
        return {
            "question": question,
            "is_oos": is_oos,
            "confidence": confidence,
            "interpretation": (
                f"{'OUT-OF-SCOPE' if is_oos else 'IN-SCOPE'} "
                f"(confidence: {confidence:.2%})"
            ),
        }
    
    except Exception as exc:
        logger.error(f"Failed to test OOS detection: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test OOS detection: {str(exc)}",
        ) from exc
