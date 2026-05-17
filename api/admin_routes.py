"""Admin API routes for managing real-time adaptive features.

Endpoints for:
- Prompt injection pattern management
- MongoDB index advisor
- Session store metrics
- Query rewriter feedback and stats
- Request queue monitoring
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.adaptive_query_rewriter import adaptive_query_rewriter
from services.adaptive_queue import adaptive_queue
from services.mongo_index_advisor import MongoIndexAdvisor
from services.mongo_service import mongo_service
from services.prompt_injection_detector import injection_detector
from services.session_hybrid import hybrid_session_service
from models.config import settings
from utils.logger import logger

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# Prompt Injection Defense
# ============================================================================


class InjectionExemplarsRequest(BaseModel):
    """Request to add new injection patterns."""

    patterns: list[str] = Field(..., min_length=1, max_length=50)


@router.post("/injection/exemplars")
async def add_injection_exemplars(request: InjectionExemplarsRequest) -> dict[str, Any]:
    """Add new injection patterns for detection.

    Real-time update: No redeployment needed.
    """
    if not settings.injection_detection_enabled:
        raise HTTPException(
            status_code=400,
            detail="Injection detection is disabled. Set INJECTION_DETECTION_ENABLED=true",
        )

    result = injection_detector.update_exemplars(request.patterns)
    return result


@router.get("/injection/exemplars")
async def get_injection_exemplars() -> dict[str, Any]:
    """Get current injection patterns."""
    if not settings.injection_detection_enabled:
        raise HTTPException(
            status_code=400,
            detail="Injection detection is disabled",
        )

    patterns = injection_detector.get_exemplars()
    return {
        "patterns": patterns,
        "count": len(patterns),
    }


# ============================================================================
# MongoDB Index Advisor
# ============================================================================


@router.get("/indexes/recommendations")
async def get_index_recommendations(collection: str | None = None) -> dict[str, Any]:
    """Get index recommendations from query pattern analysis."""
    if not hasattr(mongo_service, "index_advisor"):
        raise HTTPException(
            status_code=400,
            detail="Index advisor not initialized",
        )

    advisor: MongoIndexAdvisor = mongo_service.index_advisor
    recommendations = advisor.get_recommendations(collection)

    return {
        "recommendations": [
            {
                "collection": r.collection,
                "fields": r.fields,
                "reason": r.reason,
                "avg_exec_time_ms": r.avg_exec_time_ms,
                "query_count": r.query_count,
                "created": r.created,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recommendations
        ],
        "count": len(recommendations),
    }


@router.get("/indexes/stats/{collection}")
async def get_index_stats(collection: str) -> dict[str, Any]:
    """Get index statistics for a collection."""
    if not hasattr(mongo_service, "index_advisor"):
        raise HTTPException(
            status_code=400,
            detail="Index advisor not initialized",
        )

    advisor: MongoIndexAdvisor = mongo_service.index_advisor
    stats = advisor.get_index_stats(collection)

    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])

    return stats


@router.post("/indexes/drop-unused/{collection}")
async def drop_unused_indexes(
    collection: str, min_age_days: int = 7
) -> dict[str, Any]:
    """Drop indexes that haven't been used in min_age_days.

    WARNING: Use with caution in production.
    """
    if not settings.auto_drop_unused_indexes:
        raise HTTPException(
            status_code=400,
            detail="Auto-drop unused indexes is disabled. Set AUTO_DROP_UNUSED_INDEXES=true",
        )

    if not hasattr(mongo_service, "index_advisor"):
        raise HTTPException(
            status_code=400,
            detail="Index advisor not initialized",
        )

    advisor: MongoIndexAdvisor = mongo_service.index_advisor
    dropped = advisor.drop_unused_indexes(collection, min_age_days)

    return {
        "collection": collection,
        "dropped_indexes": dropped,
        "count": len(dropped),
    }


# ============================================================================
# Session Store Metrics
# ============================================================================


@router.get("/sessions/metrics")
async def get_session_metrics() -> dict[str, Any]:
    """Get session store metrics (cache hit rates, etc.)."""
    if settings.session_backend == "hybrid":
        metrics = hybrid_session_service.get_metrics()
    else:
        # In-memory session service doesn't have detailed metrics
        from services.session_service import session_service

        metrics = {
            "backend": "memory",
            "session_count": len(session_service._records),
        }

    return metrics


@router.post("/sessions/clear-l1")
async def clear_session_l1_cache() -> dict[str, Any]:
    """Clear L1 (in-memory) session cache."""
    if settings.session_backend != "hybrid":
        raise HTTPException(
            status_code=400,
            detail="Hybrid session backend not enabled",
        )

    count = hybrid_session_service.clear_l1()

    return {
        "status": "success",
        "cleared_sessions": count,
    }


# ============================================================================
# Query Rewriter
# ============================================================================


class QueryCorrectionRequest(BaseModel):
    """Request to teach query rewriter a correction."""

    original: str = Field(..., min_length=1, max_length=500)
    corrected: str = Field(..., min_length=1, max_length=500)


@router.post("/rewriter/feedback")
async def submit_query_correction(request: QueryCorrectionRequest) -> dict[str, Any]:
    """User corrects a query → rewriter learns for future.

    Real-time learning: No retraining needed.
    """
    if not settings.query_rewriter_enabled:
        raise HTTPException(
            status_code=400,
            detail="Query rewriter is disabled. Set QUERY_REWRITER_ENABLED=true",
        )

    result = adaptive_query_rewriter.learn_from_feedback(
        original=request.original,
        corrected=request.corrected,
    )

    return result


@router.get("/rewriter/corrections")
async def get_query_corrections() -> dict[str, Any]:
    """Get all learned query corrections."""
    if not settings.query_rewriter_enabled:
        raise HTTPException(
            status_code=400,
            detail="Query rewriter is disabled",
        )

    corrections = adaptive_query_rewriter.get_corrections()

    return {
        "corrections": corrections,
        "count": len(corrections),
    }


@router.get("/rewriter/stats")
async def get_rewriter_stats() -> dict[str, Any]:
    """Get query rewriter statistics."""
    if not settings.query_rewriter_enabled:
        raise HTTPException(
            status_code=400,
            detail="Query rewriter is disabled",
        )

    stats = adaptive_query_rewriter.get_stats()
    return stats


@router.post("/rewriter/refresh-vocabulary")
async def refresh_rewriter_vocabulary() -> dict[str, Any]:
    """Refresh vocabulary from current data.

    Call this after ingesting new data or adding new metrics.
    """
    if not settings.query_rewriter_enabled:
        raise HTTPException(
            status_code=400,
            detail="Query rewriter is disabled",
        )

    result = adaptive_query_rewriter.refresh_vocabulary()
    return result


# ============================================================================
# Request Queue
# ============================================================================


@router.get("/queue/metrics")
async def get_queue_metrics() -> dict[str, Any]:
    """Get request queue metrics."""
    if not settings.queue_enabled:
        raise HTTPException(
            status_code=400,
            detail="Request queue is disabled. Set QUEUE_ENABLED=true",
        )

    metrics = adaptive_queue.get_metrics()
    return metrics


@router.get("/queue/status/{task_id}")
async def get_queue_task_status(task_id: str) -> dict[str, Any]:
    """Get status of a queued task."""
    if not settings.queue_enabled:
        raise HTTPException(
            status_code=400,
            detail="Request queue is disabled",
        )

    status = adaptive_queue.get_status(task_id)

    if status["status"] == "unknown":
        raise HTTPException(status_code=404, detail="Task not found")

    return status


# ============================================================================
# System Health
# ============================================================================


@router.get("/health")
async def get_system_health() -> dict[str, Any]:
    """Get overall system health and feature status."""
    health = {
        "features": {
            "injection_detection": settings.injection_detection_enabled,
            "auto_create_indexes": settings.auto_create_indexes,
            "auto_drop_unused_indexes": settings.auto_drop_unused_indexes,
            "session_backend": settings.session_backend,
            "query_rewriter": settings.query_rewriter_enabled,
            "request_queue": settings.queue_enabled,
            "semantic_oos": settings.semantic_oos_enabled,
            "critic": settings.critic_enabled,
            "reranker": settings.reranker_enabled,
        },
        "status": "healthy",
    }

    # Check critical services
    try:
        # MongoDB
        mongo_service.client.admin.command("ping")
        health["mongodb"] = "connected"
    except Exception as exc:  # noqa: BLE001
        health["mongodb"] = f"error: {exc}"
        health["status"] = "degraded"

    # Redis (if hybrid sessions)
    if settings.session_backend == "hybrid" and settings.redis_url:
        try:
            redis_available = hybrid_session_service._redis_available
            health["redis"] = "connected" if redis_available else "unavailable"
        except Exception as exc:  # noqa: BLE001
            health["redis"] = f"error: {exc}"

    return health
