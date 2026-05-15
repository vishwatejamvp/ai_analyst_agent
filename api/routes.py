"""FastAPI routes for the AI Analyst Agent."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from models.enums import DataSource, MetricStatus
from models.schemas import (
    AnalystResponse,
    IngestResponse,
    MetricDefinition,
    QueryRequest,
    RoutingDecision,
)
from services.analyst_service import analyst_orchestrator
from services.awqaf_ingest import ingest_payload
from services.ingestion_service import ingestion_service
from services.knowledge_base_service import knowledge_base_service
from services.mongo_service import mongo_service
from services.mysql_service import mysql_service
from services.routing_service import routing_service
from services.vector_service import vector_service
from utils.config import settings
from utils.exceptions import AIAnalystError
from utils.logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.claude_model,
        "embedding_model": settings.embedding_model,
        "vector_index_size": _safe(lambda: vector_service.size, default=0),
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
    replace: bool = Form(default=False),
) -> IngestResponse:
    """Upload an Excel/CSV/JSON file and ingest it into Mongo + FAISS."""
    suffix = Path(file.filename or "").suffix.lower() or ".xlsx"
    if suffix not in {".xlsx", ".xls", ".csv", ".json"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=upload_dir,
        suffix=suffix,
        prefix="upload_",
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        return ingestion_service.ingest_file(
            tmp_path, collection=collection, replace=replace
        )
    except AIAnalystError as exc:
        logger.error(f"Ingestion failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ---------------------------------------------------------------------------
# AWQAF JSON ingestion (same flatten logic as scripts/ingest_awqaf.py)
# ---------------------------------------------------------------------------
_AWQAF_KINDS = ("facts", "glossary", "contents")


class AwqafIngestRequest(BaseModel):
    """Frontend payload for ``POST /api/v1/awqaf/ingest``.

    The body is one of three canonical AWQAF shapes:

    * ``kind="facts"``    — payload is the JSON **array** from ``all_records.json``;
                            ``service`` (slug) is required.
    * ``kind="glossary"`` — payload is the JSON **object** from ``glossary.json``;
                            ``service`` is required.
    * ``kind="contents"`` — payload is ``AWQAF-DATA/contents.json``; ``service``
                            is ignored.
    """

    kind: str = Field(
        ...,
        description=f"AWQAF payload kind. One of {list(_AWQAF_KINDS)}.",
    )
    service: str | None = Field(
        default=None,
        description="Service slug (e.g. 'hajj-package-service'). Required for facts/glossary.",
    )
    payload: Any = Field(..., description="The raw AWQAF JSON body.")
    replace: bool = Field(
        default=False,
        description="Drop the target facts/metadata collection before insert.",
    )
    embed: bool = Field(default=True, description="Also build FAISS vectors.")


@router.post("/awqaf/ingest", tags=["ingest"])
def awqaf_ingest_json(
    request: AwqafIngestRequest = Body(...),
) -> dict[str, Any]:
    """Ingest a JSON payload from the frontend using AWQAF flatten rules."""
    if request.kind not in _AWQAF_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kind '{request.kind}'. Allowed: {list(_AWQAF_KINDS)}",
        )
    try:
        result = ingest_payload(
            request.payload,
            kind=request.kind,
            service=request.service,
            replace=request.replace,
            embed=request.embed,
        )
    except AIAnalystError as exc:
        logger.error(f"AWQAF ingest failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@router.get("/collections", tags=["meta"])
def list_collections() -> dict[str, list[str]]:
    return {
        "mongo": _safe(mongo_service.list_collections, default=[]),
        "mysql": _safe(mysql_service.list_tables, default=[]),
    }


# ---------------------------------------------------------------------------
# Routing preview (debug helper)
# ---------------------------------------------------------------------------
@router.get("/route", response_model=RoutingDecision, tags=["debug"])
def preview_route(
    question: str = Query(..., min_length=2),
    collection: str | None = None,
    data_source: DataSource = DataSource.AUTO,
) -> RoutingDecision:
    """Show the routing decision without actually executing the query."""
    return routing_service.decide(
        question, collection=collection, data_source=data_source
    )


# ---------------------------------------------------------------------------
# Vector search (debug helper)
# ---------------------------------------------------------------------------
@router.get("/search", tags=["debug"])
def vector_search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    hits = vector_service.search(q, top_k=top_k)
    return {"query": q, "hits": [hit.model_dump() for hit in hits]}


# ---------------------------------------------------------------------------
# Glossary (curated metric definitions)
# ---------------------------------------------------------------------------
@router.get("/glossary", response_model=list[MetricDefinition], tags=["glossary"])
def list_glossary(
    status: MetricStatus | None = Query(default=None),
) -> list[MetricDefinition]:
    """List curated metric definitions. Filter by status when provided."""
    return knowledge_base_service.list(status=status)


@router.get(
    "/glossary/{definition_id}",
    response_model=MetricDefinition,
    tags=["glossary"],
)
def get_glossary_entry(definition_id: str) -> MetricDefinition:
    entry = knowledge_base_service.get(definition_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Definition '{definition_id}' not found")
    return entry


@router.post(
    "/glossary",
    response_model=MetricDefinition,
    tags=["glossary"],
    status_code=200,
)
def upsert_glossary_entry(definition: MetricDefinition) -> MetricDefinition:
    """Insert or replace a metric definition (admin)."""
    try:
        return knowledge_base_service.upsert(definition)
    except AIAnalystError as exc:
        logger.error(f"Glossary upsert failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.delete("/glossary/{definition_id}", tags=["glossary"])
def delete_glossary_entry(definition_id: str) -> dict[str, Any]:
    try:
        deleted = knowledge_base_service.delete(definition_id)
    except AIAnalystError as exc:
        logger.error(f"Glossary delete failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Definition '{definition_id}' not found")
    return {"deleted": definition_id}


# ---------------------------------------------------------------------------
# Main analyst endpoint
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=AnalystResponse, tags=["analyst"])
def analyze(request: QueryRequest) -> AnalystResponse:
    """Run the full pipeline: route → fetch → context → LLM → chart."""
    try:
        return analyst_orchestrator.answer(request)
    except AIAnalystError as exc:
        logger.error(f"Analyst pipeline failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe(callable_, default):
    try:
        return callable_()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Safe call returned default: {exc}")
        return default
