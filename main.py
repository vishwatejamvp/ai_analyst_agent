"""FastAPI application entrypoint.

Run locally:

    uvicorn main:app --reload --port 8000

Or:

    python main.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request

STATIC_DIR = Path(__file__).resolve().parent / "static"

from api.routes import router as api_router
from api.statistics_routes import router as statistics_router
from utils.config import settings
from utils.exceptions import AIAnalystError
from utils.logger import logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        f"Starting AI Analyst Agent | env={settings.app_env} "
        f"model={settings.claude_model}"
    )
    yield
    logger.info("Shutting down AI Analyst Agent")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Analyst Agent",
        description=(
            "Production-grade AI analyst that routes natural-language questions "
            "across MongoDB, MySQL, and a FAISS vector store, then uses Claude "
            "to explain database-computed insights."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AIAnalystError)
    async def _domain_error_handler(_request: Request, exc: AIAnalystError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "details": exc.details},
        )

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(statistics_router)

    @app.get("/", include_in_schema=False)
    def chat_ui():
        """Single-page chat UI for the analyst."""
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            status_code=503,
            content={
                "service": "AI Analyst Agent",
                "error": "Chat UI not found (static/index.html missing)",
                "docs": "/docs",
                "health": "/api/v1/health",
            },
        )

    @app.get("/statistics", include_in_schema=False)
    def statistics_ui():
        """Statistics dashboard UI for Zakat payments."""
        stats_page = STATIC_DIR / "statistics.html"
        if stats_page.is_file():
            return FileResponse(stats_page)
        return JSONResponse(
            status_code=404,
            content={"error": "Statistics page not found"},
        )

    @app.get("/meta", tags=["meta"])
    def meta():
        return {
            "service": "AI Analyst Agent",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "chat_ui": "/",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "local",
    )
