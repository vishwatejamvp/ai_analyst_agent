"""Domain-specific exceptions used across the AI Analyst Agent.

Mapping these to HTTP status codes happens in ``api/routes.py`` so
service layers can stay framework-agnostic.
"""

from __future__ import annotations


class AIAnalystError(Exception):
    """Base class for all application errors."""

    status_code: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(AIAnalystError):
    """Raised when uploading or parsing source files fails."""

    status_code = 400


class DatabaseError(AIAnalystError):
    """Raised on MongoDB / MySQL failures."""

    status_code = 502


class VectorStoreError(AIAnalystError):
    """Raised on FAISS / embedding failures."""

    status_code = 502


class LLMError(AIAnalystError):
    """Raised on Claude / Anthropic API failures."""

    status_code = 502


class RoutingError(AIAnalystError):
    """Raised when a query cannot be routed to any backend."""

    status_code = 422


class UnsafeQueryError(AIAnalystError):
    """Raised when a generated/user-supplied SQL query is rejected."""

    status_code = 400
