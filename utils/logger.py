"""Application-wide logger using Loguru.

Single import point: ``from utils.logger import logger``.
"""

from __future__ import annotations

import sys

from loguru import logger as _logger

from models.config import settings


def _configure() -> None:
    """Configure the global Loguru logger from environment settings."""
    _logger.remove()
    _logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        colorize=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )


_configure()
logger = _logger
