"""
Structured logging for the Enterprise RAG Assistant.
Logs document ingestion, vector index creation, query handling, and errors.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "enterprise_rag",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a structured logger.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO).
        log_file: Optional file path for file handler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "enterprise_rag") -> logging.Logger:
    """Return the application logger. Call setup_logger first if needed."""
    return logging.getLogger(name)
