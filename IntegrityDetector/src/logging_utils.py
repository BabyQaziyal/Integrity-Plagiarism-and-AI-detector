"""Consistent, timestamped logging used across the whole project."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "plagiarism", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Windows consoles default to cp1252; force UTF-8 so unicode in student
        # text or log messages never crashes the process.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
