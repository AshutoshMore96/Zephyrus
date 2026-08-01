"""Consistent, timestamped logging used across the package."""

from __future__ import annotations

import logging

from .config import get_settings


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=get_settings().log_level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    return logging.getLogger(name)
