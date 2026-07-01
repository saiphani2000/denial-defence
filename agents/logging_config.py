"""Structured logging setup."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("denial_defense")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_logging()
