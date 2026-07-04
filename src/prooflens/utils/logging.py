"""Logging setup for scripts and the eval loop.

One configured logger factory so every script logs consistently (timestamped, level-tagged) to
stderr — Slurm captures stderr to slurm-<jobid>.err, so this is where run diagnostics land.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stderr. Level from the LOG_LEVEL env var (default INFO)."""
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("prooflens")
        root.setLevel(level)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    return logging.getLogger(f"prooflens.{name}" if name != "prooflens" else name)
