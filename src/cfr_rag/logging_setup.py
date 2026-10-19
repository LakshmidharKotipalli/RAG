"""Process-wide logging configuration."""

from __future__ import annotations

import logging

from cfr_rag.config import LoggingConfig

_CONFIGURED = False


def setup_logging(cfg: LoggingConfig | None = None, level: str | None = None) -> None:
    """Configure the root logger once; later calls only adjust the level."""
    global _CONFIGURED
    cfg = cfg or LoggingConfig()
    lvl = getattr(logging, (level or cfg.level).upper(), logging.INFO)
    if not _CONFIGURED:
        logging.basicConfig(level=lvl, format=cfg.format)
        for noisy in ("httpx", "urllib3", "openai", "filelock", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    logging.getLogger().setLevel(lvl)
