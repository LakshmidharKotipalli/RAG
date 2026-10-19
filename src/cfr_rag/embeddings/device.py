"""Torch device and dtype selection shared by the embedder and reranker."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pick_device(requested: str = "auto") -> str:
    """Return ``mps`` on Apple silicon, ``cuda`` if available, else ``cpu`` (unless forced)."""
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(requested: str, device: str) -> Any:
    """float16 on GPU/MPS, float32 on CPU, unless a dtype is named explicitly."""
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    return torch.float16 if device in ("mps", "cuda") else torch.float32


def free_memory(device: str) -> None:
    """Release cached allocator memory between large batches."""
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
