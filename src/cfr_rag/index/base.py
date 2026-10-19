"""Index interfaces. All indexes speak in the integer ``chunk_id`` shared with SQLite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

Hit = tuple[int, float]


class DenseIndex(ABC):
    """Inner-product search over L2-normalised vectors (i.e. cosine similarity)."""

    @abstractmethod
    def build(self, ids: Sequence[int], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int,
               allowed_ids: Collection[int] | None = None) -> list[Hit]:
        """Top-``k`` hits; ``allowed_ids`` restricts candidates *before* scoring."""

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class LexicalIndex(ABC):
    """Keyword (BM25) search."""

    @abstractmethod
    def build(self, ids: Sequence[int], texts: Sequence[str]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int, allowed_ids: Collection[int] | None = None) -> list[Hit]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> None: ...


def top_k_from_scores(scores: np.ndarray, ids: np.ndarray, k: int,
                      mask: np.ndarray | None = None) -> list[Hit]:
    """Pick the ``k`` best (id, score) pairs, honouring an optional boolean mask."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(ids[i]), float(scores[i])) for i in idx]
