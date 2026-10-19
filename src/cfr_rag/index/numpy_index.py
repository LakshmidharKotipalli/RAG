"""Exact brute-force dense index in NumPy (no FAISS needed; for tests and tiny corpora)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

from cfr_rag.config import DenseIndexConfig
from cfr_rag.index.base import DenseIndex, Hit, top_k_from_scores

FILE = "dense.npz"


class NumpyIndex(DenseIndex):
    def __init__(self, cfg: DenseIndexConfig | None = None) -> None:
        self.ids = np.zeros(0, dtype=np.int64)
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def build(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int,
               allowed_ids: Collection[int] | None = None) -> list[Hit]:
        if not len(self.ids):
            return []
        scores = self.vectors @ query.reshape(-1).astype(np.float32)
        mask = np.isin(self.ids, list(allowed_ids)) if allowed_ids is not None else None
        return top_k_from_scores(scores, self.ids, k, mask)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / FILE, ids=self.ids, vectors=self.vectors)

    def load(self, directory: Path) -> None:
        data = np.load(directory / FILE)
        self.ids, self.vectors = data["ids"], data["vectors"]

    @property
    def size(self) -> int:
        return len(self.ids)
