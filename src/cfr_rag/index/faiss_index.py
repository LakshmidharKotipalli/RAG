"""FAISS dense index: exact ``IndexFlatIP`` by default, ``IndexHNSWFlat`` optionally.

Both are wrapped in ``IndexIDMap2`` so FAISS IDs are the SQLite chunk IDs.
Metadata filters are applied with an ``IDSelectorBatch`` inside the search, so
only allowed chunks are scored.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

from cfr_rag.config import DenseIndexConfig
from cfr_rag.index.base import DenseIndex, Hit

logger = logging.getLogger(__name__)

FILE = "dense.faiss"


class FaissIndex(DenseIndex):
    def __init__(self, cfg: DenseIndexConfig) -> None:
        import faiss

        self.faiss = faiss
        self.cfg = cfg
        self.index = None

    def build(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        faiss = self.faiss
        dim = vectors.shape[1]
        if self.cfg.type == "hnsw":
            inner = faiss.IndexHNSWFlat(dim, self.cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            inner.hnsw.efConstruction = self.cfg.hnsw_ef_construction
        else:
            inner = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(inner)
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32),
                                np.asarray(ids, dtype=np.int64))
        logger.info("Built FAISS %s index: %d x %d", self.cfg.type, self.index.ntotal, dim)

    def _params(self, allowed_ids: Collection[int] | None):
        """Search parameters plus the selector, which the caller must keep alive."""
        faiss = self.faiss
        if allowed_ids is None:
            if self.cfg.type == "hnsw":
                return faiss.SearchParametersHNSW(efSearch=self.cfg.hnsw_ef_search), None
            return None, None
        sel = faiss.IDSelectorBatch(np.asarray(sorted(allowed_ids), dtype=np.int64))
        if self.cfg.type == "hnsw":
            return faiss.SearchParametersHNSW(sel=sel, efSearch=self.cfg.hnsw_ef_search), sel
        return faiss.SearchParameters(sel=sel), sel

    def search(self, query: np.ndarray, k: int,
               allowed_ids: Collection[int] | None = None) -> list[Hit]:
        if self.index is None or self.index.ntotal == 0 or k <= 0:
            return []
        if allowed_ids is not None and not allowed_ids:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        k = min(k, self.index.ntotal)
        params, _selector = self._params(allowed_ids)
        scores, ids = (self.index.search(q, k, params=params) if params is not None
                       else self.index.search(q, k))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i != -1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(directory / FILE))

    def load(self, directory: Path) -> None:
        self.index = self.faiss.read_index(str(directory / FILE))

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)
