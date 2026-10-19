"""Reranker interface and the shared rerank-and-cut logic."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence

from cfr_rag.config import RerankConfig
from cfr_rag.retrieval.types import Candidate

logger = logging.getLogger(__name__)


class Reranker(ABC):
    def __init__(self, cfg: RerankConfig) -> None:
        self.cfg = cfg
        self.last_latency_ms = 0.0

    @abstractmethod
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score per document (higher is better)."""

    def rerank(self, query: str, candidates: list[Candidate], top_n_in: int | None = None,
               top_n_out: int | None = None, pin: bool = True) -> list[Candidate]:
        """Score the first ``top_n_in`` candidates and keep the best ``top_n_out``.

        Pinned candidates (exact citation hits) are always kept when ``pin`` is
        true; they take slots from the lowest-scoring unpinned candidates.
        """
        n_in = top_n_in or self.cfg.top_n_in
        n_out = top_n_out or self.cfg.top_n_out
        pool = candidates[:n_in]
        if not pool:
            return []
        t0 = time.perf_counter()
        scores = self.score(query, [c.chunk.embed_text for c in pool])
        self.last_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Reranked %d candidates in %.0f ms", len(pool), self.last_latency_ms)
        for c, s in zip(pool, scores, strict=True):
            c.scores["rerank"] = float(s)
        ordered = sorted(pool, key=lambda c: -c.scores["rerank"])
        for rank, c in enumerate(ordered, 1):
            c.ranks["rerank"] = rank
        if not pin:
            return ordered[:n_out]
        pinned = [c for c in ordered if c.pinned]
        rest = [c for c in ordered if not c.pinned]
        keep = pinned[:n_out] + rest[: max(0, n_out - len(pinned))]
        return sorted(keep, key=lambda c: -c.scores["rerank"])


class IdentityReranker(Reranker):
    """No-op reranker (``rerank.backend: none``): keeps fused order, scores by fused rank."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [1.0 / (i + 1) for i in range(len(documents))]


def build_reranker(cfg: RerankConfig) -> Reranker | None:
    """Instantiate the configured reranker, or ``None`` when disabled."""
    if not cfg.enabled:
        return None
    from cfr_rag.registry import resolve

    return resolve("reranker", cfg.backend)(cfg)
