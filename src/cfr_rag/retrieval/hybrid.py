"""Retriever: routing, dense + BM25 search, metadata pre-filtering and RRF fusion."""

from __future__ import annotations

import logging
import time
from collections.abc import Collection

from cfr_rag.config import RetrievalConfig
from cfr_rag.index.builder import IndexBundle
from cfr_rag.retrieval.fusion import reciprocal_rank_fusion
from cfr_rag.retrieval.query_transform import QueryTransformer
from cfr_rag.retrieval.router import paragraph_matches, route_query
from cfr_rag.retrieval.types import Candidate, Filters, RetrievalResult

logger = logging.getLogger(__name__)

MODES = ("dense", "bm25", "hybrid")


class Retriever:
    """Stateless over requests; holds the loaded indexes and config."""

    def __init__(self, bundle: IndexBundle, cfg: RetrievalConfig,
                 transformer: QueryTransformer | None = None) -> None:
        self.bundle = bundle
        self.cfg = cfg
        self.transformer = transformer

    # ------------------------------------------------------------------ primitives
    def allowed_ids(self, filters: Filters) -> list[int] | None:
        if filters.is_empty():
            return None
        return self.bundle.store.chunk_ids_matching(self.bundle.chunker_id, filters.chapters,
                                                    filters.parts)

    def dense_search(self, queries: list[str], k: int,
                     allowed: Collection[int] | None) -> dict[str, list[tuple[int, float]]]:
        vecs = self.bundle.embedder.embed_queries(queries)
        return {f"dense:{i}": self.bundle.dense.search(v, k, allowed) for i, v in enumerate(vecs)}

    def bm25_search(self, queries: list[str], k: int,
                    allowed: Collection[int] | None) -> dict[str, list[tuple[int, float]]]:
        return {f"bm25:{i}": self.bundle.lexical.search(q, k, allowed) for i, q in enumerate(queries)}

    def exact_lookup(self, query: str) -> tuple[list[Candidate], list[str], list[str]]:
        """Chunks of every section cited in ``query`` (paragraph-matching chunks first)."""
        route = route_query(query)
        found: list[Candidate] = []
        detected, missing = [], []
        seen: set[int] = set()
        for cit in route.sections:
            detected.append(cit.normalized())
            chunks = self.bundle.store.chunks_for_section(cit.section, self.bundle.chunker_id)
            if not chunks:
                missing.append(cit.normalized())
                continue
            chunks.sort(key=lambda c: (not paragraph_matches(c.paragraph_start, c.paragraph_end,
                                                             cit.paragraph), c.ordinal))
            for c in chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                found.append(Candidate(chunk_id=c.chunk_id, chunk=c, sources=["exact"],
                                       pinned=self.cfg.router.pin_exact_matches,
                                       scores={"exact": 1.0}))
        return found[: self.cfg.router.max_exact_chunks], detected, missing

    # ------------------------------------------------------------------ main entry
    def retrieve(self, query: str, filters: Filters | None = None, mode: str = "hybrid",
                 use_router: bool | None = None, provider: str | None = None,
                 model: str | None = None) -> RetrievalResult:
        """Retrieve fused candidates for ``query``.

        Args:
            mode: ``dense``, ``bm25`` or ``hybrid`` (eval uses the single-retriever modes).
            use_router: Override ``retrieval.router.enabled``.
            provider, model: LLM override used by query rewriting/expansion.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        filters = filters or Filters()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        use_router = self.cfg.router.enabled if use_router is None else use_router

        exact: list[Candidate] = []
        detected: list[str] = []
        missing: list[str] = []
        route_name = mode
        if use_router:
            exact, detected, missing = self.exact_lookup(query)
            route = route_query(query)
            if route.parts and self.cfg.router.part_as_filter and filters.is_empty():
                filters = Filters(parts=route.parts)
            timings["route_ms"] = _ms(t0)
            if exact and not self.cfg.router.exact_then_hybrid:
                return RetrievalResult(query, [query], "exact", self._hydrate(exact), detected,
                                       missing, filters, timings)
            if exact:
                route_name = f"exact+{mode}"

        t = time.perf_counter()
        queries = (self.transformer.transform(query, provider, model)
                   if self.transformer and self.transformer.active else [query])
        if len(queries) > 1 or queries[0] != query:
            timings["query_transform_ms"] = _ms(t)
        allowed = self.allowed_ids(filters)

        lists: dict[str, list[tuple[int, float]]] = {}
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            lists.update(self.dense_search(queries, self.cfg.dense_top_k, allowed))
            timings["dense_ms"] = _ms(t)
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            lists.update(self.bm25_search(queries, self.cfg.bm25_top_k, allowed))
            timings["bm25_ms"] = _ms(t)

        t = time.perf_counter()
        fused = reciprocal_rank_fusion({n: [cid for cid, _ in hits] for n, hits in lists.items()},
                                       self.cfg.fusion.weights, self.cfg.fusion.rrf_k)
        timings["fusion_ms"] = _ms(t)

        per_list = {n: {cid: (rank, score) for rank, (cid, score) in enumerate(hits, 1)}
                    for n, hits in lists.items()}
        cands: list[Candidate] = []
        by_id = {c.chunk_id: c for c in exact}
        for rank, (cid, score) in enumerate(fused[: self.cfg.fused_top_k], 1):
            cand = by_id.pop(cid, None) or Candidate(chunk_id=cid)
            cand.scores["rrf"] = score
            cand.ranks["fused"] = rank
            for name, table in per_list.items():
                if cid in table:
                    base = name.split(":")[0]
                    r, s = table[cid]
                    if base not in cand.ranks or r < cand.ranks[base]:
                        cand.ranks[base], cand.scores[base] = r, s
                    if base not in cand.sources:
                        cand.sources.append(base)
            cands.append(cand)
        # Exact hits first (in citation order; fused scores were merged into the same
        # objects above), then the remaining fused results.
        exact_ids = {c.chunk_id for c in exact}
        ordered = exact + [c for c in cands if c.chunk_id not in exact_ids]
        timings["retrieve_total_ms"] = _ms(t0)
        return RetrievalResult(query, queries, route_name, self._hydrate(ordered), detected,
                               missing, filters, timings)

    def _hydrate(self, cands: list[Candidate]) -> list[Candidate]:
        need = [c.chunk_id for c in cands if c.chunk is None]
        if need:
            chunks = self.bundle.store.get_chunks(need)
            for c in cands:
                if c.chunk is None:
                    c.chunk = chunks.get(c.chunk_id)
        return [c for c in cands if c.chunk is not None]


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 2)
