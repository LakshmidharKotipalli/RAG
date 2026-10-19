"""End-to-end RAG pipeline: route -> retrieve -> fuse -> rerank -> generate."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from cfr_rag.chunking.tokens import get_counter
from cfr_rag.config import AppConfig
from cfr_rag.generation.generator import ContextBuilder, Generator
from cfr_rag.generation.schemas import Citation, ContextBlock
from cfr_rag.index.builder import IndexBundle, load_indexes
from cfr_rag.llm.errors import LLMError
from cfr_rag.llm.factory import LLMRouter
from cfr_rag.rerank.base import Reranker, build_reranker
from cfr_rag.retrieval.hybrid import Retriever
from cfr_rag.retrieval.query_transform import QueryTransformer
from cfr_rag.retrieval.types import Candidate, Filters, RetrievalResult

logger = logging.getLogger(__name__)

ProviderName = Literal["openrouter", "ollama"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    chapters: list[str] | None = Field(None, description="Restrict to chapters, e.g. ['I']")
    parts: list[str] | None = Field(None, description="Restrict to parts, e.g. ['211']")
    provider: ProviderName | None = Field(None, description="LLM provider override")
    model: str | None = Field(None, description="LLM model override")
    top_k: int | None = Field(None, ge=1, le=50, description="Passages sent to the LLM")
    rerank: bool | None = Field(None, description="Override rerank.enabled")


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    provider: str
    model: str
    debug: dict[str, Any]


class RAGPipeline:
    """Holds every loaded component; thread-safe for concurrent read-only queries."""

    def __init__(self, cfg: AppConfig, bundle: IndexBundle, llm: LLMRouter,
                 retriever: Retriever, reranker: Reranker | None, generator: Generator) -> None:
        self.cfg = cfg
        self.bundle = bundle
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        # Local models (embedder/reranker) are not safe to run concurrently on one device.
        self._model_lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: AppConfig, bundle: IndexBundle | None = None,
                    llm: LLMRouter | None = None) -> RAGPipeline:
        """Load indexes and models named in ``cfg``."""
        bundle = bundle or load_indexes(cfg)
        llm = llm or LLMRouter(cfg.llm)
        r = cfg.retrieval
        transformer = QueryTransformer(llm, r.query_rewrite.enabled, r.multi_query.enabled,
                                       r.multi_query.num_queries)
        retriever = Retriever(bundle, r, transformer)
        reranker = build_reranker(cfg.rerank)
        counter = get_counter(cfg.chunking.tokenizer, cfg.embedding.model_name)
        ctx = ContextBuilder(cfg, bundle.store, counter, bundle.chunker_id)
        return cls(cfg, bundle, llm, retriever, reranker, Generator(cfg, llm, bundle.store, ctx))

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str, filters: Filters | None = None, mode: str = "hybrid",
                 rerank: bool | None = None, use_router: bool | None = None,
                 provider: str | None = None, model: str | None = None,
                 top_k: int | None = None) -> tuple[RetrievalResult, list[Candidate], dict]:
        """Retrieve and (optionally) rerank. Returns ``(retrieval, final_candidates, debug)``."""
        with self._model_lock:
            res = self.retriever.retrieve(question, filters, mode=mode, use_router=use_router,
                                          provider=provider, model=model)
            timings = dict(res.timings_ms)
            do_rerank = self.reranker is not None and (rerank is None or rerank)
            n_out = top_k or self.cfg.rerank.top_n_out
            if do_rerank and res.candidates:
                final = self.reranker.rerank(question, list(res.candidates),
                                             self.cfg.rerank.top_n_in, n_out,
                                             pin=self.cfg.retrieval.router.pin_exact_matches)
                timings["rerank_ms"] = self.reranker.last_latency_ms
            else:
                final = res.candidates[:n_out]
        return res, final, timings

    # ------------------------------------------------------------------ answering
    def query(self, req: QueryRequest) -> QueryResponse:
        """Answer a question (non-streaming)."""
        t0 = time.perf_counter()
        filters = Filters(chapters=req.chapters, parts=req.parts)
        res, final, timings = self.retrieve(req.question, filters, rerank=req.rerank,
                                            provider=req.provider, model=req.model, top_k=req.top_k)
        gen = self.generator.generate(req.question, final, req.provider, req.model)
        timings.update({k: v for k, v in gen.debug.items() if k.endswith("_ms")})
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        debug = self._debug(res, final, gen.context, timings, gen.provider, gen.model)
        if "usage" in gen.debug:
            debug["usage"] = gen.debug["usage"]
        return QueryResponse(answer=gen.answer, citations=gen.citations, refused=gen.refused,
                             provider=gen.provider, model=gen.model, debug=debug)

    def stream(self, req: QueryRequest) -> Iterator[dict[str, Any]]:
        """Event stream: ``retrieval`` (sources + debug), ``token``…, then ``done`` or ``error``."""
        t0 = time.perf_counter()
        try:
            filters = Filters(chapters=req.chapters, parts=req.parts)
            res, final, timings = self.retrieve(req.question, filters, rerank=req.rerank,
                                                provider=req.provider, model=req.model,
                                                top_k=req.top_k)
            prov, mname, _ = self.llm.resolve("generation", req.provider, req.model)
            yield {"type": "retrieval", "provider": prov.name, "model": mname,
                   "debug": self._debug(res, final, [], timings, prov.name, mname)}
            for event in self.generator.stream(req.question, final, req.provider, req.model):
                if event["type"] == "done":
                    gen = event["result"]
                    timings.update({k: v for k, v in gen.debug.items() if k.endswith("_ms")})
                    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    resp = QueryResponse(answer=gen.answer, citations=gen.citations,
                                         refused=gen.refused, provider=gen.provider,
                                         model=gen.model,
                                         debug=self._debug(res, final, gen.context, timings,
                                                           gen.provider, gen.model))
                    yield {"type": "done", "result": resp.model_dump()}
                else:
                    yield event
        except Exception as exc:
            if isinstance(exc, LLMError):
                logger.warning("Streaming query failed: %s", exc)
            else:
                logger.exception("Streaming query failed")
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    def _debug(self, res: RetrievalResult, final: list[Candidate], context: list[ContextBlock],
               timings: dict, provider: str, model: str) -> dict[str, Any]:
        return {
            "route": res.route, "queries": res.queries,
            "detected_citations": res.detected_citations,
            "missing_citations": res.missing_citations,
            "filters": {"chapters": res.filters.chapters, "parts": res.filters.parts},
            "provider": provider, "model": model, "timings_ms": timings,
            "fused_candidates": [c.debug(160) for c in res.candidates[:20]],
            "final_candidates": [c.debug(400) for c in final],
            "context_blocks": [{"index": b.index, "section_id": b.section_id, "label": b.label,
                                "kind": b.kind, "chunk_ids": b.chunk_ids} for b in context],
            "index": {"chunker": self.bundle.chunker_id, "as_of_date": self.bundle.meta.as_of_date,
                      "embedding_model": self.bundle.meta.embedding_model,
                      "dimension": self.bundle.meta.dimension},
            "rerank_enabled": self.reranker is not None,
        }
