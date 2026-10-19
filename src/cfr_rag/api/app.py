"""FastAPI application: ``POST /query``, ``GET /section/{citation}``, ``GET /health``."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cfr_rag.api.schemas import (
    HealthResponse,
    ParagraphOut,
    QueryBody,
    QueryResponse,
    SectionResponse,
)
from cfr_rag.citations import ecfr_section_url, parse_citation
from cfr_rag.config import AppConfig, load_config
from cfr_rag.llm.errors import LLMError, ProviderNotConfiguredError, ProviderUnavailableError
from cfr_rag.llm.factory import LLMRouter
from cfr_rag.logging_setup import setup_logging
from cfr_rag.pipeline import RAGPipeline
from cfr_rag.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def create_app(cfg: AppConfig | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    """Build the app. The pipeline loads at startup; failures are reported by ``/health``."""
    cfg = cfg or load_config()
    setup_logging(cfg.logging)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline
        app.state.load_error = None
        if app.state.pipeline is None:
            try:
                app.state.pipeline = RAGPipeline.from_config(cfg)
            except Exception as exc:  # keep serving /health so the UI can explain the problem
                logger.error("Pipeline failed to load: %s", exc)
                app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="21 CFR RAG API", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    if cfg.api.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.api.cors_origins,
                           allow_methods=["*"], allow_headers=["*"])

    def get_pipeline(request: Request) -> RAGPipeline:
        pipe = request.app.state.pipeline
        if pipe is None:
            raise HTTPException(503, f"Index not loaded: {request.app.state.load_error}. "
                                     "Run `cfr-rag ingest` and `cfr-rag index`.")
        return pipe

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryBody, request: Request):
        """Answer a question. With ``stream: true`` returns NDJSON events."""
        pipe = get_pipeline(request)
        if body.stream:
            def events() -> Iterator[bytes]:
                for event in pipe.stream(body):
                    yield (json.dumps(event, default=str) + "\n").encode()
            return StreamingResponse(events(), media_type="application/x-ndjson")
        try:
            return pipe.query(body)
        except ProviderNotConfiguredError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProviderUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/section/{citation:path}", response_model=SectionResponse)
    def section(citation: str, request: Request):
        """Full text of a section, e.g. ``/section/211.68`` or ``/section/21 CFR 211.68(b)``."""
        cit = parse_citation(citation)
        if cit is None or not cit.section:
            raise HTTPException(400, f"Not a section citation: {citation!r}")
        store = SQLiteStore(cfg.paths.db_path)
        try:
            sec = store.get_section(cit.section)
        finally:
            store.close()
        if sec is None:
            raise HTTPException(404, f"Section {cit.section} not found in the corpus")
        paras = sec.paragraphs
        if cit.paragraph:
            paras = [p for p in paras if p.path.startswith(cit.paragraph)]
            if not paras:
                raise HTTPException(404, f"Paragraph {cit.paragraph} not found in § {sec.section_id}")
        tmpl = cfg.ecfr.paragraph_url_template if cit.paragraph else cfg.ecfr.section_url_template
        return SectionResponse(
            section_id=sec.section_id, citation=cit.formatted(cfg.ecfr.title),
            requested=citation, heading=sec.heading, chapter=sec.chapter,
            chapter_heading=sec.chapter_heading, subchapter=sec.subchapter,
            subchapter_heading=sec.subchapter_heading, part=sec.part, part_heading=sec.part_heading,
            subpart=sec.subpart, subpart_heading=sec.subpart_heading,
            text="\n".join(p.text for p in paras),
            paragraphs=[ParagraphOut(path=p.path, text=p.text) for p in paras], notes=sec.notes,
            source_citation=sec.source_citation, amendment_dates=sec.amendment_dates,
            part_source=sec.part_source, part_authority=sec.part_authority,
            cross_refs=sec.cross_refs, as_of_date=sec.as_of_date,
            url=ecfr_section_url(tmpl, sec.section_id, cit.paragraph, cfg.ecfr.title))

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """Index status and reachability of each LLM provider."""
        pipe = request.app.state.pipeline
        if pipe is not None:
            m = pipe.bundle.meta
            index = {"loaded": True, "chunks": m.num_chunks, "as_of_date": m.as_of_date,
                     "embedding_model": m.embedding_model, "dimension": m.dimension,
                     "dense_type": m.dense_type, "lexical_backend": m.lexical_backend,
                     "chunker": m.chunker_id, "built_at": m.built_at,
                     "reranker": cfg.rerank.model_name if pipe.reranker else None}
            router = pipe.llm
        else:
            index = {"loaded": False, "error": request.app.state.load_error}
            router = LLMRouter(cfg.llm)
        providers = {n: {"reachable": h.reachable, "detail": h.detail, "models": h.models[:50]}
                     for n, h in router.health().items()}
        gen_role = cfg.llm.roles.get("generation")
        llm = {"default_provider": cfg.llm.default_provider,
               "generation_models": gen_role.models if gen_role else {}}
        default_ok = providers.get(cfg.llm.default_provider, {}).get("reachable", False)
        status = "ok" if index["loaded"] and default_ok else "degraded"
        return HealthResponse(status=status, index=index, providers=providers, llm=llm)

    return app
