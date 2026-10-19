"""Run the configured chunker over the stored corpus."""

from __future__ import annotations

import logging

from cfr_rag.chunking.base import Chunker, extract_definitions
from cfr_rag.chunking.tokens import get_counter
from cfr_rag.config import AppConfig
from cfr_rag.registry import resolve
from cfr_rag.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def chunker_id(cfg: AppConfig) -> str:
    """Identifier for a chunker *and* its settings, e.g. ``structural-3f2a9c1b0d4e``.

    Chunks from different chunking configs coexist in the database under
    different IDs, so eval experiments never overwrite each other.
    """
    name = cfg.chunking.chunker.split(":")[-1].lower()
    return f"{name}-{cfg.fingerprint('chunking')}"


def build_chunker(cfg: AppConfig) -> Chunker:
    """Instantiate the chunker named in ``chunking.chunker``."""
    cls = resolve("chunker", cfg.chunking.chunker)
    counter = get_counter(cfg.chunking.tokenizer, cfg.embedding.model_name)
    return cls(cfg.chunking, counter)


def chunk_corpus(cfg: AppConfig, store: SQLiteStore | None = None) -> str:
    """Chunk every stored section, replace this chunker's rows and refresh definitions.

    Returns the chunker ID the chunks were stored under.
    """
    own = store is None
    store = store or SQLiteStore(cfg.paths.db_path)
    try:
        sections = list(store.iter_sections())
        if not sections:
            raise RuntimeError("No sections in the database; run `cfr-rag ingest` first.")
        cid = chunker_id(cfg)
        chunks = build_chunker(cfg).chunk(sections)
        start = store.next_chunk_id()
        store.conn.execute("DELETE FROM chunks WHERE chunker = ?", (cid,))
        for i, c in enumerate(chunks):
            c.chunk_id = start + i
            c.chunker = cid
            c.chunk_key = f"{cid}:{c.chunk_key.split(':', 1)[1]}"
        n = store.replace_chunks(chunks, cid)
        defs = [d for s in sections if s.is_definitions for d in extract_definitions(s)]
        store.replace_definitions(defs)
        store.set_meta(f"chunker:{cid}", {"chunking": cfg.chunking.model_dump(mode="json"),
                                          "num_chunks": n,
                                          "as_of_date": store.get_meta("as_of_date")})
        tokens = [c.token_count for c in chunks] or [0]
        logger.info("Chunker %s: %d chunks (mean %.0f, max %d tokens), %d definitions", cid, n,
                    sum(tokens) / len(tokens), max(tokens), len(defs))
        return cid
    finally:
        if own:
            store.close()
