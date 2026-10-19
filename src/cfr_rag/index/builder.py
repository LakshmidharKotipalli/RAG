"""Build, load and incrementally refresh the dense + lexical indexes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cfr_rag.chunking.service import chunk_corpus, chunker_id
from cfr_rag.config import AppConfig
from cfr_rag.embeddings.base import CachedEmbedder, build_embedder
from cfr_rag.index.base import DenseIndex, LexicalIndex
from cfr_rag.index.metadata import IndexMetadata
from cfr_rag.registry import resolve
from cfr_rag.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def make_dense(cfg: AppConfig) -> DenseIndex:
    return resolve("dense_index", cfg.index.dense.type)(cfg.index.dense)


def make_lexical(cfg: AppConfig) -> LexicalIndex:
    return resolve("lexical_index", cfg.index.lexical.backend)(cfg.index.lexical)


@dataclass
class IndexBundle:
    """Everything retrieval needs, loaded once and shared."""

    cfg: AppConfig
    store: SQLiteStore
    dense: DenseIndex
    lexical: LexicalIndex
    embedder: CachedEmbedder
    meta: IndexMetadata

    @property
    def chunker_id(self) -> str:
        return self.meta.chunker_id


def build_indexes(cfg: AppConfig, rechunk: bool = False,
                  embedder: CachedEmbedder | None = None) -> IndexMetadata:
    """Chunk (if needed), embed (cache-aware) and write FAISS + BM25 indexes to ``index_dir``."""
    store = SQLiteStore(cfg.paths.db_path)
    try:
        cid = chunker_id(cfg)
        if rechunk or store.count_chunks(cid) == 0:
            cid = chunk_corpus(cfg, store)
        chunks = list(store.iter_chunks(cid))
        if not chunks:
            raise RuntimeError("No chunks to index.")
        embedder = embedder or build_embedder(cfg.embedding, str(cfg.paths.cache_dir))
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        stats = embedder.last_stats
        ids = [c.chunk_id for c in chunks]
        dense = make_dense(cfg)
        dense.build(ids, vectors)
        lexical = make_lexical(cfg)
        lexical.build(ids, [c.embed_text for c in chunks])
        index_dir = Path(cfg.paths.index_dir)
        dense.save(index_dir)
        lexical.save(index_dir)
        meta = IndexMetadata(
            embedding_model=cfg.embedding.model_name, embedding_backend=cfg.embedding.backend,
            dimension=int(vectors.shape[1]), normalized=cfg.embedding.normalize,
            dense_type=cfg.index.dense.type, lexical_backend=cfg.index.lexical.backend,
            chunker_id=cid, num_chunks=len(chunks), as_of_date=store.get_meta("as_of_date", ""),
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        meta.save(index_dir)
        logger.info("Index ready at %s: %d chunks, dim %d (%d newly embedded, %d cached)",
                    index_dir, len(chunks), meta.dimension, stats.embedded, stats.cache_hits)
        return meta
    finally:
        store.close()


def load_indexes(cfg: AppConfig, embedder: CachedEmbedder | None = None) -> IndexBundle:
    """Load indexes from ``paths.index_dir`` and verify they match the embedding config."""
    index_dir = Path(cfg.paths.index_dir)
    meta = IndexMetadata.load(index_dir)
    meta.check_compatible(cfg)  # cheap check first: model name, backend, configured dim
    embedder = embedder or build_embedder(cfg.embedding, str(cfg.paths.cache_dir))
    meta.check_compatible(cfg, expected_dim=embedder.dim)  # actual query-vector dim
    dense = resolve("dense_index", meta.dense_type)(cfg.index.dense.model_copy(
        update={"type": meta.dense_type}))
    dense.load(index_dir)
    lexical = resolve("lexical_index", meta.lexical_backend)(cfg.index.lexical)
    lexical.load(index_dir)
    store = SQLiteStore(cfg.paths.db_path)
    logger.info("Loaded index %s (%d chunks, %s, dim %d, as of %s)", index_dir, meta.num_chunks,
                meta.embedding_model, meta.dimension, meta.as_of_date)
    return IndexBundle(cfg, store, dense, lexical, embedder, meta)


def reindex_since(cfg: AppConfig, since: str, date: str | None = None) -> dict[str, Any]:
    """Pull a newer snapshot, apply changed sections and rebuild the indexes.

    Only chunks whose text changed are re-embedded (the rest hit the cache).
    ``since`` is used to fetch the versioner's change records for the summary.
    """
    from cfr_rag.ingest.diff import diff_hashes
    from cfr_rag.ingest.ecfr_client import ECFRClient, SnapshotManifest
    from cfr_rag.ingest.service import parse_snapshot

    client = ECFRClient(cfg.ecfr)
    snap = client.download(cfg.paths.raw_dir, date=date)
    new_sections = parse_snapshot(cfg, snap)
    new_date = SnapshotManifest.load(snap).as_of_date
    store = SQLiteStore(cfg.paths.db_path)
    try:
        old_date = store.get_meta("as_of_date", "")
        added, removed, modified = diff_hashes(store.section_hashes(),
                                               {s.section_id: s.content_hash for s in new_sections})
        changed = set(added) | set(modified)
        store.upsert_sections([s for s in new_sections if s.section_id in changed])
        store.delete_sections(removed)
        store.set_meta("as_of_date", new_date)
    finally:
        store.close()
    try:
        api_changes = len(client.versions_since(since))
    except Exception as exc:
        logger.warning("Could not fetch versioner change records: %s", exc)
        api_changes = None
    embedder = build_embedder(cfg.embedding, str(cfg.paths.cache_dir))
    meta = build_indexes(cfg, rechunk=True, embedder=embedder)
    summary = {
        "old_as_of": old_date, "new_as_of": new_date, "since": since,
        "added": len(added), "removed": len(removed), "modified": len(modified),
        "changed_sections": sorted(changed)[:200], "versioner_change_records": api_changes,
        "chunks": meta.num_chunks, "chunks_embedded": embedder.last_stats.embedded,
    }
    logger.info("Reindex complete: %s", {k: v for k, v in summary.items() if k != "changed_sections"})
    return summary
