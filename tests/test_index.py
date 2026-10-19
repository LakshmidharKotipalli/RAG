from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURE_XML, offline_cfg

from cfr_rag.config import load_config
from cfr_rag.embeddings.base import build_embedder, project
from cfr_rag.index.builder import build_indexes, load_indexes
from cfr_rag.index.metadata import IndexMismatchError
from cfr_rag.index.numpy_index import NumpyIndex
from cfr_rag.ingest.service import ingest


def test_matryoshka_projection_renormalizes():
    v = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    p = project(v, 4)
    assert p.shape == (3, 4)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)


def test_build_load_and_cache(tmp_path: Path):
    cfg = offline_cfg(tmp_path)
    ingest(cfg, xml_path=FIXTURE_XML)
    meta = build_indexes(cfg)
    assert meta.num_chunks > 5 and meta.dimension == 256
    emb = build_embedder(cfg.embedding, str(cfg.paths.cache_dir))
    build_indexes(cfg, embedder=emb)
    assert emb.last_stats.embedded == 0  # everything served from cache on rebuild
    bundle = load_indexes(cfg)
    q = bundle.embedder.embed_queries(["automatic electronic equipment calibration"])[0]
    hits = bundle.dense.search(q, 3)
    chunks = bundle.store.get_chunks([h[0] for h in hits])
    assert any(c.section_id == "211.68" for c in chunks.values())
    lex = bundle.lexical.search("211.68 computer backup file", 3)
    assert bundle.store.get_chunks([lex[0][0]])[lex[0][0]].section_id == "211.68"


def test_prefilter_by_part(tmp_path: Path):
    cfg = offline_cfg(tmp_path)
    ingest(cfg, xml_path=FIXTURE_XML)
    build_indexes(cfg)
    b = load_indexes(cfg)
    allowed = b.store.chunk_ids_matching(b.chunker_id, parts=["1300"])
    q = b.embedder.embed_queries(["equipment"])[0]
    hits = b.dense.search(q, 10, allowed_ids=allowed)
    assert hits and all(h[0] in allowed for h in hits)
    assert all(h[0] in allowed for h in b.lexical.search("drug", 10, allowed_ids=allowed))
    assert b.dense.search(q, 10, allowed_ids=[]) == []


def test_dimension_mismatch_fails_clearly(tmp_path: Path):
    cfg = offline_cfg(tmp_path)
    ingest(cfg, xml_path=FIXTURE_XML)
    build_indexes(cfg)
    with pytest.raises(IndexMismatchError, match="dimension"):
        load_indexes(cfg.with_overrides({"embedding": {"dimension": 128}}))
    with pytest.raises(IndexMismatchError, match="model"):
        load_indexes(cfg.with_overrides({"embedding": {"model_name": "other"}}))


def test_numpy_index_roundtrip(tmp_path: Path):
    idx = NumpyIndex()
    idx.build([10, 20], np.eye(2, dtype=np.float32))
    idx.save(tmp_path)
    other = NumpyIndex()
    other.load(tmp_path)
    assert other.search(np.array([0, 1.0]), 1) == [(20, 1.0)]


def test_faiss_flat_and_hnsw_if_available(tmp_path: Path):
    pytest.importorskip("faiss")
    from cfr_rag.config import DenseIndexConfig
    from cfr_rag.index.faiss_index import FaissIndex

    vecs = np.eye(4, dtype=np.float32)
    for typ in ("flat", "hnsw"):
        idx = FaissIndex(DenseIndexConfig(type=typ))
        idx.build([5, 6, 7, 8], vecs)
        assert idx.search(vecs[2], 1)[0][0] == 7
        assert all(h[0] in {5, 6} for h in idx.search(vecs[2], 2, allowed_ids=[5, 6]))


def test_qwen3_query_prefix_and_token_budget_batches():
    from cfr_rag.embeddings.qwen3 import Qwen3Embedder

    cfg = load_config("config/default.yaml").embedding.model_copy(
        update={"batch_size": 4, "max_batch_tokens": 1000})
    emb = Qwen3Embedder(cfg)  # model loads lazily; nothing downloaded here
    q = emb.format_query("What is a batch?")
    assert q.startswith("Instruct: ") and q.endswith("\nQuery:What is a batch?")
    lengths = [900, 10, 20, 400, 400, 30, 5, 5, 5]
    batches = emb._batches(lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 4
        assert max(lengths[i] for i in b) * len(b) <= 1000 or len(b) == 1
