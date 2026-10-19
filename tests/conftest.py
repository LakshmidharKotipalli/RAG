"""Shared test helpers."""

from pathlib import Path

from cfr_rag.config import load_config

FIXTURE_XML = Path(__file__).parent / "fixtures" / "title-21-sample.xml"


def offline_cfg(tmp_path: Path, **over):
    base = {
        "paths": {"db_path": str(tmp_path / "db.sqlite"), "index_dir": str(tmp_path / "index"),
                  "cache_dir": str(tmp_path / "cache")},
        "chunking": {"tokenizer": "approx"},
        "embedding": {"backend": "hash", "model_name": "hash-test"},
        "index": {"dense": {"type": "numpy"}, "lexical": {"backend": "simple"}},
    }
    return load_config("config/default.yaml", overrides=base).with_overrides(over)


def fake_llm_overrides() -> dict:
    """Route both providers to the in-process fake (see tests/fakes.py)."""
    return {"llm": {"providers": {
        "openrouter": {"kind": "fakes:FakeProvider", "base_url": "http://fake"},
        "ollama": {"kind": "fakes:FakeProvider", "base_url": "http://fake"}}},
        "rerank": {"backend": "none"}}


def build_offline_pipeline(tmp_path: Path, **over):
    """Ingest the fixture, index it offline and return a pipeline using the fake LLM."""
    from cfr_rag.index.builder import build_indexes
    from cfr_rag.ingest.service import ingest
    from cfr_rag.pipeline import RAGPipeline

    cfg = offline_cfg(tmp_path).with_overrides(fake_llm_overrides()).with_overrides(over)
    ingest(cfg, date="2026-10-01", xml_path=FIXTURE_XML)
    build_indexes(cfg)
    return RAGPipeline.from_config(cfg)
