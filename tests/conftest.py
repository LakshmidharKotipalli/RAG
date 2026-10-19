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
