from pathlib import Path

from cfr_rag.config import deep_merge, load_config


def test_default_config_loads():
    cfg = load_config("config/default.yaml")
    assert cfg.ecfr.title == 21
    assert cfg.llm.default_provider == "openrouter"
    assert cfg.embedding.dimension is None


def test_overlay_and_overrides(tmp_path: Path):
    overlay = tmp_path / "exp.yaml"
    overlay.write_text("chunking:\n  chunker: fixed\n")
    cfg = load_config("config/default.yaml", overlays=[overlay], overrides={"rerank": {"top_n_out": 5}})
    assert cfg.chunking.chunker == "fixed"
    assert cfg.chunking.max_tokens == 512  # untouched sibling survives the merge
    assert cfg.rerank.top_n_out == 5


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 3}})
    assert out == {"a": {"b": 3, "c": 2}} and base["a"]["b"] == 1


def test_with_overrides_returns_new_config():
    cfg = load_config("config/default.yaml")
    new = cfg.with_overrides({"retrieval": {"fusion": {"rrf_k": 10}}})
    assert new.retrieval.fusion.rrf_k == 10 and cfg.retrieval.fusion.rrf_k == 60
