"""Typed configuration loaded from YAML, with deep-merged overrides.

Every tunable value in the system lives in ``config/default.yaml``. Experiment
files (``config/experiments/*.yaml``) and per-call dictionaries are deep-merged
on top, so a single field can be changed without repeating the rest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CONFIG_ENV_VAR = "CFR_RAG_CONFIG"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    data_dir: Path
    raw_dir: Path
    db_path: Path
    index_dir: Path
    cache_dir: Path
    eval_dir: Path
    reports_dir: Path


class LoggingConfig(_Base):
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class ECFRConfig(_Base):
    base_url: str
    titles_endpoint: str
    full_endpoint: str
    structure_endpoint: str
    versions_endpoint: str
    section_url_template: str
    paragraph_url_template: str
    part_url_template: str
    title: int = 21
    chapters: list[str] = Field(default_factory=lambda: ["I", "II", "III"])
    download_mode: str = "full"
    timeout_s: float = 300
    max_retries: int = 4
    backoff_s: float = 2.0
    user_agent: str = "cfr21-rag"


class FixedChunkConfig(_Base):
    size_tokens: int = 400
    overlap_tokens: int = 50


class DefinitionsChunkConfig(_Base):
    mode: str = "per_term"
    heading_pattern: str = r"(?i)\bdefinitions?\b"
    min_terms: int = 2


class ChunkingConfig(_Base):
    chunker: str = "structural"
    tokenizer: str = "hf"
    max_tokens: int = 512
    overlap_tokens: int = 48
    min_tokens: int = 64
    merge_max_tokens: int = 384
    include_notes: bool = True
    header_template: str = "21 CFR Part {part}{subpart_part} > {section_id} {heading}"
    fixed: FixedChunkConfig = Field(default_factory=FixedChunkConfig)
    definitions: DefinitionsChunkConfig = Field(default_factory=DefinitionsChunkConfig)


class EmbeddingConfig(_Base):
    backend: str = "qwen3"
    model_name: str
    query_instruction: str
    query_template: str = "Instruct: {instruction}\nQuery:{query}"
    dimension: int | None = None
    max_length: int = 2048
    batch_size: int = 16
    max_batch_tokens: int = 16384
    device: str = "auto"
    dtype: str = "auto"
    normalize: bool = True


class DenseIndexConfig(_Base):
    type: str = "flat"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128


class LexicalIndexConfig(_Base):
    backend: str = "bm25s"
    k1: float = 1.5
    b: float = 0.75
    stopwords: str | None = "en"
    stemmer: str | None = "english"
    token_pattern: str = r"(?u)\b\w\w+\b"


class IndexConfig(_Base):
    dense: DenseIndexConfig = Field(default_factory=DenseIndexConfig)
    lexical: LexicalIndexConfig = Field(default_factory=LexicalIndexConfig)


class FusionConfig(_Base):
    method: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 1.0})


class RouterConfig(_Base):
    enabled: bool = True
    pin_exact_matches: bool = True
    exact_then_hybrid: bool = True
    part_as_filter: bool = True
    max_exact_chunks: int = 6


class ToggleConfig(_Base):
    enabled: bool = False


class MultiQueryConfig(_Base):
    enabled: bool = False
    num_queries: int = 3


class RetrievalConfig(_Base):
    dense_top_k: int = 50
    bm25_top_k: int = 50
    fused_top_k: int = 50
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    query_rewrite: ToggleConfig = Field(default_factory=ToggleConfig)
    multi_query: MultiQueryConfig = Field(default_factory=MultiQueryConfig)


class RerankConfig(_Base):
    enabled: bool = True
    backend: str = "qwen3"
    model_name: str = ""
    instruction: str = ""
    top_n_in: int = 50
    top_n_out: int = 8
    batch_size: int = 8
    max_length: int = 4096
    device: str = "auto"
    dtype: str = "auto"


class GenDefinitionsConfig(_Base):
    enabled: bool = True
    max_definitions: int = 3


class GenerationConfig(_Base):
    max_context_tokens: int = 6000
    expand_to_parent: bool = True
    max_parent_tokens: int = 1500
    definitions: GenDefinitionsConfig = Field(default_factory=GenDefinitionsConfig)
    refusal_text: str = "The retrieved sections of 21 CFR do not answer this question."
    snippet_chars: int = 320


class ProviderConfig(_Base):
    kind: str
    base_url: str
    api_key_env: str | None = None
    timeout_s: float = 60
    max_retries: int = 3
    backoff_s: float = 1.5
    extra_headers: dict[str, str] = Field(default_factory=dict)


class RoleConfig(_Base):
    provider: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    models: dict[str, str] = Field(default_factory=dict)


class LLMConfig(_Base):
    default_provider: str = "openrouter"
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleConfig]


class APIConfig(_Base):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)


class UIConfig(_Base):
    api_url: str = "http://127.0.0.1:8000"
    title: str = "21 CFR Assistant"
    request_timeout_s: float = 300


class GoldGenConfig(_Base):
    citation: int = 30
    topical: int = 40
    definitional: int = 35
    multi_section: int = 25
    unanswerable: int = 25
    use_llm: bool = True


class EvalConfig(_Base):
    gold_path: Path = Path("eval/gold.jsonl")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    ndcg_k: int = 10
    retrieval_modes: list[str] = Field(
        default_factory=lambda: ["dense", "bm25", "hybrid", "hybrid_rerank"]
    )
    generation_sample: int | None = None
    seed: int = 13
    goldgen: GoldGenConfig = Field(default_factory=GoldGenConfig)


class AppConfig(_Base):
    paths: PathsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ecfr: ECFRConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    llm: LLMConfig
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def fingerprint(self, *sections: str) -> str:
        """Return a short stable hash of the given top-level sections (all if none)."""
        data = self.model_dump(mode="json")
        if sections:
            data = {k: data[k] for k in sections}
        blob = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_overrides(self, overrides: dict[str, Any] | None) -> AppConfig:
        """Return a new config with ``overrides`` deep-merged on top of this one."""
        if not overrides:
            return self
        merged = deep_merge(self.model_dump(mode="json"), overrides)
        return AppConfig.model_validate(merged)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``; dicts merge, others replace."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(
    path: str | Path | None = None,
    overlays: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load the base YAML config, apply overlay files then a dict of overrides.

    Args:
        path: Base config file. Defaults to ``$CFR_RAG_CONFIG`` or ``config/default.yaml``.
        overlays: Partial YAML files merged in order (e.g. experiment configs).
        overrides: Final dictionary of overrides (e.g. from CLI flags or API requests).
    """
    load_dotenv()
    base_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not base_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {base_path}. Run from the repo root or set {CONFIG_ENV_VAR}."
        )
    data = _read_yaml(base_path)
    for overlay in overlays or []:
        data = deep_merge(data, _read_yaml(Path(overlay)))
    if overrides:
        data = deep_merge(data, overrides)
    return AppConfig.model_validate(data)
