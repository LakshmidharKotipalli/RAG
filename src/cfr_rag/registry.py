"""Name-based component registry so every pipeline stage is swappable from config.

A config value such as ``chunking.chunker: structural`` resolves to a built-in
alias below. Any value of the form ``package.module:ClassName`` is imported
directly, which lets experiments plug in their own components without code
changes here.
"""

from __future__ import annotations

import importlib
from typing import Any

BUILTINS: dict[str, dict[str, str]] = {
    "chunker": {
        "structural": "cfr_rag.chunking.structural:StructuralChunker",
        "fixed": "cfr_rag.chunking.fixed:FixedSizeChunker",
    },
    "embedder": {
        "qwen3": "cfr_rag.embeddings.qwen3:Qwen3Embedder",
        "hash": "cfr_rag.embeddings.hashing:HashingEmbedder",
    },
    "dense_index": {
        "flat": "cfr_rag.index.faiss_index:FaissIndex",
        "hnsw": "cfr_rag.index.faiss_index:FaissIndex",
        "numpy": "cfr_rag.index.numpy_index:NumpyIndex",
    },
    "lexical_index": {
        "bm25s": "cfr_rag.index.bm25_index:BM25sIndex",
        "simple": "cfr_rag.index.bm25_index:SimpleBM25Index",
    },
    "reranker": {
        "qwen3": "cfr_rag.rerank.qwen3_reranker:Qwen3Reranker",
        "none": "cfr_rag.rerank.base:IdentityReranker",
    },
    "llm_provider": {
        "openai_compatible": "cfr_rag.llm.openrouter:OpenAICompatibleProvider",
        "ollama": "cfr_rag.llm.ollama:OllamaProvider",
    },
}


def resolve(kind: str, name: str) -> type[Any]:
    """Return the class registered as ``name`` for component ``kind``.

    Raises:
        KeyError: if the alias is unknown and is not a ``module:Class`` path.
    """
    target = BUILTINS.get(kind, {}).get(name)
    if target is None:
        if ":" not in name:
            known = ", ".join(sorted(BUILTINS.get(kind, {})))
            raise KeyError(f"Unknown {kind} '{name}'. Built-ins: {known}; or use 'module:Class'.")
        target = name
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def register(kind: str, name: str, target: str) -> None:
    """Register an extra alias at runtime (``target`` is ``module:Class``)."""
    BUILTINS.setdefault(kind, {})[name] = target
