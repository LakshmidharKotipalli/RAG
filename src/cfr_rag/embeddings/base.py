"""Embedder interface, Matryoshka projection and the on-disk-cached wrapper."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cfr_rag.config import EmbeddingConfig
from cfr_rag.embeddings.cache import EmbeddingCache, cache_key

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)


def project(vectors: np.ndarray, dim: int | None, normalize: bool = True) -> np.ndarray:
    """Matryoshka truncation to ``dim`` followed by re-normalisation."""
    if dim is not None and dim < vectors.shape[1]:
        vectors = vectors[:, :dim]
    return l2_normalize(vectors) if normalize else vectors.astype(np.float32)


class Embedder(ABC):
    """Produces full-dimension, L2-normalised vectors. Queries and documents differ."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    @property
    def model_id(self) -> str:
        return self.cfg.model_name

    @property
    @abstractmethod
    def full_dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode already-formatted texts into full-dimension normalised vectors."""

    def format_query(self, query: str) -> str:
        """Apply the model's query instruction prefix (documents get none)."""
        return self.cfg.query_template.format(instruction=self.cfg.query_instruction, query=query)


@dataclass
class EmbedStats:
    total: int = 0
    cache_hits: int = 0

    @property
    def embedded(self) -> int:
        return self.total - self.cache_hits


class CachedEmbedder:
    """Wraps an :class:`Embedder` with a disk cache and Matryoshka projection."""

    def __init__(self, embedder: Embedder, cache: EmbeddingCache | None) -> None:
        self.embedder = embedder
        self.cache = cache
        self.cfg = embedder.cfg
        self.last_stats = EmbedStats()

    @property
    def model_id(self) -> str:
        return self.embedder.model_id

    @property
    def dim(self) -> int:
        d = self.cfg.dimension
        return min(d, self.embedder.full_dim) if d else self.embedder.full_dim

    def _encode_cached(self, texts: Sequence[str], kind: str) -> np.ndarray:
        keys = [cache_key(self.model_id, self.cfg.max_length, kind, t) for t in texts]
        cached = self.cache.get_many(keys) if self.cache is not None else {}
        missing = [i for i, k in enumerate(keys) if k not in cached]
        self.last_stats = EmbedStats(total=len(texts), cache_hits=len(texts) - len(missing))
        if missing:
            logger.info("Embedding %d texts (%d served from cache)", len(missing),
                        len(texts) - len(missing))
            fresh = self.embedder.encode([texts[i] for i in missing])
            if self.cache is not None:
                self.cache.put_many({keys[i]: fresh[j] for j, i in enumerate(missing)})
            for j, i in enumerate(missing):
                cached[keys[i]] = fresh[j]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        full = np.stack([cached[k] for k in keys]).astype(np.float32)
        return project(full, self.cfg.dimension, self.cfg.normalize)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents verbatim (no instruction prefix)."""
        return self._encode_cached(texts, "doc")

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Embed queries with the instruction prefix; not cached on disk."""
        formatted = [self.embedder.format_query(q) for q in queries]
        full = self.embedder.encode(formatted)
        return project(full, self.cfg.dimension, self.cfg.normalize)


def build_embedder(cfg: EmbeddingConfig, cache_dir: str | None = None) -> CachedEmbedder:
    """Instantiate the configured embedder wrapped with the disk cache."""
    from cfr_rag.registry import resolve

    cls = resolve("embedder", cfg.backend)
    cache = EmbeddingCache.for_model(cache_dir, cfg.model_name) if cache_dir else None
    return CachedEmbedder(cls(cfg), cache)
