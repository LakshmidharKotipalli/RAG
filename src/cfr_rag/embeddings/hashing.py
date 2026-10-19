"""Tiny deterministic bag-of-words embedder for offline tests and smoke runs.

Not a semantic model: it hashes lowercase word unigrams and bigrams into a
fixed number of buckets. Select it with ``embedding.backend: hash``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from cfr_rag.embeddings.base import Embedder, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class HashingEmbedder(Embedder):
    @property
    def model_id(self) -> str:
        return f"hash-{self.full_dim}"

    @property
    def full_dim(self) -> int:
        return 256

    def format_query(self, query: str) -> str:
        return query

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = _TOKEN.findall(text.lower())
            for gram in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]:
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "little")
                out[row, h % self.full_dim] += 1.0 if h & 1 << 31 else -1.0
        return l2_normalize(out + 1e-6)
