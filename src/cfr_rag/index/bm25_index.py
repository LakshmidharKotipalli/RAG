"""BM25 lexical indexes over the same chunk IDs as the dense index.

``BM25sIndex`` (default) uses the ``bm25s`` library with stopwords and
PyStemmer. ``SimpleBM25Index`` is a small NumPy implementation for tests and
environments without bm25s. Both tokenise with ``index.lexical.token_pattern``,
which keeps section numbers such as ``211.68`` as single tokens.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

from cfr_rag.config import LexicalIndexConfig
from cfr_rag.index.base import Hit, LexicalIndex, top_k_from_scores

logger = logging.getLogger(__name__)

_EN_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or such that the "
    "their then there these they this to was were will with which who whom shall may any all "
    "each other than under".split()
)


class BM25sIndex(LexicalIndex):
    def __init__(self, cfg: LexicalIndexConfig) -> None:
        import bm25s

        self.bm25s = bm25s
        self.cfg = cfg
        self.retriever = None
        self.ids = np.zeros(0, dtype=np.int64)
        self.stemmer = None
        if cfg.stemmer:
            try:
                import Stemmer

                self.stemmer = Stemmer.Stemmer(cfg.stemmer)
            except ImportError:
                logger.warning("PyStemmer not installed; BM25 runs without stemming")

    def _tokenize(self, texts: Sequence[str], return_ids: bool):
        return self.bm25s.tokenize(list(texts), lower=True, token_pattern=self.cfg.token_pattern,
                                   stopwords=self.cfg.stopwords, stemmer=self.stemmer,
                                   return_ids=return_ids, show_progress=False)

    def build(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.retriever = self.bm25s.BM25(k1=self.cfg.k1, b=self.cfg.b)
        self.retriever.index(self._tokenize(texts, return_ids=True), show_progress=False)
        logger.info("Built bm25s index over %d chunks", len(self.ids))

    def search(self, query: str, k: int, allowed_ids: Collection[int] | None = None) -> list[Hit]:
        if self.retriever is None or not len(self.ids):
            return []
        tokens = self._tokenize([query], return_ids=False)[0]
        if not tokens:
            return []
        scores = np.asarray(self.retriever.get_scores(tokens), dtype=np.float32)
        mask = np.isin(self.ids, list(allowed_ids)) if allowed_ids is not None else None
        hits = top_k_from_scores(scores, self.ids, k, mask)
        return [h for h in hits if h[1] > 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory / "bm25s"))
        np.save(directory / "bm25_ids.npy", self.ids)

    def load(self, directory: Path) -> None:
        self.retriever = self.bm25s.BM25.load(str(directory / "bm25s"))
        self.ids = np.load(directory / "bm25_ids.npy")


class SimpleBM25Index(LexicalIndex):
    """Okapi BM25 in NumPy with sparse postings; adequate for tests and small corpora."""

    def __init__(self, cfg: LexicalIndexConfig) -> None:
        self.cfg = cfg
        self.pattern = re.compile(cfg.token_pattern)
        self.ids = np.zeros(0, dtype=np.int64)
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_len = np.zeros(0, dtype=np.float32)

    def tokenize(self, text: str) -> list[str]:
        toks = [t.lower() for t in self.pattern.findall(text)]
        if self.cfg.stopwords:
            toks = [t for t in toks if t not in _EN_STOPWORDS]
        return toks

    def build(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.postings = {}
        lens = []
        for row, text in enumerate(texts):
            toks = self.tokenize(text)
            lens.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, {})[row] = tf
        self.doc_len = np.asarray(lens, dtype=np.float32)

    def search(self, query: str, k: int, allowed_ids: Collection[int] | None = None) -> list[Hit]:
        n = len(self.ids)
        if not n:
            return []
        avgdl = float(self.doc_len.mean()) or 1.0
        scores = np.zeros(n, dtype=np.float32)
        for term in set(self.tokenize(query)):
            post = self.postings.get(term)
            if not post:
                continue
            rows = np.fromiter(post.keys(), dtype=np.int64)
            tf = np.fromiter(post.values(), dtype=np.float32)
            idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            denom = tf + self.cfg.k1 * (1 - self.cfg.b + self.cfg.b * self.doc_len[rows] / avgdl)
            scores[rows] += idf * tf * (self.cfg.k1 + 1) / denom
        mask = np.isin(self.ids, list(allowed_ids)) if allowed_ids is not None else None
        return [h for h in top_k_from_scores(scores, self.ids, k, mask) if h[1] > 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "bm25_ids.npy", self.ids)
        np.save(directory / "bm25_doclen.npy", self.doc_len)
        (directory / "bm25_postings.json").write_text(json.dumps(self.postings))

    def load(self, directory: Path) -> None:
        self.ids = np.load(directory / "bm25_ids.npy")
        self.doc_len = np.load(directory / "bm25_doclen.npy")
        raw = json.loads((directory / "bm25_postings.json").read_text())
        self.postings = {t: {int(r): tf for r, tf in p.items()} for t, p in raw.items()}
