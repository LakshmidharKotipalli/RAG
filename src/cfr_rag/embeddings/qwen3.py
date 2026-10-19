"""Qwen3-Embedding (local) with last-token pooling, as described on the model card.

* Queries are formatted ``Instruct: {task}\\nQuery:{query}``; documents are not.
* The tokenizer pads on the left so the final position is always the last real
  token, whose hidden state is the embedding.
* Batches are built under a token budget (``max_batch_tokens``) after sorting
  by length, which keeps peak memory predictable on Apple-silicon unified memory.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

from cfr_rag.config import EmbeddingConfig
from cfr_rag.embeddings.base import Embedder, l2_normalize
from cfr_rag.embeddings.device import free_memory, pick_device, pick_dtype

logger = logging.getLogger(__name__)


class Qwen3Embedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        super().__init__(cfg)
        self._model = None
        self._tok = None
        self.device = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = pick_device(self.cfg.device)
        dtype = pick_dtype(self.cfg.dtype, self.device)
        logger.info("Loading %s on %s (%s)", self.cfg.model_name, self.device, dtype)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name, padding_side="left")
        self._model = AutoModel.from_pretrained(self.cfg.model_name, torch_dtype=dtype)
        self._model.to(self.device).eval()
        torch.set_grad_enabled(False)

    @property
    def full_dim(self) -> int:
        self._load()
        return int(self._model.config.hidden_size)

    def _batches(self, lengths: list[int]) -> list[list[int]]:
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        batches: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            longest = lengths[cur[0]] if cur else lengths[i]
            if cur and (len(cur) >= self.cfg.batch_size
                        or longest * (len(cur) + 1) > self.cfg.max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        if not texts:
            return np.zeros((0, self.full_dim), dtype=np.float32)
        lengths = [min(len(ids), self.cfg.max_length) for ids in
                   self._tok(list(texts), add_special_tokens=True, truncation=True,
                             max_length=self.cfg.max_length)["input_ids"]]
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        batches = self._batches(lengths)
        t0 = time.perf_counter()
        for b_idx, idx in enumerate(batches):
            enc = self._tok([texts[i] for i in idx], padding=True, truncation=True,
                            max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self._model(**enc).last_hidden_state
            pooled = hidden[:, -1].float().cpu().numpy()  # left padding => last token
            out[idx] = l2_normalize(pooled)
            if len(batches) > 20 and (b_idx + 1) % 20 == 0:
                done = sum(len(b) for b in batches[: b_idx + 1])
                logger.info("Embedded %d/%d texts (%.1f texts/s)", done, len(texts),
                            done / (time.perf_counter() - t0))
                free_memory(self.device)
        return out
