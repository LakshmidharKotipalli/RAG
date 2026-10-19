"""Qwen3-Reranker scoring exactly as on the model card.

The reranker is a causal LM prompted to answer "yes" or "no" to whether a
document satisfies a query under an instruction. The relevance score is
``P(yes) = softmax([logit_no, logit_yes])[1]`` taken at the final position::

    <|im_start|>system
    Judge whether the Document meets the requirements based on the Query and the
    Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
    <|im_start|>user
    <Instruct>: {instruction}
    <Query>: {query}
    <Document>: {doc}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

This is *not* a cross-encoder with a regression head; using one would ignore
the model's training format.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cfr_rag.config import RerankConfig
from cfr_rag.embeddings.device import free_memory, pick_device, pick_dtype
from cfr_rag.rerank.base import Reranker

logger = logging.getLogger(__name__)

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class Qwen3Reranker(Reranker):
    def __init__(self, cfg: RerankConfig) -> None:
        super().__init__(cfg)
        self._model = None
        self._tok = None
        self.device = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(self.cfg.device)
        dtype = pick_dtype(self.cfg.dtype, self.device)
        logger.info("Loading %s on %s (%s)", self.cfg.model_name, self.device, dtype)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, torch_dtype=dtype)
        self._model.to(self.device).eval()
        self.yes_id = self._tok.convert_tokens_to_ids("yes")
        self.no_id = self._tok.convert_tokens_to_ids("no")
        self.prefix_ids = self._tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self._tok.encode(SUFFIX, add_special_tokens=False)

    def _inputs(self, pairs: list[str]):
        body_max = self.cfg.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self._tok(pairs, padding=False, truncation="longest_first",
                        return_attention_mask=False, max_length=body_max)
        enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
        batch = self._tok.pad(enc, padding=True, return_tensors="pt",
                              max_length=self.cfg.max_length)
        return {k: v.to(self.device) for k, v in batch.items()}

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        import torch

        self._load()
        out: list[float] = []
        for i in range(0, len(documents), self.cfg.batch_size):
            pairs = [format_pair(self.cfg.instruction, query, d)
                     for d in documents[i:i + self.cfg.batch_size]]
            with torch.inference_mode():
                logits = self._model(**self._inputs(pairs)).logits[:, -1, :]
            pair_logits = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            probs = torch.nn.functional.log_softmax(pair_logits.float(), dim=1)[:, 1].exp()
            out.extend(probs.cpu().tolist())
        free_memory(self.device)
        return out
