"""Token counting for chunk sizing.

``hf`` mode uses the embedding model's own tokenizer so the token cap matches
what the embedder sees. ``approx`` mode (no downloads) estimates 1.3 tokens
per whitespace-delimited word.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans of the countable units in ``text``."""
        ...

    @property
    def tokens_per_span(self) -> float: ...

    def tail(self, text: str, n_tokens: int) -> str:
        """The last ~``n_tokens`` tokens of ``text`` (used for chunk overlap)."""
        ...


class ApproxCounter:
    """Word-based estimate; deterministic and dependency-free."""

    ratio = 1.3

    def count(self, text: str) -> int:
        return math.ceil(len(_WORD_RE.findall(text)) * self.ratio)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _WORD_RE.finditer(text)]

    @property
    def tokens_per_span(self) -> float:
        return self.ratio

    def tail(self, text: str, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        spans = self.spans(text)
        keep = max(1, int(n_tokens / self.ratio))
        if len(spans) <= keep:
            return text
        return text[spans[-keep][0]:]


class HFCounter:
    """Exact counts with a Hugging Face tokenizer."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]

    @property
    def tokens_per_span(self) -> float:
        return 1.0

    def tail(self, text: str, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        spans = self.spans(text)
        if len(spans) <= n_tokens:
            return text
        return text[spans[-n_tokens][0]:]


@lru_cache(maxsize=4)
def get_counter(mode: str, model_name: str = "") -> TokenCounter:
    """Return a cached counter; falls back to ``approx`` if the tokenizer cannot load."""
    if mode == "hf":
        try:
            return HFCounter(model_name)
        except Exception as exc:
            logger.warning("Could not load tokenizer %s (%s); using approximate counts",
                           model_name, exc)
    return ApproxCounter()
