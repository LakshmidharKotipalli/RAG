"""Optional LLM-based query rewriting and multi-query expansion (toggled in config)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cfr_rag.llm.factory import LLMRouter

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "You rewrite user questions into precise search queries for the U.S. Code of Federal "
    "Regulations, Title 21 (Food and Drugs). Keep every section number, part number, defined "
    "term and product type from the question. Expand abbreviations (e.g. cGMP, IND, NDA, 510(k), "
    "DEA). Do not answer the question. Reply with the rewritten query only."
)
MULTI_SYSTEM = (
    "You generate alternative search queries for retrieving passages from 21 CFR. Each "
    "alternative should approach the question from a different angle (regulatory terminology, "
    "likely part or subpart topic, key obligations). Keep section numbers unchanged. Reply with "
    "a JSON array of strings and nothing else."
)


class QueryTransformer:
    """Applies the configured rewrite and/or expansion; never raises on LLM failure."""

    def __init__(self, llm: LLMRouter | None, rewrite: bool, multi_query: bool,
                 num_queries: int = 3) -> None:
        self.llm = llm
        self.rewrite_enabled = rewrite
        self.multi_enabled = multi_query
        self.num_queries = num_queries

    @property
    def active(self) -> bool:
        return self.llm is not None and (self.rewrite_enabled or self.multi_enabled)

    def transform(self, query: str, provider: str | None = None,
                  model: str | None = None) -> list[str]:
        """Return the list of queries to search with; the first is the primary query."""
        if not self.active:
            return [query]
        queries = [query]
        if self.rewrite_enabled:
            try:
                resp = self.llm.complete("rewrite", [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": query}], provider=provider, model=model)
                rewritten = _strip_thinking(resp.text).strip().strip('"')
                if rewritten and rewritten.lower() != query.lower():
                    queries = [rewritten, query]
            except Exception as exc:
                logger.warning("Query rewrite failed, using original query: %s", exc)
        if self.multi_enabled:
            try:
                resp = self.llm.complete("rewrite", [
                    {"role": "system", "content": MULTI_SYSTEM},
                    {"role": "user", "content": f"Question: {query}\nGenerate "
                                                f"{self.num_queries} alternatives."}],
                    provider=provider, model=model, json_mode=True)
                for q in parse_query_list(resp.text)[: self.num_queries]:
                    if q.lower() not in {x.lower() for x in queries}:
                        queries.append(q)
            except Exception as exc:
                logger.warning("Multi-query expansion failed: %s", exc)
        return queries


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def parse_query_list(text: str) -> list[str]:
    """Parse a JSON array (or object with a list value, or bullet lines) into strings."""
    text = _strip_thinking(text).strip()
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    candidates: list = []
    for blob in ([m.group(0)] if m else []) + [text]:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if isinstance(data, list):
            candidates = data
            break
    if not candidates:
        candidates = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln) for ln in text.splitlines()]
    return [str(q).strip() for q in candidates if str(q).strip()]
