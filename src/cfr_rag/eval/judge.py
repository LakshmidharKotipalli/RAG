"""LLM-as-judge prompts and parsing (runs through the same provider layer, role ``judge``)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from cfr_rag.llm.factory import LLMRouter

logger = logging.getLogger(__name__)

ANSWER_JUDGE_SYSTEM = """You are a strict evaluator of answers about U.S. federal regulations (21 CFR).
You receive a QUESTION, the CONTEXT passages the assistant was given, and the assistant's ANSWER.

1. Break the ANSWER into atomic factual claims (ignore citations, hedges and statements that the context does not cover).
2. For each claim decide if it is fully supported by the CONTEXT (not by your own knowledge).
3. Rate how well the ANSWER addresses the QUESTION from 1 (irrelevant) to 5 (fully and directly answers it). A clear, correct statement that the context does not answer the question scores 5 when the context indeed lacks the answer.

Reply with JSON only:
{"claims": [{"claim": "...", "supported": true}], "relevance": 1-5, "relevance_reason": "..."}"""

CITATION_JUDGE_SYSTEM = """You check citations in an answer about 21 CFR.
For each CITATION you get the cited section's TEXT and the SENTENCES of the answer that cite it.
Decide whether the section text supports those sentences.

Reply with JSON only:
{"citations": [{"citation": "...", "supported": true, "reason": "..."}]}"""


def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}")
    return json.loads(m.group(0))


class Judge:
    def __init__(self, llm: LLMRouter, provider: str | None = None, model: str | None = None) -> None:
        self.llm = llm
        self.provider = provider
        self.model = model

    def _ask(self, system: str, user: str) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(2):  # one re-ask on malformed JSON
            try:
                resp = self.llm.complete("judge", [{"role": "system", "content": system},
                                                   {"role": "user", "content": user}],
                                         provider=self.provider, model=self.model, json_mode=True)
                return _parse_json(resp.text)
            except (ValueError, json.JSONDecodeError) as exc:
                last = exc
        raise ValueError(f"Judge failed to return valid JSON: {last}")

    def judge_answer(self, question: str, context: str, answer: str) -> dict[str, Any]:
        """Faithfulness (supported claims / claims) and relevance (normalised to 0-1)."""
        data = self._ask(ANSWER_JUDGE_SYSTEM, f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                                              f"ANSWER:\n{answer}")
        claims = data.get("claims") or []
        supported = sum(1 for c in claims if c.get("supported") is True)
        rel = float(data.get("relevance", 1))
        return {
            "faithfulness": supported / len(claims) if claims else 1.0,
            "claims": len(claims), "unsupported_claims": [c.get("claim") for c in claims
                                                         if c.get("supported") is not True],
            "relevance": max(0.0, min(1.0, (rel - 1) / 4)),
            "relevance_reason": data.get("relevance_reason", ""),
        }

    def judge_citations(self, items: list[dict[str, str]]) -> list[bool]:
        """``items``: ``{"citation", "text", "sentences"}``; returns supported flags in order."""
        if not items:
            return []
        body = "\n\n".join(f"CITATION: {it['citation']}\nTEXT:\n{it['text']}\nSENTENCES:\n"
                           f"{it['sentences']}" for it in items)
        data = self._ask(CITATION_JUDGE_SYSTEM, body)
        verdicts = {str(v.get("citation", "")).strip(): bool(v.get("supported"))
                    for v in data.get("citations", [])}
        flags = []
        for i, it in enumerate(items):
            if it["citation"] in verdicts:
                flags.append(verdicts[it["citation"]])
            else:
                lst = data.get("citations", [])
                flags.append(bool(lst[i].get("supported")) if i < len(lst) else False)
        return flags
