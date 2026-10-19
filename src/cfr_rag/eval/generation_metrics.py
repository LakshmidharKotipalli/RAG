"""Generation metrics: faithfulness, citation accuracy, answer relevance, refusal behaviour."""

from __future__ import annotations

import logging
import re
from typing import Any

from cfr_rag.citations import extract_citations
from cfr_rag.eval.judge import Judge
from cfr_rag.pipeline import QueryRequest, QueryResponse, RAGPipeline

logger = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?])\s+")


def citing_sentences(answer: str, section_id: str) -> str:
    """Sentences of ``answer`` that cite ``section_id``."""
    out = [s for s in _SENT.split(answer)
           if any(c.section == section_id for c in extract_citations(s))]
    return " ".join(out)[:1500]


def evaluate_answer(pipe: RAGPipeline, judge: Judge, item: dict[str, Any],
                    max_section_chars: int = 6000) -> dict[str, Any]:
    """Run one gold question end to end and score it."""
    resp: QueryResponse = pipe.query(QueryRequest(question=item["question"]))
    row: dict[str, Any] = {
        "id": item["id"], "category": item["category"], "answerable": item["answerable"],
        "refused": resp.refused, "answer": resp.answer, "provider": resp.provider,
        "model": resp.model, "cited": [c.citation for c in resp.citations],
        "latency_ms": resp.debug.get("timings_ms", {}).get("total_ms"),
    }
    context = "\n\n".join(f"[{b['index']}] {b['label']}\n{b['text']}"
                            for b in resp.debug.get("context_blocks", []))  # exactly what the LLM saw
    try:
        row.update(judge.judge_answer(item["question"], context, resp.answer))
    except Exception as exc:
        logger.warning("Answer judge failed for %s: %s", item["id"], exc)
        row["judge_error"] = str(exc)

    # Citation accuracy: the cited section must exist AND support the citing sentences.
    by_section: dict[str, Any] = {}
    for c in resp.citations:
        by_section.setdefault(c.section_id, c)
    to_judge, exist_flags = [], []
    for sid, c in by_section.items():
        exist_flags.append(c.exists)
        if c.exists:
            sec = pipe.bundle.store.get_section(sid)
            to_judge.append({"citation": c.citation, "text": sec.full_text()[:max_section_chars],
                             "sentences": citing_sentences(resp.answer, sid) or resp.answer[:800]})
    supported: list[bool] = []
    try:
        supported = judge.judge_citations(to_judge)
    except Exception as exc:
        logger.warning("Citation judge failed for %s: %s", item["id"], exc)
        row["citation_judge_error"] = str(exc)
    n_cit = len(by_section)
    row["citations_total"] = n_cit
    row["citations_exist"] = sum(exist_flags)
    row["citations_supported"] = sum(supported)
    row["citation_accuracy"] = (sum(supported) / n_cit) if n_cit else None
    expected = set(item.get("expected_sections", []))
    row["cited_expected"] = bool(expected & set(by_section)) if expected else None
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-answer rows into report metrics."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    return {
        "n": len(rows),
        "faithfulness": mean(r.get("faithfulness") for r in answered),
        "citation_accuracy": mean(r.get("citation_accuracy") for r in answered),
        "citation_exists_rate": mean((r["citations_exist"] / r["citations_total"])
                                     if r.get("citations_total") else None for r in answered),
        "answer_relevance": mean(r.get("relevance") for r in ans),
        "cited_expected_rate": mean(float(r["cited_expected"]) if r.get("cited_expected")
                                    is not None else None for r in answered),
        "correct_refusal_rate": mean(float(r["refused"]) for r in unans),
        "false_refusal_rate": mean(float(r["refused"]) for r in ans),
        "uncited_answer_rate": mean(float(r.get("citations_total", 0) == 0) for r in answered),
        "mean_latency_ms": mean(r.get("latency_ms") for r in rows),
    }
