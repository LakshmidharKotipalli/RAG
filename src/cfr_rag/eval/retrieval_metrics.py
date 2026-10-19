"""Section-level retrieval metrics: Recall@k, MRR, nDCG@k.

Chunks are mapped to the sections they contain and de-duplicated in rank
order, so a section split into several chunks is counted once.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def section_ranking(chunk_section_ids: Iterable[Sequence[str]]) -> list[str]:
    """Flatten per-chunk section lists into a de-duplicated section ranking."""
    seen: set[str] = set()
    out: list[str] = []
    for ids in chunk_section_ids:
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    for i, sid in enumerate(ranking, 1):
        if sid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, sid in enumerate(ranking[:k], 1) if sid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_query(ranking: Sequence[str], relevant: Iterable[str], k_values: Sequence[int],
                ndcg_k: int) -> dict[str, float]:
    rel = set(relevant)
    out = {f"recall@{k}": recall_at_k(ranking, rel, k) for k in k_values}
    out["mrr"] = reciprocal_rank(ranking, rel)
    out[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranking, rel, ndcg_k)
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
