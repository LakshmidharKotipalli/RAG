"""Rank fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(ranked: Mapping[str, Sequence[int]], weights: Mapping[str, float],
                           k: int = 60) -> list[tuple[int, float]]:
    """Weighted RRF: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks.

    ``ranked`` maps a list name (e.g. ``dense`` or ``dense:2`` for a query
    variant) to chunk IDs in rank order. The weight is looked up by the name
    before any ``:`` suffix; unknown names default to 1.0.
    """
    scores: dict[int, float] = {}
    for name, ids in ranked.items():
        w = weights.get(name.split(":")[0], 1.0)
        if w == 0:
            continue
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
