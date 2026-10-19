"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cfr_rag.store.sqlite_store import Chunk


@dataclass
class Filters:
    chapters: list[str] | None = None
    parts: list[str] | None = None

    def is_empty(self) -> bool:
        return not (self.chapters or self.parts)


@dataclass
class Candidate:
    """A chunk moving through the pipeline, with its score and rank at every stage."""

    chunk_id: int
    chunk: Chunk | None = None
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    pinned: bool = False

    @property
    def section_id(self) -> str:
        return self.chunk.section_id if self.chunk else ""

    def debug(self, text_chars: int = 240) -> dict[str, Any]:
        c = self.chunk
        return {
            "chunk_id": self.chunk_id, "section_id": c.section_id if c else None,
            "section_ids": c.section_ids if c else [], "header": c.header if c else None,
            "is_definition": bool(c and c.is_definition),
            "text": (c.text[:text_chars] + ("…" if len(c.text) > text_chars else "")) if c else None,
            "scores": {k: round(v, 5) for k, v in self.scores.items()}, "ranks": self.ranks,
            "sources": self.sources, "pinned": self.pinned,
        }


@dataclass
class RetrievalResult:
    query: str
    queries: list[str]
    route: str
    candidates: list[Candidate]
    detected_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)
    timings_ms: dict[str, float] = field(default_factory=dict)
