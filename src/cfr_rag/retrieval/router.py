"""Query router: exact citation lookup first, hybrid search otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field

from cfr_rag.citations import CFRCitation, extract_citations


@dataclass
class Route:
    sections: list[CFRCitation] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    @property
    def has_exact(self) -> bool:
        return bool(self.sections)


def route_query(query: str) -> Route:
    """Find section citations (for exact lookup) and bare part references (for filtering)."""
    cits = extract_citations(query)
    sections = [c for c in cits if c.section]
    parts = [c.part for c in cits if not c.section]
    return Route(sections=sections, parts=parts)


def paragraph_matches(chunk_start: str, chunk_end: str, paragraph: str) -> bool:
    """True if a chunk covering ``chunk_start``..``chunk_end`` plausibly contains ``paragraph``."""
    if not paragraph:
        return True
    if not chunk_start:
        return True  # whole-section chunk
    top = paragraph.split(")")[0] + ")"
    return chunk_start.startswith(top) or chunk_end.startswith(top) or (
        chunk_start[:3] <= top[:3] <= chunk_end[:3])
