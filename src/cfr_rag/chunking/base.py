"""Chunker interface and helpers shared by all chunkers."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from cfr_rag.chunking.tokens import TokenCounter
from cfr_rag.config import ChunkingConfig
from cfr_rag.ingest.models import Section
from cfr_rag.store.sqlite_store import Chunk, Definition

_SENT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"“])")


class Chunker(ABC):
    """Turns sections into chunks. Subclasses set ``name`` and implement :meth:`chunk`."""

    name = "base"

    def __init__(self, cfg: ChunkingConfig, counter: TokenCounter) -> None:
        self.cfg = cfg
        self.counter = counter

    @abstractmethod
    def chunk(self, sections: Iterable[Section]) -> list[Chunk]:
        """Return chunks with ``chunk_id`` left at 0; IDs are assigned by the caller."""

    # -------------------------------------------------------------- shared helpers
    def header(self, sec: Section, paragraph_range: str = "") -> str:
        """Contextual breadcrumb prepended to each chunk before embedding."""
        subpart_part = f" > Subpart {sec.subpart}" if sec.subpart else ""
        h = self.cfg.header_template.format(
            part=sec.part, part_heading=sec.part_heading, subpart=sec.subpart,
            subpart_heading=sec.subpart_heading, subpart_part=subpart_part, chapter=sec.chapter,
            section_id=sec.section_id, heading=sec.heading,
        )
        return f"{h} {paragraph_range}".strip() if paragraph_range else h

    def make_chunk(self, sec: Section, ordinal: int, text: str, header: str,
                   section_ids: list[str] | None = None, paragraph_start: str = "",
                   paragraph_end: str = "", is_definition: bool = False,
                   term: str | None = None) -> Chunk:
        key = f"{self.name}:{sec.section_id}:{ordinal}" + (f":{term}" if term else "")
        return Chunk(
            chunk_id=0, chunk_key=key, chunker=self.name, section_id=sec.section_id,
            section_ids=section_ids or [sec.section_id], ordinal=ordinal, part=sec.part,
            chapter=sec.chapter, subpart=sec.subpart, header=header, text=text,
            paragraph_start=paragraph_start, paragraph_end=paragraph_end,
            token_count=self.counter.count(f"{header}\n{text}"), is_definition=is_definition,
            term=term, content_hash=hashlib.sha256(f"{header}\n{text}".encode()).hexdigest()[:16],
        )

    def split_long_text(self, text: str, max_tokens: int) -> list[str]:
        """Split one oversized paragraph at sentence boundaries, then by tokens."""
        pieces: list[str] = []
        cur = ""
        for sent in _SENT_RE.split(text):
            cand = f"{cur} {sent}".strip()
            if cur and self.counter.count(cand) > max_tokens:
                pieces.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            pieces.append(cur)
        out: list[str] = []
        for p in pieces:
            if self.counter.count(p) <= max_tokens:
                out.append(p)
                continue
            spans = self.counter.spans(p)
            step = max(1, int(max_tokens / self.counter.tokens_per_span))
            for i in range(0, len(spans), step):
                window = spans[i:i + step]
                out.append(p[window[0][0]:window[-1][1]])
        return out


def extract_definitions(sec: Section) -> list[Definition]:
    """Collect ``term -> text`` pairs from a definitions section.

    A term's text runs from the paragraph naming it until the next term or the
    next paragraph at the same or a shallower level.
    """
    defs: list[Definition] = []
    paras = sec.paragraphs
    for i, p in enumerate(paras):
        if not p.term:
            continue
        body = [p.text]
        for q in paras[i + 1:]:
            if q.term or (q.marker and q.level <= p.level and p.level >= 0):
                break
            if p.level < 0 and q.marker and q.level == 0:
                break
            body.append(q.text)
        defs.append(Definition(term=p.term, section_id=sec.section_id, part=sec.part,
                               chapter=sec.chapter, paragraph=p.path, text="\n".join(body)))
    return defs
