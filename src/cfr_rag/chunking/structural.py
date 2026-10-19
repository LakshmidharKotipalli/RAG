"""Structure-aware chunker (default).

* One chunk per section when it fits under ``max_tokens``.
* Longer sections split at paragraph boundaries, preferring top-level
  paragraphs ((a), (b), ...), with ``overlap_tokens`` carried between pieces.
* Consecutive tiny sections in the same part and subpart are merged.
* Definitions sections are emitted as one chunk per defined term (tagged
  ``is_definition``) according to ``definitions.mode``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from cfr_rag.chunking.base import Chunker, extract_definitions
from cfr_rag.ingest.models import Paragraph, Section
from cfr_rag.store.sqlite_store import Chunk


@dataclass
class _Unit:
    text: str
    start: str
    end: str
    tokens: int


def _top_label(path: str) -> str:
    m = re.match(r"\([^)]+\)", path)
    return m.group(0) if m else ""


class StructuralChunker(Chunker):
    name = "structural"

    def chunk(self, sections: Iterable[Section]) -> list[Chunk]:
        out: list[Chunk] = []
        pending: list[Section] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if len(pending) == 1:
                out.extend(self._section_chunks(pending[0]))
            elif pending:
                out.append(self._merged_chunk(pending))
            pending, pending_tokens = [], 0

        for sec in sections:
            body = sec.full_text(self.cfg.include_notes).strip()
            if sec.reserved or not body:
                flush()
                continue
            if sec.is_definitions and self.cfg.definitions.mode in ("per_term", "both"):
                def_chunks = self._definition_chunks(sec)
                if sum(c.is_definition for c in def_chunks) >= self.cfg.definitions.min_terms:
                    flush()
                    out.extend(def_chunks)
                    if self.cfg.definitions.mode == "both":
                        out.extend(self._section_chunks(sec, ordinal_offset=len(def_chunks)))
                    continue
            n = self.counter.count(body)
            if n < self.cfg.min_tokens and not sec.is_definitions:
                same_group = pending and (pending[-1].part, pending[-1].subpart) == (sec.part, sec.subpart)
                if same_group and pending_tokens + n <= self.cfg.merge_max_tokens:
                    pending.append(sec)
                    pending_tokens += n
                    continue
                flush()
                pending, pending_tokens = [sec], n
                continue
            flush()
            out.extend(self._section_chunks(sec))
        flush()
        return out

    # ------------------------------------------------------------------ pieces
    def _section_chunks(self, sec: Section, ordinal_offset: int = 0) -> list[Chunk]:
        body = sec.full_text(self.cfg.include_notes)
        budget = self.cfg.max_tokens
        if self.counter.count(body) <= budget:
            return [self.make_chunk(sec, ordinal_offset, body, self.header(sec))]
        units = self._units(sec, budget)
        pieces: list[list[_Unit]] = []
        cur: list[_Unit] = []
        cur_tokens = 0
        for u in units:
            if cur and cur_tokens + u.tokens > budget:
                pieces.append(cur)
                cur, cur_tokens = [], 0
            cur.append(u)
            cur_tokens += u.tokens
        if cur:
            pieces.append(cur)
        chunks: list[Chunk] = []
        prev_text = ""
        for i, piece in enumerate(pieces):
            text = "\n".join(u.text for u in piece)
            overlap = self.counter.tail(prev_text, self.cfg.overlap_tokens) if prev_text else ""
            full = f"[...] {overlap}\n{text}" if overlap else text
            start, end = piece[0].start, piece[-1].end
            rng = start if start == end else f"{start}-{end}"
            chunks.append(self.make_chunk(sec, ordinal_offset + i, full, self.header(sec, rng),
                                          paragraph_start=start, paragraph_end=end))
            prev_text = text
        return chunks

    def _units(self, sec: Section, budget: int) -> list[_Unit]:
        """Top-level paragraph groups, broken down further when a group exceeds the budget."""
        groups: list[list[Paragraph]] = []
        for p in sec.paragraphs:
            top = _top_label(p.path)
            if groups and _top_label(groups[-1][0].path) == top and (top or not p.marker):
                groups[-1].append(p)
            else:
                groups.append([p])
        if self.cfg.include_notes and sec.notes:
            last = sec.paragraphs[-1].path if sec.paragraphs else ""
            groups.append([Paragraph(path=last, marker="", level=-1,
                                     text="\n".join(f"Note: {n}" for n in sec.notes))])
        units: list[_Unit] = []
        for g in groups:
            text = "\n".join(p.text for p in g)
            n = self.counter.count(text)
            if n <= budget:
                units.append(_Unit(text, g[0].path, g[-1].path, n))
                continue
            for p in g:
                n = self.counter.count(p.text)
                if n <= budget:
                    units.append(_Unit(p.text, p.path, p.path, n))
                else:
                    for piece in self.split_long_text(p.text, budget):
                        units.append(_Unit(piece, p.path, p.path, self.counter.count(piece)))
        return units

    def _merged_chunk(self, secs: list[Section]) -> Chunk:
        first = secs[0]
        ids = [s.section_id for s in secs]
        text = "\n\n".join(f"§ {s.section_id} {s.heading}.\n{s.full_text(self.cfg.include_notes)}"
                           for s in secs)
        subpart = f" > Subpart {first.subpart}" if first.subpart else ""
        header = f"21 CFR Part {first.part}{subpart} > §§ {', '.join(ids)}"
        return self.make_chunk(first, 0, text, header, section_ids=ids)

    def _definition_chunks(self, sec: Section) -> list[Chunk]:
        """One chunk per term, plus a preamble chunk for scope text that names no term."""
        defs = extract_definitions(sec)
        if not defs:
            return []
        chunks = []
        covered = {line for d in defs for line in d.text.split("\n")}
        preamble = "\n".join(p.text for p in sec.paragraphs if p.text not in covered)
        if preamble.strip():
            chunks.append(self.make_chunk(sec, 0, preamble, f"{self.header(sec)} (scope)"))
        offset = len(chunks)
        for i, d in enumerate(defs, start=offset):
            header = f"{self.header(sec)} > Definition: {d.term}"
            chunks.append(self.make_chunk(sec, i, d.text, header, paragraph_start=d.paragraph,
                                          paragraph_end=d.paragraph, is_definition=True,
                                          term=d.term))
        return chunks
