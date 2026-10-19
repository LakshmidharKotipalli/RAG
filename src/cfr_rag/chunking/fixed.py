"""Fixed-size baseline chunker for comparison in eval.

Concatenates each part's sections and slides a window of ``fixed.size_tokens``
with ``fixed.overlap_tokens`` overlap, ignoring paragraph structure. Windows may
span section boundaries; every section a window touches is recorded in
``section_ids`` and the section contributing the most text becomes the parent.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable

from cfr_rag.chunking.base import Chunker
from cfr_rag.ingest.models import Section
from cfr_rag.store.sqlite_store import Chunk


class FixedSizeChunker(Chunker):
    name = "fixed"

    def chunk(self, sections: Iterable[Section]) -> list[Chunk]:
        by_part: dict[str, list[Section]] = defaultdict(list)
        order: list[str] = []
        for s in sections:
            if s.reserved or not s.text.strip():
                continue
            if s.part not in by_part:
                order.append(s.part)
            by_part[s.part].append(s)
        out: list[Chunk] = []
        for part in order:
            out.extend(self._chunk_part(by_part[part]))
        return out

    def _chunk_part(self, secs: list[Section]) -> list[Chunk]:
        texts, starts = [], []
        pos = 0
        for s in secs:
            t = f"§ {s.section_id} {s.heading}.\n{s.full_text(self.cfg.include_notes)}\n"
            starts.append(pos)
            texts.append(t)
            pos += len(t)
        doc = "".join(texts)
        spans = self.counter.spans(doc)
        if not spans:
            return []
        per = self.counter.tokens_per_span
        size = max(1, int(self.cfg.fixed.size_tokens / per))
        step = max(1, size - int(self.cfg.fixed.overlap_tokens / per))
        chunks: list[Chunk] = []
        for ordinal, i in enumerate(range(0, len(spans), step)):
            window = spans[i:i + size]
            a, b = window[0][0], window[-1][1]
            first = bisect.bisect_right(starts, a) - 1
            last = bisect.bisect_right(starts, max(a, b - 1)) - 1
            touched = list(range(first, last + 1))
            overlap = {j: min(b, starts[j] + len(texts[j])) - max(a, starts[j]) for j in touched}
            parent = secs[max(overlap, key=overlap.get)]
            header = f"21 CFR Part {parent.part}"
            c = self.make_chunk(parent, ordinal, doc[a:b], header,
                                section_ids=[secs[j].section_id for j in touched])
            c.chunk_key = f"{self.name}:part-{parent.part}:{ordinal}"
            chunks.append(c)
            if i + size >= len(spans):
                break
        return chunks
