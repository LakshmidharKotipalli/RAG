"""Grounded answer generation: context assembly, LLM call, citation extraction."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from cfr_rag.chunking.tokens import TokenCounter
from cfr_rag.citations import ecfr_section_url, extract_citations
from cfr_rag.config import AppConfig
from cfr_rag.generation.prompts import build_messages, format_block
from cfr_rag.generation.schemas import Citation, ContextBlock, GenerationResult
from cfr_rag.llm.factory import LLMRouter
from cfr_rag.retrieval.types import Candidate
from cfr_rag.store.sqlite_store import Definition, SQLiteStore

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


class ContextBuilder:
    """Turns reranked candidates into numbered context blocks under a token budget.

    * Split sections are expanded to the full section when it fits in
      ``generation.max_parent_tokens`` (each section appears once).
    * Defined terms mentioned in the question are pulled in from the
      definitions table, preferring definitions from parts already in context.
    """

    def __init__(self, cfg: AppConfig, store: SQLiteStore, counter: TokenCounter,
                 chunker_id: str) -> None:
        self.cfg = cfg
        self.store = store
        self.counter = counter
        self.chunker_id = chunker_id
        self._definitions: list[Definition] | None = None

    def definitions(self) -> list[Definition]:
        if self._definitions is None:
            self._definitions = self.store.all_definitions()
        return self._definitions

    def build(self, question: str, candidates: list[Candidate]) -> list[ContextBlock]:
        g = self.cfg.generation
        blocks: list[ContextBlock] = []
        used_sections: set[str] = set()
        used_chunks: set[int] = set()
        budget = g.max_context_tokens
        for cand in candidates:
            c = cand.chunk
            if c is None or c.chunk_id in used_chunks:
                continue
            sid = c.section_id
            text, kind, label, chunk_ids = c.text, "chunk", c.header, [c.chunk_id]
            if c.is_definition:
                kind = "definition"
            elif g.expand_to_parent and len(c.section_ids) == 1:
                if sid in used_sections:
                    continue
                siblings = self.store.chunks_for_section(sid, self.chunker_id)
                if len(siblings) > 1:
                    sec = self.store.get_section(sid)
                    full = sec.full_text(self.cfg.chunking.include_notes) if sec else ""
                    if sec and self.counter.count(full) <= g.max_parent_tokens:
                        text, kind = full, "section"
                        label = self._section_label(sec)
                        chunk_ids = [s.chunk_id for s in siblings]
            cost = self.counter.count(text) + 20
            if cost > budget:
                if blocks:
                    continue
                text = _head(self.counter, text, budget)  # first block is never dropped
                cost = budget
            budget -= cost
            used_chunks.update(chunk_ids)
            if kind != "definition":
                used_sections.update(c.section_ids)
            blocks.append(ContextBlock(index=len(blocks) + 1, section_id=sid, label=label,
                                       text=text, chunk_ids=chunk_ids, kind=kind))
        if g.definitions.enabled:
            for d in self.matching_definitions(question, blocks)[: g.definitions.max_definitions]:
                cost = self.counter.count(d.text) + 20
                if cost > budget:
                    break
                budget -= cost
                blocks.append(ContextBlock(
                    index=len(blocks) + 1, section_id=d.section_id,
                    label=f"21 CFR {d.section_id}{d.paragraph} (definition of \"{d.term}\")",
                    text=d.text, kind="definition"))
        return blocks

    def matching_definitions(self, question: str, blocks: list[ContextBlock]) -> list[Definition]:
        """Definitions whose term appears in the question and is not already in context."""
        q = " " + " ".join(_WORD.findall(question.lower())) + " "
        parts_in_context = {b.section_id.split(".")[0] for b in blocks}
        have = {(b.section_id, b.text[:60]) for b in blocks}
        hits: list[tuple[int, int, Definition]] = []
        for d in self.definitions():
            term = " ".join(_WORD.findall(d.term.lower()))
            if len(term) < 3 or f" {term} " not in q:
                continue
            if (d.section_id, d.text[:60]) in have:
                continue
            in_scope = 0 if d.part in parts_in_context else 1
            hits.append((in_scope, -len(term), d))
        hits.sort(key=lambda x: (x[0], x[1]))
        seen_terms: set[str] = set()
        out = []
        for _, _, d in hits:
            if d.term.lower() not in seen_terms:
                seen_terms.add(d.term.lower())
                out.append(d)
        return out

    @staticmethod
    def _section_label(sec) -> str:
        sub = f" > Subpart {sec.subpart}" if sec.subpart else ""
        return f"21 CFR Part {sec.part}{sub} > {sec.section_id} {sec.heading} (full section)"


class Generator:
    """Calls the LLM with grounded context and post-processes citations."""

    def __init__(self, cfg: AppConfig, llm: LLMRouter, store: SQLiteStore,
                 context_builder: ContextBuilder) -> None:
        self.cfg = cfg
        self.llm = llm
        self.store = store
        self.context_builder = context_builder

    def _prepare(self, question: str, candidates: list[Candidate]):
        blocks = self.context_builder.build(question, candidates)
        labelled = [format_block(b.index, b.label, b.text) for b in blocks]
        return blocks, build_messages(question, labelled, self.cfg.generation.refusal_text)

    def generate(self, question: str, candidates: list[Candidate], provider: str | None = None,
                 model: str | None = None) -> GenerationResult:
        """Non-streaming generation."""
        blocks, messages = self._prepare(question, candidates)
        prov, mname, _ = self.llm.resolve("generation", provider, model)
        if not blocks:
            return self.finalize(self.cfg.generation.refusal_text, blocks, prov.name, mname,
                                 {"llm_skipped": "no context retrieved"})
        t0 = time.perf_counter()
        resp = self.llm.complete("generation", messages, provider=provider, model=model)
        return self.finalize(resp.text, blocks, resp.provider, resp.model,
                             {"llm_ms": round((time.perf_counter() - t0) * 1000, 1),
                              "usage": resp.usage})

    def stream(self, question: str, candidates: list[Candidate], provider: str | None = None,
               model: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``{"type": "token"}`` events, then a final ``{"type": "done"}`` event."""
        blocks, messages = self._prepare(question, candidates)
        prov, mname, _ = self.llm.resolve("generation", provider, model)
        if not blocks:
            text = self.cfg.generation.refusal_text
            yield {"type": "token", "text": text}
            yield {"type": "done", "result": self.finalize(text, blocks, prov.name, mname,
                                                           {"llm_skipped": "no context"})}
            return
        t0 = time.perf_counter()
        first_token_ms = None
        parts: list[str] = []
        for delta in self.llm.stream("generation", messages, provider=provider, model=model):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
            parts.append(delta)
            yield {"type": "token", "text": delta}
        yield {"type": "done", "result": self.finalize(
            "".join(parts).strip(), blocks, prov.name, mname,
            {"llm_ms": round((time.perf_counter() - t0) * 1000, 1),
             "first_token_ms": first_token_ms})}

    # ------------------------------------------------------------------ post-processing
    def finalize(self, answer: str, blocks: list[ContextBlock], provider: str, model: str,
                 debug: dict[str, Any]) -> GenerationResult:
        refusal = self.cfg.generation.refusal_text
        refused = _normalize(refusal) in _normalize(answer)
        citations = self.extract_citations(answer, blocks)
        return GenerationResult(answer=answer, citations=citations, refused=refused,
                                provider=provider, model=model, context=blocks, debug=debug)

    def extract_citations(self, answer: str, blocks: list[ContextBlock]) -> list[Citation]:
        """One :class:`Citation` per distinct cited section+paragraph, in order of appearance."""
        ctx_sections = {b.section_id for b in blocks}
        out: list[Citation] = []
        seen: set[str] = set()
        for cit in extract_citations(answer):
            if not cit.section:
                continue
            key = cit.normalized()
            if key in seen:
                continue
            seen.add(key)
            sec = self.store.get_section(cit.section)
            snippet = ""
            if sec is not None and cit.paragraph:
                para = next((p for p in sec.paragraphs if p.path.startswith(cit.paragraph)), None)
                snippet = para.text if para else ""
            if not snippet:
                block = next((b for b in blocks if b.section_id == cit.section), None)
                snippet = block.text if block else (sec.text if sec else "")
            n = self.cfg.generation.snippet_chars
            snippet = snippet[:n] + ("…" if len(snippet) > n else "")
            url = ecfr_section_url(self.cfg.ecfr.paragraph_url_template if cit.paragraph
                                   else self.cfg.ecfr.section_url_template, cit.section,
                                   cit.paragraph, self.cfg.ecfr.title)
            out.append(Citation(section_id=cit.section, citation=cit.formatted(self.cfg.ecfr.title),
                                heading=sec.heading if sec else "", snippet=snippet, url=url,
                                in_context=cit.section in ctx_sections, exists=sec is not None))
        return out


def _head(counter: TokenCounter, text: str, n_tokens: int) -> str:
    spans = counter.spans(text)
    keep = max(1, int(n_tokens / counter.tokens_per_span))
    return text if len(spans) <= keep else text[: spans[keep - 1][1]]


def _normalize(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))
