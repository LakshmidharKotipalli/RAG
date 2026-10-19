"""Generate the gold evaluation set from the parsed corpus.

Expected section IDs come from the corpus structure itself (the section a
question was written from, the section defining a term, a section and the
section it cross-references), so labels are correct by construction. The LLM,
when enabled, only rephrases questions into natural language; any rephrasing
that leaks a citation into a question meant to test retrieval is discarded in
favour of the template.

Categories: ``citation``, ``topical``, ``definitional``, ``multi_section``,
``unanswerable``. Output is JSONL, one object per line, meant for hand editing.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cfr_rag.citations import extract_citations
from cfr_rag.config import AppConfig
from cfr_rag.ingest.models import Section
from cfr_rag.store.sqlite_store import Definition, SQLiteStore

logger = logging.getLogger(__name__)

UNANSWERABLE = [
    "What are the medical certificate requirements for commercial airline pilots?",
    "How many hours of rest must a commercial truck driver take after an 11-hour driving shift?",
    "What is the maximum civil penalty for a securities fraud violation under the Exchange Act?",
    "What forklift operator training does OSHA require?",
    "What are the FCC licensing requirements for an AM radio station?",
    "How do I calculate the home office tax deduction?",
    "What emission limits apply to coal-fired power plants under the Clean Air Act?",
    "What visa categories allow foreign nurses to work in the United States?",
    "What is the federal minimum wage for tipped employees?",
    "What are the requirements for a national bank to open a new branch?",
    "How long must employers keep I-9 employment eligibility forms?",
    "What are the lead paint disclosure rules for selling a pre-1978 house?",
    "What is the annual salary of the FDA Commissioner?",
    "Which pharmaceutical company had the most FDA warning letters last year?",
    "What is the current retail price of insulin in the United States?",
    "How many employees does the Drug Enforcement Administration have?",
    "What are the copyright registration fees for a sound recording?",
    "What are the Medicare Part D premiums for 2026?",
    "What safety standards apply to residential swimming pool drains?",
    "What are the airworthiness directive requirements for Boeing 737 aircraft?",
    "What does the European Medicines Agency require for a marketing authorisation application?",
    "What are Health Canada's natural health product licensing rules?",
    "What railroad track inspection frequencies does the FRA require?",
    "What are the USDA organic certification requirements for livestock pasture?",
    "What are the rules for federal student loan forgiveness for teachers?",
    "Which state has the strictest cannabis advertising laws?",
    "What is the statute of limitations for medical malpractice in California?",
    "How are patent term extensions calculated for utility patents in Europe?",
]

QUESTION_SYSTEM = (
    "You write evaluation questions for a search system over 21 CFR (U.S. food and drug "
    "regulations). Write ONE natural question a compliance professional might ask that is "
    "answered by the provided text. Do NOT mention any section, part or CFR number, and do not "
    "copy the heading verbatim. Reply as JSON: {\"question\": \"...\"}"
)
MULTI_SYSTEM = (
    "You write evaluation questions for a search system over 21 CFR. Write ONE natural question "
    "whose complete answer needs BOTH passages. Do NOT mention any section, part or CFR number. "
    "Reply as JSON: {\"question\": \"...\"}"
)


def _sections(store: SQLiteStore) -> list[Section]:
    return [s for s in store.iter_sections()
            if not s.reserved and s.kind == "section" and len(s.text.split()) >= 25]


def _stratified(rng: random.Random, items: list[Any], n: int, key) -> list[Any]:
    """Round-robin sample across groups (e.g. parts) so the set spans the title."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups)
    rng.shuffle(order)
    out: list[Any] = []
    while len(out) < n and any(groups.values()):
        for g in order:
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    return out


class GoldSetBuilder:
    def __init__(self, cfg: AppConfig, store: SQLiteStore, use_llm: bool) -> None:
        self.cfg = cfg
        self.store = store
        self.rng = random.Random(cfg.eval.seed)
        self.llm = None
        if use_llm:
            from cfr_rag.llm.factory import LLMRouter

            self.llm = LLMRouter(cfg.llm)

    # ------------------------------------------------------------------ llm helper
    def _llm_question(self, system: str, text: str) -> str | None:
        if self.llm is None:
            return None
        try:
            resp = self.llm.complete("judge", [{"role": "system", "content": system},
                                               {"role": "user", "content": text[:6000]}],
                                     json_mode=True, max_tokens=200)
            m = re.search(r"\{.*\}", resp.text, re.DOTALL)
            q = json.loads(m.group(0))["question"].strip() if m else resp.text.strip()
        except Exception as exc:
            logger.warning("LLM question generation failed (%s); using template", exc)
            return None
        if not q or extract_citations(q) or len(q) > 400:
            return None
        return q

    # ------------------------------------------------------------------ categories
    def citation(self, secs: list[Section], n: int) -> list[dict]:
        out = []
        for s in _stratified(self.rng, secs, n, key=lambda s: s.part):
            paras = [p for p in s.paragraphs if p.marker and p.level == 0]
            if paras and self.rng.random() < 0.5:
                p = self.rng.choice(paras)
                q = f"What does 21 CFR {s.section_id}{p.path} provide?"
                note = f"paragraph {p.path}"
            else:
                q = self.rng.choice([f"What does 21 CFR {s.section_id} require?",
                                     f"Summarize § {s.section_id}.",
                                     f"What is covered by section {s.section_id} of 21 CFR?"])
                note = ""
            out.append(self._item("citation", q, [s.section_id], "template",
                                  f"{s.heading}. {note}".strip()))
        return out

    def topical(self, secs: list[Section], n: int) -> list[dict]:
        out = []
        pool = [s for s in secs if len(s.text.split()) >= 60 and not s.is_definitions]
        for s in _stratified(self.rng, pool, n, key=lambda s: s.part):
            q = self._llm_question(QUESTION_SYSTEM, f"Heading: {s.heading}\n\n{s.text}")
            by = "llm"
            if q is None:
                by = "template"
                topic = s.heading.rstrip(".").lower()
                q = self.rng.choice([
                    f"What do the FDA regulations require regarding {topic}?",
                    f"Under the rules on {s.part_heading.lower()}, what applies to {topic}?",
                    f"What are the requirements for {topic}?"])
            out.append(self._item("topical", q, [s.section_id], by, s.heading))
        return out

    def definitional(self, defs: list[Definition], n: int) -> list[dict]:
        by_term_part: dict[tuple[str, str], list[str]] = defaultdict(list)
        for d in defs:
            by_term_part[(d.term.lower(), d.part)].append(d.section_id)
        pool = [d for d in defs if 3 <= len(d.term) <= 60 and len(d.text.split()) >= 8]
        out, seen = [], set()
        for d in _stratified(self.rng, pool, n * 2, key=lambda d: d.part):
            k = (d.term.lower(), d.part)
            if k in seen:
                continue
            seen.add(k)
            q = self.rng.choice([f"How is \"{d.term}\" defined in 21 CFR part {d.part}?",
                                 f"What does the term \"{d.term}\" mean under part {d.part}?",
                                 f"Define \"{d.term}\" as used in 21 CFR part {d.part}."])
            expected = sorted(set(by_term_part[k]))
            out.append(self._item("definitional", q, expected, "template", f"term: {d.term}"))
            if len(out) >= n:
                break
        return out

    def multi_section(self, secs: list[Section], n: int) -> list[dict]:
        by_id = {s.section_id: s for s in secs}
        pairs = []
        for s in secs:
            for ref in s.cross_refs:
                cit = extract_citations(f"§ {ref}")
                target = cit[0].section if cit else None
                if target and target in by_id and target != s.section_id:
                    pairs.append((s, by_id[target]))
        out = []
        for a, b in _stratified(self.rng, pairs, n, key=lambda p: p[0].part):
            q = self._llm_question(MULTI_SYSTEM, f"Passage A ({a.heading}):\n{a.text[:2500]}\n\n"
                                                 f"Passage B ({b.heading}):\n{b.text[:2500]}")
            by = "llm"
            if q is None:
                by = "template"
                q = (f"How do the rules on {a.heading.rstrip('.').lower()} relate to the "
                     f"requirements for {b.heading.rstrip('.').lower()}?")
            out.append(self._item("multi_section", q, [a.section_id, b.section_id], by,
                                  f"{a.section_id} cross-references {b.section_id}"))
        return out

    def unanswerable(self, n: int) -> list[dict]:
        qs = UNANSWERABLE[:]
        self.rng.shuffle(qs)
        return [self._item("unanswerable", q, [], "curated", "outside 21 CFR scope",
                           answerable=False) for q in qs[:n]]

    @staticmethod
    def _item(category: str, question: str, expected: list[str], by: str, notes: str,
              answerable: bool = True) -> dict:
        return {"category": category, "question": question, "expected_sections": expected,
                "answerable": answerable, "generated_by": by, "notes": notes}

    def build(self) -> list[dict]:
        g = self.cfg.eval.goldgen
        secs = _sections(self.store)
        defs = self.store.all_definitions()
        if not secs:
            raise RuntimeError("No sections in the database; run `cfr-rag ingest` first.")
        items = (self.citation(secs, g.citation) + self.topical(secs, g.topical)
                 + self.definitional(defs, g.definitional) + self.multi_section(secs, g.multi_section)
                 + self.unanswerable(g.unanswerable))
        counters: dict[str, int] = defaultdict(int)
        for it in items:
            counters[it["category"]] += 1
            it["id"] = f"{it['category'][:4]}-{counters[it['category']]:03d}"
        return [{"id": it.pop("id"), **it} for it in items]


def build_gold_set(cfg: AppConfig, out: Path | None = None, use_llm: bool | None = None,
                   overwrite: bool = False) -> Path:
    """Write the gold set JSONL and return its path. Refuses to clobber hand edits by default."""
    path = Path(out or cfg.eval.gold_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (it may contain hand edits). Use --overwrite.")
    use_llm = cfg.eval.goldgen.use_llm if use_llm is None else use_llm
    store = SQLiteStore(cfg.paths.db_path)
    try:
        items = GoldSetBuilder(cfg, store, use_llm).build()
        as_of = store.get_meta("as_of_date", "")
    finally:
        store.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({**it, "as_of_date": as_of}, ensure_ascii=False) + "\n")
    cats: dict[str, int] = defaultdict(int)
    for it in items:
        cats[it["category"]] += 1
    logger.info("Wrote %d gold questions to %s: %s", len(items), path, dict(cats))
    return path


def load_gold(path: Path) -> list[dict]:
    """Read a gold JSONL file, skipping blank lines and ``//`` comment lines."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
    return items
