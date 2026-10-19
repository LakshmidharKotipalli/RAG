"""Parse eCFR versioner XML into :class:`Section` objects.

Expected layout (eCFR "full" XML, GPO-derived)::

    <ECFR>
      <DIV1 TYPE="TITLE" N="21"><HEAD>Title 21 - Food and Drugs</HEAD>
        <DIV3 TYPE="CHAPTER" N="I"><HEAD>CHAPTER I - FOOD AND DRUG ...</HEAD>
          <DIV4 TYPE="SUBCHAP" N="C"> ...
            <DIV5 TYPE="PART" N="211"><HEAD>PART 211 - ...</HEAD><AUTH/><SOURCE/>
              <DIV6 TYPE="SUBPART" N="D"> ...
                <DIV8 TYPE="SECTION" N="211.68"><HEAD>§ 211.68 Automatic ...</HEAD>
                  <P>(a) ...</P> <CITA>[43 FR ..., as amended at ...]</CITA>

The parser keys on the ``TYPE`` attribute rather than the DIV number, so it
tolerates missing levels (parts with no subpart, subject groups, etc.) and
the per-part XML returned by ``?part=`` requests.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from cfr_rag.citations import extract_citations
from cfr_rag.ingest.models import Paragraph, Section
from cfr_rag.ingest.paragraphs import (
    ITALIC_CLOSE,
    ITALIC_OPEN,
    ParagraphNester,
    split_leading_markers,
)

logger = logging.getLogger(__name__)

HIERARCHY_TYPES = {"TITLE", "SUBTITLE", "CHAPTER", "SUBCHAP", "PART", "SUBPART", "SUBJGRP"}
LEAF_TYPES = {"SECTION", "APPENDIX"}
ITALIC_TAGS = {"I", "E"}
SKIP_TAGS = {"HEAD", "CITA", "EDNOTE", "EFFDNOT", "SECAUTH", "FTREF"}
NOTE_TAGS = {"EDNOTE", "EFFDNOT", "NOTE"}
BLOCK_TAGS = {"P", "FP", "HD", "PSPACE"}
DASHES = re.compile(r"\s*[—–-]{1,2}\s*")
WS = re.compile(r"[ \t\r\f\v ]+")
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sept?|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}"
)
_TERM_VERBS = r"(?:means?|mean|refers? to|is defined as|includes?|shall mean|has the meaning)"
_QUOTED_TERM_RE = re.compile(
    rf"^[\"“]([^\"”]{{1,80}})[\"”]\s*(?:or\s+[\"“][^\"”]{{1,80}}[\"”]\s*)?,?\s*{_TERM_VERBS}\b"
)
_ITALIC_TERM_RE = re.compile(
    rf"^{ITALIC_OPEN}([^{ITALIC_CLOSE}]{{1,100}}){ITALIC_CLOSE}\s*[.,:]?\s*(?:[-—–]\s*)?(?:{_TERM_VERBS}\b|[A-Z(])"
)


def clean(text: str) -> str:
    """Collapse whitespace while keeping newlines meaningful."""
    lines = [WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def strip_italics(text: str) -> str:
    return text.replace(ITALIC_OPEN, "").replace(ITALIC_CLOSE, "")


def inline_text(el: etree._Element, mark_italic: bool = False) -> str:
    """Flatten an element's text; optionally wrap italic runs in sentinel characters."""
    parts: list[str] = [el.text or ""]
    for child in el:
        if not isinstance(child.tag, str):  # comments / processing instructions
            parts.append(child.tail or "")
            continue
        tag = child.tag
        if tag in {"FTREF"}:
            pass
        elif tag == "SU":
            parts.append(f"^{inline_text(child)}")
        elif mark_italic and tag in ITALIC_TAGS:
            parts.append(f"{ITALIC_OPEN}{inline_text(child)}{ITALIC_CLOSE}")
        elif tag == "GPOTABLE":
            parts.append("\n" + table_text(child) + "\n")
        else:
            parts.append(inline_text(child, mark_italic))
        parts.append(child.tail or "")
    return "".join(parts)


def block_text(el: etree._Element) -> str:
    """Text of a container whose children are separate blocks (HED, PSPACE, P)."""
    if not len(el):
        return clean(inline_text(el))
    pieces = [clean(el.text or "")] + [clean(inline_text(c)) for c in el]
    return " ".join(p for p in pieces if p)


def table_text(table: etree._Element) -> str:
    """Render a GPOTABLE as pipe-separated rows (header row first)."""
    rows: list[str] = []
    title = table.find("TTITLE")
    if title is not None and inline_text(title).strip():
        rows.append(clean(inline_text(title)))
    heads = [clean(inline_text(h)) for h in table.iter("CHED")]
    if heads:
        rows.append(" | ".join(heads))
    for row in table.iter("ROW"):
        cells = [clean(inline_text(c)) for c in row.findall("ENT")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def split_head(head: str, prefix_re: str) -> tuple[str, str]:
    """Split ``"PART 211 - CURRENT GOOD ..."`` into ``("211", "CURRENT GOOD ...")``."""
    head = clean(head)
    m = re.match(prefix_re + r"\s*(?P<num>[\w.\-]+)\s*(?:[—–\-]+|\s)\s*(?P<rest>.*)$", head,
                 re.IGNORECASE)
    if m:
        return m.group("num").strip(" .—–-"), m.group("rest").strip(" .—–-")
    return "", head


def section_heading(head: str, section_id: str) -> str:
    """Strip ``§ 211.68`` from a section HEAD, leaving the caption."""
    text = clean(head)
    text = re.sub(rf"^§+\s*{re.escape(section_id)}\s*", "", text)
    text = re.sub(r"^§+\s*[\d.]+[a-z]?\s*", "", text)
    return text.strip().rstrip(".").strip()


@dataclass
class _Context:
    title: int = 21
    chapter: str = ""
    chapter_heading: str = ""
    subchapter: str = ""
    subchapter_heading: str = ""
    part: str = ""
    part_heading: str = ""
    part_source: str = ""
    part_authority: str = ""
    subpart: str = ""
    subpart_heading: str = ""
    subject_group: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def copy(self, **changes: str) -> _Context:
        data = {k: v for k, v in self.__dict__.items() if k != "extra"}
        data.update(changes)
        return _Context(**data, extra=dict(self.extra))


class ECFRParser:
    """Stream sections out of an eCFR title XML file (or a directory of part files)."""

    def __init__(self, chapters: Iterable[str] | None = None, as_of_date: str = "",
                 title: int = 21) -> None:
        self.chapters = {c.upper() for c in chapters} if chapters else None
        self.as_of_date = as_of_date
        self.title = title
        self.part_ancestry: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ entry points
    def parse_path(self, path: str | Path) -> Iterator[Section]:
        """Parse a title XML file, or a directory holding ``part-*.xml`` and ``structure.json``."""
        path = Path(path)
        if path.is_dir():
            structure = path / "structure.json"
            if structure.exists():
                self.part_ancestry = part_ancestry_from_structure(json.loads(structure.read_text()))
            files = sorted(path.glob("part-*.xml"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
            for f in files:
                yield from self.parse_file(f)
        else:
            yield from self.parse_file(path)

    def parse_file(self, path: str | Path) -> Iterator[Section]:
        parser = etree.XMLParser(huge_tree=True, recover=True, remove_comments=True)
        tree = etree.parse(str(path), parser)
        yield from self.parse_element(tree.getroot())

    def parse_bytes(self, data: bytes) -> list[Section]:
        parser = etree.XMLParser(huge_tree=True, recover=True, remove_comments=True)
        root = etree.fromstring(data, parser)
        return list(self.parse_element(root))

    def parse_element(self, root: etree._Element) -> Iterator[Section]:
        # Wrap the root so a document whose root is itself a typed DIV (e.g. the per-part
        # XML from ``?part=``) is entered like any other hierarchy level.
        wrapper = etree.Element("ROOT")
        wrapper.append(root)
        yield from self._walk(wrapper, _Context(title=self.title))

    # ------------------------------------------------------------------ traversal
    def _walk(self, el: etree._Element, ctx: _Context) -> Iterator[Section]:
        for child in el:
            if not isinstance(child.tag, str):
                continue
            typ = (child.get("TYPE") or "").upper()
            if child.tag.startswith("DIV") and typ in HIERARCHY_TYPES:
                new_ctx = self._enter(child, typ, ctx)
                if new_ctx is None:
                    continue
                yield from self._walk(child, new_ctx)
            elif child.tag.startswith("DIV") and typ in LEAF_TYPES:
                sec = self._section(child, typ, ctx)
                if sec is not None:
                    yield sec
            elif child.tag.startswith("DIV") or child.tag in {"ECFR", "ECFRBRWS"}:
                yield from self._walk(child, ctx)

    def _enter(self, el: etree._Element, typ: str, ctx: _Context) -> _Context | None:
        head = inline_text(el.find("HEAD")) if el.find("HEAD") is not None else ""
        n = (el.get("N") or "").strip()
        if typ == "CHAPTER":
            num, rest = split_head(head, r"CHAPTER")
            chapter = (n or num).upper()
            if self.chapters and chapter not in self.chapters:
                logger.info("Skipping chapter %s (not in configured chapters)", chapter)
                return None
            return ctx.copy(chapter=chapter, chapter_heading=rest, subchapter="",
                            subchapter_heading="", part="", subpart="", subpart_heading="",
                            subject_group="")
        if typ == "SUBCHAP":
            num, rest = split_head(head, r"SUBCHAPTER")
            return ctx.copy(subchapter=n or num, subchapter_heading=rest)
        if typ == "PART":
            num, rest = split_head(head, r"PART")
            part = n or num
            anc = self.part_ancestry.get(part, {})
            if anc and not ctx.chapter:
                if self.chapters and anc.get("chapter", "").upper() not in self.chapters:
                    return None
                ctx = ctx.copy(**anc)
            return ctx.copy(
                part=part, part_heading=rest, subpart="", subpart_heading="", subject_group="",
                part_source=_labelled_block(el, "SOURCE"), part_authority=_labelled_block(el, "AUTH"),
            )
        if typ == "SUBPART":
            num, rest = split_head(head, r"Subpart")
            return ctx.copy(subpart=n or num, subpart_heading=rest, subject_group="")
        if typ == "SUBJGRP":
            return ctx.copy(subject_group=clean(head))
        return ctx

    def _section(self, el: etree._Element, typ: str, ctx: _Context) -> Section | None:
        head_el = el.find("HEAD")
        head = inline_text(head_el) if head_el is not None else ""
        n = (el.get("N") or "").strip()
        if typ == "APPENDIX":
            label = clean(head).split("—")[0].split(" to ")[0].strip() or n
            section_id = f"{ctx.part} {label}".strip()
            heading = clean(head)
        else:
            section_id = n.replace("§", "").strip() or _section_num_from_head(head)
            if not section_id:
                logger.warning("Section without N attribute in part %s: %r", ctx.part, head[:60])
                return None
            heading = section_heading(head, section_id)
        reserved = "[reserved]" in clean(head).lower()
        paragraphs = self._paragraphs(el)
        notes = [block_text(n_el) for tag in NOTE_TAGS for n_el in el.iter(tag)
                 if n_el.getparent() is el]
        cita = " ".join(clean(inline_text(c)) for c in el.findall("CITA"))
        sec = Section(
            section_id=section_id, heading=heading, title=ctx.title, chapter=ctx.chapter,
            chapter_heading=ctx.chapter_heading, subchapter=ctx.subchapter,
            subchapter_heading=ctx.subchapter_heading, part=ctx.part or section_id.split(".")[0],
            part_heading=ctx.part_heading, subpart=ctx.subpart, subpart_heading=ctx.subpart_heading,
            subject_group=ctx.subject_group, kind=typ.lower(), reserved=reserved and not paragraphs,
            paragraphs=paragraphs, notes=[n for n in notes if n], source_citation=cita,
            amendment_dates=_DATE_RE.findall(cita), part_source=ctx.part_source,
            part_authority=ctx.part_authority, as_of_date=self.as_of_date,
        )
        sec.cross_refs = outbound_refs(sec)
        return sec

    def _paragraphs(self, el: etree._Element) -> list[Paragraph]:
        nester = ParagraphNester()
        out: list[Paragraph] = []
        for block in _iter_blocks(el):
            raw = inline_text(block, mark_italic=True) if block.tag != "GPOTABLE" else table_text(block)
            raw = clean(raw)
            if not raw:
                continue
            if block.tag == "HD":
                out.append(Paragraph(path=nester.path, marker="", level=-1, text=strip_italics(raw)))
                continue
            markers, rest = split_leading_markers(raw) if block.tag == "P" else ([], raw)
            level = -1
            for marker, italic in markers:
                level = nester.push(marker, italic)
            term = _defined_term(rest)
            text = strip_italics(raw)
            if markers:
                out.append(Paragraph(path=nester.path, marker=markers[-1][0], level=level,
                                     text=text, term=term))
            elif out and block.tag != "GPOTABLE" and out[-1].marker and not term:
                # Undesignated continuation text belongs to the open paragraph.
                out.append(Paragraph(path=nester.path, marker="", level=-1, text=text))
            else:
                out.append(Paragraph(path=nester.path, marker="", level=-1, text=text, term=term))
        return out


def _iter_blocks(el: etree._Element) -> Iterator[etree._Element]:
    """Yield text-bearing blocks in document order, descending into EXTRACT/NOTE-like wrappers."""
    for child in el:
        if not isinstance(child.tag, str) or child.tag in SKIP_TAGS or child.tag in NOTE_TAGS:
            continue
        if child.tag in BLOCK_TAGS or child.tag == "GPOTABLE":
            yield child
        elif child.tag in {"EXTRACT", "APPRO", "FTNT", "DIV", "SECTION"} or len(child):
            yield from _iter_blocks(child)
        elif (child.text or "").strip():
            yield child


def _labelled_block(el: etree._Element, tag: str) -> str:
    node = el.find(tag)
    if node is None:
        return ""
    text = clean(" ".join(inline_text(c) for c in node if c.tag != "HED"))
    return text or block_text(node)


def _section_num_from_head(head: str) -> str:
    m = re.search(r"§+\s*(\d+\.\d+[a-z]?)", head)
    return m.group(1) if m else ""


def _defined_term(text: str) -> str | None:
    m = _ITALIC_TERM_RE.match(text)
    if m:
        term = m.group(1).strip(" .,:;")
        if 0 < len(term) <= 100 and not term[0].isdigit():
            return term
    m = _QUOTED_TERM_RE.match(strip_italics(text))
    if m:
        return m.group(1).strip(" .,")
    return None


def outbound_refs(sec: Section) -> list[str]:
    """Normalised outbound references (``211.68(b)``, ``part 820``) excluding self-references."""
    refs: list[str] = []
    for cit in extract_citations(sec.text + "\n" + "\n".join(sec.notes)):
        if cit.section == sec.section_id or (not cit.section and cit.part == sec.part):
            continue
        norm = cit.normalized()
        if norm not in refs:
            refs.append(norm)
    return refs


def part_ancestry_from_structure(structure: dict) -> dict[str, dict[str, str]]:
    """Map part number to its chapter/subchapter, from the versioner structure JSON."""
    out: dict[str, dict[str, str]] = {}

    def walk(node: dict, ctx: dict[str, str]) -> None:
        typ = node.get("type", "")
        ctx = dict(ctx)
        if typ == "chapter":
            ctx.update(chapter=str(node.get("identifier", "")),
                       chapter_heading=node.get("label_description", ""))
        elif typ == "subchapter":
            ctx.update(subchapter=str(node.get("identifier", "")),
                       subchapter_heading=node.get("label_description", ""))
        elif typ == "part":
            out[str(node.get("identifier"))] = ctx
            return
        for child in node.get("children", []) or []:
            walk(child, ctx)

    walk(structure, {})
    return out


def parts_from_structure(structure: dict, chapters: Iterable[str] | None = None) -> list[str]:
    """List part identifiers (optionally restricted to ``chapters``) in document order."""
    anc = part_ancestry_from_structure(structure)
    wanted = {c.upper() for c in chapters} if chapters else None
    return [p for p, a in anc.items() if not wanted or a.get("chapter", "").upper() in wanted]
