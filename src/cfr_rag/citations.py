"""Parsing, normalising and linking CFR citations.

Recognised forms (case-insensitive):

* ``21 CFR 211.68(b)(1)``, ``21 C.F.R. § 211.68``, ``21 CFR part 211``
* ``§ 211.68``, ``§§ 211.68 and 211.100``, ``section 211.68(a)``
* ``part 211``, ``parts 210 and 211``
* bare ``211.68(b)`` (only when a paragraph marker follows, to avoid matching doses)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_NUM = r"\d{1,4}\.\d{1,4}[a-z]?"
PARAGRAPH = r"(?:\((?:[a-z]{1,4}|\d{1,3}|[A-Z]{1,3})\))+"

_LIST_SEP = r"\s*(?:,\s*(?:and|or|through)?|\band\b|\bor\b|\bthrough\b|\bto\b|-|–)\s*"
_SECTION_RE = re.compile(rf"(?P<section>{SECTION_NUM})(?P<para>{PARAGRAPH})?")
_PREFIXED_SECTION_RE = re.compile(
    rf"(?:\b21\s*C\.?\s*F\.?\s*R\.?\s*(?:§§?|sections?|sec\.)?\s*|§§?\s*|\bsections?\s+|\bsec\.\s*)"
    rf"(?P<list>{SECTION_NUM}(?:{PARAGRAPH})?"
    rf"(?:{_LIST_SEP}(?:§\s*)?(?:{SECTION_NUM}(?:{PARAGRAPH})?|{PARAGRAPH}))*)",
    re.IGNORECASE,
)
_BARE_WITH_PARA_RE = re.compile(rf"(?<![\w.]){SECTION_NUM}{PARAGRAPH}")
_PART_RE = re.compile(
    rf"(?:\b21\s*C\.?\s*F\.?\s*R\.?\s*)?\bparts?\s+(?P<list>\d{{1,4}}(?:{_LIST_SEP}\d{{1,4}})*)\b"
    r"(?!\.\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CFRCitation:
    """A citation to a section (optionally a paragraph) or to a whole part."""

    part: str
    section: str | None = None
    paragraph: str = ""

    @property
    def section_id(self) -> str | None:
        return self.section

    def normalized(self) -> str:
        """Canonical short form, e.g. ``211.68(b)(1)`` or ``part 211``."""
        if self.section:
            return f"{self.section}{self.paragraph}"
        return f"part {self.part}"

    def formatted(self, title: int = 21) -> str:
        """Human citation, e.g. ``21 CFR 211.68(b)(1)`` or ``21 CFR Part 211``."""
        if self.section:
            return f"{title} CFR {self.section}{self.paragraph}"
        return f"{title} CFR Part {self.part}"


def _from_section_token(token: str) -> CFRCitation | None:
    m = _SECTION_RE.fullmatch(token.strip())
    if not m:
        return None
    section = m.group("section")
    return CFRCitation(part=section.split(".")[0], section=section, paragraph=m.group("para") or "")


def extract_citations(text: str) -> list[CFRCitation]:
    """Return all section/part citations found in ``text``, in order, de-duplicated."""
    found: list[tuple[int, CFRCitation]] = []
    spans: list[tuple[int, int]] = []
    for m in _PREFIXED_SECTION_RE.finditer(text):
        spans.append(m.span())
        for sm in _SECTION_RE.finditer(m.group("list")):
            cit = _from_section_token(sm.group(0))
            if cit:
                found.append((m.start() + sm.start(), cit))
    for m in _BARE_WITH_PARA_RE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        cit = _from_section_token(m.group(0))
        if cit:
            found.append((m.start(), cit))
    for m in _PART_RE.finditer(text):
        for num in re.findall(r"\d{1,4}", m.group("list")):
            found.append((m.start(), CFRCitation(part=num)))
    found.sort(key=lambda x: x[0])
    seen: set[CFRCitation] = set()
    out: list[CFRCitation] = []
    for _, cit in found:
        if cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out


def parse_citation(text: str) -> CFRCitation | None:
    """Parse a single citation string such as ``211.68(b)`` or ``21 CFR 211.68``."""
    direct = _from_section_token(text.replace("§", "").strip())
    if direct:
        return direct
    cits = extract_citations(text)
    return cits[0] if cits else None


def section_sort_key(section_id: str) -> tuple[int, int, str]:
    """Sort key ordering ``1.1 < 1.2 < 1.10 < 2.1``; appendix IDs sort after sections."""
    m = re.match(r"(\d+)\.(\d+)([a-z]?)", section_id)
    if not m:
        part = re.match(r"(\d+)", section_id)
        return (int(part.group(1)) if part else 0, 10**6, section_id)
    return (int(m.group(1)), int(m.group(2)), m.group(3))


def ecfr_section_url(template: str, section: str, paragraph: str = "", title: int = 21) -> str:
    """Build an eCFR link for a section, optionally anchored at a paragraph."""
    return template.format(title=title, section=section, paragraph=paragraph)
