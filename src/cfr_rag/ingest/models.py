"""Data models for parsed regulation text."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Paragraph:
    """One designated (or undesignated) paragraph within a section.

    Attributes:
        path: Full designation, e.g. ``(b)(1)(ii)``; empty for undesignated intro text.
        marker: The last marker only, e.g. ``ii``.
        level: Nesting depth (0 = (a), 1 = (1), 2 = (i), 3 = (A), 4 = (1) italic, 5 = (i) italic).
        text: Paragraph text including its leading markers.
        term: A defined term if the paragraph opens with an italicised or quoted term.
    """

    path: str
    marker: str
    level: int
    text: str
    term: str | None = None


@dataclass
class Section:
    """A CFR section (or appendix) with its full hierarchy context."""

    section_id: str
    heading: str
    title: int = 21
    chapter: str = ""
    chapter_heading: str = ""
    subchapter: str = ""
    subchapter_heading: str = ""
    part: str = ""
    part_heading: str = ""
    subpart: str = ""
    subpart_heading: str = ""
    subject_group: str = ""
    kind: str = "section"  # section | appendix
    reserved: bool = False
    paragraphs: list[Paragraph] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_citation: str = ""  # section-level [CITA] amendment history
    amendment_dates: list[str] = field(default_factory=list)
    part_source: str = ""
    part_authority: str = ""
    cross_refs: list[str] = field(default_factory=list)
    as_of_date: str = ""
    is_definitions: bool = False

    @property
    def citation(self) -> str:
        if self.kind == "appendix":
            return f"{self.title} CFR Part {self.part}, {self.section_id.split(' ', 1)[-1]}"
        return f"{self.title} CFR {self.section_id}"

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    def full_text(self, include_notes: bool = True) -> str:
        """Section body, optionally followed by effective-date/editorial notes."""
        body = self.text
        if include_notes and self.notes:
            body += "\n" + "\n".join(f"Note: {n}" for n in self.notes)
        return body

    @property
    def content_hash(self) -> str:
        """Hash of the regulatory content, used to detect changed sections."""
        blob = "\x1f".join([self.heading, self.text, "\x1e".join(self.notes)]).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        data = dict(data)
        data["paragraphs"] = [Paragraph(**p) for p in data.get("paragraphs", [])]
        return cls(**data)
