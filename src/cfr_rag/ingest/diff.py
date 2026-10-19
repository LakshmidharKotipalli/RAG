"""Compare two snapshots of the title and report changed sections."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field

from cfr_rag.citations import section_sort_key
from cfr_rag.ingest.models import Section


@dataclass
class SnapshotDiff:
    old_date: str
    new_date: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)  # section_id -> unified diff excerpt

    @property
    def changed(self) -> list[str]:
        return sorted(set(self.added) | set(self.modified), key=section_sort_key)

    @property
    def changed_parts(self) -> list[str]:
        ids = self.added + self.removed + self.modified
        return sorted({s.split(".")[0].split(" ")[0] for s in ids}, key=lambda p: int(p or 0))

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def to_markdown(self, max_details: int = 25, api_versions: list[dict] | None = None) -> str:
        """Render a human-readable change report."""
        lines = [f"# 21 CFR changes: {self.old_date} → {self.new_date}", "",
                 f"- Added: **{len(self.added)}**", f"- Removed: **{len(self.removed)}**",
                 f"- Modified: **{len(self.modified)}**",
                 f"- Parts affected: {', '.join(self.changed_parts) or 'none'}", ""]
        for label, ids in (("Added", self.added), ("Removed", self.removed),
                           ("Modified", self.modified)):
            if ids:
                lines += [f"## {label}", "", ", ".join(f"§ {s}" for s in ids), ""]
        if self.details:
            lines += ["## Text changes (excerpts)", ""]
            for sid in list(self.details)[:max_details]:
                lines += [f"### § {sid}", "", "```diff", self.details[sid], "```", ""]
        if api_versions:
            lines += ["## eCFR versioner change records", "",
                      "| Section | Issue date | Amendment date | Name | Removed |",
                      "|---|---|---|---|---|"]
            for v in api_versions[:500]:
                lines.append(f"| {v.get('identifier', '')} | {v.get('issue_date', '')} | "
                             f"{v.get('amendment_date', '')} | {v.get('name', '')} | "
                             f"{v.get('removed', '')} |")
        return "\n".join(lines) + "\n"


def diff_snapshots(old: Iterable[Section], new: Iterable[Section], old_date: str = "",
                   new_date: str = "", context_lines: int = 1) -> SnapshotDiff:
    """Classify each section as added, removed or modified (by content hash)."""
    old_map = {s.section_id: s for s in old}
    new_map = {s.section_id: s for s in new}
    out = SnapshotDiff(old_date, new_date)
    out.added = sorted(set(new_map) - set(old_map), key=section_sort_key)
    out.removed = sorted(set(old_map) - set(new_map), key=section_sort_key)
    for sid in sorted(set(old_map) & set(new_map), key=section_sort_key):
        a, b = old_map[sid], new_map[sid]
        if a.content_hash != b.content_hash:
            out.modified.append(sid)
            udiff = difflib.unified_diff(a.full_text().splitlines(), b.full_text().splitlines(),
                                         lineterm="", n=context_lines)
            out.details[sid] = "\n".join(list(udiff)[2:40])
    return out


def diff_hashes(old: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Added, removed, modified section IDs given ``section_id -> content_hash`` maps."""
    added = sorted(set(new) - set(old), key=section_sort_key)
    removed = sorted(set(old) - set(new), key=section_sort_key)
    modified = sorted((s for s in set(old) & set(new) if old[s] != new[s]), key=section_sort_key)
    return added, removed, modified
