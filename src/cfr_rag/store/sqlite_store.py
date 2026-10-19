"""SQLite store for sections, chunks and definitions.

``chunks.chunk_id`` is the integer ID shared with the FAISS and BM25 indexes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cfr_rag.citations import section_sort_key
from cfr_rag.ingest.models import Paragraph, Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    part TEXT, chapter TEXT, subchapter TEXT, subpart TEXT,
    heading TEXT, kind TEXT, reserved INTEGER, is_definitions INTEGER,
    content_hash TEXT, as_of_date TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS idx_sections_part ON sections(part);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    chunk_key TEXT UNIQUE,
    chunker TEXT,
    section_id TEXT,
    section_ids TEXT,
    ordinal INTEGER,
    part TEXT, chapter TEXT, subpart TEXT,
    header TEXT, text TEXT,
    paragraph_start TEXT, paragraph_end TEXT,
    token_count INTEGER,
    is_definition INTEGER, term TEXT,
    content_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_part ON chunks(part);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter);
CREATE TABLE IF NOT EXISTS definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT, term_lower TEXT, section_id TEXT, part TEXT, chapter TEXT,
    paragraph TEXT, text TEXT
);
CREATE INDEX IF NOT EXISTS idx_definitions_term ON definitions(term_lower);
"""


@dataclass
class Chunk:
    """A retrievable unit of text; ``embed_text`` is what the embedder sees."""

    chunk_id: int
    chunk_key: str
    chunker: str
    section_id: str
    section_ids: list[str]
    ordinal: int
    part: str
    chapter: str
    subpart: str
    header: str
    text: str
    paragraph_start: str = ""
    paragraph_end: str = ""
    token_count: int = 0
    is_definition: bool = False
    term: str | None = None
    content_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Definition:
    term: str
    section_id: str
    part: str
    chapter: str
    paragraph: str
    text: str


class SQLiteStore:
    """All persistent corpus data. Safe to open from multiple threads read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                          (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ------------------------------------------------------------------ sections
    def upsert_sections(self, sections: Iterable[Section]) -> int:
        rows = [
            (s.section_id, s.part, s.chapter, s.subchapter, s.subpart, s.heading, s.kind,
             int(s.reserved), int(s.is_definitions), s.content_hash, s.as_of_date,
             json.dumps(s.to_dict()))
            for s in sections
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def replace_sections(self, sections: Iterable[Section]) -> int:
        self.conn.execute("DELETE FROM sections")
        return self.upsert_sections(sections)

    def delete_sections(self, section_ids: Iterable[str]) -> None:
        self.conn.executemany("DELETE FROM sections WHERE section_id = ?", [(s,) for s in section_ids])
        self.conn.commit()

    def get_section(self, section_id: str) -> Section | None:
        row = self.conn.execute("SELECT data FROM sections WHERE section_id = ?",
                                (section_id,)).fetchone()
        return _section_from_row(row["data"]) if row else None

    def iter_sections(self, part: str | None = None) -> Iterator[Section]:
        sql, args = "SELECT data FROM sections", ()
        if part:
            sql, args = sql + " WHERE part = ?", (part,)
        rows = self.conn.execute(sql, args).fetchall()
        secs = [_section_from_row(r["data"]) for r in rows]
        secs.sort(key=lambda s: section_sort_key(s.section_id))
        yield from secs

    def section_hashes(self) -> dict[str, str]:
        return {r["section_id"]: r["content_hash"]
                for r in self.conn.execute("SELECT section_id, content_hash FROM sections")}

    def count_sections(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]

    def section_exists(self, section_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sections WHERE section_id = ?",
                                 (section_id,)).fetchone() is not None

    def parts(self) -> list[tuple[str, str]]:
        """Distinct ``(chapter, part)`` pairs."""
        rows = self.conn.execute("SELECT DISTINCT chapter, part FROM sections").fetchall()
        return sorted(((r["chapter"], r["part"]) for r in rows), key=lambda x: int(x[1] or 0))

    # ------------------------------------------------------------------ chunks
    def replace_chunks(self, chunks: Iterable[Chunk], chunker: str) -> int:
        self.conn.execute("DELETE FROM chunks WHERE chunker = ?", (chunker,))
        rows = [
            (c.chunk_id, c.chunk_key, c.chunker, c.section_id, json.dumps(c.section_ids), c.ordinal,
             c.part, c.chapter, c.subpart, c.header, c.text, c.paragraph_start, c.paragraph_end,
             c.token_count, int(c.is_definition), c.term, c.content_hash)
            for c in chunks
        ]
        self.conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def next_chunk_id(self) -> int:
        row = self.conn.execute("SELECT MAX(chunk_id) FROM chunks").fetchone()
        return int(row[0] or 0) + 1

    def get_chunks(self, chunk_ids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(chunk_ids)
        out: dict[int, Chunk] = {}
        for i in range(0, len(ids), 900):
            batch = ids[i:i + 900]
            q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})"
            for row in self.conn.execute(q, batch):
                c = _chunk_from_row(row)
                out[c.chunk_id] = c
        return out

    def iter_chunks(self, chunker: str) -> Iterator[Chunk]:
        for row in self.conn.execute("SELECT * FROM chunks WHERE chunker = ? ORDER BY chunk_id",
                                     (chunker,)):
            yield _chunk_from_row(row)

    def chunks_for_section(self, section_id: str, chunker: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE chunker = ? AND (section_id = ? OR section_ids LIKE ?) "
            "ORDER BY ordinal", (chunker, section_id, f'%"{section_id}"%')).fetchall()
        return [c for c in map(_chunk_from_row, rows) if section_id in c.section_ids]

    def chunk_ids_matching(self, chunker: str, chapters: list[str] | None = None,
                           parts: list[str] | None = None) -> list[int]:
        """Chunk IDs whose chapter/part matches the filters (used for pre-filtering)."""
        sql = "SELECT chunk_id FROM chunks WHERE chunker = ?"
        args: list[Any] = [chunker]
        if chapters:
            sql += f" AND chapter IN ({','.join('?' * len(chapters))})"
            args += [c.upper() for c in chapters]
        if parts:
            sql += f" AND part IN ({','.join('?' * len(parts))})"
            args += [str(p) for p in parts]
        return [r[0] for r in self.conn.execute(sql, args)]

    def count_chunks(self, chunker: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks WHERE chunker = ?",
                                 (chunker,)).fetchone()[0]

    # ------------------------------------------------------------------ definitions
    def replace_definitions(self, defs: Iterable[Definition]) -> int:
        self.conn.execute("DELETE FROM definitions")
        rows = [(d.term, d.term.lower(), d.section_id, d.part, d.chapter, d.paragraph, d.text)
                for d in defs]
        self.conn.executemany(
            "INSERT INTO definitions(term, term_lower, section_id, part, chapter, paragraph, text) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def all_definitions(self) -> list[Definition]:
        return [Definition(r["term"], r["section_id"], r["part"], r["chapter"], r["paragraph"],
                           r["text"])
                for r in self.conn.execute("SELECT * FROM definitions ORDER BY id")]


def _section_from_row(data: str) -> Section:
    d = json.loads(data)
    d["paragraphs"] = [Paragraph(**p) for p in d.get("paragraphs", [])]
    return Section(**d)


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"], chunk_key=row["chunk_key"], chunker=row["chunker"],
        section_id=row["section_id"], section_ids=json.loads(row["section_ids"]),
        ordinal=row["ordinal"], part=row["part"], chapter=row["chapter"], subpart=row["subpart"],
        header=row["header"], text=row["text"], paragraph_start=row["paragraph_start"],
        paragraph_end=row["paragraph_end"], token_count=row["token_count"],
        is_definition=bool(row["is_definition"]), term=row["term"],
        content_hash=row["content_hash"],
    )
