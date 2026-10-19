from pathlib import Path

from conftest import FIXTURE_XML

from cfr_rag.chunking.base import extract_definitions
from cfr_rag.chunking.fixed import FixedSizeChunker
from cfr_rag.chunking.service import chunk_corpus, chunker_id
from cfr_rag.chunking.structural import StructuralChunker
from cfr_rag.chunking.tokens import ApproxCounter
from cfr_rag.config import load_config
from cfr_rag.ingest.parser import ECFRParser
from cfr_rag.ingest.service import ingest, mark_definitions
from cfr_rag.store.sqlite_store import SQLiteStore


def secs():
    return mark_definitions(ECFRParser().parse_path(FIXTURE_XML), r"(?i)\bdefinitions?\b")


def cfg(**chunking):
    return load_config("config/default.yaml", overrides={"chunking": {"tokenizer": "approx", **chunking}})


def test_header_and_whole_section():
    chunks = StructuralChunker(cfg().chunking, ApproxCounter()).chunk(secs())
    c = next(c for c in chunks if c.section_id == "211.68")
    assert c.header == "21 CFR Part 211 > Subpart D > 211.68 Automatic, mechanical, and electronic equipment"
    assert c.embed_text.startswith(c.header + "\n")


def test_long_section_split_at_paragraphs_with_overlap():
    c = cfg(max_tokens=120, overlap_tokens=10).chunking
    chunks = [x for x in StructuralChunker(c, ApproxCounter()).chunk(secs()) if x.section_id == "211.68"]
    assert len(chunks) >= 2
    assert chunks[0].paragraph_start == "(a)"
    assert chunks[1].text.startswith("[...] ")
    assert all(x.section_ids == ["211.68"] for x in chunks)
    assert "(b)" in chunks[1].header or "(c)" in chunks[1].header


def test_tiny_sections_merge_within_subpart():
    c = cfg(min_tokens=60, merge_max_tokens=400).chunking
    chunks = StructuralChunker(c, ApproxCounter()).chunk(secs())
    merged = [x for x in chunks if len(x.section_ids) > 1]
    assert merged, "expected at least one merged chunk"
    for m in merged:
        parts = {s.split(".")[0] for s in m.section_ids}
        assert len(parts) == 1
    assert all("211.64" not in x.section_ids for x in chunks)  # reserved skipped


def test_definitions_chunked_per_term():
    chunks = StructuralChunker(cfg().chunking, ApproxCounter()).chunk(secs())
    defs = [x for x in chunks if x.is_definition]
    assert {x.term for x in defs} >= {"Batch", "Component", "Controlled substance"}
    batch = next(x for x in defs if x.term == "Batch")
    assert batch.section_id == "210.3" and batch.paragraph_start == "(b)(2)"
    assert batch.header.endswith("> Definition: Batch")
    scope = [x for x in chunks if x.section_id == "210.3" and not x.is_definition]
    assert scope and "apply to parts 211 through 226" in scope[0].text


def test_extract_definitions_text_boundaries():
    s = next(x for x in secs() if x.section_id == "210.3")
    d = {x.term: x for x in extract_definitions(s)}
    assert d["Batch"].text.startswith("(2) Batch means") and "Component" not in d["Batch"].text


def test_fixed_chunker_windows_and_parents():
    c = cfg(fixed={"size_tokens": 80, "overlap_tokens": 10}).chunking
    chunks = FixedSizeChunker(c, ApproxCounter()).chunk(secs())
    assert len(chunks) > 5
    assert all(len({s.split('.')[0].split(' ')[0] for s in x.section_ids}) == 1 for x in chunks)
    assert any(len(x.section_ids) > 1 for x in chunks)
    assert all(x.section_id in x.section_ids for x in chunks)


def test_chunk_corpus_ids_coexist(tmp_path: Path):
    base = load_config("config/default.yaml", overrides={
        "paths": {"db_path": str(tmp_path / "db.sqlite")}, "chunking": {"tokenizer": "approx"}})
    ingest(base, xml_path=FIXTURE_XML)
    a = chunk_corpus(base)
    fixed = base.with_overrides({"chunking": {"chunker": "fixed"}})
    b = chunk_corpus(fixed)
    assert a != b and a == chunker_id(base)
    store = SQLiteStore(base.paths.db_path)
    ids_a = {c.chunk_id for c in store.iter_chunks(a)}
    ids_b = {c.chunk_id for c in store.iter_chunks(b)}
    assert ids_a and ids_b and not (ids_a & ids_b)
    assert store.all_definitions()
    assert store.chunks_for_section("211.68", a)
