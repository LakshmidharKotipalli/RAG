from conftest import FIXTURE_XML

from cfr_rag.ingest.paragraphs import ParagraphNester, split_leading_markers
from cfr_rag.ingest.parser import ECFRParser
from cfr_rag.ingest.service import mark_definitions


def sections():
    return {s.section_id: s for s in ECFRParser(as_of_date="2026-10-01").parse_path(FIXTURE_XML)}


def test_hierarchy_fields():
    s = sections()["211.68"]
    assert (s.chapter, s.subchapter, s.part, s.subpart) == ("I", "C", "211", "D")
    assert s.heading == "Automatic, mechanical, and electronic equipment"
    assert s.part_heading.startswith("CURRENT GOOD MANUFACTURING PRACTICE FOR FINISHED")
    assert s.subpart_heading == "Equipment"
    assert s.citation == "21 CFR 211.68"
    assert "43 FR 45077" in s.part_source
    assert s.amendment_dates[-1] == "Sept. 8, 2008"


def test_all_chapters_parsed_and_filter_works():
    chapters = {s.chapter for s in sections().values()}
    assert chapters == {"I", "II", "III"}
    only_ii = list(ECFRParser(chapters=["II"]).parse_path(FIXTURE_XML))
    assert {s.chapter for s in only_ii} == {"II"}


def test_cross_references():
    s = sections()["211.68"]
    assert s.cross_refs == ["211.101(c)", "211.103", "211.182", "211.188(b)(11)"]
    assert "210.3" in sections()["211.3"].cross_refs


def test_paragraph_nesting():
    paths = [p.path for p in sections()["211.69"].paragraphs if p.marker]
    assert paths[:7] == ["(a)", "(a)(1)", "(a)(1)(i)", "(a)(1)(ii)", "(a)(1)(ii)(A)",
                         "(a)(1)(ii)(A)(1)", "(a)(2)"]
    assert "(i)" in paths  # letter (i) after (h), not a roman numeral
    assert "(j)(1)" in paths


def test_tables_notes_reserved_appendix():
    secs = sections()
    s = secs["211.69"]
    assert any("Balances | 6" in p.text for p in s.paragraphs)
    assert s.notes and s.notes[0].startswith("Effective Date Note: At 89 FR")
    assert secs["211.64"].reserved
    assert "101 Appendix A" in secs and secs["101 Appendix A"].kind == "appendix"


def test_defined_terms_and_definition_sections():
    secs = sections()
    terms = [p.term for p in secs["210.3"].paragraphs if p.term]
    assert terms == ["Act", "Batch", "Component", "Drug product"]
    assert [p.term for p in secs["1300.01"].paragraphs if p.term] == [
        "Administration", "Controlled substance"]
    marked = mark_definitions(secs.values(), r"(?i)\bdefinitions?\b")
    flagged = {s.section_id for s in marked if s.is_definitions}
    assert flagged == {"210.3", "211.3", "1300.01"}


def test_nester_roman_vs_letter():
    n = ParagraphNester()
    assert [n.push(m) for m in ["a", "1", "i", "ii", "2", "b"]] == [0, 1, 2, 2, 1, 0]
    n = ParagraphNester()
    for m in "abcdefgh":
        n.push(m)
    assert n.push("i") == 0 and n.path == "(i)"


def test_split_leading_markers():
    markers, rest = split_leading_markers("(b)(1) Text here")
    assert markers == [("b", False), ("1", False)] and rest == "Text here"


def test_part_level_document_with_structure_ancestry(tmp_path):
    import json

    from lxml import etree

    root = etree.parse(str(FIXTURE_XML)).getroot()
    part = next(d for d in root.iter("DIV5") if d.get("N") == "211")
    (tmp_path / "part-211.xml").write_bytes(etree.tostring(part))
    (tmp_path / "structure.json").write_text(json.dumps({"type": "title", "children": [
        {"type": "chapter", "identifier": "I", "label_description": "FDA", "children": [
            {"type": "subchapter", "identifier": "C", "label_description": "Drugs", "children": [
                {"type": "part", "identifier": "211"}]}]}]}))
    secs = {s.section_id: s for s in ECFRParser(chapters=["I"]).parse_path(tmp_path)}
    s = secs["211.68"]
    assert (s.chapter, s.subchapter, s.part, s.subpart) == ("I", "C", "211", "D")
    assert s.part_heading.startswith("CURRENT GOOD")
