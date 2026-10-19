from cfr_rag.citations import CFRCitation, extract_citations, parse_citation, section_sort_key


def norm(text: str) -> list[str]:
    return [c.normalized() for c in extract_citations(text)]


def test_full_citation_with_paragraphs():
    assert norm("What does 21 CFR 211.68(b)(1) require?") == ["211.68(b)(1)"]


def test_section_symbol_lists():
    assert norm("see §§ 211.68 and 211.100(a)") == ["211.68", "211.100(a)"]


def test_parts():
    assert norm("parts 210 and 211 apply") == ["part 210", "part 211"]
    assert norm("Give 2.5 mg per 21 CFR part 211") == ["part 211"]


def test_cfr_with_dots_and_symbol():
    assert norm("21 C.F.R. § 11.10(a)") == ["11.10(a)"]


def test_bare_number_without_paragraph_is_ignored():
    assert norm("Take 2.5 mg twice daily") == []


def test_parse_and_format():
    cit = parse_citation("211.68(b)")
    assert cit == CFRCitation(part="211", section="211.68", paragraph="(b)")
    assert cit.formatted() == "21 CFR 211.68(b)"
    assert CFRCitation(part="820").formatted() == "21 CFR Part 820"


def test_sort_key_numeric():
    ids = ["1.10", "1.2", "2.1", "1.1"]
    assert sorted(ids, key=section_sort_key) == ["1.1", "1.2", "1.10", "2.1"]
