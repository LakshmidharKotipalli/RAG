from pathlib import Path

from conftest import build_offline_pipeline

from cfr_rag.pipeline import QueryRequest


def test_query_returns_structured_answer_with_citations(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    resp = pipe.query(QueryRequest(question="What does 21 CFR 211.68 require for backups?"))
    assert resp.provider == "openrouter" and resp.model == "inkling"
    first = resp.citations[0]
    assert first.section_id == "211.68" and first.citation == "21 CFR 211.68(a)"
    assert first.in_context and first.exists
    assert first.url == "https://www.ecfr.gov/current/title-21/section-211.68#p-211.68(a)"
    assert first.snippet.startswith("(a) Automatic")
    bogus = next(c for c in resp.citations if c.section_id == "999.1")
    assert not bogus.exists and not bogus.in_context
    d = resp.debug
    assert d["route"] == "exact+hybrid" and d["final_candidates"]
    assert {"retrieve_total_ms", "total_ms"} <= set(d["timings_ms"])
    assert d["index"]["as_of_date"] == "2026-10-01"


def test_provider_override_per_request(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    resp = pipe.query(QueryRequest(question="equipment calibration", provider="ollama",
                                   model="llama3.1:8b"))
    assert resp.provider == "ollama" and resp.model == "llama3.1:8b"


def test_refusal_detected(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    resp = pipe.query(QueryRequest(question="unanswerable: FAA pilot medical rules?"))
    assert resp.refused and resp.citations == []


def test_streaming_events_and_think_stripped(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    events = list(pipe.stream(QueryRequest(question="What does 21 CFR 211.63 require?")))
    assert events[0]["type"] == "retrieval"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "<think>" not in tokens and "21 CFR 211.63" in tokens
    done = events[-1]
    assert done["type"] == "done" and done["result"]["citations"][0]["section_id"] == "211.63"


def test_definitions_pulled_in(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    res, final, _ = pipe.retrieve("Who may change master production records for a batch?",
                                  use_router=False)
    blocks = pipe.generator.context_builder.build("What is a batch under cGMP?", final[:2])
    assert any(b.kind == "definition" and "Batch" in b.text for b in blocks)


def test_parent_expansion(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path, chunking={"max_tokens": 120, "overlap_tokens": 10})
    _, final, _ = pipe.retrieve("21 CFR 211.68(b)")
    blocks = pipe.generator.context_builder.build("21 CFR 211.68(b)", final)
    full = [b for b in blocks if b.section_id == "211.68"]
    assert len(full) == 1 and full[0].kind == "section" and "(c) Such automated" in full[0].text


def test_error_event_when_provider_fails(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    events = list(pipe.stream(QueryRequest(question="x", provider="ollama", model="m")))
    assert events[-1]["type"] in ("done", "error")
    pipe.llm.cfg.providers.pop("ollama")
    pipe.llm._providers.clear()
    events = list(pipe.stream(QueryRequest(question="x", provider="ollama")))
    assert events[-1]["type"] == "error" and "Unknown provider" in events[-1]["message"]
