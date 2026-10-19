from pathlib import Path

from conftest import FIXTURE_XML, offline_cfg

from cfr_rag.index.builder import build_indexes, load_indexes
from cfr_rag.ingest.service import ingest
from cfr_rag.llm.base import LLMResponse
from cfr_rag.retrieval.fusion import reciprocal_rank_fusion
from cfr_rag.retrieval.hybrid import Retriever
from cfr_rag.retrieval.query_transform import QueryTransformer, parse_query_list
from cfr_rag.retrieval.router import route_query
from cfr_rag.retrieval.types import Filters


def retriever(tmp_path: Path, **over) -> Retriever:
    cfg = offline_cfg(tmp_path, **over)
    ingest(cfg, xml_path=FIXTURE_XML)
    build_indexes(cfg)
    return Retriever(load_indexes(cfg), cfg.retrieval)


def test_rrf_weights_and_ties():
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3], "bm25": [3, 2, 9]},
                                   {"dense": 1.0, "bm25": 1.0}, k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] in (2, 3) and set(ids) == {1, 2, 3, 9}
    only_dense = reciprocal_rank_fusion({"dense": [1, 2], "bm25": [2, 1]},
                                        {"dense": 1.0, "bm25": 0.0}, k=60)
    assert [c for c, _ in only_dense] == [1, 2]
    multi = reciprocal_rank_fusion({"dense:0": [5], "dense:1": [5]}, {"dense": 2.0}, k=0)
    assert multi == [(5, 4.0)]


def test_router_detects_sections_and_parts():
    r = route_query("Under 21 CFR 211.68(b), and part 820, what backups are needed?")
    assert [c.normalized() for c in r.sections] == ["211.68(b)"]
    assert r.parts == ["820"]


def test_exact_lookup_pinned_first(tmp_path: Path):
    r = retriever(tmp_path)
    res = r.retrieve("What does 21 CFR 211.63 say about equipment?")
    assert res.route == "exact+hybrid"
    assert res.candidates[0].section_id == "211.63" and res.candidates[0].pinned
    assert res.detected_citations == ["211.63"]


def test_missing_citation_reported(tmp_path: Path):
    res = retriever(tmp_path).retrieve("Explain 21 CFR 999.99")
    assert res.missing_citations == ["999.99"]


def test_hybrid_and_single_modes(tmp_path: Path):
    r = retriever(tmp_path)
    q = "backup file of data entered into the computer"
    for mode in ("dense", "bm25", "hybrid"):
        res = r.retrieve(q, mode=mode, use_router=False)
        assert res.candidates, mode
        assert any(c.section_id == "211.68" for c in res.candidates[:3]), mode
    hy = r.retrieve(q, use_router=False)
    top = hy.candidates[0]
    assert "rrf" in top.scores and set(top.sources) <= {"dense", "bm25"}


def test_filters_restrict_results(tmp_path: Path):
    r = retriever(tmp_path)
    res = r.retrieve("drug definitions", filters=Filters(chapters=["II"]), use_router=False)
    assert res.candidates and all(c.chunk.chapter == "II" for c in res.candidates)
    res = r.retrieve("What is in part 101 about display panels?")
    assert res.filters.parts == ["101"]
    assert all(c.chunk.part == "101" for c in res.candidates)


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, role, messages, provider=None, model=None, json_mode=False, **kw):
        self.calls.append((role, provider, model))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(reply, provider or "fake", model or "m", 1.0)


def test_query_transform_rewrite_and_multi():
    llm = FakeLLM(["cGMP computer system backup requirements",
                   '["electronic records controls", "automated equipment calibration"]'])
    qt = QueryTransformer(llm, rewrite=True, multi_query=True, num_queries=2)
    qs = qt.transform("backup rules?", provider="ollama")
    assert qs[0] == "cGMP computer system backup requirements" and "backup rules?" in qs
    assert len(qs) == 4 and llm.calls[0][1] == "ollama"


def test_query_transform_falls_back_on_error():
    qt = QueryTransformer(FakeLLM([RuntimeError("down"), RuntimeError("down")]), True, True)
    assert qt.transform("q") == ["q"]


def test_parse_query_list_formats():
    assert parse_query_list('{"queries": ["a", "b"]}') == ["a", "b"]
    assert parse_query_list("1. first\n- second") == ["first", "second"]
    assert parse_query_list('<think>x</think>["z"]') == ["z"]


def test_retriever_uses_transformer(tmp_path: Path):
    r = retriever(tmp_path)
    r.transformer = QueryTransformer(FakeLLM(['["calibration of balances"]']), False, True, 1)
    res = r.retrieve("equipment checks", use_router=False)
    assert res.queries == ["equipment checks", "calibration of balances"]
    assert "query_transform_ms" in res.timings_ms
