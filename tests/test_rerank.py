from cfr_rag.config import RerankConfig
from cfr_rag.rerank.base import IdentityReranker, Reranker, build_reranker
from cfr_rag.rerank.qwen3_reranker import PREFIX, SUFFIX, format_pair
from cfr_rag.retrieval.types import Candidate
from cfr_rag.store.sqlite_store import Chunk


class OverlapReranker(Reranker):
    def score(self, query, documents):
        q = set(query.lower().split())
        return [len(q & set(d.lower().split())) / (1 + len(q)) for d in documents]


def cand(i, text, pinned=False):
    c = Chunk(i, f"k{i}", "t", f"1.{i}", [f"1.{i}"], 0, "1", "I", "", "h", text)
    return Candidate(chunk_id=i, chunk=c, pinned=pinned)


def test_rerank_orders_and_cuts():
    r = OverlapReranker(RerankConfig(top_n_in=10, top_n_out=2))
    out = r.rerank("calibration of balances", [cand(1, "nothing here"),
                                               cand(2, "calibration of balances weekly"),
                                               cand(3, "balances")])
    assert [c.chunk_id for c in out] == [2, 3]
    assert out[0].ranks["rerank"] == 1 and "rerank" in out[0].scores
    assert r.last_latency_ms >= 0


def test_pinned_candidates_survive():
    r = OverlapReranker(RerankConfig(top_n_in=10, top_n_out=2))
    cs = [cand(1, "unrelated", pinned=True), cand(2, "calibration"), cand(3, "calibration balances")]
    out = r.rerank("calibration balances", cs)
    assert {c.chunk_id for c in out} == {1, 3}
    out = r.rerank("calibration balances", cs, pin=False)
    assert 1 not in {c.chunk_id for c in out}


def test_top_n_in_limits_scoring():
    r = OverlapReranker(RerankConfig(top_n_in=2, top_n_out=5))
    out = r.rerank("x", [cand(i, "x") for i in range(5)])
    assert len(out) == 2


def test_identity_and_disabled():
    assert build_reranker(RerankConfig(enabled=False)) is None
    out = IdentityReranker(RerankConfig(top_n_out=2)).rerank("q", [cand(1, "a"), cand(2, "b"), cand(3, "c")])
    assert [c.chunk_id for c in out] == [1, 2]


def test_qwen3_prompt_format_matches_model_card():
    assert PREFIX.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert 'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' in PREFIX
    assert SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert format_pair("I", "Q", "D") == "<Instruct>: I\n<Query>: Q\n<Document>: D"
