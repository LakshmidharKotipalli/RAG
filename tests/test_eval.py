import json
from pathlib import Path

import pytest
from conftest import build_offline_pipeline

from cfr_rag.eval.generation_metrics import citing_sentences, summarize
from cfr_rag.eval.goldgen import build_gold_set, load_gold
from cfr_rag.eval.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
    section_ranking,
)
from cfr_rag.eval.runner import run_eval


def test_section_ranking_dedupes():
    assert section_ranking([["1.1"], ["1.2", "1.1"], ["1.3"]]) == ["1.1", "1.2", "1.3"]


def test_metrics_values():
    ranking = ["a", "b", "c", "d"]
    assert recall_at_k(ranking, {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(ranking, {"c"}) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranking, {"a"}, 10) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(0.6309, rel=1e-3)
    m = score_query(ranking, ["d"], [1, 5], 10)
    assert m["recall@1"] == 0 and m["recall@5"] == 1 and "ndcg@10" in m


def test_citing_sentences():
    ans = "Records are required [21 CFR 211.68(a)]. Other text. Backups too [21 CFR 211.68(b)]."
    assert "Other text" not in citing_sentences(ans, "211.68")


def test_summarize_refusal_rates():
    rows = [
        {"answerable": False, "refused": True}, {"answerable": False, "refused": False},
        {"answerable": True, "refused": False, "faithfulness": 1.0, "citation_accuracy": 0.5,
         "citations_total": 2, "citations_exist": 2, "relevance": 1.0, "cited_expected": True},
        {"answerable": True, "refused": True, "relevance": 0.0},
    ]
    s = summarize(rows)
    assert s["correct_refusal_rate"] == 0.5 and s["false_refusal_rate"] == 0.5
    assert s["faithfulness"] == 1.0 and s["citation_accuracy"] == 0.5


def _small_gold_cfg(tmp_path: Path, pipe):
    return pipe.cfg.with_overrides({
        "paths": {"reports_dir": str(tmp_path / "reports")},
        "eval": {"gold_path": str(tmp_path / "gold.jsonl"),
                 "goldgen": {"citation": 4, "topical": 3, "definitional": 3, "multi_section": 2,
                             "unanswerable": 3}}})


def test_gold_set_generation_template_and_llm(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    cfg = _small_gold_cfg(tmp_path, pipe)
    path = build_gold_set(cfg, use_llm=False)
    gold = load_gold(path)
    cats = {g["category"] for g in gold}
    assert cats == {"citation", "topical", "definitional", "multi_section", "unanswerable"}
    assert all(g["expected_sections"] for g in gold if g["answerable"])
    assert all(not g["expected_sections"] for g in gold if not g["answerable"])
    assert len({g["id"] for g in gold}) == len(gold)
    defin = [g for g in gold if g["category"] == "definitional"]
    assert all(set(g["expected_sections"]) <= {"210.3", "1300.01"} for g in defin)
    with pytest.raises(FileExistsError):
        build_gold_set(cfg, use_llm=False)
    gold_llm = load_gold(build_gold_set(cfg, use_llm=True, overwrite=True))
    assert any(g["generated_by"] == "llm" for g in gold_llm)


def test_full_eval_writes_report(tmp_path: Path):
    pipe = build_offline_pipeline(tmp_path)
    cfg = _small_gold_cfg(tmp_path, pipe)
    build_gold_set(cfg, use_llm=False)
    exp = tmp_path / "bm25_heavy.yaml"
    exp.write_text("retrieval:\n  fusion:\n    weights:\n      bm25: 3.0\n")
    report = run_eval(cfg, experiments=[str(exp)])
    text = report.read_text()
    assert "## Retrieval" in text and "| base | dense |" in text and "| bm25_heavy | full |" in text
    assert "## Generation (LLM judge)" in text and "Correct refusal" in text
    raw = json.loads(next((tmp_path / "reports").glob("eval_*.json")).read_text())
    gen = raw["base"]["generation"]["summary"]
    assert gen["correct_refusal_rate"] is not None and gen["faithfulness"] == 0.5
    assert raw["base"]["retrieval"]["full"]["overall"]["recall@20"] > 0.5
