"""Run the full evaluation over one or more configs and write a markdown report."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cfr_rag.config import AppConfig
from cfr_rag.eval.generation_metrics import evaluate_answer, summarize
from cfr_rag.eval.goldgen import load_gold
from cfr_rag.eval.judge import Judge
from cfr_rag.eval.report import write_report
from cfr_rag.eval.retrieval_metrics import mean_metrics, score_query, section_ranking
from cfr_rag.index.builder import build_indexes
from cfr_rag.index.metadata import IndexNotFoundError
from cfr_rag.pipeline import RAGPipeline
from cfr_rag.retrieval.types import Filters

logger = logging.getLogger(__name__)

# mode -> (retriever mode, use router, rerank)
MODE_SPECS: dict[str, tuple[str, bool, bool]] = {
    "dense": ("dense", False, False),
    "bm25": ("bm25", False, False),
    "hybrid": ("hybrid", False, False),
    "hybrid_rerank": ("hybrid", False, True),
    "full": ("hybrid", True, True),  # production path: router + hybrid + rerank
}


def _load_pipeline(cfg: AppConfig, build_missing: bool) -> RAGPipeline:
    try:
        return RAGPipeline.from_config(cfg)
    except IndexNotFoundError:
        if not build_missing:
            raise
        logger.info("Index missing at %s; building it for this experiment", cfg.paths.index_dir)
        build_indexes(cfg)
        return RAGPipeline.from_config(cfg)


def eval_retrieval(pipe: RAGPipeline, gold: list[dict], modes: list[str]) -> dict[str, Any]:
    """Per-mode aggregate and per-category metrics over answerable gold questions."""
    ecfg = pipe.cfg.eval
    depth = max(max(ecfg.k_values), ecfg.ndcg_k)
    items = [g for g in gold if g["answerable"] and g.get("expected_sections")]
    results: dict[str, Any] = {}
    for mode in modes:
        if mode not in MODE_SPECS:
            raise ValueError(f"Unknown eval mode {mode!r}; choose from {list(MODE_SPECS)}")
        rmode, router, rerank = MODE_SPECS[mode]
        if rerank and pipe.reranker is None:
            logger.warning("Skipping %s: reranker disabled in this config", mode)
            continue
        rows, per_cat, latencies, misses = [], {}, [], []
        for g in items:
            t0 = time.perf_counter()
            _, final, _ = pipe.retrieve(g["question"], Filters(), mode=rmode, rerank=rerank,
                                        use_router=router, top_k=depth)
            latencies.append((time.perf_counter() - t0) * 1000)
            ranking = section_ranking(c.chunk.section_ids for c in final)
            m = score_query(ranking, g["expected_sections"], ecfg.k_values, ecfg.ndcg_k)
            rows.append(m)
            per_cat.setdefault(g["category"], []).append(m)
            if m[f"recall@{max(ecfg.k_values)}"] == 0:
                misses.append({"id": g["id"], "question": g["question"],
                               "expected": g["expected_sections"], "got": ranking[:5]})
        results[mode] = {
            "overall": mean_metrics(rows),
            "by_category": {c: mean_metrics(v) for c, v in per_cat.items()},
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "misses": misses[:15], "n": len(rows),
        }
        logger.info("[%s] %s", mode, {k: round(v, 3) for k, v in results[mode]["overall"].items()})
    return results


def eval_generation(pipe: RAGPipeline, gold: list[dict]) -> dict[str, Any]:
    sample = pipe.cfg.eval.generation_sample
    items = gold
    if sample and sample < len(gold):
        import random

        rng = random.Random(pipe.cfg.eval.seed)
        items = rng.sample(gold, sample)
    judge = Judge(pipe.llm)
    rows = []
    for i, g in enumerate(items, 1):
        try:
            rows.append(evaluate_answer(pipe, judge, g))
        except Exception as exc:
            logger.error("Generation eval failed for %s: %s", g["id"], exc)
            rows.append({"id": g["id"], "category": g["category"], "answerable": g["answerable"],
                         "refused": False, "error": str(exc)})
        if i % 10 == 0:
            logger.info("Generation eval: %d/%d", i, len(items))
    ok = [r for r in rows if "error" not in r]
    return {"summary": summarize(ok) if ok else {}, "errors": len(rows) - len(ok), "rows": rows}


def run_eval(cfg: AppConfig, experiments: list[str] | None = None, skip_generation: bool = False,
             limit: int | None = None, build_missing: bool = True) -> Path:
    """Evaluate the base config plus each experiment overlay; return the report path."""
    gold = load_gold(Path(cfg.eval.gold_path))
    if limit:
        gold = gold[:limit]
    if not gold:
        raise RuntimeError(f"Gold set {cfg.eval.gold_path} is empty; run `cfr-rag eval build-gold`.")
    configs: list[tuple[str, AppConfig]] = [("base", cfg)]
    for exp in experiments or []:
        overlay = yaml.safe_load(Path(exp).read_text()) or {}
        configs.append((Path(exp).stem, cfg.with_overrides(overlay)))

    results: dict[str, Any] = {}
    for name, c in configs:
        logger.info("=== Evaluating config '%s' (index %s) ===", name, c.paths.index_dir)
        pipe = _load_pipeline(c, build_missing)
        entry: dict[str, Any] = {"index": vars(pipe.bundle.meta), "retrieval":
                                 eval_retrieval(pipe, gold, c.eval.retrieval_modes)}
        if not skip_generation:
            prov, model, _ = pipe.llm.resolve("generation")
            jprov, jmodel, _ = pipe.llm.resolve("judge")
            entry["llm"] = {"generation": f"{prov.name}/{model}", "judge": f"{jprov.name}/{jmodel}"}
            entry["generation"] = eval_generation(pipe, gold)
        results[name] = entry

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(cfg.paths.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"eval_{stamp}.json"
    raw.write_text(json.dumps(results, indent=2, default=str))
    report = write_report(results, gold, cfg, out_dir / f"eval_{stamp}.md")
    (out_dir / "latest.md").write_text(report.read_text())
    logger.info("Raw results: %s", raw)
    return report
