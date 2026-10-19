"""Markdown report comparing configs and retrieval modes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from cfr_rag.config import AppConfig


def _f(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def write_report(results: dict[str, Any], gold: list[dict], cfg: AppConfig, path: Path) -> Path:
    ks = cfg.eval.k_values
    nk = cfg.eval.ndcg_k
    cats = Counter(g["category"] for g in gold)
    first = next(iter(results.values()))
    lines = [
        "# 21 CFR RAG evaluation report", "",
        f"- Corpus as of: **{first['index'].get('as_of_date', '?')}**",
        f"- Gold questions: **{len(gold)}** ({', '.join(f'{c}: {n}' for c, n in sorted(cats.items()))})",
        "- Retrieval metrics are section-level, over answerable questions only. Modes `dense`, "
        "`bm25`, `hybrid` and `hybrid_rerank` bypass the citation router; `full` is the "
        "production path (router + hybrid + rerank).", "",
        "## Configs", "",
        "| Config | Chunker | Chunks | Embedding | Dim | Dense index | Generation LLM | Judge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r["index"]
        llm = r.get("llm", {})
        lines.append(f"| {name} | `{m['chunker_id']}` | {m['num_chunks']:,} | {m['embedding_model']} "
                     f"| {m['dimension']} | {m['dense_type']} | {llm.get('generation', '-')} | "
                     f"{llm.get('judge', '-')} |")

    lines += ["", "## Retrieval", "",
              "| Config | Mode | " + " | ".join(f"R@{k}" for k in ks) + f" | MRR | nDCG@{nk} | ms/query |",
              "|---|---|" + "---|" * (len(ks) + 3)]
    for name, r in results.items():
        for mode, res in r["retrieval"].items():
            o = res["overall"]
            lines.append(f"| {name} | {mode} | " + " | ".join(_f(o.get(f'recall@{k}')) for k in ks)
                         + f" | {_f(o.get('mrr'))} | {_f(o.get(f'ndcg@{nk}'))} | "
                         f"{res['mean_latency_ms']:.0f} |")

    kmid = 10 if 10 in ks else ks[-1]
    lines += ["", f"### Recall@{kmid} by question category", ""]
    categories = sorted({c for r in results.values() for res in r["retrieval"].values()
                         for c in res["by_category"]})
    lines += ["| Config | Mode | " + " | ".join(categories) + " |",
              "|---|---|" + "---|" * len(categories)]
    for name, r in results.items():
        for mode, res in r["retrieval"].items():
            bc = res["by_category"]
            lines.append(f"| {name} | {mode} | " + " | ".join(
                _f(bc.get(c, {}).get(f"recall@{kmid}")) for c in categories) + " |")

    gen = {n: r["generation"] for n, r in results.items() if "generation" in r}
    if gen:
        lines += ["", "## Generation (LLM judge)", "",
                  "| Config | n | Faithfulness | Citation accuracy | Citations exist | "
                  "Answer relevance | Cited a gold section | Correct refusal | False refusal | "
                  "Uncited answers | ms/answer | Errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, g in gen.items():
            s = g["summary"]
            lines.append(
                f"| {name} | {s.get('n', 0)} | {_f(s.get('faithfulness'))} | "
                f"{_f(s.get('citation_accuracy'))} | {_f(s.get('citation_exists_rate'))} | "
                f"{_f(s.get('answer_relevance'))} | {_f(s.get('cited_expected_rate'), True)} | "
                f"{_f(s.get('correct_refusal_rate'), True)} | {_f(s.get('false_refusal_rate'), True)} | "
                f"{_f(s.get('uncited_answer_rate'), True)} | "
                f"{(s.get('mean_latency_ms') or 0):.0f} | {g['errors']} |")
        lines += ["", "Definitions: *faithfulness* = supported claims / claims (judge, against the "
                  "exact context sent to the LLM); *citation accuracy* = cited sections that exist "
                  "in the corpus **and** support the citing sentences / all cited sections; "
                  "*answer relevance* = judge 1-5 rescaled to 0-1; *correct refusal* = refusals on "
                  "unanswerable questions; *false refusal* = refusals on answerable questions."]

    lines += ["", "## Hardest retrieval misses", ""]
    for name, r in results.items():
        best_mode = max(r["retrieval"], key=lambda m: r["retrieval"][m]["overall"].get("mrr", 0),
                        default=None)
        if not best_mode:
            continue
        misses = r["retrieval"][best_mode]["misses"]
        lines.append(f"**{name} / {best_mode}**: {len(misses)} shown (zero recall@{max(ks)})")
        lines.append("")
        for m in misses[:10]:
            lines.append(f"- `{m['id']}` {m['question']}  \n  expected {', '.join(m['expected'])}; "
                         f"got {', '.join(m['got']) or 'nothing'}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
