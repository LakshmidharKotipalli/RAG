"""Command-line entry point: ``cfr-rag <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from cfr_rag.config import AppConfig, load_config
from cfr_rag.logging_setup import setup_logging

logger = logging.getLogger("cfr_rag.cli")


def _load(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config, overlays=args.overlay or [])
    setup_logging(cfg.logging, "DEBUG" if args.verbose else None)
    return cfg


# ---------------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace) -> int:
    from cfr_rag.ingest.service import ingest

    cfg = _load(args)
    sections = ingest(cfg, date=args.date, xml_path=Path(args.xml) if args.xml else None,
                      force=args.force)
    logger.info("Ingested %d sections", len(sections))
    return 0


def cmd_check_updates(args: argparse.Namespace) -> int:
    from cfr_rag.ingest.service import check_updates

    cfg = _load(args)
    diff, versions = check_updates(cfg, new_date=args.date,
                                   old_snapshot=Path(args.old) if args.old else None)
    report = diff.to_markdown(api_versions=versions)
    out = Path(args.out or cfg.paths.reports_dir / f"changes_{diff.old_date}_{diff.new_date}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    logger.info("%d added, %d removed, %d modified; report written to %s", len(diff.added),
                len(diff.removed), len(diff.modified), out)
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from cfr_rag.chunking.service import chunk_corpus

    cfg = _load(args)
    chunk_corpus(cfg)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from cfr_rag.index.builder import build_indexes

    cfg = _load(args)
    build_indexes(cfg, rechunk=args.rechunk)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    from cfr_rag.index.builder import reindex_since

    cfg = _load(args)
    summary = reindex_since(cfg, since=args.since, date=args.date)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from cfr_rag.pipeline import QueryRequest, RAGPipeline

    cfg = _load(args)
    pipe = RAGPipeline.from_config(cfg)
    req = QueryRequest(question=args.question, provider=args.provider, model=args.model,
                       parts=args.part or None, chapters=args.chapter or None)
    if args.stream:
        for event in pipe.stream(req):
            if event["type"] == "token":
                sys.stdout.write(event["text"])
                sys.stdout.flush()
            elif event["type"] == "done":
                print("\n\nCitations:")
                for c in event["result"]["citations"]:
                    print(f"  - {c['citation']}  {c['url']}")
        return 0
    result = pipe.query(req)
    print(result.answer)
    print("\nCitations:")
    for c in result.citations:
        print(f"  - {c.citation}  {c.heading}  {c.url}")
    if args.debug:
        print(json.dumps(result.debug, indent=2, default=str))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = _load(args)
    uvicorn.run("cfr_rag.api.app:create_app", factory=True, host=args.host or cfg.api.host,
                port=args.port or cfg.api.port, reload=False)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).parent / "ui" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_eval_gold(args: argparse.Namespace) -> int:
    from cfr_rag.eval.goldgen import build_gold_set

    cfg = _load(args)
    path = build_gold_set(cfg, out=Path(args.out) if args.out else None,
                          use_llm=not args.no_llm, overwrite=args.overwrite)
    logger.info("Gold set written to %s", path)
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from cfr_rag.eval.runner import run_eval

    cfg = _load(args)
    report = run_eval(cfg, experiments=args.experiment or [], skip_generation=args.retrieval_only,
                      limit=args.limit, build_missing=not args.no_build)
    logger.info("Report written to %s", report)
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfr-rag", description="21 CFR retrieval-augmented QA")
    p.add_argument("--config", help="Base YAML config (default: config/default.yaml)")
    p.add_argument("--overlay", action="append", help="Extra YAML merged on top (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="Download and parse Title 21 into the database")
    s.add_argument("--date", help="As-of date YYYY-MM-DD (default: latest)")
    s.add_argument("--xml", help="Parse a local XML file instead of downloading")
    s.add_argument("--force", action="store_true", help="Re-download an existing snapshot")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("check-updates", help="Download a newer snapshot and report changed sections")
    s.add_argument("--date", help="New as-of date (default: latest)")
    s.add_argument("--old", help="Old snapshot directory (default: the ingested database)")
    s.add_argument("--out", help="Markdown report path")
    s.set_defaults(func=cmd_check_updates)

    s = sub.add_parser("chunk", help="Chunk stored sections with the configured chunker")
    s.set_defaults(func=cmd_chunk)

    s = sub.add_parser("index", help="Embed chunks and build FAISS + BM25 indexes")
    s.add_argument("--rechunk", action="store_true", help="Run the chunker first")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("reindex", help="Pull a newer snapshot and re-index changed sections")
    s.add_argument("--since", required=True, help="Report eCFR changes issued since YYYY-MM-DD")
    s.add_argument("--date", help="Snapshot date to pull (default: latest)")
    s.set_defaults(func=cmd_reindex)

    s = sub.add_parser("query", help="Ask a question from the command line")
    s.add_argument("question")
    s.add_argument("--provider", choices=["openrouter", "ollama"])
    s.add_argument("--model")
    s.add_argument("--part", action="append")
    s.add_argument("--chapter", action="append")
    s.add_argument("--stream", action="store_true")
    s.add_argument("--debug", action="store_true")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("serve", help="Run the FastAPI server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ui", help="Run the Streamlit chat UI")
    s.set_defaults(func=cmd_ui)

    ev = sub.add_parser("eval", help="Evaluation harness").add_subparsers(dest="eval_cmd",
                                                                         required=True)
    s = ev.add_parser("build-gold", help="Generate the gold question set (JSONL)")
    s.add_argument("--out")
    s.add_argument("--no-llm", action="store_true", help="Template questions only")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_eval_gold)
    s = ev.add_parser("run", help="Run retrieval + generation eval and write a markdown report")
    s.add_argument("--experiment", action="append",
                   help="Experiment overlay YAML to compare against the base config (repeatable)")
    s.add_argument("--retrieval-only", action="store_true")
    s.add_argument("--limit", type=int, help="Use only the first N gold questions")
    s.add_argument("--no-build", action="store_true",
                   help="Fail instead of building missing experiment indexes")
    s.set_defaults(func=cmd_eval_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # surface a clean message, full trace with -v
        logging.getLogger("cfr_rag").error("%s: %s", type(exc).__name__, exc,
                                           exc_info=getattr(args, "verbose", False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
