# 21 CFR RAG

Question answering over the U.S. Code of Federal Regulations, **Title 21 (Food and Drugs), Chapters I, II and III**, grounded only in retrieved regulation text and cited to the exact section and paragraph (for example `21 CFR 211.68(b)`).

Every stage (chunker, embedder, dense index, lexical index, reranker, LLM provider) is selected by name in one YAML file, so you can swap components and compare them in the eval harness without touching code.

```
eCFR XML ──► parser ──► SQLite (sections, chunks, definitions)
                             │
              chunker (structural | fixed)
                             │
     Qwen3-Embedding ──► FAISS (Flat IP | HNSW)      bm25s ◄── same chunk IDs
                             │                         │
query ──► router ──► exact citation lookup ─┐   dense top-k + BM25 top-k
                │                           │          │  weighted RRF
                └─► (optional) rewrite / multi-query ──┘
                                            ▼
                         Qwen3-Reranker (50 in → 8 out, yes/no logits)
                                            ▼
          context builder (full-section expansion + defined terms)
                                            ▼
            LLM via OpenRouter (default) or Ollama ──► answer + citations + debug
```

---

## 1. Setup

Requirements: Python 3.11+, about 4 GB free RAM for the two 0.6B models, and about 3 GB of disk for the XML, database, indexes and model weights.

```bash
git clone <this repo> && cd RAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

**Apple silicon.** PyTorch uses the `mps` device automatically (`embedding.device: auto`), with float16 weights. Embedding batches are length-sorted and capped by `embedding.max_batch_tokens` (default 16384) as well as `embedding.batch_size`. If you hit memory pressure on an 8 GB machine, lower `max_batch_tokens` to 8192 and `rerank.batch_size` to 4.

### Add your OpenRouter key

Edit `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

The key is read only from the environment (the variable name is `llm.providers.openrouter.api_key_env`). If it is missing, any OpenRouter call fails with `OPENROUTER_API_KEY is not set. Add it to .env ...`, and `/health` reports the provider as not configured.

The default OpenRouter model is set in `config/default.yaml` under `llm.roles.*.models.openrouter` (currently `inkling`). OpenRouter model IDs usually look like `vendor/model`, so check the exact ID at https://openrouter.ai/models and update all three roles (generation, rewrite, judge).

## 2. Build the corpus and indexes

```bash
cfr-rag ingest                 # latest eCFR date; or: cfr-rag ingest --date 2026-10-01
python scripts/inspect_xml.py data/raw/<date>/title-21.xml   # sanity-check the XML vs the parser
cfr-rag chunk                  # structure-aware chunks (fast)
cfr-rag index                  # embeds (cached) and builds FAISS + BM25
```

- `ingest` downloads `GET /api/versioner/v1/full/{date}/title-21.xml` into `data/raw/<date>/` with a `manifest.json` recording the as-of date, size and SHA-256. If the full-title request fails (it is large), it falls back to one request per part. `--xml path` parses a local file instead.
- `index` takes roughly 20 to 60 minutes the first time on an M-series Mac. Re-running it embeds only chunks whose text changed (see the embedding cache below).

## 3. Run it

```bash
cfr-rag serve                  # FastAPI on http://127.0.0.1:8000  (docs at /docs)
cfr-rag ui                     # Streamlit chat on http://localhost:8501 (needs the API running)
cfr-rag query "What does 21 CFR 211.68(b) require?" --stream
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /query` | `{"question", "chapters"?, "parts"?, "provider"?: "openrouter"\|"ollama", "model"?, "top_k"?, "rerank"?, "stream"?}` |
| `GET /section/{citation}` | `/section/211.68`, `/section/21 CFR 211.68(b)` returns text, hierarchy, notes, cross-references, eCFR URL |
| `GET /health` | index status (model, dimension, chunker, as-of date) and reachability of each LLM provider |

Non-streaming responses contain `answer`, `citations` (section ID, formatted citation, heading, snippet, eCFR URL, and the `in_context` / `exists` flags), `refused`, `provider`, `model`, and `debug`. The `debug` field holds the route taken, query variants, fused and final candidates with dense, BM25, RRF and rerank scores and ranks, the exact context blocks sent to the LLM, and per-stage timings. With `"stream": true` the response is NDJSON: one `retrieval` event, then `token` events, then `done` (the full response) or `error`.

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How long must batch production records be retained?","parts":["211"]}' | jq .answer
```

## 4. Providers: OpenRouter (default) and Ollama

All LLM calls (generation, query rewriting, eval judge) go through `cfr_rag.llm.factory.LLMRouter`. The provider for a call is chosen in this order:

1. The per-request override (`provider` in `/query`, the UI sidebar, or `--provider` on the CLI).
2. The role's `provider` in config (`llm.roles.<role>.provider`).
3. `llm.default_provider` (`openrouter`).

The model is the per-request `model` override, else `llm.roles.<role>.models.<provider>`.

### Running with Ollama locally

```bash
brew install ollama        # or download from https://ollama.com
ollama serve               # or open the Ollama app
ollama pull qwen3:8b       # the default Ollama model in config
cfr-rag query "What is a batch?" --provider ollama
```

The base URL is `llm.providers.ollama.base_url` (default `http://localhost:11434`). If Ollama is not running you get `Ollama is not running at http://localhost:11434. Start it with ollama serve ...`. If the model is not installed you get `Run ollama pull <model>`. `<think>` blocks emitted by reasoning models are stripped, including from streams.

### Switching providers

| Scope | How |
|---|---|
| Everything | `llm.default_provider: ollama`, or `--overlay config/experiments/ollama_local.yaml` |
| One role (e.g. judge only) | `llm.roles.judge.provider: ollama` |
| One request | `"provider": "ollama"` (and optionally `"model": "llama3.1:8b"`) in `/query`; sidebar radio in the UI |

Timeouts, retry counts and backoff are per provider (`timeout_s`, `max_retries`, `backoff_s`). Timeouts, connection errors, 429 and 5xx responses are retried with exponential backoff plus jitter. Streams retry only before the first token.

## 5. Configuration

`config/default.yaml` holds all models, URLs, paths and tunables. Overlay files are deep-merged on top:

```bash
cfr-rag --overlay config/experiments/mrl_256.yaml index
```

Key knobs:

| Setting | Default | Notes |
|---|---|---|
| `chunking.chunker` | `structural` | or `fixed`, or any `module:Class` |
| `chunking.max_tokens` / `overlap_tokens` / `min_tokens` | 512 / 48 / 64 | measured with the Qwen tokenizer (`tokenizer: hf`) |
| `embedding.dimension` | `null` (1024) | Matryoshka output size, e.g. 512 or 256 |
| `index.dense.type` | `flat` | `hnsw` for approximate search |
| `retrieval.dense_top_k` / `bm25_top_k` | 50 / 50 | |
| `retrieval.fusion.weights` | dense 1.0, bm25 1.0 | weighted RRF, `rrf_k: 60` |
| `retrieval.query_rewrite.enabled` / `multi_query.enabled` | false / false | LLM query transforms |
| `rerank.top_n_in` / `top_n_out` | 50 / 8 | |
| `generation.expand_to_parent` | true | send the whole section when a split chunk is retrieved and it fits |

Indexes record the embedding model, backend, dimension and chunker. Loading an index with a config that would produce different vectors or chunks fails with an explicit `IndexMismatchError`, rather than returning silently wrong results. Give each experiment its own `paths.index_dir`.

### Embedding cache

Vectors are cached in `data/cache/embeddings-<model>.sqlite`, keyed by a SHA-256 of model, max length and document text. The full-dimension vector is stored and Matryoshka truncation is applied afterwards, so changing `embedding.dimension` never re-embeds. Only the instruction-prefixed queries bypass the cache.

## 6. Keeping up with eCFR changes

```bash
python scripts/check_updates.py              # pull latest, report added/removed/modified sections
cfr-rag reindex --since 2026-10-01           # apply changes and rebuild (re-embeds changed chunks only)
```

`check-updates` writes `eval/reports/changes_<old>_<new>.md`. The report contains text-diff excerpts per modified section and the versioner API's change records since the old date. It does not modify the database.

## 7. Evaluation

```bash
cfr-rag eval build-gold                       # writes eval/gold.jsonl (~155 questions)
cfr-rag eval run                              # base config, retrieval + generation
cfr-rag eval run --experiment config/experiments/fixed_chunker.yaml \
                 --experiment config/experiments/mrl_256.yaml \
                 --experiment config/experiments/bm25_heavy.yaml
cfr-rag eval run --retrieval-only --limit 50  # quick, no LLM cost
```

**Gold set** (`eval/gold.jsonl`, one JSON object per line, safe to edit by hand; lines starting with `//` are ignored):

| Category | Count | Expected sections come from |
|---|---|---|
| `citation` | 30 | the cited section ("What does 21 CFR 211.68(a) provide?") |
| `topical` | 40 | the section the question was written from (no citation in the question) |
| `definitional` | 35 | every section in that part defining the term |
| `multi_section` | 25 | a section and the section it cross-references |
| `unanswerable` | 25 | none (questions outside Title 21's scope) |

Labels are derived from the corpus structure, so they are correct by construction. With `eval.goldgen.use_llm: true` the LLM only rephrases topical and multi-section questions, and a rephrasing that leaks a section number is replaced by the template. `build-gold` refuses to overwrite an existing file unless you pass `--overwrite`.

**Retrieval metrics** are Recall@{1,3,5,10,20}, MRR and nDCG@10. They are computed at section level (chunks map to their sections, de-duplicated in rank order) over answerable questions, for these modes:

| Mode | What runs |
|---|---|
| `dense` | dense search only, router off |
| `bm25` | BM25 only, router off |
| `hybrid` | dense + BM25 with RRF, router off |
| `hybrid_rerank` | hybrid, then rerank, router off |
| `full` | router + hybrid + rerank (the production path) |

**Generation metrics** use the `judge` role through the same provider layer:

| Metric | Definition |
|---|---|
| Faithfulness | supported atomic claims / claims, judged against the exact context sent to the LLM |
| Citation accuracy | cited sections that exist in the corpus and support the sentences citing them / all citations |
| Answer relevance | judge score 1 to 5, rescaled to 0 to 1 |
| Correct refusal rate | refusals on unanswerable questions |
| False refusal rate | refusals on answerable questions |

The report also shows the share of answers that cite a gold section, the share of uncited answers and latency. Each run writes `eval/reports/eval_<timestamp>.md` (also copied to `latest.md`) and a raw JSON file.

Using the same model as generator and judge inflates scores. For a stricter judge, set `llm.roles.judge.models.openrouter` to a different, stronger model.

## 8. How the eCFR XML is parsed (and what to check)

The development sandbox for this repo had no network access to `www.ecfr.gov`, so the parser was written against the documented layout of the eCFR versioner "full" XML (GPO-derived `DIV1`…`DIV9` elements with a `TYPE` attribute), not against a downloaded copy. Before trusting the index, run `python scripts/inspect_xml.py` on your download. It lists the DIV types and section-level tags and marks anything the parser does not handle.

The parser is written to tolerate variation:

- **Hierarchy comes from `TYPE`**, not the DIV number: TITLE, CHAPTER, SUBCHAP, PART, SUBPART, SUBJGRP, SECTION, APPENDIX. Missing levels (parts with no subpart) and the per-part XML returned by the `?part=` fallback both work; for per-part files, chapter and subchapter come from `structure.json`.
- **Paragraphs are flat `<P>` elements** whose leading markers imply nesting. `ingest/paragraphs.py` rebuilds the full path (e.g. `(b)(1)(ii)(A)`), resolving letter-vs-roman ambiguity: `(i)` after `(h)` is a letter, `(i)` after `(1)` is roman. It uses `<I>`/`<E>` italics to recognise level-5 and level-6 markers, and handles chained markers such as `(b)(1)`.
- **Captured per section:** citation, headings at every level, paragraph tree, `CITA` amendment history (with parsed dates), part `SOURCE` and `AUTH`, `EFFDNOT`/`EDNOTE`/`NOTE` notes, outbound cross-references (`§ 211.68`, `§§ 211.101(c), 211.103`, `parts 211 through 226`), and defined terms.
- **Rendering:** `GPOTABLE` tables become pipe-separated rows, `EXTRACT` content is flattened in order, and reserved sections are kept in the database but not chunked.

If `inspect_xml.py` flags a structural difference (for example a different `TYPE` value for subparts), add it to `HIERARCHY_TYPES`/`LEAF_TYPES` or the block-tag sets in `src/cfr_rag/ingest/parser.py` and extend `tests/fixtures/title-21-sample.xml`.

## 9. Development

```bash
pytest -q                 # offline: hashing embedder, NumPy dense index, simple BM25, fake LLM
ruff check src tests
```

The tests run without network or model downloads. They use test-only backends (`embedding.backend: hash`, `index.dense.type: numpy`, `index.lexical.backend: simple`, and a fake provider in `tests/fakes.py`). Tests that need `faiss`, `openai` or `fastapi` are skipped if those packages are missing.

```
config/                 default.yaml + experiments/*.yaml overlays
scripts/                check_updates.py, inspect_xml.py
src/cfr_rag/
  config.py registry.py citations.py cli.py pipeline.py
  ingest/      ecfr_client, parser, paragraphs, models, diff, service
  chunking/    base, structural, fixed, tokens, service
  store/       sqlite_store (sections, chunks, definitions; chunk_id shared with indexes)
  embeddings/  qwen3, cache, hashing, device
  index/       faiss_index, bm25_index, numpy_index, metadata, builder
  retrieval/   router, hybrid, fusion, query_transform, types
  rerank/      qwen3_reranker, base
  llm/         factory (router), openrouter (OpenAI-compatible), ollama, errors
  generation/  prompts, generator (context builder, citation extraction), schemas
  api/         FastAPI app
  ui/          Streamlit app
  eval/        goldgen, retrieval_metrics, judge, generation_metrics, runner, report
tests/                  fixture XML + unit and end-to-end tests
```

Adding a component: implement the interface (`Chunker`, `Embedder`, `DenseIndex`, `LexicalIndex`, `Reranker`, or a provider with `complete`/`stream`/`health`) and reference it from config as `package.module:ClassName`, or register an alias in `registry.BUILTINS`.

## 10. Limitations

- Answers are only as current as the ingested snapshot (`/health` shows the as-of date). This is not legal advice.
- Section-level labels in the gold set do not grade paragraph-level precision. The citation judge does that.
- `IndexHNSWFlat` with a restrictive part filter can return fewer than k results; use `flat` (the default) when filtering heavily.
