"""Streamlit chat UI for the 21 CFR RAG API.

Run with ``cfr-rag ui`` (or ``streamlit run src/cfr_rag/ui/streamlit_app.py``)
after starting the API with ``cfr-rag serve``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st

from cfr_rag.config import load_config

logger = logging.getLogger(__name__)

CFG = load_config()
PROVIDERS = ["openrouter", "ollama"]


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_url: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        logger.warning("Health check failed: %s", exc)
        return None


def stream_query(api_url: str, payload: dict[str, Any], state: dict[str, Any]) -> Iterator[str]:
    """POST /query with streaming; yields answer tokens and fills ``state`` with metadata."""
    try:
        with requests.post(f"{api_url}/query", json={**payload, "stream": True}, stream=True,
                           timeout=CFG.ui.request_timeout_s) as r:
            if r.status_code != 200:
                state["error"] = f"HTTP {r.status_code}: {r.text[:500]}"
                return
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                kind = event.get("type")
                if kind == "token":
                    yield event["text"]
                elif kind == "retrieval":
                    state["retrieval"] = event
                elif kind == "done":
                    state["result"] = event["result"]
                elif kind == "error":
                    state["error"] = event["message"]
    except requests.RequestException as exc:
        state["error"] = f"Could not reach the API at {api_url}: {exc}"


def render_sources(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        flag = "" if c.get("in_context", True) else " (not in retrieved context)"
        if not c.get("exists", True):
            flag = " (section not found in corpus)"
        title = f"{c['citation']} · {c.get('heading') or ''}{flag}"
        with st.expander(title, expanded=False):
            if c.get("snippet"):
                st.markdown(f"> {c['snippet']}")
            if c.get("url"):
                st.markdown(f"[Open on eCFR]({c['url']})")


def render_debug(debug: dict[str, Any]) -> None:
    with st.expander("Debug: retrieval, scores and timings", expanded=False):
        st.markdown(f"**Route:** `{debug.get('route')}` · **Provider:** `{debug.get('provider')}` · "
                    f"**Model:** `{debug.get('model')}`")
        if debug.get("queries"):
            st.markdown("**Queries:** " + " | ".join(f"`{q}`" for q in debug["queries"]))
        if debug.get("detected_citations"):
            st.markdown("**Detected citations:** " + ", ".join(debug["detected_citations"]))
        if debug.get("missing_citations"):
            st.warning("Cited but not found: " + ", ".join(debug["missing_citations"]))
        st.markdown("**Timings (ms)**")
        st.json(debug.get("timings_ms", {}))
        rows = [{"rank": i + 1, "section": c["section_id"], "pinned": c["pinned"],
                 "sources": ",".join(c["sources"]), **{f"{k}": v for k, v in c["scores"].items()},
                 "header": c["header"]} for i, c in enumerate(debug.get("final_candidates", []))]
        st.markdown("**Final passages (after rerank)**")
        st.dataframe(rows, use_container_width=True)
        fused = [{"rank": i + 1, "section": c["section_id"], **c["ranks"],
                  "rrf": c["scores"].get("rrf")} for i, c in enumerate(debug.get("fused_candidates", []))]
        st.markdown("**Fused candidates (top 20)**")
        st.dataframe(fused, use_container_width=True)
        st.markdown("**Index**")
        st.json(debug.get("index", {}))


def sidebar(health: dict[str, Any] | None) -> dict[str, Any]:
    st.sidebar.header("Settings")
    api_url = st.sidebar.text_input("API URL", CFG.ui.api_url)
    if health is None:
        st.sidebar.error("API not reachable. Start it with `cfr-rag serve`.")
    else:
        idx = health["index"]
        if idx.get("loaded"):
            st.sidebar.success(f"Index: {idx['chunks']:,} chunks · as of {idx['as_of_date']}")
        else:
            st.sidebar.error(f"Index not loaded: {idx.get('error')}")
        for name, p in health["providers"].items():
            icon = "🟢" if p["reachable"] else "🔴"
            st.sidebar.caption(f"{icon} {name}: {p['detail'][:120]}")

    default = CFG.llm.default_provider
    provider = st.sidebar.radio("LLM provider", PROVIDERS,
                                index=PROVIDERS.index(default) if default in PROVIDERS else 0,
                                horizontal=True)
    models = (health or {}).get("llm", {}).get("generation_models") or \
        CFG.llm.roles["generation"].models
    default_model = models.get(provider, "")
    installed = (health or {}).get("providers", {}).get("ollama", {}).get("models", [])
    if provider == "ollama" and installed:
        options = sorted(set(installed) | {default_model} - {""})
        model = st.sidebar.selectbox("Model", options,
                                     index=options.index(default_model) if default_model in options else 0)
    else:
        model = st.sidebar.text_input("Model", default_model,
                                      help="OpenRouter model ID, e.g. as listed on openrouter.ai/models")

    st.sidebar.subheader("Filters")
    chapters = st.sidebar.multiselect("Chapters", ["I", "II", "III"],
                                      help="I: FDA · II: DEA · III: ONDCP")
    parts_raw = st.sidebar.text_input("Parts (comma-separated)", "", placeholder="e.g. 211, 820")
    parts = [p.strip() for p in parts_raw.split(",") if p.strip()]
    rerank = st.sidebar.toggle("Rerank", value=CFG.rerank.enabled)
    debug = st.sidebar.toggle("Show debug view", value=False)
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
    return {"api_url": api_url, "provider": provider, "model": model or None,
            "chapters": chapters or None, "parts": parts or None, "rerank": rerank, "debug": debug}


def main() -> None:
    st.set_page_config(page_title=CFG.ui.title, page_icon="⚖️", layout="wide")
    st.title(CFG.ui.title)
    st.caption("Answers are grounded only in retrieved text of 21 CFR (Chapters I-III), with "
               "section citations. Not legal advice; verify on eCFR.")
    health = fetch_health(CFG.ui.api_url)
    opts = sidebar(health)
    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("citations", []))
                if opts["debug"] and msg.get("debug"):
                    render_debug(msg["debug"])

    question = st.chat_input("Ask about 21 CFR, e.g. 'What does 21 CFR 211.68(b) require?'")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    payload = {"question": question, "provider": opts["provider"], "model": opts["model"],
               "chapters": opts["chapters"], "parts": opts["parts"], "rerank": opts["rerank"]}
    with st.chat_message("assistant"):
        state: dict[str, Any] = {}
        with st.spinner("Retrieving…"):
            tokens = stream_query(opts["api_url"], payload, state)
            first = next(tokens, None)
        if first is None and state.get("error"):
            st.error(state["error"])
            return

        def rest() -> Iterator[str]:
            if first is not None:
                yield first
            yield from tokens

        answer = st.write_stream(rest())
        if state.get("error"):
            st.error(state["error"])
        result = state.get("result") or {}
        citations = result.get("citations", [])
        render_sources(citations)
        debug = result.get("debug") or state.get("retrieval", {}).get("debug", {})
        if opts["debug"] and debug:
            render_debug(debug)
    st.session_state.messages.append({"role": "assistant", "content": result.get("answer") or answer,
                                      "citations": citations, "debug": debug})


main()
