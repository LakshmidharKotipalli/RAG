"""API tests (skipped when FastAPI is not installed)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conftest import build_offline_pipeline  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cfr_rag.api.app import create_app  # noqa: E402


def client(tmp_path: Path) -> TestClient:
    pipe = build_offline_pipeline(tmp_path)
    return TestClient(create_app(pipe.cfg, pipeline=pipe))


def test_query_and_debug(tmp_path: Path):
    with client(tmp_path) as c:
        r = c.post("/query", json={"question": "What does 21 CFR 211.68 require?",
                                   "provider": "ollama", "model": "x"})
        assert r.status_code == 200
        body = r.json()
        assert body["provider"] == "ollama" and body["citations"][0]["section_id"] == "211.68"
        assert body["debug"]["final_candidates"] and "timings_ms" in body["debug"]


def test_query_stream_ndjson(tmp_path: Path):
    with client(tmp_path) as c:
        r = c.post("/query", json={"question": "21 CFR 211.63", "stream": True})
        events = [json.loads(line) for line in r.text.splitlines() if line]
        assert events[0]["type"] == "retrieval" and events[-1]["type"] == "done"


def test_section_lookup(tmp_path: Path):
    with client(tmp_path) as c:
        r = c.get("/section/21 CFR 211.68(b)")
        assert r.status_code == 200
        body = r.json()
        assert body["section_id"] == "211.68" and body["paragraphs"][0]["path"] == "(b)"
        assert body["url"].endswith("#p-211.68(b)")
        assert c.get("/section/999.99").status_code == 404
        assert c.get("/section/hello").status_code == 400


def test_health(tmp_path: Path):
    with client(tmp_path) as c:
        body = c.get("/health").json()
        assert body["index"]["loaded"] and body["providers"]["openrouter"]["reachable"]
        assert body["llm"]["default_provider"] == "openrouter"


def test_validation_rejects_bad_provider(tmp_path: Path):
    with client(tmp_path) as c:
        assert c.post("/query", json={"question": "q", "provider": "gpt"}).status_code == 422
