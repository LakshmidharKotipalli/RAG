import os

import pytest

from cfr_rag.config import load_config
from cfr_rag.llm.errors import ProviderNotConfiguredError
from cfr_rag.llm.factory import LLMRouter, ThinkFilter, strip_think
from cfr_rag.llm.ollama import OllamaProvider
from cfr_rag.llm.openrouter import OpenAICompatibleProvider


def llm_cfg(**over):
    return load_config("config/default.yaml").with_overrides({"llm": over}).llm


def test_default_routes_to_openrouter_and_override_to_ollama():
    router = LLMRouter(llm_cfg())
    prov, model, _ = router.resolve("generation")
    assert prov.name == "openrouter" and model == "inkling"
    prov, model, _ = router.resolve("generation", provider="ollama")
    assert prov.name == "ollama" and model == "qwen3:8b"
    _, model, _ = router.resolve("judge", provider="ollama", model="llama3.1:8b")
    assert model == "llama3.1:8b"


def test_role_level_provider_from_config():
    router = LLMRouter(llm_cfg(roles={"rewrite": {"provider": "ollama"}}))
    assert router.resolve("rewrite")[0].name == "ollama"
    assert router.resolve("generation")[0].name == "openrouter"


def test_unknown_provider_and_missing_model():
    router = LLMRouter(llm_cfg())
    with pytest.raises(ProviderNotConfiguredError, match="Unknown provider"):
        router.resolve("generation", provider="nope")
    router = LLMRouter(llm_cfg(roles={"judge": {"models": {"openrouter": ""}}}))
    with pytest.raises(ProviderNotConfiguredError, match="No model configured"):
        router.resolve("judge")


def test_missing_openrouter_key_is_clear():
    cfg = llm_cfg().providers["openrouter"]
    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(ProviderNotConfiguredError, match="OPENROUTER_API_KEY is not set"):
            OpenAICompatibleProvider("openrouter", cfg).api_key()
    finally:
        if old is not None:
            os.environ["OPENROUTER_API_KEY"] = old


def test_ollama_not_running_health():
    cfg = llm_cfg().providers["ollama"].model_copy(update={"base_url": "http://127.0.0.1:9"})
    h = OllamaProvider("ollama", cfg).health()
    assert not h.reachable and "Ollama is not running" in h.detail


def test_ollama_base_urls():
    p = OllamaProvider("ollama", llm_cfg().providers["ollama"])
    assert p.api_base == "http://localhost:11434/v1" and p.root == "http://localhost:11434"


def test_strip_think_and_stream_filter():
    assert strip_think("<think>a\nb</think>\nAnswer") == "Answer"
    chunks = ["Hel", "lo <thi", "nk>secret", " stuff</th", "ink> world"]
    assert "".join(ThinkFilter().filter(iter(chunks))) == "Hello world"
    assert "".join(ThinkFilter().filter(iter(["no tags at all"]))) == "no tags at all"


def test_retries_then_succeeds(monkeypatch=None):
    openai = pytest.importorskip("openai")
    import httpx

    cfg = llm_cfg().providers["openrouter"].model_copy(update={"backoff_s": 0.0, "max_retries": 2})
    p = OpenAICompatibleProvider("openrouter", cfg)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
        return "ok"

    assert p._with_retries(flaky, "m") == "ok" and calls["n"] == 3
