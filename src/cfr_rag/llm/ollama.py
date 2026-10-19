"""Ollama provider: OpenAI-compatible ``/v1`` endpoint plus native ``/api/tags`` health check."""

from __future__ import annotations

import requests

from cfr_rag.llm.base import ProviderHealth
from cfr_rag.llm.errors import LLMError, ProviderUnavailableError
from cfr_rag.llm.openrouter import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    @property
    def root(self) -> str:
        return self.cfg.base_url.rstrip("/").removesuffix("/v1")

    @property
    def api_base(self) -> str:
        return f"{self.root}/v1"

    def api_key(self) -> str:
        return "ollama"

    def _not_running(self) -> ProviderUnavailableError:
        return ProviderUnavailableError(
            f"Ollama is not running at {self.root}. Start it with `ollama serve` (or open the "
            f"Ollama app), or switch the provider to 'openrouter'.")

    def _translate(self, exc: Exception, model: str) -> Exception:
        err = super()._translate(exc, model)
        if isinstance(err, ProviderUnavailableError):
            return self._not_running()
        if isinstance(err, LLMError) and "not found" in str(err):
            return LLMError(f"Ollama model '{model}' is not installed. Run `ollama pull {model}`.")
        return err

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.root}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def health(self) -> ProviderHealth:
        try:
            models = self.list_models()
            return ProviderHealth(self.name, True, "ok", models)
        except requests.RequestException:
            return ProviderHealth(self.name, False, str(self._not_running()))
