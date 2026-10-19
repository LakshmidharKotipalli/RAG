"""OpenAI-compatible chat provider (used for OpenRouter, and as the base for Ollama).

Retries timeouts, connection errors, 429 and 5xx responses with exponential
backoff plus jitter. The OpenAI client's own retries are disabled so the policy
here is the only one in effect.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Iterator
from typing import Any

import requests

from cfr_rag.config import ProviderConfig
from cfr_rag.llm.base import LLMResponse, ProviderHealth
from cfr_rag.llm.errors import (
    LLMError,
    LLMTimeoutError,
    ProviderNotConfiguredError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """Chat completions against any OpenAI-compatible endpoint."""

    def __init__(self, name: str, cfg: ProviderConfig) -> None:
        self.name = name
        self.cfg = cfg
        self._client = None

    # ------------------------------------------------------------------ setup
    @property
    def api_base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def api_key(self) -> str:
        if not self.cfg.api_key_env:
            return "not-needed"
        key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not key:
            raise ProviderNotConfiguredError(
                f"{self.cfg.api_key_env} is not set. Add it to .env (copy .env.example) or select "
                f"the 'ollama' provider in config or per request.")
        return key

    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key(),
                                  timeout=self.cfg.timeout_s, max_retries=0,
                                  default_headers=self.cfg.extra_headers or None)
        return self._client

    # ------------------------------------------------------------------ errors
    def _translate(self, exc: Exception, model: str) -> Exception:
        """Map SDK exceptions to our error types; return a retryable marker or a final error."""
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return ProviderNotConfiguredError(
                f"{self.name} rejected the API key ({self.cfg.api_key_env}). Check .env.")
        if isinstance(exc, openai.NotFoundError):
            return LLMError(f"{self.name}: model '{model}' not found. Check the model name in "
                            f"config (llm.roles.*.models.{self.name}).")
        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(f"{self.name} timed out after {self.cfg.timeout_s}s")
        if isinstance(exc, openai.APIConnectionError):
            return ProviderUnavailableError(f"Cannot reach {self.name} at {self.api_base}: {exc}")
        return LLMError(f"{self.name} error: {exc}")

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        import openai

        if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError | openai.RateLimitError
                      | openai.InternalServerError):
            return True
        return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500

    def _with_retries(self, fn, model: str):
        attempts = self.cfg.max_retries + 1
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                if not self._retryable(exc) or attempt == attempts - 1:
                    raise self._translate(exc, model) from exc
                delay = self.cfg.backoff_s * (2**attempt) * (0.75 + random.random() / 2)
                logger.warning("%s call failed (%s); retry %d/%d in %.1fs", self.name,
                               type(exc).__name__, attempt + 1, attempts - 1, delay)
                time.sleep(delay)
        raise LLMError("unreachable")

    # ------------------------------------------------------------------ calls
    def complete(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, json_mode: bool = False) -> LLMResponse:
        """Single chat completion; falls back to plain text if JSON mode is unsupported."""
        import openai

        client = self.client()
        kwargs: dict[str, Any] = {"model": model, "messages": messages,
                                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            resp = self._with_retries(lambda: client.chat.completions.create(**kwargs), model)
        except LLMError as exc:
            if json_mode and isinstance(exc.__cause__, openai.BadRequestError):
                kwargs.pop("response_format")
                resp = self._with_retries(lambda: client.chat.completions.create(**kwargs), model)
            else:
                raise
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return LLMResponse(text=text, provider=self.name, model=model,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1), usage=usage)

    def stream(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0,
               max_tokens: int = 1024) -> Iterator[str]:
        """Yield text deltas. Retries only happen before the first token arrives."""
        client = self.client()
        stream = self._with_retries(lambda: client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            stream=True), model)
        try:
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise self._translate(exc, model) from exc

    def health(self) -> ProviderHealth:
        """Check reachability via ``GET /models`` (does not spend tokens)."""
        headers = {}
        detail = ""
        try:
            headers["Authorization"] = f"Bearer {self.api_key()}"
        except ProviderNotConfiguredError as exc:
            detail = str(exc)
        try:
            r = requests.get(f"{self.api_base}/models", headers=headers, timeout=5)
            ok = r.status_code < 400
            if not ok:
                detail = detail or f"HTTP {r.status_code}"
            return ProviderHealth(self.name, ok and not detail, detail or "ok")
        except requests.RequestException as exc:
            return ProviderHealth(self.name, False, f"unreachable: {exc}")
