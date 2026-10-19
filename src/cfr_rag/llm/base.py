"""Provider-agnostic LLM interface used for generation, query rewriting and judging."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    provider: str
    reachable: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)


class LLMProvider(Protocol):
    """A chat-completion backend (OpenRouter, Ollama, ...)."""

    name: str

    def complete(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, json_mode: bool = False) -> LLMResponse: ...

    def stream(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0,
               max_tokens: int = 1024) -> Iterator[str]: ...

    def health(self) -> ProviderHealth: ...
