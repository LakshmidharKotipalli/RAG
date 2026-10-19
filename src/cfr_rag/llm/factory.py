"""Routes each LLM call (by role) to a provider and model.

Resolution order for the provider: per-request override, then the role's
``provider``, then ``llm.default_provider`` (OpenRouter). The model is the
per-request override or the role's model for that provider.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from cfr_rag.config import LLMConfig, RoleConfig
from cfr_rag.llm.base import LLMProvider, LLMResponse, ProviderHealth
from cfr_rag.llm.errors import ProviderNotConfiguredError
from cfr_rag.registry import resolve

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._providers: dict[str, LLMProvider] = {}

    def provider(self, name: str) -> LLMProvider:
        if name not in self.cfg.providers:
            raise ProviderNotConfiguredError(
                f"Unknown provider '{name}'. Configured: {', '.join(self.cfg.providers)}")
        if name not in self._providers:
            pcfg = self.cfg.providers[name]
            self._providers[name] = resolve("llm_provider", pcfg.kind)(name, pcfg)
        return self._providers[name]

    def role(self, role: str) -> RoleConfig:
        if role not in self.cfg.roles:
            raise ProviderNotConfiguredError(f"No LLM role '{role}' in config (llm.roles)")
        return self.cfg.roles[role]

    def resolve(self, role: str, provider: str | None = None,
                model: str | None = None) -> tuple[LLMProvider, str, RoleConfig]:
        """Return ``(provider, model, role_config)`` for a call."""
        rcfg = self.role(role)
        pname = provider or rcfg.provider or self.cfg.default_provider
        prov = self.provider(pname)
        mname = model or rcfg.models.get(pname)
        if not mname:
            raise ProviderNotConfiguredError(
                f"No model configured for role '{role}' on provider '{pname}' "
                f"(set llm.roles.{role}.models.{pname} or pass a model override)")
        return prov, mname, rcfg

    def complete(self, role: str, messages: list[dict[str, str]], provider: str | None = None,
                 model: str | None = None, json_mode: bool = False,
                 max_tokens: int | None = None) -> LLMResponse:
        prov, mname, rcfg = self.resolve(role, provider, model)
        resp = prov.complete(messages, mname, rcfg.temperature, max_tokens or rcfg.max_tokens,
                             json_mode)
        resp.text = strip_think(resp.text)
        logger.debug("LLM %s via %s/%s in %.0f ms", role, prov.name, mname, resp.latency_ms)
        return resp

    def stream(self, role: str, messages: list[dict[str, str]], provider: str | None = None,
               model: str | None = None) -> Iterator[str]:
        prov, mname, rcfg = self.resolve(role, provider, model)
        return ThinkFilter().filter(prov.stream(messages, mname, rcfg.temperature, rcfg.max_tokens))

    def health(self) -> dict[str, ProviderHealth]:
        out = {}
        for name in self.cfg.providers:
            try:
                out[name] = self.provider(name).health()
            except Exception as exc:
                out[name] = ProviderHealth(name, False, str(exc))
        return out


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks some local models emit inline."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


class ThinkFilter:
    """Streaming counterpart of :func:`strip_think`."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def filter(self, deltas: Iterator[str]) -> Iterator[str]:
        for d in deltas:
            self.buf += d
            while True:
                if self.inside:
                    end = self.buf.find("</think>")
                    if end < 0:
                        self.buf = self.buf[-8:]
                        break
                    self.buf = self.buf[end + 8:].lstrip()
                    self.inside = False
                else:
                    start = self.buf.find("<think>")
                    if start >= 0:
                        if start:
                            yield self.buf[:start]
                        self.buf = self.buf[start + 7:]
                        self.inside = True
                        continue
                    safe = len(self.buf) - 7  # keep a possible partial "<think>" tag
                    if safe > 0:
                        yield self.buf[:safe]
                        self.buf = self.buf[safe:]
                    break
        if self.buf and not self.inside:
            yield self.buf
