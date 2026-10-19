"""Fake LLM provider used by tests (referenced from config as ``fakes:FakeProvider``)."""

from __future__ import annotations

import re

from cfr_rag.llm.base import LLMResponse, ProviderHealth

CALLS: list[dict] = []


class FakeProvider:
    """Answers by citing the first section that appears in the context passages."""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg

    def _answer(self, messages):
        user = messages[-1]["content"]
        CALLS.append({"messages": messages})
        if "(no passages retrieved)" in user or "unanswerable" in user.lower():
            return "The retrieved sections of 21 CFR do not answer this question."
        m = re.search(r"\[1\] .*?(\d+\.\d+)", user)
        sec = m.group(1) if m else "211.68"
        return f"The rule requires written records of checks [21 CFR {sec}(a)]. See also [21 CFR 999.1]."

    def complete(self, messages, model, temperature=0.0, max_tokens=1024, json_mode=False):
        return LLMResponse(self._answer(messages), self.name, model, 1.0)

    def stream(self, messages, model, temperature=0.0, max_tokens=1024):
        text = "<think>hmm</think>" + self._answer(messages)
        for i in range(0, len(text), 7):
            yield text[i:i + 7]

    def health(self):
        return ProviderHealth(self.name, True, "fake")
