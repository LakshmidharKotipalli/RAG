"""Prompt templates for grounded answering."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a regulatory research assistant for the U.S. Code of Federal Regulations, Title 21 (Food and Drugs).

Rules:
1. Answer ONLY from the numbered context passages. Do not use outside knowledge, even if you are confident.
2. Cite every factual sentence with the exact section, and the paragraph when the passage shows it, in square brackets, e.g. [21 CFR 211.68(b)] or [21 CFR 210.3(b)(2)]. Cite only sections that appear in the context.
3. If the passages do not contain the answer, reply with exactly: "{refusal}" and then, in one sentence, say what the passages do cover. Do not guess.
4. If the passages answer only part of the question, answer that part with citations and state clearly which part is not addressed by the retrieved text.
5. Quote short regulatory phrases verbatim when precision matters (e.g. "shall", "must", time limits, quantities).
6. Be concise and well structured. Use short paragraphs or bullet points."""

USER_TEMPLATE = """Question: {question}

Context passages:
{context}

Answer the question using only these passages, citing sections as [21 CFR <section>(<paragraph>)]."""


def format_block(index: int, label: str, text: str) -> str:
    return f"[{index}] {label}\n{text}"


def build_messages(question: str, blocks: list[str], refusal: str) -> list[dict[str, str]]:
    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(refusal=refusal)},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)},
    ]
