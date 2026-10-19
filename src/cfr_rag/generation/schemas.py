"""Structured outputs of the generation stage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    section_id: str
    citation: str = Field(description="Formatted citation, e.g. '21 CFR 211.68(b)'")
    heading: str = ""
    snippet: str = ""
    url: str = ""
    in_context: bool = Field(True, description="The cited section was among the retrieved passages")
    exists: bool = Field(True, description="The cited section exists in the corpus")


class ContextBlock(BaseModel):
    index: int
    section_id: str
    label: str
    text: str
    chunk_ids: list[int] = Field(default_factory=list)
    kind: str = "chunk"  # chunk | section | definition


class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    provider: str = ""
    model: str = ""
    context: list[ContextBlock] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
