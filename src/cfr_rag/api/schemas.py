"""API request/response models (query models are shared with the pipeline)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cfr_rag.pipeline import QueryRequest, QueryResponse

__all__ = ["QueryRequest", "QueryResponse", "QueryBody", "SectionResponse", "HealthResponse"]


class QueryBody(QueryRequest):
    stream: bool = Field(False, description="Stream newline-delimited JSON events")


class ParagraphOut(BaseModel):
    path: str
    text: str


class SectionResponse(BaseModel):
    section_id: str
    citation: str
    requested: str
    heading: str
    chapter: str
    chapter_heading: str
    subchapter: str
    subchapter_heading: str
    part: str
    part_heading: str
    subpart: str
    subpart_heading: str
    text: str
    paragraphs: list[ParagraphOut]
    notes: list[str]
    source_citation: str
    amendment_dates: list[str]
    part_source: str
    part_authority: str
    cross_refs: list[str]
    as_of_date: str
    url: str


class HealthResponse(BaseModel):
    status: str
    index: dict[str, Any]
    providers: dict[str, dict[str, Any]]
    llm: dict[str, Any]
