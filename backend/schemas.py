from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: list[dict]


class ExtractedFacts(BaseModel):
    facts: list[str]


class NoteDraft(BaseModel):
    keywords: list[str]
    tags: list[str]
    context: str


class WriteDecision(BaseModel):
    action: Literal["add", "update", "noop"]
    target_note_id: int | None = None
    merged_content: str | None = None
    reason: str


class LinkDecision(BaseModel):
    note_id: int
    should_link: bool
    reason: str


class Evolution(BaseModel):
    links: list[LinkDecision]
    evolved_context: str | None = None
    evolved_tags: list[str] | None = None
    evolve_note_id: int | None = None
