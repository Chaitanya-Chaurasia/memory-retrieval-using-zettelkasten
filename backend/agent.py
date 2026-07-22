"""LLM layer: chat streaming plus the A-MEM note pipeline (construct, link, evolve)."""

import json
from typing import Iterator

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-4-8"

client = anthropic.Anthropic()


# ---------- structured output schemas ----------

class ExtractedFacts(BaseModel):
    facts: list[str]


class NoteDraft(BaseModel):
    keywords: list[str]
    tags: list[str]
    context: str


class LinkDecision(BaseModel):
    note_id: int
    should_link: bool
    reason: str


class Evolution(BaseModel):
    links: list[LinkDecision]
    evolved_context: str | None = None
    evolved_tags: list[str] | None = None
    evolve_note_id: int | None = None


# ---------- chat ----------

CHAT_SYSTEM = """You are a personal assistant with a long-term memory.
Relevant memories retrieved for this turn are provided below. Use them naturally when
they help; never claim to remember something that is not in the memories or the
conversation. If memories conflict, prefer the most recently updated one.

<memories>
{memories}
</memories>"""


def format_memories(memories: list[dict]) -> str:
    if not memories:
        return "(no relevant memories found)"
    lines = []
    for m in memories:
        lines.append(f"- [note {m['id']}] {m['content']} (context: {m['context']})")
    return "\n".join(lines)


def chat_stream(messages: list[dict], memories: list[dict]) -> Iterator[dict]:
    """Yields {"type": "thinking_delta"|"token_delta", "text": ...} events."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": "summarized"},
        system=CHAT_SYSTEM.format(memories=format_memories(memories)),
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "thinking_delta" and event.delta.thinking:
                    yield {"type": "thinking_delta", "text": event.delta.thinking}
                elif event.delta.type == "text_delta":
                    yield {"type": "token_delta", "text": event.delta.text}


# ---------- A-MEM pipeline ----------

def extract_facts(user_msg: str, assistant_msg: str) -> list[str]:
    """Pull memorable, atomic facts about the user out of the latest exchange."""
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        output_format=ExtractedFacts,
        messages=[{
            "role": "user",
            "content": (
                "Extract durable, memorable facts about the user from this exchange — "
                "preferences, biographical details, projects, goals, relationships, "
                "corrections to earlier beliefs. Each fact must be a short standalone "
                "sentence. Skip small talk, transient states, and anything already implied "
                "by an extracted fact. Return an empty list if nothing is worth remembering.\n\n"
                f"USER: {user_msg}\n\nASSISTANT: {assistant_msg}"
            ),
        }],
    )
    return resp.parsed_output.facts if resp.parsed_output else []


def construct_note(fact: str) -> NoteDraft:
    """A-MEM note construction: LLM generates keywords, tags, and a context sentence."""
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=512,
        output_format=NoteDraft,
        messages=[{
            "role": "user",
            "content": (
                "You maintain a Zettelkasten-style memory. For the memory below, produce:\n"
                "- keywords: 3-6 specific search keywords\n"
                "- tags: 2-4 broad category tags (lowercase, single words)\n"
                "- context: one sentence situating this memory (why it matters, what it relates to)\n\n"
                f"Memory: {fact}"
            ),
        }],
    )
    return resp.parsed_output or NoteDraft(keywords=[], tags=[], context="")


def decide_links_and_evolution(note: dict, neighbors: list[dict]) -> Evolution:
    """A-MEM link generation + memory evolution over the nearest existing notes."""
    if not neighbors:
        return Evolution(links=[])
    neighbor_text = "\n".join(
        f"- note_id={n['id']}: {n['content']} (context: {n['context']}, tags: {n['tags']})"
        for n in neighbors
    )
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        output_format=Evolution,
        messages=[{
            "role": "user",
            "content": (
                "You maintain a Zettelkasten memory network. A new note was just added:\n\n"
                f"NEW NOTE: {note['content']} (context: {note['context']})\n\n"
                f"Its nearest existing notes:\n{neighbor_text}\n\n"
                "1. For each neighbor, decide if a bidirectional link to the new note is "
                "genuinely useful (shared topic, same entity, cause/effect, contradiction). "
                "Give a short reason.\n"
                "2. Memory evolution: if the new note meaningfully changes how ONE existing "
                "neighbor should be understood (e.g. supersedes it, adds crucial context), set "
                "evolve_note_id to that neighbor's id and provide evolved_context (a rewritten "
                "context sentence) and optionally evolved_tags. Otherwise leave them null."
            ),
        }],
    )
    return resp.parsed_output or Evolution(links=[])
