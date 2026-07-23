from typing import Iterator

import anthropic

import prompts
from schemas import Evolution, ExtractedFacts, NoteDraft, WriteDecision

CHAT_MODEL = "claude-sonnet-5"
MEMORY_MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic()


def format_memories(memories: list[dict]) -> str:
    if not memories:
        return "(no relevant memories found)"
    return "\n".join(
        f"- [note {m['id']}] {m['content']} (context: {m['context']})" for m in memories
    )


def chat_stream(messages: list[dict], memories: list[dict]) -> Iterator[dict]:
    with client.messages.stream(
        model=CHAT_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": "summarized"},
        system=prompts.CHAT_SYSTEM.format(memories=format_memories(memories)),
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "thinking_delta" and event.delta.thinking:
                    yield {"type": "thinking_delta", "text": event.delta.thinking}
                elif event.delta.type == "text_delta":
                    yield {"type": "token_delta", "text": event.delta.text}


def extract_facts(user_msg: str, assistant_msg: str) -> list[str]:
    resp = client.messages.parse(
        model=MEMORY_MODEL,
        max_tokens=1024,
        output_format=ExtractedFacts,
        messages=[{
            "role": "user",
            "content": prompts.EXTRACT_FACTS.format(
                user_msg=user_msg, assistant_msg=assistant_msg
            ),
        }],
    )
    return resp.parsed_output.facts if resp.parsed_output else []


def decide_write(fact: str, candidates: list[dict]) -> WriteDecision:
    if not candidates:
        return WriteDecision(action="add", reason="no similar notes exist")
    candidate_text = "\n".join(
        f"- note_id={c['id']}: {c['content']} (context: {c['context']})" for c in candidates
    )
    resp = client.messages.parse(
        model=MEMORY_MODEL,
        max_tokens=512,
        output_format=WriteDecision,
        messages=[{
            "role": "user",
            "content": prompts.DECIDE_WRITE.format(fact=fact, candidates=candidate_text),
        }],
    )
    return resp.parsed_output or WriteDecision(
        action="add", reason="decision failed; defaulting to add"
    )


def construct_note(fact: str) -> NoteDraft:
    resp = client.messages.parse(
        model=MEMORY_MODEL,
        max_tokens=512,
        output_format=NoteDraft,
        messages=[{"role": "user", "content": prompts.CONSTRUCT_NOTE.format(fact=fact)}],
    )
    return resp.parsed_output or NoteDraft(keywords=[], tags=[], context="")


def decide_links_and_evolution(note: dict, neighbors: list[dict]) -> Evolution:
    if not neighbors:
        return Evolution(links=[])
    neighbor_text = "\n".join(
        f"- note_id={n['id']}: {n['content']} (context: {n['context']}, tags: {n['tags']})"
        for n in neighbors
    )
    resp = client.messages.parse(
        model=MEMORY_MODEL,
        max_tokens=1024,
        output_format=Evolution,
        messages=[{
            "role": "user",
            "content": prompts.LINK_AND_EVOLVE.format(
                content=note["content"], context=note["context"], neighbors=neighbor_text
            ),
        }],
    )
    return resp.parsed_output or Evolution(links=[])
