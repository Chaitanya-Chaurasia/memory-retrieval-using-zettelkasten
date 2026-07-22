"""FastAPI backend: SSE chat endpoint that streams the full thinking process —
memory retrieval, Claude's reasoning + answer, then the A-MEM write pipeline."""

import json
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()  # reads backend/.env (ANTHROPIC_API_KEY) before the anthropic client is created

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import agent
from memory import MemoryStore

app = FastAPI(title="amem-chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MemoryStore()


class ChatRequest(BaseModel):
    messages: list[dict]  # [{role, content}, ...]


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def run_turn(messages: list[dict]) -> Iterator[str]:
    user_msg = messages[-1]["content"]

    # --- 1. memory retrieval ---
    yield sse({"type": "stage", "name": "retrieval"})
    yield sse({"type": "memory_search", "query": user_msg, "method": "hybrid (vector + BM25, RRF)"})
    memories = store.hybrid_search(user_msg, k=6)
    store.touch([m["id"] for m in memories])
    for m in memories:
        yield sse({
            "type": "memory_hit",
            "id": m["id"],
            "content": m["content"],
            "context": m["context"],
            "score": m["score"],
            "sources": m["sources"],
        })
    if not memories:
        yield sse({"type": "memory_miss"})

    # --- 2. answer with Claude (thinking + text streamed) ---
    yield sse({"type": "stage", "name": "answering"})
    assistant_text = []
    for ev in agent.chat_stream(messages, memories):
        if ev["type"] == "token_delta":
            assistant_text.append(ev["text"])
        yield sse(ev)
    answer = "".join(assistant_text)

    # --- 3. A-MEM write pipeline ---
    yield sse({"type": "stage", "name": "memorizing"})
    try:
        facts = agent.extract_facts(user_msg, answer)
    except Exception as e:
        yield sse({"type": "error", "where": "extract_facts", "message": str(e)})
        facts = []
    if not facts:
        yield sse({"type": "no_facts"})
    for fact in facts:
        yield sse({"type": "fact_extracted", "fact": fact})
        try:
            draft = agent.construct_note(fact)
            note_id = store.add_note(fact, draft.context, draft.keywords, draft.tags)
            note = store.note_dict(store.get_note(note_id))
            yield sse({"type": "note_created", "note": note})

            nbrs = store.neighbors(note_id, k=5)
            if nbrs:
                yield sse({"type": "evolution_check", "note_id": note_id,
                           "neighbor_ids": [n["id"] for n in nbrs]})
                evo = agent.decide_links_and_evolution(note, nbrs)
                for link in evo.links:
                    if link.should_link:
                        store.link(note_id, link.note_id, link.reason)
                        yield sse({"type": "link_created", "a": note_id, "b": link.note_id,
                                   "reason": link.reason})
                if evo.evolve_note_id is not None and evo.evolved_context:
                    store.evolve_note(evo.evolve_note_id, context=evo.evolved_context,
                                      tags=evo.evolved_tags)
                    yield sse({"type": "note_evolved", "note_id": evo.evolve_note_id,
                               "new_context": evo.evolved_context,
                               "new_tags": evo.evolved_tags})
        except Exception as e:
            yield sse({"type": "error", "where": "memorize", "message": str(e)})

    yield sse({"type": "done"})


@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(run_turn(req.messages), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/memories")
def memories():
    return store.all_notes()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
