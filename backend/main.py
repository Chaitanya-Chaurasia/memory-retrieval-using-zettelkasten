import json
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import agent
from memory import MemoryStore
from schemas import ChatRequest

app = FastAPI(title="long-term memory retrieval using zettelkasten")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MemoryStore()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def run_turn(messages: list[dict]) -> Iterator[str]:
    user_msg = messages[-1]["content"]

    yield sse({"type": "stage", "name": "retrieval"})
    yield sse({"type": "memory_search", "query": user_msg, "method": "hybrid (vector + BM25, RRF)"})
    yield sse({"type": "embed", "model": "all-MiniLM-L6-v2", "dims": 384})
    memories, trace = store.hybrid_search_traced(user_msg, k=6)
    yield sse({"type": "ranker_results", "ranker": "vector", **trace["vector"]})
    yield sse({"type": "ranker_results", "ranker": "bm25", **trace["bm25"]})
    if trace["fusion"]:
        yield sse({"type": "rrf_table", "rows": trace["fusion"]})
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

    yield sse({"type": "stage", "name": "answering"})
    assistant_text = []
    for ev in agent.chat_stream(messages, memories):
        if ev["type"] == "token_delta":
            assistant_text.append(ev["text"])
        yield sse(ev)
    answer = "".join(assistant_text)

    yield sse({"type": "stage", "name": "memorizing"})
    yield sse({"type": "extracting_facts", "model": agent.MEMORY_MODEL})
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
            candidates = store.hybrid_search(fact, k=4)
            decision = agent.decide_write(fact, candidates)
            yield sse({"type": "write_decision", "action": decision.action,
                       "target": decision.target_note_id, "reason": decision.reason})
            if decision.action == "noop":
                if decision.target_note_id:
                    store.touch([decision.target_note_id])
                continue
            if decision.action == "update" and decision.target_note_id:
                store.evolve_note(decision.target_note_id,
                                  content=decision.merged_content or fact)
                updated = store.note_dict(store.get_note(decision.target_note_id))
                yield sse({"type": "note_updated", "note": updated})
                continue

            draft = agent.construct_note(fact)
            note_id = store.add_note(fact, draft.context, draft.keywords, draft.tags)
            note = store.note_dict(store.get_note(note_id))
            yield sse({"type": "note_created", "note": note})

            nbrs = store.neighbors(note_id, k=5)
            if nbrs:
                yield sse({"type": "evolution_check", "note_id": note_id,
                           "neighbors": [{"id": n["id"], "snippet": n["content"][:80],
                                          "distance": n["distance"]} for n in nbrs]})
                evo = agent.decide_links_and_evolution(note, nbrs)
                for link in evo.links:
                    if link.should_link:
                        store.link(note_id, link.note_id, link.reason)
                    yield sse({"type": "link_decision", "a": note_id, "b": link.note_id,
                               "linked": link.should_link, "reason": link.reason})
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
