# long-term memory retrieval using zettelkasten

a chat app that actually remembers you. everything you tell it gets distilled into small
linked notes in a local sqlite file, and the whole thought process (search math, model
reasoning, write decisions) streams to the ui as it happens.

**byok** (bring your own anthropic key).

<img width="1396" height="534" alt="Screenshot 2026-08-07 at 6 55 39 PM" src="https://github.com/user-attachments/assets/f3e7eb35-3a9b-4c2d-a05a-37653683b1c1" />


## the idea

llms forget everything between conversations. the usual fix is rag over raw chat logs,
which works badly: transcript chunks mix noise with signal, stored text never gets
updated when you contradict yourself, and related facts sit around as disconnected
islands.

this project glues together two papers that attack that from different angles:

- [a-mem](https://arxiv.org/abs/2502.12110) treats memory like a zettelkasten. each fact
  becomes a note with llm-written keywords, tags, and a context sentence. new notes get
  linked to old ones the model thinks are genuinely related, and a new note can trigger
  "evolution": rewriting an older note whose meaning just changed.
- [mem0](https://arxiv.org/abs/2504.19413) is about write hygiene. before storing
  anything, compare it against what you already know and decide: add, update, or noop.
  without this you end up with five copies of "my name is chai" (ask me how i know).

mem0 keeps the store clean, a-mem keeps it structured. the fusion is a write gate in
front of a note graph.

## what happens per message

1. **retrieval.** your message is embedded locally (all-minilm-l6-v2, 384 dims, no api
   call) and searched two ways in one sqlite file: knn over sqlite-vec, and bm25 over
   fts5. the two rankings get merged with reciprocal rank fusion: each note scores the
   sum of 1/(60+rank) over the rankers it appears in, so a note ranked well by both
   beats a note ranked top by only one. embeddings catch paraphrase ("my dog" finds
   rex), bm25 catches exact names that embeddings blur. fused candidates then get a
   [memorybank](https://arxiv.org/abs/2305.10250)-style rescore:

   ```
   final = 0.75*rel + 0.25*e^(-age/S),  S = 72h * (1 + ln(1 + recalls))
   ```

   rel is the minmax-normalized rrf score, age is hours since the note was last
   recalled or edited, and S is a stability that grows with recall count. so memories
   you actually use fade slower, and a stales lose to a fresh and popular
   note. the top 6 by final go into the system prompt. the write gate below skips this
   rescore on purpose: it needs pure similarity, or old duplicates would fade out of
   its candidate list and sneak back in as "new" facts.
2. **answer.** claude sonnet 5 replies with those memories in context. its thinking
   stream and the answer tokens go straight to the timeline.
3. **memorize.** haiku pulls durable facts out of the exchange. each fact hits the mem0
   gate first. if it survives as an "add", it becomes a note, gets compared to its 5
   nearest neighbors, and the model decides which links are real and whether any old
   note needs rewriting. every decision, including the rejected links, shows up in the
   timeline with a reason.

the llm never scans the whole store. embeddings propose candidates, the model judges
them. per-turn cost stays flat no matter how many notes you have.

## storage

one sqlite file, four tables: notes (the graph nodes), links (edges with the model's
reason attached), a vec0 virtual table for vectors, an fts5 table for keyword search.
no vector db, no graph db. delete `backend/memory.db` to wipe your assistant's brain.

## running it

put your anthropic api key in `backend/.env`, then:

```sh
./run.sh
```

that's it. first run installs everything (the embedding model plus pytorch, so give it
a few minutes), then it starts the backend on :8000 and the ui on :3000. ctrl-c kills
both.

you need a python whose sqlite3 can load extensions. homebrew python works, the one
xcode ships does not. the script defaults to `/opt/homebrew/bin/python3.13`, override
with `PY=/path/to/python ./run.sh`.

chat runs on sonnet 5, the memory pipeline on haiku, so a turn costs a cent or two.
