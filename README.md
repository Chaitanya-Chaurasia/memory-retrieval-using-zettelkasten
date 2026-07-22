# amem-chat

Personal AI chat with **A-MEM agentic memory** ([arXiv 2502.12110](https://arxiv.org/abs/2502.12110)) — a Zettelkasten-style long-term memory in local SQLite, with the backend's full thinking process streamed live to the UI.

## How it works

Every turn runs three visible stages, streamed as SSE events:

1. **Retrieval** — the user message is embedded locally (`all-MiniLM-L6-v2`) and searched against the note store with hybrid retrieval: `sqlite-vec` KNN + SQLite FTS5 BM25, merged with Reciprocal Rank Fusion.
2. **Answering** — Claude (`claude-opus-4-8`, adaptive thinking) answers with retrieved memories in the system prompt. Its thinking summary and answer tokens stream live.
3. **Memorizing (A-MEM)** — memorable facts are extracted from the exchange; each becomes a *note* (LLM-generated keywords, tags, context), gets linked to its nearest existing notes, and may trigger *memory evolution* — retroactively rewriting an old note's context when the new one changes its meaning.

## Run

Backend (needs `ANTHROPIC_API_KEY` in the environment):

```sh
cd backend
# NOTE: must be a Python whose sqlite3 supports loadable extensions
# (Homebrew python works; the python.org/Xcode build does not)
/opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py          # http://localhost:8000
```

Frontend:

```sh
cd frontend
npm install
npm run dev                       # http://localhost:3000
```

First backend start downloads the ~90MB embedding model. Memory lives in `backend/memory.db` — delete it to wipe.
