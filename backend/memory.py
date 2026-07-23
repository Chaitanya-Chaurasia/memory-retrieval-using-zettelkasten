"""A-MEM style memory store: SQLite + sqlite-vec (KNN) + FTS5 (BM25), hybrid via RRF.

Each memory is a Zettelkasten-style "note" (arXiv 2502.12110): content plus
LLM-generated keywords, tags, and a context sentence, with bidirectional links
to related notes. Old notes can be retroactively evolved as new ones arrive.
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent / "memory.db"
EMBED_DIM = 384

_model = None


def embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    return embedder().encode([text], normalize_embeddings=True)[0].astype(np.float32)


class MemoryStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS links (
                a INTEGER NOT NULL REFERENCES notes(id),
                b INTEGER NOT NULL REFERENCES notes(id),
                reason TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (a, b)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content, context, keywords
            );
            """
        )
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(embedding float[{EMBED_DIM}])"
        )
        self.db.commit()

    # ---------- write path ----------

    def add_note(self, content: str, context: str, keywords: list[str], tags: list[str]) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO notes (content, context, keywords, tags, created_at, updated_at, last_accessed)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, context, json.dumps(keywords), json.dumps(tags), now, now, now),
        )
        note_id = cur.lastrowid
        self.db.execute(
            "INSERT INTO notes_fts (rowid, content, context, keywords) VALUES (?, ?, ?, ?)",
            (note_id, content, context, " ".join(keywords)),
        )
        vec = embed(self._note_text(content, context, keywords))
        self.db.execute(
            "INSERT INTO notes_vec (rowid, embedding) VALUES (?, ?)",
            (note_id, sqlite_vec.serialize_float32(vec)),
        )
        self.db.commit()
        return note_id

    def evolve_note(self, note_id: int, content: str | None = None,
                    context: str | None = None, tags: list[str] | None = None):
        """A-MEM memory evolution / Mem0-style UPDATE: rewrite a note in place
        and re-sync its FTS and vector representations."""
        note = self.get_note(note_id)
        if note is None:
            return
        new_content = content if content is not None else note["content"]
        new_context = context if context is not None else note["context"]
        new_tags = json.dumps(tags) if tags is not None else note["tags"]
        self.db.execute(
            "UPDATE notes SET content = ?, context = ?, tags = ?, updated_at = ? WHERE id = ?",
            (new_content, new_context, new_tags, time.time(), note_id),
        )
        keywords = json.loads(note["keywords"])
        self.db.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
        self.db.execute(
            "INSERT INTO notes_fts (rowid, content, context, keywords) VALUES (?, ?, ?, ?)",
            (note_id, new_content, new_context, " ".join(keywords)),
        )
        vec = embed(self._note_text(new_content, new_context, keywords))
        self.db.execute("DELETE FROM notes_vec WHERE rowid = ?", (note_id,))
        self.db.execute(
            "INSERT INTO notes_vec (rowid, embedding) VALUES (?, ?)",
            (note_id, sqlite_vec.serialize_float32(vec)),
        )
        self.db.commit()

    def link(self, a: int, b: int, reason: str = ""):
        lo, hi = sorted((a, b))
        self.db.execute(
            "INSERT OR IGNORE INTO links (a, b, reason) VALUES (?, ?, ?)", (lo, hi, reason)
        )
        self.db.commit()

    # ---------- read path ----------

    def get_note(self, note_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()

    def touch(self, note_ids: list[int]):
        now = time.time()
        self.db.executemany(
            "UPDATE notes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            [(now, i) for i in note_ids],
        )
        self.db.commit()

    def vector_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        vec = embed(query)
        rows = self.db.execute(
            "SELECT rowid, distance FROM notes_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(vec), k),
        ).fetchall()
        return [(r["rowid"], r["distance"]) for r in rows]

    def bm25_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        # FTS5 MATCH syntax chokes on punctuation; keep alphanumeric tokens only
        tokens = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 1]
        if not tokens:
            return []
        match = " OR ".join(tokens)
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(notes_fts) AS score FROM notes_fts WHERE notes_fts MATCH ?"
                " ORDER BY score LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], r["score"]) for r in rows]

    def hybrid_search_traced(self, query: str, k: int = 6) -> tuple[list[dict], dict]:
        """Like hybrid_search, but also returns the full computation trace:
        raw ranker outputs and the per-note RRF arithmetic."""
        t0 = time.perf_counter()
        vec_hits = self.vector_search(query, k=k * 2)
        vec_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        bm25_hits = self.bm25_search(query, k=k * 2)
        bm25_ms = (time.perf_counter() - t0) * 1000

        def snippet(rid):
            n = self.get_note(rid)
            return (n["content"][:80] if n else "?")

        vec_rank = {rid: r for r, (rid, _) in enumerate(vec_hits)}
        bm25_rank = {rid: r for r, (rid, _) in enumerate(bm25_hits)}
        fusion = []
        for rid in set(vec_rank) | set(bm25_rank):
            vc = 1.0 / (60 + vec_rank[rid] + 1) if rid in vec_rank else 0.0
            bc = 1.0 / (60 + bm25_rank[rid] + 1) if rid in bm25_rank else 0.0
            fusion.append({
                "id": rid,
                "snippet": snippet(rid),
                "vec_rank": vec_rank.get(rid),
                "bm25_rank": bm25_rank.get(rid),
                "vec_contrib": round(vc, 5),
                "bm25_contrib": round(bc, 5),
                "total": round(vc + bc, 5),
            })
        fusion.sort(key=lambda r: -r["total"])
        for i, row in enumerate(fusion):
            row["selected"] = i < k

        trace = {
            "vector": {
                "ms": round(vec_ms, 1),
                "hits": [{"id": rid, "snippet": snippet(rid), "distance": round(d, 4)}
                         for rid, d in vec_hits],
            },
            "bm25": {
                "ms": round(bm25_ms, 1),
                "hits": [{"id": rid, "snippet": snippet(rid), "score": round(s, 3)}
                         for rid, s in bm25_hits],
            },
            "fusion": fusion,
        }
        results = []
        for row in fusion:
            if not row["selected"]:
                continue
            note = self.get_note(row["id"])
            if note:
                sources = (["vector"] if row["vec_rank"] is not None else []) + \
                          (["bm25"] if row["bm25_rank"] is not None else [])
                results.append({**self.note_dict(note), "score": row["total"], "sources": sources})
        return results, trace

    def hybrid_search(self, query: str, k: int = 6) -> list[dict]:
        """Reciprocal Rank Fusion of vector and BM25 rankings (k=60 constant)."""
        vec_hits = self.vector_search(query, k=k * 2)
        bm25_hits = self.bm25_search(query, k=k * 2)
        scores: dict[int, float] = {}
        sources: dict[int, list[str]] = {}
        for rank, (rid, _) in enumerate(vec_hits):
            scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank + 1)
            sources.setdefault(rid, []).append("vector")
        for rank, (rid, _) in enumerate(bm25_hits):
            scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank + 1)
            sources.setdefault(rid, []).append("bm25")
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        results = []
        for rid, score in ranked:
            note = self.get_note(rid)
            if note:
                results.append({**self.note_dict(note), "score": round(score, 5), "sources": sources[rid]})
        return results

    def neighbors(self, note_id: int, k: int = 5) -> list[dict]:
        """Nearest notes to an existing note (for link generation / evolution)."""
        row = self.db.execute(
            "SELECT embedding FROM notes_vec WHERE rowid = ?", (note_id,)
        ).fetchone()
        if row is None:
            return []
        rows = self.db.execute(
            "SELECT rowid, distance FROM notes_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (row["embedding"], k + 1),
        ).fetchall()
        out = []
        for r in rows:
            if r["rowid"] == note_id:
                continue
            note = self.get_note(r["rowid"])
            if note:
                out.append({**self.note_dict(note), "distance": round(r["distance"], 4)})
        return out[:k]

    def all_notes(self) -> list[dict]:
        notes = [self.note_dict(r) for r in self.db.execute("SELECT * FROM notes ORDER BY id DESC")]
        links = [dict(r) for r in self.db.execute("SELECT a, b, reason FROM links")]
        return {"notes": notes, "links": links}

    def links_of(self, note_id: int) -> list[int]:
        rows = self.db.execute(
            "SELECT a, b FROM links WHERE a = ? OR b = ?", (note_id, note_id)
        ).fetchall()
        return [r["b"] if r["a"] == note_id else r["a"] for r in rows]

    @staticmethod
    def note_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["keywords"] = json.loads(d["keywords"])
        d["tags"] = json.loads(d["tags"])
        return d

    @staticmethod
    def _note_text(content: str, context: str, keywords: list[str]) -> str:
        return f"{content}\n{context}\n{' '.join(keywords)}"
