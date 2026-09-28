"""The index: SQLite for records and BM25, a vector store beside it for dense search.

Derived, rebuildable state (ADR 0001). Nothing here is a source of truth, so a corrupted
index is a rebuild rather than a loss, and that is what licenses the cheerful discarding
in ``open_or_create`` when an embedder changes.

Chunk text is stored twice, once in ``chunks`` and once in the FTS5 table. External-content
FTS5 would avoid that, but it requires an integer rowid and chunk ids are content-derived
strings; at this scale the duplication costs a few megabytes and buys a much simpler
write path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chunking import Chunk
from .documents import Document

SCHEMA_VERSION = 1
DB_FILE = "index.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    rel          TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    trust        TEXT NOT NULL,
    size         INTEGER NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    indexed_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_rel ON documents(rel);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_rel         TEXT NOT NULL,
    citation        TEXT NOT NULL,
    section_ref     TEXT,
    heading_path    TEXT NOT NULL,
    text            TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    ordinal         INTEGER NOT NULL,
    part            INTEGER NOT NULL,
    parts           INTEGER NOT NULL,
    parent_chunk_id TEXT,
    trust           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_parent ON chunks(parent_chunk_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    headings,
    tokenize = 'porter unicode61'
);
"""

# FTS5 treats a pile of characters as syntax. A raw user query containing a quote, a
# hyphen or a colon is a syntax error rather than a search, so queries are rebuilt from
# extracted terms instead of passed through.
FTS_TERM = re.compile(r"[\w]+", re.UNICODE)

# Function words carry almost no retrieval signal but do drag in candidates: an OR query
# containing "about" matches most of a prose corpus, and the noise then competes for the
# candidate budget that fusion and reranking work within. BM25's IDF discounts them in
# *scoring*, which is not the same as keeping them out of *candidate selection*.
# Deliberately short and English-only -- an aggressive list starts discarding real terms.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "again",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "at",
        "be",
        "been",
        "being",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "each",
        "few",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "on",
        "only",
        "or",
        "other",
        "our",
        "over",
        "own",
        "same",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "then",
        "these",
        "this",
        "those",
        "to",
        "too",
        "under",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)
MIN_TERM_LENGTH = 2


class IndexError_(Exception):
    """Raised for index problems that are the operator's to fix."""


def fts_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression from arbitrary user input.

    Terms are extracted, quoted and OR-ed. Quoting each term means punctuation cannot
    escape into FTS5's grammar; OR rather than AND keeps recall up, and ranking sorts
    out which of the loose matches actually mattered.
    """
    terms = FTS_TERM.findall(text or "")
    if not terms:
        return ""
    meaningful = [t for t in terms if len(t) >= MIN_TERM_LENGTH and t.lower() not in STOPWORDS]
    # If a query is nothing but function words, honour it literally rather than returning
    # no results at all -- somebody searching for "who" deserves an answer, not silence.
    chosen = meaningful or terms
    return " OR ".join(f'"{t}"' for t in chosen)


@dataclass(frozen=True)
class IndexStats:
    documents: int
    chunks: int
    trust: dict[str, int]
    embedder: str
    schema_version: int

    def render(self) -> str:
        trust = ", ".join(f"{k}={v}" for k, v in sorted(self.trust.items())) or "-"
        return (
            f"  {self.documents:>6}  documents\n"
            f"  {self.chunks:>6}  chunks\n"
            f"  trust: {trust}\n"
            f"  embedder: {self.embedder or 'none'}"
        )


class Index:
    """Records and lexical search. The vector store is managed alongside, not inside."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---- lifecycle -----------------------------------------------------------

    @classmethod
    def open(cls, directory: Path) -> Index:
        directory.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(directory / DB_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        idx = cls(conn)
        stored = idx.get_meta("schema_version")
        if stored is None:
            idx.set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(stored) != SCHEMA_VERSION:
            raise IndexError_(
                f"index at {directory} uses schema version {stored}, this build expects "
                f"{SCHEMA_VERSION}. Delete the directory and re-index."
            )
        conn.commit()
        return idx

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- meta ----------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ---- writes --------------------------------------------------------------

    def known_hashes(self) -> dict[str, str]:
        """``{doc_id: content_hash}`` for the ingest diff."""
        return {
            str(r["doc_id"]): str(r["content_hash"])
            for r in self.conn.execute("SELECT doc_id, content_hash FROM documents")
        }

    def drop_documents(self, doc_ids: Iterable[str]) -> list[str]:
        """Remove documents and their chunks. Returns the chunk ids that went away."""
        ids = list(doc_ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        gone = [
            str(r["chunk_id"])
            for r in self.conn.execute(
                f"SELECT chunk_id FROM chunks WHERE doc_id IN ({marks})", ids
            )
        ]
        if gone:
            cmarks = ",".join("?" * len(gone))
            self.conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({cmarks})", gone)
        self.conn.execute(f"DELETE FROM chunks WHERE doc_id IN ({marks})", ids)
        self.conn.execute(f"DELETE FROM documents WHERE doc_id IN ({marks})", ids)
        return gone

    def put_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        """Insert or replace a document and all of its chunks."""
        self.drop_documents([doc.doc_id])
        self.conn.execute(
            "INSERT INTO documents(doc_id, rel, content_hash, trust, size, chunk_count, "
            "indexed_at) VALUES(?,?,?,?,?,?,?)",
            (
                doc.doc_id,
                doc.rel,
                doc.content_hash,
                doc.trust,
                doc.size,
                len(chunks),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.doc_rel,
                c.citation,
                c.section_ref,
                json.dumps(list(c.heading_path)),
                c.text,
                c.start_line,
                c.end_line,
                c.ordinal,
                c.part,
                c.parts,
                c.parent_chunk_id,
                c.trust,
            )
            for c in chunks
        ]
        self.conn.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, doc_rel, citation, section_ref, "
            "heading_path, text, start_line, end_line, ordinal, part, parts, "
            "parent_chunk_id, trust) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.executemany(
            "INSERT INTO chunks_fts(chunk_id, text, headings) VALUES(?,?,?)",
            [(c.chunk_id, c.text, " / ".join(c.heading_path)) for c in chunks],
        )

    def commit(self) -> None:
        self.conn.commit()

    # ---- reads ---------------------------------------------------------------

    def chunk(self, chunk_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        return {
            str(r["chunk_id"]): r
            for r in self.conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", list(chunk_ids)
            )
        }

    def all_chunk_ids(self) -> list[str]:
        return [str(r["chunk_id"]) for r in self.conn.execute("SELECT chunk_id FROM chunks")]

    def iter_chunk_text(self) -> Iterable[tuple[str, str]]:
        for r in self.conn.execute("SELECT chunk_id, text FROM chunks ORDER BY rowid"):
            yield str(r["chunk_id"]), str(r["text"])

    def search_lexical(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """BM25 search. Returns ``(chunk_id, score)`` best first, score higher-is-better.

        SQLite's ``bm25()`` returns negative numbers where more negative is a better
        match. It is negated here so every retriever in this project agrees that bigger
        means better — mixing the two conventions is a silent ranking inversion.
        """
        expression = fts_query(query)
        if not expression or k <= 0:
            return []
        try:
            rows = self.conn.execute(
                "SELECT chunk_id, bm25(chunks_fts, 1.0, 0.5) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (expression, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            raise IndexError_(f"lexical search failed for {query!r}: {exc}") from exc
        return [(str(r["chunk_id"]), -float(r["score"])) for r in rows]

    def stats(self) -> IndexStats:
        docs = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        trust = {
            str(r["trust"]): int(r["n"])
            for r in self.conn.execute("SELECT trust, COUNT(*) AS n FROM chunks GROUP BY trust")
        }
        return IndexStats(
            documents=int(docs),
            chunks=int(chunks),
            trust=trust,
            embedder=self.get_meta("embedder_signature") or "",
            schema_version=SCHEMA_VERSION,
        )
