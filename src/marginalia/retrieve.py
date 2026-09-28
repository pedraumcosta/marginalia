"""Hybrid retrieval: BM25 and dense search, fused by reciprocal rank.

Two retrievers fail differently. BM25 finds the exact term and misses the paraphrase;
dense search finds the paraphrase and drifts on rare proper nouns and identifiers — of
which a technical wiki is largely made. Fusing them covers both, and **reciprocal rank
fusion** is used rather than a weighted score sum because BM25 scores and cosine
similarities are not on comparable scales; combining them numerically means inventing a
calibration nobody measured. RRF only reads positions, so there is nothing to calibrate.

Single-pass by design: one query, retrieve, fuse, rank. No query rewriting, no multi-hop.
ADR 0006 keeps that until an evaluation shows it earning its cost.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .documents import TRUST_TIERS
from .embeddings import Embedder
from .index import Index
from .store import VectorStore

# The usual RRF constant. Larger flattens the contribution of rank position; 60 is the
# value from the original paper and there is no measurement here that justifies tuning it.
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with where it came from and why it ranked."""

    chunk_id: str
    citation: str
    doc_rel: str
    text: str
    trust: str
    score: float
    heading_path: tuple[str, ...] = ()
    section_ref: str | None = None
    start_line: int = 0
    end_line: int = 0
    parent_chunk_id: str | None = None
    retrievers: tuple[str, ...] = ()
    ranks: Mapping[str, int] = field(default_factory=dict)

    @property
    def found_by_both(self) -> bool:
        return len(self.retrievers) > 1

    def snippet(self, width: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float, tuple[str, ...], dict[str, int]]]:
    """Fuse ranked id lists. Returns ``(id, score, retrievers, ranks)`` best first.

    Each list contributes ``1 / (k + rank)`` per item, ranks being 1-based.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    positions: dict[str, dict[str, int]] = {}

    for name, ids in rankings.items():
        for position, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + position)
            sources.setdefault(cid, []).append(name)
            positions.setdefault(cid, {})[name] = position

    ordered = sorted(
        scores.items(),
        # Ties broken by how many retrievers agreed, then by id for determinism.
        key=lambda kv: (-kv[1], -len(sources[kv[0]]), kv[0]),
    )
    return [(cid, score, tuple(sources[cid]), positions[cid]) for cid, score in ordered]


def _trust_rank(tier: str) -> int:
    try:
        return TRUST_TIERS.index(tier)
    except ValueError:
        return len(TRUST_TIERS)


def _row_to_hit(
    row: sqlite3.Row,
    score: float,
    retrievers: tuple[str, ...],
    ranks: dict[str, int],
) -> Hit:
    try:
        path = tuple(json.loads(row["heading_path"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        path = ()
    return Hit(
        chunk_id=str(row["chunk_id"]),
        citation=str(row["citation"]),
        doc_rel=str(row["doc_rel"]),
        text=str(row["text"]),
        trust=str(row["trust"]),
        score=score,
        heading_path=path,
        section_ref=row["section_ref"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        parent_chunk_id=row["parent_chunk_id"],
        retrievers=retrievers,
        ranks=ranks,
    )


class Retriever:
    """Lexical, dense, or both."""

    def __init__(
        self,
        index: Index,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        *,
        candidates: int = 50,
        prefer_trust: bool = False,
    ) -> None:
        self.index = index
        self.store = store
        self.embedder = embedder
        self.candidates = max(1, candidates)
        self.prefer_trust = prefer_trust

    @property
    def has_dense(self) -> bool:
        return self.store is not None and self.embedder is not None and len(self.store) > 0  # type: ignore[arg-type]

    # ---- individual retrievers ------------------------------------------------

    def lexical(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        return self.index.search_lexical(query, k or self.candidates)

    def dense(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        if not self.has_dense:
            return []
        assert self.store is not None and self.embedder is not None
        vector = self.embedder.encode([query])
        if not len(vector):
            return []
        return self.store.search(vector[0], k or self.candidates)

    # ---- fused ----------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not query.strip() or k <= 0:
            return []

        lex = self.lexical(query)
        den = self.dense(query)

        rankings: dict[str, Sequence[str]] = {}
        if lex:
            rankings["bm25"] = [cid for cid, _ in lex]
        if den:
            rankings["dense"] = [cid for cid, _ in den]
        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        # Fetch a little beyond k: trust preference may reorder the head.
        head = fused[: max(k * 2, k)]
        rows = self.index.chunks_by_id([cid for cid, _, _, _ in head])

        hits = [
            _row_to_hit(rows[cid], score, retrievers, ranks)
            for cid, score, retrievers, ranks in head
            if cid in rows
        ]

        if self.prefer_trust:
            # Only a tie-break: score dominates, so this cannot promote a bad match just
            # for being well-sourced.
            hits.sort(key=lambda h: (-h.score, _trust_rank(h.trust)))

        return hits[:k]

    def with_context(self, hit: Hit) -> str:
        """Expand a hit to its parent section, for answer context (ADR 0007)."""
        if not hit.parent_chunk_id:
            return hit.text
        parent = self.index.chunk(hit.parent_chunk_id)
        if parent is None:
            return hit.text
        return f"{parent['text']}\n\n{hit.text}"


def build_dense_index(
    index: Index,
    store: VectorStore,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    only: Sequence[str] | None = None,
) -> int:
    """Embed chunk text into the vector store. Returns how many were embedded."""
    wanted = set(only) if only is not None else None
    pending: list[tuple[str, str]] = [
        (cid, text) for cid, text in index.iter_chunk_text() if wanted is None or cid in wanted
    ]
    if not pending:
        return 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.encode([text for _, text in batch])
        store.upsert([cid for cid, _ in batch], np.asarray(vectors, dtype=np.float32))
    return len(pending)
