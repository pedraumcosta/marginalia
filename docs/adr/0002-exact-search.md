# 0002 — Exact search, not approximate nearest neighbour

## Context
Reaching for an ANN index (HNSW, IVF) is the reflex in RAG work. ANN trades recall for speed, and
that trade is worth it at a scale this corpus does not reach: the indexable set is on the order of
10^4 documents, not 10^7.

Two reference points argue the same way. A published knowledge-graph study over the full PubMed
baseline ran **exact dense search over 28.3M abstracts with no ANN index**, on the grounds that the
recall loss was removable and the provenance claim needed it. A small Go RAG sample in the same
collection implements retrieval as hand-written `dotproduct` / `magnitude` — brute-force cosine,
no index at all.

## Decision
Exact search. Dense retrieval is a full cosine scan over a `numpy` matrix; lexical retrieval is
SQLite FTS5. No ANN index in v1.

## Consequences
- Zero recall loss from indexing, so retrieval-quality numbers measure the retriever, not the index.
- No index build step, no index parameters to tune, no staleness between index and store.
- Latency grows linearly with corpus size. This is acceptable at 10^4 and will not be at 10^6;
  0004 keeps the escape hatch open.
- Memory holds the full embedding matrix. At 10^4 chunks × 384 dims × float32 this is ~15 MB.

## Rejected
- **ANN from the start.** Unjustifiable complexity at this scale, and it would make every recall
  measurement ambiguous between retriever quality and index error.
