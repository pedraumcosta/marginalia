# 0004 — Vector store behind a protocol

## Context
0002 chooses brute-force exact search, which is right at the current scale and wrong at a larger
one. That decision should be reversible without touching ingestion, retrieval or evaluation.

## Decision
Define a narrow `VectorStore` protocol — `upsert`, `search`, `delete`, `stats` — and make the
brute-force implementation one conforming adapter among several. Selecting a store is a
configuration value, not a code change.

## Consequences
- Swapping in pgvector, Qdrant or Chroma touches one module.
- The protocol is the narrowest interface that serves retrieval, so adapters stay small.
- Store-specific features (payload filtering, quantisation, hybrid scoring) are not exposed
  through the protocol; using them would mean widening it deliberately.

## Rejected
- **Depend on a vector database directly.** Adds infrastructure before there is a measured need,
  and couples eval results to a service version.
- **Abstract over everything.** A general storage abstraction would be larger than the retrieval
  code it serves.
