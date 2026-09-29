# Roadmap

What is built, what is not, and what is deliberately excluded. The architecture decisions
behind each item — including the rejected alternatives — are in [`docs/adr/`](docs/adr/).

## Built

| | |
|---|---|
| **Scoping** | Deny-by-default allowlist. Nothing is indexed unless its root is named. Nested git repositories are pruned by rule, with individual carve-outs. Secret-bearing patterns are excluded unconditionally. `marginalia scope` reports the effective scope without reading a single file. |
| **Profiles** | Named scopes with separate indexes, so a narrow corpus and a wide one get separate numbers rather than one blended figure. |
| **Ingestion** | Markdown chunked on its own heading structure, with the heading path carried on every chunk and section references recovered from headings, so a result cites itself as `File.md §6.5`. Hierarchical: retrieve the chunk, expand to the parent section. Incremental on content hash. |
| **Provenance** | A trust tier per document, assigned at ingestion from configuration. |
| **Index** | SQLite with FTS5 for BM25; vectors in a numpy matrix beside it. Exact cosine, no ANN. Derived state — a corrupt index is a rebuild. |
| **Retrieval** | Hybrid, fused by reciprocal rank, behind a swappable store protocol. Single pass. |
| **Embeddings** | Local only, always. No hosted option exists. A dependency-free hashing embedder lets CI exercise the pipeline offline. |
| **Evaluation** | Recall, precision, MRR and NDCG at k over a committed golden set. Runs cache to JSON; metrics recompute from cache. A CI gate floors the best retriever. |
| **Answering** | Grounded answers with citations. Default provider makes no model call and no network request. Answers are checked against the passages actually supplied, and an invented citation is surfaced rather than trusted. |

## Not built

- **A local generative model.** The default answer is extractive. A hosted model is opt-in; a
  local one would be the better default and is not wired up.
- **Reranking.** A cross-encoder stage is sketched and disabled. It goes in when an evaluation
  shows the fused ranking failing in a way reranking fixes.
- **Incremental dense updates on a changed document.** A changed document re-embeds its own
  chunks; an embedder change re-embeds everything. Both are correct and the second is slow.
- **Watch mode.** Reindexing is manual.
- **Anything multi-hop.** One query, one retrieval pass. See
  [ADR 0006](docs/adr/0006-plain-before-agentic.md) for why that is a decision rather than a gap.

## Deliberately excluded

- **Hosted embeddings.** Indexing touches every chunk, so this is the widest exposure in the
  system. There is no configuration path to a hosted embedder, and a test asserts that asking
  for one raises ([ADR 0003](docs/adr/0003-local-embeddings.md)).
- **An ANN index.** At 10⁴ chunks it would trade recall for latency this corpus does not need,
  and make every measurement ambiguous between retriever quality and index error
  ([ADR 0002](docs/adr/0002-exact-search.md)).
- **Opt-out scoping.** An exclusion list fails unsafely: anything forgotten gets indexed
  ([ADR 0005](docs/adr/0005-allowlist-scope.md)).
- **A single blended quality number.** Which retriever wins depends on the embedder and on the
  corpus, so every figure names both.

## Next, in order

1. **A local generative model**, so the good default and the useful default are the same one.
2. **Reranking, measured.** Add the stage, run the harness, keep it only if it earns its cost.
3. **A second corpus.** Every finding here is conditioned on two corpora. A third would say
   more about which conclusions travel.
4. **Query analysis.** The evaluation names which questions each retriever misses; those misses
   are a shortlist, and nothing has been done with it yet.
