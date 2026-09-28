# Architecture decisions

Each record states the context, the decision, what it costs, and what was rejected.
Rejected options are kept deliberately: a decision with no discarded alternative is a
preference, not a decision.

| # | Decision |
|---|---|
| [0001](0001-query-time-layer.md) | Build the query-time layer over an already-compiled wiki |
| [0002](0002-exact-search.md) | Exact search, not approximate nearest neighbour |
| [0003](0003-local-embeddings.md) | Embeddings always local; generation opt-in |
| [0004](0004-swappable-store.md) | Vector store behind a protocol |
| [0005](0005-allowlist-scope.md) | Allowlist scoping, deny-by-default |
| [0006](0006-plain-before-agentic.md) | Plain retrieval before agentic retrieval |
| [0007](0007-heading-chunking.md) | Heading-aware chunks, section-path citations |
| [0008](0008-eval-split.md) | Split eval: public set in CI, private set local, cached artifacts |
