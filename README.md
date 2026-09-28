# marginalia

Retrieval over a Markdown knowledge wiki — local embeddings, citations in the wiki's own
`file §section` idiom, and an evaluation harness that reports numbers rather than claims.

> **Status: in development.** Phase 1 of 7. See [`docs/adr/`](docs/adr/) for the decisions
> already made and what was rejected.

## Why this exists

Most "chat with your documents" tools assume the documents are unstructured and the index is the
product. This one assumes the opposite: the corpus is a wiki that a human already organised —
topic maps, numbered sections, one canonical home per item — and that structure is signal worth
using rather than chunking away.

So the files stay canonical and the index is derived, rebuildable state. See
[ADR 0001](docs/adr/0001-query-time-layer.md).

## Data-flow boundary

This tool is built to run over corpora that include private material, so the boundary is explicit:

| Stage | Where the text goes |
|---|---|
| Indexing / embeddings | **Never leaves the machine.** Local model, CPU. No configuration option exists to embed through a hosted API. |
| Retrieval | Local. |
| Generation | **Local or extractive by default.** A hosted model is used only when explicitly enabled in ignored local config. |

Nothing is indexed unless you list its root explicitly — scope is an allowlist, not an
exclusion list. See [ADR 0005](docs/adr/0005-allowlist-scope.md).

## License

MIT
