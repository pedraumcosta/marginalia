# 0007 — Heading-aware chunks, section-path citations

## Context
Chunking strategies trade retrieval precision against answer context: small chunks retrieve well
and answer poorly, large chunks the reverse. The usual defaults — fixed-size with overlap, or
recursive splitting on separators — are corpus-agnostic.

This corpus is not agnostic. It is Markdown organised under numbered headings, and its own
convention is to cite material as `file §n`. That structure is authored signal, and splitting
across it would discard information a human already encoded.

## Decision
- Split on heading boundaries, carrying the full heading path with each chunk.
- **Hierarchical**: retrieve on the small chunk, expand to the enclosing section for answer context.
- Citations are emitted in the corpus's own idiom — file plus section path — so a result can be
  checked against the source by eye.
- Oversized sections fall back to fixed-size splitting *within* the section, never across it.

## Consequences
- Citations are verifiable and idiomatic, which is most of what makes retrieval trustworthy.
- Heading structure becomes a retrieval feature: a path is matchable text.
- Documents with no headings degrade to fixed-size chunking, and lose the citation precision.

## Rejected
- **Fixed-size with overlap.** Simple and robust, but discards authored structure and yields
  citations that cannot be located by a reader.
- **Whole-document chunks.** Precision collapses, and answer context blows the window.
