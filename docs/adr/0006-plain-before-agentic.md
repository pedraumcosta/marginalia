# 0006 — Plain retrieval before agentic retrieval

## Context
Agentic RAG — letting the model decide whether and what to retrieve, iterating over several
searches — is the current default ambition. A practitioner session in this collection, which
builds an agentic RAG coding agent from scratch, reaches the opposite conclusion:
**"Most people should not build agentic RAG systems for their workflows."** Its finding was that
the complexity lived in unglamorous tool details, not in the agentic loop.

The same instinct appears in the context-engineering material: "architectural sophistication is a
hypothesis until a controlled evaluation supports it."

## Decision
Ship single-pass retrieval first: one query, hybrid retrieve, rank, answer. Agentic retrieval
(query rewriting, multi-hop, self-correction) is added only after the evaluation harness exists
and shows single-pass retrieval failing on a measurable class of question.

## Consequences
- A working, measurable system arrives sooner, and becomes the baseline any later sophistication
  must beat.
- Some multi-hop questions will fail in v1. That is a documented limitation, not a defect.
- Every later addition has a number attached to it.

## Rejected
- **Agentic retrieval in v1.** No baseline to justify it against, and it would make failures hard
  to attribute between retrieval, planning and generation.
