# 0001 — Build the query-time layer over an already-compiled wiki

## Context
Karpathy's "LLM Wiki" (April 2026) proposes that an agent incrementally compile raw sources into a
persistent, interlinked Markdown wiki — understanding compiled at *write* time. The counterpart
position, argued for tools like OpenBrain, is to keep sources raw and synthesise at *query* time.
The two are usually presented as rivals.

The corpus this tool targets is a knowledge wiki that has been maintained by hand for years:
topic maps with stable numbered sections, per-shelf inventories, one canonical location per item,
and a filing procedure that enforces placement. The write-time arm already exists and is good.

## Decision
Do not rebuild write-time compilation. Build the **query-time retrieval layer** over the corpus,
treating the Markdown files as the canonical source of truth and the index as derived,
rebuildable state.

## Consequences
- The index is disposable by construction; a corrupted index is a rebuild, never a data loss.
- Answers cite the file and section, so the wiki stays authoritative and the tool stays a lens.
- We inherit the corpus's structure as retrieval signal instead of inferring it (see 0007).
- We are coupled to Markdown with heading structure. A flat or binary corpus would need new work.

## Rejected
- **Rebuild as write-time compilation.** It would duplicate a working system, and the published
  critique of LLM-maintained wikis is that the agent "forgets where it wrote things and creates
  disorganized, duplicate, and contradictory information."
- **Shift-worker / handoff model.** That approach replaces memory with structured handoff
  artifacts between short-lived agent runs. It answers a continuity problem. We have a durable
  corpus and a retrieval problem, so it does not apply — but see 0005, where its "continuity lives
  in the environment" instinct survives as explicit, inspectable configuration.
