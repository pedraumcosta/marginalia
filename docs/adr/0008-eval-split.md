# 0008 — Split eval: public set in CI, private set local, cached artifacts

## Context
Evaluation needs questions with known answers. Over a private corpus, those question–answer pairs
*are* private content — a golden set is a verbatim extract of the material it tests. So the
evaluation that matters cannot be committed, and the evaluation that can be committed does not
run over the real corpus.

Separately, evaluation has to be cheap enough to run often. A knowledge-graph study in this
collection separates an expensive pipeline from an `analysis/run_all.py` that reproduces **every
published number in about ten seconds on CPU from cached posteriors**.

## Decision
- **Two golden sets.** A public set over a synthetic corpus, committed, run in CI. A private set
  over the real corpus, ignored by git, run locally.
- **Cached artifacts.** Retrieval runs write their results; metric computation reads the cache.
  Recomputing metrics never requires re-running retrieval or re-embedding.
- Metrics: recall@k, MRR and NDCG for retrieval; faithfulness, answer relevance, context precision
  and context recall end to end.
- Published numbers state which set produced them. A number from the synthetic corpus is never
  presented as a result about the real one.

## Consequences
- Anyone who clones the repository can reproduce the public numbers exactly, with no private data.
- CI protects against regressions without ever seeing the corpus.
- The synthetic corpus must be good enough to be a real test, which is work; a toy corpus would
  make CI green and meaningless.
- Two sets can drift apart. The public set is the contract for behaviour, the private one for quality.

## Rejected
- **Only a private eval set.** Nothing reproducible, no CI, and the honest-numbers claim rests on
  trust alone.
- **Only a public synthetic set.** Measures the code, not the retrieval problem the tool exists for.
