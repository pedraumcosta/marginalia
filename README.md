# marginalia

Retrieval over a Markdown knowledge wiki — local embeddings, citations in the wiki's own
`file §section` idiom, and an evaluation harness that reports numbers rather than claims.

> **Status: in development.** Scoping, safety rails, ingestion, indexing, hybrid retrieval
> and evaluation all work. Numbers below are measured and reproducible; the generation
> layer is not built yet. See [`docs/adr/`](docs/adr/) for the decisions
> already made and, as importantly, what was rejected.

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

## Scope profiles

A profile is a named scope. Two are usually wanted: the documents you wrote, and the wider
library you collected. They answer different questions, so they get separate numbers and
separate indexes rather than one blended figure.

```console
$ marginalia scope --list-profiles
authored  (default)
library

$ marginalia scope
Effective scope:
       8  …/Practices/AI
       6  …/Practices/Coding
      19  …/Practices/Mngmnt
       2  …/Practices/TechTrading
  ------
      35  files, 0.8 MiB
  index:  ~/.local/share/marginalia/authored
```

`marginalia scope` reads directory entries only, never file contents, so you can see exactly
what would be indexed before anything is. Add `--files` for the full list.

### Nested repositories are pruned

Any subdirectory that is itself a git repository is skipped: a cloned dependency or a course's
sample code is not your writing. This is a rule rather than an exclusion list because the corpus
it was developed against contains 182 of them, and a list would be stale within a week. Carve
out individual ones with `allow_repos` when a clone genuinely belongs in the corpus.

## Chunking

Chunks follow the document's own heading structure rather than a fixed byte size, so a result
cites itself the way the corpus does:

```console
$ marginalia chunks
profile: authored
      35  documents
     958  chunks
          chars: min 40, median 884, max 1799
     315  carry a section reference (32%)
     410  are part of a split section
     775  have a parent section for context
  trust: authored=958

  sample citations (6 of 958):
    Mngmnt.md §1
    NLP/Courses.md > Agent (1/3)
    Effective_Engineer.md > Adopt the Right Mindsets > Invest in your team's Growth
```

A section reference is recovered from the heading text, so `## 6.5 Agent Memory` cites as
`LLM.md §6.5`. Where headings are not numbered the heading path is used instead. Oversized
sections are split *within* the section and never across a heading boundary, and a document
with no headings at all falls back to fixed-size windows.

Each chunk records the nearest ancestor section, so retrieval can match a small chunk and
expand to its parent for answer context.

### Provenance

Every document carries a trust tier — `authored`, `curated`, `upstream`, `untrusted` or
`unknown` — assigned at ingestion from configuration, not guessed at query time. A corpus
mixing your own writing with collected third-party text mixes text you wrote with text
somebody else could have written, and persistent retrieval turns a one-shot prompt injection
into a durable one. The default tier is `unknown`, because over-trusting by default is the
failure mode that matters.

## Searching

```console
$ marginalia index
profile: authored
      35  added
     958  chunks written
     958  chunks embedded

$ marginalia search --why "memory injection attack numbers"
  1. NLP/LLM.md §6.7 (5/6)   [0.0323]
     via bm25#1, dense#2; trust=authored
  2. NLP/LLM.md §6.5 (1/6)   [0.0315]
     via bm25#3, dense#1; trust=authored
```

Indexing is incremental on content hash, so a second run over an unchanged corpus writes
nothing. `--rebuild` starts over; `--dry-run` reports the plan and writes nothing.

### How retrieval works

Two retrievers, fused by **reciprocal rank fusion**:

- **BM25** via SQLite FTS5 with porter stemming. Finds the exact term, misses the paraphrase.
- **Dense** exact cosine over a numpy matrix — no ANN index, because at 10⁴ chunks the recall
  loss would make every measurement ambiguous between retriever quality and index error
  ([ADR 0002](docs/adr/0002-exact-search.md)).

RRF rather than a weighted sum of scores: BM25 scores and cosine similarities are not on
comparable scales, so combining them numerically means inventing a calibration nobody
measured. RRF reads only rank positions.

Retrieval is single-pass — one query, retrieve, fuse, rank. No query rewriting or multi-hop
until an evaluation shows them earning their cost ([ADR 0006](docs/adr/0006-plain-before-agentic.md)).

## Evaluation

25 questions over the synthetic corpus, committed, so these numbers are reproducible by
anyone with no private data:

```console
$ marginalia index -c fixtures/config.yaml
$ marginalia eval  -c fixtures/config.yaml -k 5

  retriever           hit@5     recall@5  precision@5          mrr       ndcg@5
  lexical             0.960        0.960        0.216        0.861        0.880
  dense               0.800        0.780        0.176        0.607        0.649
  fused               0.880        0.880        0.200        0.783        0.803
```

Metrics are recall, precision, MRR and NDCG at k, with the metric functions tested against
values computed by hand. Runs are cached as JSON and metrics recompute from the cache, so a
figure can be re-derived with no index present and nothing re-embedded:

```console
$ marginalia eval --load eval/public/results-hashing-k5.json
```

Relevance is judged on document plus section, not on the citation string — citations carry a
part suffix for split sections, and pinning ground truth to `(2/6)` would break the golden
set every time chunk sizing changed.

### The measured result contradicts the hybrid default

**Fusion loses to BM25 alone, on every metric, at k = 3, 5 and 10.**

| k | lexical ndcg | fused ndcg | lexical hit | fused hit |
|---|---|---|---|---|
| 3 | **0.855** | 0.775 | **0.920** | 0.840 |
| 5 | **0.880** | 0.803 | **0.960** | 0.880 |
| 10 | **0.893** | 0.843 | 1.000 | 1.000 |

The reason is not that hybrid retrieval is a bad idea. It is that RRF assumes **both** inputs
carry signal, and the table above was produced with the dependency-free `hashing` embedder,
which is lexical by construction. Fusing a retriever that contributes noise costs real
accuracy. Whether a genuine embedding model reverses this is measured separately and reported
with the embedder named, per [ADR 0008](docs/adr/0008-eval-split.md) — a number from one
embedder is never presented as a claim about another.

The CI gate floors the *best* retriever rather than a named one, because which mode wins is
itself a finding here.

One earlier fix, also measured: **function words had to be filtered out of candidate
selection**. BM25's IDF discounts common terms when scoring, which is not the same as keeping
them out of the candidate set. Before the filter, *"what did I conclude about agentic RAG"*
returned a management playbook's hiring section first.

## Development

```console
$ pip install -e '.[dev]'
$ # For real embeddings, install CPU-only torch first. The default Linux torch wheel
$ # pulls several GB of NVIDIA CUDA packages that this project never uses; the CPU
$ # index is ~196 MB.
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install -e '.[embed]'
$ sh tools/install-hooks.sh      # refuses commits containing credentials
$ pytest -q
$ ruff check . && ruff format --check .
```

The pre-commit hook and CI both run `tools/scan_secrets.py`. Generic credential patterns are
committed; project-specific forbidden terms load from a gitignored `.forbidden-terms.local`,
because a scanner in a public repository cannot contain the private strings it looks for. Mark a
deliberate fake with a trailing `marginalia:allow-secret` comment — suppressions are counted and
reported so they cannot accumulate unnoticed.

Run against the synthetic corpus in [`fixtures/`](fixtures/), which is fiction and needs no
configuration of your own:

```console
$ marginalia scope -c fixtures/config.yaml
```

## License

MIT
