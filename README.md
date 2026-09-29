# marginalia

Retrieval over a Markdown knowledge wiki — local embeddings, citations in the wiki's own
`file §section` idiom, and an evaluation harness that reports numbers rather than claims.

> **Status: in development.** Scoping, safety rails, ingestion, indexing, hybrid retrieval,
> evaluation and answering all work. Numbers below are measured and reproducible. See [`docs/adr/`](docs/adr/) for the decisions
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
       8  ~/wiki/ai
       6  ~/wiki/engineering
      19  ~/wiki/management
       2  ~/wiki/trading
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

## Asking

```console
$ marginalia ask "what did I conclude about agentic RAG"
No model was called; these are the passages that matched, most relevant first.

NLP/LLM.md §5.1
  Agentic RAG (the model decides whether to retrieve) vs. retrieving every time, and
  when *not* to go agentic — see §6.6 ...

Sources:
  - NLP/LLM.md §5.1
  - NLP/AI Engineering Skills Map.md §3.2
```

The default provider makes **no model call and no network request** — it returns the
passages and the sentences that match, with citations. A hosted model is used only when
local config says so.

### Retrieved text is data, never instruction

This is where the provenance recorded at ingestion earns its place. A corpus assembled partly
from saved articles contains text somebody else wrote, so a retrieval system that pipes it
into a prompt is a prompt-injection channel — and persistent retrieval turns a one-shot
injection into one that fires on **every future query** that retrieves the poisoned passage.

Four measures, each with tests:

- Passages travel in a delimited, provenance-labelled block in the **user** turn. Corpus text
  never enters the system prompt, where injected text would read as policy.
- The system prompt states that passage content is data, and that an instruction found inside
  a passage is to be reported rather than followed.
- **Answers are checked against the passages actually supplied.** A citation naming anything
  else is surfaced as a warning instead of trusted — `Answer.grounded` is false and the
  invented citation is printed.
- `--trust-floor` withholds passages below a provenance tier before generating at all.

None of this makes injection impossible. It makes it visible, and keeps the blast radius
inside one answer rather than inside the index.

### The data-flow boundary, restated

`marginalia ask` prints to stderr how many passages and characters are about to leave the
machine, every time the hosted provider is used, and every answer records whether it did.

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

### What actually wins depends on two things, and neither is the design

The same 25 questions, run under two embedders. Then the same harness against a second,
private corpus of roughly a thousand chunks — a hand-maintained reference wiki, dense with
proper nouns, identifiers and cross-references.

| corpus | embedder | lexical | dense | fused | winner |
|---|---|---|---|---|---|
| synthetic, 56 chunks | `hashing` | 0.880 | 0.649 | 0.803 | **lexical** |
| synthetic, 56 chunks | MiniLM | 0.880 | 0.888 | 0.938 | **fused** |
| private, ~1000 chunks | MiniLM | 0.796 | 0.582 | 0.733 | **lexical** |

*ndcg@5. The private row is not reproducible from this repository — the corpus is not
public — and is reported as measured, per [ADR 0008](docs/adr/0008-eval-split.md).*

**Two conditions decide it.**

*The embedder has to carry signal.* Swap MiniLM for the dependency-free `hashing` embedder
and fusion goes from beating both its inputs to losing to one of them. RRF assumes each
ranking it fuses is informative; fusing noise costs real accuracy.

*The corpus has to suit the retriever.* With the same good embedder, fusion wins on synthetic
prose and loses on a reference corpus. BM25 is at home among exact identifiers, section
numbers and proper nouns; a general-purpose sentence model is not, and fusing a weaker
retriever drags the ranking down again.

Where fusion does win, the reason is visible in the misses: **lexical and dense fail on
different questions.** BM25 misses *"What should I read first and why?"*; dense misses
*"Which facts are recorded as unresolved?"*. That complementarity is the whole premise of
hybrid retrieval, and it is the thing to check rather than assume.

So there is no headline number here, and that is the finding. Measure on your corpus with
your embedder; the CI gate floors the *best* retriever rather than a named one for exactly
this reason.

### A negative result: folder-level prose did not help

The private corpus is organised in four top-level practice folders. Each was given a README
— what the folder is, what questions it answers, which documents to start from, and what is
deliberately elsewhere — on the theory that folder-level orientation would improve retrieval.

Measured properly: READMEs removed, full rebuild, measured, restored, re-indexed, measured
again, nothing else changed.

| retriever | Δ hit@5 | Δ recall@5 | Δ precision@5 | Δ mrr | Δ ndcg@5 |
|---|---|---|---|---|---|
| lexical | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| dense | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| fused | +0.000 | +0.000 | **−0.017** | +0.000 | +0.000 |

Nothing. The prediction recorded beforehand — that semantic retrieval would gain most — was
simply wrong, and fused precision fell slightly because the new pages take result slots
without being relevant to those questions.

The limitation matters as much as the result: those questions ask about **content inside**
documents, while the READMEs answer **routing** questions, which the set barely contains. A
separate routing set, written afterwards and therefore weaker evidence, finds the READMEs
retrievable for that purpose (`hit@5` 1.000). So the narrow conclusion is that folder prose
does not improve content retrieval and is retrievable for navigation — not that it is
worthless, and not that it helped.

The first run of this experiment was **invalid**, and finding out why was worth more than the
experiment. Three of the four READMEs were missing from the index: each sat at the top of its
own scope root, every root-relative path came out as `README.md`, and since document identity
derives from that path, three were silently overwritten. The index reported "4 added" while
holding one. Checking whether the new documents were retrieved *at all* is what surfaced it.

## Limitations

- **Two corpora.** Every finding above is conditioned on a synthetic corpus of 56 chunks and
  one private corpus of about a thousand. Neither is large, and two is not many.
- **The public golden set is 25 questions**, written by the same person who wrote the corpus.
  It measures whether retrieval finds known passages, not whether answers are good.
- **No local generative model.** The default answer is extractive.
- **Answer quality is unmeasured.** Faithfulness and answer relevance are named in
  [ADR 0008](docs/adr/0008-eval-split.md) and not yet implemented; only retrieval is measured.
- **The injection defences are mitigations, not guarantees.** They make a poisoned passage
  visible and keep its blast radius inside one answer. They do not make injection impossible.
- **Latency is unmeasured.** Exact search is linear in corpus size and fine at this scale;
  nobody has timed it.

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
