# marginalia

Retrieval over a Markdown knowledge wiki — local embeddings, citations in the wiki's own
`file §section` idiom, and an evaluation harness that reports numbers rather than claims.

> **Status: in development.** Scoping and safety rails are in place; ingestion, indexing,
> retrieval and evaluation are not yet built. See [`docs/adr/`](docs/adr/) for the decisions
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

## Development

```console
$ pip install -e '.[dev]'
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
