# 0005 — Allowlist scoping, deny-by-default

## Context
The tool is pointed at a directory tree that mixes material of very different sensitivity:
public study notes beside confidential work product and personal records. A denylist fails
unsafely — anything the author forgets to exclude gets indexed.

The corpus also contains many third-party repository clones. Indexing them would swamp retrieval
with vendor documentation that answers none of the questions the tool exists to answer.

## Decision
- **Nothing is indexed unless a root is explicitly listed.** Adding a root is an affirmative act.
- **Any subdirectory that is itself a git repository is pruned.** A cloned dependency or a
  course's sample code is not the operator's writing. This is a rule rather than a list because
  the corpus it was built against contains 182 nested repositories; enumerating them would be
  stale within a week. Measured effect on that corpus: 6,252 candidate files down to 58 authored
  ones. Opt back in with `skip_nested_repos: false`.
- Deny globs apply on top, for vendored directories and for secret-bearing files
  (`.env`, credentials, key material) that could otherwise be embedded and later surfaced verbatim.
- The effective scope is printable: the tool can report exactly which roots and how many files it
  would index, before it indexes them.

## Consequences
- Forgetting to add a root means missing results — a visible, harmless failure.
- Forgetting to add an exclusion cannot silently ingest a sensitive tree.
- Scope is inspectable configuration rather than behaviour buried in code, which is what makes the
  privacy claim auditable rather than asserted.

## Rejected
- **Index the tree and exclude the sensitive parts.** Fails unsafely by default.
- **Ask interactively per directory.** Unauditable and unrepeatable; scope must be a file that can
  be reviewed and diffed.
