# 0003 — Embeddings always local; generation opt-in

## Context
The corpus contains confidential and personally identifying material. Indexing requires embedding
**every chunk**; answering requires sending only the *retrieved* chunks to a generator. Those are
very different exposure profiles, and collapsing them is how privacy leaks get designed in.

A well-known "Hello World" RAG walkthrough in the same collection requires an OpenAI API key and
sends every chunk out during ingestion. That is a reasonable default for public documents and an
unacceptable one here.

## Decision
- **Embeddings run locally, always.** `sentence-transformers` on CPU. There is no configuration
  option to embed through a hosted API.
- **Generation is pluggable and defaults to local/extractive.** A fresh clone answers with no
  network access and no API key. A hosted model is enabled only by explicit local configuration.
- The data-flow boundary is documented in the README, not left implicit.

## Consequences
- Ingestion is slower and embedding quality is bounded by what runs on CPU.
- The repository is runnable by anyone who clones it, with no account anywhere.
- Enabling a hosted generator is a deliberate, reviewable act recorded in ignored local config.

## Rejected
- **Hosted embeddings.** Would export the entire corpus, which is the one thing that must not happen.
- **Hosted generation by default.** Convenient, but it makes the privacy-preserving path the one
  you have to opt into rather than the one you get.
