"""Command line entry point.

`marginalia scope` exists before any indexing command on purpose: the tool should be able
to tell you exactly what it would read before it reads anything (ADR 0005).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config, ConfigError
from .documents import chunk_of, discover, plan_ingest
from .embeddings import EmbeddingError, make_embedder
from .index import Index, IndexError_
from .retrieve import Retriever, build_dense_index
from .scope import ScopeError
from .store import NumpyVectorStore, StoreError


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="config file (default: config.local.yaml, then config.yaml, in the cwd)",
    )
    p.add_argument(
        "-p",
        "--profile",
        metavar="NAME",
        help="scope profile to use (default: the config's default_profile)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marginalia",
        description="Retrieval over a Markdown knowledge wiki.",
    )
    parser.add_argument("--version", action="version", version=f"marginalia {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_scope = sub.add_parser(
        "scope",
        help="report what would be indexed, without reading file contents",
        description=(
            "Print the effective scope: which roots, how many files, how large. "
            "Reads directory entries only, never file contents."
        ),
    )
    _add_common(p_scope)
    p_scope.add_argument(
        "--list-profiles", action="store_true", help="list the profiles this config declares"
    )
    p_scope.add_argument(
        "--files", action="store_true", help="also list every file that would be indexed"
    )
    p_scope.add_argument(
        "--all-profiles", action="store_true", help="report every profile, not just one"
    )
    p_scope.set_defaults(func=cmd_scope)

    p_chunks = sub.add_parser(
        "chunks",
        help="chunk the corpus and report what came out, without building an index",
        description=(
            "Chunk every document in scope and summarise the result: counts, size "
            "distribution, how many carry a section reference, and sample citations. "
            "Reads file contents but writes nothing."
        ),
    )
    _add_common(p_chunks)
    p_chunks.add_argument("--sample", type=int, default=5, metavar="N", help="sample N citations")
    p_chunks.add_argument("--doc", metavar="REL", help="only this document (relative path)")
    p_chunks.add_argument("--max-chars", type=int, metavar="N", help="override chunk size")
    p_chunks.set_defaults(func=cmd_chunks)

    p_index = sub.add_parser(
        "index",
        help="build or update the index",
        description=(
            "Chunk everything in scope and write it to the index. Documents whose content "
            "has not changed are skipped. Pass --rebuild to discard and start over."
        ),
    )
    _add_common(p_index)
    p_index.add_argument("--rebuild", action="store_true", help="discard the index first")
    p_index.add_argument("--no-dense", action="store_true", help="lexical only, skip embedding")
    p_index.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser(
        "search",
        help="search the index",
        description=(
            "Hybrid retrieval: BM25 and dense search fused by reciprocal rank. Prints "
            "ranked citations. Makes no model call beyond embedding the query."
        ),
    )
    _add_common(p_search)
    p_search.add_argument("query", nargs="+", help="what to search for")
    p_search.add_argument("-k", type=int, default=8, metavar="N", help="results to show")
    p_search.add_argument("--lexical", action="store_true", help="BM25 only")
    p_search.add_argument("--dense", action="store_true", help="dense only")
    p_search.add_argument("--snippets", action="store_true", help="show a snippet per hit")
    p_search.add_argument("--why", action="store_true", help="show which retriever found each hit")
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="summarise the index")
    _add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    return parser


def cmd_scope(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)

    if args.list_profiles:
        default = cfg.default_profile
        for name in cfg.profile_names:
            marker = "  (default)" if name == default else ""
            print(f"{name}{marker}")
        return 0

    names = list(cfg.profile_names) if args.all_profiles else [args.profile or cfg.default_profile]

    for i, name in enumerate(names):
        if len(names) > 1:
            print(f"{'' if i == 0 else chr(10)}profile: {name}")
        policy = cfg.scope(name)
        report = policy.describe()
        print(report.render())
        print(f"  index:  {cfg.index_dir(name)}")
        if args.files:
            for f in sorted(policy.iter_files()):
                print(f"    {f}")
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    profile = args.profile or cfg.default_profile
    policy = cfg.scope(profile)
    trust = cfg.trust_policy(profile)

    docs = list(discover(policy, trust))
    if args.doc:
        docs = [d for d in docs if d.rel == args.doc or d.rel.endswith(args.doc)]
        if not docs:
            print(
                f"error: no document matching {args.doc!r} in profile {profile!r}", file=sys.stderr
            )
            return 1

    chunks = []
    for doc in docs:
        chunks.extend(chunk_of(doc, max_chars=args.max_chars))

    if not chunks:
        print(f"profile {profile!r}: {len(docs)} document(s), no chunks produced")
        return 0

    sizes = sorted(len(c.text) for c in chunks)
    n = len(sizes)
    with_ref = sum(1 for c in chunks if c.section_ref)
    split = sum(1 for c in chunks if c.is_split)
    parented = sum(1 for c in chunks if c.parent_chunk_id)

    print(f"profile: {profile}")
    print(f"  {len(docs):>6}  documents")
    print(f"  {n:>6}  chunks")
    print(f"  {'':>6}  chars: min {sizes[0]}, median {sizes[n // 2]}, max {sizes[-1]}")
    print(f"  {with_ref:>6}  carry a section reference ({with_ref * 100 // n}%)")
    print(f"  {split:>6}  are part of a split section")
    print(f"  {parented:>6}  have a parent section for context")

    tiers: dict[str, int] = {}
    for c in chunks:
        tiers[c.trust] = tiers.get(c.trust, 0) + 1
    print("  trust: " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())))

    if args.sample:
        print(f"\n  sample citations ({min(args.sample, n)} of {n}):")
        step = max(1, n // args.sample)
        for c in chunks[::step][: args.sample]:
            print(f"    {c.citation}")
    return 0


def _open_index(cfg: Config, profile: str) -> Index:
    return Index.open(cfg.index_dir(profile))


def cmd_index(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    profile = args.profile or cfg.default_profile
    policy = cfg.scope(profile)
    trust = cfg.trust_policy(profile)

    docs = list(discover(policy, trust))
    directory = cfg.index_dir(profile)

    with _open_index(cfg, profile) as index:
        if args.rebuild:
            index.drop_documents(list(index.known_hashes()))
            index.commit()
        plan = plan_ingest(docs, index.known_hashes())

        print(f"profile: {profile}")
        print(plan.render())
        if args.dry_run:
            print("  (dry run, nothing written)")
            return 0

        if plan.removed:
            index.drop_documents(plan.removed)

        written = 0
        for doc in plan.work:
            chunks = chunk_of(doc)
            index.put_document(doc, chunks)
            written += len(chunks)
        index.commit()

        if args.no_dense:
            index.set_meta("embedder_signature", "")
            index.commit()
            print(f"  {written:>6}  chunks written (lexical only)")
            return 0

        try:
            embedder = make_embedder(cfg.embeddings)
            store = NumpyVectorStore.open_or_create(
                directory, dim=embedder.dim, signature=embedder.signature
            )
        except (EmbeddingError, StoreError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            print(
                "       (records are indexed; re-run with --no-dense to skip embedding)",
                file=sys.stderr,
            )
            return 1

        previous = index.get_meta("embedder_signature")
        full = args.rebuild or previous != embedder.signature or len(store) == 0
        if full and previous and previous != embedder.signature:
            print(f"  embedder changed ({previous} -> {embedder.signature}); rebuilding vectors")

        only = None if full else [c.chunk_id for d in plan.work for c in chunk_of(d)]
        if full:
            store.clear()
        embedded = build_dense_index(index, store, embedder, only=only)
        store.save(directory)
        index.set_meta("embedder_signature", embedder.signature)
        index.commit()

        print(f"  {written:>6}  chunks written")
        print(f"  {embedded:>6}  chunks embedded ({embedder.signature})")
    return 0


def _retriever(cfg: Config, profile: str, index: Index, *, want_dense: bool) -> Retriever:
    store = None
    embedder = None
    if want_dense:
        try:
            embedder = make_embedder(cfg.embeddings)
            store = NumpyVectorStore.load(cfg.index_dir(profile))
            if store.signature != embedder.signature:
                print(
                    f"warning: index was embedded with {store.signature!r} but configuration "
                    f"says {embedder.signature!r}; dense search disabled. Re-run `index`.",
                    file=sys.stderr,
                )
                store, embedder = None, None
        except (EmbeddingError, StoreError):
            store, embedder = None, None
    return Retriever(
        index,
        store,
        embedder,
        candidates=int(cfg.retrieval.get("candidates", 50)),
        prefer_trust=bool(cfg.retrieval.get("prefer_trust", False)),
    )


def cmd_search(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    profile = args.profile or cfg.default_profile
    query = " ".join(args.query)

    with _open_index(cfg, profile) as index:
        if index.stats().chunks == 0:
            print(
                f"error: index for profile {profile!r} is empty. Run `marginalia index`.",
                file=sys.stderr,
            )
            return 1
        retriever = _retriever(cfg, profile, index, want_dense=not args.lexical)

        if args.lexical:
            pairs = retriever.lexical(query, args.k)
        elif args.dense:
            if not retriever.has_dense:
                print("error: no dense index available. Run `marginalia index`.", file=sys.stderr)
                return 1
            pairs = retriever.dense(query, args.k)
        else:
            pairs = None

        if pairs is not None:
            rows = index.chunks_by_id([cid for cid, _ in pairs])
            for rank, (cid, score) in enumerate(pairs, start=1):
                row = rows.get(cid)
                if row is None:
                    continue
                print(f"{rank:>3}. {row['citation']}   [{score:.4f}]")
                if args.snippets:
                    print(f"     {' '.join(str(row['text']).split())[:160]}")
            return 0

        hits = retriever.search(query, k=args.k)
        if not hits:
            print("no results")
            return 0
        for rank, hit in enumerate(hits, start=1):
            print(f"{rank:>3}. {hit.citation}   [{hit.score:.4f}]")
            if args.why:
                detail = ", ".join(f"{name}#{pos}" for name, pos in sorted(hit.ranks.items()))
                print(f"     via {detail}; trust={hit.trust}")
            if args.snippets:
                print(f"     {hit.snippet()}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    profile = args.profile or cfg.default_profile
    with _open_index(cfg, profile) as index:
        print(f"profile: {profile}")
        print(index.stats().render())
        try:
            store = NumpyVectorStore.load(cfg.index_dir(profile))
            st = store.stats()
            print(f"  vectors: {st['count']} x {st['dim']} ({int(st['bytes']) // 1024} KiB)")
        except StoreError:
            print("  vectors: none")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    try:
        return int(args.func(args))
    except (ConfigError, ScopeError, IndexError_, StoreError, EmbeddingError) as exc:
        # These are user-facing configuration problems, not bugs; a traceback would
        # bury the one line that says what to fix.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
