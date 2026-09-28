"""Command line entry point.

`marginalia scope` exists before any indexing command on purpose: the tool should be able
to tell you exactly what it would read before it reads anything (ADR 0005).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config, ConfigError
from .documents import chunk_of, discover
from .scope import ScopeError


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    try:
        return int(args.func(args))
    except (ConfigError, ScopeError) as exc:
        # These are user-facing configuration problems, not bugs; a traceback would
        # bury the one line that says what to fix.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
