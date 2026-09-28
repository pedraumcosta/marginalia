"""Command line entry point.

`marginalia scope` exists before any indexing command on purpose: the tool should be able
to tell you exactly what it would read before it reads anything (ADR 0005).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config, ConfigError
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
