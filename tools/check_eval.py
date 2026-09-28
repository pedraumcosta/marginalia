#!/usr/bin/env python3
"""Fail the build when retrieval regresses below a recorded baseline.

A number nobody checks becomes decoration. This reads a cached run and asserts the best
retriever clears a floor, so a change that quietly degrades ranking stops being invisible.

The floor is deliberately on the *best* retriever rather than on a named one: which mode
wins is itself a finding here, and pinning the check to "fused" would bake in an
assumption the measurements do not currently support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marginalia.evaluate import EvalError, EvalRun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="cached eval run (JSON)")
    ap.add_argument("--min-ndcg", type=float, default=0.0)
    ap.add_argument("--min-hit", type=float, default=0.0)
    ap.add_argument("--min-mrr", type=float, default=0.0)
    args = ap.parse_args(argv)

    try:
        run = EvalRun.load(args.run)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = run.summary()
    if not summary:
        print("error: cached run has no outcomes", file=sys.stderr)
        return 1

    ndcg_key, hit_key = f"ndcg@{run.k}", f"hit@{run.k}"
    best = max(summary.items(), key=lambda kv: kv[1].get(ndcg_key, 0.0))
    name, metrics = best

    print(f"best retriever: {name}")
    for key in (ndcg_key, hit_key, "mrr"):
        print(f"  {key}: {metrics.get(key, 0.0):.3f}")

    failures = []
    for key, floor in ((ndcg_key, args.min_ndcg), (hit_key, args.min_hit), ("mrr", args.min_mrr)):
        value = metrics.get(key, 0.0)
        if floor and value < floor:
            failures.append(f"{key} {value:.3f} < floor {floor:.3f}")

    if failures:
        print("\nREGRESSION:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("\nno regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
