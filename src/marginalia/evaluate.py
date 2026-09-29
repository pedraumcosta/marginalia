"""Measuring retrieval, and caching the measurements.

Two golden sets, per ADR 0008: a public one over the synthetic corpus, committed and run
in CI, and a private one over the real corpus that cannot be committed because the
question–answer pairs *are* extracts of the material they test.

Retrieval runs are cached to disk and metrics are computed from the cache, so
recomputing a number never re-embeds anything. That split is borrowed from a
knowledge-graph study in the same collection whose ``analysis/run_all.py`` reproduces
every published figure in about ten seconds on CPU while the pipeline that produced them
needs a GPU and the corpus.

Relevance is judged on *document plus section*, not on the citation string: citations
carry a part suffix for split sections, and pinning a golden set to "(2/6)" would make it
break whenever chunk sizing changes, which is a knob we expect to turn.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

RETRIEVERS = ("lexical", "dense", "fused")
DEFAULT_K = 10


class EvalError(Exception):
    pass


# ---------------------------------------------------------------------------
# golden sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One passage a question should retrieve."""

    doc: str
    section: str | None = None
    heading: str | None = None

    def matches(self, row: Mapping[str, object] | sqlite3.Row) -> bool:
        if str(row["doc_rel"]) != self.doc:
            return False
        if self.section is not None:
            return str(row["section_ref"] or "") == self.section
        if self.heading is not None:
            try:
                path = json.loads(str(row["heading_path"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                path = []
            return any(self.heading.lower() in str(h).lower() for h in path)
        # Document-level target: any chunk of the document counts.
        return True

    def label(self) -> str:
        if self.section:
            return f"{self.doc} §{self.section}"
        if self.heading:
            return f"{self.doc} > {self.heading}"
        return self.doc


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    targets: tuple[Target, ...]
    tags: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.targets:
            raise EvalError(f"question {self.id!r} has no targets, so it can never be scored")


def load_questions(path: Path) -> list[Question]:
    """Load a golden set from YAML."""
    if not path.is_file():
        raise EvalError(f"no golden set at {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or []
    except yaml.YAMLError as exc:
        raise EvalError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Sequence):
        raise EvalError(f"{path}: expected a list of questions")

    questions: list[Question] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise EvalError(f"{path}: entry {i} is not a mapping")
        qid = str(entry.get("id") or f"q{i + 1}")
        if qid in seen:
            raise EvalError(f"{path}: duplicate question id {qid!r}")
        seen.add(qid)
        text = str(entry.get("question") or "").strip()
        if not text:
            raise EvalError(f"{path}: question {qid!r} has no text")

        targets: list[Target] = []
        for t in entry.get("relevant") or []:
            if isinstance(t, str):
                targets.append(Target(doc=t))
            elif isinstance(t, Mapping):
                targets.append(
                    Target(
                        doc=str(t.get("doc") or ""),
                        section=None if t.get("section") is None else str(t["section"]),
                        heading=None if t.get("heading") is None else str(t["heading"]),
                    )
                )
        questions.append(
            Question(
                id=qid,
                question=text,
                targets=tuple(targets),
                tags=tuple(str(x) for x in (entry.get("tags") or ())),
                note=str(entry.get("note") or ""),
            )
        )
    if not questions:
        raise EvalError(f"{path}: golden set is empty")
    return questions


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def reciprocal_rank(positions: Sequence[int]) -> float:
    """1 / rank of the first relevant result. 0 when none was retrieved."""
    return 1.0 / min(positions) if positions else 0.0


def hit_rate(positions: Sequence[int], k: int) -> float:
    """Did anything relevant land in the top k. Binary, per query."""
    return 1.0 if any(p <= k for p in positions) else 0.0


def recall_at_k(target_positions: Sequence[int], total_relevant: int, k: int) -> float:
    """Fraction of the distinct relevant *targets* that made the top k.

    Takes one position per target -- the rank where that target was first found -- not
    every matching chunk. A target names a document section, a section may be split
    across several chunks, and counting each matching chunk made this exceed 1.0 on a
    corpus with split sections (observed at 1.583 before the fix).
    """
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for p in target_positions if p <= k)
    return min(found, total_relevant) / total_relevant


def precision_at_k(positions: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for p in positions if p <= k) / k


def ndcg_at_k(target_positions: Sequence[int], total_relevant: int, k: int) -> float:
    """Binary-relevance NDCG over distinct targets.

    Gain 1 per target found, discounted by log2(rank + 1) at the rank where it was first
    retrieved. Like recall, this counts targets rather than matching chunks: summing a gain
    per chunk let a split section earn several gains for one target and pushed the result
    above 1.0 (observed at 1.139 before the fix).
    """
    if total_relevant <= 0 or k <= 0:
        return 0.0
    ranks = sorted(p for p in target_positions if p <= k)[:total_relevant]
    dcg = sum(1.0 / math.log2(p + 1) for p in ranks)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@dataclass
class QueryOutcome:
    """One question against one retriever: what came back, and where the answers were."""

    question_id: str
    question: str
    retriever: str
    ranked: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    # One entry per distinct target that was found, at the rank where it first appeared.
    # Kept separate from `positions` (every matching chunk) because recall and NDCG are
    # about targets while precision and MRR are about retrieved results.
    target_positions: list[int] = field(default_factory=list)
    total_relevant: int = 0
    targets: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.positions)

    @property
    def first_rank(self) -> int | None:
        return min(self.positions) if self.positions else None

    def to_json(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "retriever": self.retriever,
            "ranked": self.ranked,
            "citations": self.citations,
            "positions": self.positions,
            "target_positions": self.target_positions,
            "total_relevant": self.total_relevant,
            "targets": self.targets,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> QueryOutcome:
        return cls(
            question_id=str(data["question_id"]),
            question=str(data.get("question", "")),
            retriever=str(data.get("retriever", "")),
            ranked=[str(x) for x in (data.get("ranked") or [])],
            citations=[str(x) for x in (data.get("citations") or [])],
            positions=[int(x) for x in (data.get("positions") or [])],
            target_positions=[int(x) for x in (data.get("target_positions") or [])],
            total_relevant=int(data.get("total_relevant") or 0),
            targets=[str(x) for x in (data.get("targets") or [])],
        )


def aggregate(outcomes: Sequence[QueryOutcome], k: int = DEFAULT_K) -> dict[str, float]:
    """Mean metrics over a set of outcomes."""
    if not outcomes:
        return {}
    n = len(outcomes)
    return {
        f"hit@{k}": sum(hit_rate(o.positions, k) for o in outcomes) / n,
        f"recall@{k}": sum(recall_at_k(o.target_positions, o.total_relevant, k) for o in outcomes)
        / n,
        f"precision@{k}": sum(precision_at_k(o.positions, k) for o in outcomes) / n,
        "mrr": sum(reciprocal_rank(o.positions) for o in outcomes) / n,
        f"ndcg@{k}": sum(ndcg_at_k(o.target_positions, o.total_relevant, k) for o in outcomes) / n,
    }


def evaluate_question(
    retriever,
    index,
    question: Question,
    *,
    mode: str = "fused",
    k: int = DEFAULT_K,
) -> QueryOutcome:
    """Run one question through one retrieval mode and locate the relevant results."""
    if mode == "lexical":
        pairs = retriever.lexical(question.question, k)
        ranked = [cid for cid, _ in pairs]
    elif mode == "dense":
        pairs = retriever.dense(question.question, k)
        ranked = [cid for cid, _ in pairs]
    elif mode == "fused":
        ranked = [h.chunk_id for h in retriever.search(question.question, k=k)]
    else:
        raise EvalError(f"unknown retrieval mode {mode!r}. Known: {', '.join(RETRIEVERS)}")

    rows = index.chunks_by_id(ranked)
    citations: list[str] = []
    positions: list[int] = []
    first_seen: dict[str, int] = {}
    for rank, cid in enumerate(ranked, start=1):
        row = rows.get(cid)
        if row is None:
            continue
        citations.append(str(row["citation"]))
        matched = [t for t in question.targets if t.matches(row)]
        if matched:
            positions.append(rank)
            for t in matched:
                first_seen.setdefault(t.label(), rank)

    return QueryOutcome(
        question_id=question.id,
        question=question.question,
        retriever=mode,
        ranked=ranked,
        citations=citations,
        positions=positions,
        target_positions=sorted(first_seen.values()),
        total_relevant=len(question.targets),
        targets=[t.label() for t in question.targets],
    )


@dataclass
class EvalRun:
    """A complete run: every question against every requested retrieval mode."""

    profile: str
    embedder: str
    k: int
    corpus_chunks: int
    outcomes: list[QueryOutcome] = field(default_factory=list)
    created_at: str = ""
    golden_set: str = ""

    def by_retriever(self) -> dict[str, list[QueryOutcome]]:
        grouped: dict[str, list[QueryOutcome]] = {}
        for o in self.outcomes:
            grouped.setdefault(o.retriever, []).append(o)
        return grouped

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: aggregate(group, self.k) for name, group in self.by_retriever().items()}

    def to_json(self) -> dict:
        return {
            "profile": self.profile,
            "embedder": self.embedder,
            "k": self.k,
            "corpus_chunks": self.corpus_chunks,
            "created_at": self.created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "golden_set": self.golden_set,
            "outcomes": [o.to_json() for o in self.outcomes],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> EvalRun:
        if not path.is_file():
            raise EvalError(f"no cached run at {path}")
        data = json.loads(path.read_text())
        return cls(
            profile=str(data.get("profile", "")),
            embedder=str(data.get("embedder", "")),
            k=int(data.get("k", DEFAULT_K)),
            corpus_chunks=int(data.get("corpus_chunks", 0)),
            outcomes=[QueryOutcome.from_json(o) for o in data.get("outcomes", [])],
            created_at=str(data.get("created_at", "")),
            golden_set=str(data.get("golden_set", "")),
        )


def run_eval(
    retriever,
    index,
    questions: Sequence[Question],
    *,
    modes: Sequence[str] = RETRIEVERS,
    k: int = DEFAULT_K,
    profile: str = "",
    embedder: str = "",
) -> EvalRun:
    """Run every question against every mode. Modes needing an absent retriever are skipped."""
    usable = [m for m in modes if m != "dense" or retriever.has_dense]
    if "fused" in usable and not retriever.has_dense:
        # Without vectors, "fused" would be lexical wearing a different label, and
        # reporting it as a distinct number would overstate what was measured.
        usable = [m for m in usable if m != "fused"]
    if not usable:
        raise EvalError("no usable retrieval mode: the index has no chunks")

    run = EvalRun(
        profile=profile,
        embedder=embedder,
        k=k,
        corpus_chunks=index.stats().chunks,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    for mode in usable:
        for q in questions:
            run.outcomes.append(evaluate_question(retriever, index, q, mode=mode, k=k))
    return run


def render_summary(run: EvalRun) -> str:
    """A table comparing retrieval modes, plus the queries nothing found."""
    summary = run.summary()
    if not summary:
        return "no outcomes"

    metric_names = list(next(iter(summary.values())).keys())
    width = max(len(m) for m in metric_names) + 2
    lines = [
        f"profile={run.profile or '-'}  embedder={run.embedder or '-'}  "
        f"k={run.k}  chunks={run.corpus_chunks}",
        f"questions: {len(run.outcomes) // max(1, len(summary))}",
        "",
        "  " + "retriever".ljust(12) + "".join(m.rjust(width) for m in metric_names),
    ]
    for name in RETRIEVERS:
        if name not in summary:
            continue
        row = summary[name]
        lines.append(
            "  " + name.ljust(12) + "".join(f"{row[m]:.3f}".rjust(width) for m in metric_names)
        )

    for name, group in run.by_retriever().items():
        missed = [o for o in group if not o.found]
        if missed:
            lines.append("")
            lines.append(f"  {name}: {len(missed)} question(s) found nothing relevant")
            for o in missed[:5]:
                lines.append(f"    - {o.question_id}: {o.question[:64]}")
    return "\n".join(lines)
