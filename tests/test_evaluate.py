"""Tests for the evaluation harness.

The metric tests use hand-computed values. A metric that is quietly wrong is worse than
no metric at all, because every later decision gets justified by it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from marginalia.chunking import chunk_document
from marginalia.documents import Document, document_id
from marginalia.embeddings import HashingEmbedder
from marginalia.evaluate import (
    DEFAULT_K,
    EvalError,
    EvalRun,
    QueryOutcome,
    Question,
    Target,
    aggregate,
    evaluate_question,
    hit_rate,
    load_questions,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_summary,
    run_eval,
)
from marginalia.index import Index
from marginalia.retrieve import Retriever
from marginalia.store import NumpyVectorStore

PUBLIC_SET = Path(__file__).resolve().parent.parent / "eval" / "public" / "questions.yaml"


# ---- metrics: reciprocal rank --------------------------------------------------


@pytest.mark.parametrize(
    ("positions", "expected"),
    [([1], 1.0), ([2], 0.5), ([4], 0.25), ([3, 7], 1 / 3), ([], 0.0)],
)
def test_reciprocal_rank(positions, expected):
    assert reciprocal_rank(positions) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_first_relevant_not_the_best():
    assert reciprocal_rank([5, 2, 9]) == pytest.approx(0.5)


# ---- metrics: hit rate ---------------------------------------------------------


def test_hit_rate_is_binary():
    assert hit_rate([3], 5) == 1.0
    assert hit_rate([7], 5) == 0.0
    assert hit_rate([], 5) == 0.0


def test_hit_rate_boundary_is_inclusive():
    assert hit_rate([5], 5) == 1.0
    assert hit_rate([6], 5) == 0.0


# ---- metrics: recall and precision ---------------------------------------------


def test_recall_counts_relevant_found_over_relevant_total():
    assert recall_at_k([1, 3], total_relevant=2, k=5) == pytest.approx(1.0)
    assert recall_at_k([1], total_relevant=2, k=5) == pytest.approx(0.5)
    assert recall_at_k([1, 9], total_relevant=2, k=5) == pytest.approx(0.5)


def test_recall_with_no_relevant_passages_is_zero_not_an_error():
    assert recall_at_k([], total_relevant=0, k=5) == 0.0


def test_precision_counts_relevant_found_over_k():
    assert precision_at_k([1, 2], k=10) == pytest.approx(0.2)
    assert precision_at_k([], k=10) == 0.0
    assert precision_at_k([1], k=0) == 0.0


# ---- metrics: NDCG -------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1], total_relevant=1, k=10) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], total_relevant=2, k=10) == pytest.approx(1.0)


def test_ndcg_matches_the_formula_by_hand():
    """One relevant passage at rank 3: DCG = 1/log2(4) = 0.5, ideal = 1/log2(2) = 1."""
    assert ndcg_at_k([3], total_relevant=1, k=10) == pytest.approx(0.5)


def test_ndcg_second_position_by_hand():
    """Rank 2: DCG = 1/log2(3) ~= 0.6309."""
    assert ndcg_at_k([2], total_relevant=1, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_two_relevant_split_by_hand():
    """Relevant at 1 and 4 out of 2 total.

    DCG = 1/log2(2) + 1/log2(5) = 1 + 0.4307 = 1.4307
    ideal = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    """
    dcg = 1.0 + 1 / math.log2(5)
    ideal = 1.0 + 1 / math.log2(3)
    assert ndcg_at_k([1, 4], total_relevant=2, k=10) == pytest.approx(dcg / ideal)


def test_ndcg_ignores_results_past_k():
    assert ndcg_at_k([11], total_relevant=1, k=10) == 0.0


def test_ndcg_degrades_monotonically_with_rank():
    scores = [ndcg_at_k([p], total_relevant=1, k=10) for p in range(1, 11)]
    assert scores == sorted(scores, reverse=True)


def test_ndcg_ideal_is_capped_at_k():
    """With more relevant passages than slots, a perfect top-k must still score 1."""
    assert ndcg_at_k([1, 2, 3], total_relevant=10, k=3) == pytest.approx(1.0)


def test_ndcg_with_nothing_relevant_is_zero():
    assert ndcg_at_k([], total_relevant=3, k=10) == 0.0
    assert ndcg_at_k([1], total_relevant=0, k=10) == 0.0


# ---- targets -------------------------------------------------------------------


def row(doc_rel: str, section: str | None = None, path: list[str] | None = None) -> dict:
    return {
        "doc_rel": doc_rel,
        "section_ref": section,
        "heading_path": json.dumps(path or []),
    }


def test_section_target_matches_document_and_section():
    t = Target(doc="a/B.md", section="6.5")
    assert t.matches(row("a/B.md", "6.5"))
    assert not t.matches(row("a/B.md", "6.6"))
    assert not t.matches(row("other.md", "6.5"))


def test_section_target_ignores_the_part_suffix():
    """Citations carry '(2/6)'; pinning a golden set to it would break on resizing."""
    t = Target(doc="a/B.md", section="6.5")
    assert t.matches(row("a/B.md", "6.5"))


def test_heading_target_matches_anywhere_in_the_path():
    t = Target(doc="a/B.md", heading="Timekeeping")
    assert t.matches(row("a/B.md", None, ["Inventory", "2. Timekeeping"]))
    assert not t.matches(row("a/B.md", None, ["Inventory", "1. Angle-measuring"]))


def test_heading_target_is_case_insensitive():
    assert Target(doc="a.md", heading="timekeeping").matches(row("a.md", None, ["Timekeeping"]))


def test_document_target_matches_any_chunk():
    t = Target(doc="notes/Field Notes.md")
    assert t.matches(row("notes/Field Notes.md", None, []))
    assert not t.matches(row("notes/Other.md", None, []))


def test_target_with_malformed_heading_path_does_not_raise():
    t = Target(doc="a.md", heading="x")
    assert not t.matches({"doc_rel": "a.md", "section_ref": None, "heading_path": "{bad"})


def test_target_labels_are_readable():
    assert Target("a.md", section="1.2").label() == "a.md §1.2"
    assert Target("a.md", heading="Intro").label() == "a.md > Intro"
    assert Target("a.md").label() == "a.md"


# ---- golden set loading --------------------------------------------------------


def test_public_golden_set_loads():
    qs = load_questions(PUBLIC_SET)
    assert len(qs) >= 20
    assert all(q.targets for q in qs)


def test_public_golden_set_has_multi_target_and_unstructured_cases():
    qs = load_questions(PUBLIC_SET)
    assert any(len(q.targets) > 1 for q in qs), "cross-document retrieval must be measured"
    assert any("unstructured" in q.tags for q in qs), "the no-headings path must be measured"
    assert any("contradiction" in q.tags for q in qs)


def test_question_without_targets_is_rejected():
    with pytest.raises(EvalError, match="never be scored"):
        Question(id="x", question="y", targets=())


def test_missing_golden_set_is_an_error(tmp_path):
    with pytest.raises(EvalError, match="no golden set"):
        load_questions(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- [unclosed\n")
    with pytest.raises(EvalError, match="invalid YAML"):
        load_questions(p)


def test_empty_golden_set_is_rejected(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("[]\n")
    with pytest.raises(EvalError, match="empty"):
        load_questions(p)


def test_duplicate_question_ids_are_rejected(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(
        "- id: a\n  question: x\n  relevant: [d.md]\n- id: a\n  question: y\n  relevant: [d.md]\n"
    )
    with pytest.raises(EvalError, match="duplicate"):
        load_questions(p)


def test_question_without_text_is_rejected(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  relevant: [d.md]\n")
    with pytest.raises(EvalError, match="no text"):
        load_questions(p)


def test_string_shorthand_becomes_a_document_target(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- id: a\n  question: x\n  relevant:\n    - some/Doc.md\n")
    q = load_questions(p)[0]
    assert q.targets == (Target(doc="some/Doc.md"),)


# ---- running against a real index ----------------------------------------------

CORPUS = {
    "nav/Navigation.md": (
        "# Navigation\n\nEntry point for position finding, a map not a notebook.\n\n"
        "## 1. The problem\n\nLongitude requires knowing the time at a reference meridian, "
        "which is why it remained unsolved for three centuries while latitude was routine.\n\n"
        "## 5.3 Compass deviation\n\nThe vessel's own iron deflects the compass. Deviation "
        "belongs to the ship; variation belongs to the Earth and is printed on the chart.\n"
    ),
    "charts/Cartography.md": (
        "# Cartography\n\nHow charts are made and what they distort.\n\n"
        "## 2. What to preserve\n\nFor navigation the answer is angle, so Mercator remains "
        "the sea chart even though its area distortion is notorious.\n"
    ),
}


@pytest.fixture
def harness(tmp_path):
    index = Index.open(tmp_path / "idx")
    for rel, text in CORPUS.items():
        doc = Document(
            path=Path("/c") / rel,
            rel=rel,
            root=Path("/c"),
            doc_id=document_id(rel),
            content_hash="h" + rel,
            size=len(text),
            trust="authored",
        )
        index.put_document(
            doc, chunk_document(text, doc_id=doc.doc_id, doc_rel=rel, trust="authored")
        )
    index.commit()

    emb = HashingEmbedder(dim=128)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    from marginalia.retrieve import build_dense_index

    build_dense_index(index, store, emb)
    yield Retriever(index, store, emb, candidates=20), index
    index.close()


def test_evaluate_question_finds_the_target(harness):
    retriever, index = harness
    q = Question(
        id="longitude",
        question="why was longitude unsolved",
        targets=(Target(doc="nav/Navigation.md", section="1"),),
    )
    out = evaluate_question(retriever, index, q, mode="lexical", k=5)
    assert out.found
    assert out.first_rank == 1
    assert out.total_relevant == 1
    assert out.targets == ["nav/Navigation.md §1"]


def test_evaluate_question_records_a_miss_without_failing(harness):
    retriever, index = harness
    q = Question(
        id="absent",
        question="sourdough fermentation schedule",
        targets=(Target(doc="nav/Navigation.md", section="9.9"),),
    )
    out = evaluate_question(retriever, index, q, mode="lexical", k=5)
    assert not out.found
    assert out.first_rank is None


def test_unknown_mode_is_rejected(harness):
    retriever, index = harness
    q = Question(id="a", question="x", targets=(Target(doc="nav/Navigation.md"),))
    with pytest.raises(EvalError, match="unknown retrieval mode"):
        evaluate_question(retriever, index, q, mode="telepathy")


def test_run_eval_covers_every_mode_and_question(harness):
    retriever, index = harness
    qs = [
        Question("a", "longitude reference meridian", (Target("nav/Navigation.md", "1"),)),
        Question("b", "Mercator angle", (Target("charts/Cartography.md", "2"),)),
    ]
    run = run_eval(retriever, index, qs, k=5, profile="t", embedder="hashing:128")
    assert set(run.by_retriever()) == {"lexical", "dense", "fused"}
    assert len(run.outcomes) == 6
    assert run.corpus_chunks > 0


def test_fused_is_omitted_without_vectors(harness):
    """Reporting fused-without-dense as a separate number overstates what was measured."""
    _, index = harness
    lexical_only = Retriever(index)
    qs = [Question("a", "longitude", (Target("nav/Navigation.md", "1"),))]
    run = run_eval(lexical_only, index, qs, k=5)
    assert set(run.by_retriever()) == {"lexical"}


def test_run_eval_on_an_empty_index_raises(tmp_path):
    index = Index.open(tmp_path / "empty")
    qs = [Question("a", "x", (Target("d.md"),))]
    with pytest.raises(EvalError, match="no usable retrieval mode"):
        run_eval(Retriever(index), index, qs, modes=("dense",))
    index.close()


# ---- aggregation and caching ---------------------------------------------------


def outcome(positions, total=1, retriever="lexical", target_positions=None) -> QueryOutcome:
    return QueryOutcome(
        question_id="q",
        question="?",
        retriever=retriever,
        positions=list(positions),
        target_positions=list(target_positions if target_positions is not None else positions),
        total_relevant=total,
    )


def test_aggregate_averages_over_questions():
    agg = aggregate([outcome([1]), outcome([])], k=10)
    assert agg["hit@10"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([], k=10) == {}


def test_aggregate_reports_every_metric():
    agg = aggregate([outcome([1])], k=DEFAULT_K)
    assert set(agg) == {
        f"hit@{DEFAULT_K}",
        f"recall@{DEFAULT_K}",
        f"precision@{DEFAULT_K}",
        "mrr",
        f"ndcg@{DEFAULT_K}",
    }


def test_run_round_trips_through_the_cache(tmp_path, harness):
    retriever, index = harness
    qs = [Question("a", "longitude", (Target("nav/Navigation.md", "1"),))]
    run = run_eval(retriever, index, qs, k=5, profile="p", embedder="e")
    path = tmp_path / "run.json"
    run.save(path)

    again = EvalRun.load(path)
    assert again.profile == "p"
    assert again.embedder == "e"
    assert again.k == 5
    assert again.summary() == run.summary(), "metrics must recompute from cache identically"


def test_loading_an_absent_cache_is_an_error(tmp_path):
    with pytest.raises(EvalError, match="no cached run"):
        EvalRun.load(tmp_path / "nope.json")


def test_metrics_recompute_from_cache_without_retrieval(tmp_path):
    """The point of the cache: a number can be recomputed with no index present."""
    run = EvalRun(profile="p", embedder="e", k=5, corpus_chunks=10, outcomes=[outcome([2])])
    path = tmp_path / "r.json"
    run.save(path)
    assert EvalRun.load(path).summary()["lexical"]["mrr"] == pytest.approx(0.5)


def test_render_summary_shows_modes_and_misses():
    run = EvalRun(
        profile="p",
        embedder="e",
        k=5,
        corpus_chunks=1,
        outcomes=[outcome([1], retriever="lexical"), outcome([], retriever="dense")],
    )
    text = render_summary(run)
    assert "lexical" in text and "dense" in text
    assert "found nothing relevant" in text


def test_render_summary_of_an_empty_run():
    assert render_summary(EvalRun(profile="", embedder="", k=5, corpus_chunks=0)) == "no outcomes"


# ---- metric bounds (regression) -------------------------------------------------


def test_recall_cannot_exceed_one_when_one_target_matches_many_chunks():
    """Regression: a target names a document section, and a section may be split across
    several chunks. Counting each matching chunk gave recall 1.583 on the real corpus."""
    assert recall_at_k([1], total_relevant=1, k=5) == pytest.approx(1.0)
    assert recall_at_k([1, 2, 3, 4, 5], total_relevant=1, k=5) <= 1.0


def test_ndcg_cannot_exceed_one_when_one_target_matches_many_chunks():
    """Regression: ndcg reached 1.139 for the same reason."""
    assert ndcg_at_k([1, 2, 3, 4, 5], total_relevant=1, k=5) <= 1.0


@pytest.mark.parametrize("total", [1, 2, 3, 5])
@pytest.mark.parametrize("positions", [[1], [1, 2], [1, 2, 3, 4, 5], [2, 4], [5]])
def test_recall_and_ndcg_are_always_in_range(positions, total):
    r = recall_at_k(positions, total_relevant=total, k=5)
    n = ndcg_at_k(positions, total_relevant=total, k=5)
    assert 0.0 <= r <= 1.0, f"recall {r} out of range"
    assert 0.0 <= n <= 1.0, f"ndcg {n} out of range"


def test_aggregate_metrics_are_all_in_range():
    outcomes = [
        outcome([1, 2, 3, 4, 5], total=1, target_positions=[1]),
        outcome([2], total=2, target_positions=[2]),
        outcome([], total=3, target_positions=[]),
    ]
    agg = aggregate(outcomes, k=5)
    for name, value in agg.items():
        assert 0.0 <= value <= 1.0, f"{name} = {value} out of range"


def test_target_positions_record_the_first_rank_per_target(harness):
    """One entry per distinct target, at the rank it first appeared."""
    retriever, index = harness
    q = Question(
        id="both",
        question="longitude meridian Mercator angle",
        targets=(
            Target(doc="nav/Navigation.md", section="1"),
            Target(doc="charts/Cartography.md", section="2"),
        ),
    )
    out = evaluate_question(retriever, index, q, mode="lexical", k=10)
    assert len(out.target_positions) <= out.total_relevant
    assert out.target_positions == sorted(out.target_positions)
    assert recall_at_k(out.target_positions, out.total_relevant, 10) <= 1.0


def test_precision_still_counts_retrieved_chunks_not_targets():
    """Precision is about what was returned, so every matching chunk counts there."""
    assert precision_at_k([1, 2, 3], k=10) == pytest.approx(0.3)
