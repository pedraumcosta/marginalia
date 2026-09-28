"""Tests for the index, lexical search, and hybrid fusion."""

from __future__ import annotations

import numpy as np
import pytest

from marginalia.chunking import chunk_document
from marginalia.documents import Document, document_id
from marginalia.embeddings import HashingEmbedder
from marginalia.index import Index, IndexError_, fts_query
from marginalia.retrieve import RRF_K, Retriever, build_dense_index, reciprocal_rank_fusion
from marginalia.store import NumpyVectorStore

DOCS = {
    "navigation/Navigation.md": (
        "# Navigation\n\n"
        "Entry point for position-finding at sea. A map rather than a notebook, holding "
        "framings and citations into the instrument catalogue.\n\n"
        "## 1. The problem\n\n"
        "Latitude follows from the altitude of a known body. Longitude requires knowing "
        "the time at a reference meridian, which is why it stayed unsolved.\n\n"
        "## 5.3 Compass deviation\n\n"
        "The vessel's own iron deflects the compass. Deviation is not variation: variation "
        "is a property of the Earth and appears on the chart.\n"
    ),
    "cartography/Cartography.md": (
        "# Cartography\n\n"
        "## 2. What a chart must preserve\n\n"
        "For navigation the answer is angle. The Mercator projection remains the sea chart "
        "because a straight line on it is a rhumb line of constant bearing.\n"
    ),
    "lighthouses/Lighthouses.md": (
        "# Lighthouses\n\n"
        "## 3.1 Nominal and actual range\n\n"
        "Nominal range assumes clear air. In haze, expect about one third of the nominal "
        "range for a coastal light.\n"
    ),
}


def make_doc(rel: str, text: str, trust: str = "authored") -> Document:
    from pathlib import Path

    return Document(
        path=Path("/corpus") / rel,
        rel=rel,
        root=Path("/corpus"),
        doc_id=document_id(rel),
        content_hash=f"hash-of-{rel}-{len(text)}",
        size=len(text),
        trust=trust,
    )


@pytest.fixture
def index(tmp_path):
    idx = Index.open(tmp_path / "idx")
    for rel, text in DOCS.items():
        doc = make_doc(rel, text)
        chunks = chunk_document(text, doc_id=doc.doc_id, doc_rel=rel, trust=doc.trust)
        idx.put_document(doc, chunks)
    idx.commit()
    yield idx
    idx.close()


# ---- FTS5 query building -------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        'a "quoted" phrase',
        "NEAR(a b)",
        "col:value",
        "trailing*",
        "a AND b OR c NOT d",
        "paren(theses)",
        "hyphen-ated term",
        "^caret",
        'unbalanced "quote',
        "semi; colon",
        "back\\slash",
    ],
)
def test_hostile_input_becomes_a_safe_expression(index, hostile):
    """Raw user text in a MATCH clause is a syntax error, not a search."""
    assert index.search_lexical(hostile, k=3) is not None


def test_query_terms_are_quoted_and_ored():
    assert fts_query("compass deviation") == '"compass" OR "deviation"'


def test_punctuation_only_query_yields_no_expression():
    assert fts_query("!!! ???") == ""
    assert fts_query("") == ""
    assert fts_query(None) == ""  # type: ignore[arg-type]


def test_empty_expression_searches_nothing_rather_than_erroring(index):
    assert index.search_lexical("!!!", k=5) == []


def test_unicode_terms_survive():
    assert fts_query("variação magnética") == '"variação" OR "magnética"'


# ---- lexical search ------------------------------------------------------------


def test_lexical_finds_the_right_section(index):
    hits = index.search_lexical("compass deviation variation", k=3)
    assert hits
    row = index.chunk(hits[0][0])
    assert row["section_ref"] == "5.3"


def test_lexical_scores_are_higher_is_better(index):
    """SQLite's bm25() is negative and lower-is-better; every retriever here agrees that
    bigger means better, so mixing the conventions would invert ranking silently."""
    hits = index.search_lexical("range chart navigation", k=5)
    assert len(hits) >= 2
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0


def test_lexical_respects_k(index):
    assert len(index.search_lexical("the", k=2)) <= 2


def test_lexical_with_zero_k_is_empty(index):
    assert index.search_lexical("compass", k=0) == []


def test_porter_stemming_matches_inflections(index):
    """'assumes' should find 'assume'-family text; the tokenizer choice is load-bearing."""
    assert index.search_lexical("assumed", k=5)


def test_headings_are_searchable(index):
    hits = index.search_lexical("Nominal", k=5)
    assert any(index.chunk(cid)["section_ref"] == "3.1" for cid, _ in hits)


# ---- index writes and incrementality -------------------------------------------


def test_known_hashes_round_trips(index):
    known = index.known_hashes()
    assert len(known) == len(DOCS)
    assert all(v.startswith("hash-of-") for v in known.values())


def test_put_document_replaces_rather_than_accumulating(index):
    rel = "navigation/Navigation.md"
    new_text = "# Replaced\n\n## 9. Only section\n\n" + "fresh words " * 30
    doc = make_doc(rel, new_text)
    index.put_document(
        doc, chunk_document(new_text, doc_id=doc.doc_id, doc_rel=rel, trust="authored")
    )
    index.commit()

    assert index.stats().documents == len(DOCS), "no duplicate document row"
    refs = {
        r["section_ref"]
        for r in index.conn.execute("SELECT section_ref FROM chunks WHERE doc_rel = ?", (rel,))
    }
    assert refs == {"9"}, f"old sections survived: {refs}"
    # And gone from the FTS table too, not just from `chunks`.
    assert index.search_lexical("unsolved reference meridian", k=5) == []


def test_dropping_a_document_removes_it_from_fts_too(index):
    rel = "lighthouses/Lighthouses.md"
    gone = index.drop_documents([document_id(rel)])
    index.commit()
    assert gone
    assert index.search_lexical("haze nominal", k=5) == []
    assert index.stats().documents == len(DOCS) - 1


def test_dropping_nothing_is_a_no_op(index):
    assert index.drop_documents([]) == []


def test_stats_counts_trust_tiers(tmp_path):
    idx = Index.open(tmp_path / "i")
    for rel, tier in (("a.md", "authored"), ("b.md", "upstream")):
        text = "# T\n\n## 1. S\n\n" + "words " * 40
        doc = make_doc(rel, text, trust=tier)
        idx.put_document(doc, chunk_document(text, doc_id=doc.doc_id, doc_rel=rel, trust=tier))
    idx.commit()
    assert idx.stats().trust == {"authored": 1, "upstream": 1}
    idx.close()


def test_schema_version_mismatch_refuses_to_open(tmp_path):
    idx = Index.open(tmp_path / "i")
    idx.set_meta("schema_version", "999")
    idx.commit()
    idx.close()
    with pytest.raises(IndexError_, match="schema version"):
        Index.open(tmp_path / "i")


def test_meta_round_trips(tmp_path):
    with Index.open(tmp_path / "i") as idx:
        assert idx.get_meta("nope") is None
        idx.set_meta("k", "v1")
        idx.set_meta("k", "v2")
        assert idx.get_meta("k") == "v2"


def test_reopening_preserves_contents(tmp_path, index):
    path = tmp_path / "idx"
    index.commit()
    with Index.open(path) as again:
        assert again.stats().documents == len(DOCS)


# ---- reciprocal rank fusion ----------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    """An item both retrievers rank mid beats items only one ranks first."""
    fused = reciprocal_rank_fusion({"bm25": ["x", "a", "b"], "dense": ["y", "a", "b"]})
    assert [cid for cid, _, _, _ in fused][0] == "a"


def test_rrf_mildly_favours_polarising_items_over_middling_ones():
    """A property worth knowing, not a defect.

    ``1 / (k + rank)`` is convex in rank, so an item ranked 1st by one retriever and 3rd
    by the other scores *above* an item both rank 2nd: 1/61 + 1/63 > 2/62. Under reversed
    rankings the extremes tie with each other and beat the middle. It means RRF gives a
    slight edge to items one retriever loves over items both find merely acceptable.
    """
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["c", "b", "a"]})
    scores = {cid: score for cid, score, _, _ in fused}
    assert scores["a"] == pytest.approx(scores["c"])
    assert scores["a"] > scores["b"]


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion({"x": ["a"]}, k=RRF_K)
    assert fused[0][1] == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_records_which_retrievers_found_each_item():
    fused = reciprocal_rank_fusion({"bm25": ["a"], "dense": ["a", "b"]})
    by_id = {cid: (srcs, ranks) for cid, _, srcs, ranks in fused}
    assert set(by_id["a"][0]) == {"bm25", "dense"}
    assert by_id["a"][1] == {"bm25": 1, "dense": 1}
    assert set(by_id["b"][0]) == {"dense"}


def test_rrf_is_deterministic_on_ties():
    a = reciprocal_rank_fusion({"x": ["p", "q"], "y": ["q", "p"]})
    b = reciprocal_rank_fusion({"x": ["p", "q"], "y": ["q", "p"]})
    assert [i for i, *_ in a] == [i for i, *_ in b]


def test_rrf_of_one_ranking_preserves_its_order():
    fused = reciprocal_rank_fusion({"only": ["a", "b", "c"]})
    assert [cid for cid, *_ in fused] == ["a", "b", "c"]


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"x": []}) == []


# ---- hybrid retrieval ----------------------------------------------------------


@pytest.fixture
def hybrid(index, tmp_path):
    emb = HashingEmbedder(dim=128)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    n = build_dense_index(index, store, emb)
    assert n > 0
    return Retriever(index, store, emb, candidates=20)


def test_hybrid_has_dense_when_vectors_exist(hybrid):
    assert hybrid.has_dense


def test_lexical_only_retriever_reports_no_dense(index):
    assert not Retriever(index).has_dense


def test_hybrid_search_returns_hits_with_citations(hybrid):
    hits = hybrid.search("compass deviation", k=3)
    assert hits
    assert hits[0].citation.startswith("navigation/Navigation.md")
    assert hits[0].trust == "authored"


def test_hits_record_which_retrievers_matched(hybrid):
    hits = hybrid.search("Mercator rhumb line", k=3)
    assert hits
    assert any(h.found_by_both for h in hits)


def test_hybrid_beats_lexical_only_on_a_paraphrase(index, hybrid):
    """Not a quality claim about the hashing embedder -- just that fusion consults both."""
    hits = hybrid.search("angle preserving projection for sea charts", k=3)
    assert hits
    assert all(h.retrievers for h in hits), "every hit must record its source"
    assert any("Cartography" in h.doc_rel for h in hits)


def test_search_with_empty_query_is_empty(hybrid):
    assert hybrid.search("   ", k=5) == []
    assert hybrid.search("anything", k=0) == []


def test_search_respects_k(hybrid):
    assert len(hybrid.search("the range of a light", k=2)) <= 2


def test_snippet_is_flattened_and_bounded(hybrid):
    hit = hybrid.search("compass", k=1)[0]
    s = hit.snippet(width=50)
    assert "\n" not in s
    assert len(s) <= 50


def test_with_context_expands_to_the_parent_section(hybrid, index):
    hits = [h for h in hybrid.search("deviation variation", k=5) if h.parent_chunk_id]
    assert hits, "expected at least one hit with a parent"
    hit = hits[0]
    expanded = hybrid.with_context(hit)
    assert len(expanded) > len(hit.text)
    assert hit.text in expanded


def test_with_context_of_a_parentless_hit_is_itself(hybrid):
    hits = hybrid.search("navigation", k=8)
    orphan = next((h for h in hits if not h.parent_chunk_id), None)
    if orphan is not None:
        assert hybrid.with_context(orphan) == orphan.text


def test_trust_preference_only_breaks_ties(index):
    """It must not promote a poor match merely for being well-sourced."""
    emb = HashingEmbedder(dim=128)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    build_dense_index(index, store, emb)
    plain = Retriever(index, store, emb).search("compass deviation", k=5)
    trusted = Retriever(index, store, emb, prefer_trust=True).search("compass deviation", k=5)
    assert plain[0].chunk_id == trusted[0].chunk_id


# ---- dense index building ------------------------------------------------------


def test_build_dense_index_embeds_every_chunk(index):
    emb = HashingEmbedder(dim=64)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    n = build_dense_index(index, store, emb)
    assert n == index.stats().chunks == len(store)


def test_build_dense_index_can_be_restricted(index):
    emb = HashingEmbedder(dim=64)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    subset = index.all_chunk_ids()[:2]
    assert build_dense_index(index, store, emb, only=subset) == 2
    assert len(store) == 2


def test_build_dense_index_on_an_empty_selection_does_nothing(index):
    emb = HashingEmbedder(dim=64)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    assert build_dense_index(index, store, emb, only=[]) == 0


def test_dense_search_alone_returns_vectors_not_rows(hybrid):
    pairs = hybrid.dense("haze nominal range", k=3)
    assert pairs
    assert all(isinstance(s, float) for _, s in pairs)


def test_dense_search_without_a_store_is_empty(index):
    assert Retriever(index).dense("anything", k=3) == []


def test_embedding_batches_do_not_change_the_result(index):
    emb = HashingEmbedder(dim=64)
    one = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    many = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    build_dense_index(index, one, emb, batch_size=1)
    build_dense_index(index, many, emb, batch_size=1000)
    q = emb.encode(["compass deviation"])[0]
    assert [c for c, _ in one.search(q, k=5)] == [c for c, _ in many.search(q, k=5)]


def test_vectors_are_float32_to_keep_the_index_small(index):
    emb = HashingEmbedder(dim=64)
    store = NumpyVectorStore(dim=emb.dim, signature=emb.signature)
    build_dense_index(index, store, emb)
    assert np.dtype(np.float32) == store._matrix.dtype  # noqa: SLF001


# ---- stopword handling ---------------------------------------------------------


def test_function_words_are_dropped_from_the_expression():
    """IDF discounts them when scoring; it does not keep them out of candidate selection."""
    assert fts_query("what did I conclude about agentic RAG") == '"conclude" OR "agentic" OR "RAG"'


def test_single_characters_are_dropped():
    assert fts_query("a b compass") == '"compass"'


def test_an_all_stopword_query_is_honoured_literally():
    """Silence would be a worse answer than a loose one."""
    assert fts_query("who and why") == '"who" OR "and" OR "why"'


def test_stopword_filtering_is_case_insensitive():
    assert fts_query("The Compass") == '"Compass"'


def test_domain_terms_are_not_mistaken_for_stopwords():
    for term in ("range", "chart", "light", "memory", "index", "trust", "scope"):
        assert term in fts_query(f"the {term}")


def test_stopwords_improve_a_natural_language_query(index):
    """The noisy-question case that motivated the filter."""
    hits = index.search_lexical("what did I conclude about the Mercator projection", k=3)
    assert hits
    top = index.chunk(hits[0][0])
    assert "Cartography" in str(top["doc_rel"])
