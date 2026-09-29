"""Tests for embedders and the vector store."""

from __future__ import annotations

import numpy as np
import pytest

from marginalia.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
)
from marginalia.store import NumpyVectorStore, StoreError

# ---- embedders -----------------------------------------------------------------


def test_hashing_embedder_shape_and_normalisation():
    emb = HashingEmbedder(dim=64)
    out = emb.encode(["hello world", "goodbye moon"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic():
    """CI reproducibility depends on this, and on it being stable across processes."""
    a = HashingEmbedder(dim=64).encode(["the same text"])
    b = HashingEmbedder(dim=64).encode(["the same text"])
    np.testing.assert_array_equal(a, b)


def test_hashing_embedder_separates_different_text():
    emb = HashingEmbedder(dim=256)
    out = emb.encode(["navigation by celestial observation", "baking sourdough bread"])
    assert float(out[0] @ out[1]) < 0.3


def test_hashing_embedder_finds_lexical_overlap():
    emb = HashingEmbedder(dim=256)
    out = emb.encode(["compass deviation and variation", "deviation of the compass"])
    assert float(out[0] @ out[1]) > 0.4


def test_hashing_embedder_is_not_semantic():
    """Recorded as a property, not a defect: it is why it must not be the default."""
    emb = HashingEmbedder(dim=256)
    out = emb.encode(["car", "automobile"])
    assert float(out[0] @ out[1]) == pytest.approx(0.0, abs=1e-6)


def test_empty_text_gives_a_zero_vector_not_nan():
    out = HashingEmbedder(dim=32).encode(["", "!!! ???"])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 0.0)


def test_encoding_nothing_is_allowed():
    assert HashingEmbedder(dim=16).encode([]).shape == (0, 16)


def test_hashing_dim_must_be_positive():
    with pytest.raises(EmbeddingError):
        HashingEmbedder(dim=0)


def test_signature_identifies_the_vector_space():
    assert HashingEmbedder(dim=64).signature != HashingEmbedder(dim=128).signature
    assert SentenceTransformerEmbedder(model="a").signature != (
        SentenceTransformerEmbedder(model="b").signature
    )


# ---- factory -------------------------------------------------------------------


def test_factory_defaults_to_the_real_model_not_the_test_one():
    """The dependency-free embedder is a testing affordance; defaulting to it would
    quietly degrade anyone who forgot to configure."""
    assert make_embedder({}).name == "sentence-transformers"
    assert make_embedder(None).name == "sentence-transformers"


def test_factory_builds_the_hashing_embedder_on_request():
    emb = make_embedder({"provider": "hashing", "dim": 32})
    assert emb.name == "hashing"
    assert emb.dim == 32


def test_factory_rejects_an_unknown_provider():
    with pytest.raises(EmbeddingError, match="unknown embeddings provider"):
        make_embedder({"provider": "magic"})


def test_factory_never_offers_a_hosted_option():
    """ADR 0003: there is deliberately no configuration path to hosted embeddings."""
    for hosted in ("openai", "cohere", "voyage", "anthropic", "api"):
        with pytest.raises(EmbeddingError):
            make_embedder({"provider": hosted})


def test_missing_sentence_transformers_gives_actionable_advice(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_st(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_st)
    with pytest.raises(EmbeddingError, match=r"\[embed\]"):
        SentenceTransformerEmbedder().encode(["x"])


# ---- store: writes -------------------------------------------------------------


def vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_upsert_then_search_returns_the_nearest():
    store = NumpyVectorStore(dim=8)
    v = vectors(5, 8)
    store.upsert([f"c{i}" for i in range(5)], v)
    hits = store.search(v[2], k=1)
    assert hits[0][0] == "c2"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_upsert_overwrites_rather_than_duplicating():
    store = NumpyVectorStore(dim=4)
    store.upsert(["a"], vectors(1, 4, seed=1))
    store.upsert(["a"], vectors(1, 4, seed=2))
    assert len(store) == 1


def test_upsert_id_count_must_match_vector_count():
    store = NumpyVectorStore(dim=4)
    with pytest.raises(StoreError, match="ids for"):
        store.upsert(["a", "b"], vectors(1, 4))


def test_upsert_rejects_the_wrong_dimension():
    store = NumpyVectorStore(dim=4)
    with pytest.raises(StoreError, match="dim 4"):
        store.upsert(["a"], vectors(1, 7))


def test_upsert_of_nothing_is_a_no_op():
    store = NumpyVectorStore(dim=4)
    store.upsert([], np.zeros((0, 4), np.float32))
    assert len(store) == 0


def test_delete_removes_and_keeps_the_rest_searchable():
    store = NumpyVectorStore(dim=8)
    v = vectors(4, 8)
    store.upsert(["a", "b", "c", "d"], v)
    store.delete(["b"])
    assert len(store) == 3
    assert "b" not in store.ids
    assert store.search(v[2], k=1)[0][0] == "c"


def test_delete_of_an_absent_id_is_harmless():
    store = NumpyVectorStore(dim=4)
    store.upsert(["a"], vectors(1, 4))
    store.delete(["nope"])
    assert len(store) == 1


def test_clear_empties_the_store():
    store = NumpyVectorStore(dim=4)
    store.upsert(["a", "b"], vectors(2, 4))
    store.clear()
    assert len(store) == 0
    assert store.search(vectors(1, 4)[0], k=3) == []


# ---- store: reads --------------------------------------------------------------


def test_search_on_an_empty_store_is_empty():
    assert NumpyVectorStore(dim=4).search(vectors(1, 4)[0], k=5) == []


def test_search_respects_k_and_orders_by_similarity():
    store = NumpyVectorStore(dim=16)
    v = vectors(20, 16)
    store.upsert([f"c{i}" for i in range(20)], v)
    hits = store.search(v[0], k=5)
    assert len(hits) == 5
    assert hits[0][0] == "c0"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_search_k_larger_than_the_corpus_is_fine():
    store = NumpyVectorStore(dim=4)
    store.upsert(["a", "b"], vectors(2, 4))
    assert len(store.search(vectors(1, 4)[0], k=99)) == 2


def test_search_with_zero_k_returns_nothing():
    store = NumpyVectorStore(dim=4)
    store.upsert(["a"], vectors(1, 4))
    assert store.search(vectors(1, 4)[0], k=0) == []


def test_allow_filter_restricts_candidates():
    store = NumpyVectorStore(dim=8)
    v = vectors(5, 8)
    store.upsert([f"c{i}" for i in range(5)], v)
    hits = store.search(v[0], k=5, allow={"c3", "c4"})
    assert {cid for cid, _ in hits} == {"c3", "c4"}


def test_empty_allow_filter_returns_nothing():
    store = NumpyVectorStore(dim=8)
    store.upsert(["a"], vectors(1, 8))
    assert store.search(vectors(1, 8)[0], k=3, allow=set()) == []


def test_search_rejects_a_mismatched_query_dimension():
    store = NumpyVectorStore(dim=8)
    store.upsert(["a"], vectors(1, 8))
    with pytest.raises(StoreError, match="query dim"):
        store.search(vectors(1, 3)[0], k=1)


# ---- store: persistence --------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    store = NumpyVectorStore(dim=8, signature="hashing:8")
    v = vectors(6, 8)
    store.upsert([f"c{i}" for i in range(6)], v)
    store.save(tmp_path)

    again = NumpyVectorStore.load(tmp_path)
    assert again.ids == store.ids
    assert again.signature == "hashing:8"
    np.testing.assert_allclose(again.search(v[1], k=1)[0][1], 1.0, atol=1e-5)


def test_load_from_an_empty_directory_fails(tmp_path):
    with pytest.raises(StoreError, match="no vector store"):
        NumpyVectorStore.load(tmp_path)


def test_load_detects_an_inconsistent_sidecar(tmp_path):
    store = NumpyVectorStore(dim=4, signature="s")
    store.upsert(["a", "b"], vectors(2, 4))
    store.save(tmp_path)
    # Edit the JSON rather than the text: indent=0 puts each id on its own line, so a
    # naive string replacement silently matches nothing and the test proves nothing.
    import json as _json

    sidecar = tmp_path / "vectors.json"
    meta = _json.loads(sidecar.read_text())
    meta["ids"] = meta["ids"][:1]
    sidecar.write_text(_json.dumps(meta))
    with pytest.raises(StoreError, match="inconsistent"):
        NumpyVectorStore.load(tmp_path)


def test_open_or_create_discards_on_signature_change(tmp_path):
    """Vectors from two embedders are not comparable; mixing them is wrong but not
    obviously broken, so the old ones are discarded rather than migrated."""
    old = NumpyVectorStore(dim=4, signature="hashing:4")
    old.upsert(["a"], vectors(1, 4))
    old.save(tmp_path)

    fresh = NumpyVectorStore.open_or_create(tmp_path, dim=4, signature="other:4")
    assert len(fresh) == 0


def test_open_or_create_discards_on_dim_change(tmp_path):
    old = NumpyVectorStore(dim=4, signature="s")
    old.upsert(["a"], vectors(1, 4))
    old.save(tmp_path)
    assert len(NumpyVectorStore.open_or_create(tmp_path, dim=8, signature="s")) == 0


def test_open_or_create_reuses_a_matching_store(tmp_path):
    old = NumpyVectorStore(dim=4, signature="s")
    old.upsert(["a"], vectors(1, 4))
    old.save(tmp_path)
    assert len(NumpyVectorStore.open_or_create(tmp_path, dim=4, signature="s")) == 1


def test_open_or_create_survives_a_corrupt_sidecar(tmp_path):
    (tmp_path / "vectors.json").write_text("{not json")
    (tmp_path / "vectors.npy").write_bytes(b"garbage")
    assert len(NumpyVectorStore.open_or_create(tmp_path, dim=4, signature="s")) == 0


def test_stats_reports_the_backend_and_size():
    store = NumpyVectorStore(dim=8, signature="hashing:8")
    store.upsert(["a", "b"], vectors(2, 8))
    st = store.stats()
    assert st["backend"] == "numpy-exact"
    assert st["count"] == 2 and st["dim"] == 8
    assert int(st["bytes"]) > 0


def test_dimension_works_under_either_library_method_name():
    """sentence-transformers renamed this; the store signature depends on it."""

    class NewApi:
        def get_embedding_dimension(self):
            return 384

    class OldApi:
        def get_sentence_embedding_dimension(self):
            return 768

    for stub, expected in ((NewApi(), 384), (OldApi(), 768)):
        emb = SentenceTransformerEmbedder()
        emb._model = stub  # noqa: SLF001
        assert emb.dim == expected
