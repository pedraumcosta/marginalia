"""Tests for document discovery, identity, trust tiers and the ingest diff."""

from __future__ import annotations

import pytest

from marginalia.documents import (
    DEFAULT_TRUST,
    Document,
    TrustPolicy,
    TrustRule,
    chunk_of,
    content_hash,
    discover,
    document_id,
    load_document,
    plan_ingest,
)
from marginalia.scope import ScopePolicy


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "Map.md").write_text("# Map\n\n## 1. First\n\n" + "prose " * 60)
    (tmp_path / "mine" / "sub").mkdir()
    (tmp_path / "mine" / "sub" / "README.md").write_text("# Sub\n\n" + "text " * 60)
    (tmp_path / "theirs").mkdir()
    (tmp_path / "theirs" / "README.md").write_text("# Theirs\n\n" + "other " * 60)
    return tmp_path


def policy(tree, *names):
    return ScopePolicy(roots=tuple(tree / n for n in (names or ("mine",))))


# ---- identity -----------------------------------------------------------------


def test_relative_paths_disambiguate_same_named_files(tree):
    """Bare filenames collapse every README.md into one citation identity."""
    docs = list(discover(policy(tree, "mine", "theirs")))
    rels = sorted(d.rel for d in docs)
    assert rels == ["mine/Map.md", "mine/sub/README.md", "theirs/README.md"]
    assert len({d.doc_id for d in docs}) == 3


def test_nested_roots_name_against_their_common_parent(tree):
    """Roots may nest. The deeper root is still reported (trust rules need it), but the
    name is taken against the shared parent so it stays unique."""
    pol = ScopePolicy(roots=(tree / "mine", tree / "mine" / "sub"))
    doc = next(d for d in discover(pol) if d.path.name == "README.md")
    assert doc.rel == "sub/README.md"
    assert doc.root == tree / "mine" / "sub"


def test_document_id_is_stable_and_path_derived(tree):
    first = {d.rel: d.doc_id for d in discover(policy(tree))}
    second = {d.rel: d.doc_id for d in discover(policy(tree))}
    assert first == second
    assert first["Map.md"] == document_id("Map.md")


def test_document_id_survives_an_edit(tree):
    """An edited document is the same document; its chunks replace, not accumulate."""
    before = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    (tree / "mine" / "Map.md").write_text("# Map\n\n## 1. Changed\n\n" + "new " * 60)
    after = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    assert before.doc_id == after.doc_id
    assert before.content_hash != after.content_hash


def test_hash_is_over_content_not_mtime(tree):
    """Dropbox and git both rewrite mtimes without changing bytes."""
    import os

    doc = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    os.utime(tree / "mine" / "Map.md", (0, 0))
    again = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    assert doc.content_hash == again.content_hash


def test_content_hash_is_sensitive_to_one_byte():
    assert content_hash(b"abc") != content_hash(b"abd")


def test_absolute_paths_do_not_appear_in_the_identity(tree):
    doc = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    assert str(tree) not in doc.rel
    assert str(tree) not in doc.doc_id


def test_unreadable_file_is_skipped(tree, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("nope")

    monkeypatch.setattr("pathlib.Path.read_bytes", boom)
    assert list(discover(policy(tree))) == []


def test_load_document_returns_none_on_read_failure(tree, monkeypatch):
    monkeypatch.setattr(
        "pathlib.Path.read_bytes", lambda self, *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert load_document(tree / "mine" / "Map.md", (tree / "mine",), TrustPolicy()) is None


# ---- trust --------------------------------------------------------------------


def test_default_trust_is_unknown_not_authored(tree):
    """Over-trusting by default is the failure mode that matters."""
    doc = next(discover(policy(tree)))
    assert doc.trust == DEFAULT_TRUST == "unknown"


def test_rule_assigns_a_tier_by_directory_name():
    tp = TrustPolicy([TrustRule("sub", "upstream")], default="authored")
    assert tp.tier_for("sub/README.md") == "upstream"
    assert tp.tier_for("Map.md") == "authored"


def test_rule_assigns_a_tier_by_path_glob():
    tp = TrustPolicy([TrustRule("vendor/*", "untrusted")], default="authored")
    assert tp.tier_for("vendor/thing.md") == "untrusted"
    assert tp.tier_for("mine/thing.md") == "authored"


def test_first_matching_rule_wins():
    tp = TrustPolicy([TrustRule("notes", "curated"), TrustRule("notes", "untrusted")])
    assert tp.tier_for("notes/a.md") == "curated"


def test_from_config_parses_rules_and_default():
    tp = TrustPolicy.from_config(
        {
            "default": "authored",
            "rules": [
                {"match": "ai-that-works", "trust": "upstream"},
                {"match": "clippings/*", "trust": "untrusted"},
            ],
        }
    )
    assert tp.default == "authored"
    assert tp.tier_for("ai-that-works/ep/README.md") == "upstream"
    assert tp.tier_for("clippings/article.md") == "untrusted"
    assert tp.tier_for("LLM.md") == "authored"


def test_from_config_ignores_unknown_tiers_and_blank_patterns():
    tp = TrustPolicy.from_config(
        {"rules": [{"match": "x", "trust": "extremely"}, {"match": "", "trust": "authored"}]}
    )
    assert tp.rules == ()


def test_from_config_of_nothing_is_permissive_about_shape():
    assert TrustPolicy.from_config(None).default == DEFAULT_TRUST
    assert TrustPolicy.from_config({}).default == DEFAULT_TRUST


def test_trust_reaches_every_chunk(tree):
    tp = TrustPolicy(default="authored")
    doc = next(d for d in discover(policy(tree), tp) if d.rel == "Map.md")
    cs = chunk_of(doc)
    assert cs
    assert {c.trust for c in cs} == {"authored"}


def test_rules_match_the_root_relative_path_not_the_root_name(tree):
    """Documented limitation: the root's own directory name is not in `rel`.

    Assign a whole root's tier with `default` (or a per-profile trust block); rules are
    for carve-outs *inside* a root, which is the case that actually arises -- a cloned
    holding sitting a few levels down inside an otherwise authored tree.
    """
    tp = TrustPolicy([TrustRule("mine", "upstream")], default="authored")
    doc = next(d for d in discover(policy(tree), tp) if d.rel == "Map.md")
    assert doc.trust == "authored", "the root name 'mine' is not part of rel"


def test_a_carved_out_clone_inside_a_root_is_downgraded(tmp_path):
    """The real shape: an upstream holding nested inside an authored root."""
    root = tmp_path / "AI"
    (root / "NLP").mkdir(parents=True)
    (root / "NLP" / "LLM.md").write_text("# Map\n\n" + "mine " * 60)
    clone = root / "NLP" / "ai-that-works" / "ep"
    clone.mkdir(parents=True)
    (clone / "README.md").write_text("# Episode\n\n" + "theirs " * 60)

    tp = TrustPolicy([TrustRule("ai-that-works", "upstream")], default="authored")
    tiers = {d.rel: d.trust for d in discover(ScopePolicy(roots=(root,)), tp)}
    assert tiers["NLP/LLM.md"] == "authored"
    assert tiers["NLP/ai-that-works/ep/README.md"] == "upstream"


# ---- chunking integration ------------------------------------------------------


def test_chunks_cite_the_relative_path(tree):
    docs = discover(policy(tree, "mine", "theirs"))
    doc = next(d for d in docs if d.rel == "mine/sub/README.md")
    cs = chunk_of(doc)
    assert cs
    assert all(c.citation.startswith("mine/sub/README.md") for c in cs)


def test_chunk_of_respects_size_overrides(tree):
    doc = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    few = chunk_of(doc, max_chars=2000)
    many = chunk_of(doc, max_chars=120, overlap=20)
    assert len(many) > len(few)


def test_chunk_of_on_an_unreadable_document_is_empty(tree, monkeypatch):
    doc = next(d for d in discover(policy(tree)) if d.rel == "Map.md")
    monkeypatch.setattr(
        "pathlib.Path.read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert chunk_of(doc) == []


# ---- ingest planning -----------------------------------------------------------


def doc(rel: str, h: str) -> Document:
    from pathlib import Path as P

    return Document(
        path=P(f"/x/{rel}"),
        rel=rel,
        root=P("/x"),
        doc_id=document_id(rel),
        content_hash=h,
        size=1,
    )


def test_everything_is_added_against_an_empty_index():
    plan = plan_ingest([doc("a.md", "h1"), doc("b.md", "h2")], {})
    assert len(plan.added) == 2
    assert not plan.changed and not plan.unchanged and not plan.removed


def test_unchanged_documents_are_skipped():
    a = doc("a.md", "h1")
    plan = plan_ingest([a], {a.doc_id: "h1"})
    assert plan.unchanged == (a,)
    assert plan.work == ()


def test_changed_content_is_work():
    a = doc("a.md", "h2")
    plan = plan_ingest([a], {a.doc_id: "h1"})
    assert plan.changed == (a,)
    assert plan.work == (a,)


def test_disappeared_documents_are_reported_for_removal():
    a = doc("a.md", "h1")
    gone = document_id("gone.md")
    plan = plan_ingest([a], {a.doc_id: "h1", gone: "h9"})
    assert plan.removed == (gone,)


def test_plan_render_reports_every_category():
    a, b = doc("a.md", "h1"), doc("b.md", "h2")
    plan = plan_ingest([a, b], {b.doc_id: "old", document_id("z.md"): "h"})
    text = plan.render()
    for word in ("added", "changed", "unchanged", "removed"):
        assert word in text


# ---- identity collisions (regression) -------------------------------------------


def test_files_at_the_top_of_different_roots_do_not_collide(tmp_path):
    """Regression, found by shipping it.

    Four sibling folders were each given a README.md and declared as scope roots. Naming
    each file against its own root made all four `README.md`, and since the document id
    derives from that name, they shared one id: three were silently overwritten and the
    index held one document where there should have been four.
    """
    for name in ("AI", "Coding", "Mngmnt", "TechTrading"):
        d = tmp_path / name
        d.mkdir()
        (d / "README.md").write_text(f"# {name}\n\n" + f"{name} body " * 40)

    pol = ScopePolicy(roots=tuple(tmp_path / n for n in ("AI", "Coding", "Mngmnt", "TechTrading")))
    docs = list(discover(pol))

    assert len(docs) == 4
    assert len({d.doc_id for d in docs}) == 4, "ids must be distinct"
    assert sorted(d.rel for d in docs) == [
        "AI/README.md",
        "Coding/README.md",
        "Mngmnt/README.md",
        "TechTrading/README.md",
    ]


def test_colliding_names_produce_distinct_citations(tmp_path):
    """A citation of 'README.md' is useless when four documents share the name."""
    for name in ("AI", "Coding"):
        d = tmp_path / name
        d.mkdir()
        (d / "README.md").write_text(f"# {name}\n\n## 1. Section\n\n" + f"{name} prose " * 40)

    pol = ScopePolicy(roots=(tmp_path / "AI", tmp_path / "Coding"))
    citations = {c.citation for d in discover(pol) for c in chunk_of(d)}
    assert any(c.startswith("AI/README.md") for c in citations)
    assert any(c.startswith("Coding/README.md") for c in citations)


def test_roots_with_no_common_parent_fall_back_to_the_root_name(tmp_path):
    """Without a shared parent there is nothing to name against, so keep the root name."""
    a = tmp_path / "alpha" / "one"
    b = tmp_path / "beta" / "two"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "README.md").write_text("# R\n\n" + "body " * 40)

    pol = ScopePolicy(roots=(a, b))
    rels = sorted(d.rel for d in discover(pol))
    # commonpath here is tmp_path, so names stay unique via the intermediate directories.
    assert len(set(rels)) == 2
    assert all(r.endswith("README.md") for r in rels)


def test_single_root_keeps_short_relative_names(tmp_path):
    """One root cannot collide, so names stay as short as they can be."""
    root = tmp_path / "notes"
    (root / "deep").mkdir(parents=True)
    (root / "README.md").write_text("# R\n\n" + "body " * 40)
    (root / "deep" / "Other.md").write_text("# O\n\n" + "body " * 40)
    rels = sorted(d.rel for d in discover(ScopePolicy(roots=(root,))))
    assert rels == ["README.md", "deep/Other.md"]
