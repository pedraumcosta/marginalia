"""Tests for Markdown section parsing and chunking.

Weighted toward the cases that silently corrupt a section tree rather than raising:
a ``#`` inside a fenced code block, and a ``---`` line that is a horizontal rule in most
of this corpus but a setext underline in scraped articles.
"""

from __future__ import annotations

import pytest

from marginalia.chunking import (
    MIN_CHUNK_CHARS,
    chunk_document,
    parse_sections,
    strip_frontmatter,
)


def chunks(text: str, **kw):
    return chunk_document(text, doc_id="d1", doc_rel="notes/Doc.md", **kw)


def body(n: int = 80, word: str = "prose") -> str:
    return " ".join([word] * n)


# ---- frontmatter --------------------------------------------------------------


def test_frontmatter_is_stripped_and_counted():
    text = "---\ntitle: X\ntags: [a]\n---\n# Heading\n\nbody\n"
    out, removed = strip_frontmatter(text)
    assert out.startswith("# Heading")
    assert removed == 4


def test_unterminated_frontmatter_is_left_alone():
    text = "---\ntitle: X\n# Heading\n"
    out, removed = strip_frontmatter(text)
    assert out == text
    assert removed == 0


def test_document_not_starting_with_delimiter_is_untouched():
    text = "# Heading\n\n---\n\nmore\n"
    out, removed = strip_frontmatter(text)
    assert out == text and removed == 0


def test_line_numbers_account_for_stripped_frontmatter():
    text = "---\ntitle: X\n---\n## 1. First\n\n" + body()
    secs = [s for s in parse_sections(text) if s.title]
    assert secs[0].start_line == 4


# ---- fenced code blocks -------------------------------------------------------


def test_hash_inside_a_fence_is_not_a_heading():
    text = f"## 1. Real\n\n```python\n# not a heading\n#### also not\n```\n\n{body()}\n"
    titles = [s.title for s in parse_sections(text) if s.title]
    assert titles == ["1. Real"]


def test_tilde_fences_are_honoured():
    text = f"## 1. Real\n\n~~~\n# not a heading\n~~~\n\n{body()}\n"
    assert [s.title for s in parse_sections(text) if s.title] == ["1. Real"]


def test_a_shorter_run_does_not_close_a_longer_fence():
    text = f"## 1. Real\n\n````\n```\n# still inside\n````\n\n{body()}\n"
    assert [s.title for s in parse_sections(text) if s.title] == ["1. Real"]


def test_headings_after_a_closed_fence_are_found_again():
    text = f"## 1. One\n\n```\n# hidden\n```\n\n{body()}\n\n## 2. Two\n\n{body()}\n"
    assert [s.title for s in parse_sections(text) if s.title] == ["1. One", "2. Two"]


# ---- setext vs horizontal rule ------------------------------------------------


def test_dashes_after_blank_line_are_a_horizontal_rule():
    """This corpus uses '---' as a separator; treating it as a heading would be wrong."""
    text = f"## 1. Section\n\n{body()}\n\n---\n\n## 2. Next\n\n{body()}\n"
    titles = [s.title for s in parse_sections(text) if s.title]
    assert titles == ["1. Section", "2. Next"]


def test_dashes_directly_under_text_are_a_setext_heading():
    """Scraped articles in the same corpus use this form."""
    text = f"Zones\n---\n\n{body()}\n"
    secs = [s for s in parse_sections(text) if s.title]
    assert [s.title for s in secs] == ["Zones"]
    assert secs[0].level == 2


def test_equals_underline_is_a_setext_h1():
    text = f"Title\n===\n\n{body()}\n"
    secs = [s for s in parse_sections(text) if s.title]
    assert secs[0].level == 1


def test_a_list_item_above_dashes_is_not_a_heading():
    text = f"## 1. S\n\n- an item\n---\n\n{body()}\n"
    assert [s.title for s in parse_sections(text) if s.title] == ["1. S"]


# ---- section refs and citations -----------------------------------------------


@pytest.mark.parametrize(
    ("heading", "ref"),
    [
        ("## 6.5 Agent Memory", "6.5"),
        ("## 3. Dead reckoning", "3"),
        ("### 2.1 Meridian altitude", "2.1"),
        ("#### 1.2.3 Deeply numbered", "1.2.3"),
        ("## Unnumbered section", None),
        ("## 2026 in review", "2026"),
    ],
)
def test_section_reference_is_recovered_from_the_heading(heading, ref):
    text = f"{heading}\n\n{body()}\n"
    secs = [s for s in parse_sections(text) if s.title]
    assert secs[0].ref == ref


def test_citation_uses_the_section_idiom():
    out = chunks(f"# Doc\n\n## 6.5 Memory\n\n{body()}\n")
    memory = next(c for c in out if c.section_ref == "6.5")
    assert memory.citation == "notes/Doc.md §6.5"


def test_citation_falls_back_to_the_heading_path():
    out = chunks(f"# Doc\n\n## Development\n\n{body()}\n")
    c = next(c for c in out if c.heading_path and c.heading_path[-1] == "Development")
    assert c.citation == "notes/Doc.md > Development"


def test_citation_of_an_unstructured_document_is_the_path():
    out = chunks(body(200))
    assert out[0].citation.startswith("notes/Doc.md")


def test_split_chunks_declare_their_part_in_the_citation():
    out = chunks(f"## 1. Big\n\n{body(2000)}\n", max_chars=400)
    assert len(out) > 1
    assert out[0].citation.endswith(f"(1/{out[0].parts})")


# ---- heading paths ------------------------------------------------------------


def test_heading_path_accumulates_ancestors():
    text = f"# Map\n\n## 2. Celestial\n\n{body()}\n\n### 2.1 Meridian\n\n{body()}\n"
    out = chunks(text)
    leaf = next(c for c in out if c.section_ref == "2.1")
    assert leaf.heading_path == ("Map", "2. Celestial", "2.1 Meridian")


def test_sibling_does_not_inherit_the_previous_subtree():
    text = f"# Map\n\n## 1. A\n\n{body()}\n\n### 1.1 A-child\n\n{body()}\n\n## 2. B\n\n{body()}\n"
    out = chunks(text)
    b = next(c for c in out if c.section_ref == "2")
    assert b.heading_path == ("Map", "2. B")


def test_deeper_heading_after_shallower_nests_correctly():
    text = f"## 1. A\n\n{body()}\n\n#### 1.1.1 Deep\n\n{body()}\n"
    out = chunks(text)
    deep = next(c for c in out if c.section_ref == "1.1.1")
    assert deep.heading_path == ("1. A", "1.1.1 Deep")


# ---- container sections -------------------------------------------------------


def test_container_heading_with_no_prose_yields_no_chunk():
    """Its heading still travels in descendants' paths, so nothing is lost."""
    text = f"## 2. Container\n\n### 2.1 Real\n\n{body()}\n"
    out = chunks(text)
    refs = {c.section_ref for c in out}
    assert refs == {"2.1"}
    assert out[0].heading_path == ("2. Container", "2.1 Real")


def test_sections_shorter_than_the_minimum_are_dropped():
    text = f"## 1. Tiny\n\nshort\n\n## 2. Real\n\n{body()}\n"
    out = chunks(text, min_chars=MIN_CHUNK_CHARS)
    assert {c.section_ref for c in out} == {"2"}


def test_empty_document_yields_nothing():
    assert chunks("") == []
    assert chunks("\n\n   \n") == []


# ---- splitting within a section ----------------------------------------------


def test_oversized_section_is_split_into_parts():
    out = chunks(f"## 1. Big\n\n{body(3000)}\n", max_chars=500, overlap=50)
    assert len(out) > 1
    assert all(c.section_ref == "1" for c in out)
    assert {c.parts for c in out} == {len(out)}
    assert [c.part for c in out] == list(range(len(out)))


def test_splitting_never_crosses_a_heading_boundary():
    text = f"## 1. One\n\n{body(1500)}\n\n## 2. Two\n\n{body(1500)}\n"
    out = chunks(text, max_chars=400)
    for c in out:
        assert c.section_ref in ("1", "2")
        # A chunk from section 1 must not contain section 2's heading text.
        other = "2. Two" if c.section_ref == "1" else "1. One"
        assert other not in c.text


def test_split_windows_overlap_so_a_seam_fact_is_not_orphaned():
    out = chunks(f"## 1. Big\n\n{body(1200)}\n", max_chars=400, overlap=100)
    assert len(out) > 1
    # Consecutive windows should share some text.
    assert any(
        out[i].text[-40:] in out[i + 1].text or out[i + 1].text[:40] in out[i].text
        for i in range(len(out) - 1)
    )


def test_a_section_with_no_paragraph_breaks_still_splits():
    out = chunks("## 1. Wall\n\n" + ("x" * 5000) + "\n", max_chars=500)
    assert len(out) > 1


# ---- parent linkage -----------------------------------------------------------


def test_child_chunk_points_at_its_nearest_chunked_ancestor():
    text = f"# Map\n\n{body()}\n\n## 2. Parent\n\n{body()}\n\n### 2.1 Child\n\n{body()}\n"
    out = chunks(text)
    parent = next(c for c in out if c.section_ref == "2")
    child = next(c for c in out if c.section_ref == "2.1")
    assert child.parent_chunk_id == parent.chunk_id


def test_parent_linkage_skips_a_container_that_produced_no_chunk():
    text = f"# Map\n\n{body()}\n\n## 2. Container\n\n### 2.1 Child\n\n{body()}\n"
    out = chunks(text)
    root = next(c for c in out if c.heading_path == ("Map",))
    child = next(c for c in out if c.section_ref == "2.1")
    assert child.parent_chunk_id == root.chunk_id


def test_top_level_chunk_has_no_parent():
    out = chunks(f"# Map\n\n{body()}\n")
    assert out[0].parent_chunk_id is None


# ---- identity and metadata ----------------------------------------------------


def test_chunk_ids_are_stable_across_runs():
    text = f"## 1. A\n\n{body()}\n"
    assert [c.chunk_id for c in chunks(text)] == [c.chunk_id for c in chunks(text)]


def test_chunk_ids_are_unique_within_a_document():
    text = "".join(f"## {i}. S{i}\n\n{body()}\n\n" for i in range(1, 12))
    ids = [c.chunk_id for c in chunks(text)]
    assert len(ids) == len(set(ids))


def test_trust_tier_is_carried_onto_every_chunk():
    out = chunks(f"## 1. A\n\n{body()}\n", trust="upstream")
    assert {c.trust for c in out} == {"upstream"}


def test_line_numbers_are_recorded():
    out = chunks(f"# Map\n\n{body()}\n\n## 2. Next\n\n{body()}\n")
    nxt = next(c for c in out if c.section_ref == "2")
    assert nxt.start_line > 1
    assert nxt.end_line >= nxt.start_line


def test_preamble_before_the_first_heading_is_kept():
    text = f"{body()}\n\n## 1. Later\n\n{body()}\n"
    out = chunks(text)
    assert any(c.heading_path == () and "prose" in c.text for c in out)


# ---- regressions --------------------------------------------------------------


def test_early_break_point_with_large_overlap_terminates():
    """Regression: an early newline plus a large overlap once looped forever.

    A heading puts a newline ~10 chars in. The previous implementation advanced by
    ``cut - overlap``, which went negative, left the buffer unchanged, and appended
    windows until the process was killed.
    """
    out = chunks(f"## 1. Big\n\n{body(2000)}\n", max_chars=400, overlap=180)
    assert 1 < len(out) < 200


def test_overlap_larger_than_the_window_is_clamped():
    out = chunks(f"## 1. Big\n\n{body(500)}\n", max_chars=200, overlap=5000)
    assert 1 < len(out) < 200


def test_overlap_equal_to_max_chars_still_advances():
    out = chunks(f"## 1. Big\n\n{body(500)}\n", max_chars=300, overlap=300)
    assert 1 < len(out) < 200


def test_zero_overlap_is_allowed():
    out = chunks(f"## 1. Big\n\n{body(600)}\n", max_chars=300, overlap=0)
    assert len(out) > 1
    assert sum(len(c.text) for c in out) >= 300


def test_text_with_no_break_characters_at_all_terminates():
    out = chunks("## 1. Wall\n\n" + ("x" * 3000), max_chars=250, overlap=100)
    assert 1 < len(out) < 200


def test_windows_cover_the_whole_body():
    """Splitting may duplicate, but must not drop text."""
    marker_body = " ".join(f"w{i}" for i in range(400))
    out = chunks(f"## 1. Big\n\n{marker_body}\n", max_chars=400, overlap=80)
    joined = " ".join(c.text for c in out)
    for i in (0, 137, 399):
        assert f"w{i}" in joined


def test_split_remainder_shorter_than_the_minimum_is_dropped():
    """A windowing remainder can be a few characters; it must not reach the index."""
    out = chunks("## 1. Big\n\n" + ("y " * 900) + "tail\n", max_chars=400, overlap=40)
    assert out
    assert min(len(c.text.strip()) for c in out) >= MIN_CHUNK_CHARS


def test_a_document_that_is_only_a_runt_still_yields_one_chunk():
    """Dropping runts must not silently erase a short document entirely."""
    out = chunks("ab", min_chars=40)
    assert len(out) <= 1


# ---- citation readability -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("**Bold heading**", "Bold heading"),
        ("~~Struck out~~", "Struck out"),
        ("`code heading`", "code heading"),
        ("_Emphasised_", "Emphasised"),
        ("Mixed **bold** and `code`", "Mixed bold and code"),
        ("Indexed verticals *(depth deferred)*", "Indexed verticals"),
        ("Section *(added 2026-08-11)*", "Section"),
        ("Spaced   out    words", "Spaced out words"),
    ],
)
def test_heading_is_cleaned_for_display(raw, clean):
    from marginalia.chunking import clean_heading

    assert clean_heading(raw) == clean


def test_long_heading_is_truncated_in_a_citation():
    from marginalia.chunking import MAX_CITATION_SEGMENT, clean_heading

    out = clean_heading("A" * 200)
    assert len(out) == MAX_CITATION_SEGMENT
    assert out.endswith("…")


def test_heading_path_stays_verbatim_for_matching():
    """Only the citation is cleaned; the path is content a retriever can match on."""
    out = chunks(f"# Doc\n\n## **Bold** section\n\n{body()}\n")
    c = next(c for c in out if len(c.heading_path) > 1)
    assert c.heading_path[-1] == "**Bold** section"
    assert "**" not in c.citation
