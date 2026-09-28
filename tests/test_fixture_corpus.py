"""Guard the properties the synthetic corpus is relied on for.

The corpus exists so CI and the public evaluation set can run without private data
(ADR 0008), and it only earns that role if it keeps exercising the awkward cases. These
tests fail if someone tidies those cases away.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from marginalia.config import Config

FIXTURE_CONFIG = Path(__file__).resolve().parent.parent / "fixtures" / "config.yaml"
CORPUS = FIXTURE_CONFIG.parent / "corpus"
HEADING = re.compile(r"^#{1,6}\s+\S", re.M)


@pytest.fixture(scope="module")
def files() -> list[Path]:
    return sorted(Config.load(FIXTURE_CONFIG).scope().iter_files())


def test_corpus_is_discovered_through_the_real_scope_machinery(files):
    assert len(files) == 9


def test_filenames_with_spaces_are_handled(files):
    """The corpus this mirrors has 'Study Guide.md' and 'Coding Inventory.md'."""
    assert any(" " in f.name for f in files)


def test_documents_are_nested_not_flat(files):
    """Retrieval should not quietly depend on everything sitting in one directory."""
    assert len({f.parent for f in files}) >= 4


def test_one_document_has_no_headings(files):
    """Exercises the fixed-size fallback path in ADR 0007."""
    without = [f for f in files if not HEADING.findall(f.read_text())]
    assert len(without) == 1, "the unstructured document is the fallback test case"
    assert without[0].name == "Field Notes.md"


def test_maps_carry_numbered_sections(files):
    """The heading-path citation scheme depends on this shape."""
    nav = next(f for f in files if f.name == "Navigation.md")
    numbered = re.findall(r"^#{2,3}\s+\d+(?:\.\d+)?\.?\s+\S", nav.read_text(), re.M)
    assert len(numbered) >= 10


def test_inventories_are_dense_one_line_entries(files):
    """Inventories retrieve very differently from prose, so keep one in the fixture."""
    inv = next(f for f in files if f.name == "Instruments Inventory.md")
    bullets = [ln for ln in inv.read_text().splitlines() if ln.startswith("- **")]
    assert len(bullets) >= 12


def test_cross_references_use_the_section_idiom(files):
    """Citations are emitted as 'file §n', so the fixture must contain some to resolve."""
    refs = sum(len(re.findall(r"§\d", f.read_text())) for f in files)
    assert refs >= 10


def test_the_deliberate_contradiction_survives(files):
    """Two documents disagree on purpose; retrieval over a real wiki meets this."""
    nav = next(f for f in files if f.name == "Navigation.md").read_text()
    lights = next(f for f in files if f.name == "Lighthouses.md").read_text()
    assert "one half" in nav and "half" in nav
    assert "one third" in lights
    plan = next(f for f in files if f.name == "Study Plan.md").read_text()
    assert "unresolved" in plan.lower(), "the conflict should be logged, not silently fixed"


def test_corpus_contains_no_real_identifiers(files):
    """It is fiction, and the secret scanner should never have anything to find here."""
    import sys

    sys.path.insert(0, str(FIXTURE_CONFIG.parent.parent / "tools"))
    from scan_secrets import scan_paths

    findings, _, scanned = scan_paths(files)
    assert scanned == 9
    assert findings == []
