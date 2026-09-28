# Harbour Practice — a synthetic knowledge wiki

This corpus exists so the test suite and the public evaluation set can run without
touching anybody's real notes. It is invented in full: the harbours, instruments, people
and measurements below are fiction.

It deliberately imitates the *shape* of a hand-maintained wiki rather than a pile of
scraped articles, because that shape is what the retriever is built to exploit:

- **Maps** carry numbered sections and prose (`Navigation.md`, `Cartography.md`, `Lighthouses.md`).
- **Inventories** are one-line-per-item catalogues, which retrieve very differently from prose.
- **Cross-references** use the `file §n` idiom, so citation handling has something to chew on.
- Some facts appear in **two places with different emphasis**, and two are in **open conflict**
  (see `Navigation.md` §5.2 against `Lighthouses.md` §3.1) — retrieval over a real wiki has to
  cope with both, so the fixture has both.
- `Field Notes.md` has **no headings at all**, exercising the fallback path in ADR 0007.

## Documents

| File | What it answers |
|---|---|
| `navigation/Navigation.md` | How position is found at sea |
| `navigation/Instruments Inventory.md` | What instruments exist, one line each |
| `navigation/Study Plan.md` | What to read in what order |
| `navigation/Field Notes.md` | Unstructured working notes |
| `cartography/Cartography.md` | How charts are made |
| `cartography/Projections Inventory.md` | Which projection for which purpose |
| `lighthouses/Lighthouses.md` | How lights are built, run and identified |
| `lighthouses/Lighthouse Inventory.md` | The lights of the fictional coast |
