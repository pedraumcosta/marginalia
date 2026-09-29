"""Discovering documents, identifying them, and deciding how far to trust them.

Three jobs, all of which the chunker assumes somebody else has done:

**Root-relative paths.** A citation has to name a document unambiguously. Using the bare
filename collapses every ``README.md`` in the corpus into one identity — on the corpus this
was built against that merged several distinct files under a single citation. Paths are
therefore relative to the scope root that matched, which also keeps absolute machine paths
out of the index.

**Content hashing.** The index is derived state (ADR 0001), so re-indexing must be cheap
and must skip what has not changed. The hash is over content, not mtime: Dropbox and git
both rewrite mtimes without changing bytes.

**Trust tiers.** A corpus that mixes your own writing with scraped third-party articles
mixes text you authored with text an attacker could have authored. Persistent retrieval
turns a one-shot injection into a durable one, so provenance is recorded at ingestion
rather than guessed at query time.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .chunking import Chunk, chunk_document
from .scope import ScopePolicy

# Ordered from most to least trusted. A retriever may prefer higher tiers when scores
# are close, and a generator should treat anything below "authored" as data, not instruction.
TRUST_TIERS: tuple[str, ...] = ("authored", "curated", "upstream", "untrusted", "unknown")
DEFAULT_TRUST = "unknown"

READ_ERRORS = (OSError, UnicodeError)


@dataclass(frozen=True)
class Document:
    """One source file, identified by content."""

    path: Path
    rel: str
    root: Path
    doc_id: str
    content_hash: str
    size: int
    trust: str = DEFAULT_TRUST

    @property
    def citation_name(self) -> str:
        return self.rel


@dataclass(frozen=True)
class TrustRule:
    """Assign a trust tier to paths matching a glob, relative to the root."""

    pattern: str
    tier: str

    def matches(self, rel: str) -> bool:
        if "/" in self.pattern:
            return fnmatch(rel, self.pattern)
        return any(fnmatch(part, self.pattern) for part in Path(rel).parts)


class TrustPolicy:
    """Resolve a document's trust tier from configuration.

    First matching rule wins, so the config reads top-down like a firewall. Absent any
    rule the tier is "unknown" rather than "authored": over-trusting by default is the
    failure mode that matters.
    """

    def __init__(self, rules: Sequence[TrustRule] = (), default: str = DEFAULT_TRUST) -> None:
        self.rules = tuple(rules)
        self.default = default if default in TRUST_TIERS else DEFAULT_TRUST

    @classmethod
    def from_config(cls, data: Mapping[str, object] | None) -> TrustPolicy:
        if not data:
            return cls()
        raw = data.get("rules") or []
        rules: list[TrustRule] = []
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            for entry in raw:
                if isinstance(entry, Mapping):
                    pattern = str(entry.get("match", "")).strip()
                    tier = str(entry.get("trust", DEFAULT_TRUST)).strip().lower()
                    if pattern and tier in TRUST_TIERS:
                        rules.append(TrustRule(pattern, tier))
        default = str(data.get("default", DEFAULT_TRUST)).strip().lower()
        return cls(rules, default)

    def tier_for(self, rel: str) -> str:
        for rule in self.rules:
            if rule.matches(rel):
                return rule.tier
        return self.default

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrustPolicy(rules={len(self.rules)}, default={self.default!r})"


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_id(rel: str) -> str:
    """Stable identity for a document, derived from its relative path.

    Deliberately not derived from content: a document that is edited is the same document,
    and its chunks should replace the previous ones rather than accumulate beside them.
    """
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]


def display_base(roots: Sequence[Path]) -> Path | None:
    """The directory that relative paths are named against.

    With several roots this is their common parent, so a file sitting *at* the top of a
    root keeps that root's name: four roots each holding a ``README.md`` must not all be
    called ``README.md``. Naming each file against its own root did exactly that, and
    since the document id is derived from the relative path, the four collapsed into one
    and three were silently overwritten.

    Returns None when the roots share no common parent, in which case the caller falls
    back to prefixing the root's own name.
    """
    if not roots:
        return None
    if len(roots) == 1:
        return roots[0]
    try:
        return Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError:
        # Different drives, or an empty set: no shared parent to name against.
        return None


def _relative_to_root(path: Path, roots: Sequence[Path]) -> tuple[str, Path]:
    """Return (relative path for display and identity, the root that matched).

    The relative path is taken against the roots' common parent rather than against the
    matching root, so it is unique across roots. The matched root is still reported, since
    trust rules are evaluated against the path within it.
    """
    matched: Path | None = None
    deepest = -1
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if len(root.parts) > deepest:
            deepest = len(root.parts)
            matched = root
    if matched is None:
        return path.name, path.parent

    base = display_base(roots)
    if base is not None:
        try:
            return path.relative_to(base).as_posix(), matched
        except ValueError:
            pass
    # No usable common parent: keep the root's name as a discriminator.
    return f"{matched.name}/{path.relative_to(matched).as_posix()}", matched


def load_document(path: Path, roots: Sequence[Path], trust: TrustPolicy) -> Document | None:
    """Read and identify one document. Returns None if it cannot be read."""
    try:
        data = path.read_bytes()
    except READ_ERRORS:
        return None
    rel, root = _relative_to_root(path, roots)
    return Document(
        path=path,
        rel=rel,
        root=root,
        doc_id=document_id(rel),
        content_hash=content_hash(data),
        size=len(data),
        trust=trust.tier_for(rel),
    )


def discover(policy: ScopePolicy, trust: TrustPolicy | None = None) -> Iterator[Document]:
    """Yield a Document for every file the scope admits."""
    tp = trust or TrustPolicy()
    roots = policy.roots
    for path in policy.iter_files():
        doc = load_document(path, roots, tp)
        if doc is not None:
            yield doc


def chunk_of(
    doc: Document, *, max_chars: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    """Chunk a document, carrying its identity and trust tier onto every chunk."""
    try:
        text = doc.path.read_text(errors="replace")
    except READ_ERRORS:
        return []
    kwargs: dict[str, int] = {}
    if max_chars is not None:
        kwargs["max_chars"] = max_chars
    if overlap is not None:
        kwargs["overlap"] = overlap
    return chunk_document(
        text,
        doc_id=doc.doc_id,
        doc_rel=doc.rel,
        trust=doc.trust,
        **kwargs,
    )


@dataclass(frozen=True)
class IngestPlan:
    """What a re-index would do, decided by comparing content hashes."""

    added: tuple[Document, ...]
    changed: tuple[Document, ...]
    unchanged: tuple[Document, ...]
    removed: tuple[str, ...]

    @property
    def work(self) -> tuple[Document, ...]:
        return (*self.added, *self.changed)

    def render(self) -> str:
        return (
            f"  {len(self.added):>5}  added\n"
            f"  {len(self.changed):>5}  changed\n"
            f"  {len(self.unchanged):>5}  unchanged (skipped)\n"
            f"  {len(self.removed):>5}  removed"
        )


def plan_ingest(docs: Sequence[Document], known: Mapping[str, str]) -> IngestPlan:
    """Diff discovered documents against ``{doc_id: content_hash}`` already indexed."""
    added: list[Document] = []
    changed: list[Document] = []
    unchanged: list[Document] = []

    seen: set[str] = set()
    for doc in docs:
        seen.add(doc.doc_id)
        previous = known.get(doc.doc_id)
        if previous is None:
            added.append(doc)
        elif previous != doc.content_hash:
            changed.append(doc)
        else:
            unchanged.append(doc)

    removed = tuple(sorted(doc_id for doc_id in known if doc_id not in seen))
    return IngestPlan(tuple(added), tuple(changed), tuple(unchanged), removed)
