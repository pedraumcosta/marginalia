"""Split Markdown into retrievable chunks along its own heading structure.

The corpus is organised under numbered headings and cites itself as ``file §n``, so the
structure is authored signal and splitting across it would discard information a human
already encoded (ADR 0007). The chunker therefore:

* parses the document into a tree of sections;
* emits one chunk per section that has prose of its own, carrying the full heading path;
* recovers the section reference from the heading text, so a citation reads
  ``Navigation.md §2.1`` rather than a byte offset;
* splits only *within* an oversized section, never across a heading boundary;
* falls back to fixed-size windows for documents that have no headings at all.

Two details matter more than they look. A ``#`` inside a fenced code block is not a
heading, and a ``---`` line is a horizontal rule in most of this corpus but a setext
underline in scraped articles — getting either wrong silently mangles the section tree.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

# ``## 6.5 Agent Memory`` / ``## 3. Dead reckoning`` -> "6.5" / "3"
SECTION_REF = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*)$")
ATX = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
FENCE = re.compile(r"^(?P<indent>\s{0,3})(?P<fence>```+|~~~+)\s*(?P<info>.*)$")
SETEXT_H1 = re.compile(r"^=+\s*$")
SETEXT_H2 = re.compile(r"^-{2,}\s*$")
# Lines that cannot be a setext heading's text.
NOT_SETEXT_TEXT = re.compile(r"^\s*(?:[-*+>]\s|\d+[.)]\s|#|\||$)")

# Inline Markdown that belongs in prose but not in a citation.
INLINE_MARKUP = re.compile(r"(\*\*|__|~~|\*|_|`)")
TRAILING_NOTE = re.compile(
    r"\s*\([^)]*\b(?:added|flattened|deferred|revised|resolved|updated)\b[^)]*\)\s*$", re.I
)
MAX_CITATION_SEGMENT = 48

DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP = 180
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Section:
    """One heading and the prose directly beneath it, before any child heading."""

    level: int
    title: str
    ref: str | None
    body: str
    start_line: int
    end_line: int
    path: tuple[str, ...]
    parent_index: int | None


def clean_heading(text: str) -> str:
    """Render a heading for display in a citation.

    Heading text is kept verbatim in `heading_path` because it is matchable content, but
    a citation is read by a human: emphasis markers, strikethrough, code ticks and dated
    editorial asides make an unreadable reference out of a perfectly good heading.
    """
    out = INLINE_MARKUP.sub("", text)
    out = TRAILING_NOTE.sub("", out)
    out = " ".join(out.split())
    if len(out) > MAX_CITATION_SEGMENT:
        out = out[: MAX_CITATION_SEGMENT - 1].rstrip() + "\u2026"
    return out


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with everything needed to cite it."""

    doc_id: str
    doc_rel: str
    chunk_id: str
    text: str
    heading_path: tuple[str, ...]
    section_ref: str | None
    start_line: int
    end_line: int
    ordinal: int = 0
    part: int = 0
    parts: int = 1
    parent_chunk_id: str | None = None
    trust: str = "unknown"
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """How this chunk refers to itself, in the corpus's own idiom."""
        if self.section_ref:
            base = f"{self.doc_rel} §{self.section_ref}"
        elif len(self.heading_path) > 1:
            trail = " > ".join(clean_heading(h) for h in self.heading_path[1:])
            base = f"{self.doc_rel} > {trail}"
        elif self.heading_path:
            base = self.doc_rel
        else:
            base = self.doc_rel
        if self.parts > 1:
            base = f"{base} ({self.part + 1}/{self.parts})"
        return base

    @property
    def is_split(self) -> bool:
        return self.parts > 1


# ---------------------------------------------------------------------------
# frontmatter and fences
# ---------------------------------------------------------------------------


def strip_frontmatter(text: str) -> tuple[str, int]:
    """Remove a leading YAML frontmatter block. Returns (body, lines_removed)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text, 0
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "\n".join(lines[i + 1 :]), i + 1
    return text, 0


def _fence_state(lines: Sequence[str]) -> list[bool]:
    """For each line, whether it sits inside a fenced code block.

    Tracks the opening fence's character and length, since a shorter run of the same
    character does not close a longer fence.
    """
    inside: list[bool] = []
    open_char: str | None = None
    open_len = 0
    for line in lines:
        m = FENCE.match(line)
        if m:
            fence = m.group("fence")
            char, length = fence[0], len(fence)
            if open_char is None:
                # Opening fence: the line itself is part of the block.
                open_char, open_len = char, length
                inside.append(True)
                continue
            if char == open_char and length >= open_len and not m.group("info").strip():
                open_char, open_len = None, 0
                inside.append(True)
                continue
        inside.append(open_char is not None)
    return inside


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _heading_at(
    lines: Sequence[str], i: int, inside: Sequence[bool]
) -> tuple[int, str, int] | None:
    """Return (level, text, lines_consumed) if a heading starts at line `i`."""
    if inside[i]:
        return None
    line = lines[i]

    m = ATX.match(line)
    if m and m.group("text").strip():
        return len(m.group("hashes")), m.group("text").strip(), 1

    # Setext: this line is the text, the next is the underline.
    if i + 1 < len(lines) and not inside[i + 1] and not NOT_SETEXT_TEXT.match(line):
        nxt = lines[i + 1]
        if SETEXT_H1.match(nxt):
            return 1, line.strip(), 2
        # A '---' run is only a setext underline when real text precedes it; otherwise
        # it is a horizontal rule, which this corpus uses heavily as a separator.
        if SETEXT_H2.match(nxt) and line.strip():
            return 2, line.strip(), 2
    return None


def parse_sections(text: str) -> list[Section]:
    """Parse Markdown into a flat list of sections in document order."""
    body, offset = strip_frontmatter(text)
    lines = body.splitlines()
    inside = _fence_state(lines)

    sections: list[Section] = []
    stack: list[int] = []  # indices into `sections`, ancestors of the current one
    pending: list[str] = []  # body lines of the section being accumulated
    current: dict[str, object] | None = None
    current_start = 0

    def close(end_line: int) -> None:
        nonlocal pending, current
        if current is None:
            if any(ln.strip() for ln in pending):
                # Preamble before the first heading.
                sections.append(
                    Section(
                        level=0,
                        title="",
                        ref=None,
                        body="\n".join(pending).strip("\n"),
                        start_line=current_start + offset + 1,
                        end_line=end_line + offset,
                        path=(),
                        parent_index=None,
                    )
                )
            pending = []
            return
        level = int(current["level"])  # type: ignore[index]
        title = str(current["title"])  # type: ignore[index]
        ref = current["ref"]  # type: ignore[index]
        path = tuple(current["path"])  # type: ignore[arg-type,index]
        sections.append(
            Section(
                level=level,
                title=title,
                ref=ref,  # type: ignore[arg-type]
                body="\n".join(pending).strip("\n"),
                start_line=int(current["start"]) + offset + 1,  # type: ignore[index]
                end_line=end_line + offset,
                path=path,
                parent_index=current["parent"],  # type: ignore[arg-type,index]
            )
        )
        pending = []

    i = 0
    while i < len(lines):
        head = _heading_at(lines, i, inside)
        if head is None:
            pending.append(lines[i])
            i += 1
            continue

        level, htext, consumed = head
        close(i)

        # Pop ancestors at or below this level.
        while stack and sections[stack[-1]].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        parent_path = sections[parent].path if parent is not None else ()

        m = SECTION_REF.match(htext)
        ref = m.group(1) if m else None

        current = {
            "level": level,
            "title": htext,
            "ref": ref,
            "path": (*parent_path, htext),
            "parent": parent,
            "start": i,
        }
        # The section just opened will be at this index once closed.
        stack.append(len(sections))
        i += consumed

    close(len(lines))
    return sections


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def _hard_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Window a single oversized paragraph by index, with guaranteed forward progress.

    Index arithmetic rather than repeatedly re-slicing a buffer: an earlier version
    advanced by ``cut - overlap``, which goes negative when the only break point sits
    near the start, leaving the buffer unchanged and the loop appending windows until
    the process was killed.
    """
    windows: list[str] = []
    step = max(1, max_chars - overlap)
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            # Prefer a break after the minimum step, so a window is never degenerate.
            cut = text.rfind("\n", i + step, end)
            if cut == -1:
                cut = text.rfind(" ", i + step, end)
            if cut > i:
                end = cut
        windows.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)  # strictly increasing, so this always terminates
    return windows


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Window an oversized body, preferring paragraph boundaries.

    Paragraph seams carry no overlap: a blank line is already a semantic boundary, so
    duplicating across it buys nothing. Overlap is applied only where a single paragraph
    has to be cut mid-flow, which is where a fact can genuinely straddle the seam.
    """
    if len(text) <= max_chars:
        return [text]

    # Overlap larger than half the window makes windows that barely advance.
    overlap = max(0, min(overlap, max_chars // 2))

    windows: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        if len(para) > max_chars:
            if buf:
                windows.append(buf)
                buf = ""
            windows.extend(_hard_window(para, max_chars, overlap))
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                windows.append(buf)
            buf = para
    if buf.strip():
        windows.append(buf)
    return windows or [text[:max_chars]]


def _chunk_id(doc_id: str, ordinal: int, part: int) -> str:
    return hashlib.sha256(f"{doc_id}:{ordinal}:{part}".encode()).hexdigest()[:16]


def _drop_runts(windows: list[str], min_chars: int) -> list[str]:
    """Discard split windows too short to carry meaning.

    Windowing leaves a remainder, and a remainder can be a few characters long. The
    per-section minimum is applied before splitting, so without this a 3-character tail
    reaches the index and occupies a retrieval slot it cannot possibly earn.
    """
    if len(windows) <= 1:
        return windows
    kept = [w for w in windows if len(w.strip()) >= min_chars]
    return kept or windows[:1]


def chunk_document(
    text: str,
    *,
    doc_id: str,
    doc_rel: str,
    trust: str = "unknown",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk one document.

    Sections that are pure containers — a heading whose prose is only its children — do
    not become chunks of their own; their heading still travels in every descendant's
    path, so nothing is lost and an empty chunk is not indexed.
    """
    sections = parse_sections(text)

    if not sections or all(s.level == 0 for s in sections):
        # No headings: fall back to fixed-size windows over the whole document (ADR 0007).
        body, offset = strip_frontmatter(text)
        body = body.strip("\n")
        if not body.strip():
            return []
        windows = _drop_runts(_split_long(body, max_chars, overlap), min_chars)
        total = len(windows)
        return [
            Chunk(
                doc_id=doc_id,
                doc_rel=doc_rel,
                chunk_id=_chunk_id(doc_id, 0, part),
                text=w,
                heading_path=(),
                section_ref=None,
                start_line=offset + 1,
                end_line=offset + len(body.splitlines()),
                ordinal=0,
                part=part,
                parts=total,
                trust=trust,
            )
            for part, w in enumerate(windows)
        ]

    # Map section index -> chunk id of its first emitted chunk, for parent linkage.
    first_chunk: dict[int, str] = {}
    chunks: list[Chunk] = []

    for idx, sec in enumerate(sections):
        body = sec.body.strip()
        if len(body) < min_chars:
            continue

        # Nearest ancestor that produced a chunk becomes the parent for context expansion.
        parent_id: str | None = None
        walker = sec.parent_index
        while walker is not None:
            if walker in first_chunk:
                parent_id = first_chunk[walker]
                break
            walker = sections[walker].parent_index

        heading_line = sec.title if sec.title else ""
        payload = f"{heading_line}\n\n{body}".strip() if heading_line else body
        windows = _drop_runts(_split_long(payload, max_chars, overlap), min_chars)
        total = len(windows)

        for part, w in enumerate(windows):
            cid = _chunk_id(doc_id, idx, part)
            if part == 0:
                first_chunk[idx] = cid
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_rel=doc_rel,
                    chunk_id=cid,
                    text=w,
                    heading_path=sec.path,
                    section_ref=sec.ref,
                    start_line=sec.start_line,
                    end_line=sec.end_line,
                    ordinal=idx,
                    part=part,
                    parts=total,
                    parent_chunk_id=parent_id,
                    trust=trust,
                )
            )

    return chunks


def iter_chunks(chunks: Sequence[Chunk]) -> Iterator[tuple[str, str]]:
    """(chunk_id, text) pairs, the shape an embedder or FTS writer wants."""
    for c in chunks:
        yield c.chunk_id, c.text
