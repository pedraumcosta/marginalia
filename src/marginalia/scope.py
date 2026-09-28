"""Which files may be indexed.

The rule is deny-by-default: nothing is a candidate unless it sits under a root the
operator listed explicitly (ADR 0005). Exclusions then apply on top, and a set of
secret-bearing patterns is always excluded whether or not the config mentions them —
an embedded credential is a credential that can be surfaced verbatim by a search.

The effective scope is reportable before anything is read, so "what would you index?"
is answerable without indexing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

# Applied unconditionally, unioned with whatever the config excludes. These are files
# whose contents must never enter an index, and directories whose contents are never
# the operator's own writing.
ALWAYS_EXCLUDE: tuple[str, ...] = (
    # secrets
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "token.json",
    "*.credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    # version control and vendored trees
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "site-packages",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
)

DEFAULT_SUFFIXES: tuple[str, ...] = (".md", ".markdown")


class ScopeError(Exception):
    """Raised when a scope is unusable — most often because no root was declared."""


@dataclass(frozen=True)
class ScopeReport:
    """What a scope would cover, without having read any file content."""

    roots: tuple[Path, ...]
    file_count: int
    total_bytes: int
    per_root: tuple[tuple[Path, int], ...]
    skipped_roots: tuple[tuple[Path, str], ...] = ()

    def render(self) -> str:
        lines = ["Effective scope:"]
        for root, count in self.per_root:
            lines.append(f"  {count:>6}  {root}")
        for root, why in self.skipped_roots:
            lines.append(f"  {'--':>6}  {root}  ({why})")
        mb = self.total_bytes / 1_048_576
        lines.append(f"  {'-' * 6}")
        lines.append(f"  {self.file_count:>6}  files, {mb:.1f} MiB")
        return "\n".join(lines)


@dataclass(frozen=True)
class ScopePolicy:
    """An allowlist of roots plus exclusion patterns."""

    roots: tuple[Path, ...]
    suffixes: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_SUFFIXES))
    exclude: tuple[str, ...] = ()
    follow_symlinks: bool = False
    max_file_bytes: int = 2_000_000
    # A nested git repository is, almost by definition, somebody else's writing: a
    # cloned dependency, a course's sample code, a vendored tool. Pruning by rule
    # scales where an exclusion list does not — the corpus this was built against
    # holds 182 such repositories, and enumerating them would go stale immediately.
    skip_nested_repos: bool = True

    def __post_init__(self) -> None:
        if not self.roots:
            raise ScopeError(
                "no roots declared: scope is deny-by-default, so at least one root "
                "must be listed explicitly before anything can be indexed"
            )

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, data: Mapping[str, object], *, base: Path | None = None) -> ScopePolicy:
        """Build from a parsed config mapping.

        Relative roots resolve against `base` (normally the config file's directory) so a
        config is portable and never needs an absolute path committed to it.
        """
        raw_roots = data.get("roots") or []
        if isinstance(raw_roots, str):
            raw_roots = [raw_roots]
        if not isinstance(raw_roots, Sequence):
            raise ScopeError("'roots' must be a list of paths")

        roots: list[Path] = []
        for entry in raw_roots:
            p = Path(str(entry)).expanduser()
            if not p.is_absolute() and base is not None:
                p = (base / p).resolve()
            roots.append(p)

        raw_suffixes = data.get("suffixes") or DEFAULT_SUFFIXES
        if isinstance(raw_suffixes, str):
            raw_suffixes = [raw_suffixes]
        suffixes = frozenset(
            s if str(s).startswith(".") else f".{s}" for s in raw_suffixes  # type: ignore[union-attr]
        )

        raw_exclude = data.get("exclude") or []
        if isinstance(raw_exclude, str):
            raw_exclude = [raw_exclude]

        return cls(
            roots=tuple(roots),
            suffixes=suffixes,
            exclude=tuple(str(e) for e in raw_exclude),  # type: ignore[union-attr]
            follow_symlinks=bool(data.get("follow_symlinks", False)),
            skip_nested_repos=bool(data.get("skip_nested_repos", True)),
            max_file_bytes=int(data.get("max_file_bytes", 2_000_000)),  # type: ignore[arg-type]
        )

    # ---- matching -----------------------------------------------------------

    @property
    def effective_exclude(self) -> tuple[str, ...]:
        """Config exclusions unioned with the unconditional ones."""
        return tuple(dict.fromkeys((*self.exclude, *ALWAYS_EXCLUDE)))

    def _matches(self, name: str, relpath: str) -> bool:
        for pattern in self.effective_exclude:
            # A bare name (``node_modules``, ``.env``) matches any path component;
            # a pattern with a separator is matched against the path from the root.
            if "/" in pattern:
                if fnmatch(relpath, pattern):
                    return True
            elif fnmatch(name, pattern):
                return True
        return False

    def is_excluded(self, path: Path, *, root: Path | None = None) -> bool:
        """True when any component of `path` is excluded."""
        try:
            rel = path.relative_to(root) if root else path
        except ValueError:
            rel = path
        relpath = rel.as_posix()
        if self._matches(path.name, relpath):
            return True
        return any(self._matches(part, relpath) for part in rel.parts)

    def wants_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    @staticmethod
    def _is_repo(path: Path) -> bool:
        """True when `path` is the top of a git repository (worktrees included)."""
        marker = path / ".git"
        return marker.is_dir() or marker.is_file()

    # ---- walking ------------------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        """Yield indexable files, pruning excluded directories as it walks.

        Pruning matters: an unpruned walk of a tree containing vendored dependency
        directories spends most of its time in files that can never be indexed.
        """
        for root in self.roots:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
                here = Path(dirpath)
                # Prune in place so os.walk does not descend.
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not self.is_excluded(here / d, root=root)
                    and not (self.skip_nested_repos and self._is_repo(here / d))
                ]
                for fname in filenames:
                    fpath = here / fname
                    if not self.wants_file(fpath):
                        continue
                    if self.is_excluded(fpath, root=root):
                        continue
                    if not self.follow_symlinks and fpath.is_symlink():
                        continue
                    try:
                        if fpath.stat().st_size > self.max_file_bytes:
                            continue
                    except OSError:
                        continue
                    yield fpath

    def describe(self) -> ScopeReport:
        """Count what would be indexed, reading no file contents."""
        per_root: list[tuple[Path, int]] = []
        skipped: list[tuple[Path, str]] = []
        total_files = 0
        total_bytes = 0

        for root in self.roots:
            if not root.exists():
                skipped.append((root, "does not exist"))
                continue
            if not root.is_dir():
                skipped.append((root, "not a directory"))
                continue
            count = 0
            single = ScopePolicy(
                roots=(root,),
                suffixes=self.suffixes,
                exclude=self.exclude,
                follow_symlinks=self.follow_symlinks,
                max_file_bytes=self.max_file_bytes,
                skip_nested_repos=self.skip_nested_repos,
            )
            for f in single.iter_files():
                count += 1
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
            per_root.append((root, count))
            total_files += count

        return ScopeReport(
            roots=self.roots,
            file_count=total_files,
            total_bytes=total_bytes,
            per_root=tuple(per_root),
            skipped_roots=tuple(skipped),
        )
