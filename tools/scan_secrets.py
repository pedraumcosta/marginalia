#!/usr/bin/env python3
"""Refuse to commit credentials, personal identifiers or machine-specific paths.

Runs in two places: as a pre-commit hook over staged files, and in CI over every
tracked file.

A deliberate split, because a scanner that ships in a public repository cannot contain
the private strings it is looking for — writing an employer's name into the pattern list
leaks the employer's name:

  * **Generic patterns** live here and are committed. Credentials, private keys, tokens,
    absolute home paths, e-mail addresses, national identifiers.
  * **Project-specific terms** live in a gitignored file (default `.forbidden-terms.local`,
    one term per line). Present locally, absent in CI, which is the correct asymmetry: the
    terms are themselves sensitive, and CI has no private data to compare against anyway.

Suppress an intentional match with a trailing ``marginalia:allow-secret`` comment on the
same line. Suppressions are reported so they cannot pile up unnoticed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ALLOW_MARKER = "marginalia:allow-secret"

# (name, pattern, hint). Patterns are written so they do not match their own source text.
PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
        "a private key block",
    ),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "an AWS access key id"),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "a GitHub token"),
    ("openai-key", re.compile(r"\bsk-(?!ant)[A-Za-z0-9]{20,}\b"), "an OpenAI-style API key"),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "an Anthropic API key"),
    ("notion-token", re.compile(r"\bntn_[A-Za-z0-9]{30,}\b"), "a Notion integration token"),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "a Slack token"),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "a Google API key"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "a JSON Web Token",
    ),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token)\b"
            r"\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"
        ),
        "a secret assigned inline",
    ),
    (
        "home-path",
        re.compile(r"(?:/home/(?!user\b|runner\b)[a-z][a-z0-9_-]{1,31}|/Users/(?!user\b)[A-Za-z])"),
        "an absolute path naming a real account",
    ),
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b|test\b|localhost\b)"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "an e-mail address",
    ),
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "a Brazilian CPF"),
    ("us-ssn", re.compile(r"\b(?!000|666)\d{3}-\d{2}-\d{4}\b"), "a US social security number"),
)

# Suffixes worth reading. Anything else is treated as binary and skipped.
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".json",
        ".xml",
        ".sh",
        ".bash",
        ".zsh",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".sql",
        ".csv",
        ".tsv",
        ".env",
        ".example",
        ".gitignore",
    }
)
MAX_BYTES = 1_000_000


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    rule: str
    hint: str
    excerpt: str

    def render(self) -> str:
        return f"{self.path}:{self.line_no}: {self.rule}: {self.hint}\n    {self.excerpt}"


def _redact(line: str, limit: int = 100) -> str:
    """Show enough of the line to locate it, never the whole secret."""
    stripped = line.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "…"


def load_terms(path: Path | None) -> tuple[str, ...]:
    """Load project-specific forbidden terms, if the local file exists."""
    if path is None or not path.is_file():
        return ()
    terms = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return tuple(terms)


def scan_text(text: str, path: Path, terms: tuple[str, ...] = ()) -> tuple[list[Finding], int]:
    """Return findings and the number of suppressed lines."""
    findings: list[Finding] = []
    suppressed = 0
    term_res = [(t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for t in terms]

    for i, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            suppressed += 1
            continue
        for rule, pattern, hint in PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path, i, rule, hint, _redact(line)))
        for term, term_re in term_res:
            if term_re.search(line):
                findings.append(
                    Finding(path, i, "forbidden-term", f"the term {term!r}", _redact(line))
                )
    return findings, suppressed


def is_scannable(path: Path) -> bool:
    if path.name in (".gitignore", ".dockerignore"):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_paths(paths: list[Path], terms: tuple[str, ...] = ()) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    suppressed = 0
    scanned = 0
    for p in paths:
        if not p.is_file() or not is_scannable(p):
            continue
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_text(errors="replace")
        except OSError:
            continue
        scanned += 1
        f, s = scan_text(text, p, terms)
        findings.extend(f)
        suppressed += s
    return findings, suppressed, scanned


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=False)  # noqa: S603
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def staged_files() -> list[Path]:
    return [Path(p) for p in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")]


def tracked_files() -> list[Path]:
    return [Path(p) for p in _git("ls-files")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan for credentials and personal identifiers.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="scan staged files (pre-commit)")
    group.add_argument("--tracked", action="store_true", help="scan every tracked file (CI)")
    ap.add_argument("paths", nargs="*", type=Path, help="explicit paths to scan")
    ap.add_argument(
        "--terms",
        type=Path,
        default=Path(".forbidden-terms.local"),
        help="file of project-specific forbidden terms (gitignored; default: %(default)s)",
    )
    args = ap.parse_args(argv)

    if args.staged:
        paths = staged_files()
    elif args.tracked:
        paths = tracked_files()
    elif args.paths:
        paths = args.paths
    else:
        paths = tracked_files()

    terms = load_terms(args.terms)
    findings, suppressed, scanned = scan_paths(paths, terms)

    note = f"scanned {scanned} file(s)"
    if terms:
        note += f", {len(terms)} project term(s)"
    else:
        note += ", no project term file (generic patterns only)"
    if suppressed:
        note += f", {suppressed} suppressed line(s)"
    print(note, file=sys.stderr)

    if findings:
        print("", file=sys.stderr)
        for f in findings:
            print(f.render(), file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). Remove the value, or append "
            f"'{ALLOW_MARKER}' to the line if it is deliberately fake.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
