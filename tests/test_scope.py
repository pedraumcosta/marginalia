"""Tests for the scope allowlist.

These are the tests that matter most in the project: they are what makes the privacy
claim in the README checkable rather than asserted. The important cases are the
negative ones — a secret that must not be indexed even when the config forgets it.
"""

from __future__ import annotations

import pytest

from marginalia.scope import ALWAYS_EXCLUDE, ScopeError, ScopePolicy


@pytest.fixture
def tree(tmp_path):
    """A corpus that mixes indexable notes, vendored trees and secrets."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "map.md").write_text("# Map\n\nContent.\n")
    (tmp_path / "notes" / "deep").mkdir()
    (tmp_path / "notes" / "deep" / "nested.md").write_text("# Nested\n")
    (tmp_path / "notes" / "notes.txt").write_text("not markdown\n")

    # vendored: must be pruned even though it contains markdown
    vendor = tmp_path / "notes" / "node_modules" / "pkg"
    vendor.mkdir(parents=True)
    (vendor / "README.md").write_text("# vendor readme\n")

    # secrets: must never be indexed
    (tmp_path / "notes" / ".env").write_text("SECRET=abc123\n")
    (tmp_path / "notes" / "token.json").write_text('{"t":"x"}\n')

    # a root the operator did NOT declare
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "confidential.md").write_text("# do not index\n")

    return tmp_path


def policy_for(tree, **kw):
    return ScopePolicy(roots=(tree / "notes",), **kw)


def test_requires_at_least_one_root():
    with pytest.raises(ScopeError, match="deny-by-default"):
        ScopePolicy(roots=())


def test_indexes_markdown_under_declared_root(tree):
    found = {p.name for p in policy_for(tree).iter_files()}
    assert found == {"map.md", "nested.md"}


def test_undeclared_root_is_never_reached(tree):
    """The central guarantee: not listed means not indexed."""
    found = {p.name for p in policy_for(tree).iter_files()}
    assert "confidential.md" not in found


def test_vendored_directories_are_pruned(tree):
    found = {p.name for p in policy_for(tree).iter_files()}
    assert "README.md" not in found, "node_modules should be pruned even if it holds markdown"


def test_secrets_excluded_even_when_config_is_silent(tree):
    """ALWAYS_EXCLUDE applies whether or not the operator remembered it."""
    pol = policy_for(tree, exclude=())  # config mentions no exclusions at all
    assert pol.is_excluded(tree / "notes" / ".env", root=tree / "notes")
    assert pol.is_excluded(tree / "notes" / "token.json", root=tree / "notes")


def test_secret_suffixes_are_not_indexable_anyway(tree):
    """Defence in depth: .env is excluded, and also not an indexable suffix."""
    pol = policy_for(tree)
    assert not pol.wants_file(tree / "notes" / ".env")


def test_non_markdown_is_skipped(tree):
    found = {p.name for p in policy_for(tree).iter_files()}
    assert "notes.txt" not in found


def test_suffixes_are_configurable(tree):
    pol = policy_for(tree, suffixes=frozenset({".txt"}))
    found = {p.name for p in pol.iter_files()}
    assert found == {"notes.txt"}


def test_oversized_files_are_skipped(tree):
    big = tree / "notes" / "big.md"
    big.write_text("x" * 5000)
    pol = policy_for(tree, max_file_bytes=1000)
    assert "big.md" not in {p.name for p in pol.iter_files()}


def test_symlinks_are_not_followed_by_default(tree):
    """A symlink is an escape route out of the allowlist, so ignore it by default."""
    link = tree / "notes" / "escape.md"
    link.symlink_to(tree / "private" / "confidential.md")
    found = {p.name for p in policy_for(tree).iter_files()}
    assert "escape.md" not in found


def test_describe_counts_without_reading_content(tree):
    report = policy_for(tree).describe()
    assert report.file_count == 2
    assert report.total_bytes > 0
    assert len(report.per_root) == 1
    assert "2" in report.render()


def test_describe_reports_missing_roots(tree):
    pol = ScopePolicy(roots=(tree / "notes", tree / "nope"))
    report = pol.describe()
    assert report.file_count == 2
    assert report.skipped_roots
    assert "does not exist" in report.render()


def test_effective_exclude_unions_and_dedupes(tree):
    pol = policy_for(tree, exclude=("node_modules", "Papers"))
    eff = pol.effective_exclude
    assert "Papers" in eff
    assert eff.count("node_modules") == 1, "union must not duplicate"
    assert all(p in eff for p in ALWAYS_EXCLUDE)


def test_path_glob_excludes_match_from_root(tree):
    (tree / "notes" / "drafts").mkdir()
    (tree / "notes" / "drafts" / "wip.md").write_text("# wip\n")
    pol = policy_for(tree, exclude=("drafts/*",))
    assert "wip.md" not in {p.name for p in pol.iter_files()}


def test_from_config_resolves_relative_roots(tree):
    pol = ScopePolicy.from_config({"roots": ["notes"]}, base=tree)
    assert pol.roots == (tree / "notes",)
    assert {p.name for p in pol.iter_files()} == {"map.md", "nested.md"}


def test_from_config_accepts_bare_suffixes(tree):
    pol = ScopePolicy.from_config({"roots": [str(tree / "notes")], "suffixes": ["txt"]})
    assert ".txt" in pol.suffixes


def test_from_config_rejects_empty_roots():
    with pytest.raises(ScopeError):
        ScopePolicy.from_config({"roots": []})


# ---- nested repository pruning -------------------------------------------------


@pytest.fixture
def tree_with_clone(tree):
    """A cloned dependency sitting inside the declared root."""
    clone = tree / "notes" / "vendored-course"
    (clone / ".git").mkdir(parents=True)
    (clone / "lesson.md").write_text("# somebody else's lesson\n")
    worktree = tree / "notes" / "linked-worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    (worktree / "notes.md").write_text("# worktree content\n")
    return tree


def test_nested_repos_pruned_by_default(tree_with_clone):
    found = {p.name for p in policy_for(tree_with_clone).iter_files()}
    assert "lesson.md" not in found
    assert found == {"map.md", "nested.md"}


def test_nested_worktree_also_pruned(tree_with_clone):
    """A linked worktree has .git as a *file*, not a directory."""
    found = {p.name for p in policy_for(tree_with_clone).iter_files()}
    assert "notes.md" not in found


def test_nested_repos_can_be_opted_back_in(tree_with_clone):
    pol = policy_for(tree_with_clone, skip_nested_repos=False)
    found = {p.name for p in pol.iter_files()}
    assert "lesson.md" in found


def test_declared_root_may_itself_be_a_repo(tmp_path):
    """Pruning applies to subdirectories; a root the operator named is honoured."""
    root = tmp_path / "wiki"
    (root / ".git").mkdir(parents=True)
    (root / "map.md").write_text("# map\n")
    pol = ScopePolicy(roots=(root,))
    assert {p.name for p in pol.iter_files()} == {"map.md"}


def test_describe_agrees_with_iter_files(tree_with_clone):
    """describe() builds a per-root probe policy; if it drops a setting, counts lie."""
    pol = policy_for(tree_with_clone)
    assert pol.describe().file_count == len(list(pol.iter_files()))


def test_describe_agrees_when_nested_repos_included(tree_with_clone):
    pol = policy_for(tree_with_clone, skip_nested_repos=False)
    assert pol.describe().file_count == len(list(pol.iter_files()))


def test_from_config_defaults_to_skipping_nested_repos(tree_with_clone):
    pol = ScopePolicy.from_config({"roots": [str(tree_with_clone / "notes")]})
    assert pol.skip_nested_repos is True
    assert "lesson.md" not in {p.name for p in pol.iter_files()}
