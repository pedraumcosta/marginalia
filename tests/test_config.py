"""Tests for configuration loading and scope profiles."""

from __future__ import annotations

import pytest

from marginalia.config import Config, ConfigError
from marginalia.scope import ScopeError


@pytest.fixture
def corpus(tmp_path):
    for name in ("alpha", "beta"):
        d = tmp_path / name
        d.mkdir()
        (d / f"{name}.md").write_text(f"# {name}\n")
    # a nested clone inside alpha
    clone = tmp_path / "alpha" / "vendored"
    (clone / ".git").mkdir(parents=True)
    (clone / "upstream.md").write_text("# upstream\n")
    return tmp_path


def write_config(tmp_path, body: str, name: str = "config.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return p


# ---- discovery ----------------------------------------------------------------


def test_load_explicit_path(corpus):
    p = write_config(corpus, "scope:\n  roots: [alpha]\n")
    cfg = Config.load(p)
    assert cfg.path == p.resolve()


def test_load_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "nope.yaml")


def test_discovery_prefers_local_over_committed(corpus):
    write_config(corpus, "scope:\n  roots: [alpha]\n", "config.yaml")
    write_config(corpus, "scope:\n  roots: [beta]\n", "config.local.yaml")
    cfg = Config.load(start=corpus)
    assert cfg.path.name == "config.local.yaml"
    assert cfg.scope().roots == (corpus / "beta",)


def test_discovery_failure_names_what_it_looked_for(tmp_path):
    with pytest.raises(ConfigError, match="config.example.yaml"):
        Config.load(start=tmp_path)


def test_invalid_yaml_is_reported_with_the_path(corpus):
    p = write_config(corpus, "scope: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        Config.load(p)


def test_non_mapping_top_level_rejected(corpus):
    p = write_config(corpus, "- a\n- b\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.load(p)


# ---- flat config (no profiles) ------------------------------------------------


def test_flat_config_exposes_one_implicit_profile(corpus):
    cfg = Config.load(write_config(corpus, "scope:\n  roots: [alpha]\n"))
    assert cfg.profile_names == ("default",)
    assert cfg.default_profile == "default"
    assert {p.name for p in cfg.scope().iter_files()} == {"alpha.md"}


def test_flat_config_rejects_a_named_profile(corpus):
    cfg = Config.load(write_config(corpus, "scope:\n  roots: [alpha]\n"))
    with pytest.raises(ConfigError, match="declares no profiles"):
        cfg.scope("library")


def test_config_with_no_roots_at_all(corpus):
    cfg = Config.load(write_config(corpus, "scope: {}\n"))
    with pytest.raises(ConfigError, match="Nothing can be indexed"):
        _ = cfg.default_profile


# ---- profiles -----------------------------------------------------------------


PROFILES = """
scope:
  default_profile: narrow
  defaults:
    suffixes: [.md]
    exclude: [Papers]
  profiles:
    narrow:
      roots: [alpha]
    wide:
      roots: [alpha, beta]
      exclude: [figures]
"""


def test_profiles_are_listed_and_defaulted(corpus):
    cfg = Config.load(write_config(corpus, PROFILES))
    assert set(cfg.profile_names) == {"narrow", "wide"}
    assert cfg.default_profile == "narrow"


def test_profile_selects_its_own_roots(corpus):
    cfg = Config.load(write_config(corpus, PROFILES))
    assert {p.name for p in cfg.scope("narrow").iter_files()} == {"alpha.md"}
    assert {p.name for p in cfg.scope("wide").iter_files()} == {"alpha.md", "beta.md"}


def test_exclusions_accumulate_rather_than_replace(corpus):
    """A profile adding an exclusion must not silently drop the shared ones."""
    cfg = Config.load(write_config(corpus, PROFILES))
    wide = cfg.scope("wide")
    assert "Papers" in wide.exclude, "defaults' exclusion was dropped"
    assert "figures" in wide.exclude


def test_non_additive_keys_are_overridden(corpus):
    body = """
scope:
  defaults:
    suffixes: [.md]
  profiles:
    other:
      roots: [alpha]
      suffixes: [.txt]
"""
    cfg = Config.load(write_config(corpus, body))
    assert cfg.scope("other").suffixes == frozenset({".txt"})


def test_unknown_profile_lists_the_available_ones(corpus):
    cfg = Config.load(write_config(corpus, PROFILES))
    with pytest.raises(ConfigError, match="narrow"):
        cfg.scope("nope")


def test_bad_default_profile_is_rejected(corpus):
    body = "scope:\n  default_profile: ghost\n  profiles:\n    real:\n      roots: [alpha]\n"
    cfg = Config.load(write_config(corpus, body))
    with pytest.raises(ConfigError, match="not defined"):
        _ = cfg.default_profile


def test_empty_roots_in_a_profile_still_raises(corpus):
    body = "scope:\n  profiles:\n    empty:\n      roots: []\n"
    cfg = Config.load(write_config(corpus, body))
    with pytest.raises(ScopeError):
        cfg.scope("empty")


# ---- index directories --------------------------------------------------------


def test_profiles_get_separate_index_dirs(corpus):
    """Sharing one index would make profiles return each other's results."""
    cfg = Config.load(write_config(corpus, PROFILES))
    assert cfg.index_dir("narrow") != cfg.index_dir("wide")
    assert cfg.index_dir("narrow").name == "narrow"


def test_index_path_is_configurable_and_expanded(corpus):
    body = "scope:\n  roots: [alpha]\nindex:\n  path: ~/somewhere\n"
    cfg = Config.load(write_config(corpus, body))
    assert cfg.index_root.is_absolute()
    assert "somewhere" in str(cfg.index_root)


def test_relative_index_path_resolves_against_the_config(corpus):
    body = "scope:\n  roots: [alpha]\nindex:\n  path: ./state\n"
    cfg = Config.load(write_config(corpus, body))
    assert cfg.index_root == (corpus / "state").resolve()


# ---- generation ---------------------------------------------------------------


def test_generation_defaults_to_none(corpus):
    """The privacy-preserving default must be what you get by omission (ADR 0003)."""
    cfg = Config.load(write_config(corpus, "scope:\n  roots: [alpha]\n"))
    assert cfg.generation_provider == "none"


def test_generation_provider_is_read_and_lowercased(corpus):
    body = "scope:\n  roots: [alpha]\ngeneration:\n  provider: Anthropic\n"
    cfg = Config.load(write_config(corpus, body))
    assert cfg.generation_provider == "anthropic"


# ---- nested repo carve-out ----------------------------------------------------


def test_nested_clone_pruned_by_default(corpus):
    cfg = Config.load(write_config(corpus, "scope:\n  roots: [alpha]\n"))
    assert "upstream.md" not in {p.name for p in cfg.scope().iter_files()}


def test_allow_repos_carves_out_one_clone(corpus):
    body = "scope:\n  roots: [alpha]\n  allow_repos: [vendored]\n"
    cfg = Config.load(write_config(corpus, body))
    assert "upstream.md" in {p.name for p in cfg.scope().iter_files()}


def test_allow_repos_does_not_reopen_other_clones(corpus):
    """The carve-out must be surgical: naming one clone must not admit another."""
    other = corpus / "alpha" / "another"
    (other / ".git").mkdir(parents=True)
    (other / "nope.md").write_text("# nope\n")
    body = "scope:\n  roots: [alpha]\n  allow_repos: [vendored]\n"
    cfg = Config.load(write_config(corpus, body))
    names = {p.name for p in cfg.scope().iter_files()}
    assert "upstream.md" in names
    assert "nope.md" not in names


def test_allow_repos_accepts_a_path_glob(corpus):
    """A directory of clones is carved out by pattern, not one entry per repo."""
    holder = corpus / "alpha" / "holdings"
    for name in ("one", "two"):
        r = holder / name
        (r / ".git").mkdir(parents=True)
        (r / f"{name}.md").write_text(f"# {name}\n")
    body = "scope:\n  roots: [alpha]\n  allow_repos: ['holdings/*']\n"
    cfg = Config.load(write_config(corpus, body))
    names = {p.name for p in cfg.scope().iter_files()}
    assert {"one.md", "two.md"} <= names


def test_describe_agrees_with_iter_files_under_carve_out(corpus):
    body = "scope:\n  roots: [alpha]\n  allow_repos: [vendored]\n"
    cfg = Config.load(write_config(corpus, body))
    pol = cfg.scope()
    assert pol.describe().file_count == len(list(pol.iter_files()))
