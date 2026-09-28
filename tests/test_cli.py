"""Tests for the command line interface."""

from __future__ import annotations

import pytest

from marginalia.cli import main


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "one.md").write_text("# One\n")
    (tmp_path / "notes" / "two.md").write_text("# Two\n")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "three.md").write_text("# Three\n")
    (tmp_path / "config.local.yaml").write_text(
        "scope:\n"
        "  default_profile: small\n"
        "  profiles:\n"
        "    small:\n"
        "      roots: [notes]\n"
        "    big:\n"
        "      roots: [notes, other]\n"
        "index:\n"
        "  path: ./state\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_scope_reports_default_profile(workspace, capsys):
    assert main(["scope"]) == 0
    out = capsys.readouterr().out
    assert "2  files" in out.replace("  ", "  ")
    assert "state/small" in out


def test_scope_honours_profile_flag(workspace, capsys):
    assert main(["scope", "--profile", "big"]) == 0
    assert "3" in capsys.readouterr().out


def test_scope_lists_profiles_and_marks_default(workspace, capsys):
    assert main(["scope", "--list-profiles"]) == 0
    out = capsys.readouterr().out
    assert "small  (default)" in out
    assert "big" in out


def test_scope_all_profiles_reports_each(workspace, capsys):
    assert main(["scope", "--all-profiles"]) == 0
    out = capsys.readouterr().out
    assert "profile: small" in out
    assert "profile: big" in out


def test_scope_files_lists_paths(workspace, capsys):
    assert main(["scope", "--files"]) == 0
    out = capsys.readouterr().out
    assert "one.md" in out and "two.md" in out


def test_unknown_profile_exits_nonzero_without_traceback(workspace, capsys):
    assert main(["scope", "--profile", "ghost"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_missing_config_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["scope"]) == 1
    assert "error:" in capsys.readouterr().err


# ---- ask ----------------------------------------------------------------------


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    """A small real index, built through the CLI, so `ask` is exercised end to end."""
    corpus = tmp_path / "notes"
    corpus.mkdir()
    (corpus / "Navigation.md").write_text(
        "# Navigation\n\nA map of position finding.\n\n"
        "## 1. The problem\n\nLongitude requires knowing the time at a reference "
        "meridian, which is why it stayed unsolved while latitude was routine.\n"
    )
    (tmp_path / "config.local.yaml").write_text(
        "scope:\n  roots: [notes]\n"
        "index:\n  path: ./state\n"
        "embeddings:\n  provider: hashing\n  dim: 64\n"
        "generation:\n  provider: extractive\n"
        "trust:\n  default: authored\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["index"]) == 0
    return tmp_path


def test_ask_answers_with_a_citation(indexed, capsys):
    assert main(["ask", "why", "was", "longitude", "unsolved"]) == 0
    out = capsys.readouterr().out
    assert "Navigation.md" in out
    assert "Sources:" in out


def test_ask_makes_no_model_call_by_default(indexed, capsys):
    assert main(["ask", "longitude"]) == 0
    out = capsys.readouterr().out
    assert "No model was called" in out


def test_ask_can_show_the_passages_it_would_send(indexed, capsys):
    assert main(["ask", "--show-passages", "longitude"]) == 0
    out = capsys.readouterr().out
    assert "[authored]" in out


def test_ask_trust_floor_is_accepted(indexed, capsys):
    assert main(["ask", "--trust-floor", "authored", "longitude"]) == 0
    assert "Sources:" in capsys.readouterr().out


def test_ask_rejects_an_unknown_trust_floor(indexed, capsys):
    assert main(["ask", "--trust-floor", "extremely", "longitude"]) == 1
    assert "unknown trust floor" in capsys.readouterr().err


def test_ask_on_an_empty_index_is_a_clean_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "notes").mkdir()
    (tmp_path / "config.local.yaml").write_text(
        "scope:\n  roots: [notes]\nindex:\n  path: ./state\n"
        "embeddings:\n  provider: hashing\n  dim: 32\n"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["ask", "anything"]) == 1
    assert "is empty" in capsys.readouterr().err


def test_stats_reports_the_index(indexed, capsys):
    assert main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "documents" in out and "chunks" in out
