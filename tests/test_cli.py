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
