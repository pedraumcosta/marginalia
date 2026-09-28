"""Tests for the secret scanner.

A scanner that returns clean on everything is worse than no scanner, because it buys
false confidence. So most of these plant realistically shaped fakes and assert they are
caught, and two assert the opposite: that the scanner does not cry wolf on placeholder
e-mail domains or on CI's own home paths.

The fakes below are deliberately invalid credentials.
"""

from __future__ import annotations

import pytest
from scan_secrets import ALLOW_MARKER, load_terms, main, scan_paths, scan_text


def find(text: str, name: str = "sample.py"):
    from pathlib import Path

    findings, _ = scan_text(text, Path(name))
    return {f.rule for f in findings}


# ---- detection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "sample"),
    [
        ("private-key", "-----BEGIN RSA PRIVATE KEY-----"),  # marginalia:allow-secret
        ("aws-access-key", "AKIAQQQQWWWWEEEERRRR"),  # marginalia:allow-secret
        ("github-token", "ghp_" + "a" * 36),
        ("openai-key", "sk-" + "b" * 32),
        ("anthropic-key", "sk-ant-" + "c" * 30),
        ("notion-token", "ntn_" + "d" * 40),
        ("slack-token", "xoxb-1111111111-aaaaaaaaaa"),  # marginalia:allow-secret
        ("google-key", "AIza" + "e" * 35),
        (
            "jwt",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abcdefghij",  # marginalia:allow-secret
        ),  # marginalia:allow-secret
        ("assigned-secret", 'api_key = "hunter2hunter2"'),  # marginalia:allow-secret
        ("assigned-secret", "password: 'correcthorsebattery'"),  # marginalia:allow-secret
        ("cpf", "123.456.789-01"),  # marginalia:allow-secret
        ("us-ssn", "123-45-6789"),  # marginalia:allow-secret
    ],
)
def test_catches_credential_shapes(rule, sample):
    assert rule in find(sample), f"{rule} not detected in {sample[:20]!r}"


def test_catches_absolute_home_path():
    assert "home-path" in find("corpus: /home/someone/Documents/notes")  # marginalia:allow-secret


def test_catches_email_address():
    assert "email" in find("contact: real.person@somecompany.co")  # marginalia:allow-secret


# ---- restraint ---------------------------------------------------------------


@pytest.mark.parametrize(
    "benign",
    [
        "user@example.com",
        "someone@example.org",
        "a@test",
    ],
)
def test_placeholder_emails_are_not_flagged(benign):
    assert "email" not in find(f"see {benign} for details")


@pytest.mark.parametrize("ci_path", ["/home/runner/work/repo", "/home/user/project"])
def test_ci_and_placeholder_home_paths_are_not_flagged(ci_path):
    """CI runners and documentation placeholders would otherwise fail every build."""
    assert "home-path" not in find(f"path: {ci_path}")


def test_short_assignments_are_not_flagged():
    """An 8-character floor keeps 'token = \"x\"' style examples quiet."""
    assert "assigned-secret" not in find('token = "abc"')


def test_the_scanner_does_not_flag_its_own_pattern_source():
    """The regex literals must not match themselves, or the tool fails on itself."""
    import pathlib

    import scan_secrets

    source = pathlib.Path(scan_secrets.__file__)
    findings, _, _ = scan_paths([source])
    assert findings == [], f"scanner flags its own source: {[f.rule for f in findings]}"


# ---- suppression -------------------------------------------------------------


def test_allow_marker_suppresses_a_line():
    text = f'api_key = "deliberatelyfake"  # {ALLOW_MARKER}\n'  # marginalia:allow-secret
    findings, suppressed = scan_text(text, __import__("pathlib").Path("x.py"))
    assert findings == []
    assert suppressed == 1


def test_suppressions_are_counted_so_they_cannot_pile_up(capsys, tmp_path, monkeypatch):
    f = tmp_path / "s.py"
    f.write_text(f'api_key = "fakefakefake"  # {ALLOW_MARKER}\n')  # marginalia:allow-secret
    monkeypatch.chdir(tmp_path)
    assert main([str(f)]) == 0
    assert "1 suppressed line" in capsys.readouterr().err


# ---- redaction ---------------------------------------------------------------


def test_excerpt_is_truncated_so_the_secret_is_not_reprinted():
    from pathlib import Path

    long_secret = "k" * 400
    findings, _ = scan_text(f'api_key = "{long_secret}"', Path("x.py"))  # marginalia:allow-secret
    assert findings
    assert long_secret not in findings[0].excerpt
    assert findings[0].excerpt.endswith("…")


# ---- project-specific terms --------------------------------------------------


def test_terms_file_is_loaded_ignoring_comments_and_blanks(tmp_path):
    p = tmp_path / ".forbidden-terms.local"
    p.write_text("# a comment\n\nAcmeCorp\nProjectNeptune\n")
    assert load_terms(p) == ("AcmeCorp", "ProjectNeptune")


def test_missing_terms_file_is_not_an_error(tmp_path):
    assert load_terms(tmp_path / "absent") == ()


def test_forbidden_term_is_caught_case_insensitively():
    from pathlib import Path

    findings, _ = scan_text("we deployed to acmecorp last week", Path("x.md"), ("AcmeCorp",))
    assert [f.rule for f in findings] == ["forbidden-term"]


def test_forbidden_term_respects_word_boundaries():
    """Substring matching would flag 'clones' for a term like 'Clone'."""
    from pathlib import Path

    findings, _ = scan_text("anyone who clones the repo", Path("x.md"), ("Clone",))
    assert findings == []


# ---- file selection ----------------------------------------------------------


def test_binary_suffixes_are_skipped(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"AKIAQQQQWWWWEEEERRRR")  # marginalia:allow-secret
    findings, _, scanned = scan_paths([p])
    assert scanned == 0
    assert findings == []


def test_oversized_files_are_skipped(tmp_path):
    p = tmp_path / "big.md"
    p.write_text("x" * 1_200_000)
    _, _, scanned = scan_paths([p])
    assert scanned == 0


def test_gitignore_is_scanned_despite_having_no_suffix(tmp_path):
    p = tmp_path / ".gitignore"
    p.write_text("/home/someone/state\n")  # marginalia:allow-secret
    findings, _, scanned = scan_paths([p])
    assert scanned == 1
    assert {f.rule for f in findings} == {"home-path"}


# ---- exit codes --------------------------------------------------------------


def test_exit_zero_when_clean(tmp_path, capsys):
    p = tmp_path / "ok.md"
    p.write_text("# Just prose\n")
    assert main([str(p)]) == 0


def test_exit_one_and_reports_when_dirty(tmp_path, capsys):
    p = tmp_path / "bad.md"
    p.write_text("AKIAQQQQWWWWEEEERRRR\n")  # marginalia:allow-secret
    assert main([str(p)]) == 1
    err = capsys.readouterr().err
    assert "aws-access-key" in err
    assert "1 finding" in err


def test_reports_when_no_term_file_is_present(tmp_path, capsys):
    p = tmp_path / "ok.md"
    p.write_text("# prose\n")
    main([str(p), "--terms", str(tmp_path / "absent")])
    assert "generic patterns only" in capsys.readouterr().err
