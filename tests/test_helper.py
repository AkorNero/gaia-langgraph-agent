"""Unit tests for helper.py: text/table helpers and the run_python runners."""

import subprocess

import numpy as np
import pandas as pd
import pytest

import helper
from helper import (
    clean_markdown,
    clean_table,
    docker_available,
    header,
    keyword_lines,
    outline,
    prose_preview,
    run_in_docker,
    run_local,
    section_text,
    sections,
    slice_with_note,
)

WIKI = """Intro paragraph.
== History ==
History text.
=== Early years ===
Early text.
== Discography ==
Disco text.
== Discography notes ==
Notes text.
"""


# --- header / sections / outline ----------------------------------------------
@pytest.mark.parametrize(
    "line, expected",
    [
        ("== History ==", (2, "History")),
        ("=== Early years ===", (3, "Early years")),
        ("  == Padded ==  ", (2, "Padded")),
        ("# Title", (1, "Title")),
        ("### Sub ###", (3, "Sub")),
        ("plain text", None),
        ("#nospace", None),
        ("####### seven", None),
    ],
)
def test_header(line, expected):
    assert header(line) == expected


def test_sections_lists_every_heading_in_order():
    assert sections(WIKI) == [
        (2, "History"),
        (3, "Early years"),
        (2, "Discography"),
        (2, "Discography notes"),
    ]


def test_outline_indents_relative_to_top_level():
    assert outline(WIKI) == (
        "- History\n  - Early years\n- Discography\n- Discography notes"
    )


def test_outline_deduplicates_and_caps():
    content = "## A\n## A\n## B\n## C\n## D"
    assert outline(content, max_items=2) == "- A\n- B\n... 2 more headings"


def test_outline_without_headings():
    assert outline("just text") == "(no headings)"


# --- section_text ----------------------------------------------------------------
def test_section_text_includes_subsections_and_stops_at_same_level():
    assert section_text(WIKI, "History") == (
        "== History ==\nHistory text.\n=== Early years ===\nEarly text."
    )


def test_section_text_exact_match_beats_partial_match():
    assert section_text(WIKI, "discography") == "== Discography ==\nDisco text."


def test_section_text_partial_match():
    assert section_text(WIKI, "notes") == "== Discography notes ==\nNotes text."


def test_section_text_last_section_runs_to_end():
    assert section_text(WIKI, "Early years") == "=== Early years ===\nEarly text."


def test_section_text_not_found():
    assert section_text(WIKI, "Awards") is None


# --- keyword_lines ---------------------------------------------------------------
def test_keyword_lines_prose_prefixes_section():
    text = "Top line 2000.\n== Life ==\nBorn 1935.\nDied 2009.\nOther."
    assert keyword_lines(text, "1935, 2009, 2000", is_table=False) == (
        "Top line 2000.\n[Life] Born 1935.\n[Life] Died 2009."
    )


def test_keyword_lines_is_case_insensitive_and_skips_empty_keywords():
    assert keyword_lines("Alpha\nbeta\ngamma", " , ALPHA, ,", is_table=False) == "Alpha"


def test_keyword_lines_table_keeps_header_rows():
    table = "| Year | Album |\n|---|---|\n| 1999 | A |\n| 2009 | B |"
    assert keyword_lines(table, "2009", is_table=True) == (
        "| Year | Album |\n|---|---|\n| 2009 | B |"
    )


def test_keyword_lines_no_match():
    assert keyword_lines("a\nb", "zzz", is_table=False) == ""


# --- slice_with_note -------------------------------------------------------------
def test_slice_with_note_short_text_has_no_note():
    assert slice_with_note("abc", 0, 10, "k") == "abc"


def test_slice_with_note_adds_continuation_note():
    out = slice_with_note("abcdefghij", 0, 4, "k")
    assert out.startswith("abcd\n\n[6 more chars.")
    assert "read_stored(key='k', offset=4)" in out


def test_slice_with_note_with_offset():
    out = slice_with_note("abcdefghij", 4, 4, "k")
    assert out.startswith("efgh\n\n[2 more chars.")
    assert "offset=8" in out


def test_slice_with_note_reads_to_end_without_note():
    assert slice_with_note("abcdefghij", 8, 4, "k") == "ij"


def test_slice_with_note_offset_past_end():
    assert slice_with_note("abc", 10, 4, "k") == ""


# --- clean_markdown --------------------------------------------------------------
def test_clean_markdown_strips_images_citations_and_link_urls():
    md = (
        "# Title\n"
        "Text ![logo](http://x/logo.png) here.[[12]](#cite_note-12)\n"
        'See [the docs](http://x/docs "Docs") and [home](/home).'
    )
    assert clean_markdown(md) == "# Title\n\nText  here.\n\nSee the docs and home."


def test_clean_markdown_drops_empty_headings_and_blank_lines():
    md = "## Menu\n## Nav\n\n\n## Real\nBody   \n### Sub\nSub body\n## Trailing"
    assert clean_markdown(md) == "## Real\n\nBody\n\n### Sub\n\nSub body"


# --- prose_preview ---------------------------------------------------------------
def test_prose_preview_skips_short_lines_and_headings():
    long1, long2 = "a" * 100, "b" * 100
    text = f"Home\nAbout\n## {'h' * 100}\n{long1}\nshort\n{long2}"
    assert prose_preview(text) == f"{long1}\n{long2}"


def test_prose_preview_respects_limit():
    text = "\n".join(["x" * 100] * 50)
    out = prose_preview(text, limit=250)
    assert len(out) == 250


def test_prose_preview_falls_back_to_raw_text():
    assert prose_preview("tiny\nlines", limit=5) == "tiny\n"


# --- clean_table -----------------------------------------------------------------
def test_clean_table_flattens_multiindex_headers():
    cols = pd.MultiIndex.from_tuples(
        [("Year", "Year"), ("Album", "Title"), ("Unnamed: 2_level_0", "Label")]
    )
    df = pd.DataFrame([[1999, "A", "L"]], columns=cols)
    assert list(clean_table(df).columns) == ["Year", "Album / Title", "Label"]


def test_clean_table_removes_citations_from_headers_and_cells():
    df = pd.DataFrame(
        {"Album[1]": ["Foo[2]", "Bar[citation needed]"], "Year[a]": [1, 2]}
    )
    out = clean_table(df)
    assert list(out.columns) == ["Album", "Year"]
    assert out["Album"].tolist() == ["Foo", "Bar"]


def test_clean_table_drops_empty_and_duplicate_columns_and_empty_rows():
    df = pd.DataFrame(
        [["a", np.nan, "x"], [np.nan, np.nan, np.nan], ["b", np.nan, "y"]],
        columns=["A", "Empty", "A"],
    )
    out = clean_table(df)
    assert list(out.columns) == ["A"]
    assert out["A"].tolist() == ["a", "b"]


# --- docker_available ------------------------------------------------------------
def test_docker_available_false_without_cli(monkeypatch):
    monkeypatch.setattr(helper.shutil, "which", lambda _: None)
    assert docker_available() is False


def test_docker_available_false_when_image_missing(monkeypatch):
    monkeypatch.setattr(helper.shutil, "which", lambda _: "/usr/bin/docker")

    def fake_run(cmd, **_):
        if cmd[:3] == ["docker", "image", "inspect"]:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    assert docker_available() is False


def test_docker_available_true_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(helper.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(helper.subprocess, "run", lambda cmd, **_: calls.append(cmd))
    assert docker_available() is True
    assert docker_available() is True
    assert len(calls) == 2  # docker info + image inspect, checked only once


# --- run_in_docker ---------------------------------------------------------------
def test_run_in_docker_builds_locked_down_command(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, "out", "err")

    monkeypatch.setattr(helper, "SANDBOX_DATA_DIRS", ["/data"])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    assert run_in_docker("print(1)") == (0, "out", "err")

    cmd = " ".join(seen["cmd"])
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--user 65534:65534",
        "-v /data:/data:ro",
        "python -I -",
    ):
        assert flag in cmd
    assert seen["kwargs"]["input"] == "print(1)"


def test_run_in_docker_timeout_kills_container(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    assert run_in_docker("while True: pass", timeout=3) == "ERROR: timeout after 3s"

    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "kill", name]


# --- run_local (runs a real subprocess) ----------------------------------------
def test_run_local_returns_stdout():
    assert run_local("print(1 + 2)") == (0, "3\n", "")


def test_run_local_returns_error():
    code, out, err = run_local("raise ValueError('boom')")
    assert code != 0
    assert out == ""
    assert "ValueError: boom" in err


def test_run_local_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    _, out, _ = run_local("import os; print(os.environ.get('SECRET_TOKEN'))")
    assert out == "None\n"


def test_run_local_timeout():
    assert run_local("while True: pass", timeout=1) == "ERROR: timeout after 1s"
