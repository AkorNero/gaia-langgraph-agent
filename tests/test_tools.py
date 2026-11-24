"""Unit tests for the agent tools in tools.py. The cached fetchers, external
clients and the Python runners are replaced by fakes, so no network is used."""

from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from wikipedia.exceptions import DisambiguationError, PageError

import tools
from tools import (
    fetch_page,
    get_youtube_transcript,
    read_excel,
    read_file,
    read_stored,
    run_python,
    search_web,
    search_wiki,
    transcribe_audio,
    wikipedia_tables,
)
from variables import READ_CHARS, SNIPPET_CHARS


def raises(exc):
    def fn(*_, **__):
        raise exc

    return fn


# --- search_web ------------------------------------------------------------------
def test_search_web_formats_and_truncates_snippets(monkeypatch):
    long = "s" * (SNIPPET_CHARS + 10)
    monkeypatch.setattr(
        tools, "cached_web_search", lambda q: (("T1", "u1", "short"), ("T2", "u2", long))
    )
    out = search_web.invoke({"query": "q"})
    assert out == (
        "1. T1\n   URL: u1\n   short\n\n"
        f"2. T2\n   URL: u2\n   {'s' * SNIPPET_CHARS}..."
    )


def test_search_web_no_results(monkeypatch):
    monkeypatch.setattr(tools, "cached_web_search", lambda q: ())
    assert search_web.invoke({"query": "q"}) == "No results."


def test_search_web_error(monkeypatch):
    monkeypatch.setattr(tools, "cached_web_search", raises(RuntimeError("quota")))
    assert search_web.invoke({"query": "q"}).startswith(
        "ERROR: web search failed (RuntimeError: quota)"
    )


# --- fetch_page ------------------------------------------------------------------
def test_fetch_page_stores_text_and_returns_preview(monkeypatch, store):
    text = "## Intro\n\n" + "p" * 120 + "\n\n## Details\n\nmore"
    monkeypatch.setattr(tools, "cached_fetch_page", lambda url: ("My Page", text))
    out = fetch_page.invoke({"url": "http://x"})
    assert store.get("web:http://x") == text
    assert out.startswith("# My Page\nURL: http://x")
    assert "p" * 120 in out
    assert "## Headings\n- Intro\n- Details" in out
    assert f"Full text ({len(text)} chars) stored as key 'web:http://x'" in out


def test_fetch_page_empty_text(monkeypatch, store):
    monkeypatch.setattr(tools, "cached_fetch_page", lambda url: ("T", ""))
    assert "no readable text" in fetch_page.invoke({"url": "http://x"})
    assert store.keys() == []


def test_fetch_page_error(monkeypatch):
    monkeypatch.setattr(tools, "cached_fetch_page", raises(RuntimeError("blocked")))
    assert fetch_page.invoke({"url": "http://x"}).startswith("ERROR: blocked.")


# --- search_wiki -----------------------------------------------------------------
WIKI = "Sosa was a singer.\n\n== Life ==\nBorn 1935.\n\n== Discography ==\nAlbums."


def test_search_wiki_returns_intro_and_sections(monkeypatch, store):
    monkeypatch.setattr(tools, "cached_wiki_search", lambda q: ("Mercedes Sosa", "Other"))
    monkeypatch.setattr(tools, "cached_wiki_page", lambda t: (t, "http://w/MS", WIKI))
    out = search_wiki.invoke({"query": "sosa"})
    assert store.get("wiki:Mercedes Sosa") == WIKI
    assert out.startswith("# Mercedes Sosa\nURL: http://w/MS\nOther results: ['Other']")
    assert "## Intro\nSosa was a singer.\n\n" in out
    assert "## Sections\n- Life\n- Discography" in out
    assert "Born 1935" not in out


def test_search_wiki_no_results(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_search", lambda q: ())
    assert search_wiki.invoke({"query": "q"}) == "No results."


def test_search_wiki_disambiguation(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_search", lambda q: ("Mercury",))
    monkeypatch.setattr(
        tools, "cached_wiki_page", raises(DisambiguationError("Mercury", ["Planet", "Element"]))
    )
    assert search_wiki.invoke({"query": "q"}) == (
        "'Mercury' is ambiguous. Options: ['Planet', 'Element']"
    )


def test_search_wiki_page_not_found(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_search", lambda q: ("A", "B"))
    monkeypatch.setattr(tools, "cached_wiki_page", raises(PageError("A")))
    assert search_wiki.invoke({"query": "q"}) == "Page 'A' not found. Other results: ['B']"


def test_search_wiki_other_error(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_search", raises(ConnectionError()))
    assert search_wiki.invoke({"query": "q"}).startswith(
        "ERROR: Wikipedia request failed (ConnectionError)"
    )


# --- wikipedia_tables ------------------------------------------------------------
NAV = pd.DataFrame({"vte Navigation": ["links"]})
ALBUMS = pd.DataFrame({"Year": [1999, 2009], "Album": ["Studio One", "Live Two"]})
AWARDS = pd.DataFrame({"Award": ["Grammy"]})


@pytest.fixture
def tables(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_tables", lambda t: (NAV, ALBUMS, AWARDS))


def test_wikipedia_tables_overview_skips_nav_boxes(tables):
    out = wikipedia_tables.invoke({"title": "X"})
    assert "Table 0" not in out
    assert "Table 1: 2 rows x 2 cols. Columns: ['Year', 'Album']" in out
    assert "First row: 1999 | Studio One" in out
    assert "Table 2: 1 rows x 1 cols" in out
    assert out.endswith("to get one table.")


def test_wikipedia_tables_match_filter(tables):
    out = wikipedia_tables.invoke({"title": "X", "match": "STUDIO"})
    assert "Table 1" in out and "Table 2" not in out


def test_wikipedia_tables_match_nothing(tables):
    assert wikipedia_tables.invoke({"title": "X", "match": "zzz"}) == (
        "No table contains 'zzz'. The page has 3 tables."
    )


def test_wikipedia_tables_single_table_is_stored(tables, store):
    out = wikipedia_tables.invoke({"title": "X", "table_index": 1})
    assert "Year" in out and "Album" in out and "Live Two" in out
    assert store.get("table:X:1") == ALBUMS.to_markdown(index=False)


def test_wikipedia_tables_index_out_of_range(tables):
    assert wikipedia_tables.invoke({"title": "X", "table_index": 3}) == (
        "ERROR: table_index out of range, page has 3 tables (0-2)."
    )


def test_wikipedia_tables_no_tables(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_tables", lambda t: ())
    assert wikipedia_tables.invoke({"title": "X"}) == "No tables found on this page."


def test_wikipedia_tables_http_error(monkeypatch):
    resp = requests.Response()
    resp.status_code = 404
    monkeypatch.setattr(tools, "cached_wiki_tables", raises(requests.HTTPError(response=resp)))
    assert wikipedia_tables.invoke({"title": "Nope"}).startswith(
        "ERROR: HTTP 404 for page 'Nope'"
    )


def test_wikipedia_tables_request_error(monkeypatch):
    monkeypatch.setattr(tools, "cached_wiki_tables", raises(requests.ConnectionError()))
    assert wikipedia_tables.invoke({"title": "X"}) == (
        "ERROR: request failed (ConnectionError). Try again."
    )


# --- read_stored -----------------------------------------------------------------
def test_read_stored_missing_key_lists_keys(store):
    store.put("wiki:A", "text")
    assert read_stored.invoke({"key": "nope"}) == (
        "No stored result 'nope'. Stored keys: ['wiki:A']. Rerun the original tool."
    )


def test_read_stored_whole_text(store):
    store.put("k", "hello")
    assert read_stored.invoke({"key": "k"}) == "hello"


def test_read_stored_section(store):
    store.put("k", WIKI)
    assert read_stored.invoke({"key": "k", "section": "life"}) == "== Life ==\nBorn 1935."


def test_read_stored_section_not_found(store):
    store.put("k", WIKI)
    assert read_stored.invoke({"key": "k", "section": "Awards"}) == (
        "Section 'Awards' not found. Sections: ['Life', 'Discography']"
    )


def test_read_stored_keyword(store):
    store.put("k", WIKI)
    assert read_stored.invoke({"key": "k", "keyword": "1935"}) == "[Life] Born 1935."


def test_read_stored_section_and_keyword(store):
    store.put("k", WIKI)
    out = read_stored.invoke({"key": "k", "section": "Discography", "keyword": "born"})
    assert out == "No lines contain any of: born"


def test_read_stored_table_keyword_keeps_header(store):
    store.put("table:X:1", "| Year | Album |\n|---|---|\n| 1999 | A |\n| 2009 | B |")
    assert read_stored.invoke({"key": "table:X:1", "keyword": "2009"}) == (
        "| Year | Album |\n|---|---|\n| 2009 | B |"
    )


def test_read_stored_paginates_with_offset(store):
    text = "a" * READ_CHARS + "b" * 10
    store.put("k", text)
    first = read_stored.invoke({"key": "k"})
    assert first.startswith("a" * READ_CHARS + "\n\n[10 more chars.")
    assert f"offset={READ_CHARS}" in first
    assert read_stored.invoke({"key": "k", "offset": READ_CHARS}) == "b" * 10


# --- get_youtube_transcript ------------------------------------------------------
def test_youtube_transcript_joins_and_stores(monkeypatch, store):
    class FakeApi:
        def fetch(self, video_id):
            assert video_id == "abc"
            return SimpleNamespace(to_raw_data=lambda: [{"text": "hello"}, {"text": "world"}])

    monkeypatch.setattr(tools, "YouTubeTranscriptApi", FakeApi)
    assert get_youtube_transcript.invoke({"video_id": "abc"}) == "hello world"
    assert store.get("yt:abc") == "hello world"


def test_youtube_transcript_error(monkeypatch):
    class FakeApi:
        def fetch(self, video_id):
            raise LookupError("disabled")

    monkeypatch.setattr(tools, "YouTubeTranscriptApi", FakeApi)
    assert get_youtube_transcript.invoke({"video_id": "abc"}) == (
        "ERROR: no transcript available (LookupError)"
    )


# --- transcribe_audio ------------------------------------------------------------
def test_transcribe_audio_uses_whisper(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3")
    seen = {}

    def create(model, file):
        seen["model"], seen["data"] = model, file.read()
        return SimpleNamespace(text="spoken words")

    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    monkeypatch.setattr(tools, "client", fake)
    assert transcribe_audio.invoke({"file_path": str(audio)}) == "spoken words"
    assert seen == {"model": "whisper-1", "data": b"ID3"}


# --- read_excel ------------------------------------------------------------------
def test_read_excel_csv(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1], "b": ["x"]}).to_csv(path, index=False)
    out = read_excel.invoke({"file_path": str(path)})
    assert out.splitlines()[0].split("|")[2:4] == ["   a ", " b   "]
    assert "| x" in out


def test_read_excel_xlsx_all_sheets(tmp_path):
    path = tmp_path / "d.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"food": ["burger"]}).to_excel(w, sheet_name="Food", index=False)
        pd.DataFrame({"drink": ["soda"]}).to_excel(w, sheet_name="Drinks", index=False)
    out = read_excel.invoke({"file_path": str(path)})
    assert "## Sheet: Food" in out and "burger" in out
    assert "## Sheet: Drinks" in out and "soda" in out


def test_read_excel_unsupported(tmp_path):
    assert read_excel.invoke({"file_path": str(tmp_path / "d.TXT")}) == (
        "Unsupported file type: .txt"
    )


# --- read_file -------------------------------------------------------------------
def test_read_file_text(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("print('hi')\n")
    assert read_file.invoke({"file_path": str(path)}) == "print('hi')\n"


def test_read_file_image_returns_base64_block(tmp_path):
    path = tmp_path / "pic.png"
    path.write_bytes(b"\x89PNG")
    assert read_file.invoke({"file_path": str(path)}) == [
        {"type": "text", "text": "Image pic.png:"},
        {"type": "image", "base64": "iVBORw==", "mime_type": "image/png"},
    ]


# --- run_python ------------------------------------------------------------------
@pytest.fixture
def runner(monkeypatch):
    """Configure run_python's runner: runner(result, docker=False)."""
    used = []

    def configure(result, docker=False):
        monkeypatch.setattr(tools, "docker_available", lambda: docker)
        monkeypatch.setattr(tools, "run_in_docker", lambda code: used.append("docker") or result)
        monkeypatch.setattr(tools, "run_local", lambda code: used.append("local") or result)
        return used

    return configure


def test_run_python_prefers_docker(runner):
    used = runner((0, "out\n", ""), docker=True)
    assert run_python.invoke({"code": "x"}) == "out\n"
    assert used == ["docker"]


def test_run_python_falls_back_to_local(runner):
    used = runner((0, "out\n", ""), docker=False)
    run_python.invoke({"code": "x"})
    assert used == ["local"]


def test_run_python_appends_error(runner):
    runner((1, "partial\n", "Traceback...\nValueError: bad"))
    assert run_python.invoke({"code": "x"}) == (
        "partial\n\nERROR (exit 1):\nTraceback...\nValueError: bad"
    )


def test_run_python_truncates_long_output(runner):
    runner((0, "a" * 6000 + "END", ""))
    out = run_python.invoke({"code": "x"})
    assert len(out) == 5000 and out.endswith("END")


def test_run_python_timeout(runner):
    runner("ERROR: timeout after 30s")
    assert run_python.invoke({"code": "x"}) == "ERROR: timeout after 30s"


def test_run_python_no_output(runner):
    runner((0, "", ""))
    assert run_python.invoke({"code": "x"}) == "No output. Use print()."
