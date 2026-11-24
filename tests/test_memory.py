"""Unit tests for memory.py: ResultStore, stash and the cached fetchers.
Every external call (Tavily, Wikipedia, HTTP) is replaced by a fake."""

import threading
from types import SimpleNamespace

import pytest
import requests

import memory
from memory import (
    ResultStore,
    cached_fetch_page,
    cached_web_search,
    cached_wiki_page,
    cached_wiki_search,
    cached_wiki_tables,
    stash,
)


class FakeTool:
    """Stands in for TavilySearch / TavilyExtract: records calls, returns canned results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        res = self.results.pop(0)
        if isinstance(res, Exception):
            raise res
        return res


# --- ResultStore -----------------------------------------------------------------
def test_store_put_get_and_keys():
    s = ResultStore()
    s.put("a", "1")
    s.put("b", "2")
    assert s.get("a") == "1"
    assert s.get("missing") is None
    assert s.keys() == ["b", "a"]  # get moved "a" to most recent


def test_store_overwrite_replaces_text_and_size():
    s = ResultStore()
    s.put("a", "xxxx")
    s.put("a", "y")
    assert s.get("a") == "y"
    assert s._size == 1


def test_store_truncates_long_entries():
    s = ResultStore(max_total_chars=100, max_entry_chars=5)
    s.put("a", "abcdefgh")
    assert s.get("a") == "abcde"


def test_store_evicts_least_recently_used():
    s = ResultStore(max_total_chars=10, max_entry_chars=10)
    s.put("a", "aaaa")
    s.put("b", "bbbb")
    s.get("a")  # "b" is now the oldest
    s.put("c", "cccc")
    assert s.keys() == ["a", "c"]
    assert s._size == 8


def test_store_keeps_single_entry_over_total_limit():
    s = ResultStore(max_total_chars=3, max_entry_chars=10)
    s.put("a", "aa")
    s.put("big", "x" * 8)
    assert s.keys() == ["big"]


def test_store_is_thread_safe():
    s = ResultStore(max_total_chars=1_000, max_entry_chars=100)

    def worker(n):
        for i in range(200):
            s.put(f"{n}-{i}", "x" * (i % 50))
            s.get(f"{n}-{i // 2}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s._size == sum(len(v) for v in s._data.values())
    assert s._size <= 1_000


# --- stash -----------------------------------------------------------------------
def test_stash_stores_full_text_and_returns_preview(store):
    text = "x" * 50
    out = stash("k", text, preview=10)
    assert store.get("k") == text
    assert out.startswith("x" * 10 + "\n\n[40 more chars.")


def test_stash_short_text_returned_whole(store):
    assert stash("k", "short") == "short"
    assert store.get("k") == "short"


# --- cached_web_search -----------------------------------------------------------
def test_web_search_parses_results(monkeypatch):
    fake = FakeTool(
        {
            "results": [
                {"title": "T1", "url": "u1", "content": "c1"},
                {"title": None, "url": "u2"},
            ]
        }
    )
    monkeypatch.setattr(memory, "web_search", fake)
    assert cached_web_search("q") == (("T1", "u1", "c1"), ("", "u2", ""))
    assert fake.calls == [{"query": "q"}]


def test_web_search_is_cached(monkeypatch):
    fake = FakeTool({"results": []})
    monkeypatch.setattr(memory, "web_search", fake)
    cached_web_search("q")
    cached_web_search("q")
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "response, error",
    [({"error": "rate limited"}, RuntimeError), ("not a dict", TypeError)],
)
def test_web_search_errors_raise_and_are_not_cached(monkeypatch, response, error):
    fake = FakeTool(response, {"results": []})
    monkeypatch.setattr(memory, "web_search", fake)
    with pytest.raises(error):
        cached_web_search("q")
    assert cached_web_search("q") == ()  # retried, not cached


# --- cached_fetch_page -----------------------------------------------------------
def test_fetch_page_returns_title_and_cleaned_markdown(monkeypatch):
    fake = FakeTool(
        {"results": [{"title": "Page", "raw_content": "Hi ![img](x.png) [link](http://a)"}]}
    )
    monkeypatch.setattr(memory, "web_extract", fake)
    assert cached_fetch_page("http://x") == ("Page", "Hi  link")
    assert fake.calls == [{"urls": ["http://x"]}]


def test_fetch_page_title_falls_back_to_url(monkeypatch):
    monkeypatch.setattr(memory, "web_extract", FakeTool({"results": [{"raw_content": None}]}))
    assert cached_fetch_page("http://x") == ("http://x", "")


def test_fetch_page_failure_raises(monkeypatch):
    fake = FakeTool({"results": [], "failed_results": [{"error": "403"}]})
    monkeypatch.setattr(memory, "web_extract", fake)
    with pytest.raises(RuntimeError, match="could not fetch http://x.*403"):
        cached_fetch_page("http://x")


# --- Wikipedia -------------------------------------------------------------------
def test_wiki_search_returns_tuple_of_top_3(monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "search", lambda q, n: calls.append((q, n)) or ["A", "B"])
    assert cached_wiki_search("q") == ("A", "B")
    assert calls == [("q", 3)]


def test_wiki_page_disables_auto_suggest(monkeypatch):
    calls = []

    def fake_page(title, auto_suggest):
        calls.append((title, auto_suggest))
        return SimpleNamespace(title="T", url="http://w/T", content="body")

    monkeypatch.setattr(memory, "page", fake_page)
    assert cached_wiki_page("t") == ("T", "http://w/T", "body")
    assert calls == [("t", False)]


class FakeResponse:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


TABLE_HTML = """<html><body><table>
<tr><th>Year[1]</th><th>Album</th></tr>
<tr><td>1999</td><td>Foo[2]</td></tr>
</table></body></html>"""


def test_wiki_tables_fetches_and_cleans(monkeypatch):
    urls = []

    def fake_get(url, **_):
        urls.append(url)
        return FakeResponse(TABLE_HTML)

    monkeypatch.setattr(memory.requests, "get", fake_get)
    (table,) = cached_wiki_tables("Mercedes Sosa")
    assert urls == ["https://en.wikipedia.org/wiki/Mercedes_Sosa"]
    assert list(table.columns) == ["Year", "Album"]
    assert table.iloc[0].tolist() == [1999, "Foo"]


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="BUG: read_html falls back to bs4/html5lib when lxml finds no table; "
    "html5lib is not installed, so ImportError escapes instead of ValueError. "
    "Fix: pd.read_html(..., flavor='lxml').",
)
def test_wiki_tables_no_tables(monkeypatch):
    monkeypatch.setattr(memory.requests, "get", lambda *_, **__: FakeResponse("<p>none</p>"))
    assert cached_wiki_tables("X") == ()


def test_wiki_tables_http_error_raises(monkeypatch):
    monkeypatch.setattr(memory.requests, "get", lambda *_, **__: FakeResponse(status=404))
    with pytest.raises(requests.HTTPError):
        cached_wiki_tables("Nope")
