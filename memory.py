"""In-process memory: the result STORE and the lru_cache'd fetchers.

Full tool results live in STORE; the LLM only gets a preview + key.
For multi-worker production, swap ResultStore for a Redis-backed class with the
same put/get/keys interface, and the lru_caches for a shared Redis cache.
"""

import threading
from collections import OrderedDict
from functools import lru_cache
from io import StringIO

import pandas as pd
import requests
from wikipedia import page, search
from youtube_transcript_api import YouTubeTranscriptApi

from helper import clean_markdown, clean_table, slice_with_note
from variables import (
    HEADERS,
    PREVIEW_CHARS,
    STORE_MAX_ENTRY_CHARS,
    STORE_MAX_TOTAL_CHARS,
    client,
    web_extract,
    web_search,
)


class ResultStore:
    """Bounded key -> text store, evicts least recently used entries by total size."""

    def __init__(
        self,
        max_total_chars: int = STORE_MAX_TOTAL_CHARS,
        max_entry_chars: int = STORE_MAX_ENTRY_CHARS,
    ):
        self._data: OrderedDict[str, str] = OrderedDict()
        self._size = 0
        self._max_total = max_total_chars
        self._max_entry = max_entry_chars
        self._lock = threading.Lock()  # ToolNode runs parallel tool calls in threads

    def put(self, key: str, text: str) -> None:
        text = text[: self._max_entry]
        with self._lock:
            if key in self._data:
                self._size -= len(self._data.pop(key))
            self._data[key] = text
            self._size += len(text)
            while self._size > self._max_total and len(self._data) > 1:
                _, old = self._data.popitem(last=False)
                self._size -= len(old)

    def get(self, key: str) -> str | None:
        with self._lock:
            text = self._data.get(key)
            if text is not None:
                self._data.move_to_end(key)
            return text

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)


STORE = ResultStore()


def stash(key: str, text: str, preview: int = PREVIEW_CHARS) -> str:
    """Store the full text and return only a preview."""
    STORE.put(key, text)
    return slice_with_note(text, 0, preview, key)


# --- cached fetchers (lru_cache does not cache exceptions, so failures are retried)
@lru_cache(maxsize=128)
def cached_wiki_search(query: str) -> tuple[str, ...]:
    return tuple(search(query, 3))


@lru_cache(maxsize=64)
def cached_wiki_page(title: str) -> tuple[str, str, str]:
    """(title, url, plain text content) of a Wikipedia page."""
    p = page(title, auto_suggest=False)
    return p.title, p.url, p.content


@lru_cache(maxsize=32)
def cached_wiki_tables(title: str) -> tuple[pd.DataFrame, ...]:
    """Cleaned tables of an English Wikipedia page."""
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    try:
        return tuple(clean_table(t) for t in pd.read_html(StringIO(res.text)))
    except ValueError:  # no tables on the page
        return ()


@lru_cache(maxsize=256)
def cached_web_search(query: str) -> tuple[tuple[str, str, str], ...]:
    """(title, url, snippet) per result. Cached: every Tavily call costs credits."""
    res = web_search.invoke({"query": query})
    if not isinstance(res, dict):
        raise TypeError(str(res)[:200])
    if res.get("error"):
        raise RuntimeError(str(res["error"])[:200])
    return tuple(
        (r.get("title") or "", r.get("url") or "", r.get("content") or "")
        for r in res.get("results", [])
    )


@lru_cache(maxsize=64)
def cached_fetch_page(url: str) -> tuple[str, str]:
    """(title, cleaned markdown) of a web page."""
    res = web_extract.invoke({"urls": [url]})
    if not isinstance(res, dict) or not res.get("results"):
        failed = res.get("failed_results") if isinstance(res, dict) else res
        raise RuntimeError(f"could not fetch {url}: {str(failed)[:200]}")
    r = res["results"][0]
    return r.get("title") or url, clean_markdown(r.get("raw_content") or "")


@lru_cache(maxsize=32)
def cached_youtube_transcript(video_id: str) -> str:
    """Full transcript text of a YouTube video."""
    data = YouTubeTranscriptApi().fetch(video_id=video_id).to_raw_data()
    return " ".join(s["text"] for s in data)


@lru_cache(maxsize=32)
def cached_transcription(file_path: str) -> str:
    """Whisper transcript of an audio file. Cached: every call costs money and time."""
    with open(file_path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f).text
