"""Shared fixtures. Dummy API keys are set before any project module is imported,
so building the OpenAI/Tavily clients in variables.py never needs real keys and
no test can reach a paid API by accident."""

import os

os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"

import pytest

import helper
import memory
import tools

CACHED = (
    memory.cached_wiki_search,
    memory.cached_wiki_page,
    memory.cached_wiki_tables,
    memory.cached_web_search,
    memory.cached_fetch_page,
    helper.docker_available,
)


def _clear_caches() -> None:
    for fn in CACHED:
        fn.cache_clear()


@pytest.fixture(autouse=True)
def store(monkeypatch) -> memory.ResultStore:
    """A fresh, empty STORE and empty caches for every test."""
    fresh = memory.ResultStore()
    monkeypatch.setattr(memory, "STORE", fresh)
    monkeypatch.setattr(tools, "STORE", fresh)
    _clear_caches()
    yield fresh
    _clear_caches()
