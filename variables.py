"""Settings, limits and shared clients used by the tools."""

import os
import re

import wikipedia
from dotenv import load_dotenv
from langchain_tavily import TavilyExtract, TavilySearch
from openai import OpenAI

load_dotenv()

# --- clients -----------------------------------------------------------------
client = OpenAI()
web_search = TavilySearch(max_results=5, topic="general")
web_extract = TavilyExtract(extract_depth="basic", format="markdown")

HEADERS = {"User-Agent": "gaia-agent/1.0 (HF agents course)"}
wikipedia.set_user_agent(HEADERS["User-Agent"])

# --- how much a tool may put into the LLM context ----------------------------
PREVIEW_CHARS = 2000  # max chars a tool puts into the LLM context in one call
READ_CHARS = 4000  # max chars read_stored returns in one call
SNIPPET_CHARS = 400  # max chars per search_web result snippet

# --- result store limits -----------------------------------------------------
STORE_MAX_TOTAL_CHARS = 20_000_000
STORE_MAX_ENTRY_CHARS = 500_000

# --- text patterns -----------------------------------------------------------
CITATION = r"\[(?:\d+|[a-z]|note \d+|citation needed|nb \d+)\]"
WIKI_HEADER_RE = re.compile(r"^(=+)\s*(.+?)\s*\1$")
MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# --- run_python sandbox -------------------------------------------------------
RUN_TIMEOUT = 30  # wall-clock seconds per run
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agent-sandbox")  # see sandbox.Dockerfile
# Host dirs mounted read-only at the same path in the container, so file paths
# given to the agent work unchanged. The whole HF hub cache is mounted because
# dataset snapshot files are symlinks into its blobs/ folder.
SANDBOX_DATA_DIRS = [
    d
    for d in os.getenv(
        "SANDBOX_DATA_DIRS",
        os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
    ).split(os.pathsep)
    if d and os.path.isdir(d)
]
