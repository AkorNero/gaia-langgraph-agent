"""Tools the agents call. Logic lives in helper.py, memory.py and variables.py."""

import base64
import mimetypes
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests
from langchain_core.tools import tool
from wikipedia.exceptions import DisambiguationError, PageError
from youtube_transcript_api import YouTubeTranscriptApi

from helper import (
    keyword_lines,
    outline,
    prose_preview,
    section_text,
    sections,
    slice_with_note,
)
from memory import (
    STORE,
    cached_fetch_page,
    cached_web_search,
    cached_wiki_page,
    cached_wiki_search,
    cached_wiki_tables,
    stash,
)
from variables import PREVIEW_CHARS, READ_CHARS, SNIPPET_CHARS, client


@tool
def search_web(query: str) -> str:
    """
    Search the web. Returns the top 5 results (title, URL, short snippet).
    Snippets are cut: use fetch_page(url) to read a full page.
    """
    try:
        results = cached_web_search(query)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: web search failed ({type(e).__name__}: {e}). Try again or rephrase."
    if not results:
        return "No results."
    return "\n\n".join(
        f"{i}. {title}\n   URL: {url}\n   "
        + (snip[:SNIPPET_CHARS] + "..." if len(snip) > SNIPPET_CHARS else snip)
        for i, (title, url, snip) in enumerate(results, 1)
    )


@tool
def fetch_page(url: str) -> str:
    """
    Read a web page (e.g. a URL from search_web). Returns the start of the page
    and its list of headings. The full text is stored: read a section or
    keyword matches with read_stored.
    """
    try:
        title, text = cached_fetch_page(url)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}. Try another URL from the search results."
    if not text:
        return f"Page '{url}' has no readable text. Try another URL."
    key = f"web:{url}"
    STORE.put(key, text)
    return (
        f"# {title}\nURL: {url}\n\n## Main text (preview)\n{prose_preview(text)}\n\n"
        f"## Headings\n{outline(text, max_items=25)}\n\n"
        f"Full text ({len(text)} chars) stored as key '{key}'. Use "
        f"read_stored(key='{key}', keyword='a, b') or read_stored(key='{key}', section='...')."
    )


@tool
def search_wiki(query: str) -> str:
    """
    Search Wikipedia. Returns the top page's intro and its list of sections.
    The full text is stored: read a section or keyword matches with read_stored.
    Tables are not included: use wikipedia_tables for those.
    """
    titles: tuple[str, ...] = ()
    try:
        titles = cached_wiki_search(query)
        if not titles:
            return "No results."
        title, url, content = cached_wiki_page(titles[0])
    except DisambiguationError as e:
        return f"'{titles[0]}' is ambiguous. Options: {e.options[:15]}"
    except PageError:
        return f"Page '{titles[0]}' not found. Other results: {list(titles[1:])}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: Wikipedia request failed ({type(e).__name__}). Try again or use search_web."

    key = f"wiki:{title}"
    STORE.put(key, content)
    first_header = re.search(r"^=+ .+ =+$", content, flags=re.MULTILINE)
    intro = content[: first_header.start() if first_header else len(content)].strip()
    sections = outline(content, max_items=60)
    return (
        f"# {title}\nURL: {url}\nOther results: {list(titles[1:])}\n\n"
        f"## Intro\n{intro[:PREVIEW_CHARS]}\n\n## Sections\n{sections}\n\n"
        f"Full text ({len(content)} chars) stored as key '{key}'. Use "
        f"read_stored(key='{key}', section='...') or read_stored(key='{key}', keyword='a, b')."
    )


@tool
def wikipedia_tables(title: str, table_index: int = -1, match: str = "") -> str:
    """
    Tables of an English Wikipedia page. Give the exact page title, e.g. "Mercedes Sosa".
    Without table_index: returns a short overview of the tables (size, columns, first row).
    With table_index: returns that table as markdown.
    match: only list tables whose text contains this string (case-insensitive).
    """
    try:
        tables = cached_wiki_tables(title)
    except requests.HTTPError as e:
        return f"ERROR: HTTP {e.response.status_code} for page '{title}'. Check the exact title."  # ty: ignore[missing-union-attribute]
    except requests.RequestException as e:
        return f"ERROR: request failed ({type(e).__name__}). Try again."
    if not tables:
        return "No tables found on this page."

    if table_index >= 0:
        if table_index >= len(tables):
            return f"ERROR: table_index out of range, page has {len(tables)} tables (0-{len(tables) - 1})."
        md = tables[table_index].to_markdown(index=False)
        return stash(f"table:{title}:{table_index}", md, preview=READ_CHARS)

    lines = []
    for i, t in enumerate(tables):
        if str(t.columns[0]).startswith("vte"):  # navigation boxes, never useful
            continue
        if match and match.casefold() not in t.to_string().casefold():
            continue
        cols = [str(c) for c in t.columns[:8]]
        first = " | ".join(str(v) for v in t.iloc[0].tolist()[:8]) if len(t) else ""
        lines.append(
            f"Table {i}: {t.shape[0]} rows x {t.shape[1]} cols. "
            f"Columns: {cols}{' ...' if t.shape[1] > 8 else ''}\n  First row: {first[:200]}"
        )
    if not lines:
        return f"No table contains '{match}'. The page has {len(tables)} tables."
    return (
        "\n".join(lines)
        + "\n\nCall wikipedia_tables(title, table_index=N) to get one table."
    )


@tool
def read_stored(key: str, section: str = "", keyword: str = "", offset: int = 0) -> str:
    """
    Read more of a stored tool result (keys are given by search_wiki, fetch_page,
    wikipedia_tables and get_youtube_transcript).
    section: a heading name, returns only that section.
    keyword: comma-separated words, returns only the lines/table rows containing any of them.
    offset: start position, for reading on after a truncated result.
    """
    text = STORE.get(key)
    if text is None:
        return f"No stored result '{key}'. Stored keys: {STORE.keys()[-20:]}. Rerun the original tool."
    if section:
        sec = section_text(text, section)
        if sec is None:
            names = list(dict.fromkeys(n for _, n in sections(text)))[:60]
            return f"Section '{section}' not found. Sections: {names}"
        text = sec
    if keyword:
        text = keyword_lines(text, keyword, is_table=key.startswith("table:"))
        if not text.strip():
            return f"No lines contain any of: {keyword}"
    return slice_with_note(text, offset, READ_CHARS, key)


@tool
def get_youtube_transcript(video_id: str) -> str:
    """
    Return the transcript text of a YouTube video (long transcripts are stored,
    read more with read_stored).
    Give only the video ID, for example "1htKBjuUWec" from
    https://www.youtube.com/watch?v=1htKBjuUWec. Not the full URL.
    """
    try:
        data = YouTubeTranscriptApi().fetch(video_id=video_id).to_raw_data()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: no transcript available ({type(e).__name__})"
    return stash(
        f"yt:{video_id}", " ".join(s["text"] for s in data), preview=READ_CHARS
    )


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Use this tool to transcribe a audio file
    """
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text


@tool
def read_excel(file_path: str) -> str:
    """
    Read a CSV or Excel file and return all sheets as markdown tables.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path).to_markdown()
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(file_path, sheet_name=None)
        return "\n\n".join(
            f"## Sheet: {name}\n{df.to_markdown()}" for name, df in sheets.items()
        )
    return f"Unsupported file type: {ext}"


@tool
def read_file(file_path: str) -> str | list[dict]:
    """
    Read a file. Text files return their text; images (.png, .jpg, .jpeg, .gif, .webp)
    are returned as an image you can see. Not for Excel/CSV: use read_excel.
    """
    path = Path(file_path)
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        data = base64.b64encode(path.read_bytes()).decode()
        return [
            {"type": "text", "text": f"Image {path.name}:"},
            {"type": "image", "base64": data, "mime_type": mime},
        ]
    return path.read_text()


RUN_TIMEOUT = 30  # wall-clock seconds


# Runs inside the child before the LLM's code. Soft and hard limits are both set,
# and an unprivileged process can't raise a hard limit, so the code can't undo them.
# (Done here rather than with preexec_fn, which can deadlock in threaded callers.)
_RUN_PRELUDE = """\
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))                # CPU seconds
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))       # 2 GiB memory
resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))  # 50 MB per file
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))                 # no core dumps
_code = sys.argv.pop(1)
exec(compile(_code, "<code>", "exec"), {"__name__": "__main__"})
"""


@tool
def run_python(code: str) -> str:
    """
    Run Python code and return stdout and stderr.
    Use print() to show results. Pandas and standard library are available.
    """
    # Not a sandbox: the code can still read files and use the network.
    # This only keeps secrets out of the child and caps its resources.
    with tempfile.TemporaryDirectory() as tmp:
        env = {
            "PATH": "/usr/bin:/bin",  # no API keys from .env reach the child
            "HOME": tmp,
            "OMP_NUM_THREADS": "1",  # keep numpy/pandas inside the memory cap
            "OPENBLAS_NUM_THREADS": "1",
        }
        proc = subprocess.Popen(
            # -I: ignore PYTHON* vars and user site-packages
            [sys.executable, "-I", "-c", _RUN_PRELUDE, code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp,
            env=env,
            start_new_session=True,  # own process group, so a timeout kills children too
        )
        try:
            stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return f"ERROR: timeout after {RUN_TIMEOUT}s"
    out = stdout[-5000:]
    if proc.returncode != 0:
        out += f"\nERROR (exit {proc.returncode}):\n{stderr[-3000:]}"
    return out or "No output. Use print()."


if __name__ == "__main__":
    import os

    from huggingface_hub import snapshot_download

    data_dir = (
        Path(
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
                local_files_only=True,
            )
        )
        / "2023"
        / "validation"
    )

    tests = [
        (search_web, {"query": "Mercedes Sosa studio albums"}),
        (fetch_page, {"url": "https://www.nasa.gov/missions/apollo/apollo-11/"}),
        (
            read_stored,
            {
                "key": "web:https://www.nasa.gov/missions/apollo/apollo-11/",
                "keyword": "Armstrong",
            },
        ),
        (search_wiki, {"query": "Mercedes Sosa"}),
        (read_stored, {"key": "wiki:Mercedes Sosa", "section": "Discography"}),
        (read_stored, {"key": "wiki:Mercedes Sosa", "keyword": "2000, 2009"}),
        (wikipedia_tables, {"title": "Mercedes Sosa"}),
        (wikipedia_tables, {"title": "Mercedes Sosa", "match": "studio"}),
        (wikipedia_tables, {"title": "Mercedes Sosa", "table_index": 0}),
        (read_stored, {"key": "table:Mercedes Sosa:0", "keyword": "200"}),
        (get_youtube_transcript, {"video_id": "1htKBjuUWec"}),
        (
            transcribe_audio,
            {"file_path": str(data_dir / "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3")},
        ),
        (
            read_excel,
            {"file_path": str(data_dir / "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx")},
        ),
        (
            read_file,
            {"file_path": str(data_dir / "f918266a-b3e0-4914-865d-4faa564f1aef.py")},
        ),
        (run_python, {"code": "print(sum(range(10)))"}),
        (run_python, {"code": "raise ValueError('test error')"}),
        (run_python, {"code": "while True: pass"}),
    ]

    for t, args in tests:
        print(f"\n===== {t.name} {args} =====")
        try:
            print(str(t.invoke(args))[:500])
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")
