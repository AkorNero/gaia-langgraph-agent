import base64
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import wikipedia
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import tavily_search
from openai import OpenAI
from wikipedia import page, search
from wikipedia.exceptions import DisambiguationError, PageError
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

client = OpenAI()

web_search = tavily_search.TavilySearch(max_results=2, topic="general")

HEADERS = {"User-Agent": "gaia-agent/1.0 (HF agents course)"}
wikipedia.set_user_agent(HEADERS["User-Agent"])


@tool
def search_web(query: str) -> str:
  """
  Search the web and return the top 2 results (title, URL, content).
  """
  res = web_search.invoke({"query": query})
  if isinstance(res, dict) and res.get("results"):
    return "\n\n---\n\n".join(
      f"# {r.get('title')}\nURL: {r.get('url')}\n{r.get('content')}"
      for r in res["results"]
    )
  return str(res) or "No results."


@tool
def search_wiki(query: str) -> str:
  """
  Search Wikipedia and return the full text of the top page.
  The text has no tables. Use wikipedia_tables to get tables.
  """
  try:
    titles = search(query, 3)
    if not titles:
      return "No results."
    p = page(titles[0], auto_suggest=False)
  except DisambiguationError as e:
    return f"'{titles[0]}' is ambiguous. Options: {e.options[:15]}"  # ty: ignore[possibly-unresolved-reference]
  except PageError:
    return f"Page '{titles[0]}' not found. Other results: {titles[1:]}"  # ty: ignore[possibly-unresolved-reference]
  except Exception as e:
    return f"ERROR: Wikipedia request failed ({type(e).__name__}). Try again or use search_web."
  return (
    f"# {p.title}\nURL: {p.url}\nOther results: {titles[1:]}\n\n{p.content[:20000]}"
  )


@tool
def wikipedia_tables(title: str) -> str:
  """
  Return the tables of an English Wikipedia page as markdown.
  Give the exact page title, for example "Mercedes Sosa".
  """
  url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
  res = requests.get(url, headers=HEADERS, timeout=30)
  if res.status_code != 200:
    return f"ERROR: HTTP {res.status_code} for {url}"
  try:
    tables = pd.read_html(StringIO(res.text))
  except ValueError:
    return "No tables found on this page."
  return "\n\n".join(
    f"## Table {i}\n{t.to_markdown()}" for i, t in enumerate(tables[:10])
  )[:30000]


@tool
def get_youtube_transcript(video_id: str) -> str:
  """
  Return the transcript text of a YouTube video.
  Give only the video ID, for example "1htKBjuUWec" from
  https://www.youtube.com/watch?v=1htKBjuUWec. Not the full URL.
  """
  try:
    data = YouTubeTranscriptApi().fetch(video_id=video_id).to_raw_data()
  except Exception as e:
    return f"ERROR: no transcript available ({type(e).__name__})"
  return " ".join(s["text"] for s in data)


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
def read_file(file_path: str) -> str | bytes:
  """
  Use this tool if you want to get the text content of a file. Note: Can be used to read image as bytes. Can't be used with excel files
  """
  ext = Path(file_path).suffix.lower()
  if ext in {".png", ".jpg"}:
    with open(file_path, "rb") as f:
      return base64.b64encode(f.read()).decode()
  else:
    with open(file_path) as f:
      return f.read()


@tool
def run_python(code: str) -> str:
  """
  Run Python code and return stdout and stderr.
  Use print() to show results. Pandas and standard library are available.
  """
  with tempfile.TemporaryDirectory() as tmp:
    try:
      res = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp,
      )
    except subprocess.TimeoutExpired:
      return "ERROR: timeout after 30s"
  out = res.stdout[-5000:]
  if res.returncode != 0:
    out += f"\nERROR (exit {res.returncode}):\n{res.stderr[-3000:]}"
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
    (search_wiki, {"query": "Mercedes Sosa"}),
    (wikipedia_tables, {"title": "Mercedes Sosa"}),
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
    except Exception as e:
      print(f"FAILED: {type(e).__name__}: {e}")
