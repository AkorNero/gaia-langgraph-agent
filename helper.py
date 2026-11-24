"""Text/table helpers (cleaning, headings, filtering, slicing) and the
run_python runners (Docker sandbox, local fallback)."""

import functools
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

import pandas as pd

from variables import (
    CITATION,
    MD_HEADER_RE,
    PREVIEW_CHARS,
    RUN_TIMEOUT,
    SANDBOX_DATA_DIRS,
    SANDBOX_IMAGE,
    WIKI_HEADER_RE,
)


def header(line: str) -> tuple[int, str] | None:
    """(level, name) if the line is a wiki '== X ==' or markdown '## X' heading."""
    line = line.strip()
    m = WIKI_HEADER_RE.match(line) or MD_HEADER_RE.match(line)
    return (len(m.group(1)), m.group(2)) if m else None


def sections(content: str) -> list[tuple[int, str]]:
    """(level, name) of every heading line."""
    return [h for line in content.splitlines() if (h := header(line))]


def outline(content: str, max_items: int = 40) -> str:
    """Indented list of headings, de-duplicated and capped."""
    secs = list(dict.fromkeys(sections(content)))
    if not secs:
        return "(no headings)"
    top = min(lvl for lvl, _ in secs)
    out = [f"{'  ' * (lvl - top)}- {name}" for lvl, name in secs[:max_items]]
    if len(secs) > max_items:
        out.append(f"... {len(secs) - max_items} more headings")
    return "\n".join(out)


def section_text(content: str, section: str) -> str | None:
    """Text from the matching heading up to the next heading of the same or higher level.
    An exact heading name wins over a partial match."""
    lines = content.splitlines()
    want = section.strip().casefold()
    heads = [(i, h) for i, line in enumerate(lines) if (h := header(line))]
    hit = next(((i, h) for i, h in heads if h[1].casefold() == want), None) or next(
        ((i, h) for i, h in heads if want in h[1].casefold()), None
    )
    if hit is None:
        return None
    start, (level, _) = hit
    end = next((i for i, h in heads if i > start and h[0] <= level), len(lines))
    return "\n".join(lines[start:end]).strip()


def keyword_lines(text: str, keywords: str, is_table: bool) -> str:
    """Lines matching any comma-separated keyword. Tables keep their header rows;
    prose lines are prefixed with the section they come from."""
    kws = [k.strip().casefold() for k in keywords.split(",") if k.strip()]
    lines = text.splitlines()
    out = lines[:2] if is_table else []
    section = ""
    for line in lines[2:] if is_table else lines:
        if h := header(line):
            section = h[1]
            continue
        if any(k in line.casefold() for k in kws):
            out.append(line if is_table or not section else f"[{section}] {line}")
    return "\n".join(out)


def slice_with_note(text: str, offset: int, limit: int, key: str) -> str:
    """Return text[offset:offset+limit] with a note telling the LLM how to read more."""
    chunk = text[offset : offset + limit]
    rest = len(text) - (offset + len(chunk))
    if rest > 0:
        chunk += (
            f"\n\n[{rest} more chars. Call read_stored(key='{key}', "
            f"offset={offset + len(chunk)}) or narrow with section=/keyword=.]"
        )
    return chunk


def clean_markdown(md: str) -> str:
    """Strip images, citation links and link URLs; drop empty (nav menu) headings."""
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images
    md = re.sub(r"\[\[[^\]]*\]\]\([^)]*\)", "", md)  # citations [[12]](#cite_note-12)
    md = re.sub(
        r'\[([^\]]*)\]\((?:\S*\s+"[^"]*"|\S*?)\)', r"\1", md
    )  # links -> their text
    md = re.sub(r"[ \t]+\n", "\n", md)
    # drop empty headings (nav menus): a heading directly followed by a heading
    # of the same or higher level, or by the end of the page
    lines = [ln for ln in md.splitlines() if ln.strip()]
    keep = []
    for i, ln in enumerate(lines):
        h = header(ln)
        nxt = header(lines[i + 1]) if i + 1 < len(lines) else (0, "")
        if h and nxt and nxt[0] <= h[0]:
            continue
        keep.append(ln)
    return "\n\n".join(keep)


def prose_preview(text: str, limit: int = PREVIEW_CHARS, min_len: int = 80) -> str:
    """First paragraphs of real prose, skipping short menu/link lines."""
    out, size = [], 0
    for ln in text.splitlines():
        if len(ln) < min_len or header(ln):
            continue
        out.append(ln)
        size += len(ln)
        if size >= limit:
            break
    return "\n".join(out)[:limit] or text[:limit]


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten headers, drop citations and empty/duplicate columns."""
    if isinstance(df.columns, pd.MultiIndex):  # flatten repeated multi-level headers
        df.columns = [
            " / ".join(
                dict.fromkeys(str(c) for c in col if not str(c).startswith("Unnamed"))
            )
            for col in df.columns
        ]
    df.columns = [re.sub(CITATION, "", str(c)).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    return df.replace(CITATION, "", regex=True)


# --- run_python runners ------------------------------------------------------
# Each runner returns (exit code, stdout, stderr), or an error string on timeout.
RunResult = tuple[int, str, str] | str


@functools.cache
def docker_available() -> bool:
    """True if the docker CLI exists, the daemon answers and the sandbox image is
    built. Checked once per process."""
    if not shutil.which("docker"):
        return False
    try:
        for cmd in (["docker", "info"], ["docker", "image", "inspect", SANDBOX_IMAGE]):
            subprocess.run(cmd, capture_output=True, check=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def run_in_docker(code: str, timeout: int = RUN_TIMEOUT) -> RunResult:
    """Run code in a throwaway container: no network, no host files except the
    data dirs (read-only), capped memory/CPU/processes, unprivileged user."""
    name = f"run-python-{uuid.uuid4().hex[:12]}"
    mounts = [arg for d in SANDBOX_DATA_DIRS for arg in ("-v", f"{d}:{d}:ro")]
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--memory", "2g", "--memory-swap", "2g",
        "--cpus", "1", "--pids-limit", "128",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--workdir", "/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
        *mounts,
        SANDBOX_IMAGE, "python", "-I", "-",  # code is read from stdin
    ]  # fmt: skip
    try:
        p = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Killing the docker CLI leaves the container running: kill it by name.
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        return f"ERROR: timeout after {timeout}s"
    return p.returncode, p.stdout, p.stderr


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


def run_local(code: str, timeout: int = RUN_TIMEOUT) -> RunResult:
    """Fallback when Docker is missing. Not a sandbox: the code can still read
    files and use the network. This only keeps secrets out of the child and caps
    its resources."""
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
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return f"ERROR: timeout after {timeout}s"
    return proc.returncode, stdout, stderr
