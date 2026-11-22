"""Pure text/table helpers: cleaning, headings, filtering, slicing. No I/O, no state."""

import re

import pandas as pd

from variables import CITATION, MD_HEADER_RE, PREVIEW_CHARS, WIKI_HEADER_RE


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
