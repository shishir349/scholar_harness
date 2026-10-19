"""Turn a parsed thesis (markdown) into classified chapters, units and retrieval chunks."""
import re

from ..prompts.sections import classify_title  # noqa: F401  (re-exported: one vocabulary for corpus and runtime)

# Candidate chapter headings. Dotted numbers (2.1 ...) are sub-sections, never chapters.
_HEADING = re.compile(
    r"^(?:#{1,2}\s+(?P<md>(?!\d+\.\d+)[^\n]+)"
    r"|(?P<ch>(?:chapter|CHAPTER)\s+(?:\d+|[IVXL]+)\b.*)"
    r"|(?P<num>\d{1,2}\.?\s+[A-Z][^\n]{2,80}))$")
_SUBHEAD = re.compile(r"^(?:#{1,6}\s+(?P<md>.+)|(?P<num>\d+\.\d+\.?\s+[A-Z][^\n]{2,80}))$")
_REFS = re.compile(r"^(?:#{1,6}\s*)?(?:\d{1,2}\.?\s+)?(references|bibliography|works cited)\s*$", re.I)
_CAPTION = re.compile(r"^\s*(Figure|Table|Fig\.)\s+\d", re.I)
_IMAGE = re.compile(r"^\s*!\[.*\]\(.*\)\s*$")


def n_words(text: str) -> int:
    return len(text.split())


def clean_thesis_text(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    counts: dict[str, int] = {}
    for ln in lines:
        s = ln.strip()
        if s and len(s) < 120:
            counts[s] = counts.get(s, 0) + 1
    out, in_refs = [], False
    for ln in lines:
        s = ln.strip()
        if _REFS.match(s):
            in_refs = True
            continue
        if in_refs:
            # references end at the next chapter-level heading that is not itself a reference-ish line
            if _HEADING.match(s) and classify_title(_heading_text(_HEADING.match(s))) is not None:
                in_refs = False
            else:
                continue
        if _IMAGE.match(ln) or s.startswith("|") or _CAPTION.match(ln) or re.fullmatch(r"\d{1,4}", s):
            continue
        if s and len(s) < 120 and counts.get(s, 0) >= 5 and not s.startswith("#"):
            continue  # running header / footer
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _heading_text(m: re.Match) -> str:
    head = (m.group("md") or m.group("ch") or m.group("num")).strip().lstrip("#").strip()
    head = re.sub(r"^[*_]+|[*_]+$", "", head).strip()  # strip **bold**/_italic_ decoration
    return re.sub(r"\s{2,}", " ", head)


_FALLBACK_MIN_WORDS = 500  # below this, an unclassified heading is a stray sub-heading, not a real chapter
_PAGE_MARK = re.compile(r"<!-- page (\d+) -->\s*")
_NUM_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_BARE_CHAPTER = re.compile(rf"^chapter\s+(?:\d+|[ivxl]+|{_NUM_WORDS})\W*$", re.I)
_FIRST_CHAPTER = re.compile(r"^chapter\s+(?:1|i|one)\b", re.I)
_ANY_CHAPTER = re.compile(rf"^chapter\s+(?:\d+|[ivxl]+|{_NUM_WORDS})\b", re.I)
_SUBNUM = re.compile(r"^\d+\.\d+")  # 2.1 / 3.4.1 ...: a sub-section, never a chapter
_CAPTION_HEAD = re.compile(r"^(table|figure|fig\.)\s*\d", re.I)


def _is_chapter_heading(head: str) -> bool:
    """Filters the false positives the raw heading regex lets through: dotted sub-sections
    (even when bold-wrapped), table/figure captions, and numbered list items / sentences."""
    if not head or _SUBNUM.match(head) or _CAPTION_HEAD.match(head):
        return False
    if _BARE_CHAPTER.match(head):
        return True
    return len(head.split()) <= 12 and not head.endswith((".", ";", ",", ":")) and not re.search(r"[=≤≥<>]", head)


def _titlecase(head: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", head) if len(w) > 3]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= 0.6


def _chapter_like(head: str) -> bool:
    """Would this unclassified heading plausibly be a chapter title (vs a sub-heading inside a chapter)?"""
    return bool(_ANY_CHAPTER.match(head) or head.isupper()
                or (re.match(r"^\d{1,2}\.?\s", head) and _titlecase(head)))


def _finalize_chapters(raw: list[dict]) -> list[dict]:
    """raw: [{heading, type, text}] in document order, type possibly None. Applies the findings
    fallback and drops front matter/appendices. Shared by the TOC-based and heading-heuristic splitters."""
    # A chapter titled outside the known vocabulary (narrative/qualitative theses: "Identity
    # Journeys", "Thesis Reflection"...) is still real thesis body content, not noise. Anything
    # sitting between a recognised introduction and a recognised conclusion (or the end, if no
    # conclusion heading was found) is counted as findings instead of silently dropped -
    # otherwise a genuinely well-structured thesis fails the gate purely for using its own words.
    intro_idx = next((k for k, c in enumerate(raw) if c["type"] == "introduction"), None)
    numbered = [k for k, c in enumerate(raw) if c["type"] is None and _ANY_CHAPTER.match(c["heading"])
                and n_words(c["text"]) >= _FALLBACK_MIN_WORDS]
    if intro_idx is None:
        # Untitled "Chapter 1" is the introduction by convention.
        first = next((k for k in numbered if _FIRST_CHAPTER.match(raw[k]["heading"])), None)
        if first is not None:
            raw[first]["type"], intro_idx = "introduction", first
    concl_idx = next((k for k in range(len(raw) - 1, -1, -1) if raw[k]["type"] == "conclusion"), None)
    if concl_idx is None and intro_idx is not None:
        # ...and the last numbered chapter of a 4+ chapter thesis is its conclusion.
        later = [k for k in numbered if k > intro_idx]
        if len(later) >= 3:
            raw[later[-1]]["type"], concl_idx = "conclusion", later[-1]
    if intro_idx is not None:
        hi = concl_idx if concl_idx is not None else len(raw)
        for c in raw[intro_idx + 1:hi]:
            if c["type"] is None and n_words(c["text"]) >= _FALLBACK_MIN_WORDS:
                c["type"] = "findings"

    chapters: list[dict] = []
    for c in raw:
        if c["type"] == "appendix":
            if chapters:  # an "appendix" heading before any real chapter is front matter (list of tables in appendix...)
                break
            continue
        if c["type"] is None:
            continue
        chapters.append(c)
    return chapters


def split_chapters(md: str) -> list[dict]:
    """Classified chapters [{heading, type, text}] from heading-detection heuristics on the body text.
    Used when the PDF has no usable embedded table of contents (see split_chapters_from_toc)."""
    lines = clean_thesis_text(md).split("\n")
    marks = []
    for i, ln in enumerate(lines):
        m = _HEADING.match(ln.strip())
        head = _heading_text(m) if m else ""
        # Every chapter-like heading is a boundary, classified or not - otherwise an unclassified
        # chapter's text silently bleeds into whichever classified chapter came before it (real bug:
        # mislabels e.g. literature-review content as "conclusion"). Unclassified headings that do
        # NOT look like chapters ("Study site", "Ziquala district") are sub-headings and stay inside
        # the chapter they sit in.
        if m and _is_chapter_heading(head) and (classify_title(head) is not None or _chapter_like(head)):
            marks.append((i, head))
    raw = []
    for k, (i, head) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        if _BARE_CHAPTER.match(head):  # "Chapter 3" alone: the title is the next short line
            first = next((re.sub(r"^[#*_\s]+|[*_\s]+$", "", s) for s in body.split("\n")[:4] if s.strip()), "")
            if first and len(first.split()) <= 12 and not first.endswith((".", ";", ",")):
                head = f"{head} {first}"
        raw.append({"heading": head, "type": classify_title(head), "text": body})
    return _finalize_chapters(raw)


def _split_pages(md: str) -> list[str]:
    """md must carry the '<!-- page N -->' markers parse_pdf writes. Returns page texts, index 0 = page 1."""
    parts = _PAGE_MARK.split(md)
    if len(parts) < 3:
        return []
    pages: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        pages[int(parts[i])] = parts[i + 1]
    return [pages.get(p, "") for p in range(1, max(pages) + 1)]


def split_chapters_from_toc(md: str, toc: list) -> list[dict] | None:
    """Classified chapters straight from the PDF's own embedded outline/bookmarks (toc entries:
    [level, title, page]), when there is one. Exact chapter names and boundaries - no heading
    guesswork. None when the PDF has fewer than 3 top-level outline entries (no real per-chapter TOC)."""
    top = [(str(title).strip(), page) for level, title, page in toc if level == 1 and page and page > 0]
    if len(top) < 3:
        return None
    pages = _split_pages(clean_thesis_text(md))
    if not pages:
        return None
    raw = []
    for k, (title, start) in enumerate(top):
        end = top[k + 1][1] - 1 if k + 1 < len(top) else len(pages)
        start = max(1, min(start, len(pages)))
        end = max(start, min(end, len(pages)))
        text = "\n\n".join(pages[start - 1:end]).strip()
        raw.append({"heading": title, "type": classify_title(title), "text": text})
    return _finalize_chapters(raw)


def _windows(text: str, size: int = 1200) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur, cw = [], [], 0
    for p in paras:
        cur.append(p)
        cw += n_words(p)
        if cw >= size:
            out.append("\n\n".join(cur))
            cur, cw = [], 0
    if cur:
        out.append("\n\n".join(cur))
    return out


def split_units(chapter: dict, min_words: int = 300) -> list[dict]:
    """Sub-sections of a chapter (>= min_words). A chapter without sub-headings is cut into ~1200-word windows."""
    lines = chapter["text"].split("\n")
    marks = [(i, (m.group("md") or m.group("num")).strip()) for i, ln in enumerate(lines)
             if (m := _SUBHEAD.match(ln.strip()))]
    units: list[dict] = []
    if marks:
        pre = "\n".join(lines[:marks[0][0]]).strip()
        if pre:
            units.append({"heading": chapter["heading"], "text": pre})
        for k, (i, head) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
            units.append({"heading": head, "text": "\n".join(lines[i + 1:end]).strip()})
    else:
        units = [{"heading": chapter["heading"], "text": w} for w in _windows(chapter["text"])]
    return [u for u in units if n_words(u["text"]) >= min_words]


def chunk_unit(unit: dict, target: int = 450, max_words: int = 650) -> list[dict]:
    """Greedy paragraph chunks; first = open, last = close, rest = mid; a lone chunk is open."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", unit["text"]) if p.strip()]
    chunks: list[list[str]] = []
    cur: list[str] = []
    cw = 0
    for p in paras:
        pw = n_words(p)
        if cur and (cw >= target or cw + pw > max_words):
            chunks.append(cur)
            cur, cw = [], 0
        cur.append(p)
        cw += pw
    if cur:
        # fold a tiny tail into the previous chunk instead of leaving a fragment
        if chunks and cw < target // 3 and n_words("\n\n".join(chunks[-1])) + cw <= max_words:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    out = []
    for i, c in enumerate(chunks):
        pos = "open" if i == 0 else ("close" if i == len(chunks) - 1 else "mid")
        text = "\n\n".join(c)
        out.append({"heading": unit["heading"], "chunk_idx": i, "position": pos, "text": text, "word_count": n_words(text)})
    return out


def build_sections(md: str, toc: list | None = None) -> dict:
    """Everything the corpus needs from one thesis: chapters, units and chunks."""
    chapters = (split_chapters_from_toc(md, toc) if toc else None) or split_chapters(md)
    out_ch, chunk_rows = [], []
    for ch in chapters:
        units = split_units(ch)
        out_ch.append({"heading": ch["heading"], "type": ch["type"], "words": n_words(ch["text"]),
                       "units": [{"heading": u["heading"], "text": u["text"]} for u in units]})
        for u in units:
            for c in chunk_unit(u):
                chunk_rows.append({**c, "section_type": ch["type"]})
    return {"chapters": out_ch, "chunks": chunk_rows}
