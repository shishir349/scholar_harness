"""Reduce chapter markup to plain prose for style measurement."""
import re

_ENTITIES = (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"))


def strip_for_style(text: str) -> str:
    """Prose paragraphs only: no figure/table blocks, tags, markdown tables or headings."""
    t = text or ""
    t = re.sub(r"\[FIGURE_START\][\s\S]*?\[FIGURE_END\]", "\n\n", t)
    t = re.sub(r"\[TABLE_START\][\s\S]*?\[TABLE_END\]", "\n\n", t)
    t = re.sub(r"<div[^>]*data-thesis-figure[^>]*>\s*</div>", "\n\n", t, flags=re.I)
    t = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", t, flags=re.I)
    # headings become their own marker lines first so they can be dropped or used to slice
    t = re.sub(r"<h[1-6][^>]*>([\s\S]*?)</h[1-6]>", r"\n\n# \1\n\n", t, flags=re.I)
    t = re.sub(r"</p\s*>|<br\s*/?>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in _ENTITIES:
        t = t.replace(a, b)
    lines = [ln for ln in t.split("\n") if not ln.lstrip().startswith(("|", "#"))]
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def headings_and_text(text: str) -> str:
    """Like strip_for_style but keeps heading lines ("# 2.1 Title") for section slicing."""
    t = text or ""
    t = re.sub(r"\[FIGURE_START\][\s\S]*?\[FIGURE_END\]", "\n\n", t)
    t = re.sub(r"\[TABLE_START\][\s\S]*?\[TABLE_END\]", "\n\n", t)
    t = re.sub(r"<div[^>]*data-thesis-figure[^>]*>\s*</div>", "\n\n", t, flags=re.I)
    t = re.sub(r"<h[1-6][^>]*>([\s\S]*?)</h[1-6]>", r"\n\n# \1\n\n", t, flags=re.I)
    t = re.sub(r"</p\s*>|<br\s*/?>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in _ENTITIES:
        t = t.replace(a, b)
    t = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", t))
    return t.strip()
