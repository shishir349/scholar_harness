"""Integrity gate for any rewrite: what must come out exactly as it went in.

Hard checks (a failure rejects the rewrite): citations (author + year), every number, quotations word for word,
figure/table blocks, headings, paragraph count. Soft checks (reported only): hedge words."""
import re
from collections import Counter

from .citations import NARRATIVE_RE, extract_citations, first_author

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
QUOTE = re.compile(r'"([^"\n]{4,})"|“([^”\n]{4,})”')
BLOCK = re.compile(r"\[(?:FIGURE|TABLE)_START\].*?\[(?:FIGURE|TABLE)_END\]", re.S)
HEDGE = re.compile(r"\b(?:may|might|could|appears? to|seems? to|suggests?|tends? to|likely|unlikely|perhaps|possibly|limited|only)\b", re.I)


_STRAIGHT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def norm(text: str) -> str:
    """Curly quotes and apostrophes read the same as straight ones (a rewriter often swaps them)."""
    return text.translate(_STRAIGHT)


_GROUP = re.compile(r"\(([^()]*?\d{4}[a-z]?[^()]*)\)")
_ITEM = re.compile(r"([A-Z][\w'-]+)(?: et al\.?| (?:and|&) [A-Z][\w'-]+)?,?\s+((?:19|20)\d\d)[a-z]?")


def _cites(text: str) -> set:
    """(first author, year) pairs: the shared extractor plus every item of a grouped '(A, 2018; B, 2021)' citation."""
    found = {(first_author(c["author"]).lower(), c["year"]) for c in extract_citations(text)}
    for g in _GROUP.findall(text):
        found |= {(a.lower(), int(y)) for a, y in _ITEM.findall(g)}
    return found


def numbers(text: str) -> Counter:
    """Every number except the years inside citations (those are guarded by _cites). A rewrite that splits a sentence may cite the
    work in both halves; counting the year twice must not read as a changed number."""
    plain = _GROUP.sub(lambda m: _ITEM.sub(" ", m.group(0)), text)
    return Counter(NUMBER.findall(NARRATIVE_RE.sub(lambda m: m.group(1), plain)))


def _paras(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _quotes(text: str) -> list[str]:
    return [a or b for a, b in QUOTE.findall(text)]


def _headings(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("#")]


def compare(before: str, after: str) -> list[str]:
    """[] when every protected span survived, else one short reason per failure."""
    before, after = norm(before), norm(after)
    out = []
    lost = _cites(before) - _cites(after)
    new = _cites(after) - _cites(before)
    if lost or new:
        out.append("citation changed: lost " + ", ".join(f"{a} {y}" for a, y in sorted(lost)) + "; added " + ", ".join(f"{a} {y}" for a, y in sorted(new)))
    nb, na = numbers(before), numbers(after)
    if nb != na:
        out.append("number changed: lost " + ", ".join(sorted((nb - na).elements())) + "; added " + ", ".join(sorted((na - nb).elements())))
    gone = [q for q in _quotes(before) if q not in after]
    if gone:
        out.append("quote changed: " + "; ".join(gone[:3]))
    if BLOCK.findall(before) != BLOCK.findall(after):
        out.append("figure/table block changed")
    if _headings(before) != _headings(after):
        out.append("heading changed")
    if len(_paras(before)) != len(_paras(after)):
        out.append(f"paragraph count changed: {len(_paras(before))} -> {len(_paras(after))}")
    return out


def soft(before: str, after: str) -> dict:
    """Reported, never failing: (before, after) counts."""
    return {"hedges": (len(HEDGE.findall(before)), len(HEDGE.findall(after)))}
