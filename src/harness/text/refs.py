"""Spot reference-list text that leaked into thesis chunks (found by the phrase-gap analysis: 'Journal of', 'Cited on')."""
import re

# "Ackrich, M., 1992. ..." / "19. ASCO (2020) ..." style entries, and lines carrying a retrieval note or URL
_ENTRY = re.compile(r"^\s*[-*\d.\s]*[A-Z][\w'’\-]+,\s+(?:[A-Z]\.[\s,&]*){1,4}.{0,200}?(?:19|20)\d\d[a-z]?")
_NOTE = re.compile(r"Available (?:at|from)|Retrieved|Accessed|Cited on|https?://|doi:|\bJournal of\b")


def looks_like_references(text: str) -> bool:
    """True when at least 3 lines, and a quarter of all lines, read as bibliography entries."""
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    hits = sum(1 for ln in lines if _ENTRY.match(ln) or _NOTE.search(ln))
    return hits >= 3 and hits / len(lines) >= 0.25
