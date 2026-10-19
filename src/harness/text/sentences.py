"""Sentence / word / paragraph splitting shared by validators and the claim verifier."""
import re

_ABBR = ["et al.", "e.g.", "i.e.", "vs.", "cf.", "Fig.", "No.", "pp.", "p.", "Dr.", "Prof.", "approx.", "Ltd.", "Inc."]
_DOT = "․"  # one-dot leader, stands in for a protected period
_SPLIT = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z0-9"(\[])')


def _protect(text: str) -> str:
    for a in _ABBR:
        text = re.sub(re.escape(a), a.replace(".", _DOT), text)
    return re.sub(r"(\d)\.(\d)", rf"\1{_DOT}\2", text)


def split_sentences(text: str) -> list[str]:
    text = _protect(re.sub(r"\s+", " ", text or "").strip())
    if not text:
        return []
    parts = _SPLIT.split(text)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'’\-]+", text or "")


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
