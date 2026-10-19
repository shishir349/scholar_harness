"""Title token overlap. Port of titleSimilarity in acdyon-scholar crossref.service.js."""
import re


def _normalise(s) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def title_similarity(a, b):
    """Shared long words / smaller set. None when either title is too short to compare."""
    A = {w for w in _normalise(a).split(" ") if len(w) > 3}
    B = {w for w in _normalise(b).split(" ") if len(w) > 3}
    if not A or not B:
        return None
    return len(A & B) / min(len(A), len(B))
