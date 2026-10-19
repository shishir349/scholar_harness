"""Copy guard: flag a draft that reproduces exemplar wording."""
import re


def _norm(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def overlap_report(draft: str, exemplar_texts: list[str], n: int = 8) -> dict:
    """{max_run, shared_ratio, flagged}. flagged = a 12+ word verbatim run or >2% of draft 8-grams shared."""
    d = _norm(draft)
    grams = {tuple(d[i:i + n]) for i in range(len(d) - n + 1)}
    total = max(1, len(d) - n + 1)
    shared_pos = [False] * len(d)
    seen: set[tuple] = set()
    for ex in exemplar_texts or []:
        e = _norm(ex)
        eg = {tuple(e[i:i + n]) for i in range(len(e) - n + 1)}
        hits = grams & eg
        seen |= hits
    for i in range(len(d) - n + 1):
        if tuple(d[i:i + n]) in seen:
            for j in range(i, i + n):
                shared_pos[j] = True
    run = best = 0
    for s in shared_pos:
        run = run + 1 if s else 0
        best = max(best, run)
    n_shared = sum(1 for i in range(len(d) - n + 1) if tuple(d[i:i + n]) in seen)
    ratio = n_shared / total if len(d) >= n else 0.0
    return {"max_run": best, "shared_ratio": round(ratio, 4), "flagged": best >= 12 or ratio > 0.02}


def exemplar_ngrams(exemplar_texts: list[str], n: int = 8) -> set[tuple]:
    """The n-gram vocabulary of the exemplars, once, so many passages can be checked against it cheaply."""
    out: set[tuple] = set()
    for ex in exemplar_texts or []:
        e = _norm(ex)
        out |= {tuple(e[i:i + n]) for i in range(len(e) - n + 1)}
    return out


def passage_overlaps(passage: str, exemplar_grams: set[tuple], n: int = 8) -> bool:
    """Does this one passage (e.g. a paragraph) share an n-gram with the exemplar vocabulary?
    Lets a revision localize copied wording to the specific paragraph instead of the whole chunk."""
    d = _norm(passage)
    return any(tuple(d[i:i + n]) in exemplar_grams for i in range(len(d) - n + 1))
