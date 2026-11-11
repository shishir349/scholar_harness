"""CONSTRAINTS.md as code: `section_violations` measures a finished section against every hard gate and returns what breaks.
Pure code, no model calls. The generator's own repairs and checks use the same numbers (harness/limits.py); this is the independent
count that tests, the live script and any release check read, so a gate cannot pass because the code that enforces it was weakened."""
import re
import statistics

from .. import limits as L
from ..text import cleanup
from ..text.citations import NARRATIVE_RE
from ..text.sentences import split_sentences

_ITEM = re.compile(r"[A-Z][\w'’-]+(?: et al\.?| (?:&|and) [A-Z][\w'’-]+)?,\s*(?:19|20)\d\d[a-z]?")
_GROUP = re.compile(r"\(([^()]*(?:19|20)\d\d[^()]*)\)")


def narrative_n(sentence: str) -> int:
    return sum(1 for m in NARRATIVE_RE.finditer(sentence) if sentence[max(0, m.start() - 1):m.start()] != "(")


def parenthetical_n(text: str) -> int:
    return sum(len(_ITEM.findall(g)) for g in _GROUP.findall(text))


def _body(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text or "") if p.strip() and not cleanup._BLOCKISH.match(p)]


def paragraph_issues(p: str) -> list[str]:
    n = [len(s.split()) for s in split_sentences(p)]
    out = []
    if len(n) < L.MIN_SENTS or len(p.split()) < L.MIN_WORDS:
        out.append("micro paragraph")
    if len(n) >= L.MIN_SENTS:
        if statistics.pstdev(n) < L.STD_MIN:
            out.append(f"std {statistics.pstdev(n):.1f} < {L.STD_MIN}")
        if max(n) > L.MAX_SENT:
            out.append(f"sentence of {max(n)} words > {L.MAX_SENT}")
        if min(n) > L.SHORT_MAX:
            out.append(f"no sentence of {L.SHORT_MAX} words or fewer")
    return out


def section_violations(chunks) -> dict:
    """chunks: {open, mid, close} (or one string). -> {gate: detail}, with "ok": True only when every gate holds."""
    chunks = chunks if isinstance(chunks, dict) else {"open": chunks, "mid": "", "close": ""}
    full = "\n\n".join(c for c in chunks.values() if c and c.strip())
    paras = _body(full)
    micro = [i for i, p in enumerate(paras) if "micro paragraph" in paragraph_issues(p)]
    cadence = [(i, w) for i, p in enumerate(paras) for w in paragraph_issues(p) if w != "micro paragraph"]
    per_call = {pos: sum(narrative_n(s) for p in _body(chunks.get(pos) or "") for s in split_sentences(p)) for pos in cleanup.POSITIONS}
    narrative, paren = sum(per_call.values()), parenthetical_n(full)
    share = paren / (paren + narrative) if paren + narrative else 1.0
    sents = [s for p in paras for s in split_sentences(p)]
    sets = [cleanup._content(s) for s in sents]
    dup = [(i, j) for i in range(len(sents)) for j in range(i + 1, len(sents))
           if len(sents[i].split()) >= 8 and len(sents[j].split()) >= 8 and cleanup.same_idea(sets[i], sets[j], L.DUP_JACCARD)]
    seen, twice = {}, []
    for s in sents:
        for m in NARRATIVE_RE.finditer(s):
            if s[max(0, m.start() - 1):m.start()] == "(":
                continue
            key = (cleanup.first_author(m.group(1)).lower(), m.group(2))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 2:
                twice.append(key)
    out = {"micro_paragraphs": micro, "cadence": cadence, "narrative_per_call": per_call, "narrative_section": narrative,
           "paren_share": round(share, 2), "duplicate_sentence_pairs": dup, "author_year_summarised_twice": twice}
    out["ok"] = not (micro or cadence or any(n > L.NARR_CALL for n in per_call.values()) or narrative > L.NARR_SECTION
                     or share < L.PAREN_SHARE or dup or twice)
    return out
