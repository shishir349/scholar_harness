"""Shares of paragraphs that carry a number, a first-person word or an example, the real-thesis targets for each, and the
numbers the harness is allowed to use. Measured on the corpus (paragraphs of 40+ words): numbers sit in 52-59% of real
paragraphs, but only 4-13% of numbered sentences also carry a citation, the first number is in the opening sentence half the
time, and 19% of numbered sentences point at the thesis itself ("Chapter 3", "Table 2", "RQ1"). Real numbers mostly come from the
author's own data, which the harness does not have, so the targets sit below the real share."""
import math
import re

from .cleanup import EXAMPLE, FIGURE, POSITIONS, paras

# share of real paragraphs (40+ words) with a number / a first-person word, by section type
NUM_PARA_REAL = {"introduction": 0.52, "literature_review": 0.46, "methodology": 0.51, "discussion": 0.58}
NUM_PARA_TARGET = {"introduction": 0.38, "literature_review": 0.38, "methodology": 0.38, "discussion": 0.42}
FP_PARA_REAL = {"introduction": 0.27, "literature_review": 0.07, "methodology": 0.36, "discussion": 0.29}
FP_PARA_TARGET = {"introduction": 0.24, "literature_review": 0.08, "methodology": 0.32, "discussion": 0.26}   # ~22% averaged over sections
EXAMPLES_PER_1000 = 1.5          # real theses 1.97, AI drafts 0.55
FP_VERBS = "chose, decided, used, treat, found, wanted, consider, observed, selected, read, compared, focus on, count, kept"
MIN_WORDS = 40
_STRUCTURE_TYPES = ("introduction", "methodology", "discussion", "literature_review")


def body_paras(chunks: dict) -> list[tuple[str, int, str]]:
    """[(pos, idx, text)] of body paragraphs with 40+ words (the idx matches graph.py: blank-line split, empty parts removed)."""
    return [(pos, i, p) for pos in POSITIONS for i, p in enumerate(paras(chunks[pos]))
            if not p.startswith("#") and len(p.split()) >= MIN_WORDS]


def has_figure(p: str) -> bool:
    return bool(FIGURE.search(re.sub(r"\bpp?\. ?\d[\d–-]*", "", re.sub(r"\[\d[\d,\s–-]*\]", "", p))))


def has_first_person(p: str) -> bool:
    return bool(re.search(r"\b(?:I|my|me|we|our|us)\b", p)) or bool(re.search(r"\b(?:We|Our|Us|My|Me)\b", p))


def share(chunks: dict, test) -> float | None:
    ps = body_paras(chunks)
    return sum(1 for _, _, p in ps if test(p)) / len(ps) if len(ps) >= 5 else None


def _gap(chunks: dict, test, target: float, cap: int = 3) -> int:
    """How many more paragraphs should satisfy `test` to reach `target`; 0 when the share is already at 70% of it or the draft is
    too small (under five paragraphs) to judge."""
    ps = body_paras(chunks)
    if len(ps) < 5:
        return 0
    have = sum(1 for _, _, p in ps if test(p))
    if have / len(ps) >= 0.7 * target:
        return 0
    return min(cap, math.ceil(target * len(ps)) - have)


def number_gap(chunks: dict, ptype: str) -> int:
    return _gap(chunks, has_figure, NUM_PARA_TARGET.get(ptype, 0.35))


def fp_gap(chunks: dict, ptype: str) -> int:
    return _gap(chunks, has_first_person, FP_PARA_TARGET.get(ptype, 0.22), cap=2)


def example_gap(chunks: dict) -> int:
    text = " ".join(chunks[pos] for pos in POSITIONS)
    words = len(text.split())
    if words < 600:
        return 0
    have = len(EXAMPLE.findall(text))
    return max(0, min(2, math.ceil(EXAMPLES_PER_1000 * words / 1000) - have)) if have < 0.75 * EXAMPLES_PER_1000 * words / 1000 else 0


def _text(v) -> str:
    return " ".join(str(v).split()).strip(' ."\'') if isinstance(v, (str, int, float)) else ""


def structure_facts(thesis: dict, chapter: dict, section: dict, ptype: str) -> list[dict]:
    """Numbers that are true by construction because the request holds them: the chapter number, the other sections' numbers, the
    objective and research-question numbers. Pointing at 'Research question 2' is what real authors do; there is no Table or
    Figure here because the request supplies none."""
    if ptype not in _STRUCTURE_TYPES:
        return []
    out = []
    qs, os_ = (thesis or {}).get("questions"), (thesis or {}).get("objectives")
    for i, q in enumerate(qs if isinstance(qs, list) else [], 1):
        if _text(q):
            out.append({"text": f'Research question {i} of this thesis is: "{_text(q)[:140]}". Refer to it as "Research question {i}"', "kind": "structure"})
    for i, o in enumerate(os_ if isinstance(os_, list) else [], 1):
        if _text(o):
            out.append({"text": f'Objective {i} of this thesis is: "{_text(o)[:140]}". Refer to it as "Objective {i}"', "kind": "structure"})
    n = (chapter or {}).get("chapter_number")
    if n:
        out.append({"text": f'This section is in Chapter {n}; mention the chapter by its number', "kind": "structure"})
    secs = (chapter or {}).get("sections") or ((chapter or {}).get("toc_sections") or {}).get("sections") or []
    for s in secs:
        if isinstance(s, dict) and s.get("number") and s.get("number") != (section or {}).get("number") and _text(s.get("title")):
            out.append({"text": f'Section {s["number"]} of this chapter is "{_text(s["title"])[:80]}"; point to it as "Section {s["number"]}"', "kind": "structure"})
    return out


def brief_facts(thesis: dict) -> list[dict]:
    """Sentences of the author's own brief that hold a figure (a sample size, a duration), so a methodology paragraph can state it."""
    out = []
    for k, v in (thesis or {}).items():
        if k in ("title", "objectives", "questions"):
            continue
        for item in (v if isinstance(v, list) else [v]):
            t = _text(item)
            if t and 3 <= len(t.split()) <= 40 and FIGURE.search(t):
                out.append({"text": f'The author\'s brief says: "{t}"', "kind": "brief"})
    return out


def fact_used(fact: dict, low_text: str) -> bool:
    """True when the draft already states this fact: its figures (source, brief) or its 'Research question 2' style pointer."""
    if fact.get("kind") == "structure":
        m = re.search(r"(?:Research question|Objective|Section|Chapter) [\d.]+", fact["text"])
        return bool(m) and bool(re.search(r"(?<![\w.])" + re.escape(m.group(0).lower()) + r"(?![\d])", low_text))
    figs = [t for t in re.findall(r"\d[\d.,]*%?", re.sub(r"\b(?:19|20)\d\d[a-z]?\b", "", fact["text"])) if t.strip(".,")]
    return bool(figs) and any(re.search(r"(?<![\d.])" + re.escape(t.strip(".,")) + r"(?![\d])", low_text) for t in figs)
