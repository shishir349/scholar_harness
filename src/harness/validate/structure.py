"""Deterministic structure checks per section type (keyword/regex level, no LLM)."""
import re

from ..text.citations import extract_citations
from ..text.sentences import words
from ..text.strip import headings_and_text, strip_for_style

GAP = re.compile(r"\b(gap|remains (un|under)\w+|little is known|has not been|few studies|limited research|scarce)\b", re.I)
PREVIEW = re.compile(r"\bchapter (two|three|four|five|\d)\b", re.I)
_STOP = set("that this with from have been were their which about would there these those into than then".split())

METHOD_GROUPS = {
    "design/philosophy": r"\b(design|philosoph\w+|paradigm|positivis\w+|interpretiv\w+|pragmati\w+)\b",
    "sample/participants": r"\b(sampl\w+|participants?|respondents?|population)\b",
    "data collection": r"\b(data collection|collected|survey|interview\w*|questionnaire|observation)\b",
    "analysis": r"\b(analys[ie]s|regression|thematic|coding|statistical|correlation|SEM|ANOVA)\b",
    "ethics": r"\bethic\w*\b",
    "validity/reliability": r"\b(validity|reliab\w+|trustworth\w+|credibility)\b",
}


def _check(cid: str, passed: bool, detail: str = "") -> dict:
    return {"id": cid, "passed": bool(passed), "detail": detail}


def _q_list(thesis: dict, key: str) -> list[str]:
    return list(thesis.get(key) or (thesis.get("toc_data") or {}).get(key) or [])


def _content_words(s: str) -> set[str]:
    return {w.lower() for w in words(s) if len(w) > 3 and w.lower() not in _STOP}


def check_structure(content: str, chapter: dict, thesis: dict, section_type: str, full_chapter: bool = True) -> dict:
    """full_chapter=False when `content` is only one section of a multi-section chapter (e.g. an eval
    fixture): checks that only make sense once the whole chapter is written (a closing "chapter preview",
    "answers every research question", methodology covering every method group) are skipped instead of
    being counted as automatic fails against partial content."""
    marked = headings_and_text(content)
    text = strip_for_style(content)
    n = len(words(text))
    checks: list[dict] = []

    # all sections
    for sec in (chapter.get("toc_sections") or {}).get("sections", []) if isinstance(chapter.get("toc_sections"), dict) \
            else (chapter.get("toc_sections") or []):
        num, title = str(sec.get("number", "")), str(sec.get("title", ""))
        present = (num and num in marked) or (title and title.lower() in marked.lower())
        checks.append(_check(f"section_present:{num or title}", present, title))
    target = chapter.get("target_words")
    if target:
        ok = 0.75 * target <= n <= 1.25 * target
        checks.append(_check("word_count", ok, f"{n} words vs target {target}"))

    lower = text.lower()
    tail = text[int(len(text) * 0.8):]
    cited = {(c["author"], c["year"]) for c in extract_citations(text)}

    if section_type == "introduction":
        checks.append(_check("states_rq_or_objectives", re.search(r"research questions?|objectives?|\baims?\b", lower)))
        if full_chapter:
            checks.append(_check("gap_stated", GAP.search(text)))
            checks.append(_check("chapter_preview", PREVIEW.search(text)))
    elif section_type == "literature_review":
        need = n / 250
        checks.append(_check("distinct_sources", len(cited) >= need, f"{len(cited)} sources, need {need:.0f}"))
        if full_chapter:
            checks.append(_check("gap_near_end", GAP.search(tail)))
    elif section_type == "methodology":
        if full_chapter:
            for name, rx in METHOD_GROUPS.items():
                checks.append(_check(f"covers:{name}", re.search(rx, text, re.I)))
    elif section_type == "findings":
        if re.search(r"quantitative|mixed", str(thesis.get("methodology", "")), re.I):
            digits = sum(1 for t in text.split() if re.search(r"\d", t))
            checks.append(_check("numbers_present", n and digits / n >= 0.01, f"{digits} numeric tokens"))
    elif section_type == "discussion":
        checks.append(_check("refers_to_findings", re.search(r"\b(finding|result)s?\b", lower)))
        if full_chapter:
            checks.append(_check("distinct_citations", len(cited) >= 5, f"{len(cited)}"))
    elif section_type == "conclusion" and full_chapter:
        for k, rx in (("contribution", r"contribut"), ("implication", r"implication"),
                      ("future_research", r"future (research|studies|work)|further research"), ("limitation", r"limitation")):
            checks.append(_check(f"mentions:{k}", re.search(rx, lower)))

    if full_chapter and section_type in ("discussion", "conclusion"):
        have = _content_words(text)
        for i, q in enumerate(_q_list(thesis, "questions"), 1):
            qw = _content_words(q)
            cov = len(qw & have) / len(qw) if qw else 1.0
            checks.append(_check(f"answers_rq{i}", cov >= 0.4, f"{cov:.0%} overlap"))

    passed = sum(1 for c in checks if c["passed"])
    return {"checks": checks, "pass_rate": (passed / len(checks)) if checks else 1.0}
