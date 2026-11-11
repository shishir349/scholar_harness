"""Source ledger used by the research agent."""
import re

from ..text.cleanup import FIGURE as _FIGURE
from ..text.similarity import title_similarity

CITATION_RULES = (
    "CITATION RULES: cite works in APA 7 (Author, Year) form. Prefer the works listed in REFERENCES. You may also "
    "cite a well-known foundational work that you are certain exists, giving its real first author and year exactly; "
    "every citation is looked up afterwards and any that cannot be verified is removed, so never guess an author or "
    "year, and never invent a source, quotation or statistic. When you attribute a finding, number or claim "
    "to a listed source, state only what that source's notes support, with the plain verb the note uses, and put your own "
    "inference in a separate sentence with no citation (a sentence that cites a source and then reads more into it is a "
    "made-up claim). Never copy a sentence from the notes. Your own analysis, judgements, comparisons and "
    "concrete examples need no citation: state them plainly, specifically and with conviction instead of hedging. "
    "If a claim is not supported by a listed source, present it as the researcher's own argument without a citation."
)


class Ledger:
    """Sources seen this run. Ids (S1..Sn) are assigned on first sight; the model may only pick from these."""

    def __init__(self):
        self.items: list[dict] = []
        self._by_oa: dict[str, dict] = {}

    def add(self, ref: dict) -> str:
        key = ref.get("openalex_id") or ref.get("doi") or ref["title"]
        if key in self._by_oa:
            return self._by_oa[key]["ledger_id"]
        ref = {**ref, "ledger_id": f"S{len(self.items) + 1}"}
        self.items.append(ref)
        self._by_oa[key] = ref
        return ref["ledger_id"]

    def get(self, ledger_id: str) -> dict | None:
        return next((r for r in self.items if r["ledger_id"] == ledger_id), None)


def _short_authors(authors: str) -> str:
    """'Smith, J. Q., Lee, K., & Park, C.' -> 'Smith et al.'; a single author -> 'Smith'."""
    surname = authors.split(",")[0].strip()
    multi = bool(re.search(r"\.,\s|\s&\s|\sand\s", authors))
    return f"{surname} et al." if multi else surname


def compact(ref: dict) -> dict:
    return {"id": ref["ledger_id"], "title": ref["title"], "authors": _short_authors(ref["authors"]),
            "year": ref["year"], "venue": ref.get("journal"), "cited_by": ref.get("cited_by", 0),
            "abstract": (ref.get("abstract") or "")[:300]}


def merge_refs(library_refs: list, sources: list[dict]) -> list:
    """Library references first; a source is dropped if the library already has it (DOI, OpenAlex id, or ~same title)."""
    lib = [r for r in (library_refs or []) if isinstance(r, dict)]
    dois = {str(r.get("doi") or "").lower() for r in lib} - {""}
    oa_ids = {r.get("openalex_id") for r in lib} - {None}
    out = list(library_refs or [])
    for s in sources:
        if (s.get("doi") and s["doi"].lower() in dois) or s.get("openalex_id") in oa_ids:
            continue
        if any((title_similarity(s["title"], r.get("title") or "") or 0) >= 0.9 for r in lib):
            continue
        out.append(s)
    return out


MAX_NOTES = 8         # source notes per section: each note invites a citation, and real theses cite far less than the drafts do
NOTE_CHARS = 420     # abstract characters per source in the writer's notes; 250 left too few facts, so drafts stayed vague
_NAMED = re.compile(r"(?<=[a-z,] )[A-Z][A-Za-z0-9-]{2,}")


def _sentences(abstract: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", abstract.strip())


def _note(abstract: str, limit: int = NOTE_CHARS) -> str:
    """The abstract's opening, plus up to two later sentences that carry figures (else names), within `limit` characters.
    Real theses are full of exact numbers and named things; a short note loses both and the draft turns vague."""
    sents = _sentences(abstract)
    rest = sents[1:]
    rich = [x for x in rest if _FIGURE.search(x)] + [x for x in rest if _NAMED.search(x) and not _FIGURE.search(x)]
    rich = rich[:2]
    if not rich:
        return abstract[:limit]
    each = (limit - 60) // len(rich)
    tail = " ".join(x[:each] for x in rich)
    head = sents[0][:max(40, limit - len(tail) - 1)]
    return f"{head} {tail}"[:limit]


def _chosen(section: dict, sources: list[dict], by_section: dict | None) -> list[dict]:
    ids = (by_section or {}).get(str(section.get("number", "")))
    chosen = [s for s in sources if s.get("ledger_id") in (ids or [])]
    return chosen or sorted(sources, key=lambda s: -(s.get("cited_by") or 0))[:6]


def facts_for(section: dict, sources: list[dict], by_section: dict | None, limit: int = 12) -> list[dict]:
    """[{text, cite}]: the abstract sentences that carry a figure (then a named thing) from this section's sources, each with its
    (Author, year). The paragraph plan hands one to a paragraph so the draft states an exact number instead of gesturing."""
    figures, names = [], []
    for s in _chosen(section, sources, by_section):
        cite = f"({_short_authors(s['authors'])}, {s['year']})"
        for x in _sentences(s.get("abstract") or ""):
            if len(x.split()) < 5:
                continue
            (figures if _FIGURE.search(x) else names if _NAMED.search(x) else []).append({"text": x.strip(), "cite": cite, "kind": "source" if _FIGURE.search(x) else "name"})
    return (figures + names)[:limit]


_SAMPLE = re.compile(r"\b(?:n\s*=\s*\d[\d,]*|\d[\d,]*\s+(?:[A-Za-z-]+\s+){0,2}(?:participants|respondents|employees|patients|firms|"
                     r"studies|articles|papers|students|nurses|managers|cases|interviews|countries|hospitals|sites|samples?|observations|"
                     r"organi[sz]ations|companies|schools|households|workers|teachers|farmers|bridges|sensors))\b", re.I)
_METHOD = re.compile(r"\b(?:survey|questionnaire|interviews?|focus groups?|experiment\w*|trials?|regression|case stud(?:y|ies)|"
                     r"ethnograph\w*|content analysis|thematic analysis|systematic review|meta-analysis|simulation|longitudinal|"
                     r"cross-sectional|mixed[- ]methods?|qualitative|quantitative|grounded theory|structural equation\w*|ANOVA|"
                     r"machine learning|deep learning|field study|panel data)\b", re.I)
_RESULT = re.compile(r"\b(?:found|show(?:s|ed|n)?|reveal(?:s|ed)?|indicat\w+|results?|increas\w+|decreas\w+|improv\w+|reduc\w+|"
                     r"associated|correlat\w+|signific\w+|predict\w+|higher|lower|effects?)\b", re.I)
_LIMIT = re.compile(r"\b(?:limitations?|limited|small sample|future research|self-report\w*|generali[sz]\w+|caution)\b", re.I)
_LEAD = re.compile(r"^(?:(?:this|the|our|these) (?:study|paper|article|research|review|results|findings|analysis)\s+(?:also )?"
                   r"(?:found|shows?|showed|reveals?|revealed|indicates?|indicated|demonstrates?|suggests?)\s+(?:that\s+)?)", re.I)


_SAID = re.compile(r"\b(?:found|shows?|showed|reveal(?:s|ed)?|indicat(?:es|ed)|demonstrat(?:es|ed))\s+that\s+(.+)", re.I)


def _clip_words(s: str, n: int) -> str:
    return " ".join(s.strip().rstrip(".").split()[:n])


def _window(sentence: str, rx: re.Pattern, side: int = 5) -> str:
    """The match plus `side` words either side: a limit named in a few words, not a whole abstract sentence."""
    w = sentence.split()
    for i, tok in enumerate(w):
        if rx.search(tok):
            return " ".join(w[max(0, i - side):i + side + 1]).strip(" .,;")
    return ""


def _telegraph(abstract: str) -> str:
    """method / sample / finding / limit as short fragments, so the writer has facts to use and no abstract prose to copy or
    lightly reword. Pure code. The claim check still reads the full abstract."""
    sents = [x for x in _sentences(abstract) if x]
    text = " ".join(sents)
    methods = list(dict.fromkeys(m.group(0).lower() for m in _METHOD.finditer(text)))[:3]
    sample = _SAMPLE.search(text)
    results = [x for x in sents if _RESULT.search(x)]
    best = next((x for x in results if _FIGURE.search(x)), results[0] if results else (sents[0] if sents else ""))
    limit = next((_window(x, _LIMIT) for x in sents if _LIMIT.search(x)), "")
    said = _SAID.search(best)
    start = _RESULT.search(best)
    finding = said.group(1) if said else best[start.start():] if start else best      # from the result on, never the abstract's opening clause
    parts = [("method", ", ".join(methods)), ("sample", sample.group(0) if sample else ""),
             ("finding", _clip_words(_LEAD.sub("", finding), 14)), ("limit", limit)]
    return " | ".join(f"{k}: {v}" for k, v in parts if v)


def source_notes_for(section: dict, sources: list[dict], by_section: dict | None, telegraph: bool = False) -> str:
    """What each mapped source actually reports. Falls back to the top 6 by citations. `telegraph` gives key-value fragments
    (method | sample | finding | limit) in place of the abstract's own sentences."""
    lines = []
    for s in sorted(_chosen(section, sources, by_section), key=lambda x: -(x.get("cited_by") or 0))[:MAX_NOTES]:
        if not s.get("abstract"):
            continue
        first = s["authors"].split(",")[0]
        lines.append(f"- ({first}, {s['year']}): {_telegraph(s['abstract']) if telegraph else _note(s['abstract'])}")
    if not lines:
        return ""
    if telegraph:
        return ("SOURCE NOTES (facts only, as fragments: method | sample | finding | limit. Write every sentence yourself; never "
                "rebuild a note into a sentence about that source, and make claims about a source only within its note):\n"
                + "\n".join(lines))
    return ("SOURCE NOTES (what each source actually reports; make claims about a source only within this, in your own words):\n"
            + "\n".join(lines))
