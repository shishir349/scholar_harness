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


def source_notes_for(section: dict, sources: list[dict], by_section: dict | None) -> str:
    """What each mapped source actually reports (abstract start). Falls back to the top 6 by citations."""
    lines = []
    for s in sorted(_chosen(section, sources, by_section), key=lambda x: -(x.get("cited_by") or 0))[:MAX_NOTES]:
        if not s.get("abstract"):
            continue
        first = s["authors"].split(",")[0]
        lines.append(f"- ({first}, {s['year']}): {_note(s['abstract'])}")
    if not lines:
        return ""
    return ("SOURCE NOTES (what each source actually reports; make claims about a source only within this, in your own words):\n"
            + "\n".join(lines))
