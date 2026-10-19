"""APA in-text citation extraction and audit. Port of acdyon-scholar citationExtractor.js."""
import re
from datetime import date

_N = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-']+"
PAREN_RE = re.compile(
    rf"\(\s*({_N}(?:\s*(?:,\s*{_N}|\s*&\s*{_N}|,?\s*et al\.?))*)\s*,\s*(\d{{4}}[a-z]?)\s*\)"
)
NARRATIVE_RE = re.compile(
    rf"\b({_N}(?:\s+(?:and|&)\s+{_N})?(?:\s+et al\.?)?)\s+\((\d{{4}}[a-z]?)\)"
)
_SPLIT_AUTHOR = re.compile(r"\s*(?:&|,|\band\b|et al)", re.I)   # "Macey and Schneider" -> "Macey"


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def extract_citations(text: str) -> list[dict]:
    """Unique {raw, author, year} citations, in first-seen order."""
    if not text:
        return []
    plain = strip_html(text)
    found: dict[str, dict] = {}

    def add(raw, author, year_str):
        year = int(re.match(r"\d{4}", year_str).group(0))
        key = f"{author.strip()}|{year}"
        if key not in found:
            found[key] = {"raw": raw.strip(), "author": author.strip(), "year": year}

    for m in PAREN_RE.finditer(plain):
        add(m.group(0), m.group(1), m.group(2))
    for m in NARRATIVE_RE.finditer(plain):
        if plain[max(0, m.start() - 1):m.start()] == "(":
            continue
        add(m.group(0), m.group(1), m.group(2))
    return list(found.values())


def first_author(author: str) -> str:
    """Surname of the first author; "Kahn's (1990)" is Kahn."""
    return re.sub(r"['’]s$", "", _SPLIT_AUTHOR.split(author)[0].strip())


def _ref_year(ref):
    y = ref.get("year")
    try:
        return int(y)
    except (TypeError, ValueError):
        return None


def match_ref(cit: dict, refs: list[dict]):
    """The stored ref a citation points at, by year + first-author surname."""
    fa = first_author(cit["author"]).lower()
    for ref in refs:
        if _ref_year(ref) != cit["year"]:
            continue
        if fa in (ref.get("authors") or "").lower():
            return ref
    return None


def audit_citations(citations: list[dict], refs: list[dict]) -> dict:
    current = date.today().year
    matched, unmatched, suspicious = [], [], []
    for cit in citations:
        if cit["year"] > current or cit["year"] < 1900:
            suspicious.append({**cit, "reason": "Future publication year" if cit["year"] > current else "Year before 1900"})
            continue
        ref = match_ref(cit, refs)
        if ref:
            matched.append({**cit, "reference": ref})
        else:
            unmatched.append(cit)

    def unused(ref):
        fa = re.split(r"[,;&]", ref.get("authors") or "")[0].strip().lower()
        ry = _ref_year(ref)
        return not any(first_author(c["author"]).lower() == fa and c["year"] == ry for c in citations)

    return {"matched": matched, "unmatched": unmatched, "suspicious": suspicious,
            "unused_refs": [r for r in refs if unused(r)]}
