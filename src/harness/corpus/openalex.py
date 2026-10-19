"""Thin OpenAlex client. Free API key required since 2026-02-13 (OPENALEX_API_KEY).
List calls cost 1 credit, single-entity calls are free."""
import re
import time
import unicodedata

import httpx

from .. import config
from ..text.apa import build_apa

BASE = "https://api.openalex.org"
MAILTO = "support@acdyon.io"


class OpenAlexError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    return bool(config.get("OPENALEX_API_KEY"))


def get(path: str, params: dict | None = None, *, timeout: float = 15.0, retries: int = 2, client=None) -> dict:
    key = config.get("OPENALEX_API_KEY")
    if not key:
        raise OpenAlexError("OPENALEX_API_KEY is not set (free key: openalex.org/settings/api)")
    q = {**(params or {}), "api_key": key, "mailto": MAILTO}
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(retries + 1):
            try:
                r = client.get(BASE + path, params=q, timeout=timeout)
            except httpx.TransportError as e:
                if attempt >= retries:
                    raise OpenAlexError(f"OpenAlex unreachable: {e}") from e
                time.sleep(1 * 2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt >= retries:
                    raise OpenAlexError(f"OpenAlex {r.status_code}", r.status_code)
                time.sleep(1 * 2 ** attempt)
                continue
            if r.status_code >= 400:
                raise OpenAlexError(f"OpenAlex {r.status_code}: {r.text[:200]}", r.status_code)
            return r.json()
    finally:
        if own:
            client.close()
    raise OpenAlexError("unreachable")  # pragma: no cover


def short_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def abstract_from_inverted_index(idx: dict | None) -> str | None:
    if not idx:
        return None
    slots: dict[int, str] = {}
    for word, positions in idx.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots)) or None


def _apa_author(display_name: str) -> str:
    parts = display_name.replace(",", " ").split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    surname, given = parts[-1], parts[:-1]
    initials = " ".join(f"{g[0].upper()}." for g in given if g and g[0].isalpha())
    return f"{surname}, {initials}" if initials else surname


def _join_authors(names: list[str]) -> str:
    names = [n for n in names if n]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]  # build_apa turns the final "and" into "&"


def work_to_ref(work: dict) -> dict | None:
    title = (work.get("title") or work.get("display_name") or "").strip()
    year = work.get("publication_year")
    names = [a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])][:20]
    authors = _join_authors([_apa_author(n) for n in names])
    if not title or not year or not authors:
        return None
    src = ((work.get("primary_location") or {}).get("source") or {})
    biblio = work.get("biblio") or {}
    fp, lp = biblio.get("first_page"), biblio.get("last_page")
    pages = f"{fp}-{lp}" if fp and lp else (fp or None)
    doi = re.sub(r"^https?://doi\.org/", "", work.get("doi") or "", flags=re.I) or None
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    ref = {
        "openalex_id": short_id(work.get("id")), "doi": doi, "title": title, "year": year, "authors": authors,
        "journal": src.get("display_name"), "volume": biblio.get("volume"), "issue": biblio.get("issue"),
        "pages": pages, "type": "JOURNAL",
        "url": None if doi else ((work.get("primary_location") or {}).get("landing_page_url")),
        "abstract": abstract[:2000] if abstract else None,
        "cited_by": work.get("cited_by_count") or 0,
    }
    ref["apa"] = build_apa(ref)
    return ref


_SEARCH_SELECT = ("id,doi,title,publication_year,authorships,primary_location,biblio,cited_by_count,"
                  "abstract_inverted_index,is_retracted")


def search_works(query: str, from_year: int | None = None, limit: int = 8, *, client=None) -> list[dict]:
    flt = "type:article|review,is_retracted:false,has_abstract:true"
    if from_year:
        flt += f",from_publication_date:{int(from_year)}-01-01"
    data = get("/works", {"search": query, "filter": flt, "per-page": limit, "select": _SEARCH_SELECT}, client=client)
    out = []
    for w in data.get("results", []):
        ref = work_to_ref(w)
        if ref and not w.get("is_retracted"):
            out.append(ref)
    return out


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().strip()


_STOP = frozenset("""about above after also among because been before being between both could does during each from
have having however into more most other over same should since some such than that their them then there these they
this those through under until upon very were what when where which while will with within without would""".split())
MIN_SHARED_WORDS = 2


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in _STOP}


def find_cited_work(surname: str, year: int, hint: str = "", *, limit: int = 25, client=None) -> dict | None:
    """A real work whose first author has this surname and that was published in `year` - i.e. the thing a
    writer's "(Surname, Year)" citation points at. Surname + year alone match namesakes ("Lazarus 2011" is also an
    enzyme paper), so a candidate must share a word with `hint` (the citing context) in its title and at least
    MIN_SHARED_WORDS in title + abstract; the most cited such work wins. Measured on 47 real unmatched citations:
    title-word + 2 overall removes most namesakes. None when nothing fits: the citation cannot be verified."""
    name = re.sub(r"[,:|]", " ", surname or "").strip()
    hint_words = _words(hint)
    if not name or not year or len(hint_words) < MIN_SHARED_WORDS:
        return None
    flt = f"raw_author_name.search:{name},publication_year:{int(year)},is_retracted:false"
    # OpenAlex's text search wants every word to match, so a long context returns nothing: fetch by author + year
    # (most cited first), and when a common surname fills the page, retry narrowed by a few distinctive words.
    narrow = " ".join(sorted(hint_words, key=len, reverse=True)[:3])
    best = None
    for extra in ({}, {"search": narrow}):
        data = get("/works", {"filter": flt, "sort": "cited_by_count:desc", "per-page": limit, "select": _SEARCH_SELECT,
                              **extra}, client=client)
        results = data.get("results", [])
        for w in results:
            ref = work_to_ref(w)
            if not ref or _ascii(ref["authors"].split(",")[0]) != _ascii(name):
                continue
            shared = len(hint_words & _words(f"{ref['title']} {ref.get('abstract') or ''}"))
            if (hint_words & _words(ref["title"]) and shared >= MIN_SHARED_WORDS
                    and (best is None or ref["cited_by"] > best["cited_by"])):
                best = ref                            # on-topic namesakes: the most cited is the one people cite
        if best or len(results) < limit:             # found one, or the page was not full so nothing was missed
            break
    return best
