"""APA 7 reference strings. Port of acdyon-scholar backend/src/utils/apa.js."""
import re


def _clean(v) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v))


def _normalise_authors(authors) -> str:
    a = _clean(authors)
    if not a:
        return ""
    # Only the FINAL " and " becomes "&"; an existing comma is absorbed.
    a = re.sub(r",?\s+and\s+(?![\s\S]*\s+and\s+)", ", & ", a, flags=re.I)
    return re.sub(r"\s{2,}", " ", a)


def _terminate(s) -> str:
    t = _clean(s)
    if not t:
        return ""
    return t if re.search(r"[.?!]$", t) else f"{t}."


def _source_link(ref) -> str:
    d = _clean(ref.get("doi"))
    if d:
        return " https://doi.org/" + re.sub(r"^https?://doi\.org/", "", d, flags=re.I)
    u = _clean(ref.get("url"))
    return f" {u}" if u else ""


def build_apa(ref: dict) -> str:
    authors = _normalise_authors(ref.get("authors"))
    year = _clean(ref.get("year")) or "n.d."
    title = _clean(ref.get("title"))
    source = _clean(ref.get("journal"))
    link = _source_link(ref)
    typ = _clean(ref.get("type")).upper() or "JOURNAL"

    head = " ".join(x for x in [authors and _terminate(authors), f"({year})."] if x)

    if typ in ("BOOK", "REPORT", "WEBSITE"):
        return f"{head} {_terminate(title)}{' ' + _terminate(source) if source else ''}{link}".strip()

    vol, iss, pgs = _clean(ref.get("volume")), _clean(ref.get("issue")), _clean(ref.get("pages"))
    locator = source
    if locator and vol:
        locator += f", {vol}"
    if locator and iss:
        locator += f"({iss})"
    if pgs:
        locator = f"{locator}, {pgs}" if locator else pgs

    return f"{head} {_terminate(title)}{' ' + _terminate(locator) if locator else ''}{link}".strip()
