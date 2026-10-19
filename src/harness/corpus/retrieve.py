"""Exemplar retrieval: (section type, position, field) filter + FTS5 rank, one chunk per thesis.
Never raises: any failure yields no exemplars and the caller writes exactly as before."""
import logging
import re
import sqlite3

from . import store
from ..prompts.sections import profiles
from ..text.refs import looks_like_references

log = logging.getLogger("harness.retrieve")

_STOP = set("""about above after again against also among because been before being below between both could does doing
down during each from further have having here into itself just more most other over same should some such than that
their them then there these they this those through under until very were what when where which while will with would
your study analysis research thesis effect effects impact based using towards toward""".split())

_cache: dict = {}
_CACHE_MAX = 200


def topic_query(text: str | None) -> str:
    """FTS5 OR-query from content words. Alphanumerics only, so hostile input cannot inject syntax."""
    seen: list[str] = []
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.append(w)
        if len(seen) == 8:
            break
    return " OR ".join(seen)


def map_to_field(title: str, domain: str, fields: list[dict]) -> str | None:
    """Best-overlap field name, or None when the evidence is thin (then all fields are pooled)."""
    toks = {w for w in re.findall(r"[a-z0-9]+", f"{title} {domain}".lower()) if len(w) > 3 and w not in _STOP}
    best, best_n = None, 0
    for f in fields:
        vocab = {w for w in re.findall(r"[a-z0-9]+", " ".join([f["name"], *f.get("keywords", [])]).lower())
                 if len(w) > 3 and w not in _STOP}
        n = len(toks & vocab)
        if n > best_n:
            best, best_n = f["name"], n
    return best if best_n >= 2 else None


def _rows(con, q: str, section_type: str, position: str, field: str | None, limit: int):
    where = "c.section_type=? AND c.position=? AND t.holdout=0"
    args: list = [section_type, position]
    if field:
        where += " AND c.field_name=?"
        args.append(field)
    rows = []
    if q:
        rows = con.execute(
            f"""SELECT c.id, c.text, c.heading, c.field_name, c.thesis_id FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid JOIN theses t ON t.id = c.thesis_id
                WHERE chunks_fts MATCH ? AND {where} ORDER BY bm25(chunks_fts) LIMIT 30""", [q, *args]).fetchall()
    return rows


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3 and w not in _STOP}


def _rerank(rows: list, context_tokens: set[str]) -> list:
    """FTS5's bm25 only sees the OR-query terms; this re-scores the same candidates by real overlap
    with the full topic context (title+chapter+section+heading), so among several keyword-matching
    chunks the one that's actually closest in subject wins, not just the one bm25 ranked first."""
    if not context_tokens or not rows:
        return rows
    scored = [(len(context_tokens & _tokens(f"{r['heading']} {r['text'][:600]}")), i, r) for i, r in enumerate(rows)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, __, r in scored]


def _topup(con, section_type, position, field, exclude_ids, need):
    where = "c.section_type=? AND c.position=? AND t.holdout=0"
    args: list = [section_type, position]
    if field:
        where += " AND c.field_name=?"
        args.append(field)
    return con.execute(
        f"""SELECT c.id, c.text, c.heading, c.field_name, c.thesis_id FROM chunks c JOIN theses t ON t.id=c.thesis_id
            WHERE {where} ORDER BY t.score DESC, c.id LIMIT 60""", args).fetchall()


def get_exemplars(*, con, fields: list[dict], title: str, domain: str, chapter_title: str, section_title: str,
                  section_type: str, position: str, limit: int = 2) -> list[dict]:
    key = (id(con), title, domain, chapter_title, section_title, section_type, position, limit)
    if key in _cache:
        return _cache[key]
    try:
        field = map_to_field(title, domain, fields)
        q = topic_query(f"{title} {chapter_title} {section_title}")
        context = _tokens(f"{title} {chapter_title} {section_title}")
        picked: list[dict] = []
        used_theses: set[int] = set()

        def take(rows):
            for r in rows:
                if len(picked) >= limit:
                    return
                if r["thesis_id"] in used_theses or looks_like_references(r["text"]):   # also guards older databases
                    continue
                used_theses.add(r["thesis_id"])
                picked.append({"id": r["id"], "text": r["text"], "field": r["field_name"], "heading": r["heading"]})

        take(_rerank(_rows(con, q, section_type, position, field, limit), context))
        if len(picked) < limit:
            take(_topup(con, section_type, position, field, used_theses, limit))
        if len(picked) < limit and field:  # field too thin: pool every field
            take(_topup(con, section_type, position, None, used_theses, limit))
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = picked
        return picked
    except (sqlite3.Error, KeyError, TypeError) as e:
        log.warning("exemplar retrieval failed, continuing without: %s", e)
        return []


def build_exemplar_block(exemplars: list[dict], section_type: str, position: str) -> str:
    if not exemplars:
        return ""
    label = profiles().get(section_type, {}).get("label", section_type)
    head = (f"STYLE EXEMPLARS: real doctoral prose from published theses, shown ONLY so you can study how researchers "
            f"write the {position} of a {label} passage: how claims are hedged, how evidence is introduced, how "
            f"paragraphs open and close, how sentence length varies, where the writer commits to a position. Do NOT "
            f"copy or paraphrase their wording. Do NOT reuse their content, facts, data or citations. Your content "
            f"comes only from this thesis and the REFERENCES list.")
    parts = [head]
    for i, e in enumerate(exemplars, 1):
        parts.append(f"[EXEMPLAR {i}: {e['field']}, {position} of a {label} section]\n\"\"\"\n{e['text']}\n\"\"\"")
    return "\n".join(parts)


def clear_cache() -> None:
    _cache.clear()


__all__ = ["topic_query", "map_to_field", "get_exemplars", "build_exemplar_block", "clear_cache", "store"]
