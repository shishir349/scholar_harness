"""Research agent: plan searches, run them, then pick sources - two cheap-model calls, no back-and-forth loop."""
import json
import logging

from .. import llm
from ..corpus import openalex
from ..text.jsonparse import parse_json
from .tools import Ledger, compact

log = logging.getLogger("harness.agent")

PLAN_SYSTEM = (
    "You are a research librarian planning searches for one chapter of a doctoral thesis. List up to {n} distinct "
    "search queries (5-8 is ideal) that together would find real, peer-reviewed sources covering the chapter's "
    "sections. At least three of the queries must be the EXACT titles of the most-cited foundational or seminal works "
    "in this area that you are confident exist (a real thesis on this topic would cite them); if you are not sure of "
    "a title, use a focused topical query instead and never invent a title. The remaining queries are focused topical "
    "queries that vary the angle (theory, method, geography, critique) and prefer terms found in real paper titles. "
    'Reply with ONLY this JSON: {"queries": [{"q": "...", "from_year": 2015}, ...]} - omit from_year (or use null) '
    "for foundational titles; use it only to focus a topical query on recent work."
)
SELECT_SYSTEM = (
    "You are a research librarian selecting sources for one chapter of a doctoral thesis from the CANDIDATES list "
    "below (found by earlier searches). Keep the well-known foundational works among them when they fit the chapter. "
    "Never invent a source: only pick ids that appear in CANDIDATES. Reply with "
    'ONLY this JSON: {"selected": [{"id": "S3", "sections": ["2.1", "2.2"]}]} listing at most {max_sources} source '
    "ids, each with the section numbers it supports."
)


def _user_message(thesis: dict, chapter: dict, sections: list[dict]) -> str:
    q = lambda k: thesis.get(k) or (thesis.get("toc_data") or {}).get(k) or []
    lines = [
        f'Thesis: "{thesis.get("title")}"',
        f"Domain: {thesis.get('domain')} | Methodology: {thesis.get('methodology')} | "
        f"Geographic focus: {thesis.get('geographic_focus') or 'not specified'}",
        "Objectives: " + "; ".join(q("objectives")) if q("objectives") else "",
        "Research questions: " + "; ".join(q("questions")) if q("questions") else "",
        f"Chapter {chapter.get('chapter_number')}: {chapter.get('title')}",
        "Sections:",
        *[f"  {s.get('number')} {s.get('title')}" for s in sections],
    ]
    return "\n".join(x for x in lines if x)


def _plan_queries(thesis: dict, chapter: dict, sections: list[dict], n: int, spend) -> list[tuple[str, int | None]]:
    text = llm.ask("haiku", PLAN_SYSTEM.replace("{n}", str(n)), _user_message(thesis, chapter, sections),
                    max_tokens=500, spend=spend)
    data = parse_json(text)
    out = []
    if isinstance(data, dict):
        for item in (data.get("queries") or [])[:n]:
            if isinstance(item, dict) and item.get("q"):
                out.append((item["q"], item.get("from_year")))
    return out


def _select(text: str, ledger: Ledger, max_sources: int):
    data = parse_json(text)
    chosen, by_section = [], {}
    if isinstance(data, dict):
        for item in data.get("selected") or []:
            ref = ledger.get(item.get("id")) if isinstance(item, dict) else None
            if ref and ref not in chosen:
                chosen.append(ref)
                for sec in item.get("sections") or []:
                    by_section.setdefault(str(sec), []).append(ref["ledger_id"])
    if not chosen:  # parse failure / empty selection: best-cited sources seen
        chosen = sorted(ledger.items, key=lambda r: -(r.get("cited_by") or 0))
        by_section = {}
    return chosen[:max_sources], by_section


def research_chapter(thesis: dict, chapter: dict, sections: list[dict], *, spend=None, max_rounds: int = 8,
                     max_sources: int = 12, search_fn=None) -> dict | None:
    """{sources, by_section, rounds, searches} | {skipped: 'no_key'} | None on any failure (caller proceeds without).
    Two AI calls total (cheap model): plan the searches, run them all, then pick sources from what came back -
    no back-and-forth agent loop, so cost does not grow with the number of searches."""
    if search_fn is None and not openalex.enabled():
        return {"sources": [], "by_section": {}, "skipped": "no_key"}
    search = search_fn or (lambda q, y, n: openalex.search_works(q, from_year=y, limit=n))
    try:
        ledger = Ledger()
        queries = _plan_queries(thesis, chapter, sections, max_rounds, spend)
        searches = 0
        for q, from_year in queries:
            try:
                for r in search(q, from_year, 12):
                    ledger.add(r)
                searches += 1
            except Exception as e:  # a single bad query does not sink the others
                log.warning("search %r failed, continuing: %s", q, e)
        candidates = sorted(ledger.items, key=lambda r: -(r.get("cited_by") or 0))[:60]
        select_msg = (_user_message(thesis, chapter, sections) + "\n\nCANDIDATES:\n"
                      + json.dumps([compact(r) for r in candidates]))
        reply = llm.ask("haiku", SELECT_SYSTEM.replace("{max_sources}", str(max_sources)), select_msg,
                        max_tokens=800, spend=spend)
        sources, by_section = _select(reply, ledger, max_sources)
        return {"sources": sources, "by_section": by_section, "rounds": 2, "searches": searches}
    except Exception as e:  # noqa: BLE001 - fail open by design; BudgetExceeded is re-raised below
        from ..spend import BudgetExceeded
        if isinstance(e, BudgetExceeded):
            raise
        log.warning("research agent failed, continuing without sources: %s", e)
        return None
