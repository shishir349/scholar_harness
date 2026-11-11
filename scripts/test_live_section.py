"""Live check of one section with the real Anthropic API: what the fixes changed, with numbers.

    python scripts/test_live_section.py [--fixture fx03] [--title "Research Philosophy"] [--number 3.1] [--budget 2.0]

Costs real money (research ~2 Haiku calls, three Sonnet drafts, a patch, claim checks, one LAMP call): about 0.3-0.6 USD.
Prints, for the text entering the LAMP tail ("before LAMP") and the final text ("after"):
  - the sentence-length array of every paragraph
  - narrative vs parenthetical citation counts
  - sentences that repeat an earlier one
and for LAMP: spans flagged / sent / accepted / rejected (and why)."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import graph  # noqa: E402
from harness.text import cleanup  # noqa: E402
from harness.text.citations import NARRATIVE_RE, PAREN_RE  # noqa: E402
from harness.text.sentences import split_sentences  # noqa: E402
from harness.validate import constraints, lamp_detector  # noqa: E402


def _paras(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip() and not p.lstrip().startswith("#")]


def diagnose(text) -> dict:
    """Numbers about a section. `text` is the section string, or {open, mid, close} to count repeats across chunks."""
    chunks = text if isinstance(text, dict) else {"open": text, "mid": "", "close": ""}
    full = "\n\n".join(c for c in chunks.values() if c.strip())
    paras = _paras(full)
    lengths = [[len(s.split()) for s in split_sentences(p)] for p in paras]
    sents = [s for p in paras for s in split_sentences(p)]
    narrative = sum(1 for s in sents for m in NARRATIVE_RE.finditer(s) if s[max(0, m.start() - 1):m.start()] != "(")
    parenthetical = sum(len(PAREN_RE.findall(s)) for s in sents)
    seen, frames, repeats, cross = [], set(), 0, 0
    for pos in cleanup.POSITIONS:
        before = len(seen)
        for p in _paras(chunks[pos]):
            for s in split_sentences(p):
                c, fr = cleanup._content(s), cleanup._frame(s)
                if len(s.split()) >= 8 and (any(cleanup.same_idea(c, o, cleanup.SECTION_DUP) for o in seen) or (fr and fr in frames)):
                    repeats += 1
                    cross += any(cleanup.same_idea(c, o, cleanup.SECTION_DUP) for o in seen[:before]) or bool(fr and fr in frames)
                seen.append(c)
                if fr:
                    frames.add(fr)
    return {"gates": constraints.section_violations(chunks), "paragraphs": lengths, "words": len(full.split()), "narrative": narrative, "parenthetical": parenthetical,
            "short_paragraphs": sum(1 for n in lengths if len(n) < 3), "longest_sentence": max((max(n) for n in lengths if n), default=0),
            "over_42": sum(1 for n in lengths for x in n if x > 42),
            "no_short_sentence": sum(1 for n in lengths if len(n) >= 3 and min(n) > 11),
            "idioms": len(lamp_detector._IDIOM.findall(full)), "tells": graph.tell_count(full),
            "repeats_in_section": repeats, "cross_chunk_repeats": cross}


def show(title: str, d: dict) -> None:
    print(f"\n── {title} ──  {d['words']} words, {len(d['paragraphs'])} paragraphs")
    for i, n in enumerate(d["paragraphs"], 1):
        flag = "  <-- under 3 sentences" if len(n) < 3 else "  <-- over 42" if max(n) > 42 else "  <-- none of 11 words or fewer" if len(n) >= 3 and min(n) > 11 else ""
        print(f"  P{i:<2} {n}{flag}")
    print(f"  narrative citations {d['narrative']}  parenthetical {d['parenthetical']}  "
          f"| paragraphs <3 sentences {d['short_paragraphs']}  sentences >42 words {d['over_42']}  no-short-sentence paragraphs {d['no_short_sentence']}")
    g = d["gates"]
    print(f"  CONSTRAINTS.md: {'ALL GATES HOLD' if g['ok'] else 'BROKEN'}  micro-paragraphs {len(g['micro_paragraphs'])}  cadence issues {len(g['cadence'])}  "
          f"narrative per call {g['narrative_per_call']} (section {g['narrative_section']})  parenthetical share {g['paren_share']}  "
          f"duplicate pairs {len(g['duplicate_sentence_pairs'])}  author-year summarised twice {len(g['author_year_summarised_twice'])}")
    print(f"  stock idioms {d['idioms']}  tells {d['tells']}  | repeated sentences: {d['repeats_in_section']} in section, {d['cross_chunk_repeats']} across chunks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="fx03")
    ap.add_argument("--title", default="Research Philosophy")
    ap.add_argument("--number", default="3.1")
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()

    from harness import config
    from harness.corpus import openalex, store
    from harness.sources import agent as source_agent
    from harness.spend import Spend
    if not config.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; this script needs the real API.")
        return 2
    fx = next(f for f in json.loads((ROOT / "eval" / "fixtures.json").read_text(encoding="utf-8")) if f["id"] == a.fixture)
    section = {**fx["chapter"]["sections"][0], "title": a.title, "number": a.number, "pages": 5}
    chapter = {**fx["chapter"], "sections": [section], "toc_sections": {"sections": [section]}}
    spend = Spend(budget_usd=a.budget)
    con = store.connect()
    ledger = source_agent.research_chapter(fx["thesis"], chapter, [section], spend=spend) or {}
    print(f"thesis: {fx['thesis']['title']} | section {a.number} {a.title} | sources found: {len(ledger.get('sources', []))}"
          + ("" if openalex.enabled() else "  (no OPENALEX_API_KEY: writing without sources)"))

    captured = {}
    real_tail = graph._lamp_tail

    def tail(chunks, state, deps):
        captured["before"] = dict(chunks)
        out, rep = real_tail(chunks, state, deps)
        captured["after_lamp"] = dict(out)
        return out, rep

    graph._lamp_tail = tail
    req = {"thesis": fx["thesis"], "chapter": chapter, "section": section, "references": fx.get("references", []),
           "flags": {"pool": True, "sources": bool(ledger.get("sources")), "revise": True}, "n_sections": 1, "ledger": ledger}
    res = graph.write_section(req, graph.Deps(spend=spend, con=con, fields=store.get_meta(con, "fields", []),
                                              lookup=openalex.find_cited_work if openalex.enabled() else None))
    graph._lamp_tail = real_tail

    rep = res["reports"].get("lamp") or {}
    print("\nfirst problems found by validate:", {k: v for k, v in (res["reports"].get("first_problems") or {}).items() if v})
    if "before" in captured:
        show("before LAMP (drafts + revise + claim check, short paragraphs merged)", diagnose(captured["before"]))
    show("final text", diagnose(res["text"]))
    print("\n── LAMP ──")
    for k in ("skipped", "error", "flagged", "sent", "sent_words", "accepted", "straddled_seam", "kinds", "rejected_by"):
        if k in rep:
            print(f"  {k}: {rep[k]}")
    for r in rep.get("rejected", []):
        print(f"  rejected span {r['span']} {r['types']}: {r['why']}")
    print(f"\ncost: ${spend.total:.3f} (budget {a.budget})  " + ", ".join(f"{m.split('-')[1] if '-' in m else m}: {v['calls']} calls" for m, v in spend.by_model.items()))
    out = ROOT / "eval" / "out" / "live_section.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res["text"], encoding="utf-8")
    print(f"text saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
