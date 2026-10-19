"""One-section probe: did each change take effect? Old app vs harness (polish off), one fixture.
    python eval/probe.py [--fixture fx06]        (live keys; about 0.15 USD)"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import evaluate, graph  # noqa: E402
from harness.corpus import openalex, store  # noqa: E402
from harness.sources import agent as source_agent  # noqa: E402
from harness.spend import Spend  # noqa: E402
from harness.text.strip import strip_for_style  # noqa: E402
from harness.validate.features import style_features  # noqa: E402
from harness.validate.judge import judge_h2h  # noqa: E402
from harness.validate.gapfeats import feats as gap_feats  # noqa: E402
from harness.validate.profile import load_profile, profile_metrics  # noqa: E402
from harness.validate.style import feature_stats  # noqa: E402

SPENDS: list = []           # every per-run Spend, in creation order: [baseline, harness]
EX_COUNTS: list = []        # exemplars returned per retrieval call during the harness run


class _Spend(Spend):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        SPENDS.append(self)


evaluate.Spend = _Spend
_get_ex = graph.retrieve.get_exemplars
graph.retrieve.get_exemplars = lambda **k: (lambda r: (EX_COUNTS.append(len(r)), r)[1])(_get_ex(**k))

HARNESS = "harness"          # polish is off by default now, so this is just the shipped "full" arm
evaluate.ARMS.pop("full", None)
evaluate.ARMS[HARNESS] = {"pool": True, "sources": True, "revise": True, "humanize": False}


def feats(text: str) -> dict:
    f = style_features(strip_for_style(text)) or {}
    paras = [p for p in strip_for_style(text).split("\n\n") if p.strip()]
    return {"first_person": f.get("first_person_per_100", 0), "cites": f.get("citation_per_100", 0),
            "sent_len": f.get("sent_len_mean", 0), "para_cv": f.get("para_len_cv", 0),
            "dashes": text.count("—"), "formulaic": sum(1 for p in paras if graph._FORMULAIC.search(p))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="fx06")
    fx = next(f for f in json.loads((ROOT / "eval" / "fixtures.json").read_text(encoding="utf-8"))
              if f["id"] == ap.parse_args().fixture)
    con = store.connect()
    fields = store.get_meta(con, "fields", [])
    lookup = openalex.find_cited_work if openalex.enabled() else None
    sp_research, sp_judge = Spend(budget_usd=1), Spend(budget_usd=1)

    ledger = source_agent.research_chapter(fx["thesis"], fx["chapter"], fx["chapter"]["sections"], spend=sp_research) or {}
    rows = {}
    for arm in ("baseline", HARNESS):
        if arm == HARNESS:
            EX_COUNTS.clear()
        rows[arm] = evaluate.run_fixture(fx, arm, con=con, fields=fields, ledger=ledger, lookup=lookup)
    hs = SPENDS[-1].by_model                       # harness run: calls and cache tokens per model
    son = sum(m["calls"] for k, m in hs.items() if "sonnet" in k)
    cache_read = sum(m["cache_read"] for m in hs.values())
    b, h = rows["baseline"], rows[HARNESS]
    fb, fh = feats(b["text"]), feats(h["text"])
    cheap = judge_h2h(evaluate._clip(h["text"]), evaluate._clip(b["text"]), h["section_type"], sp_judge, repeats=3)
    forced = judge_h2h(evaluate._clip(h["text"]), evaluate._clip(b["text"]), h["section_type"], sp_judge, repeats=3, forced=True)
    strong = judge_h2h(evaluate._clip(h["text"]), evaluate._clip(b["text"]), h["section_type"], sp_judge, repeats=1, tier="sonnet")

    # Real-thesis numbers for THIS section type (not the all-sections average) and the same floors the pipeline uses.
    real = {n: (feature_stats(h["section_type"], n) or {}).get("mean", 0) for n in (*graph.VOICE_FLOOR, "citation_per_100")}
    cite_cap = real["citation_per_100"] * graph.CITE_CEILING
    tells = lambda t: graph.tell_count(t) * 1000 / max(1, len(t.split()))
    fp_floor = real["first_person_per_100"] * graph.VOICE_FLOOR["first_person_per_100"]
    pv_floor = real["para_len_cv"] * graph.VOICE_FLOOR["para_len_cv"]
    prof = load_profile() or {}
    mb, mh = profile_metrics(b["text"], prof), profile_metrics(h["text"], prof)
    sign_real = (prof.get("signposts") or {}).get("real_rate", 0)
    nar_real = (prof.get("citations") or {}).get("narrative_share_real")
    GAP12 = ("connectives", "that", "of", "word_len", "booster", "passive", "digits", "capitalised_mid", "acronyms",
             "short_sent_share", "the_field_lit", "ttr_1000")
    tg = ((prof.get("targets") or {}).get(h["section_type"]) or (prof.get("targets") or {}).get("all") or {})
    gh, gb = gap_feats(h["text"]) or {}, gap_feats(b["text"]) or {}
    inside = lambda g: [k for k in GAP12 if k in tg and k in g and tg[k]["p10"] <= g[k] <= tg[k]["p90"]]
    out_h = [f"{k} {gh[k]:.1f} (real {tg[k]['p10']:.1f}-{tg[k]['p90']:.1f})" for k in GAP12 if k in tg and k in gh and k not in inside(gh)]
    sources = ledger.get("sources", [])
    found = [s for s in sources if (s.get("year") or 9999) < 2015 or (s.get("cited_by") or 0) >= 1000]
    checks = [
        ("Examples: 3 per draft call", EX_COUNTS and max(EX_COUNTS) <= 3 and 3 in EX_COUNTS, f"per retrieval: {EX_COUNTS} (max {graph.N_EXEMPLARS}; thin corpus can return fewer)"),
        ("Polish off, no Sonnet rewrite", son == 3 and h["revisions"] <= 1, f"{son} Sonnet calls (3 = drafts only), {h['revisions']} revision(s)"),
        ("Prompt cache is hit", cache_read > 0, f"{cache_read} cached tokens read (needs a shared block of ~1K+ tokens; 0 = block too small)"),
        ("Cost within 2x of old app", h["usd"] <= 2 * b["usd"], f"${h['usd']:.3f} vs ${b['usd']:.3f} = {h['usd'] / max(b['usd'], 1e-9):.2f}x"),
        ("Planner finds foundational works", len(found) >= 2, f"{len(found)} of {len(sources)} sources are old/heavily cited"),
        # Nothing left to look up is a pass: the planner already supplied the real works. The look-up itself is unit-tested.
        ("Fake citations gone (look-up adopts what is left)", h["unmatched"] <= 1 or len(h["added_refs"]) >= 1,
         f"adopted {len(h['added_refs'])}, fake citations left {h['unmatched']} (old app: {b['unmatched']})"),
        ("Voice: first person", fh["first_person"] >= fp_floor,
         f"{fh['first_person']:.2f} per 100 words (old app {fb['first_person']:.2f}, real {real['first_person_per_100']:.2f}, floor {fp_floor:.2f})"),
        ("Voice: citations near real rate", fh["cites"] <= cite_cap,
         f"{fh['cites']:.2f} per 100 words (old app {fb['cites']:.2f}, real {real['citation_per_100']:.2f}, cap {cite_cap:.2f})"),
        ("Mined AI-only phrases below old app", mh["tells_per_1000"] <= mb["tells_per_1000"],
         f"{mh['tells_per_1000']} per 1000 words vs old app {mb['tells_per_1000']} (real theses ~0.5)"),
        ("Plain signposts (For example, However...)", mh["signpost_rate"] >= 0.5 * sign_real,
         f"{100 * mh['signpost_rate']:.1f}% of sentences vs old app {100 * mb['signpost_rate']:.1f}% (real {100 * sign_real:.1f}%)"),
        ("Narrative citation share near real", nar_real is None or mh["narrative_share"] is None or abs(mh["narrative_share"] - nar_real) <= 0.25,
         f"{mh['narrative_share']} vs old app {mb['narrative_share']} (real {nar_real})"),
        ("Gap features inside the real range (of 12)", len(inside(gh)) >= max(8, len(inside(gb))),
         f"{len(inside(gh))} of 12 (old app {len(inside(gb))}); outside: " + "; ".join(out_h)[:300]),
        ("Fewer AI tells than old app (humanizer list)", tells(h["text"]) <= tells(b["text"]),
         f"{tells(h['text']):.1f} per 1000 words vs old app {tells(b['text']):.1f}"),
        ("Rhythm: sentence length", fh["sent_len"] >= 26, f"{fh['sent_len']:.1f} words (old app {fb['sent_len']:.1f}, real 30)"),
        ("Rhythm: paragraph variety", fh["para_cv"] >= pv_floor,
         f"{fh['para_cv']:.2f} (old app {fb['para_cv']:.2f}, real {real['para_len_cv']:.2f}, floor {pv_floor:.2f})"),
        ("Dash cleanup", fh["dashes"] == 0, f"{fh['dashes']} long dashes (old app {fb['dashes']})"),
        ("Stock-phrase fix", fh["formulaic"] <= 1, f"{fh['formulaic']} paragraphs (old app {fb['formulaic']})"),
        ("Made-up-claim fix", h["unsupported"] == 0, f"{h['unsupported']} unsupported of {h['checked']} checked; "
                                                     f"{h['claims_fixed']} sentences rewritten"),
        ("Sounds human (pick-one judge)", forced["score"] >= 0.5, f"{forced['score']:.2f} over 6 votes (0.5 = even; tie-allowed judge {cheap['score']:.2f})"),
        ("Sounds human (strong judge)", strong["score"] >= 0.5, f"{strong['score']:.2f} (cheap judge {cheap['score']:.2f}; pick-one judge {forced['score']:.2f}; 0.5 = even)"),
    ]
    print(f"\n=== PROBE {fx['id']} {h['section_type']} (polish off) ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:42} {detail}")
    print("\nProblems found on the first check (harness):", h.get("first_problems"))
    print("\nAdopted works (check by eye that each is on-topic):")
    for r in h["added_refs"]:
        print(f"   {r['cite']:28} -> {r['title'][:80]}")
    for c in h["weak_claims"]:
        print(f"   WEAK {c['verdict']}: {c['sentence'][:110]} | {c['reason'][:90]}")
    total = sp_research.total + sp_judge.total + b["usd"] + h["usd"]
    print(f"\ncost: ${total:.3f} (old app {b['usd']:.3f}, harness {h['usd']:.3f}, research {sp_research.total:.3f}, judges {sp_judge.total:.3f})")

    out = ROOT / "eval" / "out"
    evaluate.save_texts(list(rows.values()), out / "thesis_samples" / f"probe_{datetime.now():%Y%m%d_%H%M%S}")
    (out / "probe.json").write_text(json.dumps({"checks": [[n, ok, d] for n, ok, d in checks], "rows": list(rows.values()),
                                                "cheap": cheap, "strong": strong, "forced": forced, "cost": total}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
