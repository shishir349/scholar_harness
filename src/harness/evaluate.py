"""A/B evaluation: baseline (ported Scholar prompts, no harness features) vs full (pool + sources + revise).
Automated proxies only; a blind human comparison remains the final gate."""
import re
import statistics
from datetime import datetime, timezone

from .corpus import store
from .graph import Deps, write_section
from .prompts.sections import detect_profile
from .sources import agent as source_agent
from .sources.verifier import audit_chapter
from .spend import Spend
from .text.strip import strip_for_style
from .validate.judge import judge_h2h, judge_pair
from .validate.structure import check_structure

ARMS = {
    "baseline": {"pool": False, "sources": False, "revise": False, "humanize": False},
    "full": {"pool": True, "sources": True, "revise": True, "humanize": False},   # polish: opt-in, doubled cost for no style gain
}


def run_fixture(fx: dict, arm: str, *, con, fields, ledger, spend_budget: float = 3.0, baseline=None,
                lookup=None) -> dict:
    spend = Spend(budget_usd=spend_budget)
    chapter, section = fx["chapter"], fx["chapter"]["sections"][0]
    section = {**section, "pages": section.get("pages", 5)}
    uses_sources = bool(ARMS[arm].get("sources"))          # any harness arm, not only the one named "full"
    req = {"thesis": fx["thesis"], "chapter": {**chapter, "toc_sections": {"sections": chapter["sections"]}},
           "section": section, "references": fx.get("references", []), "flags": ARMS[arm], "n_sections": 1,
           "ledger": ledger if uses_sources else {}}
    res = write_section(req, Deps(spend=spend, con=con, fields=fields, baseline=baseline,
                                  lookup=lookup if uses_sources else None))
    text = res["text"]
    ptype = detect_profile(chapter["title"], chapter.get("chapter_number"))["id"]
    refs = (list(fx.get("references", [])) + (ledger.get("sources", []) if uses_sources else [])
            + res.get("added_refs", []))
    cit = audit_chapter(text, [r for r in refs if isinstance(r, dict)], spend, verify=True)
    # A fixture writes one section of a multi-section chapter: grade it as partial (full_chapter=False)
    # so whole-chapter-only checks (chapter preview, "answers every RQ", ...) aren't counted as fails
    # against content that was never meant to contain them. Still check that section's own heading landed.
    structure = check_structure(text, {"toc_sections": [section]}, fx["thesis"], ptype,
                                full_chapter=len(chapter["sections"]) == 1)
    style = res["reports"].get("style", {})
    ref_chunk = store.random_holdout_chunk(con, ptype, "open") if con is not None else None
    judge = (judge_pair(_clip(text), ref_chunk["text"], ptype, spend)["result"]
             if ref_chunk else None)
    weak_claims = [{k: c[k] for k in ("sentence", "cite", "verdict", "reason")}
                   for c in cit["claim_checks"] if c["verdict"] in ("UNSUPPORTED", "PARTIAL")]
    return {
        "fixture": fx["id"], "arm": arm, "section_type": ptype, "words": len(text.split()),
        "style_score": style.get("style_score") if style.get("available") else None,
        "style_passed": style.get("passed"), "structure_pass_rate": structure["pass_rate"], "judge": judge,
        "unmatched": cit["stats"]["unmatched_count"], "checked": cit["stats"]["checked"], "unsupported": cit["stats"]["unsupported"],
        "overlap_flagged": bool(res["reports"]["overlap"]["flagged"]), "revisions": res["revisions"],
        "usd": spend.total, "text": text, "weak_claims": weak_claims,
        "added_refs": [{"cite": f"{r['authors'].split(',')[0]} ({r['year']})", "title": r["title"]}
                       for r in res.get("added_refs", [])],
        "claims_fixed": res["reports"].get("claims", {}).get("fixed_unsupported", 0),
        "first_problems": res["reports"].get("first_problems"),
    }


def _clip(text: str) -> str:
    """First ~450 words for the judge, cut back to the last full sentence so neither text trails off mid-sentence."""
    clipped = " ".join(strip_for_style(text).split()[:450])
    ends = [m.end() for m in re.finditer(r"[.!?][\"')\]]?(?=\s|$)", clipped)]
    return clipped[:ends[-1]] if ends and ends[-1] > 0.6 * len(clipped) else clipped


def run_ab(fixtures: list[dict], *, con, fields, search_fn=None, baseline=None, lookup=None) -> list[dict]:
    rows = []
    for fx in fixtures:
        spend = Spend(budget_usd=3.0)
        chapter = fx["chapter"]
        ledger = source_agent.research_chapter(fx["thesis"], chapter, chapter["sections"], spend=spend, search_fn=search_fn) or {}
        base_row = None
        for arm in ARMS:
            row = run_fixture(fx, arm, con=con, fields=fields, ledger=ledger, baseline=baseline, lookup=lookup)
            if arm == "baseline":
                base_row = row
            else:
                # Head-to-head on the same fixture: 'win' = full read as the human one in both A/B orders.
                a, b = _clip(row["text"]), _clip(base_row["text"])
                row["h2h"] = judge_h2h(a, b, row["section_type"], spend)["result"]
                row["h2h_forced"] = judge_h2h(a, b, row["section_type"], spend, forced=True)["score"]   # pick-one mode, informational
                row["usd"] += spend.total          # research + head-to-head cost belong to the full arm
            rows.append(row)
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _paired_diffs(rows: list[dict], metric: str) -> dict:
    """{fixture_id: full[metric] - baseline[metric]}, fixtures with both arms and a real value only.
    A pooled mean lets one outlier fixture swing the whole verdict; comparing each fixture to itself
    doesn't."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["fixture"], {})[r["arm"]] = r
    return {fid: d["full"][metric] - d["baseline"][metric] for fid, d in by.items()
            if "full" in d and "baseline" in d and d["full"].get(metric) is not None and d["baseline"].get(metric) is not None}


def _not_worse_rate(diffs: dict, slack: float) -> float | None:
    return (sum(1 for d in diffs.values() if d >= slack) / len(diffs)) if diffs else None


def summarise(rows: list[dict]) -> dict:
    style_nw = _not_worse_rate(_paired_diffs(rows, "style_score"), slack=-5)
    struct_nw = _not_worse_rate(_paired_diffs(rows, "structure_pass_rate"), slack=-0.15)
    out = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        judged = [x for x in r if x["judge"]]
        h2h = [x for x in r if x.get("h2h")]
        checked = sum(x["checked"] for x in r)
        out[arm] = {
            "n": len(r), "style_mean": _mean([x["style_score"] for x in r]),
            "structure_mean": _mean([x["structure_pass_rate"] for x in r]),
            "win_rate": (sum(1 for x in judged if x["judge"] == "win") / len(judged)) if judged else None,
            "tie_rate": (sum(1 for x in judged if x["judge"] == "tie") / len(judged)) if judged else None,
            "lose_rate": (sum(1 for x in judged if x["judge"] == "lose") / len(judged)) if judged else None,
            # full vs baseline head-to-head, ties count half; only the full arm has one
            "h2h_score": ((sum(1 for x in h2h if x["h2h"] == "win") + 0.5 * sum(1 for x in h2h if x["h2h"] == "tie")) / len(h2h)) if h2h else None,
            "h2h_forced": _mean([x.get("h2h_forced") for x in r]),
            "unmatched_total": sum(x["unmatched"] for x in r),
            "unsupported_rate": (sum(x["unsupported"] for x in r) / checked) if checked else 0.0,
            "overlap_flags": sum(1 for x in r if x["overlap_flagged"]), "revisions": sum(x["revisions"] for x in r),
            "usd_total": sum(x["usd"] for x in r),
            # per-fixture paired comparison (full vs baseline on the same fixture); only meaningful for "full"
            "style_not_worse_rate": style_nw if arm == "full" else None,
            "structure_not_worse_rate": struct_nw if arm == "full" else None,
        }
    return out


def evaluate(s: dict) -> list[dict]:
    """The G3 pass criteria. Each: {name, passed, detail}. A missing metric fails (unknown is not a pass)."""
    b, f = s["baseline"], s["full"]
    num = lambda v: "n/a" if v is None else f"{v:.2f}"
    h2h = f["h2h_score"]
    # Why these bars: style is already near its ceiling for the baseline, so the criterion is "no
    # regression" (a +5 bar is unreachable at ~85/100); the old win-rate-vs-a-human-excerpt bar
    # saturated at 1.0 for both arms, so quality is judged head-to-head on the same fixtures instead;
    # "zero unmatched citations" became "at least halve them" because zero across all fixtures is luck.
    # style/structure gate on the per-fixture paired rate, not the pooled mean, so one noisy fixture
    # can't flip the verdict (see _paired_diffs/_not_worse_rate).
    return [
        {"name": "style_score: full not much worse than baseline, most fixtures (>=50%, slack -5)",
         "passed": f["style_not_worse_rate"] is not None and f["style_not_worse_rate"] >= 0.5,
         "detail": f"{num(f['style_not_worse_rate'])} of paired fixtures (means: full {num(f['style_mean'])} vs baseline {num(b['style_mean'])})"},
        {"name": "head-to-head: full reads more human than baseline (>= 0.60, ties = half)",
         "passed": h2h is not None and h2h >= 0.60, "detail": num(h2h)},
        {"name": "unmatched citations at least halved vs baseline",
         "passed": f["unmatched_total"] <= b["unmatched_total"] // 2,
         "detail": f"full {f['unmatched_total']} vs baseline {b['unmatched_total']}"},
        {"name": "unsupported rate <= 5%", "passed": f["unsupported_rate"] <= 0.05, "detail": f"{f['unsupported_rate']:.1%}"},
        {"name": "no overlap flags (full)", "passed": f["overlap_flags"] == 0, "detail": f"{f['overlap_flags']}"},
        {"name": "structure: full not much worse than baseline, most fixtures (>=50%, slack -0.15)",
         "passed": f["structure_not_worse_rate"] is not None and f["structure_not_worse_rate"] >= 0.5,
         "detail": f"{num(f['structure_not_worse_rate'])} of paired fixtures (means: full {num(f['structure_mean'])} vs baseline {num(b['structure_mean'])})"},
    ]


def report_md(rows: list[dict], s: dict, criteria: list[dict]) -> str:
    L = [f"# A/B evaluation ({datetime.now(timezone.utc).date()})", "",
         "| metric | baseline | full |", "|---|---|---|"]
    for k in ("style_mean", "style_not_worse_rate", "structure_mean", "structure_not_worse_rate", "h2h_score", "h2h_forced",
              "win_rate", "tie_rate", "lose_rate", "unmatched_total", "unsupported_rate", "overlap_flags",
              "revisions", "usd_total"):
        fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        L.append(f"| {k} | {fmt(s['baseline'][k])} | {fmt(s['full'][k])} |")
    L += ["", "## G3 criteria", ""]
    L += [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['name']} ({c['detail']})" for c in criteria]
    n = s["full"]["n"]
    L += ["", f"**Overall: {'PASS' if all(c['passed'] for c in criteria) else 'FAIL'}**", "",
          "_Automated proxies only. A blind human comparison remains the final quality gate. The judge is the same vendor "
          "family as the generator and measures human-likeness, not correctness. win/tie/lose_rate compare each arm to a real "
          "thesis excerpt and are informational (they saturate); h2h_score is the head-to-head that gates._"]
    if n < 10:
        L.append(f"\n**Only {n} fixture(s): differences at this size are noise, not a verdict.**")
    return "\n".join(L)


def save_texts(rows: list[dict], folder) -> int:
    """One text file per (fixture, arm) so generated sections survive the next run overwriting results.json."""
    from pathlib import Path
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        head = (f"Topic: {r['fixture']} | Section: {r['section_type']} | Arm: {r['arm']} | Words: {r['words']} | "
                f"style: {r.get('style_score')} | h2h: {r.get('h2h')} | unmatched cites: {r['unmatched']} | "
                f"unsupported: {r['unsupported']} | cost: ${r['usd']:.3f}")
        (out / f"{r['fixture']}_{r['section_type']}__{r['arm']}.txt").write_text(head + "\n\n" + r["text"], encoding="utf-8")
    return len(rows)
