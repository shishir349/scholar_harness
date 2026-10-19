"""Where do saved AI drafts differ from real theses, on ~45 measurable features? (no API calls)
    python eval/gap_report.py      -> eval/out/gap_report.md
For each feature: how well it alone separates real text from AI text (AUC, 0.5 = no difference, 1.0 = perfect), how many real
standard deviations the AI mean sits from the real mean, and which lever in the harness already targets it."""
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness.corpus import store  # noqa: E402
from harness.validate.gapfeats import feats  # noqa: E402

WINDOW, PER_THESIS = 3, 8
TYPES = ("introduction", "literature_review", "methodology", "discussion")

# feature -> the lever that already targets it ("" = nothing does yet)
LEVER = {
    "sent_len_mean": "style target", "sent_len_cv": "style target", "short_sent_share": "voice rules + patch",
    "para_per_1000": "paragraph plan", "para_len_cv": "paragraph plan + merge", "first_person": "patch for missing first person",
    "cite_narrative_share": "plan + patch", "cite_per_100": "citation patch", "signpost_open": "plan + patch",
    "dash": "dash cleanup", "hedge": "style target", "digits": "source notes keep figures", "capitalised_mid": "source notes keep names",
    "acronyms": "define-once rule", "connectives": "mined target + patch", "that": "mined target + patch", "of": "mined target",
    "word_len": "mined plain-word list", "long_word_share": "mined plain-word list", "booster": "mined avoid list",
    "passive": "mined target (per section)", "the_field_lit": "meta-phrase patch", "ttr_1000": "repeat-the-term rule",
}


def auc(real: list, ai: list) -> float:
    """P(an AI value is larger than a real value), ties half. 0.5 = no difference."""
    wins = sum((a > r) + 0.5 * (a == r) for a in ai for r in real)
    return wins / (len(ai) * len(real))


def main() -> None:
    con = store.connect()
    rows = con.execute(
        f"select c.thesis_id, c.section_type, c.chunk_idx, c.text from chunks c where c.section_type in ({','.join('?' * len(TYPES))}) "
        "order by c.thesis_id, c.section_type, c.chunk_idx", TYPES).fetchall()
    groups: dict = {}
    for tid, st, idx, text in rows:
        groups.setdefault((tid, st), []).append(text)
    rng = random.Random(7)
    per_thesis: dict = {}
    for (tid, st), texts in groups.items():
        for i in range(0, len(texts) - WINDOW + 1, WINDOW):
            per_thesis.setdefault(tid, []).append("\n\n".join(texts[i:i + WINDOW]))
    real_f = []
    for tid, wins in per_thesis.items():
        rng.shuffle(wins)
        real_f += [f for f in (feats(w) for w in wins[:PER_THESIS]) if f]
    ai = {"baseline": [], "harness": []}
    for p in sorted((ROOT / "eval" / "out" / "thesis_samples").rglob("*.txt")):
        arm = "baseline" if "baseline" in p.name or p.parent.name == "baseline" else "harness"
        body = p.read_text(encoding="utf-8", errors="replace").partition("\n\n")[2]
        f = feats(body)
        if f:
            ai[arm].append(f)
    names = [k for k in real_f[0] if k != "ai_phrases"]
    stats_r = {k: (statistics.fmean(f[k] for f in real_f), statistics.pstdev([f[k] for f in real_f]) or 1e-9) for k in names}
    md = [f"# Gap report: real theses vs saved AI drafts", "",
          f"Real: {len(real_f)} samples (~{WINDOW * 440} words each, <= {PER_THESIS} per thesis, from {len(per_thesis)} theses, section types "
          f"{', '.join(TYPES)}). AI: {len(ai['baseline'])} old-app drafts, {len(ai['harness'])} harness drafts (older harness versions included).",
          "", "AUC: 0.5 = this feature does not tell AI from real, 1.0 / 0.0 = it tells them apart perfectly. z = AI mean minus real mean, in real standard deviations.", ""]
    ranked = []
    for k in names:
        rv = [f[k] for f in real_f]
        row = {"k": k}
        for arm in ai:
            av = [f[k] for f in ai[arm]]
            row[arm + "_auc"] = auc(rv, av)
            row[arm + "_z"] = (statistics.fmean(av) - stats_r[k][0]) / stats_r[k][1]
        row["real"] = stats_r[k][0]; row["real_sd"] = stats_r[k][1]
        row["harness_mean"] = statistics.fmean(f[k] for f in ai["harness"]); row["baseline_mean"] = statistics.fmean(f[k] for f in ai["baseline"])
        ranked.append(row)
    ranked.sort(key=lambda r: -abs(r["harness_auc"] - 0.5))
    md += ["## Features ranked by how well they expose the harness drafts", "",
           "| feature | real mean (sd) | old app | harness | harness z | harness AUC | lever now |", "|---|---|---|---|---|---|---|"]
    for r in ranked:
        md.append(f"| {r['k']} | {r['real']:.2f} ({r['real_sd']:.2f}) | {r['baseline_mean']:.2f} | {r['harness_mean']:.2f} | {r['harness_z']:+.1f} | "
                  f"{r['harness_auc']:.2f} | {LEVER.get(r['k'], '')} |")
    # overall distance: mean |z| over all features, real samples scored against the rest of real (leave-one-in is fine at this size)
    dist = lambda f: statistics.fmean(min(4.0, abs((f[k] - stats_r[k][0]) / stats_r[k][1])) for k in names)
    rd = sorted(dist(f) for f in real_f)
    md += ["", "## Overall distance from real prose (mean |z| over all features, capped at 4)", "",
           f"- real samples: median {rd[len(rd) // 2]:.2f}, 90th percentile {rd[int(len(rd) * 0.9)]:.2f}",
           f"- old app drafts: median {statistics.median(dist(f) for f in ai['baseline']):.2f}",
           f"- harness drafts: median {statistics.median(dist(f) for f in ai['harness']):.2f}"]
    strong = [r for r in ranked if abs(r["harness_auc"] - 0.5) >= 0.4]
    md += ["", f"{len(strong)} of {len(names)} features separate harness drafts from real text almost perfectly (AUC >= 0.9 or <= 0.1)."]
    out = ROOT / "eval" / "out" / "gap_report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
