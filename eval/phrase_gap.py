"""Mine the real-thesis corpus against every saved AI draft (no API calls).
    python eval/phrase_gap.py     -> data/style_profile.json (used by the harness) and eval/out/phrase_gap.md (for people)
Phrase lists compare drafts with real chunks of the SAME section types; paragraph shape uses every section type."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import config  # noqa: E402
from harness.corpus import store  # noqa: E402
from harness.corpus.profile import TARGET_FEATURES, feature_targets, mine  # noqa: E402
from harness.validate.gapfeats import feats  # noqa: E402


def load_ai():
    """[(fixture, section_type, text)] from every saved draft; identical texts counted once, short ones skipped."""
    seen, out = set(), []
    for p in sorted((ROOT / "eval" / "out" / "thesis_samples").rglob("*.txt")):
        head, _, body = p.read_text(encoding="utf-8", errors="replace").partition("\n\n")
        m = re.match(r"(fx\d+)", p.name)
        sec = re.search(r"Section: (\w+)", head)
        if m and sec and body not in seen and len(body.split()) >= 300:
            seen.add(body)
            out.append((m.group(1), sec.group(1), body))
    return out


def main() -> None:
    ai = load_ai()
    types = sorted({t for _, t, _ in ai})
    con = store.connect()
    rows = con.execute("select c.thesis_id, c.section_type, c.text from chunks c join theses t on t.id = c.thesis_id "
                       "where t.holdout = 0").fetchall()
    real_all = [(r[0], r[1], r[2]) for r in rows]
    real_match = [r for r in real_all if r[1] in types]
    print(f"AI drafts {len(ai)} | real chunks used for phrases {len(real_match)} of {len(real_all)} (types {types})")
    prof = mine(ai, real_match, ratio_real=real_all)
    prof["real_chunks_all_types"] = len(real_all)
    ordered = con.execute("select c.thesis_id, c.section_type, c.chunk_idx, c.text from chunks c join theses t on t.id = c.thesis_id "
                          "where t.holdout = 0").fetchall()
    prof["targets"] = feature_targets([(r[0], r[1], r[2], r[3]) for r in ordered])
    af = [f for f in (feats(t) for _, _, t in ai) if f]
    prof["ai_means"] = {k: round(sum(f[k] for f in af) / len(af), 3) for k in TARGET_FEATURES} if af else {}
    prof["section_types_compared"] = types
    (config.DATA / "style_profile.json").write_text(json.dumps(prof, indent=1), encoding="utf-8")

    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    md = ["# Style profile mined from the real-thesis corpus", "",
          f"AI: {prof['n_ai_docs']} drafts, {prof['ai_words']:,} words, {prof['n_fixtures']} fixtures. Real: {prof['n_real_theses']} theses, "
          f"{prof['n_real_chunks']:,} chunks of the same section types ({prof['real_words']:,} words); paragraph shape from "
          f"{prof['real_chunks_all_types']:,} chunks of every type.", "",
          "## Phrases the AI drafts overuse (rates per 1000 words)", "", "| phrase | AI | real |", "|---|---|---|"]
    md += [f"| {e['p']} | {e['ai']:.2f} | {e['real']:.3f} |" for e in prof["avoid"]]
    md += ["", "## Phrases real theses use and the AI drafts do not", "", "| phrase | real | AI |", "|---|---|---|"]
    md += [f"| {e['p']} | {e['real']:.2f} | {e['ai']:.3f} |" for e in prof["prefer"]]
    md += ["", f"**Sentence openers AI overuses:** {', '.join(prof['avoid_openers'])}", "",
           f"**Sentence openers real theses prefer:** {', '.join(prof['prefer_openers'])}", "",
           f"**Plain signposts** ({', '.join(prof['signposts']['list'])}) open {pct(prof['signposts']['real_rate'])} of real sentences "
           f"and {pct(prof['signposts']['ai_rate'])} of AI sentences.", "",
           f"**Narrative citations** (\"Smith (2010) argues\"): {pct(prof['citations']['narrative_share_real'])} of real citations, "
           f"{pct(prof['citations']['narrative_share_ai'])} of AI citations.", "",
           "## Paragraph length relative to the chunk's mean paragraph (p5, p15, p30, p50, p70, p85, p95)", ""]
    md += [f"- {t}: {v}" for t, v in sorted(prof["paragraph_ratios"].items())]
    out = ROOT / "eval" / "out" / "phrase_gap.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", config.DATA / "style_profile.json", "and", out)


if __name__ == "__main__":
    main()
