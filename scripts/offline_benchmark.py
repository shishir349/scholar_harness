"""Offline verification of the free repairs and the local surprisal scorer. No API calls.

    python scripts/offline_benchmark.py [--text eval/fixtures/section_2_1.txt] [--human 60]

Prints: (1) skeleton counts, (2) the deterministic cleanup on the flagged section, (3) surprisal on it vs real human paragraphs."""
import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import limits  # noqa: E402
from harness.text import cleanup as C, protect  # noqa: E402
from harness.text.sentences import split_sentences  # noqa: E402
from harness.validate import constraints as K, surprisal  # noqa: E402


def skeleton_counts(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["skeletons"]
    by = Counter((s["section_type"], s["position"]) for s in d)
    micro = sum(1 for s in d if any(len(p["sentence_lengths"]) < 3 or p["word_count"] < limits.MIN_SKELETON_WORDS for p in s["paragraphs"]))
    return len(d), len(by), min(by.values()), micro


def blocks(text):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip() and not p.startswith("#")]
    return {"open": "\n\n".join(paras[:2]), "mid": paras[2], "close": paras[3]}


def n_paras(chunks):
    return sum(len(C.paras(v)) for v in chunks.values())


def n_long(chunks):
    return sum(1 for v in chunks.values() for s in split_sentences(v) if len(s.split()) > limits.MAX_SENT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=str(ROOT / "eval" / "fixtures" / "section_2_1.txt"))
    ap.add_argument("--human", type=int, default=60)
    ap.add_argument("--before-skeletons", default="")
    a = ap.parse_args()
    text = Path(a.text).read_text(encoding="utf-8")
    chunks = blocks(text)
    n_sent = sum(len(split_sentences(v)) for v in chunks.values())

    print("== 1. skeletons (data/human_skeletons.json)")
    if a.before_skeletons:
        print("   before: total %d, buckets %d, smallest bucket %d, with a micro-paragraph %d" % skeleton_counts(a.before_skeletons))
    print("   now:    total %d, buckets %d, smallest bucket %d, with a micro-paragraph %d" % skeleton_counts(ROOT / "data" / "human_skeletons.json"))

    print("\n== 2. deterministic cleanup on the flagged section (%d sentences, %d words)" % (n_sent, len(text.split())))
    stress = {"open": "\n\n".join(s for s in split_sentences(chunks["open"])), "mid": "\n\n".join(split_sentences(chunks["mid"])),
              "close": "\n\n".join(split_sentences(chunks["close"]))}
    merged = C.merge_short_paragraphs(chunks)
    print("   paragraphs: as extracted %d -> %d after merge_short_paragraphs (the PDF keeps 4 blocks; GPTZero's export drops the real breaks)" % (n_paras(chunks), n_paras(merged)))
    print("   stress test, one paragraph per sentence: %d -> %d after merge_short_paragraphs (cap %d words)" % (n_paras(stress), n_paras(C.merge_short_paragraphs(stress)), C.MAX_MERGED_WORDS))
    before = K.section_violations(chunks)
    print("   narrative / parenthetical citations before: %d / %.0f%% parenthetical" % (before["narrative_section"], 100 * before["paren_share"]))
    for label, src in (("limit_narrative (per call %d, section %d)" % (limits.NARR_CALL, limits.NARR_SECTION), chunks),
                       ("drop_repeats, then limit_narrative", C.drop_repeats(chunks, [])[0])):
        for sec in (limits.NARR_SECTION, 4):
            out = C.limit_narrative(src, per_section=sec)
            v = K.section_violations(out)
            safe = protect.compare(" ".join(chunks.values()), " ".join(out.values())) == [] or "drop_repeats" in label
            print("     %-44s section limit %d -> narrative %2d, parenthetical %3.0f%%, sentences %d, citations/numbers kept: %s"
                  % (label, sec, v["narrative_section"], 100 * v["paren_share"], sum(len(split_sentences(x)) for x in out.values()), safe))
    print("   sentences over %d words: %d -> %d after split_long" % (limits.MAX_SENT, n_long(chunks), n_long(C.split_long(chunks, limits.MAX_SENT))))

    print("\n== 3. surprisal (distilgpt2 int8, masked tokens) on the flagged section vs %d human paragraphs of data/corpus.db" % a.human)
    sc = surprisal.get_scorer()
    if sc is None:
        print("   no scorer (model files or onnxruntime missing)")
        return 0
    import calibrate_surprisal as cal
    t = time.perf_counter()
    surprisal.clear_cache()
    r = surprisal.score_section_surprisal(text, sc)
    cold = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    surprisal.score_section_surprisal(text, sc)
    cached = (time.perf_counter() - t) * 1000
    human, n_par = cal.human_sentences(sc, a.human, 7)
    ai = [s for s in r["sentences"] if s["n"] >= surprisal.MIN_TOKENS]
    tm, tw = r["threshold"]["mean"], r["threshold"]["window"]
    hflag = sum(1 for s in human if s["mean"] < tm or s["min_window"] < tw)
    med = statistics.median
    print("   shipped thresholds: mean < %.2f or 6-token window < %.2f nats/token" % (tm, tw))
    print("   %-26s %12s %12s" % ("", "flagged text", "human corpus"))
    print("   %-26s %12.2f %12.2f" % ("masked_mean_surprisal (med)", med(s["mean"] for s in ai), med(s["mean"] for s in human)))
    print("   %-26s %12.2f %12.2f" % ("min_window_surprisal (med)", med(s["min_window"] for s in ai), med(s["min_window"] for s in human)))
    print("   %-26s %12.2f %12.2f" % ("lstr, top-5 share (med)", med(s["lstr"] for s in ai), med(s["lstr"] for s in human)))
    print("   %-26s %9d/%d %9d/%d  (%.0f%% / %.0f%%)" % ("sentences flagged", len(r["low"]), len(r["sentences"]), hflag, len(human), 100 * len(r["low"]) / len(r["sentences"]), 100 * hflag / len(human)))
    for want in (0.5, 0.65):
        best = min(((sum(1 for s in human if s["mean"] < m or s["min_window"] < w) / len(human), m, w) for m in [x / 20 for x in range(30, 140)] for w in [x / 20 for x in range(10, 120)]
                    if sum(1 for s in ai if s["mean"] < m or s["min_window"] < w) / len(ai) >= want), default=None)
        print("   to flag %d%% of the flagged text's sentences a threshold must also flag %.0f%% of human sentences (mean<%.2f, window<%.2f)" % (100 * want, 100 * best[0], best[1], best[2]))
    print("   top 10 predictable spans of the flagged text (lowest window first):")
    for s in sorted(ai, key=lambda s: s["min_window"])[:10]:
        print("     %.2f  %s" % (s["min_window"], s["predictable_span"]))
    print("   latency, %d words: %.0f ms cold (all %d tokens through the model), %.0f ms when every paragraph is cached" % (len(text.split()), cold, r["n_tokens"], cached))
    return 0


if __name__ == "__main__":
    sys.exit(main())
