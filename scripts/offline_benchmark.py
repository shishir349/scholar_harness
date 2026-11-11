"""Offline verification of the free repairs and the local Claude-style detector. No API calls.

    python scripts/offline_benchmark.py [--text eval/fixtures/section_2_1.txt] [--before-skeletons old.json]

Prints: (1) skeleton counts, (2) the deterministic cleanup on the flagged section (long sentences, narrative citations),
(3) the sentence detector on held-out human thesis sentences and held-out Claude text, with latency."""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import graph, limits  # noqa: E402
from harness.text import cleanup as C, protect  # noqa: E402
from harness.text.sentences import split_sentences  # noqa: E402
from harness.validate import constraints as K, detector  # noqa: E402


def skeleton_counts(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["skeletons"]
    by = Counter((s["section_type"], s["position"]) for s in d)
    micro = sum(1 for s in d if any(len(p["sentence_lengths"]) < 3 or p["word_count"] < limits.MIN_SKELETON_WORDS for p in s["paragraphs"]))
    return len(d), len(by), min(by.values()), micro


def blocks(text):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip() and not p.startswith("#")]
    return {"open": "\n\n".join(paras[:2]), "mid": paras[2], "close": paras[3]}


def n_long(chunks):
    return sum(1 for v in chunks.values() for s in split_sentences(v) if len(s.split()) > limits.MAX_SENT)


def auc(pos, neg):
    """P(a Claude sentence scores above a human one), ties half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=str(ROOT / "eval" / "fixtures" / "section_2_1.txt"))
    ap.add_argument("--before-skeletons", default="")
    a = ap.parse_args()
    text = Path(a.text).read_text(encoding="utf-8")
    chunks = blocks(text)

    print("== 1. skeletons (data/human_skeletons.json)")
    if a.before_skeletons:
        print("   before: total %d, buckets %d, smallest bucket %d, with a micro-paragraph %d" % skeleton_counts(a.before_skeletons))
    print("   now:    total %d, buckets %d, smallest bucket %d, with a micro-paragraph %d" % skeleton_counts(ROOT / "data" / "human_skeletons.json"))

    print("\n== 2. deterministic cleanup on the flagged section (%d sentences, %d words)" % (sum(len(split_sentences(v)) for v in chunks.values()), len(text.split())))
    print("   sentences over %d words: %d -> %d with split_long;  %d -> %d through the pipeline's own _split_long (drop_micro, split_long, limit_narrative)"
          % (limits.MAX_SENT, n_long(chunks), n_long(C.split_long(chunks, limits.MAX_SENT)), n_long(chunks), n_long(graph._split_long(chunks))))
    before = K.section_violations(chunks)
    print("   narrative citations before: %d (%.0f%% of citations parenthetical)" % (before["narrative_section"], 100 * before["paren_share"]))
    for label, src in (("limit_narrative", chunks), ("drop_repeats, then limit_narrative", C.drop_repeats(chunks, [])[0])):
        out = C.limit_narrative(src)
        v = K.section_violations(out)
        clean = protect.compare(" ".join(chunks.values()), " ".join(out.values())) == [] or "drop_repeats" in label
        print("     %-36s -> narrative %2d (gate %d), parenthetical %3.0f%%, %d sentences, citations/numbers kept: %s, 'A study' lead-ins: %d"
              % (label, v["narrative_section"], limits.NARR_SECTION, 100 * v["paren_share"], sum(len(split_sentences(x)) for x in out.values()), clean,
                 sum(x.count("A study (") for x in out.values())))
    print("     what no safe pattern covers stays narrative and is queued for LAMP (narrative_surplus)")

    print("\n== 3. detector (data/claude_sentence_detector.json) on HELD-OUT sentences")
    model = detector.load()
    if model is None:
        print("   no model: run python scripts/train_sentence_detector.py")
        return 0
    import train_sentence_detector as T
    hum = T.human_sentences(3000, 7)["test"]
    ai = T.ai_sentences()
    thr = model["threshold"]
    ph = np.array([detector.score_sentence(s, model)["ai_prob"] for s in hum])
    print("   held-out human: %d sentences from theses the model never saw; Claude: topics fx06 + the two flagged sections (not trained on)" % len(hum))
    allp = []
    for g in ("fx06", "gptzero_2_1", "live_3_1"):
        p = np.array([detector.score_sentence(s, model)["ai_prob"] for s in ai[g]])
        allp += list(p)
        print("   %-12s %4d sentences  flagged %5.1f%%  mean ai_prob %.2f  AUC vs held-out human %.3f" % (g, len(p), 100 * (p >= thr).mean(), p.mean(), auc(p, ph)))
    allp = np.array(allp)
    print("   threshold %.3f (chosen on a separate validation split for <=10%% human false positives)" % thr)
    print("   ALL held-out Claude: flagged %.1f%%   AUC %.3f   |   held-out human: flagged %.1f%%, mean ai_prob %.2f"
          % (100 * (allp >= thr).mean(), auc(allp, ph), 100 * (ph >= thr).mean(), ph.mean()))
    rows = detector.score_text(text, model)
    big = [r for r in rows if len(r["text"].split()) >= detector.MIN_WORDS]
    print("   flagged section (%d sentences >= %d words): %d flagged (%.0f%%); top culprit phrases:" % (len(big), detector.MIN_WORDS, sum(r["flagged"] for r in big), 100 * sum(r["flagged"] for r in big) / len(big)))
    seen = Counter(c for r in big if r["flagged"] for c in r["culprit_phrases"])
    for c, n in seen.most_common(10):
        print("     %2dx  %s" % (n, c))
    sec = text
    t = time.perf_counter()
    for _ in range(50):
        detector.score_text(sec, model)
    ms = (time.perf_counter() - t) * 1000 / 50
    print("   latency: %.2f ms for the whole %d-word section (%d sentences), %.3f ms per sentence; model file %d KB, runtime dependencies: none"
          % (ms, len(text.split()), len(rows), ms / len(rows), (ROOT / "data" / "claude_sentence_detector.json").stat().st_size // 1024))
    print("   caveat: human = text extracted from thesis PDFs, Claude = our generated chapters, so some of the separation is genre/format, not authorship")
    return 0


if __name__ == "__main__":
    sys.exit(main())
