"""Sets data/surprisal_baseline.json: the two thresholds of validate/surprisal.py, fitted offline (no API calls, local model only).

    python scripts/calibrate_surprisal.py [--n 100] [--seed 7] [--ai eval/fixtures/section_2_1.txt] [--max-human-fpr 0.19]

Human sentences: N random body paragraphs (60-250 words) of data/corpus.db. AI sentences: the text GPTZero scored 100% AI (44 of 51 sentences).
Searches (mean threshold, window threshold) for the highest share of AI sentences flagged while at most --max-human-fpr of human sentences are.
One AI text of ~50 sentences is a small sample: the fit is tuned to it, so check the result on other generated text before trusting it."""
import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from harness.validate import surprisal  # noqa: E402


def sentences(text, sc):
    r = surprisal.score_section_surprisal(text, sc, low=-1, window=-1)      # thresholds off: only the features
    return [s for s in (r["sentences"] if r else []) if s["n"] >= surprisal.MIN_TOKENS]


def human_sentences(sc, n, seed):
    rows = sqlite3.connect(ROOT / "data" / "corpus.db").execute(
        "select text from chunks where section_type not in ('references','abstract') and word_count between 150 and 600").fetchall()
    random.Random(seed).shuffle(rows)
    paras = [p.strip() for (t,) in rows for p in re.split(r"\n\s*\n", t) if 60 <= len(p.split()) <= 250 and not p.lstrip().startswith(("#", "|"))][:n]
    return [s for p in paras for s in sentences(p, sc)], len(paras)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ai", default=str(ROOT / "eval" / "fixtures" / "section_2_1.txt"))
    ap.add_argument("--max-human-fpr", type=float, default=0.19)
    a = ap.parse_args()
    sc = surprisal.get_scorer()
    if sc is None:
        print("no scorer: put lm-q.onnx and tokenizer.json in data/models/ and install onnxruntime, numpy, tokenizers")
        return 2
    human, n_par = human_sentences(sc, a.n, a.seed)
    ai = sentences(Path(a.ai).read_text(encoding="utf-8"), sc)

    def rate(rows, tm, tw):
        return sum(1 for s in rows if s["mean"] < tm or s["min_window"] < tw) / len(rows)

    best = None
    for tm in [x / 20 for x in range(30, 100)]:
        for tw in [x / 20 for x in range(10, 90)]:
            fpr, tpr = rate(human, tm, tw), rate(ai, tm, tw)
            if fpr <= a.max_human_fpr and (best is None or (tpr, -fpr) > (best[0], -best[1])):
                best = (tpr, fpr, tm, tw)
    tpr, fpr, tm, tw = best
    med = lambda rows, k: round(statistics.median(s[k] for s in rows), 3)      # noqa: E731
    out = {"mean": tm, "window": tw, "human_flagged": round(fpr, 3), "ai_flagged": round(tpr, 3), "ai_sentences": len(ai), "human_sentences": len(human),
           "human_paragraphs": n_par, "model": "distilgpt2-int8", "unit": "nats per unmasked token",
           "human_median_mean": med(human, "mean"), "ai_median_mean": med(ai, "mean"),
           "human_median_window": med(human, "min_window"), "ai_median_window": med(ai, "min_window"),
           "human_median_lstr": med(human, "lstr"), "ai_median_lstr": med(ai, "lstr")}
    (ROOT / "data" / "surprisal_baseline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
