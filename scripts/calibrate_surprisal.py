"""Sets data/surprisal_baseline.json from real human theses in data/corpus.db (needs the local model, see validate/surprisal.py).

    python scripts/calibrate_surprisal.py [--n 60] [--seed 7]

Scores N random body paragraphs of 60-250 words and writes the distribution of their sentence means. `low` is the 15th percentile of
human sentences: a sentence below it is more predictable than 85% of what human thesis authors wrote."""
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


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    sc = surprisal.get_scorer()
    if sc is None:
        print("no scorer: put lm-q.onnx and tokenizer.json in data/models/ and install onnxruntime, numpy, tokenizers")
        return 2
    rows = sqlite3.connect(ROOT / "data" / "corpus.db").execute(
        "select text from chunks where section_type not in ('references','abstract') and word_count between 150 and 600").fetchall()
    random.Random(a.seed).shuffle(rows)
    paras = [p.strip() for (t,) in rows for p in re.split(r"\n\s*\n", t) if 60 <= len(p.split()) <= 250 and not p.lstrip().startswith(("#", "|"))]
    paras = paras[:a.n]
    sent, paramean = [], []
    for p in paras:
        r = surprisal.score_section_surprisal(p, sc, low=0)
        if r:
            sent += [s["mean"] for s in r["sentences"] if s["n"] >= 8]
            paramean.append(r["mean"])
    out = {"low": round(pct(sent, 0.15), 3), "sentence_median": round(statistics.median(sent), 3), "sentence_p85": round(pct(sent, 0.85), 3),
           "paragraph_mean_median": round(statistics.median(paramean), 3), "paragraphs": len(paramean), "sentences": len(sent),
           "model": "distilgpt2-int8", "unit": "nats per token"}
    (ROOT / "data" / "surprisal_baseline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
