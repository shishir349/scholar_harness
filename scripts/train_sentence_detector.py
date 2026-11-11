"""Trains data/claude_sentence_detector.json, the weights validate/detector.py scores sentences with. Offline, no API calls.

    pip install scikit-learn        # training only; the runtime needs nothing
    python scripts/train_sentence_detector.py [--human 3000] [--fpr 0.10] [--out data/claude_sentence_detector.json]

Human class: sentences (>= 8 words, citations/numbers masked by the shared tokenizer) of data/corpus.db theses (holdout=0, score >= 4), split BY THESIS
60/20/20 into train / validation / test so no thesis leaks across. Claude class: every sentence of the generated texts under eval/out/thesis_samples
(baseline / harness / polish runs of ten fixtures), the two texts GPTZero and the live run flagged, split BY TOPIC: fx03 validates, fx06 + the two
flagged sections test, the rest train. Features: 1-3 grams of abstract tokens (function words and academic verbs spelled out, content words
replaced by a class), top K by chi2, plus the 12 structural features and the high-ratio "avoid" phrases of data/style_profile.json.
The threshold is the lowest one whose human FPR on the validation split is <= --fpr. All metrics are printed and stored in the JSON ("meta")."""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from harness.text.sentences import split_sentences  # noqa: E402
from harness.validate import detector as D  # noqa: E402

TEST_GROUPS = {"fx06", "gptzero_2_1", "live_3_1"}
VAL_GROUPS = {"fx03"}


def sentences_of(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or para.startswith(("#", "|", "![", "Topic:", "```")):
            continue
        out += [s for s in split_sentences(para) if len(s.split()) >= D.MIN_WORDS]
    return out


def ai_sentences() -> dict:
    """{group: [sentence]} from every generated text on disk (deduplicated within a group)."""
    groups: dict = {}
    for p in sorted((ROOT / "eval" / "out" / "thesis_samples").rglob("*.txt")):
        m = re.search(r"fx\d\d", p.name)
        groups.setdefault(m.group() if m else p.stem, []).extend(sentences_of(p.read_text(encoding="utf-8")))
    for name, rel in (("gptzero_2_1", "eval/fixtures/section_2_1.txt"), ("live_3_1", "eval/out/live_section.md")):
        f = ROOT / rel
        if f.exists():
            groups[name] = sentences_of(f.read_text(encoding="utf-8"))
    return {g: list(dict.fromkeys(v)) for g, v in groups.items()}


def clean_human(s: str) -> bool:
    letters = sum(c.isalpha() for c in s)
    return (8 <= len(s.split()) <= 60 and s[0].isupper() and s[-1] in ".!?" and letters / len(s) > 0.75 and s.isascii()
            and not re.search(r"[|_=<>\[\]{}]|\bhttp|\bwww\.|\bFigure\b|\bTable\b|\bet al\.$", s))


def human_sentences(n: int, seed: int) -> dict:
    """{'train'|'val'|'test': [sentence]} split by thesis; at most ~n sentences in all, spread evenly over theses."""
    con = sqlite3.connect(ROOT / "data" / "corpus.db")
    thesis = [r[0] for r in con.execute("select id from theses where holdout=0 and score>=4 order by id")]
    per = max(5, n // max(1, len(thesis)) + 3)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for tid in thesis:
        bucket = int(hashlib.md5(str(tid).encode()).hexdigest(), 16) % 10
        split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        rows = con.execute("select text from chunks where thesis_id=? and section_type not in ('references','abstract')", (tid,)).fetchall()
        pool = [s for (t,) in rows for s in sentences_of(t) if clean_human(s)]
        rng.shuffle(pool)
        out[split] += pool[:per]
    return out


def avoid_phrases() -> list[str]:
    d = json.loads((ROOT / "data" / "style_profile.json").read_text(encoding="utf-8"))
    return [x["p"] for x in d.get("avoid", []) if x["ai"] >= 0.2 and x["ai"] >= 5 * max(x["real"], 0.001) and len(x["p"].split()) >= 2]


def featurize_all(rows: list[str], avoid: list[str]):
    out = []
    for s in rows:
        grams, dense = D.featurize(s)
        low = s.lower()
        keys = set(grams) | {"avoid:" + p for p in avoid if p in low}
        out.append((keys, [dense[n] for n in D.DENSE]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=int, default=3000)
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(D.PATH))
    a = ap.parse_args()
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    hum, ai = human_sentences(a.human, a.seed), ai_sentences()
    av = avoid_phrases()
    split_ai = {"train": [s for g, v in ai.items() if g not in TEST_GROUPS | VAL_GROUPS for s in v],
                "val": [s for g in VAL_GROUPS for s in ai.get(g, [])],
                "test": [s for g in TEST_GROUPS for s in ai.get(g, [])]}
    print("human sentences train/val/test:", {k: len(v) for k, v in hum.items()})
    print("claude sentences train/val/test:", {k: len(v) for k, v in split_ai.items()}, "| groups:", {g: len(v) for g, v in ai.items()})
    print("avoid phrases (ai/real >= 5):", len(av))
    F = {sp: featurize_all(hum[sp] + split_ai[sp], av) for sp in hum}
    y = {sp: np.array([0] * len(hum[sp]) + [1] * len(split_ai[sp])) for sp in hum}

    # vocabulary: keys seen in >= 3 training sentences, top K by chi2 (plus every avoid phrase)
    df: dict = {}
    for keys, _ in F["train"]:
        for k in keys:
            df[k] = df.get(k, 0) + 1
    vocab = sorted(k for k, c in df.items() if c >= 3)
    index = {k: i for i, k in enumerate(vocab)}

    def sparse(rows):
        r, c = [], []
        for i, (keys, _) in enumerate(rows):
            for k in keys:
                j = index.get(k)
                if j is not None:
                    r.append(i)
                    c.append(j)
        return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(vocab)))

    Xtr = sparse(F["train"])
    scores, _ = chi2(Xtr, y["train"])
    scores = np.nan_to_num(scores)
    keep = sorted(set(np.argsort(-scores)[:a.k]) | {index["avoid:" + p] for p in av if "avoid:" + p in index})
    kept = [vocab[i] for i in keep]
    kidx = {k: i for i, k in enumerate(kept)}
    dense_tr = np.array([d for _, d in F["train"]])
    mean, std = dense_tr.mean(axis=0), dense_tr.std(axis=0)
    std[std == 0] = 1.0

    def matrix(rows):
        r, c = [], []
        for i, (keys, _) in enumerate(rows):
            for k in keys:
                j = kidx.get(k)
                if j is not None:
                    r.append(i)
                    c.append(j)
        sp = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(kept)))
        return hstack([sp, csr_matrix((np.array([d for _, d in rows]) - mean) / std)]).tocsr()

    X = {sp: matrix(F[sp]) for sp in F}
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear").fit(X["train"], y["train"])
        auc = roc_auc_score(y["val"], clf.decision_function(X["val"]))
        print(f"  C={C:<5} validation AUC {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    auc_val, C, clf = best

    def prob(sp):
        return 1 / (1 + np.exp(-clf.decision_function(X[sp])))

    pv = prob("val")
    hv = np.sort(pv[y["val"] == 0])
    thr = float(hv[int(np.ceil((1 - a.fpr) * len(hv))) - 1]) + 1e-6          # lowest threshold with validation human FPR <= --fpr
    pt, yt = prob("test"), y["test"]
    meta = {"trained": time.strftime("%Y-%m-%d"), "C": C, "k": len(kept), "human_sentences": {k: len(v) for k, v in hum.items()},
            "claude_sentences": {k: len(v) for k, v in split_ai.items()}, "claude_groups": {g: len(v) for g, v in ai.items()},
            "val_auc": round(auc_val, 3), "val_human_fpr_target": a.fpr, "test_auc": round(float(roc_auc_score(yt, pt)), 3),
            "test_human_flagged": round(float((pt[yt == 0] >= thr).mean()), 3), "test_claude_flagged": round(float((pt[yt == 1] >= thr).mean()), 3),
            "test_by_group": {}}
    for g in sorted(TEST_GROUPS):
        if g in ai:
            Fg = featurize_all(ai[g], av)
            pg = 1 / (1 + np.exp(-clf.decision_function(matrix(Fg))))
            meta["test_by_group"][g] = {"n": len(pg), "flagged": round(float((pg >= thr).mean()), 3), "mean_prob": round(float(pg.mean()), 3)}
    coef = clf.coef_[0]
    model = {"version": 1, "bias": float(clf.intercept_[0]), "threshold": round(thr, 4),
             "ngrams": {k: round(float(coef[i]), 4) for i, k in enumerate(kept) if not k.startswith("avoid:") and abs(coef[i]) > 1e-4},
             "avoid": {k[6:]: round(float(coef[i]), 4) for i, k in enumerate(kept) if k.startswith("avoid:") and abs(coef[i]) > 1e-4},
             "dense": {"names": list(D.DENSE), "mean": [round(float(x), 5) for x in mean], "std": [round(float(x), 5) for x in std],
                       "w": [round(float(x), 4) for x in coef[len(kept):]]},
             "meta": meta}
    Path(a.out).write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print("top Claude n-grams:", sorted(model["ngrams"].items(), key=lambda kv: -kv[1])[:15])
    print("top human n-grams: ", sorted(model["ngrams"].items(), key=lambda kv: kv[1])[:10])
    print("dense weights:", dict(zip(D.DENSE, model["dense"]["w"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
