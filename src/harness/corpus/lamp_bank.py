"""Human span examples for the LAMP rewrite: real thesis sentence pairs with sharp length contrast, mined from data/corpus.db.

    python -m harness.corpus.lamp_bank [--min-score 4] [--out data/lamp_span_examples.json]

Two kinds, five per section type (25 in all, each from a different thesis within its type):
  long_then_blunt  a dense 30-45 word sentence followed at once by a 5-11 word blunt one
  abrupt_close     a paragraph that ends on a short sentence holding a figure or a citation, with no wrap-up
The examples go to the rewriter as a model of rhythm only; lamp_rewrite rejects any rewrite that copies their wording."""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from .. import config
from ..text.cleanup import CLOSER
from ..text.sentences import split_sentences
from ..validate.features import _STOCK_RX
from . import skeletons as SK, store

PATH = config.DATA / "lamp_span_examples.json"
TYPES = ("introduction", "literature_review", "methodology", "findings", "discussion")
PER_KIND = {"long_then_blunt": 3, "abrupt_close": 2}
LONG, BLUNT = (30, 45), (5, 11)
CITE = re.compile(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)")
FIGURE = re.compile(r"(?<![\w.])(?!(?:19|20)\d\d[a-z]?\b)\d")


def _clean(s: str) -> bool:
    """A sentence worth showing: capitalised, ends with a full stop, no markup, no stock phrase."""
    return (s[:1].isupper() and s.endswith(".") and s.isascii() and not re.search(r"[\[\]|<>\n_~*]|https?:|www\.|\.com|\bfig(?:ure)?\b", s, re.I)
            and not _STOCK_RX.search(s) and len(s.split()) <= 60)


def pairs_in(text: str):
    """(kind, [sentences], score) for every qualifying pair in a chunk's paragraphs."""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if para.startswith("#"):
            continue
        sents = split_sentences(para)
        n = [len(s.split()) for s in sents]
        for j in range(len(sents) - 1):
            a, b = sents[j], sents[j + 1]
            if LONG[0] <= n[j] <= LONG[1] and BLUNT[0] <= n[j + 1] <= BLUNT[1] and _clean(a) and _clean(b):
                yield "long_then_blunt", [a, b], n[j] - n[j + 1] + (6 if FIGURE.search(b) or CITE.search(b) else 0)
        if len(sents) >= 3 and 5 <= n[-1] <= 12 and 25 <= n[-2] <= 50 and (FIGURE.search(sents[-1]) or CITE.search(sents[-1])) and "communication" not in sents[-2] \
                and not CLOSER.search(sents[-1]) and _clean(sents[-2]) and _clean(sents[-1]):
            yield "abrupt_close", sents[-2:], n[-2] - n[-1]


def build(con, min_score: float = 4) -> list[dict]:
    """The example list. Deterministic: best thesis score first, then sharpest contrast."""
    best: dict = {}          # (type, kind) -> {thesis_id: example}
    rows = con.execute("SELECT c.section_type, c.heading, c.text, t.id AS tid, t.score FROM chunks c JOIN theses t ON t.id = c.thesis_id "
                       "WHERE t.holdout = 0 AND t.score >= ? ORDER BY t.score DESC, c.id", (min_score,))
    for r in rows:
        typ = SK.reclassify(r["heading"], r["section_type"])
        if typ not in TYPES:
            continue
        for kind, sents, contrast in pairs_in(r["text"]):
            slot = best.setdefault((typ, kind), {})
            cur = slot.get(r["tid"])
            if cur is None or contrast > cur["contrast"]:
                slot[r["tid"]] = {"section_type": typ, "kind": kind, "sentences": sents, "lengths": [len(s.split()) for s in sents],
                                  "thesis_id": r["tid"], "score": r["score"], "contrast": contrast}
    out = []
    for typ in TYPES:
        for kind, k in PER_KIND.items():
            ranked = sorted(best.get((typ, kind), {}).values(), key=lambda e: (-e["contrast"], -e["score"], e["thesis_id"]))
            out += [{x: e[x] for x in ("section_type", "kind", "sentences", "lengths", "thesis_id")} for e in ranked[:k]]
    return out


@lru_cache(maxsize=2)
def load(path: str | None = None) -> list:
    try:
        data = json.loads(Path(path or PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def pick(examples: list, section_type: str, n: int = 8) -> list:
    """n examples, this section type's first, then the rest in file order (mixed kinds), so a thin type still gets a full set."""
    mine = [e for e in examples or [] if e.get("section_type") == section_type]
    rest = [e for e in examples or [] if e.get("section_type") != section_type]
    return (mine + rest)[:n]


def text_of(example: dict) -> str:
    return " ".join(example.get("sentences") or [])


def main():
    ap = argparse.ArgumentParser(description="Build data/lamp_span_examples.json from data/corpus.db")
    ap.add_argument("--min-score", type=float, default=4)
    ap.add_argument("--out", default=str(PATH))
    a = ap.parse_args()
    con = store.connect()
    ex = build(con, a.min_score)
    Path(a.out).write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
    by = {}
    for e in ex:
        by.setdefault(e["section_type"], []).append(e["kind"])
    print(f"{len(ex)} examples -> {a.out}")
    for t, ks in by.items():
        print(f"  {t}: {len(ks)}  ({', '.join(sorted(set(ks)))})")


if __name__ == "__main__":
    main()
