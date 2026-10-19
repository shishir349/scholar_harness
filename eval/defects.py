"""Free check, no API calls: defects a reader (or the judge) trips on, real theses vs saved baseline vs saved harness drafts.
    python eval/defects.py [file.txt ...]        (with files: print the defects found in just those)
The patterns are the ones harness/text/cleanup.py removes or flags, so a draft that went through it reads near 0 here."""
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness.corpus import store  # noqa: E402
from harness.text.cleanup import CLOSER, EXAMPLE, REPORT as REPORT_VERB, STANCE  # noqa: E402
from harness.text.numbers import has_first_person  # noqa: E402
from harness.validate.features import style_features  # noqa: E402
from harness.validate.profile import signpost_counts  # noqa: E402
from harness.sources.tools import _FIGURE  # noqa: E402
from harness.text.citations import extract_citations  # noqa: E402
from harness.text.sentences import split_sentences  # noqa: E402
from harness.text.strip import strip_for_style  # noqa: E402

STOP = set("the a an of in on to and or for with by is are was were that this these those as at from it its be been which than but not".split())


def _content(s: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP)


def defects(text: str) -> dict:
    t = strip_for_style(text)
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    sents = [s for p in paras for s in split_sentences(p)]
    words = max(1, len(t.split()))
    sets = [(s, _content(s)) for s in sents if len(s.split()) >= 8]
    dup = sum(1 for i in range(len(sets)) for j in range(i + 1, len(sets))
              if len(sets[i][1] | sets[j][1]) and len(sets[i][1] & sets[j][1]) / len(sets[i][1] | sets[j][1]) >= 0.5)
    stems: dict = {}
    for s in sents:
        k = " ".join(s.lower().split()[:5])
        stems[k] = stems.get(k, 0) + 1
    grams: dict = {}
    for s in sents:
        w = s.lower().split()
        for i in range(len(w) - 5):
            grams[" ".join(w[i:i + 6])] = grams.get(" ".join(w[i:i + 6]), 0) + 1
    lens = [len(s.split()) for s in sents]
    last = [split_sentences(p)[-1] for p in paras if split_sentences(p)]
    gate = style_features(t) or {}
    return {
        "dup_sentence_pairs": dup,
        "fragment_paras": sum(1 for p in paras if p[0].islower() or p.rstrip()[-1] not in '.!?)"”’'),
        "repeated_openers": sum(1 for k, n in stems.items() if n >= 2 and len(k.split()) == 5),
        "repeated_phrases": sum(1 for n in grams.values() if n >= 2),
        "stance_stock": len(STANCE.findall(t)),
        "closer_paras": sum(1 for s in last if CLOSER.search(s)),
        "report_verb_cites": len([s for s in sents if REPORT_VERB.search(s)]),
        "short_sents": sum(1 for n in lens if n <= 8),
        "sent_cv": round(statistics.pstdev(lens) / statistics.fmean(lens), 2) if len(lens) > 2 else 0.0,
        "cite_sent_share": round(sum(1 for s in sents if extract_citations(s)) / max(1, len(sents)), 2),
        "digit_para_share": round(sum(1 for p in paras if len(p.split()) >= 40 and _FIGURE.search(p)) / max(1, sum(1 for p in paras if len(p.split()) >= 40)), 2),
        "cite_para_share": round(sum(1 for p in paras if len(p.split()) >= 40 and extract_citations(p)) / max(1, sum(1 for p in paras if len(p.split()) >= 40)), 2),
        "examples_per_1000": round(1000 * len(EXAMPLE.findall(t)) / words, 1),
        "fp_para_share": round(sum(1 for p in paras if len(p.split()) >= 40 and has_first_person(p)) / max(1, sum(1 for p in paras if len(p.split()) >= 40)), 2),
        "signpost_share": round(signpost_counts(t)[0] / max(1, signpost_counts(t)[1]), 3),
        # the gate's own measure: real chunks hold captions and cut-off pieces that inflate short_sents / sent_cv above
        "gate_short_ratio": round(gate.get("short_sent_ratio", 0.0), 3), "gate_sent_cv": round(gate.get("sent_len_cv", 0.0), 2),
    }


def _drafts(arm: str) -> list[str]:
    root = ROOT / "eval" / "out" / "thesis_samples"
    files = [p for p in sorted(root.rglob("*.txt")) if arm in p.name or p.parent.name == arm]
    return [t for t in (p.read_text(encoding="utf-8", errors="replace").partition("\n\n")[2] for p in files) if len(t.split()) > 300]


def _real() -> list[str]:
    con = store.connect()
    rows = con.execute("select c.text from chunks c join theses t on t.id=c.thesis_id where t.holdout=1 and c.section_type in "
                       "('introduction','literature_review','methodology','discussion')").fetchall()
    random.Random(1).shuffle(rows)
    return ["\n\n".join(r[0] for r in rows[i:i + 3]) for i in range(0, min(len(rows), 90), 3)]


def main() -> None:
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            print(f, defects(Path(f).read_text(encoding="utf-8", errors="replace").partition("\n\n")[2] or Path(f).read_text(encoding="utf-8")))
        return
    arms = {"real (holdout)": _real(), "baseline (old app)": _drafts("baseline"), "harness": _drafts("harness")}
    rows = {k: [defects(t) for t in v] for k, v in arms.items()}
    keys = list(next(iter(rows.values()))[0])
    print(f"{'metric (mean per draft)':26}" + "".join(f"{k + f' n={len(v)}':>26}" for k, v in rows.items()))
    for key in keys:
        print(f"{key:26}" + "".join(f"{statistics.fmean(d[key] for d in v):26.2f}" for v in rows.values()))


if __name__ == "__main__":
    main()
