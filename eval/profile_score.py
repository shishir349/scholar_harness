"""Free check, no API calls: score real thesis text and every saved AI draft on the mined profile's scale.
    python eval/profile_score.py [file.txt ...]        (with files: score just those)
Caveat: the saved AI drafts were also used to mine the profile, so their tell rates are in-sample. Real text is scored on
the 30 holdout theses, which the miner never saw."""
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness.corpus import store  # noqa: E402
from harness.validate.profile import load_profile, profile_metrics  # noqa: E402


def row(label: str, texts: list[str], prof: dict) -> None:
    ms = [profile_metrics(t, prof) for t in texts]
    nar = [m["narrative_share"] for m in ms if m["narrative_share"] is not None]
    print(f"{label:34} n={len(ms):3}  tells/1000 {statistics.fmean(m['tells_per_1000'] for m in ms):5.1f}   "
          f"signpost sentences {100 * statistics.fmean(m['signpost_rate'] for m in ms):4.1f}%   "
          f"narrative cites {100 * statistics.fmean(nar) if nar else float('nan'):4.0f}%")


def main() -> None:
    prof = load_profile()
    if not prof:
        sys.exit("no data/style_profile.json: run python eval/phrase_gap.py first")
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            row(f, [Path(f).read_text(encoding="utf-8")], prof)
        return
    con = store.connect()
    types = prof.get("section_types_compared", ["introduction"])
    rows = con.execute(f"select c.text from chunks c join theses t on t.id=c.thesis_id where t.holdout=1 and c.section_type in "
                       f"({','.join('?' * len(types))})", types).fetchall()
    random.Random(1).shuffle(rows)
    # real chunks are ~440 words; join 3 so they are comparable in length to a drafted section
    real = ["\n\n".join(r[0] for r in rows[i:i + 3]) for i in range(0, min(len(rows), 300), 3)]
    row("REAL theses (holdout, 3 chunks each)", real, prof)
    for arm in ("baseline", "harness"):
        texts = [p.read_text(encoding="utf-8", errors="replace").partition("\n\n")[2]
                 for p in sorted((ROOT / "eval" / "out" / "thesis_samples").rglob(f"*{arm}*.txt"))]
        texts += [p.read_text(encoding="utf-8", errors="replace").partition("\n\n")[2]
                  for p in sorted((ROOT / "eval" / "out" / "thesis_samples" / arm).glob("*.txt"))] if (ROOT / "eval" / "out" / "thesis_samples" / arm).is_dir() else []
        row(f"AI drafts, {arm} arm (saved)", [t for t in texts if len(t.split()) > 300], prof)


if __name__ == "__main__":
    main()
