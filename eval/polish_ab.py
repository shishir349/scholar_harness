"""One-off: polish (humanize) on vs off, same fixtures, same research/ledger shared across both arms.
    python eval/polish_ab.py [--fixtures eval/peek_fixtures.json]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import evaluate  # noqa: E402
from harness.corpus import openalex, store  # noqa: E402

evaluate.ARMS["full_polish"] = {**evaluate.ARMS["full"], "humanize": True}   # "full" has polish off by default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(ROOT / "eval" / "peek_fixtures.json"))
    args = ap.parse_args()
    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    con = store.connect()
    fields = store.get_meta(con, "fields", [])
    rows = evaluate.run_ab(fixtures, con=con, fields=fields, lookup=openalex.find_cited_work if openalex.enabled() else None)
    s = evaluate.summarise(rows)
    out = ROOT / "eval" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "polish_compare.json").write_text(json.dumps({"rows": rows, "summary": s}, indent=1), encoding="utf-8")
    from datetime import datetime
    evaluate.save_texts(rows, out / "thesis_samples" / f"polish_run_{datetime.now():%Y%m%d_%H%M%S}")
    print(f"{'arm':16} {'style':>7} {'struct':>7} {'h2h':>6} {'unmatched':>9} {'usd':>7}")
    for arm in ("baseline", "full", "full_polish"):
        a = s[arm]
        num = lambda v, f="{:.2f}": "n/a" if v is None else f.format(v)
        print(f"{arm:16} {num(a['style_mean']):>7} {num(a['structure_mean']):>7} {num(a['h2h_score']):>6} "
              f"{a['unmatched_total']:>9} {num(a['usd_total']):>7}")


if __name__ == "__main__":
    main()
