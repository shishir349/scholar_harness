"""python eval/ab.py [--fixtures eval/fixtures.json] [--limit N]   (live keys required; cost about 1-3 USD)"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import config, evaluate  # noqa: E402
from harness.corpus import openalex, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(ROOT / "eval" / "fixtures.json"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    if args.limit:
        fixtures = fixtures[:args.limit]
    if not config.db_path().exists():
        print("corpus db missing: run the corpus stages first (python -m harness.corpus.ingest)")
        return 2
    con = store.connect()
    fields = store.get_meta(con, "fields", [])
    rows = evaluate.run_ab(fixtures, con=con, fields=fields, lookup=openalex.find_cited_work if openalex.enabled() else None)
    s = evaluate.summarise(rows)
    criteria = evaluate.evaluate(s)
    out = ROOT / "eval" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"rows": rows, "summary": s, "criteria": criteria}, indent=1), encoding="utf-8")
    from datetime import datetime
    evaluate.save_texts(rows, out / "thesis_samples" / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    md = evaluate.report_md(rows, s, criteria)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0 if all(c["passed"] for c in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
