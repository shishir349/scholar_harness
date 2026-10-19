"""Task 1.7: load the reviewed manifest into data/corpus.db and write data/style_baseline.json.
    python -m harness.corpus.ingest"""
import json

from .. import config
from ..text.refs import looks_like_references
from ..validate.style import build_baseline
from . import store
from .acquire import RAW


def ingest(manifest: list[dict], raw_root, con, baseline_path, fields: list[dict]) -> dict:
    units_by_type: dict[str, list[str]] = {}
    n_theses = n_chunks = 0
    for row in manifest:
        if row.get("exclude"):
            continue
        built = json.loads((raw_root / row["field_slug"] / row["slug"] / "sections.json").read_text(encoding="utf-8"))
        tid = store.upsert_thesis(con, {
            "openalex_id": row["openalex_id"], "doi": row.get("doi"), "title": row["title"], "author": row.get("author"),
            "university": row.get("university"), "year": row.get("year"), "field_id": row["field_id"],
            "field_name": row["field"], "license": row.get("license"), "score": row.get("score"),
            "pages": row.get("pages"), "holdout": row.get("holdout", False), "meta": {"types": row.get("types", {})}})
        store.replace_thesis_chunks(con, tid, row["field"], built["chunks"])
        n_theses += 1
        n_chunks += len(built["chunks"])
        for ch in built["chapters"]:   # style baseline: bibliography text is not prose either
            units_by_type.setdefault(ch["type"], []).extend(u["text"] for u in ch["units"] if not looks_like_references(u["text"]))
    baseline = build_baseline(units_by_type)
    baseline["n_theses"] = n_theses
    from datetime import datetime, timezone
    baseline["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline_path.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    store.set_meta(con, "fields", fields)
    thin = {t: e["n"] for t, e in baseline["by_section"].items() if t != "all" and e["n"] < 20}
    return {"theses": n_theses, "chunks": n_chunks, "units": baseline["n_units"], "thin_types": thin}


def main() -> None:
    manifest = json.loads((config.DATA / "manifest.json").read_text(encoding="utf-8"))
    fields = json.loads((config.DATA / "fields.selected.json").read_text(encoding="utf-8"))["fields"]
    con = store.connect()
    stats = ingest(manifest, RAW, con, config.DATA / "style_baseline.json", fields)
    print(stats)
    for r in con.execute("SELECT section_type, position, count(*) n FROM chunks GROUP BY 1,2 ORDER BY 1,2"):
        print(f"  {r['section_type']:18} {r['position']:6} {r['n']}")
    if stats["thin_types"]:
        print("WARNING: section types with < 20 baseline units:", stats["thin_types"])


if __name__ == "__main__":
    main()
