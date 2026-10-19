"""Task 1.2: pick the top OpenAlex fields by open-access dissertation count.
    python -m harness.corpus.topics
Writes data/fields.selected.json; prints the table and the coverage the selection gives."""
import json
from datetime import datetime, timezone

from .. import config
from . import openalex as oa
from .settings import DISSERTATION_FILTER, FIELDS_TARGET, slugify


def select_fields(groups: list[dict], n: int = FIELDS_TARGET) -> dict:
    rows = [g for g in groups if g.get("key") and "unknown" not in str(g["key"]).lower()]
    rows.sort(key=lambda g: -g["count"])
    total = sum(g["count"] for g in rows) or 1
    fields, cum = [], 0
    for g in rows[:n]:
        cum += g["count"]
        fields.append({"id": oa.short_id(g["key"]), "name": g["key_display_name"], "count": g["count"],
                       "share": round(g["count"] / total, 4), "slug": slugify(g["key_display_name"]), "keywords": []})
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "total_counted": total,
            "coverage_top_n": round(cum / total, 4), "n": len(fields), "fields": fields}


def main() -> None:
    data = oa.get("/works", {"filter": DISSERTATION_FILTER, "group_by": "primary_topic.field.id"})
    sel = select_fields(data.get("group_by", []))
    out = config.DATA / "fields.selected.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    cum = 0.0
    print(f"{'#':>2}  {'field':45} {'count':>9} {'share':>7} {'cum':>7}")
    for i, f in enumerate(sel["fields"], 1):
        cum += f["share"]
        print(f"{i:>2}  {f['name'][:45]:45} {f['count']:>9} {f['share']:>7.1%} {cum:>7.1%}")
    print(f"\ncoverage of top {sel['n']} fields: {sel['coverage_top_n']:.1%} of {sel['total_counted']} open-access dissertations")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
