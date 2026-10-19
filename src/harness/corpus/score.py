"""Task 1.6: score passing theses, pick 5 exemplars + 1 holdout per field, write data/manifest.json.
    python -m harness.corpus.score [--reselect]      (--reselect: reuse cached rubric scores, honour "exclude")"""
import argparse
import json
import math

from .. import config, llm
from ..spend import Spend
from ..text.jsonparse import parse_json
from .acquire import RAW
from .settings import MIN_WORDS_PER_TYPE

# The exemplar pool ends up thin for these two types even when a field has plenty of passing theses overall,
# because many real theses skip a distinct literature_review/methodology chapter (they fold it into other
# chapters). Bias picking toward theses that DO have one, so the pool grows for the types that need it most.
STRUCTURE_BONUS_TYPES = ["literature_review", "methodology"]

RUBRIC_KEYS = ["rq_clear", "methods_justified", "limitations_stated", "argument_coherent", "claims_hedged"]
SAMPLE = [("introduction", 1200), ("literature_review", 1200), ("methodology", 800), ("discussion", 800)]
RUBRIC_SYSTEM = (
    "You assess the writing quality of excerpts from a doctoral thesis. Score each item from 1 (poor) to 5 (excellent): "
    "rq_clear (the research question or aim is clearly stated), methods_justified (methodological choices are "
    "justified), limitations_stated (limitations are acknowledged), argument_coherent (the argument is coherent and "
    "well structured), claims_hedged (claims are proportionate to the evidence). "
    'Reply with ONLY JSON: {"rq_clear": n, "methods_justified": n, "limitations_stated": n, "argument_coherent": n, "claims_hedged": n}.'
)


def sample_text(built: dict) -> str:
    """First N words of each key chapter type, labelled, for the rubric call."""
    out = []
    for typ, n in SAMPLE:
        words = " ".join(u["text"] for c in built["chapters"] if c["type"] == typ for u in c["units"]).split()
        if words:
            out.append(f"=== {typ.upper()} ===\n" + " ".join(words[:n]))
    return "\n\n".join(out)


def rubric_points(built: dict, spend: Spend | None) -> tuple[float, bool]:
    """(0..25, fallback_used). Two attempts, then a neutral 12.5."""
    text = sample_text(built)
    for _ in range(2):
        data = parse_json(llm.ask("haiku", RUBRIC_SYSTEM, text, max_tokens=200, spend=spend))
        if isinstance(data, dict) and all(isinstance(data.get(k), (int, float)) for k in RUBRIC_KEYS):
            mean = sum(min(5, max(1, data[k])) for k in RUBRIC_KEYS) / len(RUBRIC_KEYS)
            return mean / 5 * 25, False
    return 12.5, True


def pct_rank(value: float, values: list[float]) -> float:
    """Share of the field's candidates at or below `value` (0..1)."""
    return sum(1 for v in values if v <= value) / len(values) if values else 0.0


def composite(meta: dict, rubric: float, inst_pool: list[float], cite_pool: list[float], types: dict | None = None) -> float:
    inst = 25 * pct_rank(math.log1p(meta.get("institution_weight") or 0), inst_pool)
    cit = 20 * pct_rank(meta.get("cited_by") or 0, cite_pool)
    spin = 20 * min(meta.get("spinoffs") or 0, 3) / 3
    lic = 10 if str(meta.get("license") or "").lower().startswith("cc") else 0
    structure = 5 * sum(1 for t in STRUCTURE_BONUS_TYPES if (types or {}).get(t, 0) >= MIN_WORDS_PER_TYPE)
    return round(rubric + inst + cit + spin + lic + structure, 2)


def select(rows: list[dict], per_field: int = 6) -> list[dict]:
    """Top (per_field - 1) exemplars, then one holdout; needs >= 4 to spare a holdout. Input rows carry field, score."""
    out = []
    for f in sorted({r["field_slug"] for r in rows}):
        pool = sorted((r for r in rows if r["field_slug"] == f and not r.get("exclude")), key=lambda r: -r["score"])[:per_field]
        for i, r in enumerate(pool):
            out.append({**r, "holdout": len(pool) >= 4 and i == len(pool) - 1})
        # Keep excluded rows in the manifest (ingest skips them) so the flag survives the next --reselect.
        out += [{**r, "holdout": False} for r in rows if r["field_slug"] == f and r.get("exclude")]
    return out


def _load_passers() -> list[dict]:
    rows = []
    for meta_path in sorted(RAW.glob("*/*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        m["_dir"] = meta_path.parent
        rows.append(m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reselect", action="store_true")
    ap.add_argument("--per-field", type=int, default=6)
    args = ap.parse_args()
    manifest_path = config.DATA / "manifest.json"
    old = {r["openalex_id"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))} if manifest_path.exists() else {}
    spend = Spend(budget_usd=5)
    passers = _load_passers()
    by_field: dict[str, list[dict]] = {}
    for m in passers:
        by_field.setdefault(m["field_slug"], []).append(m)
    scored = []
    for slug, group in by_field.items():
        inst_pool = [math.log1p(m.get("institution_weight") or 0) for m in group]
        cite_pool = [m.get("cited_by") or 0 for m in group]
        for m in group:
            if "rubric" not in m:
                if args.reselect:
                    m["rubric"] = 12.5
                else:
                    built = json.loads((m["_dir"] / "sections.json").read_text(encoding="utf-8"))
                    m["rubric"], m["rubric_fallback"] = rubric_points(built, spend)
                (m["_dir"] / "meta.json").write_text(json.dumps({k: v for k, v in m.items() if k != "_dir"}), encoding="utf-8")
            built = json.loads((m["_dir"] / "sections.json").read_text(encoding="utf-8"))
            scored.append({
                "openalex_id": m["openalex_id"], "field": m["field"], "field_id": m["field_id"], "field_slug": slug,
                "slug": m["_dir"].name, "title": m["title"], "author": m.get("author"), "university": m.get("university"),
                "year": m["year"], "doi": m.get("doi"), "pages": m.get("pages"), "license": m.get("license"),
                "score": composite(m, m["rubric"], inst_pool, cite_pool, built.get("type_words", {})),
                "types": built.get("type_words", {}),
                "exclude": bool(old.get(m["openalex_id"], {}).get("exclude", False)),
            })
    chosen = select(scored, per_field=args.per_field)
    manifest_path.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print(f"{'field':32} {'score':>6} {'H':>2} {'pp':>4}  title")
    for r in chosen:
        print(f"{r['field_slug'][:32]:32} {r['score']:>6.1f} {'*' if r['holdout'] else ' ':>2} {r['pages'] or 0:>4}  {r['title'][:70]}")
    print(f"\n{len(chosen)} theses in data/manifest.json. Review it and set \"exclude\": true where wrong,")
    print("then run `python -m harness.corpus.score --reselect` to backfill. LLM spend:", spend.summary()["total_usd"], "USD")


if __name__ == "__main__":
    main()
