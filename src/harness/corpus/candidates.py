"""Task 1.3: candidate dissertations per field, with quality signals.
    python -m harness.corpus.candidates [--field slug] [--candidates-max N]"""
import argparse
import json
from collections import Counter

from .. import config
from ..text.similarity import title_similarity
from . import openalex as oa
from .settings import CANDIDATES_MAX, DISSERTATION_FILTER, WORK_SELECT

CAND_DIR = config.DATA / "candidates"


def pdf_urls(work: dict) -> list[str]:
    urls = []
    for loc in [work.get("best_oa_location"), *(work.get("locations") or [])]:
        u = (loc or {}).get("pdf_url")
        if u and u not in urls:
            urls.append(u)
    return urls


def candidate_from_work(w: dict) -> dict | None:
    """None when unusable: retracted, no title/year, or no PDF link."""
    urls = pdf_urls(w)
    if w.get("is_retracted") or not urls or not w.get("title") or not w.get("publication_year"):
        return None
    auth = (w.get("authorships") or [{}])[0]
    inst = (auth.get("institutions") or [{}])[0]
    topic = w.get("primary_topic") or {}
    return {
        "openalex_id": oa.short_id(w.get("id")), "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": w["title"], "year": w["publication_year"], "author": (auth.get("author") or {}).get("display_name"),
        "author_id": oa.short_id((auth.get("author") or {}).get("id")),
        "university": inst.get("display_name"), "institution_id": oa.short_id(inst.get("id")),
        "license": (w.get("best_oa_location") or {}).get("license"), "cited_by": w.get("cited_by_count") or 0,
        "pdf_urls": urls, "subfield": (topic.get("subfield") or {}).get("display_name"), "topic": topic.get("display_name"),
    }


def count_spinoffs(thesis_title: str, article_titles: list[str], threshold: float = 0.5) -> int:
    return sum(1 for t in article_titles if (title_similarity(thesis_title, t) or 0) >= threshold)


def _institution_weight(inst_id: str | None, cache: dict) -> int:
    if not inst_id:
        return 0
    if inst_id not in cache:
        try:
            cache[inst_id] = oa.get(f"/institutions/{inst_id}", {"select": "id,cited_by_count"}).get("cited_by_count") or 0
        except oa.OpenAlexError:
            cache[inst_id] = 0
    return cache[inst_id]


def gather_field(field: dict, cap: int, inst_cache: dict) -> list[dict]:
    out: list[dict] = []
    for page in (1, 2, 3):
        data = oa.get("/works", {"filter": f"{DISSERTATION_FILTER},primary_topic.field.id:{field['id']}",
                                 "sort": "cited_by_count:desc", "per-page": 50, "page": page, "select": WORK_SELECT})
        rows = data.get("results", [])
        out += [c for c in (candidate_from_work(w) for w in rows) if c]
        if len(out) >= cap or len(rows) < 50:
            break
    out = out[:cap]
    for c in out:
        c["institution_weight"] = _institution_weight(c["institution_id"], inst_cache)
        c["spinoffs"] = 0
        if c["author_id"]:
            try:
                arts = oa.get("/works", {"filter": f"authorships.author.id:{c['author_id']},type:article,"
                                                   f"from_publication_date:{c['year'] - 1}-01-01",
                                         "select": "title", "per-page": 25}).get("results", [])
                c["spinoffs"] = count_spinoffs(c["title"], [a.get("title") or "" for a in arts])
            except oa.OpenAlexError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field")
    ap.add_argument("--candidates-max", type=int, default=CANDIDATES_MAX)
    args = ap.parse_args()
    sel_path = config.DATA / "fields.selected.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    inst_path = CAND_DIR / "_institutions.json"
    inst_cache = json.loads(inst_path.read_text()) if inst_path.exists() else {}
    for f in sel["fields"]:
        if args.field and f["slug"] != args.field:
            continue
        cands = gather_field(f, args.candidates_max, inst_cache)
        (CAND_DIR / f"{f['slug']}.json").write_text(json.dumps(cands, indent=1), encoding="utf-8")
        f["keywords"] = [k for k, _ in Counter(x for c in cands for x in (c["subfield"], c["topic"]) if x).most_common(60)]
        print(f"{f['slug']:45} {len(cands):3} candidates")
        if len(cands) < 20:
            print(f"  WARNING: only {len(cands)} for {f['name']}: rerun with a larger --candidates-max")
        inst_path.write_text(json.dumps(inst_cache), encoding="utf-8")
        sel_path.write_text(json.dumps(sel, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
