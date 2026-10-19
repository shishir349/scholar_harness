"""Live smoke test: `python -m harness.corpus.smoke`. Confirms the OpenAlex params this repo
relies on are accepted and the response shapes match what the code reads."""
import sys

from . import openalex as oa

DISS = "type:dissertation,open_access.is_oa:true,from_publication_date:2015-01-01,language:en"
WORK_SELECT = ("id,doi,title,publication_year,authorships,open_access,best_oa_location,locations,cited_by_count,"
               "primary_topic,is_retracted,abstract_inverted_index")


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main() -> None:
    check(oa.enabled(), "OPENALEX_API_KEY present")

    g = oa.get("/works", {"filter": DISS, "group_by": "primary_topic.field.id"})
    groups = g.get("group_by", [])
    check(bool(groups) and {"key", "key_display_name", "count"} <= set(groups[0]),
          f"group_by fields ({len(groups)} groups, top: {groups[0]['key_display_name']} {groups[0]['count']})")
    field_id = oa.short_id(groups[0]["key"])

    p = oa.get("/works", {"filter": f"{DISS},primary_topic.field.id:{field_id}", "sort": "cited_by_count:desc",
                          "per-page": 5, "select": WORK_SELECT})
    res = p.get("results", [])
    check(bool(res), f"dissertation page for field {field_id} ({len(res)} results)")
    w = res[0]
    check({"id", "title", "publication_year", "authorships", "best_oa_location", "locations", "cited_by_count"} <= set(w),
          "dissertation work has expected keys")
    check("pdf_url" in (w.get("best_oa_location") or {"pdf_url": None}), "best_oa_location.pdf_url readable")
    inst = next((i for a in w.get("authorships", []) for i in a.get("institutions", [])), None)
    if inst:
        ent = oa.get(f"/institutions/{oa.short_id(inst['id'])}", {"select": "id,display_name,cited_by_count"})
        check("cited_by_count" in ent, f"institution entity ({ent.get('display_name')})")

    refs = oa.search_works("self-efficacy academic performance", from_year=2015, limit=3)
    check(bool(refs) and all(r["apa"] and r["openalex_id"] for r in refs), f"search_works ({len(refs)} refs)")
    print("sample APA:", refs[0]["apa"])
    print("sample abstract:", (refs[0]["abstract"] or "")[:120])
    print("SMOKE OK")


if __name__ == "__main__":
    main()
