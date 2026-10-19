import json
import math

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import llm
from harness.corpus import acquire, candidates, ingest, score, store, topics
from harness.corpus.retrieve import get_exemplars


# ── topics ───────────────────────────────────────────────────────────────────
def test_select_fields_orders_computes_coverage_and_drops_unknown():
    groups = [{"key": "https://openalex.org/fields/11", "key_display_name": "Medicine", "count": 500},
              {"key": "https://openalex.org/fields/17", "key_display_name": "Computer Science", "count": 300},
              {"key": "unknown", "key_display_name": "unknown", "count": 9999},
              {"key": "https://openalex.org/fields/33", "key_display_name": "Social Sciences", "count": 200}]
    sel = topics.select_fields(groups, n=2)
    assert [f["name"] for f in sel["fields"]] == ["Medicine", "Computer Science"]
    assert sel["fields"][0]["id"] == "11" and sel["fields"][0]["slug"] == "medicine"
    assert sel["total_counted"] == 1000 and sel["coverage_top_n"] == 0.8


# ── candidates ───────────────────────────────────────────────────────────────
WORK = {"id": "https://openalex.org/W5", "doi": "https://doi.org/10.1/x", "title": "A thesis", "publication_year": 2019,
        "authorships": [{"author": {"display_name": "Ann Lee", "id": "https://openalex.org/A9"},
                         "institutions": [{"display_name": "Uni X", "id": "https://openalex.org/I3"}]}],
        "best_oa_location": {"pdf_url": "https://x/a.pdf", "license": "cc-by"},
        "locations": [{"pdf_url": "https://x/a.pdf"}, {"pdf_url": "https://y/b.pdf"}, {"pdf_url": None}],
        "cited_by_count": 7, "primary_topic": {"display_name": "T", "subfield": {"display_name": "S"}}}


def test_candidate_from_work_and_rejections():
    c = candidates.candidate_from_work(WORK)
    assert c["openalex_id"] == "W5" and c["pdf_urls"] == ["https://x/a.pdf", "https://y/b.pdf"]
    assert c["institution_id"] == "I3" and c["author_id"] == "A9" and c["license"] == "cc-by" and c["doi"] == "10.1/x"
    assert candidates.candidate_from_work({**WORK, "is_retracted": True}) is None
    assert candidates.candidate_from_work({**WORK, "best_oa_location": None, "locations": []}) is None


def test_spinoff_count():
    arts = ["Digital adoption in small firms", "Unrelated cooking paper here", "Small firms digital adoption revisited"]
    assert candidates.count_spinoffs("Digital adoption in small firms: a thesis", arts) == 2


# ── acquire gate ─────────────────────────────────────────────────────────────
def _para(n):
    return " ".join(f"w{i}" for i in range(n)) + "."


def good_md():
    parts = []
    for i, t in enumerate(["Introduction", "Literature Review", "Research Methodology", "Results and Discussion", "Conclusion"], 1):
        parts += [f"# Chapter {i} {t}", _para(900), ""]
    return "\n".join(parts)


def test_gate_pass_and_each_failure_reason():
    ok, reason, built = acquire.gate(good_md(), pages=120, chars=120 * 2500)
    assert ok and built["type_words"]["introduction"] >= 800 and built["chunks"]
    assert not acquire.gate(good_md(), 20, 999999)[0] and "pages" in acquire.gate(good_md(), 20, 999999)[1]
    assert "scan" in acquire.gate(good_md(), 120, 120 * 100)[1]
    thin = "# Chapter 1 Introduction\n" + _para(900)
    assert "only 1 chapter types" in acquire.gate(thin, 120, 120 * 2500)[1]


# ── score ────────────────────────────────────────────────────────────────────
def test_pct_rank_and_composite_maths():
    assert score.pct_rank(5, [1, 5, 9, 10]) == 0.5 and score.pct_rank(5, []) == 0.0
    meta = {"institution_weight": 99, "cited_by": 10, "spinoffs": 5, "license": "CC-BY"}
    inst_pool = [math.log1p(x) for x in (0, 9, 99)]
    val = score.composite(meta, 20.0, inst_pool, [1, 5, 10])
    assert val == pytest.approx(20 + 25 + 20 + 20 + 10)                       # top of every pool, spinoffs capped at 3
    assert score.composite({}, 0.0, [0.0], [0]) == pytest.approx(25 * 1 + 20 * 1)  # single-candidate pools rank 1.0


def test_composite_rewards_having_a_real_lit_review_or_methodology_chapter():
    base = score.composite({}, 0.0, [0.0], [0])
    thin = score.composite({}, 0.0, [0.0], [0], {"literature_review": 50})            # under the word bar: no bonus
    one = score.composite({}, 0.0, [0.0], [0], {"literature_review": 900})
    both = score.composite({}, 0.0, [0.0], [0], {"literature_review": 900, "methodology": 900})
    assert thin == base and one == base + 5 and both == base + 10


def test_select_top_five_plus_holdout_and_exclusions():
    rows = [{"field_slug": "a", "score": s, "openalex_id": f"a{s}", "exclude": s == 90} for s in (90, 80, 70, 60, 50, 40, 30)]
    rows += [{"field_slug": "b", "score": s, "openalex_id": f"b{s}"} for s in (10, 20)]
    out = score.select(rows)
    a = [r for r in out if r["field_slug"] == "a" and not r.get("exclude")]
    assert [r["score"] for r in a] == [80, 70, 60, 50, 40, 30] and [r["holdout"] for r in a] == [False] * 5 + [True]
    assert [r["score"] for r in out if r.get("exclude")] == [90]               # kept in the manifest, flagged, never a holdout
    assert not any(r["holdout"] for r in out if r["field_slug"] == "b")        # < 4 theses: keep all as exemplars


def test_rubric_points_parses_retries_and_falls_back():
    built = {"chapters": [{"type": "introduction", "units": [{"text": "word " * 2000}]}]}
    assert "=== INTRODUCTION ===" in score.sample_text(built) and len(score.sample_text(built).split()) < 1300

    def fake(*texts):
        m = GenericFakeChatModel(messages=iter([AIMessage(content=t) for t in texts]))
        llm.set_fake(lambda tier: m)
    try:
        fake('{"rq_clear":5,"methods_justified":5,"limitations_stated":5,"argument_coherent":5,"claims_hedged":5}')
        assert score.rubric_points(built, None) == (25.0, False)
        fake("garbage", '{"rq_clear":1,"methods_justified":1,"limitations_stated":1,"argument_coherent":1,"claims_hedged":1}')
        assert score.rubric_points(built, None) == (5.0, False)
        fake("garbage", "still garbage")
        assert score.rubric_points(built, None) == (12.5, True)
    finally:
        llm.set_fake(None)


# ── ingest ───────────────────────────────────────────────────────────────────
def test_ingest_end_to_end(tmp_path):
    ok, _, built = acquire.gate(good_md(), 120, 120 * 2500)
    raw = tmp_path / "raw"
    manifest = []
    for i in range(3):
        d = raw / "cs" / f"thesis-{i}"
        d.mkdir(parents=True)
        (d / "sections.json").write_text(json.dumps(built), encoding="utf-8")
        manifest.append({"openalex_id": f"W{i}", "field": "Computer Science", "field_id": "17", "field_slug": "cs",
                         "slug": f"thesis-{i}", "title": f"T{i}", "score": 50 + i, "holdout": i == 2, "exclude": i == 1})
    con = store.connect(":memory:")
    bp = tmp_path / "baseline.json"
    stats = ingest.ingest(manifest, raw, con, bp, [{"name": "Computer Science", "keywords": ["software"]}])
    assert stats["theses"] == 2 and stats["chunks"] == 2 * len(built["chunks"])          # excluded thesis skipped
    baseline = json.loads(bp.read_text())
    assert baseline["n_theses"] == 2 and "all" in baseline["by_section"] and "introduction" in baseline["by_section"]
    assert store.get_meta(con, "fields")[0]["name"] == "Computer Science"
    # holdout excluded from retrieval, and re-ingesting replaces rather than duplicates
    ingest.ingest(manifest, raw, con, bp, [])
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == stats["chunks"]
    got = get_exemplars(con=con, fields=[], title="x", domain="", chapter_title="", section_title="",
                        section_type="introduction", position="open", limit=5)
    holdout_id = con.execute("SELECT id FROM theses WHERE holdout=1").fetchone()[0]
    assert got and all(con.execute("SELECT thesis_id FROM chunks WHERE id=?", (g["id"],)).fetchone()[0] != holdout_id for g in got)
