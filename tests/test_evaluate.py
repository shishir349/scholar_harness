import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import evaluate as E
from harness import llm
from harness.corpus import store

STYLE = "style_score: full not much worse than baseline, most fixtures (>=50%, slack -5)"
H2H = "head-to-head: full reads more human than baseline (>= 0.60, ties = half)"
CITES = "unmatched citations at least halved vs baseline"
STRUCT = "structure: full not much worse than baseline, most fixtures (>=50%, slack -0.15)"


def row(arm, fixture="f1", **kw):
    base = dict(fixture=fixture, arm=arm, section_type="introduction", words=500, style_score=60.0, style_passed=True,
                structure_pass_rate=0.5, judge="tie", unmatched=0, checked=4, unsupported=0, overlap_flagged=False,
                revisions=0, usd=0.1)
    base.update(kw)
    return base


def good_rows():
    """Two paired fixtures (f1, f2): full beats or matches baseline on both, so every criterion passes."""
    return [row("baseline", "f1", style_score=50, judge="lose", unmatched=3),
            row("baseline", "f2", style_score=54, judge="tie", unmatched=2),
            row("full", "f1", style_score=62, judge="win", h2h="win", structure_pass_rate=0.7),
            row("full", "f2", style_score=60, judge="win", h2h="win")]


def test_summarise_metrics():
    s = E.summarise(good_rows())
    assert s["baseline"]["style_mean"] == 52 and s["full"]["style_mean"] == 61
    assert s["full"]["win_rate"] == 1.0 and s["baseline"]["lose_rate"] == 0.5
    assert s["full"]["h2h_score"] == 1.0 and s["baseline"]["h2h_score"] is None
    assert s["baseline"]["unmatched_total"] == 5 and s["full"]["unmatched_total"] == 0
    assert s["full"]["usd_total"] == pytest.approx(0.2)
    assert s["full"]["style_not_worse_rate"] == 1.0 and s["baseline"]["style_not_worse_rate"] is None


def test_all_criteria_pass_on_a_clear_win():
    crit = E.evaluate(E.summarise(good_rows()))
    assert len(crit) == 6 and all(c["passed"] for c in crit), crit


@pytest.mark.parametrize("mutation, failing", [
    (dict(style_score=40), STYLE),                               # -10 and -14 vs baseline: past the -5 slack, both fixtures
    (dict(h2h="lose"), H2H),
    (dict(unmatched=3), CITES),                                  # 6 vs baseline 5: not halved
    (dict(unsupported=1), "unsupported rate <= 5%"),
    (dict(overlap_flagged=True), "no overlap flags (full)"),
    (dict(structure_pass_rate=0.1), STRUCT),                     # -0.4 vs baseline on both fixtures: past the -0.15 slack
])
def test_each_criterion_can_fail(mutation, failing):
    rows = good_rows()
    rows[2] = {**rows[2], **mutation}
    rows[3] = {**rows[3], **mutation}
    crit = {c["name"]: c["passed"] for c in E.evaluate(E.summarise(rows))}
    assert crit[failing] is False


def test_one_bad_fixture_does_not_flip_the_verdict():
    """The pooled mean would fail this (one fixture tanks it), but the paired majority-vote doesn't."""
    rows = good_rows()
    rows[2] = {**rows[2], "style_score": 5}                      # f1: 5 vs baseline 50, a huge single-fixture drop
    s = E.summarise(rows)
    assert s["full"]["style_mean"] < s["baseline"]["style_mean"]  # the naive pooled-mean view looks like a regression
    assert {c["name"]: c["passed"] for c in E.evaluate(s)}[STYLE] is True  # f2 (60 vs 54) still carries the majority


def test_the_old_saturating_criteria_no_longer_decide_the_outcome():
    """Both arms beating a human excerpt every time (win_rate 1.0 vs 1.0) must not fail G3 by itself,
    and a style score a little under baseline is no regression."""
    rows = good_rows()
    rows[0] = {**rows[0], "judge": "win"}
    rows[1] = {**rows[1], "judge": "win"}
    rows[2] = {**rows[2], "style_score": 51}
    rows[3] = {**rows[3], "style_score": 51}
    crit = {c["name"]: c["passed"] for c in E.evaluate(E.summarise(rows))}
    assert crit[STYLE] and all(crit.values())


def test_head_to_head_counts_ties_as_half():
    rows = good_rows()
    rows[2] = {**rows[2], "h2h": "win"}
    rows[3] = {**rows[3], "h2h": "tie"}
    assert E.summarise(rows)["full"]["h2h_score"] == 0.75
    rows[2] = {**rows[2], "h2h": "tie"}
    s = E.summarise(rows)
    assert s["full"]["h2h_score"] == 0.5 and not {c["name"]: c["passed"] for c in E.evaluate(s)}[H2H]


def test_baseline_with_no_citation_problems_needs_full_to_have_none():
    rows = [r for r in good_rows()]
    rows[0] = {**rows[0], "unmatched": 0}
    rows[1] = {**rows[1], "unmatched": 0}
    assert {c["name"]: c["passed"] for c in E.evaluate(E.summarise(rows))}[CITES] is True
    rows[2] = {**rows[2], "unmatched": 1}
    assert {c["name"]: c["passed"] for c in E.evaluate(E.summarise(rows))}[CITES] is False


def test_missing_metrics_fail_rather_than_pass():
    rows = [row("baseline", style_score=None, judge=None), row("full", style_score=None, judge=None)]
    crit = {c["name"]: c["passed"] for c in E.evaluate(E.summarise(rows))}
    assert crit[STYLE] is False and crit[H2H] is False


def test_report_marks_overall_and_warns_on_tiny_samples():
    s = E.summarise(good_rows())
    md = E.report_md(good_rows(), s, E.evaluate(s))
    assert "**Overall: PASS**" in md and "Automated proxies only" in md and "h2h_score" in md
    assert "style_not_worse_rate" in md and "structure_not_worse_rate" in md
    assert "Only 2 fixture(s)" in md                              # good_rows has 2 per arm


def test_full_run_with_fake_models_end_to_end():
    """Two fixtures, scripted models: proves the harness wiring (research -> both arms -> judge -> h2h -> summary)."""
    from langchain_core.messages import AIMessage as A

    class M(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    text = " ".join(["Small firms adopted cloud accounting software during recent years.",   # distinct sentences: repeats are deleted
                     "Retail owners reported faster invoicing after the migration finished.",
                     "Manufacturers struggled with legacy machinery connected to modern dashboards.",
                     "Training budgets differed widely between urban and rural enterprises.",
                     "Regulators published guidance covering payment data and customer privacy.",
                     "Several interviewees mentioned unreliable broadband as the main obstacle.",
                     "Consultants recommended gradual rollouts with clear ownership inside teams.",
                     "Survey respondents rated support quality higher than software price."]) + " "
    sonnet = M(messages=itertools.cycle([A(content=text)]))
    haiku = M(messages=itertools.cycle([A(content='{"human": "A", "verdict": "x"}')]))
    llm.set_fake(lambda tier: sonnet if tier == "sonnet" else haiku)
    try:
        con = store.connect(":memory:")
        tid = store.upsert_thesis(con, dict(openalex_id="W1", title="T", field_id="1", field_name="Business", score=9, holdout=1))
        store.insert_chunks(con, tid, "Business", [dict(section_type="introduction", position="open", heading="H", chunk_idx=0, text="holdout reference text " * 30)])
        con.commit()
        fx = {"id": "f1", "thesis": {"title": "Digital tools in firms", "domain": "Business", "methodology": "Survey",
                                     "geographic_focus": "India", "university": "U", "programme": "P"},
              "chapter": {"chapter_number": 1, "title": "Introduction", "sections": [{"number": "1.1", "title": "Background"}]}}
        rows = E.run_ab([fx], con=con, fields=[], search_fn=lambda q, y, n: [], baseline={})
        assert [r["arm"] for r in rows] == ["baseline", "full"]
        assert all(r["words"] > 100 and r["judge"] in ("win", "lose", "tie") and r["text"] for r in rows)
        assert rows[0].get("h2h") is None and rows[1]["h2h"] in ("win", "lose", "tie")
        s = E.summarise(rows)
        assert s["baseline"]["n"] == 1 and s["full"]["h2h_score"] is not None and E.evaluate(s)
    finally:
        llm.set_fake(None)


def test_clip_ends_on_a_full_sentence_for_the_judge():
    from harness.evaluate import _clip
    text = ("Firms grew quickly. " * 100) + "This last sentence has no ending and gets cut off here"
    out = _clip(text)
    assert out.endswith(".") and len(out.split()) <= 450
    assert _clip("Short one. Another that stops") == "Short one. Another that stops"       # never chops most of a short text
