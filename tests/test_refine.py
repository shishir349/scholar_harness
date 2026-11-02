"""Long-text refine stage (harness.refine): chunking, the integrity gate (text/protect.py) and recovery to the original."""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import llm, refine as R
from harness.spend import BudgetExceeded
from harness.text import protect as P

ORIG = ("Lee et al. (2015) measured 120 nodes over 6 months, and the boards missed low frequency vibration. "
        "Bridge decks \"flex under load in ways a bench test cannot show\" (Kim, 2016, p. 4).\n\n"
        "Power draw decides whether a node survives a long winter, so the plan costs about 3.5 W per node (Hassani & Dackermann, 2023). "
        "That may not hold for short spans.")


def fake(replies):
    calls = []

    class M(GenericFakeChatModel):
        def _generate(self, messages, *a, **k):
            calls.append("\n".join(llm.text_of(m) for m in messages))
            return super()._generate(messages, *a, **k)
    m = M(messages=iter([AIMessage(content=t) for t in replies]))
    llm.set_fake(lambda tier: m)
    return calls


@pytest.fixture(autouse=True)
def reset():
    yield
    llm.set_fake(None)


# ── gate ──
def test_gate_passes_identical_and_allows_narrative_to_parenthetical():
    assert P.compare(ORIG, ORIG) == []
    moved = ORIG.replace("Lee et al. (2015) measured", "The team measured").replace("months,", "months (Lee et al., 2015),")
    assert P.compare(ORIG, moved) == []


@pytest.mark.parametrize("bad,why", [
    (lambda t: t.replace("Lee et al. (2015) ", ""), "citation"),
    (lambda t: t.replace("(Kim, 2016, p. 4)", "(Kim, 2017, p. 4)"), "changed"),    # page-number form: the number check catches it
    (lambda t: t.replace("120 nodes", "130 nodes"), "number"),
    (lambda t: t.replace("3.5 W", "3 W"), "number"),
    (lambda t: t.replace("flex under load", "bend under load"), "quote"),
    (lambda t: t.replace("\n\n", " "), "paragraph"),
])
def test_gate_fails_when_a_protected_span_changes(bad, why):
    out = P.compare(ORIG, bad(ORIG))
    assert out and any(why in f for f in out)


def test_gate_guards_figure_blocks_and_headings():
    src = "## 2.1 Sensors\n\nBody text here.\n\n[FIGURE_START]\nTitle: A\n[FIGURE_END]"
    assert P.compare(src, src) == []
    assert P.compare(src, src.replace("Title: A", "Title: B"))
    assert P.compare(src, src.replace("## 2.1 Sensors", "## 2.1 Nodes"))


def test_hedge_loss_is_reported_but_does_not_fail():
    soft = P.soft(ORIG, ORIG.replace("may not hold", "does not hold"))
    assert soft["hedges"] == (1, 0)
    assert P.compare(ORIG, ORIG.replace("may not hold", "does not hold")) == []


# ── chunking ──
def test_chunks_keep_headings_with_next_paragraph_and_never_split_a_paragraph():
    paras = ["## H", ("word " * 120).strip(), ("word " * 120).strip(), ("word " * 120).strip(), ("word " * 500).strip()]
    chunks = R.chunk_paragraphs("\n\n".join(paras), max_words=250)
    assert [len(c) for c in chunks] == [3, 1, 1]                    # heading + 1 para, +1 para(240 words) fits, long para alone
    assert chunks[0][0] == "## H" and all(len(p.split()) in (2, 120, 500) for c in chunks for p in c)


def test_empty_and_single_chunk_documents():
    assert R.chunk_paragraphs("", 300) == [] and len(R.chunk_paragraphs("one short paragraph", 300)) == 1


# ── refine ──
def good(text):
    return text.replace("Power draw decides", "Whether a node survives a long winter comes down to power draw, and")   # keeps all spans


def test_a_valid_rewrite_replaces_the_chunk_and_reports_the_change():
    new = ORIG.replace("measured 120 nodes", "tracked 120 nodes")
    calls = fake([new])
    out = R.refine(ORIG)
    assert out["changed"] and "tracked 120 nodes" in out["text"] and len(calls) == 1
    assert out["chunks"][0]["status"] == "rewritten" and P.compare(ORIG, out["text"]) == []


def test_a_rewrite_that_loses_a_citation_is_retried_once_then_the_original_is_kept():
    bad = ORIG.replace("(Hassani & Dackermann, 2023)", "")
    calls = fake([bad, bad])
    out = R.refine(ORIG)
    assert out["text"].strip() == ORIG.strip() and not out["changed"] and len(calls) == 2
    assert out["chunks"][0]["status"] == "kept_original" and "citation" in " ".join(out["chunks"][0]["failures"])
    assert "citation" in calls[1].lower()                                                  # the retry names what went wrong


def test_the_retry_can_succeed():
    bad = ORIG.replace("120 nodes", "130 nodes")
    ok = ORIG.replace("measured", "tracked")
    fake([bad, ok])
    out = R.refine(ORIG)
    assert out["changed"] and out["chunks"][0]["status"] == "rewritten" and out["chunks"][0]["retried"]


def test_merged_paragraphs_are_rejected():
    fake([ORIG.replace("\n\n", " "), ORIG.replace("\n\n", " ")])
    assert R.refine(ORIG)["chunks"][0]["status"] == "kept_original"


def test_long_documents_are_chunked_and_later_chunks_see_earlier_openings():
    doc = "\n\n".join(f"Paragraph {i} opens with a distinct start and then runs on. " + "filler words here " * 30 for i in range(6))
    calls = fake([p for p in [doc]] * 0 or [doc.split("\n\n")[0] + "\n\n" + doc.split("\n\n")[1]] * 3)
    out = R.refine(doc, max_words=250)
    assert len(out["chunks"]) == 3 and len(calls) >= 3
    assert "Paragraph 0 opens" in calls[1] or "Paragraph 0" in calls[1]                    # openings so far are passed forward


def test_nothing_to_rewrite_makes_no_call():
    calls = fake([])
    assert R.refine("")["changed"] is False and R.refine("## Only a heading")["text"] == "## Only a heading" and calls == []


def test_budget_stop_propagates_and_other_errors_keep_the_original():
    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise BudgetExceeded("over")
    llm.set_fake(lambda tier: Boom(messages=iter([])))
    with pytest.raises(BudgetExceeded):
        R.refine(ORIG)

    class Down(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("api down")
    llm.set_fake(lambda tier: Down(messages=iter([])))
    out = R.refine(ORIG)
    assert out["text"].strip() == ORIG.strip() and out["chunks"][0]["status"] == "error"


def test_refine_is_a_no_op_on_its_own_output_when_the_model_returns_it_unchanged():
    fake([ORIG, ORIG])
    out = R.refine(ORIG)
    assert out["changed"] is False


def test_curly_and_straight_quotes_are_the_same_to_the_gate():
    a = "Kahn's (1990) work held. \"flex under load\" was said."
    b = "Kahn\u2019s (1990) work held. \u201cflex under load\u201d was said."
    assert P.compare(a, b) == [] and P.compare(b, a) == []


def test_grouped_citations_are_all_counted():
    a = "Both hold (Azhdari et al., 2018; Boonjob & Lee, 2021)."
    assert P.compare(a, a) == []
    assert P.compare(a, a.replace("Boonjob & Lee, 2021", "Boonjob & Lee, 2020"))
    assert P.compare(a, "Azhdari et al. (2018) and Boonjob and Lee (2021) agree.") == []
