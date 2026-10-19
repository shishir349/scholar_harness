"""The free wording rewrites in text/cleanup.polish: each one hits its pattern, leaves a miss alone, keeps figures and
citations, and gives the same text when run twice."""
import pytest

from harness.text import cleanup as C


def _c(open_="", mid="", close=""):
    return {"open": open_, "mid": mid, "close": close}


def _one(fn, text, *a):
    out, n = fn(_c(text), *a)
    return out["open"], n


def test_stance_is_stripped_and_the_claim_kept_with_a_capital():
    t, n = _one(C.strip_stance, "I argue that power draw decides survival. The data agree. In the end, I would argue that costs matter more.")
    assert t == "Power draw decides survival. The data agree. In the end, costs matter more." and n == 2
    t, _ = _one(C.strip_stance, "This study takes the position that field tests beat lab tests. Next point here.")
    assert t.startswith("Field tests beat lab tests.")


@pytest.mark.parametrize("text", ["I argued about it for years.", "Some argue that nodes drift.", "He argues that nodes drift.", "# I argue that"])
def test_stance_leaves_other_uses_alone(text):
    assert _one(C.strip_stance, text) == (text, 0)


def test_what_cleft_becomes_a_plain_statement_only_when_a_long_clause_follows():
    t, n = _one(C.what_cleft, "What the sources show together, read against one another, is that platforms are not converging on one standard. Next sentence follows.")
    assert t.startswith("Platforms are not converging on one standard.") and n == 1
    for text in ("What matters here is the sampling rate of the board.", "What is that? Nobody knows for sure about this thing.",
                 "What the sources show is that nodes fail."):          # no 'is that' / clause under six words
        assert _one(C.what_cleft, text) == (text, 0)


def test_report_lines_become_parentheticals_and_according_to_without_losing_the_citation():
    text = ("Lee et al. (2015) show that cheap boards miss the low frequency vibration of long spans. "
            "Smith and Kim (2019) note that nodes drift over time in the field. "
            "Yu (2020) argues that gateways fail first under heavy winter load.")
    out, n = _one(C.report_to_cite, text)
    assert n == 3 and out.count("According to") == 2
    assert "According to Lee et al. (2015), cheap boards" in out and "Nodes drift over time in the field (Smith & Kim, 2019)." in out
    assert "(2020)" in out or "Yu, 2020" in out
    again, n2 = _one(C.report_to_cite, out)
    assert again == out and n2 == 0


@pytest.mark.parametrize("text", ["Lee et al. (2015) trace the development of boards.", "Lee (2015) show that x.",
                                  "As Lee et al. (2015) show that nodes drift, the point stands here for now."])
def test_report_leaves_sentences_it_cannot_convert_cleanly(text):
    assert _one(C.report_to_cite, text) == (text, 0)


def test_closer_is_dropped_only_when_plain_and_two_sentences_remain():
    body = "Nodes drift over a long winter. Power draw decides whether a node lives. Taken together, power matters most."
    assert _one(C.drop_closers, body) == ("Nodes drift over a long winter. Power draw decides whether a node lives.", 1)
    assert _one(C.drop_closers, "Nodes drift. Taken together, power matters most.")[1] == 0            # would leave one sentence
    assert _one(C.drop_closers, "A one. A two. Overall, 94% of nodes survived (Lee, 2015).")[1] == 0      # figure and citation stay
    assert _one(C.drop_closers, "A one. A two. Overall, 94% of nodes survived.")[1] == 0


def test_boosters_go_but_the_sentence_survives():
    t, n = _one(C.drop_boosters, "Clearly, the nodes genuinely fail. The gap is fundamentally about power, and deeply felt.")
    assert t == "The nodes fail. The gap is fundamentally about power, and deeply felt." and n == 1      # adjective uses ('deeply felt') stay
    assert _one(C.drop_boosters, "Clearly defined boundaries matter. Deeply held beliefs shape this.")[1] == 0
    assert _one(C.drop_boosters, "The result was significant. Nodes fail, genuinely, in winter.")[0].endswith("Nodes fail, in winter.")


def test_that_is_thinned_only_after_a_saying_verb_and_only_above_the_limit():
    text = ("Lee found that the nodes drift. Kim shows that these boards fail. Wu noted that it rains. Yu argues that they leave. "
            "The idea that guides this work is simple.") * 3
    out, n = C.thin_that(_c(text), per_1000_limit=1)
    assert n >= 1 and out["open"].count("idea that guides") == 3                      # relative-clause 'that' is never touched
    assert C.thin_that(_c(text), per_1000_limit=500) == (_c(text), 0)
    assert C.thin_that(_c(""), 1) == (_c(""), 0)


def test_polish_is_idempotent_keeps_headings_figures_and_citations_and_handles_empty_chunks():
    text = ("# 2.1 Sensors\n\nI argue that 94% of nodes survived (Lee, 2015). Lee et al. (2015) show that cheap boards miss vibration in long spans. "
            "Clearly, field tests matter. Taken together, power matters most.\n\nA short second paragraph stays. It has two sentences.")
    chunks = _c(text, "", "")
    once, counts = C.polish(chunks, that_limit=10)
    twice, again = C.polish(once, that_limit=10)
    assert once == twice and not any(again.values())
    assert once["open"].startswith("# 2.1 Sensors\n\n") and "94% of nodes survived (Lee, 2015)" in once["open"]
    assert once["mid"] == "" and counts["stance"] == 1 and counts["booster"] == 1


def test_crlf_whitespace_and_multiline_paragraphs_survive():
    text = "- bullet one\n- bullet two\n\nI argue that  nodes\ndrift over time in the field. Another sentence closes it."
    out, n = _one(C.strip_stance, text)
    assert n == 1 and out.startswith("- bullet one\n- bullet two\n\n") and "Nodes" in out


# ── review findings ──
def test_what_cleft_never_deletes_a_citation_or_a_figure_from_its_prefix():
    for text in ("What Lee (2015) found is that wear rises with the applied load in all tests.",
                 "What 94% of respondents reported was that the sensors drifted after a week of use."):
        assert _one(C.what_cleft, text) == (text, 0)


def test_closer_words_need_a_word_boundary():
    text = "A is one thing here. B is two things here. In summer 2019 the sensors failed badly."
    assert _one(C.drop_closers, text)[1] == 0 and not C.repeat_flags(_c(text))["closer"]


def test_thin_that_is_idempotent_and_leaves_ambiguous_cases():
    text = " ".join(["Lee found that the nodes drift. Kim shows that this approach fails. Wu found that a threshold of 5 was exceeded."] * 4)
    once, n = C.thin_that(_c(text), 5)
    twice, n2 = C.thin_that(once, 5)
    assert n >= 1 and n2 == 0 and twice == once
    assert "shows that this approach" in once["open"] and "found that a threshold" in once["open"]


def test_report_with_a_trailing_citation_does_not_stack_parentheses():
    out, n = _one(C.report_to_cite, "Lee (2015) argues that wear rises with load in every test run (Kim, 2016).")
    assert n == 1 and out.startswith("According to Lee (2015), wear rises") and out.count("(") == 2
