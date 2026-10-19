"""Number, first-person and example shares (text/numbers.py), the facts the harness may use, and how they reach the plan and the patch."""
import re

import pytest

from harness import graph as G
from harness.text import numbers as N
from tests.test_graph import PROMPTS, models, request

def _w(n):                                      # distinct made-up content words, so no two paragraphs restate each other
    return "".join(chr(97 + (n * 7 // 26 ** k) % 26) for k in range(5)) + "ic"


def _paras(n, fig=0, fp=0, ex=0):
    out = []
    for i in range(n):
        w = [_w(i * 12 + j) for j in range(12)]
        p = (f"The {w[0]} {w[1]} of {w[2]} {w[3]} was {w[4]} by the {w[5]} {w[6]} for every {w[7]} and for each {w[8]} under {w[9]} "
             f"with {w[10]} {w[11]} {w[0]}s. " + " ".join(_w(500 + i * 30 + j) for j in range(30)) + ".")
        if i < fig:
            p += " The panel held 120 firms."
        if n - 1 - i < fp:
            p += " I chose the panel myself."
        if i < ex:
            p += " For example, a small plant closes first."
        out.append(p)
    return out


def _chunks(paras):
    k = max(1, len(paras) // 3)
    return {"open": "\n\n".join(paras[:k]), "mid": "\n\n".join(paras[k:2 * k]), "close": "\n\n".join(paras[2 * k:])}


def test_body_paras_skip_headings_and_short_paragraphs():
    ch = {"open": "# Heading\n\nToo short to count.\n\n" + _paras(1)[0], "mid": "", "close": ""}
    assert [i for _, i, _ in N.body_paras(ch)] == [2]


@pytest.mark.parametrize("have,want", [(0, True), (1, True), (3, False), (5, False)])
def test_number_gap_asks_for_figures_only_below_70_percent_of_the_target(have, want):
    gap = N.number_gap(_chunks(_paras(10, fig=have)), "introduction")        # target 38%: 70% of it is 2.7 paragraphs of 10
    assert bool(gap) is want and gap <= 3


def test_number_gap_never_judges_a_draft_of_under_five_paragraphs():
    assert N.number_gap(_chunks(_paras(4)), "introduction") == 0 and N.number_gap(_chunks([]), "introduction") == 0


def test_a_year_a_citation_year_and_a_bracket_number_are_not_figures():
    assert not N.has_figure("Lee et al. (2015a) found it in 2019 [12, 13] and again in 1999.")
    assert N.has_figure("Research question 2 asks about it.") and N.has_figure("It fell 3.5 mm.")


def test_first_person_gap_follows_the_section_type():
    assert N.fp_gap(_chunks(_paras(10)), "introduction") == 2                  # capped at 2
    assert N.fp_gap(_chunks(_paras(10, fp=3)), "introduction") == 0
    assert N.fp_gap(_chunks(_paras(10)), "literature_review") == 1             # real share is only 7%
    assert N.has_first_person("I chose it.") and N.has_first_person("Our panel held.") and not N.has_first_person("The panel held, as a user said.")


def test_example_gap_needs_a_long_enough_draft_and_counts_for_example_such_as_and_eg():
    short = _chunks(_paras(5))
    assert N.example_gap(short) == 0                                            # under 600 words
    long = _chunks(_paras(20))
    assert N.example_gap(long) == 2
    assert N.example_gap(_chunks(_paras(20, ex=3))) == 0
    assert N.example_gap({"open": "x such as y, e.g. z, for instance q " * 150, "mid": "", "close": ""}) == 0


def test_structure_facts_use_only_numbers_the_request_holds():
    thesis = {"questions": ["What explains X?", "", 5], "objectives": ["To examine Y"]}
    chapter = {"chapter_number": 2, "sections": [{"number": "2.1", "title": "A"}, {"number": "2.2", "title": "B"}, {"title": "no number"}]}
    texts = [f["text"] for f in N.structure_facts(thesis, chapter, {"number": "2.1"}, "introduction")]
    assert any("Research question 1" in t for t in texts) and any("Objective 1" in t for t in texts)
    assert any("Chapter 2" in t for t in texts) and any("Section 2.2" in t for t in texts)
    assert not any("Section 2.1" in t for t in texts) and not any("no number" in t for t in texts)
    assert not any("Table" in t or "Figure" in t for t in texts)
    assert N.structure_facts(thesis, chapter, {}, "findings") == [] and N.structure_facts({}, {}, {}, "introduction") == []
    assert N.structure_facts(None, None, None, "introduction") == []


def test_brief_facts_take_a_figure_from_the_authors_own_words_and_never_the_title_or_a_year():
    thesis = {"title": "Sensors on 12 bridges", "sample": "We recruited 120 participants over 6 months", "note": "Started in 2019", "x": 5}
    assert [f["text"] for f in N.brief_facts(thesis)] == ['The author\'s brief says: "We recruited 120 participants over 6 months"']
    assert N.brief_facts({}) == [] and N.brief_facts(None) == []


def test_fact_used_matches_figures_and_structure_pointers_and_ignores_years():
    src = {"text": "Over 25% of bridges were rated deficient in 2009.", "cite": "(Harms et al., 2010)", "kind": "source"}
    assert N.fact_used(src, "about 25% of bridges failed") and not N.fact_used(src, "bridges failed in 2009")
    st = {"text": 'Research question 2 of this thesis is: "x". Refer to it as "Research question 2"', "kind": "structure"}
    assert N.fact_used(st, "this answers research question 2 directly") and not N.fact_used(st, "this answers research question 1")
    assert not N.fact_used({"text": "No digits here at all", "kind": "source"}, "anything")


def _plans(chapter_title, thesis=None, sources=None):
    PROMPTS.clear()
    models(["a", "b", "c"])
    req = request(flags={"pool": True, "sources": bool(sources), "revise": False},
                  chapter={"chapter_number": 2, "title": chapter_title, "estimated_pages": 5},
                  ledger={"sources": sources or [], "by_section": {}})
    if thesis:
        req["thesis"] = {**req["thesis"], **thesis}
    G.write_section(req, G.Deps(baseline={}, profile={}))
    return ["\n".join(ln for ln in p.split("PARAGRAPH PLAN")[1].splitlines() if " words:" in ln) for p in PROMPTS[:3]]


def test_every_passage_plan_holds_an_example_paragraph_and_the_number_rule():
    for title in ("Introduction", "Literature Review", "Methodology", "Discussion"):
        plans = _plans(title)
        assert all("plain example from ordinary practice" in p and 'For example,' in p for p in plans), title
    assert "Use a number only where" in PROMPTS[0]


def test_the_thesis_own_numbers_reach_the_plan_and_no_number_is_invented():
    plans = _plans("Introduction", {"questions": ["What explains X?", "How does Y change?"], "objectives": ["To examine Y"]})
    text = " ".join(plans)
    assert re.search(r"Research question \d|Objective \d|Chapter 2", text)
    nums = set(re.findall(r"\d+", text)) - {str(n) for n in range(0, 2500)}
    assert not nums


def test_first_person_paragraphs_name_plain_verbs_and_forbid_the_announcement():
    seen = False
    for title in ("Introduction", "Methodology", "Discussion"):
        for p in _plans(title):
            if "first person" in p:
                seen = True
                assert "Never \"I argue\"" in p and "chose" in p
    assert seen


def _flag_prompt(paras, **over):
    chunks = _chunks(paras)
    PROMPTS.clear()
    models([chunks["open"] + "\n\n", chunks["mid"] + "\n\n", chunks["close"]], haiku=["[]"])
    req = request(flags={"pool": True, "sources": False, "revise": True}, **over)
    req["thesis"] = {**req["thesis"], "questions": ["What explains X?"], "objectives": ["To examine Y"]}
    return G.write_section(req, G.Deps(baseline={}, profile={}))


def test_a_draft_with_no_figure_no_example_and_no_first_person_is_sent_to_the_patch_for_each():
    _flag_prompt(_paras(12), chapter={"chapter_number": 2, "title": "Introduction", "estimated_pages": 5})
    patch = PROMPTS[3]
    assert "states no exact figure" in patch and "has no example" in patch and "first person" in patch
    assert "Add no other number" in patch
    assert patch.count("Problem:") <= 8                                          # one patch call never carries more than 8 paragraphs


def test_a_draft_that_already_has_them_gets_no_patch_for_them():
    _flag_prompt(_paras(12, fig=6, fp=4, ex=6), chapter={"chapter_number": 2, "title": "Introduction", "estimated_pages": 5})
    assert len(PROMPTS) == 3 or not any(w in PROMPTS[3] for w in ("states no exact figure", "has no example"))


def test_the_free_polish_runs_inside_write_section():
    para = "I argue that power draw decides whether a node survives a long winter in the field. " + \
        "Lee et al. (2015) show that cheap boards miss the low frequency vibration that long spans produce at all. " * 1
    models([para + "\n\n", "Second chunk is plain text with enough words in it to stand as a sentence by itself here.\n\n", "Third chunk follows on from it."])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}, profile={}))
    assert "I argue" not in out["text"] and "Power draw decides" in out["text"] and "Lee et al. (2015)" in out["text"]


def test_baseline_arm_text_is_never_polished():
    models(["I argue that power draw decides whether a node survives.\n\n", "b\n\n", "c"])
    out = G.write_section(request(flags={"pool": False, "sources": False, "revise": False}), G.Deps(baseline={}))
    assert "I argue that power draw" in out["text"]


# ── review findings ──
def test_fact_used_needs_a_whole_number_not_a_digit_inside_another():
    f = {"text": "Wear fell by 3 mm", "kind": "source"}
    assert not N.fact_used(f, "lee (2013) found wear") and N.fact_used(f, "wear fell 3 mm") and not N.fact_used(f, "wear fell 30 mm")
    ch = {"text": "Chapter 3 is here", "kind": "structure"}
    assert N.fact_used(ch, "see chapter 3.") and not N.fact_used(ch, "see chapter 30")


def test_a_page_number_is_not_a_figure():
    assert not N.has_figure("Lee said so (Lee, 2015, p. 45) and again (Kim, 2016, pp. 3-7).")


def test_string_questions_and_objectives_are_ignored_not_split_into_letters():
    assert N.structure_facts({"questions": "What is X?", "objectives": "To do Y"}, {"chapter_number": 1}, {}, "introduction")[0]["text"].startswith("This section")


def test_name_only_source_facts_are_not_number_facts():
    from harness.sources.tools import facts_for
    src = [{"ledger_id": "S1", "authors": "Lee, K.", "year": 2020, "cited_by": 1,
            "abstract": "Opening line about the topic here. The ADXL355 board was used in the trial runs. Over 25% of nodes failed in winter."}]
    kinds = {f["text"]: f["kind"] for f in facts_for({"number": "1"}, src, {"1": ["S1"]})}
    assert kinds["Over 25% of nodes failed in winter."] == "source" and kinds["The ADXL355 board was used in the trial runs."] == "name"
    assert not any(f["kind"] == "name" for f in G._numeric({"facts": [{"kind": "name", "text": "x"}, {"kind": "structure", "text": "y"}], "rules": "r"}))


def test_the_appendix_is_never_polished():
    models(["I argue that power draw decides whether a node survives.\n\n", "b\n\n", "c"])
    out = G.write_section(request(chapter={"chapter_number": 9, "title": "Appendix A", "estimated_pages": 2},
                                  flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}, profile={}))
    assert "I argue that power draw" in out["text"]
