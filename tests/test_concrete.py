"""Concrete detail and fewer decorative citations: richer source notes, a fact assigned to paragraphs, an uncited-paragraph plan,
Scholar's repeat-licensing noise lines removed, and an evidence rule that stops invented disagreement between sources."""
import math
import re

from harness import graph as G
from harness.prompts import sections as P
from harness.sources.tools import NOTE_CHARS, facts_for, source_notes_for
from tests.test_graph import PROMPTS, models, request

SRC = [
    {"ledger_id": "S1", "authors": "Harms, T., & Lee, K.", "year": 2010, "cited_by": 5,
     "abstract": "Bridges matter to every region. Over 25% of bridges in the United States were rated deficient in 2009. "
                 "The ADXL355 sensor costs under 20 dollars."},
    {"ledger_id": "S2", "authors": "Yu, X.", "year": 2023, "cited_by": 9,
     "abstract": "Nodes have advanced in a decade of research. The study ran from 2015 to 2019."},
    {"ledger_id": "S3", "authors": "Kim, J.", "year": 2021, "cited_by": 2,
     "abstract": "Packet loss rose to 12% when nodes shared one channel. Battery life fell by 30% in winter."},
]
BY = {"2.1": ["S1", "S2", "S3"]}


def test_facts_are_figure_or_name_sentences_with_their_source_and_skip_year_only_sentences():
    facts = facts_for({"number": "2.1"}, SRC, BY)
    texts = [f["text"] for f in facts]
    assert "Over 25% of bridges in the United States were rated deficient in 2009." in texts
    assert "Battery life fell by 30% in winter." in texts
    assert not any("ran from 2015 to 2019" in t for t in texts)                        # a year is not a figure
    assert facts[0]["cite"] == "(Harms et al., 2010)" and facts_for({"number": "2.1"}, SRC, BY, limit=1) == facts[:1]
    assert facts_for({"number": "2.1"}, [], BY) == []


def test_figure_sentences_come_before_name_only_sentences():
    texts = [f["text"] for f in facts_for({"number": "2.1"}, SRC[:1], BY)]
    assert texts.index("Over 25% of bridges in the United States were rated deficient in 2009.") < texts.index("The ADXL355 sensor costs under 20 dollars.")


def test_a_note_keeps_two_figure_sentences_when_there_is_room():
    abstract = ("Opening sentence about the topic of this study at large. First result: accuracy was 94% over 120 samples. "
                "Second result: drift was 3 mm per year at the old bridge. Closing remark follows here.")
    srcs = [{"ledger_id": "S1", "authors": "Lee, K.", "year": 2020, "abstract": abstract, "cited_by": 3}]
    note = source_notes_for({"number": "2.1"}, srcs, {"2.1": ["S1"]}).splitlines()[-1]
    assert "94%" in note and "3 mm" in note and len(note.split("): ", 1)[1]) <= NOTE_CHARS


def _plan(prompt):
    return [ln for ln in prompt.split("PARAGRAPH PLAN")[1].splitlines() if " words:" in ln]


def _run(chapter_title, sources=SRC):
    PROMPTS.clear()
    models(["a", "b", "c"])
    req = request(flags={"pool": True, "sources": True, "revise": False}, ledger={"sources": sources, "by_section": BY},
                  chapter={"chapter_number": 2, "title": chapter_title, "estimated_pages": 5})
    req["section"] = {**req["section"], "number": "2.1"}
    G.write_section(req, G.Deps(baseline={}, profile={}))
    return PROMPTS[:3]


def test_each_passage_gets_a_different_fact_to_state_in_its_own_words():
    plans = [_plan(p) for p in _run("Literature Review")]
    used = [re.findall(r'this fact[^"]*"([^"]+)"', " ".join(lines)) for lines in plans]
    assert all(any("this fact" in ln or "number exact" in ln for ln in lines) for lines in plans)     # every passage states a figure
    flat = [f for u in used for f in u]
    assert len(flat) == len(set(flat))                                                  # never the same fact twice in a section


def test_the_plan_leaves_a_real_share_of_paragraphs_without_a_citation_and_never_mixes_the_two_instructions():
    for title, real in (("Literature Review", 0.47), ("Introduction", 0.24)):
        lines = [ln for p in _run(title) for ln in _plan(p)]
        none = [ln for ln in lines if "carries no citation" in ln]
        assert abs(len(none) / len(lines) - (1 - real)) <= 0.2, title
        assert not any("carries no citation" in ln and ("in parentheses" in ln or "this fact" in ln) for ln in lines)


def test_without_sources_the_plan_says_nothing_about_citations_or_facts():
    PROMPTS.clear(); models(["a", "b", "c"])
    G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}, profile={}))
    assert not any("carries no citation" in ln or "this fact" in ln for p in PROMPTS[:3] for ln in _plan(p))


def _cited_chunks(n_cited, n_total):
    para = lambda i, cited: " ".join(["Costs fell in region number five last year as firms cut output again"] * 5) + (f" (Smith, 20{10 + i})." if cited else ".")
    ps = [para(i, i < n_cited) for i in range(n_total)]
    return {"open": "\n\n".join(ps[:4]), "mid": "\n\n".join(ps[4:8]), "close": "\n\n".join(ps[8:])}


def test_enough_cited_paragraphs_are_flagged_to_bring_the_share_back_to_real_and_no_more():
    assert G.cite_para_targets(_cited_chunks(7, 10), "literature_review") and len(G.cite_para_targets(_cited_chunks(7, 10), "literature_review")) == 2
    assert len(G.cite_para_targets(_cited_chunks(10, 10), "literature_review")) == 5          # capped
    assert G.cite_para_targets(_cited_chunks(5, 10), "literature_review") == []                # inside real + 0.06


def test_a_paragraph_that_carries_a_figure_is_never_asked_to_drop_its_citation():
    chunks = _cited_chunks(10, 10)
    chunks = {k: v.replace("Costs fell", "Costs fell by 30%") for k, v in chunks.items()}
    assert G.cite_para_targets(chunks, "literature_review") == []


def test_noise_lines_that_license_repeating_a_point_are_removed_for_every_register():
    for p in P.profiles().values():
        out = G._trim_noise(P.noise_block(p))
        assert not re.search(r"restat|circle back|backward glance|natural repetition|recursive|redundant|over-explain|elaborate than strictly", out, re.I), p["id"]
    lit = G._trim_noise(P.noise_block(P.profiles()["literature_review"]))
    assert lit.startswith("NATURAL ACADEMIC REALISM") and "Not every paragraph needs equal strength" in lit
    assert G._trim_noise("None") == "None"


def test_the_harness_path_never_tells_the_writer_to_invent_disagreement_or_results():
    lit = _run("Literature Review")[0]
    assert "genuinely mixed" not in lit and "replications fail" not in lit and "only where a SOURCE NOTE says so" in lit
    disc = _run("Discussion")[0]
    assert "partially confirmed" not in disc and "do not state effect sizes" in disc
    assert "r = 0.20" not in _run("Findings")[0]


def test_scholars_own_prompt_is_untouched_without_the_harness_flag():
    PROMPTS.clear(); models(["a", "b", "c"])
    G.write_section(request(chapter={"chapter_number": 2, "title": "Literature Review", "estimated_pages": 5},
                            flags={"pool": False, "sources": False, "revise": False}), G.Deps(baseline={}))
    assert "genuinely mixed" in PROMPTS[0] and "Occasional conceptual restatement" in PROMPTS[0]


def test_lines_that_license_drift_or_critique_without_evidence_are_gone_on_the_harness_path():
    gone = ("Allow occasional conceptual drift", "Slight imperfections in flow", "Expose limitations, contradictions, or oversimplifications",
            "Critique literature that presents overly neat", "Acknowledge unresolved debates instead of presenting stable consensus",
            "Acknowledge what remains genuinely contested or unresolved")
    lit = _run("Literature Review")
    for prompt in lit:
        assert not any(g in prompt for g in gone), [g for g in gone if g in prompt]
    assert "against itself" not in lit[1] and "competing schools" not in lit[1]
    assert "only where a source note or your own reasoning shows it" in lit[1]
