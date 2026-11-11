"""Offline fixes: the narrative-to-parenthetical converter on the text GPTZero scored 100% AI, and skeletons built from merged paragraphs.
No API calls: llm.ask is replaced by a function that fails the test if anything reaches it."""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from harness import llm, limits
from harness.corpus import skeletons as SK
from harness.text import cleanup as C, protect
from harness.validate import constraints as K

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "section_2_1.txt"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an API call was made")
    monkeypatch.setattr(llm, "ask", boom)


def _chunks():
    paras = [p for p in re.split(r"\n\s*\n", FIXTURE.read_text(encoding="utf-8")) if p.strip() and not p.startswith("#")]
    return {"open": "\n\n".join(paras[:2]), "mid": paras[2], "close": paras[3]}


# ── limit_narrative patterns ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("src,want", [
    ("Maarouf (2019) argued that retention depends on pay.", "Retention depends on pay (Maarouf, 2019)."),                              # A
    ("Sun and Zuo (2024) found that turnover fell after the change.", "Turnover fell after the change (Sun & Zuo, 2024)."),
    ("Jackson et al. (2013) noted the field had expanded widely.", "The field had expanded widely (Jackson et al., 2013)."),            # A, no "that"
    ("Ananthram, Xerri, Teo, and Connell (2018) found that effects were mediated.", "Effects were mediated (Ananthram, Xerri, Teo, & Connell, 2018)."),
    ("Wright and McMahan (1992) described SHRM as a pattern of planned deployments.", "SHRM was described as a pattern of planned deployments (Wright & McMahan, 1992)."),   # D
    ("The result, as Brown and Dueñas (2019) noted, holds only in large firms.", "The result (Brown & Dueñas, 2019) holds only in large firms."),                 # B
    ("Sun and Zuo (2024) conducted a critical review of the field and found that turnover fell.", "Turnover fell (Sun & Zuo, 2024)."),   # C: no number in the description, dropped
    ("Chang (2005) surveyed 235 Taiwanese firms and found that strategy moderated the effect.", "Strategy moderated the effect (Chang, 2005; surveyed 235 Taiwanese firms)."),  # C: the number stays
    ("The gap persists, as noted by Jackson et al. (2013), in most settings.", "The gap persists, as noted (Jackson et al., 2013), in most settings."),
])
def test_one_sentence_is_restated_with_the_citation_in_parentheses(src, want):
    assert C._restate(src) == want
    assert protect.compare(src, want) == []


def test_sentences_it_cannot_restate_safely_are_left_alone():
    assert C._restate("Smith (2010), writing before the reform, found that staffing ratios rose.") is None
    assert C._restate("Heffernan and Dundon (2016) found cross-level effects on well-being, mediated by justice.") is None     # verb + noun phrase: no safe form
    assert C._restate("Ananthram et al. (2018) measured effects in call centres and found mediated relationships.") is None   # "found" + noun phrase, not a clause
    assert C._restate("Smith (2010) and Lee (2011) disagree about retention.") is None                                         # two narrative citations


def test_last_resort_lead_in_only_when_still_over_the_limit():
    s = "Truss and Gratton (1994) distinguished hard from soft SHRM across a conceptual framework."
    assert C._restate(s) is None and C._restate(s, "A study") == "A study (Truss & Gratton, 1994) distinguished hard from soft SHRM across a conceptual framework."
    two = {"open": s + " Lado and Jackson (2000) pointed to boundary conditions in how constructs are operationalised across firms.", "mid": "", "close": ""}
    assert C.limit_narrative(two, per_call=2, per_section=5, paren_share=0.0) == two                      # within the limits: untouched
    out = C.limit_narrative(two, per_call=1, per_section=5, paren_share=0.0)["open"]
    assert out.count("A study") == 1 and "A study (Lado & Jackson, 2000)" in out    # latest first; one flip is enough
    assert out.startswith("Truss and Gratton (1994)")


# ── the flagged section, end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_flagged_section_goes_from_21_narrative_citations_to_four_with_no_api_calls():
    chunks = _chunks()
    before = K.section_violations(chunks)
    assert before["narrative_section"] >= 8 and before["paren_share"] < 0.2
    out = C.limit_narrative(chunks, per_call=limits.NARR_CALL, per_section=4)
    after = K.section_violations(out)
    assert after["narrative_section"] <= 4 and all(n <= limits.NARR_CALL for n in after["narrative_per_call"].values())
    assert after["paren_share"] >= limits.PAREN_SHARE
    assert protect.compare(" ".join(chunks.values()), " ".join(out.values())) == []        # every citation, number and name survives
    assert C.limit_narrative(out, per_call=limits.NARR_CALL, per_section=4) == out          # idempotent


def test_dropping_repeated_summaries_first_leaves_a_section_that_meets_the_shipped_gate():
    out = C.limit_narrative(C.drop_repeats(_chunks(), [])[0])
    v = K.section_violations(out)
    assert v["narrative_section"] <= limits.NARR_SECTION and v["paren_share"] >= limits.PAREN_SHARE


# ── skeletons: merge short paragraphs BEFORE the staccato filter ────────────────────────────────────────────────────────
def _sent(tag, n):
    return f"Sentence {tag} " + " ".join(f"word{k}" for k in range(n - 3)) + " ends."


def _para(tag, lens):
    return " ".join(_sent(f"{tag}{i}", n) for i, n in enumerate(lens))


def test_a_chunk_with_one_short_paragraph_still_yields_a_skeleton_after_the_merge():
    text = "\n\n".join([_para("a", [30, 8, 36]), _para("b", [34]), _para("c", [28, 9, 40]), _para("d", [33, 7, 31])])
    sk = SK.skeleton_of(text, "findings", "mid", "x")
    assert sk is not None
    assert all(len(p["sentence_lengths"]) >= limits.MIN_SENTS and p["word_count"] >= limits.MIN_SKELETON_WORDS for p in sk["paragraphs"])
    assert len(sk["paragraphs"]) == 3 and max(p["word_count"] for p in sk["paragraphs"]) <= SK.MAX_MERGED + 80


def test_a_chunk_that_cannot_be_repaired_is_still_dropped():
    text = "\n\n".join([_para("a", [30]), _para("b", [30]), _para("c", [30])])        # 3 one-sentence paragraphs merge into one: under MIN_PARAS
    assert SK.skeleton_of(text, "findings", "mid", "x") is None


def test_shipped_skeletons_are_plentiful_in_every_bucket_and_have_no_micro_paragraph():
    d = json.loads((ROOT / "data" / "human_skeletons.json").read_text(encoding="utf-8"))["skeletons"]
    by = Counter((s["section_type"], s["position"]) for s in d)
    assert len(d) >= 1200 and len(by) == 18 and min(by.values()) >= 15
    assert not [s for s in d if any(len(p["sentence_lengths"]) < 3 or p["word_count"] < limits.MIN_SKELETON_WORDS for p in s["paragraphs"])]
