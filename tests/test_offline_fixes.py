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
from harness.text.sentences import split_sentences
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


def test_no_a_study_lead_in_exists_any_more():
    s = "Truss and Gratton (1994) distinguished hard from soft SHRM across a conceptual framework."
    assert C._restate(s) is None                                      # no safe form: left for LAMP, never "A study (...) distinguished"
    out = C.limit_narrative({"open": s, "mid": "", "close": ""}, per_call=0, per_section=0, paren_share=1.0)["open"]
    assert out == s and "A study" not in out


@pytest.mark.parametrize("src,want", [
    ("Sun and Zuo (2024) describe SHRM as a bundle of practices.", "SHRM is described as a bundle of practices (Sun & Zuo, 2024)."),
    ("Maarouf (2019) examined retention in call centres and concluded that pay mattered most.", "Pay mattered most (Maarouf, 2019)."),
])
def test_present_tense_and_examined_and_concluded_forms(src, want):
    assert C._restate(src) == want


# ── the flagged section, end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_flagged_section_narrative_citations_fall_below_the_gate_with_no_api_calls():
    chunks = _chunks()
    before = K.section_violations(chunks)
    assert before["narrative_section"] >= 8 and before["paren_share"] < 0.2
    out = C.limit_narrative(chunks, per_call=limits.NARR_CALL, per_section=limits.NARR_SECTION)
    after = K.section_violations(out)
    assert after["narrative_section"] <= 11 and before["narrative_section"] >= 20      # what no safe pattern covers is left for LAMP and the validator
    assert after["paren_share"] >= 0.45 and "A study" not in " ".join(out.values())
    assert protect.compare(" ".join(chunks.values()), " ".join(out.values())) == []        # every citation, number and name survives
    assert C.limit_narrative(out, per_call=limits.NARR_CALL, per_section=limits.NARR_SECTION) == out          # idempotent


def test_dropping_repeated_summaries_first_then_limiting_leaves_few_narrative_citations():
    out = C.limit_narrative(C.drop_repeats(_chunks(), [])[0])
    assert K.section_violations(out)["narrative_section"] <= 6


# ── long sentences ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _filler(n, tag="x"):
    return " ".join(f"{tag}{k}" for k in range(n))


@pytest.mark.parametrize("marker,joined,starts", [
    ("; ", None, "Y"),
    (": ", None, "Y"),
    (", whereas ", "In contrast, ", None),
    (", while ", None, "Y"),
    (", though ", None, "Y"),
    (", where ", "There, ", None),
])
def test_long_sentence_is_cut_at_each_marker(marker, joined, starts):
    head = "The first part of the sentence runs for about twenty words and says one complete thing about " + _filler(8, "h") + ""
    tail = "the second part is also a full clause that is has " + _filler(14, "t") + " done"
    s = head + marker + tail + "."
    out = C._split_sentence(s, 42 - 14)
    parts = split_sentences(out)
    assert len(parts) == 2 and parts[1][0].isupper() and max(len(x.split()) for x in parts) < len(s.split())
    if joined:
        assert parts[1].startswith(joined)


def test_which_clause_becomes_a_this_sentence_only_late_and_long():
    head = "The study surveyed a very large number of firms across several national contexts and sectors in a single year " + _filler(6, "h")
    s = head + ", which means the gains " + _filler(12, "t") + " are conditional."
    out = C._split_sentence(s, 30)
    assert ". This means the gains" in out and out.count(".") == 2
    short = "A short sentence, which means little, ends here."
    assert C._split_sentence(short, 3) == short                      # head under 22 words: not cut


def test_because_and_appositive_and_who_tails():
    head = "The measure varied so widely across instruments that comparing results became a category error " + _filler(10, "h")
    assert ". This is because researchers" in C._split_sentence(head + ", because researchers labelled " + _filler(12, "t") + " names.", 30)
    assert ". This is partly because the instruments" in C._split_sentence(head + ", partly because the instruments " + _filler(12, "t") + " ask.", 30)
    out = C._split_sentence(head + ", a design choice that allows rich description but " + _filler(12, "t") + " limits.", 30)
    assert ". This design choice allows rich description" in out
    kept = C._split_sentence(head + ", a company that employs " + _filler(8, "q") + ", is large and " + _filler(10, "r") + " done.", 30)
    assert ". This company" not in kept                               # the appositive continues into the main clause: not cut
    who = C._split_sentence(head + " such as the SMEs studied by Pittino (2016), who found that practices " + _filler(12, "t") + " held.", 30)
    assert ". They found that practices" in who


def test_flagged_section_long_sentences_fall_from_12_to_at_most_2():
    chunks = C.limit_narrative(_chunks())
    over = lambda c: sum(1 for v in c.values() for x in split_sentences(v) if len(x.split()) > limits.MAX_SENT)   # noqa: E731
    assert over(_chunks()) == 12
    out = C.split_long(chunks, limits.MAX_SENT)
    assert over(out) <= 2
    assert protect.compare(" ".join(chunks.values()), " ".join(out.values())) == []
    assert C.split_long(out, limits.MAX_SENT) == out                  # idempotent


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
