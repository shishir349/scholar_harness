"""CONSTRAINTS.md pinned: the five production leaks of the 3.1 Research Philosophy run, written before the fixes."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from harness import graph as G, limits as L
from harness.corpus import skeletons as SK
from harness.sources.tools import source_notes_for
from harness.text import cleanup as C, lamp_rewrite as LR, protect
from harness.validate import constraints as K, lamp_detector as D, style as V

from .test_graph import clean  # noqa: F401  (autouse fixture reused)
from .test_lamp import EX, para, sent

ROOT = Path(__file__).resolve().parents[1]


def _cites(n, year0=2000):
    return [f"Writer{chr(97 + k)}" for k in range(n)], [year0 + k for k in range(n)]


def narrative(who, year, tag, verb="found"):
    return f"{who} ({year}) {verb} that " + " ".join(f"{tag}{k}" for k in range(11)) + "."


# ── gates as numbers ───────────────────────────────────────────────────────────────────────────────────────────────
def test_the_gate_numbers_are_the_ones_in_constraints_md():
    assert (L.MIN_SENTS, L.MIN_WORDS, L.STD_MIN, L.MAX_SENT, L.SHORT_MAX) == (3, 70, 6.8, 42, 11)
    assert (L.NARR_CALL, L.NARR_SECTION, L.PAREN_SHARE, L.DUP_JACCARD) == (2, 5, 0.70, 0.35)
    assert (V.FLAT_STD, V.MAX_SENT_WORDS, V.SHORT_SENT_WORDS, D.FLAT_STD, D.NARRATIVE_CAP) == (6.8, 42, 11, 6.8, 5)
    assert (G.MAX_NARRATIVE_PASSAGE, G.MAX_NARRATIVE_SECTION, C.MIN_SENTS, C.MIN_WORDS, LR.MAX_SPANS) == (2, 5, 3, 70, 14)
    text = open(ROOT / "CONSTRAINTS.md", encoding="utf-8").read()
    for token in ("70 words", "6.8", "42 words", "11 words", "<= 2", "<= 5", "70%", "0.35"):
        assert token in text, token


def test_the_repairs_and_checks_are_on_by_default(monkeypatch):
    for k in ("HARNESS_SHAPE", "HARNESS_LAMP", "HARNESS_SKELETONS"):
        monkeypatch.delenv(k, raising=False)
    long_ = ("The positivist stance does not sit without friction against the operational realities of district hospitals in the region, "
             "and this divergence is not a minor calibration problem but a question about what the independent variable is in this particular study of staffing across the wards.")
    assert len(G._split_long({"open": long_, "mid": "", "close": ""})["open"].split(". ")) == 2


def test_golden_files_are_untouched():
    if not shutil.which("git"):
        pytest.skip("git not available")
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/harness/prompts/sections.py", "tests/golden/cases.json"], cwd=ROOT)
    assert r.returncode == 0


def test_section_violations_measures_every_gate():
    good = "\n\n".join(para(38, 7, 33, 9, 35, tag=t) for t in ("a", "b", "c"))
    ok = K.section_violations({"open": good, "mid": "", "close": ""})
    assert ok["ok"] and ok["micro_paragraphs"] == [] and ok["cadence"] == [] and ok["paren_share"] == 1.0
    bad = para(30, 8) + "\n\n" + para(24, 25, 27, 26, tag="b") + "\n\n" + para(60, 8, 33, tag="c")
    v = K.section_violations(bad)
    assert not v["ok"] and v["micro_paragraphs"] == [0] and {w.split()[0] for _, w in v["cadence"]} >= {"std", "sentence", "no"}


# ── 1. zero micro-paragraphs ───────────────────────────────────────────────────────────────────────────────────────────
def test_merge_short_paragraphs_takes_the_spec_arguments_and_folds_into_a_neighbour():
    long_ = para(38, 7, 33, 9, tag="b")                                         # 87 words
    chunks = {"open": long_ + "\n\n" + sent(9, "x") + " " + sent(12, "y") + "\n\n" + long_, "mid": "## 3.1 T\n\n" + sent(15, "z") + "\n\n" + long_, "close": ""}
    out = C.merge_short_paragraphs(chunks, min_sents=3, min_words=70)
    assert len(C.paras(out["open"])) == 2 and len(C.paras(out["mid"])) == 2 and out["mid"].startswith("## 3.1 T")
    assert K.section_violations(out)["micro_paragraphs"] == []
    two = {"open": para(30, 8, 34, 9, tag="a") + "\n\n" + para(35, 9, 30, tag="b"), "mid": "", "close": ""}        # 80 and 74 words
    assert C.merge_short_paragraphs(two, min_sents=3, min_words=80) != two and C.merge_short_paragraphs(two, min_words=70) == two


def test_a_mined_skeleton_is_dropped_when_any_body_paragraph_is_short():
    ok = "\n\n".join(para(38, 7, 33, 9, tag=t) for t in ("a", "b", "c"))
    short = ok + "\n\n" + para(30, 8, tag="s")
    assert SK.skeleton_of(ok, "methodology", "mid", "1_0") is not None
    assert SK.skeleton_of(short, "methodology", "mid", "1_1") is None                    # a 2-sentence paragraph
    thin = "\n\n".join([para(30, 8, 12, tag="a")] + [para(38, 7, 33, 9, tag=t) for t in ("b", "c")])      # 3 sentences but 50 words
    assert SK.skeleton_of(thin, "methodology", "mid", "1_2") is None


def test_the_shipped_skeleton_file_holds_no_short_paragraph():
    data = json.loads((ROOT / "data" / "human_skeletons.json").read_text(encoding="utf-8"))["skeletons"]
    assert len(data) >= 150
    for sk in data:
        for p in sk["paragraphs"]:
            assert len(p["sentence_lengths"]) >= L.MIN_SENTS and p["word_count"] >= L.MIN_SKELETON_WORDS, sk["skeleton_id"]
    kinds = {(s["section_type"], s["position"]) for s in data}
    assert len(kinds) == 18                                                              # the rebuild did not empty whole groups


def test_staccato_openers_are_formulaic():
    for t in ("One practical limit stands here.", "One design decision matters here.", "One constraint remains here.", "The design is settled.",
              "Two phases follow.", "The labels hold in the sense that they apply."):
        assert G._FORMULAIC.search(t), t
    assert not G._FORMULAIC.search("The design was piloted with 12 nurses.")


# ── 2. cadence ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cadence_gate_is_std_6_8_max_42_and_a_short_sentence_of_11():
    assert V.burst_flags(para(24, 28, 33, 11, tag="a")) == []                            # std 7.9, has an 11-word sentence
    assert V.burst_flags(para(26, 27, 25, 11, tag="b")) and D.paragraph_is_flat(para(26, 27, 25, 11, tag="b"))      # std 6.5: flat at 6.8
    assert V.burst_flags(para(30, 36, 33, 12, tag="c")) and "none" in V.burst_flags(para(30, 36, 33, 12, tag="c"))[0][1]   # no sentence <= 11
    assert not V.burst_flags(para(38, 36, 33, 11, tag="d")) and V.burst_flags(para(43, 8, 30, tag="e"))


# ── 3. citation form ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_limit_narrative_enforces_two_per_call_five_per_section_and_seventy_percent_parenthetical():
    names = ["Smith", "Lee and Park", "Brown et al.", "Diaz", "Khan", "Moyo", "Ali", "Chen", "Wu"]
    mk = lambda i: narrative(names[i], 2001 + i, f"w{i}x")
    par = " Plain claim (Ng, 2010; Ode, 2011)."
    chunks = {"open": " ".join(mk(i) for i in (0, 1, 2)) + par, "mid": " ".join(mk(i) for i in (3, 4, 5)) + par, "close": " ".join(mk(i) for i in (6, 7, 8)) + par}
    out = C.limit_narrative(chunks)
    v = K.section_violations(out)
    assert all(n <= L.NARR_CALL for n in v["narrative_per_call"].values()) and v["narrative_section"] <= L.NARR_SECTION
    assert v["paren_share"] >= L.PAREN_SHARE
    assert protect.compare(" ".join(chunks.values()), " ".join(out.values())) == []       # every citation and number survives
    assert C.limit_narrative(out) == out                                                    # idempotent


def test_limit_narrative_leaves_a_sentence_it_cannot_restate_and_validate_still_flags_it():
    odd = [f"{w} ({y}), writing before the reform, found that staffing ratios " + " ".join(f"q{k}" for k in range(12)) + "." for w, y in
           (("Smith", 2001), ("Lee", 2002), ("Park", 2003))]
    chunks = {"open": " ".join(odd), "mid": "", "close": ""}
    out = C.limit_narrative(chunks)
    assert out == chunks and G._narrative_surplus(out)                                       # not the plain pattern: not rewritten, still reported


def test_source_notes_are_pipe_fragments_without_abstract_verbs():
    abstract = ("Employee engagement was examined in retail. The authors conducted a cross-sectional survey of 412 employees across 38 stores and "
                "noted that engagement was associated with 12% higher sales. A limitation is the single-country sample.")
    line = source_notes_for({"number": "3.1"}, [{"ledger_id": "S1", "authors": "Brown, A.", "year": 2019, "abstract": abstract}],
                            {"3.1": ["S1"]}, telegraph=True).splitlines()[-1]
    assert line.startswith("- (Brown, 2019): ") and "412 employees" in line and "12%" in line
    assert not re.search(r"\b(?:examined|conducted|noted|argued)\b", line, re.I)
    assert abstract.split(". ")[1][:50] not in line


# ── 4. whole-section deduplication ─────────────────────────────────────────────────────────────────────────────────────
def test_a_work_summarised_in_open_is_not_summarised_again_in_close_whatever_the_verb():
    a = narrative("Brown and Dueñas", 2019, "aa", "noted")
    b = narrative("Brown and Dueñas", 2019, "zz", "observed")                              # new words, same work, another verb
    keep = sent(15, "kk") + " " + sent(12, "ll")
    chunks = {"open": sent(14, "o") + " " + a + " " + sent(11, "p"), "mid": sent(16, "m") + " " + sent(12, "n") + " " + sent(10, "q"),
              "close": sent(13, "c") + " " + b + " " + keep}
    out, _ = C.drop_repeats(chunks)
    assert a in out["open"] and b not in out["close"] and keep in out["close"]
    assert K.section_violations(out)["author_year_summarised_twice"] == [] and K.section_violations(chunks)["author_year_summarised_twice"]


def test_a_near_duplicate_sentence_across_chunks_is_dropped_at_jaccard_035():
    base = "Positivism holds that social phenomena can be measured quantified and generalised across a defined population of hospitals"
    near = "Social phenomena can be measured quantified and generalised across defined hospital populations according to positivism"
    chunks = {"open": sent(14, "o") + " " + base + ". " + sent(11, "p"), "mid": "", "close": sent(13, "c") + " " + near + ". " + sent(12, "d") + " " + sent(9, "e")}
    out, _ = C.drop_repeats(chunks)
    assert near not in out["close"] and K.section_violations(out)["duplicate_sentence_pairs"] == []


def test_the_writer_is_told_which_works_are_already_summarised():
    text = "## 3.1 T\n\n" + sent(10, "z") + " " + narrative("Brown and Dueñas", 2019, "bb") + " " + sent(8, "y") + " (Turyahikayo, 2021)."
    out = G._made(text)
    assert "ALREADY_SUMMARIZED_SOURCES" in out and "Brown and Dueñas (2019)" in out and "Turyahikayo (2021)" in out and "do not summarise" in out.lower()


# ── 5. LAMP ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_lamp_takes_fourteen_spans_and_accepts_a_split_that_keeps_the_citation():
    assert LR.MAX_SPANS == 14
    a = ("Brown and Dueñas (2021) noted that despite the importance of research paradigms in shaping inquiry across twelve studies of hospital staffing "
         "the choice of philosophical stance is rarely justified in the methods that follow it and that this omission leaves readers unable to judge the design.")
    b = ("Despite the importance of research paradigms in shaping inquiry across twelve studies of hospital staffing, the stance is rarely justified in the methods (Brown and Dueñas, 2021). "
         "The omission leaves readers unable to judge the design.")
    assert len(a.split()) > 40 and protect.compare(a, b) == []
    both = b.replace("The omission leaves", "That gap (Brown and Dueñas, 2021) leaves")
    assert protect.compare(a, both) == []                                                  # the citation in both halves
    assert protect.compare(a, b.replace("2021", "2022")) and protect.compare(a, b.replace("twelve", "thirteen")) == []
