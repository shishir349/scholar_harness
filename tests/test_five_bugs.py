"""The five production bugs found in section 3.1: staccato paragraphs, abstract paraphrase, cross-call repeats, starved LAMP,
and the run that shows them (scripts/test_live_section.py)."""
import json

from harness import graph as G, llm
from harness.corpus import skeletons as SK
from harness.sources.tools import _telegraph, source_notes_for
from harness.text import cleanup as C, lamp_rewrite as LR, protect
from harness.validate import lamp_detector as D, style as V

from .test_graph import clean  # noqa: F401  (autouse fixture reused)
from .test_lamp import EX, para, sent


# ── bug 1: staccato micro-paragraphs ─────────────────────────────────────────────────────────────────────────────────
def test_one_and_two_sentence_paragraphs_are_folded_into_a_neighbour_and_headings_stay():
    body = para(30, 8, 34, tag="a")                                   # 72 words: short by words too
    long_ = para(30, 8, 34, 9, 33, tag="b")                           # 4+ sentences, 114 words
    chunks = {"open": "## 3.1 Title\n\n" + sent(9, "x") + "\n\n" + long_, "mid": long_ + "\n\n" + sent(9, "y") + " " + sent(12, "z") + "\n\n" + long_,
              "close": body + "\n\n" + long_}
    out = C.merge_short(chunks)
    assert out["open"].startswith("## 3.1 Title") and len(C.paras(out["open"])) == 2          # the lone sentence joined the next paragraph
    assert len(C.paras(out["mid"])) == 2 and any(sent(9, "y") in p for p in C.paras(out["mid"]))   # the 2-sentence block joined a neighbour
    assert all(len(C.split_sentences(p)) >= 3 for pos in out for p in C.paras(out[pos]) if not p.startswith("#"))
    assert C.merge_short(out) == out                                                           # idempotent


def test_skeleton_specs_never_hold_a_short_paragraph_or_a_long_or_all_long_sentence_target():
    sk = {"skeleton_id": "1_0", "section_type": "methodology", "position": "mid", "total_words": 300, "paragraphs": [
        {"word_count": 20, "sentence_lengths": [20], "sent_len_cv": 0, "opener_type": "domain_noun_subject",
         "cite_positions": {"sentences": [], "mode": "none"}, "closer_type": "comparative_assertion"},
        {"word_count": 120, "sentence_lengths": [55, 30, 35], "sent_len_cv": 0.3, "opener_type": "author_narrative_cite",
         "cite_positions": {"sentences": [1], "mode": "narrative"}, "closer_type": "raw_data_or_cite_cutoff"},
        {"word_count": 40, "sentence_lengths": [18, 22], "sent_len_cv": 0.1, "opener_type": "empirical_figure_or_metric",
         "cite_positions": {"sentences": [2], "mode": "parenthetical"}, "closer_type": "limitation_or_caveat"},
        {"word_count": 120, "sentence_lengths": [30, 28, 31, 30], "sent_len_cv": 0.04, "opener_type": "domain_noun_subject",
         "cite_positions": {"sentences": [], "mode": "none"}, "closer_type": "comparative_assertion"}]}
    specs = SK.scale(sk, 300)
    assert len(specs) == 2                                                  # 4 -> 2: the 1- and 2-sentence ones were merged
    for s in specs:
        t = s["target_sentences"]
        assert len(t) >= 3 and s["target_words"] >= 75 and max(t) <= 42 and min(t) < 10 and min(t[1:]) < 10
    assert specs[0]["cite_positions"]["mode"] == "narrative" and specs[0]["cite_positions"]["sentences"] == [2, 6]    # shifted by the merges
    assert "3+ sentences" in SK.blueprint(specs)


def test_burst_flags_catch_a_long_sentence_and_a_paragraph_with_no_short_one():
    text = para(44, 8, 20, tag="a") + "\n\n" + para(30, 14, 36, tag="b") + "\n\n" + para(38, 7, 33, tag="c")
    notes = dict(V.burst_flags(text))
    assert "44-word" in notes[0] and "none of 11 words or fewer" in notes[1] and 2 not in notes


def test_lamp_flat_note_enforces_the_same_limits_so_a_rewrite_cannot_trade_one_defect_for_another():
    assert D.paragraph_is_flat(para(44, 8, 20)) and D.paragraph_is_flat(para(30, 14, 36)) and not D.paragraph_is_flat(para(38, 7, 33))


def test_rhetorical_micro_openers_are_formulaic():
    for t in ("One practical limit stands here. Next.", "The design is settled. Two phases follow.",
              "The labels apply in the sense that they hold."):
        assert G._FORMULAIC.search(t), t
    assert not G._FORMULAIC.search("The design was piloted with 12 nurses.")


# ── bug 2: abstract paraphrase ─────────────────────────────────────────────────────────────────────────────────────────
ABSTRACT = ("Employee engagement predicts store outcomes in retail. Using a cross-sectional survey of 412 employees across 38 stores, "
            "this study found that engagement was associated with 12% higher sales. A limitation is the single-country sample. "
            "Future research should test other regions.")


def test_source_notes_are_telegraphic_facts_not_abstract_sentences():
    line = source_notes_for({"number": "3.1"}, [{"ledger_id": "S1", "authors": "Brown, A.", "year": 2019, "abstract": ABSTRACT, "cited_by": 3}],
                            {"3.1": ["S1"]}, telegraph=True).splitlines()[-1]
    assert line.startswith("- (Brown, 2019): ") and "412 employees" in line and "12%" in line and "cross-sectional" in line.lower()
    assert "method:" in line and "finding:" in line and "limit:" in line
    assert ABSTRACT.split(". ")[1][:40] not in line                          # no abstract sentence survives whole
    assert "finding: engagement was associated with 12% higher sales" in line
    plain = source_notes_for({"number": "3.1"}, [{"ledger_id": "S1", "authors": "Brown, A.", "year": 2019, "abstract": ABSTRACT}], {"3.1": ["S1"]})
    assert "Using a cross-sectional survey" in plain                          # the default (Scholar's own) form is unchanged


def test_more_than_two_narrative_citations_in_a_passage_are_flagged_with_their_paragraph():
    p = lambda cite, tag: sent(10, tag + "a") + f" Smith ({cite}) found that " + sent(8, tag + "b")
    chunks = {"open": p(2001, "a") + "\n\n" + p(2002, "b") + "\n\n" + p(2003, "c") + "\n\n" + sent(9, "plain") + " (Lee, 2004).", "mid": p(2005, "d"), "close": ""}
    flags = G._narrative_surplus(chunks)
    assert [(pos, i) for pos, i, _ in flags] == [("open", 2)] and G.narrative_count("Smith (2001) found it (Lee, 2004).") == 1


def test_lamp_flags_narrative_citations_past_six_and_stock_idioms_and_names_the_phrase():
    cites = "\n\n".join(sent(30, f"s{k}") + f" Writer{chr(97 + k)} ({2000 + k}) " + sent(7, f"t{k}") for k in range(8))
    spans = D.detect_ai_spans(cites + "\n\nThe model carries evidential weight in this field and sits at the intersection of theory and practice.")
    kinds = {t for s in spans for t in s["types"]}
    assert {"narrative_surplus", "stock_idiom"} <= kinds
    idiom = next(s for s in spans if "stock_idiom" in s["types"])
    assert "carries evidential weight" in idiom["instruction"]
    assert sum("narrative_surplus" in s["types"] for s in spans) == 3


# ── bug 3: repeats across open / mid / close ──────────────────────────────────────────────────────────────────────────
def test_a_sentence_repeated_from_open_in_close_is_dropped_and_the_citation_frame_too():
    dup = "Brown and Dueñas (2019) noted that despite the importance of research paradigms in shaping inquiry the choice of stance is rarely justified."
    reword = "Brown and Dueñas (2019) noted that stance is seldom justified although paradigms matter for the inquiry that researchers design."
    keep = sent(15, "keepa") + " " + sent(12, "keepb")
    chunks = {"open": sent(14, "o") + " " + dup + " " + sent(11, "p"), "mid": sent(16, "m") + " " + sent(12, "n") + " " + sent(10, "q"),
              "close": sent(13, "c") + " " + dup + " " + reword + " " + keep}
    out, _ = C.drop_repeats(chunks)
    assert out["open"] == chunks["open"]
    assert dup not in out["close"] and reword not in out["close"] and keep in out["close"]


def test_the_repetition_ledger_lists_each_source_already_used():
    text = "## 3.1 T\n\n" + sent(10, "z") + " Brown and Dueñas (2019) noted that stance is rarely stated. " + sent(8, "y") + " (Lee et al., 2020)."
    out = G._made(text)
    assert "ALREADY_SUMMARIZED_SOURCES" in out and "Brown and Dueñas (2019): " in out and "Lee et al. (2020)" in out


# ── bug 4: LAMP starvation and silent rejections ─────────────────────────────────────────────────────────────────────
def test_lamp_takes_fifteen_spans_by_default_and_reports_why_each_rejection_happened(monkeypatch):
    text = "\n\n".join(para(24, 27, 25, 28, tag=t) for t in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"))
    fixed = para(40, 7, 36, tag="new")
    monkeypatch.setattr(llm, "ask", lambda *a, **k: json.dumps([fixed + f" {i}" if i % 2 else para(24, 27, 25, 28, tag=f"z{i}") for i in range(14)]))
    out, rep = LR.lamp_rewrite_section(text, "introduction", EX, None)
    assert rep["flagged"] == 18 and rep["sent"] == 14 and rep["applied"] + len(rep["rejected"]) == 14
    assert rep["rejected_by"] and sum(rep["rejected_by"].values()) == len(rep["rejected"]) and rep["sent_words"] > 0


def test_splitting_a_cited_sentence_into_two_halves_that_both_cite_is_accepted():
    a = "Brown and Dueñas (2019) noted that despite the importance of research paradigms across 12 studies, stance is rarely justified in the methods."
    b = "Brown and Dueñas (2019) found paradigms matter across 12 studies. Yet stance is rarely justified (Brown and Dueñas, 2019)."
    assert protect.compare(a, b) == []
    assert protect.compare(a, b.replace("12", "13")) and protect.compare(a, b.replace("2019)", "2018)", 1))      # numbers and citations still guarded


# ── bug 5 helper: the diagnostic counts what it claims to count ─────────────────────────────────────────────────────
def test_diagnostic_counts_sentence_lengths_citations_and_cross_chunk_repeats():
    from scripts import test_live_section as T
    text = para(30, 8, 34) + " Smith (2001) found it (Lee, 2004).\n\n" + para(30, 8, 34, tag="b")
    d = T.diagnose({"open": text, "mid": "", "close": ""})
    assert d["paragraphs"][0][:3] == [30, 8, 34] and d["narrative"] == 1 and d["parenthetical"] == 1 and d["cross_chunk_repeats"] == 0


# ── found by the first live run ────────────────────────────────────────────────────────────────────────────────────────
def test_merging_never_builds_a_giant_paragraph_and_leaves_a_short_one_it_cannot_place():
    big = para(30, 8, 34, 9, 33, 30, 12, tag="b")                           # 7 sentences, ~156 words
    chunks = {"open": big + "\n\n" + sent(9, "x") + " " + sent(8, "y") + "\n\n" + big, "mid": "", "close": ""}
    out = C.merge_short(chunks)
    assert max(len(p.split()) for p in C.paras(out["open"])) <= C.MAX_MERGED_WORDS
    run = {"open": sent(10, "a") + "\n\n" + sent(11, "b") + "\n\n" + big, "mid": "", "close": ""}
    merged = C.paras(C.merge_short(run)["open"])
    assert len(merged) == 1 and len(merged[0].split()) <= C.MAX_MERGED_WORDS          # two lone sentences became one block, then joined the big one


def test_a_sentence_over_42_words_is_cut_at_a_clause_not_in_a_list_and_citations_survive():
    long_ = ("The positivist stance does not sit without friction against the operational realities of district hospitals in the region, "
             "and this divergence is not a minor calibration problem but a question about what the independent variable is in this particular study of staffing (Long, 2018).")
    out = C.split_long({"open": long_, "mid": "", "close": ""})["open"]
    assert len(C.split_sentences(out)) == 2 and "(Long, 2018)" in out and out.count("2018") == 1 and all(len(x.split()) <= 42 for x in C.split_sentences(out))
    lst = "The panel covered " + ", ".join(f"ward {w}" for w in "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda".split()) + ", and mu."
    assert C.split_long({"open": lst + " " + lst, "mid": "", "close": ""})["open"] == lst + " " + lst        # nothing to split safely
    assert protect.compare(long_, out) == []


def test_stacked_fragments_do_not_satisfy_the_one_short_sentence_rule():
    gamed = sent(30, "a") + " " + sent(3, "b") + " " + sent(4, "c") + " " + sent(34, "d")
    assert D.paragraph_is_flat(gamed) and V.burst_flags(gamed) and "fragments" in V.burst_flags(gamed)[0][1]
    assert not D.paragraph_is_flat(para(38, 7, 33))


def test_rhetorical_micro_sentences_are_dropped_but_a_short_fact_stays():
    p = sent(30, "a") + " It is not decoration. " + sent(34, "b") + " Choice matters here. The sample was 412 nurses. " + sent(28, "c")
    out = C.drop_micro({"open": p, "mid": "", "close": ""})["open"]
    assert "decoration" not in out and "Choice matters" not in out and "412 nurses" in out
    two = sent(30, "a") + " It does, considerably. " + sent(30, "b")
    assert C.drop_micro({"open": two, "mid": "", "close": ""})["open"] == two          # the paragraph would fall under 3 sentences


def test_a_surplus_narrative_citation_is_turned_into_a_parenthetical_one_and_the_first_two_stay():
    mk = lambda who, y, tag: f"{who} ({y}) found that " + " ".join(f"{tag}{k}" for k in range(12)) + "."
    p = " ".join([mk("Smith", 2001, "aa"), mk("Lee and Park", 2002, "bb"), mk("Brown et al.", 2003, "cc"), mk("Diaz", 2004, "dd")])
    out = C.demote_narrative({"open": p, "mid": "", "close": ""})["open"]
    assert "(Brown et al., 2003)." in out and "(Diaz, 2004)." in out and out.count("found that") <= 2
    assert protect.compare(p, out) == [] and G._narrative_surplus({"open": out, "mid": "", "close": ""}) == []


def test_a_long_sentence_is_its_own_span_and_a_paragraph_is_accepted_when_it_breaks_fewer_rules(monkeypatch):
    p = para(30, 52, 8, 29, tag="a")
    spans = D.detect_ai_spans(p)
    assert [(s["kind"], s["types"]) for s in spans] == [("sentence", ["long_sentence"])] and "under 40 words" in spans[0]["instruction"]
    gamed = para(30, 4, 3, 36)                          # stacked fragments, 2 rules broken
    assert D.paragraph_violations(gamed) >= 1 and D.paragraph_violations(para(38, 7, 33)) == 0
    assert D.paragraph_violations(para(24, 27, 25, 28)) == 3 and D.paragraph_violations(para(40, 7, 36, 9, tag="b")) == 0   # flat, run, no short one
    assert D.paragraph_violations(para(24, 27, 6, 28, tag="c")) < D.paragraph_violations(para(24, 27, 25, 28))               # better, not perfect: accepted
