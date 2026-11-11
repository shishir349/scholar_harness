import json
import sqlite3

import pytest

from harness import config, graph as G, llm
from harness.corpus import lamp_bank as LB, store
from harness.text import lamp_rewrite as LR
from harness.validate import lamp_detector as D

from .test_graph import PROMPTS, clean, models, request  # noqa: F401  (clean: autouse fixture reused)


def _w(i):
    s, i = "", i + 1
    while i:
        s, i = chr(97 + (i - 1) % 26) + s, (i - 1) // 26
    return s


def sent(n, tag="alpha"):
    """A sentence of n distinct letter-only words (no digits, so number checks stay out of the way)."""
    return " ".join([tag.capitalize() + _w(0)] + [tag + _w(k) for k in range(1, n)]) + "."


def para(*lengths, tag="alpha"):
    return " ".join(sent(n, tag + _w(j)) for j, n in enumerate(lengths))


# ── detector ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_flat_paragraph_that_section_wide_checks_miss_is_flagged_and_a_bimodal_one_passes():
    flat = para(24, 27, 25, 28)
    ok = para(40, 7, 36, 9, tag="beta")
    spans = D.detect_ai_spans(flat + "\n\n" + ok)
    assert [(s["kind"], s["para"]) for s in spans] == [("paragraph", 0)] and "flat_paragraph" in spans[0]["types"]
    assert "42 words" in spans[0]["instruction"] and "[24, 27, 25, 28]" in spans[0]["instruction"]


def test_three_sentences_within_five_words_of_each_other_are_flagged_even_with_a_high_overall_spread():
    p = para(45, 8, 20, 24, 22)               # std is large, but 20, 24, 22 run together
    assert D.flat_note([45, 8, 20, 24, 22]) == "sentences 3-5 run 20, 24, 22"
    assert D.paragraph_is_flat(p) and not D.paragraph_is_flat(para(40, 7, 36, 9)) and not D.paragraph_is_flat(para(20, 21))


def test_each_sentence_level_artifact_is_found():
    long, short = sent(40, "lead"), sent(6, "tail")
    cases = {
        "epilogue_closer": "Taken together, these results point to a field that is still settling on its terms and tools.",
        "symmetric_hedge": "While the first design reached more sites, the second design reached them with less cost and more care.",
        "participial_tail": "The sensors drifted over the winter, thereby reducing the usable record to a few weeks of clean data.",
        "vague_attribution": "The effect of tillage on carbon storage has been extensively studied in temperate soils and elsewhere.",
        "nominalization_stack": "The implementation of the assessment of the adaptation of the organisation required considerable commitment.",
    }
    for kind, s in cases.items():
        text = long + " " + short + " " + s if kind == "epilogue_closer" else long + " " + s + " " + short
        got = [sp for sp in D.detect_ai_spans(text) if kind in sp["types"]]
        assert got and got[0]["kind"] == "sentence" and got[0]["text"] == s, kind
    mono = "The plots dried early in July after the rains stopped. The team measured moisture at forty sites across the region. The data showed a clear split between the wettest and driest plots."
    assert any("rhythm_monotony" in sp["types"] for sp in D.detect_ai_spans(mono))


def test_a_closer_that_ends_on_a_figure_or_citation_and_a_participle_like_including_are_not_flagged():
    long = sent(40, "lead")
    assert not D.detect_ai_spans(long + " " + sent(6, "mid") + " Overall, 94% of nodes survived the winter.")
    assert not D.detect_ai_spans(long + " The kit held several parts, including a spare battery for the field team. " + sent(6, "tail"))


def test_sentences_the_claim_fixer_wrote_are_always_queued_first():
    ok = para(40, 7, 36, 9)
    fixed = sent(18, "fixer")
    text = para(24, 27, 25, 28, tag="flat") + "\n\n" + ok + " " + fixed.replace("Fixer", "Fixer") + "\n\n" + para(40, 7, 36, 9, tag="zeta")
    spans = D.detect_ai_spans(text, patched=[fixed])
    assert spans[0]["types"][0] == "haiku_verify_patch" and spans[0]["text"] == fixed
    assert spans[1]["kind"] == "paragraph"


def test_headings_figure_blocks_and_tables_are_skipped():
    text = "# Heading\n\n[FIGURE_START] " + para(24, 27, 25, 28) + " [FIGURE_END]\n\n| a | b |\n| c | d |"
    assert D.detect_ai_spans(text) == []


# ── span bank ───────────────────────────────────────────────────────────────────────────────────────────────────────
PAIR = sent(34, "dense") + " " + sent(7, "blunt")[:-1] + " (Lee, 2015)."


def test_pairs_in_finds_long_then_blunt_and_abrupt_close_only_when_clean():
    kinds = {k for k, _, _ in LB.pairs_in(PAIR)}
    assert kinds == {"long_then_blunt"}
    close = sent(30, "aa") + " " + sent(30, "bb") + " " + sent(8, "cc")[:-1] + " in Table 3."
    assert "abrupt_close" in {k for k, _, _ in LB.pairs_in(close)}
    assert not list(LB.pairs_in(sent(34, "dense") + " See https://x.org now for more notes here."))
    assert not list(LB.pairs_in(sent(34, "dense") + " " + "The result is shown in Fig 2 here now."))


def test_build_keeps_each_type_to_five_examples_from_separate_theses_and_skips_holdouts():
    con = store.connect(":memory:")
    ids = [store.upsert_thesis(con, dict(openalex_id=f"W{i}", title="T", field_id="1", field_name="F", score=50 + i, holdout=(i == 0)))
           for i in range(8)]
    for tid in ids:
        store.insert_chunks(con, tid, "F", [dict(section_type="introduction", position="open", heading="Intro", chunk_idx=0,
                                                 text=PAIR.replace("dense", f"dense{_w(tid)}"))])
    low = store.upsert_thesis(con, dict(openalex_id="WL", title="L", field_id="1", field_name="F", score=1))
    store.insert_chunks(con, low, "F", [dict(section_type="introduction", position="open", heading="x", chunk_idx=0, text=PAIR)])
    ex = LB.build(con, min_score=4)
    assert len(ex) == 3 and {e["kind"] for e in ex} == {"long_then_blunt"}            # three per kind; the other kind found none
    assert len({e["thesis_id"] for e in ex}) == 3 and ids[0] not in {e["thesis_id"] for e in ex} and low not in {e["thesis_id"] for e in ex}


def test_the_shipped_bank_has_25_examples_across_the_five_types():
    ex = LB.load(str(config.DATA / "lamp_span_examples.json"))
    if not ex:
        pytest.skip("no data/lamp_span_examples.json")
    assert len(ex) == 25
    assert {e["section_type"] for e in ex} == set(LB.TYPES)
    assert all(len(e["sentences"]) == 2 and e["lengths"] == [len(s.split()) for s in e["sentences"]] for e in ex)
    assert len(LB.pick(ex, "methodology", 8)) == 8 and LB.pick(ex, "methodology", 8)[0]["section_type"] == "methodology"


# ── rewrite ─────────────────────────────────────────────────────────────────────────────────────────────────────────
EX = [{"section_type": "introduction", "kind": "long_then_blunt", "sentences": [sent(34, "human"), sent(7, "human")], "lengths": [34, 7]}]
GOOD = para(40, 7, 36, 9, tag="newa")


def _ask(reply, seen=None):
    def ask(tier, system, user, **k):
        if seen is not None:
            seen.update(k, tier=tier, user=user)
        return reply
    return ask


def test_a_flat_paragraph_is_rewritten_in_one_sonnet_call_at_temperature_085(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "ask", _ask(json.dumps([GOOD]), seen))
    flat = para(24, 27, 25, 28)
    out, rep = LR.lamp_rewrite_section("# H\n\n" + flat, "introduction", EX, None)
    assert out == "# H\n\n" + GOOD and rep["applied"] == 1 and rep["rejected"] == []
    assert seen["tier"] == "sonnet" and seen["temperature"] == 0.85
    assert flat in seen["user"] and sent(34, "human") in seen["user"] and "Edit:" in seen["user"]


def test_one_bad_span_keeps_its_original_while_the_good_ones_land(monkeypatch):
    a, b, c, d = (para(24, 27, 25, 28, tag=t) for t in ("aa", "bb", "cc", "dd"))
    cited = "Smith (2010) measured it. " + b
    text = "\n\n".join([a, cited.replace("Smith (2010) measured it. ", ""), c, d])
    replies = [
        GOOD,                                                       # fine
        "Smith (2010) argued otherwise. " + para(40, 7, 36, 9),     # adds a citation
        para(24, 27, 25, 28, tag="again"),                          # still flat
        "One sentence.\n\nTwo paragraphs.",                         # splits the paragraph
    ]
    monkeypatch.setattr(llm, "ask", _ask(json.dumps(replies)))
    out, rep = LR.lamp_rewrite_section(text, "introduction", EX, None)
    assert rep["applied"] == 1 and len(rep["rejected"]) == 3 and out.split("\n\n")[0] == GOOD
    assert out.split("\n\n")[1:] == text.split("\n\n")[1:]
    whys = " | ".join(r["why"] for r in rep["rejected"])
    assert "citation changed" in whys and "still flat" in whys and "more than one paragraph" in whys


def test_numbers_quotes_and_citations_are_held_exactly(monkeypatch):
    base = para(40, 7, 36, 9, tag="keep")
    flat = para(24, 27, 25, 28, tag="num").replace("Numaa ", "Numaa 12 ", 1) + " (Lee, 2015)"
    for bad in (GOOD, GOOD + " (Lee, 2016)", GOOD + " (Lee, 2015) 13", "Only (Lee, 2015)."):
        monkeypatch.setattr(llm, "ask", _ask(json.dumps([bad])))
        out, rep = LR.lamp_rewrite_section(base + "\n\n" + flat, "introduction", EX, None)
        assert out == base + "\n\n" + flat and rep["applied"] == 0, bad
    good = GOOD.replace("Newaaa ", "Newaaa 12 ", 1) + " (Lee, 2015)"
    monkeypatch.setattr(llm, "ask", _ask(json.dumps([good])))
    out, rep = LR.lamp_rewrite_section(base + "\n\n" + flat, "introduction", EX, None)
    assert rep["applied"] == 1 and out.endswith(good)


def test_a_closing_summary_sentence_can_be_deleted_and_a_copy_of_an_example_cannot(monkeypatch):
    long, short = sent(40, "lead"), sent(6, "tail")
    closer = "Taken together, these results point to a field that is still settling on its terms and tools."
    text = long + " " + short + " " + closer
    monkeypatch.setattr(llm, "ask", _ask(json.dumps([""])))
    out, rep = LR.lamp_rewrite_section(text, "introduction", EX, None)
    assert out == long + " " + short and rep["applied"] == 1
    copy = sent(34, "human") + " " + sent(7, "human")
    monkeypatch.setattr(llm, "ask", _ask(json.dumps([copy])))
    out, rep = LR.lamp_rewrite_section(para(24, 27, 25, 28), "introduction", EX, None)
    assert rep["applied"] == 0 and rep["rejected"][0]["why"] in ("copies an example", "still flat")


def test_spans_are_capped_and_a_bad_reply_changes_nothing(monkeypatch):
    text = "\n\n".join(para(24, 27, 25, 28, tag=t) for t in ("a", "b", "c", "d", "e", "f", "g", "h"))
    seen = {}
    monkeypatch.setattr(llm, "ask", _ask("not json", seen))
    out, rep = LR.lamp_rewrite_section(text, "introduction", EX, None, max_spans=6)
    assert out == text and rep["flagged"] == 8 and rep["sent"] == 6 and "error" in rep
    assert LR.MAX_SPANS == 14
    assert seen["user"].count("(paragraph) Edit:") == 6
    monkeypatch.setattr(llm, "ask", _ask(json.dumps(["x"])))
    assert LR.lamp_rewrite_section(text, "introduction", EX, None)[0] == text            # wrong count
    assert LR.lamp_rewrite_section(GOOD, "introduction", EX, None)[1]["sent"] == 0       # nothing flagged: no call needed


# ── in the graph ────────────────────────────────────────────────────────────────────────────────────────────────────
ON = {"pool": True, "sources": False, "revise": False}


def _flat3():
    return ["\n\n".join(para(24, 27, 25, 28, tag=f"p{_w(k)}{_w(j)}") for k in range(2)) for j in range(3)]


def test_lamp_runs_after_the_drafts_before_the_free_rewrites_and_lands_in_the_report(monkeypatch):
    monkeypatch.setenv("HARNESS_LAMP", "on")
    flat = _flat3()
    spans = D.detect_ai_spans("\n\n".join(flat))
    models(flat + [json.dumps([para(40, 7, 36, 9, tag=f"r{_w(i)}") for i in range(len(spans))])])
    out = G.write_section(request(flags=ON), G.Deps(baseline={}, skeletons=[], lamp_examples=EX))
    assert len(spans) == 6 and out["reports"]["lamp"]["applied"] == 6 and out["reports"]["lamp"]["sent"] == 6
    assert "edits" not in out["reports"]["lamp"] and "paragraph) Edit:" in PROMPTS[3]
    assert all(not D.paragraph_is_flat(p) for p in out["text"].split("\n\n") if p.strip())


def test_lamp_can_be_switched_off_and_never_runs_on_scholars_own_path(monkeypatch):
    monkeypatch.setenv("HARNESS_LAMP", "off")
    models(_flat3())
    out = G.write_section(request(flags=ON), G.Deps(baseline={}, skeletons=[], lamp_examples=EX))
    assert out["reports"]["lamp"] == {"skipped": "off"} and len(PROMPTS) == 3
    monkeypatch.setenv("HARNESS_LAMP", "on")
    PROMPTS.clear()
    models(["a", "b", "c"])
    out = G.write_section(request(), G.Deps(baseline={}))
    assert "lamp" not in out["reports"] and len(PROMPTS) == 3


def test_a_model_failure_in_the_lamp_call_keeps_the_text(monkeypatch):
    monkeypatch.setenv("HARNESS_LAMP", "on")
    flat = _flat3()
    models(flat)                                       # no fourth reply: the fake raises StopIteration inside the call
    out = G.write_section(request(flags=ON), G.Deps(baseline={}, skeletons=[], lamp_examples=EX))
    assert "error" in out["reports"]["lamp"] and out["text"].strip()


def test_the_claim_fixer_rewrites_are_handed_to_lamp(monkeypatch):
    seen = []
    monkeypatch.setattr(LR, "lamp_rewrite_section",
                        lambda text, typ, ex, spend=None, patched=(): seen.append(list(patched)) or (text, {"applied": 0, "edits": []}))
    monkeypatch.setenv("HARNESS_LAMP", "on")
    state = {"flags": ON, "profile": {"id": "introduction"}, "reports": {"claims": {"patched_sentences": ["A fixed one."]}}}
    G._lamp_tail({"open": "a", "mid": "b", "close": "c"}, state, G.Deps(lamp_examples=EX))
    assert seen == [["A fixed one."]]


def test_drafts_run_at_temperature_08_and_the_claim_checks_stay_at_zero(monkeypatch):
    temps = []
    real = llm.chat
    monkeypatch.setattr(G.llm, "chat", lambda tier, **k: temps.append((tier, k.get("temperature"))) or real(tier, **k))
    models(["a", "b", "c"])
    G.write_section(request(flags=ON), G.Deps(baseline={}, skeletons=[]))
    assert temps[:3] == [("sonnet", 0.8)] * 3
    import inspect
    from harness.sources import verifier
    assert "temperature=0)" in inspect.getsource(verifier)


def test_a_sentence_the_haiku_fixer_wrote_goes_through_lamp_and_keeps_its_citation(monkeypatch):
    from .test_graph import SOURCES
    monkeypatch.setenv("HARNESS_LAMP", "on")
    claim = "Cloud firms doubled their output (Smith, 2019)."
    models([claim + "\n\n", "B one.\n\n", "C one.", json.dumps(["Cloud firms grew faster in this sample (Smith, 2019)."])],
           haiku=['[{"id": 1, "verdict": "UNSUPPORTED", "reason": "abstract only says faster"}]',
                  '["Cloud firms grew faster (Smith, 2019)."]',
                  '[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'])
    out = G.write_section(request(flags={"pool": True, "sources": True, "revise": True}, ledger={"sources": SOURCES, "by_section": {}}),
                          G.Deps(baseline={}, skeletons=[], lamp_examples=EX))
    assert out["reports"]["claims"]["patched_sentences"] == ["Cloud firms grew faster (Smith, 2019)."]
    assert out["reports"]["lamp"]["applied"] == 1 and out["reports"]["lamp"]["kinds"]["haiku_verify_patch"] == 1
    assert "Cloud firms grew faster in this sample (Smith, 2019)." in out["text"]
