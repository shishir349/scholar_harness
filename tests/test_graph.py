import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import graph as G
from harness import llm
from harness.corpus import retrieve, store
from harness.spend import BudgetExceeded, Spend

GOLD = json.loads((Path(__file__).parent / "golden" / "cases.json").read_text(encoding="utf-8"))["cases"][0]
PROMPTS: list[str] = []


class Rec(GenericFakeChatModel):
    def _generate(self, messages, *a, **k):
        PROMPTS.append("\n\n".join(llm.text_of(m) for m in messages))   # cached system block + user prompt
        return super()._generate(messages, *a, **k)


@pytest.fixture(autouse=True)
def clean():
    PROMPTS.clear()
    retrieve.clear_cache()
    yield
    llm.set_fake(None)


def models(sonnet: list[str], haiku: list[str] | None = None):
    s = Rec(messages=iter([AIMessage(content=t) for t in sonnet]))
    h = Rec(messages=iter([AIMessage(content=t) for t in (haiku or [])]))
    llm.set_fake(lambda tier: s if tier == "sonnet" else h)


def request(**over):
    chapter = {**GOLD["chapter"], "estimated_pages": 5}
    section = {**GOLD["section"], "pages": 5.39}       # -> words_target 450, same as the golden prompts
    base = {"thesis": GOLD["thesis"], "chapter": chapter, "section": section, "references": GOLD["references"],
            "ledger": {}, "flags": {"pool": False, "sources": False, "revise": True}, "n_sections": 1}
    base.update(over)
    return base


def tenpad(s):  # >=150 words so style/overlap can measure, while keeping the golden chunk text as the prefix
    return s


def test_words_target_matches_scholar_sizing():
    assert G.words_target({"estimated_pages": 9}, {}, 3) == (250, 400)          # 3 pages -> 750 words -> 250/chunk
    assert G.words_target({"estimated_pages": 100}, {"pages": 40}, 3)[0] == 700  # clamped high
    assert G.words_target({"estimated_pages": 1}, {}, 10)[0] == 200              # clamped low


def test_no_flags_prompts_equal_scholars_and_text_is_concatenation():
    models([GOLD["chunk1"], GOLD["chunk2"], "THIRD"])
    out = G.write_section(request(), G.Deps(baseline={}))
    assert PROMPTS[0] == GOLD["call1"]
    assert PROMPTS[1] == GOLD["call2"]
    assert PROMPTS[2].startswith(GOLD["call3"][:300]) and GOLD["chunk1"] in PROMPTS[2]
    assert out["text"] == GOLD["chunk1"] + GOLD["chunk2"] + "THIRD" and out["revisions"] == 0
    assert len(PROMPTS) == 3                                            # no verifier call without abstracts


def _pool():
    con = store.connect(":memory:")
    tid = store.upsert_thesis(con, dict(openalex_id="W1", title="T", field_id="1", field_name="Business", score=90))
    for pos, txt in (("open", "OPEN-EXEMPLAR words"), ("mid", "MID-EXEMPLAR words"), ("close", "CLOSE-EXEMPLAR words")):
        store.insert_chunks(con, tid, "Business", [dict(section_type="introduction", position=pos, heading="H", chunk_idx=0, text=txt)])
    con.commit()
    return con


def test_exemplars_go_to_the_matching_call_only_when_pool_flag_on():
    models(["a", "b", "c"])
    G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(con=_pool(), baseline={}))
    assert "OPEN-EXEMPLAR" in PROMPTS[0] and "MID-EXEMPLAR" not in PROMPTS[0]
    assert "MID-EXEMPLAR" in PROMPTS[1] and "CLOSE-EXEMPLAR" in PROMPTS[2]
    PROMPTS.clear()
    models(["a", "b", "c"])
    G.write_section(request(flags={"pool": False}), G.Deps(con=_pool(), baseline={}))
    assert not any("EXEMPLAR" in p for p in PROMPTS)


def test_retrieval_error_means_no_exemplars_not_a_failure():
    class Boom:
        def execute(self, *a, **k):
            import sqlite3
            raise sqlite3.OperationalError("x")
    models(["a", "b", "c"])
    out = G.write_section(request(flags={"pool": True}), G.Deps(con=Boom(), baseline={}))
    assert out["text"] == "a b c" and not any("EXEMPLAR" in p for p in PROMPTS)


SOURCES = [{"ledger_id": "S1", "openalex_id": "W1", "title": "Cloud growth", "year": 2019, "authors": "Smith, J.", "doi": "10.1/s",
            "abstract": "Firms that adopted cloud tools grew faster.", "cited_by": 9, "apa": "Smith, J. (2019). Cloud growth."}]


def test_ledger_adds_references_notes_and_rules_to_every_call():
    models(["a (Smith, 2019).", "b", "c"], haiku=['[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'])
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": True},
                                  ledger={"sources": SOURCES, "by_section": {GOLD["section"]["number"]: ["S1"]}}), G.Deps(baseline={}))
    for p in PROMPTS[:3]:
        assert "CITATION RULES" in p and "SOURCE NOTES" in p
    assert "Smith, J. (2019). Cloud growth." in PROMPTS[0]           # ledger APA line merged into REFERENCES
    assert "REFERENCES (cite these;" in PROMPTS[1] and "REFERENCES (cite these;" in PROMPTS[2]
    assert out["reports"]["claims"]["stats"]["supported"] == 1 and out["revisions"] == 0


def test_unsupported_claim_sentence_is_rewritten_in_place_on_the_cheap_model():
    claim = "Cloud firms doubled their output (Smith, 2019)."
    models([claim + "\n\n", "B one.\n\n", "C one."],
           haiku=['[{"id": 1, "verdict": "UNSUPPORTED", "reason": "abstract only says faster"}]',
                  '["Cloud firms grew faster (Smith, 2019)."]',
                  '[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'])
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": True},
                                  ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}))
    assert out["text"] == "Cloud firms grew faster (Smith, 2019).\n\nB one.\n\nC one."
    assert out["reports"]["claims"]["fixed_unsupported"] == 1 and len(PROMPTS) == 6   # 3 drafts + verify + fix + re-verify


def test_unsupported_claim_left_alone_when_revise_is_off():
    claim = "Cloud firms doubled their output (Smith, 2019)."
    models([claim + "\n\n", "B one.\n\n", "C one."], haiku=['[{"id": 1, "verdict": "UNSUPPORTED", "reason": "r"}]'])
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": False},
                                  ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}))
    assert out["text"] == claim + "\n\nB one.\n\nC one." and len(PROMPTS) == 4
    assert "fixed_unsupported" not in out["reports"]["claims"]



def test_unmatched_citation_is_patched_in_place_without_a_full_rewrite():
    """The bad citation sits in one paragraph (the 'open' chunk): a single patch call fixes just that
    paragraph, mid/close are never touched or regenerated."""
    ghost = "Claims (Ghost, 2015) appear here."
    models([ghost, "mid1", "close1"],                                 # first attempt
           haiku=['["Claims appear here without the bad citation."]'])  # patch reply: one flagged paragraph
    out = G.write_section(request(flags={"sources": True, "revise": True}, ledger={"sources": SOURCES}), G.Deps(baseline={}))
    assert out["revisions"] == 1
    assert out["text"] == "Claims appear here without the bad citation.\n\nmid1 close1"
    assert len(PROMPTS) == 4                                          # 3 drafting calls + 1 patch call


def test_patch_failure_falls_back_to_the_full_chunk_rewrite():
    ghost = "Claims (Ghost, 2015) appear here."
    models([ghost, "mid1", "close1", "open2 clean", "mid2", "close2"],
           haiku=["not valid json"])                                  # patch reply unparseable -> fallback
    out = G.write_section(request(flags={"sources": True, "revise": True}, ledger={"sources": SOURCES}), G.Deps(baseline={}))
    assert out["revisions"] == 1 and out["text"] == "open2 clean mid2 close2"
    revise_prompt = PROMPTS[4]
    assert "PREVIOUS ATTEMPT PROBLEMS" in revise_prompt and "(Ghost, 2015)" in revise_prompt
    assert "PREVIOUS ATTEMPT PROBLEMS" not in PROMPTS[5] and len(PROMPTS) == 7


def test_second_failure_does_not_loop():
    ghost = "Claims (Ghost, 2015) appear here."
    models([ghost, "m", "c", ghost, "m2", "c2"], haiku=["not json"])
    out = G.write_section(request(flags={"sources": True, "revise": True}, ledger={"sources": SOURCES}), G.Deps(baseline={}))
    assert out["revisions"] == 1 and len(PROMPTS) == 7 and out["reports"]["citations"]["unmatched"] == ["(Ghost, 2015)"]


def test_revise_flag_off_skips_revision():
    models(["Claims (Ghost, 2015).", "m", "c"])
    out = G.write_section(request(flags={"sources": True, "revise": False}, ledger={"sources": SOURCES}), G.Deps(baseline={}))
    assert out["revisions"] == 0 and len(PROMPTS) == 3


def test_copied_exemplar_wording_is_patched_in_place():
    copied = "the quick brown fox jumps over the lazy dog while the cat watches quietly from the old wooden fence nearby"
    con = store.connect(":memory:")
    tid = store.upsert_thesis(con, dict(openalex_id="W1", title="T", field_id="1", field_name="Business", score=90))
    store.insert_chunks(con, tid, "Business", [dict(section_type="introduction", position="open", heading="H", chunk_idx=0, text=copied)])
    con.commit()
    models(["Intro " + copied, "m", "c"], haiku=['["Intro entirely original wording here."]'])
    out = G.write_section(request(flags={"pool": True, "revise": True}), G.Deps(con=con, baseline={}))
    assert out["revisions"] == 1 and out["text"] == "Intro entirely original wording here.\n\nm c"
    assert out["reports"]["overlap"]["flagged"] is False
    assert len(PROMPTS) == 4


def test_humanize_pass_rewrites_all_three_chunks_in_one_call():
    models(["openp", "midp", "closep", '{"open": "OPEN-H", "mid": "MID-H", "close": "CLOSE-H"}'])
    out = G.write_section(request(flags={"pool": True, "humanize": True, "revise": False}), G.Deps(con=_pool(), baseline={}))
    assert out["text"] == "OPEN-H MID-H CLOSE-H" and len(PROMPTS) == 4


def test_humanize_off_by_default_and_skipped_without_style_signal():
    models(["a", "b", "c"])
    out = G.write_section(request(flags={"pool": False, "humanize": True, "revise": False}), G.Deps(baseline={}))
    assert out["text"] == "abc" and len(PROMPTS) == 3           # no exemplars, no baseline: nothing to match, skip


def test_humanize_skips_the_rewrite_when_the_draft_already_reads_fine(monkeypatch):
    monkeypatch.setattr(G, "score_section", lambda text, ptype, baseline: {"available": True, "passed": True})
    models(["a", "b", "c"])
    out = G.write_section(request(flags={"pool": True, "humanize": True, "revise": False}), G.Deps(con=_pool(), baseline={}))
    assert out["text"] == "a b c" and len(PROMPTS) == 3           # already passes: no rewrite call spent on it


def test_humanize_bad_reply_falls_back_to_original_chunks():
    models(["openp", "midp", "closep", "not json"])
    out = G.write_section(request(flags={"pool": True, "humanize": True, "revise": False}), G.Deps(con=_pool(), baseline={}))
    assert out["text"] == "openp midp closep" and len(PROMPTS) == 4


def test_budget_exceeded_surfaces():
    class Over(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise BudgetExceeded("over")
    llm.set_fake(lambda tier: Over(messages=iter([])))
    with pytest.raises(BudgetExceeded):
        G.write_section(request(), G.Deps(baseline={}))


def test_verifier_exception_does_not_lose_the_text(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("verifier down")
    monkeypatch.setattr(G, "audit_chapter", boom)
    models(["a", "b", "c"])
    out = G.write_section(request(), G.Deps(baseline={}))
    assert out["text"] == "abc" and "verifier down" in out["reports"]["claims"]["error"]


def test_spend_counts_graph_calls():
    s = Spend(budget_usd=5)
    models(["a", "b", "c"])
    # real spend accounting is exercised in test_spend; here just make sure deps carries it into chat()
    seen = []
    orig = llm.chat
    llm.chat = lambda tier, **kw: (seen.append(kw.get("spend")), orig(tier, **kw))[1]
    try:
        G.write_section(request(), G.Deps(spend=s, baseline={}))
    finally:
        llm.chat = orig
    assert seen and all(x is s for x in seen)


def test_tidy_replaces_em_dashes_and_keeps_ranges_and_paragraph_breaks():
    assert G._tidy("Firms grew \u2014 slowly \u2014 in 2019.") == "Firms grew, slowly, in 2019."
    assert G._tidy("word\u2014word and 1990\u20132000 and a \u2013 b") == "word, word and 1990\u20132000 and a, b"
    assert G._tidy("End.\n\n\u2014 Next").startswith("End.\n\n")          # never swallows a paragraph break
    assert G._tidy("stop \u2014.") == "stop."


def test_dash_cleanup_and_voice_rules_only_with_the_harness_flag(monkeypatch):
    monkeypatch.setattr(G, "style_target", lambda ptype, baseline: "STYLE TARGET x")
    models(["Firms grew \u2014 slowly.", "b.", "c."])
    out = G.write_section(request(flags={"pool": True, "revise": False}), G.Deps(baseline={}))
    assert "\u2014" not in out["text"] and all("VOICE RULES" in p and "STYLE TARGET x" in p for p in PROMPTS[:3])
    PROMPTS.clear()
    models(["Firms grew \u2014 slowly.", "b.", "c."])
    plain = G.write_section(request(flags={"pool": False, "revise": False}), G.Deps(baseline={}))
    assert "\u2014" in plain["text"] and not any("VOICE RULES" in p or "STYLE TARGET" in p for p in PROMPTS)


def test_a_pattern_of_formulaic_phrases_triggers_a_cheap_targeted_patch_but_one_does_not():
    two = ["The problem becomes clearer when firms scale.\n\n", "It is important to note that scale matters.\n\n", "Done."]
    models(two, haiku=['["Scale changes costs.", "Scale matters for cost."]'])
    out = G.write_section(request(flags={"pool": False, "revise": True}), G.Deps(baseline={}))
    assert out["revisions"] == 1 and out["text"] == "Scale changes costs.Scale matters for cost.Done."
    assert len(PROMPTS) == 4                                             # 3 drafts + 1 patch on the cheap tier
    PROMPTS.clear()
    models(["The problem becomes clearer when firms scale.\n\n", "Plain point.\n\n", "Done."])
    one = G.write_section(request(flags={"pool": False, "revise": True}), G.Deps(baseline={}))
    assert one["revisions"] == 0 and len(PROMPTS) == 3                   # a single stock phrase is left alone


def test_formulaic_patterns_catch_the_phrases_the_judge_called_out():
    for phrase in ("The picture becomes more complex still", "Furthermore, firms grew", "plays a crucial role in",
                   "not only faster but also cheaper", "remains genuinely contested", "does not speak with one voice"):
        assert G._FORMULAIC.search(phrase), phrase
    assert not G._FORMULAIC.search("Firms grew, and Kahn (1990) explains why engagement varies.")


KAHN = {"openalex_id": "W99", "title": "Psychological conditions of personal engagement", "year": 1990,
        "authors": "Kahn, W. A.", "abstract": "Engagement varies with psychological safety.", "cited_by": 9000,
        "apa": "Kahn, W. A. (1990). Psychological conditions of personal engagement."}


def test_a_citation_from_the_writers_own_knowledge_is_looked_up_and_adopted_when_real():
    seen = []

    def lookup(surname, year, hint):
        seen.append((surname, year))
        return KAHN if (surname, year) == ("Kahn", 1990) else None
    models(["Engagement varies with safety (Kahn, 1990).\n\n", "B one.\n\n", "C one."],
           haiku=['[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'])
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": True},
                                  ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}, lookup=lookup))
    assert seen == [("Kahn", 1990)] and out["reports"]["citations"]["unmatched"] == []
    assert out["revisions"] == 0 and [r["title"] for r in out["added_refs"]] == [KAHN["title"]]
    assert out["added_refs"][0]["resolved"] is True and len(PROMPTS) == 4      # 3 drafts + claim check, no fix-up


def test_a_citation_that_cannot_be_verified_still_goes_through_the_normal_fix():
    ghost = "Claims (Ghost, 2015) appear here."
    for lookup in (lambda *a: None, lambda *a: (_ for _ in ()).throw(RuntimeError("openalex down"))):
        PROMPTS.clear()
        models([ghost, "mid1", "close1"], haiku=['["Claims appear here without the bad citation."]'])
        out = G.write_section(request(flags={"sources": True, "revise": True}, ledger={"sources": SOURCES}),
                              G.Deps(baseline={}, lookup=lookup))
        assert out["revisions"] == 1 and out["added_refs"] == [] and len(PROMPTS) == 4


def test_no_lookup_without_the_sources_flag_or_a_lookup_function():
    calls = []
    models(["Claims (Ghost, 2015) appear here.", "b", "c"])
    G.write_section(request(flags={"sources": False, "revise": False}), G.Deps(baseline={}, lookup=lambda *a: calls.append(a)))
    assert calls == []


def _even(n):
    return " ".join(["word"] * n) + "."


def _stats(para_mean, fp_mean):
    f = {"para_len_cv": {"mean": para_mean, "std": 0.5}, "first_person_per_100": {"mean": fp_mean, "std": 0.5}}
    return {"by_section": {"all": {"n": 1, "features": f}}, "score_p10": {}}


def test_even_paragraphs_get_merged_for_free_when_variation_is_below_the_real_floor():
    chunks = ["## 1.1 H\n\n" + _even(60) + "\n\n" + _even(60) + "\n\n", _even(60) + "\n\n" + _even(60) + "\n\n",
              _even(60) + "\n\n" + _even(60)]
    models(chunks)
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline=_stats(0.7, 0.0)))
    assert out["text"].count("\n\n") < "".join(chunks).count("\n\n") and len(PROMPTS) == 3     # merged, zero extra calls


def test_missing_first_person_triggers_one_cheap_patch_not_a_rewrite():
    chunks = [_even(60) + "\n\n" + _even(60) + "\n\n", _even(60) + "\n\n", _even(60)]
    models(chunks, haiku=['["I argue this holds.", "I take the opposite view."]'])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(baseline=_stats(0.01, 0.6)))
    assert out["revisions"] == 1 and "I argue this holds." in out["text"] and len(PROMPTS) == 4


def test_paragraph_plan_and_cached_shared_block_and_three_exemplars():
    models(["a", "b", "c"])
    G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(con=_pool(), baseline={}))
    assert all("PARAGRAPH PLAN" in p and "VOICE RULES" in p for p in PROMPTS[:3]) and G.N_EXEMPLARS == 3



# --- humanizer-derived AI tells (https://github.com/blader/humanizer, patterns 1-5, 12, 13, 15, 18) ---------------
@pytest.mark.parametrize("text", [
    "It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere.",
    "This does not mean every choice is equal. It means there is no external system that confirms which is right.",
    "Caching cuts repeat work. That is the real win.",
    "The real question is whether teams can adapt. At its core, readiness matters.",
    "The gallery serves as an exhibition space and boasts four rooms.",
    "These sensors showcase a pivotal and intricate interplay between cost and coverage.",
    "Costs fell sharply, highlighting the importance of early planning.",
    "Despite these challenges, the field continues to thrive.",
    "To be clear, I am not claiming the method is new.",
])
def test_tell_count_catches_humanizer_patterns(text):
    assert G.tell_count(text) >= 1


@pytest.mark.parametrize("text", [
    "Strain gauges measure local deformation (Farrar & Worden, 2007), and fibre sensors extend that range.",
    "I chose a mixed design because the survey alone could not explain the interview results.",
    "Accelerometers are cheap but drift; displacement transducers are accurate but need a fixed reference.",
])
def test_tell_count_leaves_plain_academic_prose_alone(text):
    assert G.tell_count(text) == 0


def _cited(n):
    names = ["Smith", "Jones", "Lee", "Chen", "Patel", "Kim", "Brown", "Garcia"]
    return " ".join(f"One study ({names[i]}, 20{10 + i}) found an effect." for i in range(n))


def test_too_many_citations_trigger_one_cheap_patch_not_a_rewrite():
    # 8 citations in ~70 words, real mean 0.1 per 100 words -> far over the 1.5x ceiling
    heavy = _cited(8) + " " + " ".join(["word"] * 40) + "."
    chunks = [heavy + "\n\n" + _even(60) + "\n\n", _even(60) + "\n\n", _even(60)]
    models(chunks, haiku=['["Several studies found an effect (Smith, 2010; Jones, 2011)."]'])
    base = _stats(0.01, 0.0)
    base["by_section"]["all"]["features"]["citation_per_100"] = {"mean": 0.1, "std": 0.5}
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(baseline=base))
    assert out["revisions"] == 1 and "Several studies found an effect" in out["text"] and len(PROMPTS) == 4
    assert "too many sources" in PROMPTS[3] and "REFERENCES" not in PROMPTS[3]       # asks to merge/trim, never to add sources



# --- judge-cue slice: tidy triads, signposted transitions, source-by-source chains, chunk seams -------------------
@pytest.mark.parametrize("text", [
    "The appeal is obvious: small form factor, low power draw, and integration with cheap microcontrollers.",
    "What these demonstrations share is that they used one structure under one set of conditions.",
    "This work opens a path toward continuous monitoring.",
])
def test_tell_count_catches_judge_cues_found_in_a_real_probe(text):
    assert G.tell_count(text) >= 1


def test_tell_count_ignores_an_ordinary_list_inside_a_sentence():
    assert G.tell_count("Sensors, cables, and loggers were bought from three suppliers over two years.") == 0


def test_chunks_are_glued_with_a_paragraph_break_or_a_space_when_the_harness_is_on():
    models(["First chunk ends here.", "Second chunk starts here and stops at remains", "largely unnoticed."])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}))
    assert out["text"] == "First chunk ends here.\n\nSecond chunk starts here and stops at remains largely unnoticed."
    PROMPTS.clear()
    models(["A.", "B.", "C."])                                           # plain Scholar path keeps its exact concatenation
    assert G.write_section(request(), G.Deps(baseline={}))["text"] == "A.B.C."


def test_a_run_of_sentences_that_each_open_with_a_source_gets_one_cheap_patch():
    chain = ("Smith (2010) found an effect. Jones (2011) extended it. Lee et al. (2012) confirmed it. "
             "Chen & Patel (2013) disputed it. " + " ".join(["word"] * 30) + ".")
    chunks = [chain + "\n\n" + _even(60) + "\n\n", _even(60) + "\n\n", _even(60)]
    models(chunks, haiku=['["Several studies agree on an effect (Smith, 2010; Jones, 2011; Lee et al., 2012), though Chen and Patel (2013) dispute it."]'])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(baseline=_stats(0.01, 0.0)))
    assert out["revisions"] == 1 and "Several studies agree" in out["text"] and len(PROMPTS) == 4
    assert "make the researcher the subject" in PROMPTS[3]


def test_failed_patch_for_stock_phrases_never_escalates_to_a_sonnet_rewrite():
    two = ["The problem becomes clearer when firms scale.\n\n", "It is important to note that scale matters.\n\n", "Done."]
    models(two, haiku=["not json"])
    out = G.write_section(request(flags={"pool": False, "revise": True}), G.Deps(baseline={}))
    assert out["revisions"] == 1 and len(PROMPTS) == 4                    # 3 drafts + the failed patch, no rewrite



# --- ChatGPT review slice: no fixed claim -> citation -> limitation -> gap template, varied paragraph moves -------
@pytest.mark.parametrize("text", [
    "The problem emerges when sensors are deployed at scale.",
    "A first sentence.\n\nThis study takes the position that cost is secondary.",
    "The gap between laboratory and field performance is large.",
])
def test_tell_count_catches_the_template_openers_chatgpt_flagged(text):
    assert G.tell_count(text) >= 1


def test_voice_rules_do_not_prescribe_a_claim_to_implication_template():
    assert "Take a stance early" not in G.VOICE_RULES and "close by saying what it implies" not in G.VOICE_RULES
    assert "at least one candid limitation" not in G.VOICE_RULES
    assert "evidence" in G.VOICE_RULES and "do not force" in G.VOICE_RULES.lower()


def test_paragraph_plan_gives_each_paragraph_its_own_move_and_never_repeats_one_in_a_passage():
    models(["a", "b", "c"])
    G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}))
    plan = PROMPTS[0].split("PARAGRAPH PLAN")[1]
    moves = [line.split(": ", 1)[1] for line in plan.splitlines() if line.strip()[:2] in {f"{i}." for i in range(1, 8)}]
    assert len(moves) >= 2 and len(set(moves)) == len(moves)



# --- made-up claims -> zero: re-check every rewrite, delete what still cannot be backed -----------------------------
UNSUP = '[{"id": 1, "verdict": "UNSUPPORTED", "reason": "r"}]'
SUPP = '[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'
CLAIM = "Cloud firms doubled their output (Smith, 2019)."


def _claim_run(haiku):
    models([CLAIM + " Plain context sentence.\n\n", "B one.\n\n", "C one."], haiku=haiku)
    return G.write_section(request(flags={"pool": False, "sources": True, "revise": True},
                                   ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}))


def test_a_sentence_that_is_still_unsupported_after_the_rewrite_is_deleted():
    out = _claim_run([UNSUP, '["Cloud firms outgrew peers (Smith, 2019)."]', UNSUP])        # check, rewrite, re-check
    c = out["reports"]["claims"]
    assert "Smith" not in out["text"] and "Plain context sentence." in out["text"] and "B one." in out["text"]
    assert c["fixed_unsupported"] == 1 and c["dropped_unsupported"] == 1 and c["stats"]["unsupported"] == 0


def test_a_rewrite_that_passes_the_re_check_is_kept_and_nothing_is_deleted():
    out = _claim_run([UNSUP, '["Cloud firms grew faster (Smith, 2019)."]', SUPP])
    c = out["reports"]["claims"]
    assert "Cloud firms grew faster (Smith, 2019)." in out["text"] and c.get("dropped_unsupported", 0) == 0
    assert c["stats"]["unsupported"] == 0


def test_a_failed_rewrite_reply_goes_straight_to_deletion():
    out = _claim_run([UNSUP, "not json"])                                                    # no claims left, so no re-check call
    c = out["reports"]["claims"]
    assert "Smith" not in out["text"] and c["dropped_unsupported"] == 1 and c["stats"]["unsupported"] == 0


def test_deletion_finds_a_sentence_even_when_whitespace_differs():
    models(["Cloud firms\n doubled  their output (Smith, 2019). Kept.\n\n", "B.\n\n", "C."], haiku=[UNSUP, "not json"])
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": True},
                                  ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}))
    assert "Smith" not in out["text"] and "Kept." in out["text"]


def test_claim_checks_run_at_temperature_zero_so_the_same_sentence_gets_the_same_verdict(monkeypatch):
    from harness.sources import verifier
    seen = {}
    monkeypatch.setattr(verifier.llm, "ask", lambda tier, system, user, **k: seen.update(k) or "[]")
    verifier.verify_claims([{"id": 1, "sentence": "s", "ref": {"abstract": "a"}}])
    assert seen["temperature"] == 0


def test_chat_passes_temperature_only_when_asked(monkeypatch):
    import langchain_anthropic
    got = []
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **k: got.append(k) or object())
    llm.chat("haiku", temperature=0)
    llm.chat("haiku")
    assert got[0]["temperature"] == 0 and "temperature" not in got[1]


def test_new_refs_lists_only_cited_works_that_are_not_in_the_library():
    other = {**SOURCES[0], "ledger_id": "S2", "openalex_id": "W2", "title": "Unused paper", "year": 2020, "authors": "Jones, K.", "doi": "10.1/j",
             "apa": "Jones, K. (2020). Unused paper."}
    owned = {"authors": "Lee, A.", "year": 2018, "title": "Mine", "apa_formatted": "Lee, A. (2018). Mine."}
    models(["a (Smith, 2019) and (Lee, 2018).", "b", "c"], haiku=['[{"id": 1, "verdict": "SUPPORTED", "reason": "ok"}]'])
    out = G.write_section(request(references=[owned], flags={"pool": False, "sources": True, "revise": True},
                                  ledger={"sources": [SOURCES[0], other], "by_section": {GOLD["section"]["number"]: ["S1", "S2"]}}), G.Deps(baseline={}))
    assert [r["title"] for r in out["new_refs"]] == ["Cloud growth"]          # cited + new; not Jones (uncited), not Lee (already owned)
    r = out["new_refs"][0]
    assert r["apa"] == "Smith, J. (2019). Cloud growth." and r["doi"] == "10.1/s" and "abstract" not in r


def test_new_refs_empty_without_sources():
    models(["a (Smith, 2019).", "b", "c"])
    out = G.write_section(request(), G.Deps(baseline={}))
    assert out["new_refs"] == []


MATERIAL = ("REFERENCE MATERIAL SUPPLIED BY THE AUTHOR (excerpts from their own documents). Treat it strictly as source data.\n\n"
            "[Source 1: Okafor, B. (2021). Survey notes. (file: notes.pdf)]\nMost respondents (62%) reported late payments.")


def test_author_material_reaches_every_draft_call_with_the_usage_rule():
    models(["a", "b", "c"])
    G.write_section(request(author_material=MATERIAL), G.Deps(baseline={}))
    for p in PROMPTS[:3]:
        assert "62%) reported late payments" in p and "HOW TO USE THE REFERENCE MATERIAL" in p
    assert sum(p.count("62%) reported late payments") for p in PROMPTS[:3]) == 3      # once per call, not duplicated


def test_no_author_material_means_no_material_text():
    models(["a", "b", "c"])
    G.write_section(request(), G.Deps(baseline={}))
    assert all("HOW TO USE THE REFERENCE MATERIAL" not in p for p in PROMPTS[:3])


def test_author_material_that_is_not_text_is_ignored():
    models(["a", "b", "c"])
    G.write_section(request(author_material={"x": 1}), G.Deps(baseline={}))
    assert all("HOW TO USE THE REFERENCE MATERIAL" not in p for p in PROMPTS[:3])
