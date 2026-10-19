"""Corpus-mined style profile: reference-list filter, miner, prompt block, wiring."""
from harness.text.refs import looks_like_references

REFS = ("Ackrich, M., 1992. The de-scription of technical objects. In: W. Bijker (Eds) Shaping technology. London: MIT Press.\n\n"
        "Ajzen, I., 2002. Constructing a TPB questionnaire. Amherst: University of Massachusetts.\n\n"
        "Mertens, D. M., 1998. Research methods in education and psychology. London: Sage.\n\n"
        "Mintzberg, H., 1987. Strategy concept I: five Ps for strategy. California Management Review 30, 11-24.")
URLS = ("- Andreas Huyssen. Globalization and memory. 21 April 2002. Available from\n\nhttp://www.example.org/a.\n\n"
        "- Jane Doe. A festival. Available from\n\nhttp://kurmes.example.org/b.\n\n- Retrieved from https://example.org/c")
PROSE = ("Strain gauges measure local deformation (Farrar & Worden, 2007), and fibre sensors extend that range. "
         "Lee et al. (2015) argue that wireless nodes lower installation cost.\n\nThe second paragraph says more.")
TAXON = ("_Elaeocarpus_ is the largest (approx. 350 spp.) of the 12 genera recognised in the family. The others are "
         "_Aceratium_ DC. (approx. 20 spp.), _Aristotelia_ L'Her (approx. 5 spp.), _Crinodendron_ Molina (1782).")


def test_reference_lists_are_recognised_and_ordinary_prose_is_not():
    assert looks_like_references(REFS) and looks_like_references(URLS)
    assert not looks_like_references(PROSE) and not looks_like_references(TAXON) and not looks_like_references("")


# --- slice 1b: bibliography text never enters the pool ------------------------------------------------------------
from harness.corpus import retrieve, store  # noqa: E402


def _con_with(texts):
    retrieve.clear_cache()
    con = store.connect(":memory:")
    tid = store.upsert_thesis(con, dict(openalex_id="W1", title="T", field_id="1", field_name="Business", score=90))
    store.insert_chunks(con, tid, "Business", [dict(section_type="conclusion", position="open", heading="H", chunk_idx=i, text=t)
                                                for i, t in enumerate(texts)])
    con.commit()
    return con


def test_insert_chunks_skips_reference_lists():
    con = _con_with([REFS, PROSE])
    assert [r[0] for r in con.execute("select text from chunks")] == [PROSE]


def test_drop_reference_chunks_cleans_an_old_database_and_its_search_index():
    con = _con_with([PROSE])
    for i, t in ((50, REFS), (51, URLS)):                       # simulate rows loaded before the filter existed
        con.execute("insert into chunks(id, thesis_id, field_name, section_type, position, heading, chunk_idx, word_count, text)"
                    " values(?,1,'Business','conclusion','open','H',?,10,?)", (i, i, t))
        con.execute("insert into chunks_fts(rowid, heading, text) values(?,?,?)", (i, "H", t))
    assert store.drop_reference_chunks(con) == 2
    assert con.execute("select count(*) from chunks").fetchone()[0] == 1
    assert con.execute("select count(*) from chunks_fts where chunks_fts match 'Ackrich'").fetchone()[0] == 0


def test_exemplar_retrieval_never_returns_a_reference_list():
    con = _con_with([PROSE])
    con.execute("insert into chunks(id, thesis_id, field_name, section_type, position, heading, chunk_idx, word_count, text)"
                " values(60,1,'Business','conclusion','open','H',60,10,?)", (REFS,))
    con.execute("insert into chunks_fts(rowid, heading, text) values(60,'H',?)", (REFS,))
    got = retrieve.get_exemplars(con=con, fields=[], title="Wireless strain gauges", domain="", chapter_title="Conclusion",
                                 section_title="Sensors", section_type="conclusion", position="open", limit=3)
    assert [g["text"] for g in got] == [PROSE]


# --- slice 2: the miner -------------------------------------------------------------------------------------------
from harness.corpus.profile import mine  # noqa: E402

AI_TEXT = ("The field has long assumed that cost decides everything. The question of what counts as adequate remains open. "
           "The literature has not settled the matter and the field has moved on. Sensors fail in ways that surprise teams. "
           "However the evidence tends to favour cheaper devices. ")
REAL_TEXT = ("For example the survey used in order to compare sites as well as costs. Such results were reported by Lee (2010) "
             "and the team was able to repeat them. For instance we measured the number of faults. ")


def _docs():
    ai = [(f"fx{i:02d}", "introduction", (AI_TEXT * 8) + (" bridge cables corrode." * 6 if i < 2 else "")) for i in range(1, 7)]
    real = [(t, "introduction", (REAL_TEXT * 4) + "\n\n" + (REAL_TEXT * (1 + t % 5)) + "\n\n" + REAL_TEXT) for t in range(1, 31)]
    return ai, real


def test_miner_finds_phrases_the_ai_overuses_and_real_theses_prefer():
    ai, real = _docs()
    prof = mine(ai, real, min_real_count=5, min_real_theses=5, min_ai_count=4)
    avoid = [e["p"] for e in prof["avoid"]]
    prefer = [e["p"] for e in prof["prefer"]]
    assert any("the field has" in p or "field has" in p for p in avoid)
    assert not any("bridge" in p for p in avoid)                       # topic of 2 fixtures only: not a style tell
    assert any("for example" in p or "in order to" in p for p in prefer)
    assert not any(set(p.split()) <= {"of", "the", "in", "to", "a"} for p in avoid + prefer)   # bare function words are noise


def test_miner_reports_signposts_citation_share_and_paragraph_ratios():
    ai, real = _docs()
    prof = mine(ai, real, min_real_count=5, min_real_theses=5, min_ai_count=4)
    assert prof["signposts"]["real_rate"] > prof["signposts"]["ai_rate"]
    assert 0 <= prof["citations"]["narrative_share_real"] <= 1
    r = prof["paragraph_ratios"]["introduction"]
    assert len(r) == 7 and r == sorted(r) and 0.3 <= r[0] and r[-1] <= 3.0
    assert prof["n_real_theses"] == 30 and prof["n_fixtures"] == 6


# --- slice 3: the profile steers the prompt, the plan and the patch ----------------------------------------------
from harness.validate.profile import avoid_hits, plan_ratios, profile_block  # noqa: E402

PROF = {
    "n_real_theses": 196,
    "avoid": [{"p": "the field has", "ai": 0.6, "real": 0.001}, {"p": "in ways that", "ai": 1.0, "real": 0.02},
              {"p": "the literature", "ai": 2.3, "real": 0.28}],
    "avoid_words": ["rarely", "tends"],
    "prefer": [{"p": "for example", "ai": 0.02, "real": 0.66}, {"p": "in order to", "ai": 0.0, "real": 0.45}],
    "prefer_words": ["however", "therefore"],
    "avoid_openers": ["the question"], "prefer_openers": ["in addition"],
    "signposts": {"list": ["for example", "however", "in addition"], "real_rate": 0.053, "ai_rate": 0.0},
    "citations": {"narrative_share_real": 0.57, "narrative_share_ai": 0.76},
    "paragraph_ratios": {"all": [1.0] * 7, "introduction": [0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]},
}
BASE = {"by_section": {"all": {"n": 1, "features": {
    "sent_len_mean": {"mean": 29, "std": 5, "p10": 22, "p90": 36}, "short_sent_ratio": {"mean": 0.03, "std": 0.05, "p10": 0.0, "p90": 0.08},
    "first_person_per_100": {"mean": 0.3, "std": 0.5, "p10": 0.0, "p90": 0.6}, "hedge_per_100": {"mean": 0.4, "std": 0.4, "p10": 0.0, "p90": 0.9},
    "para_len_cv": {"mean": 0.7, "std": 0.4, "p10": 0.3, "p90": 1.2}}}}, "score_p10": {}}


def test_profile_block_turns_the_mined_numbers_into_instructions():
    b = profile_block("introduction", PROF, BASE)
    for needle in ("196 real theses", "the field has", "in ways that", "for example", "in order to", "rarely", "however",
                   "5%", "1 sentence in 19", "57%", "22-36", "first person"):
        assert needle in b, needle
    assert profile_block("introduction", {}, BASE) == "" and profile_block("introduction", None, BASE) == ""


def test_avoid_hits_matches_whole_phrases_and_skips_phrases_real_writers_also_use():
    text = "The field has changed. It shifted in ways that matter. The literature agrees. Nothing in waysides."
    assert avoid_hits(text, PROF) == ["the field has", "in ways that"]       # 'the literature' is common in real theses: not a tell
    assert avoid_hits("plain text", PROF) == [] and avoid_hits(text, {}) == []


def test_plan_ratios_fall_back_to_all_section_types():
    assert plan_ratios("introduction", PROF) == PROF["paragraph_ratios"]["introduction"]
    assert plan_ratios("conclusion", PROF) == [1.0] * 7 and plan_ratios("x", {}) is None


import harness.graph as G2  # noqa: E402
from tests.test_graph import PROMPTS, models, request  # noqa: E402


import pytest as _pt  # noqa: E402


@_pt.fixture(autouse=True)
def _repeat_logic_off(monkeypatch):
    """These tests feed one sentence many times to trip a single gap detector; the repeat logic (tests/test_cleanup.py) would
    delete or flag that filler, so it is switched off here."""
    monkeypatch.setattr(G2.cleanup, "drop_repeats", lambda chunks, abstracts=(): (chunks, []))
    monkeypatch.setattr(G2.cleanup, "repeat_flags", lambda chunks: {})
    # the number / first-person / example shares and the free polish are tested in tests/test_numbers.py and tests/test_polish.py
    for name in ("number_gap", "fp_gap"):
        monkeypatch.setattr(G2.numbers, name, lambda chunks, ptype: 0)
    monkeypatch.setattr(G2.numbers, "example_gap", lambda chunks: 0)
    monkeypatch.setattr(G2.cleanup, "polish", lambda chunks, that_limit=None: (chunks, {}))


def test_graph_puts_the_profile_in_the_cached_block_and_follows_its_paragraph_shape():
    PROMPTS.clear(); models(["a", "b", "c"])
    flat = {**PROF, "paragraph_ratios": {"all": [1.0] * 7, "introduction": [1.0] * 7}}
    G2.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G2.Deps(baseline={}, profile=flat))
    assert all("REAL THESIS PHRASING" in p and "the field has" in p for p in PROMPTS[:3])
    plan = PROMPTS[0].split("PARAGRAPH PLAN")[1]
    sizes = {line.split("about ")[1].split(" words")[0] for line in plan.splitlines() if " words:" in line}
    assert len(sizes) == 1                                                  # all-equal ratios -> equal sizes
    PROMPTS.clear(); models(["a", "b", "c"])
    G2.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G2.Deps(baseline={}, profile={}))
    assert not any("REAL THESIS PHRASING" in p for p in PROMPTS)


def test_a_paragraph_stacked_with_ai_only_phrases_gets_one_cheap_patch_naming_them():
    bad = "The field has moved on. It shifted in ways that surprised teams. " + " ".join(["word"] * 50) + "."
    chunks = [bad + "\n\n", "A plain second part. " + " ".join(["word"] * 50) + ".\n\n", "A plain third part. " + " ".join(["word"] * 50) + "."]
    PROMPTS.clear(); models(chunks, haiku=['["The field moved on. Teams were surprised by the shifts."]'])
    out = G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline={}, profile=PROF))
    assert out["revisions"] == 1 and "Teams were surprised" in out["text"] and len(PROMPTS) == 4
    assert '"the field has"' in PROMPTS[3] and '"in ways that"' in PROMPTS[3]


def test_profile_block_also_lists_sentence_openers_to_drop():
    assert '"the question"' in profile_block("introduction", PROF, BASE)


from harness.validate.profile import profile_metrics  # noqa: E402


def test_profile_metrics_scores_any_text_without_an_api_call():
    text = ("The field has changed in ways that matter. For example, Lee (2010) measured it. However the effect was small. "
            "Chen and Patel argue otherwise (Chen & Patel, 2012). The field has not settled it.")
    m = profile_metrics(text, PROF)
    assert m["tells_per_1000"] == 96.8 and m["signpost_rate"] == 0.4 and m["narrative_share"] == 0.5
    assert profile_metrics("", PROF) == {"tells_per_1000": 0.0, "signpost_rate": 0.0, "narrative_share": None}



# --- concrete rules: exact signpost openers + a parenthetical share in the plan, and patches when the draft ignores them ---
from tests.test_graph import SOURCES  # noqa: E402


def _plan_lines(prompt):
    plan = prompt.split("PARAGRAPH PLAN")[1]
    return [ln for ln in plan.splitlines() if " words:" in ln]


def test_plan_names_exact_signpost_openers_and_a_parenthetical_share_of_citations():
    PROMPTS.clear(); models(["a", "b", "c"])
    G2.write_section(request(flags={"pool": True, "sources": True, "revise": False},
                             ledger={"sources": SOURCES, "by_section": {}}), G2.Deps(baseline={}, profile=PROF))
    lines = [ln for p in PROMPTS[:3] for ln in _plan_lines(p)]
    openers = {ln.split('begin one sentence with "')[1].split('"')[0] for ln in lines if 'begin one sentence with "' in ln}
    assert len(openers) >= 2                                              # different signpost per passage, from the mined list
    paren = [ln for ln in lines if "in parentheses" in ln]
    assert paren and len(paren) < len(lines)                              # a share of paragraphs, not all of them


def test_without_sources_the_plan_still_names_signposts_but_says_nothing_about_citations():
    PROMPTS.clear(); models(["a", "b", "c"])
    G2.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G2.Deps(baseline={}, profile=PROF))
    lines = [ln for p in PROMPTS[:3] for ln in _plan_lines(p)]
    assert any('begin one sentence with "' in ln for ln in lines) and not any("parentheses" in ln for ln in lines)


def _plain(n):
    return " ".join(f"Costs fell in region number {i} that year." for i in range(n))


def test_a_draft_with_no_signposts_gets_one_cheap_patch_asking_for_them():
    chunks = [_plain(5) + "\n\n", _plain(5) + "\n\n", _plain(5)]
    PROMPTS.clear(); models(chunks, haiku=['["For example, costs fell in region 1 that year.", "However, costs fell in region 2 that year.", "In addition, costs fell in region 3 that year."]'])
    out = G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline={}, profile=PROF))
    assert out["revisions"] == 1 and "For example, costs fell" in out["text"] and len(PROMPTS) == 4
    assert "plain signpost" in PROMPTS[3]


def test_a_draft_that_cites_everything_in_the_narrative_form_gets_half_moved_into_parentheses():
    names = ["Smith", "Jones", "Lee", "Chen", "Patel", "Kim"]
    para = lambda a: " ".join(f"Costs fell after the change, as {n} (20{10 + i}) found." for i, n in enumerate(a))
    chunks = [para(names[:3]) + "\n\n", para(names[3:]) + "\n\n", "Plain closing note."]
    PROMPTS.clear(); models(chunks, haiku=['["Costs fell after the change (Smith, 2010; Jones, 2011).", "Costs fell too (Chen, 2013; Patel, 2014)."]'])
    out = G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline={}, profile=PROF))
    assert out["revisions"] == 1 and "(Smith, 2010; Jones, 2011)" in out["text"] and len(PROMPTS) == 4
    assert "into parentheses" in PROMPTS[3]


def test_a_draft_that_already_mixes_citation_forms_is_left_alone():
    para = lambda n1, n2: f"Costs fell after the change, as {n1} (2010) found. Output rose too (Jones, 2011). Another view held (Lee, 2012). It held as {n2} (2013) wrote."
    chunks = [para("Smith", "Chen") + "\n\n", para("Kim", "Patel") + "\n\n", "Plain closing note."]
    PROMPTS.clear(); models(chunks)
    out = G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline={}, profile=PROF))
    assert out["revisions"] == 0 and len(PROMPTS) == 3


# --- T2: mined per-feature targets (the gap report's features) become plain instructions -------------------------------
from harness.corpus.profile import TARGET_FEATURES, feature_targets  # noqa: E402

_SENT = "We measured the soil at 12 sites that were chosen at random. However, the results were also stable. Costs rose sharply here. "


def test_feature_targets_summarise_real_windows_per_section_type():
    real = [(t, st, i, _SENT * 14) for t in range(1, 5) for st in ("introduction", "methodology") for i in range(6)]
    tg = feature_targets(real, window=3, per_thesis=8)
    assert set(tg) == {"all", "introduction", "methodology"}
    f = tg["all"]["connectives"]
    assert f["p10"] <= f["p50"] <= f["p90"] and f["n"] >= 8 and set(TARGET_FEATURES) <= set(tg["all"])
    assert tg["all"]["that"]["p50"] > 0


GAP = {**PROF,
       "targets": {"all": {"connectives": {"p10": 3.0, "p50": 6.0, "p90": 9.0}, "that": {"p10": 6.0, "p50": 10.7, "p90": 16.0},
                           "of": {"p10": 26.0, "p50": 38.0, "p90": 50.0}, "word_len": {"p10": 4.9, "p50": 5.3, "p90": 5.8},
                           "booster": {"p10": 0.0, "p50": 0.0, "p90": 0.5}, "passive": {"p10": 3.0, "p50": 9.0, "p90": 15.0},
                           "digits": {"p10": 1.0, "p50": 12.0, "p90": 80.0}, "capitalised_mid": {"p10": 8.0, "p50": 28.0, "p90": 55.0},
                           "acronyms": {"p10": 1.0, "p50": 9.0, "p90": 28.0}, "the_field_lit": {"p10": 0.0, "p50": 0.2, "p90": 1.0},
                           "ttr_1000": {"p10": 0.34, "p50": 0.40, "p90": 0.47}}},
       "ai_means": {"connectives": 1.5, "that": 24.9, "of": 21.0, "word_len": 6.0, "booster": 1.4, "passive": 4.3, "digits": 0.7,
                    "capitalised_mid": 9.0, "acronyms": 4.9, "the_field_lit": 3.6, "ttr_1000": 0.50}}


def test_profile_block_states_each_gap_as_a_number_the_model_can_aim_at():
    b = profile_block("introduction", GAP, BASE)
    for needle in ("connectives", "also, however, therefore", "about 6 per 1000", "AI drafts 1.5", '"that"', "about 11 per 1000",
                   "AI drafts 24.9", "5.3 letters", "was collected", "exact figures", "never invent", "acronym", "repeat the same",
                   "unique-word"):
        assert needle in b, needle
    assert "Join claims" not in profile_block("introduction", PROF, BASE)             # no targets mined -> no invented numbers


# --- T3: long AI words that real theses use far less ---------------------------------------------------------------
def _long_docs():
    ai_text = "We comprehensive evaluate accelerometers straightforwardly in every case. " * 10
    ai = [(f"fx{i:02d}", "introduction", ai_text) for i in range(1, 7)]
    real = [(t, "introduction", ("For example the survey used a comprehensive design. " if t % 3 == 0 else "The survey used a design. ")
             + "The survey used a design. " * 3) for t in range(1, 31)]
    return ai, real


def test_miner_lists_long_words_real_writers_share_but_use_far_less_not_topic_terms():
    ai, real = _long_docs()
    prof = mine(ai, real, min_real_count=5, min_ai_count=4, min_fixtures_long=5)
    assert "comprehensive" in prof["avoid_long"]                 # used in 10 of 30 real theses, 10x less often
    assert "accelerometers" not in prof["avoid_long"]            # a topic term: real theses almost never use it
    assert "straightforwardly" not in prof["avoid_long"]         # AI-only word, but only 6 fixtures (< 8): could be topic


def test_ai_only_long_words_count_when_they_show_up_across_nearly_every_topic():
    ai, real = _long_docs()
    ai += [(f"fx{i:02d}", "introduction", "straightforwardly plain text here. " * 10) for i in range(7, 11)]
    prof = mine(ai, real, min_real_count=5, min_ai_count=4, min_fixtures_long=5)
    assert "straightforwardly" in prof["avoid_long"] and "accelerometers" not in prof["avoid_long"]


def test_profile_block_lists_the_long_words_to_swap_for_plain_ones():
    b = profile_block("introduction", {**PROF, "avoid_long": ["comprehensive", "facilitate"]}, BASE)
    assert '"comprehensive"' in b and '"facilitate"' in b and "shorter plain word" in b



# --- T5: measured gaps the model still misses get one cheap patch each --------------------------------------------------
def _targets(**k):
    keys = {"that": 10.0, "connectives": 0.0, "short_sent_share": 0.07, "the_field_lit": 0.2}
    keys.update(k)
    return {"targets": {"all": {n: {"p10": v / 2, "p50": v, "p90": v * 2} for n, v in keys.items()}}}


def _run_patch(rep, haiku, profile, n=14):
    chunks = [rep * n + "\n\n", rep * n + "\n\n", rep * n]
    PROMPTS.clear(); models(chunks, haiku=[haiku])
    return G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline={}, profile=profile))


def test_heavy_use_of_that_gets_one_patch_that_drops_it():
    out = _run_patch("We also found that costs that firms reported that year fell. ", '["P1 fixed.", "P2 fixed."]', _targets())
    assert out["revisions"] == 1 and "P1 fixed." in out["text"] and len(PROMPTS) == 4 and 'drop "that"' in PROMPTS[3]


def test_a_draft_without_plain_connectives_gets_one_patch_asking_for_them():
    out = _run_patch("Costs fell in region number five last year as firms cut output. ", '["P1 fixed.", "P2 fixed.", "P3 fixed."]',
                     _targets(connectives=6.0))
    assert out["revisions"] == 1 and len(PROMPTS) == 4 and "also / however / therefore / thus" in PROMPTS[3]


def test_a_run_of_very_short_sentences_gets_one_patch_that_joins_them():
    out = _run_patch("Costs rose here. Output fell too. ", '["P1 fixed.", "P2 fixed."]', _targets(), n=30)
    assert out["revisions"] == 1 and len(PROMPTS) == 4 and "under ten words" in PROMPTS[3]


def test_writing_about_the_field_as_a_character_gets_one_patch_naming_the_studies():
    out = _run_patch("The field has changed a great deal over recent years in many ways. ", '["P1 fixed.", "P2 fixed."]', _targets())
    assert out["revisions"] == 1 and len(PROMPTS) == 4 and "as a person" in PROMPTS[3]


def test_a_draft_inside_the_real_ranges_is_left_alone_and_no_targets_means_no_patch():
    plain = "Costs fell in region number five last year as firms cut output and moved on. "
    assert _run_patch(plain, '[]', _targets()).get("revisions") == 0
    assert _run_patch("We also found that costs that firms reported that year fell. ", '[]', {}).get("revisions") == 0


# --- the patch must fit what it is asked to rewrite (the live probe showed it silently returning nothing) ----------
from harness import llm as _llm  # noqa: E402


def test_patch_gets_a_token_budget_that_fits_the_paragraphs_it_rewrites(monkeypatch):
    seen = {}
    monkeypatch.setattr(_llm, "ask", lambda tier, system, user, **k: seen.update(k) or '["P1 fixed.", "P2 fixed."]')
    out = _run_patch("We also found that costs that firms reported that year fell. ", "[]", _targets())
    assert out["revisions"] == 1 and "P1 fixed." in out["text"]
    assert seen["max_tokens"] > 720            # two 154-word paragraphs need more than the 720 a whole chunk gets


def test_pruning_keeps_hard_problems_first_then_the_paragraphs_with_most_reasons():
    mk = lambda reasons, n=50: {"idx": 0, "text": "w " * n, "reasons": reasons}
    hard = mk(["cites a work not in REFERENCES"])
    many = mk(["a", "b", "c"])
    soft = [mk(["x"]) for _ in range(6)]
    by_pos = {"open": [soft[0], soft[1], soft[2]], "mid": [hard, soft[3]], "close": [many, soft[4], soft[5]]}
    kept = G2.prune_flagged(by_pos, limit=3)
    flat = [f for pos in ("open", "mid", "close") for f in kept.get(pos, [])]
    assert len(flat) == 3 and hard in flat and many in flat
    assert G2.prune_flagged({"open": [soft[0]]}, limit=3) == {"open": [soft[0]]}          # under the limit: untouched


# --- found by asking the judge why it preferred the other draft ---------------------------------------------------------
import pytest as _pytest  # noqa: E402
from tests.test_graph import _stats  # noqa: E402


@_pytest.mark.parametrize("text", [
    "Taken together, the sources suggest a modest effect.",
    "What the sources mean by the term is worth pinning down, because it varies.",
    "It is worth noting the cost differs by site.",
])
def test_judge_flagged_openers_and_hedges_count_as_tells(text):
    assert G2.tell_count(text) >= 1


def test_the_first_person_patch_does_not_hand_the_model_a_stock_stance_to_copy():
    PROMPTS.clear()
    chunks = [(" ".join(["word"] * 60) + ".\n\n"), (" ".join(["word"] * 60) + ".\n\n"), (" ".join(["word"] * 60) + ".")]
    models(chunks, haiku=['["I treat it differently.", "I chose this."]'])
    G2.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G2.Deps(baseline=_stats(0.01, 0.6), profile={}))
    assert len(PROMPTS) == 4 and "I argue" not in PROMPTS[3] and "I chose" in PROMPTS[3]
