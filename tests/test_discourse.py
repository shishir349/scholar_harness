import json
import re

from harness import graph as G
from harness.text import cleanup as C
from harness.validate import style as V

from .test_graph import GOLD, PROMPTS, clean, models, request  # noqa: F401  (clean: autouse fixture reused)
from .test_skeletons import _profile_id, _sk

ON = {"pool": True, "sources": False, "revise": False}


def _run(**over):
    models(["a", "b", "c"])
    return G.write_section(request(flags=ON, **over), G.Deps(baseline={}, skeletons=[]))


# ── 1. uneven call budgets, bimodal rule ────────────────────────────────────────────────────────────────────────────
def test_budgets_are_uneven_deterministic_and_keep_the_total():
    seen = set()
    for n in range(40):
        b = G.call_budgets(f"T|C|S{n}", 450)
        assert b == G.call_budgets(f"T|C|S{n}", 450)
        w = [b[p][0] for p in G.POSITIONS]
        assert len(set(w)) > 1 and abs(sum(w) - 1350) <= 3 and all(b[p][1] >= w_ for p, w_ in zip(G.POSITIONS, w))
        seen.add(tuple(round(x / 1350, 2) for x in w))
    assert seen == {tuple(round(r, 2) for r in prof) for prof in G.BUDGET_PROFILES}        # all three profiles get drawn


def test_each_draft_call_asks_for_its_own_word_count():
    _run()
    want = G.call_budgets("Test Thesis|%s|%s" % (GOLD["chapter"]["title"], GOLD["section"]["title"]), 450)
    for p, pos in zip(PROMPTS[:3], G.POSITIONS):
        assert f"{want[pos][0]} words" in p
    assert len({w for w, _ in want.values()}) > 1


def test_voice_rules_carry_the_bimodal_rule_and_not_the_flat_average():
    assert G.BIMODAL in G.VOICE_RULES
    assert "32–48 words" in G.VOICE_RULES and "5–11 words" in G.VOICE_RULES and "18–26 words" in G.VOICE_RULES
    assert "near 29" not in G.VOICE_RULES and "avg ~29" not in G.VOICE_RULES
    assert G._plain("- Mix very short sentences (under 8 words) with long analytical ones. Include at least two short sentences per response").count("32–48") == 1


def test_scholars_own_path_keeps_one_target_for_all_three_calls():
    models(["a", "b", "c"])
    G.write_section(request(), G.Deps(baseline={}))
    assert all("450 words" in p for p in PROMPTS[:3])


# ── 2. discourse moves ──────────────────────────────────────────────────────────────────────────────────────────────
def test_the_plan_asks_for_friction_a_callback_named_anchors_and_fact_endings():
    _run()
    mid, close = PROMPTS[1], PROMPTS[2]
    assert "set two cited works or observations against each other" in mid and "set two cited works" in close
    assert "set two cited works" not in PROMPTS[0]
    assert "point back to an earlier section" in close and "point back to an earlier section" not in PROMPTS[0]
    for p in PROMPTS[:3]:
        assert "concrete proper noun" in p and "End on a specific number" in p


def test_epilogue_ban_and_anchors_cover_the_asked_share_of_paragraphs():
    _run()
    for p in PROMPTS[:3]:
        paras = re.findall(r"^\d+\. about \d+ words:.*$", p, re.M)
        assert paras and sum("End on a specific number" in x for x in paras) * 3 >= 2 * len(paras)
        assert sum("proper noun" in x for x in paras) * 10 >= 4 * len(paras)


def test_the_skeleton_plan_gets_the_same_discourse_moves():
    typ = _profile_id()
    pool = [_sk(i, typ, pos, 450 + i) for i in range(6) for pos in ("open", "mid", "close")]
    models(["a", "b", "c"])
    G.write_section(request(flags=ON), G.Deps(baseline={}, skeletons=pool))
    assert all("HUMAN DISCOURSE SKELETON" in p and "DISCOURSE MOVES" in p and "proper noun" in p for p in PROMPTS[:3])
    assert "set two cited works" in PROMPTS[1] and "point back to an earlier section" in PROMPTS[2]


def _one(fn, text):
    out, n = fn({"open": text, "mid": "", "close": ""})
    return out["open"], n


def test_drop_closers_strips_the_new_takeaway_openers():
    body = "Nodes drift over a long winter. Power draw decides whether a node lives."
    for lead in ("In sum", "Overall", "Ultimately", "This indicates that", "These findings collectively", "Such dynamics"):
        text, n = _one(C.drop_closers, f"{body} {lead}, the network is fragile and needs care.")
        assert n == 1 and text.strip() == body, lead
    text, n = _one(C.drop_closers, f"{body} Such dynamics held in 94% of nodes.")
    assert n == 0                                                                    # a figure keeps the sentence
    text, n = _one(C.drop_closers, f"{body} Such dynamics held in the sensors (Lee, 2015).")
    assert n == 0                                                                    # so does a citation


# ── 4. validate and prune priority ──────────────────────────────────────────────────────────────────────────────────
def _sent(n, k):
    return " ".join([f"W{k}"] + [f"w{k}x{i}" for i in range(n - 1)]) + "."


def test_burst_flags_find_three_similar_sentences_and_pass_a_bimodal_paragraph():
    mid = " ".join(_sent(22, k) for k in range(4))
    bimodal = " ".join([_sent(40, 0), _sent(7, 1), _sent(36, 2), _sent(9, 3)])
    out = V.burst_flags("# Head\n\n" + mid + "\n\n" + bimodal)
    assert [i for i, _ in out] == [1] and "22, 22, 22" in out[0][1]
    assert V.burst_flags(" ".join([_sent(20, 0), _sent(21, 1)])) == []             # two sentences: nothing to judge


def test_a_flat_draft_gets_a_burst_problem_and_a_reason_in_the_patch(monkeypatch):
    monkeypatch.setattr(G, "burst_flags", V.burst_flags)
    flat = "\n\n".join(" ".join(_sent(22, k * 10 + j) for j in range(4)) for k in range(3))
    models([flat] * 3, haiku=['["r1","r2","r3","r4","r5","r6","r7","r8"]'])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(baseline={}, skeletons=[]))
    assert out["reports"]["first_problems"]["burst"] == 9
    assert any("no bimodal burstiness" in p for p in PROMPTS) and out["revisions"] == 1


def test_prune_puts_flat_paragraphs_and_takeaway_closers_right_after_citations_and_exemplar_copying():
    mk = lambda *reasons, n=50: {"idx": 0, "text": "w " * n, "reasons": list(reasons)}
    cite = mk("cites a work not in REFERENCES")
    copy = mk("wording copied from a style exemplar")
    flat = mk("has no bimodal burstiness (x): rule")
    flat2 = mk("has flat sentence lengths (x): rewrite")
    close = mk('ends on a summarising verdict ("Overall"): cut')
    stock = mk("contains a generic/stock AI phrase")
    minor = mk("a", "b", "c", "d", n=200)
    by_pos = {"open": [minor, stock, close], "mid": [flat, cite], "close": [copy, flat2]}
    kept = G.prune_flagged(by_pos, limit=5)
    got = [f for pos in G.POSITIONS for f in kept.get(pos, [])]
    assert len(got) == 5 and stock not in got and minor not in got
    assert all(x in got for x in (cite, copy, flat, flat2, close))
    kept = G.prune_flagged(by_pos, limit=3)
    got = [f for pos in G.POSITIONS for f in kept.get(pos, [])]
    assert cite in got and copy in got and len(got) == 3 and stock not in got


# ── 3. memory and history ───────────────────────────────────────────────────────────────────────────────────────────
HIST = [{"opening": "Smith (2010) measured soil moisture across forty plots.", "closing": "Mean yield fell by 12 percent.", "paragraphs": 4},
        {"opening": "In 2019 the plots dried early in July.", "closing": "The second site was never resampled.", "paragraphs": 5}]


def test_blocks_and_section_history_reach_every_draft_call_and_forbid_repeating_the_last_shape():
    _run(memory_block="MEMBLOCK earlier chapters said X", ecology_block="ECOBLOCK soil", temporal_block="TEMPBLOCK 2024", section_history=HIST)
    for p in PROMPTS[:3]:
        assert "MEMBLOCK" in p and "ECOBLOCK" in p and "TEMPBLOCK" in p
        assert "EARLIER SECTIONS OF THIS CHAPTER" in p and "The second site was never resampled." in p
        assert "empirical figure or metric" in p                    # the last section opened on a year
        assert "must not total 5 paragraphs" in p and "different kind of sentence" in p


def test_no_history_and_no_blocks_change_nothing():
    _run()
    assert not any("EARLIER SECTIONS" in p or "MEMBLOCK" in p for p in PROMPTS[:3])


def test_history_and_blocks_are_ignored_on_scholars_own_path():
    models(["a", "b", "c"])
    G.write_section(request(memory_block="MEMBLOCK", section_history=HIST), G.Deps(baseline={}))
    assert not any("MEMBLOCK" in p or "EARLIER SECTIONS" in p for p in PROMPTS[:3])


def test_a_note_about_the_source_is_never_pasted_into_the_text():
    from .test_graph import SOURCES
    claim = "Cloud firms doubled their output (Smith, 2019)."
    note = "The source abstract addresses caesarean section rates, not cloud firms. This claim cannot be verified."
    models([claim + "\n\n", "B one.\n\n", "C one."],
           haiku=['[{"id": 1, "verdict": "UNSUPPORTED", "reason": "off topic"}]', json.dumps([note]),
                  '[{"id": 1, "verdict": "UNSUPPORTED", "reason": "off topic"}]'] + ['[{"id": 1, "verdict": "UNSUPPORTED", "reason": "x"}]'] * 4)
    out = G.write_section(request(flags={"pool": False, "sources": True, "revise": True}, ledger={"sources": SOURCES, "by_section": {}}), G.Deps(baseline={}))
    assert "cannot be verified" not in out["text"] and "source abstract" not in out["text"]
    assert "Cloud firms doubled" not in out["text"]              # the unbacked sentence was deleted instead
