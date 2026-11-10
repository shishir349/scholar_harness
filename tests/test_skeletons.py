import json
import sqlite3

import pytest

from harness import config, graph as G, llm
from harness.corpus import skeletons as SK, store
from harness.prompts import sections as P

from .test_graph import GOLD, PROMPTS, clean, models, request  # noqa: F401  (clean: autouse fixture reused)

# One paragraph: narrative cite in sentence 1, a blunt short sentence, a parenthetical cite, a figure to close on.
VARIED = ("Smith (2010) measured soil moisture across forty plots over three growing seasons and found that the wettest plots "
          "also carried the heaviest weed load. Rain mattered less. Where drainage was poor, the yield penalty grew with every "
          "additional week of standing water, a pattern that held in both regions (Lee, 2015). Mean yield fell by 12 percent.")
def flat(k):
    """A paragraph of four sentences of 15 words each (no spread), with words unique to k: drop_repeats deletes restated sentences."""
    return " ".join("The team " + " ".join(f"item{k}{j}{m}" for m in range(12)) + " finished." for j in range(4))


FLAT = flat(0)


def _chunk(n=4):
    return "\n\n".join([VARIED.replace("forty", f"{40 + i}") for i in range(n)])


def test_reclassify_moves_only_catch_all_findings_by_heading():
    assert SK.reclassify("2.2 Related Work", "findings") == "literature_review"
    assert SK.reclassify("**Experimental setup**", "findings") == "methodology"
    assert SK.reclassify("Limitations of the study", "findings") == "discussion"
    assert SK.reclassify("Conclusions", "findings") == "conclusion"
    assert SK.reclassify("Soil moisture results", "findings") == "findings"
    assert SK.reclassify("Related Work", "introduction") == "introduction"     # only the catch-all moves
    assert SK.reclassify(None, "findings") == "findings"


def test_paragraph_shape_reads_cadence_citations_opener_and_closer():
    s = SK.paragraph_shape(VARIED)
    assert s["sentence_lengths"][1] < 5 and len(s["sentence_lengths"]) == 4 and s["word_count"] == sum(s["sentence_lengths"])
    assert s["opener_type"] == "author_narrative_cite"
    assert s["cite_positions"] == {"sentences": [1, 3], "mode": "narrative"} or s["cite_positions"]["sentences"] == [1, 3]
    assert s["closer_type"] == "raw_data_or_cite_cutoff"
    assert s["sent_len_cv"] > 0.38
    assert SK.opener_type("However the plots dried early in July.") == "contrastive_pivot"
    assert SK.opener_type("In 2019 the plots dried early in July.") == "empirical_figure_or_metric"
    assert SK.opener_type("We used a split plot design here.") == "methodological_action"


def test_a_skeleton_keeps_numbers_and_labels_never_the_text():
    sk = SK.skeleton_of(_chunk(), "findings", "open", "1_0_1")
    assert sk["total_words"] == sum(p["word_count"] for p in sk["paragraphs"]) and len(sk["paragraphs"]) == 4
    strings = [v for p in sk["paragraphs"] for v in (p["opener_type"], p["closer_type"], p["cite_positions"]["mode"])]
    assert set(strings) <= {"author_narrative_cite", "empirical_figure_or_metric", "contrastive_pivot", "methodological_action",
                            "domain_noun_subject", "raw_data_or_cite_cutoff", "limitation_or_caveat", "cross_ref_or_transition",
                            "comparative_assertion", "narrative", "parenthetical", "none"}
    blob = json.dumps(sk).lower()
    assert "smith" not in blob and "moisture" not in blob
    assert SK.skeleton_of("\n\n".join([VARIED] * 2), "findings", "open", "x") is None        # under 3 paragraphs


def test_build_applies_filters_reclassifies_and_skips_holdouts():
    con = store.connect(":memory:")
    good = store.upsert_thesis(con, dict(openalex_id="W1", title="T", field_id="1", field_name="F", score=80))
    held = store.upsert_thesis(con, dict(openalex_id="W2", title="H", field_id="1", field_name="F", score=90, holdout=True))
    low = store.upsert_thesis(con, dict(openalex_id="W3", title="L", field_id="1", field_name="F", score=10))
    row = lambda h, text=None: dict(section_type="findings", position="open", heading=h, chunk_idx=0, text=text or _chunk())
    store.insert_chunks(con, good, "F", [row("Related Work")])
    store.insert_chunks(con, good, "F", [dict(row("Plain"), chunk_idx=1, text=FLAT + "\n\n" + FLAT + "\n\n" + FLAT)])   # flat: filtered out
    store.insert_chunks(con, held, "F", [row("Related Work")])
    store.insert_chunks(con, low, "F", [row("Related Work")])
    res = SK.build(con, min_score=50)
    assert [s["section_type"] for s in res["skeletons"]] == ["literature_review"]
    assert res["skeletons"][0]["skeleton_id"].startswith(f"{good}_0_") and res["stats"]["reclassified_from_findings"] == 1


def test_real_corpus_slice_yields_valid_skeletons():
    db = config.db_path()
    if not db.exists():
        pytest.skip("no data/corpus.db")
    src = sqlite3.connect(db)
    src.row_factory = sqlite3.Row
    ids = [r[0] for r in src.execute("SELECT id FROM theses WHERE holdout=0 ORDER BY score DESC LIMIT 25")]
    mem = store.connect(":memory:")
    for tid in ids:
        t = dict(src.execute("SELECT * FROM theses WHERE id=?", (tid,)).fetchone())
        new = store.upsert_thesis(mem, {k: t[k] for k in ("openalex_id", "title", "field_id", "field_name", "score", "holdout")})
        store.insert_chunks(mem, new, t["field_name"], [dict(r) for r in src.execute(
            "SELECT section_type, position, heading, chunk_idx, text FROM chunks WHERE thesis_id=?", (tid,))])
    res = SK.build(mem, min_score=0)
    assert res["skeletons"], "no skeleton came out of 25 real theses"
    seen = set()
    for s in res["skeletons"]:
        assert s["skeleton_id"] not in seen
        seen.add(s["skeleton_id"])
        assert s["section_type"] in SK.TYPES and s["position"] in ("open", "mid", "close")
        assert len(s["paragraphs"]) >= SK.MIN_PARAS and s["total_words"] == sum(p["word_count"] for p in s["paragraphs"])
        for p in s["paragraphs"]:
            assert p["word_count"] == sum(p["sentence_lengths"]) and max(p["sentence_lengths"]) <= SK.MAX_SENT
            assert all(1 <= i <= len(p["sentence_lengths"]) for i in p["cite_positions"]["sentences"])
        assert SK._mean_sent_cv(s) >= SK.MIN_SENT_CV


def _sk(i, typ="introduction", pos="open", words=450):
    paras = [{"word_count": words // 3, "sentence_lengths": [34, 8, 29, 14], "sent_len_cv": 0.5, "opener_type": "domain_noun_subject",
              "cite_positions": {"sentences": [3], "mode": "parenthetical"}, "closer_type": "comparative_assertion"}] * 3
    return {"skeleton_id": f"{i}_0_{i}", "section_type": typ, "position": pos, "total_words": words, "paragraphs": paras}


def test_pick_is_deterministic_and_gives_each_section_of_a_chapter_its_own_skeleton():
    pool = [_sk(i, words=400 + i) for i in range(20)]
    ids = [SK.pick(pool, "introduction", "open", 450, "T|C|open", n)["skeleton_id"] for n in range(6)]
    assert len(set(ids)) == 6
    assert ids == [SK.pick(pool, "introduction", "open", 450, "T|C|open", n)["skeleton_id"] for n in range(6)]
    assert SK.pick(pool[:2], "introduction", "open", 450, "s") is None            # too few to choose from
    assert SK.pick(pool, "methodology", "open", 450, "s") is None


def test_scale_keeps_cadence_ratio_opener_citations_and_closer():
    specs = SK.scale(_sk(1, words=300), 600)
    assert abs(sum(s["target_words"] for s in specs) - 600) <= 30
    s = specs[0]
    assert s["opener_type"] == "domain_noun_subject" and s["cite_positions"]["sentences"] == [3] and s["closer_type"] == "comparative_assertion"
    a, b, c, d = s["target_sentences"]
    assert b < a and b < c and d < c and abs(a / b - 34 / 8) < 1.2          # long, blunt, long, shorter survives scaling


def test_blueprint_text_names_count_cadence_blunt_sentence_opener_citations_and_closer():
    text = SK.blueprint(SK.scale(_sk(1), 450))
    assert "HUMAN DISCOURSE SKELETON" in text and "Write exactly 3 paragraphs" in text
    assert "target lengths [~" in text and "blunt" in text and "sentence 3, parenthetical" in text
    assert "opens on its subject noun" in text or "Open with a sentence that opens on its subject noun" in text
    assert "End on a plain comparison" in text


def test_adherence_flags_a_paragraph_that_collapsed_to_flat_sentence_lengths():
    specs = SK.scale(_sk(1), 450)
    flat = SK.adherence("# Heading\n\n" + FLAT + "\n\n" + VARIED + "\n\n" + FLAT, specs)
    assert flat["paragraphs"] == [3, 3] and [i for i, _, _ in flat["flat"]] == [1, 3]      # index counts the heading paragraph
    assert SK.sent_std(VARIED) > SK.FLAT_STD


def _profile_id():
    return P.detect_profile(GOLD["chapter"]["title"], GOLD["chapter"]["chapter_number"])["id"]


def test_para_plan_emits_the_skeleton_block_and_flat_drafts_are_flagged():
    typ = _profile_id()
    pool = [_sk(i, typ, pos, 450 + i) for i in range(6) for pos in ("open", "mid", "close")]
    models([FLAT + "\n\n" + FLAT + "\n\n" + FLAT] * 3)
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}, skeletons=pool))
    assert all("HUMAN DISCOURSE SKELETON" in p and "PARAGRAPH PLAN" not in p for p in PROMPTS[:3])
    assert out["reports"]["skeleton"]["open"][1] == 3
    assert out["reports"]["first_problems"]["skeleton"] >= 3


def test_without_skeletons_the_old_paragraph_plan_runs_and_nothing_is_flagged():
    models(["a", "b", "c"])
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": False}), G.Deps(baseline={}, skeletons=[]))
    assert all("PARAGRAPH PLAN" in p and "HUMAN DISCOURSE SKELETON" not in p for p in PROMPTS[:3])
    assert out["reports"]["skeleton"] == {} and out["reports"]["first_problems"]["skeleton"] == 0


def test_a_flat_paragraph_is_patched_in_place_with_its_planned_cadence_in_the_reason():
    typ = _profile_id()
    pool = [_sk(i, typ, pos, 450 + i) for i in range(6) for pos in ("open", "mid", "close")]
    models([VARIED.replace("forty", str(i)) + "\n\n" + FLAT + "\n\n" + VARIED for i in range(3)], haiku=['["rewritten flat one"]'] * 3)
    out = G.write_section(request(flags={"pool": True, "sources": False, "revise": True}), G.Deps(baseline={}, skeletons=pool))
    prompts = [p for p in PROMPTS if "has flat sentence lengths" in p]
    assert prompts and "plan [~" in prompts[0] and out["revisions"] == 1
