import json
from pathlib import Path

from harness import graph as G
from harness.text import cleanup as C

ABS = ("Low-cost accelerometers based on Arduino or Raspberry Pi platforms present several drawbacks for structural health "
       "monitoring, including high noise density, low sampling frequency, and synchronization issues.")
GOOD = ("Komarizadehasl et al. (2022) found that cheap boards miss the low-frequency vibration that long spans produce, which "
        "laboratory characterisation alone does not reveal.")


def _c(open_="", mid="", close=""):
    return {"open": open_, "mid": mid, "close": close}


def test_heal_drops_an_orphan_fragment_and_fixes_a_glued_seam():
    out = C.heal(_c("Nodes fail often.The cause is power.\n\nconsidered when setting acceptance thresholds for a given bridge class.\n\nNext paragraph."))
    assert "fail often. The cause" in out["open"] and "considered when" not in out["open"] and "Next paragraph." in out["open"]


def test_heal_leaves_stubs_continuations_and_name_particles_alone():
    chunks = _c("First chunk ends here.", "Second chunk starts here and stops at remains", "largely unnoticed.\n\nvan der Berg et al. (2019) argue the opposite of this claim.")
    assert C.heal(chunks) == chunks


def test_heal_cuts_a_last_sentence_the_token_limit_left_unfinished():
    body = " ".join(["Sensors drift over long deployments."] * 6) + " The cross-sectional structure,"
    assert C.heal(_c("", "", body))["close"].endswith("deployments.")


def test_drop_repeats_deletes_a_pasted_source_sentence_and_a_restated_one():
    para = f"{GOOD} {ABS} Power draw decides whether a node survives a winter. Whether a node survives a winter depends on power draw."
    out, left = C.drop_repeats(_c(para), [ABS])
    assert ABS not in out["open"] and GOOD in out["open"] and left == []
    assert out["open"].count("winter") == 1


def test_drop_repeats_keeps_two_sentences_and_reports_what_it_could_not_delete():
    out, left = C.drop_repeats(_c(f"{GOOD} {ABS}"), [ABS])
    assert out["open"] == f"{GOOD} {ABS}" and left == [("open", 0)]


def test_repeat_flags_find_restated_openers_stance_closers_and_what_openers():
    a = "What the sources show together, read against one another, is that sensor platforms are not converging on one standard."
    b = "What the sources show together, read as a body of work, is that the trade-off depends on the deployment."
    p = f"I argue that field performance matters more. {GOOD}\n\n{a}\n\n{b} I contend that this needs a system view."
    f = C.repeat_flags(_c(p))
    assert f["repeat"] and f["stance"] and f["whatopen"]
    assert C.repeat_flags(_c(f"{GOOD} Power draw decides the winter. Taken together, power matters most."))["closer"]


def test_scholars_stock_seeds_are_replaced_on_the_harness_path_only():
    gold = json.loads((Path(__file__).parent / "golden" / "cases.json").read_text(encoding="utf-8"))["cases"][0]
    for key in ("call1", "call2", "call3"):
        plain = G._plain(gold[key])
        assert plain != gold[key] and not any(old in plain for old, _ in G._SEEDS)
    assert "This study takes the position" not in G._plain(gold["call3"])


def test_report_verb_pairs_count_as_a_source_by_source_chain():
    para = ("Sabato et al. (2016) trace the development of boards. Yu et al. (2023) situate these constraints in a decade of work. "
            "A node in isolation behaves differently from one in a network.")
    assert G._is_chain(para) and not G._is_chain("Sabato et al. (2016) trace the boards. Nodes drift (Yu et al., 2023).")


def test_a_pasted_sentence_that_carries_a_figure_is_kept_and_sent_to_the_rewrite_not_deleted():
    fact = "Over 25% of bridges in the United States were rated deficient in 2009 by the federal highway agency."
    out, left = C.drop_repeats(_c(f"{GOOD} Power draw decides the winter. {fact} Nodes drift."), [fact])
    assert fact in out["open"] and left == [("open", 0)]                  # the number survives; the cheap patch rewords the sentence
