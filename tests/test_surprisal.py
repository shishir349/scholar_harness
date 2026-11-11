"""Token surprisal scorer: offline, with a stub scorer (no model file needed) plus one test that uses the real model when present."""
import math
import re

import pytest

from harness.text import lamp_rewrite
from harness.validate import lamp_detector, surprisal

COMMON = {"the", "of", "and", "is", "a", "to", "in", "that", "this", "it", "are", "for", "as", "with", "sense", "why", "employees"}


class Stub:
    """Common words cost 1 nat (rank 1), everything else 7 (rank 50), a word starting 'zz' costs 12. Offsets are character spans, like the real scorer."""
    calls = 0
    paragraphs: list = []

    def token_logprobs(self, text):
        Stub.calls += 1
        Stub.paragraphs.append(text)
        out = []
        for m in re.finditer(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text):
            w = m.group().lower()
            lp, rank = (-1.0, 1) if w in COMMON else (-12.0, 900) if w.startswith("zz") else (-7.0, 50)
            out.append((m.start(), m.end(), lp, rank))
        return out


@pytest.fixture(autouse=True)
def _fresh():
    Stub.calls, Stub.paragraphs = 0, []
    surprisal.clear_cache()


PLAIN = "The data is a sample of the firms in the study and it is for the test. This is the case for the model in the study as it is."
ODD = "Quokka tessellations perturb zygomatic ledgers. Mulch frescoes defenestrate gantry hymns."
TEXT = PLAIN + "\n\n" + ODD


def test_scores_each_sentence_and_the_section():
    r = surprisal.score_section_surprisal(TEXT, Stub(), low=3.0, window=3.0)
    plain = [s for s in r["sentences"] if s["text"].startswith(("The data", "This is"))]
    odd = [s for s in r["sentences"] if s["text"].startswith(("Quokka", "Mulch"))]
    assert plain and odd and max(s["mean"] for s in plain) < min(s["mean"] for s in odd)
    assert r["perplexity"] == pytest.approx(math.exp(r["mean"]))
    assert r["burstiness"] > 0 and r["n_tokens"] > 20


def test_flags_low_surprisal_sentences_only():
    r = surprisal.score_section_surprisal(TEXT, Stub(), low=3.0, window=3.0)
    assert {s["text"][:5] for s in r["low"]} == {"The d", "This "}


def test_citations_numbers_acronyms_and_rare_tokens_do_not_hide_predictable_prose():
    plain = "The data is a sample of the firms in the study and it is for the test"
    cited = plain + " (Turyahikayo & Maarouf, 2021). SHRM zzrare 2024 Dueñas."
    s = surprisal.score_section_surprisal(cited, Stub(), low=3.0, window=3.0)["sentences"]
    assert s[0]["raw_mean"] > s[0]["mean"] + 0.5        # unmasked tokens only: the cited sentence still reads as predictable
    assert s[0]["mean"] < 3.0 and s[0]["flagged"]
    junk = surprisal.score_section_surprisal("SHRM 2024 zzrare Dueñas (Smith, 2020) !!", Stub(), low=3.0, window=3.0)
    assert junk is None or junk["sentences"][0]["n"] == 0 or not junk["sentences"][0]["flagged"]     # nothing unmasked: never flagged


def test_mid_sentence_capitals_are_masked_but_the_first_word_is_not():
    r = surprisal.score_section_surprisal("Quokka tessellations perturb Zygomatic ledgers in Leeds.", Stub(), low=3.0, window=3.0)
    s = r["sentences"][0]
    assert s["n"] == 5           # Quokka (first word, kept), tessellations, perturb, ledgers, in; Zygomatic and Leeds (mid-sentence capitals) and "." are masked


def test_min_window_finds_the_predictable_glue_inside_a_technical_sentence():
    text = "Quokka tessellations perturb in the sense that why employees zygomatic ledgers frescoes."
    s = surprisal.score_section_surprisal(text, Stub(), low=3.0, window=3.0)["sentences"][0]
    assert s["min_window"] == pytest.approx(1.0)
    assert s["predictable_span"] == "in the sense that why employees"
    assert s["mean"] > 3.0 and s["flagged"]             # the sentence mean hides it; the window catches it


def test_lstr_is_the_share_of_top5_tokens():
    s = surprisal.score_section_surprisal("The data is a quokka tessellations perturb ledgers.", Stub(), low=3.0, window=3.0)["sentences"][0]
    assert s["lstr"] == pytest.approx(3 / 8, abs=0.01)     # The, is, a are rank 1; data, quokka, tessellations, perturb, ledgers are not


def test_no_scorer_means_none_and_never_raises(monkeypatch):
    monkeypatch.setenv("HARNESS_SURPRISAL", "off")
    assert surprisal.get_scorer() is None
    assert surprisal.score_section_surprisal(TEXT, None) is None
    monkeypatch.delenv("HARNESS_SURPRISAL")
    monkeypatch.setenv("HARNESS_SURPRISAL_DIR", "/nonexistent")
    surprisal._SCORER = None
    assert surprisal.get_scorer() is None          # model files missing -> safe fallback


def test_a_broken_scorer_is_a_none_not_an_exception():
    class Bad:
        def token_logprobs(self, text):
            raise RuntimeError("boom")
    assert surprisal.score_section_surprisal(TEXT, Bad()) is None


def test_long_text_is_scored_one_paragraph_at_a_time():
    surprisal.score_section_surprisal("\n\n".join(f"{PLAIN} v{i}" for i in range(60)), Stub())
    assert Stub.calls == 60 and max(len(p) for p in Stub.paragraphs) < 400


def test_unchanged_paragraphs_are_not_scored_twice():
    sc = Stub()
    surprisal.score_section_surprisal(TEXT, sc)
    n = Stub.calls
    surprisal.score_section_surprisal(TEXT + "\n\nA new paragraph about the study.", sc)
    assert Stub.calls == n + 1                     # only the new paragraph went through the model


def test_detector_adds_low_surprisal_type_and_names_the_phrase():
    text = "Quokka tessellations perturb as it is that why employees zygomatic ledgers frescoes."
    r = surprisal.score_section_surprisal(text, Stub(), low=3.0, window=3.0)
    spans = lamp_detector.detect_ai_spans(text, surprisal=r)
    hit = [sp for sp in spans if "low_surprisal" in sp["types"]]
    assert hit and "as it is that why employees" in hit[0]["instruction"]
    assert not any("low_surprisal" in sp["types"] for sp in lamp_detector.detect_ai_spans(text))      # off without scores


def test_rewrite_must_raise_surprisal(monkeypatch):
    low = "The data is a sample of the firms in the study and it is for the test."
    text = low + " " + "Quokka tessellations perturb zygomatic ledgers in the Q3 audit (Smith, 2020)."
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["%s"]' % low)           # no gain
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, window=3.0, rounds=1)
    assert out == text and rep["rejected_by"].get("surprisal not higher") == 1
    better = "Firms sampled by purposive recruitment yielded sparse panels."
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["%s"]' % better)
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, window=3.0, rounds=1)
    assert better in out and rep["surprisal_after"]["mean"] > rep["surprisal_before"]["mean"]


def test_rescoring_after_a_rewrite_only_touches_the_changed_paragraph(monkeypatch):
    p1 = "The data is a sample of the firms in the study and it is for the test."
    p2 = "Quokka tessellations perturb zygomatic ledgers in the Q3 audit (Smith, 2020)."
    text = p1 + "\n\n" + p2
    better = "Firms sampled by purposive recruitment yielded sparse panels."
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["%s"]' % better)
    lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, window=3.0, rounds=1)
    assert Stub.paragraphs.count(p2) == 1          # scored once, though the section was scored before and after


def test_second_round_retries_only_the_rejected(monkeypatch):
    low = "The data is a sample of the firms in the study and it is for the test."
    text = low + " Quokka tessellations perturb zygomatic ledgers in the Q3 audit (Smith, 2020)."
    replies = iter(['["%s"]' % low, '["Purposive recruitment yielded sparse firm panels."]'])
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: next(replies))
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, window=3.0, rounds=2)
    assert "Purposive recruitment" in out and rep["rounds"] == 2


def test_without_scorer_rewrite_is_unchanged(monkeypatch):
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["x"]')
    _, rep = lamp_rewrite.lamp_rewrite_section("Short.", "methodology", [], scorer=None)
    assert "surprisal_before" not in rep


@pytest.mark.skipif(not (surprisal.model_dir() / "lm-q.onnx").exists(), reason="local model not downloaded")
def test_real_model_reads_stock_prose_as_more_predictable(monkeypatch):
    monkeypatch.delenv("HARNESS_SURPRISAL", raising=False)
    surprisal._SCORER = None
    sc = surprisal.get_scorer()
    assert sc is not None
    stock = "In recent years, there has been a growing interest in the role of technology in the field of education."
    odd = "Quokka tessellations perturb zygomatic ledgers while mulch frescoes defenestrate gantry hymns."
    a, b = (surprisal.score_section_surprisal(x, sc)["mean"] for x in (stock, odd))
    assert a < b
    toks = sc.token_logprobs(stock)
    assert all(len(t) == 4 and t[3] >= 1 for t in toks)       # (start, end, ln p, rank)


def test_surprisal_is_opt_in_for_the_rewrite(monkeypatch):
    called = []
    monkeypatch.setattr(surprisal, "get_scorer", lambda: called.append(1))
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["x"]')
    monkeypatch.delenv("HARNESS_SURPRISAL", raising=False)
    lamp_rewrite.lamp_rewrite_section("Short.", "methodology", [])
    assert not called                                            # default off: the model is never touched
    monkeypatch.setenv("HARNESS_SURPRISAL", "on")
    lamp_rewrite.lamp_rewrite_section("Short.", "methodology", [])
    assert called
