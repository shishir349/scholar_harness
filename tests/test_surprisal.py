"""Token surprisal scorer: offline, with a stub scorer (no model file needed)."""
import math
import re

import pytest

from harness.text import lamp_rewrite
from harness.validate import lamp_detector, surprisal

COMMON = {"the", "of", "and", "is", "a", "to", "in", "that", "this", "it", "are", "for", "as", "with"}


class Stub:
    """Common words cost 1 nat, everything else 7. Offsets are character spans, like the real scorer returns."""
    calls = 0

    def token_logprobs(self, text):
        Stub.calls += 1
        return [(m.start(), m.end(), -1.0 if m.group().lower() in COMMON else -7.0) for m in re.finditer(r"[A-Za-z']+", text)]


PLAIN = "The data is a sample of the firms in the study and it is for the test. This is the case for the model in the study as it is."
ODD = "Quokka tessellations perturb zygomatic ledgers. Mulch frescoes defenestrate gantry hymns."
TEXT = PLAIN + "\n\n" + ODD


def test_scores_each_sentence_and_the_section():
    r = surprisal.score_section_surprisal(TEXT, Stub())
    plain = [s for s in r["sentences"] if s["text"].startswith(("The data", "This is"))]
    odd = [s for s in r["sentences"] if s["text"].startswith(("Quokka", "Mulch"))]
    assert plain and odd and max(s["mean"] for s in plain) < min(s["mean"] for s in odd)
    assert r["perplexity"] == pytest.approx(math.exp(r["mean"]))
    assert r["burstiness"] > 0 and r["n_tokens"] > 20


def test_flags_low_surprisal_sentences_only():
    r = surprisal.score_section_surprisal(TEXT, Stub(), low=3.0)
    assert {s["text"][:5] for s in r["low"]} == {"The d", "This "}


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


def test_long_text_is_scored_in_bounded_windows():
    seen = []

    class Spy(Stub):
        def token_logprobs(self, text):
            seen.append(len(text))
            return super().token_logprobs(text)
    surprisal.score_section_surprisal("\n\n".join([PLAIN] * 60), Spy())
    assert len(seen) == 60 and max(seen) < 400           # one call per paragraph, never the whole section


def test_detector_adds_low_surprisal_type():
    r = surprisal.score_section_surprisal(TEXT, Stub(), low=3.0)
    spans = lamp_detector.detect_ai_spans(TEXT, surprisal=r)
    assert any("low_surprisal" in sp["types"] for sp in spans)
    assert not any("low_surprisal" in sp["types"] for sp in lamp_detector.detect_ai_spans(TEXT))      # off without scores


def test_rewrite_must_raise_surprisal(monkeypatch):
    low = "The data is a sample of the firms in the study and it is for the test."
    text = low + " " + "Quokka tessellations perturb zygomatic ledgers in the Q3 audit (Smith, 2020)."
    flat_new = "The data is a sample of the firms in the study and it is for the test."          # no gain
    better = "Firms sampled by purposive recruitment yielded sparse panels."
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["%s"]' % flat_new)
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, rounds=1)
    assert out == text and rep["rejected_by"].get("surprisal not higher") == 1
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: '["%s"]' % better)
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, rounds=1)
    assert better in out and rep["surprisal_after"]["mean"] > rep["surprisal_before"]["mean"]


def test_second_round_retries_only_the_rejected(monkeypatch):
    low = "The data is a sample of the firms in the study and it is for the test."
    text = low + " Quokka tessellations perturb zygomatic ledgers in the Q3 audit (Smith, 2020)."
    replies = iter(['["%s"]' % low, '["Purposive recruitment yielded sparse firm panels."]'])
    monkeypatch.setattr(lamp_rewrite.llm, "ask", lambda *a, **k: next(replies))
    out, rep = lamp_rewrite.lamp_rewrite_section(text, "methodology", [], scorer=Stub(), low=3.0, rounds=2)
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
