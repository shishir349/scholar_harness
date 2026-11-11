"""The Claude-style detector inside LAMP: flagged sentences become spans that name the exact phrases to delete, and a rewrite is accepted only if
those phrases are gone and the local score went down. No API calls: llm.ask is replaced."""
import pytest

from harness.text import lamp_rewrite as LR
from harness.validate import detector as DT, lamp_detector as LD

from .test_detector import SENSE, model

PLAIN = "Wards were staffed by nurses on every shift of the year, as the rota shows."
TEXT = SENSE + " " + PLAIN


def rows(m=None):
    return DT.score_text(TEXT, m or model())


def test_a_flagged_sentence_becomes_a_span_that_names_the_phrase_to_delete():
    spans = LD.detect_ai_spans(TEXT, claude=rows())
    hit = [s for s in spans if "claude_style" in s["types"]]
    assert len(hit) == 1 and hit[0]["text"] == SENSE and hit[0]["culprits"][0] == "in the sense"
    assert '"in the sense"' in hit[0]["instruction"] and hit[0]["ai_prob"] > 0.5
    assert not [s for s in LD.detect_ai_spans(TEXT) if "claude_style" in s["types"]]          # no scores, no span


@pytest.mark.parametrize("culprit,directive", [
    ("avg_word_len", "Rewrite using concrete, everyday academic vocabulary; avoid heavy multi-syllabic nominalizations."),
    ("commas>=5", "Split this compound sentence at the midpoint into two separate sentences."),
    ("colon_semicolon", "Remove the colon/semicolon and structure as two independent sentences."),
])
def test_structural_culprits_become_explicit_directives_not_quoted_phrases(culprit, directive):
    ask = LD.claude_ask([culprit])
    assert directive in ask and '"' not in ask and culprit not in ask


def test_literal_phrases_stay_forbidden_and_mix_with_directives():
    ask = LD.claude_ask(["the field", "has not", "avg_word_len"])
    assert '"the field"' in ask and '"has not"' in ask and "everyday academic vocabulary" in ask


def test_structural_culprits_reach_the_span_instruction():
    r = [{"text": SENSE, "para": 0, "ai_prob": 0.9, "flagged": True, "culprit_phrases": ["commas>=6", "subordinators>=3"]}]
    ins = LD.detect_ai_spans(TEXT, claude=r)[0]["instruction"]
    assert "Split this compound sentence" in ins and "which/where/while" in ins


def _run(monkeypatch, reply):
    monkeypatch.setattr(LR.llm, "ask", lambda *a, **k: '["%s"]' % reply)
    return LR.lamp_rewrite_section(TEXT, "methodology", [], detector_model=model())


def test_rewrite_that_keeps_the_flagged_phrase_is_rejected(monkeypatch):
    out, rep = _run(monkeypatch, "The gap exists in the sense that fit and performance are rarely measured together (Smith, 2021).".replace("The gap", "A gap"))
    assert out == TEXT and rep["rejected_by"].get("kept a flagged phrase") == 1


def test_rewrite_that_does_not_lower_the_score_is_rejected(monkeypatch):
    from harness.validate import detector
    m = model(ngrams={"in the W": 3.0, "rather than": 3.0})
    monkeypatch.setattr(LR.llm, "ask", lambda *a, **k: '["Fit and performance are rarely measured together, rather than apart (Smith, 2021)."]')
    out, rep = LR.lamp_rewrite_section(TEXT, "methodology", [], detector_model=m)
    assert out == TEXT and rep["rejected_by"].get("detector score not lower") == 1


def test_clean_rewrite_is_accepted_and_the_report_shows_before_and_after(monkeypatch):
    new = "Fit and performance are rarely measured together (Smith, 2021)."
    out, rep = _run(monkeypatch, new)
    assert new in out and rep["applied"] == 1
    assert rep["detector"]["flagged_before"] == 1 and rep["detector"]["flagged_after"] == 0
    assert rep["detector"]["mean_prob_after"] < rep["detector"]["mean_prob_before"]


def test_off_switch_and_missing_model_leave_the_rewrite_alone(monkeypatch):
    monkeypatch.setattr(LR.llm, "ask", lambda *a, **k: "[]")
    monkeypatch.setenv("HARNESS_DETECTOR", "off")
    _, rep = LR.lamp_rewrite_section(TEXT, "methodology", [])
    assert "detector" not in rep and "claude_style" not in rep["kinds"]
    monkeypatch.setenv("HARNESS_DETECTOR", "on")
    monkeypatch.setattr(DT, "load", lambda path=None: None)
    _, rep = LR.lamp_rewrite_section(TEXT, "methodology", [])
    assert "detector" not in rep
