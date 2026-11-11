"""validate/detector.py: the sentence-level Claude-vs-human scorer. Pure Python at runtime; these tests use a tiny hand-made model."""
import math
import time

import pytest

from harness.validate import detector as D

SENSE = "The gap exists in the sense that fit and performance are rarely measured together (Smith, 2021)."


def model(**kw):
    m = {"version": 1, "bias": -1.0, "threshold": 0.5,
         "ngrams": {"in the W": 3.0, "rather than": 2.0, "is the W": -1.0},
         "avoid": {"the literature": 1.5},
         "dense": {"names": list(D.DENSE), "mean": [0.0] * len(D.DENSE), "std": [1.0] * len(D.DENSE), "w": [0.0] * len(D.DENSE)}}
    m.update(kw)
    return m


def test_citations_numbers_and_names_are_masked_and_surface_text_is_kept():
    toks = D.tokenize("Smith and Lee (2021) found 235 firms (Jones, 2020) in Leeds, rather than elsewhere.")
    assert [t[0] for t in toks[:3]] == ["<CITE>", "found", "<NUM>"]
    assert ("<CITE>", "(Jones, 2020)") in [(t[0], t[1]) for t in toks]
    assert [t[0] for t in toks if t[0] == "NAME"] == ["NAME"]                       # Leeds: capitalised mid-sentence
    assert toks[0][1] == "Smith and Lee (2021)"


def test_function_words_and_academic_verbs_stay_and_content_words_become_classes():
    ab = [t[0] for t in D.tokenize("The organisation quickly demonstrated that implementation was considered desirable.")]
    assert ab == ["the", "NOM", "ADV", "demonstrated", "that", "NOM", "was", "ED", "ADJ"]
    assert "organisation" not in ab and "implementation" not in ab


def test_culprit_phrases_are_exact_substrings_in_weight_order():
    r = D.score_sentence(SENSE, model())
    assert r["culprit_phrases"][0] == "in the sense"
    assert all(p in SENSE or ">=" in p for p in r["culprit_phrases"])
    assert r["ai_prob"] == pytest.approx(1 / (1 + math.exp(-(-1.0 + 3.0))), abs=1e-6)
    assert r["flagged"]


def test_avoid_phrases_and_negative_weights():
    r = D.score_sentence("The literature is the W of nothing.", model())
    assert "The literature" in r["culprit_phrases"]            # surface text keeps the sentence's own capitalisation
    plain = D.score_sentence("Wards were staffed by nurses.", model())
    assert plain["ai_prob"] < 0.5 and not plain["flagged"] and plain["culprit_phrases"] == []


def test_structural_features_are_counted_and_named_when_they_push_toward_claude():
    names = list(D.DENSE)
    w = [0.0] * len(names)
    w[names.index("commas")] = 1.0
    m = model(ngrams={}, avoid={}, dense={"names": names, "mean": [0.0] * len(names), "std": [1.0] * len(names), "w": w}, bias=-3.0)
    s = "First, second, third, fourth, and fifth things were measured, noted, and compared by the team."
    r = D.score_sentence(s, m)
    assert any(p.startswith("commas>=") for p in r["culprit_phrases"]) and r["ai_prob"] > 0.5


def test_features_cover_the_twelve_requested_signals():
    f = D.dense_features("Although it is, which rather than that, a measured participation, studies suggest (Lee, 2020) 12 things; yes: done, being.")
    assert len(D.DENSE) == 12 and set(f) == set(D.DENSE)
    assert f["colon_semicolon"] == 1 and f["subordinators"] >= 2 and f["cites"] == 1 and f["nums"] == 1 and f["ing_clauses"] >= 1


def test_no_model_means_none(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "PATH", tmp_path / "missing.json")
    D.load.cache_clear()
    assert D.load() is None and D.score_sentence(SENSE) is None and D.score_text(SENSE) is None
    D.load.cache_clear()


def test_score_text_skips_headings_and_tables_and_numbers_paragraphs():
    text = "## 2.1 Heading\n\n" + SENSE + " Wards were staffed by nurses.\n\n| a | b |\n\n" + SENSE
    rows = D.score_text(text, model())
    assert [r["para"] for r in rows] == [1, 1, 3] and rows[0]["flagged"] and not rows[1]["flagged"]


def test_fifty_sentences_score_in_under_ten_milliseconds():
    text = " ".join([SENSE, "Wards were staffed by nurses on every shift of the year."] * 25)
    D.score_text(text, model())
    t = time.perf_counter()
    rows = D.score_text(text, model())
    ms = (time.perf_counter() - t) * 1000
    assert len(rows) == 50 and ms < 10
