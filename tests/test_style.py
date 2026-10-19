import random

import pytest

from harness.validate.features import FEATURES, style_features
from harness.validate.style import build_baseline, score_section, split_chapter_by_sections, score_features, style_target

HUMAN = [
    "Adoption is uneven.",
    "Smith (2019) reports large gains in output, although the sample was drawn almost entirely from firms that already had "
    "some digital capability, which makes the headline figure hard to trust.",
    "I am not convinced.",
    "The literature rarely addresses what happens after the first year, when the novelty fades and the staff who championed "
    "the change have moved on to other projects (Jones, 2020).",
    "That may be too generous a reading.",
    "In practice, managers describe the tools as helpful but tiring, a mixed verdict that the survey instruments in most "
    "studies were not built to capture.",
    "Why does this matter?",
    "Because the cost-benefit story depends on a timescale that firms seldom share with researchers, and because the "
    "evidence is, at best, suggestive rather than settled (Lee et al., 2018).",
    "We return to this below.",
    "Perhaps the more useful question concerns capability: what a firm can actually do with the system once it exists, "
    "and who inside the firm is permitted to try.",
    "The evidence appears thin.",
    "Brown (2021) tends to agree, yet the argument leans on a single case that could be atypical in ways the author only "
    "partly acknowledges.",
    "Our own data point the same way, somewhat unevenly.",
    "It seems likely that scale matters, though the direction of the effect is not obvious from the tables alone and the "
    "confidence intervals are wide enough to embarrass a stronger claim.",
]

AI = [
    "Digital transformation plays a crucial role in the ever-evolving landscape of modern business.",
    "Furthermore, organizations must navigate the complexities of a multifaceted technological environment today.",
    "Moreover, the adoption of digital tools demonstrates significant benefits for organizational performance overall.",
    "Additionally, it is important to note that leadership commitment is a testament to successful change outcomes.",
    "Consequently, firms that embrace a holistic approach to innovation will clearly outperform their competitors always.",
    "Ultimately, the integration of technology underscores the importance of strategic alignment across all departments.",
    "The results delve into a rich tapestry of interconnected factors that shape the digital transformation journey.",
    "In conclusion, digital transformation offers undoubtedly transformative opportunities for organizations everywhere.",
]


def make(pool, seed, n=14, sep=" "):
    rng = random.Random(seed)
    sents = [rng.choice(pool) for _ in range(n)]
    paras, i = [], 0
    while i < len(sents):
        k = rng.choice([2, 3, 5])
        paras.append(sep.join(sents[i:i + k]))
        i += k
    return "\n\n".join(paras)


def test_none_under_150_words():
    assert style_features("Too short. " * 20) is None
    assert style_features("word " * 149) is None
    assert style_features("word. " * 151) is not None


def test_known_feature_values():
    # 10 sentences of exactly 20 words => mean 20, cv 0, no short/long
    sent = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty."
    f = style_features(" ".join([sent] * 10))
    assert f["sent_len_mean"] == 20 and f["sent_len_cv"] == 0
    assert f["short_sent_ratio"] == 0 and f["long_sent_ratio"] == 0
    assert f["opener_repeat"] == 1.0            # every sentence opens "one two"
    assert set(FEATURES) == set(f)
    hedged = "This may hold. " + "It is fine here today. " * 40
    g = style_features(hedged)
    n = len(hedged.split())
    assert g["hedge_per_100"] == pytest.approx(100 / n)
    assert style_features("Furthermore, moreover — dash. " + "filler words go here now. " * 40)["stock_per_1000"] > 0
    assert style_features("As Smith (2019) noted, and (Lee, 2020). " + "plain words continue on here. " * 40)["citation_per_100"] > 0


def test_ai_like_differs_from_human_like_in_expected_direction():
    h = style_features(make(HUMAN, 1))
    a = style_features(make(AI, 1, n=20))
    assert a["stock_per_1000"] > h["stock_per_1000"] + 20
    assert a["sent_len_cv"] < h["sent_len_cv"]
    assert a["hedge_per_100"] < h["hedge_per_100"]
    assert h["short_sent_ratio"] > a["short_sent_ratio"]


def test_figure_blocks_and_html_ignored():
    body = make(HUMAN, 2)
    noisy = "[FIGURE_START]\nTYPE: mermaid\nCODE:\ngraph TD\n[FIGURE_END]\n\n<h2>2.1 Heading</h2>\n\n" + body
    assert style_features(noisy)["sent_len_mean"] == pytest.approx(style_features(body)["sent_len_mean"])


@pytest.fixture(scope="module")
def baseline():
    units = [make(HUMAN, s) for s in range(30)]
    return build_baseline({"literature_review": units})


def test_baseline_shape(baseline):
    assert baseline["n_units"] == 30 and "all" in baseline["by_section"]
    e = baseline["by_section"]["literature_review"]
    assert e["n"] == 30 and set(e["features"]) == set(FEATURES)
    assert baseline["score_p10"]["literature_review"] < 100


def test_corpus_like_passes_ai_like_fails_with_expected_flags(baseline):
    ok = score_section(make(HUMAN, 999), "literature_review", baseline)
    assert ok["available"] and ok["passed"] and ok["style_score"] > 60
    bad = score_section(make(AI, 5, n=20), "literature_review", baseline)
    assert bad["available"] and not bad["passed"]
    flagged = {f["feature"]: f["direction"] for f in bad["flags"]}
    assert flagged.get("stock_per_1000") == "high"
    assert bad["style_score"] < ok["style_score"]


def test_unknown_section_falls_back_to_pooled_and_missing_baseline_unavailable(baseline):
    assert score_section(make(HUMAN, 7), "methodology", baseline)["available"]      # pooled 'all'
    assert score_section(make(HUMAN, 7), "methodology", {})["available"] is False   # empty baseline
    assert score_section("too short", "literature_review", baseline)["available"] is False


def test_style_target_reads_real_numbers_off_the_baseline(baseline):
    t = style_target("literature_review", baseline)
    mean = baseline["by_section"]["literature_review"]["features"]["sent_len_mean"]["mean"]
    assert f"{mean:.0f}" in t and "STYLE TARGET" in t
    assert style_target("literature_review", {}) == ""       # empty baseline -> no target, never raises
    assert style_target("nope", baseline) != ""               # falls back to pooled 'all'


def test_missing_baseline_file_never_raises(tmp_path):
    from harness.validate import style
    style.load_baseline.cache_clear()
    assert style.load_baseline(str(tmp_path / "nope.json")) is None


def test_one_sided_features_do_not_penalise_low(baseline):
    entry = baseline["by_section"]["literature_review"]
    feats = style_features(make(HUMAN, 3))
    low = {**feats, "stock_per_1000": 0.0, "opener_repeat": 0.0, "em_dash_per_para": 0.0}
    assert score_features(low, entry)["style_score"] >= score_features(feats, entry)["style_score"] - 0.5


def test_split_chapter_by_sections():
    content = ("<h2>2.1 Background</h2><p>" + make(HUMAN, 1).replace("\n\n", "</p><p>") + "</p>"
               "<h2>2.2 Debates</h2><p>debates text here</p>")
    parts = split_chapter_by_sections(content, [{"number": "2.1", "title": "Background"}, {"number": "2.2", "title": "Debates"}])
    assert [p["number"] for p in parts] == ["2.1", "2.2"]
    assert "debates text here" in parts[1]["text"] and "debates text" not in parts[0]["text"]
    whole = split_chapter_by_sections("just prose here", [{"number": "9.9", "title": "Nope"}])
    assert whole == [{"number": "", "title": "chapter", "text": "just prose here"}]
