"""Plain-word swaps (cleanup.plain_words): shorter twins for generic academic words, nothing else touched."""
from harness.text import cleanup as C


def _run(text):
    return C.plain_words({"open": text, "mid": "", "close": ""})


def test_swaps_words_and_phrases_and_keeps_capitals():
    out, n = _run("We utilize two boards in order to compare them. Furthermore, sufficient data was obtained.")
    assert out["open"] == "We use two boards to compare them. Also, enough data was got." and n == 5


def test_leaves_quotes_citations_and_domain_words_alone():
    for t in ('He wrote "utilize it" here.', "Lee (2015) measured bandwidth and monitoring quality."):
        out, n = _run(t)
        assert out["open"] == t and n == 0


def test_is_idempotent_and_ignores_embedded_words():
    once, _ = _run("The utilization of this unutilized board was approximately fine.")
    twice, n = C.plain_words(once)
    assert once == twice and n == 0 and "unutilized" in once["open"]
