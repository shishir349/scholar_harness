"""validate/gapfeats.py: the 50-feature extractor behind eval/gap_report.py and the mined targets."""
import pytest

from harness.validate.gapfeats import feats

S1 = "We measured the soil at 12 sites that were chosen at random. "          # 11 words, 1 digit token, 'that', first person
S2 = "However, the results were also stable. "                                  # 6 words, 2 connectives
S3 = "Costs rose sharply here. "                                                # 4 words
TEXT = (S1 + S2 + S3) * 30                                                      # 630 words, 90 sentences


def test_short_texts_are_not_measured():
    assert feats("Too short to measure.") is None


def test_counts_per_1000_words_and_shares_match_a_hand_count():
    f = feats(TEXT)
    assert f["that"] == pytest.approx(30 / 630 * 1000, rel=0.01)
    assert f["connectives"] == pytest.approx(60 / 630 * 1000, rel=0.01)
    assert f["digits"] == pytest.approx(30 / 630 * 1000, rel=0.01)
    assert f["first_person"] == pytest.approx(30 / 630 * 1000, rel=0.01)
    assert f["short_sent_share"] == pytest.approx(60 / 90, rel=0.01)
    assert f["sent_len_mean"] == pytest.approx(21 / 3, rel=0.01)


def test_numbers_inside_citations_do_not_count_as_data():
    f = feats(TEXT + " Lee (2015) argued otherwise (Kim, 2016). " * 5)
    assert f["digits"] == pytest.approx(30 / (630 + 5 * 4) * 1000, rel=0.02)
