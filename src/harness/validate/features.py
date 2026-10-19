"""Deterministic style features. The same code measures corpus units (baseline) and drafts."""
import re
import statistics

from ..text.citations import extract_citations
from ..text.sentences import paragraphs, split_sentences, words
from ..text.strip import strip_for_style

MIN_WORDS = 150

HEDGES = ["may", "might", "could", "suggests", "appears", "seems", "arguably", "potentially", "likely",
          "tends", "somewhat", "possibly", "perhaps", "indicates"]
BOOSTERS = ["clearly", "undoubtedly", "demonstrates", "proves", "obviously", "certainly", "significantly"]
STOCK_PHRASES = ["delve", "tapestry", "testament to", "plays a crucial role", "it is important to note",
                 "it is worth noting", "underscores", "multifaceted", "holistic", "navigate the complexities",
                 "in today's", "ever-evolving", "landscape of", "furthermore", "moreover", "additionally",
                 "consequently", "ultimately", "in conclusion"]
FIRST_PERSON = {"i", "my", "we", "our", "me", "us"}

FEATURES = ["sent_len_mean", "sent_len_cv", "short_sent_ratio", "long_sent_ratio", "para_len_cv",
            "hedge_per_100", "booster_per_100", "citation_per_100", "first_person_per_100",
            "opener_repeat", "stock_per_1000", "em_dash_per_para"]
# only "too high" is a warning sign for these
ONE_SIDED_HIGH = {"stock_per_1000", "opener_repeat", "em_dash_per_para"}

_HEDGE_RX = re.compile(r"\b(" + "|".join(HEDGES) + r")\b", re.I)
_BOOST_RX = re.compile(r"\b(" + "|".join(BOOSTERS) + r")\b", re.I)
_STOCK_RX = re.compile(r"\b(" + "|".join(re.escape(p) for p in STOCK_PHRASES) + r")", re.I)


def _cv(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    m = statistics.fmean(values)
    return statistics.pstdev(values) / m if m else 0.0


def style_features(text: str) -> dict | None:
    """None when the passage is under 150 words (too short to measure reliably)."""
    plain = strip_for_style(text)
    toks = words(plain)
    n = len(toks)
    if n < MIN_WORDS:
        return None
    sents = split_sentences(plain)
    lens = [len(words(s)) for s in sents] or [n]
    paras = [len(words(p)) for p in paragraphs(plain)] or [n]

    openers = [" ".join(w.lower() for w in words(s)[:2]) for s in sents if len(words(s)) >= 2]
    counts: dict[str, int] = {}
    for o in openers:
        counts[o] = counts.get(o, 0) + 1
    opener_repeat = sum(1 for o in openers if counts[o] > 1) / len(openers) if openers else 0.0

    per100 = 100.0 / n
    return {
        "sent_len_mean": statistics.fmean(lens),
        "sent_len_cv": _cv(lens),
        "short_sent_ratio": sum(1 for x in lens if x < 8) / len(lens),
        "long_sent_ratio": sum(1 for x in lens if x > 35) / len(lens),
        "para_len_cv": _cv(paras),
        "hedge_per_100": len(_HEDGE_RX.findall(plain)) * per100,
        "booster_per_100": len(_BOOST_RX.findall(plain)) * per100,
        "citation_per_100": len(extract_citations(plain)) * per100,
        "first_person_per_100": sum(1 for t in toks if (t == "I" or t.lower() in FIRST_PERSON)) * per100,
        "opener_repeat": opener_repeat,
        "stock_per_1000": len(_STOCK_RX.findall(plain)) * 1000.0 / n,
        "em_dash_per_para": plain.count("—") / max(1, len(paras)),
    }
