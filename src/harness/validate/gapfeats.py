"""50 measurable text features, shared by eval/gap_report.py (finds the gaps), corpus/profile.py (mines real targets) and
the probe. Rates are per 1000 words unless the name says share / cv / mean. None for text under 400 words."""
import re
import statistics
from collections import Counter

from ..text.citations import extract_citations
from ..text.strip import strip_for_style

TOK = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")
CONNECTIVES = ("however", "therefore", "thus", "moreover", "furthermore", "also", "additionally", "hence")
BOOSTERS = ("clearly", "obviously", "certainly", "definitely", "undoubtedly", "genuinely", "fundamentally", "deeply")
HEDGES = ("may", "might", "could", "perhaps", "possibly", "appears", "seems", "suggests", "likely", "tend", "tends")
_CITE_YEARS = re.compile(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)")
_SIGNPOST = re.compile(r"(?:for (?:example|instance)|in addition|however|according to|based on|as a result|in contrast|in particular)\b", re.I)


def short_sentences(text: str) -> tuple[int, int]:
    """(sentences under ten words, all sentences of three words or more) in `text`."""
    slen = [len(TOK.findall(s)) for s in SENT.split(strip_for_style(text))]
    slen = [x for x in slen if x >= 3]
    return sum(1 for x in slen if x < 10), len(slen)


def feats(text: str) -> dict | None:
    plain = strip_for_style(text)
    toks = TOK.findall(plain)
    n = len(toks)
    if n < 400:
        return None
    low = [t.lower() for t in toks]
    sents = [s for s in SENT.split(plain) if len(TOK.findall(s)) >= 3]
    if not sents:
        return None
    slen = [len(TOK.findall(s)) for s in sents]
    paras = [len(TOK.findall(p)) for p in re.split(r"\n\s*\n", plain) if len(TOK.findall(p)) >= 5]
    c = Counter(low)
    per = lambda k: 1000 * k / n
    cnt = lambda rx: len(re.findall(rx, plain, re.I))
    firsts = [TOK.findall(s)[0].lower() for s in sents]
    fc = Counter(firsts)
    cites = extract_citations(plain)
    nar = sum(1 for x in cites if not x["raw"].startswith("("))
    first1000 = low[:1000]
    return {
        "sent_len_mean": statistics.fmean(slen), "sent_len_cv": statistics.pstdev(slen) / statistics.fmean(slen),
        "short_sent_share": sum(1 for x in slen if x < 10) / len(slen), "long_sent_share": sum(1 for x in slen if x > 40) / len(slen),
        "sent_len_max": max(slen),
        "para_per_1000": 1000 * len(paras) / n, "para_len_cv": (statistics.pstdev(paras) / statistics.fmean(paras)) if len(paras) > 1 else 0,
        "para_short_share": sum(1 for x in paras if x < 40) / max(1, len(paras)),
        "opener_distinct": len(fc) / len(firsts), "opener_top_share": fc.most_common(1)[0][1] / len(firsts),
        "opener_the_share": fc["the"] / len(firsts), "opener_this_share": fc["this"] / len(firsts),
        "ttr_1000": len(set(first1000)) / len(first1000), "word_len": statistics.fmean(len(t) for t in toks),
        "long_word_share": sum(1 for t in toks if len(t) >= 9) / n,
        "comma": per(plain.count(",")), "semicolon": per(plain.count(";")), "colon": per(plain.count(":")),
        "paren": per(plain.count("(")), "quote": per(plain.count('"') + plain.count("“")), "question": per(plain.count("?")),
        "digits": per(len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", _CITE_YEARS.sub("", plain)))),
        "percent": per(plain.count("%")), "dash": per(plain.count("—") + plain.count("–")),
        "hyphen_words": per(len(re.findall(r"\w-\w", plain))),
        "acronyms": per(len(re.findall(r"\b[A-Z]{2,6}s?\b", plain))), "capitalised_mid": per(len(re.findall(r"(?<=[a-z,] )[A-Z][a-z]{2,}", plain))),
        "figure_table_refs": per(cnt(r"\b(?:figure|fig\.|table|chapter|section|appendix)\s+\d")),
        "eg_ie": per(cnt(r"\b(?:e\.g\.|i\.e\.|etc\.)")),
        "first_person": per(sum(c[w] for w in ("i", "my", "me", "we", "our", "us"))),
        "passive": per(len(re.findall(r"\b(?:was|were|been|being|is|are)\s+\w+ed\b", plain))),
        "past_aux": per(c["was"] + c["were"] + c["had"] + c["did"]), "future": per(c["will"]),
        "modal": per(sum(c[w] for w in ("could", "would", "might", "should", "may", "can"))),
        "hedge": per(sum(c[w] for w in HEDGES)), "booster": per(sum(c[w] for w in BOOSTERS)),
        "and": per(c["and"]), "of": per(c["of"]), "that": per(c["that"]), "which": per(c["which"]), "the": per(c["the"]),
        "connectives": per(sum(c[w] for w in CONNECTIVES)),
        "nominalisation": per(sum(1 for t in low if re.search(r"(?:tion|ity|ment|ness|ism)s?$", t))),
        "meta_study": per(cnt(r"\b(?:this (?:study|thesis|chapter|section|research)|the (?:present|current) study)\b")),
        "the_field_lit": per(cnt(r"\bthe (?:literature|field)\b")),
        "signpost_open": sum(1 for s in sents if _SIGNPOST.match(s)) / len(sents),
        "cite_per_100": 100 * len(cites) / n, "cite_narrative_share": (nar / len(cites)) if cites else 0.0,
        "et_al": per(cnt(r"\bet al\b")),
        "contractions": per(len(re.findall(r"\b\w+n['’]t\b", plain))),
    }
