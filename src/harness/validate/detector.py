"""Sentence-level "does this read as Claude?" scorer: a linear model over function-word n-grams plus 12 structural features.

Pure Python, no ML library at runtime: data/claude_sentence_detector.json holds the weights, written by scripts/train_sentence_detector.py.
~0.1 ms per sentence. Domain nouns cannot leak in: every content word is replaced by a class (NOM, ADV, ING, ED, ADJ, NAME, W), citations
become <CITE> and numbers <NUM>; only function words, academic verbs and discourse words keep their spelling.

score_sentence(s) -> {ai_prob, flagged, culprit_phrases}: culprit_phrases are the (up to 3) exact stretches of the sentence, or structural
features such as "commas>=5", that push the score most toward Claude. None when no model file exists (callers skip)."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from ..text.sentences import split_sentences

PATH = Path(__file__).resolve().parents[3] / "data" / "claude_sentence_detector.json"
MIN_WORDS = 8
DENSE = ("words", "commas", "colon_semicolon", "subordinators", "nominal_ratio", "passive_ratio", "ing_clauses", "avg_word_len", "cites",
         "nums", "connective_open", "comma_per_word")
_COUNT_FEATURES = {"commas", "subordinators", "ing_clauses", "cites", "nums", "words"}

_FUNCTION = frozenset("""a an the this that these those it its they them their we our i you he she his her there here who whom whose which what
where when while whereas although though if unless because since as so than then thus hence yet but and or nor not no nor either neither both
each every all any some most many much more less few several other another such same own only also even just still already ever never
of in on at by for with without within into onto upon over under between among across through during before after about above below against
toward towards from to up down out off per via is are was were be been being am do does did done doing have has had having can could may
might will would shall should must ought one two three first second third whether how why rather instead despite regarding given
across along around beyond behind beside besides""".split())
_ACADEMIC = frozenset("""argue argues argued suggest suggests suggested demonstrate demonstrates demonstrated indicate indicates indicated reveal
reveals revealed highlight highlights highlighted underscore underscores underscored emphasise emphasises emphasised emphasize emphasizes
emphasized examine examines examined explore explores explored show shows showed shown find finds found note notes noted report reports
reported claim claims claimed maintain maintains maintained contend contends contended propose proposes proposed identify identifies identified
describe describes described define defines defined conceptualise conceptualised conceptualize conceptualized observe observes observed
conclude concludes concluded establish establishes established illustrate illustrates illustrated reflect reflects reflected imply implies
implied require requires required allow allows allowed produce produces produced remain remains remained mean means meant carry carries carried
shape shapes shaped depend depends depended rest rests rested sit sits""".split())
_DISCOURSE = frozenset("""however moreover furthermore therefore consequently nevertheless nonetheless notably indeed accordingly thus overall
importantly crucially essentially ultimately similarly conversely meanwhile additionally specifically particularly broadly largely partly
arguably clearly rather instead""".split())
KEEP = _FUNCTION | _ACADEMIC | _DISCOURSE
_SUBORD = re.compile(r"\b(?:which|where|whereas|while|although|though|despite|rather than|in the sense|whereby|insofar)\b", re.I)
_PASSIVE = re.compile(r"\b(?:is|are|was|were|been|be|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.I)
_ING_CLAUSE = re.compile(r",\s+(?:\w+ly\s+)?\w+ing\b", re.I)
_NOMINAL = re.compile(r"(?:tion|tions|sion|ment|ments|ity|ities|ance|ances|ence|ences|ness|ism)$", re.I)
_CONNECTIVE = frozenset("however moreover furthermore therefore consequently nevertheless nonetheless notably indeed accordingly thus overall importantly similarly conversely meanwhile additionally".split())
_TOK = re.compile(
    r"(?P<cite>\([^()]*(?:19|20)\d\d[a-z]?[^()]*\))"
    r"|(?P<narr>[A-Z][\w'’-]+(?:\s+(?:and|&)\s+[A-Z][\w'’-]+|\s+et\s+al\.?)?\s*\((?:19|20)\d\d[a-z]?\))"
    r"|(?P<num>\d[\d.,]*%?)"
    r"|(?P<word>[A-Za-z][A-Za-z'’-]*)"
    r"|(?P<punc>[,;:])")
_BLOCK = re.compile(r"\[(?:FIGURE|TABLE)_START\]|^\s*[|!<]", re.M)


@lru_cache(maxsize=60000)
def _class(w: str, first: bool) -> str:
    low = w.lower()
    if low in KEEP:
        return low
    if w[0].isupper() and not first:
        return "NAME"
    if _NOMINAL.search(low):
        return "NOM"
    if low.endswith("ly"):
        return "ADV"
    if low.endswith("ing"):
        return "ING"
    if low.endswith(("ed", "en")):
        return "ED"
    if low.endswith(("ous", "ive", "al", "ic", "ful", "less", "able", "ible")):
        return "ADJ"
    return "W"


def tokenize(s: str) -> list[tuple[str, str, int, int]]:
    """[(abstract token, surface text, start, end)]: citations -> <CITE>, numbers -> <NUM>, content words -> a class."""
    out = []
    for m in _TOK.finditer(s):
        kind = m.lastgroup
        if kind in ("cite", "narr"):
            tok = "<CITE>"
        elif kind == "num":
            tok = "<NUM>"
        elif kind == "punc":
            tok = m.group()
        else:
            tok = _class(m.group(), not out)
        out.append((tok, m.group(), m.start(), m.end()))
    return out


def ngrams(toks: list) -> dict:
    """{key: surface text of its first occurrence} for every 1-3 gram of abstract tokens."""
    out = {}
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            key = " ".join(t[0] for t in toks[i:i + n])
            if key not in out:
                out[key] = (toks[i][2], toks[i + n - 1][3])
    return out


def dense_features(s: str, toks: list | None = None) -> dict:
    toks = tokenize(s) if toks is None else toks
    words = [t for t in toks if t[0] not in ("<CITE>", "<NUM>", ",", ";", ":")]
    n = max(1, len(words))
    commas = s.count(",")
    nom = sum(1 for t in words if t[0] == "NOM")
    return {"words": float(len(s.split())), "commas": float(commas), "colon_semicolon": float(bool(re.search(r"[;:]", s))),
            "subordinators": float(len(_SUBORD.findall(s))), "nominal_ratio": nom / n, "passive_ratio": len(_PASSIVE.findall(s)) / n,
            "ing_clauses": float(len(_ING_CLAUSE.findall(s))), "avg_word_len": sum(len(t[1]) for t in words) / n,
            "cites": float(sum(1 for t in toks if t[0] == "<CITE>")), "nums": float(sum(1 for t in toks if t[0] == "<NUM>")),
            "connective_open": float(bool(toks) and toks[0][1].lower() in _CONNECTIVE), "comma_per_word": commas / n}


def featurize(s: str) -> tuple[dict, dict]:
    """(n-gram key -> (start, end), dense feature dict)."""
    toks = tokenize(s)
    return ngrams(toks), dense_features(s, toks)


@lru_cache(maxsize=2)
def _load(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(path=None):
    """The shipped model dict, or None when data/claude_sentence_detector.json is missing or unreadable."""
    return _load(str(path or PATH))


load.cache_clear = _load.cache_clear           # type: ignore[attr-defined]


_PUNCT = frozenset((",", ";", ":"))


def score_sentence(s: str, model=None, threshold: float | None = None) -> dict | None:
    """-> {ai_prob, flagged, culprit_phrases} or None without a model. culprit_phrases: up to 3 stretches of the sentence (2-3 words, no
    citation or number, no punctuation at the ends) or structural features ("commas>=5") that push the score toward Claude, strongest first."""
    m = model if model is not None else load()
    if m is None:
        return None
    toks = tokenize(s)
    dense = dense_features(s, toks)
    z = m["bias"]
    contrib = []                                                      # (contribution, label)
    ngw, seen = m["ngrams"], set()
    ab = [t[0] for t in toks]
    for n in (1, 2, 3):
        for i in range(len(ab) - n + 1):
            key = ab[i] if n == 1 else " ".join(ab[i:i + n])
            w = ngw.get(key)
            if w and key not in seen:                                 # binary presence: each key counts once, as in training
                seen.add(key)
                z += w
                if w > 0 and n > 1 and ab[i] not in _PUNCT and ab[i + n - 1] not in _PUNCT and "<CITE>" not in key and "<NUM>" not in key:
                    contrib.append((w, s[toks[i][2]:toks[i + n - 1][3]]))
    low = s.lower()
    for phrase, w in m.get("avoid", {}).items():
        j = low.find(phrase)
        if j >= 0:
            z += w
            if w > 0:
                contrib.append((w, s[j:j + len(phrase)]))
    d = m["dense"]
    for name, mean, std, w in zip(d["names"], d["mean"], d["std"], d["w"]):
        v = w * (dense[name] - mean) / (std or 1.0)
        z += v
        if v > 0:
            contrib.append((v, f"{name}>={int(dense[name])}" if name in _COUNT_FEATURES else name))
    contrib.sort(key=lambda c: -c[0])
    culprits, seen_l = [], []
    for _, label in contrib:
        lab = label.lower()
        if not any(lab in x or x in lab for x in seen_l):
            culprits.append(label)
            seen_l.append(lab)
        if len(culprits) == 3:
            break
    p = 1 / (1 + math.exp(-z)) if z > -700 else 0.0
    t = m.get("threshold", 0.5) if threshold is None else threshold
    return {"ai_prob": p, "flagged": p >= t, "culprit_phrases": culprits}


def score_text(text: str, model=None, threshold: float | None = None) -> list[dict] | None:
    """One row per sentence (headings, tables and figure blocks skipped): {text, para, ai_prob, flagged, culprit_phrases}."""
    m = model if model is not None else load()
    if m is None:
        return None
    rows = []
    for i, para in enumerate(q.strip() for q in re.split(r"\n\s*\n", text or "") if q.strip()):
        if para.startswith("#") or _BLOCK.search(para):
            continue
        for s in split_sentences(para):
            rows.append({"text": s, "para": i, **score_sentence(s, m, threshold)})
    return rows
