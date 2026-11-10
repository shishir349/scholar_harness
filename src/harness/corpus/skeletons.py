"""Human discourse skeletons: the SHAPE of real thesis passages (paragraph sizes, sentence-length cadence, how a paragraph
opens and closes, where it cites), mined from data/corpus.db with no thesis text kept. graph._para_plan hands one to the
writer in place of the invented plan.

    python -m harness.corpus.skeletons [--min-score N] [--out data/human_skeletons.json]

Everything here is pure (no model calls). A skeleton holds numbers and labels only; tests assert that."""
import argparse
import json
import random
import re
import statistics
from functools import lru_cache
from pathlib import Path

from .. import config
from ..text.citations import NARRATIVE_RE, PAREN_RE
from ..text.cleanup import CLOSER
from ..text.sentences import split_sentences
from ..validate.features import _STOCK_RX
from . import store

PATH = config.DATA / "human_skeletons.json"
TYPES = ("introduction", "literature_review", "methodology", "findings", "discussion", "conclusion")
MIN_PARAS = 3          # body paragraphs a chunk needs to yield a skeleton
MIN_PARA_WORDS = 20    # shorter lines are captions, list items, headings
MAX_SENT = 90
MIN_SENT_CV = 0.38     # mean within-paragraph sentence-length variation of an elite skeleton
KEEP_SHARE = 0.40      # top share of each (type, position) group by structural diversity
MAX_KEEP = 2500
POOL = 12              # candidates nearest in length that one chapter draws from, so its sections differ

# ── 1. heading re-classifier ──────────────────────────────────────────────────────────────────────────────────────────
# corpus/sections.py files every unrecognised body chapter under "findings" (65% of the corpus). Headings often still say
# what the chunk is. First match wins; only chunks labelled "findings" are moved.
_REHEAD = (
    ("conclusion", re.compile(r"\bconclu|\bconcluding\b|\bsummary\b", re.I)),
    ("discussion", re.compile(r"implication|limitation|future (?:work|research)|critique|\bdiscussion\b", re.I)),
    ("literature_review", re.compile(r"related work|background|prior (?:research|work)|theoretical|conceptual framework|"
                                     r"literature|\breview\b|state of the art", re.I)),
    ("methodology", re.compile(r"experimental|materials|architecture|implementation|data collection|\bdesign\b|formulation|"
                               r"protocol|\bmethods?\b|methodolog|sampling|procedure|instrument", re.I)),
)


def reclassify(heading: str | None, section_type: str) -> str:
    """The section type to index a chunk under: its stored type, except a catch-all "findings" chunk whose heading says more."""
    if section_type != "findings":
        return section_type
    for typ, rx in _REHEAD:
        if rx.search(heading or ""):
            return typ
    return section_type


# ── 2. paragraph analysis ─────────────────────────────────────────────────────────────────────────────────────────────
_CONTRAST = re.compile(r"^(?:however|yet|but|although|though|while|whereas|despite|in contrast|conversely|nevertheless|nonetheless|"
                       r"instead|unlike|on the other hand|by contrast|still)\b", re.I)
_METHOD = re.compile(r"^(?:i|we|to|using|data|participants|interviews?|samples?|respondents?|(?:this|the|my|our) "
                     r"(?:study|thesis|research|chapter|researcher|survey|sample|analysis|approach|design))\b", re.I)
_NARR = re.compile(r"^[A-Z][\w'’-]+(?: et al\.?)?(?: (?:and|&) [A-Z][\w'’-]+)? \((?:19|20)\d\d")   # "Smith (2010)", never "In 2019"
_CROSSREF = re.compile(r"\b(?:chapter|section|table|figure|next|following|below|above|subsequent|turns? to|will (?:be )?"
                       r"(?:discuss|examin|present|explor|describ))", re.I)
_LIMIT = re.compile(r"\b(?:limit\w*|however|although|though|caveat|cannot|only|may not|beyond the scope|constrain\w*|"
                    r"unclear|uncertain|not (?:be|been|yet)|remains? (?:un\w+|limited|open))\b", re.I)


def opener_type(sentence: str) -> str:
    first5 = " ".join(sentence.split()[:5])
    if _NARR.match(first5):
        return "author_narrative_cite"
    if re.search(r"\d", first5):
        return "empirical_figure_or_metric"
    if _CONTRAST.match(first5):
        return "contrastive_pivot"
    if _METHOD.match(first5):
        return "methodological_action"
    return "domain_noun_subject"


def closer_type(sentence: str) -> str:
    if _CROSSREF.search(sentence):
        return "cross_ref_or_transition"
    if re.search(r"\d", sentence) or re.search(r"\([^()]*(?:19|20)\d\d[a-z]?\)\W*$", sentence):
        return "raw_data_or_cite_cutoff"
    if _LIMIT.search(sentence):
        return "limitation_or_caveat"
    return "comparative_assertion"


def _cv(vals: list) -> float:
    m = statistics.fmean(vals) if vals else 0.0
    return statistics.pstdev(vals) / m if len(vals) > 1 and m else 0.0


def _body(text: str) -> list[str]:
    out = []
    for p in re.split(r"\n\s*\n", text or ""):
        p = re.sub(r"\*{1,3}", "", p).strip()
        if p and not p.startswith(("#", "|", "![")) and len(p.split()) >= MIN_PARA_WORDS:
            out.append(p)
    return out


def paragraph_shape(p: str) -> dict | None:
    sents = split_sentences(p)
    if not sents:
        return None
    lens = [len(s.split()) for s in sents]
    narrative, paren, at = 0, 0, []
    for i, s in enumerate(sents, 1):
        n = sum(1 for m in NARRATIVE_RE.finditer(s) if s[max(0, m.start() - 1):m.start()] != "(")
        k = len(PAREN_RE.findall(s))
        if n or k:
            at.append(i)
            narrative, paren = narrative + n, paren + k
    mode = "none" if not at else "narrative" if narrative > paren else "parenthetical"
    return {"word_count": len(p.split()), "sentence_lengths": lens, "sent_len_cv": round(_cv(lens), 3),
            "opener_type": opener_type(sents[0]),
            "cite_positions": {"sentences": at, "mode": mode},
            "closer_type": closer_type(sents[-1])}


def skeleton_of(text: str, section_type: str, position: str, sid: str) -> dict | None:
    """The shape of one chunk, or None when it has under MIN_PARAS body paragraphs. Keeps no words of `text`."""
    paras = [x for x in (paragraph_shape(p) for p in _body(text)) if x]
    if len(paras) < MIN_PARAS or any(max(p["sentence_lengths"]) > MAX_SENT for p in paras):   # a 100-word "sentence" is a parse failure
        return None
    return {"skeleton_id": sid, "section_type": section_type, "position": position,
            "total_words": sum(p["word_count"] for p in paras), "paragraphs": paras}


def _formulaic_closer(text: str) -> bool:
    return any(CLOSER.search(split_sentences(p)[-1]) for p in _body(text) if split_sentences(p)) or bool(_STOCK_RX.search(text))


def diversity(sk: dict) -> float:
    return _cv([p["word_count"] for p in sk["paragraphs"]]) + _mean_sent_cv(sk)


def _mean_sent_cv(sk: dict) -> float:
    cvs = [p["sent_len_cv"] for p in sk["paragraphs"] if len(p["sentence_lengths"]) >= 2]
    return statistics.fmean(cvs) if cvs else 0.0


# ── 3. build the index from corpus.db ─────────────────────────────────────────────────────────────────────────────────
def build(con, min_score: float | None = None, keep_share: float = KEEP_SHARE, max_keep: int = MAX_KEEP) -> dict:
    """-> {"skeletons": [...], "stats": {...}}. Reads non-holdout chunks of theses scoring >= min_score (default 0: every ingested
    thesis already passed the score.py gate; theses.score runs 0-100, median 53). Keeps the top `keep_share` of each (type, position) group by diversity after the
    hard filters: mean sentence-length variation >= MIN_SENT_CV and no formulaic closer or stock phrase."""
    min_score = 0.0 if min_score is None else min_score
    rows = con.execute("""SELECT c.id, c.thesis_id, c.chunk_idx, c.section_type, c.position, c.heading, c.text
                          FROM chunks c JOIN theses t ON t.id=c.thesis_id
                          WHERE t.holdout=0 AND t.score >= ? ORDER BY c.id""", (min_score,)).fetchall()
    groups: dict = {}
    moved = scanned = 0
    for r in rows:
        typ = reclassify(r["heading"], r["section_type"])
        if typ not in TYPES:
            continue
        scanned += 1
        moved += typ != r["section_type"]
        sk = skeleton_of(r["text"], typ, r["position"], f"{r['thesis_id']}_{r['chunk_idx']}_{r['id']}")
        if sk and _mean_sent_cv(sk) >= MIN_SENT_CV and not _formulaic_closer(r["text"]):
            groups.setdefault((typ, r["position"]), []).append(sk)
    elite, per = [], {}
    for key, items in sorted(groups.items()):
        items.sort(key=lambda s: (-diversity(s), s["skeleton_id"]))
        top = items[:max(1, int(len(items) * keep_share))]
        per[f"{key[0]}/{key[1]}"] = [len(items), len(top)]
        elite += top
    elite = sorted(elite, key=lambda s: -diversity(s))[:max_keep]
    return {"skeletons": sorted(elite, key=lambda s: s["skeleton_id"]),
            "stats": {"min_score": min_score, "chunks_scanned": scanned, "reclassified_from_findings": moved,
                      "passed_filters_and_kept_by_group": per, "kept": len(elite)}}


# ── 4. use: pick, scale, describe ─────────────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _load(path: str) -> tuple:
    try:
        return tuple(json.loads(Path(path).read_text(encoding="utf-8"))["skeletons"])
    except (OSError, ValueError, KeyError):
        return ()


def load(path=None) -> list:
    """data/human_skeletons.json (or HARNESS_SKELETONS=<path>); HARNESS_SKELETONS=off or a missing file gives []."""
    env = config.get("HARNESS_SKELETONS")
    if env.lower() == "off":
        return []
    return list(_load(str(path or env or PATH)))


def pick(skeletons: list, section_type: str, position: str, words_target: int, seed: str, index: int = 0) -> dict | None:
    """One skeleton for (type, position): the POOL nearest `words_target` in length, shuffled by `seed` (same for every section
    of a chapter), then taken at `index` (the section's place in its chapter), so sections of one chapter never share one."""
    cands = sorted((s for s in skeletons if s["section_type"] == section_type and s["position"] == position),
                   key=lambda s: (abs(s["total_words"] - words_target), s["skeleton_id"]))[:POOL]
    if len(cands) < 3:
        return None
    return random.Random(seed).sample(cands, len(cands))[index % len(cands)]


def scale(sk: dict, words_target: int) -> list[dict]:
    """Paragraph specs sized so they sum to about `words_target`; sentence lengths keep their ratios inside each paragraph;
    opener, citation positions and closer are carried over unchanged."""
    f = words_target / max(1, sk["total_words"])
    out = []
    for p in sk["paragraphs"]:
        wc = max(30, int(round(p["word_count"] * f, -1)))
        g = wc / max(1, sum(p["sentence_lengths"]))
        out.append({**p, "target_words": wc, "target_sentences": [max(4, round(n * g)) for n in p["sentence_lengths"]]})
    return out


_OPEN = {"author_narrative_cite": "a sentence that opens on a source (\"Smith (2010) found ...\")",
         "empirical_figure_or_metric": "a sentence that opens on a figure or measured quantity",
         "contrastive_pivot": "a sentence that opens on a contrast (However / Although / While ...)",
         "methodological_action": "a sentence that opens on what was done or chosen (\"I used ...\", \"The sample ...\")",
         "domain_noun_subject": "a sentence that opens on its subject noun (\"The ...\", \"In ...\")"}
_END = {"raw_data_or_cite_cutoff": "a concrete figure or a cited finding, stated and left there",
        "limitation_or_caveat": "a limit or caveat on what was just said",
        "cross_ref_or_transition": "a pointer to another section, table or the next step",
        "comparative_assertion": "a plain comparison or claim, with no verdict phrase"}


def cadence(spec: dict) -> str:
    return "[" + ", ".join(f"~{n}w" for n in spec["target_sentences"]) + "]"


def blueprint(specs: list[dict]) -> str:
    """The prompt block. Structure only: the skeleton carries no content, so the writer takes every fact from its sources."""
    lines = ["HUMAN DISCOURSE SKELETON for this passage (the shape of a real doctoral passage; it carries no content, so take "
             f"every fact from the notes and references). Write exactly {len(specs)} paragraphs, in this order:"]
    for i, s in enumerate(specs, 1):
        sents, cp = s["target_sentences"], s["cite_positions"]
        short = [j for j, n in enumerate(sents, 1) if j > 1 and n <= 0.5 * sents[j - 2] and n <= 14]
        note = (f" Sentence {short[0]} is a blunt {sents[short[0] - 1]}-word statement right after a long one." if short else "")
        cite = ("no citation in this paragraph" if cp["mode"] == "none" else
                f"citations in sentence{'s' if len(cp['sentences']) > 1 else ''} {', '.join(map(str, cp['sentences']))}, "
                f"{'narrative (\"Author (year)\")' if cp['mode'] == 'narrative' else 'parenthetical (\"(Author, year)\")'}; "
                "cite only works listed in REFERENCES")
        lines.append(f"Paragraph {i}: about {s['target_words']} words in {len(sents)} sentence{"s" if len(sents) != 1 else ""} with target lengths "
                     f"{cadence(s)}.{note} Open with {_OPEN[s['opener_type']]}. {cite[0].upper() + cite[1:]}. "
                     f"End on {_END[s['closer_type']]}.")
    lines.append("Keep each sentence length near its target and do not even them out: the contrast between long and short "
                 "sentences is part of the plan. The heading line is not counted.")
    return "\n\n" + "\n".join(lines)


# ── 5. adherence ──────────────────────────────────────────────────────────────────────────────────────────────────────
FLAT_STD = 6.0


def sent_std(p: str) -> float | None:
    lens = [len(s.split()) for s in split_sentences(p)]
    return statistics.pstdev(lens) if len(lens) >= 3 else None


def adherence(text: str, specs: list[dict]) -> dict:
    """{paragraphs: [got, want], flat: [(body_index, std, cadence)]}. A paragraph is flat when the plan gave it 3+ sentences with
    real spread yet its sentences came out within FLAT_STD words of each other. body_index counts the non-heading paragraphs."""
    paras = [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    body = [(i, p) for i, p in enumerate(paras) if not p.lstrip().startswith("#")]
    flat = []
    for (idx, p), spec in zip(body, specs):
        planned = statistics.pstdev(spec["target_sentences"]) if len(spec["target_sentences"]) >= 3 else 0.0
        std = sent_std(p)
        if std is not None and std < FLAT_STD and planned >= FLAT_STD:
            flat.append((idx, round(std, 1), cadence(spec)))
    return {"paragraphs": [len(body), len(specs)], "flat": flat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build data/human_skeletons.json from data/corpus.db")
    ap.add_argument("--min-score", type=float, default=None, help="thesis score floor (0-100); default 0")
    ap.add_argument("--out", default=str(PATH))
    a = ap.parse_args(argv)
    con = store.connect(config.db_path())
    res = build(con, a.min_score)
    con.close()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(res["stats"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
