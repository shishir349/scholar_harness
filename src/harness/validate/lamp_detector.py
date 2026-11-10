"""Where a finished section still reads as machine prose, found at the level of one paragraph or one sentence.

Section-wide sentence-length variation lets a paragraph of four 24-28 word sentences through; a detector that reads each
paragraph on its own does not. `detect_ai_spans` returns the spans text/lamp_rewrite.py edits (LAMP, span-level rewriting).
Pure code: no model calls."""
import re
import statistics

from ..text.cleanup import CLOSER
from ..text.sentences import split_sentences

FLAT_STD = 6.5     # intra-paragraph sentence-length std below this (3+ sentences) is flat
NEAR = 5           # three sentences in a row each within this many words of the one before are monotone
CITE = re.compile(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)")
FIGURE = re.compile(r"(?<![\w.])(?!(?:19|20)\d\d[a-z]?\b)\d")
_BLOCK = re.compile(r"\[(?:FIGURE|TABLE)_START\]|^\s*[|!<]", re.M)

# the paragraph-level and sentence-level kinds, most urgent first
PRIORITY = ("haiku_verify_patch", "flat_paragraph", "rhythm_monotony", "epilogue_closer", "symmetric_hedge", "participial_tail",
            "vague_attribution", "nominalization_stack")

_CLOSE = re.compile(r"^(?:This (?:underscores|highlights|reflects|demonstrates|illustrates|points to|shows|reveals)\b|"
                    r"(?:Consequently|Thus|Therefore|Hence|As such|In this way|In turn),?\s)", re.I)
_HEDGE = re.compile(r"^(?:While|Although|Though|Whereas)\b[^,]{4,},|,\s+(?:while|although|whereas)\s+[^,]{4,}[.!?]$", re.I)
_VAGUE = re.compile(r"\b(?:has|have) (?:been )?(?:extensively|widely|well|increasingly|long|often) (?:studied|documented|examined|discussed|"
                    r"researched|debated|noted)\b|\b(?:researchers|scholars|authors|studies) have (?:noted|observed|argued|shown|suggested|"
                    r"found|pointed out)\b|\bit has been (?:argued|noted|suggested|shown|observed)\b|\bthe literature (?:suggests|indicates|shows)\b", re.I)
_NOT_PARTICIPLE = frozenset("including according following during concerning regarding being nothing something anything "
                            "building training testing engineering ongoing existing".split())
_NOMINAL = re.compile(r"\b[a-z]{3,}(?:tion|tions|ment|ments|ance|ances|ence|ences|ity|ities)\b", re.I)
_NOMINAL_STOP = frozenset("question questions section sections position positions".split())

INSTRUCTIONS = {
    "haiku_verify_patch": "A cheap model patched this sentence so it stays within its source. Keep its claim, numbers and citation exactly; "
                          "reword it and change its rhythm so it reads as the researcher's own sentence.",
    "flat_paragraph": "The sentences are too even in length ({lengths}). Rework into a bimodal paragraph: join neighbours into one dense "
                      "32-45 word multi-clause sentence and make another a blunt 5-9 word statement; no three sentences within 5 words "
                      "of each other. Keep every fact, number and citation.",
    "rhythm_monotony": "Three sentences in a row open and run alike. Change how two of them open and how long they are (one blunt, "
                       "one long and dense). Keep every fact, number and citation.",
    "epilogue_closer": "This closing sentence only sums up. Reply with an empty string to delete it.",
    "symmetric_hedge": "Replace the balanced While/Although clause with a plain statement of the one point the evidence supports, "
                       "and give the limit as its own short sentence.",
    "participial_tail": "Cut the trailing -ing clause, or make it a short sentence of its own. Keep every fact and citation.",
    "vague_attribution": "Drop the 'has been studied' / 'researchers have noted' framing and state the claim directly. Add no citation.",
    "nominalization_stack": "Too many abstract nouns: turn some back into verbs and name who does what. Keep every fact and citation.",
}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def flat_note(lengths: list[int]) -> str | None:
    """Why a paragraph with these sentence lengths is flat, or None."""
    if len(lengths) < 3:
        return None
    if statistics.pstdev(lengths) < FLAT_STD:
        return f"std {statistics.pstdev(lengths):.1f}"
    for j in range(len(lengths) - 2):
        a, b, c = lengths[j:j + 3]
        if abs(a - b) <= NEAR and abs(b - c) <= NEAR:
            return f"sentences {j + 1}-{j + 3} run {a}, {b}, {c}"
    return None


def paragraph_is_flat(text: str) -> bool:
    return flat_note([len(s.split()) for s in split_sentences(text)]) is not None


def _monotone(sents: list[str]) -> bool:
    """Three declarative sentences in a row that open on the same word (the same subject-verb-object frame)."""
    for j in range(len(sents) - 2):
        three = sents[j:j + 3]
        if all(len(s.split()) >= 6 and not s.rstrip().endswith("?") for s in three) and len({s.split()[0].lower() for s in three}) == 1:
            return True
    return False


def sentence_types(sent: str, *, last: bool, n_sents: int, patched: set) -> list[str]:
    out = []
    if _norm(sent) in patched:
        out.append("haiku_verify_patch")
    if last and n_sents >= 3 and (CLOSER.search(sent) or _CLOSE.search(sent)) and not FIGURE.search(sent) and not CITE.search(sent):
        out.append("epilogue_closer")
    if _HEDGE.search(sent):
        out.append("symmetric_hedge")
    tail = sent.rstrip(" .!?").rsplit(", ", 1)
    if len(tail) == 2:
        first = tail[1].split()[:2]
        word = first[-1] if first and first[0].lower() in ("thereby", "thus", "hence") and len(first) > 1 else (first[0] if first else "")
        if re.fullmatch(r"[A-Za-z]{4,}ing", word) and word.lower() not in _NOT_PARTICIPLE:
            out.append("participial_tail")
    if _VAGUE.search(sent) and not CITE.search(sent):
        out.append("vague_attribution")
    if len([w for w in _NOMINAL.findall(sent) if w.lower() not in _NOMINAL_STOP]) >= 4:
        out.append("nominalization_stack")
    return out


def detect_ai_spans(text: str, patched=()) -> list[dict]:
    """Spans to rewrite, most urgent first: [{kind: 'paragraph'|'sentence', text, types, para, instruction}].
    `patched` holds sentences the claim verifier rewrote (always queued). A flagged paragraph is one span; a paragraph with
    only sentence-level flags yields one span per flagged sentence. Headings, figure/table blocks and tables are skipped."""
    patched_n = {_norm(s) for s in patched or ()}
    spans = []
    for i, para in enumerate(q.strip() for q in re.split(r"\n\s*\n", text or "") if q.strip()):
        if para.startswith("#") or _BLOCK.search(para):
            continue
        sents = split_sentences(para)
        lengths = [len(s.split()) for s in sents]
        per = [sentence_types(s, last=(k == len(sents) - 1), n_sents=len(sents), patched=patched_n) for k, s in enumerate(sents)]
        note = flat_note(lengths)
        types = (["flat_paragraph"] if note else []) + (["rhythm_monotony"] if _monotone(sents) else [])
        if types:
            more = sorted({t for ts in per for t in ts}, key=PRIORITY.index)
            alltypes = types + [t for t in more if t not in types]
            asks = [INSTRUCTIONS[t].format(lengths=lengths) for t in alltypes[:2] if t != "epilogue_closer"]
            if "epilogue_closer" in alltypes:
                asks.append("Also cut the closing sentence that only sums the paragraph up, or end on its last concrete number or limitation.")
            spans.append({"kind": "paragraph", "text": para, "types": alltypes, "para": i, "instruction": " ".join(asks)})
            continue
        for s, ts in zip(sents, per):
            if ts:
                ts = sorted(ts, key=PRIORITY.index)
                spans.append({"kind": "sentence", "text": s, "types": ts, "para": i, "instruction": INSTRUCTIONS[ts[0]]})
    spans.sort(key=lambda sp: (min(PRIORITY.index(t) for t in sp["types"]), sp["para"]))
    return spans
