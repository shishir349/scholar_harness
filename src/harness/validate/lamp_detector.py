"""Where a finished section still reads as machine prose, found at the level of one paragraph or one sentence.

Section-wide sentence-length variation lets a paragraph of four 24-28 word sentences through; a detector that reads each
paragraph on its own does not. `detect_ai_spans` returns the spans text/lamp_rewrite.py edits (LAMP, span-level rewriting).
Pure code: no model calls."""
import re
import statistics

from ..text.citations import NARRATIVE_RE
from ..text.cleanup import CLOSER
from ..text.sentences import split_sentences
from .. import limits
from .style import FLAT_STD, FRAGMENT_WORDS, MAX_SENT_WORDS, SHORT_SENT_WORDS, STACK_WORDS

NEAR = 5           # three sentences in a row each within this many words of the one before are monotone
NARRATIVE_CAP = limits.NARR_SECTION  # "Author (year) found ..." sentences a section may hold; later ones are rewritten with the citation in parentheses
CITE = re.compile(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)")
FIGURE = re.compile(r"(?<![\w.])(?!(?:19|20)\d\d[a-z]?\b)\d")
_BLOCK = re.compile(r"\[(?:FIGURE|TABLE)_START\]|^\s*[|!<]", re.M)

# the paragraph-level and sentence-level kinds, most urgent first
PRIORITY = ("haiku_verify_patch", "flat_paragraph", "rhythm_monotony", "long_sentence", "narrative_surplus", "stock_idiom", "epilogue_closer",
            "symmetric_hedge", "participial_tail", "vague_attribution", "nominalization_stack", "claude_style")

_CLOSE = re.compile(r"^(?:This (?:underscores|highlights|reflects|demonstrates|illustrates|points to|shows|reveals)\b|"
                    r"(?:Consequently|Thus|Therefore|Hence|As such|In this way|In turn),?\s)", re.I)
_HEDGE = re.compile(r"^(?:While|Although|Though|Whereas)\b[^,]{4,},|,\s+(?:while|although|whereas)\s+[^,]{4,}[.!?]$", re.I)
_VAGUE = re.compile(r"\b(?:has|have) (?:been )?(?:extensively|widely|well|increasingly|long|often) (?:studied|documented|examined|discussed|"
                    r"researched|debated|noted)\b|\b(?:researchers|scholars|authors|studies) have (?:noted|observed|argued|shown|suggested|"
                    r"found|pointed out)\b|\bit has been (?:argued|noted|suggested|shown|observed)\b|\bthe literature (?:suggests|indicates|shows)\b", re.I)
_NOT_PARTICIPLE = frozenset("including according following during concerning regarding being nothing something anything "
                            "building training testing engineering ongoing existing".split())
# Phrases a draft of section 3.1 leaned on: high-probability academic filler that a detector reads as machine prose.
_IDIOM = re.compile(r"\b(?:carr(?:y|ies|ied) (?:\w+ )?(?:evidential|analytical|explanatory|considerable) weight|"
                    r"sits? at the intersection of|not an? ?(?:neutral|value-free|innocent) (?:act|choice|stance)|resists? reduction to|"
                    r"not a failure of (?:the )?\w+|in the sense that|a nuanced understanding|sheds? light on|at the heart of|"
                    r"offers? a (?:lens|window) (?:on|onto|for)|speaks? to the (?:broader|wider)|a complex (?:interplay|web) of)\b", re.I)
_NOMINAL = re.compile(r"\b[a-z]{3,}(?:tion|tions|ment|ments|ance|ances|ence|ences|ity|ities)\b", re.I)
_NOMINAL_STOP = frozenset("question questions section sections position positions".split())

INSTRUCTIONS = {
    "haiku_verify_patch": "A cheap model patched this sentence so it stays within its source. Keep its claim, numbers and citation exactly; "
                          "reword it and change its rhythm so it reads as the researcher's own sentence.",
    "flat_paragraph": "The sentence lengths ({lengths}) break the pattern: no sentence may be longer than " + str(MAX_SENT_WORDS) +
                      " words, one sentence must be a blunt 5-9 word statement inside the paragraph, and no three sentences may sit within 5 "
                      "words of each other. Rework into a bimodal paragraph (dense multi-clause sentences beside the blunt one); keep it one "
                      "paragraph. Keep every fact, number and citation.",
    "rhythm_monotony": "Three sentences in a row open and run alike. Change how two of them open and how long they are (one blunt, "
                       "one long and dense). Keep every fact, number and citation.",
    "long_sentence": "This sentence is longer than " + str(MAX_SENT_WORDS) + " words. Split it into two sentences of under 40 words each "
                     "(the citation may stay in one half or appear in both). Keep every fact, number and name.",
    "narrative_surplus": "This sentence names an author as its subject, like a book report. State the claim yourself and put the "
                         "citation in parentheses at its end (Author, year); better, set it against another source already cited nearby. "
                         "Keep every citation and number.",
    "stock_idiom": "Predictable academic phrasing ({found}). Say the same thing in direct, un-hedged technical language, with the "
                   "concrete subject and verb. Keep every fact, number and citation.",
    "epilogue_closer": "This closing sentence only sums up. Reply with an empty string to delete it.",
    "symmetric_hedge": "Replace the balanced While/Although clause with a plain statement of the one point the evidence supports, "
                       "and give the limit as its own short sentence.",
    "participial_tail": "Cut the trailing -ing clause, or make it a short sentence of its own. Keep every fact and citation.",
    "vague_attribution": "Drop the 'has been studied' / 'researchers have noted' framing and state the claim directly. Add no citation.",
    "claude_style": "A local detector trained on Claude's own output flags this sentence. {found} "
                    "Say the same thing another way. Keep every fact, number and citation.",
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
    if max(lengths) > MAX_SENT_WORDS:
        return f"a {max(lengths)}-word sentence (limit {MAX_SENT_WORDS})"
    if sum(1 for x in lengths if x < FRAGMENT_WORDS) >= 2 or any(all(x < STACK_WORDS for x in lengths[j:j + 3]) for j in range(len(lengths) - 2)):
        return "stacked fragments"
    if min(lengths) > SHORT_SENT_WORDS:
        return f"no sentence of {SHORT_SENT_WORDS} words or fewer"
    return None


def paragraph_note(lengths: list[int]) -> str | None:
    """flat_note without the long-sentence test: a long sentence is a span of its own, not a reason to rewrite the paragraph."""
    return flat_note([x for x in lengths if x <= MAX_SENT_WORDS])


def violations(lengths: list[int]) -> int:
    """How many rhythm rules these sentence lengths break (0 = clean): flat spread, a monotone run, sentences over the limit,
    stacked fragments, no short sentence. A rewrite is accepted when it breaks fewer than the original did."""
    if len(lengths) < 3:
        return 0
    n = int(statistics.pstdev(lengths) < FLAT_STD) + sum(1 for x in lengths if x > MAX_SENT_WORDS)
    n += int(any(abs(a - b) <= NEAR and abs(b - c) <= NEAR for a, b, c in zip(lengths, lengths[1:], lengths[2:])))
    n += int(sum(1 for x in lengths if x < FRAGMENT_WORDS) >= 2 or any(all(x < STACK_WORDS for x in lengths[j:j + 3]) for j in range(len(lengths) - 2)))
    return n + int(min(lengths) > SHORT_SENT_WORDS)


def paragraph_is_flat(text: str) -> bool:
    return flat_note([len(s.split()) for s in split_sentences(text)]) is not None


def paragraph_violations(text: str) -> int:
    return violations([len(s.split()) for s in split_sentences(text)])


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
    if len(sent.split()) > MAX_SENT_WORDS:
        out.append("long_sentence")
    if _IDIOM.search(sent):
        out.append("stock_idiom")
    if _VAGUE.search(sent) and not CITE.search(sent):
        out.append("vague_attribution")
    if len([w for w in _NOMINAL.findall(sent) if w.lower() not in _NOMINAL_STOP]) >= 4:
        out.append("nominalization_stack")
    return out


def narrative_count(sent: str) -> int:
    return sum(1 for m in NARRATIVE_RE.finditer(sent) if sent[max(0, m.start() - 1):m.start()] != "(")


_DIRECTIVE = {      # structural detector feature -> what the rewriter must do (never quoted as a phrase)
    "avg_word_len": "Rewrite using concrete, everyday academic vocabulary; avoid heavy multi-syllabic nominalizations.",
    "nominal_ratio": "Rewrite using concrete, everyday academic vocabulary; avoid heavy multi-syllabic nominalizations.",
    "commas": "Split this compound sentence at the midpoint into two separate sentences.",
    "comma_per_word": "Split this compound sentence at the midpoint into two separate sentences.",
    "colon_semicolon": "Remove the colon/semicolon and structure as two independent sentences.",
    "subordinators": "Cut the which/where/while clauses: make each one its own short sentence.",
    "ing_clauses": "Cut the trailing -ing clause, or make it a short sentence of its own.",
    "passive_ratio": "Use the active voice: name who did what.",
    "words": "Shorten the sentence.",
}
_STRUCTURAL = _DIRECTIVE | {"nums": "", "cites": ""}     # recognised, no directive: never ask to drop a figure or citation


def claude_ask(culprits: list[str]) -> str:
    """Exact phrases to delete (quoted) plus one explicit directive per structural feature; each directive once."""
    feat = lambda c: c.split(">=")[0]
    phrases = [f'"{c}"' for c in culprits if feat(c) not in _STRUCTURAL]
    rules = list(dict.fromkeys(_DIRECTIVE[feat(c)] for c in culprits if feat(c) in _DIRECTIVE))
    head = ["Delete these exact phrases (none may remain, in any form): " + "; ".join(phrases) + "."] if phrases else []
    return " ".join(head + rules) or "Rewrite without its stock phrasing."


def _ask(t: str, sent: str = "", phrase: str = "") -> str:
    """The edit instruction for one span type; the idiom instruction names the phrases found, the detector one the exact Claude phrases."""
    ins = INSTRUCTIONS[t]
    if t == "claude_style":
        ins = ins.replace("{found}", phrase)
    if t == "stock_idiom":
        ins = ins.replace("{found}", "; ".join(dict.fromkeys(m.group(0) for m in _IDIOM.finditer(sent)))[:120])
    return ins


def detect_ai_spans(text: str, patched=(), claude=None) -> list[dict]:
    """Spans to rewrite, most urgent first: [{kind: 'paragraph'|'sentence', text, types, para, instruction}].
    `patched` holds sentences the claim verifier rewrote (always queued). A flagged paragraph is one span; a paragraph with
    only sentence-level flags yields one span per flagged sentence. Headings, figure/table blocks and tables are skipped.
    `claude` is validate/detector.score_text(text) (None = the detector is off): a flagged sentence of 8+ words gets the type "claude_style",
    and the span carries `culprits` (the exact phrases) and `ai_prob`."""
    patched_n = {_norm(s) for s in patched or ()}
    cl = {_norm(r["text"]): r for r in claude or [] if r["flagged"] and len(r["text"].split()) >= 8}
    spans, nar = [], 0
    for i, para in enumerate(q.strip() for q in re.split(r"\n\s*\n", text or "") if q.strip()):
        if para.startswith("#") or _BLOCK.search(para):
            continue
        sents = split_sentences(para)
        lengths = [len(s.split()) for s in sents]
        per = [sentence_types(s, last=(k == len(sents) - 1), n_sents=len(sents), patched=patched_n) for k, s in enumerate(sents)]
        for k, s in enumerate(sents):
            if _norm(s) in cl:
                per[k].append("claude_style")
            nar += narrative_count(s)
            if narrative_count(s) and nar > NARRATIVE_CAP:
                per[k].append("narrative_surplus")
        note = paragraph_note(lengths)
        types = (["flat_paragraph"] if note else []) + (["rhythm_monotony"] if _monotone(sents) else [])
        if types:
            more = sorted({t for ts in per for t in ts}, key=PRIORITY.index)
            alltypes = types + [t for t in more if t not in types]
            asks = [INSTRUCTIONS[t].format(lengths=lengths) if t == "flat_paragraph" else _ask(t, para)
                    for t in alltypes[:2] if t != "epilogue_closer"]
            if "epilogue_closer" in alltypes:
                asks.append("Also cut the closing sentence that only sums the paragraph up, or end on its last concrete number or limitation.")
            hit = [cl[_norm(x)] for x in sents if _norm(x) in cl]
            span = {"kind": "paragraph", "text": para, "types": alltypes, "para": i}
            if hit:
                culprits = list(dict.fromkeys(c for r in hit for c in r["culprit_phrases"]))[:5]
                asks.append(claude_ask(culprits))
                span.update(culprits=culprits, ai_prob=max(r["ai_prob"] for r in hit))
            spans.append({**span, "instruction": " ".join(asks)})
            continue
        for s, ts in zip(sents, per):
            if ts:
                ts = sorted(ts, key=PRIORITY.index)
                r = cl.get(_norm(s))
                culprits = r["culprit_phrases"] if r else []
                ask = _ask(ts[0], s, claude_ask(culprits))
                if r and ts[0] != "claude_style":
                    ask += " " + claude_ask(culprits)
                span = {"kind": "sentence", "text": s, "types": ts, "para": i, "instruction": ask}
                if r:
                    span.update(culprits=culprits, ai_prob=r["ai_prob"])
                spans.append(span)
    spans.sort(key=lambda sp: (min(PRIORITY.index(t) for t in sp["types"]), -sp.get("ai_prob", 0.0), sp["para"]))
    return spans
