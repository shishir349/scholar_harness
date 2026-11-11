"""Free repairs and repeat detectors for a drafted section (no model calls).
`heal` fixes chunk seams, `drop_repeats` deletes pasted source sentences and restated sentences, `repeat_flags` lists what only a
rewrite can fix. Paragraph indexes match graph.py: blank-line split, empty parts removed."""
import re

from .. import limits
from .citations import NARRATIVE_RE, first_author
from .sentences import split_sentences

POSITIONS = ("open", "mid", "close")
_STOP = frozenset("the a an of in on to and or for with by is are was were that this these those as at from it its be been which than but not".split())
_PARTICLES = frozenset("van von de der den da di du la le el al bin ibn ten".split())
_END = '.!?)"”’'
_GLUE = re.compile(r"(?<=[a-z][.!?])(?=[A-Z][a-z])")                 # "remains.The" -> "remains. The"
STANCE = re.compile(r"\b(?:I (?:argue|contend|would argue|maintain|take the view)|this study takes the position)\b", re.I)
CLOSER = re.compile(r"^(?:What (?:the|these|this) \w+(?: \w+){0,3} (?:show|suggest|mean|share)|The (?:answer|upshot)\b|Taken together\b|"
                    r"This (?:means|suggests|matters|indicates that)\b|In (?:short|sum)\b|Overall\b|Ultimately\b|The (?:implication|lesson|takeaway)\b|"
                    r"These findings collectively\b|Such dynamics\b)", re.I)
REPORT = re.compile(r"^[A-Z][\w'’-]+(?: et al\.)?(?: (?:and|&) [A-Z][\w'’-]+)? \((?:19|20)\d\d[a-z]?\) "
                    r"(?:trace|note|offer|reinforce|situate|frame|argue|show|suggest|highlight|emphasi[sz]e|review|examine|explore|report|describe)s?\b")
# a figure is a number that is not a year and not inside a word ("94%", "12 mm", "120 samples"; not "2019", not "ADXL355")
FIGURE = re.compile(r"(?<![\w.])(?!(?:19|20)\d\d[a-z]?\b)\d")
SAME_IDEA = 0.45        # content-word overlap above which two sentences say the same thing
NL2 = "\n\n"


def paras(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _content(s: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in _STOP)


SECTION_DUP = limits.DUP_JACCARD      # the same test across the whole assembled section (open + mid + close)
MIN_SENTS, MIN_WORDS = limits.MIN_SENTS, limits.MIN_WORDS     # a body paragraph with fewer sentences or words is a staccato filler block
_BLOCKISH = re.compile(r"^\s*(?:#|\||!\[|[-*•]\s|\d+[.)]\s|\[(?:FIGURE|TABLE)_)")


def same_idea(a: frozenset, b: frozenset, thr: float = SAME_IDEA) -> bool:
    """Two sentences with at least five distinct content words and a large overlap."""
    return len(a) >= 5 and len(b) >= 5 and len(a & b) / len(a | b) >= thr


def _frame(s: str):
    """(first author, year, the word after the citation) of a sentence's narrative citation: 'Brown and Dueñas (2019) noted'."""
    m = NARRATIVE_RE.search(s)
    if not m:
        return None
    nxt = s[m.end():].split()
    return (first_author(m.group(1)).lower(), m.group(2), nxt[0].lower().strip(",;") if nxt else "")


def _works(s: str) -> set:
    """{(first author, year)} the sentence summarises narratively ("Smith (2010) found ..."), not cites in parentheses."""
    return {(first_author(m.group(1)).lower(), m.group(2)) for m in NARRATIVE_RE.finditer(s) if s[max(0, m.start() - 1):m.start()] != "("}


def _short_para(p: str, min_sents: int = MIN_SENTS, min_words: int = MIN_WORDS) -> bool:
    return not _BLOCKISH.match(p) and (len(split_sentences(p)) < min_sents or len(p.split()) < min_words)


MAX_MERGED_WORDS = 200      # a merge never builds a paragraph longer than this; a lone short paragraph is left rather than bloat one


def merge_short_paragraphs(chunks: dict, min_sents: int = MIN_SENTS, min_words: int = MIN_WORDS) -> dict:
    """Join body paragraphs under MIN_SENTS sentences or MIN_WORDS words so no 1-2 sentence filler block survives: first a run of
    short ones together, then each one still short into the neighbour it fits with (the shorter, never past MAX_MERGED_WORDS).
    Headings, tables, lists and figure blocks are left alone."""
    out = dict(chunks)
    for pos in POSITIONS:
        before = [x.strip() for x in paras(chunks[pos])]
        res: list[str] = []
        for p in before:                      # 1. runs of short paragraphs become one
            if res and _short_para(p, min_sents, min_words) and _short_para(res[-1], min_sents, min_words) and len((res[-1] + " " + p).split()) <= MAX_MERGED_WORDS:
                res[-1] += " " + p
            else:
                res.append(p)
        i = 0
        while i < len(res):                   # 2. a short paragraph still left joins the neighbour it fits with
            if not _short_para(res[i], min_sents, min_words):
                i += 1
                continue
            w = len(res[i].split())
            prev = res[i - 1] if i > 0 and not _BLOCKISH.match(res[i - 1]) else None
            nxt = res[i + 1] if i + 1 < len(res) and not _BLOCKISH.match(res[i + 1]) else None
            fits = [(len(n.split()), k) for k, n in (("prev", prev), ("next", nxt)) if n is not None and len(n.split()) + w <= MAX_MERGED_WORDS]
            if not fits:
                i += 1
                continue
            if min(fits)[1] == "prev":
                res[i - 1] += " " + res[i]
                del res[i]
            else:
                res[i + 1] = res[i] + " " + res[i + 1]
                del res[i]
        if res != before:
            out[pos] = _rebuild(chunks[pos], res)
    return out


merge_short = merge_short_paragraphs


_SPLIT_AT = re.compile(r"(?:; |, (?:and|but|yet|so|while|whereas) )")
_FINITE = re.compile(r"\b(?:is|are|was|were|be|been|has|have|had|do|does|did|can|could|may|might|will|would|must|should|means|requires?|"
                     r"depends?|remains?|reflects?|makes?|allows?|produces?|shows?|holds?|creates?|carries|rests?|sits?|needs?)\b", re.I)


def _split_sentence(s: str, limit: int) -> str:
    """One sentence over `limit` words cut at the clause boundary nearest its middle: '; ' or ', and/but/yet/so/while/whereas '
    outside any bracket, both halves at least 9 words, the second with a finite verb (so a list's last item is never cut off).
    The conjunction is dropped. Returns the sentence unchanged when no such boundary exists."""
    if len(s.split()) <= limit:
        return s
    depth, cuts = 0, []
    for m in re.finditer(r"[()]|; |, (?:and|but|yet|so|while|whereas) ", s):
        if m.group(0) == "(":
            depth += 1
        elif m.group(0) == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            head, tail = s[:m.start()], s[m.end():]
            if len(head.split()) >= 9 and len(tail.split()) >= 9 and _FINITE.search(tail) and head.count("(") == head.count(")"):
                cuts.append((abs(len(head.split()) - len(tail.split())), m, head, tail))
    if not cuts:
        return s
    _, m, head, tail = min(cuts, key=lambda c: c[0])
    conj = m.group(0).strip(" ,;")
    tail = tail[:1].upper() + tail[1:] if conj in ("and", "but", "yet", "so", "") else tail[:1].upper() + tail[1:]
    return _split_sentence(head.rstrip(" ,;") + ". ", limit).rstrip() + " " + _split_sentence(tail, limit)


def split_long(chunks: dict, limit: int = 42) -> dict:
    """Free: cut every sentence over `limit` words at a clause boundary (see _split_sentence). Citations, numbers and quotes are
    untouched (only a conjunction or semicolon is replaced by a full stop). Headings and blocks are skipped."""
    out = dict(chunks)
    for pos in POSITIONS:
        ps = paras(chunks[pos])
        changed = False
        for i, p in enumerate(ps):
            if _BLOCKISH.match(p):
                continue
            sents = split_sentences(p)
            new = [_split_sentence(x, limit) for x in sents]
            if new != sents:
                ps[i], changed = " ".join(new), True
        if changed:
            out[pos] = _rebuild(chunks[pos], ps)
    return out


def _grams(s: str) -> set:
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}


def copied_from(sentence: str, source_grams: set) -> bool:
    """True when half or more of the sentence's 5-word runs appear in a source abstract: pasted, not written."""
    g = _grams(sentence)
    return len(sentence.split()) >= 8 and len(g) > 0 and len(g & source_grams) / len(g) >= 0.5


def _drop_fragment(p: str) -> str:
    """A paragraph that starts mid-sentence (a cut-off continuation): drop up to the first full stop, '' if none is left."""
    parts = split_sentences(p)
    return " ".join(parts[1:]) if len(parts) > 1 else ""


def _rebuild(old: str, parts: list[str]) -> str:
    return NL2.join(parts) + old[len(old.rstrip()):]


def heal(chunks: dict) -> dict:
    """Seams the model leaves between chunks: "remains.The", a lowercase paragraph that continues a sentence the previous
    chunk already finished, and a last sentence cut off by the token limit. Short stubs and name particles are left alone."""
    out = {}
    for k, pos in enumerate(POSITIONS):
        before = [p.strip() for p in paras(chunks[pos])]
        tail = out[POSITIONS[k - 1]].rstrip() if k else ""
        prev_done = not tail or tail[-1] in _END
        res = []
        for i, p in enumerate(_GLUE.sub(" ", x) for x in before):
            words = p.split()
            if (not p.startswith("#") and p[:1].islower() and words[0].lower() not in _PARTICLES and (i > 0 or prev_done)
                    and 6 <= len(split_sentences(p)[0].split()) <= 40):      # a short orphan, not a paragraph that merely starts lowercase
                p = _drop_fragment(p)
            if p:
                res.append(p)
        if res and not res[-1].startswith("#") and res[-1][-1] not in _END and len(res[-1].split()) >= 25:
            nxt = chunks[POSITIONS[k + 1]].lstrip() if k + 1 < len(POSITIONS) else ""
            cut = max(res[-1].rfind(c + " ") for c in ".!?")
            if cut > 0 and not nxt[:1].islower():       # cut mid-sentence and the next chunk does not finish it
                res[-1] = res[-1][:cut + 1]
        out[pos] = chunks[pos] if res == before else _rebuild(chunks[pos], res)
    return out


def drop_repeats(chunks: dict, abstracts=()) -> tuple[dict, list]:
    """Delete, for free, a sentence that (a) is pasted from a source abstract or (b) restates a sentence in the same or the
    previous paragraph, as long as the paragraph keeps two sentences. Returns (chunks, [(pos, idx)] of pasted sentences that
    could not be deleted, for a rewrite)."""
    src = set().union(*(_grams(a) for a in abstracts)) if abstracts else set()
    out, left, seen, frames, works = dict(chunks), [], [], set(), set()    # kept so far in the WHOLE section; works: narrative (author, year)
    for pos in POSITIONS:
        ps = paras(chunks[pos])
        changed = False
        for i, p in enumerate(ps):
            if p.startswith("#"):
                continue
            sents = split_sentences(p)
            keep, here = [], []
            for j, s in enumerate(sents):
                c, fr = _content(s), _frame(s)
                long = len(s.split()) >= 8
                ay = _works(s)
                dup = long and (any(same_idea(c, o, SECTION_DUP) for o in here + seen) or (fr is not None and fr in frames) or bool(ay & works))
                paste = bool(src) and copied_from(s, src)
                if (dup or paste) and len(keep) + len(sents) - j - 1 >= 2 and not (paste and FIGURE.search(s)):
                    continue                    # a pasted sentence with a figure is reworded by the patch, never deleted: the number stays
                if paste:
                    left.append((pos, i))
                keep.append(s)
                here.append(c)
                if fr is not None:
                    frames.add(fr)
                works |= ay
            if len(keep) != len(sents):
                ps[i], changed = " ".join(keep), True
            seen += here
        if changed:
            out[pos] = _rebuild(chunks[pos], ps)
    return out, sorted(set(left))


def repeat_flags(chunks: dict) -> dict:
    """What only a rewrite can fix, as {kind: [(pos, idx, note)]}: a sentence that restates one from an earlier paragraph or
    opens like one, 'I argue' announcements after the first, paragraphs ending on a verdict line, and 'What ...' openers
    (real theses: 2% of paragraphs, AI drafts: 11-14%)."""
    out = {"repeat": [], "stance": [], "closer": [], "whatopen": []}
    seen, stems, stance_n, what, frames, works = [], {}, 0, [], {}, {}
    for pos in POSITIONS:
        for i, p in enumerate(paras(chunks[pos])):
            sents = split_sentences(p) if not p.startswith("#") else []
            if not sents:
                continue
            if p.startswith("What "):
                what.append((pos, i))
            if CLOSER.search(sents[-1]) and len(sents) >= 2:
                out["closer"].append((pos, i, sents[-1][:60]))
            if STANCE.search(p):
                stance_n += 1
                if stance_n > 1:
                    out["stance"].append((pos, i, ""))
            mine = []
            for s in sents:
                c, stem = _content(s), " ".join(s.lower().split()[:5])
                long = len(s.split()) >= 8
                hit = next((o for o, c0 in seen if same_idea(c, c0, SECTION_DUP)), None) if long else None
                fr = _frame(s)
                if hit is None and long and fr is not None:
                    hit = frames.get(fr)
                if hit is None and long:
                    hit = next((works[k] for k in _works(s) if k in works), None)
                if hit is None and long and len(set(stem.split())) >= 4:
                    hit = stems.get(stem)
                for k in _works(s):
                    works.setdefault(k, s)
                if fr is not None:
                    frames.setdefault(fr, s)
                if hit:
                    out["repeat"].append((pos, i, hit[:70]))
                    break
                mine.append((s, c))
                stems.setdefault(stem, s)
            seen += mine
    if len(what) >= 2:
        out["whatopen"] = [(pos, i, "") for pos, i in what]
    return {k: sorted(set(v)) for k, v in out.items()}


# ── free polish: wording rewrites that need no model ───────────────────────────────────────────────────────────────────
# Measured on saved drafts (real / harness): "I argue" lines 0.13 / 1.26 per draft, "Author (year) notes ..." lines 0.27 / 1.37,
# paragraphs opening "What ..." 2% / 14%, verdict-closer paragraphs 0.10 / 0.16, boosters 0.2 / 1.5 per 1000 words, "that" 10.7 / 24.9.
# Each rewrite is narrow, leaves the text alone when its pattern is not found, never removes a figure, and is safe to run twice.
STANCE_LEAD = re.compile(r"\b(?:I (?:would )?(?:argue|contend|maintain)(?: here| below)?|this (?:study|thesis|chapter) "
                         r"(?:takes the position|argues|contends|maintains)) that\s+", re.I)
WHAT_CLEFT = re.compile(r"^What\b[^.;:]{3,230}?\b(?:is|are|was|were)\s+that\s+(.+)$", re.S)
_WHO = r"(?P<who>[A-Z][\w'’-]+(?: et al\.)?(?: (?:and|&) [A-Z][\w'’-]+)?) \((?P<yr>(?:19|20)\d\d[a-z]?)\)"
REPORT_THAT = re.compile(rf"^{_WHO} (?:note|argue|show|suggest|highlight|emphasi[sz]e|report|find|observe|demonstrate|indicate|conclude|stress)s?"
                         r"(?:, [^,.;]{3,60},)? that (?P<rest>.+)$", re.S)
BOOSTER_WORDS = "clearly|obviously|certainly|definitely|undoubtedly|genuinely|fundamentally|deeply|actually|adequately|analytically"
_BOOST_LEAD = re.compile(rf"^(?:{BOOSTER_WORDS}),\s+", re.I)       # only 'Clearly, X': 'Clearly defined' is an adjective phrase
_BOOST_MID = re.compile(r"[ \t]+(?:obviously|certainly|definitely|undoubtedly|genuinely|actually|adequately|analytically)(?=[ \t,])", re.I)
_THAT_AFTER = re.compile(r"\b(found|finds?|shows?|showed|suggest(?:s|ed)?|indicat(?:e|es|ed)|not(?:e|es|ed)|argue[sd]?|"
                         r"observ(?:e|es|ed)|assum(?:e|es|ed)|believ(?:e|es|ed)|conclud(?:e|es|ed)|means?) that "
                         r"(?=(?:the|these|those|it|its|they|their|there|such|most|many|some|no|all|each|both|we|I|our|my)\b)")
EXAMPLE = re.compile(r"\b(?:for example|for instance|such as|e\.g\.)", re.I)

# Plain-word swaps: generic academic words and phrases with a shorter plain twin. Domain terms are never listed.
PLAIN = {"utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using", "utilization": "use", "demonstrate": "show",
         "demonstrates": "shows", "demonstrated": "showed", "numerous": "many", "commence": "begin", "facilitate": "help",
         "facilitates": "helps", "subsequently": "later", "additionally": "also", "furthermore": "also", "approximately": "about",
         "in order to": "to", "a number of": "several", "due to the fact that": "because", "in the event that": "if",
         "with regard to": "on", "with respect to": "on", "prior to": "before", "in addition to": "besides", "endeavour": "try",
         "ascertain": "find out", "obtain": "get", "obtained": "got", "sufficient": "enough", "assist": "help", "assists": "helps"}
_PLAIN = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, PLAIN), key=len, reverse=True)) + r")(?![\w-])", re.I)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _swap(p: str, old: str, new: str) -> str:
    """Replace sentence `old` in paragraph `p` with `new` ('' deletes it). split_sentences collapses whitespace, so a miss on
    the exact text retries tolerant of any whitespace. The paragraph is returned unchanged when the sentence is not found."""
    if old in p:
        out = p.replace(old, new, 1)
    else:
        m = re.search(r"\s+".join(map(re.escape, old.split())), p)
        if not m:
            return p
        out = p[:m.start()] + new + p[m.end():]
    return re.sub(r"[ \t]{2,}", " ", out).strip() if not new else out


def _each(chunks: dict, fn) -> tuple[dict, int]:
    """Apply fn(paragraph) -> (paragraph, n_changes) to every body paragraph of every chunk."""
    out, total = dict(chunks), 0
    for pos in POSITIONS:
        ps, changed = paras(chunks[pos]), False
        for i, p in enumerate(ps):
            if p.startswith("#"):
                continue
            new, n = fn(p)
            if n:
                ps[i], changed, total = new, True, total + n
        if changed:
            out[pos] = _rebuild(chunks[pos], ps)
    return out, total


def strip_stance(chunks: dict) -> tuple[dict, int]:
    """'I argue that X' / 'This study takes the position that X' -> 'X': the claim without the announcement."""
    def one(p):
        n = 0
        for s in split_sentences(p):
            new = STANCE_LEAD.sub("", s)
            if new != s:
                new = _cap(new) if STANCE_LEAD.match(s) else new
                p, n = _swap(p, s, new), n + 1
        return p, n
    return _each(chunks, one)


def what_cleft(chunks: dict) -> tuple[dict, int]:
    """A paragraph opening 'What the sources show is that X.' -> 'X.' (real theses: 2% of paragraphs open on 'What', AI drafts 14%)."""
    def one(p):
        sents = split_sentences(p)
        m = WHAT_CLEFT.match(sents[0]) if sents else None
        if not m or len(m.group(1).split()) < 6:
            return p, 0
        prefix = sents[0][:m.start(1)]
        if FIGURE.search(prefix) or re.search(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)", prefix):
            return p, 0                             # the prefix holds a figure or a citation: the rewrite would delete it
        return _swap(p, sents[0], _cap(m.group(1))), 1
    return _each(chunks, one)


def _paren(who: str, yr: str) -> str:
    return f"({who.replace(' and ', ' & ')}, {yr})"


def report_to_cite(chunks: dict, according: int = 2) -> tuple[dict, int]:
    """'Lee et al. (2015) show that X.' -> 'X (Lee et al., 2015).' so a paragraph is not a list of what each source did. Every
    other conversion becomes 'According to Lee et al. (2015), X.' (a plain signpost, still narrative), up to `according` of them."""
    state = {"n": 0, "acc": 0}

    def one(p):
        n = 0
        for s in split_sentences(p):
            m = REPORT_THAT.match(s)
            if not m or len(m.group("rest").split()) < 5:
                continue
            rest, tail = m.group("rest"), ""
            if rest[-1] in ".!?":
                rest, tail = rest[:-1], rest[-1]
            if re.search(r"\)\s*$", rest) or (state["n"] % 2 == 0 and state["acc"] < according):
                new, state["acc"] = f"According to {m.group('who')} ({m.group('yr')}), {rest}{tail}", state["acc"] + 1
            else:
                new = f"{_cap(rest)} {_paren(m.group('who'), m.group('yr'))}{tail}"
            state["n"] += 1
            p, n = _swap(p, s, new), n + 1
        return p, n
    return _each(chunks, one)


def drop_closers(chunks: dict) -> tuple[dict, int]:
    """Delete a paragraph's last sentence when it is a verdict line ('Taken together ...', 'The upshot ...') that carries no citation
    and no figure, as long as two sentences remain."""
    def one(p):
        sents = split_sentences(p)
        if len(sents) >= 3 and CLOSER.search(sents[-1]) and not FIGURE.search(sents[-1]) and not re.search(r"\((?:[^()]*?)(?:19|20)\d\d[a-z]?\)", sents[-1]):
            return _swap(p, sents[-1], ""), 1
        return p, 0
    return _each(chunks, one)


def drop_boosters(chunks: dict) -> tuple[dict, int]:
    """'genuinely contested' -> 'contested': emphasis adverbs run 1.5 per 1000 words in AI drafts and 0.2 in real theses."""
    def one(p):
        n = 0
        for s in split_sentences(p):
            new = _BOOST_LEAD.sub("", s, count=1)
            new = _cap(new) if new != s else new
            new = re.sub(r",\s*,", ",", _BOOST_MID.sub("", new))
            if new != s:
                p, n = _swap(p, s, new), n + 1
        return p, n
    return _each(chunks, one)


def thin_that(chunks: dict, per_1000_limit: float) -> tuple[dict, int]:
    """Drop 'that' after a saying verb before a determiner or pronoun ('found that the sensors' -> 'found the sensors') while the
    draft's rate is over `per_1000_limit`. Real theses use 'that' ~11 per 1000 words, AI drafts ~25."""
    words = max(1, sum(len(chunks[pos].split()) for pos in POSITIONS))
    over = sum(len(re.findall(r"\bthat\b", chunks[pos], re.I)) for pos in POSITIONS) - per_1000_limit * words / 1000
    state = {"drop": int(over), "n": 0}
    if state["drop"] <= 0:
        return chunks, 0

    def one(p):
        def sub(m):
            if state["drop"] > 0:                               # first ones first; the count is cut to the limit, so a second run is a no-op
                state["drop"] -= 1
                return m.group(1) + " "
            return m.group(0)
        new = _THAT_AFTER.sub(sub, p)
        return new, int(new != p)
    return _each(chunks, one)


def plain_words(chunks: dict) -> tuple[dict, int]:
    """'utilize' -> 'use', 'in order to' -> 'to': real theses average 5.25 letters a word, AI drafts 5.97. Text inside quotation
    marks is left alone, and a swap that starts a sentence keeps its capital."""
    def sub(m):
        new = PLAIN[m.group(1).lower()]
        return _cap(new) if m.group(1)[:1].isupper() else new

    def one(p):
        if '"' in p or "“" in p:
            return p, 0
        new, n = _PLAIN.subn(sub, p)
        return new, n
    return _each(chunks, one)


def polish(chunks: dict, that_limit: float | None = None) -> tuple[dict, dict]:
    """All free wording rewrites, in order. -> (chunks, {name: count})."""
    counts = {}
    for name, fn in (("stance", strip_stance), ("what", what_cleft), ("report", report_to_cite), ("closer", drop_closers),
                     ("booster", drop_boosters), ("plain", plain_words)):
        chunks, counts[name] = fn(chunks)
    if that_limit:
        chunks, counts["that"] = thin_that(chunks, that_limit)
    return chunks, counts


_HAS_FACT = re.compile(r"\d|\([^)]*(?:19|20)\d\d|[a-z] [A-Z][a-z]")      # a figure, a citation, or a proper noun inside the sentence
MICRO_WORDS = 4


def drop_micro(chunks: dict) -> dict:
    """Free: delete a rhetorical punch of MICRO_WORDS words or fewer ("It is not decoration." "Choice matters here.") that holds no
    figure, citation or name, while its paragraph keeps three sentences. A writer told to include a short sentence stacks these."""
    out = dict(chunks)
    for pos in POSITIONS:
        ps = paras(chunks[pos])
        changed = False
        for i, p in enumerate(ps):
            if _BLOCKISH.match(p):
                continue
            sents = split_sentences(p)
            keep = [x for x in sents if len(x.split()) > MICRO_WORDS or _HAS_FACT.search(x)]
            if len(keep) != len(sents) and len(keep) >= MIN_SENTS:
                ps[i], changed = " ".join(keep), True
        if changed:
            out[pos] = _rebuild(chunks[pos], ps)
    return out


_NARR_OPEN = re.compile(r"^((?:[A-Z][\w'\u2019-]+(?: et al\.)?)(?: (?:and|&) [A-Z][\w'\u2019-]+)?) \(((?:19|20)\d\d[a-z]?)\) "
                        r"(?:also |then |further )?(?:found|noted|argued|showed|reported|observed|suggested|demonstrated|concluded|"
                        r"identified|stated|claimed|reveal(?:ed)?|indicated|highlighted) that (.+?)([.!?])$")
_PAREN_ITEM = re.compile(r"[A-Z][\w'\u2019-]+(?: et al\.?| (?:&|and) [A-Z][\w'\u2019-]+)?,\s*(?:19|20)\d\d[a-z]?")
_PAREN_GROUP = re.compile(r"\(([^()]*(?:19|20)\d\d[^()]*)\)")


def _narr(x: str) -> int:
    return sum(1 for m in NARRATIVE_RE.finditer(x) if x[max(0, m.start() - 1):m.start()] != "(")


def _restate(x: str):
    """'Smith (2010) found that X.' -> 'X (Smith, 2010).'  None when the sentence is not that plain pattern."""
    m = _NARR_OPEN.match(x.strip())
    if not m or _narr(x) != 1:
        return None
    body = m.group(3)
    return f"{body[:1].upper()}{body[1:]} ({m.group(1).replace(' and ', ' & ')}, {m.group(2)}){m.group(4)}"


def limit_narrative(chunks: dict, per_call: int = limits.NARR_CALL, per_section: int = limits.NARR_SECTION,
                   paren_share: float = limits.PAREN_SHARE) -> dict:
    """Free: turn "Smith (2010) found that X." into "X (Smith, 2010)." until a passage holds at most `per_call` narrative citations,
    the section at most `per_section`, and at least `paren_share` of its citations are parenthetical. Latest sentences go first.
    The citation, its words and every number are unchanged; only the order moves. A sentence that is not the plain pattern is left
    alone (the validator still reports it)."""
    doc = {pos: [[split_sentences(p) if not _BLOCKISH.match(p) else [p]] for p in paras(chunks[pos])] for pos in POSITIONS}
    doc = {pos: [x[0] for x in ps] for pos, ps in doc.items()}
    where = [(pos, i, j) for pos in POSITIONS for i, sents in enumerate(doc[pos]) for j, x in enumerate(sents) if _narr(x)]

    def flip(k):
        pos, i, j = k
        new = _restate(doc[pos][i][j])
        if new:
            doc[pos][i][j] = new
        return bool(new)

    for pos in POSITIONS:                                           # 1. per passage
        for k in [w for w in where if w[0] == pos][per_call:]:
            flip(k)
    left = [w for w in where if _narr(doc[w[0]][w[1]][w[2]])]
    for k in left[per_section:]:                                    # 2. per section
        flip(k)
    left = [w for w in where if _narr(doc[w[0]][w[1]][w[2]])]
    paren = sum(len(_PAREN_ITEM.findall(g)) for pos in POSITIONS for sents in doc[pos] for x in sents for g in _PAREN_GROUP.findall(x))
    for k in reversed(left):                                        # 3. parenthetical share
        if paren / max(1, paren + len(left)) >= paren_share:
            break
        if flip(k):
            paren, left = paren + 1, [w for w in left if w != k]
    out = dict(chunks)
    for pos in POSITIONS:
        ps = paras(chunks[pos])
        new = [" ".join(sents) if not _BLOCKISH.match(ps[i]) else ps[i] for i, sents in enumerate(doc[pos])]
        if new != ps:
            out[pos] = _rebuild(chunks[pos], new)
    return out


demote_narrative = limit_narrative
