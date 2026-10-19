"""Mine what real theses do differently from saved AI drafts -> data/style_profile.json (see eval/phrase_gap.py).

Pure functions, no API calls. Method: n-gram log-odds with an informative prior (Monroe et al. 2008). A phrase only counts
as an AI tell when it shows up across several different fixtures (topic words cannot pose as style) and a phrase only
counts as a real-thesis habit when many different theses use it (one author's quirk cannot)."""
import math
import random
import re
from collections import Counter, defaultdict

from ..text.citations import extract_citations
from ..text.strip import strip_for_style
from ..validate.gapfeats import feats

A0 = 300                                           # prior strength
FUNC = frozenset("the of to in a an and for on by with as be is are was were been that this these those it its or at from "
                 "not but which what than so if there their they we i you he she his her our can may will would should".split())
ARTIFACT = frozenset("journal cited retrieved available pp doi http https www vol al isbn edn proceedings chapter".split())
SIGNPOSTS = ("for example", "for instance", "in addition", "however", "according to", "based on", "as a result",
             "in contrast", "in particular", "in other words", "for this reason")
STYLE_SUFFIX = ("ly", "ize", "ise", "izes", "ises", "ate", "ates", "ated", "ating", "ive", "ous", "ful", "ify", "ifies")
QUANT = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)    # paragraph length relative to the chunk's mean paragraph
SENT = re.compile(r"(?<=[.!?])\s+")
TOK = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


def _sentences(text: str):
    for s in SENT.split(strip_for_style(text)):
        toks = TOK.findall(s.lower())
        if len(toks) >= 3:
            yield toks


def _grams(toks, n):
    return (" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _z(a: int, r: int, na: int, nr: int) -> float:
    aw = A0 * (a + r) / (na + nr)
    d = math.log((a + aw) / (na - a + A0 - aw)) - math.log((r + aw) / (nr - r + A0 - aw))
    return d / math.sqrt(1 / (a + aw) + 1 / (r + aw))


def _quantile(vals: list, q: float) -> float:
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _plain(g: str) -> bool:
    """False for bare function words, one-letter tokens and bibliography debris: nothing a writer can act on."""
    toks = g.split()
    return not all(t in FUNC for t in toks) and not any(t in ARTIFACT or len(t) < 2 for t in toks)


def _dedupe(scored: list, top: int) -> list:
    """scored: [(z, phrase)]. Best z first; drop any phrase that contains, or sits inside, one already kept."""
    kept: list = []
    # phrases first (a bare word like "field" tells a writer nothing), best z first, ties to the shorter phrase
    for z, g in sorted(scored, key=lambda t: (" " not in t[1], -round(t[0], 3), t[1].count(" "), t[1])):
        padded = f" {g} "
        if not any(padded in f" {k} " or f" {k} " in padded for _, k in kept):
            kept.append((z, g))
    return kept[:top]


BAD_START = frozenset("is are was were be been has have had not and or but which when if than so to it there can may will would "
                      "should does do did we i".split())
ALLOWED_TAIL = frozenset(["such as", "according to", "based on", "due to", "because of", "related to", "compared to"])


def _edge_ok(phrase: str) -> bool:
    """A phrase a writer can use as is: not opening on a verb helper or conjunction, not stopping on a dangling
    function word (unless it is a fixed connective such as 'according to' or already 3+ words)."""
    w = phrase.split()
    return w[0] not in BAD_START and not (w[-1] in FUNC and len(w) < 3 and phrase not in ALLOWED_TAIL)


def _extend(frags: list, docs: list, share: float = 0.7, max_words: int = 6) -> dict:
    """{fragment: longest phrase that contains it and that >= `share` of its occurrences extend to}.
    Turns "well as" / "as well" into "as well as" and "study takes" into "study takes the position that" - only where
    the corpus text itself repeats it, never by gluing unrelated fragments."""
    by_first: dict = defaultdict(list)
    for f in frags:
        by_first[f.split()[0]].append(f.split())
    total: Counter = Counter()
    wins: dict = defaultdict(Counter)
    for _, _, text in docs:
        for toks in _sentences(text):
            for i, t in enumerate(toks):
                for fw in by_first.get(t, ()):
                    j = i + len(fw)
                    if toks[i:j] != fw:
                        continue
                    key = " ".join(fw)
                    total[key] += 1
                    for left in range(0, 4):
                        for right in range(0, 4):
                            if (left or right) and i - left >= 0 and j + right <= len(toks) and len(fw) + left + right <= max_words:
                                wins[key][" ".join(toks[i - left:j + right])] += 1
    out = {}
    for f in frags:
        best = f
        for w, c in wins[f].items():
            if c >= share * total[f] and (len(w.split()), c) > (len(best.split()), 0):
                best = w
        out[f] = best
    return out


def _occurrences(docs: list, phrases: list) -> Counter:
    """How often each phrase occurs inside a sentence (docs: [(id, type, text)])."""
    padded = [(p, f" {p} ") for p in phrases]
    out: Counter = Counter()
    for _, _, text in docs:
        for toks in _sentences(text):
            sent = " " + " ".join(toks) + " "
            for p, q in padded:
                if q in sent:
                    out[p] += sent.count(q)
    return out


# features the gap report found far from real prose; their real ranges become prompt instructions
TARGET_FEATURES = ("connectives", "that", "of", "word_len", "long_word_share", "nominalisation", "booster", "passive", "digits",
                   "capitalised_mid", "acronyms", "short_sent_share", "the_field_lit", "ttr_1000", "figure_table_refs", "paren",
                   "first_person", "hedge")


def feature_targets(real: list, window: int = 3, per_thesis: int = 8, seed: int = 7) -> dict:
    """real: [(thesis_id, section_type, chunk_idx, text)]. -> {section_type | 'all': {feature: {p10, p50, p90, n}}}, measured on
    windows of `window` consecutive chunks (a drafted section's size), at most `per_thesis` per thesis so no author dominates."""
    groups: dict = defaultdict(list)
    for tid, st, idx, text in sorted(real, key=lambda r: (r[0], r[1], r[2])):
        groups[(tid, st)].append(text)
    wins: dict = defaultdict(list)
    for (tid, st), texts in groups.items():
        for i in range(0, len(texts) - window + 1, window):
            wins[tid].append((st, "\n\n".join(texts[i:i + window])))
    rng = random.Random(seed)
    vals: dict = defaultdict(lambda: defaultdict(list))
    for tid, ws in wins.items():
        rng.shuffle(ws)
        for st, text in ws[:per_thesis]:
            f = feats(text)
            if f:
                for k in TARGET_FEATURES:
                    vals[st][k].append(f[k]); vals["all"][k].append(f[k])
    out = {}
    for st, d in vals.items():
        if len(d[TARGET_FEATURES[0]]) >= 8:
            out[st] = {k: {"p10": round(_quantile(sorted(v), 0.1), 3), "p50": round(_quantile(sorted(v), 0.5), 3),
                           "p90": round(_quantile(sorted(v), 0.9), 3), "n": len(v)} for k, v in d.items()}
    return out


def _citation_counts(text: str) -> tuple[int, int]:
    nar = par = 0
    for c in extract_citations(text):
        if c["raw"].startswith("("):
            par += 1
        else:
            nar += 1
    return nar, par


def mine(ai: list, real: list, *, top: int = 40, min_fixtures: int = 5, min_ai_count: int = 4, min_real_count: int = 60,
         min_real_theses: int | None = None, min_tri_support: int = 25, ratio_real: list | None = None,
         min_fixtures_word: int = 7, min_fixtures_long: int = 5, min_fixtures_ai_only: int = 8) -> dict:
    """ai: [(fixture, section_type, text)]; real: [(thesis_id, section_type, text)] (re-iterable).
    -> {avoid, prefer, avoid_openers, prefer_openers, signposts, citations, paragraph_ratios, ...counts}"""
    cnt_a = {n: Counter() for n in (1, 2, 3)}
    fix_a = {n: defaultdict(set) for n in (1, 2, 3)}
    open_a: Counter = Counter()
    wa = sa = sign_a = nar_a = par_a = 0
    for fx, _, text in ai:
        for toks in _sentences(text):
            wa += len(toks); sa += 1
            open_a[" ".join(toks[:2])] += 1
            sign_a += any(" ".join(toks).startswith(p + " ") for p in SIGNPOSTS)
            for n in (1, 2, 3):
                for g in _grams(toks, n):
                    cnt_a[n][g] += 1; fix_a[n][g].add(fx)
        a, b = _citation_counts(text); nar_a += a; par_a += b

    cnt_r = {n: Counter() for n in (1, 2, 3)}
    open_r: Counter = Counter()
    wr = sr = sign_r = nar_r = par_r = 0
    ratios: dict = defaultdict(list)
    for tid, st, text in real:                                            # pass 0: words, 2-word phrases, shape
        for toks in _sentences(text):
            wr += len(toks); sr += 1
            open_r[" ".join(toks[:2])] += 1
            sign_r += any(" ".join(toks).startswith(p + " ") for p in SIGNPOSTS)
            for n in (1, 2):
                for g in _grams(toks, n):
                    cnt_r[n][g] += 1
        a, b = _citation_counts(text); nar_r += a; par_r += b
    for tid, st, text in (real if ratio_real is None else ratio_real):    # paragraph shape: every section type, not only the ones drafted
        paras = [len(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip() and not p.lstrip().startswith("#")]
        if len(paras) >= 3:
            m = sum(paras) / len(paras)
            ratios[st].extend(w / m for w in paras); ratios["all"].extend(w / m for w in paras)
    ok_bi = {g for g, c in cnt_r[2].items() if c >= min_tri_support}
    ai_tri = cnt_a[3]
    for tid, st, text in real:                                            # pass 1: 3-word phrases, pruned
        for toks in _sentences(text):
            for i in range(len(toks) - 2):
                a, b, c = toks[i], toks[i + 1], toks[i + 2]
                g = f"{a} {b} {c}"
                if g in ai_tri or (f"{a} {b}" in ok_bi and f"{b} {c}" in ok_bi):
                    cnt_r[3][g] += 1
    common = {n: {g for g, c in cnt_r[n].items() if c >= min_real_count} for n in (1, 2, 3)}
    df: dict = {n: defaultdict(set) for n in (1, 2, 3)}
    for tid, st, text in real:                                            # pass 2: how many theses use each common phrase
        for toks in _sentences(text):
            for n in (1, 2, 3):
                for g in _grams(toks, n):
                    if g in common[n]:
                        df[n][g].add(tid)

    n_theses = len({t for t, _, _ in real})
    thr = min_real_theses or max(15, n_theses // 4)           # a phrase real writers share, not one author's habit
    word_thr = int(0.6 * n_theses)                              # a bare word must be in most theses to count as style, not topic
    rate = lambda c, n: 1000 * c / n
    avoid_c, prefer_c = [], []
    for n in (1, 2, 3):
        for g, a in cnt_a[n].items():
            if a >= min_ai_count and len(fix_a[n][g]) >= min_fixtures and _plain(g) and (n > 1 or len(g) >= 4):
                r = cnt_r[n].get(g, 0)
                z = _z(a, r, wa, wr)
                if z > 0 and rate(a, wa) / max(rate(r, wr), 0.01) >= (3 if n > 1 else 15):
                    avoid_c.append((z, g))
        for g in common[n]:
            r = cnt_r[n][g]
            if len(df[n][g]) >= (thr if n > 1 else word_thr) and _plain(g) and (n > 1 or len(g) >= 4):
                a = cnt_a[n].get(g, 0)
                z = _z(a, r, wa, wr)
                if z < 0 and rate(r, wr) / max(rate(a, wa), 0.01) >= 3:
                    prefer_c.append((-z, g))
    def finish(cands: list, docs: list, n_top: int) -> list:
        frags = [g for _, g in _dedupe([t for t in cands if " " in t[1]], n_top * 2)]
        z_of = {g: z for z, g in cands}
        best: dict = {}
        for f, e in _extend(frags, docs).items():
            if _edge_ok(e):
                best[e] = max(best.get(e, -1e9), z_of[f])
        return _dedupe([(z, g) for g, z in best.items()], n_top)

    avoid_ph, prefer_ph = finish(avoid_c, ai, top), finish(prefer_c, real, top)
    inside = lambda w, phs: any(f" {w} " in f" {g} " for _, g in phs)
    avoid_w = [g for _, g in sorted((t for t in avoid_c if " " not in t[1] and len(fix_a[1][t[1]]) >= min_fixtures_word
                                     and not inside(t[1], avoid_ph)), reverse=True)][:12]
    prefer_w = [g for _, g in sorted((t for t in prefer_c if " " not in t[1] and len(df[1][t[1]]) >= word_thr
                                      and not inside(t[1], prefer_ph)), reverse=True)][:10]
    # long words (8+ letters): either real writers use them across many theses but far less often than the drafts do, or only the
    # drafts use them and they appear in nearly every fixture (so they are a habit, not a topic)
    shared = max(5, int(0.2 * n_theses))
    longw = []
    for g, a in cnt_a[1].items():
        if len(g) < 8 or a < min_ai_count or not _plain(g) or not g.endswith(STYLE_SUFFIX):   # adverbs, adjectives, verbs: crutches, not topics
            continue
        r, nfix = cnt_r[1].get(g, 0), len(fix_a[1][g])
        z = _z(a, r, wa, wr)
        if z <= 0:
            continue
        if g in common[1] and len(df[1][g]) >= shared and rate(a, wa) / max(rate(r, wr), 0.01) >= 6 and nfix >= min_fixtures_long:
            longw.append((z, g))
        elif g not in common[1] and nfix >= min_fixtures_ai_only:
            longw.append((z, g))
    avoid_long = [g for _, g in sorted(longw, reverse=True) if g not in avoid_w][:14]
    names = [g for _, g in avoid_ph + prefer_ph]
    occ_a, occ_r = _occurrences(ai, names), _occurrences(real, names)
    entry = lambda g: {"p": g, "ai": round(rate(occ_a[g], wa), 3), "real": round(rate(occ_r[g], wr), 3)}
    avoid = [entry(g) for _, g in avoid_ph]
    prefer = [entry(g) for _, g in prefer_ph]

    def openers(c_hi: Counter, n_hi: int, c_lo: Counter, n_lo: int, min_hi: int, hi_is_ai: bool) -> list:
        out = []
        for g, a in c_hi.items():
            if a >= min_hi and _plain(g):
                b = c_lo.get(g, 0)
                z = _z(a, b, n_hi, n_lo)
                if z > 0 and (a / n_hi) / max(b / n_lo, 1e-5) >= 3:
                    out.append((z, g))
        return [g for _, g in sorted(out, reverse=True)[:12]]

    avoid_open = openers(open_a, sa, open_r, sr, 5, True)
    prefer_open = openers(open_r, sr, open_a, sa, min_real_count * 3, False)
    para = {}
    for st, vals in ratios.items():
        if len(vals) >= 20:
            vals.sort()
            para[st] = [round(min(3.0, max(0.3, _quantile(vals, q))), 2) for q in QUANT]
    share = lambda n, p: round(n / (n + p), 3) if n + p else None
    return {
        "version": 1, "n_ai_docs": len(ai), "n_fixtures": len({fx for fx, _, _ in ai}), "ai_words": wa,
        "n_real_theses": len({t for t, _, _ in real}), "n_real_chunks": len(real), "real_words": wr,
        "avoid": avoid, "avoid_words": avoid_w, "avoid_long": avoid_long, "prefer": prefer, "prefer_words": prefer_w, "avoid_openers": avoid_open, "prefer_openers": prefer_open,
        "signposts": {"list": list(SIGNPOSTS), "real_rate": round(sign_r / max(sr, 1), 4), "ai_rate": round(sign_a / max(sa, 1), 4)},
        "citations": {"narrative_share_real": share(nar_r, par_r), "narrative_share_ai": share(nar_a, par_a)},
        "paragraph_ratios": para,
    }
