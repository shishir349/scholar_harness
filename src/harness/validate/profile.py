"""Use data/style_profile.json (mined by eval/phrase_gap.py) in the harness: a prompt block, a phrase detector and the
paragraph-shape numbers. Everything is optional: no file means no block, no detector, the old fixed paragraph plan."""
import json
import re
from functools import lru_cache
from pathlib import Path

from .. import config
from ..corpus.profile import SIGNPOSTS, _citation_counts, _sentences
from .style import feature_stats

PROFILE_PATH = config.DATA / "style_profile.json"
TELL_RATIO = 10     # a phrase counts as a tell when the AI drafts use it >= 10x more than real theses do


@lru_cache(maxsize=2)
def load_profile(path: str | None = None) -> dict | None:
    try:
        return json.loads(Path(path or PROFILE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tells(profile: dict) -> list[str]:
    return [e["p"] for e in (profile or {}).get("avoid", []) if e["ai"] / max(e["real"], 0.01) >= TELL_RATIO]


def avoid_hits(text: str, profile: dict | None) -> list[str]:
    """Distinct tell phrases in `text`, in order of first appearance (whole words, any case)."""
    phrases = _tells(profile)
    if not phrases or not text:
        return []
    rx = re.compile(r"(?<!\w)(?:" + "|".join(r"\s+".join(map(re.escape, p.split())) for p in phrases) + r")(?!\w)", re.I)
    seen: list[str] = []
    for m in rx.finditer(text):
        p = " ".join(m.group(0).lower().split())
        if p not in seen:
            seen.append(p)
    return seen


def plan_ratios(section_type: str, profile: dict | None) -> list[float] | None:
    pr = (profile or {}).get("paragraph_ratios") or {}
    return pr.get(section_type) or pr.get("all")


def _range(section_type: str, name: str, baseline, fmt: str, scale: float = 1.0) -> str | None:
    st = feature_stats(section_type, name, baseline)
    if not st or "p10" not in st or "p90" not in st:
        return None
    return f"{fmt.format(st['p10'] * scale)}-{fmt.format(st['p90'] * scale)}"


def _gap_lines(section_type: str, profile: dict) -> list[str]:
    """One plain instruction per measured gap, with the real number to aim at and the AI drafts' number for contrast."""
    tg = profile.get("targets") or {}
    t = tg.get(section_type) or tg.get("all") or {}
    ai = profile.get("ai_means") or {}

    def say(k: str, fmt: str, text: str):
        if k not in t:
            return None
        mid = fmt.format(t[k]["p50"])
        contrast = f" (AI drafts {('{:.2f}' if fmt == '{:.2f}' else '{:.1f}').format(ai[k])})" if k in ai else ""
        return text.replace("{r}", mid).replace("{ai}", contrast)

    rules = (
        ("connectives", "{:.0f}", "- Join claims with plain connectives (also, however, therefore, thus): about {r} per 1000 words in real theses{ai}."),
        ("that", "{:.0f}", '- The word "that": about {r} per 1000 words in real theses{ai}. Drop it wherever the sentence reads fine without it.'),
        ("of", "{:.0f}", '- Plain "the X of Y" phrasing is normal: "of" about {r} per 1000 words{ai}.'),
        ("word_len", "{:.1f}", "- Average word length is {r} letters in real theses{ai}: choose the short plain word (use, show, need, help) over "
                               "the long one (utilise, demonstrate, facilitate)."),
        ("booster", "{:.1f}", "- Emphasis words (clearly, genuinely, fundamentally, obviously): about {r} per 1000 words in real theses{ai}. Leave them out."),
        ("passive", "{:.0f}", "- Passive verbs (was collected, were tested, is shown) about {r} per 1000 words{ai}; use them for what was done or found."),
        ("digits", "{:.0f}", "- Real theses give exact figures: about {r} numbers per 1000 words{ai}. State every figure, sample size and date the "
                             "source notes give, exactly; never invent one."),
        ("capitalised_mid", "{:.0f}", "- Name things: instruments, datasets, places, organisations (about {r} capitalised names per 1000 words{ai}). "
                                      "Write an acronym out once, then use the acronym."),
        ("the_field_lit", "{:.1f}", '- Do not write about "the field" or "the literature" as if it were a person ({r} per 1000 words in real theses{ai}); '
                                    "name the studies instead."),
        ("ttr_1000", "{:.2f}", "- Words: repeat the same technical term instead of swapping synonyms; unique-word ratio is {r} in real theses{ai}."),
    )
    return [x for x in (say(k, fmt, text) for k, fmt, text in rules) if x]


def profile_block(section_type: str, profile: dict | None, baseline: dict | None = None) -> str:
    """Prompt text from the mined profile; "" when there is none."""
    if not profile:
        return ""
    quote = lambda items: ", ".join(f'"{x}"' for x in items)
    strong = _tells(profile)
    softer = [e["p"] for e in profile.get("avoid", []) if e["p"] not in strong][:10] + list(profile.get("avoid_words", []))[:10]
    prefer = [e["p"] for e in profile.get("prefer", [])][:18]
    lines = [f"REAL THESIS PHRASING (mined from {profile.get('n_real_theses', 0)} real theses; follow the pattern, never quote them):"]
    if strong:
        lines.append(f"- Real thesis writers almost never write these, and AI drafts lean on them. Say the point another way: {quote(strong[:24])}.")
    if softer:
        lines.append(f"- Use these far less often, and only when the topic needs them: {quote(softer)}.")
    if profile.get("avoid_long"):
        lines.append(f"- Long words AI drafts overuse; use a shorter plain word instead: {quote(profile['avoid_long'][:14])}.")
    if prefer or profile.get("prefer_words"):
        lines.append(f"- Real writers reach for plain connectives and framing: {quote(prefer)}; and words like "
                     f"{quote(profile.get('prefer_words', [])[:8])}.")
    sp = profile.get("signposts") or {}
    if sp.get("real_rate"):
        lines.append(f"- About {sp['real_rate']:.0%} of real sentences open with a plain signpost ({quote(sp['list'][:6])}, ...); AI drafts "
                     f"do it {sp.get('ai_rate', 0):.1%} of the time. Aim for about 1 sentence in {round(1 / sp['real_rate'])}.")
    if profile.get("avoid_openers"):
        lines.append(f"- Do not start sentences with these (AI drafts do, real theses rarely): {quote(profile['avoid_openers'][:10])}.")
    lines += _gap_lines(section_type, profile)
    share = (profile.get("citations") or {}).get("narrative_share_real")
    if share is not None:
        lines.append(f"- About {share:.0%} of citations in real theses are narrative (\"Smith (2010) argues\"); the rest sit in parentheses.")
    pieces = []
    for name, fmt, scale, tmpl in (("sent_len_mean", "{:.0f}", 1, "average sentence {} words"),
                                   ("short_sent_ratio", "{:.0f}", 100, "short sentences (under ten words) {}% of sentences"),
                                   ("first_person_per_100", "{:.1f}", 1, "first person {} per 100 words"),
                                   ("hedge_per_100", "{:.1f}", 1, "hedges {} per 100 words")):
        r = _range(section_type, name, baseline, fmt, scale)
        if r:
            pieces.append(tmpl.format(r))
    if pieces:
        lines.append("- Middle 80% of real theses of this section type: " + "; ".join(pieces) + ".")
    return "\n".join(lines)


def signpost_counts(text: str) -> tuple[int, int]:
    """(sentences opening with a plain signpost, all sentences) in `text`."""
    signs = sents = 0
    for toks in _sentences(text):
        sents += 1
        signs += any(" ".join(toks).startswith(p + " ") for p in SIGNPOSTS)
    return signs, sents


def profile_metrics(text: str, profile: dict | None) -> dict:
    """Free scores for any text, on the same scale the profile was mined on: tell phrases per 1000 words, the share of
    sentences opening with a plain signpost, and the narrative share of its citations."""
    phrases = _tells(profile)
    words = sents = signs = 0
    for toks in _sentences(text):
        words += len(toks); sents += 1
        signs += any(" ".join(toks).startswith(p + " ") for p in SIGNPOSTS)
    hits = 0
    if phrases and text:
        rx = re.compile(r"(?<!\w)(?:" + "|".join(r"\s+".join(map(re.escape, p.split())) for p in phrases) + r")(?!\w)", re.I)
        hits = len(rx.findall(text))
    nar, par = _citation_counts(text)
    return {"tells_per_1000": round(1000 * hits / words, 1) if words else 0.0, "signpost_rate": round(signs / sents, 3) if sents else 0.0,
            "narrative_share": round(nar / (nar + par), 3) if nar + par else None}
