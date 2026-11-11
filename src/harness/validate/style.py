"""Style validation against the corpus baseline (data/style_baseline.json)."""
import json
import re
import statistics
from functools import lru_cache
from pathlib import Path

from .. import config
from .. import limits
from ..text.sentences import split_sentences
from ..text.strip import headings_and_text, strip_for_style
from .features import FEATURES, ONE_SIDED_HIGH, style_features

FLAT_STD = limits.STD_MIN              # a paragraph of 3+ sentences with a sentence-length std under this is flat
MAX_SENT_WORDS = limits.MAX_SENT       # a longer sentence is a compound Claude sentence ("... in the sense that ... and in the sense that ...")
SHORT_SENT_WORDS = limits.SHORT_MAX    # every paragraph of 3+ sentences holds a sentence of this many words or fewer
FRAGMENT_WORDS = limits.FRAGMENT       # ...but two sentences under this are stacked fragments ("The data are numerical. The variables are defined.")
STACK_WORDS = limits.STACK             # ...as are three short sentences in a row
BASELINE_PATH = config.DATA / "style_baseline.json"


@lru_cache(maxsize=4)
def load_baseline(path: str | None = None) -> dict | None:
    p = Path(path) if path else BASELINE_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _entry(baseline: dict, section_type: str) -> dict | None:
    by = baseline.get("by_section", {})
    return by.get(section_type) or by.get("all")


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


# Smallest believable spread per feature. A corpus where a feature barely varies (e.g. stock phrases: ~0
# everywhere) must still be able to flag a draft that has many, without flagging trivial deviations.
STD_FLOOR = {"sent_len_mean": 1.0, "sent_len_cv": 0.05, "short_sent_ratio": 0.03, "long_sent_ratio": 0.02,
             "para_len_cv": 0.05, "hedge_per_100": 0.2, "booster_per_100": 0.1, "citation_per_100": 0.2,
             "first_person_per_100": 0.2, "opener_repeat": 0.05, "stock_per_1000": 1.0, "em_dash_per_para": 0.1}


def score_features(features: dict, entry: dict) -> dict:
    """z-scores of `features` against a baseline entry -> {style_score, flags, max_abs_z}."""
    penalties, flags, max_abs = [], [], 0.0
    for name in FEATURES:
        stats = entry["features"].get(name)
        if not stats or features.get(name) is None:
            continue
        z = (features[name] - stats["mean"]) / max(stats.get("std", 0.0), STD_FLOOR[name])
        eff = max(z, 0.0) if name in ONE_SIDED_HIGH else abs(z)
        penalties.append(min(eff, 3.0) / 3.0)
        max_abs = max(max_abs, eff)
        if eff > 2:
            flags.append({"feature": name, "value": round(features[name], 4), "mean": round(stats["mean"], 4),
                          "z": round(z, 2), "direction": "high" if z > 0 else "low"})
    score = 100.0 - 100.0 * (statistics.fmean(penalties) if penalties else 0.0)
    return {"style_score": score, "flags": flags, "max_abs_z": max_abs}


def score_section(text: str, section_type: str, baseline: dict | None = None) -> dict:
    """{available, style_score, passed, flags, features}. Never raises; unavailable when no baseline/too short."""
    baseline = baseline if baseline is not None else load_baseline()
    if not baseline:
        return {"available": False, "reason": "no baseline"}
    entry = _entry(baseline, section_type)
    feats = style_features(text)
    if entry is None or feats is None:
        return {"available": False, "reason": "no baseline entry" if entry is None else "under 150 words"}
    r = score_features(feats, entry)
    p10 = baseline.get("score_p10", {}).get(section_type, baseline.get("score_p10", {}).get("all"))
    passed = r["max_abs_z"] <= 3 and (p10 is None or r["style_score"] >= p10)
    return {"available": True, "style_score": round(r["style_score"], 2), "passed": passed,
            "flags": r["flags"], "features": feats}


def feature_stats(section_type: str, name: str, baseline: dict | None = None) -> dict | None:
    """{mean, std, p10, p90} of one feature in real theses of this section type, or None."""
    baseline = baseline if baseline is not None else load_baseline()
    entry = _entry(baseline, section_type) if baseline else None
    return (entry or {}).get("features", {}).get(name)


def style_target(section_type: str, baseline: dict | None = None) -> str:
    """Plain-language target numbers pulled straight from the corpus baseline (mean sentence length,
    hedge rate, citation rate) - a concrete number the model can aim for, instead of 2-4 example
    paragraphs it has to infer a style from on its own."""
    baseline = baseline if baseline is not None else load_baseline()
    if not baseline:
        return ""
    entry = _entry(baseline, section_type)
    if not entry:
        return ""
    f = entry["features"]

    def mean(name: str) -> float | None:
        return f.get(name, {}).get("mean")

    sl, cv, hedge, cite = mean("sent_len_mean"), mean("sent_len_cv"), mean("hedge_per_100"), mean("citation_per_100")
    if sl is None:
        return ""
    return (f"STYLE TARGET (measured from real theses of this section type, aim near these, don't force exact "
            f"numbers): average sentence length around {sl:.0f} words with real variation - mix short and long "
            f"sentences, not uniform length (spread: {cv:.2f}). Hedge words like 'may', 'suggests', 'appears to' "
            f"about {hedge:.1f} per 100 words. Citations about {cite:.1f} per 100 words.")


def build_baseline(units_by_type: dict[str, list[str]]) -> dict:
    """Aggregate corpus units into the baseline JSON structure (used by corpus.ingest)."""
    by_section: dict = {}
    feats_by_type: dict[str, list[dict]] = {}
    for typ, texts in units_by_type.items():
        rows = [f for f in (style_features(t) for t in texts) if f]
        if rows:
            feats_by_type[typ] = rows
    feats_by_type["all"] = [r for rows in list(feats_by_type.values()) for r in rows]
    for typ, rows in feats_by_type.items():
        stats = {}
        for name in FEATURES:
            vals = sorted(r[name] for r in rows)
            stats[name] = {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                           "p10": _percentile(vals, 0.10), "p90": _percentile(vals, 0.90)}
        by_section[typ] = {"n": len(rows), "features": stats}
    score_p10 = {}
    for typ, rows in feats_by_type.items():
        scores = sorted(score_features(r, by_section[typ])["style_score"] for r in rows)
        score_p10[typ] = _percentile(scores, 0.10)
    return {"version": 1, "n_units": len(feats_by_type["all"]), "by_section": by_section, "score_p10": score_p10}


def split_chapter_by_sections(content: str, toc_sections: list[dict] | None) -> list[dict]:
    """[{number, title, text}] by locating each section heading; whole chapter when none are found."""
    marked = headings_and_text(content)
    lines = marked.split("\n")
    starts = []
    for sec in toc_sections or []:
        num, title = str(sec.get("number", "")).strip(), str(sec.get("title", "")).strip()
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("#") and ((num and re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", ln)) or
                                                (title and title.lower() in ln.lower())):
                if all(i != s[0] for s in starts):
                    starts.append((i, num, title))
                break
    starts.sort()
    if not starts:
        return [{"number": "", "title": "chapter", "text": strip_for_style(content)}]
    out = []
    for k, (i, num, title) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out.append({"number": num, "title": title, "text": strip_for_style("\n".join(lines[i + 1:end]))})
    return out


def burst_flags(text: str) -> list[tuple[int, str]]:
    """[(paragraph index, note)] for paragraphs without bimodal burstiness: three sentences in a row of 18-26 words, or four or
    more sentences with almost no spread. Headings and short paragraphs are skipped. The index counts every paragraph, headings too."""
    out = []
    for i, p in enumerate(q for q in re.split(r"\n\s*\n", text) if q.strip()):
        if p.lstrip().startswith("#"):
            continue
        n = [len(s.split()) for s in split_sentences(p)]
        if len(n) < 3:
            continue
        run = next((j for j in range(len(n) - 2) if all(18 <= x <= 26 for x in n[j:j + 3])), None)
        if run is not None:
            out.append((i, f"sentences {run + 1}-{run + 3} run {n[run]}, {n[run + 1]}, {n[run + 2]} words"))
        elif statistics.pstdev(n) < FLAT_STD:
            out.append((i, f"sentence lengths {n} barely vary (std {statistics.pstdev(n):.1f}, need {FLAT_STD})"))
        elif max(n) > MAX_SENT_WORDS:
            out.append((i, f"a {max(n)}-word sentence, the limit is {MAX_SENT_WORDS}"))
        elif sum(1 for x in n if x < FRAGMENT_WORDS) >= 2 or any(all(x < STACK_WORDS for x in n[j:j + 3]) for j in range(len(n) - 2)):
            out.append((i, f"sentence lengths {n}: stacked fragments, keep one blunt {FRAGMENT_WORDS}-9 word sentence"))
        elif min(n) > SHORT_SENT_WORDS:
            out.append((i, f"sentence lengths {n}: none of {SHORT_SENT_WORDS} words or fewer"))
    return out
