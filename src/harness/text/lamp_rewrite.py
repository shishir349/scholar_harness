"""LAMP span-level rewrite (Chakrabarty et al. 2025): edit only the spans a detector flagged, not whole chunks.

One batched Sonnet call at temperature 0.85 gets the numbered spans, a few real human sentence pairs for rhythm, and one edit
instruction per span. Every returned span is then checked on the spot; a span that fails keeps its original text, so one bad
edit never costs the section. Runs after claim verification: protect.compare guarantees no citation, number or quote moves."""
import re

from .. import config, llm
from ..corpus import lamp_bank
from ..validate import detector
from ..validate.lamp_detector import MAX_SENT_WORDS, detect_ai_spans, paragraph_violations
from ..text.sentences import split_sentences
from ..validate.features import _STOCK_RX
from ..validate.overlap import exemplar_ngrams, passage_overlaps
from . import protect
from .jsonparse import parse_json

MAX_SPANS = 14
TEMPERATURE = 0.85
N_EXAMPLES = 8
SYSTEM = ("You edit thesis prose at the level of single sentences and paragraphs. Match the rhythm of the human examples (a dense, "
          "multi-clause sentence beside a blunt short one; a paragraph that stops on a fact) without reusing their words or topic. "
          "Also break predictable wording: replace stock academic phrasing (\"carries evidential weight\", \"sits at the intersection of\", "
          "\"not a neutral act\", \"resists reduction to\", \"not a failure of the paradigm\", \"in the sense that\") with direct, un-hedged "
          "technical language that names who did what. You may split a long cited sentence into two; the citation may stay in one half "
          "or appear in both. Never change a citation, a number, a quotation or a name, and never add one. When an instruction lists exact phrases to delete, "
          "none of them may appear in your rewrite, inflected or not.")


def _prompt(spans: list[dict], examples: list[dict]) -> str:
    ex = "\n".join(f"- ({e.get('kind', '').replace('_', ' ')}, {' + '.join(str(n) for n in e.get('lengths', []))} words) {lamp_bank.text_of(e)}"
                   for e in examples)
    items = "\n\n".join(f"[{i}] ({sp['kind']}) Edit: {sp['instruction']}\n\"\"\"\n{sp['text']}\n\"\"\"" for i, sp in enumerate(spans, 1))
    return (f"Human examples of the rhythm to aim for (imitate the length pattern only):\n{ex}\n\n"
            f"Rewrite each numbered span below as instructed. A sentence span comes back as one or two sentences, a paragraph span as "
            f"one paragraph with no blank line. Keep every citation, number, quotation and name exactly.\n\n{items}\n\n"
            f"Reply with ONLY a JSON array of {len(spans)} strings, one per span, same order.")


def _check(sp: dict, new, grams: set) -> str | None:
    """Why this replacement is rejected, or None."""
    if not isinstance(new, str):
        return "not text"
    new = new.strip()
    if not new:
        return None if "epilogue_closer" in sp["types"] and sp["kind"] == "sentence" else "empty"
    if re.search(r"\n\s*\n", new):
        return "more than one paragraph"
    bad = protect.compare(sp["text"], new)
    if bad:
        return bad[0]
    if sp["kind"] == "paragraph" and paragraph_violations(new) >= paragraph_violations(sp["text"]):
        return "still flat"          # not better than the original: it breaks as many rhythm rules as before
    if "long_sentence" in sp["types"] and any(len(x.split()) > MAX_SENT_WORDS for x in split_sentences(new)):
        return "still over the sentence limit"
    if passage_overlaps(new, grams):
        return "copies an example"
    if _STOCK_RX.search(new) and not _STOCK_RX.search(sp["text"]):
        return "adds a stock phrase"
    ratio = len(new.split()) / max(1, len(sp["text"].split()))
    if sp["kind"] == "paragraph" and not 0.6 <= ratio <= 1.5:
        return f"length changed x{ratio:.1f}"
    return None


def _claude_check(sp: dict, new: str, model) -> str | None:
    """A sentence span the local detector flagged: none of its exact Claude phrases may remain and the best score of the rewrite must be lower."""
    if model is None or "claude_style" not in sp["types"] or sp["kind"] != "sentence" or not new.strip():
        return None
    low = new.lower()
    if any(c.lower() in low for c in sp.get("culprits", []) if ">=" not in c and c in sp["text"]):
        return "kept a flagged phrase"
    rows = detector.score_text(new, model)
    if rows and max(r["ai_prob"] for r in rows) >= sp.get("ai_prob", 1.0):
        return "detector score not lower"
    return None


def _detector_stats(text: str, model) -> dict | None:
    rows = detector.score_text(text, model) if model is not None else None
    rows = [r for r in rows or [] if len(r["text"].split()) >= detector.MIN_WORDS]
    if not rows:
        return None
    return {"n": len(rows), "flagged": sum(r["flagged"] for r in rows), "mean_prob": round(sum(r["ai_prob"] for r in rows) / len(rows), 3)}


def lamp_rewrite_section(text: str, section_type: str, exemplars: list, spend=None, patched=(), max_spans: int = MAX_SPANS,
                         detector_model="auto") -> tuple[str, dict]:
    """-> (text, report). `exemplars` is lamp_bank.load(); `patched` the sentences the claim verifier rewrote. `detector_model` is the
    local Claude-style detector ("auto": data/claude_sentence_detector.json unless HARNESS_DETECTOR=off; None: off)."""
    if detector_model == "auto":
        detector_model = None if config.get("HARNESS_DETECTOR", "on").lower() == "off" else detector.load()
    before = _detector_stats(text, detector_model)
    found = detect_ai_spans(text, patched, detector.score_text(text, detector_model) if detector_model is not None else None)
    spans = found[:max_spans]
    kinds: dict = {}
    for sp in found:
        for t in sp["types"]:
            kinds[t] = kinds.get(t, 0) + 1
    rep = {"flagged": len(found), "sent": len(spans), "applied": 0, "rejected": [], "rejected_by": {}, "kinds": kinds, "edits": [],
           "sent_words": sum(len(sp["text"].split()) for sp in spans)}   # edits: (old, new) pairs; rejected_by: reason -> count
    if before:
        rep["detector"] = {"flagged_before": before["flagged"], "mean_prob_before": before["mean_prob"], "sentences": before["n"]}
    if not spans:
        return text, rep
    examples = lamp_bank.pick(exemplars, section_type, N_EXAMPLES)
    budget = min(9000, 300 + int(2.2 * sum(len(sp["text"].split()) for sp in spans)))
    data = parse_json(llm.ask("sonnet", SYSTEM, _prompt(spans, examples), max_tokens=budget, spend=spend, temperature=TEMPERATURE))
    if not isinstance(data, list) or len(data) != len(spans):
        rep["error"] = "reply did not parse into one string per span"
        return text, rep
    grams = exemplar_ngrams([lamp_bank.text_of(e) for e in examples])
    out = text
    for i, (sp, new) in enumerate(zip(spans, data), 1):
        why = _check(sp, new, grams) or (None if sp["text"] in out else "span not found") or _claude_check(sp, new, detector_model)
        if why:
            rep["rejected"].append({"span": i, "types": sp["types"], "why": why})
            key = re.sub(r"[:\d].*$", "", why).strip() or why
            rep["rejected_by"][key] = rep["rejected_by"].get(key, 0) + 1
            continue
        out = out.replace(sp["text"], new.strip(), 1)
        rep["edits"].append((sp["text"], new.strip()))
        rep["applied"] += 1
    out = re.sub(r"[ \t]+(?=\n|$)", "", out)
    after = _detector_stats(out, detector_model)
    if before and after:
        rep["detector"].update(flagged_after=after["flagged"], mean_prob_after=after["mean_prob"])
    return out, rep
