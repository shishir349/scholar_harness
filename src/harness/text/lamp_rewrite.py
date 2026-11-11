"""LAMP span-level rewrite (Chakrabarty et al. 2025): edit only the spans a detector flagged, not whole chunks.

One batched Sonnet call at temperature 0.85 gets the numbered spans, a few real human sentence pairs for rhythm, and one edit
instruction per span. Every returned span is then checked on the spot; a span that fails keeps its original text, so one bad
edit never costs the section. Runs after claim verification: protect.compare guarantees no citation, number or quote moves."""
import re

from .. import config, llm
from ..corpus import lamp_bank
from ..validate import surprisal
from .. import limits
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
          "or appear in both. Never change a citation, a number, a quotation or a name, and never add one.")


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


def _gain(sp: dict, new: str, scorer, low: float, window: float) -> str | None:
    """Local check for a sentence span flagged as predictable: the rewrite's unmasked tokens must read less predictable (mean up by
    SURPRISAL_GAIN) and its most predictable 6-token stretch must not get worse. Only the old and the new sentence are scored."""
    if "low_surprisal" not in sp["types"] or sp["kind"] != "sentence" or not new.strip():
        return None
    old, cur = surprisal.score_section_surprisal(sp["text"], scorer, low, window), surprisal.score_section_surprisal(new, scorer, low, window)
    if old is None or cur is None:
        return None
    o, c = old["sentences"][0], cur["sentences"][0]
    return None if c["mean"] >= o["mean"] + limits.SURPRISAL_GAIN and c["min_window"] >= o["min_window"] else "surprisal not higher"


def lamp_rewrite_section(text: str, section_type: str, exemplars: list, spend=None, patched=(), max_spans: int = MAX_SPANS,
                         scorer="auto", low: float | None = None, rounds: int = 2, window: float | None = None) -> tuple[str, dict]:
    """-> (text, report). `exemplars` is lamp_bank.load(); `patched` the sentences the claim verifier rewrote.
    `scorer` (default: the shared local surprisal scorer, None = off) adds predictable sentences to the spans and rejects a rewrite
    of one that does not read less predictable; round 2 retries only the spans rejected for that."""
    if scorer == "auto":     # opt-in: HARNESS_SURPRISAL=on. Offline benchmark (scripts/offline_benchmark.py): flagged AI text is LESS predictable than human thesis text
        scorer = surprisal.get_scorer() if config.get("HARNESS_SURPRISAL", "off").lower() == "on" else None
    before = surprisal.score_section_surprisal(text, scorer, low, window) if scorer is not None else None
    found = detect_ai_spans(text, patched, before)
    spans = found[:max_spans]
    kinds: dict = {}
    for sp in found:
        for t in sp["types"]:
            kinds[t] = kinds.get(t, 0) + 1
    rep = {"flagged": len(found), "sent": len(spans), "applied": 0, "rejected": [], "rejected_by": {}, "kinds": kinds, "edits": [],
           "sent_words": sum(len(sp["text"].split()) for sp in spans)}   # edits: (old, new) pairs; rejected_by: reason -> count
    if before is not None:
        rep["surprisal_before"] = surprisal.summary(before)
    if not spans:
        return text, rep
    examples = lamp_bank.pick(exemplars, section_type, N_EXAMPLES)
    grams = exemplar_ngrams([lamp_bank.text_of(e) for e in examples])
    out = text
    for rnd in range(1, max(1, rounds if scorer is not None else 1) + 1):
        budget = min(9000, 300 + int(2.2 * sum(len(sp["text"].split()) for sp in spans)))
        data = parse_json(llm.ask("sonnet", SYSTEM, _prompt(spans, examples), max_tokens=budget, spend=spend, temperature=TEMPERATURE))
        rep["rounds"] = rnd
        if not isinstance(data, list) or len(data) != len(spans):
            rep["error"] = "reply did not parse into one string per span"
            for k, sp in enumerate(spans, 1):          # spans deferred from round 1 keep their rejection on record
                if rnd > 1:
                    rep["rejected"].append({"span": k, "types": sp["types"], "why": "surprisal not higher"})
                    rep["rejected_by"]["surprisal not higher"] = rep["rejected_by"].get("surprisal not higher", 0) + 1
            break
        retry = []
        for i, (sp, new) in enumerate(zip(spans, data), 1):
            why = _check(sp, new, grams) or (None if sp["text"] in out else "span not found") or (_gain(sp, new, scorer, before and before['threshold']['mean'], before and before['threshold']['window']) if scorer else None)
            if why:
                if why == "surprisal not higher":
                    retry.append({**sp, "instruction": sp["instruction"] + " Your previous attempt was still predictable: change the vocabulary, not just the order."})
                    if rnd < rounds:
                        continue                   # not final yet: counted only if round 2 also fails
                rep["rejected"].append({"span": i, "types": sp["types"], "why": why})
                key = re.sub(r"[:\d].*$", "", why).strip() or why
                rep["rejected_by"][key] = rep["rejected_by"].get(key, 0) + 1
                continue
            out = out.replace(sp["text"], new.strip(), 1)
            rep["edits"].append((sp["text"], new.strip()))
            rep["applied"] += 1
        spans = retry if rnd < rounds else []
        if not spans:
            break
    out = re.sub(r"[ \t]+(?=\n|$)", "", out)
    after = surprisal.score_section_surprisal(out, scorer, low, window) if scorer is not None else None       # paragraph cache: only the edited paragraphs go through the model
    if after is not None:
        rep["surprisal_after"] = surprisal.summary(after)
    return out, rep
