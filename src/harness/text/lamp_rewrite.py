"""LAMP span-level rewrite (Chakrabarty et al. 2025): edit only the spans a detector flagged, not whole chunks.

One batched Sonnet call at temperature 0.85 gets the numbered spans, a few real human sentence pairs for rhythm, and one edit
instruction per span. Every returned span is then checked on the spot; a span that fails keeps its original text, so one bad
edit never costs the section. Runs after claim verification: protect.compare guarantees no citation, number or quote moves."""
import re

from .. import llm
from ..corpus import lamp_bank
from ..validate.lamp_detector import detect_ai_spans, paragraph_is_flat
from ..validate.features import _STOCK_RX
from ..validate.overlap import exemplar_ngrams, passage_overlaps
from . import protect
from .jsonparse import parse_json

MAX_SPANS = 6
TEMPERATURE = 0.85
N_EXAMPLES = 8
SYSTEM = ("You edit thesis prose at the level of single sentences and paragraphs. Match the rhythm of the human examples (a dense, "
          "multi-clause sentence beside a blunt short one; a paragraph that stops on a fact) without reusing their words or topic. "
          "Never change a citation, a number, a quotation or a name, and never add one.")


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
    if sp["kind"] == "paragraph" and paragraph_is_flat(new):
        return "still flat"
    if passage_overlaps(new, grams):
        return "copies an example"
    if _STOCK_RX.search(new) and not _STOCK_RX.search(sp["text"]):
        return "adds a stock phrase"
    ratio = len(new.split()) / max(1, len(sp["text"].split()))
    if sp["kind"] == "paragraph" and not 0.6 <= ratio <= 1.5:
        return f"length changed x{ratio:.1f}"
    return None


def lamp_rewrite_section(text: str, section_type: str, exemplars: list, spend=None, patched=(), max_spans: int = MAX_SPANS) -> tuple[str, dict]:
    """-> (text, report). `exemplars` is lamp_bank.load(); `patched` the sentences the claim verifier rewrote."""
    found = detect_ai_spans(text, patched)
    spans = found[:max_spans]
    kinds: dict = {}
    for sp in found:
        for t in sp["types"]:
            kinds[t] = kinds.get(t, 0) + 1
    rep = {"flagged": len(found), "sent": len(spans), "applied": 0, "rejected": [], "kinds": kinds, "edits": []}   # edits: (old, new) pairs
    if not spans:
        return text, rep
    examples = lamp_bank.pick(exemplars, section_type, N_EXAMPLES)
    budget = min(4000, 300 + int(2 * sum(len(sp["text"].split()) for sp in spans)))
    data = parse_json(llm.ask("sonnet", SYSTEM, _prompt(spans, examples), max_tokens=budget, spend=spend, temperature=TEMPERATURE))
    if not isinstance(data, list) or len(data) != len(spans):
        rep["error"] = "reply did not parse into one string per span"
        return text, rep
    grams = exemplar_ngrams([lamp_bank.text_of(e) for e in examples])
    out = text
    for i, (sp, new) in enumerate(zip(spans, data), 1):
        why = _check(sp, new, grams) or (None if sp["text"] in out else "span not found")
        if why:
            rep["rejected"].append({"span": i, "types": sp["types"], "why": why})
            continue
        out = out.replace(sp["text"], new.strip(), 1)
        rep["edits"].append((sp["text"], new.strip()))
        rep["applied"] += 1
    return re.sub(r"[ \t]+(?=\n|$)", "", out), rep
