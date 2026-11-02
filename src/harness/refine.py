"""Long-text refinement: split a document into paragraph chunks, rewrite each chunk once, keep the rewrite only if the integrity
gate (text/protect.py) passes. A failed or errored chunk is retried at most `retries` times, then the original chunk stays.
No word limit: length is handled by chunking, and the model sees the previous chunk's last sentence and the openings used so far."""
import re

from . import llm
from .spend import BudgetExceeded
from .text import cleanup, protect

SYSTEM = (
    "You revise passages of a doctoral thesis so they read like a careful human researcher wrote them. "
    "Rewrite only the wording and the sentence structure.\n"
    "KEEP EXACTLY: every citation (author names and year), every number, date, unit and percentage, every quotation word for word, "
    "every technical term, acronym and proper name.\n"
    "DO NOT: add a fact or a source; strengthen or weaken a claim; remove a limitation or a hedge (may, suggests, appears to); "
    "merge, split, add or reorder paragraphs; add a heading or any comment.\n"
    "DO: vary sentence length widely (some under 10 words beside some over 35); vary how sentences and paragraphs open; prefer plain "
    "exact words to impressive ones; use plain connectives (so, but, because, however) now and then, and the passive where the doer "
    "does not matter; keep the author's position and the order of ideas.\n"
    "AVOID: stock phrases (\"it is worth noting\", \"in accordance with\", \"it is reasonable to consider\", \"an examination of the "
    "sources reveals\", \"plays a crucial role\"), em dashes, announcing a claim (\"I argue that\"), and ending a paragraph on a verdict.\n"
    "Reply with ONLY the rewritten paragraphs, separated by one blank line, the same number of paragraphs as given."
)
_NOT_PROSE = re.compile(r"^\s*(?:#|\[(?:FIGURE|TABLE)_START\])")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _prose(p: str) -> bool:
    return not _NOT_PROSE.match(p) and len(p.split()) >= 8


def chunk_paragraphs(text: str, max_words: int = 350) -> list[list[str]]:
    """Paragraph groups of about `max_words`. A heading stays with the paragraph after it; a paragraph is never split."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    units, i = [], 0
    while i < len(paras):
        unit = [paras[i]]
        while paras[i].startswith("#") and i + 1 < len(paras):
            i += 1
            unit.append(paras[i])
        units.append(unit)
        i += 1
    chunks, cur, n = [], [], 0
    for u in units:
        w = sum(len(p.split()) for p in u)
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = [], 0
        cur += u
        n += w
    if cur:
        chunks.append(cur)
    return chunks


def _user(prose: list[str], tail: str, openings: list[str], feedback: list[str] | None) -> str:
    parts = []
    if tail:
        parts.append(f"The passage before this one ended with: \"{tail}\" (for flow only; do not rewrite it).")
    if openings:
        parts.append("Paragraph openings already used, do not repeat them: " + " | ".join(openings[-12:]))
    if feedback:
        parts.append("Your last attempt was rejected because: " + "; ".join(feedback) + ". Fix exactly this and keep everything else.")
    parts.append(f"Rewrite these {len(prose)} paragraph(s):\n\n" + "\n\n".join(prose))
    return "\n\n".join(parts)


def _rewrite(chunk: list[str], tail: str, openings: list[str], *, spend, tier: str, retries: int) -> tuple[list[str], dict]:
    idx = [i for i, p in enumerate(chunk) if _prose(p)]
    info = {"status": "skipped", "retried": False, "failures": []}
    if not idx:
        return chunk, info
    prose = [chunk[i] for i in idx]
    before = "\n\n".join(chunk)
    feedback = None
    for attempt in range(retries + 1):
        info["retried"] = attempt > 0
        try:
            reply = llm.ask(tier, SYSTEM, _user(prose, tail, openings, feedback), max_tokens=max(1024, int(sum(len(p.split()) for p in prose) * 2.2) + 200), spend=spend)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - one bad call must not lose the chunk
            info.update(status="error", failures=[f"{type(e).__name__}: {e}"[:160]])
            return chunk, info
        got = [p.strip() for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
        new = list(chunk)
        if len(got) == len(idx):
            for i, p in zip(idx, got):
                new[i] = p
        after = "\n\n".join(new)
        failures = protect.compare(before, after) if len(got) == len(idx) else [f"paragraph count changed: {len(idx)} -> {len(got)}"]
        if not failures:
            info.update(status="unchanged" if after == before else "rewritten", failures=[], soft=protect.soft(before, after))
            return new, info
        info["failures"] = failures
        feedback = failures
    info["status"] = "kept_original"
    return chunk, info


def refine(text: str, *, spend=None, tier: str = "sonnet", max_words: int = 350, retries: int = 1, polish: bool = True) -> dict:
    """-> {text, changed, chunks:[{index, words, status, retried, failures}]}. Never returns less protected content than it got."""
    out_paras, report, openings, tail = [], [], [], ""
    for n, chunk in enumerate(chunk_paragraphs(text, max_words)):
        new, info = _rewrite(chunk, tail, openings, spend=spend, tier=tier, retries=retries)
        report.append({"index": n, "words": sum(len(p.split()) for p in chunk), **info})
        out_paras += new
        openings += [" ".join(p.split()[:5]) for p in new if _prose(p)]
        last = next((p for p in reversed(new) if _prose(p)), "")
        tail = (_SENT_END.split(last.strip()) or [""])[-1][:200]
    result = "\n\n".join(out_paras)
    original = "\n\n".join(p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip())
    return {"text": result if report else (text or ""), "changed": result != original, "chunks": report}
