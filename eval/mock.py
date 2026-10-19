"""Mock run, no API calls, no money: push every saved draft through the harness's free repair steps and show what changed.
    python eval/mock.py
It replays saved drafts as the "model output", runs the real write_section graph with scripted fake models (so the wiring is
exercised end to end), and counts reader-visible defects (eval/defects.py) before and after. It predicts nothing about the judge."""
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import defects as D  # noqa: E402
from harness import graph as G, llm  # noqa: E402
from harness.text import cleanup  # noqa: E402


def split3(text: str) -> list[str]:
    ps = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    k = max(1, len(ps) // 3)
    return ["\n\n".join(ps[:k]) + "\n\n", "\n\n".join(ps[k:2 * k]) + "\n\n", "\n\n".join(ps[2 * k:])]


def run(text: str, ptype_title: str) -> tuple[str, dict]:
    """The real graph with a fake writer that returns `text` in three chunks and a fake cheap model that returns nothing usable."""
    sonnet = GenericFakeChatModel(messages=iter([AIMessage(content=c) for c in split3(text)]))
    haiku = GenericFakeChatModel(messages=iter([AIMessage(content="[]") for _ in range(6)]))
    llm.set_fake(lambda tier: sonnet if tier == "sonnet" else haiku)
    req = {"thesis": {"title": "Low-cost sensors for bridges", "domain": "Civil engineering", "methodology": "Review"},
           "chapter": {"chapter_number": 2, "title": ptype_title, "estimated_pages": 5},
           "section": {"number": "2.1", "title": "Sensor Technologies", "pages": 5.39}, "references": [], "ledger": {},
           "flags": {"pool": True, "sources": False, "revise": False}, "n_sections": 1}
    out = G.write_section(req, G.Deps(baseline=None))
    return out["text"], out["reports"].get("first_problems", {})


def main() -> None:
    llm.set_fake(None)
    rows = {}
    for arm in ("baseline", "harness"):
        before, after, flags = [], [], []
        for t in D._drafts(arm):
            try:
                new, first = run(t, "Literature Review")
            except Exception as e:  # noqa: BLE001
                print("run failed:", arm, type(e).__name__, e)
                continue
            before.append(D.defects(t))
            after.append(D.defects(new))
            flags.append(first)
        rows[arm] = (before, after, flags)
    keys = ["fragment_paras", "repeated_openers", "repeated_phrases", "stance_stock", "closer_paras", "report_verb_cites",
            "signpost_share", "gate_short_ratio", "gate_sent_cv"]
    print(f"{'defect per draft':22}" + "".join(f"{a + ' before':>18}{a + ' after':>16}" for a in rows))
    for k in keys:
        print(f"{k:22}" + "".join(f"{statistics.fmean(d[k] for d in b):18.2f}{statistics.fmean(d[k] for d in a):16.2f}" for b, a, _ in rows.values()))
    print("\nproblems the cheap patch would still be asked to fix (share of drafts with at least one):")
    for arm, (_, _, flags) in rows.items():
        n = max(1, len(flags))
        for k in ("citedpara", "repeat", "stance", "closer", "whatopen", "chains", "formulaic", "copied", "voice", "numbers", "examples",
                  "signposts"):
            share = sum(1 for f in flags if f.get(k)) / n
            print(f"  {arm:9} {k:10} {share:4.0%}")
    print(f"\nprompt rules: {len(G.VOICE_RULES.split())} words (was 670, then 374)")


if __name__ == "__main__":
    main()
