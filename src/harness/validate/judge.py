"""Pairwise style judge: does the draft read like a human researcher next to real thesis prose?
Used by evaluation only. Instructions go in the system message; both A/B orders are run to cancel position bias."""
from .. import llm
from ..text.jsonparse import parse_json

SYSTEM = (
    "You compare two passages, A and B, taken from doctoral theses. Exactly one may have been written by an AI model. "
    "Decide which passage reads as written by a human researcher. Base the decision on concrete cues: sentence rhythm "
    "and variation, hedging that fits the evidence, commitment to a position, specific detail versus generic phrasing, "
    "and natural (not mechanical) transitions. Ignore topic, correctness and length. "
    'Reply with ONLY JSON: {"human": "A" or "B", "cues": ["short cue", ...]}.'
)


SYSTEM_H2H = (
    "You compare two passages, A and B, written for the same section of a doctoral thesis. Either may have been "
    "produced with AI help. Decide which reads more like natural writing by a human researcher, judging concrete "
    "cues: sentence rhythm and variation, hedging that fits the evidence, commitment to a position, specific detail "
    "versus generic phrasing, and natural (not mechanical) transitions. Ignore topic, correctness, length and "
    'formatting. If neither is clearly more human, answer TIE. Reply with ONLY JSON: {"human": "A" or "B" or "TIE"}.'
)


# Same as SYSTEM_H2H but no TIE: two good AI drafts get TIE almost every time, which pins the score at 0.5.
# Both A/B orders and repeats still cancel position bias; this only makes the judge sensitive, not generous.
SYSTEM_H2H_FORCED = SYSTEM_H2H.replace(
    "If neither is clearly more human, answer TIE. ", "Even if the difference is small, you must pick one. ").replace(
    ' or "TIE"}', '}')


def _once(first: str, second: str, section_type: str, spend, system: str = SYSTEM, tier: str = "haiku") -> str | None:
    user = f"Section type: {section_type}\n\n=== PASSAGE A ===\n{first}\n\n=== PASSAGE B ===\n{second}"
    data = parse_json(llm.ask(tier, system, user, max_tokens=100 if system in (SYSTEM_H2H, SYSTEM_H2H_FORCED) else 400, spend=spend))
    v = data.get("human") if isinstance(data, dict) else None
    return v if v in ("A", "B", "TIE") else None


def judge_h2h(draft: str, other: str, section_type: str, spend=None, repeats: int = 3, tier: str = "haiku",
              forced: bool = False) -> dict:
    """Head-to-head of two AI-produced drafts. Each repeat asks in both orders; a vote is 1 for `draft`, 0 for
    `other`, 0.5 for TIE or an unusable reply. {result: win|tie|lose, score 0..1}. Repeats average out the
    randomness of a single call; the bands leave a wide tie zone so noise is not read as a difference.
    forced=True removes the TIE option (pick-one mode); report it next to the normal score, never instead of it."""
    system = SYSTEM_H2H_FORCED if forced else SYSTEM_H2H
    votes = []
    for _ in range(repeats):
        a = _once(draft, other, section_type, spend, system, tier)   # draft is A
        b = _once(other, draft, section_type, spend, system, tier)   # draft is B
        votes.append(1.0 if a == "A" else 0.0 if a == "B" else 0.5)
        votes.append(1.0 if b == "B" else 0.0 if b == "A" else 0.5)
    score = sum(votes) / len(votes)
    return {"result": "win" if score > 0.6 else "lose" if score < 0.4 else "tie", "score": round(score, 3)}


def judge_pair(draft: str, reference: str, section_type: str, spend=None) -> dict:
    """{result: win|lose|tie, cues}. 'win' only if the draft is picked as human in BOTH orders."""
    a = _once(draft, reference, section_type, spend)       # draft is A
    b = _once(reference, draft, section_type, spend)       # draft is B
    picked = [a == "A" if a else None, b == "B" if b else None]
    if picked == [True, True]:
        result = "win"
    elif picked == [False, False]:
        result = "lose"
    else:
        result = "tie"
    return {"result": result, "picks": {"draft_first": a, "draft_second": b}}
