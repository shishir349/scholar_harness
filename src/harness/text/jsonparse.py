"""Tolerant JSON extraction from model output (fences, preamble, trailing chatter)."""
import json
import re


def _try(t: str, opener: str, closer: str):
    start = t.find(opener)
    if start == -1:
        return None
    end = t.rfind(closer)
    # widest span first, then shrink to the previous closer until something parses
    while end > start:
        try:
            return json.loads(t[start:end + 1])
        except ValueError:
            end = t.rfind(closer, start, end)
    return None


def parse_json(text: str | None):
    """First JSON object/array found in `text` (whichever opens first), or None."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda p: t.find(p[0]) if t.find(p[0]) != -1 else len(t) + 1)
    for opener, closer in pairs:
        got = _try(t, opener, closer)
        if got is not None:
            return got
    return None
