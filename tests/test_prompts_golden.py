"""Ported prompts must equal Scholar's real builders byte for byte (tests/golden/cases.json,
regenerated with tests/golden/gen.mjs)."""
import json
from pathlib import Path

import pytest

from harness.prompts import sections as S

DATA = json.loads((Path(__file__).parent / "golden" / "cases.json").read_text(encoding="utf-8"))
CASES = DATA["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_calls_match_scholar(case):
    p = S.detect_profile(case["chapter"]["title"], case["chapter"]["chapter_number"])
    assert p["id"] == case["profileId"]
    used = S.extract_used_concepts(case["chunk1"] + " " + case["chunk2"])
    assert used == case["usedConcepts"]

    common = dict(thesis=case["thesis"], chapter=case["chapter"], section=case["section"], words_target=case["wordsTarget"])
    c1 = S.build_call1(**common, references=case["references"], profile_block_=S.profile_block(p),
                       noise_block_=S.noise_block(p), empirical_block_=S.empirical_block(p))
    c2 = S.build_call2(**common, chunk1=case["chunk1"], used_concepts=used, profile_note=S.call2_note(p))
    c3 = S.build_call3(**common, chunk1=case["chunk1"], chunk2=case["chunk2"], used_concepts=used,
                       profile_note=S.call3_note(p))
    assert c1 == case["call1"]
    assert c2 == case["call2"]
    assert c3 == case["call3"]


def test_title_classification_matches_scholar():
    unmatched = {"References", "Acknowledgements", "Background of the Study", "Summary", "Abstract"}
    for title, js_id in DATA["titles"]:
        assert S.detect_profile(title, 99)["id"] == js_id, title
        got = S.classify_title(title)
        if title in unmatched:
            assert got is None, title
        else:
            assert got == js_id, title


def test_optional_blocks_appear_only_when_given():
    case = CASES[1]
    p = S.detect_profile(case["chapter"]["title"], 2)
    base = dict(thesis=case["thesis"], chapter=case["chapter"], section=case["section"], words_target=450)
    plain = S.build_call1(**base, references=case["references"])
    rich = S.build_call1(**base, references=case["references"], exemplar_block="EXEMPLAR-X",
                         source_notes_block="NOTES-X", citation_rules="RULES-X", feedback="FEEDBACK-X")
    for marker in ("EXEMPLAR-X", "NOTES-X", "RULES-X", "FEEDBACK-X"):
        assert marker not in plain and marker in rich
    assert rich.index("EXEMPLAR-X") < rich.index("YOUR TASK, OPENING PASSAGE")
    assert rich.index("NOTES-X") > rich.index("REFERENCES:")
    for fn in (S.build_call2, S.build_call3):
        kw = dict(**base, chunk1="a", used_concepts="none yet", exemplar_block="EX", references_block="RB",
                  source_notes_block="SN", citation_rules="CR", feedback="FB")
        if fn is S.build_call3:
            kw["chunk2"] = "b"
        out = fn(**kw)
        assert all(m in out for m in ("EX", "RB", "SN", "CR", "FB"))
