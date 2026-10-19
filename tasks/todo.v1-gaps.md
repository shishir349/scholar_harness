# Todo

## Phase A: foundation
- [x] T1 Move the 50-feature extractor into the package (`validate/gapfeats.py`); `eval/gap_report.py` uses it.
  - AC: same numbers as before on a fixed text; unit tests on 6 features. Verify: `pytest tests/test_gapfeats.py`.
- [x] T2 Miner writes `targets` (real p10/p50/p90 per section type for 14 features); profile block prints them as plain instructions.
  - AC: profile has targets; block mentions connectives, "that", "of", boosters, passive, numbers, word length; <= 700 tokens.
  - Verify: `pytest tests/test_profile.py`; rebuild `python eval/phrase_gap.py`.
- [x] Checkpoint A

## Phase B: levers
- [x] T3 Mined long-word and booster avoid words (len >= 9 or booster list; ratio >= 4; >= 5 fixtures) added to the profile and block.
- [x] T4 Source notes keep number/name-bearing sentences first; prompt lines for exact figures, acronym definition, repeat-the-term.
  - AC: a 600-char abstract with one numeric sentence puts that sentence in the note; no digits invented.
- [x] T5 Patches (one Haiku call, max 2 paragraphs each): plain connectives and fewer "that"; sentences under ten words; "the field/literature" as a character.
  - AC: each fires only past a threshold; none escalates to a Sonnet rewrite.
- [x] Checkpoint B

## Phase C: proof
- [x] T6 Probe checks for the new features; one live probe (~$0.14); gap report on the new draft.
- [x] Checkpoint C
