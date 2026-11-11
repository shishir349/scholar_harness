# CONSTRAINTS: quality gates for generated sections

Hard, non-negotiable. A section that breaks one is not done. The numbers live in `src/harness/limits.py` (one place); the independent
measurement is `src/harness/validate/constraints.py:section_violations`; `tests/test_production_leaks.py` pins both.
Source of the gates: the 3.1 Research Philosophy live test (GPTZero: 44 of 51 sentences AI).

## The gates

Body sections = `introduction`, `literature_review`, `methodology`, `findings`, `discussion`, `conclusion`.

| # | Gate | Limit |
|---|------|-------|
| 1 | **Zero micro-paragraphs** | 0 paragraphs with fewer than 3 sentences or fewer than 70 words |
| 2 | **Intra-paragraph cadence** | every paragraph of 3+ sentences: sentence-length std >= 6.8 words, longest sentence <= 42 words, at least one sentence of <= 11 words inside it (never a 1-2 sentence paragraph of its own) |
| 3 | **Narrative citation ceiling** | <= 2 "Author (Year) verb ..." sentences per call (open / mid / close), <= 5 per section; >= 70% of citations parenthetical, concept-first `(Author, Year; Author, Year)` |
| 4 | **Zero cross-chunk duplicate claims** | content-word Jaccard < 0.35 between any two sentences of the assembled section; no `(Author, Year)` summarised in a narrative sentence twice |
| 5 | **Golden tests untouched** | `src/harness/prompts/sections.py` and `tests/golden/cases.json` are byte-identical to `HEAD` |

## Rules that keep the gates honest

- **No silencing.** A gate is never relaxed, skipped or catch-and-ignored to make a section or a test pass. The repairs are on by default
  (`HARNESS_SHAPE`, `HARNESS_LAMP`, `HARNESS_SKELETONS` default `on`); fixtures switch them off only inside `tests/conftest.py`, and
  `tests/test_production_leaks.py` asserts the defaults.
- **One source of numbers.** Change a limit in `limits.py` only. Prompt text, validators, repairs and `section_violations` read it.
- **Repair order** for a draft that breaks a gate: the free deterministic repair (merge, split, drop, convert) -> the one `revise` patch ->
  the LAMP span rewrite. A rewrite that does not reduce the violations is rejected and the original kept.
- **Meaning is protected, not the wording.** `text/protect.py:compare` guards citations (author + year), numbers (a citation year counted
  once per citation), quotes, headings, tables. A repair may reorder or split a sentence; it may not change those.
- **A gate that cannot be met says so.** If a section still breaks a gate after all repairs, `reports` shows which, and the caller sees it.
  Do not add a fallback that hides it.
- **Measure live before claiming a gain.** `scripts/test_live_section.py` prints the gates for the text before the LAMP tail and the final
  text. Offline tests prove the code does what it says; only a live run shows the model obeys.
