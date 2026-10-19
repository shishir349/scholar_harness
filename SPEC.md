# SPEC: close the real-thesis gaps, round 2 (tier 1 + tier 2)

Status: built in one pass on the user's explicit request; no live API call is made to prove it (see Boundaries).

## 1. Objective
Move harness drafts toward real-thesis numbers on the metrics below without inventing facts, without raising cost, and
without touching prod Scholar prompts. Users: thesis writers (readers of the draft) and the G3 judge.
Success = each metric moves toward REAL in the free mock (`eval/mock.py`, `eval/defects.py`) and in unit tests. The judge
effect stays unproven until the user approves `eval/probe.py`.

### Targets (REAL = real theses, H = harness before this spec)
| # | Metric | REAL | H now | Target | Lever |
|---|---|---|---|---|---|
| 1 | paragraphs with a citation | 27% (lit review 47) | 65% | <= REAL+6 pts | tighter quota, fewer notes, patch |
| 2a | source-summary lines / draft | 0.27 | 1.37 | <= 0.5 | free rewrite to a parenthetical / "According to" |
| 2b | "I argue" lines / draft | 0.13 | 1.26 | <= 0.3 | free strip of the announcement |
| 2c | "What..." paragraph openers | 2% | 14% | <= 4% | free "What X is that Y" rewrite + patch |
| 3 | paragraphs with a number | 52% (loose) | 8% | >= 35%, never a made-up number | typed facts + placement + patch |
| 4a | plain signposts / sentence | 5.3% | ~0% | >= 4% | plan + free "According to" + patch |
| 4b | examples / 1000 words | 1.97 | 0.55 | >= 1.5 | example paragraph per passage + patch |
| 5 | paragraphs with first person | 24% (intro 27, lit 7, method 36, disc 29) | 8% | 20-22% overall, per section | quota + verbs + patch |
| 6a | short sentences / draft | 10.1 | 4.4 | **no change** (see Findings) | - |
| 6b | sentence variety (CV) | 0.62 | 0.48 | **no change** (see Findings) | - |
| 7a | verdict-closer paragraphs | 0.10 | 0.16 | <= 0.10 | free drop + patch |
| 7b | narrative-cite share | 0.33 mean / 0.57 median | 0.77-0.79 | 0.5 or lower | report rewrite goes to parentheses |
| 7c | AI tells / 1000 words | 0.5 | 1.7 | <= 1.0 | boosters/adverbs dropped free, seeds |
| 8 | gap features in range | 12 of 12 | 7-8 | >= 10 | digits, first person, signposts, boosters, "that" |

### Findings that change the brief (data, not opinion)
1. **Short sentences / CV**: `eval/defects.py` counts real chunks that hold captions, lists and cut-off pieces (8 fragments
   per real draft), so real 10.1 and 0.62 are inflated. On the gate's own measure (`style_features`, `gap_report`) harness
   has MORE short sentences than real (share 0.08 to 0.23 vs real 0.04 to 0.07) and CV 0.48 is inside the real p10-p90 band
   (0.35-0.84). Raising short sentences would move away from real. Decision: no push; add gate-grade columns to defects.py.
2. **Numbers**: 19% of real numbered sentences point at thesis structure (Chapter 3, Table 2, Figure 1, RQ2, Phase 1), only
   4-13% of numbered sentences also carry a citation, the first number sits in the opening sentence 50% of the time, and
   numbered paragraphs run 15-20 words longer. Real numbers mostly come from the author's own data, which the harness does
   not have, so 35% (not 52%) is the honest ceiling.
3. **First person** differs by section (lit review only 7%), so one 22% flat target would hurt the lit review.
4. **Narrative share**: the tier-2 ask (lower it) is right; the earlier scoreboard row that showed H below R was wrong.

## 2. Commands
```
pytest -q                                   # offline suite (free)
python eval/defects.py                      # free: real vs saved drafts
python eval/mock.py                         # free: replay saved drafts through the real graph
python eval/probe.py --fixture fx06         # PAID, only when the user says go
```

## 3. Project structure (touch list)
```
src/harness/text/cleanup.py      + polish(): strip_stance, what_cleft, report_to_cite, drop_closers, drop_boosters, thin_that
src/harness/text/numbers.py      NEW  structure_facts, brief_facts, figure_share, first_person_share, example_count, targets
src/harness/sources/tools.py     NOTE cap, facts carry kind
src/harness/graph.py             plan quotas, moves reworded, validate wiring, new patch reasons, final polish
eval/defects.py                  gate-grade columns, strict examples regex shared
tests/test_polish.py, tests/test_numbers.py   NEW
```

## 4. Code style
Match the surrounding file: short docstrings that say why, module constants in CAPS with the measured number in a comment,
regexes at module top, no new dependency, no class where a function works. Every new free rewrite returns the text unchanged
when its pattern is not found.

## 5. Testing strategy
- Unit test per rewrite: hit, miss, edge (et al., two authors, no "that", heading lines, 1-2 sentence paragraphs, empty chunk,
  `\r\n`, text with figures, name particles).
- Property: every polish step is idempotent and never removes a figure or a citation that was not the thing being rewritten.
- Wiring: a fake-model run of `write_section` proves new problem keys reach the patch and that totals fall.
- Existing 259 tests keep passing; golden prompt tests stay byte-identical (changes live on the harness path only).
- Live probe: not run.

## 6. Boundaries
Always: harness path only (`pool: True`); keep every citation unless it is the thing being rewritten; numbers come only from
source notes, the thesis brief, the chapter/section/objective/question numbers, or arithmetic shown in an uncited example.
Ask first: any paid call (probe, judge); changing Scholar prod prompts; changing the gate.
Never: invent a source, figure, table, quote or effect size; point to a Table or Figure that the request does not supply;
add a number just to hit the share; commit without being asked.
