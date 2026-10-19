# Plan: real-thesis gaps, round 2 (spec: ../SPEC.md)

## Dependency graph
```
cleanup.py free rewrites ──┬─> graph.validate wiring ──> final polish in write_section ──> mock + defects
numbers.py (facts, shares) ─┤                                  ^
tools.py (note cap, kinds) ─┘                                  |
graph._concrete_rules (plan quotas) ───────────────────────────┘
patch reasons (numbers, examples, first person, signposts) depend on numbers.py shares
```
Free rewrites first (no model, no cost, testable alone), then the prompt plan, then the patch fall-back for what free cannot fix.

## Slices (each leaves the suite green)
| # | Slice | Done when |
|---|---|---|
| S1 | `cleanup.polish`: strip_stance, what_cleft, report_to_cite (paren, cap 2 "According to"), drop_closers, drop_boosters, thin_that | unit tests; idempotent; keeps figures/citations |
| S2 | `numbers.py`: structure/brief facts, shares (figure, first person, example), targets per section | unit tests on 4 section types |
| S3 | Plan quotas in `_concrete_rules`: number facts (typed, placed start/mid), first-person quota + verbs, example paragraph, tighter no-cite quota; moves reworded (no "What"); note cap in tools.py | tests: plan text contains fact/verbs/example, never an invented number |
| S4 | validate wiring: polish after drop_repeats; new problems `numbers`, `examples`, `fp`, widened `signposts`/`whatopen`; patch reasons; route; prune priority | fake-model run shows keys reach the patch |
| S5 | final polish in `write_section`; defects.py gate columns + shared regexes; mock.py prints new rows | mock before/after table |
| S6 | Review pass (five axes) + edge-case tests + README note | review notes in todo.md; full offline suite green |

## Checkpoints
- After S1: run test_polish only. After S3: golden prompts still byte-identical. After S5: full suite + mock. After S6: report.

## Risks
| Risk | Mitigation |
|---|---|
| Free rewrite breaks grammar | rewrite only on narrow patterns, tests per pattern, idempotent, unchanged on a miss |
| Numbers pushed where not needed | quota by section share, facts only from allowed origins, patch fires only below 70% of target and never above |
| More patch paragraphs overflow the reply | `prune_flagged` limit stays 8; new reasons rank below HARD |
| First person turns into "I argue" | verbs listed, stance stripped free |
| Judge effect unknown | disclosed; probe waits for the user |
