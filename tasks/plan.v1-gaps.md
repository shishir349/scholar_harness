# Plan: close the measured gaps between real theses and harness drafts

No SPEC.md exists. The spec is `eval/out/gap_report.md` (50 features, real theses vs saved drafts) and the 12 gaps listed by the user.

## Overview
Eleven features still separate the harness drafts from 1,057 real-thesis samples almost perfectly. Each gap gets the cheapest lever
that works without inventing facts: a mined number in the prompt, a mined word list, a source-note change, or a cheap Haiku patch.
No change may break citation checks (`verify` loop), cost (target <= 2x old app) or the 217 existing tests.

## Gap -> lever map
| Gap (real vs harness) | Lever | Task |
|---|---|---|
| Numbers 34 vs 0.7, names 31 vs 9, acronyms 12 vs 5 | source notes keep number/name sentences; "use exact figures and names from the notes, never invent"; define acronyms once | T4 |
| Connectives 6.0 vs 1.5, "that" 10.7 vs 24.9, "of" 38 vs 21 | mined per-section targets in the prompt block + one cheap patch | T2, T5 |
| Word length 5.3 vs 6.0, vocabulary variety 0.40 vs 0.50, boosters 0.2 vs 1.4 | mined long-word/booster avoid list; "repeat the term, do not swap synonyms" | T3 |
| Short sentences 7% vs 23% | rule already corrected; cheap patch when > 2x real | T5 |
| Passive 9.2 vs 4.3 | per-section passive target in the block (methods/results wording) | T2 |
| "the field/literature" 0.4 vs 3.6 | meta-phrase patch | T5 |
| Figure/table refs 2.4 vs 0.2 | NOT built: pointing at a figure that does not exist is a made-up claim | skipped |

## Dependency graph
```
T1 feature module (gapfeats)  ->  T2 mined targets + prompt lines  ->  T5 patches (use the targets)
                              ->  T3 mined word lists
T4 source notes (independent)
T6 probe checks + live probe  <- all
```

## Phases and checkpoints
**Phase A foundation:** T1, T2.  Checkpoint A: full suite green; profile rebuilds; block under ~700 tokens.
**Phase B levers:** T3, T4, T5.  Checkpoint B: suite green; offline prompt size check; cost estimate.
**Phase C proof:** T6 (probe + report). Checkpoint C: live probe, compare gap features, report honestly.

## Risks
| Risk | Mitigation |
|---|---|
| Numbers invented to close the gap | notes carry only digits already in the abstract; prompt says never invent; claim loop deletes unsupported sentences |
| Patches stack and cost grows | all patches go through the one existing Haiku `_patch` call, max 2 paragraphs per kind |
| Over-fitting to 40 saved drafts | targets come from 1,057 real samples; word lists need >= 5 fixtures |
| Prompt block grows | cached; check size in T2 and T3 |

## Open questions
None blocking. User approved scope ("build 1 to 4, do anything to reduce the gaps").
