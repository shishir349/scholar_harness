# Todo: round 2

- [x] S1 cleanup.polish (free rewrites)  AC: tests/test_polish.py green; idempotent; figures + citations kept
- [x] S2 text/numbers.py                 AC: tests/test_numbers.py green
- [x] S3 plan quotas + moves + note cap  AC: tests/test_concrete.py extended; golden prompts unchanged
- [x] S4 validate/route/patch wiring     AC: fake-model graph test shows new keys flagged then fixed
- [x] S5 final polish, defects/mock cols AC: `python eval/mock.py` prints before/after rows
- [x] S6 review + edge cases + README    AC: full offline suite green; review notes below

## Review notes
Independent review (15 findings) fixed: What-cleft no longer drops a citation/figure; CLOSER word boundaries; appendix not polished;
boosters keep adjective uses; thin_that idempotent and safer; fact_used whole-number match; page numbers not figures; name-only facts
excluded from number targets; plan and patch quotas no longer pick the same paragraph; string questions guarded; signpost list guarded.
Left as is (low): 'In this chapter, I argue that X' becomes 'In this chapter, X'. Single 'What' opener left to the free rewrite (no extra patch call).
Not proven: judge effect (needs probe, waits for the user).
