# scholar-harness

Reference corpus, agentic sources and validators for Acdyon Scholar. Separate from `acdyon-scholar`
(read-only reference; nothing here touches its code, prod Supabase or Render).

Python 3.12, LangGraph 1.2.12, langchain-anthropic 1.7.2, langchain-core 1.6.3 (pinned in `pyproject.toml`).
Prompts ported from `acdyon-scholar` @ `2ec345557486d64681fdcfe71d0b68f0552565a1`
(golden-tested: `tests/golden/cases.json`, regenerate with `node tests/golden/gen.mjs` if Scholar's prompts change).

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[corpus,dev]"     # corpus extra = pymupdf4llm (AGPL, corpus build only)
copy .env.example .env                                     # then fill ANTHROPIC_API_KEY and OPENALEX_API_KEY
.venv\Scripts\python -m pytest                             # offline, no keys needed
```
OpenAlex needs a free key: https://openalex.org/settings/api . Use a separate Anthropic key with a spend cap.

## Build the corpus (live keys; each stage is resumable)
```
python -m harness.corpus.smoke        # 0. confirms OpenAlex params/shapes (fix here first if it fails)
python -m harness.corpus.topics       # 1. top-20 fields -> data/fields.selected.json   [review: gate G1]
python -m harness.corpus.candidates   # 2. candidates + quality signals per field
python -m harness.corpus.acquire      # 3. download, parse, gate -> data/raw/<field>/<thesis>/
python -m harness.corpus.score        # 4. rubric + score, 5 exemplars + 1 holdout per field -> data/manifest.json
                                      #    [review manifest, set "exclude": true, rerun with --reselect: gate G2]
python -m harness.corpus.ingest       # 5. data/corpus.db (FTS5) + data/style_baseline.json
python eval/ab.py                     # 6. A/B report -> eval/out/report.md   [gate G3]
python -m harness.corpus.skeletons   # 7. data/human_skeletons.json: paragraph/sentence-shape blueprints (no text) that graph._para_plan hands the writer
```
`data/raw/`, `data/candidates/`, `data/corpus.db` are gitignored (copyrighted text). Commit `fields.selected.json`,
`manifest.json`, `style_baseline.json`.

## CLI contract (what Scholar's Node backend will spawn later)
`python -m harness.run research | write-section | audit` reads one JSON object on stdin and writes NDJSON events on
stdout (`status`, then a final `ledger` / `section` / `audit`, or `error`). Exit 0 ok, 2 bad request, 1 failure.
See `src/harness/run.py`. Each call pays ~3 s of Python/LangGraph import time; if that matters at merge, add a
long-lived worker mode.

## Layout
`prompts/` ported prompts + profiles, `corpus/` build + retrieval, `sources/` research agent + claim verifier,
`validate/` style/structure/overlap/judge, `graph.py` the write_section LangGraph, `evaluate.py` A/B logic.

## Style profile (mined from the real-thesis corpus, no API calls)
`python eval/phrase_gap.py` compares every saved AI draft in `eval/out/thesis_samples/` with the real theses (holdout excluded)
and writes `data/style_profile.json` (+ `eval/out/phrase_gap.md` for people): phrases AI overuses, plain phrasing real writers
use, signpost and narrative-citation rates, real paragraph-length shape. The harness puts it in the cached system block, draws the
paragraph plan from it, and patches paragraphs stacked with AI-only phrases. `python eval/profile_score.py` scores any text on
that scale for free. Re-run `phrase_gap.py` after each batch of new drafts. Bibliography text inside chapters is filtered at
ingest, at retrieval and in the miner (`text/refs.py`; `store.drop_reference_chunks` cleans an old database).

## Reader-visible defects (free, no API calls)
`python eval/defects.py` counts what trips a reader or the judge in real theses vs saved baseline vs saved harness drafts: pasted or
restated sentences, source-by-source "Author (year) traces ..." summaries, "I argue" announcements, verdict closers, "What ..." openers.
`src/harness/text/cleanup.py` fixes or flags the same things inside `write_section`: chunk seams and orphan fragments are healed, a
sentence pasted from a source abstract or restating a neighbour is deleted, and the rest goes to the one cheap Haiku patch. Scholar's
own prompt lines that seed stock phrases are replaced on the harness path only (`graph._SEEDS`), so the golden tests stay byte-identical.

### Round 2: closing the real-thesis gaps (see SPEC.md, tasks/plan.md)
Free wording rewrites (`cleanup.polish`): "I argue that X" becomes "X", "What the sources show is that X" becomes "X", "Author (year)
shows that X" becomes "X (Author, year)" or "According to Author (year), X", verdict-closer sentences, emphasis adverbs and extra
"that" are dropped. The plan (`graph._concrete_rules`) now sets per-section quotas for exact numbers (source figures, plus the
thesis's own chapter, objective and question numbers), first-person paragraphs, one example paragraph per passage and plain
signposts. `text/numbers.py` measures the shares; below 70% of target, the one cheap patch is asked to add the missing piece.
`python eval/defects.py` and `python eval/mock.py` show the effect for free. The judge effect is unproven until a probe runs.
