# SCHOLAR_ARCHITECTURE_AUDIT

Read-only audit. No source file was modified; this report is the only file written.
Date: 2026-11-10. Scholar HEAD `2d5c2ce` (branch `crm`). Harness HEAD `256cd6e`.

## 0. Two codebases, one pipeline

| Repo | Path | Stack | Role |
|---|---|---|---|
| acdyon-scholar | `AcdyOn/acdyon io/acdyon-scholar` | Express (Render), React/Vite (Hostinger), Supabase | Owns thesis/ToC/chapter state, SSE to browser, legacy 3-call writer, export |
| scholar-harness | `AcdyOn/scholar-harness` | Python 3.12, LangGraph, langchain-anthropic, SQLite FTS5, stdlib HTTP | Section writer, OpenAlex research, corpus exemplars, validators, polish. Called over HTTP by Scholar |
| corpus-pack | `AcdyOn/corpus-pack/corpus-pack.jsonl.gz` | gz JSONL | Packed corpus; harness boots it from a private URL |

Scholar calls the harness through three Node adapters: `backend/src/utils/harnessWriter.js` (`/research`, `/write-section`), `harnessRefine.js` (`/refine`), `harnessRefs.js` (saves cited works to Supabase).
Harness HTTP surface: `harness/server.py` `ROUTES = /refine /research /write-section /audit`, Bearer `HARNESS_TOKEN`, `MAX_BODY=300_000`, `SLOTS=BoundedSemaphore(HARNESS_MAX_CONCURRENT=4)` (busy returns 429).

**Unverified from this checkout:** whether `HARNESS_URL`/`HARNESS_TOKEN` are set on Render prod. If unset, `writerAllowed()` is false and every chapter uses the legacy 3-call writer (section 4A).

---

## 1. Directory and pipeline flow

### 1.1 Execution path (topic to compiled chapter)

| # | Stage | File : function | Notes |
|---|---|---|---|
| 1 | Thesis create | `routes/thesis.routes.js` `POST /` to `thesisController.create` | Stores title, university, programme, domain, methodology, geographic_focus, keywords, `toc_data.{objectives,questions,hypotheses}` |
| 2 | Title ideas (optional) | `POST /ai/suggest-titles` | |
| 3 | ToC generation | `thesisController.generateTOC` (`thesis.controller.js:~440-540`), prompt `services/toc.service.js:buildToCPrompt` | SSE; `orchestrator.streamGenerate`, feature `toc_generate`, `MAX_TOKENS.TOC=24000`; JSON repair via `utils/jsonRepair.js` |
| 4 | ToC validate | `toc.service.js:validateToC` | >=5 chapters, >=200 pages, sections non-empty |
| 5 | Finalise ToC | `thesisController.finaliseTOC` to `toc.service.js:chaptersFromToC` | Inserts `chapters` rows: `toc_sections={sections,ai_notes}`, `estimated_pages`, `target_words = pages*250`, `status NOT_STARTED` |
| 6 | Sources (optional) | `uploadSource`, `utils/extractText.js`, `sourceContext.js`, `sourceMeta.js` | Author documents into `thesis_sources` |
| 7 | Chapter generate | `POST /:id/chapters/:chapterId/generate` to `thesisController.generateChapter` (`thesis.controller.js:~640-1002`) | SSE; per-section loop; resume/cancel via `utils/generationRuns.js` |
| 7a | Per-chapter context | same fn, lines 767-803 | memory, profile, noise, empirical, ecology, temporal blocks; source rows; visual plan |
| 7b | Harness research | `harnessWriter.js:harnessResearch` to harness `/research` to `run.cmd_research` to `sources/agent.py:research_chapter` | Once per chapter (section 5) |
| 7c | Per-section writer | **A:** `harnessWriter.js:harnessWriteSection` to `/write-section` to `graph.py:write_section`. **B (fallback):** `buildSectionCall1/2/3Prompt` + `streamPassage` | Section 4 |
| 7d | Visuals | `directiveFor`, `writeVisualBlocks`, `weaveVisuals`, `streamPassage` repair | Harness path writes prose only; figures/tables requested separately |
| 7e | Optional polish | `harnessRefine.js:refineSection` to `/refine` to `refine.py:refine` | OFF unless user toggle (`harnessPolish:true`) or `HARNESS_REFINE=on` |
| 7f | Persist | `fullContent += sectionText + '\n\n'`; `chapters.update({content, word_count})` after **every** section | Final status `DRAFT` |
| 8 | Whole-thesis review | `POST /:id/review` to `reviewThesis` + `prompts/review.prompt.js` + `utils/reviewIntegrity.js:applyFixes` | Section 6 |
| 9 | Export | `exportChapter` / `exportFullThesis`, `services/export.service.js`; client `frontend/src/utils/docxExporter.js`, `contentSanitizer.js` | |

### 1.2 Harness internal graph (`harness/graph.py:build_graph`)

```
START -> context -> retrieve -> draft_open -> draft_mid -> draft_close -> humanize -> validate
validate --(problems && revisions==0 && flags.revise)--> revise -> validate
validate --(else)--> verify -> END
```
`write_section()` post-step (after graph): `cleanup.heal`, `cleanup.drop_repeats`, `cleanup.polish`, `_tidy` (em-dash), `_glue`, then `_new_refs`.

### 1.3 Core files, one line each

Scholar backend: `controllers/thesis.controller.js` (2010 lines, orchestration), `services/claude.service.js` (SDK + logging), `services/inferenceOrchestrator.service.js` (kill switch, rate limit, cache, retry), `services/modelGovernance.service.js` (ceilings), `prompts/chapter.prompt.js`, `prompts/toc.prompt.js` + `services/toc.service.js`, `utils/sectionProfiles.js`, `utils/thesisMemory.js`, `utils/literatureEcology.js`, `utils/visualBudget.js`, `utils/reviewIntegrity.js`, `utils/citationExtractor.js`, `services/crossref.service.js`.
Harness: `graph.py` (1114), `prompts/{base,sections}.py` + `prompts/profiles.json`, `corpus/{retrieve,store,sections,ingest,score,profile,openalex,boot,pack}.py`, `sources/{agent,tools,verifier}.py`, `text/{cleanup,numbers,protect,citations,apa,sentences,strip}.py`, `validate/{features,style,profile,overlap,structure,judge,gapfeats}.py`, `refine.py`, `spend.py`, `llm.py`.

---

## 2. Outline and structural planning

### 2.1 Chapter and section outline (ToC)
Single Claude call, `toc.service.js:buildToCPrompt(thesis)`. Persona: "PhD thesis architect with 40 years of experience". Inputs: candidate profile, thesis parameters, objectives/questions/hypotheses, methodology-specific guidance block, a typical 6-chapter skeleton. Output contract is JSON only:

```json
{ "suggested_title": "", "total_estimated_pages": 230,
  "sections": [
    {"id":"front_matter","type":"front_matter","title":"Front Matter","items":[{"id","title","pages"}]},
    {"id":"chapter_1","type":"chapter","number":1,"title":"","estimated_pages":25,"ai_notes":"",
     "sections":[{"id":"1.1","number":"1.1","title":"","pages":5,
                  "subsections":[{"id":"1.1.1","number":"1.1.1","title":"","pages":2}]}]},
    {"id":"references","type":"references","title":"","estimated_pages":15},
    {"id":"appendices","type":"appendices","items":[...]} ] }
```
Rules in prompt: total >= 200 pages, >=5 chapters, every section >=2 subsections, no generic titles. `ai_notes` is stored in `chapters.toc_sections.ai_notes` but **is not injected into any draft prompt** (grep: only `toc.service.js` writes it; `generateChapter` never reads it).
`prompts/toc.prompt.js` (`buildTOCPrompt`) is a second, older ToC prompt; it is not imported anywhere.

### 2.2 Paragraph outline: there is none persisted
No outline below subsection title exists in the DB. The only paragraph-level plan is generated at draft time in the harness, `graph.py:_para_plan(state, which)`:
- `k = clamp(round(words_target/140), 2, 6)` paragraphs per passage.
- RNG seeded `random.Random(f"{section.title}|{which}")` (deterministic per section and position).
- Lengths from `style_profile.json.paragraph_ratios[section_type]` (real-thesis ratios), `sizes = round(words_target*m/sum(m), -1)`, floor 30.
- Move per paragraph sampled without repeat from `PARAGRAPH_MOVES` (8 strings, e.g. "open on one specific study...", "qualify one finding...").
- `_concrete_rules` appends per-paragraph add-ons: required plain signpost ("For example,", "However," from `SIGNPOST_HINTS`), one `EXAMPLE_MOVE` paragraph, `FP_MOVE` first-person paragraphs (share `numbers.FP_PARA_TARGET`), exact-figure facts (`_mix(_numeric(state))`), "carries no citation" paragraphs (share `1 - CITE_PARA_REAL`), parenthetical-citation instruction (from `narrative_share_real`).
Emitted as text block "PARAGRAPH PLAN for this passage ..." appended to the prompt.

### 2.3 Section sizing
`thesis.controller.js:825-828` and mirrored `graph.words_target()`:
`wordsPerSection = ceil(section.pages*250)`; `wordsTarget = clamp(ceil(wordsPerSection/3), 200, 700)` per call; `max_tokens = ceil(wordsTarget*1.6)` (+ visual tokens on legacy path). A section = 3 calls ("open", "mid", "close"), so section length is at most about 2100 words regardless of `pages`.

### 2.4 Section-type profile (register)
`utils/sectionProfiles.js:detectProfile(chapterTitle, chapterNumber)`, mirrored in `harness/prompts/sections.py:detect_profile` and `prompts/profiles.json`. Regex on chapter title: introduction, literature_review, methodology, findings, discussion, conclusion, limitations, appendix (default `literature_review`; chapter 1 defaults to introduction). Each profile carries `instruction`, `noiseInstruction`, `empiricalInstruction`, `call2Note`, `call3Note` and numeric knobs (`citationDensityLevel`, `noiseIntensity`, `optimizationLevel`, `evidenceConsistency`, ...). The numeric knobs are **not read by any code**; only the text fields reach prompts.

### 2.5 Cross-chapter memory
`utils/thesisMemory.js`: `seedMemory(thesis)` (locked topic, methodology, objectives), `buildMemoryBlock`, `updateThesisMemory` (Haiku, `MAX_TOKENS.MEMORY_UPDATE=450`). **Background updates are disabled** unless `ENABLE_MEMORY_UPDATES=true` (burn incident noted in code). Memory block is used only by the legacy writer; `harnessWriter.slim()` drops `thesis_memory` and `content`.

---

## 3. Human thesis corpus

### 3.1 Storage
- Working copy: `scholar-harness/data/corpus.db` (SQLite). Gitignored (copyrighted text). Backups `corpus.db.before-load52`, `.before-refs`.
- Raw PDFs/markdown: `data/raw/<field>/<slug>/` (gitignored).
- Committed metadata: `data/manifest.json` (212 rows), `data/fields.selected.json`, `data/style_baseline.json`, `data/style_profile.json`.
- Prod distribution: `corpus-pack/corpus-pack.jsonl.gz`; `corpus/pack.py` builds, `corpus/boot.py:ensure_corpus(HARNESS_CORPUS_URL, HARNESS_CORPUS_SHA256, db_path)` downloads at server start (called in `server.main()`); image keeps corpus out (commit `4a5605b`).

### 3.2 Measured contents of `data/corpus.db` (queried this session)
- **226 theses** (30 flagged `holdout=1`, excluded from retrieval), **20 OpenAlex fields**, **20,088 chunks**.
- Chunks by section_type (open+mid+close): findings 13,020 (65%), introduction 2,473, conclusion 1,835, discussion 1,076, literature_review 818, methodology 826, limitations 40.
- Average chunk 460-510 words (open ~465, mid ~500, close ~335).
- `style_baseline.json`: `n_units=10146`, `by_section` keys introduction/findings/conclusion/discussion/methodology/literature_review/limitations/all.
- Caveat: `corpus/sections.py:_finalize_chapters` labels any unclassified body chapter of >=500 words as `findings`, so "findings" is a catch-all and thin for literature_review/methodology (818/826 chunks across 20 fields, about 40 per field).

### 3.3 Parsing and indexing (offline pipeline, `harness/corpus/*`)
`topics.py` (top 20 OpenAlex fields) to `candidates.py` to `acquire.py` (download, `pymupdf4llm`, gates) to `score.py` (Haiku rubric, 5 exemplar + 1 holdout per field, writes manifest) to `ingest.py`:
1. `sections.py:build_sections(md, toc)`: `split_chapters_from_toc` (PDF outline) else `split_chapters` (heading regex), `clean_thesis_text` removes references, tables, captions, running headers; `classify_title` assigns section_type.
2. `split_units` (sub-sections >=300 words, else ~1200-word windows), `chunk_unit(target=450, max=650)`: greedy paragraph chunks, first=`open`, last=`close`, rest=`mid`.
3. `store.py` schema: `theses(id, openalex_id, field_name, score, holdout, ...)`, `chunks(thesis_id, field_name, section_type, position, heading, chunk_idx, word_count, text)`, FTS5 `chunks_fts(heading,text)` with `porter unicode61`. Bibliography-looking chunks dropped (`text/refs.py:looks_like_references`).
4. `validate/style.py:build_baseline` computes per-section mean/std/p10/p90 for 12 features (`validate/features.py:FEATURES`).
5. `eval/phrase_gap.py` mines `style_profile.json`: AI-overused phrases (`avoid`, ratio >=10x = "tells"), `prefer`, `avoid_openers`, `signposts`, narrative-citation share, `paragraph_ratios`, per-feature `targets` p50, `ai_means`.

### 3.4 Injection into generation (retrieval + stats, no embeddings)
`graph.py:retrieve_node` to `corpus/retrieve.py:get_exemplars` per position (`open|mid|close`), N=`HARNESS_EXEMPLARS` (default 3):
1. `map_to_field(title, domain, fields)`: token overlap with field keywords, needs >=2 shared tokens else pool all fields.
2. `topic_query`: up to 8 content words as an FTS5 OR query on `title + chapter + section`.
3. `_rows`: `WHERE section_type=? AND position=? AND field_name=? AND holdout=0 ORDER BY bm25 LIMIT 30`, `_rerank` by token overlap with `heading + text[:600]`, one chunk per thesis.
4. Top-ups: best-scored chunks in field, then all fields. In-process LRU `_cache` (200).
5. `build_exemplar_block`: "STYLE EXEMPLARS: ... Do NOT copy or paraphrase..." plus `[EXEMPLAR i: field, position of a <label> section]` full chunk text, injected in the **user prompt** of call 1/2/3 (position-matched).
6. Guard: `validate/overlap.py` (8-gram; flagged if a 12+ word run or >2% shared) in `validate()`; `passage_overlaps` per paragraph for the patch.

Additional corpus-derived context (system block, cached): `validate/style.py:style_target` (mean sentence length, hedge rate, citation rate), `validate/profile.py:profile_block` (tell phrases, signpost rate, ranges, "that"/"of"/word-length/digit targets), `graph.VOICE_RULES`.
`humanize` node reuses up to 4 exemplars (500 chars each) as style examples, **but only when `flags.humanize` is set** (see 6.4).

Legacy writer (section 4B) uses **no corpus at all**.

---

## 4. Drafting and Claude API integration

### 4.A Harness writer (default when configured)
- Request from Node (`harnessWriter.js:harnessWriteSection`): `{thesis(slim), chapter(slim), section, references[], ledger, author_material, n_sections, flags:{pool:true, sources:true, revise:true}}`; `minWords = floor(wordsPerSection*0.5)`, else rejected and legacy writer runs. Timeout `HARNESS_WRITER_TIMEOUT_MS` (300000).
- Models: `config.py` `SONNET_MODEL` default `claude-sonnet-4-6`, `HAIKU_MODEL` default `claude-haiku-4-5-20251001`.
- `llm.py:chat(tier, max_tokens, spend, temperature=None)`: **temperature is not set for drafting** (Anthropic default). Only the claim verifier sets `temperature=0`. No top_p, no stop sequences.
- Draft call = `graph.py:gen(prompt, max_tokens, shared)`: `SystemMessage` with `cache_control: ephemeral` containing `_shared(state)` = style_target + profile_block + `VOICE_RULES` + REFERENCES + SOURCE NOTES + `CITATION_RULES`; `HumanMessage` = `P.build_call{1,2,3}` output + `_para_plan`.
- `max_tokens = ceil(words_target*1.6)`; `words_target` 200-700.
- Prompt text is `harness/prompts/sections.py` (byte-identical port of `chapter.prompt.js`, golden-tested) then `graph._plain()` applies `_SEEDS` replacements (section 6.3) that overwrite Scholar's own anti-detection lines.
- Prompt skeleton (all 3 calls): `BASE_PERSONA` ("distinguished veteran scholar ... NOT an AI content generator") / THESIS block / CHAPTER block / profile+noise+empirical blocks / exemplar block / TASK / `ANTI_AI_RULES` / `NONE_DIRECTIVE` / REFERENCES / feedback / "Write N words. No preamble."
- Call roles: 1 = theoretical foundation (ends on a point), 2 = friction/critique (continues from chunk1 + `used_concepts` + `POINTS ALREADY MADE`), 3 = synthesis (sees chunk1+chunk2).
- Guard rails: `spend.py:Spend` callback, per-request budget `HARNESS_BUDGET_USD` (default $2; HTTP 402 on breach); rates hard-coded for sonnet-4-6 and haiku-4-5.
- Harness path **does not receive**: `memory_block`, `ecology_block`, `temporal_block` (not passed in `_build`), `thesis_memory` (dropped by `slim`).

### 4.B Legacy Scholar writer (fallback)
- `thesis.controller.js:buildSectionCall1/2/3Prompt` then `streamPassage` to `streamChunkWithRetry` to `orchestrator.streamGenerate({prompt, maxTokens, feature:'chapter_generate'})`.
- `claude.service.js:streamGenerate`: `anthropic.messages.stream({model, max_tokens, messages:[{role:'user', content: prompt}]})`. **No system prompt, no temperature, no cache_control.** Model = `FEATURE_MODELS.chapter_generate = claude-sonnet-4-6`.
- `inferenceOrchestrator.service.js`: kill switch, 10 calls/60s per thesis, 5 calls/30s per user+feature, 2-min result cache (non-stream only), `safeRetry` (non-stream only). `modelGovernance.service.js`: Opus blocked, ceiling by class (generation 60k, research 20k, validation 12k, background 8k).
- Per-call blocks: `BASE_PERSONA`, `ANTI_AI_RULES`, `buildVisualRules(directive)`, `sourceRules`, `extractUsedConcepts` (capitalised words repeated >=2).
- Chunks are concatenated `chunk1+chunk2+chunk3` with no cleanup.

### 4.C Other Claude features (`claude.service.js` `FEATURE_MODELS`, `MAX_TOKENS`)
Sonnet: `chapter_generate` (16000 const, per-call computed), `toc_generate` (24000), `research_paper` (20000), `section_regenerate` (3000), `sop` (8000), `synopsis` (9000). Haiku: `inline_transform` (800), `memory_update` (450), `research_chat` (8000), `abstract` (1000). Review pass `thesis_review` (8000).
Frontend also has a browser-direct Anthropic fallback (`VITE_ANTHROPIC_KEY`, `frontend/src/prompts/*`): not audited here.

---

## 5. OpenAlex citation engine

OpenAlex lives **only in the harness** (`harness/corpus/openalex.py`, `sources/*`). Scholar's own CrossRef code (`services/crossref.service.js`, `reference.controller.js`) only verifies the user's pasted reference list; it is not used in drafting.

Auth: `OPENALEX_API_KEY` (`openalex.enabled()`); `mailto=support@acdyon.io`; retries 429/5xx twice.

| Phase | When | Code | Behaviour |
|---|---|---|---|
| 1. Research (pre-draft) | Once per chapter, before the section loop | `run.cmd_research` to `sources/agent.py:research_chapter` | Haiku `PLAN_SYSTEM`: up to 8 queries (>=3 are exact titles of seminal works). Each: `openalex.search_works(q, from_year, 12)` with filter `type:article\|review, is_retracted:false, has_abstract:true`. Dedup into `tools.Ledger` (S1..Sn). Top 60 by `cited_by` go to Haiku `SELECT_SYSTEM`: pick <=12 ids with section numbers. Fallback: top-cited. Returns `{sources, by_section}`. Fail-open (returns None, writer proceeds without) and skipped when no key |
| 2. In-draft grounding | Each section | `tools.merge_refs` (library first, dedup by DOI/OpenAlex id/title sim 0.9), `source_notes_for` (<=8 notes, abstract opening + up to 2 figure/name sentences, 420 chars), `facts_for` | Sources enter system block as SOURCE NOTES and REFERENCES; `CITATION_RULES` allows citing listed works plus well-known foundational works, "every citation is looked up afterwards" |
| 3. Post-hoc resolve | `validate` node | `graph._resolve` to `openalex.find_cited_work(surname, year, hint)` | For each `(Author, Year)` not in refs: filter `raw_author_name.search:<surname>,publication_year:<year>,is_retracted:false`, sort `cited_by desc`; accept only if first-author matches and the hint (thesis title + 200 chars of citing context) shares >=1 title word and >=2 words overall. Max 15 lookups per graph run. Hits are adopted as refs (`added_refs`) |
| 4. Unmatched audit | `validate` | `_unmatched` via `text/citations.py:audit_citations` | Regex `PAREN_RE`/`NARRATIVE_RE`; match = year + first-author surname substring. Unmatched triggers revise (patch reason "cites a work not in REFERENCES") |
| 5. Claim verification | `verify` node | `sources/verifier.py:audit_chapter` | Every citing sentence (cap `MAX_CLAIMS=40`, evenly sampled) checked against the cited work's **abstract only** (<=1500 chars) by Haiku, batches of 12, `temperature=0`, verdicts SUPPORTED/PARTIAL/UNSUPPORTED/CANNOT_TELL |
| 6. Repair | `verify` | `_fix_unsupported`, `_drop_sentences` | Sequence rewrite (UNSUPPORTED+PARTIAL via Haiku) then delete UNSUPPORTED sentences up to 3 rounds. Text that leaves has no UNSUPPORTED sentence |
| 7. Bibliography sync | After write | `graph._new_refs` to Node `harnessRefs.js:saveNewReferences` | Cited works not in the author's library are inserted into Supabase `references` + `thesis_references` (dedupe DOI/title) |

Observations: the citation engine is plain regex APA-parenthetical matching, so it is blind to numeric `[1]`, group `(A, 2018; B, 2021)` only partly (`protect._GROUP` handles it, `extract_citations` does not). Verification is abstract-level, so PARTIAL/CANNOT_TELL claims pass through unless rewritten. Section writers run in separate graph invocations, so `tried` lookup cache resets per section.

---

## 6. Existing linting, humanization and post-processing

### 6.1 Prompt-level (before generation)
- Scholar `chapter.prompt.js:ANTI_AI_RULES` (all three calls): em-dash budget (1 per 3 paragraphs), mix short (<8 words, "at least two") and long sentences, uneven paragraphs, banned transitions (furthermore, moreover, additionally, consequently, this demonstrates, this highlights, this suggests, to summarise, ultimately), suggested natural transitions ("What matters here is...", "The issue becomes more complex when...", "The problem emerges when..."), allow conceptual drift.
- Profile `noiseInstruction` / `empiricalInstruction` per section type (asks for restatement, imperfection, mixed evidence, effect sizes 0.20-0.45).
- Harness `graph._SEEDS` (15 string replacements, harness path only) reverses the lines above that seed stock phrases ("The problem emerges when", "This study takes the position") or license drift/restatement; `_trim_noise` drops sentences matching `_REPEAT_LICENCE`; `_EVIDENCE` swaps empirical-realism text for source-grounded text.
- Harness `graph.VOICE_RULES`: first person, cite sparingly, concrete figures, sentence rhythm (avg ~29 words, spread 10-55, under-8-word sentence ~1 in 20), subject-first paragraph openers (never "What ..."), plain connectives ~1 per 150 words, hedge ~1 per 250 words, "say each point once", banned machine tells.
- Mined `profile_block` (tell phrases, long-word list, signpost rate, ranges) and the paragraph plan (section 2.2).

### 6.2 Deterministic detectors in `validate` (no model call)
| Detector | Where | What |
|---|---|---|
| `_FORMULAIC` regex (~45 patterns) | `graph.py:100` | Furthermore/Moreover openers, "plays a crucial role", "not only ... but also", delve/pivotal/intricate/meticulous/vibrant/showcase/underscore/interplay/bolster/garner, "serves as", ", highlighting ...", "Taken together,", "it is worth noting", tidy colon triads, "genuinely contested", etc. Trigger when >=2 paragraphs match |
| `STOCK_PHRASES` / `_STOCK_RX` | `validate/features.py` | 19 phrases (delve, tapestry, testament to, furthermore, moreover, ...) |
| `tell_count(text)` | `graph.py:272` | `_FORMULAIC` + `_STOCK_RX` hits (eval metric) |
| `avoid_hits`, `profile_metrics` | `validate/profile.py` | Mined tell phrases (AI/real ratio >=10); tells per 1000, signpost rate, narrative-citation share |
| `style_features` (12 features) | `validate/features.py` | sent_len_mean/cv, short/long sentence ratio, para_len_cv, hedge/booster/citation/first-person per 100, opener_repeat, stock_per_1000, em_dash_per_para. This is the closest thing to burstiness; there is **no perplexity** model |
| `score_section` | `validate/style.py` | z-score vs baseline per section type, `passed` if max|z|<=3 and score >= p10; `STD_FLOOR` per feature |
| `_voice_targets`, `_cite_targets`, `cite_para_targets`, `_style_targets`, `_plain_targets`, `_gap_targets` | `graph.py` | first-person gap, citation over `CITE_CEILING=1.25`x real, cited-paragraph share vs `CITE_PARA_REAL`, signpost and narrative-citation gaps, "that"/connective/short-sentence/"the field" gaps, missing exact figures and examples |
| `_is_chain` | `graph.py` | 3+ sentences opening "Author (year)" or 2 `cleanup.REPORT` lines |
| `cleanup.repeat_flags` | `text/cleanup.py` | restated sentence (`same_idea` content-word Jaccard >=0.45), >1 "I argue", verdict-closer, >=2 "What..." paragraph openers |
| `overlap_report`, `passage_overlaps` | `validate/overlap.py` | exemplar copying |
| `check_structure` | `validate/structure.py` | per-section-type keyword checks (used by `/audit` only, not in the write loop) |

### 6.3 Free rewrites (no model), `text/cleanup.py` and `graph.py`
`heal` (seams, orphan fragments, truncated last sentence), `drop_repeats` (pasted-from-abstract and restated sentences), `polish` = `strip_stance` ("I argue that X" to "X"), `what_cleft`, `report_to_cite` ("Lee (2015) shows that X" to "X (Lee, 2015)" / "According to..."), `drop_closers`, `drop_boosters`, `plain_words` (`PLAIN` dict: utilize/use, furthermore/also, ...), `thin_that`; `_vary` (merge shortest paragraphs to raise `para_len_cv`); `graph._tidy` (em dash and spaced en dash to ", ", applied to final text).

### 6.4 Model-based passes
| Pass | Model | Trigger | Detail |
|---|---|---|---|
| `humanize` node | Sonnet | **`flags.humanize` only; Node never sends it** (`harnessWriter.js` sends `{pool, sources, revise}`), so this node is dead in production. `evaluate.py` also sets `humanize: False` in the "full" config ("doubled cost for no style gain"), so it looks deliberate | Rewrites all 3 chunks as JSON with 4 exemplars + style target + `VOICE_RULES`; skipped if `score_section` already passes |
| `revise` node (patch) | Haiku | `route()` when any problem flag and `revisions==0` | `_patch`: one call rewrites only flagged paragraphs (<=8, `prune_flagged`), reasons enumerated in `_flagged_paragraphs`; falls back to Sonnet rewrite of the worst chunk and following chunks only for hard problems (unmatched cite, overlap, hard style z). **One revision max** |
| `verify` node | Haiku | always | claim check + rewrite/delete (section 5) |
| `/refine` | Sonnet (tier flag) | OFF by default; user toggle `harnessPolish` or `HARNESS_REFINE=on`; `shadow` mode logs only | `refine.py`: 350-word paragraph chunks, SYSTEM prompt (vary sentence length, plain connectives, AVOID stock phrases and em dashes), gate `text/protect.py:compare` (citations, every number, quotes, figure blocks, headings, paragraph count); 1 retry, else original kept. Node also requires length ratio 0.7-1.4. The `polish` argument of `refine()` is accepted but unused in the body |
| Stitch pass | n/a | `buildStitchRefinementPrompt` exists in `chapter.prompt.js` but `generateChapter` never calls it | dead code |
| Thesis review | Sonnet | user clicks review | `review.prompt.js`: JSON issue list with exact `find`/`replace`; `reviewIntegrity.js:applyFixes` rejects changes to citations, numbers, headings |

### 6.5 Client-side
- `frontend/src/utils/contentSanitizer.js:sanitizeChapterContent` (used by `docxExporter.js`): broken-sentence repair, placeholder strip, mermaid cleanup, and an em-dash budget (`PAIRED_DASH`/`SINGLE_DASH` to commas, numeric ranges and headings kept). Commit `204b186` found no watermark/invisible characters; the signal was em-dash density (13.68 per 1000 words).
- `frontend/src/components/HumanizePanel.jsx`: third-party Clever Humanizer widget; user copies text in and pastes the result back by hand. Text leaves the system.
- `utils/export/humanizationFinalPass.js` (282 lines) was deleted in `204b186`.

### 6.6 Evaluation (offline, not in the request path)
`eval/ab.py`, `eval/polish_ab.py`, `eval/probe.py`, `eval/defects.py`, `eval/gap_report.py`, `eval/profile_score.py`; `validate/judge.py` (Haiku/Sonnet pairwise "which reads human", both orders, forced-choice variant). Judge is another LLM, not an external AI-detector. No GPTZero/Turnitin/Originality or open-weights detector call exists anywhere in either repo.

---

## 7. Integration surface for three new anti-detection modules

Lowest-risk hook points, in order of the pipeline:
1. **Pre-draft constraints:** `graph._shared(state)` (cached system block) and `graph._SEEDS` for prompt-side rules. Changing `ANTI_AI_RULES` in `chapter.prompt.js` breaks golden tests (`tests/golden/cases.json`); do it harness-side.
2. **Free deterministic stage:** `cleanup.polish()` list `(name, fn)` is the natural plug-in point (each fn is `(chunks) -> (chunks, count)`, idempotent, never removes figures/citations).
3. **Detector and router:** `graph.validate()` builds `problems`; `route()` sends to `revise`; `_flagged_paragraphs` turns problems into per-paragraph reasons for `_patch`. A new detector adds a key to `problems`, a clause in `route()` and a reason in `_flagged_paragraphs`.
4. **Final whole-section pass:** `write_section()` tail (`heal`, `drop_repeats`, `polish`, `_tidy`) or `refine.refine()` (has the integrity gate `protect.compare`, reuse it for any rewriting module).
5. **Chapter-level:** only `/audit` (`run.cmd_audit`) and Node `reviewThesis` see a whole chapter; nothing in the write loop does.

Known issues that will affect the new modules:
- `humanize` is unreachable from Node (flag never sent) and `flags.pool` is required for exemplars; confirm the intended defaults before relying on either.
- Revisions are capped at 1, so a new detector's findings share one patch budget with 20+ existing problem types (`prune_flagged` limit 8).
- No temperature is set for drafting, so sampling variance is whatever the SDK default gives; drafting is stateless across sections (no cross-section repetition or opener control beyond `POINTS ALREADY MADE` within a section).
- The corpus is 65% "findings" chunks; literature_review/methodology exemplars are thin (about 40 chunks per field).
- Harness path loses `thesis_memory`, ecology and temporal blocks that the legacy writer has.
- Citation matching is surname+year substring, so false matches are possible for common surnames.
- Prod env flags (`HARNESS_URL`, `HARNESS_REFINE`, `HARNESS_WRITER`, `ENABLE_MEMORY_UPDATES`, `OPENALEX_API_KEY` on the harness) could not be verified from the repos.
