# Node adapter contract

Source of truth: `src/harness/run.py`. Stateless process per call. No server.

## Invocation

```
python -m harness.run <research|write-section|audit>
```

- stdin: ONE JSON object, then close stdin.
- stdout: NDJSON, one JSON object per line, flushed per line. Every line parses. Last line is the terminal event.
- stderr: logs only. Never parse. Capture for diagnostics.
- cwd: repo root (or set `HARNESS_DB` absolute).

## Exit codes

| code | meaning | terminal event |
|------|---------|----------------|
| 0 | ok | `ledger` / `section` / `audit` |
| 2 | bad request (bad usage, invalid JSON, missing/mistyped field) | `{"type":"error","message":...}` |
| 1 | failure | `{"type":"error","message":...}`; budget stop adds `"code":"budget_exceeded"` |

Adapter rule: exit 0 + last line type != `error` = success. Anything else = failure. Exit 2 = do not retry (fix the request). Exit 1 = retry at most once, except `budget_exceeded` = never retry.

## Events

| type | when | shape |
|------|------|-------|
| `status` | progress | `{"type":"status","value":"researching"}`; write-section: `drafting`, `validating`, `revising`, `verifying` |
| `ledger` | research done | `{sources[], by_section{}, rounds?, searches?, skipped?, usage}` |
| `section` | write-section done | `{text, reports{style,overlap,citations,claims}, revisions, usage}` |
| `audit` | audit done | `{style[], structure, citations{matched,unmatched[],suspicious}, claims{stats,checks}, usage}` |
| `error` | failure | `{message, code?}` |

`usage` = `{total_usd, by_model{model:{calls,in,out,usd}}}`, present on every success event.

## Requests

### research
```json
{"thesis": {}, "chapter": {"chapter_number": 2, "title": "..."}, "sections": [{"number": "2.1", "title": "..."}]}
```
Fail-open: no `OPENALEX_API_KEY` -> exit 0, `ledger` with `"skipped":"no_key"`. Agent failure -> exit 0, `"skipped":"agent_error"`. Node should proceed without sources on any `skipped`.

### write-section
```json
{"thesis": {}, "chapter": {}, "section": {}, "references": [], "sources": [], "by_section": {},
 "flags": {"pool": false, "sources": false, "revise": true}}
```
`flags.pool: true` needs the corpus DB (`HARNESS_DB`, default `data/corpus.db`). Missing DB = no exemplars, not an error.

### audit
```json
{"content": "chapter text", "chapter": {}, "thesis": {}, "references": [], "claims": true}
```
`claims: false` skips LLM claim verification (no cost).

## Timeouts (adapter side; harness sets none on LLM calls)

Harness has no wall-clock limit. Node MUST enforce, then `SIGKILL` the child:

| command | suggested hard timeout |
|---------|------------------------|
| research | 120 s |
| audit | 120 s (`claims:false`: 30 s) |
| write-section | 300 s |

Treat timeout as failure (exit 1 semantics). Nothing is persisted by the harness, so kill is safe.
Idle watchdog: any NDJSON line resets it if you prefer streaming semantics; `status` events arrive between stages, not during a stage.

## Budget

- Env `HARNESS_BUDGET_USD` (default 2) = per-process hard cap, set per call.
- Counted across all LLM calls in the process (Sonnet + Haiku, rates in `spend.py`; unknown model = Sonnet rate).
- Exceeded -> stop, `error` with `code:"budget_exceeded"`, exit 1. Partial output is not emitted.
- Node owns cross-call budgets: sum `usage.total_usd` from each success event, pass a smaller `HARNESS_BUDGET_USD` per spawn.
- Cost of a failed/killed run is unreported. Reserve headroom.

## Environment passed by Node

`ANTHROPIC_API_KEY`, `OPENALEX_API_KEY`, `HARNESS_BUDGET_USD`, optional `SONNET_MODEL`, `HAIKU_MODEL`, `HARNESS_DB`. Real env wins over `.env`. Never log them.
`HARNESS_FAKE_LLM=1` is test-only; must be unset in prod.

## Parsing rule for Node

Read stdout by line, `JSON.parse` each, forward `status` to the UI, keep the last event. Ignore blank lines.
