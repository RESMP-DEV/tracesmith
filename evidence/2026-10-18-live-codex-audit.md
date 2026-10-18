# Live Codex corpus audit

Date: 2026-10-18

## Scope

This audits the current TraceSmith pipeline against recent local Codex
rollouts and records the accepted extractor/statistics repair at commit
`f1abfa0`. It does not publish a dataset or claim that aggregate token usage
is authoritative when the source telemetry is inconsistent.

## Live input

The audit staged 37 Codex rollout files from October 16 through October 18,
90 MB total, under:

`~/.cache/alphaheng/tracesmith-runs/20261018-live-audit/stage`

The explicit pipeline was extract, redact, verify, export, and stats. The
redaction run replaced 3,155 private terms, 1,108 home paths, 55 private URLs,
279 other URLs, 110 emails, six IP addresses, and one environment secret
assignment. `tracesmith verify` scanned the redacted tree and reported no
leftovers.

The export retained 36 conversations with assistant output, 80 ShareGPT
pairs, and dropped one resume shell with no assistant turn.

## Extractor defect and repair

Before the repair, TraceSmith counted `function_call` and `custom_tool_call`
response items but ignored Codex `web_search_call` response items. The staged
files contained 279 native web-search calls. Seven cited research
conversations therefore appeared to have no tools even though their assistant
answers referred to live search evidence.

Commit `f1abfa0` adds `web_search_call` to the deduplicated Codex tool-call
count and adds a focused regression test.

Corrected aggregate results:

- conversations with tools: 36/36;
- total tool calls: 2,377;
- tool calls per assistant message: 4.1775;
- GLM-5.3 density: 4.0817 calls per assistant message;
- GLM-5.3-Flash density: 4.4121 calls per assistant message.

The old count was 2,098 calls, 29 conversations with tools, and 3.6872 calls
per assistant message. The missing native searches had made Flash appear less
tool-first than it actually was.

## Statistics improvements

`tracesmith stats` now reports:

- total assistant messages, tool calls, and tool results;
- tool-call density;
- zero-tool conversation count;
- per-model conversation, character, assistant-message, tool-call, density,
  and zero-tool metrics;
- reported cache ratio;
- the number of traces where cached input exceeds input.

These fields make corpus-quality regressions visible without a separate
ad hoc analysis script.

## Token telemetry anomaly

In 7 of 36 exported conversations, Codex reports
`cached_input_tokens > input_tokens`. The aggregate reported cache ratio is
therefore 1.0348, which is not physically valid for a component split.
The anomaly occurs in both `ccr_worker` and `claude-code-router` providers.

TraceSmith now exposes this condition instead of silently treating the number
as a valid hit rate. Correcting the upstream telemetry counter is outside this
repository.

## Tool-required export check

After the repair, `tracesmith export --require-tools --variant both --dedup`
retained all 36 conversations with assistant output and all 80 ShareGPT
pairs. Its only reported filter drop was the one resume shell with no
assistant output. Before native search calls were counted, the seven cited
research conversations would have been excluded by this filter.

## Validation

- Baseline repository: clean at `81ed537`, identical to `origin/main`.
- Baseline suite: 136 passed.
- Repaired suite: 138 passed.
- Focused Ruff and formatting checks passed for `stats.py` and `tests/test_stats.py`.
- `git diff --check` passed before commit.
- Full formatting of the legacy Codex extractor was not attempted; it has
  pre-existing style debt. The changed branch is covered by the new focused
  extractor test and the full suite.

## Non-claims

- No trace data or derived corpus was uploaded.
- No claim is made about other agent sources from this Codex-only live run.
- Aggregate token totals remain source-reported values, not corrected billing
  totals.
