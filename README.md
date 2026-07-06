# agent-trace-share

Extract, redact, and export AI coding-agent traces into DistillKit-ready datasets.

`agent-trace-share` (`ats`) is a CLI that turns the local conversation logs left
behind by AI coding agents (Claude Code, Codex, Cursor, Gemini CLI, Continue,
OpenCode, Trae, Windsurf) into clean, redacted, fine-tuning-ready data. It runs
four stages — **extract → redact → export → manifest** — and can publish the
result straight to the HuggingFace Hub.

The goal is a reproducible, privacy-first pipeline for building instruction /
chat datasets from your own agent usage. Nothing about your identity is ever
baked into the tool: every secret, path, email, and identifier is replaced with
a stable placeholder before any export is written.

> **Design spec:** the full architecture, locked decisions, and stage contracts
> live in
> [`docs/superpowers/specs/2026-07-06-agent-trace-share-design.md`](../docs/superpowers/specs/2026-07-06-agent-trace-share-design.md).

---

## Status

**v0.1.0** — extract (8 sources), layered redaction, two export variants,
MANIFEST provenance, verify, stats, sample, publish. This is the first
publishable milestone.

---

## Install

Requires Python ≥ 3.10.

```bash
# Core (click only).
pip install -e .

# Optional extras:
pip install -e ".[publish]"          # huggingface_hub, for `ats publish`
pip install -e ".[privacy-filter]"   # transformers + torch, for the optional privacy filter
pip install -e ".[dev]"              # pytest + pytest-cov, for running the suite
```

The optional redaction layers have extra, non-Python dependencies:

- **gitleaks** — the external `gitleaks` binary (the `--gitleaks` / `--gitleaks-fix`
  flags). Install separately, e.g. `brew install gitleaks`.
- **LLM residue pass** — any OpenAI-compatible endpoint, passed via
  `--llm-residue http://localhost:8000`.

These are all opt-in. The default redaction path needs nothing but `click`.

---

## Quick start

One command runs the whole pipeline end to end:

```bash
# Extract from ~, redact, export both variants, write MANIFEST.json.
ats run --out ./ats_output --user "$USER"
```

This produces:

```
ats_output/
├── raw_extracted/<source>.jsonl   # normalized conversations, one per source
├── redacted/<source>.jsonl        # PII/secrets replaced with placeholders
├── REDACTION_REPORT.json          # per-source redaction counts + config snapshot
├── export/
│   ├── messages.jsonl             # DistillKit "messages" variant
│   └── sharegpt.jsonl             # DistillKit "ShareGPT" variant
└── MANIFEST.json                  # provenance: counts, config, file hashes
```

Then inspect before you publish:

```bash
ats verify --in ./ats_output/redacted     # scan for leftover PII (exits 1 on findings)
ats stats  --in ./ats_output/export       # corpus metrics as JSON
ats sample --in ./ats_output/export -n 5 --out ./preview   # random 5-row preview
```

Happy? Publish:

```bash
ats publish --repo youruser/my-traces --in ./ats_output/export --variant both
```

---

## Supported sources

Eight extractors, all auto-discovered from the local home directory:

| Source | Key |
|---|---|
| Claude Code | `claude_code` |
| Codex (OpenAI) | `codex` |
| Continue | `continue` |
| Cursor | `cursor` |
| Gemini CLI | `gemini` |
| OpenCode | `opencode` |
| Trae | `trae` |
| Windsurf | `windsurf` |

Each source contributes its own `<source>.jsonl` in every stage, so datasets can
be sliced or dropped per-source (see `--drop-sources` under `export`).

---

## CLI reference

All eight commands. Defaults are shown.

### `ats run` — the whole pipeline

```
ats run [--sources a,b] [--root ~] [--out ./ats_output]
        [--variant messages|sharegpt|both] [--user NAME] [--home DIR]
```

Chains **extract → redact → export → MANIFEST.json**, calling the stage
functions directly so the per-stage counts feed the manifest. Use `--user` and
`--home` so the redactor knows which username/home path to scrub as private
identifiers (defaults to `$USER` / `$HOME`).

### `ats extract`

```
ats extract [--sources a,b] [--root ~] [--out ./ats_output]
```

Scan `--root` for installed agents and write `raw_extracted/<source>.jsonl`.
Sources with no installation contribute an empty file (downstream stages always
find a file). With no `--sources`, all 8 are tried.

### `ats redact`

```
ats redact [--in ./ats_output/raw_extracted] [--out ./ats_output/redacted]
           [--allow-public-urls] [--allow-domain D]... [--private-term T]...
           [--private-domain D]... [--user NAME] [--home DIR]
           [--privacy-filter] [--privacy-filter-device auto|cpu|cuda|cuda:N]
           [--gitleaks] [--gitleaks-fix]
           [--llm-residue URL]
```

Applies the layered redactor (rules → optional privacy filter → optional
gitleaks fix → optional LLM residue → optional final gitleaks scan) and writes
`redacted/<source>.jsonl` plus a sibling `REDACTION_REPORT.json`.

### `ats export`

```
ats export [--in ./ats_output/redacted] [--out ./ats_output/export]
           [--variant messages|sharegpt|both]
           [--min-turns N] [--max-turns N] [--min-assistant-chars N]
           [--drop-sources a,b] [--dedup]
```

Flatten redacted conversations into DistillKit-ready JSONL (see *Output format*
below). Quality filters drop conversations with no assistant turn, apply
turn-count bounds, require a minimum assistant message length, drop whole
sources, and optionally deduplicate.

### `ats verify`

```
ats verify [--in ./ats_output/redacted]
```

Scan the redacted tree for leftover PII/secrets that survived the pipeline.
Exits `0` if clean, `1` if anything is found (prints up to 20 findings).

### `ats publish`

```
ats publish --repo USER/DATASET [--in ./ats_output/export]
            [--variant messages|sharegpt|both] [--private]
```

Upload the chosen variant(s) to the HuggingFace Hub with an auto-generated
dataset card (provenance, redaction summary, attribution). Requires the
`[publish]` extra and a logged-in `huggingface_hub` token.

### `ats stats`

```
ats stats [--in ./ats_output/export]
```

Print corpus metrics (conversation counts, token/char totals, per-source and
per-variant breakdowns) as JSON.

### `ats sample`

```
ats sample [--in ./ats_output/export] [-n 10] [--seed 0] [--out DIR]
```

Sample `n` random conversations for pre-publish inspection. With `--out`, also
writes `sample.jsonl`. Seeded for reproducibility.

---

## Output format

`ats` writes **two interchangeable variants** of the same flattened data, so you
can point DistillKit at whichever one your recipe expects.

### `messages` variant — `export/messages.jsonl`

One JSON object per conversation, DistillKit's flat `messages` schema. Each
message's rich fields (`tool_use`, `tool_results`, `suggested_diffs`, …) are
flattened into a single string `content`, and the original `role` is preserved.

```json
{"messages": [
  {"role": "user", "content": "add a greeting function to the project"},
  {"role": "assistant", "content": "I'll create it.\n\n<tool_use name=\"write_file\">\n{...}\n</tool_use>"}
]}
```

### `sharegpt` variant — `export/sharegpt.jsonl`

One JSON object per **user→assistant instruction pair** (independent pairs, no
rolling context). A leading `system` message, if present, is attached to the
first pair. Uses the classic `from`/`value` shape:

```json
{"conversations": [
  {"from": "human", "value": "add a greeting function to the project"},
  {"from": "gpt",   "value": "I'll create it.\n\n<tool_use name=\"write_file\">\n{...}\n</tool_use>"}
]}
```

Worked, redacted examples of both variants live in
[`examples/`](examples/) — `sample_input.jsonl` (raw) plus the golden
`messages_expected.jsonl` and `sharegpt_expected.jsonl` the pipeline produces
from it.

### `MANIFEST.json`

Provenance for reproducibility: a UTC `created_at`, the `tool_version`, the
`config` snapshot used, a per-`sources` block (raw record counts, redaction
counts, exported message counts), and a `files` block with `sha256`/`bytes`/
`rows` for each export artifact. Generated automatically by `ats run`.

---

## DistillKit integration

Point DistillKit at the exported JSONL and pick the matching variant. Minimal
config snippet:

```yaml
# distillkit.yaml
dataset:
  # Use exactly one of the two below.
  path_messages: ./ats_output/export/messages.jsonl   # flat messages variant
  path_sharegpt: ./ats_output/export/sharegpt.jsonl   # from/value pairs

  # Quality knobs that mirror `ats export` so train/eval stay consistent.
  min_turns: 2
  min_assistant_chars: 50
  dedup: true

output:
  dir: ./distilled
```

The two variants are drop-in for DistillKit's expected schemas: `messages.jsonl`
→ the `{"messages": [{"role","content"}, ...]}` chat format; `sharegpt.jsonl`
→ the `{"conversations": [{"from","value"}, ...]}` instruction-pair format.

---

## Privacy model

Redaction is the core safety boundary. It is **layered and opt-in**, with the
rule layer always on:

1. **Rule layer (always on).** Regex-driven, ported from the upstream
   `agent-trace-redaction-methodology` and split by family — identifiers,
   secrets, paths, URLs, plus your private terms/domains. Every match becomes a
   stable, reversible placeholder like `[GENERIC_HOME_PATH:0001:58c69070ae]`
   (family + index + short hash) so structure is preserved but identity is not.
2. **Privacy filter (optional, `--privacy-filter`).** A transformers NER model
   that catches names/PII the regex layer misses. Needs the `[privacy-filter]`
   extra.
3. **gitleaks fix (optional, `--gitleaks-fix`).** Iteratively redacts findings
   from a live gitleaks scan, in place, before the final pass.
4. **LLM residue pass (optional, `--llm-residue URL`).** An OpenAI-compatible
   model call that re-reads the redacted text for anything still leaking.
5. **Final gitleaks scan (optional, `--gitleaks`).** A read-only last line of
   defense that fails loudly if anything remains.

Always run `ats verify` on the redacted tree before publishing — it's a cheap,
dependency-free safety net that exits non-zero on any leftover.

You control your own scope: `--user` and `--home` tell the redactor which
username and home directory to treat as yours; `--private-term` /
`--private-domain` add custom scrub targets; `--allow-public-urls` /
`--allow-domain` loosen URL scrubbing for domains you've decided are fine.

---

## Attribution

This project builds on two upstream MIT-licensed projects, vendored and
refactored in place:

- **0xSero/ai-data-extraction** — the eight source extractors.
- **RodriMora/agent-trace-redaction-methodology** — the layered redactor and
  canary test suite.

File-by-file provenance (upstream path, commit SHA, license, and a summary of
changes) is recorded in [`ATTRIBUTION.md`](ATTRIBUTION.md). Licensed under the
[MIT License](LICENSE).

---

## Design spec

The architecture, locked decisions, redaction rule inventory, export contracts,
and the full task breakdown are documented in
[`docs/superpowers/specs/2026-07-06-agent-trace-share-design.md`](../docs/superpowers/specs/2026-07-06-agent-trace-share-design.md).
