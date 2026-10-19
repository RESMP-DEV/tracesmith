# TraceSmith agent instructions

TraceSmith is the canonical private tool and authority for turning local agent
conversation histories into publishable fine-tuning corpora. Its jurisdiction
covers source discovery/extraction, redaction and sanitization, normalization,
trace corpus consolidation and deduplication, provenance manifests, publication
cards, and trace-hosting receipts.

## Ownership corrections

- Do not create or extend a separate repository as the durable home for a trace
  extraction, sanitization, consolidation, or publication capability. Add it to
  TraceSmith instead.
- `RESMP-DEV/codex-trace-sanitizer` and `RESMP-DEV/resmp-reprocessor` are
  legacy task-local implementations, not competing authorities. Treat their
  useful behavior and tests as absorption input; do not invest further product
  surface there except for a safety fix needed before migration.
- Model fine-tuning and hyperparameter studies remain owned by their model
  trainer repositories. TraceSmith owns the corpus and its privacy/provenance
  boundary up to the point a trainer consumes a versioned artifact.

## Operating rules

- Preserve raw traces, credentials, exported corpora, and redacted datasets
  outside Git. Keep generated outputs under ignored directories.
- Every redaction rule and source normalizer needs tests with realistic
  structure but no real secrets or private identifiers.
- A corpus publication claim requires a measured row/shard/source manifest,
  redaction report or scanner receipt, licensing boundary, and explicit
  non-claims. Package presence or a successful local import is not publication
  acceptance.
- Changes to extraction, redaction, or export behavior require the full pytest
  suite and focused fixtures covering the changed source/rule.
