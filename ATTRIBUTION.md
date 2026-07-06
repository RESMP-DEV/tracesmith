# ATTRIBUTION

File-by-file provenance. Each entry links to the upstream path and (once vendored) the commit SHA.

| File | Upstream | Upstream path | Upstream commit | License | Changes |
|---|---|---|---|---|---|
| `agent_trace_share/extract/claude_code.py` | 0xSero/ai-data-extraction | `extract_claude_code.py` | `b7520c48b2bb46d5a0d3257e80ca1a59670d5e37` | MIT | Refactored to `Extractor` protocol; logic preserved |
| `agent_trace_share/redact/rules/*.py` | RodriMora/agent-trace-redaction-methodology | `scripts/export_redacted_traces.py` (`Redactor` class, lines 261–408) | `c696bc72ad4c18f3d5b79fe4d54af00a2fd1c6b6` | MIT | Split into modules by rule family; regex/category/replacement ported verbatim, composed in upstream interleaved order via `UPSTREAM_ORDER` |
| `agent_trace_share/redact/schema.py` | RodriMora/agent-trace-redaction-methodology | `scripts/export_redacted_traces.py` (`redact_sensitive_field`) | `<pending>` | MIT | Extracted as standalone recursive walker |
| `tests/test_canary_redaction.py` | RodriMora/agent-trace-redaction-methodology | `tests/test_canary_redaction.py` | `<pending>` | MIT | Adapted to new pipeline shape |
