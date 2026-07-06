# ATTRIBUTION

File-by-file provenance. Each entry links to the upstream path and (once vendored) the commit SHA.

| File | Upstream | Upstream path | License | Changes |
|---|---|---|---|---|
| `agent_trace_share/extract/claude_code.py` | 0xSero/ai-data-extraction | `extract_claude_code.py` | MIT | Refactored to `Extractor` protocol; logic preserved |
| `agent_trace_share/redact/rules/*.py` | RodriMora/agent-trace-redaction-methodology | `scripts/export_redacted_traces.py` (`Redactor` class) | MIT | Split into modules by rule family |
| `agent_trace_share/redact/schema.py` | RodriMora/agent-trace-redaction-methodology | `scripts/export_redacted_traces.py` (`redact_sensitive_field`) | MIT | Extracted as standalone recursive walker |
| `tests/test_canary_redaction.py` | RodriMora/agent-trace-redaction-methodology | `tests/test_canary_redaction.py` | MIT | Adapted to new pipeline shape |
