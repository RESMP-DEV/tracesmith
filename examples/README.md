# Reproducible examples

Regenerate the redacted golden exports from `sample_input.jsonl` with:

```bash
python examples/regenerate.py
```

The generator uses the fixed, non-secret metadata key
`tracesmith-public-example-v1`, so the example trace and project IDs remain
stable across processes. Runtime exports should continue to use a private
`TRACESMITH_METADATA_KEY` rather than this public fixture-only value.

Use `python examples/regenerate.py --check` to verify that the tracked exports
are current without rewriting them.
