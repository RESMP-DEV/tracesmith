# uv.lock regeneration - 2026-10-24

Branch `consolidation/uv-lock-20261024` from owned `origin/main`
`e50730826adf7f5aa1368a5d56442241585c7a36`.

## Objective

Generate the complete current dependency lock from `pyproject.toml`, including
the `privacy-filter`, `publish`, `distillkit`, and `dev` extras. Do not merge
either stale historical lock.

## Result

A fresh networked `uv lock` resolved 94 package entries (84 distinct names).
The historical `recovery/2026-10-02-uv-lock` had 66 entries (62 distinct names).
The current resolution removes no package name and adds the 22 transitive names
required by the current `datasets`/DistillKit and HTTP stack. Representative
version movements include `torch 2.14.0 -> 2.14.1`,
`transformers 5.17.0 -> 5.20.2`, `certifi 2026.7.22 -> 2026.10.23`, and
`datasets` entering at `4.8.5`. Multiple versions remain only where Python
markers require them.

The first worker attempt was offline and correctly refused to fabricate a lock:
`uv lock --offline` exited 1 because current `simple-v26` metadata was missing
for `pytest-cov`, `coverage`, and `exceptiongroup`, and its sandbox could not
resolve PyPI. The owning parent then completed the same operation with network
access. No historical lock bytes were copied.

## Verification

All commands ran in the isolated integration worktree with the configured
machine cache location represented here as `<UV_CACHE_DIR>`.

| Command | Exit | Result |
| --- | ---: | --- |
| `uv lock` | 0 | Resolved 94 packages |
| `uv lock --check` | 0 | Lock is current |
| `uv sync --frozen --all-extras` | 0 | Installed the complete locked environment |
| `uv run --frozen --all-extras pytest -q` | 0 | 156 passed, 0 failed |

The two tests skipped in the worker's default-extra run were exercised by the
all-extras run; the complete suite reports 156 passed.

## Non-claims

- This proves manifest/lock agreement and the complete local test suite at the
  generated lock; it does not execute DistillKit training.
- It does not prove platform coverage beyond the universal lock metadata and
  local macOS installation.
- No historical recovery branch, review branch, or remote ref was deleted.
