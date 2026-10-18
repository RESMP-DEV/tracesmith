# DistillKit pilot on TraceSmith-exported Codex traces

Date: 2026-10-18

## Scope

This records one bounded real-model distillation pilot that consumes the
DistillKit dataset splits shipped by TraceSmith PR #2. It is a pilot receipt,
not a production fine-tune and not a quality claim about agent capability.

## Inputs

- Corpus: TraceSmith live audit of Codex rollouts from 2026-10-16 through
  2026-10-18, 37 raw records, 36 exported conversations, 80 ShareGPT pairs.
- Repair: TraceSmith `f1abfa0` counts native `web_search_call` items, so
  cited-research conversations are no longer treated as tool-free.
- Integration: TraceSmith PR #2, squash merge `aece5c7`, then README commit
  `27c907f` on `main`.
- Dataset: `tracesmith distillkit` produced 55 train rows and 25 eval rows
  grouped into 29 train and 7 eval conversations, seed 17, eval fraction 0.2.
  `verify_distillkit_dataset` matched the source export SHA-256
  `811cea24a4d52c912378c836964f08feac301b5501335d04580e1880e96a695b`.
- Student: `Qwen3-0.6B`. Teacher: `Qwen3-4B-Instruct-2507`. Tokenizer: the
  teacher tokenizer, which is what DistillKit uses.
- Training subset: the first 16 hash-shuffled train rows, one epoch, maximum
  sequence length 2,048, learning rate 5e-6 with cosine decay, cross entropy
  weight 0.5 and top-64 teacher KL weight 0.5.

## Runs

| Device | Train steps | Wall time | Final combined loss | Student weight SHA-256 |
| --- | ---: | ---: | ---: | --- |
| M4 Max, MPS | 16 | 60.25 s | 2.194 | `33fad6f582e893b63796acaa9282cb929528543cda162ee4d22d6eacdf599470` |
| B550, RTX 3090 Ti CUDA | 16 | 13.4 s | 2.196 | `cc6b4bfb184039525526140efad93237a3b66b075ee18f5ad87c2d4665f228b6` |

The two runs used different RNG streams and kernels, so the weight digests
differ, but the per-step loss traces and held-out scores agree. The B550 run
shared GPU 0 with a long-lived embedding service and logged one non-fatal CUDA
allocator OOM warning on a 1.24 GB allocation at step 7 before completing all
16 steps.

## Held-out evaluation

All evaluations use the same 25 eval rows and the teacher tokenizer at maximum
length 2,048.

| Model | Tokens | Mean cross-entropy | Perplexity |
| --- | ---: | ---: | ---: |
| Untouched Qwen3-0.6B | 20,632 | 4.060923486589784 | 58.0279 |
| Distilled pilot, MPS | 20,632 | 3.9170669561534392 | 50.2528 |
| Distilled pilot, B550 CUDA | 20,607 | 3.9164821706022037 | 50.2235 |
| Qwen3-4B teacher | 20,632 | 3.6292985780334326 | 37.6864 |

Held-out cross-entropy fell by 0.14386 nats per token, a 3.54% relative
reduction, and perplexity fell 13.43%. The distilled student still trails the
teacher by 0.28777 nats, so the pilot closed about 33.28% of the measured
student-teacher gap. The MPS and CUDA pilots agree to 0.0006 nats, which is
inside the tokenizer-length difference between the two evaluation passes.

## Receipts

- `~/.cache/alphaheng/tracesmith-runs/20261018-live-audit/distillkit-pilot/`
  - `baseline-eval-ce-teacher-tokenizer.json`
  - `pilot-16-eval-ce-teacher-tokenizer.json`
  - `b550-pilot-16-eval-ce.json`
  - `teacher-eval-ce-teacher-tokenizer.json`
  - `pilot-16.yml`, `pilot-16-b550.yml`, `pilot-16.log`, `b550-pilot-16-train.log`
- TraceSmith tests: 148 passed in the default environment and 148 passed with
  the `distillkit` extra pinned to `datasets<5`.
- Transfer verification: remote student and all three remote teacher shards
  matched local SHA-256 values on B550.

## Non-claims

- This is a single small private slice of redacted agent-trace text, not a task
  benchmark and not a representative agent-quality evaluation.
- No distilled student was published to Hugging Face or any GitHub remote.
- Nothing was pushed to an upstream remote; only `RESMP-DEV/tracesmith` was
  written.
