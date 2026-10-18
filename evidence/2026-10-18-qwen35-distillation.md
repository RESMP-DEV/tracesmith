# Qwen3.5 distillation and Infatoshi trace import

Date: 2026-10-18

## Scope

This receipt records the corrected Qwen3.5-family run, the external Infatoshi
trace import path, and the fixed held-out cross-entropy gate. It does not claim
broad coding-agent capability from cross-entropy alone.

## Architecture and baseline controls

- Student and teacher are both Qwen3.5-family models with 248,320 model rows,
  248,077 concrete tokenizer entries, and tokenizer vocabulary SHA-256
  `6dd8168c97763714677819a072bed8eb31f71fca3d1506966d7563901393e40f`.
- Live preflight and the active run both report `qwen3_5` /
  `qwen3_5_text`; receipt:
  `/home/kearm/training/runs/qwen35-0p8b-fable20k/architecture_guard.json`.
- The fixed evaluation is 200 rows, 125,004 tokens, maximum length 1,024.
- The untouched student reproduced CE `2.0192924497192455` and the Qwen3.5-4B
  teacher reproduced CE `1.8913045364823717` exactly on GPU 1:
  `/home/kearm/training/eval/qwen35-0p8b-baseline-ce200-rerun.json` and
  `/home/kearm/training/eval/qwen35-4b-teacher-ce200-rerun.json`.
- A train/eval audit found zero shared session IDs and zero shared message
  hashes across the 20,000 train and 2,000 eval rows.

## In-flight Fable run

- Detached tmux session: `qwen35-distill-full`.
- Config: `/home/kearm/training/qwen35-distill-full.yml`.
- Log: `/home/kearm/training/logs/qwen35-fable20k.log`.
- Checkpoint 1750 measured on the fixed slice at CE `1.2582032752454362`
  (perplexity `3.5190930366516113`):
  `/home/kearm/training/eval/qwen35-0p8b-fable20k-checkpoint1750-ce200.json`.
  This is a live intermediate probe, not the final-model acceptance result.

## External Infatoshi imports

- TraceSmith `main` commits:
  - `7c43376` adds the path-based Claude JSONL importer.
  - `aafb50f` accepts tool-first sessions.
  - `deb3e1d` keeps Claude tool results in Qwen-compatible user-role text.
  - `efb69f8` requires uppercase environment-secret names and fixes the four
    Hard-verifier false positives without weakening real secret detection.
- HFD fully verified:
  - `Infatoshi/kernelbench-mega-traces` at `3f86d538839844437c95b5ce45874ce362f7ce54`,
    140 files and 169,639,873 bytes.
  - `Infatoshi/kernelbench-hard-traces` at `c2c764c8b5a5fdad493592a2fde1055684e0f064`,
    745 files and 615,621,979 bytes.
- Qwen-compatible imports:
  - Mega: 137 JSONLs scanned, 125 conversations, 12 skipped, zero failed lines.
  - Hard: 740 JSONLs scanned, 696 conversations, 44 skipped, zero failed lines.
- Redaction and verification passed for both. Hard initially stopped on four
  false-positive `tokens =` matches; the corrected verifier passed without
  bypassing the gate. Hard export retained 686 conversations and split 549/137.

## Windowed training mix

- Full Mega sessions had median 169,754 tokens and only 1/125 fit 1,024 tokens,
  so whole-conversation insertion would hide assistant targets behind
  DistillKit's hard truncation.
- The final mix contains 17,569 Fable rows and 2,431 assistant-bearing
  Infatoshi windows (2,058 Hard and 373 Mega), for 20,000 train rows total.
- Validation found 2,431/2,431 windows with assistant targets, 2,354 with tool
  signal, maximum window length 979 tokens, and zero windows over 1,024.
- The 2,000 evaluation rows are exactly unchanged, with canonical SHA-256
  `53823a7bbbab2fe721b68bfe17e8b743e7279641795361ca946bf7dbdf4b5fdd`.
- Dataset and validation:
  `/home/kearm/training/data/qwen35-fable17569-infatoshi2431-windows/`.

## Non-claims

- Cross-entropy and perplexity do not prove task-level coding-agent quality.
- Windowing preserves assistant-bearing segments, not every token from every
  source conversation.
- The final Fable model and the Infatoshi-window model remain pending live
  completion and final held-out evaluation at the time of this entry.

## Learning-rate provenance and measured study

The original full-run value `3e-6` was a conservative hand choice inherited from
the earlier `5e-6` Qwen3 pilot; it was not measured for this corpus. A bounded
sequential study then ran nine successful real-model trials. Each trial started
from the same Qwen3.5-0.8B student, consumed the same 800-row windowed train
slice for 100 optimizer steps, used the same online Qwen3.5-4B teacher, and was
scored on the same fixed 200-row/125,004-token held-out slice.

- Hand-set `3e-6`: CE `1.7465055597590995`.
- Best measured LR `9.668806547836171e-6`: CE `1.446322741842183`.
- The best probe remained numerically stable; its logged gradient norms ended
  near `5.6` rather than diverging.
- Study summary:
  `/home/kearm/training/studies/qwen35-lr/summary.json`.
- The full Infatoshi-window config was updated to the measured best value and
  its pre-study config is retained as
  `/home/kearm/training/qwen35-distill-windows.pre-lr-study.yml`.
- One trial-0 launcher attempt failed before training because of a bad study
  executable path; it is recorded as failed and excluded from the nine
  successful measurements.

## Attention and linear-attention capability

The Qwen3.5-0.8B text model has 24 layers: 18 linear-attention layers and 6
full-attention layers. The active environment already uses optimized kernels
for the linear path:

- `flash-linear-attention==0.5.2`; direct wrapper introspection showed the
  FLA `chunk_gated_delta_rule` implementation is selected.
- `causal_conv1d==1.7.0`.
- PyTorch `2.14.1+cu130` with SDPA's flash backend enabled.

A restricted live SDPA probe on the RTX3090 Ti executed the PyTorch flash
forward kernel and produced finite forward/backward values at head dimensions
64, 128, and 256. At Qwen's head dimension 256, profiler evidence showed both
the split-KV flash forward kernel and its combine kernel. Receipt:
`/home/kearm/training/eval/sdpa-flash-capability-probe.json`.

FlexAttention imports and completed a small causal forward/backward probe, but
without `torch.compile` it warned that it materialized the full score matrix.
It is not wired into the Qwen3.5 model path. The separate `flash-attn` package
is also absent; installing it would not obviously improve the six full-attention
layers because SDPA is already dispatching to PyTorch's flash kernel. A live
model-level A/B roofline is the required gate before replacing SDPA or wiring
FlexAttention. Receipt:
`/home/kearm/training/eval/attention-capability-probe.json`.

The measured-LR Infatoshi-window run was launched in detached tmux session
`qwen35-infatoshi-windows` after the study. Its architecture guard passed and
early step-16 behavior was stable. This entry does not claim that run's final
quality before its identical held-out evaluation completes.
