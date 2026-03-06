---
status: complete
phase: 02-bptt-training-loop
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md]
started: 2026-03-06T14:00:00Z
updated: 2026-03-06T14:05:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Test suite passes
expected: Run the BPTT test suite. All 13 tests should pass (6 compat + 7 BPTT). Command: python -m pytest src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py src2/hashi_puzzle_solver/trainers/test_diffusion_bptt.py -v — Expected: 13 passed, 0 failed, 0 errors.
result: pass

### 2. Backward compatibility — disabled path unchanged
expected: With bptt.enabled=false (the default), a training call invokes optimizer.step() exactly once and no BPTT-specific code runs. bptt_enabled flag present with `and training` eval guard confirmed.
result: pass

### 3. BPTT enabled — window loop activates
expected: With bptt.enabled=true, the sliding-window loop runs _run_bptt_window per window and calls optimizer.step() exactly once after all windows. _run_bptt_window exists, sliding window range with stride wired in run_epoch confirmed.
result: pass

### 4. EMA loss reported
expected: When bptt.enabled=true, the batch loss returned in results["loss"] is the EMA-smoothed scalar across windows. bptt_ema with decay confirmed in run_epoch source.
result: pass

### 5. Gradient checkpointing active
expected: _run_bptt_window uses torch.utils.checkpoint.checkpoint(use_reentrant=False) per step so activation memory does not grow linearly with window size.
result: pass

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
