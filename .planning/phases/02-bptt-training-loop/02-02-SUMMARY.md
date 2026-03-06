---
phase: 02-bptt-training-loop
plan: "02"
subsystem: training
tags: [pytorch, bptt, gradient-checkpointing, sliding-window, ema, diffusion]

# Dependency graph
requires:
  - phase: 02-bptt-training-loop
    plan: "01"
    provides: "bptt_enabled dispatch, step_boundary_states cache populated with detached logit slices, NotImplementedError stub"
provides:
  - "_run_bptt_window helper method on DiffusionTrainer with gradient checkpointing"
  - "Sliding-window BPTT loop wired into run_epoch replacing NotImplementedError stub"
  - "EMA loss scalar bptt_ema reported as training loss for BPTT path"
  - "test_diffusion_bptt.py with 7 passing tests covering window helper and loop behavior"
affects:
  - "Any phase adding training features to DiffusionTrainer"
  - "Future phases tuning BPTT window/stride hyperparameters"

# Tech tracking
tech-stack:
  added: ["torch.utils.checkpoint.checkpoint (gradient checkpointing)"]
  patterns:
    - "Sliding window over step indices with configurable window and stride"
    - "Gradient checkpointing per step to bound activation memory"
    - "EMA smoothing of per-window scalar loss (not model weights)"
    - "retain_graph=True on all but the last window .backward() call"
    - "Single optimizer.step() after all windows have accumulated gradients"

key-files:
  created:
    - src2/hashi_puzzle_solver/trainers/test_diffusion_bptt.py
  modified:
    - src2/hashi_puzzle_solver/trainers/diffusion.py
    - src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py

key-decisions:
  - "checkpoint(use_reentrant=False) chosen for safer gradient checkpointing on modern PyTorch"
  - "window_loss.backward(retain_graph=more_windows) — retain_graph only when more windows remain, freeing graph memory after last window"
  - "bptt_ema initialized to None, set to first window loss.item() (no decay on first window)"
  - "total_batch_loss_value unified variable used by both BPTT and non-BPTT paths for total_loss accounting"
  - "compat test TestBpttEnabledNotImplementedError updated to TestBpttEnabledWindowLoop — stub test replaced with implementation verification"

patterns-established:
  - "TDD RED->GREEN: tests written first in test_diffusion_bptt.py, confirmed failing, then implementation added"
  - "_run_bptt_window does NOT call .backward() — caller (run_epoch) owns the backward/step cycle"
  - "step_boundary_states[start_step] restored at window start; no detach inside window (grad flows freely)"

requirements-completed: [TRN-02, TRN-03, TRN-04, TRN-05, TRN-06]

# Metrics
duration: 3min
completed: 2026-03-06
---

# Phase 02 Plan 02: BPTT Window Loop Summary

**Sliding-window BPTT with gradient checkpointing: `_run_bptt_window` helper + wired run_epoch loop accumulating gradients across overlapping windows before a single `optimizer.step()`**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T13:44:54Z
- **Completed:** 2026-03-06T13:48:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `_run_bptt_window` method added to `DiffusionTrainer`: re-runs model forward for steps [start_step, end_step) using `torch.utils.checkpoint.checkpoint` per step, returns differentiable scalar loss
- `NotImplementedError` stub replaced with sliding-window loop in `run_epoch`: windows at `[0, stride, 2*stride, ...]`, each calls `_run_bptt_window` then `.backward(retain_graph=more_windows)`, single `optimizer.step()` after all windows
- EMA scalar `bptt_ema` updated after each window and reported as the batch loss value
- 7 tests passing in `test_diffusion_bptt.py` covering scalar output, gradient flow, backward safety, window counts for stride=1 and stride=2, EMA value, and single-window equivalence

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement _run_bptt_window helper with gradient checkpointing** - `7b2c883` (feat)
2. **Task 2: Wire window loop + EMA into run_epoch, replace NotImplementedError stub** - `e3487df` (feat)

**Plan metadata:** see final docs commit

_Note: TDD tasks — tests written first (RED), then implementation (GREEN), combined into atomic task commits_

## Files Created/Modified
- `src2/hashi_puzzle_solver/trainers/diffusion.py` - Added `from torch.utils.checkpoint import checkpoint` import; added `_run_bptt_window` method; replaced NotImplementedError stub with window loop + EMA; unified total_batch_loss_value for both BPTT and non-BPTT paths
- `src2/hashi_puzzle_solver/trainers/test_diffusion_bptt.py` - New test file with 7 tests: 3 window helper tests (scalar, grad, backward) + 4 loop integration tests (stride1, stride2, EMA, single-window)
- `src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py` - Updated `TestBpttEnabledNotImplementedError` to `TestBpttEnabledWindowLoop` — stub test replaced with verification that bptt_enabled=True now succeeds

## Decisions Made
- `checkpoint(use_reentrant=False)` — safer mode for modern PyTorch, avoids reentrancy issues with nested autocast
- `retain_graph=True` only when `more_windows` remain — frees computation graph memory after last window's backward pass
- `bptt_ema` initialized to `None`; first window sets it to `wl` directly (no decay on first sample, avoids cold-start bias)
- Unified `total_batch_loss_value` variable: non-BPTT branch sets it to `total_batch_loss.item()`, BPTT branch sets it to `bptt_ema`; single `total_loss += total_batch_loss_value` line covers both paths

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated compat test that tested the now-replaced stub**
- **Found during:** Task 2 verification run (full test suite)
- **Issue:** `TestBpttEnabledNotImplementedError.test_bptt_enabled_raises_not_implemented` tested that `bptt_enabled=True` raises `NotImplementedError`. Plan 02 replaces the stub, so the test correctly failed after implementation.
- **Fix:** Renamed class to `TestBpttEnabledWindowLoop`; replaced `pytest.raises(NotImplementedError)` test with a test verifying the window loop runs successfully and `optimizer.step()` is called once
- **Files modified:** `src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py`
- **Verification:** All 13 tests pass across both compat and bptt test files
- **Committed in:** `e3487df` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — test updated to match implemented behavior)
**Impact on plan:** Required — the stub test was explicitly testing the Plan 01 stub that Plan 02 was designed to replace. Updating it is the correct action.

## Issues Encountered
None - implementation straightforward. Gradient checkpointing with `use_reentrant=False` and mock losses returning `param.sum()` provided clean autograd flow through the tests.

## Next Phase Readiness
- BPTT window loop is fully operational: `_run_bptt_window` + `run_epoch` wired path
- `bptt.enabled=true` in config now activates real gradient signal flowing across consecutive diffusion steps
- `test_diffusion_compat.py` and `test_diffusion_bptt.py` both pass (13 tests total)
- Ready for any phase that tunes BPTT hyperparameters or adds training diagnostics

## Self-Check: PASSED

- FOUND: `src2/hashi_puzzle_solver/trainers/test_diffusion_bptt.py`
- FOUND: `src2/hashi_puzzle_solver/trainers/diffusion.py`
- FOUND: `.planning/phases/02-bptt-training-loop/02-02-SUMMARY.md`
- FOUND commit `7b2c883`: feat(02-02): implement _run_bptt_window with gradient checkpointing
- FOUND commit `e3487df`: feat(02-02): wire BPTT sliding-window loop + EMA into run_epoch

---
*Phase: 02-bptt-training-loop*
*Completed: 2026-03-06*
