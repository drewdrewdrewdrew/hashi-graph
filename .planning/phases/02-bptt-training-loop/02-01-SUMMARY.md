---
phase: 02-bptt-training-loop
plan: "01"
subsystem: training
tags: [bptt, diffusion, pytorch, training-loop, tdd]

# Dependency graph
requires:
  - phase: 01-config-schema
    provides: BpttConfig dataclass with enabled/window/stride/loss_ema_decay fields
provides:
  - BPTT-aware run_epoch with disabled-path guard and step-state cache
  - Backward-compat regression tests confirming disabled path is byte-for-byte identical
  - step_boundary_states list populated with detached edge-logit tensors per inference step when enabled
  - NotImplementedError stub for Plan 02 BPTT window loop
affects:
  - 02-02 (BPTT window loop that replaces the stub)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BPTT disabled guard: bptt_enabled = (_bptt.get('enabled', False) if isinstance(_bptt, dict) else _bptt.enabled) and training"
    - "Step-state cache: step_boundary_states list of detached .clone() edge-logit slices"
    - "TDD: RED commit (failing tests) then GREEN commit (implementation) pattern"

key-files:
  created:
    - src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py
  modified:
    - src2/hashi_puzzle_solver/trainers/diffusion.py

key-decisions:
  - "bptt_enabled includes 'and training' guard so eval always uses the existing no_grad path regardless of config"
  - "step_boundary_states stores detached clones to avoid holding graph memory across steps"
  - "BPTT enabled path raises NotImplementedError stub — Plan 02 replaces the raise with the window loop"
  - "Total batch loss reference in accumulation (total_loss +=) is safe because the NotImplementedError exits the batch before that line"

patterns-established:
  - "Deviation in run_epoch gated on bptt_enabled bool derived at top of method — single computed flag, used at two sites"
  - "TDD two-commit pattern: test(RED) -> feat(GREEN) for each TDD task"

requirements-completed: [TRN-01, COMP-01, COMP-02]

# Metrics
duration: 2min
completed: 2026-03-06
---

# Phase 2 Plan 01: BPTT Dispatch and Step-State Cache Summary

**BPTT-aware run_epoch with disabled-path compatibility guard and per-step edge-logit caching using detached clones, gated by a single bptt_enabled bool**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-06T13:40:37Z
- **Completed:** 2026-03-06T13:42:39Z
- **Tasks:** 1 (TDD: 2 commits — test + feat)
- **Files modified:** 2

## Accomplishments
- `bptt_enabled` extracted at top of `run_epoch` from dict-or-dataclass config, gated on `and training` so eval is always unaffected
- `step_boundary_states` list initialized per-batch and populated with one detached `.clone()` of the `bridge_logits_idx:+3` edge-attr slice per inference step when enabled
- Backward dispatch: disabled path unchanged (stack+mean+backward+step); enabled path raises `NotImplementedError` stub for Plan 02
- 6 regression tests covering: disabled backward called, disabled zero_grad called, enabled raises NotImplementedError (not AttributeError), cache population logic, eval path unaffected

## Task Commits

Each task was committed atomically via TDD:

1. **Task 1 (RED): Failing tests** - `d188836` (test)
2. **Task 1 (GREEN): BPTT dispatch + cache** - `2abe3ee` (feat)

## Files Created/Modified
- `src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py` - 6 backward-compat regression tests for BPTT dispatch
- `src2/hashi_puzzle_solver/trainers/diffusion.py` - Added bptt_enabled extraction, step_boundary_states list, caching inside step loop, backward/optimizer dispatch block

## Decisions Made
- `bptt_enabled` includes `and training` so the eval path never enters the enabled branch — the plan specifies this explicitly and it's critical for correctness
- Detached `.clone()` chosen for cache entries to avoid holding computation graph memory across steps
- Stub raises `NotImplementedError` rather than silently no-oping so Plan 02 integration is explicit and testable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None — the minimal mock pattern from the plan worked cleanly. The two disabled-path tests passed before the implementation (they test existing behavior), and only `test_bptt_enabled_raises_not_implemented` failed in RED, confirming the correct single failing test for TDD.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 02 can now: iterate `step_boundary_states` list, implement sliding-window re-runs with gradient flow, and replace the `NotImplementedError` stub with the window-loss backward + optimizer step
- All backward-compat guarantees are verified by the regression test suite
- `bridge_logits_idx` is confirmed present and used consistently

---
*Phase: 02-bptt-training-loop*
*Completed: 2026-03-06*

## Self-Check: PASSED

- FOUND: src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py
- FOUND: src2/hashi_puzzle_solver/trainers/diffusion.py
- FOUND: .planning/phases/02-bptt-training-loop/02-01-SUMMARY.md
- FOUND commit: d188836 (test RED)
- FOUND commit: 2abe3ee (feat GREEN)
