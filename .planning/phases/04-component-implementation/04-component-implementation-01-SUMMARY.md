---
phase: 04-component-implementation
plan: 01
subsystem: training
tags: [pytorch, diffusion, trainer, mode-dispatch, tdd]

# Dependency graph
requires:
  - phase: 03-config-schema-bug-fix
    provides: scales = None fix before mode dispatch; ReasoningConfig and ReverseGnnConfig in ModelConfig
provides:
  - elif mode == "rev-reason": data = batch branch in DiffusionTrainer.run_epoch
  - tests/test_diffusion_rev_reason.py covering MODE-01 and MODE-02
affects:
  - 04-02-IterativeBackbone (will execute inside this branch in phase 5)
  - 04-03-ReverseBackbone (will execute inside this branch in phase 5)
  - 05-integration (wires components into the rev-reason elif body)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mode dispatch elif chain: diff-cont / flow-blind / rev-reason / else (diff-discrete fallback)"
    - "TDD RED-GREEN cycle: test file committed before production code"

key-files:
  created:
    - tests/test_diffusion_rev_reason.py
  modified:
    - src2/hashi_puzzle_solver/trainers/diffusion.py

key-decisions:
  - "rev-reason elif placed after flow-blind and before the else diff-discrete fallback — preserves fallthrough guard for unrecognized modes"
  - "data = batch assignment only; no noise injection, no scales computation — clean puzzle state passed directly to model"
  - "TODO(phase-5) comment inside elif body marks the exact wiring point for IterativeBackbone and ReverseBackbone"
  - "Test config fixture requires loss_weights.ce key for compute_combined_loss in legacy.py (Rule 2 auto-fix to test setup)"

patterns-established:
  - "rev-reason mode: batch passed unmodified, no noise injection functions called"
  - "Component flag independence: reasoning.enabled and reverse_gnn.enabled are orthogonal to mode dispatch"

requirements-completed: [MODE-01, MODE-02]

# Metrics
duration: 4min
completed: 2026-03-09
---

# Phase 4 Plan 01: rev-reason Mode Dispatch Summary

**`elif mode == "rev-reason": data = batch` branch inserted in DiffusionTrainer.run_epoch, establishing the clean-graph training path with no noise injection before IterativeBackbone and ReverseBackbone wiring in phase 5**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-09T09:56:45Z
- **Completed:** 2026-03-09T10:00:45Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 2

## Accomplishments

- `elif mode == "rev-reason": data = batch` inserted before the `else` catch-all in `run_epoch`, with a phase-5 TODO comment
- No noise injection function (`inject_noise`, `inject_continuous_noise`, `inject_flow_noise`) can be reached in the rev-reason path
- Both component flags (`reasoning.enabled`, `reverse_gnn.enabled`) independently settable in all four combinations without error
- Test file `tests/test_diffusion_rev_reason.py` created: 5 tests (1 for MODE-01, 4 parametrized for MODE-02)

## Task Commits

Each task was committed atomically:

1. **Task 0: Write failing test stubs for MODE-01 and MODE-02** - `41f7151` (test)
2. **Task 1: Add rev-reason elif branch to DiffusionTrainer.run_epoch** - `c4dc4ae` (feat)

_Note: TDD tasks have separate test commit (RED) and implementation commit (GREEN)_

## Files Created/Modified

- `tests/test_diffusion_rev_reason.py` — Pytest tests for MODE-01 (no noise injection) and MODE-02 (flag independence); 5 tests all GREEN
- `src2/hashi_puzzle_solver/trainers/diffusion.py` — 3-line elif insertion in the mode dispatch block (line 380)

## Decisions Made

- `elif mode == "rev-reason"` placed between `flow-blind` and `else` to preserve the diff-discrete fallthrough guard for unrecognized modes
- Only `data = batch` in the body — no scales computation, no noise call; clean puzzle state is passed directly to the model forward loop
- `TODO(phase-5)` comment inserted as the explicit wiring point for Phase 5 integration

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added `ce` key to test fixture loss_weights dict**
- **Found during:** Task 1 (GREEN phase — running tests after elif insertion)
- **Issue:** `compute_combined_loss` in `losses/legacy.py` uses `loss_weights["ce"]` directly (KeyError if absent). Minimal test config had `"loss_weights": {}`.
- **Fix:** Added `"ce": 1.0` to the `loss_weights` dict in `_make_config()` fixture
- **Files modified:** `tests/test_diffusion_rev_reason.py`
- **Verification:** All 5 tests pass after fix
- **Committed in:** `c4dc4ae` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical config key in test fixture)
**Impact on plan:** Fix was required for tests to run to completion. No scope creep.

## Issues Encountered

- Pre-existing import errors in many legacy tests (`hashi_puzzle_solver.diffusion_engine` not found, `hashi_puzzle_solver.utils` import mismatches) — out of scope, not caused by this plan's changes. Deferred to `deferred-items.md`.

## Next Phase Readiness

- The `elif mode == "rev-reason"` dispatch point exists and is tested
- Phase 5 can wire `IterativeBackbone` and `ReverseBackbone` into the TODO comment location inside this elif body
- No blockers

---
*Phase: 04-component-implementation*
*Completed: 2026-03-09*
