---
phase: 04-component-implementation
plan: 02
subsystem: model
tags: [pytorch, torch-geometric, TransformerConv, GNN, reasoning]

# Dependency graph
requires:
  - phase: 03-config-schema-bug-fix
    provides: ReasoningConfig with steps field wired into model config
provides:
  - IterativeBackbone class: shared-weight TransformerConv applied K times with residual updates
affects:
  - 05-integration (Phase 5 wires IterativeBackbone into HashiGraphModel.forward)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Iterative weight-sharing: single nn.Module instance reused in a loop rather than stacking N distinct layers"
    - "concat=False on TransformerConv fixes output dim to hidden_channels, making residual add unconditional"

key-files:
  created:
    - src2/hashi_puzzle_solver/models/iterative_backbone.py
    - tests/test_iterative_backbone.py
  modified: []

key-decisions:
  - "conv.forward spy instead of patch.object: PyTorch nn.Module rejects non-Module attribute assignment on registered submodules, so we replace the bound method directly"
  - "concat=False enforced unconditionally: guarantees in_channels == out_channels == hidden_channels so residual add needs no shape guard (unlike GraphBackbone)"
  - "steps=1 test uses backbone.eval() + p=0 dropout in manual pass to eliminate randomness — makes output deterministic and directly comparable"

patterns-established:
  - "Iterative backbone pattern: one shared conv + norm in a for-loop with unconditional residual; steps controls loop count"

requirements-completed: [REAS-01, REAS-02]

# Metrics
duration: 7min
completed: 2026-03-09
---

# Phase 4 Plan 02: IterativeBackbone Summary

**Standalone `IterativeBackbone` class delivering REAS-01 and REAS-02: single shared-weight TransformerConv (concat=False) applied K times with residual updates, steps-controlled loop, final_dim==hidden_channels**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-09T09:55:37Z
- **Completed:** 2026-03-09T10:02:00Z
- **Tasks:** 2 (Task 0: RED tests, Task 1: GREEN implementation)
- **Files modified:** 2

## Accomplishments
- Created `IterativeBackbone` — a standalone GNN module that reuses a single TransformerConv K times rather than stacking N distinct layers
- Residual add is unconditional (no shape guard) because concat=False fixes output dim to hidden_channels for any number of attention heads
- All three REAS tests pass: K-iteration call count, steps parameter/shape, and steps=1 identity against manual pass

## Task Commits

Each task was committed atomically:

1. **Task 0: Write failing test stubs for REAS-01 and REAS-02** - `bf5d262` (test)
2. **Task 1: Implement IterativeBackbone class** - `adaf39c` (feat)

_Note: TDD tasks — test committed in RED state, then implementation commit brought tests to GREEN_

## Files Created/Modified
- `src2/hashi_puzzle_solver/models/iterative_backbone.py` - IterativeBackbone class (new file, standalone)
- `tests/test_iterative_backbone.py` - Three automated tests for REAS-01 and REAS-02

## Decisions Made
- **conv.forward spy approach:** PyTorch `nn.Module.__setattr__` rejects assigning a non-Module (like `MagicMock`) to a registered submodule. Replaced the `patch.object` approach with direct `backbone.conv.forward = spy_forward` wrapping the original bound method. This avoids the TypeError while still counting calls accurately.
- **concat=False unconditional:** Unlike `GraphBackbone` which uses a shape guard before the residual add (`if h_in.shape == h.shape`), `IterativeBackbone` fixes concat=False so in_channels always equals out_channels, making the guard unnecessary.
- **No modifications to existing model files:** As per Phase 4 constraint, backbone.py, core.py, factory.py, and __init__.py were not touched.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test spy method — replaced patch.object with conv.forward wrapper**
- **Found during:** Task 1 (GREEN phase — tests failed after implementation was created)
- **Issue:** `patch.object(backbone, "conv", ...)` attempted to assign a MagicMock as a registered nn.Module submodule; PyTorch raises `TypeError: cannot assign 'MagicMock' as child module`
- **Fix:** Removed dead `patch.object` block from Task 0 test; replaced with `backbone.conv.forward = spy_forward` which wraps the original bound method and counts calls without touching the module registry
- **Files modified:** tests/test_iterative_backbone.py
- **Verification:** All 3 tests pass (GREEN)
- **Committed in:** adaf39c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test spy strategy)
**Impact on plan:** Fix required for tests to function. No scope creep; no behavioral changes to implementation.

## Issues Encountered
- Pre-existing `ModuleNotFoundError: No module named 'hashi_puzzle_solver.diffusion_engine'` in `tests/test_adaptive_sampler.py` prevents `pytest tests/ -x -q` from passing. This is unrelated to iterative_backbone changes and was failing before this plan. Logged as out-of-scope per deviation rules.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `IterativeBackbone` is complete and self-contained. Phase 5 (integration) can wire it into `HashiGraphModel.forward()` via `ReasoningConfig.steps`.
- `backbone.py`, `core.py`, `factory.py`, `__init__.py` all unchanged — clean integration surface.

---
*Phase: 04-component-implementation*
*Completed: 2026-03-09*
