---
phase: 05-integration
plan: 01
subsystem: model
tags: [pytorch, gnn, transformer-conv, iterative-backbone, reverse-backbone, tdd]

# Dependency graph
requires:
  - phase: 04-component-implementation
    provides: IterativeBackbone and ReverseBackbone standalone modules with tested contracts
provides:
  - HashiGraphModel with optional iterative_backbone and reverse_backbone submodules
  - ModelFactory computes edge_head_node_dim and builds optional components from config flags
  - Interleaved rev-reasoning composition block in HashiGraphModel.forward()
  - Integration test suite gating all four Phase 5 success criteria
affects: [training, evaluation, inference, phase-06-if-any]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - None-guard composition block in forward() keeps baseline path byte-for-byte identical
    - edge_head_node_dim computed before EdgeHead construction so dim math is centralised
    - Interleaved loop: forward conv + reverse pass + cat + project + residual per step
    - Constructor-time validation raises ValueError when project_embeddings=False with both flags

key-files:
  created:
    - tests/test_hashi_graph_model_integration.py
  modified:
    - src2/hashi_puzzle_solver/models/core.py
    - src2/hashi_puzzle_solver/models/factory.py

key-decisions:
  - "05-01: composition block placed after backbone call in step 6; all three optional paths fully gated by None checks so baseline is byte-for-byte identical"
  - "05-01: interleaved loop accesses iterative_backbone.conv directly (not forward()) to interleave reverse pass before residual; consistent with plan spec"
  - "05-01: project_embeddings=False + both flags enabled raises ValueError at construction time — fail fast before any forward pass"
  - "05-01: edge_head_node_dim computed in factory.py before EdgeHead/ProphetHead construction; reasoning.enabled alone leaves dim unchanged"

patterns-established:
  - "Phase 5 composition: optional component None-guards keep existing forward paths unchanged when flags are disabled"
  - "Factory dim math: edge_head_node_dim = hidden_channels (proj=True), 2*final_dim (proj=False), or backbone.final_dim (no reverse)"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-03-09
---

# Phase 5 Plan 01: Integration Summary

**HashiGraphModel end-to-end rev-reasoning via interleaved IterativeBackbone + ReverseBackbone composition with correct EdgeHead dim in all four flag combinations**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-09T10:34:10Z
- **Completed:** 2026-03-09T10:38:30Z
- **Tasks:** 2 (Task 0 RED + Task 1 GREEN)
- **Files modified:** 3

## Accomplishments

- HashiGraphModel now accepts optional `iterative_backbone` and `reverse_backbone` submodules; when both are None forward() is byte-for-byte identical to baseline
- Interleaved rev-reasoning loop: each iteration applies `iterative_backbone.conv + norm + relu + dropout`, then `reverse_backbone`, then `cat + projection + residual` — confirmed by spy call count
- ModelFactory builds optional components from config flags and computes `edge_head_node_dim` before EdgeHead/ProphetHead construction, correctly handling all four flag combinations
- 11-test integration suite covers all Phase 5 success criteria; all tests green

## Task Commits

1. **Task 0: Write failing integration test scaffold (Wave 0 RED)** - `a35cc43` (test)
2. **Task 1: Wire optional backbones into HashiGraphModel and ModelFactory (GREEN)** - `15b18e0` (feat)

## Files Created/Modified

- `tests/test_hashi_graph_model_integration.py` - 11 integration tests covering SC-1 through SC-4; spy-based call count verification for interleaved loop
- `src2/hashi_puzzle_solver/models/core.py` - Added optional backbone params, constructor validation, and composition block in step 6
- `src2/hashi_puzzle_solver/models/factory.py` - Added optional component construction and `edge_head_node_dim` computation

## Decisions Made

- Composition block placed after backbone call in step 6; None guards preserve baseline path exactly
- Interleaved loop accesses `iterative_backbone.conv` directly (not `forward()`) so the reverse pass can be inserted between the conv and the residual update — as specified in the plan
- `project_embeddings=False` with both flags enabled raises `ValueError` at construction time — fail fast
- `edge_head_node_dim` computed in factory.py before head construction; reasoning-only does not change node dim

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed edge feature dimension mismatch in test batch fixture**
- **Found during:** Task 0 (RED test scaffold)
- **Issue:** Initial `_make_batch` provided edge_attr with 1 column but the EdgeEncoder always has at least `inv_dx`, `inv_dy`, and `is_meta` (when categorical types disabled), plus 2 edge label columns = 5 columns total
- **Fix:** Updated `_make_batch` to produce edge_attr of shape `(n_edges, 5)` matching the EdgeFeatureManager schema for the test config
- **Files modified:** tests/test_hashi_graph_model_integration.py
- **Verification:** test_flags_disabled_baseline passes after fix
- **Committed in:** a35cc43 (Task 0 commit)

**2. [Rule 1 - Bug] Fixed EdgeHead input dim mismatch in test config**
- **Found during:** Task 0 (RED test scaffold)
- **Issue:** Default `edge_concat_global_meta=True` adds `node_hidden_dim` to the MLP input at construction, but with no global meta nodes in test graphs (node_type is None), the runtime skips that concatenation — causing a shape mismatch (48 vs 32)
- **Fix:** Added `edge_concat_global_meta=False` to the test ModelConfig so the MLP is built with 2*node_hidden_dim matching runtime behavior
- **Files modified:** tests/test_hashi_graph_model_integration.py
- **Verification:** All baseline tests pass after fix
- **Committed in:** a35cc43 (Task 0 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs in test fixture setup)
**Impact on plan:** Both fixes necessary for test correctness. No scope creep — production code unaffected.

## Issues Encountered

- EdgeFeatureManager always includes `inv_dx`, `inv_dy` and `is_meta` as base edge features (even with most features disabled), requiring test batch to provide at least 5 columns. This is pre-existing behavior, not a regression.

## Next Phase Readiness

- Phase 5 all success criteria met (SC-1 through SC-4)
- Full end-to-end rev-reasoning mode is functional; ready for training integration
- No blockers

---
*Phase: 05-integration*
*Completed: 2026-03-09*

## Self-Check: PASSED

- FOUND: tests/test_hashi_graph_model_integration.py
- FOUND: src2/hashi_puzzle_solver/models/core.py
- FOUND: src2/hashi_puzzle_solver/models/factory.py
- FOUND: .planning/phases/05-integration/05-integration-01-SUMMARY.md
- FOUND commit: a35cc43 (test RED phase)
- FOUND commit: 15b18e0 (feat GREEN phase)
