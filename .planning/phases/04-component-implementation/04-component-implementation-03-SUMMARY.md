---
phase: 04-component-implementation
plan: 03
subsystem: model
tags: [pytorch, gnn, transformer-conv, reverse-backbone, graph-neural-network]

# Dependency graph
requires:
  - phase: 03-config-schema-bug-fix
    provides: ReverseGnnConfig dataclass with enabled/separate_weights/project_embeddings fields
  - phase: 04-component-implementation
    provides: GraphBackbone in backbone.py (template for mirror construction)
provides:
  - ReverseBackbone class in src2/hashi_puzzle_solver/models/reverse_backbone.py
  - object.__setattr__ weight-sharing pattern for safe submodule reference holding
  - self.projection = Linear(2*final_dim, hidden_channels) registered for Phase 5 use
affects: [05-integration, HashiGraphModel.forward weight sharing, optimizer parameter counting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "object.__setattr__ bypass pattern for holding nn.Module reference without submodule registration"
    - "TDD RED/GREEN for standalone PyTorch module with weight-sharing semantics"

key-files:
  created:
    - src2/hashi_puzzle_solver/models/reverse_backbone.py
    - tests/test_reverse_backbone.py
  modified: []

key-decisions:
  - "forward() returns raw reverse embeddings only — concatenation and projection call deferred to Phase 5 (HashiGraphModel.forward)"
  - "separate_weights=False uses object.__setattr__(self, '_shared_backbone', fwd) to prevent double-parameter registration in optimizer"
  - "project_embeddings=True registers Linear(2*final_dim, hidden_channels) as self.projection — available but not called in Phase 4 forward()"

patterns-established:
  - "Weight-sharing without submodule registration: object.__setattr__ bypasses nn.Module.__setattr__ for plain attribute storage"
  - "Mirror construction from forward backbone: reads convs[0].in_channels, convs[0].heads, convs[0].edge_dim, len(convs), dropout, gnn_type"

requirements-completed: [REVG-01, REVG-02, REVG-03]

# Metrics
duration: 10min
completed: 2026-03-09
---

# Phase 4 Plan 03: ReverseBackbone Component Summary

**Standalone ReverseBackbone GNN module that runs GraphBackbone on reversed edges (edge_index.flip(0)) with optional weight sharing via object.__setattr__ and a projection layer stub for Phase 5 integration**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-09T09:55:39Z
- **Completed:** 2026-03-09T10:05:30Z
- **Tasks:** 2 (Task 0 RED + Task 1 GREEN)
- **Files modified:** 2

## Accomplishments
- Implemented ReverseBackbone with correct edge reversal (REVG-01: output shape equals input h)
- Established safe weight-sharing pattern: object.__setattr__ prevents double-registration in optimizer (REVG-02)
- Registered self.projection = Linear(2*final_dim, hidden_channels) as stub for Phase 5 concat+project flow (REVG-03)
- All 3 REVG tests pass; no existing files modified

## Task Commits

Each task was committed atomically:

1. **Task 0: Write failing test stubs for REVG-01, REVG-02, REVG-03** - `18c25f8` (test)
2. **Task 1: Implement ReverseBackbone class** - `af75e57` (feat)

_Note: TDD tasks committed separately (RED test file, then GREEN implementation)_

## Files Created/Modified
- `src2/hashi_puzzle_solver/models/reverse_backbone.py` - ReverseBackbone class: flips edge_index, runs backbone, optional projection layer
- `tests/test_reverse_backbone.py` - 3 tests covering REVG-01 (shape), REVG-02 (weight semantics), REVG-03 (projection dims)

## Decisions Made
- `forward()` returns raw reverse embeddings — Phase 5 handles concatenation of `[fwd_h, rev_h]` and calls `self.projection` after obtaining both halves
- `object.__setattr__` chosen over normal attribute assignment because `self.backbone = forward_backbone` would call `nn.Module.__setattr__` and register the forward backbone as a submodule of ReverseBackbone, causing its parameters to appear twice in `model.parameters()` and receive double gradient updates
- Mirror construction reads `convs[0].in_channels`, `convs[0].heads`, `convs[0].edge_dim` directly from the forward backbone's first conv layer

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing unrelated test collection errors in suite (tests importing `hashi_puzzle_solver.diffusion_engine` which doesn't exist). Not caused by this plan, not fixed (out-of-scope). The reverse-backbone-specific test suite and config/data tests all pass green.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `ReverseBackbone` class is ready for Phase 5 wiring into `HashiGraphModel.forward()`
- Phase 5 should: (1) instantiate `ReverseBackbone` from `ReverseGnnConfig`, (2) call `reverse_backbone(h, edge_index)` alongside forward backbone, (3) concatenate `[fwd_h, rev_h]`, (4) call `reverse_backbone.projection(cat)` to compress to `hidden_channels`
- No blockers

---
*Phase: 04-component-implementation*
*Completed: 2026-03-09*
