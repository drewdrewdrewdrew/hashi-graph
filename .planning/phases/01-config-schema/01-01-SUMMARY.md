---
phase: 01-config-schema
plan: "01"
subsystem: config
tags: [python, dataclass, yaml, bptt, config-schema]

# Dependency graph
requires: []
provides:
  - BpttConfig dataclass with enabled/window/stride/loss_ema_decay fields and __post_init__ validation
  - TrainingConfig.bptt typed field with default_factory
  - HashiModelConfig.from_dict bptt extraction and wiring
  - bptt: block in diffusion_solver_continuous.yaml
affects:
  - 02-bptt-training-loop

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Nested dataclass config pattern: extract sub-dict from training_dict, exclude key from base_dict, pass as BpttConfig(**bptt_dict)"
    - "TDD RED/GREEN: write failing tests before implementation"

key-files:
  created:
    - src2/hashi_puzzle_solver/models/test_bptt_config.py
  modified:
    - src2/hashi_puzzle_solver/models/config.py
    - configs/diffusion_solver_continuous.yaml

key-decisions:
  - "BpttConfig placed before LossWeightsConfig to maintain alphabetical grouping of nested config dataclasses"
  - "loss_ema_decay validation uses half-open interval [0, 1) — decay of 1.0 is excluded as it would make EMA non-decaying"
  - "bptt defaults to enabled=False so existing configs and trainers load without modification"

patterns-established:
  - "Nested config pattern: define dataclass, add default_factory field in TrainingConfig, extract in from_dict exclusion list"
  - "Config validation: __post_init__ raises ValueError with descriptive message including field name and got-value"

requirements-completed: [CFG-01, CFG-02, CFG-03, CFG-04]

# Metrics
duration: 4min
completed: 2026-03-06
---

# Phase 1 Plan 01: Config Schema Summary

**BpttConfig dataclass with window/stride/loss_ema_decay validation wired into TrainingConfig via default_factory and HashiModelConfig.from_dict**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-06T13:22:03Z
- **Completed:** 2026-03-06T13:26:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `BpttConfig` dataclass with `__post_init__` validation (window>=1, stride>=1, loss_ema_decay in [0,1))
- Wired `BpttConfig` into `TrainingConfig` as a typed `bptt` field with `default_factory`
- Updated `HashiModelConfig.from_dict` to extract and pass `bptt_dict` without breaking existing training field loading
- Added `bptt:` block to `diffusion_solver_continuous.yaml` with all four fields matching defaults
- 23 pytest tests written (TDD) covering defaults, validation, TrainingConfig wiring, and from_dict behavior

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for BpttConfig** - `3caf866` (test)
2. **Task 1 GREEN: BpttConfig implementation** - `7372945` (feat)
3. **Task 2: bptt block in YAML** - `c7fb599` (feat)

_Note: TDD task has two commits (test RED -> feat GREEN)_

## Files Created/Modified

- `src2/hashi_puzzle_solver/models/config.py` - Added BpttConfig dataclass (before LossWeightsConfig), bptt field in TrainingConfig, bptt extraction in from_dict
- `src2/hashi_puzzle_solver/models/test_bptt_config.py` - 23 tests covering all behavior cases (created)
- `configs/diffusion_solver_continuous.yaml` - Added bptt: sub-block inside training: after early_stopping:

## Decisions Made

- `BpttConfig` positioned before `LossWeightsConfig` to maintain alphabetical grouping with other nested config classes
- `loss_ema_decay` uses half-open interval `[0, 1)` — value of 1.0 excluded because it would make EMA non-decaying (infinite memory)
- `enabled` defaults to `False` so all existing configs load transparently without any YAML changes required

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `training_cfg.bptt.enabled`, `training_cfg.bptt.window`, `training_cfg.bptt.stride`, `training_cfg.bptt.loss_ema_decay` are all accessible as typed attributes
- Phase 2 (BPTT training loop) can read these fields directly from any loaded `HashiModelConfig`
- No blockers

---
*Phase: 01-config-schema*
*Completed: 2026-03-06*

## Self-Check: PASSED

- FOUND: src2/hashi_puzzle_solver/models/config.py
- FOUND: src2/hashi_puzzle_solver/models/test_bptt_config.py
- FOUND: configs/diffusion_solver_continuous.yaml
- FOUND: .planning/phases/01-config-schema/01-01-SUMMARY.md
- FOUND commit: 3caf866 (test RED)
- FOUND commit: 7372945 (feat GREEN)
- FOUND commit: c7fb599 (feat YAML)
