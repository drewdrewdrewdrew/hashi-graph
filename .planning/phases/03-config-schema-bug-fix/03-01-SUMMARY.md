---
phase: 03-config-schema-bug-fix
plan: 01
subsystem: config
tags: [dataclass, yaml, config, python, diffusion, reasoning, reverse-gnn]

# Dependency graph
requires:
  - phase: 02-bptt-training-loop
    provides: BpttConfig dataclass pattern and from_dict extraction pattern for nested training sub-configs
provides:
  - ReasoningConfig dataclass with enabled/steps fields and validation
  - ReverseGnnConfig dataclass with enabled/separate_weights/project_embeddings fields
  - ModelConfig.reasoning and ModelConfig.reverse_gnn typed fields
  - HashiModelConfig.from_dict correctly strips and parses model sub-dicts
  - configs/rev_reasoning.yaml reference config for rev-reason training mode
  - scales = None initialization in diffusion.py preventing UnboundLocalError
affects: [04-01-trainer-dispatch, 04-02-iterative-backbone, 04-03-reverse-backbone, 05-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Nested config dataclass extraction: strip sub-dicts from parent dict before **splat, pass as typed sub-config instances"
    - "forward-reference default_factory: use lambda to defer ReasoningConfig/ReverseGnnConfig construction until after classes defined"

key-files:
  created:
    - tests/test_config_reasoning.py
    - configs/rev_reasoning.yaml
  modified:
    - src2/hashi_puzzle_solver/models/config.py
    - src2/hashi_puzzle_solver/trainers/diffusion.py

key-decisions:
  - "ReasoningConfig and ReverseGnnConfig placed after BpttConfig in config.py (alphabetical placement within nested config group)"
  - "ModelConfig uses lambda default_factory for reasoning/reverse_gnn to handle forward-reference ordering"
  - "scales = None minimal one-line fix before mode branch — no other changes to diffusion.py logic"
  - "rev_reasoning.yaml comments out diffusion-specific params rather than removing to preserve discoverability"

patterns-established:
  - "Model sub-dict extraction: reasoning_dict = model_dict.get('reasoning', {}); model_base_dict strips these before ModelConfig(**model_base_dict)"
  - "New nested config dataclasses go after existing nested configs in config.py, before LossWeightsConfig"

requirements-completed: [BUG-01, CFG-05, CFG-06, CFG-07, CFG-08]

# Metrics
duration: 4min
completed: 2026-03-09
---

# Phase 3 Plan 01: Config Schema Bug Fix Summary

**ReasoningConfig and ReverseGnnConfig dataclasses added to config.py with typed ModelConfig fields, from_dict extraction, scales UnboundLocalError fixed, and rev_reasoning.yaml reference config created**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-09T09:32:31Z
- **Completed:** 2026-03-09T09:35:55Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `ReasoningConfig` (enabled=False, steps=5, validates steps>=1) and `ReverseGnnConfig` (enabled=False, separate_weights=True, project_embeddings=True) dataclasses to config.py
- Updated `ModelConfig` with typed `reasoning` and `reverse_gnn` fields; updated `HashiModelConfig.from_dict` to correctly extract and strip these sub-dicts before `**model_base_dict` splat
- Fixed `UnboundLocalError: scales` in diffusion.py by adding `scales = None` before the mode branch — prevents crash when `bptt.enabled=true` with non-diff-cont modes
- Created `configs/rev_reasoning.yaml` with `training.mode: rev-reason`, both model config blocks present, and diffusion-specific params commented out

## Task Commits

1. **Task 1 RED: Failing tests for config dataclasses** - `3cdd142` (test)
2. **Task 1 GREEN: ReasoningConfig, ReverseGnnConfig, ModelConfig, from_dict** - `844ad86` (feat)
3. **Task 2: scales fix + rev_reasoning.yaml** - `0642569` (feat)

## Files Created/Modified
- `src2/hashi_puzzle_solver/models/config.py` - Added ReasoningConfig, ReverseGnnConfig dataclasses; ModelConfig new fields; from_dict sub-dict extraction
- `src2/hashi_puzzle_solver/trainers/diffusion.py` - Added `scales = None` before mode branch
- `configs/rev_reasoning.yaml` - New reference config for rev-reason training mode
- `tests/test_config_reasoning.py` - 10 TDD tests covering all new config behavior

## Decisions Made
- `ModelConfig` uses `field(default_factory=lambda: ReasoningConfig())` rather than `field(default_factory=ReasoningConfig)` to handle the forward-reference ordering (the two new dataclasses are defined after `ModelConfig` in the file, after `BpttConfig`)
- `scales = None` minimal one-line fix only — no other refactoring of the diffusion mode branch
- Diffusion-specific training params in rev_reasoning.yaml are commented out (not removed) to preserve discoverability for future contributors

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Pre-existing test collection errors in `tests/test_adaptive_sampler.py`, `tests/test_diffusion.py`, and several other tests using old module path `hashi_puzzle_solver.*` (without `src2.` prefix). These are unrelated to this plan's changes and existed before. Logged to deferred items.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All Phase 4 plans (04-01 trainer dispatch, 04-02 IterativeBackbone, 04-03 ReverseBackbone) can now reference `cfg.model.reasoning` and `cfg.model.reverse_gnn` fields
- Phase 5 integration plan has the typed config foundation it needs
- No blockers

---
*Phase: 03-config-schema-bug-fix*
*Completed: 2026-03-09*

## Self-Check: PASSED

All files present and all commits verified.
