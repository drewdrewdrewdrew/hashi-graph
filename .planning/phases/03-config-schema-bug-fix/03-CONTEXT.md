---
phase: 3
phase_name: Config Schema + Bug Fix
created: 2026-03-09
status: ready_to_plan
---

# Phase 3 Context: Config Schema + Bug Fix

## User Constraints

- **Modular, reuse existing code** — follow BpttConfig/from_dict patterns exactly; no new abstractions

## Decisions

### Bug Fix (BUG-01)

- **Fix**: Initialize `scales = None` before the mode branch in `run_epoch` (line ~354 in `diffusion.py`)
- **Scope**: Minimal — `scales = None` only; Phase 4 will handle any BPTT+rev-reason interaction
- `_run_bptt_window` signature is left as-is for Phase 3

### ReasoningConfig (CFG-05)

- **Location**: `src2/hashi_puzzle_solver/models/config.py`, before `ModelConfig` (alphabetical with other nested configs)
- **Fields**: `enabled: bool = False`, `steps: int = 5`
- **Validation** (`__post_init__`): `steps >= 1`
- **Pattern**: Mirror `BpttConfig` exactly

### ReverseGnnConfig (CFG-06)

- **Location**: same file, before or alongside `ReasoningConfig`
- **Fields**: `enabled: bool = False`, `separate_weights: bool = True`, `project_embeddings: bool = True`
- **Validation**: none (all booleans)
- **Pattern**: Mirror `BpttConfig` style (no `__post_init__` needed)

### ModelConfig updates (CFG-07)

- Add typed fields with `field(default_factory=...)`:
  ```python
  reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
  reverse_gnn: ReverseGnnConfig = field(default_factory=ReverseGnnConfig)
  ```
- **from_dict changes in `HashiModelConfig.from_dict`**: extract `reasoning` and `reverse_gnn` sub-dicts from `model_dict` before constructing `ModelConfig`, strip them from the flat dict — same pattern as `bptt` in `training_dict`
- Keys to strip from `model_dict`: `["reasoning", "reverse_gnn"]`

### rev_reasoning.yaml (CFG-08)

- **Source**: copy from `configs/diffusion_solver_continuous_bptt.yaml`
- **Changes**:
  - `training.mode: rev-reason`
  - Comment out all diffusion-specific params (`sigma_max`, `scale_min`, `scale_max`, `alpha_power`, `zero_signal_prob`, `diffusion_step_lr`, `flush_first_step`, `eval_rollout_interval`, `diffusion_max_steps`, `use_adaptive_sampler`)
  - Drop `num_inference_steps_training`
  - Add `model.reasoning` block: `enabled: false`, `steps: 5`
  - Add `model.reverse_gnn` block: `enabled: false`, `separate_weights: true`, `project_embeddings: true`
  - Keep `bptt:` block (disabled) for future use
  - Keep all other training hyperparams as-is (lr, batch_size, n_blocks, etc.)

## Code Context

### Files modified

| File | Change |
|------|--------|
| `src2/hashi_puzzle_solver/models/config.py` | Add `ReasoningConfig`, `ReverseGnnConfig` dataclasses; add fields to `ModelConfig`; update `HashiModelConfig.from_dict` |
| `src2/hashi_puzzle_solver/trainers/diffusion.py` | Add `scales = None` before mode branch in `run_epoch` |
| `configs/rev_reasoning.yaml` | New file (CFG-08) |

### Key patterns to reuse

- `BpttConfig` dataclass → template for `ReasoningConfig` / `ReverseGnnConfig`
- `HashiModelConfig.from_dict` bptt extraction (lines 222–229) → template for model sub-dict extraction
- `diffusion_solver_continuous_bptt.yaml` → template for `rev_reasoning.yaml`

### Bug location

```python
# diffusion.py ~line 354 — scales is only set inside mode == "diff-cont" branch
# Fix: add before the if/elif/else block:
scales = None
```

All BPTT uses of `scales` are inside the `mode == "diff-cont"` path except line 545 (`_run_bptt_window`) and line 566 (`_refill_buffer`), both of which are also guarded by diff-cont checks in practice — but the variable reference itself causes the crash without the init.
