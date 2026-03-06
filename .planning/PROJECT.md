# BPTT Training for Hashi Diffusion Solver

## What This Is

An extension to the existing `DiffusionTrainer` that adds Backpropagation Through Time (BPTT) as a configurable training option. Currently, multi-step diffusion training runs each step independently under `torch.no_grad()` between transitions — gradients cannot flow across steps. This project adds a sliding-window BPTT mode that removes that barrier within a window, allowing the model to learn multi-step coordination rather than only per-step correctness.

## Core Value

The model learns to make decisions that are good for a sequence of steps, not just the next step — relaxing myopic per-step optimization with a longer-horizon gradient signal.

## Requirements

### Validated

- ✓ Multi-step diffusion training via `num_inference_steps_training` — existing
- ✓ Per-step loss accumulation (mean) with `.backward()` — existing
- ✓ Config-driven model/training via YAML + typed dataclasses (`TrainingConfig`) — existing
- ✓ `DiffusionTrainer` in `src2/hashi_puzzle_solver/trainers/diffusion.py` — existing

### Active

- [ ] BPTT sliding-window training mode (window size + stride configurable in YAML)
- [ ] Gradient checkpointing within each window (memory-first)
- [ ] Cached step-boundary states for reuse across overlapping windows
- [ ] Gradient accumulation across overlapping windows (each step gets contributions from all windows covering it)
- [ ] Window-averaged per-step loss (same per-step signal, gradients now flow through time)
- [ ] EMA smoothing on window-averaged loss value
- [ ] `bptt` config block in `diffusion_solver_continuous.yaml` and `TrainingConfig` dataclass

### Out of Scope

- Full sequence BPTT (entire `num_inference_steps_training` as one window) — memory cost is impractical; window approach is intentional
- EMA of model weights — user specified loss-value EMA only
- Changes to inference/eval rollout — BPTT is training-only

## Context

The training loop in `src2/hashi_puzzle_solver/trainers/diffusion.py` runs a `for train_step in range(num_inference_steps_training)` loop. Between steps, the next input is computed under `torch.no_grad()` (line ~355), severing the computation graph. The total loss is `torch.stack(step_losses).mean()` — steps are averaged but independent.

BPTT replaces the `no_grad` transitions within a window. The implementation must:
1. Cache the graph state (edge logits tensor) at each step boundary during the forward pass
2. For each window `[i, i+window)`, re-run the forward pass with gradient enabled, using the cached boundary state as the start point
3. Use `torch.utils.checkpoint.checkpoint` within each window to avoid storing all intermediate activations
4. Accumulate `.backward()` calls across all windows before `.step()`

## Constraints

- **Tech stack**: PyTorch + PyG; must stay compatible with existing `DiffusionTrainer` structure
- **Config**: New settings go in a `bptt:` sub-block under `training:` in YAML; mirror in `TrainingConfig` dataclass in `config.py`
- **Backward compat**: When `bptt.enabled: false` (default), training is identical to current behavior
- **Memory**: Gradient checkpointing required within windows; full BPTT across all steps is not a target

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Sliding window over full-sequence BPTT | Memory constraint; overlapping windows still propagate multi-step signal | — Pending |
| Gradient checkpointing (not detach-at-boundary) | User prioritizes memory over speed; checkpointing costs compute not memory | — Pending |
| Accumulate gradients across overlapping windows | Overlapping steps get signal from all windows covering them | — Pending |
| EMA on window loss scalar (not model weights) | Stabilize the training loss signal across windows | — Pending |

---
*Last updated: 2026-03-06 after initialization*
