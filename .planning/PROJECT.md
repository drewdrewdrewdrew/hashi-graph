# Hashi Diffusion Solver — Training Infrastructure

## What This Is

An extension to the existing `DiffusionTrainer` that adds configurable training modes on top of the base diffusion loop. v1.0 added sliding-window BPTT so gradients flow across diffusion steps. v1.1 adds two new modes: **reasoning** (iterative message passing with shared weights before each inference step) and **reverse GNN** (forward + reverse message passes to mitigate oversmoothing), composable as `rev-reasoning`.

## Current Milestone: v1.2 Constraint State Vocabulary

**Goal:** Replace the three separate `capacity`, `degree`, and `unused_capacity` node embeddings with a single `nn.Embedding` over the joint `(degree, net_capacity)` space, giving every constraint situation its own learned vector — aligned with the deductive rules a human Hashi solver applies.

**Target features:**
- `use_constraint_vocab` toggle in node encoder section of all three config files
- `ConstraintVocabEmbedding`: `nn.Embedding` over joint `(degree, net_capacity)` with 52–68 entries
- Config validation: when `use_constraint_vocab: true`, the three replaced features (`use_structural_degree`, `use_structural_degree_nsew`, `use_capacity`, `use_unused_capacity`) must all be `false` — error otherwise
- NodeEncoder wiring: when enabled, use vocab embedding instead of the three separate embeddings

## Core Value

The model learns to make decisions that are good for a sequence of steps, not just the next step — relaxing myopic per-step optimization with a longer-horizon gradient signal.

## Requirements

### Validated

- ✓ Multi-step diffusion training via `num_inference_steps_training` — existing
- ✓ Per-step loss accumulation (mean) with `.backward()` — existing
- ✓ Config-driven model/training via YAML + typed dataclasses (`TrainingConfig`) — existing
- ✓ `DiffusionTrainer` in `src2/hashi_puzzle_solver/trainers/diffusion.py` — existing
- ✓ BPTT sliding-window training mode (`bptt.enabled`, `window`, `stride`, `loss_ema_decay`) — v1.0
- ✓ Gradient checkpointing within BPTT windows — v1.0
- ✓ Gradient accumulation across overlapping windows — v1.0
- ✓ `BpttConfig` dataclass and `bptt:` YAML block in `TrainingConfig` — v1.0

### Active

- [ ] `use_constraint_vocab` toggle in node encoder toggles of all three config files
- [ ] Config validation: `use_constraint_vocab: true` requires `use_structural_degree`, `use_structural_degree_nsew`, `use_capacity`, `use_unused_capacity` all `false`
- [ ] `ConstraintVocabEmbedding`: `nn.Embedding(4 * NC_BINS, vocab_dim)` over joint `(degree, net_capacity)` space
- [ ] NodeEncoder wiring: vocab embedding replaces separate capacity/degree/unused embeddings when enabled

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
*Last updated: 2026-03-09 after v1.1 milestone start*
