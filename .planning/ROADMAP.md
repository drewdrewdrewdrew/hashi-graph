# Roadmap: BPTT Training for Hashi Diffusion Solver

## Overview

Two phases: first establish the config schema so the training loop has typed parameters to read, then implement the BPTT sliding-window training loop itself. Phase 1 is the prerequisite; Phase 2 is the entire value delivery. Backward compatibility is validated in Phase 2 because it requires a modified training loop to test.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Config Schema** - Add `bptt:` YAML block and mirror in `TrainingConfig` dataclass (completed 2026-03-06)
- [x] **Phase 2: BPTT Training Loop** - Implement sliding-window BPTT with checkpointing, accumulation, and EMA (completed 2026-03-06)

## Phase Details

### Phase 1: Config Schema
**Goal**: The `bptt:` sub-block exists in YAML and is fully typed in `TrainingConfig`, so the training loop can read all BPTT parameters
**Depends on**: Nothing (first phase)
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04
**Success Criteria** (what must be TRUE):
  1. `diffusion_solver_continuous.yaml` contains a `bptt:` block with `enabled`, `window`, `stride`, and `loss_ema_decay` fields
  2. `TrainingConfig` dataclass has a typed `bptt` field (nested dataclass) with all four parameters and correct defaults
  3. Setting `bptt.enabled: false` (the default) causes no visible change to trainer behavior — existing training runs load without error
  4. Config validation rejects malformed values (e.g., window < 1, stride < 1)
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Add BpttConfig dataclass and bptt: YAML block

### Phase 2: BPTT Training Loop
**Goal**: When `bptt.enabled: true`, the diffusion training loop uses sliding-window backpropagation through time, with gradient checkpointing and window-loss EMA, while remaining byte-for-byte equivalent to current behavior when disabled
**Depends on**: Phase 1
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04, TRN-05, TRN-06, COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. With `bptt.enabled: false`, a training run produces identical loss values and optimizer steps to the unmodified `DiffusionTrainer`
  2. With `bptt.enabled: true`, the cached edge-logit state at each step boundary is used as the start point for window re-runs, and gradients are present on model parameters after `.backward()` calls spanning multiple steps
  3. Peak GPU memory during a BPTT-enabled run is bounded by gradient checkpointing — activation memory does not grow linearly with window size
  4. Overlapping windows each call `.backward()` before the optimizer steps, so steps covered by multiple windows accumulate gradient from all covering windows
  5. The loss scalar reported per training iteration is the EMA-smoothed window-averaged step loss
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — BPTT dispatch + step-state cache + backward-compat guard (TRN-01, COMP-01, COMP-02)
- [x] 02-02-PLAN.md — Sliding-window loop with gradient checkpointing, accumulation, and EMA (TRN-02 through TRN-06)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Config Schema | 1/1 | Complete    | 2026-03-06 |
| 2. BPTT Training Loop | 2/2 | Complete    | 2026-03-06 |
